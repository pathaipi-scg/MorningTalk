import logging
from unittest.mock import Mock

import pytest

from app.azure_openai_http import (
    AzureResponseValidationError,
    extract_generated_text,
    log_azure_error,
    resolve_azure_openai_endpoint,
    response_token_usage,
)


def test_complete_responses_url_preserves_route_and_api_version():
    url = "https://resource.example/openai/responses?api-version=2025-04-01-preview"

    endpoint = resolve_azure_openai_endpoint(url, "v1", "gpt-5.4-mini")

    assert endpoint.url == url
    assert endpoint.params is None
    assert endpoint.api_mode == "responses"


@pytest.mark.parametrize(
    ("url", "version", "expected_url", "expected_mode", "expected_params"),
    [
        (
            "https://resource.example",
            "v1",
            "https://resource.example/openai/v1/chat/completions",
            "chat_completions",
            None,
        ),
        (
            "https://resource.example/openai/v1/responses",
            "v1",
            "https://resource.example/openai/v1/responses",
            "responses",
            None,
        ),
        (
            "https://resource.example",
            "2025-04-01-preview",
            "https://resource.example/openai/deployments/gpt-5.4-mini/chat/completions",
            "chat_completions",
            {"api-version": "2025-04-01-preview"},
        ),
    ],
)
def test_endpoint_classification_preserves_existing_chat_and_v1_modes(
    url, version, expected_url, expected_mode, expected_params
):
    endpoint = resolve_azure_openai_endpoint(url, version, "gpt-5.4-mini")

    assert endpoint.url == expected_url
    assert endpoint.params == expected_params
    assert endpoint.api_mode == expected_mode


def test_endpoint_classification_rejects_unknown_route_and_bad_deployment():
    with pytest.raises(ValueError, match="unrecognized"):
        resolve_azure_openai_endpoint(
            "https://resource.example/openai/something-else",
            "v1",
            "gpt-5.4-mini",
        )
    with pytest.raises(ValueError, match="does not match"):
        resolve_azure_openai_endpoint(
            "https://resource.example/openai/deployments/other/chat/completions?api-version=2025-04-01-preview",
            "v1",
            "gpt-5.4-mini",
        )


def test_responses_parser_extracts_text_usage_and_rejects_refusal_or_incomplete():
    payload = {
        "status": "completed",
        "output": [{
            "type": "message",
            "content": [{"type": "output_text", "text": "{\"ok\":true}"}],
        }],
        "usage": {"input_tokens": 23, "output_tokens": 7},
    }

    assert extract_generated_text(payload, "responses") == "{\"ok\":true}"
    assert response_token_usage(payload, "responses") == (23, 7)
    with pytest.raises(AzureResponseValidationError, match="Refusal"):
        extract_generated_text(
            {
                "status": "completed",
                "output": [{
                    "type": "message",
                    "content": [{"type": "refusal", "refusal": "not allowed"}],
                }],
            },
            "responses",
        )
    with pytest.raises(AzureResponseValidationError, match="IncompleteResponse"):
        extract_generated_text(
            {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
            "responses",
        )
    with pytest.raises(AzureResponseValidationError, match="MalformedOutput"):
        extract_generated_text({"status": "completed", "output": []}, "responses")


def test_404_diagnostics_capture_safe_metadata_without_response_or_transcript(
    caplog,
):
    response = Mock(
        status_code=404,
        headers={
            "apim-request-id": "request-123",
            "Authorization": "must-not-be-logged",
        },
    )
    response.json.return_value = {
        "error": {
            "code": "DeploymentNotFound",
            "message": "secret transcript phrase api-key=credential-value",
        }
    }

    with caplog.at_level(logging.ERROR):
        diagnostics = log_azure_error(
            logging.getLogger("test.azure"),
            response,
            "https://resource.example/openai/v1/chat/completions",
            "gpt-5.4-mini",
        )

    assert diagnostics.status == 404
    assert diagnostics.error_code == "DeploymentNotFound"
    assert diagnostics.message == "The configured deployment was not found on the Azure resource."
    assert diagnostics.request_path == "/openai/v1/chat/completions"
    assert diagnostics.api_mode == "unknown"
    assert diagnostics.deployment == "gpt-5.4-mini"
    assert diagnostics.request_id == "request-123"
    for secret in ("secret transcript phrase", "credential-value", "must-not-be-logged"):
        assert secret not in caplog.text


def test_404_diagnostics_omit_untrusted_response_message():
    response = Mock(status_code=404, headers={})
    response.json.return_value = {
        "error": {"code": "NotFound", "message": "do not log this body"}
    }

    diagnostics = log_azure_error(
        logging.getLogger("test.azure"),
        response,
        "https://resource.example/openai/deployments/deployment/chat/completions",
        "deployment",
    )

    assert diagnostics.error_code == "NotFound"
    assert diagnostics.message == "Azure returned HTTP 404; route or deployment not found. Response body omitted."
    assert "do not log this body" not in diagnostics.message


@pytest.mark.parametrize("status", [400, 401])
def test_error_diagnostics_include_api_mode_without_logging_body(status, caplog):
    response = Mock(
        status_code=status,
        headers={"x-ms-request-id": "safe-request-id"},
    )
    response.json.return_value = {
        "error": {
            "code": "InvalidRequest",
            "message": "transcript text and key=secret",
        }
    }

    with caplog.at_level(logging.ERROR):
        diagnostics = log_azure_error(
            logging.getLogger("test.azure.mode"),
            response,
            "https://resource.example/openai/responses?api-version=secret-version",
            "gpt-5.4-mini",
            "responses",
        )

    assert diagnostics.status == status
    assert diagnostics.api_mode == "responses"
    assert diagnostics.request_path == "/openai/responses"
    assert diagnostics.request_id == "safe-request-id"
    assert "secret-version" not in caplog.text
    assert "transcript text" not in caplog.text
    assert "key=secret" not in caplog.text
