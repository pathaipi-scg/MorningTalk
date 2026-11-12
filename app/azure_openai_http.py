"""Safe diagnostics for Azure OpenAI HTTP errors."""

from dataclasses import dataclass
import logging
import re
from typing import Any
from urllib.parse import parse_qsl, quote, unquote, urlsplit


_SAFE_IDENTIFIER = re.compile(r"[^A-Za-z0-9._:-]")


@dataclass(frozen=True)
class AzureErrorDiagnostics:
    status: int
    error_code: str | None
    message: str
    request_path: str
    deployment: str
    request_id: str | None
    api_mode: str


@dataclass(frozen=True)
class AzureOpenAIEndpoint:
    url: str
    params: dict[str, str] | None
    api_mode: str


class AzureResponseValidationError(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def resolve_azure_openai_endpoint(
    configured_endpoint: str,
    api_version: str,
    deployment: str,
) -> AzureOpenAIEndpoint:
    parsed = urlsplit(configured_endpoint)
    if (
        parsed.scheme.casefold() != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.fragment
    ):
        raise ValueError("AZURE_OPENAI_ENDPOINT must be an HTTPS URL without embedded credentials or fragments.")
    if not deployment:
        raise ValueError("AZURE_OPENAI_DEPLOYMENT is required.")

    path = parsed.path.rstrip("/")
    query = parse_qsl(parsed.query, keep_blank_values=True)
    query_versions = [value for key, value in query if key.casefold() == "api-version"]
    if len(query_versions) > 1:
        raise ValueError("AZURE_OPENAI_ENDPOINT contains multiple api-version parameters.")

    if path == "/openai/responses":
        if not query_versions or not query_versions[0]:
            raise ValueError("A versioned Azure OpenAI Responses endpoint requires its api-version query parameter.")
        return AzureOpenAIEndpoint(configured_endpoint, None, "responses")
    if path == "/openai/v1/responses":
        return AzureOpenAIEndpoint(configured_endpoint, None, "responses")
    if path == "/openai/v1/chat/completions":
        return AzureOpenAIEndpoint(configured_endpoint, None, "chat_completions")

    deployment_chat_path = re.fullmatch(
        r"/openai/deployments/([^/]+)/chat/completions", path
    )
    if deployment_chat_path:
        configured_deployment = unquote(deployment_chat_path.group(1))
        if configured_deployment != deployment:
            raise ValueError("AZURE_OPENAI_ENDPOINT deployment does not match AZURE_OPENAI_DEPLOYMENT.")
        if not query_versions and api_version.casefold() == "v1":
            raise ValueError("A legacy deployment-specific Chat Completions endpoint requires a dated api-version.")
        params = None if query_versions else {"api-version": api_version}
        return AzureOpenAIEndpoint(configured_endpoint, params, "chat_completions")

    if path in {"", "/"} and not query:
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if api_version.casefold() == "v1":
            return AzureOpenAIEndpoint(
                f"{origin}/openai/v1/chat/completions", None, "chat_completions"
            )
        return AzureOpenAIEndpoint(
            f"{origin}/openai/deployments/{quote(deployment, safe='')}/chat/completions",
            {"api-version": api_version},
            "chat_completions",
        )

    raise ValueError("AZURE_OPENAI_ENDPOINT uses an unsupported or unrecognized Azure OpenAI API route.")


def extract_generated_text(payload: Any, api_mode: str) -> str:
    if not isinstance(payload, dict):
        raise AzureResponseValidationError("MalformedOutput")

    if api_mode == "responses":
        status = payload.get("status")
        if status is not None and not isinstance(status, str):
            raise AzureResponseValidationError("MalformedOutput")
        if status == "incomplete" or status in {"queued", "in_progress"}:
            raise AzureResponseValidationError("IncompleteResponse")
        if status in {"failed", "cancelled"}:
            raise AzureResponseValidationError("ResponseFailed")
        if status is not None and status != "completed":
            raise AzureResponseValidationError("MalformedOutput")
        if payload.get("incomplete_details") is not None:
            raise AzureResponseValidationError("IncompleteResponse")
        if payload.get("error") is not None:
            raise AzureResponseValidationError("ResponseFailed")

        collected: list[str] = []
        output = payload.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "refusal":
                        raise AzureResponseValidationError("Refusal")
                    if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                        collected.append(part["text"])

        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text
        if collected:
            return "".join(collected)
        raise AzureResponseValidationError("MalformedOutput")

    if api_mode != "chat_completions":
        raise AzureResponseValidationError("UnsupportedResponseMode")
    try:
        choice = payload["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError):
        raise AzureResponseValidationError("MalformedOutput") from None
    if not isinstance(choice, dict) or not isinstance(message, dict):
        raise AzureResponseValidationError("MalformedOutput")
    if choice.get("finish_reason") == "length":
        raise AzureResponseValidationError("IncompleteResponse")
    if message.get("refusal"):
        raise AzureResponseValidationError("Refusal")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise AzureResponseValidationError("MalformedOutput")
    return content


def response_token_usage(payload: Any, api_mode: str) -> tuple[int | None, int | None]:
    if not isinstance(payload, dict) or not isinstance(payload.get("usage"), dict):
        return None, None
    usage = payload["usage"]
    if api_mode == "responses":
        input_value, output_value = usage.get("input_tokens"), usage.get("output_tokens")
    else:
        input_value, output_value = usage.get("prompt_tokens"), usage.get("completion_tokens")
    return (
        input_value if isinstance(input_value, int) and not isinstance(input_value, bool) else None,
        output_value if isinstance(output_value, int) and not isinstance(output_value, bool) else None,
    )


def azure_error_diagnostics(
    response: Any, request_url: str, deployment: str, api_mode: str = "unknown"
) -> AzureErrorDiagnostics:
    try:
        payload = response.json()
    except (ValueError, TypeError):
        payload = None

    error = payload.get("error") if isinstance(payload, dict) else None
    raw_code = error.get("code") if isinstance(error, dict) else None
    error_code = None
    if isinstance(raw_code, str):
        candidate = _SAFE_IDENTIFIER.sub("", raw_code)[:80]
        error_code = candidate or None

    status = int(response.status_code)
    if (error_code or "").casefold() == "deploymentnotfound":
        message = "The configured deployment was not found on the Azure resource."
    elif status == 404:
        message = "Azure returned HTTP 404; route or deployment not found. Response body omitted."
    else:
        message = f"Azure returned HTTP {status}. Response body omitted."

    headers = getattr(response, "headers", {})
    request_id = None
    if hasattr(headers, "get"):
        for header in ("apim-request-id", "x-ms-request-id", "x-request-id"):
            value = headers.get(header)
            if isinstance(value, str) and value.strip():
                request_id = _SAFE_IDENTIFIER.sub("", value.strip())[:128] or None
                if request_id:
                    break

    return AzureErrorDiagnostics(
        status=status,
        error_code=error_code,
        message=message,
        request_path=urlsplit(request_url).path,
        deployment=_SAFE_IDENTIFIER.sub("", deployment)[:128],
        request_id=request_id,
        api_mode=_SAFE_IDENTIFIER.sub("", api_mode)[:40] or "unknown",
    )


def log_azure_error(
    logger: logging.Logger,
    response: Any,
    request_url: str,
    deployment: str,
    api_mode: str = "unknown",
) -> AzureErrorDiagnostics:
    diagnostics = azure_error_diagnostics(response, request_url, deployment, api_mode)
    logger.error(
        "Azure OpenAI request failed; http_status=%s azure_error_code=%s "
        "message=%s request_path=%s api_mode=%s deployment=%s azure_request_id=%s",
        diagnostics.status,
        diagnostics.error_code or "unavailable",
        diagnostics.message,
        diagnostics.request_path,
        diagnostics.api_mode,
        diagnostics.deployment,
        diagnostics.request_id or "unavailable",
    )
    return diagnostics
