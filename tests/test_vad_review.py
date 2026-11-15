import http.client
import json
from pathlib import Path
import threading

import pytest

from app.vad_review import (
    ReviewCatalog,
    ReviewError,
    create_server,
    parse_byte_range,
    playback_bounds,
)


@pytest.fixture
def review_fixture(tmp_path):
    recording_root = tmp_path / "Record File"
    source = recording_root / "CB" / "Nov26" / "meeting.mp3"
    source.parent.mkdir(parents=True)
    source.write_bytes(bytes(range(64)))
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    adjustment_dir = tmp_path / "adjustments"
    manifest_path = manifest_dir / "meeting.vad.json"
    payload = {
        "recording_id": "a" * 16,
        "source_mp3": str(source),
        "audio_duration_ms": 10_000,
        "vad_engine": {"name": "WebRTC VAD"},
        "segments": [
            {
                "segment_id": "seg-0001",
                "start_ms": 0,
                "end_ms": 600,
                "duration_ms": 600,
                "alignment_status": "not_aligned",
                "transcript_text": None,
            },
            {
                "segment_id": "seg-0002",
                "start_ms": 1000,
                "end_ms": 2500,
                "duration_ms": 1500,
                "alignment_status": "not_aligned",
                "transcript_text": None,
            },
        ],
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    optimized_path = manifest_dir / "meeting.vad.optimized-review.json"
    optimized_payload = {
        **payload,
        "manifest_type": "optimized_review_segments",
        "segments": [
            {
                "segment_id": "review-0001",
                "start_ms": 0,
                "end_ms": 2500,
                "duration_ms": 2500,
                "source_segment_ids": ["seg-0001", "seg-0002"],
                "source_segments": [
                    {"segment_id": "seg-0001", "original_start_ms": 0, "original_end_ms": 600},
                    {"segment_id": "seg-0002", "original_start_ms": 1000, "original_end_ms": 2500},
                ],
                "merge_status": "merged",
                "alignment_status": "not_aligned",
                "transcript_text": None,
            },
            {
                "segment_id": "review-0002",
                "start_ms": 5000,
                "end_ms": 6000,
                "duration_ms": 1000,
                "source_segment_ids": ["seg-0003"],
                "source_segments": [
                    {"segment_id": "seg-0003", "original_start_ms": 5000, "original_end_ms": 6000}
                ],
                "merge_status": "unchanged",
                "alignment_status": "not_aligned",
                "transcript_text": None,
            },
        ],
    }
    optimized_path.write_text(json.dumps(optimized_payload), encoding="utf-8")
    return {
        "recording_root": recording_root,
        "source": source,
        "manifest_dir": manifest_dir,
        "manifest_path": manifest_path,
        "optimized_path": optimized_path,
        "adjustment_dir": adjustment_dir,
        "payload": payload,
    }


def make_catalog(data):
    return ReviewCatalog(
        data["manifest_dir"],
        data["recording_root"],
        data["adjustment_dir"],
    )


def test_manifest_loading_and_recording_navigation(review_fixture):
    catalog = make_catalog(review_fixture)

    assert catalog.list_recordings() == [
        {
            "recording_id": "a" * 16,
            "mode": "original",
            "plant": "CB",
            "month": "Nov26",
            "recording": "meeting.mp3",
            "audio_duration_ms": 10_000,
            "segment_count": 2,
        }
    ]
    recording = catalog.get_recording("a" * 16)
    assert recording["segments"][0]["alignment_status"] == "not_aligned"
    assert recording["segments"][0]["transcript_text"] is None


def test_original_is_default_and_optimized_catalog_has_separate_segments(review_fixture):
    catalog = make_catalog(review_fixture)
    assert catalog.list_recordings()[0]["segment_count"] == 2
    assert catalog.list_recordings("optimized")[0]["segment_count"] == 2
    optimized = catalog.get_recording("a" * 16, "optimized")
    assert optimized["segments"][0]["segment_id"] == "review-0001"
    assert optimized["segments"][0]["merge_status"] == "merged"
    assert optimized["segments"][0]["source_segment_ids"] == ["seg-0001", "seg-0002"]
    assert catalog.get_audio_path("a" * 16, "optimized") == review_fixture["source"]


def test_optimized_manifest_rejects_malformed_source_lineage(review_fixture):
    payload = json.loads(review_fixture["optimized_path"].read_text())
    payload["segments"][0]["source_segments"][0] = "not-an-object"
    review_fixture["optimized_path"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReviewError, match="lineage"):
        make_catalog(review_fixture)


def test_audio_access_and_single_range_streaming(review_fixture):
    catalog = make_catalog(review_fixture)
    server = create_server(catalog, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection(*server.server_address, timeout=5)
        connection.request(
            "GET",
            f"/api/recordings/{'a' * 16}/audio",
            headers={"Range": "bytes=3-7"},
        )
        response = connection.getresponse()
        assert response.status == 206
        assert response.getheader("Accept-Ranges") == "bytes"
        assert response.getheader("Content-Range") == "bytes 3-7/64"
        assert response.read() == bytes(range(3, 8))
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_unknown_recording_and_segment_ids_are_not_found(review_fixture):
    catalog = make_catalog(review_fixture)
    server = create_server(catalog, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection(*server.server_address, timeout=5)
        connection.request("GET", f"/api/recordings/{'a' * 16}/segments/missing/playback")
        response = connection.getresponse()
        assert response.status == 404
        response.read()
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_playback_endpoint_clamps_context_for_short_segment(review_fixture):
    server = create_server(make_catalog(review_fixture), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection(*server.server_address, timeout=5)
        connection.request(
            "GET",
            f"/api/recordings/{'a' * 16}/segments/seg-0001/playback?before=1&after=0.5",
        )
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read()) == {"start_ms": 0, "end_ms": 1100}
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_path_traversal_is_rejected_for_manifest_and_http_routes(review_fixture):
    outside = review_fixture["manifest_dir"].parent / "secret.mp3"
    outside.write_bytes(b"not-a-recording")
    malicious = dict(review_fixture["payload"], source_mp3=str(outside))
    review_fixture["manifest_path"].write_text(json.dumps(malicious), encoding="utf-8")

    with pytest.raises(ReviewError, match="outside"):
        make_catalog(review_fixture)

    review_fixture["manifest_path"].write_text(
        json.dumps(review_fixture["payload"]), encoding="utf-8"
    )
    server = create_server(make_catalog(review_fixture), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection(*server.server_address, timeout=5)
        connection.request("GET", "/api/recordings/%2e%2e%2fsecret/audio")
        response = connection.getresponse()
        assert response.status == 404
        response.read()
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize(
    "start_ms,end_ms,duration_ms",
    [(-1, 10, 100), (10, 10, 100), (10, 101, 100), (True, 20, 100)],
)
def test_invalid_manifest_timestamps_are_rejected(review_fixture, start_ms, end_ms, duration_ms):
    payload = review_fixture["payload"]
    payload["audio_duration_ms"] = duration_ms
    payload["segments"][0].update(
        start_ms=start_ms,
        end_ms=end_ms,
        duration_ms=end_ms - start_ms if type(start_ms) is int and type(end_ms) is int else 0,
    )
    review_fixture["manifest_path"].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ReviewError):
        make_catalog(review_fixture)


def test_playback_boundaries_clamp_context_and_keep_short_segments():
    assert playback_bounds(0, 600, 10_000, context_before_ms=250, context_after_ms=500) == (
        0,
        1100,
    )
    assert playback_bounds(
        9_700, 10_000, 10_000, context_before_ms=1_000, context_after_ms=1_000
    ) == (8700, 10_000)
    with pytest.raises(ReviewError):
        playback_bounds(0, 100, 100, context_before_ms=60_001)


def test_adjustment_is_separate_and_preserves_manifest_original(review_fixture):
    catalog = make_catalog(review_fixture)
    adjusted = catalog.update_segment("a" * 16, "seg-0001", 25, 625)
    adjustment_path = review_fixture["adjustment_dir"] / f"{'a' * 16}.adjustments.json"

    assert adjusted["start_ms"] == 25
    assert adjusted["original_start_ms"] == 0
    assert json.loads(review_fixture["manifest_path"].read_text())["segments"][0]["start_ms"] == 0
    assert json.loads(adjustment_path.read_text())["segments"]["seg-0001"] == {
        "original_start_ms": 0,
        "original_end_ms": 600,
        "start_ms": 25,
        "end_ms": 625,
    }
    reloaded = make_catalog(review_fixture).get_recording("a" * 16)
    assert reloaded["segments"][0]["start_ms"] == 25
    assert reloaded["segments"][0]["original_start_ms"] == 0


def test_original_and_optimized_adjustments_use_independent_sidecars(review_fixture):
    catalog = make_catalog(review_fixture)
    catalog.update_segment("a" * 16, "seg-0001", 25, 625)
    catalog.update_segment("a" * 16, "review-0001", 50, 2550, "optimized")
    original_path = review_fixture["adjustment_dir"] / f"{'a' * 16}.adjustments.json"
    optimized_path = review_fixture["adjustment_dir"] / f"{'a' * 16}.optimized.adjustments.json"
    original_before = original_path.read_bytes()

    assert optimized_path.is_file()
    assert json.loads(original_before)["segments"]["seg-0001"]["start_ms"] == 25
    assert json.loads(optimized_path.read_text())["mode"] == "optimized"
    assert original_path.read_bytes() == original_before
    reloaded = make_catalog(review_fixture)
    assert reloaded.get_recording("a" * 16)["segments"][0]["start_ms"] == 25
    assert reloaded.get_recording("a" * 16, "optimized")["segments"][0]["start_ms"] == 50


def test_http_timing_adjustment_updates_sidecar_only(review_fixture):
    catalog = make_catalog(review_fixture)
    server = create_server(catalog, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection(*server.server_address, timeout=5)
        body = json.dumps({"start_ms": 20, "end_ms": 700})
        connection.request(
            "PATCH",
            f"/api/recordings/{'a' * 16}/segments/seg-0001",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        result = json.loads(response.read())
        assert response.status == 200
        assert result["start_ms"] == 20
        assert json.loads(review_fixture["manifest_path"].read_text())["segments"][0]["start_ms"] == 0
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_http_optimized_mode_uses_optimized_playback_and_adjustment_sidecar(review_fixture):
    server = create_server(make_catalog(review_fixture), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection(*server.server_address, timeout=5)
        connection.request("GET", f"/api/recordings/{'a' * 16}/segments/review-0001/playback?mode=optimized")
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read()) == {"start_ms": 0, "end_ms": 2500}

        body = json.dumps({"start_ms": 75, "end_ms": 2600})
        connection.request(
            "PATCH",
            f"/api/recordings/{'a' * 16}/segments/review-0001?mode=optimized",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["start_ms"] == 75
        assert not (
            review_fixture["adjustment_dir"] / f"{'a' * 16}.adjustments.json"
        ).exists()
        assert (
            review_fixture["adjustment_dir"] / f"{'a' * 16}.optimized.adjustments.json"
        ).is_file()
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_byte_range_parser_supports_open_and_suffix_ranges_and_rejects_bad_ranges():
    assert parse_byte_range("bytes=2-5", 10) == (2, 5)
    assert parse_byte_range("bytes=7-", 10) == (7, 9)
    assert parse_byte_range("bytes=-3", 10) == (7, 9)
    assert parse_byte_range(None, 10) is None
    with pytest.raises(ValueError):
        parse_byte_range("bytes=0-1,4-5", 10)


def test_server_refuses_non_loopback_bind(review_fixture):
    with pytest.raises(ValueError, match="loopback"):
        create_server(make_catalog(review_fixture), host="0.0.0.0", port=0)
