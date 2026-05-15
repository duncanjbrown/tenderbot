"""Tests for tenderbot.py"""

import json
from unittest.mock import MagicMock, patch, call
import pytest
import requests

from tenderbot import (
    TENDER_TAGS,
    BatchResult,
    TenderMatch,
    evaluate,
    fetch_tenders,
    load_cache,
    load_results_cache,
    render_html,
    save_cache,
    save_results_cache,
    summarise,
)


# ---------------------------------------------------------------------------
# fetch_tenders
# ---------------------------------------------------------------------------


def _make_response(releases, next_url=None):
    resp = MagicMock(spec=requests.Response)
    resp.raise_for_status.return_value = None
    links = {"next": next_url} if next_url else {}
    resp.json.return_value = {"releases": releases, "links": links}
    return resp


def _release(tags, ocid="ocds-test-001", title="Test Tender"):
    return {"ocid": ocid, "tag": tags, "tender": {"title": title}, "buyer": {}}


class TestFetchTenders:
    def test_returns_only_tender_tagged_releases(self):
        tender = _release(["tender"])
        award = _release(["award"], ocid="ocds-test-002")
        with patch("tenderbot.requests.get") as mock_get:
            mock_get.return_value = _make_response([tender, award])
            result = fetch_tenders()
        assert len(result) == 1
        assert result[0]["ocid"] == "ocds-test-001"

    def test_excludes_cancellations(self):
        cancellation = _release(["tenderCancellation"])
        with patch("tenderbot.requests.get") as mock_get:
            mock_get.return_value = _make_response([cancellation])
            result = fetch_tenders()
        assert result == []

    def test_includes_tender_update_and_amendment(self):
        releases = [
            _release(["tenderUpdate"], ocid="ocds-001"),
            _release(["tenderAmendment"], ocid="ocds-002"),
        ]
        with patch("tenderbot.requests.get") as mock_get:
            mock_get.return_value = _make_response(releases)
            result = fetch_tenders()
        assert len(result) == 2

    def test_paginates_via_links_next(self):
        page1 = [_release(["tender"], ocid="ocds-001")]
        page2 = [_release(["tender"], ocid="ocds-002")]
        responses = [
            _make_response(page1, next_url="https://example.com/api?cursor=abc"),
            _make_response(page2),
        ]
        with patch("tenderbot.requests.get", side_effect=responses) as mock_get:
            result = fetch_tenders()
        assert len(result) == 2
        assert mock_get.call_count == 2
        # Second call should use the next URL directly
        assert mock_get.call_args_list[1] == call(
            "https://example.com/api?cursor=abc", timeout=30
        )

    def test_stops_when_empty_page(self):
        with patch("tenderbot.requests.get") as mock_get:
            mock_get.return_value = _make_response([])
            result = fetch_tenders()
        assert result == []
        assert mock_get.call_count == 1

    def test_raises_on_http_error(self):
        with patch("tenderbot.requests.get") as mock_get:
            mock_get.return_value.raise_for_status.side_effect = (
                requests.HTTPError("502")
            )
            with pytest.raises(requests.HTTPError):
                fetch_tenders()


# ---------------------------------------------------------------------------
# summarise
# ---------------------------------------------------------------------------


class TestSummarise:
    def test_extracts_key_fields(self):
        release = {
            "ocid": "ocds-abc-123",
            "tender": {
                "title": "Digital Health Platform",
                "description": "Build a new platform",
                "value": {"amount": 500000, "currency": "GBP"},
            },
            "buyer": {"name": "NHS England"},
        }
        result = summarise(release)
        assert result["ocid"] == "ocds-abc-123"
        assert result["title"] == "Digital Health Platform"
        assert result["description"] == "Build a new platform"
        assert result["buyer"] == "NHS England"
        assert result["value"] == "500000 GBP"

    def test_truncates_long_description(self):
        release = {
            "ocid": "x",
            "tender": {"description": "a" * 600},
            "buyer": {},
        }
        assert len(summarise(release)["description"]) == 400

    def test_handles_missing_fields_gracefully(self):
        result = summarise({})
        assert result["title"] == "(no title)"
        assert result["description"] == ""
        assert result["buyer"] == ""
        assert result["value"] is None

    def test_handles_none_tender_and_buyer(self):
        result = summarise({"tender": None, "buyer": None})
        assert result["title"] == "(no title)"

    def test_omits_value_when_amount_missing(self):
        release = {"tender": {"value": {"currency": "GBP"}}, "buyer": {}}
        assert summarise(release)["value"] is None


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def _make_anthropic_client(matches: list[TenderMatch]):
    """Return a mock Anthropic client whose .messages.parse() yields BatchResult."""
    parsed = BatchResult(results=matches)
    response = MagicMock()
    response.parsed_output = parsed
    client = MagicMock()
    client.messages.parse.return_value = response
    return client


class TestEvaluate:
    def test_returns_all_results_from_single_batch(self):
        releases = [_release(["tender"], ocid=f"ocds-00{i}") for i in range(3)]
        matches = [
            TenderMatch(ocid=f"ocds-00{i}", title="T", relevant=True, reason="r")
            for i in range(3)
        ]
        client = _make_anthropic_client(matches)
        result = evaluate(releases, "health", client)
        assert len(result) == 3

    def test_batches_in_groups_of_20(self):
        releases = [_release(["tender"], ocid=f"ocds-{i:03}") for i in range(45)]
        client = _make_anthropic_client([])
        evaluate(releases, "health", client)
        assert client.messages.parse.call_count == 3  # 20 + 20 + 5

    def test_passes_interests_in_system_prompt(self):
        releases = [_release(["tender"])]
        client = _make_anthropic_client([])
        evaluate(releases, "interoperability", client)
        call_kwargs = client.messages.parse.call_args.kwargs
        system_text = call_kwargs["system"][0]["text"]
        assert "interoperability" in system_text

    def test_uses_cache_control_on_system(self):
        releases = [_release(["tender"])]
        client = _make_anthropic_client([])
        evaluate(releases, "health", client)
        system_block = client.messages.parse.call_args.kwargs["system"][0]
        assert system_block["cache_control"] == {"type": "ephemeral"}

    def test_returns_empty_list_for_no_releases(self):
        client = _make_anthropic_client([])
        assert evaluate([], "health", client) == []
        client.messages.parse.assert_not_called()

    def test_accumulates_results_across_batches(self):
        releases = [_release(["tender"], ocid=f"ocds-{i:03}") for i in range(25)]
        batch1 = [TenderMatch(ocid=f"ocds-{i:03}", title="T", relevant=True, reason="r") for i in range(20)]
        batch2 = [TenderMatch(ocid=f"ocds-{i:03}", title="T", relevant=False, reason="r") for i in range(20, 25)]

        responses = [
            MagicMock(parsed_output=BatchResult(results=batch1)),
            MagicMock(parsed_output=BatchResult(results=batch2)),
        ]
        client = MagicMock()
        client.messages.parse.side_effect = responses
        result = evaluate(releases, "health", client)
        assert len(result) == 25


# ---------------------------------------------------------------------------
# save_cache / load_cache
# ---------------------------------------------------------------------------


class TestCache:
    def test_save_and_load_roundtrip(self, tmp_path):
        releases = [_release(["tender"], ocid="ocds-001"), _release(["tenderUpdate"], ocid="ocds-002")]
        path = tmp_path / "cache.json"
        save_cache(releases, str(path))
        loaded = load_cache(str(path))
        assert loaded == releases

    def test_save_creates_file(self, tmp_path):
        path = tmp_path / "cache.json"
        save_cache([], str(path))
        assert path.exists()

    def test_load_returns_none_when_file_missing(self, tmp_path):
        assert load_cache(str(tmp_path / "no_such_file.json")) is None

    def test_saved_file_is_valid_json(self, tmp_path):
        path = tmp_path / "cache.json"
        releases = [_release(["tender"])]
        save_cache(releases, str(path))
        data = json.loads(path.read_text())
        assert isinstance(data, list)


# ---------------------------------------------------------------------------
# save_results_cache / load_results_cache
# ---------------------------------------------------------------------------


class TestResultsCache:
    def test_roundtrip(self, tmp_path):
        results = [
            TenderMatch(ocid="ocds-001", title="Health Portal", relevant=True, reason="NHS"),
            TenderMatch(ocid="ocds-002", title="Road Works", relevant=False, reason="Unrelated"),
        ]
        path = str(tmp_path / "results.json")
        save_results_cache(results, "health, NHS", path)
        loaded_interests, loaded_results = load_results_cache(path)
        assert loaded_interests == "health, NHS"
        assert loaded_results == results

    def test_load_returns_none_when_missing(self, tmp_path):
        assert load_results_cache(str(tmp_path / "nope.json")) is None

    def test_saved_file_is_valid_json(self, tmp_path):
        path = tmp_path / "results.json"
        save_results_cache([], "health", str(path))
        data = json.loads(path.read_text())
        assert "interests" in data
        assert "results" in data

    def test_preserves_all_tender_match_fields(self, tmp_path):
        match = TenderMatch(ocid="ocds-xyz", title="T", relevant=True, reason="r")
        path = str(tmp_path / "r.json")
        save_results_cache([match], "health", path)
        _, loaded = load_results_cache(path)
        assert loaded[0].ocid == "ocds-xyz"
        assert loaded[0].relevant is True


# ---------------------------------------------------------------------------
# render_html
# ---------------------------------------------------------------------------


def _match(ocid="ocds-001", title="NHS Digital Platform", reason="Directly relates to health interoperability"):
    return TenderMatch(ocid=ocid, title=title, relevant=True, reason=reason)


class TestRenderHtml:
    def test_returns_string(self):
        assert isinstance(render_html([], 10, "health"), str)

    def test_contains_viewport_meta_for_mobile(self):
        html = render_html([], 0, "health")
        assert 'name="viewport"' in html

    def test_shows_interests(self):
        html = render_html([], 5, "health, NHS")
        assert "health, NHS" in html

    def test_shows_match_and_total_count(self):
        matches = [_match()]
        html = render_html(matches, 10, "health")
        assert "1" in html
        assert "10" in html

    def test_renders_each_match_title(self):
        matches = [_match(title="GP Data Services"), _match(ocid="ocds-002", title="Care Interop API")]
        html = render_html(matches, 20, "health")
        assert "GP Data Services" in html
        assert "Care Interop API" in html

    def test_renders_ocid(self):
        html = render_html([_match(ocid="ocds-b5fd17-123")], 5, "health")
        assert "ocds-b5fd17-123" in html

    def test_renders_reason(self):
        html = render_html([_match(reason="Relates to NHS prevention strategy")], 5, "health")
        assert "Relates to NHS prevention strategy" in html

    def test_no_matches_shows_empty_state(self):
        html = render_html([], 50, "health")
        assert "No matching tenders" in html

    def test_escapes_html_in_title(self):
        match = _match(title='<script>alert("xss")</script>')
        html = render_html([match], 1, "health")
        assert "<script>" not in html


# ---------------------------------------------------------------------------
# TENDER_TAGS constant
# ---------------------------------------------------------------------------


class TestTenderTags:
    def test_does_not_include_cancellation(self):
        assert "tenderCancellation" not in TENDER_TAGS

    def test_includes_core_tender_tags(self):
        assert {"tender", "tenderUpdate", "tenderAmendment"}.issubset(TENDER_TAGS)
