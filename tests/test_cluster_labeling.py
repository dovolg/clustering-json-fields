import httpx
import pytest
from google.genai import errors

from config import MAX_RETRIES
from clustering.labeling import (
    ClusterLabelingError,
    ClusterSummary,
    _build_request_items,
    _is_retryable,
    is_available,
    label_clusters,
)

STATS = {
    0: {"cluster": 1, "data": [{"tags": ["python", "pandas"]}, {"tags": ["python", "numpy"]}]},
    1: {"cluster": 2, "data": [{"tags": "single string value"}]},
}


class _FakeModels:
    """Replays a scripted sequence of outcomes (exceptions or responses),
    one per call to generate_content -- lets tests drive the real,
    tenacity-decorated _generate through its actual retry/backoff logic via
    label_clusters's client injection point, rather than bypassing it."""

    def __init__(self, outcomes, calls):
        self._outcomes = list(outcomes)
        self._calls = calls

    def generate_content(self, **kwargs):
        self._calls["n"] += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, outcomes):
        self.calls = {"n": 0}
        self.models = _FakeModels(outcomes, self.calls)


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------

def test_is_available_reflects_env_var(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert is_available() is False

    monkeypatch.setenv("GEMINI_API_KEY", "a-key")
    assert is_available() is True


# ---------------------------------------------------------------------------
# _build_request_items -- schema conformance is Gemini's own guarantee (via
# response_schema); what's actually this module's responsibility is getting
# the right text to the model in the first place.
# ---------------------------------------------------------------------------

def test_build_request_items_flattens_lists_and_keeps_strings():
    items = _build_request_items(STATS, "tags")

    assert items == [
        {"id": 1, "items": ["python pandas", "python numpy"]},
        {"id": 2, "items": ["single string value"]},
    ]


def test_build_request_items_truncates_to_top_n():
    many_rows = {0: {"cluster": 1, "data": [{"tags": [str(i)]} for i in range(50)]}}

    items = _build_request_items(many_rows, "tags")

    assert len(items[0]["items"]) == 15
    assert items[0]["items"] == [str(i) for i in range(15)]  # most-representative-first prefix, not a reorder


# ---------------------------------------------------------------------------
# _is_retryable
# ---------------------------------------------------------------------------

def test_server_network_and_rate_limit_errors_are_retryable():
    assert _is_retryable(errors.ServerError(code=503, response_json={})) is True
    assert _is_retryable(httpx.TimeoutException("timed out")) is True
    # 429 is a ClientError by HTTP status range (4xx), but it's the most
    # likely real-world transient failure against a rate-limited key, so it
    # needs its own carve-out rather than falling under "client errors".
    assert _is_retryable(errors.ClientError(code=429, response_json={})) is True


def test_other_client_errors_are_not_retryable():
    assert _is_retryable(errors.ClientError(code=400, response_json={})) is False
    assert _is_retryable(ValueError("bad schema")) is False


# ---------------------------------------------------------------------------
# label_clusters -- driven through FakeClient so these exercise the real,
# tenacity-decorated _generate (including its actual retry/backoff logic)
# without ever touching the network.
# ---------------------------------------------------------------------------

def test_label_clusters_raises_when_key_missing(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ClusterLabelingError):
        label_clusters(STATS, "tags")


def test_label_clusters_maps_response_to_cluster_id_dict():
    class FakeResponse:
        parsed = [ClusterSummary(id=1, summary="Python tooling"), ClusterSummary(id=2, summary="Misc")]

    client = FakeClient([FakeResponse()])

    result = label_clusters(STATS, "tags", client=client)

    assert result == {1: "Python tooling", 2: "Misc"}
    assert client.calls["n"] == 1


def test_label_clusters_result_omits_ids_missing_from_response():
    # response_schema enforces the *shape* of each returned item, not that
    # the model returns one item per requested cluster. A short response
    # should just mean fewer labels, not a crash or a mismatched mapping.
    class FakeResponse:
        parsed = [ClusterSummary(id=1, summary="only this one")]

    client = FakeClient([FakeResponse()])

    result = label_clusters(STATS, "tags", client=client)

    assert result == {1: "only this one"}


def test_label_clusters_retries_on_transient_error_then_succeeds(monkeypatch):
    monkeypatch.setattr("clustering.labeling._retry_sleep", lambda seconds: None)

    class FakeResponse:
        parsed = [ClusterSummary(id=1, summary="ok"), ClusterSummary(id=2, summary="ok")]

    # Fails on every attempt but the last -- MAX_RETRIES - 1 retryable errors
    # followed by a success, requiring every attempt to actually be used.
    outcomes = [errors.ServerError(code=503, response_json={})] * (MAX_RETRIES - 1) + [FakeResponse()]
    client = FakeClient(outcomes)

    result = label_clusters(STATS, "tags", client=client)

    assert result == {1: "ok", 2: "ok"}
    assert client.calls["n"] == MAX_RETRIES


def test_label_clusters_retries_on_rate_limit_error(monkeypatch):
    monkeypatch.setattr("clustering.labeling._retry_sleep", lambda seconds: None)

    class FakeResponse:
        parsed = [ClusterSummary(id=1, summary="ok")]

    client = FakeClient([errors.ClientError(code=429, response_json={}), FakeResponse()])

    result = label_clusters(STATS, "tags", client=client)

    assert result == {1: "ok"}
    assert client.calls["n"] == 2


def test_label_clusters_raises_when_response_has_no_parsed_content():
    # google-genai leaves .parsed as None (no exception) when a response
    # can't be parsed into the schema -- e.g. safety-filtered or truncated
    # content with no text at all. Regression test: this used to reach a
    # bare `for item in None` and crash with an uncaught TypeError instead
    # of the intended ClusterLabelingError.
    class FakeResponse:
        parsed = None

    client = FakeClient([FakeResponse()])

    with pytest.raises(ClusterLabelingError):
        label_clusters(STATS, "tags", client=client)


def test_label_clusters_does_not_retry_non_retryable_error():
    client = FakeClient([errors.ClientError(code=400, response_json={})])

    with pytest.raises(ClusterLabelingError):
        label_clusters(STATS, "tags", client=client)

    assert client.calls["n"] == 1


def test_label_clusters_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr("clustering.labeling._retry_sleep", lambda seconds: None)

    client = FakeClient([errors.ServerError(code=503, response_json={})] * MAX_RETRIES)

    with pytest.raises(ClusterLabelingError):
        label_clusters(STATS, "tags", client=client)

    assert client.calls["n"] == MAX_RETRIES
