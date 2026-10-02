"""Optional LLM-generated cluster summaries via Gemini's structured output.

Pipeline-agnostic: works off the {cluster id, member rows} shape produced by
app.py's build_cluster_stats, regardless of which clustering algorithm
produced it.

Schema conformance (the model returning valid {id, summary} objects) is
Gemini's own guarantee via response_schema, so it isn't re-tested here --
tests cover request shaping, retry behaviour, and response-to-dict mapping,
all of which are this module's own responsibility.
"""
import json
import os

import httpx
import tenacity
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from config import (
    API_KEY_ENV_VAR,
    GEMINI_MODEL,
    MAX_RETRIES,
    REQUEST_TIMEOUT_MS,
    RETRY_BACKOFF_SECONDS,
    SYSTEM_INSTRUCTION,
    TOP_N_ITEMS,
)

# Indirection so tests can mock out the actual wait (tenacity.nap.sleep is
# itself "the default strategy, and may be mocked out for unit testing" per
# its own docstring) without reaching into tenacity's internals directly.
_retry_sleep = tenacity.nap.sleep


class ClusterSummary(BaseModel):
    id: int
    summary: str


class ClusterLabelingError(Exception):
    pass


def is_available() -> bool:
    return bool(os.environ.get(API_KEY_ENV_VAR))


def _flatten(value) -> str:
    return " ".join(value) if isinstance(value, list) else str(value)


def _build_request_items(stats: dict, field_name: str) -> list[dict]:
    return [
        {
            "id": entry["cluster"],
            "items": [_flatten(row[field_name]) for row in entry["data"][:TOP_N_ITEMS]],
        }
        for entry in stats.values()
    ]


def _is_retryable(exc: Exception) -> bool:
    # Server-side (5xx), network/timeout, and rate-limit (429) failures are
    # worth retrying; other client errors (bad request, auth, ...) won't be
    # fixed by trying again.
    if isinstance(exc, (errors.ServerError, httpx.HTTPError)):
        return True
    return isinstance(exc, errors.ClientError) and exc.code == 429


@tenacity.retry(
    retry=tenacity.retry_if_exception(_is_retryable),
    stop=tenacity.stop_after_attempt(MAX_RETRIES),
    wait=tenacity.wait_exponential(multiplier=RETRY_BACKOFF_SECONDS),
    sleep=lambda seconds: _retry_sleep(seconds),
    reraise=True,
)
def _generate(client, request_items: list[dict]):
    return client.models.generate_content(
        model=GEMINI_MODEL,
        contents=json.dumps(request_items),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=list[ClusterSummary],
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
        ),
    )


class _ClientCache:
    """Lazily constructs a genai.Client once and reuses it for the rest of
    the process's lifetime."""

    def __init__(self):
        self._client = None

    def get(self):
        if self._client is None:
            if not is_available():
                raise ClusterLabelingError(f"{API_KEY_ENV_VAR} is not set.")
            self._client = genai.Client(api_key=os.environ[API_KEY_ENV_VAR])
        return self._client


_client_cache = _ClientCache()


def label_clusters(stats: dict, field_name: str, client=None) -> dict[int, str]:
    """Generates a one-line summary per cluster. Raises ClusterLabelingError
    on any failure (missing/invalid key, exhausted retries, ...) -- callers
    should treat labeling as a non-blocking enhancement and keep showing the
    rest of the review UI either way."""
    if client is None:
        client = _client_cache.get()

    request_items = _build_request_items(stats, field_name)
    try:
        parsed = _generate(client, request_items).parsed
    except Exception as exc:
        raise ClusterLabelingError(str(exc)) from exc

    # google-genai doesn't raise when a response can't be parsed into the
    # schema (e.g. safety-filtered or truncated content with no text at
    # all) -- it silently leaves .parsed as None instead. Treat that the
    # same as any other failure rather than crashing on the next line.
    if parsed is None:
        raise ClusterLabelingError("Gemini returned no parseable structured output.")

    return {item.id: item.summary for item in parsed}
