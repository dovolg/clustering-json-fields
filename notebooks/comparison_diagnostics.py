"""Standalone diagnostics for the final lexical-vs-semantic comparison
notebook. Deliberately independent of clustering/labeling.py -- that module
is production code for a different purpose (generating one-line cluster
summaries), not for judging cluster quality; this builds its own minimal
structured-output call for that.

The production clustering pipelines themselves (clustering.lexical,
clustering.semantic, clustering.kmeans) ARE imported directly by the
notebook that uses this module -- the point of that notebook is to
benchmark the real shipped code, same as notebooks/03.
"""
import os
from typing import Literal

import httpx
import pandas as pd
import tenacity
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

GEMINI_MODEL = "gemini-2.5-flash"
API_KEY_ENV_VAR = "GEMINI_API_KEY"
REQUEST_TIMEOUT_MS = 90_000
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2
TOP_N_ITEMS_PER_CLUSTER = 10
# Caps prompt size for clusterings with many clusters (found k has run as high as the
# 20s-40s for titles in earlier notebooks) -- the largest clusters are the most
# representative of the clustering's overall character, and an unbounded prompt risks
# exactly the timeout this cap is meant to avoid.
MAX_CLUSTERS_SHOWN = 15


def _is_retryable(exc: Exception) -> bool:
    # Same reasoning as clustering/labeling.py: server-side (5xx), network/timeout, and
    # rate-limit (429) failures are worth retrying; other client errors won't be fixed
    # by trying again.
    if isinstance(exc, (errors.ServerError, httpx.HTTPError)):
        return True
    return isinstance(exc, errors.ClientError) and exc.code == 429

JUDGE_SYSTEM_INSTRUCTION = (
    "You are judging the quality of two different clusterings of the same kind of short "
    "text items. Clustering A and Clustering B group similar items into topic clusters, "
    "but were produced by different methods -- they are not guaranteed to agree in cluster "
    "count or membership. For each clustering, rate its overall internal coherence (do "
    "items within each of its clusters share a clear, specific theme, as opposed to being "
    "a loose or arbitrary grouping?) on a 1-5 scale. Then say which clustering is more "
    "coherent overall, or 'tie' if they're comparable, and give one or two sentences of "
    "reasoning referencing specific clusters."
)


class PairwiseVerdict(BaseModel):
    coherence_a: int
    coherence_b: int
    more_coherent: Literal["A", "B", "tie"]
    reasoning: str


def is_available() -> bool:
    return bool(os.environ.get(API_KEY_ENV_VAR))


def _format_clustering(texts: pd.Series, labels, label: str) -> str:
    texts = pd.Series(texts).reset_index(drop=True)
    labels = pd.Series(labels).reset_index(drop=True)
    sizes = labels.value_counts()
    shown_clusters = sizes.index[:MAX_CLUSTERS_SHOWN]
    lines = [f"Clustering {label} ({len(sizes)} clusters total, showing the {len(shown_clusters)} largest):"]
    for cluster_id in shown_clusters:
        members = texts[labels == cluster_id].tolist()[:TOP_N_ITEMS_PER_CLUSTER]
        lines.append(f"Cluster {cluster_id} ({sizes[cluster_id]} items): " + " | ".join(members))
    return "\n".join(lines)


@tenacity.retry(
    retry=tenacity.retry_if_exception(_is_retryable),
    stop=tenacity.stop_after_attempt(MAX_RETRIES),
    wait=tenacity.wait_exponential(multiplier=RETRY_BACKOFF_SECONDS),
    reraise=True,
)
def _call_judge(client, texts_a, labels_a, texts_b, labels_b) -> PairwiseVerdict:
    prompt = "\n\n".join([
        _format_clustering(texts_a, labels_a, "A"),
        _format_clustering(texts_b, labels_b, "B"),
    ])
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=JUDGE_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=PairwiseVerdict,
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
        ),
    )
    if response.parsed is None:
        raise RuntimeError("Gemini returned no parseable structured output.")
    return response.parsed


def judge_pairwise_with_swap_check(name_1, texts_1, labels_1, name_2, texts_2, labels_2, client=None):
    """Judges two clusterings of the same texts against each other, once
    with each possible A/B assignment -- isolates each verdict to its own
    call (so it can't be swayed by a prior call's framing) while checking
    whether the verdict is just a position-bias artifact: if the same
    pipeline wins regardless of which letter it was given, that's a real
    signal; if the winner flips with the letter, it isn't.
    """
    if client is None:
        if not is_available():
            raise RuntimeError(f"{API_KEY_ENV_VAR} is not set.")
        client = genai.Client(api_key=os.environ[API_KEY_ENV_VAR])

    results = []
    for order_name, (name_a, texts_a, labels_a, name_b, texts_b, labels_b) in [
        ("1_as_A", (name_1, texts_1, labels_1, name_2, texts_2, labels_2)),
        ("2_as_A", (name_2, texts_2, labels_2, name_1, texts_1, labels_1)),
    ]:
        verdict = _call_judge(client, texts_a, labels_a, texts_b, labels_b)
        winner_name = {"A": name_a, "B": name_b, "tie": "tie"}.get(verdict.more_coherent, "unclear")
        results.append({
            "order": order_name,
            "a": name_a, "b": name_b,
            "coherence_a": verdict.coherence_a, "coherence_b": verdict.coherence_b,
            "winner": winner_name,
            "reasoning": verdict.reasoning,
        })

    winners = {r["winner"] for r in results}
    consistent = len(winners) == 1
    return {"results": results, "consistent": consistent, "agreed_winner": results[0]["winner"] if consistent else None}
