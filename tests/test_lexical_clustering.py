import random

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import silhouette_score

from clustering import NotEnoughDataException
from clustering.kmeans import run_kmeans
from clustering.lexical import find_best_kmeans_k, per_cluster_silhouette, vectorize_lexical


# ---------------------------------------------------------------------------
# vectorize_lexical
# ---------------------------------------------------------------------------

def test_vectorize_lexical_raises_for_tiny_vocabulary():
    with pytest.raises(NotEnoughDataException):
        vectorize_lexical(pd.Series(["a", "b"]))


def test_vectorize_lexical_small_sample_does_not_crash():
    # Regression test: n_samples < n_components used to crash PCA/SVD before
    # max_components was bounded by row count, not just vocabulary size.
    data = pd.Series([
        ["python", "pandas", "dataframe"], ["python", "numpy", "dataframe"],
        ["java", "spring", "maven"], ["java", "spring-boot", "maven"],
        ["javascript", "react", "redux"], ["javascript", "react", "hooks"],
    ])
    vect = vectorize_lexical(data)
    assert vect.shape[0] == len(data)
    assert vect.shape[1] >= 1


def test_strip_punctuation_off_keeps_short_symbol_tokens():
    # "c#" has only one word-character -- strip_punctuation=True's token
    # pattern requires 2+, so it would vanish entirely, dropping the
    # vocabulary from 3 distinct tokens to 2. With strip_punctuation off
    # (the default), it's kept intact and the vocabulary stays at 3.
    data = pd.Series(["c#", "go", "js"])

    vectorize_lexical(data, strip_punctuation=False)  # should not raise

    with pytest.raises(NotEnoughDataException):
        vectorize_lexical(data, strip_punctuation=True)


def test_split_camel_case_increases_distinct_token_count():
    # Without splitting, "userIdentifier" and "userAccount" are 2 distinct
    # whole tokens. Splitting turns them into {"user", "identifier",
    # "account"} -- "user" becomes shared, so the vocabulary grows to 3.
    data = pd.Series(["userIdentifier", "userAccount"])

    with pytest.raises(NotEnoughDataException):
        vectorize_lexical(data, split_camel_case=False, lowercase=True)

    vectorize_lexical(data, split_camel_case=True, lowercase=True)  # should not raise


def test_vectorize_lexical_warns_when_target_not_reached():
    # With only 2 rows and max_components capped at 1, this data's first
    # component happens to explain 0% of the variance -- a real, observed
    # trigger for the fallback, not a contrived one.
    data = pd.Series(["userIdentifier", "userAccount"])

    with pytest.warns(UserWarning, match="Could not reach"):
        vectorize_lexical(data, split_camel_case=True, lowercase=True)


# ---------------------------------------------------------------------------
# per_cluster_silhouette
# ---------------------------------------------------------------------------

def test_per_cluster_silhouette_single_cluster_returns_none():
    vect = np.random.default_rng(0).normal(size=(10, 2))
    labels = np.zeros(10, dtype=int)

    result = per_cluster_silhouette(vect, labels)

    assert result == {0: None}


def test_per_cluster_silhouette_normal_case():
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=0, scale=0.1, size=(10, 2))
    cluster_b = rng.normal(loc=10, scale=0.1, size=(10, 2))
    vect = np.vstack([cluster_a, cluster_b])
    labels = np.array([0] * 10 + [1] * 10)

    result = per_cluster_silhouette(vect, labels)

    assert set(result.keys()) == {0, 1}
    assert result[0] > 0.9
    assert result[1] > 0.9


# ---------------------------------------------------------------------------
# find_best_kmeans_k -- the centerpiece: rather than validating each of the
# search's internal constants individually, check that its output is close
# to an independently-known-good answer on controlled synthetic data. If
# this holds, the internal constants don't need separate justification.
# ---------------------------------------------------------------------------

GROUP_VOCAB = {
    "python": ["python", "pandas", "numpy", "flask", "django", "dataframe", "array", "api", "asyncio", "pytest"],
    "java": ["java", "spring", "maven", "hibernate", "jpa", "junit", "mockito", "gradle", "tomcat", "servlet"],
    "javascript": ["javascript", "react", "redux", "node", "express", "vue", "vuex", "webpack", "babel", "jest"],
    "csharp": ["csharp", "dotnet", "xaml", "linq", "entity", "blazor", "razor", "nuget", "wpf", "asp"],
    "go": ["go", "goroutine", "channel", "gin", "mux", "docker", "kubernetes", "grpc", "etcd", "cobra"],
    "rust": ["rust", "cargo", "ownership", "tokio", "async", "serde", "json", "borrow", "trait", "wasm"],
}


def _synthetic_tag_data(rows_per_group=25, seed=0):
    rng = random.Random(seed)
    rows = []
    for _ in range(rows_per_group):
        for vocab in GROUP_VOCAB.values():
            rows.append(rng.sample(vocab, k=3))
    return pd.Series(rows)


def test_find_best_kmeans_k_is_close_to_a_known_good_value():
    data = _synthetic_tag_data()
    vect = vectorize_lexical(data)

    found_k = find_best_kmeans_k(vect)
    found_labels = run_kmeans(vect, found_k)
    found_silhouette = silhouette_score(vect, found_labels)

    true_group_count = len(GROUP_VOCAB)
    true_labels = run_kmeans(vect, true_group_count)
    true_silhouette = silhouette_score(vect, true_labels)

    # The found k shouldn't be meaningfully worse than clustering by the
    # data's actual known group structure -- it's allowed to do better
    # (e.g. by finding real sub-topics within a group), just not worse.
    assert found_silhouette >= true_silhouette * 0.8

    # Sanity bound: not degenerate (e.g. collapsing to near-duplicate
    # singleton clusters, or failing to split the data at all).
    assert 3 <= found_k <= 10 * true_group_count
