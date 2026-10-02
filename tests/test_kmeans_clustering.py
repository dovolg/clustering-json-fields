import random

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import silhouette_score

from config import OUTLIER_Z_THRESHOLD
from clustering.kmeans import (
    NotEnoughDataException,
    _join_if_list,
    centroid_distances,
    find_best_kmeans_k,
    outlier_z_scores,
    per_cluster_silhouette,
    run_kmeans,
    vectorize_lexical,
)


# ---------------------------------------------------------------------------
# _join_if_list
# ---------------------------------------------------------------------------

def test_join_if_list_joins_lists():
    data = pd.Series([["python", "pandas"], ["java", "spring"]])
    result = _join_if_list(data)
    assert list(result) == ["python pandas", "java spring"]


def test_join_if_list_leaves_strings_unchanged():
    data = pd.Series(["hello world", "another title"])
    result = _join_if_list(data)
    assert list(result) == ["hello world", "another title"]


def test_join_if_list_handles_all_null_without_crashing():
    data = pd.Series([None, None])
    result = _join_if_list(data)
    assert result.isna().all()


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
# centroid_distances
# ---------------------------------------------------------------------------

def test_centroid_distances_matches_hand_computed_values():
    vect = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [10.0, 10.0],
        [10.0, 12.0],
    ])
    labels = np.array([0, 0, 1, 1])

    distances = centroid_distances(vect, labels)

    # cluster 0 centroid = (1, 0) -> both points are distance 1 away
    assert distances[0] == pytest.approx(1.0)
    assert distances[1] == pytest.approx(1.0)
    # cluster 1 centroid = (10, 11) -> both points are distance 1 away
    assert distances[2] == pytest.approx(1.0)
    assert distances[3] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# outlier_z_scores
# ---------------------------------------------------------------------------

def test_outlier_z_scores_flags_a_known_outlier():
    # For a sample of size n, a single point's z-score (population std) is
    # mathematically bounded by sqrt(n-1), approached but never reached as
    # that point's distance grows -- so the cluster needs to be large enough
    # that this ceiling comfortably clears OUTLIER_Z_THRESHOLD, no matter how
    # far the outlier is placed. n=30 gives sqrt(29) =~ 5.4, well above 3.
    rng = np.random.default_rng(0)
    tight = rng.normal(loc=0, scale=0.05, size=(29, 2))
    outlier = np.array([[50.0, 50.0]])
    vect = np.vstack([tight, outlier])
    labels = np.zeros(30, dtype=int)

    distances = centroid_distances(vect, labels)
    z_scores = outlier_z_scores(distances, labels)

    assert z_scores[-1] > OUTLIER_Z_THRESHOLD
    assert all(z_scores[:-1] < OUTLIER_Z_THRESHOLD)


def test_outlier_z_scores_handles_zero_variance_cluster():
    # All points identical -> zero std within the cluster; must not raise
    # or produce NaN, just report no outliers.
    vect = np.zeros((5, 3))
    labels = np.zeros(5, dtype=int)

    distances = centroid_distances(vect, labels)
    z_scores = outlier_z_scores(distances, labels)

    assert not np.isnan(z_scores).any()
    assert (z_scores == 0).all()


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


# ---------------------------------------------------------------------------
# run_kmeans
# ---------------------------------------------------------------------------

def test_run_kmeans_returns_requested_number_of_clusters():
    rng = np.random.default_rng(0)
    vect = np.vstack([
        rng.normal(loc=0, scale=0.1, size=(10, 2)),
        rng.normal(loc=10, scale=0.1, size=(10, 2)),
        rng.normal(loc=-10, scale=0.1, size=(10, 2)),
    ])

    labels = run_kmeans(vect, k=3)

    assert len(labels) == 30
    assert len(set(labels)) == 3
