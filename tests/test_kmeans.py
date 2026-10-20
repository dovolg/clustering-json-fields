import numpy as np
import pytest

from config import OUTLIER_Z_THRESHOLD
from clustering.kmeans import centroid_distances, outlier_z_scores, run_kmeans


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


# find_best_k (the shared, parametrized k-search) isn't tested directly here --
# it's exercised by two independent real callers with different scoring
# functions instead: tests/test_lexical_clustering.py's
# test_find_best_kmeans_k_is_close_to_a_known_good_value (silhouette) and
# tests/test_semantic_clustering.py's
# test_find_best_semantic_k_is_close_to_a_known_good_value (c_v). Both
# exercising the same recursion/convergence/memoization logic successfully,
# under different constants and score functions, is stronger evidence for the
# shared engine than a third synthetic-scorer test would add.
