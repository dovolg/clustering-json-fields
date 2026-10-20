"""The KMeans engine shared by both clustering backends: fitting, measuring
each point's fit to its assigned cluster, flagging outliers from that fit,
and searching for a good k. None of this depends on how the input vectors
were produced or on which metric scores a candidate k -- clustering/lexical.py
and clustering/semantic.py each supply their own vectorization and scoring
function, but search the same way.
"""
import numpy as np
from sklearn.cluster import KMeans

from config import COARSE_STEP_THRESHOLD, SCORE_DECAY_STOP_RATIO


def run_kmeans(vect: np.ndarray, k: int) -> np.ndarray:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=100, random_state=42)
    return km.fit_predict(vect)


def centroid_distances(vect: np.ndarray, labels: np.ndarray) -> np.ndarray:
    distances = np.zeros(len(labels))
    for label in np.unique(labels):
        mask = labels == label
        centroid = vect[mask].mean(axis=0)
        distances[mask] = np.linalg.norm(vect[mask] - centroid, axis=1)
    return distances


def outlier_z_scores(distances: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-point z-score of its centroid distance, computed within its own
    cluster's own distribution -- flags points that are a meaningfully worse
    fit for their assigned cluster than their clustermates, even when the
    cluster's own average quality (e.g. mean silhouette) looks fine."""
    z_scores = np.zeros(len(labels), dtype=float)
    for label in np.unique(labels):
        mask = labels == label
        d = distances[mask]
        if d.std() > 0:
            z_scores[mask] = (d - d.mean()) / d.std()
    return z_scores


def _search_k(vect, score_fn, converged, best_k, best_score, min_k, max_k, step, score_cache):
    # score_cache memoizes k -> score across recursive calls, so a
    # re-narrowed range doesn't redundantly re-fit a k already tried.
    for k in range(min_k, max_k, step):
        if k in score_cache:
            continue

        km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=100, random_state=42)
        km.fit(vect)

        try:
            score = score_fn(vect, km.labels_, k)
            score_cache[k] = score

            if best_score is None or score > best_score:
                best_k, best_score = k, score

            if best_score is not None and score < best_score * SCORE_DECAY_STOP_RATIO and step >= COARSE_STEP_THRESHOLD:
                break
        except ValueError:
            score_cache[k] = -1

    if converged(step):
        return best_k
    return _search_k(
        vect, score_fn, converged, best_k, best_score,
        max(best_k - step, 1), min(best_k + step, vect.shape[0] - 1), max(step // 5, 1),
        score_cache,
    )


def find_best_k(vect: np.ndarray, score_fn, converged, min_k: int, max_k: int, step: int) -> int:
    """Coarse-to-fine recursive narrowing around the best score found so
    far. `score_fn(vect, labels, k) -> float` scores one candidate k;
    `converged(step) -> bool` decides when narrowing has gone far enough to
    stop -- both backend-specific, since silhouette and c_v disagree on
    what counts as converged (see clustering/lexical.py and
    clustering/semantic.py for each one's reasoning).

    Trusted via each backend's own test that checks its output against an
    independently-known-good answer on data with real structure, rather than
    by individually justifying every constant here -- see
    tests/test_lexical_clustering.py, tests/test_semantic_clustering.py, and
    notebooks/01_kmeans_shape_investigation.ipynb.
    """
    best_k = _search_k(vect, score_fn, converged, None, None, min_k, max_k, step, {})
    return vect.shape[1] if best_k is None else best_k
