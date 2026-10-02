import warnings

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

from config import (
    COARSE_STEP_THRESHOLD,
    SCORE_DECAY_STOP_RATIO,
    TARGET_EXPLAINED_VARIANCE,
)


class NotEnoughDataException(Exception):
    pass


def _join_if_list(data: pd.Series) -> pd.Series:
    non_null = data.dropna()
    if non_null.empty or not isinstance(non_null.iloc[0], list):
        return data
    return data.apply(lambda v: " ".join(v) if isinstance(v, list) else v)


def vectorize_lexical(
    data: pd.Series,
    split_camel_case: bool = False,
    strip_punctuation: bool = False,
    lowercase: bool = False,
) -> np.ndarray:
    data = _join_if_list(data)

    if split_camel_case:
        data = data.str.replace(r"([a-z])([A-Z])", r"\1 \2", regex=True)

    token_pattern = r"(?u)\b\w\w+\b" if strip_punctuation else r"(?u)\S+"
    vectorizer = CountVectorizer(lowercase=lowercase, token_pattern=token_pattern)
    vectors = vectorizer.fit_transform(data)
    m = vectors.shape[1]
    if m < 3:
        raise NotEnoughDataException

    max_components = max(1, min(m, vectors.shape[0]) - 1)

    # Step is deliberately not scaled up for large vocabularies: notebooks/02 found the
    # quality-optimal n_components sits in a modest, low range (~20-30) regardless of
    # vocabulary size in the thousands.
    start = min(2 if m < 10 else 10, max_components)
    step = 10
    step = 1 if start < 10 else step

    result = None
    for i in range(start, max_components, step):
        svd = TruncatedSVD(n_components=i, random_state=42)
        result = svd.fit_transform(vectors)
        if svd.explained_variance_ratio_.sum() >= TARGET_EXPLAINED_VARIANCE:
            break

    if result is None:
        # range(start, max_components, step) was empty (start >= max_components,
        # e.g. a very small sample) -- fit once directly rather than not at all.
        svd = TruncatedSVD(n_components=start, random_state=42)
        result = svd.fit_transform(vectors)

    reached_target = svd.explained_variance_ratio_.sum() >= TARGET_EXPLAINED_VARIANCE

    if not reached_target:
        warnings.warn(
            f"Could not reach {TARGET_EXPLAINED_VARIANCE:.0%} explained variance within "
            f"{max_components} components (reached {svd.explained_variance_ratio_.sum():.0%}); "
            f"using n_components={svd.n_components}.",
            stacklevel=2,
        )

    return result


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


def per_cluster_silhouette(vect: np.ndarray, labels: np.ndarray) -> dict:
    try:
        samples = silhouette_samples(vect, labels)
    except ValueError:
        return {label: None for label in np.unique(labels)}
    return {label: float(samples[labels == label].mean()) for label in np.unique(labels)}


def _search_converged(step: int, n_samples: int) -> bool:
    """True once the step size is fine enough, relative to dataset size, that
    further narrowing wouldn't meaningfully change the answer."""
    return (step < 15 and n_samples < 500) or (step < 100 and n_samples >= 500)


def _search_kmeans_k(vect, best_k, best_custom_score, min_k, max_k, step, score_cache):
    # score_cache memoizes k -> custom_score across recursive calls, so a
    # re-narrowed range doesn't redundantly re-fit a k already tried.
    k_values = range(min_k, max_k, step)

    for k in k_values:
        if k in score_cache:
            continue

        km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=100, random_state=42)
        km.fit(vect)

        try:
            # Silhouette + a parsimony term: prefers fewer clusters when quality is
            # comparable, since a smaller number of clusters is faster for a human to
            # review in the app. See notebooks/01_kmeans_shape_investigation.ipynb for search reasoning.
            new_score = silhouette_score(vect, km.labels_) + (1 - k / vect.shape[0])
            score_cache[k] = new_score

            if new_score > best_custom_score:
                best_k = k
                best_custom_score = new_score

            if new_score < best_custom_score * SCORE_DECAY_STOP_RATIO and step >= COARSE_STEP_THRESHOLD:
                break
        except ValueError:
            score_cache[k] = -1

    if _search_converged(step, vect.shape[0]):
        return best_k
    else:
        return _search_kmeans_k(
            vect, best_k, best_custom_score,
            max(best_k - step, 1), min(best_k + step, vect.shape[0] - 1), step // 5,
            score_cache,
        )


def find_best_kmeans_k(vect: np.ndarray) -> int:
    """Searches for a good k via coarse-to-fine recursive narrowing around
    the best score found so far (see _search_kmeans_k). Trusted via a test
    that checks its output against an independently-known-good answer on
    synthetic data, rather than by individually justifying every constant
    below -- see tests/test_kmeans_clustering.py and notebooks/01_....
    """
    max_k = int(vect.shape[0] * 0.8) if vect.shape[0] > 200 else vect.shape[0]
    min_k = 3 if vect.shape[0] < 100 else 15
    step = 15 if vect.shape[0] > 100 else 1
    step = vect.shape[0] // 15 if vect.shape[0] > 1000 else step

    best_k = _search_kmeans_k(vect, None, -1, min_k, max_k, step, {})
    return vect.shape[1] if best_k is None else best_k


def run_kmeans(vect: np.ndarray, k: int) -> np.ndarray:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=100, random_state=42)
    return km.fit_predict(vect)
