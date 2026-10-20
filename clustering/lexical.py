import warnings

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score, silhouette_samples

from clustering import NotEnoughDataException, prepare_text
from clustering.kmeans import find_best_k
from config import TARGET_EXPLAINED_VARIANCE


def vectorize_lexical(
    data: pd.Series,
    split_camel_case: bool = False,
    strip_punctuation: bool = False,
    lowercase: bool = False,
) -> np.ndarray:
    data = prepare_text(data)

    if split_camel_case:
        data = data.str.replace(r"([a-z])([A-Z])", r"\1 \2", regex=True)

    token_pattern = r"(?u)\b\w\w+\b" if strip_punctuation else r"(?u)\S+"
    vectorizer = CountVectorizer(lowercase=lowercase, token_pattern=token_pattern)
    try:
        vectors = vectorizer.fit_transform(data)
    except ValueError:
        # "empty vocabulary" -- e.g. a field that's present but blank on every
        # row. CountVectorizer raises rather than returning a 0-column matrix,
        # so the m < 3 check below never gets to run.
        raise NotEnoughDataException from None
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


def per_cluster_silhouette(vect: np.ndarray, labels: np.ndarray) -> dict:
    try:
        samples = silhouette_samples(vect, labels)
    except ValueError:
        return {label: None for label in np.unique(labels)}
    return {label: float(samples[labels == label].mean()) for label in np.unique(labels)}


def _score(vect: np.ndarray, labels: np.ndarray, k: int) -> float:
    # Silhouette + a parsimony term: prefers fewer clusters when quality is
    # comparable, since a smaller number of clusters is faster for a human to
    # review in the app. See notebooks/01_kmeans_shape_investigation.ipynb for search reasoning.
    return silhouette_score(vect, labels) + (1 - k / vect.shape[0])


def _converged(step: int, n_samples: int) -> bool:
    """True once the step size is fine enough, relative to dataset size, that
    further narrowing wouldn't meaningfully change the answer."""
    return (step < 15 and n_samples < 500) or (step < 100 and n_samples >= 500)


def find_best_kmeans_k(vect: np.ndarray) -> int:
    n = vect.shape[0]
    max_k = int(n * 0.8) if n > 200 else n
    min_k = 3 if n < 100 else 15
    step = 15 if n > 100 else 1
    step = n // 15 if n > 1000 else step

    return find_best_k(vect, _score, lambda step: _converged(step, n), min_k, max_k, step)
