"""Standalone diagnostics for the KMeans lexical-clustering foundation.

Deliberately independent of clustering/kmeans.py and app.py: this notebook
investigates the constants and assumptions those modules are built on, so
it must not inherit vectorization/reduction choices from the very heuristic
under review. Any overlapping logic (e.g. join-field aggregation) is
reimplemented here standalone rather than imported.
"""
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples


def aggregate_by_user(data: pd.DataFrame, field: str, join_field: str = "user_id") -> pd.Series:
    """Combine each group's field values into one flat list of items."""
    def combine(values):
        combined = []
        for v in values:
            if isinstance(v, list):
                combined.extend(v)
            else:
                combined.append(v)
        return combined

    return data.groupby(join_field)[field].apply(combine)


def vectorize(values: pd.Series, method: str, n_components: int) -> np.ndarray:
    """Vectorize + reduce with an explicit, caller-chosen n_components.

    method: "count" or "tfidf". token_pattern keeps punctuation-bearing
    tokens intact (matches the prod default of strip_punctuation=False).
    """
    if values.map(lambda v: isinstance(v, list)).any():
        values = values.apply(lambda v: " ".join(v) if isinstance(v, list) else v)

    vectorizer_cls = TfidfVectorizer if method == "tfidf" else CountVectorizer
    vectorizer = vectorizer_cls(token_pattern=r"(?u)\S+")
    counts = vectorizer.fit_transform(values)

    n_components = max(1, min(n_components, counts.shape[0] - 1, counts.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    return svd.fit_transform(counts), counts


def explained_variance_curve(counts, max_components: int, step: int = 5) -> pd.DataFrame:
    """Cumulative explained variance vs n_components, for the vectorization
    constants question (how many components before variance saturates)."""
    max_components = min(max_components, counts.shape[0] - 1, counts.shape[1] - 1)
    rows = []
    for n in range(1, max_components, step):
        svd = TruncatedSVD(n_components=n, random_state=42)
        svd.fit(counts)
        rows.append({"n_components": n, "cumulative_variance": svd.explained_variance_ratio_.sum()})
    return pd.DataFrame(rows)


def sweep_k(vect: np.ndarray, k_values, random_state: int = 42) -> pd.DataFrame:
    """Fit KMeans at each k, recording pure silhouette and the current prod
    heuristic (silhouette + (1 - k/n)) side by side."""
    n = vect.shape[0]
    rows = []
    for k in k_values:
        if k < 2 or k >= n:
            continue
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=100, random_state=random_state)
        labels = km.fit_predict(vect)
        try:
            sil = silhouette_score(vect, labels)
        except ValueError:
            continue
        rows.append({"k": k, "silhouette": sil, "heuristic": sil + (1 - k / n)})
    return pd.DataFrame(rows)


def count_local_maxima(arr) -> int:
    """Count strict local maxima in a 1D sequence (excludes the endpoints)."""
    arr = np.asarray(arr)
    return int(np.sum((arr[1:-1] > arr[:-2]) & (arr[1:-1] > arr[2:])))


def smooth_curve_shape(k_values, y_values, s_multipliers=(0.5, 1, 2, 5, 10, 20)) -> dict:
    """Characterize a k -> score curve's shape with two rigorous (not just
    eyeballed) checks:

    - How much variance a plain two-parameter parabola explains -- a direct
      test of "is this basically one concave trend", since a parabola can
      only ever have one peak or one valley.
    - How much smoothing (as a fraction of the raw curve's own variance) a
      cubic smoothing spline needs before it collapses to a single true
      peak. Little smoothing needed => the multi-peak raw curve was mostly
      noise around one real trend. A lot needed => there isn't a strong
      single trend to begin with.
    """
    from scipy.interpolate import UnivariateSpline

    k = np.asarray(k_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    ss_tot = np.sum((y - y.mean()) ** 2)

    coeffs2 = np.polyfit(k, y, deg=2)
    fit2 = np.polyval(coeffs2, k)
    parabola_r2 = 1 - np.sum((y - fit2) ** 2) / ss_tot

    n = len(k)
    var_est = np.var(y)
    spline_trace = []
    unimodal_at = None
    for s_mult in s_multipliers:
        s_val = s_mult * n * var_est * 0.05
        spline = UnivariateSpline(k, y, s=s_val, k=3)
        fitted = spline(k)
        peaks = count_local_maxima(fitted)
        r2 = 1 - np.sum((y - fitted) ** 2) / ss_tot
        spline_trace.append({"s_mult": s_mult, "peaks": peaks, "r2": round(float(r2), 3)})
        if peaks <= 1 and unimodal_at is None:
            unimodal_at = {"s_mult": s_mult, "r2": round(float(r2), 3)}

    return {
        "parabola_r2": round(float(parabola_r2), 3),
        "spline_trace": spline_trace,
        "unimodal_at": unimodal_at,
    }


def cluster_quality_summary(vect: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Per-cluster size and mean silhouette, for flagging small/weak
    clusters worth inspecting as potential outliers."""
    try:
        samples = silhouette_samples(vect, labels)
    except ValueError:
        samples = np.full(len(labels), np.nan)
    rows = []
    for label in sorted(np.unique(labels)):
        mask = labels == label
        rows.append({
            "cluster": int(label),
            "size": int(mask.sum()),
            "mean_silhouette": round(float(np.nanmean(samples[mask])), 3),
        })
    return pd.DataFrame(rows).sort_values("mean_silhouette").reset_index(drop=True)


def dbscan_eps_sweep(vect: np.ndarray, eps_values, min_samples: int = 3) -> pd.DataFrame:
    """Sweep DBSCAN's eps to see how cluster count and noise fraction
    respond -- a quick way to pick a reasonable eps without guessing once,
    and to see whether DBSCAN finds a stable regime at all."""
    rows = []
    for eps in eps_values:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(vect)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_frac = float(np.mean(labels == -1))
        rows.append({"eps": eps, "n_clusters": n_clusters, "noise_fraction": round(noise_frac, 3)})
    return pd.DataFrame(rows)
