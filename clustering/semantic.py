"""Semantic clustering path: sentence embeddings + PCA + KMeans, scored by
c_v topic coherence rather than silhouette.

Settled by notebooks/04-07 (notebooks/04_semantic_metric_selection.ipynb through
notebooks/07_semantic_config.ipynb) and validated end to end against the lexical
pipeline in notebooks/08_lexical_vs_semantic_comparison.ipynb: semantic clustering
wins clearly on free text, lexical wins clearly on categorical fields.

- metric: c_v (gensim coherence over class-based TF-IDF top words per
  cluster), not silhouette -- on real, diffuse text, silhouette collapses to
  a degenerate k=2 answer in a way c_v doesn't (notebooks/04).
- architecture: KMeans on PCA-reduced embeddings, not UMAP+HDBSCAN --
  HDBSCAN's real c_v edge, once its own degenerate near-total-noise results
  are set aside, is under 1%, bought by excluding roughly half the data
  (notebooks/05).
- embedding model: all-MiniLM-L6-v2 -- faster AND higher-scoring than
  all-mpnet-base-v2 once both are measured at the settled n_components
  (notebooks/07).
- reduction: PCA, not SVD -- centering is the textbook-correct choice for
  dense sentence embeddings (SVD's centering-avoidance only matters for the
  lexical path's sparse count data), n_components=20 (notebooks/07).
- k-search: the same recursive coarse-to-fine strategy as the lexical path's
  find_best_kmeans_k, scored by c_v, with three differences justified in
  notebooks/06: max_k capped well below the lexical 0.8*n formula (c_v's
  evaluation cost scales badly with cluster count, unlike silhouette), the
  convergence threshold rescaled for that smaller range, and min_k lowered
  (c_v's real optimum on structured data sits below the lexical path's
  min_k=15).
"""
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

from clustering import NotEnoughDataException, prepare_text
from clustering.kmeans import find_best_k
from config import (
    SEMANTIC_COHERENCE_TOP_N,
    SEMANTIC_EMBEDDING_MODEL,
    SEMANTIC_MAX_K_CAP,
    SEMANTIC_MIN_AVG_CLUSTER_SIZE,
    SEMANTIC_MIN_K,
    SEMANTIC_N_COMPONENTS,
)


class _ModelCache:
    """Lazily loads each named SentenceTransformer once and reuses it for the
    rest of the process's lifetime -- loading weights from disk per call
    would dominate runtime otherwise."""

    def __init__(self):
        self._models = {}

    def get(self, name: str) -> SentenceTransformer:
        if name not in self._models:
            self._models[name] = SentenceTransformer(name)
        return self._models[name]


_model_cache = _ModelCache()


def embed_semantic(data: pd.Series) -> np.ndarray:
    """Sentence embeddings for the configured model, L2-normalized so
    Euclidean distance on them (what KMeans and PCA both use) is monotonic
    with cosine distance -- the standard choice for sentence embeddings."""
    data = prepare_text(data)
    if len(data) < 2:
        raise NotEnoughDataException
    model = _model_cache.get(SEMANTIC_EMBEDDING_MODEL)
    vectors = model.encode(list(data), show_progress_bar=False)
    return normalize(vectors)


def reduce_semantic(embeddings: np.ndarray) -> np.ndarray:
    # PCA's rank is bounded by n_samples-1 after centering -- mirrors
    # vectorize_lexical's same bound for its own reduction step.
    n_components = min(SEMANTIC_N_COMPONENTS, embeddings.shape[1], embeddings.shape[0] - 1)
    if n_components < 1:
        raise NotEnoughDataException
    return PCA(n_components=n_components, random_state=42).fit_transform(embeddings)


def vectorize_semantic(data: pd.Series) -> tuple:
    """Embed + reduce in one call, returning the (already list-joined) texts
    alongside the reduced vectors -- the texts are needed again by
    find_best_semantic_k and per_cluster_coherence, which score clusters
    from actual member words, not just vector geometry (unlike the lexical
    path, where vectorize_lexical's output is all find_best_kmeans_k needs)."""
    texts = prepare_text(data)
    vect = reduce_semantic(embed_semantic(texts))
    return texts, vect


def _tokenize(text) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _class_tfidf_top_words(texts: pd.Series, labels: np.ndarray, top_n: int) -> list[tuple]:
    """BERTopic-style class-based TF-IDF: treat each cluster's concatenated
    members as one pseudo-document, TF-IDF across the pseudo-document corpus
    to surface words distinctively frequent within a cluster relative to the
    others, not just frequent overall. Returns (label, words) pairs, one per
    cluster whose top words are non-empty -- a cluster with no vocabulary
    left after stopword removal contributes nothing to coherence scoring and
    is dropped rather than passed through as an empty topic."""
    texts = pd.Series(texts).reset_index(drop=True)
    labels = np.asarray(labels)
    unique_labels = sorted(set(labels))
    pseudo_docs = [" ".join(str(t) for t in texts[labels == label]) for label in unique_labels]

    vectorizer = TfidfVectorizer(stop_words="english", token_pattern=r"[a-z0-9]+", lowercase=True)
    try:
        matrix = vectorizer.fit_transform(pseudo_docs)
    except ValueError:
        # Every pseudo-document was empty after stopword removal (e.g. every
        # cluster's text is itself just stopwords) -- TfidfVectorizer raises
        # rather than returning an empty vocabulary. No topics either way.
        return []
    vocab = np.array(vectorizer.get_feature_names_out())

    result = []
    for i, label in enumerate(unique_labels):
        row = matrix[i].toarray().ravel()
        top_idx = row.argsort()[::-1][:top_n]
        words = [vocab[j] for j in top_idx if row[j] > 0]
        if words:
            result.append((label, words))
    return result


def coherence_score(texts: pd.Series, labels: np.ndarray, top_n: int = SEMANTIC_COHERENCE_TOP_N) -> float:
    """Mean c_v topic coherence across clusters -- the metric the semantic
    k-search optimizes (see module docstring)."""
    label_topics = _class_tfidf_top_words(texts, labels, top_n)
    if not label_topics:
        return -1.0

    tokenized = [_tokenize(t) for t in texts]
    dictionary = Dictionary(tokenized)
    topics = [words for _, words in label_topics]
    model = CoherenceModel(topics=topics, texts=tokenized, dictionary=dictionary, coherence="c_v")
    return model.get_coherence()


def per_cluster_coherence(texts: pd.Series, labels: np.ndarray, top_n: int = SEMANTIC_COHERENCE_TOP_N) -> dict:
    """Per-cluster c_v, for display/ordering in place of per_cluster_silhouette
    -- consistent with c_v being the metric this pipeline actually trusts."""
    all_labels = sorted(set(np.asarray(labels)))
    label_topics = _class_tfidf_top_words(texts, labels, top_n)
    if not label_topics:
        return {label: None for label in all_labels}

    tokenized = [_tokenize(t) for t in texts]
    dictionary = Dictionary(tokenized)
    topic_labels = [label for label, _ in label_topics]
    topics = [words for _, words in label_topics]
    model = CoherenceModel(topics=topics, texts=tokenized, dictionary=dictionary, coherence="c_v")

    # strict=False: if gensim ever returns fewer scores than topics, the
    # unmatched labels fall through to the setdefault below and display as
    # "n/a" -- the same graceful path an empty-vocabulary cluster already
    # takes, rather than crashing a finished clustering run.
    scores = dict(zip(topic_labels, (float(s) for s in model.get_coherence_per_topic()), strict=False))
    for label in all_labels:
        scores.setdefault(label, None)
    return scores


def _converged(step: int) -> bool:
    # Deliberately not the lexical search's step<100-ish threshold -- that
    # assumes large initial steps (hundreds) narrowing over multiple rounds.
    # Once max_k is capped to SEMANTIC_MAX_K_CAP, an initial step sized for
    # ~10 candidates is already below 100, which would stop the search after
    # one coarse pass with no real narrowing. step<=2 ensures genuine
    # narrowing still happens (see notebooks/06_semantic_search_shape.ipynb).
    return step <= 2


def find_best_semantic_k(texts: pd.Series, vect: np.ndarray) -> int:
    """Same recursive coarse-to-fine search as find_best_kmeans_k (see
    clustering/kmeans.py's find_best_k), scored by c_v instead of
    silhouette. The score function closes over `texts` rather than taking
    the vectors KMeans was fit on, because c_v is computed from cluster
    members' actual words -- unlike the lexical search, which only ever
    needs the vectors. See notebooks/06_semantic_search_shape.ipynb for why
    min_k/max_k/the convergence threshold differ from the lexical path's
    constants.
    """
    n = vect.shape[0]
    # SEMANTIC_MIN_AVG_CLUSTER_SIZE guards against c_v's small-cluster reward (see its
    # definition in config.py) -- without it, max_k would equal n for any n<=200,
    # mirroring the lexical formula's assumption that exhaustive small-n search is safe,
    # which doesn't hold for this metric.
    max_k = min(
        int(n * 0.8) if n > 200 else n,
        SEMANTIC_MAX_K_CAP,
        max(n // SEMANTIC_MIN_AVG_CLUSTER_SIZE, 1),
    )
    # SEMANTIC_MIN_K reflects the >=100-sample regime notebooks/04-06 used (known-structure
    # n=418, diffuse n=5000); below that, the lexical path's own small-n floor applies.
    min_k = 3 if n < 100 else SEMANTIC_MIN_K
    min_k = min(min_k, max(max_k - 1, 1))
    step = max(1, (max_k - min_k) // 10)

    def score_fn(vect, labels, k):
        return coherence_score(texts, labels)

    return find_best_k(vect, score_fn, _converged, min_k, max_k, step)
