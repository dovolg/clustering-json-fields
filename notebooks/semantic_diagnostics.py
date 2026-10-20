"""Standalone diagnostics shared across the semantic-clustering investigation
notebooks. Deliberately independent of production code -- no semantic
clustering module exists yet.
"""
import re
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import umap
import hdbscan


def embed(texts, model):
    """Sentence-transformers embeddings, L2-normalized so Euclidean distance
    on them is monotonic with cosine distance -- the standard choice for
    sentence embeddings, and what makes one fixed scoring space (normalized
    embeddings) valid for comparing every representation/algorithm below."""
    vectors = model.encode(list(texts), show_progress_bar=False)
    return normalize(vectors)


def build_representations(embeddings, n_components=15, random_state=42):
    """Every representation clustering is actually run on, at a fixed
    dimensionality so the comparison isn't re-confounded by also varying
    n_components per representation. 'raw' isolates whether reduction helps
    at all; 'pca' isolates linear vs non-linear reduction; the three UMAP
    variants isolate UMAP's own params -- its defaults (min_dist=0.1,
    n_neighbors=15) are tuned for visualization, not clustering; the
    established practice for feeding a density-based clusterer is a lower
    min_dist so genuinely similar points pack tightly instead of staying
    visually spread out."""
    reps = {"raw (no reduction)": embeddings}

    pca = PCA(n_components=n_components, random_state=random_state)
    reps["pca (linear)"] = pca.fit_transform(embeddings)

    reps["umap nn=15 min_dist=0.1 (default)"] = umap.UMAP(
        n_components=n_components, n_neighbors=15, min_dist=0.1, random_state=random_state
    ).fit_transform(embeddings)

    reps["umap nn=15 min_dist=0.0 (clustering-tuned)"] = umap.UMAP(
        n_components=n_components, n_neighbors=15, min_dist=0.0, random_state=random_state
    ).fit_transform(embeddings)

    reps["umap nn=30 min_dist=0.0"] = umap.UMAP(
        n_components=n_components, n_neighbors=30, min_dist=0.0, random_state=random_state
    ).fit_transform(embeddings)

    return reps


def best_kmeans(representation, original_embeddings, texts, k_values, random_state=42):
    """Best-by-c_v KMeans fit on `representation` -- c_v is the selected
    metric (see notebook 4); silhouette is computed too and reported
    alongside for visibility, not used to pick the winner."""
    best = None
    n = representation.shape[0]
    for k in k_values:
        if k < 2 or k >= n:
            continue
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=100, random_state=random_state)
        labels = km.fit_predict(representation)
        try:
            sil = silhouette_score(original_embeddings, labels)
        except ValueError:
            continue
        c_v = coherence_scores(texts, labels, top_n=10)["c_v"]
        if best is None or c_v > best["c_v"]:
            best = {
                "param": f"k={k}", "n_clusters": k, "noise_fraction": 0.0,
                "c_v": round(float(c_v), 4), "silhouette": round(float(sil), 4),
            }
    return best


def best_hdbscan(representation, original_embeddings, texts, min_cluster_size_values):
    """Best-by-c_v HDBSCAN fit on `representation` (noise points excluded
    from both c_v and silhouette, same treatment). Reports every candidate,
    not just the winner, since noise_fraction at the winning score matters
    as much as the score itself."""
    rows = []
    for mcs in min_cluster_size_values:
        labels = hdbscan.HDBSCAN(min_cluster_size=mcs).fit_predict(representation)
        noise_mask = labels == -1
        noise_fraction = float(noise_mask.mean())
        n_clusters = len(set(labels)) - (1 if noise_mask.any() else 0)
        sil, c_v = None, None
        if n_clusters >= 2 and (~noise_mask).sum() > n_clusters:
            sil = silhouette_score(original_embeddings[~noise_mask], labels[~noise_mask])
            c_v = coherence_scores(texts[~noise_mask].reset_index(drop=True), labels[~noise_mask], top_n=10)["c_v"]
        rows.append({
            "param": f"min_cluster_size={mcs}",
            "n_clusters": n_clusters,
            "noise_fraction": round(noise_fraction, 3),
            "c_v": round(float(c_v), 4) if c_v is not None else None,
            "silhouette": round(float(sil), 4) if sil is not None else None,
        })
    scored = [r for r in rows if r["c_v"] is not None]
    best = max(scored, key=lambda r: r["c_v"]) if scored else None
    return best, pd.DataFrame(rows)


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def class_tfidf_top_words(texts, labels, top_n=10):
    """BERTopic-style class-based TF-IDF: treat each cluster's concatenated
    members as one pseudo-document, TF-IDF across the pseudo-document corpus
    to surface words distinctively frequent within a cluster relative to the
    others, not just frequent overall. Deterministic, no model needed."""
    texts = pd.Series(texts).reset_index(drop=True)
    labels = np.asarray(labels)
    unique_labels = sorted(set(labels))
    pseudo_docs = [" ".join(texts[labels == label]) for label in unique_labels]

    vectorizer = TfidfVectorizer(stop_words="english", token_pattern=r"[a-z0-9]+", lowercase=True)
    matrix = vectorizer.fit_transform(pseudo_docs)
    vocab = np.array(vectorizer.get_feature_names_out())

    topics = {}
    for i, label in enumerate(unique_labels):
        row = matrix[i].toarray().ravel()
        top_idx = row.argsort()[::-1][:top_n]
        topics[label] = [vocab[j] for j in top_idx if row[j] > 0]
    return topics


_CV_SCORE_DECAY_STOP_RATIO = 0.8
_CV_COARSE_STEP_THRESHOLD = 100


def _cv_search_converged(step):
    # Deliberately NOT the lexical search's step<100 threshold -- that assumes
    # large initial steps (hundreds) narrowing over several rounds. Once max_k
    # is capped to ~150 (see find_best_cv_k), an initial step sized for ~10
    # candidates is already below 100, which would stop the search after one
    # coarse pass with no real narrowing. step<=2 ensures genuine narrowing.
    return step <= 2


def _search_cv_k(texts, vect, best_k, best_score, min_k, max_k, step, score_cache, fit_counter=None):
    for k in range(min_k, max_k, step):
        if k in score_cache:
            continue
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=100, random_state=42)
        labels = km.fit_predict(vect)
        if fit_counter is not None:
            fit_counter[0] += 1
        try:
            score = coherence_scores(texts, labels, top_n=10)["c_v"]
            score_cache[k] = score
            if best_score is None or score > best_score:
                best_k, best_score = k, score
            if (best_score is not None and score < best_score * _CV_SCORE_DECAY_STOP_RATIO
                    and step >= _CV_COARSE_STEP_THRESHOLD):
                break
        except Exception:
            score_cache[k] = -1

    if _cv_search_converged(step):
        return best_k
    return _search_cv_k(
        texts, vect, best_k, best_score,
        max(best_k - step, 1), min(best_k + step, vect.shape[0] - 1), max(step // 5, 1),
        score_cache, fit_counter,
    )


def find_best_cv_k(texts, vect, min_k=8, max_k_cap=150):
    """The semantic pipeline's k-search (see notebook 6): same recursive
    coarse-to-fine strategy as the lexical path's find_best_kmeans_k, scored
    by c_v instead of silhouette, with three tuned differences justified
    there -- max_k capped well below the lexical 0.8*n formula (c_v's
    evaluation cost scales badly with cluster count: confirmed k=50->0.8s,
    k=100->1.3s, k=200->3.0s), the convergence threshold rescaled for that
    smaller range, and min_k lowered to 8 (vs the lexical path's 15) since
    c_v's real optimum on structured data sits below 15."""
    n = vect.shape[0]
    max_k = min(int(n * 0.8) if n > 200 else n, max_k_cap)
    step = max(1, (max_k - min_k) // 10)
    fit_counter = [0]
    best_k = _search_cv_k(texts, vect, None, None, min_k, max_k, step, {}, fit_counter)
    return best_k, fit_counter[0]


def coherence_scores(texts, labels, top_n=10):
    """c_v and u_mass topic coherence of a clustering's c-TF-IDF top words,
    plus how long each took -- two different formulas (c_v: sliding-window
    NPMI + cosine similarity against the reference texts; u_mass: document
    co-occurrence counts within the same corpus, no external reference
    needed), reported side by side rather than assumed equivalent."""
    tokenized = [_tokenize(t) for t in texts]
    dictionary = Dictionary(tokenized)
    corpus = [dictionary.doc2bow(t) for t in tokenized]

    topics_by_label = class_tfidf_top_words(texts, labels, top_n=top_n)
    topics = [words for words in topics_by_label.values() if words]

    t0 = time.perf_counter()
    c_v = CoherenceModel(topics=topics, texts=tokenized, dictionary=dictionary, coherence="c_v").get_coherence()
    c_v_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    u_mass = CoherenceModel(topics=topics, corpus=corpus, dictionary=dictionary, coherence="u_mass").get_coherence()
    u_mass_time = time.perf_counter() - t0

    return {"c_v": c_v, "c_v_time": c_v_time, "u_mass": u_mass, "u_mass_time": u_mass_time}
