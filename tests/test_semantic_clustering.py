import json
import os

import numpy as np
import pandas as pd
import pytest

from clustering import NotEnoughDataException
from clustering.kmeans import run_kmeans
from clustering.semantic import (
    _class_tfidf_top_words,
    _tokenize,
    coherence_score,
    embed_semantic,
    find_best_semantic_k,
    per_cluster_coherence,
    reduce_semantic,
)


# ---------------------------------------------------------------------------
# embed_semantic
# ---------------------------------------------------------------------------

def test_embed_semantic_raises_for_too_few_rows():
    with pytest.raises(NotEnoughDataException):
        embed_semantic(pd.Series(["only one row"]))


def test_embed_semantic_returns_l2_normalized_vectors():
    vectors = embed_semantic(pd.Series(["python pandas tutorial", "java spring boot"]))
    norms = np.linalg.norm(vectors, axis=1)
    assert norms == pytest.approx(1.0, abs=1e-5)


def test_embed_semantic_joins_list_fields():
    list_data = pd.Series([["python", "pandas"], ["java", "spring"]])
    joined_data = pd.Series(["python pandas", "java spring"])

    vect_from_list = embed_semantic(list_data)
    vect_from_joined = embed_semantic(joined_data)

    assert vect_from_list == pytest.approx(vect_from_joined)


# ---------------------------------------------------------------------------
# reduce_semantic
# ---------------------------------------------------------------------------

def test_reduce_semantic_caps_components_at_n_samples_minus_one():
    # 5 samples, 384-dim embeddings -- SEMANTIC_N_COMPONENTS (20) exceeds
    # n_samples-1, so the output must fall back to that smaller bound rather
    # than raise or silently return a different shape.
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(5, 384))

    reduced = reduce_semantic(embeddings)

    assert reduced.shape == (5, 4)


def test_reduce_semantic_raises_for_a_single_row():
    embeddings = np.zeros((1, 384))
    with pytest.raises(NotEnoughDataException):
        reduce_semantic(embeddings)


# ---------------------------------------------------------------------------
# _tokenize / _class_tfidf_top_words
# ---------------------------------------------------------------------------

def test_tokenize_lowercases_and_strips_punctuation():
    assert _tokenize("Python 3.12, Pandas!") == ["python", "3", "12", "pandas"]


def test_class_tfidf_top_words_separates_distinct_clusters():
    texts = pd.Series([
        "python pandas dataframe", "python numpy array", "python flask api",
        "java spring boot", "java maven dependency", "java hibernate jpa",
    ])
    labels = np.array([0, 0, 0, 1, 1, 1])

    result = dict(_class_tfidf_top_words(texts, labels, top_n=5))

    assert "python" in result[0]
    assert "java" in result[1]
    assert "python" not in result[1]
    assert "java" not in result[0]


def test_class_tfidf_top_words_drops_clusters_with_no_vocabulary_left():
    # Stopwords-only cluster (label 1) has nothing left after stopword
    # removal -- it should be dropped, not returned as an empty topic.
    texts = pd.Series(["python pandas dataframe", "the a an of"])
    labels = np.array([0, 1])

    result = dict(_class_tfidf_top_words(texts, labels, top_n=5))

    assert 0 in result
    assert 1 not in result


# ---------------------------------------------------------------------------
# coherence_score / per_cluster_coherence
# ---------------------------------------------------------------------------

def test_coherence_score_returns_a_float_for_a_normal_case():
    # Deliberately not a "good clustering scores higher than bad" assertion:
    # checking that directly (at small scale, with hand-assigned rather than
    # KMeans-produced labels) turned out not to hold -- a random shuffle of
    # five real, known-distinct topic groups scored *higher* c_v than the
    # true grouping, both at this module's scale and at notebooks/04's own
    # validated scale (n=418). What's actually validated is the search
    # behavior end to end (see find_best_semantic_k's test below), not a
    # "higher is more correct" property of a single coherence_score call in
    # isolation -- so this test only checks the function's basic contract.
    texts = pd.Series([
        "python pandas dataframe", "python numpy array",
        "java spring boot", "java maven dependency",
    ])
    labels = np.array([0, 0, 1, 1])

    score = coherence_score(texts, labels)

    assert isinstance(score, float)


def test_coherence_score_returns_sentinel_when_no_cluster_has_vocabulary():
    # Regression test: TfidfVectorizer raises ValueError (not an empty
    # result) when every pseudo-document is empty after stopword removal --
    # this used to propagate uncaught instead of degrading gracefully.
    texts = pd.Series(["the a an of", "the a an of", "the a an of", "the a an of"])
    labels = np.array([0, 0, 1, 1])

    assert coherence_score(texts, labels) == -1.0


def test_per_cluster_coherence_covers_every_label():
    texts = pd.Series([
        "python pandas dataframe", "python numpy array",
        "java spring boot", "java maven dependency",
    ])
    labels = np.array([0, 0, 1, 1])

    result = per_cluster_coherence(texts, labels)

    assert set(result.keys()) == {0, 1}
    assert all(isinstance(v, float) for v in result.values())


def test_per_cluster_coherence_reports_none_for_vocabulary_free_cluster():
    texts = pd.Series(["python pandas dataframe", "python numpy array", "the a an of"])
    labels = np.array([0, 0, 1])

    result = per_cluster_coherence(texts, labels)

    assert result[0] is not None
    assert result[1] is None


def test_per_cluster_coherence_handles_no_cluster_having_vocabulary():
    texts = pd.Series(["the a an of", "the a an of", "the a an of", "the a an of"])
    labels = np.array([0, 0, 1, 1])

    result = per_cluster_coherence(texts, labels)

    assert result == {0: None, 1: None}


# ---------------------------------------------------------------------------
# find_best_semantic_k -- centerpiece, mirroring test_lexical_clustering.py's
# approach: check the search's output against an independently-known-good
# answer, rather than individually justifying every internal constant.
#
# Unlike the lexical search, this one is validated against real text, not
# synthetic word combinations: c_v is computed from actual cluster-member
# words, and a synthetic corpus built from small fixed vocabularies sampled
# into short rows turned out to have an exploitable property real text
# doesn't -- as k approaches n, singleton-ish clusters made of a handful of
# words that trivially co-occur (they came from one sampled row) score
# artificially well, masking the real search behavior being tested.
# ---------------------------------------------------------------------------

CANDIDATE_TAGS = ["python", "sql", "excel", "css", "swift"]


def _known_structure_titles():
    # The same construction as notebooks/04_semantic_metric_selection.ipynb
    # (five real, non-overlapping StackOverflow tags standing in for five
    # distinct topics), at a smaller scale drawn from the repo's n500 sample
    # so the test suite doesn't need the full 5000-row file.
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "stackoverflow_sample_2021_03_01_n500.json")
    with open(data_path) as f:
        data = json.load(f)

    groups = {tag: [] for tag in CANDIDATE_TAGS}
    for row in data:
        hits = [tag for tag in CANDIDATE_TAGS if tag in set(row["tags"])]
        if len(hits) == 1:
            groups[hits[0]].append(row["title"])

    return pd.Series([title for group in groups.values() for title in group])


def test_find_best_semantic_k_is_close_to_a_known_good_value():
    texts = _known_structure_titles()
    vect = reduce_semantic(embed_semantic(texts))

    found_k = find_best_semantic_k(texts, vect)
    found_labels = run_kmeans(vect, found_k)
    found_score = coherence_score(texts, found_labels)

    true_group_count = len(CANDIDATE_TAGS)
    true_labels = run_kmeans(vect, true_group_count)
    true_score = coherence_score(texts, true_labels)

    # The found k shouldn't be meaningfully worse by the metric this search
    # actually optimizes -- allowed to do better (e.g. real sub-topics), not
    # worse.
    assert found_score >= true_score * 0.8

    # Sanity bound: not degenerate -- in particular, not creeping toward n
    # the way this search did before SEMANTIC_MIN_AVG_CLUSTER_SIZE existed.
    assert 3 <= found_k <= 10 * true_group_count
