import json
import os

import pytest
from streamlit.testing.v1 import AppTest

from clustering import labeling as cluster_labeling
from config import API_KEY_ENV_VAR

HARNESS_PATH = os.path.join(os.path.dirname(__file__), "app_harness.py")

SAMPLE_DATA = [
    {"tags": ["python", "pandas", "dataframe"], "user_id": 1},
    {"tags": ["python", "numpy", "array"], "user_id": 1},
    {"tags": ["python", "flask", "api"], "user_id": 1},
    {"tags": ["java", "spring", "maven"], "user_id": 2},
    {"tags": ["java", "hibernate", "jpa"], "user_id": 2},
    {"tags": ["java", "junit", "mockito"], "user_id": 2},
    {"tags": ["javascript", "react", "redux"], "user_id": 3},
    {"tags": ["javascript", "node", "express"], "user_id": 3},
    {"tags": ["javascript", "vue", "vuex"], "user_id": 3},
]


def make_app(data=SAMPLE_DATA):
    os.environ["APP_HARNESS_DATA_JSON"] = json.dumps(data)
    return AppTest.from_file(HARNESS_PATH)


def test_invalid_field_shows_error_and_disables_button():
    at = make_app()
    at.run()
    at.text_input(key="field_name").set_value("nope").run()

    assert len(at.error) == 1
    assert "nope" in at.error[0].value
    assert at.button[0].disabled is True


def test_valid_field_clears_error_and_enables_button():
    at = make_app()
    at.run()
    at.text_input(key="field_name").set_value("tags").run()

    assert len(at.error) == 0
    assert at.button[0].disabled is False


def test_invalid_join_field_shows_error():
    at = make_app()
    at.run()
    at.text_input(key="field_name").set_value("tags").run()
    at.text_input(key="join_field").set_value("not_a_real_field").run()

    assert len(at.error) == 1
    assert at.button[0].disabled is True


def test_valid_join_field_with_invalid_field_shows_error_instead_of_crashing():
    # Regression test: a valid join field was aggregated by immediately,
    # without checking that the field being clustered exists -- so filling
    # the join field first (or at all, before a valid field name) reached
    # aggregate_by_join_field with a missing column and raised an uncaught
    # KeyError, crashing the script run instead of showing the field error
    # the app already renders.
    at = make_app()
    at.run()
    at.text_input(key="join_field").set_value("user_id").run()

    assert len(at.exception) == 0
    assert len(at.error) == 1  # the field error only -- user_id is itself valid
    assert at.button[0].disabled is True


def test_full_pipeline_without_join_field():
    at = make_app()
    at.run()
    at.text_input(key="field_name").set_value("tags").run()
    at.button[0].click().run()

    assert len(at.exception) == 0
    assert at.session_state.cluster_stats is not None
    # No aggregation requested -- one cluster-member row per original row.
    assert sum(len(c["data"]) for c in _download_payload(at)) == len(SAMPLE_DATA)


def test_full_pipeline_with_join_field_aggregates_by_group():
    at = make_app()
    at.run()
    at.text_input(key="field_name").set_value("tags").run()
    at.text_input(key="join_field").set_value("user_id").run()
    at.button[0].click().run()

    assert len(at.exception) == 0
    # 3 distinct user_ids in SAMPLE_DATA -- aggregation should collapse
    # 9 rows down to 3 groups before clustering (checked via total member
    # rows across clusters, since the number of *clusters* found depends on
    # k-search, not directly on the aggregation step).
    assert sum(len(c["data"]) for c in at.session_state.cluster_stats.values()) == 3


def test_full_pipeline_semantic_mode():
    # Longer timeouts than the lexical tests: embedding real text with
    # SentenceTransformer is slow relative to CountVectorizer+SVD.
    at = make_app()
    at.run(timeout=30)
    at.text_input(key="field_name").set_value("tags").run(timeout=30)
    at.radio(key="mode").set_value("Semantic").run(timeout=30)
    at.button[0].click().run(timeout=60)

    assert len(at.exception) == 0
    assert at.session_state.cluster_stats is not None
    assert at.session_state.quality_key == "c_v"
    # Semantic mode's quality key should actually be present on every
    # cluster entry, not silently defaulting to the lexical one.
    assert all("c_v" in entry for entry in at.session_state.cluster_stats.values())
    assert all("silhouette" not in entry for entry in at.session_state.cluster_stats.values())
    assert sum(len(c["data"]) for c in _download_payload(at)) == len(SAMPLE_DATA)


def test_semantic_mode_hides_lexical_preprocessing_checkboxes():
    at = make_app()
    at.run(timeout=30)
    at.radio(key="mode").set_value("Semantic").run(timeout=30)

    with pytest.raises(KeyError):
        at.checkbox(key="lowercase")


def test_checkbox_toggle_clears_cached_results():
    # Seeded with a plain sentinel rather than running the real pipeline: a
    # genuine segmented_control that's rendered in one run and then absent
    # in the next (exactly what happens once restart_session clears
    # cluster_stats) trips an apparent AppTest/Streamlit library bug in
    # ButtonGroup's stale-widget-state reconciliation, unrelated to this
    # app's logic. A sentinel that fails before reaching that widget avoids
    # it while still exercising the real restart_session wiring.
    at = make_app()
    at.run()
    at.session_state["cluster_stats"] = "sentinel"
    at.run()
    assert at.session_state.cluster_stats is not None

    at.checkbox(key="lowercase").set_value(True).run()
    assert at.session_state.cluster_stats is None


def test_download_output_is_1_indexed_and_well_shaped():
    at = make_app()
    at.run()
    at.text_input(key="field_name").set_value("tags").run()
    at.button[0].click().run()

    records = _download_payload(at)
    assert len(records) > 0
    for record in records:
        assert set(record.keys()) == {"cluster", "size", "silhouette", "outliers", "data"}
        # regression test: used to be 0-indexed, inconsistent with the UI's "Cluster N" display
        assert record["cluster"] >= 1
        for row in record["data"]:
            assert "distance_to_centroid" in row
            assert "outlier_z_score" in row
            assert "is_outlier" in row


def test_non_text_field_shows_error_instead_of_crashing():
    # Regression test: the field input accepts any column, but both backends
    # assume text -- a numeric field (e.g. the user_id in this app's own
    # sample data) used to reach CountVectorizer/SentenceTransformer and fail
    # with a raw TypeError, crashing the script run.
    at = make_app()
    at.run()
    at.text_input(key="field_name").set_value("user_id").run()
    at.button[0].click().run()

    assert len(at.exception) == 0
    assert len(at.error) == 1
    assert "int" in at.error[0].value
    assert at.session_state.cluster_stats is None


def test_rows_with_a_missing_field_value_are_skipped_with_a_warning():
    # Regression test: a null in the clustered field used to reach the
    # vectorizer as None and crash. Rows are dropped from the DataFrame (not
    # inside prepare_text) so the surviving rows still line up with their
    # vectors -- checked here via the member-row total.
    data = SAMPLE_DATA + [{"tags": None, "user_id": 4}]
    at = make_app(data=data)
    at.run()
    at.text_input(key="field_name").set_value("tags").run()
    at.button[0].click().run()

    assert len(at.exception) == 0
    assert len(at.warning) == 1
    assert "1 row" in at.warning[0].value
    assert sum(len(c["data"]) for c in _download_payload(at)) == len(SAMPLE_DATA)


def test_blank_field_values_show_error_instead_of_crashing():
    # Regression test: an all-blank field made CountVectorizer raise "empty
    # vocabulary" before the vocabulary-size check could report it cleanly.
    blank_data = [{"tags": ""} for _ in range(4)]
    at = make_app(data=blank_data)
    at.run()
    at.text_input(key="field_name").set_value("tags").run()
    at.button[0].click().run()

    assert len(at.exception) == 0
    assert len(at.error) == 1
    assert at.session_state.cluster_stats is None


def test_too_small_vocabulary_shows_error_instead_of_crashing():
    # Regression test: vectorize_lexical's own NotEnoughDataException (too
    # few distinct values to cluster) used to propagate uncaught, crashing
    # the whole script run instead of showing a clean error.
    too_small_data = [{"tags": ["x"]}, {"tags": ["x"]}]
    at = make_app(data=too_small_data)
    at.run()
    at.text_input(key="field_name").set_value("tags").run()
    at.button[0].click().run()

    assert len(at.exception) == 0
    assert len(at.error) == 1
    assert at.session_state.cluster_stats is None


def test_no_stray_markdown_output_with_join_field():
    # Regression test: a ternary-used-as-a-bare-statement inside
    # aggregate_by_join_field used to trigger Streamlit's "magic" commands,
    # rendering a literal "None" once per group via st.write.
    at = make_app()
    at.run()
    at.text_input(key="field_name").set_value("tags").run()
    at.text_input(key="join_field").set_value("user_id").run()

    assert len(at.exception) == 0
    assert not any(el.value == "None" for el in at.markdown)


def _download_payload(at):
    return json.loads(at.session_state.test_ready_data)


# ---------------------------------------------------------------------------
# Cluster labeling (Gemini) -- gated by GEMINI_API_KEY in the environment.
# label_clusters itself is mocked throughout: these tests exercise app.py's
# wiring (checkbox visibility, caching, failure handling), not the Gemini
# integration, which is covered separately in test_cluster_labeling.py.
# ---------------------------------------------------------------------------

def test_labeling_checkbox_hidden_without_api_key(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
    at = make_app()
    at.run()

    with pytest.raises(KeyError):
        at.checkbox(key="summarize_clusters")


def test_labeling_checkbox_shown_with_api_key(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "test-key")
    at = make_app()
    at.run()

    assert at.checkbox(key="summarize_clusters") is not None


def test_full_pipeline_with_labeling_adds_labels_to_output(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "test-key")
    monkeypatch.setattr(
        cluster_labeling,
        "label_clusters",
        lambda stats, field_name, client=None: {entry["cluster"]: "a label" for entry in stats.values()},
    )

    at = make_app()
    at.run()
    at.text_input(key="field_name").set_value("tags").run()
    at.checkbox(key="summarize_clusters").set_value(True).run()
    at.button[0].click().run()

    assert len(at.exception) == 0
    assert len(at.warning) == 0
    records = _download_payload(at)
    assert len(records) > 0
    assert all(r["label"] == "a label" for r in records)


def test_labeling_failure_does_not_block_cluster_results(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "test-key")

    def _raise(stats, field_name, client=None):
        raise cluster_labeling.ClusterLabelingError("boom")

    monkeypatch.setattr(cluster_labeling, "label_clusters", _raise)

    at = make_app()
    at.run()
    at.text_input(key="field_name").set_value("tags").run()
    at.checkbox(key="summarize_clusters").set_value(True).run()
    at.button[0].click().run()

    assert len(at.exception) == 0
    assert len(at.warning) == 1
    assert at.session_state.cluster_stats is not None
    records = _download_payload(at)
    assert all("label" not in r for r in records)
