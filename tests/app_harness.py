"""Test harness mirroring app.py's main flow.

streamlit.testing.v1.AppTest cannot drive st.file_uploader at all -- it
isn't a supported widget type in that framework, not just awkward to use --
so this harness substitutes a DataFrame read from an env var in its place.
Every actual function (aggregate_by_join_field, restart_session,
sync_checkbox, kmeans_clusters, build_cluster_stats, show_results,
convert_for_download, vectorize_lexical, etc.) is imported and called from
the real app module, not reimplemented -- only the thin top-level
orchestration here is duplicated, so logic changes in app.py can't silently
go untested.
"""
import json
import os

import pandas as pd
import streamlit as st

import app
from clustering import labeling as cluster_labeling
from clustering.kmeans import NotEnoughDataException


app.init_session_state()

data = pd.DataFrame(json.loads(os.environ["APP_HARNESS_DATA_JSON"]))

field_name = st.text_input("JSON field to cluster", key="field_name", on_change=app.restart_session)

field_valid = field_name in data.columns
if not field_valid:
    st.error(f"Field '{field_name}' not found in the uploaded file.")

join_field = st.text_input(
    "Optional field to aggregate by (e.g. a user id)",
    key="join_field",
    on_change=app.restart_session,
)

join_valid = not join_field or join_field in data.columns
if join_field and not join_valid:
    st.error(f"Join field '{join_field}' not found in the uploaded file.")
elif join_field:
    data = app.aggregate_by_join_field(data, field_name, join_field)

st.caption("Preprocessing")
split_camel_case = app.sync_checkbox("Split camelCase", "split_camel_case")
strip_punctuation = app.sync_checkbox("Strip punctuation", "strip_punctuation")
lowercase = app.sync_checkbox("Lowercase", "lowercase")

if cluster_labeling.is_available():
    st.divider()
    st.caption("AI labeling (optional, calls the Gemini API)")
    summarize_clusters = app.sync_checkbox("Generate cluster labels (Gemini)", "summarize_clusters")
else:
    summarize_clusters = False

if st.button("Run clustering", disabled=not (field_valid and join_valid)):
    field_values = data[field_name]
    try:
        vectors = app.vectorize_lexical(
            field_values,
            split_camel_case=split_camel_case,
            strip_punctuation=strip_punctuation,
            lowercase=lowercase,
        )
    except NotEnoughDataException:
        st.error(
            "Not enough distinct values to cluster with this configuration -- "
            "try a different field, join field, or preprocessing options."
        )
    else:
        with st.spinner("Clustering data...", show_time=True):
            kmeans_res = app.kmeans_clusters(vectors)

        distances = app.centroid_distances(vectors, kmeans_res)
        outlier_z = app.outlier_z_scores(distances, kmeans_res)
        silhouette = app.per_cluster_silhouette(vectors, kmeans_res)

        st.session_state.cluster_stats = app.build_cluster_stats(data, kmeans_res, distances, outlier_z, silhouette)
        st.session_state.cluster_label_summaries = None

        if summarize_clusters:
            with st.spinner("Generating cluster labels...", show_time=True):
                try:
                    st.session_state.cluster_label_summaries = cluster_labeling.label_clusters(
                        st.session_state.cluster_stats, field_name
                    )
                except cluster_labeling.ClusterLabelingError as e:
                    st.warning(f"Cluster labeling failed, showing clusters without labels: {e}")

st.divider()

if st.session_state.cluster_stats is not None:
    choices = app.show_results(st.session_state.cluster_stats, st.session_state.cluster_label_summaries)
    ready_data = app.convert_for_download(
        st.session_state.cluster_stats, choices, st.session_state.cluster_label_summaries
    )
    st.session_state.test_ready_data = ready_data  # test-only: AppTest can't read st.download_button's payload
    st.download_button(
        label="Download cluster stats",
        data=ready_data,
        file_name="test-clusters.json",
        icon=":material/download:",
    )
