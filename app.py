import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # loads GEMINI_API_KEY (etc.) from a local .env file into os.environ, if present

import json
import pandas as pd

from clustering import labeling as cluster_labeling
from config import OUTLIER_Z_THRESHOLD
from clustering.kmeans import (
    vectorize_lexical,
    find_best_kmeans_k,
    run_kmeans,
    centroid_distances,
    outlier_z_scores,
    per_cluster_silhouette,
    NotEnoughDataException,
)


def restart_session():
    st.session_state.k = -1
    st.session_state.cluster_stats = None
    st.session_state.cluster_label_summaries = None

def sync_checkbox(label, key):
    return st.checkbox(label, key=key, on_change=restart_session)

def aggregate_by_join_field(data, field_name, join_field):
    def combine(values):
        combined = []
        for v in values:
            if isinstance(v, list):
                combined.extend(v)
            else:
                combined.append(v)
        return combined

    return data.groupby(join_field)[field_name].apply(combine).reset_index()

def kmeans_clusters(vect):
    if st.session_state.k == -1:
        st.session_state.k = find_best_kmeans_k(vect)

    return run_kmeans(vect, st.session_state.k)

def build_cluster_stats(data, predictions, distances, outlier_z, silhouette_by_cluster):
    data = data.reset_index(drop=True).copy()
    data["distance_to_centroid"] = distances.round(4)
    data["outlier_z_score"] = outlier_z.round(2)
    data["is_outlier"] = outlier_z > OUTLIER_Z_THRESHOLD

    stats = dict()
    for label, silhouette in silhouette_by_cluster.items():
        members = data[predictions == label].sort_values("distance_to_centroid")
        stats[label] = {
            "cluster": int(label) + 1,
            "size": len(members),
            "silhouette": silhouette,
            "outliers": int(members["is_outlier"].sum()),
            "data": members.to_dict(orient="records"),
        }
    return stats

def show_results(stats, label_summaries=None):
    label_summaries = label_summaries or {}
    st.write(f"Found {len(stats)} clusters")

    ordered_labels = sorted(
        stats,
        key=lambda label: stats[label]["silhouette"] if stats[label]["silhouette"] is not None else float("-inf"),
        reverse=True,
    )

    choices = dict()

    for label in ordered_labels:
        entry = stats[label]
        silhouette_text = f"{entry['silhouette']:.3f}" if entry["silhouette"] is not None else "n/a"
        outlier_text = f", {entry['outliers']} flagged as outliers" if entry["outliers"] else ""
        st.write(f"Cluster {entry['cluster']}: {entry['size']} items, silhouette score {silhouette_text}{outlier_text}")

        summary = label_summaries.get(entry["cluster"])
        if summary:
            st.caption(summary)

        choices[label] = st.segmented_control(
            "Choose action", ["Keep", "Delete"], default="Keep", key=f"selector-{label}"
        )
        st.dataframe(pd.DataFrame(entry["data"]), hide_index=True)
        st.write("---")

    return choices

def init_session_state():
    defaults = {
        "k": -1,
        "fname": "",
        "field_name": "",
        "join_field": "",
        "cluster_stats": None,
        "cluster_label_summaries": None,
        "split_camel_case": False,
        "strip_punctuation": False,
        "lowercase": False,
        "summarize_clusters": False,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

def convert_for_download(stats, choices, label_summaries=None):
    label_summaries = label_summaries or {}
    records = []
    for label, entry in stats.items():
        if choices[label] != "Keep":
            continue
        record = dict(entry)
        summary = label_summaries.get(entry["cluster"])
        if summary:
            record["label"] = summary
        records.append(record)
    return json.dumps(records, indent=4)


st.set_page_config(layout="wide")

st.title('Cluster data from JSON')

init_session_state()

uploaded_file = st.file_uploader("Choose a JSON file")

if uploaded_file is not None:
    if uploaded_file.name != st.session_state.fname:
        st.session_state.fname = uploaded_file.name
        restart_session()

    data = pd.DataFrame(json.load(uploaded_file))

    field_name = st.text_input(
        "JSON field to cluster", key="field_name", on_change=restart_session
    )

    field_valid = field_name in data.columns
    if not field_valid:
        st.error(f"Field '{field_name}' not found in the uploaded file.")

    join_field = st.text_input(
        "Optional field to aggregate by (e.g. a user id)",
        key="join_field",
        on_change=restart_session,
    )

    join_valid = not join_field or join_field in data.columns
    if join_field and not join_valid:
        st.error(f"Join field '{join_field}' not found in the uploaded file.")
    elif join_field:
        data = aggregate_by_join_field(data, field_name, join_field)

    st.caption("Preprocessing")
    split_camel_case = sync_checkbox("Split camelCase", "split_camel_case")
    strip_punctuation = sync_checkbox("Strip punctuation", "strip_punctuation")
    lowercase = sync_checkbox("Lowercase", "lowercase")

    if cluster_labeling.is_available():
        st.divider()
        st.caption("AI labeling (optional, calls the Gemini API)")
        summarize_clusters = sync_checkbox("Generate cluster labels (Gemini)", "summarize_clusters")
    else:
        summarize_clusters = False

    if st.button("Run clustering", disabled=not (field_valid and join_valid)):
        field_values = data[field_name]
        try:
            vectors = vectorize_lexical(
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
                kmeans_res = kmeans_clusters(vectors)

            distances = centroid_distances(vectors, kmeans_res)
            outlier_z = outlier_z_scores(distances, kmeans_res)
            silhouette = per_cluster_silhouette(vectors, kmeans_res)

            st.session_state.cluster_stats = build_cluster_stats(data, kmeans_res, distances, outlier_z, silhouette)
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
        choices = show_results(st.session_state.cluster_stats, st.session_state.cluster_label_summaries)
        ready_data = convert_for_download(
            st.session_state.cluster_stats, choices, st.session_state.cluster_label_summaries
        )

        st.download_button(
            label="Download cluster stats",
            data=ready_data,
            file_name=f"{uploaded_file.name.split('.')[0]}-clusters.json",
            icon=":material/download:",
        )
