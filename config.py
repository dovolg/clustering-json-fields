"""Centralized, named constants for the clustering pipeline and the optional
LLM labeling feature.

Deliberately NOT used by notebooks/diagnostics.py -- the investigation
notebooks are intentionally standalone from production code (see the
containment principle in notebooks/01_kmeans_shape_investigation.ipynb), so
they keep their own independent constants rather than importing this module.
"""

# --- clustering/kmeans.py ---------------------------------------------------

# See notebooks/02_quality_and_outliers.ipynb: for count-vectorized categorical fields,
# clustering quality (silhouette, k held fixed) peaks around n_components=20-30, i.e.
# ~0.27-0.31 cumulative variance -- consistently across multiple k values tested. 0.3 is
# the numerically-justified target.
TARGET_EXPLAINED_VARIANCE = 0.3

# See notebooks/02_quality_and_outliers.ipynb: a within-cluster distance-to-centroid
# z-score above 3 reliably picked out cross-cutting/multi-topic points that are a
# meaningfully worse fit for their assigned cluster than their clustermates -- confirmed
# independently by DBSCAN flagging several of the same points as noise.
OUTLIER_Z_THRESHOLD = 3.0

# Unlike the two constants above, these two aren't independently validated -- the overall
# k-search *strategy* is trusted instead, via a test that checks its output against a
# known-good answer on synthetic data (see tests/test_kmeans_clustering.py). Named here
# for readability.
SCORE_DECAY_STOP_RATIO = 0.8  # abandon a coarse scan once score drops this far below the best found
COARSE_STEP_THRESHOLD = 100  # only apply that early-stop while still in a coarse (unrefined) scan


# --- clustering/labeling.py --------------------------------------------------

API_KEY_ENV_VAR = "GEMINI_API_KEY"
GEMINI_MODEL = "gemini-2.5-flash"

SYSTEM_INSTRUCTION = (
    "You are labeling clusters produced by a clustering algorithm. You will "
    "receive a JSON array of {id, items} objects, where items are example "
    "members of that cluster (e.g. tags or short text). For each cluster, "
    "write a short (3-15 word) summary describing what its items have in "
    "common. Return exactly one {id, summary} object per input cluster, "
    "reusing the same id."
)

# A cluster's gist is clear from its most representative members -- sending
# every member of a large cluster would waste tokens for no real gain.
# build_cluster_stats already sorts each cluster's rows by distance to
# centroid, so a prefix is the most-representative-first slice.
TOP_N_ITEMS = 15

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2
REQUEST_TIMEOUT_MS = 30_000
