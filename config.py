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
# known-good answer on synthetic data (see tests/test_lexical_clustering.py). Named here
# for readability.
SCORE_DECAY_STOP_RATIO = 0.8  # abandon a coarse scan once score drops this far below the best found
COARSE_STEP_THRESHOLD = 100  # only apply that early-stop while still in a coarse (unrefined) scan


# --- clustering/semantic.py --------------------------------------------------

# See notebooks/07_semantic_config.ipynb: all-MiniLM-L6-v2 is both faster (~9x) and
# higher-scoring (by c_v) than all-mpnet-base-v2 once both are measured at the actual
# n_components being used -- not a cost/quality tradeoff, an easy call.
SEMANTIC_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# See notebooks/05_semantic_architecture.ipynb: PCA, not SVD -- centering is the
# textbook-correct choice for dense sentence embeddings (SVD's centering-avoidance only
# matters for the lexical path's sparse, non-negative count data).
#
# See notebooks/07_semantic_config.ipynb: n_components=20 had the highest c_v in a sweep
# over 3-50, with a normal, reviewable cluster count -- unlike n_components=3/5, which
# produced unusually large cluster counts (92, 106) for no quality gain.
SEMANTIC_N_COMPONENTS = 20

# How many top class-based TF-IDF words represent each cluster when scoring coherence.
# See notebooks/04_semantic_metric_selection.ipynb: no top_n in 5/10/15/20 gave a
# consistent improvement over this mid-range default.
SEMANTIC_COHERENCE_TOP_N = 10

# See notebooks/06_semantic_search_shape.ipynb: c_v's evaluation cost scales worse than
# linearly with cluster count (confirmed k=50->0.9s, k=200->3.1s), unlike silhouette,
# which doesn't care about k -- the lexical path's max_k=0.8*n formula would let a
# semantic search attempt evaluations in the thousands, which is impractical.
SEMANTIC_MAX_K_CAP = 150

# See notebooks/06_semantic_search_shape.ipynb: lowered from the lexical path's min_k=15,
# which excludes the entire region (k=5-14) where c_v's real optimum sits on structured
# data, forcing a worse answer than necessary purely by floor placement.
SEMANTIC_MIN_K = 8

# Unlike silhouette, c_v can reward clusters shrinking toward singletons (a cluster's own
# few words trivially "cohere" with themselves). Confirmed on real data at two scales: a
# 141-title known-structure sample, where c_v climbed from 0.41 at k=5 to 0.95 at k=135
# and find_best_semantic_k landed on k=139 against a true count of 5; and again in
# notebooks/08_lexical_vs_semantic_comparison.ipynb's tags sweep at n=418, where c_v rose
# monotonically from k=5 to k=200 with no interior peak while ARI against the true groups
# fell from 0.97 to 0.06 over the same range. This floor keeps max_k from reaching that
# region for small-to-moderate n, where max_k would otherwise equal (or nearly equal) n.
# For short, repetitive categorical text like tags specifically, the lexical pipeline is
# the validated choice regardless (notebooks/08) -- this floor's job is protecting
# free-text clustering at moderate n, not making c_v behave well on tags.
SEMANTIC_MIN_AVG_CLUSTER_SIZE = 5


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
