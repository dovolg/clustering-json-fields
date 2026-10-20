# Clustering JSON Fields

Upload a JSON file, pick a field, and get its values clustered into reviewable, keep-or-delete groups.

## How it Works

Two clustering pipelines, selectable per run: **Lexical** (KMeans over count-vectorized n-grams) and **Semantic** (KMeans over sentence embeddings, scored by topic coherence instead of silhouette). Both end up at the same review/Keep-Delete/download flow. **Lexical is the better choice for categorical/tag-like fields; Semantic is the better choice for free text** — validated on real data, not just assumed (see [`notebooks/`](notebooks/)).

1.  **Upload JSON**: The user uploads a JSON file containing a list of objects.

2.  **Configure**:
    *   **Field to cluster**: the name of the JSON field whose values should be clustered.
    *   **Optional join/aggregation field** (e.g. a user id): if set, rows are grouped by this field's value first, and the selected field's values within each group are combined into one list before clustering. This clusters the *groups* (e.g. "users"), not the individual rows — the output is then one record per surviving group.
    *   List-valued fields (e.g. a JSON array like `tags`) are detected automatically from the data's actual type and joined into text — no separate config needed for this.
    *   The field has to hold text (or lists of text). A numeric or nested-object field is refused with a clear error rather than coerced with `str()` — clustering numbers as if they were words produces a confident, meaningless answer. Rows with no value in the field are skipped, with a count reported.
    *   **Clustering mode**: Lexical or Semantic (see below) — switching modes clears any cached result.
    *   In Lexical mode, three preprocessing checkboxes, **all off by default** so nothing is silently transformed: Split camelCase, Strip punctuation, Lowercase. (Not shown in Semantic mode — sentence embeddings don't need them.)
    *   Clustering only runs once you click **Run clustering** — it doesn't re-run automatically as you adjust the config above.

3.  **Vectorization & dimensionality reduction**:
    *   **Lexical**: text is vectorized with `CountVectorizer`, then reduced with `TruncatedSVD` (LSA) — the standard, sparse-friendly choice for a term-document matrix, unlike PCA which requires centering. The target amount of variance to preserve is an empirically-derived constant (see [`notebooks/`](notebooks/)), not a guess.
    *   **Semantic**: text is embedded with a sentence-transformer model (`all-MiniLM-L6-v2` — baked into the Docker image at build time, downloaded on first use when running locally), then reduced with PCA — the textbook-correct choice for dense embeddings (unlike the lexical path's sparse count data, there's no reason to avoid centering here).

4.  **Clustering**:
    *   Both modes cluster the reduced vectors with KMeans. The number of clusters (`k`) is found automatically via a coarse-to-fine search and cached for the session (recomputed only if the file or config changes) — **Lexical** scores candidate `k`s by silhouette score plus a cluster-count parsimony term; **Semantic** scores them by `c_v` topic coherence (computed from each cluster's distinctive words), since silhouette was found to collapse to a degenerate answer on real, diffuse text in a way `c_v` doesn't. See [`notebooks/`](notebooks/) for why the two paths need different metrics and search constants.

5.  **Review clusters**:
    *   Clusters are shown ordered by per-cluster quality — silhouette score in Lexical mode, `c_v` coherence in Semantic mode — best first.
    *   Each cluster shows its size, quality score, a count of flagged outliers, and its full member rows — sorted by distance to the cluster's centroid (most representative members first), with `distance_to_centroid`, `outlier_z_score`, and `is_outlier` shown as extra columns. The outlier flag is a statistical z-score of each point's distance to its own cluster's centroid — it catches individual poor-fit points even when the cluster's average quality looks fine.
    *   Per cluster, choose **Keep** or **Delete** — this is a whole-cluster decision; there's no row-level selection within a cluster.

6.  **Download Results**:
    *   The downloaded file is a list of the kept clusters. Each entry has `cluster`, `size`, `outliers` (count of flagged points), a `data` field containing the full original rows belonging to that cluster (each with its `distance_to_centroid`, `outlier_z_score`, and `is_outlier`), `label` if cluster labeling was enabled, and either `silhouette` or `c_v` depending on which mode produced it.

## Cluster Labeling (optional)

If a `GEMINI_API_KEY` is set in the environment, a "Generate cluster labels" checkbox appears and, when enabled, a short LLM-generated summary is requested per cluster (from its most representative members) via Gemini's structured output. It's otherwise invisible — no key, no checkbox, no external calls. Labeling is a non-blocking enhancement: if it fails (missing key, exhausted retries on a transient error, ...) the rest of the review flow still works, just without labels. See [`clustering/labeling.py`](clustering/labeling.py).

## Key Features

*   **Two clustering modes**: Lexical (count-vectorized n-grams) and Semantic (sentence embeddings), each with its own vectorization, reduction, and k-search metric tuned for what it's actually clustering.
*   **Configurable field + optional join/aggregation field**, with automatic list-vs-string detection for structured fields like tags.
*   **Explicit, opt-in preprocessing** in Lexical mode — no hidden or default text transformations; every step is a visible checkbox, off by default.
*   **Automatic k search** with per-cluster quality scoring (silhouette or `c_v`, depending on mode) surfaced to the user, not just used internally to pick k.
*   **Per-point outlier flagging**: a statistical z-score of each point's distance to its own cluster's centroid, surfacing individual poor fits even when a cluster's average quality looks fine.
*   **Cluster-level review**: Keep/Delete per cluster, with full member preview ordered by distance to centroid.
*   **Gated clustering**: a "Run clustering" step and session-cached `k` mean config changes don't trigger expensive recomputation until you're ready.

## Sample Data

See [`data/README.md`](data/README.md) for the sample dataset, its source, and license/attribution.

## Investigation Notebooks

See [`notebooks/`](notebooks/) for the empirical investigation behind both pipelines' constants and design choices, checked against real data rather than guessed:
*   **Lexical** (`01`-`03`): whether the k-search's concave-landscape assumption holds, the SVD variance target, the outlier z-score threshold, and whether a standard elbow method would be a simpler, comparable alternative to the custom k-search heuristic.
*   **Semantic** (`04`-`07`): which quality metric to use instead of silhouette, KMeans vs. HDBSCAN as the architecture, whether the same k-search strategy transfers, and the embedding model/reduction/`n_components` choices.
*   **Lexical vs. semantic** (`08`): the two pipelines benchmarked on the same real data, using the actual shipped code for both — Adjusted Rand Index and NMI against known categories for the `tags` field, and class-based TF-IDF word extraction plus an LLM-as-judge pairwise comparison for the `title` field (tag membership doesn't reliably transfer as ground truth for judging title clusters, so this field needed a different kind of evidence, not the same metric reused).

## Development

Tests and linting run in CI on every push (see [`.github/workflows/tests.yml`](.github/workflows/tests.yml)); to run them locally:

```bash
pip install -r tests/requirements.txt
pytest -v
ruff check .
```

Ruff is configured for correctness rules only, not formatting — see [`ruff.toml`](ruff.toml) for why.

## Running the Application

### Locally

1.  Ensure you have Python and `pip` installed.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

    On the first Semantic-mode run this downloads the sentence-transformer model (~90 MB); the Docker image below has it baked in already.

### Using Docker

1.  Navigate to the `clustering-json-fields` directory.
2.  **Build the Docker image**:
    ```bash
    make build
    ```
    This command builds the Docker image with the tag `clustering-app:latest`.

3.  **Run the Docker container**:
    ```bash
    make run
    ```
    This command runs the application in a Docker container and maps port 8501 on your host to port 8501 in the container. You can then access the app at `http://localhost:8501`. If a `.env` file exists in the project directory (see [Cluster Labeling](#cluster-labeling-optional)), it's passed through to the container automatically; its absence doesn't break this command.

4.  **Clean up (remove Docker image)**:
    ```bash
    make clean
    ```

## License

This project's own code is licensed under the [MIT License](LICENSE). The sample data has its own, separate license — see [Sample Data](#sample-data) above.