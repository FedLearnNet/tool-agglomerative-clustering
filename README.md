# tool-agglomerative-clustering

Runs hierarchical agglomerative clustering and generates a dendrogram image and summary report.

## Input and outputs

Provide a numeric CSV table as `features`. The implementation fits Ward/Euclidean
agglomerative clustering on the first four columns. Supply feature columns without
an identifier column.

- `dendrogram_png`: a PNG file containing the SciPy dendrogram.
- `report`: a plain-text summary of the fitted cluster count.
- Metrics: `samples` and `clusters`, calculated from the fitted result.

`distance_threshold` controls the clustering cut; its default `0.0` builds the full
hierarchy. `plot_title`, `dendrogram_truncate_mode`, and `dendrogram_p` control the
plot. The existing `linkage`, `metric`, `standardize`, `compute_distances`,
`use_distance_threshold`, and `n_clusters` fields are retained for compatibility;
the current fitting implementation does not apply those settings.

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
# Set APP_ID and platform URLs in .env before connecting.
python main.py
```

The Docker base image provides the FL-Net API. `requirements.txt` contains only
additional runtime dependencies; `requirements-dev.txt` includes the API for local development.

Build with `docker build -t tool-agglomerative-clustering .`.
The image runs as `nonroot` and uses Matplotlib's headless Agg backend.
