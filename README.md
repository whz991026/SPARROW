# SPARROW

**SPARROW**: Spatial Patterning and RNA-editing Regulation in Spatial Transcriptomics.

SPARROW is a Python package for spatial transcriptomics analysis, including spatial clustering, deconvolution, and spatial A-to-I RNA editing analysis.

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/yourname/SPARROW.git
```

For development:

```bash
git clone https://github.com/yourname/SPARROW.git
cd SPARROW
pip install -e .
```

## Quick start

```python
import scanpy as sc
from sparrow import SPARROW
from sparrow import clustering

adata = sc.read_h5ad("spatial_data.h5ad")

model = SPARROW(
    adata=adata,
    use_gene=True,
    use_ai=False,
    epochs=600,
)

adata = model.train()
adata = clustering(adata, n_clusters=7)
```

## A-to-I RNA editing analysis

```python
from sparrow import filter_atoi_sites, detect_spatial_atoi_sites, compute_spatial_atoi_score

adata_ai_filtered, keep_mask = filter_atoi_sites(
    adata_ai,
    min_spot_cov=10,
    min_n_spots=10,
)

sv_atoi = detect_spatial_atoi_sites(
    adata_ai_filtered,
    group_adata=adata,
    group_key="domain",
)

score = compute_spatial_atoi_score(
    adata_ai_filtered,
    sv_atoi,
)
```

## Optional dependencies

For Mclust clustering through R:

```bash
pip install rpy2
```

In R:

```r
install.packages("mclust")
```

For Moran's I and Geary's C spatial statistics:

```bash
pip install libpysal esda
```

## Repository layout

```text
SPARROW/
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── src/
│   └── sparrow/
│       ├── __init__.py
│       ├── sparrow.py
│       ├── model.py
│       ├── preprocess.py
│       ├── utils.py
│       └── atoi.py
└── examples/
```

## Citation

Coming soon.
