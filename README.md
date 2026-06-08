# SPARROW

**SPARROW**: Spatial Patterning and RNA-editing Regulation in Spatial Transcriptomics.

SPARROW is a Python package for spatial transcriptomics analysis, including spatial clustering, deconvolution, and spatial A-to-I RNA editing analysis.

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/whz991026/SPARROW.git
```

For development:

```bash
git clone https://github.com/whz991026/SPARROW.git
cd SPARROW
pip install -e .
```

## Quick start

```python
import scanpy as sc
from sparrow import SPARROW
from sparrow import clustering

adata = sc.read\_h5ad("spatial\_data.h5ad")

model = SPARROW(
    adata=adata,
    use\_gene=True,
    use\_ai=False,
    epochs=600,
)

adata = model.train()
adata = clustering(adata, n\_clusters=7)
```

## A-to-I RNA editing analysis

```python
from sparrow import filter\_atoi\_sites, detect\_spatial\_atoi\_sites, compute\_spatial\_atoi\_score

adata\_ai\_filtered, keep\_mask = filter\_atoi\_sites(
    adata\_ai,
    min\_spot\_cov=10,
    min\_n\_spots=10,
)

sv\_atoi = detect\_spatial\_atoi\_sites(
    adata\_ai\_filtered,
    group\_adata=adata,
    group\_key="domain",
)

score = compute\_spatial\_atoi\_score(
    adata\_ai\_filtered,
    sv\_atoi,
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
│       ├── \_\_init\_\_.py
│       ├── sparrow.py
│       ├── model.py
│       ├── preprocess.py
│       ├── utils.py
│       └── atoi.py
└── examples/
```

## Citation

Coming soon.

