# SPARROW

**SPARROW**: Spatial Patterning and RNA-editing Regulation in Spatial Transcriptomics.

SPARROW is a Python package for resolving tissue architecture, cellular niches, and spatial A-to-I RNA-editing programs from spatial transcriptomics data.

## Abstract

Spatial transcriptomics preserves tissue architecture during molecular profiling, but most computational analyses focus on gene-expression-defined domains and cell-type composition, leaving spatial post-transcriptional regulation less well characterized. SPARROW addresses this gap with a graph self-supervised learning framework that integrates spatial representation learning, cell-type deconvolution, and spatial A-to-I editome modeling.

SPARROW learns robust spot embeddings from gene-expression profiles and spatial coordinates using graph autoencoding, local contrastive learning, masked reconstruction, and dynamic graph updating. These embeddings support spatial domain identification, while sparse cell-to-spot projection estimates local cell-type composition from single-cell references. SPARROW further quantifies global A-to-I activity, detects spatially variable A-to-I sites, and jointly models editing signals with deconvolved cell-type modules and ADAR-family expression.

Across spatial transcriptomics benchmarks, SPARROW achieved strong clustering and deconvolution performance. In human dorsolateral prefrontal cortex, SPARROW revealed divergent ADAR-family programs, layer-associated global A-to-I activity, and a reproducible white-matter-enriched site-level editing program. Together, SPARROW provides a cell-type-aware framework for resolving tissue architecture, cellular niches, and spatial A-to-I RNA-editing programs.

## Framework

![Overview of the SPARROW framework](Figure%201.jpg)

SPARROW connects three analysis modules:

- Spatial domain identification from spatial gene-expression profiles and spot coordinates.
- Spatially informed cell-type deconvolution using single-cell reference profiles.
- Spatial A-to-I editome analysis, including global A-to-I score calculation, spatially variable A-to-I site detection, and joint modeling with cell-type composition and ADAR-family expression.

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

## Data, Code, and Examples

- Data and code archive on Figshare: 10.6084/m9.figshare.32935274
- Project repository: [https://github.com/whz991026/SPARROW](https://github.com/whz991026/SPARROW)
- Example notebook: [`examples/Example_151673.ipynb`](examples/Example_151673.ipynb)

The example notebook demonstrates a compact workflow on DLPFC sample 151673, including SPARROW clustering, visualization of precomputed deconvolution output, and global and spatially variable A-to-I editing analysis.

## Keywords

Spatial transcriptomics; graph self-supervised learning; cell-type deconvolution; spatial domain identification; A-to-I RNA editing; spatial editomics; ADAR-family regulation; spatially variable editing sites.
