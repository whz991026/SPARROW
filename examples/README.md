# SPARROW examples

This directory contains lightweight example notebooks for the SPARROW package.

## DLPFC sample 151673

`Example_151673.ipynb` demonstrates a compact single-sample workflow:

- SPARROW clustering on a DLPFC Visium sample
- visualization of precomputed deconvolution output
- global and spatially variable A-to-I editing analysis

The notebook expects the sample data directory to be available locally. Set
`SPARROW_DLPFC_151673_DIR` to the directory containing:

- `filtered_feature_bc_matrix.h5`
- `spatial/`
- `cluster_labels_151673.csv`
- `adata_obs_SPARROW.csv`
- `adata_ai.h5ad`

Large datasets and full reproduction notebooks should be distributed through the
project data archive, such as Figshare, and cited from the main repository.
