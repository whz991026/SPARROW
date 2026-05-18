"""Minimal SPARROW usage example."""

import scanpy as sc
from sparrow import SPARROW, clustering

adata = sc.read_h5ad("spatial_data.h5ad")
model = SPARROW(adata=adata, use_gene=True, use_ai=False, epochs=600)
adata = model.train()
adata = clustering(adata, n_clusters=7)
print(adata)
