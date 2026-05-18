"""SPARROW: Spatial Patterning and RNA-editing Regulation in Spatial Transcriptomics."""

try:
    from .sparrow import SPARROW
except Exception:
    # Keep package metadata importable even if optional runtime dependencies are unavailable.
    SPARROW = None

from .model import Discriminator, AvgReadout, Encoder, Encoder_sparse, Encoder_sc, Encoder_map

try:
    from .preprocess import (
        fix_seed,
        preprocess,
        preprocess_atoi,
        construct_interaction,
        construct_interaction_KNN,
        preprocess_adj,
        normalize_adj,
        sparse_mx_to_torch_sparse_tensor,
        preprocess_adj_sparse,
        permutation,
        add_contrastive_label,
        get_feature,
        filter_with_overlap_gene,
        sub_clustering,
        extract_boundary_as_new_cluster,
    )
except Exception:
    pass

try:
    from .utils import (
        mclust_R,
        search_res,
        refine_label,
        clustering_auto,
        clustering,
        extract_top_value,
        construct_cell_type_matrix,
        extract_top_k_and_renorm,
        project_cell_to_spot,
        standardize_labels,
        calculate_metrics,
        find_best_in_group,
        calculate_ari_strict,
        calculate_ari,
    )
except Exception:
    pass

try:
    from .atoi import *
except Exception:
    pass

__version__ = "0.1.0"

__all__ = [
    "SPARROW",
    "Discriminator", "AvgReadout", "Encoder", "Encoder_sparse", "Encoder_sc", "Encoder_map",
    "fix_seed", "preprocess", "preprocess_atoi", "construct_interaction", "construct_interaction_KNN",
    "preprocess_adj", "normalize_adj", "sparse_mx_to_torch_sparse_tensor", "preprocess_adj_sparse",
    "permutation", "add_contrastive_label", "get_feature", "filter_with_overlap_gene", "sub_clustering",
    "extract_boundary_as_new_cluster",
    "mclust_R", "search_res", "refine_label", "clustering_auto", "clustering", "extract_top_value",
    "construct_cell_type_matrix", "extract_top_k_and_renorm", "project_cell_to_spot", "standardize_labels",
    "calculate_metrics", "find_best_in_group", "calculate_ari_strict", "calculate_ari",
]
