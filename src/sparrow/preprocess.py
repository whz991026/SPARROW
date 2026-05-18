"""Preprocessing and spatial graph utilities for spatial transcriptomics workflows."""

import os
import random

import anndata as ad
import numpy as np
import ot
import scanpy as sc
import scipy.sparse as sp
import torch
from matplotlib import pyplot as plt
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.backends import cudnn


def fix_seed(seed):
    """
    Set random seeds and deterministic CUDA options for reproducible runs.

    Parameters
    ----------
    seed : int
        Random seed used by Python, NumPy, PyTorch, and CUDA backends.

    Returns
    -------
    None
        This function modifies global random states and environment variables in place.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def preprocess(adata, n_top_genes=3000):
    """
    Preprocess spatial gene expression data using the standard Scanpy workflow.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing raw or count-like gene expression data in ``adata.X``.
    n_top_genes : int, default=3000
        Number of highly variable genes to select with the ``seurat_v3`` method.

    Returns
    -------
    None
        The input AnnData object is modified in place. The function adds highly variable
        gene annotations to ``adata.var`` and updates ``adata.X`` after normalization,
        log-transformation, and scaling.
    """
    print(f"Preprocessing gene expression: selecting top {n_top_genes} genes...")
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)


def preprocess_atoi(
    adata_ai,
    n_top_features=1000,
    min_cells_with_G=3,
    alpha=0.5,
    beta=0.5,
):
    """
    Preprocess A-to-I editing data by filtering, smoothing, feature selection, and scaling.

    Parameters
    ----------
    adata_ai : anndata.AnnData
        AnnData object for A-to-I editing data. It must contain ``adata_ai.layers['A']``
        and ``adata_ai.layers['G']``, where ``A`` and ``G`` store reference and edited
        allele counts, respectively.
    n_top_features : int, default=1000
        Number of highly variable A-to-I sites to keep after filtering.
    min_cells_with_G : int, default=3
        Minimum number of spots/cells with at least one edited read (``G > 0``) required
        for a site to pass filtering.
    alpha : float, default=0.5
        Pseudocount added to edited reads during beta-binomial-style smoothing.
    beta : float, default=0.5
        Pseudocount added to non-edited/reference reads through the denominator.

    Returns
    -------
    anndata.AnnData
        A new AnnData object whose ``X`` matrix stores smoothed editing levels for the
        selected highly variable A-to-I sites.
    """
    print("Preprocessing A-to-I data...")

    if "A" not in adata_ai.layers or "G" not in adata_ai.layers:
        raise ValueError("adata_ai must contain layers named 'A' and 'G'.")

    A = adata_ai.layers["A"]
    G = adata_ai.layers["G"]

    if sp.issparse(A):
        A = A.toarray()
    if sp.issparse(G):
        G = G.toarray()

    cells_with_G = np.sum(G > 0, axis=0)
    keep_sites = cells_with_G >= min_cells_with_G
    n_kept = int(np.sum(keep_sites))

    print(
        f"  Filtering sites: {adata_ai.n_vars} -> {n_kept} "
        f"(min_cells_with_G={min_cells_with_G})"
    )

    if n_kept == 0:
        raise ValueError("No A-to-I sites passed filtering.")

    n_top_features_use = min(n_top_features, n_kept)
    if n_kept < n_top_features:
        print(f"  Warning: only {n_kept} sites passed filtering; using all retained sites.")

    A = A[:, keep_sites]
    G = G[:, keep_sites]
    var_filtered = adata_ai.var.iloc[keep_sites].copy()

    coverage = A + G
    editing_level = (G + alpha) / (coverage + alpha + beta)

    adata_processed = ad.AnnData(X=editing_level)
    adata_processed.obs = adata_ai.obs.copy()
    adata_processed.var = var_filtered

    print(f"  Selecting top {n_top_features_use} highly variable A-to-I sites...")
    sc.pp.highly_variable_genes(
        adata_processed,
        n_top_genes=n_top_features_use,
        flavor="seurat",
        subset=True,
    )

    print("  Scaling A-to-I editing levels with Z-score scaling...")
    sc.pp.scale(adata_processed, max_value=10)

    print(f"Final A-to-I shape: {adata_processed.shape}")
    return adata_processed


def construct_interaction(adata, n_neighbors=3):
    """
    Construct a spatial nearest-neighbor graph using pairwise Euclidean distances.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with spatial coordinates stored in ``adata.obsm['spatial']``.
    n_neighbors : int, default=3
        Number of nearest spatial neighbors to connect for each spot/cell.

    Returns
    -------
    None
        The input AnnData object is modified in place. The function stores the pairwise
        distance matrix in ``adata.obsm['distance_matrix']``, the directed neighbor graph
        in ``adata.obsm['graph_neigh']``, and the symmetrized adjacency matrix in
        ``adata.obsm['adj']``.
    """
    position = adata.obsm["spatial"]
    distances = ot.dist(position, position, metric="euclidean")
    n_spot = distances.shape[0]

    adata.obsm["distance_matrix"] = distances
    interaction = np.zeros((n_spot, n_spot))

    for i in range(n_spot):
        neighbor_order = distances[i, :].argsort()
        for t in range(1, n_neighbors + 1):
            interaction[i, neighbor_order[t]] = 1

    adata.obsm["graph_neigh"] = interaction
    adata.obsm["adj"] = np.where(interaction + interaction.T > 1, 1, interaction + interaction.T)


def construct_interaction_KNN(adata, n_neighbors=3):
    """
    Construct a spatial KNN graph using scikit-learn's nearest-neighbor search.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with spatial coordinates stored in ``adata.obsm['spatial']``.
    n_neighbors : int, default=3
        Number of nearest spatial neighbors to connect for each spot/cell.

    Returns
    -------
    None
        The input AnnData object is modified in place. The function stores the directed
        neighbor graph in ``adata.obsm['graph_neigh']`` and the symmetrized adjacency
        matrix in ``adata.obsm['adj']``.
    """
    position = adata.obsm["spatial"]
    n_spot = position.shape[0]

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(position)
    _, indices = nbrs.kneighbors(position)

    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()

    interaction = np.zeros((n_spot, n_spot))
    interaction[x, y] = 1

    adata.obsm["graph_neigh"] = interaction
    adata.obsm["adj"] = np.where(interaction + interaction.T > 1, 1, interaction + interaction.T)


def preprocess_adj(adj):
    """
    Normalize an adjacency matrix and add self-loops.

    Parameters
    ----------
    adj : array-like or scipy.sparse matrix, shape (n_nodes, n_nodes)
        Input adjacency matrix.

    Returns
    -------
    numpy.ndarray
        Symmetrically normalized adjacency matrix with identity self-loops added.
    """
    return normalize_adj(adj) + np.eye(adj.shape[0])


def normalize_adj(adj):
    """
    Symmetrically normalize an adjacency matrix as ``D^-1/2 A D^-1/2``.

    Parameters
    ----------
    adj : array-like or scipy.sparse matrix, shape (n_nodes, n_nodes)
        Input adjacency matrix.

    Returns
    -------
    numpy.ndarray
        Dense normalized adjacency matrix.
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a SciPy sparse matrix to a PyTorch sparse tensor.

    Parameters
    ----------
    sparse_mx : scipy.sparse matrix
        Sparse matrix to convert.

    Returns
    -------
    torch.Tensor
        PyTorch sparse COO tensor with ``float32`` values.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def preprocess_adj_sparse(adj):
    """
    Normalize an adjacency matrix with self-loops and return a PyTorch sparse tensor.

    Parameters
    ----------
    adj : array-like or scipy.sparse matrix, shape (n_nodes, n_nodes)
        Input adjacency matrix.

    Returns
    -------
    torch.Tensor
        Sparse PyTorch tensor storing the normalized adjacency matrix.
    """
    adj = sp.coo_matrix(adj)
    adj_with_self_loop = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_with_self_loop.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = (
        adj_with_self_loop.dot(degree_mat_inv_sqrt)
        .transpose()
        .dot(degree_mat_inv_sqrt)
        .tocoo()
    )
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def permutation(feature):
    """
    Randomly permute rows of a feature matrix.

    Parameters
    ----------
    feature : numpy.ndarray
        Feature matrix with samples/spots/cells in rows.

    Returns
    -------
    numpy.ndarray
        Row-permuted feature matrix with the same shape as the input.
    """
    ids = np.random.permutation(np.arange(feature.shape[0]))
    return feature[ids]


def add_contrastive_label(adata):
    """
    Add labels for contrastive self-supervised learning.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object whose number of observations defines the label matrix size.

    Returns
    -------
    None
        The input AnnData object is modified in place by adding ``adata.obsm['label_CSL']``.
        The first column indicates positive samples and the second column indicates
        negative/permuted samples.
    """
    n_spot = adata.n_obs
    one_matrix = np.ones((n_spot, 1))
    zero_matrix = np.zeros((n_spot, 1))
    adata.obsm["label_CSL"] = np.concatenate([one_matrix, zero_matrix], axis=1)


def get_feature(adata, deconvolution=False):
    """
    Extract model input features and a row-permuted augmented feature matrix.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing expression features in ``adata.X``. If
        ``deconvolution=False``, ``adata.var['highly_variable']`` must exist.
    deconvolution : bool, default=False
        If ``True``, use all features in ``adata.X``. If ``False``, use only highly
        variable features.

    Returns
    -------
    None
        The input AnnData object is modified in place by adding ``adata.obsm['feat']``
        and ``adata.obsm['feat_a']``.
    """
    if deconvolution:
        adata_vars = adata
    else:
        if "highly_variable" not in adata.var:
            raise ValueError("adata.var must contain 'highly_variable' when deconvolution=False.")
        adata_vars = adata[:, adata.var["highly_variable"]]

    if isinstance(adata_vars.X, (csc_matrix, csr_matrix)):
        feat = adata_vars.X.toarray()
    else:
        feat = np.asarray(adata_vars.X)

    adata.obsm["feat"] = feat
    adata.obsm["feat_a"] = permutation(feat)


def filter_with_overlap_gene(adata, adata_sc):
    """
    Keep highly variable genes that are shared between spatial and scRNA-seq data.

    Parameters
    ----------
    adata : anndata.AnnData
        Spatial transcriptomics AnnData object. It must contain
        ``adata.var['highly_variable']``.
    adata_sc : anndata.AnnData
        Single-cell RNA-seq AnnData object. It must contain
        ``adata_sc.var['highly_variable']``.

    Returns
    -------
    tuple[anndata.AnnData, anndata.AnnData]
        Filtered spatial and single-cell AnnData objects restricted to the same ordered
        set of overlapping highly variable genes. The shared gene list is also stored in
        ``.uns['overlap_genes']`` for both objects.
    """
    if "highly_variable" not in adata.var:
        raise ValueError("adata.var must contain 'highly_variable'.")
    adata = adata[:, adata.var["highly_variable"]]

    if "highly_variable" not in adata_sc.var:
        raise ValueError("adata_sc.var must contain 'highly_variable'.")
    adata_sc = adata_sc[:, adata_sc.var["highly_variable"]]

    genes = sorted(set(adata.var_names) & set(adata_sc.var_names))
    print("Number of overlap genes:", len(genes))

    adata.uns["overlap_genes"] = genes
    adata_sc.uns["overlap_genes"] = genes

    return adata[:, genes], adata_sc[:, genes]


def _load_clustering_helpers():
    """
    Load clustering helper functions used by ``sub_clustering``.

    Parameters
    ----------
    None

    Returns
    -------
    tuple[callable, callable, callable]
        ``mclust_R``, ``search_res``, and ``refine_label`` helper functions imported from
        the local ``utils`` module.
    """
    try:
        from .utils import mclust_R, refine_label, search_res
    except ImportError:
        from utils import mclust_R, refine_label, search_res

    return mclust_R, search_res, refine_label


def sub_clustering(
    adata,
    target_class,
    n_sub_clusters,
    key="domain",
    method="mclust",
    radius=None,
    refinement=True,
    resolution=None,
    n_neighbors=10,
    n_components=20,
    spatial_weight=0.0,
):
    """
    Perform sub-clustering within one selected major cluster.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing global clustering labels in ``adata.obs[key]`` and
        feature embeddings in ``adata.obsm['emb']``.
    target_class : str or int
        Major cluster label to subdivide.
    n_sub_clusters : int
        Target number of sub-clusters within ``target_class``.
    key : str, default='domain'
        Column in ``adata.obs`` that stores the major cluster labels.
    method : {'mclust', 'leiden'}, default='mclust'
        Clustering method used for sub-clustering. If ``mclust`` fails, the function
        falls back to Leiden clustering.
    radius : int or None, default=None
        Number of spatial neighbors used for label refinement. If ``None`` and
        ``refinement=True``, the function uses ``radius=10``.
    refinement : bool, default=True
        Whether to smooth sub-cluster labels using local spatial neighborhoods.
    resolution : float or None, default=None
        Leiden resolution. If ``None``, the function searches for a resolution that
        produces ``n_sub_clusters`` clusters.
    n_neighbors : int, default=10
        Number of neighbors for constructing the Leiden graph.
    n_components : int, default=20
        Number of PCA components for sub-clustering.
    spatial_weight : float, default=0.0
        Weight for injecting normalized spatial coordinates into the feature matrix. Use
        ``0.0`` for feature-only clustering; use values greater than zero to encourage
        spatially coherent sub-clusters.

    Returns
    -------
    anndata.AnnData
        The input AnnData object with updated labels in ``adata.obs[f'{key}_refined']``.
    """
    mclust_R, search_res, refine_label = _load_clustering_helpers()

    print(
        f"\n>>> Starting sub-clustering on cluster '{target_class}' "
        f"(target: {n_sub_clusters} sub-clusters)..."
    )

    subset_mask = adata.obs[key].astype(str) == str(target_class)
    if np.sum(subset_mask) == 0:
        print(f"Error: cluster '{target_class}' was not found in column '{key}'.")
        return adata

    adata_sub = adata[subset_mask].copy()
    print(f"    - Subset contains {adata_sub.n_obs} cells/spots.")

    if adata_sub.n_obs < n_sub_clusters * 5:
        print("Warning: too few cells/spots; skipping sub-clustering.")
        return adata

    X_data = adata_sub.obsm["emb"].copy()

    if spatial_weight > 0:
        print(f"    - Injecting spatial prior (weight={spatial_weight})...")
        X_spatial = adata_sub.obsm["spatial"].copy()
        X_spatial = StandardScaler().fit_transform(X_spatial)
        X_data = np.concatenate([X_data, X_spatial * spatial_weight], axis=1)
    else:
        print("    - Using feature embeddings only.")

    n_comps = min(n_components, X_data.shape[1], X_data.shape[0] - 1)
    adata_sub.obsm["emb_pca_sub"] = PCA(n_components=n_comps, random_state=42).fit_transform(X_data)

    sub_labels = None

    if method == "mclust":
        try:
            adata_sub = mclust_R(adata_sub, num_cluster=n_sub_clusters, used_obsm="emb_pca_sub")
            sub_labels = adata_sub.obs["mclust"].astype(str).values
        except Exception as exc:
            print(f"    - Mclust failed: {exc}. Switching to Leiden.")
            method = "leiden"

    if method == "leiden":
        if resolution is not None:
            print(f"    - Using provided Leiden resolution: {resolution}")
            res = resolution
            sc.pp.neighbors(adata_sub, n_neighbors=n_neighbors, use_rep="emb_pca_sub")
        else:
            res = search_res(
                adata_sub,
                n_clusters=n_sub_clusters,
                method="leiden",
                use_rep="emb_pca_sub",
                start=0.1,
                end=2.0,
                increment=0.05,
            )

        sc.tl.leiden(adata_sub, resolution=res)
        sub_labels = adata_sub.obs["leiden"].astype(str).values

    if sub_labels is None:
        raise RuntimeError(f"Unsupported clustering method: {method}")

    unique_labels = np.unique(sub_labels)
    if len(unique_labels) < 2:
        print(
            f"Warning: sub-clustering produced only {len(unique_labels)} class; "
            "keeping original labels."
        )
        return adata

    print(f"    - Preliminary split labels: {unique_labels}")

    if refinement:
        if radius is None:
            radius = 10
        print(f"    - Refining sub-cluster labels with radius={radius}...")

        adata_sub.obs["temp_sub"] = sub_labels
        refined_labels = refine_label(adata_sub, radius=radius, key="temp_sub")

        if len(set(refined_labels)) < len(unique_labels):
            print(
                "Warning: the number of classes decreased after refinement. "
                f"The radius ({radius}) may be too large."
            )

        sub_labels = np.asarray(refined_labels)

    target_col = f"{key}_refined"
    if target_col not in adata.obs.columns:
        adata.obs[target_col] = adata.obs[key].astype(str).values

    new_labels = adata.obs[target_col].astype(str).values
    final_sub_labels = [f"{target_class}_{label}" for label in sub_labels]
    new_labels[subset_mask] = final_sub_labels

    adata.obs[target_col] = new_labels.astype(str)
    adata.obs[target_col] = adata.obs[target_col].astype("category")

    unique, counts = np.unique(new_labels, return_counts=True)
    stats = dict(zip(unique, counts))
    related_stats = {k: v for k, v in stats.items() if str(k).startswith(f"{target_class}_")}

    print(f"    - Final split statistics for class {target_class}: {related_stats}")
    print(f"Done. Results were saved in '{target_col}'.")

    return adata


def extract_boundary_as_new_cluster(
    adata,
    cluster_A,
    cluster_B,
    key="domain",
    width=5,
    new_label="New_Layer",
    show_plot=True,
    selection_mode="all",
    min_component_size=10,
):
    """
    Identify the spatial boundary between two clusters and assign it a new label.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with spatial coordinates in ``adata.obsm['spatial']`` and cluster
        labels in ``adata.obs[key]``.
    cluster_A : str or int
        First cluster label used to define the boundary.
    cluster_B : str or int
        Second cluster label used to define the boundary.
    key : str, default='domain'
        Column in ``adata.obs`` that stores the cluster labels.
    width : int, default=5
        Number of nearest neighbors inspected when deciding whether a spot/cell lies on
        the boundary. Larger values produce a wider boundary.
    new_label : str, default='New_Layer'
        Label assigned to the detected boundary spots/cells.
    show_plot : bool, default=True
        Whether to display a spatial plot of the refined labels.
    selection_mode : {'all', 'largest'}, default='all'
        Boundary component selection strategy. ``'all'`` keeps all sufficiently large
        connected boundary components; ``'largest'`` keeps only the largest component.
    min_component_size : int, default=10
        Minimum connected-component size retained when ``selection_mode='all'``.

    Returns
    -------
    anndata.AnnData
        The input AnnData object with a new or updated label column named
        ``adata.obs[f'{key}_refined']``.
    """
    print(f"[Advanced] Mining boundary between cluster {cluster_A} and cluster {cluster_B}...")
    print(f"    - Strategy: keep {selection_mode} component(s).")

    if selection_mode not in {"largest", "all"}:
        raise ValueError("selection_mode must be either 'largest' or 'all'.")

    spatial = adata.obsm["spatial"]
    labels = adata.obs[key].astype(str).values

    nbrs = NearestNeighbors(n_neighbors=width * 3).fit(spatial)
    distances, indices = nbrs.kneighbors(spatial)
    median_spacing = np.median(distances[:, 1])

    boundary_mask = np.zeros(adata.n_obs, dtype=bool)
    cluster_A = str(cluster_A)
    cluster_B = str(cluster_B)

    for i in range(adata.n_obs):
        current_label = labels[i]
        if current_label not in {cluster_A, cluster_B}:
            continue

        neighbor_indices = indices[i, 1 : width + 1]
        neighbor_labels = labels[neighbor_indices]
        neighbor_dists = distances[i, 1 : width + 1]

        is_boundary = (
            current_label == cluster_A
            and cluster_B in neighbor_labels
        ) or (
            current_label == cluster_B
            and cluster_A in neighbor_labels
        )

        if is_boundary and np.mean(neighbor_dists) > median_spacing * 3.0:
            is_boundary = False

        if is_boundary:
            boundary_mask[i] = True

    n_raw = int(np.sum(boundary_mask))
    if n_raw > 0:
        sub_idx = np.where(boundary_mask)[0]
        sub_coords = spatial[sub_idx]

        k_conn = min(6, len(sub_idx) - 1)
        if k_conn > 0:
            nbrs_sub = NearestNeighbors(n_neighbors=k_conn).fit(sub_coords)
            graph = nbrs_sub.kneighbors_graph(sub_coords, mode="connectivity")

            n_cc, labels_cc = connected_components(graph)
            component_sizes = np.bincount(labels_cc)

            print(f"    - Found {n_cc} disconnected boundary segment(s).")

            if selection_mode == "largest":
                largest_cc_label = np.argmax(component_sizes)
                keep_local_mask = labels_cc == largest_cc_label
            else:
                valid_labels = np.where(component_sizes >= min_component_size)[0]
                keep_local_mask = np.isin(labels_cc, valid_labels)

                dropped = n_cc - len(valid_labels)
                if dropped > 0:
                    print(
                        f"    - Dropped {dropped} small noise segment(s) "
                        f"(< {min_component_size} cells/spots)."
                    )

            noise_indices = sub_idx[~keep_local_mask]
            boundary_mask[noise_indices] = False

    n_found = int(np.sum(boundary_mask))
    print(f"    - Finalized {n_found} valid boundary cells/spots.")

    if n_found == 0:
        print("Warning: no boundary was found.")
        return adata

    new_col_name = f"{key}_refined"
    new_labels = labels.copy()
    new_labels[boundary_mask] = new_label

    adata.obs[new_col_name] = new_labels
    adata.obs[new_col_name] = adata.obs[new_col_name].astype("category")

    if show_plot:
        plt.rcParams["figure.figsize"] = (6, 6)
        sc.pl.spatial(
            adata,
            color=new_col_name,
            title=f"Extracted '{new_label}' (mode: {selection_mode})",
            spot_size=50,
            palette="tab20",
            frameon=False,
            show=True,
        )

    print(f"Done. Results were saved in column '{new_col_name}'.")
    return adata
