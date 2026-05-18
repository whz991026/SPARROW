"""Utility functions for spatial clustering, deconvolution projection, and evaluation.

This module keeps only the imports required by the functions below. Optional
R-related dependencies are imported lazily inside ``mclust_R`` so that the rest
of the utilities can still be used when rpy2 or the R package mclust is absent.
"""

import os

import numpy as np
import pandas as pd
import scanpy as sc
import ot
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


def mclust_R(adata, num_cluster, modelNames="EEE", used_obsm="emb_pca", random_seed=2020):
    """Cluster observations using the R ``mclust`` package through rpy2.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing the input embedding in ``adata.obsm[used_obsm]``.
    num_cluster : int
        Target number of mixture-model clusters.
    modelNames : str, default="EEE"
        Gaussian mixture covariance model name passed to ``mclust::Mclust``.
        If this model fails, the function retries with automatic model selection.
    used_obsm : str, default="emb_pca"
        Key in ``adata.obsm`` containing the feature matrix used for clustering.
    random_seed : int, default=2020
        Random seed used for both NumPy and R.

    Returns
    -------
    anndata.AnnData
        The input object with ``adata.obs['mclust']`` added as categorical labels.
    """
    try:
        import rpy2.robjects as robjects
        import rpy2.robjects.numpy2ri as numpy2ri
        from rpy2.robjects.packages import importr
    except ImportError as exc:
        raise ImportError(
            "mclust_R requires rpy2 and the R package 'mclust'. "
            "Install them before using method='mclust'."
        ) from exc

    np.random.seed(random_seed)
    numpy2ri.activate()
    mclust_pkg = importr("mclust")
    robjects.r["set.seed"](random_seed)

    if used_obsm not in adata.obsm:
        raise KeyError(f"adata.obsm does not contain '{used_obsm}'.")

    data_np = np.ascontiguousarray(adata.obsm[used_obsm], dtype=np.float64)
    if np.isnan(data_np).any():
        raise ValueError("Input data contains NaNs; Mclust cannot proceed.")

    try:
        res = mclust_pkg.Mclust(data_np, G=num_cluster, modelNames=modelNames, verbose=False)
    except Exception as exc:
        print(f"R execution error: {exc}")
        res = None

    if res is None or isinstance(res, robjects.rinterface.NULLType):
        print(f"Mclust returned NULL with modelNames='{modelNames}'. Retrying with automatic model selection...")
        try:
            res = mclust_pkg.Mclust(data_np, G=num_cluster, verbose=False)
        except Exception:
            res = None

        if res is None or isinstance(res, robjects.rinterface.NULLType):
            raise RuntimeError(
                "Mclust failed to converge. The data may be singular or too high-dimensional."
            )

    adata.obs["mclust"] = np.array(res.rx2("classification")).astype(int).astype("category")
    return adata


def search_res(adata, n_clusters, method="leiden", use_rep="emb_pca", start=0.1, end=3.0, increment=0.01):
    """Find a Leiden resolution that produces the requested number of clusters.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing the representation used for neighbor graph construction.
    n_clusters : int
        Target number of clusters.
    method : {"leiden"}, default="leiden"
        Clustering method used during resolution search. Currently only Leiden is supported.
    use_rep : str, default="emb_pca"
        Key in ``adata.obsm`` used by ``scanpy.pp.neighbors``.
    start : float, default=0.1
        Lower bound of the resolution search range.
    end : float, default=3.0
        Upper bound of the resolution search range. The value is exclusive, following
        ``np.arange`` behavior.
    increment : float, default=0.01
        Step size between candidate resolutions.

    Returns
    -------
    float
        The first resolution, searched from high to low, that gives ``n_clusters`` clusters.
    """
    if method != "leiden":
        raise ValueError("search_res currently supports only method='leiden'.")
    if use_rep not in adata.obsm:
        raise KeyError(f"adata.obsm does not contain '{use_rep}'.")

    print(f"  [Search] Linear reverse search for k={n_clusters}...")
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)

    for res in sorted(np.arange(start, end, increment), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=float(res))
        if adata.obs["leiden"].nunique() == n_clusters:
            return float(res)

    raise ValueError(
        f"Resolution not found for k={n_clusters} in range [{start}, {end}). "
        "Try a wider range or a smaller increment."
    )


def refine_label(adata, radius=50, key="label"):
    """Smooth cluster labels using the majority label among spatial neighbors.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with spatial coordinates in ``adata.obsm['spatial']`` and labels
        in ``adata.obs[key]``.
    radius : int, default=50
        Number of nearest spatial neighbors used for majority voting. This is a
        neighbor count, not a physical distance threshold.
    key : str, default="label"
        Column in ``adata.obs`` containing the labels to smooth.

    Returns
    -------
    numpy.ndarray
        Smoothed labels as strings, ordered to match ``adata.obs_names``.
    """
    if key not in adata.obs:
        raise KeyError(f"adata.obs does not contain '{key}'.")
    if "spatial" not in adata.obsm:
        raise KeyError("adata.obsm does not contain 'spatial'.")

    labels = adata.obs[key].values
    positions = adata.obsm["spatial"]
    distances = ot.dist(positions, positions, metric="euclidean")
    n_cells = distances.shape[0]
    n_neighbors = min(int(radius), n_cells - 1)

    refined = []
    for i in range(n_cells):
        neighbor_idx = distances[i, :].argsort()[1 : n_neighbors + 1]
        neighbor_labels = labels[neighbor_idx]
        majority_label = max(neighbor_labels, key=list(neighbor_labels).count)
        refined.append(majority_label)

    return np.array([str(label) for label in refined])


def clustering_auto(
    adata,
    n_clusters=7,
    radius=50,
    n_components=(20,),
    key="emb",
    method="leiden",
    eval_metric="sc",
    refinement=True,
    start=0.1,
    end=3.0,
    increment=0.01,
):
    """Automatically choose PCA dimensionality for spatial clustering.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing an embedding in ``adata.obsm[key]``.
    n_clusters : int, default=7
        Target number of clusters.
    radius : int, default=50
        Number of nearest spatial neighbors used for optional label refinement.
    n_components : int or sequence of int, default=(20,)
        Candidate PCA dimensions to evaluate. If an integer is provided, only that
        value is tested.
    key : str, default="emb"
        Key in ``adata.obsm`` containing the input embedding.
    method : {"leiden", "mclust"}, default="leiden"
        Clustering method used after PCA.
    eval_metric : {"sc", "ch", "ari", "nmi"}, default="sc"
        Metric used to select the best number of PCA components. ``ari`` and ``nmi``
        require ``adata.obs['ground_truth']``.
    refinement : bool, default=True
        Whether to apply spatial majority-vote label refinement before evaluation.
    start : float, default=0.1
        Lower bound for Leiden resolution search.
    end : float, default=3.0
        Upper bound for Leiden resolution search.
    increment : float, default=0.01
        Step size for Leiden resolution search.

    Returns
    -------
    tuple
        ``(adata, best_n_components)`` where ``adata.obs['domain']`` stores the best labels.
    """
    if key not in adata.obsm:
        raise KeyError(f"adata.obsm does not contain '{key}'.")
    if method not in {"leiden", "mclust"}:
        raise ValueError("method must be either 'leiden' or 'mclust'.")
    if eval_metric not in {"sc", "ch", "ari", "nmi"}:
        raise ValueError("eval_metric must be one of {'sc', 'ch', 'ari', 'nmi'}.")

    if isinstance(n_components, int):
        n_components_list = [n_components]
    else:
        n_components_list = list(n_components)

    if eval_metric in {"ari", "nmi"} and "ground_truth" not in adata.obs.columns:
        print("Warning: 'ground_truth' not found. Falling back to silhouette score ('sc').")
        eval_metric = "sc"

    best_score = -np.inf
    best_n_comp = None
    best_labels = None
    x_raw = adata.obsm[key].copy()

    print(f"Starting auto-clustering (method={method}, metric={eval_metric})")

    for n_comp in n_components_list:
        curr_n = min(int(n_comp), x_raw.shape[1])
        adata.obsm["emb_pca"] = PCA(n_components=curr_n, random_state=42).fit_transform(x_raw)
        leiden_res = None

        try:
            if method == "leiden":
                leiden_res = search_res(
                    adata,
                    n_clusters,
                    method="leiden",
                    use_rep="emb_pca",
                    start=start,
                    end=end,
                    increment=increment,
                )
                sc.tl.leiden(adata, random_state=0, resolution=leiden_res)
                labels = adata.obs["leiden"].values.copy()
            else:
                adata = mclust_R(adata, num_cluster=n_clusters, used_obsm="emb_pca")
                labels = adata.obs["mclust"].values.copy()

            if refinement:
                adata.obs["temp_refine"] = labels
                labels = refine_label(adata, radius=radius, key="temp_refine")

            n_found = len(np.unique(labels))
            if n_found < 2:
                print(f"  [n={n_comp}] Failed: only one cluster found.")
                continue

            if eval_metric == "sc":
                idx = np.random.choice(x_raw.shape[0], min(10000, x_raw.shape[0]), replace=False)
                score = silhouette_score(x_raw[idx], labels[idx])
            elif eval_metric == "ch":
                score = calinski_harabasz_score(x_raw, labels)
            else:
                df_eval = pd.DataFrame({"pred": labels, "gt": adata.obs["ground_truth"].values})
                df_eval = df_eval[df_eval["gt"].notna() & (df_eval["gt"] != "Unknown")]
                if df_eval.empty:
                    score = 0.0
                elif eval_metric == "ari":
                    score = adjusted_rand_score(df_eval["gt"], df_eval["pred"])
                else:
                    score = normalized_mutual_info_score(df_eval["gt"], df_eval["pred"])

            res_text = f"{leiden_res:.4f}" if leiden_res is not None else "N/A"
            print(f"  [n={n_comp}] Res={res_text} | K={n_found} | {eval_metric.upper()}={score:.4f}")

            if score > best_score:
                best_score = score
                best_n_comp = int(n_comp)
                best_labels = labels

        except Exception as exc:
            print(f"  [n={n_comp}] Skipped: {exc}")

    if best_labels is None:
        raise RuntimeError("Auto-clustering failed for all n_components. Check the data and resolution range.")

    print(f"\n>>> Winner: n_components={best_n_comp}, {eval_metric.upper()}={best_score:.4f}")
    adata.obs["domain"] = best_labels
    adata.obsm["emb_pca"] = PCA(n_components=min(best_n_comp, x_raw.shape[1]), random_state=42).fit_transform(x_raw)
    return adata, best_n_comp


def clustering(
    adata,
    n_clusters=7,
    radius=50,
    n_components=20,
    key="emb",
    method="mclust",
    start=0.1,
    end=1.0,
    increment=0.01,
    refinement=False,
    spatial_weight=0.0,
):
    """Run spatial clustering with optional spatial-coordinate injection and refinement.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing an embedding in ``adata.obsm[key]``. If
        ``spatial_weight > 0`` or ``refinement=True``, it must also contain
        ``adata.obsm['spatial']``.
    n_clusters : int, default=7
        Target number of clusters.
    radius : int, default=50
        Number of nearest spatial neighbors used for optional label refinement.
    n_components : int, default=20
        Number of PCA components used before clustering.
    key : str, default="emb"
        Key in ``adata.obsm`` containing the input embedding.
    method : {"mclust", "leiden"}, default="mclust"
        Clustering method. ``mclust`` requires rpy2 and the R package mclust.
    start : float, default=0.1
        Lower bound for Leiden resolution search.
    end : float, default=1.0
        Upper bound for Leiden resolution search.
    increment : float, default=0.01
        Step size for Leiden resolution search.
    refinement : bool, default=False
        Whether to smooth labels using spatial nearest-neighbor majority voting.
    spatial_weight : float, default=0.0
        Weight for normalized spatial coordinates appended to the embedding. Use
        ``0.0`` for expression-only clustering; values above zero encourage spatially
        smoother domains.

    Returns
    -------
    anndata.AnnData
        The input object with ``adata.obs['domain']`` added.
    """
    if key not in adata.obsm:
        raise KeyError(f"adata.obsm does not contain '{key}'.")
    if method not in {"mclust", "leiden"}:
        raise ValueError("method must be either 'mclust' or 'leiden'.")

    x_data = adata.obsm[key].copy()

    if spatial_weight > 0:
        if "spatial" not in adata.obsm:
            raise KeyError("adata.obsm does not contain 'spatial'.")
        print(f"Running clustering with spatial prior (weight={spatial_weight})...")
        x_spatial = StandardScaler().fit_transform(adata.obsm["spatial"].copy())
        x_data = np.concatenate([x_data, x_spatial * spatial_weight], axis=1)
    else:
        print("Running standard clustering...")

    if x_data.shape[1] > n_components:
        embedding = PCA(n_components=n_components, random_state=42).fit_transform(x_data)
    else:
        embedding = x_data
    adata.obsm["emb_pca"] = embedding

    if method == "mclust":
        adata = mclust_R(adata, used_obsm="emb_pca", num_cluster=n_clusters)
        adata.obs["domain"] = adata.obs["mclust"]
    else:
        leiden_res = search_res(
            adata,
            n_clusters,
            use_rep="emb_pca",
            method="leiden",
            start=start,
            end=end,
            increment=increment,
        )
        sc.tl.leiden(adata, random_state=0, resolution=leiden_res)
        adata.obs["domain"] = adata.obs["leiden"]

    if refinement:
        print(f"Applying spatial label refinement with radius={radius}...")
        adata.obs["domain"] = refine_label(adata, radius=radius, key="domain")

    return adata


def extract_top_value(map_matrix, retain_percent=0.1):
    """Keep only the largest mapping values within each row.

    Parameters
    ----------
    map_matrix : numpy.ndarray
        Mapping matrix, usually with shape ``n_spots x n_cells``.
    retain_percent : float, default=0.1
        Fraction of values to retain in each row. For example, ``0.1`` keeps the
        top 10% entries per spot.

    Returns
    -------
    numpy.ndarray
        Matrix with low mapping values set to zero.
    """
    if not 0 < retain_percent <= 1:
        raise ValueError("retain_percent must be in the interval (0, 1].")

    k = max(1, int(retain_percent * map_matrix.shape[1]))
    threshold_rank = map_matrix.shape[1] - k
    return map_matrix * (np.argsort(np.argsort(map_matrix, axis=1), axis=1) >= threshold_rank)


def construct_cell_type_matrix(adata_sc):
    """Build a one-hot cell-type annotation matrix for single-cell data.

    Parameters
    ----------
    adata_sc : anndata.AnnData
        Single-cell AnnData object with ``adata_sc.obs['cell_type']``.

    Returns
    -------
    pandas.DataFrame
        One-hot matrix with cells as rows and sorted cell types as columns.
    """
    if "cell_type" not in adata_sc.obs:
        raise KeyError("adata_sc.obs does not contain 'cell_type'.")

    cell_types = adata_sc.obs["cell_type"].astype(str)
    cell_type_order = sorted(cell_types.unique())
    return pd.get_dummies(cell_types)[cell_type_order].astype(float)


def extract_top_k_and_renorm(matrix, k):
    """Keep the top-k values in each row and renormalize rows to sum to one.

    Parameters
    ----------
    matrix : numpy.ndarray
        Input matrix, usually with spots as rows and cells as columns.
    k : int
        Number of largest entries to keep per row.

    Returns
    -------
    numpy.ndarray
        Row-normalized matrix after top-k filtering.
    """
    if k < 1:
        raise ValueError("k must be at least 1.")

    k = min(int(k), matrix.shape[1])
    matrix_new = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        idx = np.argsort(matrix[i])[-k:]
        matrix_new[i, idx] = matrix[i, idx]
        matrix_new[i] /= matrix_new[i].sum() + 1e-8
    return matrix_new


def project_cell_to_spot(adata, adata_sc, retain_percent=0.1):
    """Project scRNA-seq cell-type composition onto spatial spots.

    Parameters
    ----------
    adata : anndata.AnnData
        Spatial AnnData object containing ``adata.obsm['map_matrix']`` with shape
        ``n_spots x n_cells``.
    adata_sc : anndata.AnnData
        Single-cell AnnData object containing ``adata_sc.obs['cell_type']``.
    retain_percent : float, default=0.1
        Fraction of highest-probability cells retained per spot before projection.

    Returns
    -------
    anndata.AnnData
        The input spatial object with projected cell-type proportions added to
        ``adata.obs`` as one column per cell type.
    """
    if "map_matrix" not in adata.obsm:
        raise KeyError("adata.obsm does not contain 'map_matrix'.")
    if "cell_type" not in adata_sc.obs:
        raise KeyError("adata_sc.obs does not contain 'cell_type'.")
    if not 0 < retain_percent <= 1:
        raise ValueError("retain_percent must be in the interval (0, 1].")

    map_matrix = np.nan_to_num(adata.obsm["map_matrix"])
    k = max(1, int(retain_percent * map_matrix.shape[1]))
    map_matrix = extract_top_k_and_renorm(map_matrix, k)

    cell_type_matrix = construct_cell_type_matrix(adata_sc)
    matrix_projection = map_matrix @ cell_type_matrix.values

    df_projection = pd.DataFrame(
        matrix_projection,
        index=adata.obs_names,
        columns=cell_type_matrix.columns,
    )
    df_projection = df_projection.div(df_projection.sum(axis=1).clip(lower=1e-6), axis=0)
    adata.obs[cell_type_matrix.columns] = df_projection
    return adata


def standardize_labels(labels):
    """Convert labels to an ordered categorical representation when possible.

    Parameters
    ----------
    labels : pandas.Series or array-like
        Cluster labels. Numeric labels are converted to strings and shifted from
        zero-based to one-based labels when the minimum value is 0.

    Returns
    -------
    pandas.Categorical
        Standardized categorical labels.
    """
    labels = pd.Series(labels)
    try:
        labels_int = labels.astype(int)
        if labels_int.min() == 0:
            labels_int = labels_int + 1
        labels_str = labels_int.astype(str)
        categories = sorted(labels_str.unique(), key=int)
        return pd.Categorical(labels_str, categories=categories, ordered=True)
    except (TypeError, ValueError):
        return labels.astype(str).astype("category")


def calculate_metrics(adata_obs, pred_df, pred_col, embedding=None):
    """Calculate clustering metrics against optional ground-truth labels.

    Parameters
    ----------
    adata_obs : pandas.DataFrame
        Observation metadata from an AnnData object. Must contain ``ground_truth``
        for ARI and NMI calculation.
    pred_df : pandas.DataFrame
        DataFrame containing predicted labels.
    pred_col : str
        Column in ``pred_df`` containing predicted labels.
    embedding : numpy.ndarray, optional
        Feature matrix aligned to ``adata_obs``. If provided, silhouette score and
        Calinski-Harabasz score are also calculated.

    Returns
    -------
    dict
        Dictionary with keys ``ARI``, ``NMI``, ``SC``, and ``CH``.
    """
    if "ground_truth" not in adata_obs:
        raise KeyError("adata_obs must contain 'ground_truth'.")
    if pred_col not in pred_df:
        raise KeyError(f"pred_df does not contain '{pred_col}'.")

    combined = pd.merge(
        adata_obs[["ground_truth"]],
        pred_df[[pred_col]],
        left_index=True,
        right_index=True,
        how="inner",
    )

    valid_gt = combined[combined["ground_truth"] != "Unknown"].dropna()
    if len(valid_gt) > 0:
        ari = adjusted_rand_score(valid_gt["ground_truth"], valid_gt[pred_col])
        nmi = normalized_mutual_info_score(valid_gt["ground_truth"], valid_gt[pred_col])
    else:
        ari, nmi = None, None

    sc_score, ch_score = None, None
    if embedding is not None:
        common_indices = combined.index.intersection(adata_obs.index)
        if len(common_indices) >= 2:
            try:
                valid_indices = [adata_obs.index.get_loc(idx) for idx in common_indices]
                emb_subset = embedding[valid_indices]
                labels_subset = combined.loc[common_indices, pred_col]
                if labels_subset.nunique() > 1:
                    sc_score = silhouette_score(emb_subset, labels_subset)
                    ch_score = calinski_harabasz_score(emb_subset, labels_subset)
            except Exception:
                sc_score, ch_score = None, None

    return {"ARI": ari, "NMI": nmi, "SC": sc_score, "CH": ch_score}


def find_best_in_group(file_list, adata, root_dir, sample_id, embedding=None):
    """Find the best clustering result file in a method group by ARI.

    Parameters
    ----------
    file_list : sequence of tuple[str, str]
        Sequence of ``(method_name, filename)`` pairs. Each file is expected under
        ``root_dir/sample_id/results/filename``.
    adata : anndata.AnnData
        AnnData object whose ``adata.obs['ground_truth']`` is used for evaluation.
    root_dir : str
        Root directory containing sample folders.
    sample_id : str
        Sample identifier used to construct the result-file path.
    embedding : numpy.ndarray, optional
        Embedding aligned to ``adata.obs`` for optional SC and CH metrics.

    Returns
    -------
    tuple
        ``(best_metrics, best_name, best_labels, best_filename)``.
    """
    best_ari = -1
    best_name = "None"
    best_labels = None
    best_filename = None
    best_metrics = {"ARI": None, "NMI": None, "SC": None, "CH": None}

    for method_name, filename in file_list:
        path = os.path.join(root_dir, sample_id, "results", filename)
        if not os.path.exists(path):
            continue

        try:
            df = pd.read_csv(path, index_col=0)
            pred_col = df.columns[0]
            metrics = calculate_metrics(
                adata_obs=adata.obs,
                pred_df=df,
                pred_col=pred_col,
                embedding=embedding,
            )
        except Exception:
            continue

        if metrics["ARI"] is not None and metrics["ARI"] > best_ari:
            best_ari = metrics["ARI"]
            best_name = method_name
            best_labels = df[pred_col]
            best_filename = filename
            best_metrics = metrics

    return best_metrics, best_name, best_labels, best_filename


def calculate_ari_strict(adata_obs, pred_df, pred_col):
    """Calculate ARI after strictly removing Unknown and missing ground-truth labels.

    Parameters
    ----------
    adata_obs : pandas.DataFrame
        Observation metadata containing ``ground_truth``.
    pred_df : pandas.DataFrame
        DataFrame containing predicted labels.
    pred_col : str
        Column in ``pred_df`` containing predicted labels.

    Returns
    -------
    float
        Adjusted Rand Index. Returns ``-1`` if no valid ground-truth labels remain.
    """
    if "ground_truth" not in adata_obs:
        raise KeyError("adata_obs must contain 'ground_truth'.")
    if pred_col not in pred_df:
        raise KeyError(f"pred_df does not contain '{pred_col}'.")

    combined = pd.merge(
        adata_obs[["ground_truth"]],
        pred_df[[pred_col]],
        left_index=True,
        right_index=True,
        how="inner",
    )
    combined = combined.replace("Unknown", np.nan).dropna()
    if len(combined) == 0:
        return -1
    return adjusted_rand_score(combined["ground_truth"], combined[pred_col])


def calculate_ari(adata, pred_col):
    """Calculate ARI between predicted labels and ``adata.obs['ground_truth']``.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing ``ground_truth`` and the prediction column in
        ``adata.obs``.
    pred_col : str
        Column in ``adata.obs`` containing predicted labels.

    Returns
    -------
    float
        Adjusted Rand Index. Returns ``0.0`` when ground truth or valid labels are absent.
    """
    if "ground_truth" not in adata.obs or pred_col not in adata.obs:
        return 0.0

    df = pd.DataFrame({"gt": adata.obs["ground_truth"], "pred": adata.obs[pred_col]})
    df = df[df["gt"] != "Unknown"].dropna()
    if len(df) == 0:
        return 0.0
    return adjusted_rand_score(df["gt"], df["pred"])
