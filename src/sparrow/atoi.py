# =============================================================================
# atoi.py
# Clean spatial A-to-I analysis utilities
#
# Workflow:
# 1. Reliable A-to-I site filtering
# 2. Compute global A-to-I score
# 3. Plot global A-to-I spatial pattern
# 4. Detect SV-A-to-I sites
# 5. Plot top SV-A-to-I single-site patterns
# 6. Add and plot ADAR-family spatial patterns
# 7. Analyze ADAR-family spatial correlation
# 8. Analyze global A-to-I vs ADAR-family expression
# 9. Bivariate co-localization
# 10. Analyze global A-to-I vs deconvolved cell types
# 11. Joint model: global A-to-I ~ cell types + ADAR + spatial covariates
# 12. Residual progression
# 13. Site-level ADAR correlation
# 14. Top site-ADAR co-localization examples
#
# Note:
# Deconvolution cell-type merging should be done in notebook before calling
# functions in this file.
# =============================================================================

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from scipy import sparse
from scipy.stats import kruskal, spearmanr, pearsonr
from statsmodels.stats.multitest import multipletests
from matplotlib.patches import Patch


_SPATIAL_IMAGE_CACHE = {}


# =============================================================================
# Basic helpers
# =============================================================================

def _as_csr(x):
    return x.tocsr() if sparse.issparse(x) else x


def _get_count_layers(adata_ai, a_layer: str = "A", g_layer: str = "G"):
    if a_layer not in adata_ai.layers:
        raise ValueError(f"adata_ai.layers[{a_layer!r}] not found")
    if g_layer not in adata_ai.layers:
        raise ValueError(f"adata_ai.layers[{g_layer!r}] not found")

    return _as_csr(adata_ai.layers[a_layer]), _as_csr(adata_ai.layers[g_layer])


def _dense_col(x, j: int) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x[:, j].toarray()).ravel()
    return np.asarray(x[:, j]).ravel()


def _dense_sum(x, axis: int) -> np.ndarray:
    return np.asarray(x.sum(axis=axis)).ravel()


def _coords_from_adata(adata) -> np.ndarray:
    if "spatial" in adata.obsm:
        return np.asarray(adata.obsm["spatial"], dtype=float)

    for cols in (("x_pixel", "y_pixel"), ("array_col", "array_row"), ("x", "y")):
        if set(cols).issubset(adata.obs.columns):
            return adata.obs.loc[:, list(cols)].to_numpy(dtype=float)

    raise ValueError(
        "No spatial coordinates found. Need adata.obsm['spatial'] "
        "or obs columns such as x_pixel/y_pixel."
    )


def set_spatial_background(
    adata,
    image_path: str,
    swap_x_y: bool = False,
    coordinate_scale: float = 1.0,
    crop: bool = True,
    crop_pad: float = 120,
    image_alpha: float = 0.78,
    crop_quantile: Tuple[float, float] = (0.5, 99.5),
):
    """
    Store H&E background settings used by all spatial plotting helpers.

    swap_x_y:
        Swap spot x/y coordinates before plotting. Useful when image and
        tissue_positions coordinates are transposed for a sample.

    coordinate_scale:
        Multiply spatial coordinates before plotting. For 10x hires images,
        this is usually scalefactors_json['tissue_hires_scalef'].

    crop:
        Show only the image region covered by spots, padded by crop_pad pixels.
    """
    image_path = str(image_path)

    if not Path(image_path).exists():
        warnings.warn(f"Spatial background image not found: {image_path}")

    adata.uns["spatial_background"] = {
        "image_path": image_path,
        "swap_x_y": bool(swap_x_y),
        "coordinate_scale": float(coordinate_scale),
        "crop": bool(crop),
        "crop_pad": float(crop_pad),
        "image_alpha": float(image_alpha),
        "crop_quantile": tuple(crop_quantile),
    }

    return adata


def _spatial_plot_config(adata) -> dict:
    cfg = adata.uns.get("spatial_background", {})
    return cfg if isinstance(cfg, dict) else {}


def _plot_coords_from_adata(adata) -> np.ndarray:
    coords = _coords_from_adata(adata).astype(float, copy=True)
    cfg = _spatial_plot_config(adata)

    coords *= float(cfg.get("coordinate_scale", 1.0))

    if cfg.get("swap_x_y", False):
        coords = coords[:, [1, 0]]

    return coords


def _load_spatial_image(image_path: str):
    image_path = str(image_path)

    if image_path not in _SPATIAL_IMAGE_CACHE:
        _SPATIAL_IMAGE_CACHE[image_path] = plt.imread(image_path)

    return _SPATIAL_IMAGE_CACHE[image_path]


def _draw_spatial_background(ax, adata, x, y) -> bool:
    cfg = _spatial_plot_config(adata)
    image_path = cfg.get("image_path")

    if not image_path:
        return False

    if not Path(image_path).exists():
        warnings.warn(f"Spatial background image not found: {image_path}")
        return False

    img = _load_spatial_image(image_path)
    height, width = img.shape[:2]

    ax.imshow(
        img,
        extent=(0, width, height, 0),
        origin="upper",
        alpha=cfg.get("image_alpha", 0.78),
        zorder=0,
    )

    finite = np.isfinite(x) & np.isfinite(y)

    if cfg.get("crop", True) and finite.any():
        q_low, q_high = cfg.get("crop_quantile", (0.5, 99.5))
        pad = float(cfg.get("crop_pad", 120))

        xmin, xmax = np.nanpercentile(x[finite], [q_low, q_high])
        ymin, ymax = np.nanpercentile(y[finite], [q_low, q_high])

        xmin = max(0, xmin - pad)
        xmax = min(width, xmax + pad)
        ymin = max(0, ymin - pad)
        ymax = min(height, ymax + pad)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymax, ymin)
    else:
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)

    return True


def _zscore_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.astype(float).copy()

    for col in out.columns:
        sd = out[col].std()
        if np.isfinite(sd) and sd > 0:
            out[col] = (out[col] - out[col].mean()) / sd
        else:
            out[col] = 0.0

    return out


def add_obs_zmean_score(
    adata,
    cols: Sequence[str],
    score_name: str,
    min_nonmissing: int = 1,
):
    """
    Add a row-wise mean of z-scored obs columns.

    Useful for compact modules such as ADAR-family expression when the
    cluster-level model has too few groups for many separate predictors.
    """
    cols = [c for c in cols if c in adata.obs.columns]

    if not cols:
        raise ValueError("No requested obs columns were found.")

    df = adata.obs[cols].apply(pd.to_numeric, errors="coerce")
    z = _zscore_frame(df)

    score = z.mean(axis=1, skipna=True)
    valid_n = df.notna().sum(axis=1)
    score[valid_n < min_nonmissing] = np.nan

    adata.obs[score_name] = score.astype(float)
    adata.uns[f"{score_name}_columns"] = cols

    return pd.Series(score, index=adata.obs_names, name=score_name)


def add_deconvolution_modules(
    df_deconv: pd.DataFrame,
    modules: Optional[dict] = None,
):
    """
    Add compact deconvolution modules by summing existing cell-type columns.

    Default module:
        Glial_like = Oligos + Astros + Micro/Macro + OPCs
    """
    if modules is None:
        modules = {
            "Glial_like": ("Oligos", "Astros", "Micro/Macro", "OPCs"),
        }

    out = df_deconv.copy()
    module_cols = []
    module_members = {}

    for module_name, members in modules.items():
        present = [c for c in members if c in out.columns]

        if present:
            out[module_name] = out[present].sum(axis=1)
            module_cols.append(module_name)
            module_members[module_name] = present

    return out, module_cols, module_members


def add_cluster_deconvolution_pcs(
    adata_ai,
    df_deconv: pd.DataFrame,
    celltype_cols: Sequence[str],
    group_key: str = "ground_truth",
    n_components: int = 1,
    prefix: str = "CellType_PC",
    aggfunc: str = "mean",
):
    """
    Compress all deconvolved cell-type proportions into cluster-level PCs.

    PCs are fit on the cluster x cell-type matrix and then mapped back to spots
    through group labels, so the resulting columns can be used by the existing
    cluster-level model helpers.
    """
    celltype_cols = [c for c in celltype_cols if c in df_deconv.columns]

    if not celltype_cols:
        raise ValueError("No cell-type columns were found in df_deconv.")

    cluster_mat, labels = _aggregate_analysis_frame_by_group(
        adata_ai=adata_ai,
        df_extra=df_deconv,
        columns=celltype_cols,
        group_key=group_key,
        aggfunc=aggfunc,
    )

    cluster_mat = cluster_mat.dropna(axis=0, how="all").copy()

    if cluster_mat.shape[0] < 2:
        raise ValueError("Need at least two clusters to compute cell-type PCs.")

    X = cluster_mat.astype(float).copy()
    X = X.fillna(X.mean(axis=0))
    X = _zscore_frame(X)

    max_components = min(int(n_components), X.shape[0] - 1, X.shape[1])

    if max_components < 1:
        raise ValueError("No valid cell-type PC components can be computed.")

    U, S, Vt = np.linalg.svd(X.values, full_matrices=False)

    pc_cols = [f"{prefix}{i + 1}" for i in range(max_components)]
    scores = pd.DataFrame(
        U[:, :max_components] * S[:max_components],
        index=cluster_mat.index,
        columns=pc_cols,
    )

    loadings = pd.DataFrame(
        Vt[:max_components, :].T,
        index=cluster_mat.columns,
        columns=pc_cols,
    )

    explained = (S ** 2) / np.sum(S ** 2)
    explained = pd.Series(
        explained[:max_components],
        index=pc_cols,
        name="explained_variance_ratio",
    )

    out = df_deconv.copy()

    for pc in pc_cols:
        out[pc] = np.nan

    valid = labels.notna() & labels.isin(scores.index)

    if valid.any():
        out.loc[valid, pc_cols] = scores.reindex(labels.loc[valid].values).values

    adata_ai.uns[f"{prefix}_cluster_scores"] = scores
    adata_ai.uns[f"{prefix}_loadings"] = loadings
    adata_ai.uns[f"{prefix}_explained_variance_ratio"] = explained
    adata_ai.uns[f"{prefix}_celltype_cols"] = celltype_cols

    return out, pc_cols, scores, loadings, explained


def _safe_multipletest(pvals, method="fdr_bh"):
    p = pd.Series(pvals).astype(float).fillna(1.0)
    p = p.clip(lower=np.nextafter(0, 1), upper=1.0)
    return multipletests(p.values, method=method)[1]


def _safe_corr(x, y, method="spearman"):
    df = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()

    if df.shape[0] < 3:
        return np.nan, np.nan, int(df.shape[0])

    if df["x"].std() == 0 or df["y"].std() == 0:
        return np.nan, np.nan, int(df.shape[0])

    method = str(method).lower()

    if method == "spearman":
        rho, pval = spearmanr(df["x"], df["y"])
    elif method == "pearson":
        rho, pval = pearsonr(df["x"], df["y"])
    else:
        raise ValueError("method must be 'spearman' or 'pearson'.")

    return float(rho), float(pval), int(df.shape[0])


def _valid_group_series(adata, group_key="ground_truth", index=None):
    """
    Return group labels while keeping missing labels as NaN.

    This prevents NaN / 'nan' / 'None' from becoming fake clusters.
    """
    if group_key not in adata.obs.columns:
        raise ValueError(f"{group_key!r} not found in adata.obs")

    if index is None:
        s = adata.obs[group_key].copy()
    else:
        s = adata.obs.loc[index, group_key].copy()

    s = pd.Series(s, index=s.index, name=group_key)

    s_str = s.astype(str).str.strip().str.lower()
    bad = s.isna() | s_str.isin(["nan", "none", "na", "null", ""])

    s = s.astype(object)
    s.loc[bad] = np.nan

    return s


def _map_group_mean_to_spots(values, labels):
    """
    Map group-level mean values back to each spot.
    Missing group labels remain NaN.
    """
    values = pd.Series(values, index=labels.index, dtype=float)
    df = pd.DataFrame({"value": values, "group": labels}).dropna()

    out = pd.Series(np.nan, index=labels.index, dtype=float)

    if df.empty:
        return out.values

    group_mean = df.groupby("group", observed=True)["value"].mean()

    valid = labels.notna()
    out.loc[valid] = labels.loc[valid].map(group_mean).astype(float)

    return out.values


def _aggregate_analysis_frame_by_group(
    adata_ai,
    df_extra: Optional[pd.DataFrame] = None,
    columns: Sequence[str] = (),
    group_key: str = "ground_truth",
    aggfunc: str = "mean",
):
    """
    Build a group-level analysis frame and a spot-to-group label vector.

    The returned frame is indexed by valid group labels. Numeric columns from
    adata_ai.obs and df_extra are aggregated within each group.
    """
    if group_key not in adata_ai.obs.columns:
        raise ValueError(f"{group_key!r} not found in adata_ai.obs")

    aggfunc = str(aggfunc).lower()

    if aggfunc not in ["mean", "median"]:
        raise ValueError("aggfunc must be 'mean' or 'median'.")

    labels = _valid_group_series(adata_ai, group_key=group_key)
    common = adata_ai.obs_names

    if df_extra is not None:
        common = common.intersection(df_extra.index)

    columns = list(dict.fromkeys(columns))

    df = pd.DataFrame({"group": labels.loc[common]}, index=common)

    obs_cols = [c for c in columns if c in adata_ai.obs.columns]
    extra_cols = [
        c for c in columns
        if df_extra is not None and c in df_extra.columns and c not in obs_cols
    ]

    if obs_cols:
        df = df.join(adata_ai.obs.loc[common, obs_cols], how="left")

    if extra_cols:
        df = df.join(df_extra.loc[common, extra_cols], how="left")

    value_cols = [c for c in columns if c in df.columns]
    df = df.dropna(subset=["group"])

    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if not value_cols:
        return pd.DataFrame(index=pd.Index([], name=group_key)), labels

    if aggfunc == "mean":
        out = df.groupby("group", observed=True)[value_cols].mean()
    else:
        out = df.groupby("group", observed=True)[value_cols].median()

    out.index.name = group_key

    return out, labels


def collapse_deconvolution_to_clusters(
    adata_ai,
    df_deconv: pd.DataFrame,
    celltype_cols: Optional[Sequence[str]] = None,
    group_key: str = "ground_truth",
    aggfunc: str = "mean",
    normalize: bool = True,
    min_spots_per_group: int = 2,
):
    """
    Aggregate spot-level deconvolution proportions to cluster-level proportions.

    Returns:
        cluster_deconv: group x celltype table
        spot_deconv_cluster: cluster means mapped back to each spot
        group_counts: number of spots contributing to each group
    """
    if celltype_cols is None:
        celltype_cols = [
            c for c in df_deconv.columns
            if pd.api.types.is_numeric_dtype(df_deconv[c])
        ]

    celltype_cols = [c for c in celltype_cols if c in df_deconv.columns]

    if not celltype_cols:
        raise ValueError("No numeric cell-type columns found in df_deconv.")

    cluster_df, labels = _aggregate_analysis_frame_by_group(
        adata_ai=adata_ai,
        df_extra=df_deconv,
        columns=celltype_cols,
        group_key=group_key,
        aggfunc=aggfunc,
    )

    common = adata_ai.obs_names.intersection(df_deconv.index)
    group_counts = (
        pd.DataFrame({"group": labels.loc[common]}, index=common)
        .dropna()
        .groupby("group", observed=True)
        .size()
    )

    keep_groups = group_counts[group_counts >= min_spots_per_group].index
    cluster_df = cluster_df.loc[cluster_df.index.intersection(keep_groups)].copy()
    group_counts = group_counts.loc[cluster_df.index]

    if normalize and not cluster_df.empty:
        row_sum = cluster_df.sum(axis=1).replace(0, np.nan)
        cluster_df = cluster_df.div(row_sum, axis=0)

    spot_df = pd.DataFrame(np.nan, index=adata_ai.obs_names, columns=celltype_cols, dtype=float)
    valid = labels.notna() & labels.isin(cluster_df.index)

    if valid.any():
        spot_df.loc[valid, celltype_cols] = cluster_df.reindex(labels.loc[valid].values).values

    return cluster_df, spot_df, group_counts


def transfer_obs_metadata(
    source_adata,
    target_adata,
    columns=("x_pixel", "y_pixel", "x_array", "y_array", "in_tissue", "ground_truth"),
):
    """
    Copy selected obs columns and spatial coordinates from expression AnnData
    to A-to-I AnnData.
    """
    common = target_adata.obs_names.intersection(source_adata.obs_names)

    for col in columns:
        if col in source_adata.obs.columns:
            target_adata.obs[col] = np.nan
            target_adata.obs.loc[common, col] = source_adata.obs.loc[common, col].values

    if "spatial" in source_adata.obsm:
        target_adata.obsm["spatial"] = np.zeros((target_adata.n_obs, 2), dtype=float)
        tgt_idx = target_adata.obs_names.get_indexer(common)
        src_idx = source_adata.obs_names.get_indexer(common)
        target_adata.obsm["spatial"][tgt_idx, :] = np.asarray(source_adata.obsm["spatial"])[src_idx, :]

    elif {"x_pixel", "y_pixel"}.issubset(target_adata.obs.columns):
        target_adata.obs[["x_pixel", "y_pixel"]] = (
            target_adata.obs[["x_pixel", "y_pixel"]]
            .astype(float)
            .fillna(0)
        )
        target_adata.obsm["spatial"] = target_adata.obs[["x_pixel", "y_pixel"]].values

    return target_adata


# =============================================================================
# 1. Reliable A-to-I site filtering
# =============================================================================

def filter_atoi_sites(
    adata_ai,
    min_spot_cov: int = 10,
    min_n_spots: int = 10,
    min_site_ratio: Optional[float] = None,
    max_site_ratio: float = 1.0,
    min_site_ratio_sd: Optional[float] = None,
    a_layer: str = "A",
    g_layer: str = "G",
    copy: bool = True,
    verbose: bool = True,
):
    """
    Filter reliable A-to-I sites.

    Main criteria:
        site_G_ratio is not NaN
        site_G_ratio < max_site_ratio
        n_spots_cov_gt{min_spot_cov} >= min_n_spots

    Optional:
        site_G_ratio > min_site_ratio
        site_G_ratio_sd >= min_site_ratio_sd
    """
    A, G = _get_count_layers(adata_ai, a_layer=a_layer, g_layer=g_layer)
    cov = A + G

    site_A = _dense_sum(A, axis=0)
    site_G = _dense_sum(G, axis=0)
    site_cov = site_A + site_G

    site_ratio = np.full(adata_ai.n_vars, np.nan, dtype=float)
    nonzero = site_cov > 0
    site_ratio[nonzero] = site_G[nonzero] / site_cov[nonzero]

    n_spots_cov = np.asarray((cov >= min_spot_cov).sum(axis=0)).ravel()
    cov_col = f"n_spots_cov_gt{min_spot_cov}"

    ratio_sd = np.full(adata_ai.n_vars, np.nan, dtype=float)

    for j in range(adata_ai.n_vars):
        a = _dense_col(A, j)
        g = _dense_col(G, j)
        c = a + g
        valid = c >= min_spot_cov

        if valid.sum() >= 2:
            ratio_sd[j] = np.nanstd(g[valid] / c[valid])

    adata_ai.var["total_A"] = site_A
    adata_ai.var["total_G"] = site_G
    adata_ai.var["total_cov"] = site_cov
    adata_ai.var["site_G_ratio"] = site_ratio
    adata_ai.var["site_G_ratio_sd"] = ratio_sd
    adata_ai.var[cov_col] = n_spots_cov

    keep = adata_ai.var["site_G_ratio"].notna()
    keep &= adata_ai.var["site_G_ratio"] < max_site_ratio
    keep &= adata_ai.var[cov_col] >= min_n_spots

    if min_site_ratio is not None:
        keep &= adata_ai.var["site_G_ratio"] > min_site_ratio

    if min_site_ratio_sd is not None:
        keep &= adata_ai.var["site_G_ratio_sd"] >= min_site_ratio_sd

    if verbose:
        print(f"Before: {adata_ai.n_vars}")
        print(f"After : {int(keep.sum())}")
        print(f"Removed: {int((~keep).sum())}")
        print("\nFiltering criteria:")
        print("  site_G_ratio is not NaN")
        if min_site_ratio is not None:
            print(f"  site_G_ratio > {min_site_ratio}")
        print(f"  site_G_ratio < {max_site_ratio}")
        print(f"  {cov_col} >= {min_n_spots}")
        if min_site_ratio_sd is not None:
            print(f"  site_G_ratio_sd >= {min_site_ratio_sd}")

    adata_out = adata_ai[:, keep.values].copy() if copy else adata_ai[:, keep.values]

    return adata_out, keep


# =============================================================================
# 2. Multi-site A-to-I scores
# =============================================================================

def _site_ratio_matrix(
    adata_ai,
    site_idx: Sequence[int],
    min_cov: int = 10,
    a_layer: str = "A",
    g_layer: str = "G",
):
    A, G = _get_count_layers(adata_ai, a_layer=a_layer, g_layer=g_layer)

    ratios = []
    coverages = []

    for j in site_idx:
        a = _dense_col(A, j)
        g = _dense_col(G, j)
        cov = a + g

        ratio = np.full(adata_ai.n_obs, np.nan, dtype=float)
        valid = cov >= min_cov
        ratio[valid] = g[valid] / cov[valid]

        ratios.append(ratio)
        coverages.append(cov)

    return np.vstack(ratios).T, np.vstack(coverages).T


def _resolve_site_list(
    adata_ai,
    sites: Optional[Union[pd.DataFrame, Iterable[str]]] = None,
    top_n: Optional[int] = None,
    require_sv: bool = False,
    weight_col: Optional[str] = None,
):
    weights = None

    if sites is None:
        site_names = list(adata_ai.var_names)

    elif isinstance(sites, pd.DataFrame):
        df = sites.copy()

        if require_sv:
            if "is_sv_atoi" not in df.columns:
                raise ValueError("require_sv=True but sites DataFrame has no 'is_sv_atoi' column.")

            df = df[df["is_sv_atoi"] == True].copy()

            if df.empty:
                raise ValueError("No sites with is_sv_atoi == True were found.")

        if top_n is not None:
            df = df.head(top_n)

        site_names = [s for s in df["site"].astype(str) if s in adata_ai.var_names]

        if weight_col is not None and weight_col in df.columns:
            weights = (
                df.set_index("site")
                .reindex(site_names)[weight_col]
                .astype(float)
                .fillna(1.0)
                .values
            )

    else:
        site_names = [str(s) for s in sites if str(s) in adata_ai.var_names]

        if top_n is not None:
            site_names = site_names[:top_n]

    if not site_names:
        raise ValueError("No requested A-to-I sites were found in adata_ai.var_names.")

    return site_names, weights


def select_atoi_score_sites(
    sv_atoi: pd.DataFrame,
    adata_ai=None,
    require_sv: bool = True,
    fallback_to_ranked: bool = True,
    top_n: int = 200,
    min_moran_I: float = 0.0,
):
    """
    Select sites for an A-to-I score from SV-A-to-I results.

    If require_sv=True and too few/no SV sites pass, fallback_to_ranked=True
    uses the strongest QC-passing positive-Moran sites. This is useful for an
    exploratory SV-A-to-I score under relaxed discovery thresholds.
    """
    if sv_atoi is None or sv_atoi.empty:
        raise ValueError("sv_atoi is empty.")

    df = sv_atoi.copy()

    if adata_ai is not None:
        df = df[df["site"].astype(str).isin(adata_ai.var_names)].copy()

    primary = df.copy()

    if require_sv and "is_sv_atoi" in primary.columns:
        primary = primary[primary["is_sv_atoi"] == True].copy()

    if (
        primary.empty
        and fallback_to_ranked
        and {"pass_qc", "moran_I"}.issubset(df.columns)
    ):
        primary = df[
            (df["pass_qc"] == True)
            & pd.to_numeric(df["moran_I"], errors="coerce").gt(min_moran_I)
        ].copy()
        primary["is_sv_atoi"] = True
        primary["score_site_source"] = "ranked_positive_moran_fallback"
    else:
        primary["score_site_source"] = "called_sv_atoi"

    sort_cols = []
    ascending = []

    for col, asc in [
        ("moran_fdr", True),
        ("moran_p", True),
        ("moran_I", False),
        ("sd_ratio", False),
    ]:
        if col in primary.columns:
            sort_cols.append(col)
            ascending.append(asc)

    if sort_cols:
        primary = primary.sort_values(sort_cols, ascending=ascending)

    if top_n is not None:
        primary = primary.head(top_n)

    if primary.empty:
        raise ValueError("No A-to-I score sites were selected.")

    return primary.reset_index(drop=True)


def compute_sv_atoi_score(
    adata_ai,
    sv_atoi: pd.DataFrame,
    score_name: str = "sv_atoi_score",
    min_cov: int = 10,
    top_n: int = 200,
    weight_col: str = "moran_I",
    require_sv: bool = True,
    fallback_to_ranked: bool = True,
    zscore_sites: bool = True,
    a_layer: str = "A",
    g_layer: str = "G",
):
    """
    Compute an SV-A-to-I score, with optional fallback to ranked spatial sites.
    """
    score_sites = select_atoi_score_sites(
        sv_atoi=sv_atoi,
        adata_ai=adata_ai,
        require_sv=require_sv,
        fallback_to_ranked=fallback_to_ranked,
        top_n=top_n,
    )

    score = compute_multisite_atoi_score(
        adata_ai=adata_ai,
        sites=score_sites,
        score_name=score_name,
        min_cov=min_cov,
        top_n=None,
        weight_col=weight_col,
        require_sv=False,
        a_layer=a_layer,
        g_layer=g_layer,
        zscore_sites=zscore_sites,
    )

    adata_ai.uns[f"{score_name}_site_table"] = score_sites

    return score, score_sites


def compute_multisite_atoi_score(
    adata_ai,
    sites: Optional[Union[pd.DataFrame, Iterable[str]]] = None,
    score_name: str = "global_atoi_score",
    min_cov: int = 10,
    top_n: Optional[int] = None,
    weight_col: Optional[str] = None,
    require_sv: bool = False,
    a_layer: str = "A",
    g_layer: str = "G",
    zscore_sites: bool = True,
):
    """
    Compute a spot-level multi-site A-to-I score.

    For global A-to-I score:
        sites=None
        require_sv=False
        weight_col=None

    For SV-A-to-I score:
        sites=sv_atoi
        require_sv=True
        weight_col='moran_I'
        top_n=200
    """
    site_names, weights = _resolve_site_list(
        adata_ai=adata_ai,
        sites=sites,
        top_n=top_n,
        require_sv=require_sv,
        weight_col=weight_col,
    )

    site_idx = [adata_ai.var_names.get_loc(s) for s in site_names]

    ratio_mat, cov_mat = _site_ratio_matrix(
        adata_ai,
        site_idx,
        min_cov=min_cov,
        a_layer=a_layer,
        g_layer=g_layer,
    )

    score_mat = ratio_mat.copy()

    if zscore_sites:
        mu = np.nanmean(score_mat, axis=0)
        sd = np.nanstd(score_mat, axis=0)
        sd[sd == 0] = np.nan
        score_mat = (score_mat - mu) / sd

    valid = np.isfinite(score_mat)

    if weights is None:
        score = np.nanmean(score_mat, axis=1)
    else:
        weights = np.asarray(weights, dtype=float)
        weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)

        numerator = np.nansum(np.where(valid, score_mat * weights, np.nan), axis=1)
        denominator = np.sum(valid * weights, axis=1)

        score = numerator / denominator
        score[denominator == 0] = np.nan

    adata_ai.obs[score_name] = score
    adata_ai.obs[f"{score_name}_n_sites"] = np.sum(np.isfinite(ratio_mat), axis=1)
    adata_ai.obs[f"{score_name}_mean_cov"] = np.nanmean(
        np.where(cov_mat >= min_cov, cov_mat, np.nan),
        axis=1,
    )
    adata_ai.uns[f"{score_name}_sites"] = site_names

    return pd.Series(score, index=adata_ai.obs_names, name=score_name)


def compute_global_atoi_ratio(
    adata_ai,
    ratio_key: str = "global_atoi_ratio",
    coverage_key: str = "global_atoi_cov",
    min_total_cov: int = 20,
    a_layer: str = "A",
    g_layer: str = "G",
):
    """
    Compute raw per-spot global G/(A+G) ratio across all retained sites.
    This is not z-scored.
    """
    A, G = _get_count_layers(adata_ai, a_layer=a_layer, g_layer=g_layer)

    total_A = _dense_sum(A, axis=1)
    total_G = _dense_sum(G, axis=1)
    cov = total_A + total_G

    ratio = np.full(adata_ai.n_obs, np.nan, dtype=float)
    valid = cov >= min_total_cov
    ratio[valid] = total_G[valid] / cov[valid]

    adata_ai.obs[ratio_key] = ratio
    adata_ai.obs[coverage_key] = cov

    return pd.Series(ratio, index=adata_ai.obs_names, name=ratio_key)


def get_site_editing_ratio(
    adata_ai,
    site: str,
    min_cov: int = 10,
    a_layer: str = "A",
    g_layer: str = "G",
) -> pd.Series:
    if site not in adata_ai.var_names:
        raise ValueError(f"{site!r} not found in adata_ai.var_names")

    A, G = _get_count_layers(adata_ai, a_layer=a_layer, g_layer=g_layer)

    j = adata_ai.var_names.get_loc(site)

    a = _dense_col(A, j)
    g = _dense_col(G, j)
    cov = a + g

    ratio = np.full(adata_ai.n_obs, np.nan, dtype=float)
    valid = cov >= min_cov
    ratio[valid] = g[valid] / cov[valid]

    return pd.Series(ratio, index=adata_ai.obs_names, name=site)


# =============================================================================
# 3. Spatial visualization
# =============================================================================

def _spatial_axes(
    ax,
    adata,
    values,
    title: str,
    cmap="magma",
    spot_size: int = 18,
    clip=(1, 99),
    center=None,
):
    coords = _plot_coords_from_adata(adata)

    x = coords[:, 0]
    y = coords[:, 1]

    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    has_background = _draw_spatial_background(ax, adata, x, y)

    ax.scatter(
        x[~mask],
        y[~mask],
        s=spot_size,
        c="#d9d9d9",
        alpha=0.38 if has_background else 0.22,
        linewidths=0,
        zorder=2,
    )

    if mask.sum() == 0:
        ax.set_title(title)
        if not has_background:
            ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.axis("off")
        return None

    vmin, vmax = np.nanpercentile(values[mask], clip)

    if center is not None:
        lim = max(abs(vmin - center), abs(vmax - center))
        vmin = center - lim
        vmax = center + lim

    sc = ax.scatter(
        x[mask],
        y[mask],
        s=spot_size,
        c=values[mask],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
        zorder=3,
    )

    ax.set_title(title, fontsize=11, pad=8)
    if not has_background:
        ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")

    return sc


def plot_spatial_value(
    adata,
    values,
    title="spatial value",
    cmap="magma",
    spot_size=18,
    center=None,
    figsize=(4.5, 4.2),
):
    fig, ax = plt.subplots(figsize=figsize)

    sc = _spatial_axes(
        ax,
        adata,
        values,
        title,
        cmap=cmap,
        spot_size=spot_size,
        center=center,
    )

    if sc is not None:
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)

    plt.tight_layout()
    return fig, ax


def plot_cluster_level_obs_value(
    adata,
    value_key,
    group_key="ground_truth",
    aggfunc="mean",
    spot_size=18,
    cmap="magma",
    center=None,
    figsize=(10, 4.2),
):
    """
    Show:
        1. spot-level map
        2. cluster-level map
        3. group mean barplot

    Missing group labels are ignored.
    """
    if value_key not in adata.obs.columns:
        raise ValueError(f"{value_key!r} not found in adata.obs")

    if group_key not in adata.obs.columns:
        raise ValueError(f"{group_key!r} not found in adata.obs")

    labels = _valid_group_series(adata, group_key=group_key)
    values = adata.obs[value_key].astype(float)

    df = pd.DataFrame({"value": values, "group": labels}).dropna()

    cluster_values = pd.Series(np.nan, index=adata.obs_names, dtype=float)

    if not df.empty:
        grouped = df.groupby("group", observed=True)["value"].agg(aggfunc)
        valid = labels.notna()
        cluster_values.loc[valid] = labels.loc[valid].map(grouped).astype(float)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1, 1, 0.9]},
    )

    sc0 = _spatial_axes(
        axes[0],
        adata,
        values.values,
        f"Spot-level {value_key}",
        cmap=cmap,
        spot_size=spot_size,
        center=center,
    )

    sc1 = _spatial_axes(
        axes[1],
        adata,
        cluster_values.values,
        f"Cluster-level {value_key}\n{group_key}",
        cmap=cmap,
        spot_size=spot_size,
        center=center,
    )

    if sc0 is not None:
        fig.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.02)

    if sc1 is not None:
        fig.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.02)

    summary = (
        df.groupby("group", observed=True)["value"]
        .agg(["count", "mean", "median", "std"])
        .sort_values("mean")
    )

    sns.barplot(
        data=summary.reset_index(),
        y="group",
        x="mean",
        ax=axes[2],
        color="#80b1d3",
        edgecolor="black",
        linewidth=0.4,
    )

    axes[2].set_xlabel(f"mean {value_key}")
    axes[2].set_ylabel("")
    axes[2].set_title("Mean by group", fontsize=11, pad=8)

    if center is not None:
        axes[2].axvline(center, color="black", linewidth=1)

    plt.tight_layout()

    return fig, axes, summary


def summarize_obs_by_group(adata, value_key, group_key="ground_truth"):
    if value_key not in adata.obs.columns:
        raise ValueError(f"{value_key!r} not found in adata.obs")

    labels = _valid_group_series(adata, group_key=group_key)
    values = adata.obs[value_key].astype(float)

    df = pd.DataFrame({value_key: values, group_key: labels}).dropna()

    return (
        df.groupby(group_key, observed=True)[value_key]
        .agg(["count", "mean", "median", "std"])
        .sort_values("mean", ascending=False)
    )


# =============================================================================
# 4. SV-A-to-I detection
# =============================================================================

def _safe_moran_geary(values, coords, spatial_k=6, permutations=999):
    mask = np.isfinite(values)

    if mask.sum() <= spatial_k + 2 or np.nanstd(values[mask]) == 0:
        return np.nan, np.nan, np.nan, np.nan

    try:
        from libpysal.weights import KNN
        from esda.geary import Geary
        from esda.moran import Moran

        w = KNN.from_array(coords[mask], k=min(spatial_k, mask.sum() - 1))
        w.transform = "R"

        moran = Moran(values[mask], w, permutations=permutations)
        geary = Geary(values[mask], w, permutations=permutations)

        return float(moran.I), float(moran.p_sim), float(geary.C), float(geary.p_sim)

    except Exception as exc:
        warnings.warn(f"Moran/Geary failed; returning NaN spatial statistics: {exc}")
        return np.nan, np.nan, np.nan, np.nan


def detect_spatial_atoi_sites(
    adata_ai,
    group_key="ground_truth",
    min_cov=10,
    min_valid_spots=30,
    min_total_A=30,
    min_total_G=30,
    min_ratio_sd=0.03,
    spatial_k=6,
    fdr_cutoff=0.05,
    p_cutoff=None,
    sv_call_by="fdr",
    permutations=999,
    a_layer="A",
    g_layer="G",
    store_key="sv_atoi",
    verbose=True,
):
    """
    Detect spatially variable A-to-I sites.

    sv_call_by:
        "fdr"    : moran_fdr < fdr_cutoff
        "p"      : moran_p < p_cutoff
        "either" : moran_fdr < fdr_cutoff OR moran_p < p_cutoff
    """
    if p_cutoff is None:
        p_cutoff = fdr_cutoff

    sv_call_by = str(sv_call_by).lower()

    if sv_call_by not in ["fdr", "p", "either"]:
        raise ValueError("sv_call_by must be one of: 'fdr', 'p', 'either'.")

    coords = _coords_from_adata(adata_ai)

    if group_key is not None and group_key in adata_ai.obs.columns:
        groups = _valid_group_series(adata_ai, group_key=group_key)
    else:
        groups = None

    A, G = _get_count_layers(adata_ai, a_layer=a_layer, g_layer=g_layer)

    rows = []

    for j, site in enumerate(adata_ai.var_names):
        a = _dense_col(A, j)
        g = _dense_col(G, j)
        cov = a + g

        valid = cov >= min_cov

        ratio = np.full(adata_ai.n_obs, np.nan, dtype=float)
        ratio[valid] = g[valid] / cov[valid]

        n_valid = int(valid.sum())
        total_A = float(a[valid].sum())
        total_G = float(g[valid].sum())

        mean_ratio = float(np.nanmean(ratio)) if n_valid > 0 else np.nan
        sd_ratio = float(np.nanstd(ratio)) if n_valid > 0 else np.nan

        pass_qc = (
            (n_valid >= min_valid_spots)
            and (total_A >= min_total_A)
            and (total_G >= min_total_G)
            and np.isfinite(sd_ratio)
            and (sd_ratio >= min_ratio_sd)
        )

        moran_I = moran_p = geary_C = geary_p = np.nan
        kw_stat = kw_p = np.nan
        group_n = 0

        if pass_qc:
            moran_I, moran_p, geary_C, geary_p = _safe_moran_geary(
                ratio,
                coords,
                spatial_k=spatial_k,
                permutations=permutations,
            )

            if groups is not None:
                tmp = pd.DataFrame(
                    {"ratio": ratio, "group": groups.values},
                    index=adata_ai.obs_names,
                ).dropna()

                vals = [
                    sub["ratio"].dropna().values
                    for _, sub in tmp.groupby("group", observed=True)
                ]
                vals = [v for v in vals if len(v) >= 3]
                group_n = len(vals)

                if group_n >= 2:
                    try:
                        kw_stat, kw_p = kruskal(*vals)
                        kw_stat = float(kw_stat)
                        kw_p = float(kw_p)
                    except Exception:
                        kw_stat = np.nan
                        kw_p = np.nan

        gene = ""

        if "Gene.refGene" in adata_ai.var.columns:
            gene = adata_ai.var.iloc[j].get("Gene.refGene", "")

        rows.append(
            {
                "site": site,
                "gene": gene,
                "n_valid_spots": n_valid,
                "total_A_valid": total_A,
                "total_G_valid": total_G,
                "total_cov_valid": total_A + total_G,
                "mean_ratio": mean_ratio,
                "sd_ratio": sd_ratio,
                "pass_qc": bool(pass_qc),
                "moran_I": moran_I,
                "moran_p": moran_p,
                "geary_C": geary_C,
                "geary_p": geary_p,
                "group_key": group_key,
                "group_n": group_n,
                "group_kw_stat": kw_stat,
                "group_kw_p": kw_p,
            }
        )

        if verbose and (j + 1) % 500 == 0:
            print(f"Processed {j + 1}/{adata_ai.n_vars} A-to-I sites")

    res = pd.DataFrame(rows)

    res["moran_fdr"] = np.nan
    res["group_kw_fdr"] = np.nan

    qc = res["pass_qc"].values

    if qc.any():
        res.loc[qc, "moran_fdr"] = _safe_multipletest(res.loc[qc, "moran_p"])
        res.loc[qc, "group_kw_fdr"] = _safe_multipletest(res.loc[qc, "group_kw_p"])

    if sv_call_by == "fdr":
        sig = res["moran_fdr"] < fdr_cutoff
        call_text = f"Moran FDR < {fdr_cutoff}"

    elif sv_call_by == "p":
        sig = res["moran_p"] < p_cutoff
        call_text = f"Moran p < {p_cutoff}"

    else:
        sig = (res["moran_fdr"] < fdr_cutoff) | (res["moran_p"] < p_cutoff)
        call_text = f"Moran FDR < {fdr_cutoff} OR Moran p < {p_cutoff}"

    res["is_sv_atoi"] = (
        res["pass_qc"]
        & (res["moran_I"] > 0)
        & sig.fillna(False)
    )

    res["sv_call_by"] = sv_call_by
    res["fdr_cutoff"] = fdr_cutoff
    res["p_cutoff"] = p_cutoff

    res = res.sort_values(
        ["is_sv_atoi", "moran_fdr", "moran_p", "moran_I"],
        ascending=[False, True, True, False],
    ).reset_index(drop=True)

    adata_ai.uns[store_key] = res

    if verbose:
        print(f"QC-passing sites: {int(res['pass_qc'].sum())}")
        print(f"SV-A-to-I sites ({call_text}): {int(res['is_sv_atoi'].sum())}")

    return res


def plot_sv_atoi_discovery_landscape(
    sv_atoi: pd.DataFrame,
    fdr_cutoff=0.05,
    top_n_labels=8,
    figsize=(12, 4),
):
    df = sv_atoi.copy()

    df["neglog10_fdr"] = -np.log10(df["moran_fdr"].clip(lower=1e-300))
    df["status"] = np.where(df["is_sv_atoi"], "SV-A-to-I", "not SV")

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    counts = pd.Series(
        {
            "all sites": len(df),
            "QC pass": int(df["pass_qc"].sum()),
            "SV-A-to-I": int(df["is_sv_atoi"].sum()),
        }
    )

    sns.barplot(x=counts.index, y=counts.values, ax=axes[0])
    axes[0].set_ylabel("site count")
    axes[0].set_xlabel("")
    axes[0].set_title("Discovery funnel")
    axes[0].tick_params(axis="x", rotation=25)

    colors = df["status"].map({"not SV": "#bdbdbd", "SV-A-to-I": "#d95f02"})
    sizes = np.clip(df["n_valid_spots"].astype(float) / 3, 12, 90)

    mask = df["moran_I"].notna() & df["neglog10_fdr"].notna()

    axes[1].scatter(
        df.loc[mask, "moran_I"],
        df.loc[mask, "neglog10_fdr"],
        s=sizes.loc[mask],
        c=colors.loc[mask],
        linewidths=0,
        alpha=0.75,
    )

    axes[1].axhline(-np.log10(fdr_cutoff), color="black", linestyle="--", linewidth=1)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Moran's I")
    axes[1].set_ylabel("-log10(FDR)")
    axes[1].set_title("Spatial autocorrelation")

    label_df = (
        df.sort_values(
            ["is_sv_atoi", "moran_fdr", "moran_I"],
            ascending=[False, True, False],
        )
        .head(top_n_labels)
    )

    for _, row in label_df.iterrows():
        label = row["gene"] if isinstance(row.get("gene", ""), str) and row.get("gene", "") else row["site"]

        if np.isfinite(row["moran_I"]) and np.isfinite(row["neglog10_fdr"]):
            axes[1].text(
                row["moran_I"],
                row["neglog10_fdr"],
                str(label)[:14],
                fontsize=7,
            )

    colors2 = df["status"].map({"not SV": "#bdbdbd", "SV-A-to-I": "#7570b3"})
    mask = df["n_valid_spots"].notna() & df["sd_ratio"].notna()

    axes[2].scatter(
        df.loc[mask, "n_valid_spots"],
        df.loc[mask, "sd_ratio"],
        c=colors2.loc[mask],
        s=28,
        linewidths=0,
        alpha=0.75,
    )

    axes[2].set_xlabel("valid spots")
    axes[2].set_ylabel("editing ratio SD")
    axes[2].set_title("Coverage and variability")

    plt.tight_layout()

    return fig, axes


def plot_top_sv_site_patterns(
    adata_ai,
    sv_atoi,
    group_key="ground_truth",
    top_n=6,
    min_cov=10,
    mode="cluster",
    spot_size=16,
    cmap="magma",
    figsize=None,
):
    """
    Plot top SV-A-to-I sites as spot-level or cluster-level editing-ratio maps.
    """
    mode = str(mode).lower()

    if mode not in ["spot", "cluster"]:
        raise ValueError("mode must be 'spot' or 'cluster'.")

    df = sv_atoi.copy()

    if "is_sv_atoi" in df.columns:
        df = df[df["is_sv_atoi"] == True].copy()

    if df.empty:
        raise ValueError("No SV-A-to-I sites available for plotting.")

    sites = [s for s in df["site"].head(top_n).astype(str) if s in adata_ai.var_names]

    if not sites:
        raise ValueError("No top SV-A-to-I sites found in adata_ai.var_names.")

    ncols = min(3, len(sites))
    nrows = int(np.ceil(len(sites) / ncols))

    if figsize is None:
        figsize = (4.1 * ncols, 4.0 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    if mode == "cluster":
        labels = _valid_group_series(adata_ai, group_key=group_key)
    else:
        labels = None

    for ax, site in zip(axes, sites):
        ratio = get_site_editing_ratio(adata_ai, site, min_cov=min_cov)

        if mode == "cluster":
            values = _map_group_mean_to_spots(ratio, labels)
        else:
            values = ratio.values

        if site in adata_ai.var.index and "Gene.refGene" in adata_ai.var.columns:
            gene = adata_ai.var.loc[site].get("Gene.refGene", site)
        else:
            gene = site

        sc = _spatial_axes(
            ax,
            adata_ai,
            values,
            f"{gene}\n{site}",
            cmap=cmap,
            spot_size=spot_size,
        )

        if sc is not None:
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)

    for ax in axes[len(sites):]:
        ax.axis("off")

    fig.suptitle(
        f"Top SV-A-to-I single-site editing ratios ({mode})",
        y=1.02,
        fontsize=13,
    )

    plt.tight_layout()

    return fig, axes


# =============================================================================
# 6. ADAR-family expression
# =============================================================================

def add_gene_expression_to_obs(
    adata_expr,
    target_adata,
    genes=("ADAR", "ADARB1", "ADARB2"),
    prefix="expr_",
    layer=None,
    log1p: bool = False,
):
    """
    Copy selected gene expression columns from expression AnnData into target AnnData obs.
    """
    common = target_adata.obs_names.intersection(adata_expr.obs_names)

    added = []

    for gene in genes:
        if gene not in adata_expr.var_names:
            warnings.warn(f"{gene!r} not found in expression var_names; skipped.")
            continue

        x = adata_expr[common, gene].layers[layer] if layer is not None else adata_expr[common, gene].X

        values = x.toarray().ravel() if hasattr(x, "toarray") else np.asarray(x).ravel()
        values = values.astype(float)

        if log1p:
            values = np.log1p(values)

        col = f"{prefix}{gene}"

        target_adata.obs[col] = np.nan
        target_adata.obs.loc[common, col] = values

        added.append(col)

    return added


def analyze_adar_spatial_correlation(
    adata_ai,
    adar_cols=("expr_ADAR", "expr_ADARB1", "expr_ADARB2"),
    group_key="ground_truth",
    method="spearman",
    cluster_agg="mean",
    min_cluster_groups=3,
    store_key="adar_spatial_correlation",
):
    """
    Compute ADAR-family spatial correlations at both spot level and cluster level.

    Returns dict:
        spot_corr
        spot_p
        cluster_mean
        cluster_corr
        cluster_p
        pairwise
    """
    adar_cols = [c for c in adar_cols if c in adata_ai.obs.columns]

    if len(adar_cols) < 2:
        raise ValueError(f"At least two ADAR expression columns are required. Found: {adar_cols}")

    method = str(method).lower()
    cluster_agg = str(cluster_agg).lower()

    if method not in ["spearman", "pearson"]:
        raise ValueError("method must be 'spearman' or 'pearson'.")

    if cluster_agg not in ["mean", "median"]:
        raise ValueError("cluster_agg must be 'mean' or 'median'.")

    spot_df = adata_ai.obs[adar_cols].astype(float).copy()

    spot_corr = pd.DataFrame(np.nan, index=adar_cols, columns=adar_cols)
    spot_p = pd.DataFrame(np.nan, index=adar_cols, columns=adar_cols)

    rows = []

    for i, col1 in enumerate(adar_cols):
        for j, col2 in enumerate(adar_cols):
            if i == j:
                spot_corr.loc[col1, col2] = 1.0
                spot_p.loc[col1, col2] = 0.0

            elif i < j:
                rho, pval, n = _safe_corr(spot_df[col1], spot_df[col2], method=method)

                spot_corr.loc[col1, col2] = rho
                spot_corr.loc[col2, col1] = rho

                spot_p.loc[col1, col2] = pval
                spot_p.loc[col2, col1] = pval

                rows.append(
                    {
                        "level": "spot",
                        "gene1": col1.replace("expr_", ""),
                        "gene2": col2.replace("expr_", ""),
                        "col1": col1,
                        "col2": col2,
                        "n": n,
                        "correlation": rho,
                        "pval": pval,
                        "method": method,
                        "cluster_agg": np.nan,
                    }
                )

    if group_key in adata_ai.obs.columns:
        labels = _valid_group_series(adata_ai, group_key=group_key)

        cluster_df = spot_df.copy()
        cluster_df[group_key] = labels
        cluster_df = cluster_df.dropna(subset=[group_key])

        if cluster_agg == "mean":
            cluster_mean = cluster_df.groupby(group_key, observed=True)[adar_cols].mean()
        else:
            cluster_mean = cluster_df.groupby(group_key, observed=True)[adar_cols].median()

        cluster_corr = pd.DataFrame(np.nan, index=adar_cols, columns=adar_cols)
        cluster_p = pd.DataFrame(np.nan, index=adar_cols, columns=adar_cols)

        for i, col1 in enumerate(adar_cols):
            for j, col2 in enumerate(adar_cols):
                if i == j:
                    cluster_corr.loc[col1, col2] = 1.0
                    cluster_p.loc[col1, col2] = 0.0

                elif i < j:
                    sub = cluster_mean[[col1, col2]].dropna()

                    if sub.shape[0] >= min_cluster_groups:
                        rho, pval, n = _safe_corr(sub[col1], sub[col2], method=method)
                    else:
                        rho, pval, n = np.nan, np.nan, int(sub.shape[0])

                    cluster_corr.loc[col1, col2] = rho
                    cluster_corr.loc[col2, col1] = rho

                    cluster_p.loc[col1, col2] = pval
                    cluster_p.loc[col2, col1] = pval

                    rows.append(
                        {
                            "level": "cluster",
                            "gene1": col1.replace("expr_", ""),
                            "gene2": col2.replace("expr_", ""),
                            "col1": col1,
                            "col2": col2,
                            "n": n,
                            "correlation": rho,
                            "pval": pval,
                            "method": method,
                            "cluster_agg": cluster_agg,
                        }
                    )

    else:
        cluster_mean = pd.DataFrame()
        cluster_corr = pd.DataFrame()
        cluster_p = pd.DataFrame()

    pairwise = pd.DataFrame(rows)

    if not pairwise.empty:
        pairwise["fdr"] = np.nan

        for level in pairwise["level"].dropna().unique():
            idx = pairwise["level"] == level
            pairwise.loc[idx, "fdr"] = _safe_multipletest(pairwise.loc[idx, "pval"])

        pairwise = pairwise.sort_values(
            ["level", "fdr", "correlation"],
            ascending=[True, True, False],
        ).reset_index(drop=True)

    out = {
        "spot_corr": spot_corr,
        "spot_p": spot_p,
        "cluster_mean": cluster_mean,
        "cluster_corr": cluster_corr,
        "cluster_p": cluster_p,
        "pairwise": pairwise,
    }

    adata_ai.uns[store_key] = out

    return out


def plot_adar_correlation_heatmaps(
    corr_result,
    figsize=(8.5, 3.8),
    cmap="vlag",
    vmin=-1,
    vmax=1,
):
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    spot_corr = corr_result.get("spot_corr", pd.DataFrame())
    cluster_corr = corr_result.get("cluster_corr", pd.DataFrame())

    if spot_corr.empty:
        axes[0].axis("off")
        axes[0].set_title("Spot-level ADAR correlation\n(empty)")
    else:
        sns.heatmap(
            spot_corr,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            center=0,
            vmin=vmin,
            vmax=vmax,
            square=True,
            ax=axes[0],
            cbar_kws={"shrink": 0.7},
        )
        axes[0].set_title("Spot-level ADAR correlation")

    if cluster_corr.empty:
        axes[1].axis("off")
        axes[1].set_title("Cluster-level ADAR correlation\n(empty)")
    else:
        sns.heatmap(
            cluster_corr,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            center=0,
            vmin=vmin,
            vmax=vmax,
            square=True,
            ax=axes[1],
            cbar_kws={"shrink": 0.7},
        )
        axes[1].set_title("Cluster-level ADAR correlation")

    plt.tight_layout()

    return fig, axes


def plot_cluster_adar_correlation_heatmap(
    corr_result,
    figsize=(4.6, 4.0),
    cmap="vlag",
    vmin=-1,
    vmax=1,
    title="ADAR-family cluster-level correlation",
):
    """
    Plot only the cluster-level ADAR-family correlation heatmap.
    """
    cluster_corr = corr_result.get("cluster_corr", pd.DataFrame())

    if cluster_corr.empty:
        raise ValueError("cluster_corr is empty.")

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cluster_corr,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        vmin=vmin,
        vmax=vmax,
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.75},
    )

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.tight_layout()

    return fig, ax


# =============================================================================
# 8. Global A-to-I vs ADAR-family expression
# =============================================================================

def analyze_score_vs_adar(
    adata_ai,
    score_key="global_atoi_score",
    adar_cols=("expr_ADAR", "expr_ADARB1", "expr_ADARB2"),
    covariates=("x_pixel", "y_pixel"),
    group_key="ground_truth",
    min_complete=30,
    store_key="global_atoi_vs_adar",
):
    """
    Analyze global A-to-I score versus ADAR-family expression.

    Includes:
        spot-level Spearman
        spot-level OLS adjusted for spatial covariates
        cluster-level Spearman
    """
    adar_cols = [c for c in adar_cols if c in adata_ai.obs.columns]
    covariates = [c for c in covariates if c in adata_ai.obs.columns]

    rows = []

    df = adata_ai.obs[[score_key] + adar_cols + covariates].copy()

    for adar in adar_cols:
        sub = df[[score_key, adar]].dropna()

        rho, p_spear, n_spear = _safe_corr(sub[adar], sub[score_key], method="spearman")

        model_cols = [adar] + covariates
        subm = df[[score_key] + model_cols].dropna()

        beta = p_ols = r2 = np.nan

        if (
            subm.shape[0] >= max(min_complete, len(model_cols) + 5)
            and subm[score_key].std() > 0
            and subm[adar].std() > 0
        ):
            X = _zscore_frame(subm[model_cols])
            y = subm[score_key].astype(float)
            y = (y - y.mean()) / y.std()

            fit = sm.OLS(y, sm.add_constant(X, has_constant="add")).fit()

            beta = float(fit.params[adar])
            p_ols = float(fit.pvalues[adar])
            r2 = float(fit.rsquared)

        rows.append(
            {
                "level": "spot",
                "score": score_key,
                "adar": adar.replace("expr_", ""),
                "adar_col": adar,
                "n": n_spear,
                "spearman_rho": rho,
                "spearman_p": p_spear,
                "adjusted_beta": beta,
                "adjusted_p": p_ols,
                "adjusted_r2": r2,
            }
        )

    if group_key in adata_ai.obs.columns:
        labels = _valid_group_series(adata_ai, group_key=group_key)

        cluster_df = adata_ai.obs[[score_key] + adar_cols].copy()
        cluster_df[group_key] = labels
        cluster_df = cluster_df.dropna(subset=[group_key])

        cluster_mean = cluster_df.groupby(group_key, observed=True)[[score_key] + adar_cols].mean()

        for adar in adar_cols:
            sub = cluster_mean[[score_key, adar]].dropna()
            rho, pval, n = _safe_corr(sub[adar], sub[score_key], method="spearman")

            rows.append(
                {
                    "level": "cluster",
                    "score": score_key,
                    "adar": adar.replace("expr_", ""),
                    "adar_col": adar,
                    "n": n,
                    "spearman_rho": rho,
                    "spearman_p": pval,
                    "adjusted_beta": np.nan,
                    "adjusted_p": np.nan,
                    "adjusted_r2": np.nan,
                }
            )

    res = pd.DataFrame(rows)

    if not res.empty:
        res["spearman_fdr"] = _safe_multipletest(res["spearman_p"])
        res["adjusted_fdr"] = _safe_multipletest(res["adjusted_p"])

    adata_ai.uns[store_key] = res

    return res


def plot_score_vs_adar_dashboard(
    adata_ai,
    score_key="global_atoi_score",
    adar_cols=("expr_ADAR", "expr_ADARB1", "expr_ADARB2"),
    spot_size=18,
    figsize=None,
):
    adar_cols = [c for c in adar_cols if c in adata_ai.obs.columns]

    nrows = len(adar_cols)

    if figsize is None:
        figsize = (12, 3.8 * nrows)

    fig, axes = plt.subplots(nrows, 3, figsize=figsize, squeeze=False)

    for r, adar in enumerate(adar_cols):
        gene = adar.replace("expr_", "")

        expr = adata_ai.obs[adar].astype(float).values
        score = adata_ai.obs[score_key].astype(float).values

        sc = _spatial_axes(
            axes[r, 0],
            adata_ai,
            expr,
            f"{gene} expression",
            cmap="YlGnBu",
            spot_size=spot_size,
        )
        if sc is not None:
            fig.colorbar(sc, ax=axes[r, 0], fraction=0.046, pad=0.02)

        sc = _spatial_axes(
            axes[r, 1],
            adata_ai,
            score,
            score_key,
            cmap="magma",
            spot_size=spot_size,
        )
        if sc is not None:
            fig.colorbar(sc, ax=axes[r, 1], fraction=0.046, pad=0.02)

        mask = np.isfinite(expr) & np.isfinite(score)

        if mask.sum() > 2:
            rho, pval = spearmanr(expr[mask], score[mask])
        else:
            rho, pval = np.nan, np.nan

        sns.regplot(
            x=expr[mask],
            y=score[mask],
            scatter_kws={"s": 14, "alpha": 0.45, "linewidth": 0},
            line_kws={"color": "black", "linewidth": 1.2},
            lowess=True,
            ax=axes[r, 2],
        )

        axes[r, 2].set_xlabel(f"{gene} expression")
        axes[r, 2].set_ylabel(score_key)
        axes[r, 2].set_title(f"rho={rho:.2f}, p={pval:.1e}")

    plt.tight_layout()

    return fig, axes


# =============================================================================
# 9. Bivariate co-localization
# =============================================================================

def _bivariate_classes(x, y, q=0.5):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    out = np.full(len(x), np.nan)

    if mask.sum() == 0:
        return out

    x_cut = np.nanquantile(x[mask], q)
    y_cut = np.nanquantile(y[mask], q)

    x_hi = x >= x_cut
    y_hi = y >= y_cut

    out[mask & ~x_hi & ~y_hi] = 0
    out[mask & x_hi & ~y_hi] = 1
    out[mask & ~x_hi & y_hi] = 2
    out[mask & x_hi & y_hi] = 3

    return out


def _prepare_bivariate_values(
    adata,
    x_values,
    y_values,
    mode="spot",
    group_key="ground_truth",
):
    x = pd.Series(np.asarray(x_values, dtype=float), index=adata.obs_names, name="x")
    y = pd.Series(np.asarray(y_values, dtype=float), index=adata.obs_names, name="y")

    mode = str(mode).lower()

    if mode == "spot":
        return x.values, y.values, x, y

    if mode != "cluster":
        raise ValueError("mode must be 'spot' or 'cluster'.")

    labels = _valid_group_series(adata, group_key=group_key)

    df = pd.DataFrame({"x": x, "y": y, "group": labels}).dropna()

    x_out = pd.Series(np.nan, index=adata.obs_names, dtype=float)
    y_out = pd.Series(np.nan, index=adata.obs_names, dtype=float)

    if not df.empty:
        group_mean = df.groupby("group", observed=True)[["x", "y"]].mean()

        valid = labels.notna()

        x_out.loc[valid] = labels.loc[valid].map(group_mean["x"]).astype(float)
        y_out.loc[valid] = labels.loc[valid].map(group_mean["y"]).astype(float)

    return x_out.values, y_out.values, x_out, y_out


def plot_bivariate_colocalization(
    adata,
    x_values,
    y_values,
    x_name="A-to-I",
    y_name="ADAR",
    q=0.5,
    mode="spot",
    group_key="ground_truth",
    show_legend=True,
    spot_size=18,
    figsize=(11.2, 4),
):
    """
    Plot:
        x map
        y map
        bivariate high/low co-localization map

    mode:
        "spot"    : use spot-level values
        "cluster" : aggregate by group_key, then map group means back to spots
    """
    x_plot, y_plot, _, _ = _prepare_bivariate_values(
        adata,
        x_values=x_values,
        y_values=y_values,
        mode=mode,
        group_key=group_key,
    )

    classes = _bivariate_classes(x_plot, y_plot, q=q)

    colors = ["#e8e8e8", "#64acbe", "#c85a5a", "#574249"]
    cmap_bivar = plt.matplotlib.colors.ListedColormap(colors)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    sc = _spatial_axes(
        axes[0],
        adata,
        x_plot,
        f"{x_name}\n({mode})",
        cmap="magma",
        spot_size=spot_size,
    )
    if sc is not None:
        fig.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.02)

    sc = _spatial_axes(
        axes[1],
        adata,
        y_plot,
        f"{y_name}\n({mode})",
        cmap="YlGnBu",
        spot_size=spot_size,
    )
    if sc is not None:
        fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.02)

    _spatial_axes(
        axes[2],
        adata,
        classes,
        f"Bivariate\n{x_name} x {y_name}",
        cmap=cmap_bivar,
        spot_size=spot_size,
        clip=(0, 100),
    )

    if show_legend:
        handles = [
            Patch(facecolor=colors[0], edgecolor="none", label=f"low {x_name} / low {y_name}"),
            Patch(facecolor=colors[1], edgecolor="none", label=f"high {x_name} only"),
            Patch(facecolor=colors[2], edgecolor="none", label=f"high {y_name} only"),
            Patch(facecolor=colors[3], edgecolor="none", label=f"high {x_name} / high {y_name}"),
        ]

        axes[2].legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.12),
            fontsize=7,
            frameon=False,
            ncol=1,
        )

    plt.tight_layout()

    return fig, axes


# =============================================================================
# 10. Global A-to-I vs cell types
# =============================================================================

def analyze_score_vs_celltypes(
    adata_ai,
    df_deconv: pd.DataFrame,
    score_key: str = "global_atoi_score",
    celltype_cols: Optional[Sequence[str]] = None,
    covariates=("x_pixel", "y_pixel"),
    min_complete: int = 30,
    level: str = "spot",
    group_key: str = "ground_truth",
    cluster_agg: str = "mean",
    store_key: str = "global_atoi_vs_celltypes",
):
    """
    Analyze global A-to-I score versus each deconvolved cell type proportion.

    df_deconv should already be merged/processed in notebook.
    level:
        "spot"    : spot-level association
        "cluster" : aggregate score, covariates and deconvolution by group_key
    """
    level = str(level).lower()

    if level not in ["spot", "cluster"]:
        raise ValueError("level must be 'spot' or 'cluster'.")

    common = adata_ai.obs_names.intersection(df_deconv.index)

    if len(common) == 0:
        raise ValueError("No common spots between adata_ai and df_deconv.")

    covariates = [c for c in covariates if c in adata_ai.obs.columns]

    if celltype_cols is None:
        celltype_cols = [
            c for c in df_deconv.columns
            if pd.api.types.is_numeric_dtype(df_deconv[c])
        ]

    celltype_cols = [c for c in celltype_cols if c in df_deconv.columns]

    if level == "cluster":
        df, _ = _aggregate_analysis_frame_by_group(
            adata_ai=adata_ai,
            df_extra=df_deconv,
            columns=[score_key] + list(covariates) + list(celltype_cols),
            group_key=group_key,
            aggfunc=cluster_agg,
        )
        unit_col = "n_clusters"
    else:
        df = (
            adata_ai.obs.loc[common, [score_key] + covariates]
            .join(df_deconv.loc[common, celltype_cols], how="left")
        )
        unit_col = "n_spots"

    rows = []

    for ct in celltype_cols:
        cols = [score_key, ct] + covariates
        sub = df[cols].dropna()

        if sub.shape[0] < min_complete:
            continue

        if sub[ct].std() == 0 or sub[score_key].std() == 0:
            continue

        rho, p_spear = spearmanr(sub[ct], sub[score_key])

        X = _zscore_frame(sub[[ct] + covariates])

        y = sub[score_key].astype(float)
        y = (y - y.mean()) / y.std()

        fit = sm.OLS(y, sm.add_constant(X, has_constant="add")).fit()

        rows.append(
            {
                "celltype": ct,
                "level": level,
                "group_key": group_key if level == "cluster" else np.nan,
                unit_col: int(sub.shape[0]),
                "spearman_rho": float(rho),
                "spearman_p": float(p_spear),
                "adjusted_beta": float(fit.params[ct]),
                "adjusted_p": float(fit.pvalues[ct]),
                "adjusted_r2": float(fit.rsquared),
            }
        )

    res = pd.DataFrame(rows)

    if not res.empty:
        if "n_spots" not in res.columns:
            res["n_spots"] = np.nan
        if "n_clusters" not in res.columns:
            res["n_clusters"] = np.nan
        res["spearman_fdr"] = _safe_multipletest(res["spearman_p"])
        res["adjusted_fdr"] = _safe_multipletest(res["adjusted_p"])
        res = res.sort_values(["adjusted_fdr", "spearman_fdr"]).reset_index(drop=True)

    adata_ai.uns[store_key] = res

    return res


def build_cluster_feature_matrix(
    adata_ai,
    df_deconv: Optional[pd.DataFrame] = None,
    feature_cols: Sequence[str] = (),
    group_key: str = "ground_truth",
    aggfunc: str = "mean",
):
    """
    Build a cluster-level feature matrix for descriptive heatmaps.

    feature_cols can include columns from adata_ai.obs and df_deconv.
    """
    feature_cols = list(dict.fromkeys(feature_cols))

    mat, _ = _aggregate_analysis_frame_by_group(
        adata_ai=adata_ai,
        df_extra=df_deconv,
        columns=feature_cols,
        group_key=group_key,
        aggfunc=aggfunc,
    )

    mat = mat[[c for c in feature_cols if c in mat.columns]]

    return mat


def plot_cluster_feature_heatmap(
    cluster_matrix: pd.DataFrame,
    zscore_columns: bool = True,
    figsize=(8.5, 4.4),
    cmap="vlag",
    center=0,
    title="Cluster-level feature concordance",
):
    """
    Plot a cluster x feature heatmap, optionally z-scoring each feature.
    """
    if cluster_matrix.empty:
        raise ValueError("cluster_matrix is empty.")

    plot_df = cluster_matrix.astype(float).copy()

    if zscore_columns:
        plot_df = _zscore_frame(plot_df)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        plot_df,
        cmap=cmap,
        center=center,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "column z-score" if zscore_columns else "value"},
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()

    return fig, ax, plot_df


def plot_cluster_score_comparison(
    adata,
    score_keys=("global_atoi_score", "sv_atoi_score"),
    titles=("Global A-to-I", "SV-A-to-I"),
    group_key="ground_truth",
    spot_size=18,
    cmap="magma",
    figsize=(11.5, 6.8),
):
    """
    Compare two cluster-level A-to-I scores with maps and group means.
    """
    score_keys = list(score_keys)
    titles = list(titles)

    missing = [k for k in score_keys if k not in adata.obs.columns]

    if missing:
        raise ValueError(f"Missing score columns: {missing}")

    labels = _valid_group_series(adata, group_key=group_key)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, len(score_keys), height_ratios=[1.0, 0.85])
    map_axes = [fig.add_subplot(gs[0, i]) for i in range(len(score_keys))]
    bar_ax = fig.add_subplot(gs[1, :])

    rows = []

    for ax, key, title in zip(map_axes, score_keys, titles):
        values = adata.obs[key].astype(float)
        cluster_values = _map_group_mean_to_spots(values, labels)

        sc = _spatial_axes(
            ax,
            adata,
            cluster_values,
            f"Cluster-level {title}",
            cmap=cmap,
            spot_size=spot_size,
        )

        if sc is not None:
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)

        df = pd.DataFrame({"group": labels, "value": values}).dropna()

        if not df.empty:
            group_mean = df.groupby("group", observed=True)["value"].mean()

            for group, value in group_mean.items():
                rows.append(
                    {
                        "group": group,
                        "score": title,
                        "score_key": key,
                        "value": float(value),
                    }
                )

    summary = pd.DataFrame(rows)

    if summary.empty:
        raise ValueError("No group-level score values were available.")

    first_title = titles[0]
    order = (
        summary[summary["score"] == first_title]
        .sort_values("value")["group"]
        .tolist()
    )

    sns.barplot(
        data=summary,
        x="group",
        y="value",
        hue="score",
        order=order,
        hue_order=titles,
        ax=bar_ax,
    )

    bar_ax.axhline(0, color="black", linewidth=1)
    bar_ax.set_xlabel("")
    bar_ax.set_ylabel("cluster mean score")
    bar_ax.set_title("Global vs SV-A-to-I cluster-level score comparison")
    bar_ax.tick_params(axis="x", rotation=45)
    bar_ax.legend(frameon=False)

    wide = summary.pivot(index="group", columns="score", values="value")

    if len(titles) == 2 and set(titles).issubset(wide.columns):
        sub = wide[titles].dropna()

        if sub.shape[0] >= 3:
            rho, pval = spearmanr(sub[titles[0]], sub[titles[1]])
            bar_ax.text(
                0.99,
                0.96,
                f"cluster rho={rho:.2f}, p={pval:.2g}",
                transform=bar_ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
            )

    plt.tight_layout()

    return fig, {"maps": map_axes, "bar": bar_ax}, summary


def plot_association_dotplot(
    assoc_df: pd.DataFrame,
    label_col: str,
    beta_col: str = "adjusted_beta",
    fdr_col: str = "adjusted_fdr",
    rho_col: str = "spearman_rho",
    p_col: str = "spearman_p",
    sort_col: Optional[str] = None,
    top_n: Optional[int] = None,
    show_fdr: bool = False,
    title: str = "Descriptive association with global A-to-I score",
    figsize=(7, 4),
):
    df = assoc_df.copy()

    if df.empty:
        raise ValueError("assoc_df is empty.")

    if sort_col is None:
        sort_col = rho_col if rho_col in df.columns else beta_col

    if sort_col not in df.columns:
        raise ValueError(f"{sort_col!r} not found in assoc_df.")

    df = df.sort_values(sort_col)

    if top_n is not None and df.shape[0] > top_n:
        lower = df.head(top_n // 2)
        upper = df.tail(top_n - lower.shape[0])
        df = pd.concat([lower, upper], axis=0)

    y = np.arange(df.shape[0])

    if p_col in df.columns:
        sig = -np.log10(df[p_col].clip(lower=1e-300))
    elif fdr_col in df.columns:
        sig = -np.log10(df[fdr_col].clip(lower=1e-300))
    else:
        sig = pd.Series(1.0, index=df.index)

    sizes = np.clip(30 + sig * 25, 40, 260)
    color_values = df[rho_col] if rho_col in df.columns else df[beta_col]
    colors = np.where(color_values >= 0, "#d95f02", "#1b9e77")

    fig, ax = plt.subplots(figsize=figsize)

    ax.hlines(y, 0, df[sort_col], color="#bdbdbd", linewidth=1.5)
    ax.scatter(
        df[sort_col],
        y,
        s=sizes,
        c=colors,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    ax.axvline(0, color="black", linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(df[label_col].astype(str))
    ax.set_xlabel(sort_col.replace("_", " "))
    ax.set_title(title)

    for yi, (_, row) in enumerate(df.iterrows()):
        if show_fdr and fdr_col in row:
            label = f"FDR={row[fdr_col]:.1e}"
        elif p_col in row:
            label = f"p={row[p_col]:.2g}"
        else:
            label = ""

        if label:
            ax.text(
                row[sort_col],
                yi + 0.18,
                label,
                fontsize=7,
                ha="center",
            )

    plt.tight_layout()

    return fig, ax


def plot_model_r2_comparison(
    summaries,
    labels=("Cell type", "Cell type + ADAR"),
    r2_key="r2",
    adj_r2_key="adj_r2",
    title="Cluster-level model fit",
    figsize=(5.2, 3.8),
):
    """
    Plot R2 and adjusted R2 for model summary dictionaries.
    """
    rows = []

    for label, summary in zip(labels, summaries):
        rows.append({"model": label, "metric": "R2", "value": summary.get(r2_key, np.nan)})
        rows.append({"model": label, "metric": "adjusted R2", "value": summary.get(adj_r2_key, np.nan)})

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(
        data=df,
        x="model",
        y="value",
        hue="metric",
        palette=["#4c78a8", "#f58518"],
        ax=ax,
    )

    ax.set_ylim(0, max(1.0, np.nanmax(df["value"]) * 1.08))
    ax.set_ylabel("variance explained")
    ax.set_xlabel("")
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper left")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)

    plt.tight_layout()

    return fig, ax, df


def plot_cross_sample_model_summary(
    model_df: pd.DataFrame,
    sample_col: str = "sample_id",
    figsize=(8.5, 4.2),
):
    """
    Plot cross-sample cell-type-only vs joint model fit.

    Required columns:
        sample_id, celltype_r2, joint_r2
    Optional:
        celltype_adj_r2, joint_adj_r2, n_clusters
    """
    if model_df.empty:
        raise ValueError("model_df is empty.")

    df = model_df.copy()

    required = {sample_col, "celltype_r2", "joint_r2"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["delta_r2"] = df["joint_r2"] - df["celltype_r2"]

    long_rows = []

    for _, row in df.iterrows():
        long_rows.append(
            {
                sample_col: row[sample_col],
                "model": "Cell type",
                "R2": row["celltype_r2"],
            }
        )
        long_rows.append(
            {
                sample_col: row[sample_col],
                "model": "Cell type + ADAR",
                "R2": row["joint_r2"],
            }
        )

    long_df = pd.DataFrame(long_rows)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1.55, 0.85]},
    )

    sns.barplot(
        data=long_df,
        x=sample_col,
        y="R2",
        hue="model",
        palette=["#4c78a8", "#f58518"],
        ax=axes[0],
    )

    axes[0].set_ylim(0, max(1.0, np.nanmax(long_df["R2"]) * 1.08))
    axes[0].set_xlabel("")
    axes[0].set_ylabel("R2")
    axes[0].set_title("Cross-sample model fit")
    axes[0].legend(frameon=False, fontsize=8)

    sns.barplot(
        data=df,
        x=sample_col,
        y="delta_r2",
        color="#8da0cb",
        edgecolor="black",
        linewidth=0.5,
        ax=axes[1],
    )

    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Joint R2 - cell-type R2")
    axes[1].set_title("ADAR-family gain")

    if "n_clusters" in df.columns:
        for idx, row in df.reset_index(drop=True).iterrows():
            axes[1].text(
                idx,
                row["delta_r2"],
                f"n={int(row['n_clusters'])}",
                ha="center",
                va="bottom" if row["delta_r2"] >= 0 else "top",
                fontsize=8,
            )

    plt.tight_layout()

    return fig, axes, df, long_df


# =============================================================================
# 11. Adjustment and joint model
# =============================================================================

def fit_adjustment_model(
    adata_ai,
    score_key,
    predictors,
    df_extra=None,
    covariates=(),
    residual_key="global_atoi_residual",
    min_complete=30,
    drop_reference=None,
    standardize=True,
    level: str = "spot",
    group_key: str = "ground_truth",
    cluster_agg: str = "mean",
    store_key="adjustment_model",
):
    """
    Fit OLS:
        score ~ predictors + covariates

    Adds residual to adata_ai.obs[residual_key].

    level:
        "spot"    : fit on individual spots
        "cluster" : fit on group-level means/medians and map residuals to spots

    drop_reference:
        Optional predictor to remove, useful for compositional cell-type proportions.
        Example: drop_reference="Mix"
    """
    level = str(level).lower()

    if level not in ["spot", "cluster"]:
        raise ValueError("level must be 'spot' or 'cluster'.")

    if level == "cluster":
        all_cols = [score_key] + list(predictors) + list(covariates)
        df, labels = _aggregate_analysis_frame_by_group(
            adata_ai=adata_ai,
            df_extra=df_extra,
            columns=all_cols,
            group_key=group_key,
            aggfunc=cluster_agg,
        )
    else:
        labels = None
        common = adata_ai.obs_names

        if df_extra is not None:
            common = common.intersection(df_extra.index)

        df = adata_ai.obs.loc[common, [score_key]].copy()

        obs_predictors = [c for c in predictors if c in adata_ai.obs.columns]
        obs_covariates = [c for c in covariates if c in adata_ai.obs.columns]

        if obs_predictors:
            df = df.join(adata_ai.obs.loc[common, obs_predictors], how="left")

        if obs_covariates:
            df = df.join(adata_ai.obs.loc[common, obs_covariates], how="left")

        if df_extra is not None:
            extra_cols = [
                c for c in list(predictors) + list(covariates)
                if c in df_extra.columns and c not in df.columns
            ]

            if extra_cols:
                df = df.join(df_extra.loc[common, extra_cols], how="left")

    model_cols = [
        c for c in list(predictors) + list(covariates)
        if c in df.columns
    ]

    if drop_reference is not None and drop_reference in model_cols:
        model_cols = [c for c in model_cols if c != drop_reference]

    keep_cols = []
    dropped_zero_var = []

    for c in model_cols:
        x = pd.to_numeric(df[c], errors="coerce")

        if x.dropna().shape[0] > 0 and x.std(skipna=True) > 1e-10:
            keep_cols.append(c)
        else:
            dropped_zero_var.append(c)

    model_cols = keep_cols

    sub = (
        df[[score_key] + model_cols]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    min_model_units = len(model_cols) + (2 if level == "cluster" else 5)

    if sub.shape[0] < max(min_complete, min_model_units):
        unit_name = "clusters" if level == "cluster" else "spots"
        raise ValueError(
            f"Not enough complete {unit_name} for adjustment model: "
            f"n={sub.shape[0]}, predictors={len(model_cols)}"
        )

    if sub[score_key].std() == 0:
        raise ValueError(f"{score_key} has zero variance.")

    X = sub[model_cols].astype(float).copy()

    if standardize:
        X = _zscore_frame(X)

    y = sub[score_key].astype(float)

    fit = sm.OLS(y, sm.add_constant(X, has_constant="add")).fit()

    if level == "cluster":
        cluster_residual = pd.Series(fit.resid, index=sub.index, name=f"{residual_key}_cluster")
        residual = pd.Series(np.nan, index=adata_ai.obs_names, name=residual_key)
        valid = labels.notna() & labels.isin(cluster_residual.index)
        residual.loc[valid] = labels.loc[valid].map(cluster_residual).astype(float)
        adata_ai.uns[f"{store_key}_cluster_residual"] = cluster_residual
    else:
        residual = pd.Series(np.nan, index=adata_ai.obs_names, name=residual_key)
        residual.loc[sub.index] = fit.resid

    adata_ai.obs[residual_key] = residual

    params = fit.params.to_frame("coef").join(fit.pvalues.to_frame("pval"))
    params["term"] = params.index
    params = params.reset_index(drop=True)
    params["padj"] = _safe_multipletest(params["pval"])

    try:
        condition_number = float(
            np.linalg.cond(sm.add_constant(X, has_constant="add").values)
        )
    except Exception:
        condition_number = np.nan

    summary = {
        "level": level,
        "group_key": group_key if level == "cluster" else np.nan,
        "cluster_agg": cluster_agg if level == "cluster" else np.nan,
        "n_spots": int(sub.shape[0]) if level == "spot" else np.nan,
        "n_clusters": int(sub.shape[0]) if level == "cluster" else np.nan,
        "r2": float(fit.rsquared),
        "adj_r2": float(fit.rsquared_adj),
        "residual_key": residual_key,
        "predictors": model_cols,
        "drop_reference": drop_reference,
        "dropped_zero_var": dropped_zero_var,
        "condition_number": condition_number,
    }

    adata_ai.uns[store_key] = summary
    adata_ai.uns[f"{store_key}_params"] = params

    return residual, fit, params, summary


def fit_joint_atoi_model(
    adata_ai,
    df_deconv,
    score_key="global_atoi_score",
    celltype_cols=None,
    adar_cols=("expr_ADAR", "expr_ADARB1", "expr_ADARB2"),
    covariates=("x_pixel", "y_pixel"),
    residual_key="global_atoi_residual_celltype_adar",
    drop_reference="Mix",
    min_complete=30,
    level: str = "spot",
    group_key: str = "ground_truth",
    cluster_agg: str = "mean",
    store_key="joint_global_atoi_model",
):
    """
    Joint model:
        global_atoi_score ~ cell types + ADAR-family + spatial covariates
    """
    if celltype_cols is None:
        celltype_cols = [
            c for c in df_deconv.columns
            if pd.api.types.is_numeric_dtype(df_deconv[c])
        ]

    adar_cols = [c for c in adar_cols if c in adata_ai.obs.columns]

    predictors = list(celltype_cols) + list(adar_cols)

    return fit_adjustment_model(
        adata_ai=adata_ai,
        score_key=score_key,
        predictors=predictors,
        df_extra=df_deconv,
        covariates=covariates,
        residual_key=residual_key,
        min_complete=min_complete,
        drop_reference=drop_reference,
        level=level,
        group_key=group_key,
        cluster_agg=cluster_agg,
        store_key=store_key,
    )


# =============================================================================
# 12. Residual visualization
# =============================================================================

def plot_residual_progression(
    adata_ai,
    keys=(
        "global_atoi_score",
        "global_atoi_residual_celltype",
        "global_atoi_residual_celltype_adar",
    ),
    titles=(
        "Original global A-to-I",
        "Residual after cell types",
        "Residual after cell types + ADAR",
    ),
    group_key="ground_truth",
    spot_size=18,
    kind="bar",
    figsize=(13, 7.2),
):
    """
    Show cluster-level progression:
        original global A-to-I
        cell-type-adjusted residual
        cell-type + ADAR-adjusted residual

    Top row: cluster-level spatial maps.
    Bottom row: cluster means and between-cluster range.
    """
    kind = str(kind).lower()

    if kind not in ["bar", "line"]:
        raise ValueError("kind must be 'bar' or 'line'.")

    labels = _valid_group_series(adata_ai, group_key=group_key)
    rows = []

    for i, (key, title) in enumerate(zip(keys, titles)):
        if key not in adata_ai.obs.columns:
            continue

        values = adata_ai.obs[key].astype(float)
        plot_df = pd.DataFrame(
            {
                "group": labels,
                "value": values,
            }
        ).dropna()

        if not plot_df.empty:
            group_mean = plot_df.groupby("group", observed=True)["value"].mean()

            for group, value in group_mean.items():
                rows.append(
                    {
                        "group": group,
                        "metric": title,
                        "key": key,
                        "value": float(value),
                    }
                )

    summary = pd.DataFrame(rows)

    if summary.empty:
        raise ValueError("No residual progression values were available.")

    metric_order = [title for key, title in zip(keys, titles) if key in adata_ai.obs.columns]

    if metric_order and (summary["metric"] == metric_order[0]).any():
        order = (
            summary[summary["metric"] == metric_order[0]]
            .sort_values("value")["group"]
            .tolist()
        )
    else:
        order = (
            summary.groupby("group", observed=True)["value"]
            .mean()
            .sort_values()
            .index
            .tolist()
        )

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 0.82],
        width_ratios=[1.0, 1.0, 1.0],
    )

    map_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    trend_ax = fig.add_subplot(gs[1, :2])
    range_ax = fig.add_subplot(gs[1, 2])

    for i, (key, title) in enumerate(zip(keys, titles)):
        if i >= len(map_axes):
            break

        ax = map_axes[i]

        if key not in adata_ai.obs.columns:
            ax.axis("off")
            continue

        values = adata_ai.obs[key].astype(float)
        cluster_values = _map_group_mean_to_spots(values, labels)

        center = 0 if "residual" in key else None
        cmap = "coolwarm" if "residual" in key else "magma"

        sc = _spatial_axes(
            ax,
            adata_ai,
            cluster_values,
            title,
            cmap=cmap,
            spot_size=spot_size,
            center=center,
        )

        if sc is not None:
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)

    if kind == "bar":
        sns.barplot(
            data=summary,
            x="group",
            y="value",
            hue="metric",
            order=order,
            hue_order=metric_order,
            ax=trend_ax,
        )
    else:
        wide = summary.pivot(index="group", columns="metric", values="value").reindex(order)
        wide.loc[:, metric_order].plot(marker="o", linewidth=1.8, ax=trend_ax)

    trend_ax.axhline(0, color="black", linewidth=1)
    trend_ax.set_xlabel("")
    trend_ax.set_ylabel("cluster mean value")
    trend_ax.set_title("Cluster-level A-to-I residual progression")
    trend_ax.tick_params(axis="x", rotation=45)
    trend_ax.legend(frameon=False, fontsize=8)

    range_df = (
        summary.groupby("metric", observed=True)["value"]
        .agg(lambda x: float(np.nanmax(x) - np.nanmin(x)))
        .reindex(metric_order)
        .reset_index(name="cluster_range")
    )

    sns.barplot(
        data=range_df,
        y="metric",
        x="cluster_range",
        color="#8da0cb",
        edgecolor="black",
        linewidth=0.5,
        ax=range_ax,
    )
    range_ax.set_xlabel("max - min")
    range_ax.set_ylabel("")
    range_ax.set_title("Between-cluster range")

    plt.tight_layout()

    axes = {
        "maps": map_axes,
        "trend": trend_ax,
        "range": range_ax,
    }

    return fig, axes, summary


# =============================================================================
# 13. Site-level ADAR correlation
# =============================================================================

def run_site_adar_correlation(
    adata_ai,
    candidate_sites=None,
    adar_cols=("expr_ADAR", "expr_ADARB1", "expr_ADARB2"),
    group_key="ground_truth",
    min_cov=10,
    min_valid_spots=30,
    min_cluster_groups=3,
    a_layer="A",
    g_layer="G",
    store_key="site_adar_correlation",
    verbose=True,
):
    """
    Site-level ADAR association by correlation.

    For each site and each ADAR:
        1. spot-level Spearman correlation
        2. cluster-level Spearman correlation using group-level mean editing and expression
    """
    adar_cols = [c for c in adar_cols if c in adata_ai.obs.columns]

    if len(adar_cols) == 0:
        raise ValueError("No ADAR columns found in adata_ai.obs.")

    if isinstance(candidate_sites, pd.DataFrame):
        df_sites = candidate_sites.copy()

        if "is_sv_atoi" in df_sites.columns and df_sites["is_sv_atoi"].any():
            df_sites = df_sites[df_sites["is_sv_atoi"] == True].copy()

        sites = [s for s in df_sites["site"].astype(str) if s in adata_ai.var_names]

    elif candidate_sites is None:
        sites = list(adata_ai.var_names)

    else:
        sites = [str(s) for s in candidate_sites if str(s) in adata_ai.var_names]

    labels = None

    if group_key is not None and group_key in adata_ai.obs.columns:
        labels = _valid_group_series(adata_ai, group_key=group_key)

    rows = []

    for i, site in enumerate(sites):
        ratio = get_site_editing_ratio(
            adata_ai,
            site=site,
            min_cov=min_cov,
            a_layer=a_layer,
            g_layer=g_layer,
        )

        if "Gene.refGene" in adata_ai.var.columns:
            gene = adata_ai.var.loc[site].get("Gene.refGene", site)
        else:
            gene = site

        for adar_col in adar_cols:
            expr = adata_ai.obs[adar_col].astype(float)

            spot_df = pd.DataFrame(
                {
                    "ratio": ratio,
                    "expr": expr,
                },
                index=adata_ai.obs_names,
            ).dropna()

            if (
                spot_df.shape[0] >= min_valid_spots
                and spot_df["ratio"].std() > 0
                and spot_df["expr"].std() > 0
            ):
                rho, pval = spearmanr(spot_df["expr"], spot_df["ratio"])
            else:
                rho, pval = np.nan, np.nan

            rows.append(
                {
                    "site": site,
                    "gene": gene,
                    "adar": adar_col.replace("expr_", ""),
                    "adar_col": adar_col,
                    "level": "spot",
                    "n": int(spot_df.shape[0]),
                    "spearman_rho": rho,
                    "spearman_p": pval,
                    "mean_ratio": float(spot_df["ratio"].mean()) if not spot_df.empty else np.nan,
                    "sd_ratio": float(spot_df["ratio"].std()) if not spot_df.empty else np.nan,
                }
            )

            if labels is not None:
                cl_df = pd.DataFrame(
                    {
                        "ratio": ratio,
                        "expr": expr,
                        "group": labels,
                    },
                    index=adata_ai.obs_names,
                ).dropna()

                if not cl_df.empty:
                    cl_mean = cl_df.groupby("group", observed=True)[["ratio", "expr"]].mean()
                else:
                    cl_mean = pd.DataFrame(columns=["ratio", "expr"])

                if (
                    cl_mean.shape[0] >= min_cluster_groups
                    and cl_mean["ratio"].std() > 0
                    and cl_mean["expr"].std() > 0
                ):
                    rho_c, pval_c = spearmanr(cl_mean["expr"], cl_mean["ratio"])
                else:
                    rho_c, pval_c = np.nan, np.nan

                rows.append(
                    {
                        "site": site,
                        "gene": gene,
                        "adar": adar_col.replace("expr_", ""),
                        "adar_col": adar_col,
                        "level": "cluster",
                        "n": int(cl_mean.shape[0]),
                        "spearman_rho": rho_c,
                        "spearman_p": pval_c,
                        "mean_ratio": float(cl_mean["ratio"].mean()) if not cl_mean.empty else np.nan,
                        "sd_ratio": float(cl_mean["ratio"].std()) if not cl_mean.empty else np.nan,
                    }
                )

        if verbose and (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(sites)} candidate sites")

    res = pd.DataFrame(rows)

    if not res.empty:
        res["spearman_fdr"] = np.nan

        for level in res["level"].dropna().unique():
            for adar in res["adar"].dropna().unique():
                idx = (res["level"] == level) & (res["adar"] == adar)
                res.loc[idx, "spearman_fdr"] = _safe_multipletest(res.loc[idx, "spearman_p"])

        res["abs_rho"] = res["spearman_rho"].abs()

        best = (
            res.sort_values(
                ["site", "level", "spearman_fdr", "abs_rho"],
                ascending=[True, True, True, False],
            )
            .drop_duplicates(["site", "level"])
            [["site", "level", "adar", "adar_col", "spearman_rho", "spearman_fdr"]]
            .rename(
                columns={
                    "adar": "best_adar",
                    "adar_col": "best_adar_col",
                    "spearman_rho": "best_adar_rho",
                    "spearman_fdr": "best_adar_fdr",
                }
            )
        )

        res = res.merge(best, on=["site", "level"], how="left")

        res = res.sort_values(
            ["level", "spearman_fdr", "abs_rho"],
            ascending=[True, True, False],
        ).reset_index(drop=True)

    adata_ai.uns[store_key] = res

    return res


# =============================================================================
# 14. Top site-ADAR co-localization examples
# =============================================================================

def plot_top_site_adar_colocalization(
    adata_ai,
    site_adar_assoc,
    n_examples=6,
    min_cov=10,
    q=0.5,
    level="cluster",
    mode=None,
    group_key="ground_truth",
    spot_size=16,
    figsize=None,
):
    """
    Plot top site-ADAR examples:
        site editing ratio map
        ADAR expression map
        bivariate map
        scatter
    """
    if site_adar_assoc.empty:
        raise ValueError("site_adar_assoc is empty.")

    if mode is None:
        mode = level

    df = site_adar_assoc.copy()

    if "level" in df.columns:
        df = df[df["level"] == level].copy()

    if df.empty:
        raise ValueError(f"No site-ADAR rows for level={level!r}.")

    if "spearman_fdr" in df.columns:
        fdr_col = "spearman_fdr"
    elif "padj" in df.columns:
        fdr_col = "padj"
    else:
        fdr_col = None

    if "abs_rho" not in df.columns and "spearman_rho" in df.columns:
        df["abs_rho"] = df["spearman_rho"].abs()

    sort_cols = []
    ascending = []

    if fdr_col is not None:
        sort_cols.append(fdr_col)
        ascending.append(True)

    if "abs_rho" in df.columns:
        sort_cols.append("abs_rho")
        ascending.append(False)

    if sort_cols:
        df = df.sort_values(sort_cols, ascending=ascending)

    pairs = df.drop_duplicates(["site", "adar_col"]).head(n_examples)

    if figsize is None:
        figsize = (15, 3.8 * len(pairs))

    fig, axes = plt.subplots(len(pairs), 4, figsize=figsize)
    axes = np.atleast_2d(axes)

    for r, (_, row) in enumerate(pairs.iterrows()):
        site = row["site"]
        adar_col = row["adar_col"]
        adar_name = row.get("adar", adar_col.replace("expr_", ""))

        ratio = get_site_editing_ratio(adata_ai, site=site, min_cov=min_cov)
        expr = adata_ai.obs[adar_col].astype(float)

        ratio_plot, expr_plot, _, _ = _prepare_bivariate_values(
            adata_ai,
            x_values=ratio.values,
            y_values=expr.values,
            mode=mode,
            group_key=group_key,
        )

        classes = _bivariate_classes(ratio_plot, expr_plot, q=q)

        colors = ["#e8e8e8", "#64acbe", "#c85a5a", "#574249"]
        cmap_bivar = plt.matplotlib.colors.ListedColormap(colors)

        if site in adata_ai.var.index and "Gene.refGene" in adata_ai.var.columns:
            site_gene = adata_ai.var.loc[site].get("Gene.refGene", site)
        else:
            site_gene = site

        sc = _spatial_axes(
            axes[r, 0],
            adata_ai,
            ratio_plot,
            f"{site_gene} editing\n{site}",
            cmap="magma",
            spot_size=spot_size,
        )
        if sc is not None:
            fig.colorbar(sc, ax=axes[r, 0], fraction=0.046, pad=0.02)

        sc = _spatial_axes(
            axes[r, 1],
            adata_ai,
            expr_plot,
            f"{adar_name} expression",
            cmap="YlGnBu",
            spot_size=spot_size,
        )
        if sc is not None:
            fig.colorbar(sc, ax=axes[r, 1], fraction=0.046, pad=0.02)

        _spatial_axes(
            axes[r, 2],
            adata_ai,
            classes,
            f"Bivariate map\n{mode}",
            cmap=cmap_bivar,
            spot_size=spot_size,
            clip=(0, 100),
        )

        handles = [
            Patch(facecolor=colors[0], edgecolor="none", label="low editing / low ADAR"),
            Patch(facecolor=colors[1], edgecolor="none", label="high editing only"),
            Patch(facecolor=colors[2], edgecolor="none", label="high ADAR only"),
            Patch(facecolor=colors[3], edgecolor="none", label="high editing / high ADAR"),
        ]

        axes[r, 2].legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.12),
            fontsize=7,
            frameon=False,
        )

        mask = np.isfinite(ratio_plot) & np.isfinite(expr_plot)

        if mask.sum() > 2:
            rho, pval = spearmanr(expr_plot[mask], ratio_plot[mask])
        else:
            rho, pval = np.nan, np.nan

        sns.regplot(
            x=expr_plot[mask],
            y=ratio_plot[mask],
            scatter_kws={"s": 14, "alpha": 0.45, "linewidth": 0},
            line_kws={"color": "black"},
            lowess=True,
            ax=axes[r, 3],
        )

        axes[r, 3].set_xlabel(f"{adar_name} expression")
        axes[r, 3].set_ylabel("site editing ratio")

        if fdr_col is not None:
            axes[r, 3].set_title(f"{mode} rho={rho:.2f}, FDR={row[fdr_col]:.1e}")
        else:
            axes[r, 3].set_title(f"{mode} rho={rho:.2f}, p={pval:.1e}")

    plt.tight_layout()

    return fig, axes
