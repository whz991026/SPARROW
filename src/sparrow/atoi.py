"""Functions and required helpers used by DLPFC_AtoI_improved_workflow_new_3.ipynb.

All project analysis, plotting, cell-type grouping and PDF export live here.
"""
from __future__ import annotations

import re
import gzip
import mmap
import warnings
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import sparse
from scipy.stats import spearmanr, pearsonr, mannwhitneyu, fisher_exact
from statsmodels.stats.multitest import multipletests
from matplotlib.patches import Patch
from scipy.linalg import qr
from scipy.optimize import minimize
from scipy.special import expit, logit, gammaln, betaln
from scipy.stats import chi2, norm
import unicodedata

__all__ = [
    "add_celltype_composition_modules",
    "add_gene_expression_to_obs",
    "add_obs_zmean_score",
    "analyze_adar_spatial_correlation",
    "analyze_recurrent_sv_wm_binomial",
    "analyze_score_vs_celltypes",
    "annotate_sv_atoi_sites",
    "collapse_deconvolution_to_clusters",
    "compute_global_atoi_ratio",
    "compute_multisite_atoi_score",
    "compute_obs_spatial_autocorrelation",
    "compute_sv_atoi_score",
    "detect_spatial_atoi_sites_counts",
    "filter_atoi_sites",
    "fit_adjustment_model",
    "fit_joint_atoi_model",
    "infer_sv_site_mechanism_context",
    "install_figure_pdf_export",
    "parse_dlpfc_celltype_group",
    "plot_bivariate_colocalization",
    "plot_celltype_module_deconvolution_summary",
    "plot_cluster_adar_correlation_heatmap",
    "plot_cluster_level_obs_panel",
    "plot_cluster_level_obs_value",
    "plot_cluster_score_comparison",
    "plot_count_spatial_discovery",
    "plot_cross_sample_r2_decomposition",
    "plot_cross_sample_site_spatial_ratios",
    "plot_external_overlap_summary",
    "plot_global_atoi_two_r2_donuts",
    "plot_recurrent_sv_wm_effect_heatmap",
    "plot_residual_progression",
    "plot_three_slice_sv_atoi_upset",
    "plot_top_sv_site_patterns",
    "plot_wm_aware_sv_spatial_ratios",
    "plot_wm_aware_sv_volcano",
    "run_core_analysis",
    "run_sv_external_mechanism_overlaps",
    "set_spatial_background",
    "summarize_top_sv_atoi_sites",
    "transfer_obs_metadata",
]

_SPATIAL_IMAGE_CACHE = {}


_EXTERNAL_GENE_FILTER_CACHE = {}

QUALITY_PROFILES = {
    'Q20': dict(mapq=20, baseq=20, end_distance=0, unique=False),
    'Q30_unique': dict(mapq=30, baseq=30, end_distance=0, unique=True),
    'Q30_unique_end5': dict(mapq=30, baseq=30, end_distance=5, unique=True),
}

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


def add_celltype_composition_modules(
    df_deconv: pd.DataFrame,
    module_map: Optional[dict] = None,
    prefix: str = "celltype",
    normalize_modules: bool = False,
):
    """
    Collapse deconvolved cell-type proportions into fixed biological modules.

    The default DLPFC-oriented modules are stable across samples and avoid
    outcome-driven top-celltype selection:
        glial       : Oligos, OPCs, Astro, Micro/Macro
        excitatory  : Ex* excitatory neurons
        inhibitory  : inhibitory neurons
        vascular    : Endo/pericyte/VLMC/vascular-like cells
    """
    if df_deconv is None or df_deconv.empty:
        raise ValueError("df_deconv is empty.")

    if module_map is None:
        module_map = {
            "glial_module": ["oligo", "opc", "astro", "micro", "macro"],
            "excitatory_module": ["^ex", "excit"],
            "inhibitory_module": ["inhib", "gaba"],
            "vascular_module": ["endo", "peri", "vlmc", "vasc"],
        }

    df_out = df_deconv.copy()
    module_cols = []
    rows = []
    lower_cols = {c: str(c).lower() for c in df_out.columns}

    for module_name, patterns in module_map.items():
        members = []

        for col, lower in lower_cols.items():
            for pat in patterns:
                if re.search(pat, lower, flags=re.IGNORECASE):
                    members.append(col)
                    break

        members = list(dict.fromkeys(members))

        if not members:
            continue

        col_name = f"{prefix}_{module_name}"
        df_out[col_name] = df_out[members].sum(axis=1)
        module_cols.append(col_name)
        rows.append({"module": col_name, "members": ", ".join(members), "n_members": len(members)})

    if normalize_modules and module_cols:
        module_sum = df_out[module_cols].sum(axis=1).replace(0, np.nan)
        df_out[module_cols] = df_out[module_cols].div(module_sum, axis=0)

    module_members = pd.DataFrame(rows, columns=["module", "members", "n_members"])

    return df_out, module_cols, module_members


def plot_celltype_module_deconvolution_summary(
    adata_ai,
    df_deconv: pd.DataFrame,
    module_cols: Sequence[str],
    group_key: str = "ground_truth",
    cluster_agg: str = "mean",
    cmap: str = "viridis",
    spot_size: int = 16,
    figsize=None,
):
    """
    Plot integrated deconvolution modules as spatial maps plus cluster means.
    """
    module_cols = [c for c in module_cols if c in df_deconv.columns]

    if not module_cols:
        raise ValueError("No module_cols found in df_deconv.")

    if figsize is None:
        figsize = (4.0 * len(module_cols), 7.2)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, len(module_cols), height_ratios=[1.0, 0.9])

    map_axes = []
    cluster_summary, _ = _aggregate_analysis_frame_by_group(
        adata_ai=adata_ai,
        df_extra=df_deconv,
        columns=module_cols,
        group_key=group_key,
        aggfunc=cluster_agg,
    )

    for i, col in enumerate(module_cols):
        ax = fig.add_subplot(gs[0, i])
        values = pd.Series(np.nan, index=adata_ai.obs_names, dtype=float)
        common = adata_ai.obs_names.intersection(df_deconv.index)
        values.loc[common] = pd.to_numeric(df_deconv.loc[common, col], errors="coerce")
        sc = _spatial_axes(
            ax,
            adata_ai,
            values.values,
            col.replace("celltype_", "").replace("_", " "),
            cmap=cmap,
            spot_size=spot_size,
            clip=(1, 99),
        )
        if sc is not None:
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        map_axes.append(ax)

    bar_ax = fig.add_subplot(gs[1, :])
    plot_df = cluster_summary.reset_index().melt(
        id_vars=group_key,
        value_vars=module_cols,
        var_name="module",
        value_name="proportion",
    )
    sns.barplot(
        data=plot_df,
        x=group_key,
        y="proportion",
        hue="module",
        ax=bar_ax,
    )
    bar_ax.set_xlabel("")
    bar_ax.set_ylabel("cluster mean proportion")
    bar_ax.set_title("Integrated deconvolution modules by cluster")
    bar_ax.tick_params(axis="x", rotation=25)
    bar_ax.legend(frameon=False, fontsize=8, title="")

    plt.tight_layout()

    return fig, {"maps": map_axes, "bar": bar_ax}, cluster_summary


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
    top_n: Optional[int] = None,
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

    # Count-model results retain their own statistics; never relabel as Moran I.
    if "spatial_fdr" in df.columns:
        primary = df.loc[df["fit_status"].eq("ok")].copy()
        if require_sv:
            primary = primary.loc[primary["is_sv_atoi"]].copy()
        source = "called_sv_atoi" if require_sv else "ranked_count_model"
        if primary.empty and fallback_to_ranked:
            primary = df.loc[df["pass_qc"] & df["fit_status"].eq("ok")].copy()
            source = "exploratory_count_model_fallback"
        primary["score_site_source"] = source
        primary = primary.sort_values(["spatial_fdr", "spatial_p", "spatial_effect"],
                                      ascending=[True, True, False])
        if top_n is not None:
            primary = primary.head(top_n)
        if primary.empty:
            raise ValueError("No count-model A-to-I score sites were selected.")
        return primary.reset_index(drop=True)

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
    top_n: Optional[int] = None,
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


def _spatial_axes(
    ax,
    adata,
    values,
    title: str,
    cmap="magma",
    spot_size: int = 18,
    clip=(1, 99),
    center=None,
    vmin=None,
    vmax=None,
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

    if vmin is None or vmax is None:
        auto_vmin, auto_vmax = np.nanpercentile(values[mask], clip)
        vmin = auto_vmin if vmin is None else vmin
        vmax = auto_vmax if vmax is None else vmax

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


def plot_cluster_level_obs_panel(
    adata,
    value_keys: Sequence[str],
    group_key="ground_truth",
    aggfunc="mean",
    spot_size=18,
    cmap="YlGnBu",
    figsize=None,
    title=None,
):
    """Plot several observation values as compact cluster-mean spatial maps."""
    value_keys = [key for key in value_keys if key in adata.obs.columns]
    if not value_keys:
        raise ValueError("No requested value_keys were found in adata.obs.")
    if group_key not in adata.obs.columns:
        raise ValueError(f"{group_key!r} not found in adata.obs")

    labels = _valid_group_series(adata, group_key=group_key)
    if figsize is None:
        figsize = (4.0 * len(value_keys), 4.0)
    fig, axes = plt.subplots(1, len(value_keys), figsize=figsize, squeeze=False)
    axes = axes.ravel()
    summaries = []

    for ax, value_key in zip(axes, value_keys):
        values = pd.to_numeric(adata.obs[value_key], errors="coerce")
        df = pd.DataFrame({"value": values, "group": labels}).dropna()
        grouped = df.groupby("group", observed=True)["value"].agg(aggfunc) if not df.empty else pd.Series(dtype=float)
        cluster_values = pd.Series(np.nan, index=adata.obs_names, dtype=float)
        valid = labels.notna()
        if not grouped.empty:
            cluster_values.loc[valid] = labels.loc[valid].map(grouped).astype(float)
        sc = _spatial_axes(
            ax,
            adata,
            cluster_values.values,
            value_key.replace("expr_", ""),
            cmap=cmap,
            spot_size=spot_size,
        )
        if sc is not None:
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        summary = grouped.rename("mean").reset_index()
        summary["value_key"] = value_key
        summaries.append(summary)

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    summary_df = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    return fig, axes, summary_df


def _safe_moran_geary(values, coords, spatial_k=6, permutations=999, random_state=None):
    mask = np.isfinite(values)

    if mask.sum() <= spatial_k + 2 or np.nanstd(values[mask]) == 0:
        return np.nan, np.nan, np.nan, np.nan

    try:
        from libpysal.weights import KNN
        from esda.geary import Geary
        from esda.moran import Moran

        w = KNN.from_array(coords[mask], k=min(spatial_k, mask.sum() - 1))
        w.transform = "R"

        if random_state is None:
            moran = Moran(values[mask], w, permutations=permutations)
            geary = Geary(values[mask], w, permutations=permutations)
        else:
            rng_state = np.random.get_state()
            try:
                np.random.seed(int(random_state))
                moran = Moran(values[mask], w, permutations=permutations)
                geary = Geary(values[mask], w, permutations=permutations)
            finally:
                np.random.set_state(rng_state)

        return float(moran.I), float(moran.p_sim), float(geary.C), float(geary.p_sim)

    except Exception as exc:
        warnings.warn(f"Moran/Geary failed; returning NaN spatial statistics: {exc}")
        return np.nan, np.nan, np.nan, np.nan


def compute_obs_spatial_autocorrelation(
    adata_ai,
    value_keys: Sequence[str],
    level: str = "cluster",
    group_key: str = "ground_truth",
    cluster_agg: str = "mean",
    spatial_k: int = 2,
    permutations: int = 999,
    random_state: Optional[int] = 0,
    store_key: Optional[str] = None,
):
    """
    Compute Moran's I and Geary's C for obs-level values.

    level="cluster" aggregates each value and spatial coordinate by group_key
    before computing spatial autocorrelation. This is appropriate for cluster-
    level residuals, but small DLPFC layer counts mean p values should be read
    as compact summaries rather than high-powered spatial tests.
    """
    level = str(level).lower()
    cluster_agg = str(cluster_agg).lower()

    if level not in ["spot", "cluster"]:
        raise ValueError("level must be 'spot' or 'cluster'.")

    if cluster_agg not in ["mean", "median"]:
        raise ValueError("cluster_agg must be 'mean' or 'median'.")

    value_keys = [k for k in value_keys if k in adata_ai.obs.columns]

    if not value_keys:
        raise ValueError("No requested value_keys were found in adata_ai.obs.")

    coords = _coords_from_adata(adata_ai)

    if level == "cluster":
        if group_key not in adata_ai.obs.columns:
            raise ValueError(f"{group_key!r} not found in adata_ai.obs")

        labels = _valid_group_series(adata_ai, group_key=group_key)
        df = adata_ai.obs[value_keys].astype(float).copy()
        df[group_key] = labels
        df["_x"] = coords[:, 0]
        df["_y"] = coords[:, 1]
        df = df.dropna(subset=[group_key])

        if cluster_agg == "mean":
            value_df = df.groupby(group_key, observed=True)[value_keys].mean()
        else:
            value_df = df.groupby(group_key, observed=True)[value_keys].median()

        coord_df = df.groupby(group_key, observed=True)[["_x", "_y"]].mean()
        value_df = value_df.loc[value_df.index.intersection(coord_df.index)]
        coord_df = coord_df.loc[value_df.index]

        units = value_df.index.astype(str)
        coords_use = coord_df[["_x", "_y"]].to_numpy(dtype=float)
        n_units_total = int(value_df.shape[0])
    else:
        value_df = adata_ai.obs[value_keys].astype(float).copy()
        units = value_df.index.astype(str)
        coords_use = coords
        n_units_total = int(value_df.shape[0])

    rows = []
    rng = np.random.default_rng(random_state) if random_state is not None else None

    for key in value_keys:
        values = pd.to_numeric(value_df[key], errors="coerce").to_numpy(dtype=float)
        n_valid = int(np.isfinite(values).sum())
        k_use = max(1, min(int(spatial_k), max(n_valid - 1, 1)))

        moran_I, moran_p, geary_C, geary_p = _safe_moran_geary(
            values,
            coords_use,
            spatial_k=k_use,
            permutations=permutations,
            random_state=(
                int(rng.integers(0, np.iinfo(np.int32).max))
                if rng is not None
                else None
            ),
        )

        rows.append(
            {
                "value_key": key,
                "level": level,
                "group_key": group_key if level == "cluster" else np.nan,
                "cluster_agg": cluster_agg if level == "cluster" else np.nan,
                "spatial_k": k_use,
                "permutations": permutations,
                "random_state": random_state,
                "n_units": n_valid,
                "n_units_total": n_units_total,
                "moran_I": moran_I,
                "moran_p": moran_p,
                "geary_C": geary_C,
                "geary_p": geary_p,
            }
        )

    res = pd.DataFrame(rows)
    res["moran_fdr"] = _safe_multipletest(res["moran_p"])
    res["geary_fdr"] = _safe_multipletest(res["geary_p"])

    if store_key is not None:
        adata_ai.uns[store_key] = res

    return res


def summarize_top_sv_atoi_sites(
    sv_atoi: pd.DataFrame,
    top_n: int = 20,
    require_sv: bool = True,
    fdr_cutoff: Optional[float] = None,
    min_moran_I: float = 0.0,
    sort_cols: Sequence[str] = ("moran_fdr", "moran_p", "moran_I"),
    store_in: Optional[object] = None,
    store_key: str = "top_sv_atoi_sites",
):
    """
    Create a compact table of top SV-A-to-I sites for manuscript display.
    """
    if sv_atoi is None or sv_atoi.empty:
        raise ValueError("sv_atoi is empty.")

    df = sv_atoi.copy()

    if "spatial_fdr" in df.columns:
        df = df.loc[df["fit_status"].eq("ok")].copy()
        if require_sv:
            df = df.loc[df["is_sv_atoi"]].copy()
        if fdr_cutoff is not None:
            df = df.loc[df["spatial_fdr"] < fdr_cutoff]
        df = df.sort_values(["spatial_fdr", "spatial_p", "spatial_effect"],
                            ascending=[True, True, False])
        if top_n is not None:
            df = df.head(int(top_n))
        df = df.reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1))
        if store_in is not None:
            store_in.uns[store_key] = df
        return df

    if require_sv and "is_sv_atoi" in df.columns:
        df = df[df["is_sv_atoi"] == True].copy()

    if "moran_I" in df.columns and min_moran_I is not None:
        df = df[pd.to_numeric(df["moran_I"], errors="coerce") > float(min_moran_I)].copy()

    if fdr_cutoff is not None and "moran_fdr" in df.columns:
        df = df[pd.to_numeric(df["moran_fdr"], errors="coerce") < fdr_cutoff].copy()

    if df.empty:
        return pd.DataFrame()

    ascending = []
    valid_sort_cols = []

    for col in sort_cols:
        if col in df.columns:
            valid_sort_cols.append(col)
            ascending.append(False if col in {"moran_I", "sd_ratio", "total_cov_valid"} else True)

    if valid_sort_cols:
        df = df.sort_values(valid_sort_cols, ascending=ascending)

    keep = [
        "site",
        "gene",
        "is_sv_atoi",
        "pass_qc",
        "moran_I",
        "moran_p",
        "moran_fdr",
        "geary_C",
        "geary_p",
        "group_kw_p",
        "group_kw_fdr",
        "n_valid_spots",
        "mean_ratio",
        "sd_ratio",
        "total_cov_valid",
    ]
    keep = [c for c in keep if c in df.columns]
    out = df.loc[:, keep]
    if top_n is not None:
        out = out.head(int(top_n))
    out = out.reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, out.shape[0] + 1))

    if store_in is not None:
        store_in.uns[store_key] = out

    return out


















def _fit_grouped_binomial_glm(g, a, design, term_names):
    """Fit grouped A/G binomial counts and return fit plus spot-level dispersion."""
    g = np.asarray(g, dtype=float)
    a = np.asarray(a, dtype=float)
    design = np.asarray(design, dtype=float)
    if design.ndim != 2 or design.shape[1] != len(term_names):
        raise ValueError("Grouped-binomial design and term_names do not match.")
    if len(g) != len(a) or len(g) != design.shape[0]:
        raise ValueError("Grouped-binomial inputs have incompatible lengths.")
    if np.any(g < 0) or np.any(a < 0) or np.any((g + a) <= 0):
        raise ValueError("Grouped-binomial counts must be nonnegative with positive totals.")
    fit = sm.GLM(
        np.column_stack([g, a]),
        design,
        family=sm.families.Binomial(),
    ).fit(maxiter=200, disp=0)
    if not getattr(fit, "converged", True):
        raise RuntimeError("Grouped-binomial GLM did not converge.")
    pearson = float(np.sum(np.asarray(fit.resid_pearson, dtype=float) ** 2))
    dispersion = max(1.0, pearson / max(int(fit.df_resid), 1))
    return fit, dispersion








def analyze_recurrent_sv_wm_binomial(
    sv_calls: pd.DataFrame,
    adata_by_sample: Optional[dict] = None,
    data_root: Optional[Union[str, Path]] = None,
    sample_ids: Sequence[str] = ("151673", "151671", "151507"),
    sample_col: str = "Slice",
    site_col: str = "site",
    call_col: str = "is_sv_atoi",
    group_key: str = "ground_truth",
    wm_pattern: str = r"(?:^WM$|white)",
    min_cov: int = 5,
    min_spots_per_group: int = 1,
    alpha: float = 0.05,
):
    """Run coherent single-slice and pooled three-slice WM binomial tests.

    Recurrent sites are those called as SV-A-to-I in every requested slice.
    Single-slice models use ``logit(p) = intercept + WM``. The pooled model
    uses ``logit(p) = slice + WM``; a nested slice-specific-WM model tests
    heterogeneity. No x/y terms are included because this analysis targets the
    anatomical WM contrast itself. Ordinary and Pearson-dispersion-adjusted
    likelihood-ratio p values are both returned.
    """
    samples = [str(x) for x in sample_ids]
    required = {sample_col, site_col, call_col}
    missing = required.difference(sv_calls.columns)
    if missing:
        raise ValueError(f"sv_calls is missing required columns: {sorted(missing)}")
    calls = sv_calls.copy()
    calls[sample_col] = calls[sample_col].astype(str)
    calls[site_col] = calls[site_col].astype(str)
    calls = calls.loc[calls[call_col].fillna(False).astype(bool)]
    site_sets = [set(calls.loc[calls[sample_col].eq(s), site_col]) for s in samples]
    recurrent_sites = sorted(set.intersection(*site_sets))
    if not recurrent_sites:
        raise ValueError("No called SV-A-to-I site is shared by all requested slices.")

    if adata_by_sample is None:
        if data_root is None:
            raise ValueError("Provide adata_by_sample or data_root.")
        import anndata as ad
        root = Path(data_root)
        adata_by_sample = {}
        for sample in samples:
            adata = ad.read_h5ad(root / sample / "adata_ai.h5ad")
            label_path = root / sample / f"cluster_labels_{sample}.csv"
            labels = pd.read_csv(label_path)
            labels["barcode"] = labels["key"].astype(str).str.split("_", n=1).str[-1]
            adata.obs[group_key] = labels.set_index("barcode")[group_key].reindex(
                adata.obs_names
            ).to_numpy()
            adata_by_sample[sample] = adata

    per_rows = []
    frames_by_site = {site: [] for site in recurrent_sites}
    for sample in samples:
        adata = adata_by_sample[sample]
        labels = _valid_group_series(adata, group_key=group_key)
        is_wm = labels.astype(str).str.contains(
            wm_pattern, case=False, regex=True, na=False
        ).to_numpy()
        label_valid = labels.notna().to_numpy()
        A, G = _get_count_layers(adata)
        for site in recurrent_sites:
            if site not in adata.var_names:
                continue
            j = adata.var_names.get_loc(site)
            a = _dense_col(A, j).astype(float)
            g = _dense_col(G, j).astype(float)
            valid = label_valid & ((a + g) >= float(min_cov))
            frame = pd.DataFrame({
                "A": a[valid], "G": g[valid], "is_wm": is_wm[valid],
                "sample_id": sample,
            })
            frames_by_site[site].append(frame)
            wm = frame["is_wm"].to_numpy(bool)
            n_wm = int(wm.sum())
            n_nonwm = int((~wm).sum())
            wm_a = float(frame.loc[wm, "A"].sum())
            wm_g = float(frame.loc[wm, "G"].sum())
            nonwm_a = float(frame.loc[~wm, "A"].sum())
            nonwm_g = float(frame.loc[~wm, "G"].sum())
            sufficient = n_wm >= min_spots_per_group and n_nonwm >= min_spots_per_group
            row = {
                "site": site, "sample_id": sample,
                "n_wm_spots": n_wm, "n_nonwm_spots": n_nonwm,
                "wm_A": wm_a, "wm_G": wm_g,
                "nonwm_A": nonwm_a, "nonwm_G": nonwm_g,
                "support_ok": sufficient, "fit_status": "insufficient_support",
                "wm_log_odds": np.nan, "wm_log2fc": np.nan,
                "wm_odds_ratio": np.nan, "binomial_lrt": np.nan,
                "binomial_p": np.nan, "pearson_dispersion": np.nan,
                "dispersion_adjusted_p": np.nan,
            }
            if sufficient:
                try:
                    x0 = np.ones((len(frame), 1), dtype=float)
                    x1 = np.column_stack([x0, frame["is_wm"].to_numpy(float)])
                    fit0, _ = _fit_grouped_binomial_glm(frame["G"], frame["A"], x0, ["const"])
                    fit1, dispersion = _fit_grouped_binomial_glm(
                        frame["G"], frame["A"], x1, ["const", "WM"]
                    )
                    lrt = max(0.0, 2.0 * (float(fit1.llf) - float(fit0.llf)))
                    beta = float(fit1.params[-1])
                    row.update({
                        "fit_status": "ok", "wm_log_odds": beta,
                        "wm_log2fc": beta / np.log(2.0),
                        "wm_odds_ratio": float(np.exp(np.clip(beta, -30, 30))),
                        "binomial_lrt": lrt,
                        "binomial_p": float(chi2.sf(lrt, 1)),
                        "pearson_dispersion": dispersion,
                        "dispersion_adjusted_p": float(chi2.sf(lrt / dispersion, 1)),
                    })
                except Exception:
                    row["fit_status"] = "fit_failed"
            per_rows.append(row)

    per_slice = pd.DataFrame(per_rows)
    for pcol, fdr_col in [
        ("binomial_p", "binomial_fdr_within_slice"),
        ("dispersion_adjusted_p", "dispersion_adjusted_fdr_within_slice"),
    ]:
        per_slice[fdr_col] = np.nan
        for _, index in per_slice.groupby("sample_id").groups.items():
            valid_index = per_slice.loc[index].index[per_slice.loc[index, pcol].notna()]
            if len(valid_index):
                per_slice.loc[valid_index, fdr_col] = multipletests(
                    per_slice.loc[valid_index, pcol].to_numpy(float), method="fdr_bh"
                )[1]
    per_slice["binomial_candidate"] = (
        per_slice["binomial_fdr_within_slice"].lt(alpha)
        & per_slice["wm_log_odds"].gt(0)
    )
    per_slice["dispersion_adjusted_candidate"] = (
        per_slice["dispersion_adjusted_fdr_within_slice"].lt(alpha)
        & per_slice["wm_log_odds"].gt(0)
    )

    pooled_rows = []
    for site in recurrent_sites:
        pieces = frames_by_site.get(site, [])
        if not pieces:
            continue
        frame = pd.concat(pieces, ignore_index=True)
        sample_cat = pd.Categorical(frame["sample_id"], categories=samples, ordered=True)
        dummies = pd.get_dummies(sample_cat, prefix="slice", drop_first=True, dtype=float)
        base = np.column_stack([np.ones(len(frame)), dummies.to_numpy(float)])
        base_names = ["const"] + dummies.columns.tolist()
        wm = frame["is_wm"].to_numpy(float)
        support_samples = []
        directions = []
        for sample in samples:
            sub = frame.loc[frame["sample_id"].eq(sample)]
            has_both = sub["is_wm"].any() and (~sub["is_wm"]).any()
            if has_both:
                support_samples.append(sample)
                wm_g = float(sub.loc[sub["is_wm"], "G"].sum())
                wm_a = float(sub.loc[sub["is_wm"], "A"].sum())
                nw_g = float(sub.loc[~sub["is_wm"], "G"].sum())
                nw_a = float(sub.loc[~sub["is_wm"], "A"].sum())
                odds = ((wm_g + 0.5) / (wm_a + 0.5)) / ((nw_g + 0.5) / (nw_a + 0.5))
                directions.append(odds > 1.0)
        row = {
            "site": site,
            "n_spots": int(len(frame)),
            "n_wm_spots": int(frame["is_wm"].sum()),
            "n_nonwm_spots": int((~frame["is_wm"]).sum()),
            "n_slices_with_both_regions": len(support_samples),
            "all_supported_slices_positive": bool(directions) and all(directions),
            "fit_status": "insufficient_support",
            "wm_log_odds": np.nan, "wm_log2fc": np.nan,
            "wm_odds_ratio": np.nan, "pooled_binomial_lrt": np.nan,
            "pooled_binomial_p": np.nan, "pearson_dispersion": np.nan,
            "pooled_dispersion_adjusted_p": np.nan,
            "heterogeneity_lrt": np.nan, "heterogeneity_df": np.nan,
            "heterogeneity_p": np.nan,
        }
        if len(support_samples) >= 1:
            try:
                fit0, _ = _fit_grouped_binomial_glm(
                    frame["G"], frame["A"], base, base_names
                )
                common_x = np.column_stack([base, wm])
                fit1, dispersion = _fit_grouped_binomial_glm(
                    frame["G"], frame["A"], common_x, base_names + ["WM"]
                )
                lrt = max(0.0, 2.0 * (float(fit1.llf) - float(fit0.llf)))
                beta = float(fit1.params[-1])
                row.update({
                    "fit_status": "ok", "wm_log_odds": beta,
                    "wm_log2fc": beta / np.log(2.0),
                    "wm_odds_ratio": float(np.exp(np.clip(beta, -30, 30))),
                    "pooled_binomial_lrt": lrt,
                    "pooled_binomial_p": float(chi2.sf(lrt, 1)),
                    "pearson_dispersion": dispersion,
                    "pooled_dispersion_adjusted_p": float(chi2.sf(lrt / dispersion, 1)),
                })
                if len(support_samples) >= 2:
                    wm_by_slice = np.column_stack([
                        wm * frame["sample_id"].eq(sample).to_numpy(float)
                        for sample in support_samples
                    ])
                    heter_x = np.column_stack([base, wm_by_slice])
                    heter_names = base_names + [f"WM:{s}" for s in support_samples]
                    fit2, _ = _fit_grouped_binomial_glm(
                        frame["G"], frame["A"], heter_x, heter_names
                    )
                    heter_lrt = max(0.0, 2.0 * (float(fit2.llf) - float(fit1.llf)))
                    heter_df = len(support_samples) - 1
                    row.update({
                        "heterogeneity_lrt": heter_lrt,
                        "heterogeneity_df": heter_df,
                        "heterogeneity_p": float(chi2.sf(heter_lrt, heter_df)),
                    })
            except Exception:
                row["fit_status"] = "fit_failed"
        pooled_rows.append(row)

    pooled = pd.DataFrame(pooled_rows)
    for pcol, fdr_col in [
        ("pooled_binomial_p", "pooled_binomial_fdr"),
        ("pooled_dispersion_adjusted_p", "pooled_dispersion_adjusted_fdr"),
        ("heterogeneity_p", "heterogeneity_fdr"),
    ]:
        pooled[fdr_col] = np.nan
        valid = pooled[pcol].notna()
        if valid.any():
            pooled.loc[valid, fdr_col] = multipletests(
                pooled.loc[valid, pcol].to_numpy(float), method="fdr_bh"
            )[1]
    pooled["pooled_binomial_candidate"] = (
        pooled["pooled_binomial_fdr"].lt(alpha) & pooled["wm_log_odds"].gt(0)
    )
    pooled["pooled_dispersion_adjusted_candidate"] = (
        pooled["pooled_dispersion_adjusted_fdr"].lt(alpha)
        & pooled["wm_log_odds"].gt(0)
    )
    pooled = pooled.sort_values(
        ["pooled_dispersion_adjusted_candidate", "pooled_dispersion_adjusted_fdr",
         "pooled_binomial_fdr"],
        ascending=[False, True, True], na_position="last"
    ).reset_index(drop=True)
    return per_slice, pooled


def plot_wm_aware_sv_volcano(
    wm_results: pd.DataFrame,
    site_col: str = "site",
    effect_col: str = "wm_log2fc",
    fdr_col: str = "binomial_fdr_within_slice",
    candidate_col: Optional[str] = "binomial_candidate",
    alpha: float = 0.05,
    label_top_n: int = 8,
    title: str = "WM-aware SV-A-to-I sites",
    figsize=(6.8, 5.2),
):
    """Plot WM enrichment effect size against multiplicity-adjusted significance."""
    if wm_results is None or wm_results.empty:
        raise ValueError("wm_results is empty.")
    required = {site_col, effect_col, fdr_col}
    missing = required.difference(wm_results.columns)
    if missing:
        raise ValueError(f"wm_results is missing required columns: {sorted(missing)}")

    plot_df = wm_results.copy()
    plot_df[site_col] = plot_df[site_col].astype(str)
    plot_df[effect_col] = pd.to_numeric(plot_df[effect_col], errors="coerce")
    plot_df[fdr_col] = pd.to_numeric(plot_df[fdr_col], errors="coerce")
    plot_df = plot_df.loc[
        plot_df[effect_col].notna() & plot_df[fdr_col].notna()
    ].copy()
    if plot_df.empty:
        raise ValueError("No finite WM effects and FDR values are available to plot.")

    if candidate_col is not None and candidate_col in plot_df.columns:
        positive = plot_df[candidate_col].fillna(False).astype(bool)
    else:
        positive = plot_df[fdr_col].lt(alpha) & plot_df[effect_col].gt(0)
    negative = plot_df[fdr_col].lt(alpha) & plot_df[effect_col].lt(0)
    plot_df["wm_positive_significant"] = positive.to_numpy(bool)
    plot_df["nonwm_positive_significant"] = negative.to_numpy(bool)
    plot_df["minus_log10_fdr"] = -np.log10(
        plot_df[fdr_col].clip(lower=np.finfo(float).tiny)
    )

    fig, ax = plt.subplots(figsize=figsize)
    background = ~(positive | negative)
    ax.scatter(
        plot_df.loc[background, effect_col],
        plot_df.loc[background, "minus_log10_fdr"],
        s=22,
        color="#bdbdbd",
        alpha=0.65,
        linewidths=0,
        label="Not significant",
    )
    if negative.any():
        ax.scatter(
            plot_df.loc[negative, effect_col],
            plot_df.loc[negative, "minus_log10_fdr"],
            s=30,
            color="#4e79a7",
            alpha=0.88,
            linewidths=0,
            label="non-WM > WM, FDR < 0.05",
        )
    if positive.any():
        ax.scatter(
            plot_df.loc[positive, effect_col],
            plot_df.loc[positive, "minus_log10_fdr"],
            s=34,
            color="#d95f02",
            alpha=0.92,
            linewidths=0,
            label="WM > non-WM, FDR < 0.05",
        )

    ax.axvline(0, color="#555555", lw=0.9, ls="--")
    ax.axhline(-np.log10(alpha), color="#555555", lw=0.9, ls=":")
    label_df = plot_df.loc[positive].sort_values(
        [fdr_col, effect_col], ascending=[True, False]
    ).head(int(label_top_n))
    for _, row in label_df.iterrows():
        ax.annotate(
            row[site_col],
            (row[effect_col], row["minus_log10_fdr"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("WM effect (log2 odds ratio)")
    ax.set_ylabel(f"-log10({fdr_col})")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    return fig, ax, plot_df


def plot_wm_aware_sv_spatial_ratios(
    adata_ai,
    wm_results: pd.DataFrame,
    group_key: str = "ground_truth",
    site_col: str = "site",
    effect_col: str = "wm_log2fc",
    fdr_col: str = "binomial_fdr_within_slice",
    candidate_col: Optional[str] = "binomial_candidate",
    alpha: float = 0.05,
    top_n: int = 3,
    min_cov: int = 5,
    mode: str = "cluster",
    cmap: str = "magma",
    spot_size: int = 18,
    title: str = "Top WM-enriched SV-A-to-I spatial ratios",
    figsize=None,
):
    """Plot the top WM-enriched single-site editing ratios with one colorbar per site."""
    if wm_results is None or wm_results.empty:
        raise ValueError("wm_results is empty.")
    required = {site_col, effect_col, fdr_col}
    missing = required.difference(wm_results.columns)
    if missing:
        raise ValueError(f"wm_results is missing required columns: {sorted(missing)}")
    mode = str(mode).lower()
    if mode not in {"spot", "cluster"}:
        raise ValueError("mode must be 'spot' or 'cluster'.")

    selected = wm_results.copy()
    selected[site_col] = selected[site_col].astype(str)
    selected[effect_col] = pd.to_numeric(selected[effect_col], errors="coerce")
    selected[fdr_col] = pd.to_numeric(selected[fdr_col], errors="coerce")
    if candidate_col is not None and candidate_col in selected.columns:
        mask = selected[candidate_col].fillna(False).astype(bool)
    else:
        mask = selected[fdr_col].lt(alpha) & selected[effect_col].gt(0)
    selected = selected.loc[mask].sort_values(
        [fdr_col, effect_col], ascending=[True, False]
    )
    selected = selected.loc[selected[site_col].isin(adata_ai.var_names)].head(int(top_n)).copy()
    if selected.empty:
        raise ValueError("No significant WM-enriched SV-A-to-I site is available to plot.")

    sites = selected[site_col].tolist()
    if figsize is None:
        figsize = (4.25 * len(sites), 4.2)
    fig, axes = plt.subplots(1, len(sites), figsize=figsize, squeeze=False)
    axes = axes.ravel()
    labels = _valid_group_series(adata_ai, group_key=group_key) if mode == "cluster" else None

    for ax, (_, row) in zip(axes, selected.iterrows()):
        site = row[site_col]
        ratio = get_site_editing_ratio(adata_ai, site, min_cov=min_cov)
        values = _map_group_mean_to_spots(ratio, labels) if mode == "cluster" else ratio.to_numpy()
        gene = site
        if site in adata_ai.var.index and "Gene.refGene" in adata_ai.var.columns:
            gene = str(adata_ai.var.loc[site].get("Gene.refGene", site))
        panel_title = (
            f"{gene}\n{site}\n"
            f"WM log2OR={row[effect_col]:.2f}; FDR={row[fdr_col]:.2g}"
        )
        scatter = _spatial_axes(
            ax, adata_ai, values, panel_title,
            cmap=cmap, spot_size=spot_size,
        )
        if scatter is not None:
            colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02)
            colorbar.set_label("Editing ratio", fontsize=8)

    fig.suptitle(f"{title} ({mode} level)", y=1.03, fontsize=13)
    fig.tight_layout()
    return fig, axes, selected.reset_index(drop=True)


def _load_cross_sample_spatial_adata(
    data_root,
    sample_id,
    group_key="ground_truth",
    swap_x_y=True,
):
    """Load one DLPFC A-to-I matrix with labels, full-resolution coordinates and H&E metadata."""
    import anndata as ad
    import json

    sample_id = str(sample_id)
    base = Path(data_root) / sample_id
    adata = ad.read_h5ad(base / "adata_ai.h5ad")
    positions = pd.read_csv(
        base / "spatial" / "tissue_positions_list.csv",
        header=None,
        names=["barcode", "in_tissue", "x_array", "y_array", "x_pixel", "y_pixel"],
    ).set_index("barcode")
    aligned = positions.reindex(adata.obs_names)
    for column in positions.columns:
        adata.obs[column] = aligned[column].to_numpy()
    adata.obsm["spatial"] = aligned[["x_pixel", "y_pixel"]].to_numpy(dtype=float)

    labels = pd.read_csv(base / f"cluster_labels_{sample_id}.csv")
    if "key" not in labels.columns or group_key not in labels.columns:
        raise ValueError(f"Missing key/{group_key} in cluster labels for {sample_id}.")
    labels["barcode"] = labels["key"].astype(str).str.split("_", n=1).str[-1]
    adata.obs[group_key] = labels.set_index("barcode")[group_key].reindex(
        adata.obs_names
    ).to_numpy()
    adata.obs[group_key] = adata.obs[group_key].replace(
        ["nan", "None", "NA", "null", ""], np.nan
    )

    scale_path = base / "spatial" / "scalefactors_json.json"
    if scale_path.exists():
        with scale_path.open("r", encoding="utf-8") as handle:
            coordinate_scale = float(json.load(handle).get("tissue_hires_scalef", 1.0))
    else:
        coordinate_scale = 1.0
    return set_spatial_background(
        adata,
        image_path=str(base / "spatial" / "tissue_hires_image.png"),
        swap_x_y=bool(swap_x_y),
        coordinate_scale=coordinate_scale,
        crop=True,
        crop_pad=120,
        image_alpha=0.78,
    )


def plot_cross_sample_site_spatial_ratios(
    site: str,
    data_root,
    sample_ids=("151673", "151671", "151507"),
    group_key: str = "ground_truth",
    min_cov: int = 5,
    mode: str = "cluster",
    swap_x_y_by_sample=None,
    cmap: str = "magma",
    spot_size: int = 18,
    title: Optional[str] = None,
    figsize=(12.8, 4.3),
):
    """Plot one site across slices, with independent normalization and colorbar per slice."""
    mode = str(mode).lower()
    if mode not in {"spot", "cluster"}:
        raise ValueError("mode must be 'spot' or 'cluster'.")
    samples = [str(sample) for sample in sample_ids]
    if swap_x_y_by_sample is None:
        swap_x_y_by_sample = {sample: True for sample in samples}

    fig, axes = plt.subplots(1, len(samples), figsize=figsize, squeeze=False)
    axes = axes.ravel()
    summary_rows = []
    for ax, sample in zip(axes, samples):
        swap = (
            swap_x_y_by_sample.get(sample, False)
            if isinstance(swap_x_y_by_sample, dict)
            else bool(swap_x_y_by_sample)
        )
        adata = _load_cross_sample_spatial_adata(
            data_root, sample, group_key=group_key, swap_x_y=swap
        )
        if site not in adata.var_names:
            ax.text(0.5, 0.5, f"{sample}\n{site} absent", ha="center", va="center")
            ax.axis("off")
            summary_rows.append({"sample_id": sample, "site": site, "status": "site_absent"})
            continue

        ratio = get_site_editing_ratio(adata, site, min_cov=min_cov)
        if mode == "cluster":
            labels = _valid_group_series(adata, group_key=group_key)
            values = _map_group_mean_to_spots(ratio, labels)
            frame = pd.DataFrame({"group": labels, "editing_ratio": ratio}).dropna()
            grouped = frame.groupby("group", observed=True)["editing_ratio"].agg(["mean", "size"])
            for group, row in grouped.iterrows():
                summary_rows.append({
                    "sample_id": sample, "site": site, "group": group,
                    "editing_ratio": float(row["mean"]), "n_spots": int(row["size"]),
                    "status": "ok",
                })
        else:
            values = ratio.to_numpy()
            summary_rows.append({
                "sample_id": sample, "site": site,
                "editing_ratio": float(ratio.mean()),
                "n_spots": int(ratio.notna().sum()), "status": "ok",
            })

        scatter = _spatial_axes(
            ax, adata, values, f"{sample}\n{site}",
            cmap=cmap, spot_size=spot_size,
        )
        if scatter is not None:
            colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02)
            colorbar.set_label("Editing ratio", fontsize=8)

    fig.suptitle(
        title or f"{site}: spatial editing ratio across three slices ({mode} level)",
        y=1.03,
        fontsize=13,
    )
    fig.tight_layout()
    return fig, axes, pd.DataFrame(summary_rows)


def plot_recurrent_sv_wm_effect_heatmap(
    per_slice_results: pd.DataFrame,
    pooled_results: Optional[pd.DataFrame] = None,
    sample_order=("151673", "151671", "151507"),
    site_col: str = "site",
    sample_col: str = "sample_id",
    effect_col: str = "wm_log2fc",
    fdr_col: str = "binomial_fdr_within_slice",
    pooled_fdr_col: str = "pooled_binomial_fdr",
    sites: Optional[Sequence[str]] = None,
    top_n: int = 30,
    alpha: float = 0.05,
    title: str = "Recurrent SV-A-to-I WM effects across slices",
    figsize=None,
):
    """Heatmap of slice-specific WM log2 odds ratios; stars mark within-slice FDR < alpha."""
    if per_slice_results is None or per_slice_results.empty:
        raise ValueError("per_slice_results is empty.")
    required = {site_col, sample_col, effect_col, fdr_col}
    missing = required.difference(per_slice_results.columns)
    if missing:
        raise ValueError(f"per_slice_results is missing required columns: {sorted(missing)}")
    samples = [str(sample) for sample in sample_order]

    if sites is None:
        if pooled_results is not None and not pooled_results.empty:
            ranking = pooled_results.copy()
            ranking[site_col] = ranking[site_col].astype(str)
            ranking[pooled_fdr_col] = pd.to_numeric(
                ranking[pooled_fdr_col], errors="coerce"
            )
            sites = ranking.sort_values(
                pooled_fdr_col, na_position="last"
            )[site_col].head(int(top_n)).tolist()
        else:
            sites = per_slice_results[site_col].astype(str).drop_duplicates().head(int(top_n)).tolist()
    sites = [str(site) for site in sites]
    if not sites:
        raise ValueError("No recurrent site is available for the heatmap.")

    frame = per_slice_results.copy()
    frame[site_col] = frame[site_col].astype(str)
    frame[sample_col] = frame[sample_col].astype(str)
    frame = frame.loc[frame[site_col].isin(sites)]
    effects = frame.pivot_table(
        index=site_col, columns=sample_col, values=effect_col, aggfunc="first"
    ).reindex(index=sites, columns=samples)
    fdr = frame.pivot_table(
        index=site_col, columns=sample_col, values=fdr_col, aggfunc="first"
    ).reindex(index=sites, columns=samples)
    stars = fdr.applymap(lambda value: "*" if pd.notna(value) and value < alpha else "")

    finite = np.abs(effects.to_numpy(dtype=float))
    finite = finite[np.isfinite(finite)]
    limit = float(np.quantile(finite, 0.95)) if finite.size else 1.0
    limit = max(limit, 1.0)
    if figsize is None:
        figsize = (5.8, max(4.2, 0.28 * len(sites) + 1.8))
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#eeeeee")
    sns.heatmap(
        effects,
        mask=effects.isna(),
        cmap="RdBu_r",
        center=0,
        vmin=-limit,
        vmax=limit,
        annot=stars,
        fmt="",
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "WM effect (log2 odds ratio)"},
        ax=ax,
    )
    ax.set_xlabel("Slice")
    ax.set_ylabel("Recurrent SV-A-to-I site")
    ax.set_title(title)
    ax.text(
        0,
        -0.07,
        f"* within-slice FDR < {alpha:g}; grey cells lack WM/non-WM support",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    fig.tight_layout()
    return fig, ax, effects, fdr




def annotate_sv_atoi_sites(
    adata_ai,
    sv_atoi: pd.DataFrame,
    top_n: Optional[int] = None,
    require_sv: bool = True,
    store_key: str = "sv_atoi_site_annotation",
):
    """
    Add available var-level annotations to SV-A-to-I site table.

    This uses annotations already present in adata_ai.var, for example
    ANNOVAR-style columns such as Gene.refGene, Func.refGene, ExonicFunc.refGene,
    AAChange.refGene, cytoBand, or database/repeat columns if available.
    """
    if sv_atoi is None or sv_atoi.empty:
        raise ValueError("sv_atoi is empty.")

    df = sv_atoi.copy()

    if require_sv and "is_sv_atoi" in df.columns:
        df = df[df["is_sv_atoi"] == True].copy()

    if df.empty:
        out = pd.DataFrame()
        adata_ai.uns[store_key] = out
        return out

    sort_cols = [c for c in ["spatial_fdr", "spatial_p", "spatial_effect", "moran_fdr", "moran_p", "moran_I"] if c in df.columns]
    if sort_cols:
        ascending = [False if c in {"moran_I", "spatial_effect"} else True for c in sort_cols]
        df = df.sort_values(sort_cols, ascending=ascending)

    if top_n is not None:
        df = df.head(top_n).copy()

    var = adata_ai.var.copy()
    site_index = var.index.astype(str)
    var = var.loc[site_index.isin(df["site"].astype(str))]
    var = var.reset_index()
    first_col = var.columns[0]
    if first_col != "site":
        var = var.rename(columns={first_col: "site"})

    preferred = [
        "site",
        "Gene.refGene",
        "Func.refGene",
        "ExonicFunc.refGene",
        "AAChange.refGene",
        "cytoBand",
        "avsnp150",
        "snp138",
        "REVEL_score",
        "CADD_phred",
        "Repeat",
        "repeat",
        "repName",
        "repClass",
        "repFamily",
        "REDIportal",
    ]
    annot_cols = [c for c in preferred if c in var.columns]

    if "site" not in annot_cols:
        annot_cols = ["site"] + annot_cols

    merged = df.merge(var[annot_cols].drop_duplicates("site"), on="site", how="left")

    if "gene" in merged.columns:
        merged["host_gene"] = merged["gene"].replace("", np.nan)
    elif "Gene.refGene" in merged.columns:
        merged["host_gene"] = merged["Gene.refGene"].replace("", np.nan)

    if "host_gene" in merged.columns and "Gene.refGene" in merged.columns:
        merged["host_gene"] = merged["host_gene"].fillna(merged["Gene.refGene"])

    adata_ai.uns[store_key] = merged.reset_index(drop=True)

    return adata_ai.uns[store_key]


def _site_coordinate_table(sv_annotation: pd.DataFrame, site_col: str = "site"):
    if sv_annotation is None or sv_annotation.empty or site_col not in sv_annotation.columns:
        return pd.DataFrame()

    df = sv_annotation.copy()
    coords = df[site_col].astype(str).str.extract(r"^(?P<chrom>[^_]+)_(?P<pos>\d+)$")
    df["chrom"] = coords["chrom"].astype(str).str.replace("^chr", "", regex=True)
    df["chrom_norm"] = df["chrom"].str.replace("^chr", "", regex=True)
    df["pos_1based"] = pd.to_numeric(coords["pos"], errors="coerce")
    df["bed_start"] = df["pos_1based"] - 1
    df["bed_end"] = df["pos_1based"]
    return df


def _parse_gff3_attributes(attr):
    if pd.isna(attr):
        return {}
    out = {}
    for item in str(attr).split(";"):
        if "=" in item:
            key, value = item.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def _read_external_annotation_file(path, annotation_type: str = "bed"):
    if path is None or str(path).strip() == "":
        return pd.DataFrame(), "missing_path"

    path = Path(path)
    if not path.exists():
        return pd.DataFrame(), "missing_file"
    if path.is_dir():
        return pd.DataFrame(), "directory_path"

    annotation_type = str(annotation_type).lower()

    if annotation_type in {"gff", "gff3"}:
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            comment="#",
            dtype=str,
            names=["chrom", "source", "feature_type", "gff_start", "gff_end", "score", "strand", "phase", "attributes"],
            compression="infer",
        )
        df = df.dropna(subset=["chrom", "gff_start", "gff_end"]).copy()
        attrs = df["attributes"].map(_parse_gff3_attributes)
        df["start"] = pd.to_numeric(df["gff_start"], errors="coerce") - 1
        df["end"] = pd.to_numeric(df["gff_end"], errors="coerce")
        df["name"] = attrs.map(lambda x: x.get("Name") or x.get("ID") or x.get("gene") or x.get("Alias") or "")
        df["gene"] = attrs.map(lambda x: x.get("gene") or x.get("Name") or x.get("ID") or "")
        return df[["chrom", "start", "end", "name", "score", "strand", "feature_type", "gene"]], "loaded"

    if annotation_type in {"repeatmasker", "rmsk"}:
        raw = pd.read_csv(path, sep="\t", header=None, comment="#", dtype=str, compression="infer")
        if raw.shape[1] >= 13:
            df = pd.DataFrame(
                {
                    "chrom": raw.iloc[:, 5],
                    "start": raw.iloc[:, 6],
                    "end": raw.iloc[:, 7],
                    "name": raw.iloc[:, 10],
                    "score": raw.iloc[:, 1],
                    "strand": raw.iloc[:, 9],
                    "repName": raw.iloc[:, 10],
                    "repClass": raw.iloc[:, 11],
                    "repFamily": raw.iloc[:, 12],
                }
            )
            return df, "loaded"
        return pd.DataFrame(), "unsupported_repeatmasker_format"

    if annotation_type in {"rediportal", "rediting", "radar"}:
        df0 = pd.read_csv(path, sep="\t", dtype=str, compression="infer")
        if {"Region", "Position"}.issubset(df0.columns):
            df = pd.DataFrame()
            df["chrom"] = df0["Region"].astype(str)
            df["start"] = pd.to_numeric(df0["Position"], errors="coerce") - 1
            df["end"] = pd.to_numeric(df0["Position"], errors="coerce")
            df["name"] = df0["Accession"] if "Accession" in df0.columns else "REDIportal"
            df["score"] = "."
            df["strand"] = df0["Strand"] if "Strand" in df0.columns else "."
            for col in ["repeat", "db", "type", "Gene.refGene", "Func.refGene", "Gene.wgEncodeGencodeBasicV45", "Func.wgEncodeGencodeBasicV45"]:
                if col in df0.columns:
                    df[col] = df0[col]
            return df, "loaded"
        return df0, "loaded"

    if annotation_type in {"bed", "interval"}:
        df = pd.read_csv(path, sep="\t", header=None, comment="#", dtype=str, compression="infer")
        base_cols = ["chrom", "start", "end", "name", "score", "strand"]
        cols = base_cols[: min(len(base_cols), df.shape[1])] + [f"extra_{i}" for i in range(max(0, df.shape[1] - len(base_cols)))]
        df.columns = cols
        return df, "loaded"

    df = pd.read_csv(path, sep=None, engine="python", comment="#", dtype=str, compression="infer")
    return df, "loaded"


def _site_positions_by_chrom(sv_annotation: pd.DataFrame, site_col: str = "site"):
    sites = _site_coordinate_table(sv_annotation, site_col=site_col)
    out = {}
    if sites.empty:
        return out
    for _, row in sites.dropna(subset=["pos_1based"]).iterrows():
        out.setdefault(str(row["chrom_norm"]), set()).add(int(row["pos_1based"]))
    return out


def _read_repeatmasker_filtered(path, sv_annotation: pd.DataFrame, site_col="site", flank=0):
    positions = _site_positions_by_chrom(sv_annotation, site_col=site_col)
    rows = []
    try:
        handle = _open_text_maybe_gzip(path)
    except OSError:
        return pd.DataFrame(), "read_error"
    with handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 13:
                continue
            chrom = parts[5].replace("chr", "", 1)
            if chrom not in positions:
                continue
            try:
                start, end = int(parts[6]), int(parts[7])
            except ValueError:
                continue
            if not any(start < pos + flank and end > pos - 1 - flank for pos in positions[chrom]):
                continue
            rows.append(
                {
                    "chrom": parts[5], "start": start, "end": end,
                    "name": parts[10], "score": parts[1], "strand": parts[9],
                    "repName": parts[10], "repClass": parts[11], "repFamily": parts[12],
                }
            )
    return pd.DataFrame(rows), "loaded"


def _read_rediportal_filtered(path, sv_annotation: pd.DataFrame, site_col="site"):
    positions = _site_positions_by_chrom(sv_annotation, site_col=site_col)
    if not positions:
        return pd.DataFrame(), "loaded"
    wanted_cols = {
        "Accession", "Region", "Position", "Strand", "repeat", "db", "type",
        "Gene.refGene", "Func.refGene", "Gene.wgEncodeGencodeBasicV45",
        "Func.wgEncodeGencodeBasicV45",
    }
    matched_rows = []
    with open(path, "rb") as handle:
        mm = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            header_end = mm.find(b"\n") + 1
            columns = mm[: header_end - 1].decode("utf-8", "ignore").rstrip("\r").split("\t")
            region_idx = columns.index("Region")
            position_idx = columns.index("Position")

            def parse_line(start, end):
                fields = mm[start:end].decode("utf-8", "ignore").rstrip("\r").split("\t")
                return (fields[region_idx], int(fields[position_idx])), fields

            for chrom_norm, wanted in positions.items():
                chrom = str(chrom_norm)
                if not chrom.startswith("chr"):
                    chrom = "chr" + chrom
                for pos in sorted(wanted):
                    target = (chrom, int(pos))
                    low, high = header_end, len(mm)
                    while low < high:
                        mid = (low + high) // 2
                        start = mm.rfind(b"\n", header_end, mid) + 1
                        if start < header_end:
                            start = header_end
                        end = mm.find(b"\n", start)
                        if end < 0:
                            end = len(mm)
                        key, _ = parse_line(start, end)
                        if key < target:
                            low = end + 1
                        else:
                            if start == high:
                                break
                            high = start

                    start = mm.rfind(b"\n", header_end, max(header_end, low)) + 1
                    if start < header_end:
                        start = header_end
                    while start < len(mm):
                        end = mm.find(b"\n", start)
                        if end < 0:
                            end = len(mm)
                        key, fields = parse_line(start, end)
                        if key == target:
                            matched_rows.append(dict(zip(columns, fields)))
                        elif key > target:
                            break
                        start = end + 1
        finally:
            mm.close()

    if not matched_rows:
        return pd.DataFrame(columns=["chrom", "start", "end", "name", "score", "strand"]), "loaded"
    df0 = pd.DataFrame(matched_rows)
    df = pd.DataFrame()
    df["chrom"] = df0["Region"]
    df["start"] = pd.to_numeric(df0["Position"], errors="coerce") - 1
    df["end"] = pd.to_numeric(df0["Position"], errors="coerce")
    df["name"] = df0["Accession"] if "Accession" in df0.columns else "REDIportal"
    df["score"] = "."
    df["strand"] = df0["Strand"] if "Strand" in df0.columns else "."
    for col in wanted_cols:
        if col in df0.columns and col not in {"Accession", "Region", "Position", "Strand"}:
            df[col] = df0[col]
    return df, "loaded"


def _standardize_interval_annotation(
    annot: pd.DataFrame,
    source: str,
    chrom_col: str = "chrom",
    start_col: str = "start",
    end_col: str = "end",
    name_col: Optional[str] = "name",
):
    if annot is None or annot.empty:
        return pd.DataFrame()

    df = annot.copy()
    missing = [c for c in [chrom_col, start_col, end_col] if c not in df.columns]
    if missing:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["source"] = source
    out["annot_chrom"] = df[chrom_col].astype(str)
    out["chrom_norm"] = out["annot_chrom"].str.replace("^chr", "", regex=True)
    out["annot_start"] = pd.to_numeric(df[start_col], errors="coerce")
    out["annot_end"] = pd.to_numeric(df[end_col], errors="coerce")

    if name_col is not None and name_col in df.columns:
        out["annot_name"] = df[name_col].astype(str)
    else:
        out["annot_name"] = source

    for col in [
        "strand",
        "feature_type",
        "repName",
        "repClass",
        "repFamily",
        "repeat",
        "db",
        "type",
        "gene",
        "Gene",
        "target_gene",
        "RBP",
        "rbp",
        "mirna",
        "miRNA",
        "Gene.refGene",
        "Func.refGene",
        "Gene.wgEncodeGencodeBasicV45",
        "Func.wgEncodeGencodeBasicV45",
    ]:
        if col in df.columns:
            out[f"annot_{col}"] = df[col].astype(str)

    extra_cols = [c for c in df.columns if c.startswith("extra_")]
    for col in extra_cols[:4]:
        out[f"annot_{col}"] = df[col].astype(str)

    out = out.dropna(subset=["annot_start", "annot_end"])
    return out


def overlap_sv_sites_with_interval_annotation(
    sv_annotation: pd.DataFrame,
    annot: pd.DataFrame,
    source: str,
    site_col: str = "site",
    flank: int = 0,
):
    """
    Overlap SV-A-to-I sites with interval annotations such as miRBase BED,
    miRGeneDB BED/GFF converted to BED, RepeatMasker, REDIportal, or RBP
    binding BED files.
    """
    sites = _site_coordinate_table(sv_annotation, site_col=site_col)
    annot_std = _standardize_interval_annotation(annot, source=source)

    if sites.empty:
        return pd.DataFrame()

    base_cols = [c for c in [site_col, "host_gene", "gene", "Gene.refGene", "Func.refGene", "ExonicFunc.refGene", "sample_id"] if c in sites.columns]
    rows = []

    for _, site_row in sites.iterrows():
        row_base = {c: site_row.get(c, np.nan) for c in base_cols}
        row_base.update(
            {
                "source": source,
                "query_chrom": site_row.get("chrom_norm", np.nan),
                "query_pos_1based": site_row.get("pos_1based", np.nan),
                "overlap": False,
                "n_overlaps": 0,
                "overlap_names": "",
            }
        )

        if annot_std.empty or pd.isna(site_row.get("pos_1based", np.nan)):
            rows.append(row_base)
            continue

        pos0 = float(site_row["bed_start"])
        pos1 = float(site_row["bed_end"])
        chrom = str(site_row["chrom_norm"])
        sub = annot_std[
            (annot_std["chrom_norm"].astype(str) == chrom)
            & (annot_std["annot_start"] < pos1 + flank)
            & (annot_std["annot_end"] > pos0 - flank)
        ].copy()

        if sub.empty:
            rows.append(row_base)
            continue

        names = sorted(sub["annot_name"].dropna().astype(str).unique().tolist())
        row_base["overlap"] = True
        row_base["n_overlaps"] = int(sub.shape[0])
        row_base["overlap_names"] = " | ".join(names[:20])

        for col in sub.columns:
            if col.startswith("annot_") and col not in {"annot_chrom", "annot_start", "annot_end"}:
                vals = sorted(sub[col].dropna().astype(str).unique().tolist())
                row_base[col] = ";".join(vals[:20])

        rows.append(row_base)

    return pd.DataFrame(rows)


def _open_text_maybe_gzip(path):
    path = Path(path)
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "rt", encoding="utf-8", errors="ignore")


def _bed_file_first_name_matches(path, pattern: str) -> bool:
    try:
        handle = _open_text_maybe_gzip(path)
    except OSError:
        return False
    with handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            name = parts[3] if len(parts) > 3 else ""
            return re.search(pattern, str(name), flags=re.IGNORECASE) is not None
    return False


def overlap_sv_sites_with_bed_file_collection(
    sv_annotation: pd.DataFrame,
    paths: Sequence[Union[str, Path]],
    source: str,
    site_col: str = "site",
    flank: int = 0,
    name_regex: Optional[str] = None,
    label_parser: Optional[str] = None,
    deduplicate_labels_per_site: bool = False,
):
    """
    Stream a collection of BED/narrowPeak files and collect overlaps for a small
    SV-A-to-I site set. This is mainly for ENCODE eCLIP directories, where
    loading hundreds of compressed peak files into one table is unnecessarily
    memory-heavy.
    """
    sites = _site_coordinate_table(sv_annotation, site_col=site_col)
    if sites.empty:
        return pd.DataFrame()

    base_cols = [c for c in [site_col, "host_gene", "gene", "Gene.refGene", "Func.refGene", "ExonicFunc.refGene", "sample_id"] if c in sites.columns]
    site_records = {}
    sites_by_chrom = {}
    for idx, site_row in sites.iterrows():
        key = site_row.get(site_col, idx)
        base = {c: site_row.get(c, np.nan) for c in base_cols}
        base.update(
            {
                "source": source,
                "query_chrom": site_row.get("chrom_norm", np.nan),
                "query_pos_1based": site_row.get("pos_1based", np.nan),
                "overlap": False,
                "n_overlaps": 0,
                "overlap_names": "",
            }
        )
        site_records[key] = {"base": base, "names": set(), "strands": set(), "extra_0": set(), "raw_overlaps": 0}
        chrom = str(site_row.get("chrom_norm", ""))
        if chrom and pd.notna(site_row.get("bed_start", np.nan)):
            sites_by_chrom.setdefault(chrom, []).append(
                (key, float(site_row["bed_start"]), float(site_row["bed_end"]))
            )

    for path in paths:
        file_label = Path(path).name
        try:
            handle = _open_text_maybe_gzip(path)
        except OSError:
            continue
        with handle:
            for line in handle:
                if not line.strip() or line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                chrom = parts[0].replace("chr", "", 1)
                if chrom not in sites_by_chrom:
                    continue
                try:
                    start = float(parts[1])
                    end = float(parts[2])
                except ValueError:
                    continue
                name = parts[3] if len(parts) > 3 and parts[3] else file_label
                if name_regex and re.search(name_regex, str(name), flags=re.IGNORECASE) is None:
                    continue
                if label_parser == "rbp":
                    name = re.split(r"_(?:K562|HepG2|IDR|rep\d+)", str(name), maxsplit=1, flags=re.IGNORECASE)[0]
                strand = parts[5] if len(parts) > 5 else "."
                for key, pos0, pos1 in sites_by_chrom[chrom]:
                    if start < pos1 + flank and end > pos0 - flank:
                        rec = site_records[key]
                        rec["base"]["overlap"] = True
                        rec["raw_overlaps"] += 1
                        rec["names"].add(str(name))
                        rec["strands"].add(str(strand))
                        rec["extra_0"].add(file_label)

    rows = []
    for rec in site_records.values():
        out = dict(rec["base"])
        out["n_raw_overlaps"] = int(rec["raw_overlaps"])
        out["n_overlaps"] = len(rec["names"]) if deduplicate_labels_per_site else int(rec["raw_overlaps"])
        out["overlap_names"] = " | ".join(sorted(rec["names"])[:20])
        if rec["strands"]:
            out["annot_strand"] = ";".join(sorted(rec["strands"])[:20])
        if rec["extra_0"]:
            out["annot_extra_0"] = ";".join(sorted(rec["extra_0"])[:20])
        rows.append(out)

    return pd.DataFrame(rows)


def overlap_sv_sites_with_gene_annotation(
    sv_annotation: pd.DataFrame,
    annot: pd.DataFrame,
    source: str,
    gene_cols: Sequence[str] = ("host_gene", "gene", "Gene.refGene"),
    annot_gene_col: Optional[str] = None,
    name_col: Optional[str] = None,
    site_col: str = "site",
):
    """
    Gene-level overlap for resources without site coordinates, such as
    TargetScan target tables or RBP-target gene tables.
    """
    if sv_annotation is None or sv_annotation.empty:
        return pd.DataFrame()

    sites = sv_annotation.copy()
    rows = []

    if annot is None or annot.empty:
        annot = pd.DataFrame()

    if annot_gene_col is None and not annot.empty:
        candidates = ["target_gene", "Target Gene", "Gene Symbol", "gene", "Gene", "symbol", "transcript_gene"]
        annot_gene_col = next((c for c in candidates if c in annot.columns), None)

    if name_col is None and not annot.empty:
        candidates = ["miRNA", "miRNA family", "mirna", "RBP", "rbp", "name", "Name", "database_id"]
        name_col = next((c for c in candidates if c in annot.columns), annot_gene_col)

    annot_genes = pd.Series(dtype=str)
    if annot_gene_col is not None and annot_gene_col in annot.columns:
        annot_genes = annot[annot_gene_col].dropna().astype(str).str.upper()

    for _, row in sites.iterrows():
        site_genes = []
        for col in gene_cols:
            if col in sites.columns and pd.notna(row.get(col, np.nan)):
                site_genes.extend(re.split(r"[;,|]", str(row[col])))
        site_genes = sorted({g.strip().upper() for g in site_genes if g.strip() and g.strip() != "."})

        out = {
            site_col: row.get(site_col, np.nan),
            "source": source,
            "overlap": False,
            "n_overlaps": 0,
            "overlap_names": "",
            "matched_gene": "",
        }
        for col in ["host_gene", "gene", "Gene.refGene", "Func.refGene", "ExonicFunc.refGene", "sample_id"]:
            if col in sites.columns:
                out[col] = row.get(col, np.nan)

        if not site_genes or annot.empty or annot_gene_col is None:
            rows.append(out)
            continue

        mask = annot_genes.isin(site_genes)
        sub = annot.loc[mask].copy()
        if sub.empty:
            rows.append(out)
            continue

        out["overlap"] = True
        out["n_overlaps"] = int(sub.shape[0])
        out["matched_gene"] = ";".join(sorted(set(site_genes).intersection(set(annot_genes[mask].tolist()))))
        if name_col is not None and name_col in sub.columns:
            out["overlap_names"] = " | ".join(sorted(sub[name_col].dropna().astype(str).unique().tolist())[:20])
        rows.append(out)

    return pd.DataFrame(rows)


def _extract_site_gene_set(
    sv_annotation: pd.DataFrame,
    gene_cols: Sequence[str] = ("host_gene", "gene", "Gene.refGene"),
):
    genes = set()
    if sv_annotation is None or sv_annotation.empty:
        return genes
    for _, row in sv_annotation.iterrows():
        for col in gene_cols:
            if col in sv_annotation.columns and pd.notna(row.get(col, np.nan)):
                genes.update(g.strip().upper() for g in re.split(r"[;,|]", str(row[col])) if g.strip() and g.strip() != ".")
    return genes


def _read_gene_annotation_filtered(
    path,
    target_genes: set,
    gene_col: Optional[str] = None,
    name_col: Optional[str] = None,
    filters: Optional[dict] = None,
    chunksize: int = 200000,
):
    if path is None or str(path).strip() == "":
        return pd.DataFrame(), "missing_path"
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(), "missing_file"
    if not target_genes:
        return pd.DataFrame(), "no_site_genes"

    filters = filters or {}
    cache_key = (
        str(path.resolve()),
        tuple(sorted(map(str, target_genes))),
        str(gene_col),
        str(name_col),
        tuple(sorted((str(k), str(v)) for k, v in filters.items())),
    )
    if cache_key in _EXTERNAL_GENE_FILTER_CACHE:
        return _EXTERNAL_GENE_FILTER_CACHE[cache_key].copy(), "loaded"

    header = pd.read_csv(path, sep="\t", nrows=0, dtype=str, compression="infer")
    cols = list(header.columns)
    if gene_col is None:
        candidates = ["target_gene", "Target Gene", "Gene Symbol", "gene", "Gene", "symbol", "transcript_gene"]
        gene_col = next((c for c in candidates if c in cols), None)
    if gene_col is None or gene_col not in cols:
        return pd.DataFrame(), "missing_gene_column"

    if name_col is None:
        candidates = ["Representative miRNA", "miRNA", "miRNA family", "mirna", "RBP", "rbp", "name", "Name", "database_id"]
        name_col = next((c for c in candidates if c in cols), gene_col)

    usecols = sorted({c for c in [gene_col, name_col, *filters.keys()] if c is not None and c in cols})
    pieces = []
    for chunk in pd.read_csv(path, sep="\t", dtype=str, compression="infer", chunksize=chunksize, usecols=usecols):
        mask = chunk[gene_col].fillna("").astype(str).str.upper().isin(target_genes)
        for filter_col, accepted in filters.items():
            if filter_col not in chunk.columns:
                continue
            if not isinstance(accepted, (list, tuple, set)):
                accepted = [accepted]
            mask &= chunk[filter_col].astype(str).isin([str(v) for v in accepted])
        if mask.any():
            pieces.append(chunk.loc[mask].copy())

    if not pieces:
        result = pd.DataFrame(columns=usecols)
    else:
        result = pd.concat(pieces, ignore_index=True)
    _EXTERNAL_GENE_FILTER_CACHE[cache_key] = result.copy()
    return result, "loaded"


def run_sv_external_mechanism_overlaps(
    sv_annotation: pd.DataFrame,
    annotation_paths: Optional[dict] = None,
    site_col: str = "site",
    flank: int = 0,
    store_key: Optional[str] = None,
    adata_ai=None,
):
    """
    Run optional external mechanism overlaps for all detected SV-A-to-I sites.

    annotation_paths is a dict where each key is a source name and each value is
    either a path or a config dict:

        {
          "miRBase": {"path": "...bed", "type": "bed"},
          "miRGeneDB": {"path": "...bed", "type": "bed"},
          "RepeatMasker/Alu": {"path": "...bed", "type": "bed"},
          "REDIportal": {"path": "...bed", "type": "bed"},
          "TargetScan gene information": {"path": "...zip", "type": "gene", "gene_col": "Gene symbol"},
          "TargetScan miRNA families": {"path": "...zip", "type": "catalog"},
          "TargetScan predictions": {"path": "...zip", "type": "gene", "gene_col": "Gene Symbol"},
          "RBP": {"path": "...bed", "type": "bed"}
        }

    Missing files are reported in the summary instead of failing.
    """
    default_sources = {
        "miRBase": {"path": None, "type": "bed"},
        "miRGeneDB": {"path": None, "type": "bed"},
        "RepeatMasker/Alu": {"path": None, "type": "bed"},
        "REDIportal": {"path": None, "type": "bed"},
        "TargetScan gene information": {"path": None, "type": "gene"},
        "TargetScan miRNA families": {"path": None, "type": "catalog"},
        "TargetScan predictions": {"path": None, "type": "gene"},
        "RBP": {"path": None, "type": "bed"},
    }

    configs = default_sources
    if annotation_paths:
        configs = {**default_sources, **annotation_paths}

    detail_rows = []
    summary_rows = []

    for source, cfg in configs.items():
        if isinstance(cfg, (str, Path)) or cfg is None:
            cfg = {"path": cfg, "type": "bed"}
        cfg = dict(cfg)
        path = cfg.get("path")
        annot_type = str(cfg.get("type", "bed")).lower()

        path_obj = Path(path) if path is not None and str(path).strip() else None
        detail = pd.DataFrame()
        annotation_scope = str(cfg.get("scope", "direct_site_overlap"))

        if annot_type in {"repeatmasker", "rmsk"} and path_obj is not None and path_obj.exists() and path_obj.is_file():
            annot, status = _read_repeatmasker_filtered(
                path, sv_annotation, site_col=site_col, flank=int(cfg.get("flank", flank))
            )
            available = status == "loaded"
            if available:
                detail = overlap_sv_sites_with_interval_annotation(
                    sv_annotation, annot, source=source, site_col=site_col,
                    flank=int(cfg.get("flank", flank)),
                )
        elif annot_type in {"rediportal", "rediting"} and path_obj is not None and path_obj.exists() and path_obj.is_file():
            annot, status = _read_rediportal_filtered(path, sv_annotation, site_col=site_col)
            available = status == "loaded"
            if available:
                detail = overlap_sv_sites_with_interval_annotation(
                    sv_annotation, annot, source=source, site_col=site_col,
                    flank=int(cfg.get("flank", flank)),
                )
        elif annot_type == "gene" and path_obj is not None and path_obj.exists() and path_obj.is_file():
            site_genes = _extract_site_gene_set(sv_annotation)
            annot, status = _read_gene_annotation_filtered(
                path,
                target_genes=site_genes,
                gene_col=cfg.get("gene_col"),
                name_col=cfg.get("name_col"),
                filters=cfg.get("filters"),
            )
            available = status == "loaded"
            if available:
                detail = overlap_sv_sites_with_gene_annotation(
                    sv_annotation,
                    annot,
                    source=source,
                    annot_gene_col=cfg.get("gene_col"),
                    name_col=cfg.get("name_col"),
                    site_col=site_col,
                )
        elif path_obj is not None and path_obj.exists() and path_obj.is_dir() and annot_type in {"bed", "interval", "rbp"}:
            patterns = cfg.get("patterns", ["*.bed.gz", "*.bed", "*.narrowPeak.gz", "*.narrowPeak"])
            files = []
            for pattern in patterns:
                files.extend(sorted(path_obj.glob(pattern)))
            name_regex = cfg.get("name_regex")
            if name_regex and bool(cfg.get("prefilter_files_by_first_name", False)):
                files = [p for p in files if _bed_file_first_name_matches(p, name_regex)]
            status = "loaded" if files else "empty_directory"
            annot = pd.DataFrame()
            available = bool(files)
            detail = overlap_sv_sites_with_bed_file_collection(
                sv_annotation,
                files,
                source=source,
                site_col=site_col,
                flank=int(cfg.get("flank", flank)),
                name_regex=name_regex,
                label_parser=cfg.get("label_parser"),
                deduplicate_labels_per_site=bool(cfg.get("deduplicate_labels_per_site", False)),
            ) if available else pd.DataFrame()
        else:
            annot, status = _read_external_annotation_file(path, annotation_type=annot_type)
            available = status == "loaded" and not annot.empty
            if available and annot_type not in {"catalog"}:
                detail = overlap_sv_sites_with_interval_annotation(
                    sv_annotation,
                    annot,
                    source=source,
                    site_col=site_col,
                    flank=int(cfg.get("flank", flank)),
                )
            elif available and annot_type == "catalog":
                annotation_scope = "reference_catalog_not_direct_site_overlap"

        if detail.empty:
            sites = sv_annotation[[site_col]].copy() if sv_annotation is not None and site_col in sv_annotation.columns else pd.DataFrame({site_col: []})
            detail = sites.copy()
            detail["source"] = source
            detail["overlap"] = False
            detail["n_overlaps"] = 0
            detail["overlap_names"] = ""

        if not detail.empty:
            detail["annotation_path"] = "" if path is None else str(path)
            detail["annotation_status"] = status
            detail["annotation_scope"] = annotation_scope
            detail_rows.append(detail)

        n_site_annotation_links = (
            int(pd.to_numeric(detail.get("n_overlaps", 0), errors="coerce").fillna(0).sum())
            if not detail.empty else 0
        )
        overlap_labels = set()
        if not detail.empty and "overlap_names" in detail.columns:
            for value in detail.loc[detail["overlap"] == True, "overlap_names"].dropna().astype(str):
                overlap_labels.update(x for x in value.split(" | ") if x)

        summary_rows.append(
            {
                "source": source,
                "annotation_path": "" if path is None else str(path),
                "annotation_status": status,
                "available": bool(available),
                "annotation_scope": annotation_scope,
                "n_sites": int(sv_annotation[site_col].nunique()) if sv_annotation is not None and site_col in sv_annotation.columns else 0,
                "n_overlap_sites": int(detail.loc[detail["overlap"] == True, site_col].nunique()) if not detail.empty and site_col in detail.columns else 0,
                "n_site_annotation_links": n_site_annotation_links,
                "n_unique_overlap_labels": len(overlap_labels),
                "overlap_count_definition": "sum of matched annotation rows across queried sites",
            }
        )

    detail_df = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)

    if store_key is not None and adata_ai is not None:
        adata_ai.uns[store_key] = {"detail": detail_df, "summary": summary_df}

    return detail_df, summary_df


def infer_sv_site_mechanism_context(
    sv_annotation: pd.DataFrame,
    external_overlap_detail: Optional[pd.DataFrame] = None,
    site_col: str = "site",
):
    """
    Combine existing gene/region annotation and optional external overlaps into
    a mechanism-oriented site table.
    """
    if sv_annotation is None or sv_annotation.empty:
        return pd.DataFrame()

    base = prepare_rbp_followup_table(sv_annotation)
    if base.empty:
        base = sv_annotation.copy()

    base["has_mirbase_mirgenedb_overlap"] = False
    base["has_repeatmasker_alu_overlap"] = False
    base["has_known_editing_db_overlap"] = False
    base["has_targetscan_overlap"] = False
    base["has_rbp_overlap"] = False
    base["external_overlap_sources"] = ""

    if external_overlap_detail is not None and not external_overlap_detail.empty:
        ov = external_overlap_detail[external_overlap_detail["overlap"] == True].copy()
        if not ov.empty:
            source_by_site = ov.groupby(site_col)["source"].apply(lambda x: ";".join(sorted(set(map(str, x))))).to_dict()
            base["external_overlap_sources"] = base[site_col].map(source_by_site).fillna("")
            for idx, row in base.iterrows():
                sources = str(row["external_overlap_sources"]).lower()
                base.loc[idx, "has_mirbase_mirgenedb_overlap"] = ("mirbase" in sources) or ("mirgenedb" in sources)
                base.loc[idx, "has_repeatmasker_alu_overlap"] = ("repeatmasker" in sources) or ("alu" in sources)
                base.loc[idx, "has_known_editing_db_overlap"] = "rediportal" in sources
                base.loc[idx, "has_targetscan_overlap"] = "targetscan predictions" in sources
                base.loc[idx, "has_rbp_overlap"] = "rbp" in sources

    def _bool_col(frame, col):
        if col in frame.columns:
            return frame[col].fillna(False).astype(bool)
        return pd.Series(False, index=frame.index)

    base["mechanism_class"] = "unresolved"
    base.loc[_bool_col(base, "mirna_context") | base["has_mirbase_mirgenedb_overlap"] | base["has_targetscan_overlap"], "mechanism_class"] = "miRNA/3UTR regulation"
    base.loc[_bool_col(base, "is_alu_or_repeat") | base["has_repeatmasker_alu_overlap"], "mechanism_class"] = "repeat/ADAR substrate"
    base.loc[base["has_known_editing_db_overlap"], "mechanism_class"] = "known RNA-editing site"
    base.loc[base["has_rbp_overlap"], "mechanism_class"] = "RBP binding/regulation"
    evidence_cols = [
        "has_mirbase_mirgenedb_overlap", "has_repeatmasker_alu_overlap",
        "has_known_editing_db_overlap", "has_targetscan_overlap", "has_rbp_overlap",
    ]
    base["n_external_evidence_types"] = sum(_bool_col(base, col).astype(int) for col in evidence_cols)
    base.loc[base["n_external_evidence_types"] >= 2, "mechanism_class"] = "multi-source evidence"

    notes = []
    for _, row in base.iterrows():
        row_notes = []
        if bool(row.get("mirna_context", False)):
            row_notes.append("MIR-associated host annotation motivates noncoding/miRNA follow-up but is not direct miRNA-locus overlap.")
        if bool(row.get("has_mirbase_mirgenedb_overlap", False)):
            row_notes.append("External miRBase/miRGeneDB overlap supports miRNA-gene annotation.")
        if bool(row.get("is_alu_or_repeat", False)) or bool(row.get("has_repeatmasker_alu_overlap", False)):
            row_notes.append("Repeat/Alu context supports canonical ADAR dsRNA-substrate mechanism.")
        if bool(row.get("has_known_editing_db_overlap", False)):
            row_notes.append("REDIportal overlap supports previously cataloged RNA editing.")
        if bool(row.get("has_targetscan_overlap", False)):
            row_notes.append("TargetScan provides human host-gene-level miRNA predictions, not a coordinate-level overlap with the editing site.")
        if bool(row.get("has_rbp_overlap", False)):
            row_notes.append("ENCODE eCLIP IDR peaks support RBP-binding potential; evidence is from K562/HepG2 rather than DLPFC.")
        if not row_notes:
            row_notes.append("No external mechanism overlap yet; interpret through host gene/region only.")
        notes.append(" ".join(row_notes))
    base["mechanism_note"] = notes

    return base


def plot_external_overlap_summary(
    overlap_summary: pd.DataFrame,
    figsize=(7.2, 3.8),
    title="External mechanism overlap summary",
    include_sources: Optional[Sequence[str]] = None,
):
    if overlap_summary is None or overlap_summary.empty:
        raise ValueError("overlap_summary is empty.")

    df = overlap_summary.copy()
    if include_sources is not None:
        include_sources = [str(source) for source in include_sources]
        df = df[df["source"].astype(str).isin(include_sources)].copy()
        if df.empty:
            raise ValueError("No requested include_sources were found in overlap_summary.")
        df["source"] = pd.Categorical(df["source"].astype(str), categories=include_sources, ordered=True)
    if df["source"].duplicated().any():
        df = (
            df.groupby("source", as_index=False, observed=True)
            .agg(
                n_overlap_sites=("n_overlap_sites", "sum"),
                available=("available", "all"),
                annotation_status=("annotation_status", lambda x: ";".join(sorted(set(map(str, x))))),
            )
        )
    if include_sources is not None:
        df = df.sort_values("source").reset_index(drop=True)
    df["available_label"] = np.where(df["available"], "available", "missing")
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=df,
        y="source",
        x="n_overlap_sites",
        hue="available_label",
        dodge=False,
        palette={"available": "#1b9e77", "missing": "#bdbdbd"},
        ax=ax,
    )
    ax.set_xlabel("SV-A-to-I sites with overlap")
    ax.set_ylabel("")
    ax.set_title(title)
    ax.legend(frameon=False, title="")
    for yi, (_, row) in enumerate(df.iterrows()):
        status = row.get("annotation_status", "")
        ax.text(row["n_overlap_sites"], yi, f" {status}", va="center", fontsize=8)
    plt.tight_layout()
    return fig, ax


def plot_three_slice_sv_atoi_upset(
    sv_sites: pd.DataFrame,
    sample_col: str = "sample_id",
    site_col: str = "site",
    is_sv_col: str = "is_sv_atoi",
    sample_order: Optional[Sequence[str]] = None,
    figsize=(9.5, 5.8),
    title="SV-A-to-I site overlap across three slices",
):
    """Draw a dependency-free UpSet plot of called sites in three slices."""
    if sv_sites is None or sv_sites.empty:
        raise ValueError("sv_sites is empty.")
    if sample_col not in sv_sites.columns or site_col not in sv_sites.columns:
        raise ValueError(f"sv_sites must contain {sample_col!r} and {site_col!r}.")

    df = sv_sites.copy()
    if is_sv_col in df.columns:
        df = df.loc[df[is_sv_col].fillna(False).astype(bool)].copy()
    df = df.loc[df[sample_col].notna() & df[site_col].notna(), [sample_col, site_col]]
    df[sample_col] = df[sample_col].astype(str)
    df[site_col] = df[site_col].astype(str)
    df = df.drop_duplicates()
    samples = (list(map(str, sample_order)) if sample_order is not None
               else df[sample_col].drop_duplicates().tolist())
    if len(samples) != 3:
        raise ValueError(f"Exactly three slices are required; found {len(samples)}: {samples}")
    unknown = sorted(set(df[sample_col]).difference(samples))
    if unknown:
        raise ValueError(f"sample_order omits slices present in the data: {unknown}")

    membership = (df.assign(present=True)
                  .pivot_table(index=site_col, columns=sample_col, values="present",
                               aggfunc="any", fill_value=False)
                  .reindex(columns=samples, fill_value=False).astype(bool))
    patterns = []
    for mask in range(1, 1 << len(samples)):
        flags = tuple(bool(mask & (1 << i)) for i in range(len(samples)))
        exact = np.ones(len(membership), dtype=bool)
        for sample, flag in zip(samples, flags):
            exact &= membership[sample].to_numpy() == flag
        patterns.append({"membership": flags, "intersection_size": int(exact.sum()),
                         "sites": membership.index[exact].tolist()})
    intersections = pd.DataFrame(patterns)
    intersections = intersections.loc[intersections["intersection_size"] > 0].copy()
    intersections["degree"] = intersections["membership"].map(sum)
    intersections = intersections.sort_values(
        ["intersection_size", "degree"], ascending=[False, False]
    ).reset_index(drop=True)

    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.25, 3.8), height_ratios=(3.2, 1.35),
                           hspace=0.08, wspace=0.08)
    ax_sets = fig.add_subplot(grid[1, 0])
    ax_bars = fig.add_subplot(grid[0, 1])
    ax_matrix = fig.add_subplot(grid[1, 1], sharex=ax_bars)
    x = np.arange(len(intersections))
    sizes = intersections["intersection_size"].to_numpy()
    ax_bars.bar(x, sizes, color="#4e79a7", width=0.72)
    for xi, value in zip(x, sizes):
        ax_bars.text(xi, value, str(value), ha="center", va="bottom", fontsize=8)
    ax_bars.set_ylabel("Intersection size")
    ax_bars.set_title(title)
    ax_bars.tick_params(axis="x", bottom=False, labelbottom=False)

    for xi, flags in zip(x, intersections["membership"]):
        active = [i for i, flag in enumerate(flags) if flag]
        if len(active) > 1:
            ax_matrix.plot([xi, xi], [min(active), max(active)], color="#333333", lw=1.5)
        for yi, flag in enumerate(flags):
            ax_matrix.scatter(xi, yi, s=42,
                              color="#333333" if flag else "#d9d9d9", zorder=3)
    # Matplotlib <3.5 does not support the ``labels=`` keyword in set_yticks.
    ax_matrix.set_yticks(range(3))
    ax_matrix.set_yticklabels(samples)
    ax_matrix.set_ylim(2.55, -0.55)
    ax_matrix.set_xlabel("Exclusive intersections")
    ax_matrix.spines[["top", "right", "bottom"]].set_visible(False)
    ax_matrix.tick_params(axis="x", bottom=False, labelbottom=False)

    set_sizes = membership.sum(axis=0).reindex(samples)
    ax_sets.barh(range(3), set_sizes.to_numpy(), color="#59a14f", height=0.6)
    for yi, value in enumerate(set_sizes):
        ax_sets.text(value, yi, f" {int(value)}", va="center", fontsize=8)
    ax_sets.set_yticks(range(3))
    ax_sets.set_yticklabels(samples)
    ax_sets.set_ylim(2.55, -0.55)
    ax_sets.invert_xaxis()
    ax_sets.set_xlabel("Set size")
    ax_sets.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    axes = {"set_size": ax_sets, "intersection": ax_bars, "matrix": ax_matrix}
    return fig, axes, intersections




def prepare_rbp_followup_table(
    sv_annotation: pd.DataFrame,
    utr_patterns: Sequence[str] = ("UTR", "exonic", "intronic", "ncRNA"),
):
    """
    Prioritize SV-A-to-I sites for external RBP/miRNA overlap follow-up.

    No external RBP database is queried here. The output is a focused site list
    suitable for POSTAR/RBPmap/ENCODE eCLIP or miRNA-target tools.
    """
    if sv_annotation is None or sv_annotation.empty:
        return pd.DataFrame()

    df = sv_annotation.copy()
    region_cols = [c for c in ["Func.refGene", "ExonicFunc.refGene", "GeneDetail.refGene"] if c in df.columns]

    if region_cols:
        region_text = (
            df[region_cols]
            .astype(object)
            .where(df[region_cols].notna(), "")
            .astype(str)
            .agg(";".join, axis=1)
        )
    else:
        region_text = pd.Series("", index=df.index)

    pattern = "|".join(utr_patterns)
    df["_region_text"] = region_text
    df["rbp_followup_region"] = region_text.str.contains(pattern, case=False, regex=True, na=False)
    df["rbp_region_flag"] = df["rbp_followup_region"]

    df["regulatory_region_class"] = "other"
    df.loc[region_text.str.contains("intergenic", case=False, regex=True, na=False), "regulatory_region_class"] = "intergenic"
    df.loc[region_text.str.contains("intronic|intron", case=False, regex=True, na=False), "regulatory_region_class"] = "intronic"
    df.loc[region_text.str.contains("exonic|exon", case=False, regex=True, na=False), "regulatory_region_class"] = "exonic"
    df.loc[region_text.str.contains("ncRNA", case=False, regex=True, na=False), "regulatory_region_class"] = "ncRNA"
    df.loc[region_text.str.contains("UTR5|5UTR|5'UTR", case=False, regex=True, na=False), "regulatory_region_class"] = "5UTR"
    df.loc[region_text.str.contains("UTR3|3UTR|3'UTR", case=False, regex=True, na=False), "regulatory_region_class"] = "3UTR"
    gene_text = pd.Series("", index=df.index)
    for col in ["host_gene", "gene", "Gene.refGene"]:
        if col in df.columns:
            gene_text = gene_text + ";" + df[col].astype(object).where(df[col].notna(), "").astype(str)
    df["mirna_gene_context"] = gene_text.str.contains(r"\bMIR\d+|MIRNA|MICRORNA", case=False, regex=True, na=False)
    df["mirna_regulatory_context"] = region_text.str.contains(r"UTR3|3UTR|3'UTR|miRNA|microRNA", case=False, regex=True, na=False)
    df["mirna_context"] = df["mirna_gene_context"] | df["mirna_regulatory_context"]

    repeat_cols = [c for c in ["repeat", "Repeat", "repName", "repClass", "repFamily"] if c in df.columns]
    if repeat_cols:
        repeat_text = (
            df[repeat_cols]
            .astype(object)
            .where(df[repeat_cols].notna(), "")
            .astype(str)
            .agg(";".join, axis=1)
        )
    else:
        repeat_text = pd.Series("", index=df.index)

    df["repeat_annotation"] = repeat_text.replace("", np.nan)
    df["has_repeat_annotation"] = repeat_text.str.strip().ne("")
    df["is_alu_or_repeat"] = repeat_text.str.contains("Alu|SINE|LINE|LTR|repeat", case=False, regex=True, na=False)
    df["repeat_context"] = "repeat annotation not provided"
    df.loc[df["has_repeat_annotation"] & ~df["is_alu_or_repeat"], "repeat_context"] = "annotated non-Alu/repeat"
    df.loc[df["is_alu_or_repeat"], "repeat_context"] = "Alu/repeat-supported"

    coords = df["site"].astype(str).str.extract(r"^(?P<chrom>[^_]+)_(?P<pos>\d+)$")
    df["bed_chrom"] = np.where(coords["chrom"].notna(), "chr" + coords["chrom"].astype(str).str.replace("^chr", "", regex=True), np.nan)
    df["bed_start"] = pd.to_numeric(coords["pos"], errors="coerce") - 1
    df["bed_end"] = pd.to_numeric(coords["pos"], errors="coerce")
    df["bed_name"] = df["site"].astype(str)

    df["recommended_external_resources"] = "POSTAR3; ENCODE eCLIP; RBPmap/ATtRACT"
    df.loc[df["regulatory_region_class"].isin(["3UTR", "5UTR"]), "recommended_external_resources"] += "; TargetScan/miRDB"

    df["regulatory_hypothesis"] = "lower priority: no transcript regulatory-region annotation"
    df.loc[df["regulatory_region_class"].eq("3UTR"), "regulatory_hypothesis"] = "3UTR site: candidate miRNA/RBP-mediated stability or localization effect"
    df.loc[df["regulatory_region_class"].eq("5UTR"), "regulatory_hypothesis"] = "5UTR site: candidate translation/RBP regulatory effect"
    df.loc[df["regulatory_region_class"].eq("ncRNA"), "regulatory_hypothesis"] = "ncRNA site: candidate RNA-structure/RBP interaction effect"
    df.loc[df["regulatory_region_class"].eq("intronic"), "regulatory_hypothesis"] = "intronic site: candidate splicing or nuclear RBP interaction effect"
    df.loc[df["is_alu_or_repeat"], "regulatory_hypothesis"] = (
        df.loc[df["is_alu_or_repeat"], "regulatory_hypothesis"].astype(str)
        + "; repeat/Alu context supports canonical ADAR substrate follow-up"
    )
    df.loc[df["mirna_context"], "regulatory_hypothesis"] = (
        df.loc[df["mirna_context"], "regulatory_hypothesis"].astype(str)
        + "; miRNA/3UTR context supports focused WM post-transcriptional regulation follow-up"
    )

    df["rbp_followup_priority"] = "lower"
    df.loc[df["rbp_followup_region"], "rbp_followup_priority"] = "candidate"
    df.loc[
        df["regulatory_region_class"].isin(["3UTR", "ncRNA"]) | df["is_alu_or_repeat"],
        "rbp_followup_priority",
    ] = "high"
    df.loc[df["mirna_context"], "rbp_followup_priority"] = "high"

    reason_parts = []
    for idx, row in df.iterrows():
        reasons = []
        if bool(row.get("rbp_region_flag", False)):
            reasons.append("RBP-suitable transcript region")
        if bool(row.get("mirna_gene_context", False)):
            reasons.append("MIR host-gene annotation")
        if bool(row.get("mirna_regulatory_context", False)):
            reasons.append("3UTR/miRNA regulatory context")
        if bool(row.get("is_alu_or_repeat", False)):
            reasons.append("Alu/repeat-supported ADAR substrate context")
        if not reasons:
            reasons.append("no focused regulatory-context evidence in current annotation")
        reason_parts.append("; ".join(reasons))
    df["followup_priority_reason"] = reason_parts
    df["followup_route"] = "lower-priority annotation follow-up"
    df.loc[df["rbp_region_flag"], "followup_route"] = "RBP-region follow-up"
    df.loc[df["mirna_context"], "followup_route"] = "miRNA/3UTR follow-up"
    df.loc[df["rbp_region_flag"] & df["mirna_context"], "followup_route"] = "RBP + miRNA/3UTR follow-up"
    df.loc[df["is_alu_or_repeat"], "followup_route"] = df.loc[df["is_alu_or_repeat"], "followup_route"].astype(str) + " + repeat/ADAR"

    if "spatial_fdr" in df.columns:
        priority_order = {"high": 0, "candidate": 1, "lower": 2}
        df["_priority_order"] = df["rbp_followup_priority"].map(priority_order).fillna(9)
        df = df.sort_values(["_priority_order", "spatial_fdr", "spatial_effect"], ascending=[True, True, False])
    elif "moran_fdr" in df.columns:
        priority_order = {"high": 0, "candidate": 1, "lower": 2}
        df["_priority_order"] = df["rbp_followup_priority"].map(priority_order).fillna(9)
        df = df.sort_values(["_priority_order", "moran_fdr", "moran_I"], ascending=[True, True, False])

    keep = [
        "site",
        "bed_chrom",
        "bed_start",
        "bed_end",
        "bed_name",
        "host_gene",
        "gene",
        "Gene.refGene",
        "Func.refGene",
        "ExonicFunc.refGene",
        "AAChange.refGene",
        "regulatory_region_class",
        "rbp_region_flag",
        "mirna_context",
        "mirna_gene_context",
        "mirna_regulatory_context",
        "repeat_annotation",
        "has_repeat_annotation",
        "is_alu_or_repeat",
        "repeat_context",
        "spatial_effect",
        "spatial_lrt",
        "spatial_p",
        "spatial_fdr",
        "moran_I",
        "moran_p",
        "moran_fdr",
        "rbp_followup_region",
        "rbp_followup_priority",
        "followup_priority_reason",
        "followup_route",
        "recommended_external_resources",
        "regulatory_hypothesis",
    ]
    keep = [c for c in keep if c in df.columns]

    return df.loc[:, keep].reset_index(drop=True)


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


def plot_cross_sample_r2_decomposition(
    model_df: pd.DataFrame,
    sample_col: str = "sample_id",
    celltype_col: str = "celltype_r2",
    joint_col: str = "joint_r2",
    figsize=(6.8, 4.6),
    title="Cross-sample joint R2 decomposition",
):
    """Stack cell-type R2, incremental ADAR-family R2, and unexplained variance."""
    if model_df is None or model_df.empty:
        raise ValueError("model_df is empty.")
    required = {sample_col, celltype_col, joint_col}
    missing = required - set(model_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = model_df.copy()
    df[celltype_col] = pd.to_numeric(df[celltype_col], errors="coerce").clip(0, 1)
    df[joint_col] = pd.to_numeric(df[joint_col], errors="coerce").clip(0, 1)
    df["Cell-type modules"] = df[celltype_col]
    df["ADAR-family gain"] = (df[joint_col] - df[celltype_col]).clip(lower=0)
    df["Unexplained"] = (1 - df[joint_col]).clip(lower=0)
    df = df.reset_index(drop=True)

    components = ["Cell-type modules", "ADAR-family gain", "Unexplained"]
    colors = {
        "Cell-type modules": "#4c78a8",
        "ADAR-family gain": "#f28e2b",
        "Unexplained": "#d9d9d9",
    }
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(df))
    bottom = np.zeros(len(df), dtype=float)
    for component in components:
        values = df[component].fillna(0).to_numpy(dtype=float)
        bars = ax.bar(
            x, values, bottom=bottom, width=0.68,
            color=colors[component], edgecolor="white", linewidth=0.7,
            label=component,
        )
        for xpos, value, base in zip(x, values, bottom):
            if value >= 0.035:
                ax.text(xpos, base + value / 2, f"{value:.2f}", ha="center", va="center", fontsize=8)
        bottom += values

    for xpos, joint_r2 in zip(x, df[joint_col]):
        if np.isfinite(joint_r2):
            ax.text(xpos, 1.015, f"joint R2={joint_r2:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(df[sample_col].astype(str))
    ax.set_ylim(0, 1.10)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Fraction of variance")
    ax.set_title(title)
    ax.legend(frameon=False, title="", loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    plt.tight_layout()

    long_df = df.melt(
        id_vars=[sample_col, celltype_col, joint_col],
        value_vars=components,
        var_name="component",
        value_name="variance_fraction",
    )
    return fig, ax, df, long_df


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


def plot_global_atoi_two_r2_donuts(
    adata_ai,
    original_key="global_atoi_score",
    residual_celltype_key="global_atoi_residual_celltype",
    residual_celltype_adar_key="global_atoi_residual_celltype_adar",
    celltype_store_key="cluster_celltype_only_global_atoi_model",
    joint_store_key="cluster_joint_global_atoi_model",
    use_model_summary=True,
    titles=("Cell type", "Cell type + ADAR"),
    figsize=(7.2, 3.6),
    save_path=None,
):
    """
    Plot two donut charts showing R2 of:
        1. Cell type model
        2. Cell type + ADAR-family model

    By default, R2 is read from the stored cluster-level model summaries so
    this plot matches the OLS summaries used for the residual/joint model.
    If summaries are unavailable, R2 falls back to residual variance:
        R2 = 1 - Var(residual) / Var(original)

    Required columns in adata_ai.obs:
        original_key
        residual_celltype_key
        residual_celltype_adar_key
    """

    method = "model_summary"
    adj_r2_values = [np.nan, np.nan]

    if use_model_summary:
        celltype_summary = adata_ai.uns.get(celltype_store_key, {})
        joint_summary = adata_ai.uns.get(joint_store_key, {})
        r2_values = [
            celltype_summary.get("r2", np.nan),
            joint_summary.get("r2", np.nan),
        ]
        adj_r2_values = [
            celltype_summary.get("adj_r2", np.nan),
            joint_summary.get("adj_r2", np.nan),
        ]
        r2_values = [float(v) if pd.notna(v) else np.nan for v in r2_values]
        adj_r2_values = [float(v) if pd.notna(v) else np.nan for v in adj_r2_values]
    else:
        r2_values = [np.nan, np.nan]

    if not all(np.isfinite(v) for v in r2_values):
        method = "residual_variance"
        required_keys = [
            original_key,
            residual_celltype_key,
            residual_celltype_adar_key,
        ]

        missing = [k for k in required_keys if k not in adata_ai.obs.columns]
        if missing:
            raise ValueError(f"Missing columns in adata_ai.obs: {missing}")

        df = adata_ai.obs[required_keys].astype(float).dropna()

        if df.empty:
            raise ValueError("No valid values available for R2 calculation.")

        y = df[original_key].values
        resid_celltype = df[residual_celltype_key].values
        resid_celltype_adar = df[residual_celltype_adar_key].values

        var_y = np.nanvar(y, ddof=1)

        if not np.isfinite(var_y) or var_y <= 0:
            raise ValueError("Original global A-to-I variance is zero or invalid.")

        r2_celltype = 1 - np.nanvar(resid_celltype, ddof=1) / var_y
        r2_celltype_adar = 1 - np.nanvar(resid_celltype_adar, ddof=1) / var_y
        r2_values = [r2_celltype, r2_celltype_adar]

    r2_values = [float(np.clip(v, 0, 1)) for v in r2_values]

    summary = pd.DataFrame(
        {
            "model": list(titles),
            "R2": r2_values,
            "adjusted_R2": adj_r2_values,
            "Unexplained": [1 - r for r in r2_values],
            "method": method,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, model_title, r2 in zip(axes, titles, r2_values):
        values = [r2, 1 - r2]
        labels = [
            f"Explained\n{r2 * 100:.1f}%",
            f"Unexplained\n{(1 - r2) * 100:.1f}%",
        ]

        ax.pie(
            values,
            labels=labels,
            startangle=90,
            counterclock=False,
            wedgeprops={
                "width": 0.38,
                "edgecolor": "white",
                "linewidth": 1.0,
            },
            textprops={
                "fontsize": 9,
            },
        )

        ax.text(
            0,
            0,
            f"R²\n{r2:.3f}",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
        )

        ax.set_title(model_title, fontsize=12, pad=10)
        ax.axis("equal")

    fig.suptitle(
        "Variance explained in global A-to-I",
        fontsize=13,
        y=1.03,
    )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes, summary


def _linear_design(coords, n_knots=0, reference_coords=None):
    reference = coords if reference_coords is None else reference_coords
    center = reference.mean(axis=0)
    z = coords - center
    scale = np.ptp(reference, axis=0).max()
    if scale <= 0:
        return np.ones((len(coords), 1))
    z /= scale
    if n_knots:
        support = np.unique((reference-center)/scale, axis=0)
        chosen = [int(np.argmin(np.sum(support**2, axis=1)))]
        for _ in range(min(n_knots, len(support))-1):
            distance = ((support[:,None]-support[chosen])**2).sum(axis=2).min(axis=1)
            chosen.append(int(np.argmax(distance)))
        knots = support[chosen]
        distance = np.sqrt(((knots[:,None]-knots)**2).sum(axis=2))
        positive = distance[distance>0]
        width = float(np.median(positive)) if positive.size else .5
        rbf = np.exp(-((z[:,None]-knots)**2).sum(axis=2)/(2*width**2))
        z = np.column_stack([z,rbf])
    z -= z.mean(axis=0)
    q, r, _ = qr(z, mode="economic", pivoting=True)
    diag = np.abs(np.diag(r))
    rank = int(np.sum(diag > max(z.shape) * np.finfo(float).eps * max(diag.max(), 1)))
    return np.column_stack([np.ones(len(coords)), q[:, :rank] * np.sqrt(len(coords))])


def _fit_binomial(g, n, x):
    initial = np.zeros(x.shape[1])
    initial[0] = logit(g.sum() / n.sum())
    logchoose = gammaln(n+1)-gammaln(g+1)-gammaln(n-g+1)
    def objective(beta):
        eta = x @ beta
        ll = np.sum(logchoose + g*eta - n*np.logaddexp(0, eta))
        return -ll, x.T @ (n*expit(eta)-g)
    fit = minimize(objective, initial, jac=True, method="L-BFGS-B",
                   bounds=[(-30,30)]*x.shape[1],
                   options={"maxiter":500,"maxls":50,"ftol":1e-9,"gtol":1e-5})
    if not fit.success or not np.isfinite(fit.fun):
        raise RuntimeError("optimizer did not converge")
    if np.any(np.abs(fit.x)>=29.99):
        raise RuntimeError("coefficient boundary / possible separation")
    return -float(fit.fun), expit(x @ fit.x)


def _select_common_k(adata_ai, candidates, store_key, verbose, kwargs):
    tables, configs = {}, {}
    old = {k:v for k,v in adata_ai.uns.items() if k==store_key or k.startswith(store_key+'_')}
    try:
        for k in candidates:
            if verbose:
                print('BIC selection: fitting K={}'.format(k), flush=True)
            tables[k] = detect_spatial_atoi_sites_counts(adata_ai, n_knots=k,
                store_key=store_key, verbose=False, **kwargs).set_index('site')
            configs[k] = dict(adata_ai.uns[store_key+'_model_config'])
    finally:
        for key in list(adata_ai.uns):
            if key==store_key or key.startswith(store_key+'_'):
                del adata_ai.uns[key]
        adata_ai.uns.update(old)
    common = set(tables[candidates[0]].index)
    for table in tables.values():
        common &= set(table.index[table.pass_qc & table.fit_status.eq('ok')])
    totals, audit = [], []
    for k, table in tables.items():
        bic = -2*table.loglik_spatial + (table.spatial_df+1)*np.log(table.n_valid_spots.clip(lower=1))
        used = table.index.isin(common)
        totals.append(dict(n_knots=k, n_selection_sites=len(common),
            total_bic=float(bic.loc[used].sum()) if common else np.nan))
        audit.append(pd.DataFrame(dict(site=table.index, n_knots=k, bic=bic.to_numpy(),
            used_for_selection=used, fit_status=table.fit_status.to_numpy())))
    summary = pd.DataFrame(totals)
    adata_ai.uns[store_key+'_k_selection'] = summary
    adata_ai.uns[store_key+'_k_selection_sites'] = pd.concat(audit,ignore_index=True)
    if not common:
        raise ValueError('No common successfully fitted sites across candidate K values; inspect the K-selection audit.')
    selected = int(summary.sort_values(['total_bic','n_knots']).iloc[0].n_knots)
    summary['selected'] = summary.n_knots.eq(selected)
    result = tables[selected].reset_index()
    result['p_method'] = 'asymptotic_chi2_conditional_on_selected_k'
    result['k_selection_site'] = result.site.isin(common)
    config = configs[selected]
    config.update(n_knots='auto', selected_n_knots=selected, knot_candidates=candidates,
        selection_scope='dataset', selection_adjusted=False,
        selection_criterion='sum BIC on common successfully fitted sites; smaller K breaks ties')
    adata_ai.uns[store_key] = result
    adata_ai.uns[store_key+'_model_config'] = config
    if verbose:
        print('Selected shared K={} from {} common sites; p values conditional on selected K.'.format(selected,len(common)))
    return result


def detect_spatial_atoi_sites_counts(
    adata_ai, min_cov=5, min_valid_spots=30, min_total_A=10, min_total_G=10,
    sv_call_by="p", p_cutoff=.05, fdr_cutoff=.05,
    a_layer="A", g_layer="G", spatial_coords=None, store_key="sv_atoi", verbose=True,
    n_knots=0, knot_candidates=(0, 1, 2, 3, 4, 5, 6),
):
    """Jointly model each site's G|(A+G) across covered spots using binomial counts.

    H0: logit(p)=intercept; H1: logit(p)=intercept+x+y. Coordinates are scaled
    and orthogonalized without changing the linear model. The likelihood-ratio
    chi-square degrees of freedom equal the observed spatial design rank (normally
    2 with K=0). n_knots=integer adds K Gaussian RBFs to x/y; n_knots='auto'
    selects ONE K for this dataset by summed BIC on the common set of sites
    successfully fitted for every candidate. All candidates use identical spots
    per site. Auto-mode p values condition on selected K, without selection
    calibration. Centers use deterministic farthest-point sampling of coordinate
    support; width is median inter-center distance (.5 for a single center), on
    scaled coordinates. Rank-deficient columns are removed before fitting.
    Zero coverage and missing coordinates are excluded. A=0 or G=0 observations
    remain valid when coverage passes. Require >=3 spots per fitted parameter.
    BH includes QC-passing failed fits as p=1; their displayed p/FDR remain NaN.
    Calls use raw p by default, or BH when sv_call_by='fdr'. Both are returned.
    Inference assumes binomial variance and conditionally independent spots.
    """
    if sv_call_by not in {"p","fdr"}:
        raise ValueError("sv_call_by must be 'p' or 'fdr'")
    if not 0<p_cutoff<1 or not 0<fdr_cutoff<1:
        raise ValueError("p_cutoff and fdr_cutoff must be between 0 and 1")
    if min_cov<1 or min_valid_spots<3 or min_total_A<0 or min_total_G<0:
        raise ValueError("Invalid count/spot QC thresholds")
    if isinstance(n_knots,str) and n_knots=='auto':
        candidates = list(knot_candidates)
        if not candidates or any(not isinstance(k,(int,np.integer)) or k<0 for k in candidates):
            raise ValueError('knot_candidates must be nonempty nonnegative integers')
        return _select_common_k(adata_ai,sorted(set(int(k) for k in candidates)),store_key,verbose,
            dict(min_cov=min_cov,min_valid_spots=min_valid_spots,min_total_A=min_total_A,
                min_total_G=min_total_G,sv_call_by=sv_call_by,p_cutoff=p_cutoff,fdr_cutoff=fdr_cutoff,
                a_layer=a_layer,g_layer=g_layer,spatial_coords=spatial_coords))
    if not isinstance(n_knots,(int,np.integer)) or n_knots<0:
        raise ValueError("n_knots must be a nonnegative integer or 'auto'")
    coords = _coords_from_adata(adata_ai) if spatial_coords is None else np.asarray(spatial_coords,float)
    if coords.shape != (adata_ai.n_obs,2):
        raise ValueError("Spatial coordinates must have shape (n_spots,2)")
    finite = np.isfinite(coords).all(axis=1)
    matrices=[]
    for name in (a_layer,g_layer):
        if name not in adata_ai.layers:
            raise ValueError("Missing count layer: "+name)
        m=adata_ai.layers[name]
        values=m.data if sparse.issparse(m) else np.asarray(m)
        if not np.isfinite(values).all() or np.any(values<0) or not np.allclose(values,np.round(values),rtol=0,atol=1e-8):
            raise ValueError("Count layers must contain finite nonnegative integers")
        matrices.append(m.tocsc() if sparse.issparse(m) else np.asarray(m))
    rows=[]
    for j,site in enumerate(adata_ai.var_names):
        a,g=[np.asarray(m[:,j].toarray() if sparse.issparse(m) else m[:,j],float).ravel() for m in matrices]
        n=a+g; valid=finite & (n>=min_cov)
        av,gv,nv=a[valid],g[valid],n[valid]
        ratio=gv/nv
        row=dict(site=str(site),gene=adata_ai.var.iloc[j].get("Gene.refGene",""),
                 n_valid_spots=len(nv),total_A_valid=float(av.sum()),total_G_valid=float(gv.sum()),
                 total_cov_valid=float(nv.sum()),mean_ratio=float(ratio.mean()) if len(nv) else np.nan,
                 sd_ratio=float(ratio.std()) if len(nv) else np.nan,pass_qc=False,fit_status="insufficient_counts",
                 spatial_df=np.nan,spatial_lrt=np.nan,spatial_p=np.nan,spatial_effect=np.nan,
                 null_probability=np.nan,pearson_dispersion=np.nan,loglik_null=np.nan,loglik_spatial=np.nan)
        if len(nv)>=min_valid_spots and av.sum()>=min_total_A and gv.sum()>=min_total_G and av.sum()>0 and gv.sum()>0:
            x=_linear_design(coords[valid],n_knots,coords[finite]); df=x.shape[1]-1
            row["spatial_df"]=df
            if df==0:
                row["fit_status"]="no_spatial_variation"
            elif len(nv)<3*x.shape[1]:
                row["fit_status"]="insufficient_spots_for_model"
            else:
                row["pass_qc"]=True
                try:
                    p0=gv.sum()/nv.sum()
                    ll0=float(np.sum(gammaln(nv+1)-gammaln(gv+1)-gammaln(nv-gv+1)+gv*np.log(p0)+(nv-gv)*np.log1p(-p0)))
                    ll1,p1=_fit_binomial(gv,nv,x)
                    if ll1<ll0-1e-5:
                        raise RuntimeError("alternative likelihood below null")
                    statistic=max(0.,2*(ll1-ll0))
                    row.update(fit_status="ok",spatial_lrt=statistic,spatial_p=float(chi2.sf(statistic,df)),
                               spatial_effect=float(np.quantile(p1,.95)-np.quantile(p1,.05)),
                               null_probability=float(p0),loglik_null=ll0,loglik_spatial=ll1,
                               pearson_dispersion=float(np.sum((gv-nv*p1)**2/np.maximum(nv*p1*(1-p1),1e-12))/(len(nv)-x.shape[1])))
                except (RuntimeError,ValueError,np.linalg.LinAlgError) as exc:
                    row["fit_status"]="fit_failed: "+str(exc)
        rows.append(row)
        if verbose and (j+1)%100==0:
            print("Linear binomial: {}/{} sites".format(j+1,adata_ai.n_vars))
    columns=["site","gene","n_valid_spots","total_A_valid","total_G_valid","total_cov_valid",
             "mean_ratio","sd_ratio","pass_qc","fit_status","spatial_df","spatial_lrt","spatial_p",
             "spatial_effect","null_probability","pearson_dispersion","loglik_null","loglik_spatial"]
    result=pd.DataFrame(rows,columns=columns)
    result["pass_qc"]=result.pass_qc.astype(bool)
    result["spatial_fdr"]=np.nan
    qc=result.pass_qc
    if qc.any():
        result.loc[qc,"spatial_fdr"]=multipletests(result.loc[qc,"spatial_p"].fillna(1),method="fdr_bh")[1]
        result.loc[result.spatial_p.isna(),"spatial_fdr"]=np.nan
    col="spatial_p" if sv_call_by=="p" else "spatial_fdr"
    cutoff=p_cutoff if sv_call_by=="p" else fdr_cutoff
    result["is_sv_atoi"]=qc & result[col].lt(cutoff)
    result["sv_call_by"]=col
    result["p_cutoff"]=p_cutoff
    result["fdr_cutoff"]=fdr_cutoff
    result["model_family"]="binomial"
    result["selected_n_knots"]=int(n_knots)
    result["p_method"]="asymptotic_chi2"
    result=result.sort_values(["spatial_p","spatial_effect"],ascending=[True,False]).reset_index(drop=True)
    # Remove stale auxiliary results from earlier, more complex implementations.
    for suffix in ("_k_selection","_k_selection_sites","_family_diagnostics","_family_diagnostics_config"):
        adata_ai.uns.pop(store_key+suffix,None)
    adata_ai.uns[store_key]=result
    adata_ai.uns[store_key+"_model_config"]=dict(family="binomial",null="logit(p)=intercept",
        alternative="logit(p)=intercept+x+y+Gaussian_RBFs" if n_knots else "logit(p)=intercept+x+y",
        n_knots=int(n_knots), selected_n_knots=int(n_knots),test="asymptotic likelihood-ratio chi-square",
        min_cov=min_cov,min_valid_spots=min_valid_spots,min_total_A=min_total_A,min_total_G=min_total_G,
        a_layer=a_layer,g_layer=g_layer,sv_call_by=col,p_cutoff=p_cutoff,fdr_cutoff=fdr_cutoff)
    if verbose:
        print("Linear binomial: {} fitted, {} called ({} < {})".format(int(result.fit_status.eq("ok").sum()),int(result.is_sv_atoi.sum()),col,cutoff))
    return result


def plot_count_spatial_discovery(sv_atoi, figsize=(14.5, 4.2)):
    """Plot discovery counts plus significance/effect and coverage diagnostics."""
    required = {"pass_qc", "is_sv_atoi", "spatial_effect", "spatial_p",
                "total_cov_valid", "mean_ratio"}
    missing = sorted(required.difference(sv_atoi.columns))
    if missing:
        raise ValueError(f"sv_atoi is missing required columns: {missing}")

    df = sv_atoi.copy()
    qc = df["pass_qc"].fillna(False).astype(bool)
    called = df["is_sv_atoi"].fillna(False).astype(bool)
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    ax = axes[0]
    counts = [len(sv_atoi), int(sv_atoi.pass_qc.sum()), int(sv_atoi.is_sv_atoi.sum())]
    bars = ax.bar(["all", "QC pass", "SV"], counts,
                  color=["#bdbdbd", "#f28e2b", "#59a14f"])
    ax.set(ylabel="Site count", title="Count-model discovery")
    ax.set_ylim(0, max(1, max(counts))*1.15)
    for bar, value in zip(bars, counts):
        ax.annotate(str(value), (bar.get_x()+bar.get_width()/2, value),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom")

    ax = axes[1]
    mean_ratio = pd.to_numeric(df["mean_ratio"], errors="coerce")
    pvalue = pd.to_numeric(df["spatial_p"], errors="coerce")
    fitted = np.isfinite(pvalue)
    minus_log10_p = pd.Series(0.0, index=df.index)
    minus_log10_p.loc[fitted] = -np.log10(
        pvalue.loc[fitted].clip(lower=np.finfo(float).tiny)
    )
    visible = np.isfinite(mean_ratio)
    ax.scatter(mean_ratio[visible & ~fitted], minus_log10_p[visible & ~fitted],
               s=9, color="#d9d9d9", alpha=0.45, linewidth=0,
               label=f"not fitted/QC fail (n={(visible & ~fitted).sum()})", rasterized=True)
    ax.scatter(mean_ratio[visible & fitted & ~called],
               minus_log10_p[visible & fitted & ~called], s=18,
               color="#8c8c8c", alpha=0.65, linewidth=0,
               label=f"fitted, not called (n={(visible & fitted & ~called).sum()})",
               rasterized=True)
    ax.scatter(mean_ratio[visible & called], minus_log10_p[visible & called], s=25,
               color="#59a14f", alpha=0.9, linewidth=0, label="SV-A-to-I")
    if "p_cutoff" in df.columns:
        cutoffs = pd.to_numeric(df["p_cutoff"], errors="coerce").dropna()
        if not cutoffs.empty and cutoffs.iloc[0] > 0:
            ax.axhline(-np.log10(cutoffs.iloc[0]), color="#d62728", ls="--", lw=1)
    ax.set(xlabel="Mean editing ratio across valid spots", ylabel="-log10(spatial p)",
           title="Spatial count regression")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    cov = pd.to_numeric(df["total_cov_valid"], errors="coerce")
    raw_effect = pd.to_numeric(df["spatial_effect"], errors="coerce")
    effect = raw_effect.fillna(0.0)
    valid = np.isfinite(cov) & (cov > 0)
    ax.scatter(cov[valid & ~fitted], effect[valid & ~fitted], s=9,
               color="#d9d9d9", alpha=0.45, linewidth=0, rasterized=True)
    ax.scatter(cov[valid & fitted & ~called], effect[valid & fitted & ~called], s=18,
               color="#8c8c8c", alpha=0.6, linewidth=0, rasterized=True)
    ax.scatter(cov[valid & called], effect[valid & called], s=25,
               color="#59a14f", alpha=0.9, linewidth=0, rasterized=True)
    ax.set_xscale("log")
    ax.set(xlabel="Total coverage across valid spots (log scale)",
           ylabel="Spatial effect (P95 - P5; 0 = not fitted)",
           title="Coverage and spatial effect")
    fig.tight_layout()
    return fig, axes


def validate_counts(d):
    a = d[['edited', 'unedited']].to_numpy(float)
    if not np.isfinite(a).all() or (a < 0).any() or not np.allclose(a, np.round(a)):
        raise ValueError('edited/unedited must be finite, nonnegative integer counts')
    if d.index.has_duplicates:
        raise ValueError('Duplicate spot identifiers')
    return d.assign(total=a.sum(axis=1))


def matrix_counts(adata, site):
    """Existing workflow convention: G=edited, A=unedited; strand unverified."""
    if not adata.var_names.is_unique:
        raise ValueError('Duplicate site IDs')
    j = adata.var_names.get_loc(site)
    def column(name):
        x = adata.layers[name][:, j]
        return np.asarray(x.toarray() if hasattr(x, 'toarray') else x).ravel()
    return validate_counts(pd.DataFrame({'edited': column('G'), 'unedited': column('A')},
                                       index=adata.obs_names))


def overlap_resource(path, chrom, pos1, kind='bed'):
    """BED is 0-based half-open; GTF is 1-based inclusive. Exact contig names."""
    if not path or not Path(path).is_file():
        return 'pending', ''
    opener = gzip.open if str(path).endswith('.gz') else open
    hits = set()
    seen_contig = False
    with opener(path, 'rt') as stream:
        for line in stream:
            if line.startswith(('#', 'track', 'browser')) or not line.strip():
                continue
            f = line.rstrip().split('\t')
            if f[0] != chrom:
                continue
            seen_contig = True
            if kind == 'gtf':
                if len(f) < 9 or f[2] != 'gene':
                    continue
                if int(f[3]) <= pos1 <= int(f[4]):
                    hits.add(f[8] + '; strand=' + f[6])
            elif int(f[1]) <= pos1 - 1 < int(f[2]):
                hits.add('|'.join(f[3:6]) if len(f) > 3 else f'{f[0]}:{f[1]}-{f[2]}')
    return ('queried' if seen_contig else 'contig_absent_or_empty_resource'), '; '.join(sorted(hits))


def audit_site(site, spec=None, resources=None):
    spec, resources = spec or {}, resources or {}
    row = dict(site=site, build=spec.get('build', 'pending'),
               coordinate_convention=spec.get('coordinate_convention', 'pending'),
               strand=spec.get('strand', 'pending'), reference_base='pending',
               reference_check='pending', bam_support='pending',
               independent_molecules='pending', audit_class='computational_reliability')
    if not all(k in spec for k in ('chrom', 'pos1', 'build', 'strand')):
        row.update(gene_status='pending', homolog_status='pending')
        return row
    chrom, pos = spec['chrom'], int(spec['pos1'])
    if pos < 1 or spec['strand'] not in ('+', '-'):
        raise ValueError('Explicit positive 1-based coordinate and +/- strand required')
    row.update(chrom=chrom, pos1=pos)
    for name, kind in [('gene', 'gtf'), ('homolog', 'bed')]:
        if resources.get(name + '_build') != spec['build']:
            status, hits = 'pending_build_match', ''
        else:
            status, hits = overlap_resource(resources.get(name), chrom, pos, kind)
        row[name + '_status'], row[name + '_overlaps'] = status, hits
    if resources.get('fasta') and resources.get('fasta_build') == spec['build']:
        import pysam
        with pysam.FastaFile(str(resources['fasta'])) as fa:
            base = fa.fetch(chrom, pos - 1, pos).upper()
        row['reference_base'] = base
        row['reference_check'] = 'match' if base == ('A' if spec['strand'] == '+' else 'T') else 'MISMATCH'
    return row


def bam_counts(bam_path, spec, spots, profile, unit='read'):
    """Count once per query-name or (CB,UB) at a site; discard conflicting bases.

    Requires indexed coordinate-sorted BAM and verified genomic strand/base.
    NH=1 is required in unique profiles; absent NH is not evidence of uniqueness.
    UMI mode excludes reads without UB, and is a separate counting unit.
    Duplicate-flagged reads are excluded in read mode, retained for UMI consensus.
    Quality metrics describe passing alignments, not independent observations.
    """
    import pysam
    if unit not in ('read', 'umi'):
        raise ValueError('unit must be read or umi')
    ref, alt = ('A', 'G') if spec['strand'] == '+' else ('T', 'C')
    groups, metrics = {}, []
    excluded = dict(missing_CB=0, missing_UB=0, missing_NH=0)
    spots = pd.Index(spots)
    allowed = set(spots)
    pos0 = int(spec['pos1']) - 1
    with pysam.AlignmentFile(str(bam_path), 'rb') as bam:
        for r in bam.fetch(spec['chrom'], pos0, pos0 + 1):
            if r.is_unmapped or r.is_secondary or r.is_supplementary or r.is_qcfail:
                continue
            if unit == 'read' and r.is_duplicate:
                continue
            if r.mapping_quality == 255 or r.mapping_quality < profile['mapq']:
                continue
            if profile['unique']:
                if not r.has_tag('NH'):
                    excluded['missing_NH'] += 1
                    continue
                if r.get_tag('NH') != 1:
                    continue
            if not r.has_tag('CB'):
                excluded['missing_CB'] += 1
                continue
            cb = r.get_tag('CB')
            if cb not in allowed:
                continue
            if unit == 'umi' and not r.has_tag('UB'):
                excluded['missing_UB'] += 1
                continue
            for q, p in r.get_aligned_pairs(matches_only=True):
                if p != pos0:
                    continue
                if r.query_qualities is None or r.query_sequence is None:
                    break
                bq = r.query_qualities[q]
                end = min(q, r.query_length - 1 - q)
                if bq < profile['baseq'] or end < profile['end_distance']:
                    break
                base = r.query_sequence[q].upper()
                key = (cb, r.get_tag('UB') if unit == 'umi' else r.query_name)
                groups.setdefault(key, set()).add(base)
                metrics.append((base, bq, r.mapping_quality, end, r.is_reverse))
                break
    counts = pd.DataFrame(0, index=spots, columns=['edited', 'unedited'])
    conflicts = other = 0
    for (cb, _), bases in groups.items():
        if len(bases) != 1:
            conflicts += 1
            continue
        base = next(iter(bases))
        if base in (ref, alt):
            counts.loc[cb, 'edited' if base == alt else 'unedited'] += 1
        else:
            other += 1
    info = dict(unit=unit, conflict_units=conflicts, other_base_units=other, **excluded)
    for label, base in [('edited', alt), ('unedited', ref)]:
        values = np.array([m[1:] for m in metrics if m[0] == base], dtype=float)
        info[label + '_passing_alignments'] = len(values)
        for j, name in enumerate(['baseq', 'mapq', 'end_distance', 'reverse_fraction']):
            info[label + '_' + name] = float(values[:, j].mean()) if len(values) else np.nan
    return validate_counts(counts), info


def prepare_spots(counts, obs, composition, celltypes, group_key='ground_truth',
                  xy=('x_array', 'y_array'), wm_pattern=r'(?:^WM$|white)'):
    if obs.index.has_duplicates or composition.index.has_duplicates:
        raise ValueError('Duplicate metadata/composition spot IDs')
    d = validate_counts(counts).join(obs[[group_key, *xy]], how='left')
    d['wm'] = d[group_key].astype('string').str.contains(wm_pattern, case=False, regex=True).astype(float)
    p = composition.reindex(d.index)[list(celltypes)].apply(pd.to_numeric, errors='coerce')
    if (p < 0).any().any():
        raise ValueError('Negative composition entries')
    p = p.div(p.sum(axis=1, min_count=len(celltypes)).replace(0, np.nan), axis=0)
    d = d.join(p.add_prefix('ct:'))
    return d.rename(columns={xy[0]: 'x', xy[1]: 'y'})


def _design(d, cols):
    x = pd.DataFrame({'intercept': 1., 'wm': d.wm}, index=d.index)
    for c in cols:
        values = np.log1p(d.total) if c == 'log_coverage' else d[c]
        sd = values.std()
        if not np.isfinite(sd) or sd < 1e-10:
            raise ValueError('constant_covariate:' + c)
        x[c] = (values - values.mean()) / sd
    if np.linalg.matrix_rank(x) < x.shape[1]:
        raise ValueError('rank_deficient')
    condition = float(np.linalg.cond(x))
    if condition > 1000:
        raise ValueError('ill_conditioned')
    if len(cols):
        z = x.drop(columns='wm').to_numpy()
        resid = d.wm.to_numpy() - z @ np.linalg.lstsq(z, d.wm, rcond=None)[0]
        r2 = 1 - (resid @ resid) / ((d.wm - d.wm.mean()) ** 2).sum()
    else:
        r2 = 0.
    return x, condition, float(r2)


def _fit(y, x, target):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        fit = sm.GLM(y, x, family=sm.families.Binomial()).fit(maxiter=100)
    if not fit.converged or not np.isfinite(fit.params).all() or np.max(np.abs(fit.params)) > 25:
        raise ValueError('nonconvergence_or_separation')
    x0, x1 = target.copy(), target.copy()
    x0['wm'], x1['wm'] = 0., 1.
    rd = float(np.mean(fit.predict(x1) - fit.predict(x0)))
    return rd, float(fit.params['wm']), float(fit.pearson_chi2 / fit.df_resid)


def analyze_condition(d, min_cov=5, reference_celltype=None, block_size=8,
                      n_boot=499, seed=20260906, min_spots=None, min_blocks=10):
    """Grouped binomial quasi-likelihood mean + pairs spatial block bootstrap.

    Same complete-case spots for every model. K-1 normalized fractions;
    prespecified reference is required. Intervals use jointly successful draws.
    Blocks resample entire local neighborhoods; they are NOT biological replicates.
    """
    if n_boot < 20 or block_size <= 0:
        raise ValueError('Need n_boot >=20 and positive block_size')
    ct = [c for c in d if c.startswith('ct:')]
    reference = 'ct:' + str(reference_celltype)
    if reference not in ct:
        raise ValueError('Specify an existing reference_celltype')
    ct = [c for c in ct if c != reference]
    eligible = d.loc[d.wm.notna()]
    coverage = {}
    for wm, label in [(1, 'wm'), (0, 'nonwm')]:
        g = eligible.loc[eligible.wm == wm]
        measured = g.loc[g.total >= min_cov]
        coverage.update({label + '_all_spots': len(g), label + '_measurable_spots': len(measured),
                         label + '_total_counts': measured.total.sum(),
                         label + '_edited_counts': measured.edited.sum(),
                         label + '_median_coverage_all': g.total.median()})
    d = eligible.loc[eligible.total >= min_cov].dropna(
        subset=['x', 'y', *[c for c in d if c.startswith('ct:')]]).copy()
    if not np.isfinite(d[['x', 'y']].to_numpy()).all():
        raise ValueError('Nonfinite spatial coordinates')
    for wm, label in [(1, 'wm'), (0, 'nonwm')]:
        g = d.loc[d.wm == wm]
        coverage[label + '_model_spots'] = len(g)
        coverage[label + '_model_total_counts'] = g.total.sum()
    raw = []
    for wm in (1, 0):
        g = eligible.loc[(eligible.wm == wm) & (eligible.total >= min_cov)]
        raw.append(g.edited.sum() / g.total.sum() if g.total.sum() else np.nan)
    coverage['pooled_rate_difference_all_measurable'] = raw[0] - raw[1]
    models = {'unadjusted': [], 'composition': ct, 'composition_coverage': ct + ['log_coverage']}
    rows, designs = [], {}
    blocks = np.floor(d[['x', 'y']].to_numpy(float) / block_size).astype(int)
    block_ids = np.array([f'{a}:{b}' for a, b in blocks])
    unique = np.unique(block_ids)
    wm_blocks = len(np.unique(block_ids[d.wm.to_numpy() == 1]))
    nonwm_blocks = len(np.unique(block_ids[d.wm.to_numpy() == 0]))
    y = d[['edited', 'unedited']].to_numpy()
    for model, cols in models.items():
        row = dict(model=model, status='ok', rate_difference=np.nan, ci_low=np.nan,
                   ci_high=np.nan, n_spots=len(d), n_blocks=len(unique),
                   wm_blocks=wm_blocks, nonwm_blocks=nonwm_blocks, **coverage)
        try:
            if min((d.wm == 0).sum(), (d.wm == 1).sum()) == 0:
                raise ValueError('missing_region_observations')
            if min_spots is not None and min((d.wm == 0).sum(), (d.wm == 1).sum()) < min_spots:
                raise ValueError('insufficient_spots')
            if len(d) <= len(cols) + 2:
                raise ValueError('insufficient_residual_df')
            x, cond, r2 = _design(d, cols)
            rd, beta, dispersion = _fit(y, x, x)
            row.update(rate_difference=rd, log_odds_ratio=beta, dispersion=dispersion,
                       condition_number=cond, wm_explained_R2=r2,
                       composition_overlap_warning=bool(r2 > .95))
            designs[model] = x
            if len(unique) < min_blocks:
                row['status'] = 'point_only_insufficient_spatial_blocks'
            if min(wm_blocks, nonwm_blocks) < 2:
                row['status'] = 'point_only_single_region_block'
        except (ValueError, np.linalg.LinAlgError, Warning) as exc:
            row['status'] = str(exc)
        rows.append(row)
    draws = {m: [] for m in designs}
    if len(unique) >= min_blocks and designs:
        rng = np.random.default_rng(seed)
        members = {b: np.flatnonzero(block_ids == b) for b in unique}
        for _ in range(n_boot):
            ix = np.concatenate([members[b] for b in rng.choice(unique, len(unique), replace=True)])
            values = {}
            try:
                for model, x in designs.items():
                    xb = x.iloc[ix]
                    if np.linalg.matrix_rank(xb) < xb.shape[1]:
                        raise ValueError('bootstrap_rank')
                    values[model] = _fit(y[ix], xb, xb)[0]
            except (ValueError, np.linalg.LinAlgError, Warning):
                continue
            for model, value in values.items():
                draws[model].append(value)
    for row in rows:
        values = draws.get(row['model'], [])
        if row['model'] != 'unadjusted':
            row['adjustment_change'] = row['rate_difference'] - rows[0]['rate_difference']
        row['bootstrap_success'] = len(values)
        row['bootstrap_requested'] = n_boot
        if row['status'] == 'ok':
            if len(values) >= max(20, .8 * n_boot):
                row['ci_low'], row['ci_high'] = np.quantile(values, [.025, .975])
            else:
                row['status'] = 'point_only_bootstrap_unstable'
        if row['model'] != 'unadjusted' and len(values) and len(values) == len(draws.get('unadjusted', [])):
            change = np.asarray(values) - np.asarray(draws['unadjusted'])
            row['adjustment_change'] = row['rate_difference'] - rows[0]['rate_difference']
            if row['status'] == 'ok':
                row['change_ci_low'], row['change_ci_high'] = np.quantile(change, [.025, .975])
    return pd.DataFrame(rows)


def plot_results(results, output_dir):
    """Gap + NE annotation for unestimable effects; counts retained underneath."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sites = list(results.site.unique())
    fig, axes = plt.subplots(2, len(sites), figsize=(7 * len(sites), 7), squeeze=False)
    for j, site in enumerate(sites):
        s = results.loc[(results.site == site) & (results.model == 'unadjusted') & (results.block_role == 'primary')]
        available_conditions = s.loc[s.status != 'pending_input', 'condition'].unique()
        if len(available_conditions):
            s = s.loc[s.condition.isin(available_conditions)]
        conditions = list(s.condition.unique())
        for sample, sub in s.groupby('sample_id', sort=False):
            sub = sub.set_index('condition').reindex(conditions)
            xx = np.arange(len(conditions))
            line, = axes[0, j].plot(xx, sub.rate_difference, 'o-', label=sample)
            valid_ci = np.isfinite(sub.ci_low) & np.isfinite(sub.ci_high)
            axes[0, j].vlines(xx[valid_ci], sub.ci_low[valid_ci], sub.ci_high[valid_ci], color=line.get_color())
            for k, value in enumerate(sub.rate_difference):
                if not np.isfinite(value):
                    pooled = sub.iloc[k].get('pooled_rate_difference_all_measurable', np.nan)
                    if np.isfinite(pooled):
                        axes[0, j].plot(k, pooled, 'o', markerfacecolor='none', color=line.get_color())
                    else:
                        axes[0, j].text(k, .02, 'NE', transform=axes[0, j].get_xaxis_transform(), fontsize=8)
            axes[1, j].plot(xx, sub.wm_model_spots, 'o-', color=line.get_color(), label=f'{sample} WM')
            axes[1, j].plot(xx, sub.nonwm_model_spots, 's--', color=line.get_color(), label=f'{sample} non-WM')
        for ax in axes[:, j]:
            ax.set_xticks(range(len(conditions)))
            ax.set_xticklabels(conditions, rotation=65, ha='right')
            ax.legend(fontsize=7)
        axes[0, j].axhline(0, color='grey', lw=.8)
        axes[0, j].set_title(site)
        axes[0, j].set_ylabel('WM - non-WM editing probability')
        axes[1, j].set_ylabel('Effective spots (same complete-case set)')
    fig.suptitle('Hollow points: descriptive pooled difference only; inference unavailable.\n'
                 'Missing BAM/quality inputs are pending in the table, excluded from axes.', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(output_dir / 'core_site_robustness.pdf', bbox_inches='tight')
    fig.savefig(output_dir / 'core_site_robustness.png', dpi=160, bbox_inches='tight')
    fig2, axes2 = plt.subplots(1, len(sites), figsize=(7 * len(sites), 5), squeeze=False)
    for j, site in enumerate(sites):
        s = results.loc[(results.site == site) & (results.block_role == 'primary') & (results.condition == 'matrix_cov5')]
        ax = axes2[0, j]
        for k, (_, r) in enumerate(s.iterrows()):
            color = {'unadjusted': '#3579b1', 'composition': '#d77a28',
                     'composition_coverage': '#38946a'}.get(r.model, 'grey')
            if np.isfinite(r.rate_difference):
                ax.plot(r.rate_difference, k, 'o', color=color)
                if np.isfinite(r.ci_low) and np.isfinite(r.ci_high):
                    ax.hlines(k, r.ci_low, r.ci_high, color=color)
            else:
                ax.text(.05, k, 'NE: ' + str(r.status), fontsize=8,
                        transform=ax.get_yaxis_transform(), va='center')
        ax.set_ylim(-.6, max(len(s)-.4, .6))
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels([f'{r.sample_id}: {r.model}' +
                           (' *' if str(r.status).startswith('point_only') else '')
                           for _, r in s.iterrows()])
        if np.isfinite(s.rate_difference).any():
            ax.axvline(0, color='grey', lw=.8)
        else:
            ax.set_xticks([])
        ax.set_title(site)
        ax.set_xlabel('WM - non-WM editing probability (95% block bootstrap CI)')
    fig2.suptitle('* Point estimate only: spatial uncertainty not reliably estimable.\n'
                  'See effect table for fit diagnostics and bootstrap success counts.', fontsize=10)
    fig2.tight_layout(rect=(0, 0, 1, .90))
    fig2.savefig(output_dir / 'core_site_composition_effect.pdf', bbox_inches='tight')
    fig2.savefig(output_dir / 'core_site_composition_effect.png', dpi=160, bbox_inches='tight')
    return fig, fig2


def run_core_analysis(data_root, sample_ids, sites, parse_celltype, output_dir,
                      reference_celltype='Inhib', coverages=(5, 10, 20),
                      block_sizes=(8, 12), n_boot=499, site_specs=None,
                      resources=None, bam_paths=None, bam_builds=None, units=('read', 'umi')):
    """Reload unfiltered matrices; optional BAM counts never replace matrix silently."""
    import anndata as ad
    import json
    site_specs, resources = site_specs or {}, resources or {}
    bam_paths, bam_builds = bam_paths or {}, bam_builds or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, audits, support = [], [], []
    for sample in sample_ids:
        print('Core audit:', sample, flush=True)
        base = Path(data_root) / str(sample)
        a = ad.read_h5ad(base / 'adata_ai.h5ad')
        deconv = pd.read_csv(base / 'adata_obs_SPARROW.csv', index_col=0)
        labels = pd.read_csv(base / f'cluster_labels_{sample}.csv')
        labels.index = labels['key'].str.split('_', n=1).str[1]
        if labels.index.has_duplicates:
            raise ValueError('Duplicate label barcodes')
        positions = pd.read_csv(base / 'spatial/tissue_positions_list.csv', header=None,
                                names=['barcode', 'in_tissue', 'x_array', 'y_array', 'x_pixel', 'y_pixel']).set_index('barcode')
        obs = positions.join(labels[['ground_truth']], how='left').reindex(a.obs_names)
        # Use measured tissue spots only; unknown anatomical labels remain missing.
        obs.loc[obs.in_tissue != 1, 'ground_truth'] = np.nan
        obs['ground_truth'] = obs.ground_truth.replace(['nan', 'None', 'NA', 'null', ''], np.nan)
        rawcols = [c for c in deconv if c.startswith(('Ex_', 'Inhib', 'Astro', 'Oligo', 'OPC', 'Micro', 'Macro', 'Endo', 'Mix'))]
        if not rawcols:
            raise ValueError('No recognized SPARROW cell types; configure explicit mapping')
        p = pd.DataFrame(index=deconv.index)
        for c in rawcols:
            group = parse_celltype(c)
            if group not in p:
                p[group] = 0.
            p[group] = p[group] + pd.to_numeric(deconv[c], errors='raise')
        p.to_csv(output_dir / f'{sample}_composition_input.tsv', sep='\t')
        for site in sites:
            spec = site_specs.get(site, {})
            audit = audit_site(site, spec, resources)
            audit.update(sample_id=str(sample), matrix_present=site in a.var_names,
                         matrix_count_convention='G edited / A unedited; upstream strand handling must be audited',
                         matrix_count_unit='unknown_until_upstream_provenance_checked')
            sources = {}
            if site in a.var_names:
                sources['matrix'] = matrix_counts(a, site)
                for key, value in a.var.loc[site].items():
                    if str(key).startswith(('Gene.', 'Func.', 'ExonicFunc.')) or key in (
                        'chr', 'pos', 'Region', 'Position', 'Ref', 'Ed', 'Strand',
                        'Accession', 'db', 'type', 'dbsnp', 'repeat', 'geneID', 'geneType', 'geneTag'):
                        audit['upstream_' + str(key)] = value
            else:
                support.append(dict(sample_id=str(sample), site=site, source='matrix', status='site_absent'))
            bam = bam_paths.get(str(sample))
            if bam and audit['reference_check'] == 'match' and bam_builds.get(str(sample)) == spec.get('build'):
                for unit in units:
                    for name, profile in QUALITY_PROFILES.items():
                        counts, info = bam_counts(bam, spec, a.obs_names, profile, unit)
                        source = f'{unit}_{name}'
                        sources[source] = counts
                        support.append(dict(sample_id=str(sample), site=site, source=source, status='counted', **info))
                audit['bam_support'] = 'counted; inspect support and bias metrics'
                if 'umi' in units:
                    audit['independent_molecules'] = 'CB+UB consensus attempted; inspect missing_UB and support'
            else:
                support.append(dict(sample_id=str(sample), site=site, source='bam',
                                    status='pending_BAM_verified_reference_and_build'))
            audits.append(audit)
            # Missing BAM profiles remain visible as pending rows, not fabricated zero effects.
            pending_sources = [f'{u}_{q}' for u in units for q in QUALITY_PROFILES if f'{u}_{q}' not in sources]
            if 'matrix' not in sources:
                pending_sources.insert(0, 'matrix')
            for source in pending_sources:
                for cov in coverages:
                    for model in ('unadjusted', 'composition', 'composition_coverage'):
                        records.append(dict(sample_id=str(sample), site=site, condition=f'{source}_cov{cov}',
                                            source=source, min_cov=cov, block_size=block_sizes[0], block_role='primary',
                                            model=model, status='pending_input', rate_difference=np.nan,
                                            ci_low=np.nan, ci_high=np.nan, wm_model_spots=np.nan, nonwm_model_spots=np.nan))
            for source, counts in sources.items():
                d = prepare_spots(counts, obs, p, list(p.columns))
                for cov in coverages:
                    for ib, block in enumerate(block_sizes):
                        result = analyze_condition(d, cov, reference_celltype, block, n_boot)
                        result = result.assign(sample_id=str(sample), site=site, source=source,
                                               condition=f'{source}_cov{cov}', min_cov=cov, block_size=block,
                                               block_role='primary' if ib == 0 else 'sensitivity')
                        records.extend(result.to_dict('records'))
        del a
    result, audit_df = pd.DataFrame(records), pd.DataFrame(audits)
    result.to_csv(output_dir / 'core_site_effects.tsv', sep='\t', index=False)
    audit_df.to_csv(output_dir / 'core_site_audit.tsv', sep='\t', index=False)
    pd.DataFrame(support).to_csv(output_dir / 'core_site_read_support.tsv', sep='\t', index=False)
    supplementary = result.merge(audit_df, on=['sample_id', 'site'], how='left')
    support_df = pd.DataFrame(support).rename(columns={'status': 'read_support_status'})
    if not support_df.empty:
        supplementary = supplementary.merge(support_df, on=['sample_id', 'site', 'source'], how='left')
    supplementary.to_csv(
        output_dir / 'Supplementary_core_site_audit_sensitivity.tsv', sep='\t', index=False)
    config = dict(sample_ids=list(sample_ids), sites=list(sites), reference_celltype=reference_celltype,
                  coverages=list(coverages), block_sizes=list(block_sizes), n_boot=n_boot, seed=20260906,
                  site_specs=site_specs, resources=resources, bam_paths=bam_paths, bam_builds=bam_builds,
                  quality_profiles=QUALITY_PROFILES, units=list(units),
                  estimand='equal-spot standardized editing probability difference on common complete cases',
                  inference='within-section spatial block bootstrap; no across-donor inference')
    config['minimum_spots_per_region'] = None
    (output_dir / 'core_site_run_config.json').write_text(json.dumps(config, indent=2, default=str), encoding='utf-8')
    if not result.empty:
        plot_results(result, output_dir)
    return result, audit_df


def parse_dlpfc_celltype_group(col):
    """
    Convert fine DLPFC deconvolution cell type names to analysis-level groups.

    Key rule:
        Ex_10_L2_4 -> Ex_L2_4
        Ex_1_L5_6  -> Ex_L5_6
        Ex_8_L5_6  -> Ex_L5_6
        Ex_4_L_6   -> Ex_L6

    The number after Ex_ is treated as subtype index and removed.
    Layer information is retained.
    """
    name = str(col)

    # Excitatory neurons: keep layer information
    if name.startswith("Ex_"):
        parts = name.split("_")

        # Remove "Ex" and subtype number
        if len(parts) >= 3 and parts[1].isdigit():
            rest = parts[2:]
        else:
            rest = parts[1:]

        rest = [p for p in rest if p not in ["", " "]]
        rest = [p.replace("Layer", "L") for p in rest]

        # Ex_4_L_6 -> Ex_L6
        if len(rest) >= 2 and rest[0] == "L":
            layer = "L" + "_".join(rest[1:])
            return "Ex_" + layer

        # Ex_10_L2_4 -> Ex_L2_4
        if len(rest) >= 1 and rest[0].startswith("L"):
            return "Ex_" + "_".join(rest)

        return "Ex_unknown_layer"

    if name.startswith("Inhib"):
        return "Inhib"

    if name.startswith("Astros") or name.startswith("Astro"):
        return "Astros"

    if name.startswith("Oligos") or name.startswith("Oligo"):
        return "Oligos"

    if name.startswith("OPCs") or name.startswith("OPC"):
        return "OPCs"

    if (
        name.startswith("Micro/Macro")
        or name.startswith("Micro")
        or name.startswith("Macro")
    ):
        return "Micro/Macro"

    if name.startswith("Endo"):
        return "Endo"

    if name.startswith("Mix"):
        return "Mix"

    return name


def _figure_export_title(fig):
    titles = []
    if getattr(fig, "_suptitle", None) is not None:
        text = fig._suptitle.get_text().strip()
        if text:
            titles.append(text)
    for ax in fig.axes:
        text = ax.get_title().strip()
        if text and text not in titles:
            titles.append(text)
    return " - ".join(titles[:2]) or "figure"


def _filename_slug(text, max_length=90):
    text = str(text).replace("R?", "R2").replace("?", "2")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return (text[:max_length].rstrip("._-") or "figure")


def install_figure_pdf_export(namespace, output_dir):
    """Install notebook PDF export, preserving the namespace's current sample scope."""
    original = namespace.setdefault("_SPARROW_ORIGINAL_PLT_SHOW", plt.show)
    plt.show = original
    plt.close("all")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    namespace.update(PLOT_OUTPUT_DIR=destination, PLOT_RUN_ID=run_id,
                     PLOT_EXPORT_SCOPE="workflow", PLOT_EXPORT_COUNTER=0,
                     PLOT_EXPORT_MANIFEST=[])
    def show_and_export(*args, **kwargs):
        scope = _filename_slug(namespace.get("PLOT_EXPORT_SCOPE", namespace.get("sample_id", "workflow")))
        for number in plt.get_fignums():
            fig = plt.figure(number)
            if getattr(fig, "_sparrow_pdf_export_run", None) == run_id:
                continue
            namespace["PLOT_EXPORT_COUNTER"] += 1
            count = namespace["PLOT_EXPORT_COUNTER"]
            title = _figure_export_title(fig)
            filename = "{}_{}_{scope}_{title}.pdf".format(run_id, str(count).zfill(3),
                        scope=scope, title=_filename_slug(title))
            path = destination / filename
            fig.savefig(path, format="pdf", bbox_inches="tight", dpi=300,
                        metadata={"Title": title, "Subject": "DLPFC spatial A-to-I analysis"})
            fig._sparrow_pdf_export_run = run_id
            namespace["PLOT_EXPORT_MANIFEST"].append(dict(run_id=run_id, figure_index=count,
                scope=scope, title=title, pdf_path=str(path)))
            print("Saved editable PDF:", path)
        if namespace["PLOT_EXPORT_MANIFEST"]:
            pd.DataFrame(namespace["PLOT_EXPORT_MANIFEST"]).to_csv(
                destination / (run_id + "_figure_manifest.tsv"), sep="\t", index=False)
        return original(*args, **kwargs)
    plt.show = show_and_export
    print("Automatic editable-PDF export directory:", destination)
    print("Figure export run ID:", run_id)
