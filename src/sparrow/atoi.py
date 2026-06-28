# =============================================================================
# atoi_necessary.py
# Minimal A-to-I utilities used by DLPFC_AtoI_improved_workflow.ipynb.
# Generated from atoi.py by retaining notebook-called functions and dependencies.
# =============================================================================

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
from scipy.stats import kruskal, spearmanr, pearsonr, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from matplotlib.patches import Patch


_SPATIAL_IMAGE_CACHE = {}
_EXTERNAL_GENE_FILTER_CACHE = {}


# =============================================================================
# Basic helpers
# =============================================================================


__all__ = [
    'add_celltype_composition_modules',
    'add_gene_expression_to_obs',
    'add_obs_zmean_score',
    'analyze_adar_spatial_correlation',
    'analyze_score_vs_celltypes',
    'analyze_single_site_wm_enrichment',
    'analyze_wm_enrichment_for_sv_sites',
    'annotate_sv_atoi_sites',
    'collapse_deconvolution_to_clusters',
    'compute_global_atoi_ratio',
    'compute_multisite_atoi_score',
    'compute_obs_spatial_autocorrelation',
    'compute_sv_atoi_score',
    'detect_spatial_atoi_sites',
    'filter_atoi_sites',
    'fit_adjustment_model',
    'fit_joint_atoi_model',
    'infer_sv_site_mechanism_context',
    'plot_bivariate_colocalization',
    'plot_celltype_module_deconvolution_summary',
    'plot_cluster_adar_correlation_heatmap',
    'plot_cluster_level_obs_panel',
    'plot_cluster_level_obs_value',
    'plot_cluster_score_comparison',
    'plot_cross_sample_r2_decomposition',
    'plot_cross_sample_recurrent_site_wm_boxplot',
    'plot_cross_sample_sv_score_wm_enrichment',
    'plot_external_overlap_summary',
    'plot_global_atoi_two_r2_donuts',
    'plot_residual_progression',
    'plot_site_mechanism_context',
    'plot_sv_atoi_discovery_landscape',
    'plot_top_sv_site_patterns',
    'run_sv_external_mechanism_overlaps',
    'set_spatial_background',
    'summarize_top_sv_atoi_sites',
    'transfer_obs_metadata',
]

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

def _safe_moran_geary(
    ratio,
    coords,
    spatial_k=6,
    permutations=999,
    seed=None,
):
    try:
        if seed is not None:
            np.random.seed(seed)

        x = np.asarray(ratio, dtype=float)
        coords = np.asarray(coords)

        valid = np.isfinite(x)
        x_valid = x[valid]
        coords_valid = coords[valid]

        if len(x_valid) < spatial_k + 2:
            return np.nan, np.nan, np.nan, np.nan

        if np.nanstd(x_valid) == 0:
            return np.nan, np.nan, np.nan, np.nan

        knn = libpysal.weights.KNN.from_array(coords_valid, k=spatial_k)
        knn.transform = "r"

        moran = Moran(
            x_valid,
            knn,
            permutations=permutations,
        )

        geary = Geary(
            x_valid,
            knn,
            permutations=permutations,
        )

        moran_I = float(moran.I)

        if permutations and permutations > 0:
            moran_p = float(moran.p_sim)
        else:
            moran_p = float(moran.p_norm)

        geary_C = float(geary.C)

        if permutations and permutations > 0:
            geary_p = float(geary.p_sim)
        else:
            geary_p = float(geary.p_norm)

        return moran_I, moran_p, geary_C, geary_p

    except Exception:
        return np.nan, np.nan, np.nan, np.nan

def compute_obs_spatial_autocorrelation(
    adata_ai,
    value_keys: Sequence[str],
    level: str = "cluster",
    group_key: str = "ground_truth",
    cluster_agg: str = "mean",
    spatial_k: int = 2,
    permutations: int = 999,
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

    for key in value_keys:
        values = pd.to_numeric(value_df[key], errors="coerce").to_numpy(dtype=float)
        n_valid = int(np.isfinite(values).sum())
        k_use = max(1, min(int(spatial_k), max(n_valid - 1, 1)))

        moran_I, moran_p, geary_C, geary_p = _safe_moran_geary(
            values,
            coords_use,
            spatial_k=k_use,
            permutations=permutations,
        )

        rows.append(
            {
                "value_key": key,
                "level": level,
                "group_key": group_key if level == "cluster" else np.nan,
                "cluster_agg": cluster_agg if level == "cluster" else np.nan,
                "spatial_k": k_use,
                "permutations": permutations,
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
    seed=123,
    a_layer="A",
    g_layer="G",
    store_key="sv_atoi",
    verbose=True,
):
    """
    Detect spatially variable A-to-I sites.

    Parameters
    ----------
    adata_ai : AnnData
        AnnData object containing A/G count layers and spatial coordinates.

    group_key : str or None
        Observation column used for group-level Kruskal-Wallis test.

    min_cov : int
        Minimum A + G coverage required for a spot to be valid.

    min_valid_spots : int
        Minimum number of valid spots required for one A-to-I site.

    min_total_A : int
        Minimum total A count across valid spots.

    min_total_G : int
        Minimum total G count across valid spots.

    min_ratio_sd : float
        Minimum standard deviation of editing ratio across valid spots.

    spatial_k : int
        Number of nearest spatial neighbors for Moran's I / Geary's C.

    fdr_cutoff : float
        FDR cutoff for SV-A-to-I calling.

    p_cutoff : float or None
        Raw p-value cutoff. If None, uses fdr_cutoff.

    sv_call_by : {"fdr", "p", "either"}
        "fdr"    : moran_fdr < fdr_cutoff
        "p"      : moran_p < p_cutoff
        "either" : moran_fdr < fdr_cutoff OR moran_p < p_cutoff

    permutations : int
        Number of permutations used in Moran's I / Geary's C.

    seed : int or None
        Random seed for permutation-based p-values.
        If None, results are not forced to be reproducible.
        If an integer, each site uses seed + site_index.

    a_layer : str
        Layer name for A counts.

    g_layer : str
        Layer name for G counts.

    store_key : str
        Key used to store result table in adata_ai.uns.

    verbose : bool
        Whether to print progress and summary.

    Returns
    -------
    res : pandas.DataFrame
        Table of spatial A-to-I site statistics.
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
            site_seed = None if seed is None else int(seed) + int(j)

            moran_I, moran_p, geary_C, geary_p = _safe_moran_geary(
                ratio,
                coords,
                spatial_k=spatial_k,
                permutations=permutations,
                seed=site_seed,
            )

            if groups is not None:
                tmp = pd.DataFrame(
                    {
                        "ratio": ratio,
                        "group": groups.values,
                    },
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
                "seed": seed,
                "site_seed": None if seed is None else int(seed) + int(j),
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
    res["permutations"] = permutations

    res = res.sort_values(
        ["is_sv_atoi", "moran_fdr", "moran_p", "moran_I"],
        ascending=[False, True, True, False],
    ).reset_index(drop=True)

    adata_ai.uns[store_key] = res

    if verbose:
        print(f"QC-passing sites: {int(res['pass_qc'].sum())}")
        print(f"SV-A-to-I sites ({call_text}): {int(res['is_sv_atoi'].sum())}")

        if seed is not None:
            print(f"Permutation seed: {seed}，site-level seed = seed + site_index")
        else:
            print("Permutation seed: None，results may vary between runs")

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
    out = df.loc[:, keep].head(top_n).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, out.shape[0] + 1))

    if store_in is not None:
        store_in.uns[store_key] = out

    return out

def analyze_wm_enrichment_for_sv_sites(
    adata_ai,
    sv_atoi: Optional[pd.DataFrame] = None,
    score_key: str = "sv_atoi_score",
    group_key: str = "ground_truth",
    wm_pattern: str = r"(?:^WM$|white)",
    min_cov: int = 5,
    top_n: int = 20,
    require_sv: bool = True,
    level: str = "spot",
    store_key: str = "wm_sv_atoi_enrichment",
):
    """
    Test whether SV-A-to-I sites/program score are higher in WM than non-WM.

    For site-level editing, level="spot" tests spot editing ratios. level="cluster"
    aggregates ratios by group_key before testing, but may be underpowered when
    there are few DLPFC layers/clusters.
    """
    level = str(level).lower()

    if level not in ["spot", "cluster"]:
        raise ValueError("level must be 'spot' or 'cluster'.")

    if group_key not in adata_ai.obs.columns:
        raise ValueError(f"{group_key!r} not found in adata_ai.obs")

    labels = _valid_group_series(adata_ai, group_key=group_key)
    wm = labels.astype(str).str.contains(wm_pattern, case=False, regex=True, na=False)

    if not wm.any():
        raise ValueError(f"No WM labels matched wm_pattern={wm_pattern!r}.")

    rows = []

    if sv_atoi is not None and not sv_atoi.empty:
        top_sites = summarize_top_sv_atoi_sites(
            sv_atoi,
            top_n=top_n,
            require_sv=require_sv,
            fdr_cutoff=None,
        )
        sites = top_sites["site"].tolist() if not top_sites.empty else []
    else:
        sites = []

    features = []

    for site in sites:
        try:
            values = get_site_editing_ratio(adata_ai, site=site, min_cov=min_cov)
        except Exception:
            continue
        features.append((site, "site", values))

    if score_key in adata_ai.obs.columns:
        features.append((score_key, "program_score", adata_ai.obs[score_key].astype(float)))

    for feature, feature_type, values in features:
        values = pd.Series(values, index=adata_ai.obs_names, dtype=float)
        df = pd.DataFrame(
            {
                "value": values,
                "group": labels,
                "is_wm": wm.values,
            },
            index=adata_ai.obs_names,
        ).dropna()

        if level == "cluster":
            df = (
                df.groupby("group", observed=True)
                .agg(value=("value", "mean"), is_wm=("is_wm", "max"))
                .reset_index()
            )

        wm_vals = df.loc[df["is_wm"], "value"].astype(float).dropna().values
        nonwm_vals = df.loc[~df["is_wm"], "value"].astype(float).dropna().values

        if len(wm_vals) == 0 or len(nonwm_vals) == 0:
            stat = pval = np.nan
        else:
            try:
                test = mannwhitneyu(wm_vals, nonwm_vals, alternative="two-sided")
                stat = float(test.statistic)
                pval = float(test.pvalue)
            except Exception:
                stat = pval = np.nan

        mean_wm = float(np.nanmean(wm_vals)) if len(wm_vals) else np.nan
        mean_nonwm = float(np.nanmean(nonwm_vals)) if len(nonwm_vals) else np.nan
        diff = mean_wm - mean_nonwm

        pooled_sd = np.nan
        if len(wm_vals) > 1 and len(nonwm_vals) > 1:
            pooled_sd = np.sqrt(
                ((len(wm_vals) - 1) * np.nanvar(wm_vals, ddof=1)
                 + (len(nonwm_vals) - 1) * np.nanvar(nonwm_vals, ddof=1))
                / (len(wm_vals) + len(nonwm_vals) - 2)
            )

        cohen_d = float(diff / pooled_sd) if np.isfinite(pooled_sd) and pooled_sd > 0 else np.nan

        rows.append(
            {
                "feature": feature,
                "feature_type": feature_type,
                "level": level,
                "n_wm": int(len(wm_vals)),
                "n_nonwm": int(len(nonwm_vals)),
                "mean_wm": mean_wm,
                "mean_nonwm": mean_nonwm,
                "wm_minus_nonwm": float(diff) if np.isfinite(diff) else np.nan,
                "cohen_d": cohen_d,
                "mannwhitney_u": stat,
                "pval": pval,
            }
        )

    res = pd.DataFrame(rows)

    if not res.empty:
        res["fdr"] = _safe_multipletest(res["pval"])
        res = res.sort_values(
            ["feature_type", "fdr", "wm_minus_nonwm"],
            ascending=[True, True, False],
        ).reset_index(drop=True)

    adata_ai.uns[store_key] = res

    return res

def plot_cross_sample_sv_score_wm_enrichment(
    wm_enrichment: pd.DataFrame,
    sample_col: str = "sample_id",
    feature_type_col: str = "feature_type",
    effect_col: str = "wm_minus_nonwm",
    fdr_col: str = "pval",
    figsize=(6.2, 4.4),
    title: str = "SV-A-to-I score WM enrichment",
):
    """
    Cross-sample plot for SV-A-to-I program-score WM enrichment with samples
    on the x axis.

    This is intentionally restricted to the program score, not individual sites,
    so the main figure emphasizes the reproducibility of the WM-associated
    SV-A-to-I program.
    """
    if wm_enrichment is None or wm_enrichment.empty:
        raise ValueError("wm_enrichment is empty.")

    df = wm_enrichment.copy()

    if feature_type_col in df.columns:
        df = df[df[feature_type_col].astype(str).eq("program_score")].copy()

    if df.empty:
        raise ValueError("No program_score rows to plot.")

    if sample_col not in df.columns:
        df[sample_col] = "sample"

    df[effect_col] = pd.to_numeric(df[effect_col], errors="coerce")
    if fdr_col in df.columns:
        df[fdr_col] = pd.to_numeric(df[fdr_col], errors="coerce")

    df = df.reset_index(drop=True)
    df["direction"] = np.where(df[effect_col] >= 0, "WM-high", "non-WM-high")

    fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(
        data=df,
        x=sample_col,
        y=effect_col,
        hue="direction",
        dodge=False,
        palette={"WM-high": "#d95f02", "non-WM-high": "#7570b3"},
        ax=ax,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Sample")
    ax.set_ylabel("WM mean - non-WM mean")
    ax.set_title(title)
    ax.legend(frameon=False, title="")

    ymax = pd.to_numeric(df[effect_col], errors="coerce").max()
    ymin = pd.to_numeric(df[effect_col], errors="coerce").min()
    span = ymax - ymin if np.isfinite(ymax) and np.isfinite(ymin) and ymax != ymin else max(abs(ymax), 1.0)
    offset = 0.04 * span
    for xpos, (_, row) in enumerate(df.iterrows()):
        effect = row.get(effect_col, np.nan)
        q = row.get(fdr_col, np.nan)
        if not np.isfinite(effect):
            continue
        stat_name = "p" if str(fdr_col).lower() in {"p", "pval", "p_value"} else "FDR"
        label = f"{stat_name}={q:.2g}" if np.isfinite(q) else f"{stat_name}=NA"
        va = "bottom" if effect >= 0 else "top"
        ax.text(xpos, effect + (offset if effect >= 0 else -offset), label, va=va, ha="center", fontsize=8)

    if np.isfinite(ymax) and np.isfinite(ymin):
        ax.set_ylim(min(0, ymin - 0.10 * span), max(0, ymax + 0.18 * span))

    plt.tight_layout()

    return fig, ax, df

def analyze_single_site_wm_enrichment(
    adata_ai,
    site: str,
    group_key: str = "ground_truth",
    wm_pattern: str = r"(?:^WM$|white)",
    min_cov: int = 5,
):
    """
    WM vs non-WM summary for one editing site.
    """
    if group_key not in adata_ai.obs.columns:
        raise ValueError(f"{group_key!r} not found in adata_ai.obs")

    ratio = get_site_editing_ratio(adata_ai, site=site, min_cov=min_cov)
    labels = _valid_group_series(adata_ai, group_key=group_key)
    wm = labels.astype(str).str.contains(wm_pattern, case=False, regex=True, na=False)

    df = pd.DataFrame(
        {
            "editing_ratio": ratio,
            "group": labels,
            "is_wm": wm.values,
        },
        index=adata_ai.obs_names,
    ).dropna()

    wm_vals = df.loc[df["is_wm"], "editing_ratio"].astype(float).values
    nonwm_vals = df.loc[~df["is_wm"], "editing_ratio"].astype(float).values

    if len(wm_vals) and len(nonwm_vals):
        test = mannwhitneyu(wm_vals, nonwm_vals, alternative="two-sided")
        stat = float(test.statistic)
        pval = float(test.pvalue)
    else:
        stat = pval = np.nan

    mean_wm = float(np.nanmean(wm_vals)) if len(wm_vals) else np.nan
    mean_nonwm = float(np.nanmean(nonwm_vals)) if len(nonwm_vals) else np.nan

    summary = pd.DataFrame(
        [
            {
                "site": site,
                "n_wm": int(len(wm_vals)),
                "n_nonwm": int(len(nonwm_vals)),
                "mean_wm": mean_wm,
                "mean_nonwm": mean_nonwm,
                "wm_minus_nonwm": mean_wm - mean_nonwm if np.isfinite(mean_wm) and np.isfinite(mean_nonwm) else np.nan,
                "mannwhitney_u": stat,
                "pval": pval,
            }
        ]
    )

    return df, summary

def plot_cross_sample_recurrent_site_wm_boxplot(
    ratio_df: pd.DataFrame,
    summary_df: Optional[pd.DataFrame] = None,
    sample_col: str = "sample_id",
    value_col: str = "editing_ratio",
    region_col: str = "region",
    fdr_col: str = "pval",
    figsize=(8.5, 4.4),
    title: str = "Recurrent SV-A-to-I site editing in WM vs non-WM",
    robust_ylim: bool = True,
    whisker_iqr: float = 1.5,
    display_upper_quantile: float = 0.98,
    max_points_per_group: Optional[int] = 25,
    random_state: int = 0,
):
    """
    Plot WM/non-WM editing ratio for one recurrent site across samples.
    """
    if ratio_df is None or ratio_df.empty:
        raise ValueError("ratio_df is empty.")

    df = ratio_df.copy()

    if region_col not in df.columns and "is_wm" in df.columns:
        df[region_col] = np.where(df["is_wm"], "WM", "non-WM")

    required = {sample_col, value_col, region_col}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    plot_points = df.copy()
    y_lower = y_upper = np.nan

    if robust_ylim:
        whisker_bounds = []
        for _, sub in df.groupby([sample_col, region_col], observed=True):
            values = sub[value_col].dropna().astype(float)
            if values.empty:
                continue
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = max(float(values.min()), float(q1 - whisker_iqr * iqr))
            upper = min(float(values.max()), float(q3 + whisker_iqr * iqr))
            whisker_bounds.append((lower, upper))
        if whisker_bounds:
            y_lower = min(x[0] for x in whisker_bounds)
            y_upper = max(x[1] for x in whisker_bounds)
            if 0 < float(display_upper_quantile) < 1:
                quantile_cap = float(df[value_col].dropna().quantile(display_upper_quantile))
                if np.isfinite(quantile_cap):
                    y_upper = min(y_upper, quantile_cap)
            if np.isfinite(y_lower) and np.isfinite(y_upper) and y_upper > y_lower:
                plot_points = df[df[value_col].between(y_lower, y_upper, inclusive="both")].copy()

    if max_points_per_group is not None and int(max_points_per_group) > 0 and not plot_points.empty:
        plot_points = (
            plot_points.groupby([sample_col, region_col], group_keys=False, observed=True)
            .apply(lambda x: x.sample(n=min(len(x), int(max_points_per_group)), random_state=random_state))
        )

    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(
        data=df,
        x=sample_col,
        y=value_col,
        hue=region_col,
        hue_order=["non-WM", "WM"],
        palette={"non-WM": "#bdbdbd", "WM": "#d95f02"},
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=plot_points,
        x=sample_col,
        y=value_col,
        hue=region_col,
        hue_order=["non-WM", "WM"],
        dodge=True,
        palette={"non-WM": "#4d4d4d", "WM": "#8c2d04"},
        size=2.5,
        alpha=0.45,
        linewidth=0,
        ax=ax,
    )

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:2], labels[:2], frameon=False, title="")

    if summary_df is not None and not summary_df.empty:
        s = summary_df.copy()
        if sample_col in s.columns:
            if fdr_col not in s.columns and "pval" in s.columns:
                s[fdr_col] = _safe_multipletest(s["pval"])
            ymax = y_upper if robust_ylim and np.isfinite(y_upper) else pd.to_numeric(df[value_col], errors="coerce").max()
            ymin = y_lower if robust_ylim and np.isfinite(y_lower) else pd.to_numeric(df[value_col], errors="coerce").min()
            span = ymax - ymin if np.isfinite(ymax) and np.isfinite(ymin) and ymax > ymin else 0.1
            y_text = ymax + span * 0.06
            sample_order = list(dict.fromkeys(df[sample_col].astype(str)))
            for x, sample in enumerate(sample_order):
                row = s[s[sample_col].astype(str) == str(sample)]
                if row.empty:
                    continue
                q = row[fdr_col].iloc[0] if fdr_col in row.columns else np.nan
                effect = row["wm_minus_nonwm"].iloc[0] if "wm_minus_nonwm" in row.columns else np.nan
                stat_name = "p" if str(fdr_col).lower() in {"p", "pval", "p_value"} else "FDR"
                label = f"{stat_name}={q:.2g}" if np.isfinite(q) else f"{stat_name}=NA"
                if np.isfinite(effect):
                    label = f"{label}\nΔ={effect:.2g}"
                ax.text(x, y_text, label, ha="center", va="bottom", fontsize=8)

            if robust_ylim and np.isfinite(y_lower) and np.isfinite(y_upper) and y_upper > y_lower:
                lower_pad = 0.05 * span
                upper_pad = 0.20 * span
                ax.set_ylim(max(0, y_lower - lower_pad), y_upper + upper_pad)

    ax.set_xlabel("")
    ax.set_ylabel("editing ratio")
    ax.set_title(title)
    plt.tight_layout()

    df["shown_in_stripplot"] = df.index.isin(plot_points.index)
    return fig, ax, df

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

    sort_cols = [c for c in ["moran_fdr", "moran_p", "moran_I"] if c in df.columns]
    if sort_cols:
        ascending = [False if c == "moran_I" else True for c in sort_cols]
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

def plot_site_mechanism_context(
    mechanism_df: pd.DataFrame,
    sample_col: str = "sample_id",
    site_col: str = "site",
    top_n_sites: int = 30,
    figsize=(9.5, 6.0),
    title="SV-A-to-I mechanism context",
):
    if mechanism_df is None or mechanism_df.empty:
        raise ValueError("mechanism_df is empty.")

    df = mechanism_df.copy()
    if sample_col not in df.columns:
        df[sample_col] = "sample"
    if "mechanism_class" not in df.columns:
        df["mechanism_class"] = "unresolved"

    site_order = df[site_col].astype(str).drop_duplicates().head(top_n_sites).tolist()
    df = df[df[site_col].astype(str).isin(site_order)].copy()
    df[site_col] = pd.Categorical(df[site_col].astype(str), categories=site_order[::-1], ordered=True)

    palette = {
        "miRNA/3UTR regulation": "#66a61e",
        "repeat/ADAR substrate": "#d95f02",
        "known RNA-editing site": "#1b9e77",
        "RBP binding/regulation": "#7570b3",
        "multi-source evidence": "#e7298a",
        "unresolved": "#bdbdbd",
    }
    fig, ax = plt.subplots(figsize=figsize)
    for cls, sub in df.groupby("mechanism_class", observed=True):
        ax.scatter(
            sub[sample_col].astype(str),
            sub[site_col].astype(str),
            s=82,
            color=palette.get(str(cls), "#8c8c8c"),
            edgecolor="black",
            linewidth=0.4,
            alpha=0.9,
            label=str(cls),
        )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title)
    ax.grid(axis="x", color="#eeeeee", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    return fig, ax, df

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

    if "moran_fdr" in df.columns:
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

def plot_sv_atoi_discovery_landscape(
    sv_atoi: pd.DataFrame,
    fdr_cutoff=0.05,
    p_col: str = "moran_p",
    p_cutoff: float = 0.05,
    top_n_labels=8,
    figsize=(12, 4),
):
    df = sv_atoi.copy()

    if p_col not in df.columns:
        p_col = "moran_fdr"

    df[p_col] = pd.to_numeric(df[p_col], errors="coerce")
    df["_plot_p_value"] = df[p_col].clip(lower=1e-300)
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

    mask = df["moran_I"].notna() & df["_plot_p_value"].notna()

    axes[1].scatter(
        df.loc[mask, "moran_I"],
        df.loc[mask, "_plot_p_value"],
        s=sizes.loc[mask],
        c=colors.loc[mask],
        linewidths=0,
        alpha=0.75,
    )

    axes[1].axhline(p_cutoff, color="black", linestyle="--", linewidth=1)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Moran's I")
    axes[1].set_ylabel("Moran p value" if p_col == "moran_p" else p_col)
    axes[1].set_title("Spatial autocorrelation")
    axes[1].set_yscale("log")
    axes[1].invert_yaxis()

    label_df = (
        df.sort_values(
            ["is_sv_atoi", p_col, "moran_I"],
            ascending=[False, True, False],
        )
        .head(top_n_labels)
    )

    for _, row in label_df.iterrows():
        label = row["gene"] if isinstance(row.get("gene", ""), str) and row.get("gene", "") else row["site"]

        if np.isfinite(row["moran_I"]) and np.isfinite(row["_plot_p_value"]):
            axes[1].text(
                row["moran_I"],
                row["_plot_p_value"],
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
