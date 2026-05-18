"""Spatial A-to-I editing analysis utilities.

This module provides helpers for A-to-I site filtering, spatially variable
A-to-I discovery, deconvolution association analysis, ADAR-family analyses,
and publication-style visualization.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import sparse
from scipy.stats import kruskal, mannwhitneyu, norm, spearmanr
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import multipletests

def filter_atoi_sites(adata_ai, min_spot_cov=10, min_n_spots=10, max_site_ratio=1, copy=True, verbose=True):
    """Filter A-to-I sites by site-level editing ratio and the number of sufficiently covered spots.

Parameters
----------
adata_ai : object
    AnnData object containing A-to-I count layers and spot metadata.
min_spot_cov : object
    Spot-level coverage threshold used to count valid spots per site.
min_n_spots : object
    Minimum number of valid spots required to keep a site.
max_site_ratio : object
    Maximum allowed site-level G/(A+G) ratio.
copy : object
    Whether to return a copied AnnData object instead of a view.
verbose : object
    Whether to print progress and filtering summaries.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    if 'A' not in adata_ai.layers:
        raise ValueError("adata_ai.layers['A'] not found")
    if 'G' not in adata_ai.layers:
        raise ValueError("adata_ai.layers['G'] not found")
    A = adata_ai.layers['A']
    G = adata_ai.layers['G']
    if sparse.issparse(A):
        A = A.tocsr()
    if sparse.issparse(G):
        G = G.tocsr()
    cov = A + G
    site_A = np.asarray(A.sum(axis=0)).ravel()
    site_G = np.asarray(G.sum(axis=0)).ravel()
    site_cov = site_A + site_G
    site_ratio = np.full(site_G.shape, np.nan, dtype=float)
    mask_cov = site_cov > 0
    site_ratio[mask_cov] = site_G[mask_cov] / site_cov[mask_cov]
    adata_ai.var['total_A'] = site_A
    adata_ai.var['total_G'] = site_G
    adata_ai.var['total_cov'] = site_cov
    adata_ai.var['site_G_ratio'] = site_ratio
    n_spots_cov_gt = np.asarray((cov > min_spot_cov).sum(axis=0)).ravel()
    cov_col = f'n_spots_cov_gt{min_spot_cov}'
    adata_ai.var[cov_col] = n_spots_cov_gt
    keep_mask = (adata_ai.var['site_G_ratio'] < max_site_ratio) & ~adata_ai.var['site_G_ratio'].isna() & (adata_ai.var[cov_col] >= min_n_spots)
    if verbose:
        print(f'Before: {adata_ai.n_vars}')
        print(f'After : {int(keep_mask.sum())}')
        print(f'Removed: {int((~keep_mask).sum())}')
        print('')
        print('Filtering criteria:')
        print(f'  site_G_ratio < {max_site_ratio}')
        print(f'  site_G_ratio is not NaN')
        print(f'  {cov_col} >= {min_n_spots}')
    if copy:
        adata_filtered = adata_ai[:, keep_mask].copy()
    else:
        adata_filtered = adata_ai[:, keep_mask]
    return (adata_filtered, keep_mask)

def plot_spatial_value(adata, values, title, spot_size=18, cmap='magma', percentile_clip=(1, 99)):
    """Plot a numeric spot-level value on tissue coordinates.

Parameters
----------
adata : object
    AnnData object containing spatial coordinates and observations.
values : object
    Numeric values to plot or aggregate.
title : object
    Plot title.
spot_size : object
    Marker size for spatial plots.
cmap : object
    Matplotlib or seaborn colormap name/object.
percentile_clip : object
    Lower and upper percentiles used to clip plotted values.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    x = adata.obs['x_pixel'].values
    y = adata.obs['y_pixel'].values
    values = np.asarray(values, dtype=float)
    mask = ~np.isnan(values)
    vmin = np.nanpercentile(values[mask], percentile_clip[0])
    vmax = np.nanpercentile(values[mask], percentile_clip[1])
    plt.figure(figsize=(5, 5))
    plt.scatter(x[~mask], y[~mask], s=spot_size, c='lightgrey', alpha=0.15, linewidths=0)
    sc = plt.scatter(x[mask], y[mask], s=spot_size, c=values[mask], cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')
    plt.axis('off')
    plt.title(title)
    plt.colorbar(sc, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

def run_all_sites_deconv_quasibinomial(adata_ai, df_deconv, celltype_cols, reference_celltype='Mix', min_cov=10, min_valid_spots=30, min_total_A=20, min_total_G=20, min_ratio_sd=0.03, add_log_cov=True, standardize=True, max_abs_coef=20, verbose=True):
    """Fit quasi-binomial GLMs for all A-to-I sites against deconvolved cell-type proportions.

Parameters
----------
adata_ai : object
    AnnData object containing A-to-I count layers and spot metadata.
df_deconv : object
    DataFrame of deconvolved cell-type proportions indexed by spot barcode.
celltype_cols : object
    Cell-type proportion columns to use as predictors or annotations.
reference_celltype : object
    Reference cell type excluded from the regression design matrix.
min_cov : object
    Minimum A+G coverage required for a spot-site observation.
min_valid_spots : object
    Minimum number of complete or valid spots required for analysis.
min_total_A : object
    Minimum total A counts among valid spots for each site.
min_total_G : object
    Minimum total G counts among valid spots for each site.
min_ratio_sd : object
    Minimum standard deviation of editing ratio required for each site.
add_log_cov : object
    Whether to include log10 coverage as a covariate.
standardize : object
    Whether to z-score numeric predictors before fitting models.
max_abs_coef : object
    Maximum absolute logit coefficient used when reporting capped odds ratios.
verbose : object
    Whether to print progress and filtering summaries.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    common = adata_ai.obs_names.intersection(df_deconv.index)
    if verbose:
        print('adata_ai spots:', adata_ai.n_obs)
        print('df_deconv spots:', df_deconv.shape[0])
        print('common spots:', len(common))
    if len(common) == 0:
        raise ValueError('No common spots.')
    adata_use = adata_ai[common].copy()
    df_use = df_deconv.loc[common].copy()
    use_celltypes = [c for c in celltype_cols if c != reference_celltype]
    A = adata_use.layers['A']
    G = adata_use.layers['G']
    if sparse.issparse(A):
        A = A.tocsr()
    if sparse.issparse(G):
        G = G.tocsr()
    X_base = df_use[use_celltypes].astype(float).copy()
    keep_predictors = []
    for c in use_celltypes:
        if X_base[c].std() > 1e-08:
            keep_predictors.append(c)
    if verbose:
        dropped = sorted(set(use_celltypes) - set(keep_predictors))
        if dropped:
            print('Dropped near-zero predictors:', dropped)
    use_celltypes = keep_predictors
    X_base = X_base[use_celltypes]
    results = []
    for j in range(adata_use.n_vars):
        site = adata_use.var_names[j]
        gene = ''
        if 'Gene.refGene' in adata_use.var.columns:
            gene = adata_use.var.iloc[j]['Gene.refGene']
        a = np.asarray(A[:, j].todense()).ravel() if sparse.issparse(A) else np.asarray(A[:, j]).ravel()
        g = np.asarray(G[:, j].todense()).ravel() if sparse.issparse(G) else np.asarray(G[:, j]).ravel()
        cov = a + g
        valid = cov >= min_cov
        if valid.sum() < min_valid_spots:
            continue
        total_A = float(a[valid].sum())
        total_G = float(g[valid].sum())
        if total_A < min_total_A or total_G < min_total_G:
            continue
        raw_ratio = np.full_like(g, np.nan, dtype=float)
        raw_ratio[valid] = g[valid] / cov[valid]
        if np.nanstd(raw_ratio[valid]) < min_ratio_sd:
            continue
        df_model = X_base.copy()
        df_model['A'] = a
        df_model['G'] = g
        df_model['cov'] = cov
        df_model['ratio_smooth'] = (df_model['G'] + 0.5) / (df_model['cov'] + 1.0)
        df_model['ratio_raw'] = df_model['G'] / np.maximum(df_model['cov'], 1)
        if add_log_cov:
            df_model['log_cov'] = np.log10(df_model['cov'] + 1)
        df_model = df_model.loc[valid].replace([np.inf, -np.inf], np.nan).dropna()
        if df_model.shape[0] < min_valid_spots:
            continue
        x_cols = use_celltypes.copy()
        if add_log_cov:
            x_cols.append('log_cov')
        X = df_model[x_cols].astype(float).copy()
        if standardize:
            for c in x_cols:
                sd = X[c].std()
                if sd > 0:
                    X[c] = (X[c] - X[c].mean()) / sd
                else:
                    X[c] = 0.0
        X = sm.add_constant(X, has_constant='add')
        y = df_model['ratio_smooth'].astype(float)
        weights = df_model['cov'].astype(float)
        try:
            fit = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=weights).fit(maxiter=100, disp=0)
            pearson_chi2 = fit.pearson_chi2
            df_resid = max(fit.df_resid, 1)
            phi = max(1.0, pearson_chi2 / df_resid)
            params = fit.params
            bse = fit.bse * np.sqrt(phi)
            zvals = params / bse
            pvals = 2 * norm.sf(np.abs(zvals))
            condition_number = np.linalg.cond(X.values)
        except Exception as e:
            if verbose:
                print(f'[SKIP] {site}: {e}')
            continue
        for term in params.index:
            coef = float(params[term])
            se = float(bse[term])
            pval = float(pvals[params.index.get_loc(term)])
            if coef > max_abs_coef:
                odds_ratio = np.exp(max_abs_coef)
                flagged_extreme = True
            elif coef < -max_abs_coef:
                odds_ratio = np.exp(-max_abs_coef)
                flagged_extreme = True
            else:
                odds_ratio = float(np.exp(coef))
                flagged_extreme = False
            results.append({'site': site, 'gene': gene, 'term': term, 'coef_logit': coef, 'se_quasi': se, 'z_quasi': float(zvals[term]), 'odds_ratio_capped': odds_ratio, 'pval': max(pval, np.nextafter(0, 1)), 'n_valid_spots': int(df_model.shape[0]), 'total_A_valid': total_A, 'total_G_valid': total_G, 'total_cov_valid': float(df_model['cov'].sum()), 'mean_ratio_raw': float(df_model['ratio_raw'].mean()), 'mean_ratio_smooth': float(df_model['ratio_smooth'].mean()), 'sd_ratio_raw': float(df_model['ratio_raw'].std()), 'mean_cov': float(df_model['cov'].mean()), 'phi_overdispersion': float(phi), 'condition_number': float(condition_number), 'flag_extreme_coef': flagged_extreme, 'model_type': 'quasi_binomial_glm', 'reference_celltype': reference_celltype})
        if verbose and (j + 1) % 500 == 0:
            print(f'Processed {j + 1}/{adata_use.n_vars} sites')
    res = pd.DataFrame(results)
    if res.empty:
        return res
    res['padj'] = np.nan
    for term in res['term'].unique():
        idx = res['term'] == term
        res.loc[idx, 'padj'] = multipletests(res.loc[idx, 'pval'], method='fdr_bh')[1]
    res['signed_score'] = np.sign(res['coef_logit']) * -np.log10(res['padj'].clip(lower=1e-300))
    res = res.sort_values(['padj', 'pval']).reset_index(drop=True)
    return res

def _as_csr(x):
    """Convert sparse matrices to CSR format while leaving dense arrays unchanged.

Parameters
----------
x : object
    Matrix or array-like object.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    return x.tocsr() if sparse.issparse(x) else x

def _get_count_layers(adata_ai, a_layer='A', g_layer='G'):
    """Retrieve A and G count layers from an AnnData object.

Parameters
----------
adata_ai : object
    AnnData object containing A-to-I count layers and spot metadata.
a_layer : object
    Name of the layer containing A counts.
g_layer : object
    Name of the layer containing G counts.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    if a_layer not in adata_ai.layers:
        raise ValueError(f'adata_ai.layers[{a_layer!r}] not found')
    if g_layer not in adata_ai.layers:
        raise ValueError(f'adata_ai.layers[{g_layer!r}] not found')
    return (_as_csr(adata_ai.layers[a_layer]), _as_csr(adata_ai.layers[g_layer]))

def _dense_col(x, j):
    """Extract one matrix column as a one-dimensional dense NumPy array.

Parameters
----------
x : object
    Matrix or array-like object.
j : object
    Column index.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    if sparse.issparse(x):
        return np.asarray(x[:, j].toarray()).ravel()
    return np.asarray(x[:, j]).ravel()

def _coords_from_adata(adata):
    """Extract spatial coordinates from `.obsm["spatial"]` or common coordinate columns in `.obs`.

Parameters
----------
adata : object
    AnnData object containing spatial coordinates and observations.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    if 'spatial' in adata.obsm:
        return np.asarray(adata.obsm['spatial'], dtype=float)
    for cols in (('x_pixel', 'y_pixel'), ('array_col', 'array_row'), ('x', 'y')):
        if set(cols).issubset(adata.obs.columns):
            return adata.obs.loc[:, list(cols)].to_numpy(dtype=float)
    raise ValueError("No spatial coordinates found in adata.obsm['spatial'] or obs x/y columns.")

def _site_ratio_matrix(adata_ai, site_idx, min_cov=10, a_layer='A', g_layer='G'):
    """Build spot-by-site editing ratio and coverage matrices for selected sites.

Parameters
----------
adata_ai : object
    AnnData object containing A-to-I count layers and spot metadata.
site_idx : object
    Integer indices of A-to-I sites to extract.
min_cov : object
    Minimum A+G coverage required for a spot-site observation.
a_layer : object
    Name of the layer containing A counts.
g_layer : object
    Name of the layer containing G counts.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
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
    return (np.vstack(ratios).T, np.vstack(coverages).T)

def _safe_moran_geary(values, coords, spatial_k=6):
    """Compute Moran's I and Geary's C with graceful failure handling.

Parameters
----------
values : object
    Numeric values to plot or aggregate.
coords : object
    Function argument used by this helper.
spatial_k : object
    Number of nearest neighbors used for spatial autocorrelation statistics.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    mask = np.isfinite(values)
    if mask.sum() <= spatial_k + 2 or np.nanstd(values[mask]) == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    try:
        from libpysal.weights import KNN
        from esda.geary import Geary
        from esda.moran import Moran
        w = KNN.from_array(coords[mask], k=min(spatial_k, mask.sum() - 1))
        w.transform = 'R'
        moran = Moran(values[mask], w, permutations=999)
        geary = Geary(values[mask], w, permutations=999)
        return (float(moran.I), float(moran.p_sim), float(geary.C), float(geary.p_sim))
    except Exception as exc:
        warnings.warn(f'Moran/Geary failed; returning NaN spatial statistics: {exc}')
        return (np.nan, np.nan, np.nan, np.nan)

def detect_spatial_atoi_sites(adata_ai, group_adata=None, group_key=None, min_cov=10, min_valid_spots=30, min_total_A=30, min_total_G=30, min_ratio_sd=0.03, spatial_k=6, fdr_cutoff=0.05, a_layer='A', g_layer='G', store_key='sv_atoi_sites', verbose=True):
    """Detect spatially variable A-to-I sites using QC filters and spatial autocorrelation.

Parameters
----------
adata_ai : object
    AnnData object containing A-to-I count layers and spot metadata.
group_adata : object
    Optional AnnData object containing group labels aligned by barcode.
group_key : object
    Column name containing spatial group, layer, cluster, or domain labels.
min_cov : object
    Minimum A+G coverage required for a spot-site observation.
min_valid_spots : object
    Minimum number of complete or valid spots required for analysis.
min_total_A : object
    Minimum total A counts among valid spots for each site.
min_total_G : object
    Minimum total G counts among valid spots for each site.
min_ratio_sd : object
    Minimum standard deviation of editing ratio required for each site.
spatial_k : object
    Number of nearest neighbors used for spatial autocorrelation statistics.
fdr_cutoff : object
    FDR threshold used to call significant SV-A-to-I sites.
a_layer : object
    Name of the layer containing A counts.
g_layer : object
    Name of the layer containing G counts.
store_key : object
    Key used to save result tables or summaries in `adata_ai.uns`.
verbose : object
    Whether to print progress and filtering summaries.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    if group_adata is not None:
        common = adata_ai.obs_names.intersection(group_adata.obs_names)
        adata_use = adata_ai[common].copy()
        groups = group_adata.obs.loc[common, group_key] if group_key else None
    else:
        adata_use = adata_ai
        groups = adata_use.obs[group_key] if group_key else None
    coords = _coords_from_adata(adata_use)
    A, G = _get_count_layers(adata_use, a_layer=a_layer, g_layer=g_layer)
    results = []
    for j, site in enumerate(adata_use.var_names):
        a = _dense_col(A, j)
        g = _dense_col(G, j)
        cov = a + g
        valid = cov >= min_cov
        total_A = float(a[valid].sum())
        total_G = float(g[valid].sum())
        ratio = np.full(adata_use.n_obs, np.nan, dtype=float)
        ratio[valid] = g[valid] / cov[valid]
        n_valid = int(valid.sum())
        sd_ratio = float(np.nanstd(ratio))
        mean_ratio = float(np.nanmean(ratio)) if n_valid else np.nan
        pass_qc = n_valid >= min_valid_spots and total_A >= min_total_A and (total_G >= min_total_G) and (sd_ratio >= min_ratio_sd)
        moran_i = moran_p = geary_c = geary_p = np.nan
        kw_stat = kw_p = np.nan
        n_groups = 0
        if pass_qc:
            moran_i, moran_p, geary_c, geary_p = _safe_moran_geary(ratio, coords, spatial_k=spatial_k)
            if groups is not None:
                df = pd.DataFrame({'ratio': ratio, 'group': pd.Categorical(groups)}, index=adata_use.obs_names)
                vals = [x['ratio'].dropna().values for _, x in df.groupby('group', observed=True)]
                vals = [x for x in vals if len(x) >= 3]
                n_groups = len(vals)
                if n_groups >= 2:
                    try:
                        kw_stat, kw_p = kruskal(*vals)
                        kw_stat, kw_p = (float(kw_stat), float(kw_p))
                    except Exception:
                        kw_stat = kw_p = np.nan
        gene = adata_use.var.iloc[j].get('Gene.refGene', '')
        results.append({'site': site, 'gene': gene, 'n_valid_spots': n_valid, 'total_A_valid': total_A, 'total_G_valid': total_G, 'total_cov_valid': total_A + total_G, 'mean_ratio': mean_ratio, 'sd_ratio': sd_ratio, 'pass_qc': bool(pass_qc), 'moran_I': moran_i, 'moran_p': moran_p, 'geary_C': geary_c, 'geary_p': geary_p, 'group_key': group_key, 'group_n': n_groups, 'group_kw_stat': kw_stat, 'group_kw_p': kw_p})
        if verbose and (j + 1) % 500 == 0:
            print(f'Processed {j + 1}/{adata_use.n_vars} A-to-I sites')
    res = pd.DataFrame(results)
    res['moran_fdr'] = np.nan
    res['group_kw_fdr'] = np.nan
    qc = res['pass_qc'].values
    if qc.any():
        p = res.loc[qc, 'moran_p'].fillna(1.0).values
        res.loc[qc, 'moran_fdr'] = multipletests(p, method='fdr_bh')[1]
        p_group = res.loc[qc, 'group_kw_p'].fillna(1.0).values
        res.loc[qc, 'group_kw_fdr'] = multipletests(p_group, method='fdr_bh')[1]
    res['is_sv_atoi'] = res['pass_qc'] & (res['moran_fdr'] < fdr_cutoff) & (res['moran_I'] > 0)
    res = res.sort_values(['is_sv_atoi', 'moran_fdr', 'moran_I'], ascending=[False, True, False])
    adata_ai.uns[store_key] = res
    if verbose:
        print(f"QC-passing sites: {int(res['pass_qc'].sum())}")
        print(f"SV-A-to-I sites (Moran FDR < {fdr_cutoff}): {int(res['is_sv_atoi'].sum())}")
    return res

def compute_spatial_atoi_score(adata_ai, sv_sites, score_name='sv_atoi_score', min_cov=10, top_n=None, weight_col='moran_I', a_layer='A', g_layer='G', zscore_sites=True):
    """Compute a spot-level multi-site A-to-I score from SV-A-to-I sites.

Parameters
----------
adata_ai : object
    AnnData object containing A-to-I count layers and spot metadata.
sv_sites : object
    SV-A-to-I site table or iterable of site names.
score_name : object
    Name of the output spot-level score column in `adata_ai.obs`.
min_cov : object
    Minimum A+G coverage required for a spot-site observation.
top_n : object
    Maximum number of top-ranked sites or entries to use.
weight_col : object
    Column used as site weights when computing a multi-site score.
a_layer : object
    Name of the layer containing A counts.
g_layer : object
    Name of the layer containing G counts.
zscore_sites : object
    Whether to z-score each site before aggregating into a score.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    if isinstance(sv_sites, pd.DataFrame):
        df_sites = sv_sites.copy()
        if 'is_sv_atoi' in df_sites.columns and df_sites['is_sv_atoi'].any():
            df_sites = df_sites[df_sites['is_sv_atoi']].copy()
        if top_n is not None:
            df_sites = df_sites.head(top_n)
        sites = [s for s in df_sites['site'].astype(str) if s in adata_ai.var_names]
        weights = df_sites.set_index('site').reindex(sites)[weight_col].astype(float).fillna(1.0).values if weight_col in df_sites else None
    else:
        sites = [s for s in list(sv_sites) if s in adata_ai.var_names]
        if top_n is not None:
            sites = sites[:top_n]
        weights = None
    if not sites:
        raise ValueError('No SV-A-to-I sites were found in adata_ai.var_names.')
    site_idx = [adata_ai.var_names.get_loc(s) for s in sites]
    ratio_mat, cov_mat = _site_ratio_matrix(adata_ai, site_idx, min_cov=min_cov, a_layer=a_layer, g_layer=g_layer)
    score_mat = ratio_mat.copy()
    if zscore_sites:
        mu = np.nanmean(score_mat, axis=0)
        sd = np.nanstd(score_mat, axis=0)
        sd[sd == 0] = np.nan
        score_mat = (score_mat - mu) / sd
    if weights is None:
        score = np.nanmean(score_mat, axis=1)
    else:
        weights = np.asarray(weights, dtype=float)
        weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
        valid = np.isfinite(score_mat)
        numerator = np.nansum(np.where(valid, score_mat * weights, np.nan), axis=1)
        denominator = np.sum(valid * weights, axis=1)
        score = numerator / denominator
        score[denominator == 0] = np.nan
    adata_ai.obs[score_name] = score
    adata_ai.obs[f'{score_name}_n_sites'] = np.sum(np.isfinite(ratio_mat), axis=1)
    adata_ai.obs[f'{score_name}_mean_cov'] = np.nanmean(np.where(cov_mat >= min_cov, cov_mat, np.nan), axis=1)
    adata_ai.uns[f'{score_name}_sites'] = sites
    return pd.Series(score, index=adata_ai.obs_names, name=score_name)

def analyze_atoi_deconv_association(adata_ai, df_deconv=None, score_key='sv_atoi_score', celltype_cols=None, covariates=None, group_key=None, min_complete=30, store_key='atoi_deconv_association'):
    """Associate an A-to-I score with deconvolved cell-type proportions.

Parameters
----------
adata_ai : object
    AnnData object containing A-to-I count layers and spot metadata.
df_deconv : object
    DataFrame of deconvolved cell-type proportions indexed by spot barcode.
score_key : object
    Name of the A-to-I score column in `adata_ai.obs`.
celltype_cols : object
    Cell-type proportion columns to use as predictors or annotations.
covariates : object
    Additional covariate columns to include in regression models.
group_key : object
    Column name containing spatial group, layer, cluster, or domain labels.
min_complete : object
    Function argument used by this helper.
store_key : object
    Key used to save result tables or summaries in `adata_ai.uns`.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    covariates = [] if covariates is None else list(covariates)
    if df_deconv is None:
        df = adata_ai.obs.copy()
    else:
        common = adata_ai.obs_names.intersection(df_deconv.index)
        df = adata_ai.obs.loc[common, [score_key]].join(df_deconv.loc[common], how='left')
        if group_key is not None and group_key in adata_ai.obs.columns:
            df[group_key] = adata_ai.obs.loc[common, group_key]
    if score_key not in df.columns:
        raise ValueError(f'{score_key!r} not found in adata_ai.obs or joined deconvolution table.')
    if celltype_cols is None:
        meta = {'in_tissue', 'x_array', 'y_array', 'x_pixel', 'y_pixel', 'celltype_sum', score_key}
        celltype_cols = [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    for ct in celltype_cols:
        cols = [score_key, ct] + [c for c in covariates if c in df.columns]
        sub = df.loc[:, cols].dropna()
        if sub.shape[0] < min_complete or sub[ct].std() == 0 or sub[score_key].std() == 0:
            continue
        rho, p_spear = spearmanr(sub[ct], sub[score_key])
        X = sub[[ct] + [c for c in covariates if c in sub.columns]].astype(float).copy()
        for c in X.columns:
            sd = X[c].std()
            if sd > 0:
                X[c] = (X[c] - X[c].mean()) / sd
        y = sub[score_key].astype(float)
        y = (y - y.mean()) / y.std()
        fit = sm.OLS(y, sm.add_constant(X, has_constant='add')).fit()
        rows.append({'celltype': ct, 'n_spots': int(sub.shape[0]), 'spearman_rho': float(rho), 'spearman_p': float(p_spear), 'ols_beta': float(fit.params[ct]), 'ols_p': float(fit.pvalues[ct]), 'ols_r2': float(fit.rsquared)})
    res = pd.DataFrame(rows)
    if not res.empty:
        res['spearman_fdr'] = multipletests(res['spearman_p'], method='fdr_bh')[1]
        res['ols_fdr'] = multipletests(res['ols_p'], method='fdr_bh')[1]
        res = res.sort_values(['ols_fdr', 'spearman_fdr']).reset_index(drop=True)
    if group_key is not None and group_key in df.columns:
        adata_ai.uns[f'{store_key}_group_summary'] = df.groupby(group_key, observed=True)[score_key].agg(['count', 'mean', 'median', 'std']).sort_values('mean', ascending=False)
    adata_ai.uns[store_key] = res
    return res

def compute_atoi_residual_after_deconv(adata_ai, df_deconv=None, score_key='sv_atoi_score', celltype_cols=None, covariates=None, residual_key='sv_atoi_residual', spatial_k=6, store_key='atoi_residual_model'):
    """Regress an A-to-I score on cell-type proportions and compute residual spatial signal.

Parameters
----------
adata_ai : object
    AnnData object containing A-to-I count layers and spot metadata.
df_deconv : object
    DataFrame of deconvolved cell-type proportions indexed by spot barcode.
score_key : object
    Name of the A-to-I score column in `adata_ai.obs`.
celltype_cols : object
    Cell-type proportion columns to use as predictors or annotations.
covariates : object
    Additional covariate columns to include in regression models.
residual_key : object
    Name of the residual column written to `adata_ai.obs`.
spatial_k : object
    Number of nearest neighbors used for spatial autocorrelation statistics.
store_key : object
    Key used to save result tables or summaries in `adata_ai.uns`.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    covariates = [] if covariates is None else list(covariates)
    if df_deconv is None:
        df = adata_ai.obs.copy()
    else:
        common = adata_ai.obs_names.intersection(df_deconv.index)
        df = adata_ai.obs.loc[common, [score_key]].join(df_deconv.loc[common], how='left')
    if celltype_cols is None:
        meta = {'in_tissue', 'x_array', 'y_array', 'x_pixel', 'y_pixel', 'celltype_sum', score_key}
        celltype_cols = [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]
    model_cols = [c for c in list(celltype_cols) + covariates if c in df.columns]
    sub = df[[score_key] + model_cols].dropna().copy()
    if sub.shape[0] < max(30, len(model_cols) + 5):
        raise ValueError('Not enough complete spots for residual analysis.')
    X = sub[model_cols].astype(float).copy()
    for c in X.columns:
        sd = X[c].std()
        if sd > 0:
            X[c] = (X[c] - X[c].mean()) / sd
    y = sub[score_key].astype(float)
    fit = sm.OLS(y, sm.add_constant(X, has_constant='add')).fit()
    residual = pd.Series(np.nan, index=adata_ai.obs_names, name=residual_key)
    residual.loc[sub.index] = fit.resid
    adata_ai.obs[residual_key] = residual
    coords = _coords_from_adata(adata_ai)
    moran_i, moran_p, geary_c, geary_p = _safe_moran_geary(residual.values, coords, spatial_k=spatial_k)
    summary = {'n_spots': int(sub.shape[0]), 'model_r2': float(fit.rsquared), 'model_adj_r2': float(fit.rsquared_adj), 'moran_I_residual': moran_i, 'moran_p_residual': moran_p, 'geary_C_residual': geary_c, 'geary_p_residual': geary_p, 'celltype_cols': list(celltype_cols), 'covariates': covariates}
    adata_ai.uns[store_key] = summary
    adata_ai.uns[f'{store_key}_params'] = fit.params.to_frame('coef').join(fit.pvalues.to_frame('pval'))
    return (residual, fit, summary)

def summarize_atoi_by_group(adata_ai, score_key='sv_atoi_score', group_key='ground_truth'):
    """Summarize an A-to-I score across spatial groups.

Parameters
----------
adata_ai : object
    AnnData object containing A-to-I count layers and spot metadata.
score_key : object
    Name of the A-to-I score column in `adata_ai.obs`.
group_key : object
    Column name containing spatial group, layer, cluster, or domain labels.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    if group_key not in adata_ai.obs.columns:
        raise ValueError(f'{group_key!r} not found in adata_ai.obs')
    return adata_ai.obs[[score_key, group_key]].dropna().groupby(group_key, observed=True)[score_key].agg(['count', 'mean', 'median', 'std']).sort_values('mean', ascending=False)

def _spatial_axes(ax, adata, values, title, cmap='magma', spot_size=18, clip=(1, 99), center=None):
    """Draw a reusable spatial scatter plot on a provided axis.

Parameters
----------
ax : object
    Matplotlib Axes object.
adata : object
    AnnData object containing spatial coordinates and observations.
values : object
    Numeric values to plot or aggregate.
title : object
    Plot title.
cmap : object
    Matplotlib or seaborn colormap name/object.
spot_size : object
    Marker size for spatial plots.
clip : object
    Lower and upper percentiles used to clip plotted values.
center : object
    Optional center value for symmetric color scaling.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    coords = _coords_from_adata(adata)
    x, y = (coords[:, 0], coords[:, 1])
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    ax.scatter(x[~mask], y[~mask], s=spot_size, c='#d9d9d9', alpha=0.22, linewidths=0)
    if mask.sum() == 0:
        ax.set_title(title)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.axis('off')
        return None
    vmin, vmax = np.nanpercentile(values[mask], clip)
    if center is not None:
        lim = max(abs(vmin - center), abs(vmax - center))
        vmin, vmax = (center - lim, center + lim)
    sc = ax.scatter(x[mask], y[mask], s=spot_size, c=values[mask], cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
    ax.set_title(title, fontsize=11, pad=8)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.axis('off')
    return sc

def _cluster_level_site_values(adata_ai, site, group_labels, min_cov=10, a_layer='A', g_layer='G'):
    """Aggregate one site's A/G counts by group and map group-level ratios back to spots.

Parameters
----------
adata_ai : object
    AnnData object containing A-to-I count layers and spot metadata.
site : object
    A-to-I site identifier present in `adata_ai.var_names`.
group_labels : object
    Group labels aligned to `adata_ai.obs_names`.
min_cov : object
    Minimum A+G coverage required for a spot-site observation.
a_layer : object
    Name of the layer containing A counts.
g_layer : object
    Name of the layer containing G counts.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    A, G = _get_count_layers(adata_ai, a_layer=a_layer, g_layer=g_layer)
    j = adata_ai.var_names.get_loc(site)
    a = _dense_col(A, j)
    g = _dense_col(G, j)
    cov = a + g
    df = pd.DataFrame({'group': group_labels.astype(str), 'A': a, 'G': g, 'cov': cov}, index=adata_ai.obs_names)
    agg = df.groupby('group', observed=True)[['A', 'G']].sum()
    agg['cov'] = agg['A'] + agg['G']
    agg['ratio'] = np.nan
    valid = agg['cov'] >= min_cov
    agg.loc[valid, 'ratio'] = agg.loc[valid, 'G'] / agg.loc[valid, 'cov']
    return (df['group'].map(agg['ratio']).to_numpy(dtype=float), agg)

def plot_sv_atoi_discovery_landscape(sv_atoi, fdr_cutoff=0.05, top_n_labels=8, figsize=(12, 4)):
    """Create an overview figure for SV-A-to-I discovery results.

Parameters
----------
sv_atoi : object
    DataFrame returned by `detect_spatial_atoi_sites()`.
fdr_cutoff : object
    FDR threshold used to call significant SV-A-to-I sites.
top_n_labels : object
    Number of top sites to annotate on the plot.
figsize : object
    Matplotlib figure size.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    df = sv_atoi.copy()
    df['neglog10_fdr'] = -np.log10(df['moran_fdr'].clip(lower=1e-300))
    df['status'] = np.where(df.get('is_sv_atoi', False), 'SV-A-to-I', 'not SV')
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    ax = axes[0]
    counts = pd.Series({'all sites': len(df), 'QC pass': int(df['pass_qc'].sum()) if 'pass_qc' in df else len(df), 'SV-A-to-I': int(df.get('is_sv_atoi', pd.Series(False, index=df.index)).sum())})
    sns.barplot(x=counts.index, y=counts.values, ax=ax, palette=['#8da0cb', '#66c2a5', '#fc8d62'])
    ax.set_ylabel('site count')
    ax.set_xlabel('')
    ax.set_title('Discovery funnel')
    ax.tick_params(axis='x', rotation=25)
    ax = axes[1]
    colors = df['status'].map({'not SV': '#bdbdbd', 'SV-A-to-I': '#d95f02'}).fillna('#bdbdbd')
    sizes = np.clip(df.get('n_valid_spots', pd.Series(20, index=df.index)).astype(float) / 3, 12, 90)
    mask = df['moran_I'].notna() & df['neglog10_fdr'].notna()
    ax.scatter(df.loc[mask, 'moran_I'], df.loc[mask, 'neglog10_fdr'], s=sizes.loc[mask], c=colors.loc[mask], linewidths=0, alpha=0.75)
    ax.axhline(-np.log10(fdr_cutoff), color='black', linestyle='--', linewidth=1)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel("Moran's I")
    ax.set_ylabel('-log10(FDR)')
    ax.set_title('Spatial autocorrelation')
    ax.legend(handles=[Line2D([0], [0], marker='o', color='w', label='not SV', markerfacecolor='#bdbdbd', markersize=6), Line2D([0], [0], marker='o', color='w', label='SV-A-to-I', markerfacecolor='#d95f02', markersize=6)], frameon=False, fontsize=8)
    label_df = df.sort_values(['is_sv_atoi', 'moran_fdr', 'moran_I'], ascending=[False, True, False]).head(top_n_labels)
    for _, row in label_df.iterrows():
        label = row['gene'] if isinstance(row.get('gene', ''), str) and row.get('gene', '') else row['site']
        ax.text(row['moran_I'], row['neglog10_fdr'], str(label)[:14], fontsize=7)
    ax = axes[2]
    colors2 = df['status'].map({'not SV': '#bdbdbd', 'SV-A-to-I': '#7570b3'}).fillna('#bdbdbd')
    mask = df['n_valid_spots'].notna() & df['sd_ratio'].notna()
    ax.scatter(df.loc[mask, 'n_valid_spots'], df.loc[mask, 'sd_ratio'], c=colors2.loc[mask], s=28, linewidths=0, alpha=0.75)
    ax.set_xlabel('valid spots')
    ax.set_ylabel('editing ratio SD')
    ax.set_title('Coverage and variability')
    ax.legend_.remove() if ax.legend_ else None
    plt.tight_layout()
    return (fig, axes)

def plot_cluster_level_residual_atoi(adata_ai, residual_key='sv_atoi_residual', group_key='ground_truth', adata_group=None, spot_size=18, cmap='coolwarm', figsize=(10, 4.2)):
    """Compare spot-level and cluster-level residual A-to-I signals.

Parameters
----------
adata_ai : object
    AnnData object containing A-to-I count layers and spot metadata.
residual_key : object
    Name of the residual column written to `adata_ai.obs`.
group_key : object
    Column name containing spatial group, layer, cluster, or domain labels.
adata_group : object
    Optional AnnData object containing group labels aligned by barcode.
spot_size : object
    Marker size for spatial plots.
cmap : object
    Matplotlib or seaborn colormap name/object.
figsize : object
    Matplotlib figure size.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    if residual_key not in adata_ai.obs:
        raise ValueError(f'{residual_key!r} not found in adata_ai.obs')
    if adata_group is not None:
        common = adata_ai.obs_names.intersection(adata_group.obs_names)
        adata_use = adata_ai[common].copy()
        labels = adata_group.obs.loc[common, group_key].astype(str)
    else:
        adata_use = adata_ai
        if group_key not in adata_use.obs:
            raise ValueError(f'{group_key!r} not found in adata_ai.obs')
        labels = adata_use.obs[group_key].astype(str)
    residual = adata_use.obs[residual_key].astype(float)
    group_mean = residual.groupby(labels, observed=True).mean()
    cluster_residual = labels.map(group_mean).to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=figsize, gridspec_kw={'width_ratios': [1, 1, 0.9]})
    sc0 = _spatial_axes(axes[0], adata_use, residual.values, 'Spot-level residual A-to-I', cmap=cmap, spot_size=spot_size, center=0)
    sc1 = _spatial_axes(axes[1], adata_use, cluster_residual, f'Cluster-level residual\\n{group_key}', cmap=cmap, spot_size=spot_size, center=0)
    if sc0 is not None:
        fig.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.02)
    if sc1 is not None:
        fig.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.02)
    summary = pd.DataFrame({'group': labels.values, 'residual': residual.values}).dropna().groupby('group', observed=True)['residual'].agg(['count', 'mean', 'median', 'std']).sort_values('mean')
    sns.barplot(data=summary.reset_index(), y='group', x='mean', ax=axes[2], color='#8da0cb', edgecolor='black', linewidth=0.4)
    axes[2].axvline(0, color='black', linewidth=1)
    axes[2].set_xlabel('mean residual')
    axes[2].set_ylabel('')
    axes[2].set_title('Residual by group', fontsize=11, pad=8)
    plt.tight_layout()
    return (fig, axes, summary)

def plot_spatial_atoi_overview(adata_ai, score_key='sv_atoi_score', residual_key='sv_atoi_residual', group_key='ground_truth', celltype_values=None, celltype_name=None, spot_size=18, figsize=None):
    """Create a multi-panel spatial overview of A-to-I score, residuals, cell types, and groups.

Parameters
----------
adata_ai : object
    AnnData object containing A-to-I count layers and spot metadata.
score_key : object
    Name of the A-to-I score column in `adata_ai.obs`.
residual_key : object
    Name of the residual column written to `adata_ai.obs`.
group_key : object
    Column name containing spatial group, layer, cluster, or domain labels.
celltype_values : object
    Numeric cell-type proportion values aligned to `adata_ai.obs_names`.
celltype_name : object
    Name of the cell type shown in the plot title.
spot_size : object
    Marker size for spatial plots.
figsize : object
    Matplotlib figure size.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    panels = [(score_key, adata_ai.obs[score_key].values, 'SV-A-to-I score', 'magma', None)]
    if residual_key in adata_ai.obs:
        panels.append((residual_key, adata_ai.obs[residual_key].values, 'Residual A-to-I', 'coolwarm', 0))
    if celltype_values is not None:
        panels.append(('celltype', np.asarray(celltype_values, dtype=float), f"{celltype_name or 'Cell type'} proportion", 'viridis', None))
    if group_key in adata_ai.obs:
        groups = pd.Categorical(adata_ai.obs[group_key].astype(str))
        panels.append(('group', groups.codes.astype(float), f'{group_key}', 'tab20', None))
    n = len(panels)
    if figsize is None:
        figsize = (4.2 * n, 4.2)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    axes = np.atleast_1d(axes)
    for ax, (_, values, title, cmap, center) in zip(axes, panels):
        sc = _spatial_axes(ax, adata_ai, values, title, cmap=cmap, spot_size=spot_size, center=center)
        if sc is not None and cmap != 'tab20':
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=8)
    plt.tight_layout()
    return (fig, axes)

def plot_cluster_level_atoi_site_panel(adata_ai, sv_atoi, group_key='ground_truth', adata_group=None, top_n=6, min_cov=10, spot_size=16, cmap='rocket_r', figsize=None):
    """Plot top SV-A-to-I sites as cluster-level editing-ratio maps.

Parameters
----------
adata_ai : object
    AnnData object containing A-to-I count layers and spot metadata.
sv_atoi : object
    DataFrame returned by `detect_spatial_atoi_sites()`.
group_key : object
    Column name containing spatial group, layer, cluster, or domain labels.
adata_group : object
    Optional AnnData object containing group labels aligned by barcode.
top_n : object
    Maximum number of top-ranked sites or entries to use.
min_cov : object
    Minimum A+G coverage required for a spot-site observation.
spot_size : object
    Marker size for spatial plots.
cmap : object
    Matplotlib or seaborn colormap name/object.
figsize : object
    Matplotlib figure size.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    if adata_group is not None:
        common = adata_ai.obs_names.intersection(adata_group.obs_names)
        adata_use = adata_ai[common].copy()
        labels = adata_group.obs.loc[common, group_key].astype(str)
    else:
        adata_use = adata_ai
        if group_key not in adata_use.obs:
            raise ValueError(f'{group_key!r} not found in adata_ai.obs')
        labels = adata_use.obs[group_key].astype(str)
    sites_df = sv_atoi.copy()
    if 'is_sv_atoi' in sites_df and sites_df['is_sv_atoi'].any():
        sites_df = sites_df[sites_df['is_sv_atoi']]
    sites = [s for s in sites_df['site'].head(top_n) if s in adata_use.var_names]
    if not sites:
        raise ValueError('No top SV-A-to-I sites found in adata_ai.var_names')
    ncols = min(3, len(sites))
    nrows = int(np.ceil(len(sites) / ncols))
    if figsize is None:
        figsize = (4.1 * ncols, 4.0 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    palette = sns.color_palette(cmap, as_cmap=True)
    for ax, site in zip(axes, sites):
        values, agg = _cluster_level_site_values(adata_use, site, labels, min_cov=min_cov)
        gene = adata_use.var.loc[site].get('Gene.refGene', site) if site in adata_use.var.index else site
        title = f'{gene}\\n{site}'
        sc = _spatial_axes(ax, adata_use, values, title, cmap=palette, spot_size=spot_size)
        if sc is not None:
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        ax.text(0.02, 0.02, f"groups={agg['ratio'].notna().sum()}", transform=ax.transAxes, fontsize=8, ha='left', va='bottom')
    for ax in axes[len(sites):]:
        ax.axis('off')
    fig.suptitle(f'Cluster-level A-to-I ratio for top SV sites ({group_key})', y=1.02, fontsize=13)
    plt.tight_layout()
    return (fig, axes)

def plot_atoi_group_raincloud(adata_ai, score_key='sv_atoi_score', group_key='ground_truth', order=None, figsize=(8, 4)):
    """Plot A-to-I score distributions across spatial groups.

Parameters
----------
adata_ai : object
    AnnData object containing A-to-I count layers and spot metadata.
score_key : object
    Name of the A-to-I score column in `adata_ai.obs`.
group_key : object
    Column name containing spatial group, layer, cluster, or domain labels.
order : object
    Function argument used by this helper.
figsize : object
    Matplotlib figure size.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    df = adata_ai.obs[[score_key, group_key]].dropna().copy()
    df[group_key] = df[group_key].astype(str)
    if order is None:
        order = df.groupby(group_key, observed=True)[score_key].median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(data=df, x=group_key, y=score_key, order=order, inner=None, cut=0, color='#d8ecf3', ax=ax)
    sns.boxplot(data=df, x=group_key, y=score_key, order=order, width=0.22, showcaps=False, boxprops={'facecolor': 'white', 'edgecolor': 'black', 'linewidth': 1}, whiskerprops={'linewidth': 1}, medianprops={'color': 'black', 'linewidth': 1.2}, showfliers=False, ax=ax)
    sns.stripplot(data=df, x=group_key, y=score_key, order=order, size=2, alpha=0.28, color='#333333', ax=ax)
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.45)
    ax.set_xlabel('')
    ax.set_ylabel(score_key)
    ax.set_title('SV-A-to-I regulatory score across spatial groups')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    return (fig, ax)

def plot_atoi_deconv_dotplot(assoc_df, beta_col='ols_beta', fdr_col='ols_fdr', figsize=(7, 4)):
    """Plot cell-type association coefficients and FDR values.

Parameters
----------
assoc_df : object
    Association result DataFrame.
beta_col : object
    Column containing beta coefficients.
fdr_col : object
    Column containing FDR-adjusted p-values.
figsize : object
    Matplotlib figure size.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    df = assoc_df.copy()
    if df.empty:
        raise ValueError('assoc_df is empty')
    df = df.sort_values(beta_col)
    y = np.arange(df.shape[0])
    sig = -np.log10(df[fdr_col].clip(lower=1e-300))
    sizes = np.clip(30 + sig * 25, 40, 260)
    colors = np.where(df[beta_col] >= 0, '#d95f02', '#1b9e77')
    fig, ax = plt.subplots(figsize=figsize)
    ax.hlines(y, 0, df[beta_col], color='#bdbdbd', linewidth=1.5)
    ax.scatter(df[beta_col], y, s=sizes, c=colors, edgecolor='white', linewidth=0.8, zorder=3)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df['celltype'])
    ax.set_xlabel('adjusted beta with SV-A-to-I score')
    ax.set_title('Cell-type-associated A-to-I regulatory signal')
    for yi, (_, row) in enumerate(df.iterrows()):
        ax.text(row[beta_col], yi + 0.18, f'FDR={row[fdr_col]:.1e}', fontsize=7, ha='center')
    plt.tight_layout()
    return (fig, ax)

def plot_atoi_colocalization_dashboard(adata_ai, df_deconv, celltype, score_key='sv_atoi_score', residual_key=None, spot_size=18, figsize=(14, 4)):
    """Visualize spatial co-localization between A-to-I score and one cell type.

Parameters
----------
adata_ai : object
    AnnData object containing A-to-I count layers and spot metadata.
df_deconv : object
    DataFrame of deconvolved cell-type proportions indexed by spot barcode.
celltype : object
    Cell-type column to visualize or test.
score_key : object
    Name of the A-to-I score column in `adata_ai.obs`.
residual_key : object
    Name of the residual column written to `adata_ai.obs`.
spot_size : object
    Marker size for spatial plots.
figsize : object
    Matplotlib figure size.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    common = adata_ai.obs_names.intersection(df_deconv.index)
    adata_use = adata_ai[common].copy()
    score = adata_use.obs[score_key].astype(float).values
    cell = df_deconv.loc[common, celltype].astype(float).values
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    _spatial_axes(axes[0], adata_use, score, 'SV-A-to-I score', cmap='magma', spot_size=spot_size)
    _spatial_axes(axes[1], adata_use, cell, f'{celltype} proportion', cmap='viridis', spot_size=spot_size)
    mask = np.isfinite(score) & np.isfinite(cell)
    hi_score = score >= np.nanquantile(score[mask], 0.5)
    hi_cell = cell >= np.nanquantile(cell[mask], 0.5)
    group = np.full(len(score), np.nan)
    group[mask & ~hi_score & ~hi_cell] = 0
    group[mask & hi_score & ~hi_cell] = 1
    group[mask & ~hi_score & hi_cell] = 2
    group[mask & hi_score & hi_cell] = 3
    cmap = plt.matplotlib.colors.ListedColormap(['#e8e8e8', '#64acbe', '#c85a5a', '#574249'])
    _spatial_axes(axes[2], adata_use, group, 'Bivariate co-localization', cmap=cmap, spot_size=spot_size, clip=(0, 100))
    rho, pval = spearmanr(cell[mask], score[mask]) if mask.sum() > 2 else (np.nan, np.nan)
    sns.regplot(x=cell[mask], y=score[mask], scatter_kws={'s': 14, 'alpha': 0.45, 'linewidth': 0}, line_kws={'color': 'black'}, lowess=True, ax=axes[3])
    axes[3].set_xlabel(f'{celltype} proportion')
    axes[3].set_ylabel('SV-A-to-I score')
    axes[3].set_title(f'rho={rho:.2f}, p={pval:.1e}')
    if residual_key and residual_key in adata_use.obs:
        axes[3].text(0.03, 0.97, f'Residual: {residual_key}', transform=axes[3].transAxes, va='top', fontsize=8)
    plt.tight_layout()
    return (fig, axes)

def plot_top_site_celltype_examples(adata_ai, df_deconv, res_hits, n_examples=3, min_cov=10, spot_size=16, figsize=None):
    """Plot top site-cell-type examples using editing, cell-type, and bivariate maps.

Parameters
----------
adata_ai : object
    AnnData object containing A-to-I count layers and spot metadata.
df_deconv : object
    DataFrame of deconvolved cell-type proportions indexed by spot barcode.
res_hits : object
    DataFrame of significant site-cell-type association results.
n_examples : object
    Number of top examples to plot.
min_cov : object
    Minimum A+G coverage required for a spot-site observation.
spot_size : object
    Marker size for spatial plots.
figsize : object
    Matplotlib figure size.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    pairs = res_hits.sort_values(['padj', 'abs_coef'] if 'abs_coef' in res_hits else ['padj']).drop_duplicates(['site', 'term']).head(n_examples)
    if pairs.empty:
        raise ValueError('res_hits has no site-celltype pairs')
    if figsize is None:
        figsize = (12, 3.6 * len(pairs))
    fig, axes = plt.subplots(len(pairs), 3, figsize=figsize)
    axes = np.atleast_2d(axes)
    A, G = _get_count_layers(adata_ai)
    for r, (_, row) in enumerate(pairs.iterrows()):
        site = row['site']
        celltype = row['term']
        common = adata_ai.obs_names.intersection(df_deconv.index)
        adata_use = adata_ai[common].copy()
        j = adata_use.var_names.get_loc(site)
        a = _dense_col(_as_csr(adata_use.layers['A']), j)
        g = _dense_col(_as_csr(adata_use.layers['G']), j)
        cov = a + g
        ratio = np.full(adata_use.n_obs, np.nan)
        valid = cov >= min_cov
        ratio[valid] = g[valid] / cov[valid]
        cell = df_deconv.loc[common, celltype].astype(float).values
        gene = adata_use.var.loc[site].get('Gene.refGene', site) if site in adata_use.var.index else site
        _spatial_axes(axes[r, 0], adata_use, ratio, f'{gene} editing\\n{site}', cmap='magma', spot_size=spot_size)
        _spatial_axes(axes[r, 1], adata_use, cell, f'{celltype} proportion', cmap='viridis', spot_size=spot_size)
        mask = np.isfinite(ratio) & np.isfinite(cell)
        hi_ratio = ratio >= np.nanquantile(ratio[mask], 0.5)
        hi_cell = cell >= np.nanquantile(cell[mask], 0.5)
        group = np.full(len(ratio), np.nan)
        group[mask & ~hi_ratio & ~hi_cell] = 0
        group[mask & hi_ratio & ~hi_cell] = 1
        group[mask & ~hi_ratio & hi_cell] = 2
        group[mask & hi_ratio & hi_cell] = 3
        cmap = plt.matplotlib.colors.ListedColormap(['#e8e8e8', '#64acbe', '#c85a5a', '#574249'])
        _spatial_axes(axes[r, 2], adata_use, group, f'Editing x {celltype}', cmap=cmap, spot_size=spot_size, clip=(0, 100))
        axes[r, 2].text(0.02, 0.02, f"coef={row.get('coef_logit', np.nan):.2f}\\nFDR={row.get('padj', np.nan):.1e}", transform=axes[r, 2].transAxes, fontsize=8, va='bottom')
    plt.tight_layout()
    return (fig, axes)

def add_gene_expression_to_obs(adata_expr, target_adata, genes=('ADAR', 'ADARB1', 'ADARB2'), prefix='expr_', layer=None, log1p=False):
    """Copy selected gene expression values into an A-to-I AnnData object by shared barcode.

Parameters
----------
adata_expr : object
    AnnData object containing gene expression values.
target_adata : object
    AnnData object that receives copied expression values in `.obs`.
genes : object
    Gene symbols to copy from expression AnnData.
prefix : object
    Prefix added to copied expression columns.
layer : object
    Optional expression layer to use instead of `.X`.
log1p : object
    Whether to apply log1p transformation to copied expression values.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    common = target_adata.obs_names.intersection(adata_expr.obs_names)
    added = []
    for gene in genes:
        if gene not in adata_expr.var_names:
            continue
        x = adata_expr[common, gene].layers[layer] if layer is not None else adata_expr[common, gene].X
        values = x.toarray().ravel() if hasattr(x, 'toarray') else np.asarray(x).ravel()
        values = values.astype(float)
        if log1p:
            values = np.log1p(values)
        col = f'{prefix}{gene}'
        target_adata.obs[col] = np.nan
        target_adata.obs.loc[common, col] = values
        added.append(col)
    return added

def compute_global_atoi_ratio(adata_ai, ratio_key='global_atoi_ratio', coverage_key='global_atoi_cov', min_total_cov=20, a_layer='A', g_layer='G'):
    """Compute a per-spot global A-to-I ratio across all retained sites.

Parameters
----------
adata_ai : object
    AnnData object containing A-to-I count layers and spot metadata.
ratio_key : object
    Name of the global A-to-I ratio column written to `.obs`.
coverage_key : object
    Name of the total coverage column written to `.obs`.
min_total_cov : object
    Minimum per-spot total coverage required to compute global ratio.
a_layer : object
    Name of the layer containing A counts.
g_layer : object
    Name of the layer containing G counts.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    A, G = _get_count_layers(adata_ai, a_layer=a_layer, g_layer=g_layer)
    total_A = np.asarray(A.sum(axis=1)).ravel()
    total_G = np.asarray(G.sum(axis=1)).ravel()
    cov = total_A + total_G
    ratio = np.full(adata_ai.n_obs, np.nan, dtype=float)
    valid = cov >= min_total_cov
    ratio[valid] = total_G[valid] / cov[valid]
    adata_ai.obs[ratio_key] = ratio
    adata_ai.obs[coverage_key] = cov
    return pd.Series(ratio, index=adata_ai.obs_names, name=ratio_key)

def plot_cluster_level_obs_value(adata, value_key, group_key='ground_truth', adata_group=None, aggfunc='mean', spot_size=18, cmap='magma', center=None, figsize=(10, 4.2)):
    """Visualize one observation-level value at spot and group-aggregated levels.

Parameters
----------
adata : object
    AnnData object containing spatial coordinates and observations.
value_key : object
    Observation column to visualize.
group_key : object
    Column name containing spatial group, layer, cluster, or domain labels.
adata_group : object
    Optional AnnData object containing group labels aligned by barcode.
aggfunc : object
    Aggregation function used to summarize spot values by group.
spot_size : object
    Marker size for spatial plots.
cmap : object
    Matplotlib or seaborn colormap name/object.
center : object
    Optional center value for symmetric color scaling.
figsize : object
    Matplotlib figure size.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    if value_key not in adata.obs:
        raise ValueError(f'{value_key!r} not found in adata.obs')
    if adata_group is not None:
        common = adata.obs_names.intersection(adata_group.obs_names)
        adata_use = adata[common].copy()
        labels = adata_group.obs.loc[common, group_key].astype(str)
    else:
        adata_use = adata
        if group_key not in adata_use.obs:
            raise ValueError(f'{group_key!r} not found in adata.obs')
        labels = adata_use.obs[group_key].astype(str)
    values = adata_use.obs[value_key].astype(float)
    grouped = values.groupby(labels, observed=True).agg(aggfunc)
    cluster_values = labels.map(grouped).to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=figsize, gridspec_kw={'width_ratios': [1, 1, 0.9]})
    sc0 = _spatial_axes(axes[0], adata_use, values.values, f'Spot-level {value_key}', cmap=cmap, spot_size=spot_size, center=center)
    sc1 = _spatial_axes(axes[1], adata_use, cluster_values, f'Cluster-level {value_key}\\n{group_key}', cmap=cmap, spot_size=spot_size, center=center)
    if sc0 is not None:
        fig.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.02)
    if sc1 is not None:
        fig.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.02)
    summary = pd.DataFrame({'group': labels.values, value_key: values.values}).dropna().groupby('group', observed=True)[value_key].agg(['count', 'mean', 'median', 'std']).sort_values('mean')
    sns.barplot(data=summary.reset_index(), y='group', x='mean', ax=axes[2], color='#80b1d3', edgecolor='black', linewidth=0.4)
    axes[2].set_xlabel(f'mean {value_key}')
    axes[2].set_ylabel('')
    axes[2].set_title('Mean by group', fontsize=11, pad=8)
    if center is not None:
        axes[2].axvline(center, color='black', linewidth=1)
    plt.tight_layout()
    return (fig, axes, summary)

def analyze_adar_global_atoi_relationship(adata_ai, adar_cols=('expr_ADAR', 'expr_ADARB1', 'expr_ADARB2'), atoi_keys=('global_atoi_ratio', 'sv_atoi_score'), df_deconv=None, celltype_cols=None, covariates=None, min_complete=30, store_key='adar_global_atoi_relationship'):
    """Test associations between ADAR-family expression and global or SV A-to-I metrics.

Parameters
----------
adata_ai : object
    AnnData object containing A-to-I count layers and spot metadata.
adar_cols : object
    ADAR-family expression columns available in `adata_ai.obs`.
atoi_keys : object
    A-to-I metric columns to analyze.
df_deconv : object
    DataFrame of deconvolved cell-type proportions indexed by spot barcode.
celltype_cols : object
    Cell-type proportion columns to use as predictors or annotations.
covariates : object
    Additional covariate columns to include in regression models.
min_complete : object
    Function argument used by this helper.
store_key : object
    Key used to save result tables or summaries in `adata_ai.uns`.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    adar_cols = [c for c in adar_cols if c in adata_ai.obs.columns]
    atoi_keys = [k for k in atoi_keys if k in adata_ai.obs.columns]
    covariates = [] if covariates is None else list(covariates)
    obs_covariates = [c for c in covariates if c in adata_ai.obs.columns]
    common = adata_ai.obs_names
    if df_deconv is not None:
        common = common.intersection(df_deconv.index)
        if celltype_cols is None:
            meta = {'in_tissue', 'x_array', 'y_array', 'x_pixel', 'y_pixel', 'celltype_sum'}
            celltype_cols = [c for c in df_deconv.columns if c not in meta and pd.api.types.is_numeric_dtype(df_deconv[c])]
        ct_cols = [c for c in celltype_cols if c in df_deconv.columns]
    else:
        ct_cols = []
    df = adata_ai.obs.loc[common, atoi_keys + adar_cols + obs_covariates].copy()
    if df_deconv is not None and ct_cols:
        df = df.join(df_deconv.loc[common, ct_cols], how='left')
    rows = []
    for target in atoi_keys:
        for adar in adar_cols:
            sub = df[[target, adar]].dropna()
            rho = p_spear = np.nan
            if sub.shape[0] >= min_complete and sub[target].std() > 0 and (sub[adar].std() > 0):
                rho, p_spear = spearmanr(sub[adar], sub[target])
            model_cols = [adar] + ct_cols + obs_covariates
            sub_model = df[[target] + model_cols].dropna()
            beta = p_ols = r2 = np.nan
            if sub_model.shape[0] >= max(min_complete, len(model_cols) + 5) and sub_model[target].std() > 0 and (sub_model[adar].std() > 0):
                X = sub_model[model_cols].astype(float).copy()
                for c in X.columns:
                    sd = X[c].std()
                    if sd > 0:
                        X[c] = (X[c] - X[c].mean()) / sd
                y = sub_model[target].astype(float)
                y = (y - y.mean()) / y.std()
                fit = sm.OLS(y, sm.add_constant(X, has_constant='add')).fit()
                beta = float(fit.params[adar])
                p_ols = float(fit.pvalues[adar])
                r2 = float(fit.rsquared)
            rows.append({'atoi_metric': target, 'adar': adar.replace('expr_', ''), 'adar_col': adar, 'n_spearman': int(sub.shape[0]), 'spearman_rho': float(rho) if np.isfinite(rho) else np.nan, 'spearman_p': float(p_spear) if np.isfinite(p_spear) else np.nan, 'n_model': int(sub_model.shape[0]), 'adjusted_beta': beta, 'adjusted_p': p_ols, 'adjusted_r2': r2})
    res = pd.DataFrame(rows)
    if not res.empty:
        res['spearman_fdr'] = multipletests(res['spearman_p'].fillna(1.0), method='fdr_bh')[1]
        res['adjusted_fdr'] = multipletests(res['adjusted_p'].fillna(1.0), method='fdr_bh')[1]
        res = res.sort_values(['atoi_metric', 'adjusted_fdr', 'spearman_fdr']).reset_index(drop=True)
    adata_ai.uns[store_key] = res
    return res

def find_adar_associated_atoi_sites(adata_ai, adar_cols=('expr_ADAR', 'expr_ADARB1', 'expr_ADARB2'), candidate_sites=None, min_cov=10, min_valid_spots=30, fdr_method='fdr_bh', a_layer='A', g_layer='G', store_key='adar_site_association'):
    """Find individual A-to-I sites associated with ADAR-family expression.

Parameters
----------
adata_ai : object
    AnnData object containing A-to-I count layers and spot metadata.
adar_cols : object
    ADAR-family expression columns available in `adata_ai.obs`.
candidate_sites : object
    Optional subset of candidate A-to-I sites to test.
min_cov : object
    Minimum A+G coverage required for a spot-site observation.
min_valid_spots : object
    Minimum number of complete or valid spots required for analysis.
fdr_method : object
    Multiple-testing correction method passed to `multipletests`.
a_layer : object
    Name of the layer containing A counts.
g_layer : object
    Name of the layer containing G counts.
store_key : object
    Key used to save result tables or summaries in `adata_ai.uns`.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    adar_cols = [c for c in adar_cols if c in adata_ai.obs.columns]
    if candidate_sites is None:
        sites = list(adata_ai.var_names)
    elif isinstance(candidate_sites, pd.DataFrame):
        sites = [s for s in candidate_sites['site'].astype(str) if s in adata_ai.var_names]
    else:
        sites = [s for s in candidate_sites if s in adata_ai.var_names]
    A, G = _get_count_layers(adata_ai, a_layer=a_layer, g_layer=g_layer)
    rows = []
    for site in sites:
        j = adata_ai.var_names.get_loc(site)
        a = _dense_col(A, j)
        g = _dense_col(G, j)
        cov = a + g
        ratio = np.full(adata_ai.n_obs, np.nan)
        valid_cov = cov >= min_cov
        ratio[valid_cov] = g[valid_cov] / cov[valid_cov]
        gene = adata_ai.var.loc[site].get('Gene.refGene', site)
        for adar in adar_cols:
            expr = adata_ai.obs[adar].astype(float).values
            mask = np.isfinite(ratio) & np.isfinite(expr)
            if mask.sum() < min_valid_spots or np.nanstd(ratio[mask]) == 0 or np.nanstd(expr[mask]) == 0:
                continue
            rho, pval = spearmanr(expr[mask], ratio[mask])
            rows.append({'site': site, 'gene': gene, 'adar': adar.replace('expr_', ''), 'adar_col': adar, 'n_valid_spots': int(mask.sum()), 'spearman_rho': float(rho), 'spearman_p': float(pval), 'mean_ratio': float(np.nanmean(ratio[mask])), 'sd_ratio': float(np.nanstd(ratio[mask]))})
    res = pd.DataFrame(rows)
    if not res.empty:
        res['padj'] = np.nan
        for adar in res['adar'].unique():
            idx = res['adar'] == adar
            res.loc[idx, 'padj'] = multipletests(res.loc[idx, 'spearman_p'], method=fdr_method)[1]
        res['abs_rho'] = res['spearman_rho'].abs()
        res = res.sort_values(['adar', 'padj', 'abs_rho'], ascending=[True, True, False]).reset_index(drop=True)
    adata_ai.uns[store_key] = res
    return res

def compute_adar_site_module_scores(adata_ai, adar_site_assoc, padj_cutoff=0.1, top_n=50, min_cov=10, score_prefix='adar_site_module'):
    """Compute one multi-site editing module score for each ADAR-family member.

Parameters
----------
adata_ai : object
    AnnData object containing A-to-I count layers and spot metadata.
adar_site_assoc : object
    DataFrame returned by `find_adar_associated_atoi_sites()`.
padj_cutoff : object
    Adjusted p-value threshold for selecting ADAR-associated sites.
top_n : object
    Maximum number of top-ranked sites or entries to use.
min_cov : object
    Minimum A+G coverage required for a spot-site observation.
score_prefix : object
    Prefix used for ADAR-site module score columns.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    score_cols = []
    if adar_site_assoc.empty:
        return score_cols
    for adar, df in adar_site_assoc.groupby('adar', observed=True):
        use = df[df['padj'] <= padj_cutoff].copy()
        if use.empty:
            use = df.copy()
        use = use.sort_values(['padj', 'abs_rho'], ascending=[True, False]).head(top_n)
        sites = [s for s in use['site'] if s in adata_ai.var_names]
        if not sites:
            continue
        weights = use.set_index('site').loc[sites, 'spearman_rho'].values.astype(float)
        idx = [adata_ai.var_names.get_loc(s) for s in sites]
        ratio_mat, _ = _site_ratio_matrix(adata_ai, idx, min_cov=min_cov)
        z = ratio_mat.copy()
        mu = np.nanmean(z, axis=0)
        sd = np.nanstd(z, axis=0)
        sd[sd == 0] = np.nan
        z = (z - mu) / sd
        signed_weights = np.sign(weights)
        signed_weights[signed_weights == 0] = 1
        valid = np.isfinite(z)
        score = np.nansum(np.where(valid, z * signed_weights, np.nan), axis=1) / np.sum(valid, axis=1)
        score[np.sum(valid, axis=1) == 0] = np.nan
        col = f'{score_prefix}_{adar}'
        adata_ai.obs[col] = score
        adata_ai.uns[f'{col}_sites'] = sites
        score_cols.append(col)
    return score_cols

def plot_adar_global_relationship_dashboard(adata_ai, adar_cols=('expr_ADAR', 'expr_ADARB1', 'expr_ADARB2'), atoi_keys=('global_atoi_ratio', 'sv_atoi_score'), spot_size=18, figsize=None):
    """Plot ADAR-family expression maps and expression-versus-A-to-I scatterplots.

Parameters
----------
adata_ai : object
    AnnData object containing A-to-I count layers and spot metadata.
adar_cols : object
    ADAR-family expression columns available in `adata_ai.obs`.
atoi_keys : object
    A-to-I metric columns to analyze.
spot_size : object
    Marker size for spatial plots.
figsize : object
    Matplotlib figure size.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    adar_cols = [c for c in adar_cols if c in adata_ai.obs.columns]
    atoi_keys = [k for k in atoi_keys if k in adata_ai.obs.columns]
    nrows = len(adar_cols)
    ncols = 1 + len(atoi_keys)
    if figsize is None:
        figsize = (4.2 * ncols, 3.8 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for r, adar in enumerate(adar_cols):
        gene = adar.replace('expr_', '')
        expr = adata_ai.obs[adar].astype(float).values
        sc = _spatial_axes(axes[r, 0], adata_ai, expr, f'{gene} expression', cmap='YlGnBu', spot_size=spot_size)
        if sc is not None:
            fig.colorbar(sc, ax=axes[r, 0], fraction=0.046, pad=0.02)
        for c, metric in enumerate(atoi_keys, start=1):
            x = expr
            y = adata_ai.obs[metric].astype(float).values
            mask = np.isfinite(x) & np.isfinite(y)
            rho, pval = spearmanr(x[mask], y[mask]) if mask.sum() > 2 else (np.nan, np.nan)
            sns.regplot(x=x[mask], y=y[mask], scatter_kws={'s': 13, 'alpha': 0.42, 'linewidth': 0}, line_kws={'color': 'black', 'linewidth': 1.3}, lowess=True, ax=axes[r, c])
            axes[r, c].set_xlabel(f'{gene} expression')
            axes[r, c].set_ylabel(metric)
            axes[r, c].set_title(f'{gene} vs {metric}\\nrho={rho:.2f}, p={pval:.1e}', fontsize=10)
    plt.tight_layout()
    return (fig, axes)

def plot_adar_site_module_overview(adata_ai, module_cols, adar_cols=('expr_ADAR', 'expr_ADARB1', 'expr_ADARB2'), group_key='ground_truth', spot_size=18, figsize=None):
    """Compare ADAR expression maps with ADAR-associated multi-site editing module maps.

Parameters
----------
adata_ai : object
    AnnData object containing A-to-I count layers and spot metadata.
module_cols : object
    ADAR-associated module score columns in `adata_ai.obs`.
adar_cols : object
    ADAR-family expression columns available in `adata_ai.obs`.
group_key : object
    Column name containing spatial group, layer, cluster, or domain labels.
spot_size : object
    Marker size for spatial plots.
figsize : object
    Matplotlib figure size.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    adar_cols = [c for c in adar_cols if c in adata_ai.obs.columns]
    module_cols = [c for c in module_cols if c in adata_ai.obs.columns]
    rows = []
    for adar in adar_cols:
        name = adar.replace('expr_', '')
        module = next((m for m in module_cols if m.endswith(f'_{name}')), None)
        if module:
            rows.append((adar, module))
    if not rows:
        raise ValueError('No matching ADAR module columns found.')
    if figsize is None:
        figsize = (8.4, 3.8 * len(rows))
    fig, axes = plt.subplots(len(rows), 2, figsize=figsize, squeeze=False)
    for r, (adar, module) in enumerate(rows):
        name = adar.replace('expr_', '')
        sc = _spatial_axes(axes[r, 0], adata_ai, adata_ai.obs[adar].values, f'{name} expression', cmap='YlGnBu', spot_size=spot_size)
        if sc is not None:
            fig.colorbar(sc, ax=axes[r, 0], fraction=0.046, pad=0.02)
        sc = _spatial_axes(axes[r, 1], adata_ai, adata_ai.obs[module].values, f'{name}-associated multi-site A-to-I', cmap='coolwarm', spot_size=spot_size, center=0)
        if sc is not None:
            fig.colorbar(sc, ax=axes[r, 1], fraction=0.046, pad=0.02)
    plt.tight_layout()
    return (fig, axes)

def add_wm_indicator(adata, group_key='ground_truth', wm_label='WM', wm_key='is_WM'):
    """Create a boolean white-matter indicator from a group annotation.

Parameters
----------
adata : object
    AnnData object containing spatial coordinates and observations.
group_key : object
    Column name containing spatial group, layer, cluster, or domain labels.
wm_label : object
    Group label treated as white matter.
wm_key : object
    Boolean `.obs` column indicating white-matter spots.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    if group_key not in adata.obs:
        raise ValueError(f'{group_key!r} not found in adata.obs')
    adata.obs[wm_key] = adata.obs[group_key].astype(str).str.upper().eq(str(wm_label).upper())
    return adata.obs[wm_key]

def summarize_wm_nonwm_atoi_adar(adata_ai, group_key='ground_truth', wm_label='WM', adar_cols=('expr_ADAR', 'expr_ADARB1', 'expr_ADARB2'), atoi_cols=('global_atoi_ratio', 'sv_atoi_score', 'sv_atoi_residual', 'sv_atoi_residual_deconv_adar'), module_prefix='adar_site_module_', wm_key='is_WM'):
    """Summarize ADAR expression and A-to-I metrics in WM versus non-WM spots.

Parameters
----------
adata_ai : object
    AnnData object containing A-to-I count layers and spot metadata.
group_key : object
    Column name containing spatial group, layer, cluster, or domain labels.
wm_label : object
    Group label treated as white matter.
adar_cols : object
    ADAR-family expression columns available in `adata_ai.obs`.
atoi_cols : object
    A-to-I metric columns to summarize or plot.
module_prefix : object
    Prefix used to identify ADAR-associated module columns.
wm_key : object
    Boolean `.obs` column indicating white-matter spots.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    if wm_key not in adata_ai.obs:
        add_wm_indicator(adata_ai, group_key=group_key, wm_label=wm_label, wm_key=wm_key)
    module_cols = [c for c in adata_ai.obs.columns if c.startswith(module_prefix)]
    value_cols = [c for c in list(adar_cols) + list(atoi_cols) + module_cols if c in adata_ai.obs.columns]
    rows = []
    for col in value_cols:
        wm = adata_ai.obs.loc[adata_ai.obs[wm_key], col].astype(float).dropna()
        nonwm = adata_ai.obs.loc[~adata_ai.obs[wm_key], col].astype(float).dropna()
        if len(wm) == 0 or len(nonwm) == 0:
            continue
        stat, pval = mannwhitneyu(wm, nonwm, alternative='two-sided')
        rows.append({'feature': col, 'n_WM': int(len(wm)), 'n_nonWM': int(len(nonwm)), 'mean_WM': float(wm.mean()), 'mean_nonWM': float(nonwm.mean()), 'median_WM': float(wm.median()), 'median_nonWM': float(nonwm.median()), 'delta_mean_WM_minus_nonWM': float(wm.mean() - nonwm.mean()), 'mannwhitney_p': float(pval)})
    res = pd.DataFrame(rows)
    if not res.empty:
        res['padj'] = multipletests(res['mannwhitney_p'], method='fdr_bh')[1]
        res = res.sort_values(['padj', 'feature']).reset_index(drop=True)
    adata_ai.uns['wm_nonwm_atoi_adar_summary'] = res
    return res

def analyze_wm_adar_interaction(adata_ai, target_cols=('global_atoi_ratio', 'sv_atoi_score'), adar_cols=('expr_ADAR', 'expr_ADARB1', 'expr_ADARB2'), wm_key='is_WM', df_deconv=None, celltype_cols=None, covariates=None, min_complete=30, store_key='wm_adar_interaction'):
    """Fit ADAR-by-WM interaction models for A-to-I metrics.

Parameters
----------
adata_ai : object
    AnnData object containing A-to-I count layers and spot metadata.
target_cols : object
    A-to-I metric columns used as regression outcomes.
adar_cols : object
    ADAR-family expression columns available in `adata_ai.obs`.
wm_key : object
    Boolean `.obs` column indicating white-matter spots.
df_deconv : object
    DataFrame of deconvolved cell-type proportions indexed by spot barcode.
celltype_cols : object
    Cell-type proportion columns to use as predictors or annotations.
covariates : object
    Additional covariate columns to include in regression models.
min_complete : object
    Function argument used by this helper.
store_key : object
    Key used to save result tables or summaries in `adata_ai.uns`.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    if wm_key not in adata_ai.obs:
        raise ValueError(f'{wm_key!r} not found in adata_ai.obs')
    target_cols = [c for c in target_cols if c in adata_ai.obs.columns]
    adar_cols = [c for c in adar_cols if c in adata_ai.obs.columns]
    covariates = [] if covariates is None else list(covariates)
    obs_covariates = [c for c in covariates if c in adata_ai.obs.columns]
    common = adata_ai.obs_names
    if df_deconv is not None:
        common = common.intersection(df_deconv.index)
        if celltype_cols is None:
            meta = {'in_tissue', 'x_array', 'y_array', 'x_pixel', 'y_pixel', 'celltype_sum'}
            celltype_cols = [c for c in df_deconv.columns if c not in meta and pd.api.types.is_numeric_dtype(df_deconv[c])]
        ct_cols = [c for c in celltype_cols if c in df_deconv.columns]
    else:
        ct_cols = []
    base_cols = target_cols + adar_cols + [wm_key] + obs_covariates
    df = adata_ai.obs.loc[common, base_cols].copy()
    if df_deconv is not None and ct_cols:
        df = df.join(df_deconv.loc[common, ct_cols], how='left')
    rows = []
    for target in target_cols:
        for adar in adar_cols:
            model_cols = [adar, wm_key] + ct_cols + obs_covariates
            sub = df[[target] + model_cols].dropna().copy()
            if sub.shape[0] < max(min_complete, len(model_cols) + 6):
                continue
            sub[wm_key] = sub[wm_key].astype(float)
            adar_z = sub[adar].astype(float)
            if adar_z.std() == 0 or sub[target].astype(float).std() == 0:
                continue
            sub[adar] = (adar_z - adar_z.mean()) / adar_z.std()
            interaction_col = f'{adar}:WM'
            sub[interaction_col] = sub[adar] * sub[wm_key]
            X_cols = [adar, wm_key, interaction_col] + ct_cols + obs_covariates
            X = sub[X_cols].astype(float).copy()
            for c in ct_cols + obs_covariates:
                sd = X[c].std()
                if sd > 0:
                    X[c] = (X[c] - X[c].mean()) / sd
            y = sub[target].astype(float)
            y = (y - y.mean()) / y.std()
            fit = sm.OLS(y, sm.add_constant(X, has_constant='add')).fit()
            rows.append({'target': target, 'adar': adar.replace('expr_', ''), 'n_spots': int(sub.shape[0]), 'beta_adar_nonWM': float(fit.params[adar]), 'p_adar_nonWM': float(fit.pvalues[adar]), 'beta_WM_shift': float(fit.params[wm_key]), 'p_WM_shift': float(fit.pvalues[wm_key]), 'beta_adar_x_WM': float(fit.params[interaction_col]), 'p_adar_x_WM': float(fit.pvalues[interaction_col]), 'r2': float(fit.rsquared)})
    res = pd.DataFrame(rows)
    if not res.empty:
        res['padj_adar_x_WM'] = multipletests(res['p_adar_x_WM'], method='fdr_bh')[1]
        res = res.sort_values(['target', 'padj_adar_x_WM']).reset_index(drop=True)
    adata_ai.uns[store_key] = res
    return res

def plot_wm_adar_program_summary(adata_ai, group_key='ground_truth', wm_label='WM', wm_key='is_WM', adar_cols=('expr_ADAR', 'expr_ADARB1', 'expr_ADARB2'), module_cols=None, atoi_cols=('global_atoi_ratio', 'sv_atoi_score'), spot_size=18, figsize=None):
    """Create a spatial and boxplot summary of WM/non-WM ADAR-A-to-I programs.

Parameters
----------
adata_ai : object
    AnnData object containing A-to-I count layers and spot metadata.
group_key : object
    Column name containing spatial group, layer, cluster, or domain labels.
wm_label : object
    Group label treated as white matter.
wm_key : object
    Boolean `.obs` column indicating white-matter spots.
adar_cols : object
    ADAR-family expression columns available in `adata_ai.obs`.
module_cols : object
    ADAR-associated module score columns in `adata_ai.obs`.
atoi_cols : object
    A-to-I metric columns to summarize or plot.
spot_size : object
    Marker size for spatial plots.
figsize : object
    Matplotlib figure size.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    if wm_key not in adata_ai.obs:
        add_wm_indicator(adata_ai, group_key=group_key, wm_label=wm_label, wm_key=wm_key)
    adar_cols = [c for c in adar_cols if c in adata_ai.obs.columns]
    if module_cols is None:
        module_cols = [c for c in adata_ai.obs.columns if c.startswith('adar_site_module_')]
    module_cols = [c for c in module_cols if c in adata_ai.obs.columns]
    atoi_cols = [c for c in atoi_cols if c in adata_ai.obs.columns]
    map_cols = adar_cols + module_cols + atoi_cols
    n_maps = min(len(map_cols), 8)
    if figsize is None:
        figsize = (4.0 * min(n_maps, 4), 3.8 * (int(np.ceil(n_maps / 4)) + 1))
    ncols = min(4, max(1, n_maps))
    nrows = int(np.ceil(n_maps / ncols)) + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    wm_numeric = adata_ai.obs[wm_key].astype(float).values
    _spatial_axes(axes[0], adata_ai, wm_numeric, f'{wm_label} indicator', cmap='Greys', spot_size=spot_size, clip=(0, 100))
    for ax, col in zip(axes[1:n_maps], map_cols[:max(0, n_maps - 1)]):
        cmap = 'YlGnBu' if col.startswith('expr_') else 'coolwarm' if 'module' in col or 'score' in col else 'magma'
        center = 0 if cmap == 'coolwarm' else None
        _spatial_axes(ax, adata_ai, adata_ai.obs[col].values, col, cmap=cmap, spot_size=spot_size, center=center)
    box_ax = axes[n_maps] if n_maps < len(axes) else axes[-1]
    plot_cols = adar_cols + module_cols + atoi_cols
    df = adata_ai.obs[[wm_key] + plot_cols].copy()
    df['region'] = np.where(df[wm_key], wm_label, f'non-{wm_label}')
    melt = df.melt(id_vars='region', value_vars=plot_cols, var_name='feature', value_name='value').dropna()
    sns.boxplot(data=melt, x='feature', y='value', hue='region', showfliers=False, ax=box_ax)
    box_ax.tick_params(axis='x', rotation=45)
    box_ax.set_xlabel('')
    box_ax.set_title(f'{wm_label} vs non-{wm_label}')
    for ax in axes[n_maps + 1:]:
        ax.axis('off')
    plt.tight_layout()
    return (fig, axes)

def analyze_adar_atoi_association(adata_ai, adata_expr, score_key='sv_atoi_score', genes=('ADAR', 'ADARB1', 'ADARB2'), df_deconv=None, celltype_cols=None, covariates=None, expr_prefix='expr_', layer=None, log1p=False, min_complete=30, store_key='adar_atoi_association'):
    """Test whether ADAR-family expression explains a selected SV-A-to-I score.

Parameters
----------
adata_ai : object
    AnnData object containing A-to-I count layers and spot metadata.
adata_expr : object
    AnnData object containing gene expression values.
score_key : object
    Name of the A-to-I score column in `adata_ai.obs`.
genes : object
    Gene symbols to copy from expression AnnData.
df_deconv : object
    DataFrame of deconvolved cell-type proportions indexed by spot barcode.
celltype_cols : object
    Cell-type proportion columns to use as predictors or annotations.
covariates : object
    Additional covariate columns to include in regression models.
expr_prefix : object
    Prefix added to expression columns copied into `adata_ai.obs`.
layer : object
    Optional expression layer to use instead of `.X`.
log1p : object
    Whether to apply log1p transformation to copied expression values.
min_complete : object
    Function argument used by this helper.
store_key : object
    Key used to save result tables or summaries in `adata_ai.uns`.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    added_cols = add_gene_expression_to_obs(adata_expr=adata_expr, target_adata=adata_ai, genes=genes, prefix=expr_prefix, layer=layer, log1p=log1p)
    if not added_cols:
        raise ValueError('None of the requested ADAR-family genes were found in adata_expr.var_names.')
    covariates = [] if covariates is None else list(covariates)
    model_covariates = [c for c in covariates if c in adata_ai.obs.columns]
    common = adata_ai.obs_names
    if df_deconv is not None:
        common = common.intersection(df_deconv.index)
    df = adata_ai.obs.loc[common, [score_key] + added_cols + model_covariates].copy()
    if df_deconv is not None:
        if celltype_cols is None:
            meta = {'in_tissue', 'x_array', 'y_array', 'x_pixel', 'y_pixel', 'celltype_sum'}
            celltype_cols = [c for c in df_deconv.columns if c not in meta and pd.api.types.is_numeric_dtype(df_deconv[c])]
        ct_cols = [c for c in celltype_cols if c in df_deconv.columns]
        df = df.join(df_deconv.loc[common, ct_cols], how='left')
    else:
        ct_cols = []
    rows = []
    for gene_col in added_cols:
        base_cols = [score_key, gene_col]
        sub = df.loc[:, base_cols].dropna()
        if sub.shape[0] >= min_complete and sub[gene_col].std() > 0 and (sub[score_key].std() > 0):
            from scipy.stats import spearmanr
            rho, p_spear = spearmanr(sub[gene_col], sub[score_key])
        else:
            rho, p_spear = (np.nan, np.nan)
        model_cols = [gene_col] + ct_cols + model_covariates
        sub_model = df.loc[:, [score_key] + model_cols].dropna()
        beta = p_ols = r2 = np.nan
        if sub_model.shape[0] >= max(min_complete, len(model_cols) + 5) and sub_model[gene_col].std() > 0:
            X = sub_model[model_cols].astype(float).copy()
            for c in X.columns:
                sd = X[c].std()
                if sd > 0:
                    X[c] = (X[c] - X[c].mean()) / sd
            y = sub_model[score_key].astype(float)
            if y.std() > 0:
                y = (y - y.mean()) / y.std()
                fit = sm.OLS(y, sm.add_constant(X, has_constant='add')).fit()
                beta = float(fit.params[gene_col])
                p_ols = float(fit.pvalues[gene_col])
                r2 = float(fit.rsquared)
        rows.append({'gene': gene_col.replace(expr_prefix, ''), 'expr_col': gene_col, 'n_spearman': int(sub.shape[0]), 'spearman_rho': float(rho) if np.isfinite(rho) else np.nan, 'spearman_p': float(p_spear) if np.isfinite(p_spear) else np.nan, 'n_model': int(sub_model.shape[0]), 'adjusted_beta': beta, 'adjusted_p': p_ols, 'adjusted_r2': r2, 'adjusted_for_celltypes': bool(len(ct_cols) > 0), 'celltype_cols': ct_cols, 'covariates': model_covariates})
    res = pd.DataFrame(rows)
    if not res.empty:
        res['spearman_fdr'] = multipletests(res['spearman_p'].fillna(1.0), method='fdr_bh')[1]
        res['adjusted_fdr'] = multipletests(res['adjusted_p'].fillna(1.0), method='fdr_bh')[1]
        res = res.sort_values(['adjusted_fdr', 'spearman_fdr']).reset_index(drop=True)
    adata_ai.uns[store_key] = res
    return (res, added_cols)

def compute_atoi_residual_after_deconv_adar(adata_ai, df_deconv=None, score_key='sv_atoi_score', celltype_cols=None, adar_cols=('expr_ADAR', 'expr_ADARB1', 'expr_ADARB2'), covariates=None, residual_key='sv_atoi_residual_deconv_adar', spatial_k=6, store_key='atoi_residual_deconv_adar_model'):
    """Compute A-to-I residuals after adjusting for cell types and ADAR-family expression.

Parameters
----------
adata_ai : object
    AnnData object containing A-to-I count layers and spot metadata.
df_deconv : object
    DataFrame of deconvolved cell-type proportions indexed by spot barcode.
score_key : object
    Name of the A-to-I score column in `adata_ai.obs`.
celltype_cols : object
    Cell-type proportion columns to use as predictors or annotations.
adar_cols : object
    ADAR-family expression columns available in `adata_ai.obs`.
covariates : object
    Additional covariate columns to include in regression models.
residual_key : object
    Name of the residual column written to `adata_ai.obs`.
spatial_k : object
    Number of nearest neighbors used for spatial autocorrelation statistics.
store_key : object
    Key used to save result tables or summaries in `adata_ai.uns`.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    available_adar = [c for c in adar_cols if c in adata_ai.obs.columns]
    covariates = [] if covariates is None else list(covariates)
    return compute_atoi_residual_after_deconv(adata_ai=adata_ai, df_deconv=df_deconv, score_key=score_key, celltype_cols=celltype_cols, covariates=available_adar + covariates, residual_key=residual_key, spatial_k=spatial_k, store_key=store_key)

def plot_adar_atoi_dashboard(adata_ai, adata_expr=None, gene='ADAR', score_key='sv_atoi_score', residual_key='sv_atoi_residual_deconv_adar', expr_col=None, spot_size=18, figsize=(13.5, 4)):
    """Plot ADAR expression, A-to-I score, adjusted residuals, and their scatter relationship.

Parameters
----------
adata_ai : object
    AnnData object containing A-to-I count layers and spot metadata.
adata_expr : object
    AnnData object containing gene expression values.
gene : object
    Single ADAR-family gene symbol for plotting.
score_key : object
    Name of the A-to-I score column in `adata_ai.obs`.
residual_key : object
    Name of the residual column written to `adata_ai.obs`.
expr_col : object
    Existing expression column name to use for plotting.
spot_size : object
    Marker size for spatial plots.
figsize : object
    Matplotlib figure size.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    if expr_col is None:
        expr_col = f'expr_{gene}'
    if expr_col not in adata_ai.obs:
        if adata_expr is None:
            raise ValueError(f'{expr_col!r} not found in adata_ai.obs and adata_expr was not provided.')
        add_gene_expression_to_obs(adata_expr, adata_ai, genes=(gene,), prefix='expr_')
    if expr_col not in adata_ai.obs:
        raise ValueError(f'{gene!r} was not found in expression data.')
    expr = adata_ai.obs[expr_col].astype(float).values
    score = adata_ai.obs[score_key].astype(float).values
    ncols = 4 if residual_key in adata_ai.obs else 3
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    axes = np.atleast_1d(axes)
    sc = _spatial_axes(axes[0], adata_ai, expr, f'{gene} expression', cmap='YlGnBu', spot_size=spot_size)
    if sc is not None:
        fig.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.02)
    sc = _spatial_axes(axes[1], adata_ai, score, 'SV-A-to-I score', cmap='magma', spot_size=spot_size)
    if sc is not None:
        fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.02)
    scatter_ax = axes[-1]
    if residual_key in adata_ai.obs:
        resid = adata_ai.obs[residual_key].astype(float).values
        sc = _spatial_axes(axes[2], adata_ai, resid, 'Residual after deconv + ADAR', cmap='coolwarm', spot_size=spot_size, center=0)
        if sc is not None:
            fig.colorbar(sc, ax=axes[2], fraction=0.046, pad=0.02)
    mask = np.isfinite(expr) & np.isfinite(score)
    rho, pval = spearmanr(expr[mask], score[mask]) if mask.sum() > 2 else (np.nan, np.nan)
    sns.regplot(x=expr[mask], y=score[mask], scatter_kws={'s': 14, 'alpha': 0.45, 'linewidth': 0}, line_kws={'color': 'black'}, lowess=True, ax=scatter_ax)
    scatter_ax.set_xlabel(f'{gene} expression')
    scatter_ax.set_ylabel('SV-A-to-I score')
    scatter_ax.set_title(f'{gene} vs A-to-I\\nrho={rho:.2f}, p={pval:.1e}')
    plt.tight_layout()
    return (fig, axes)

def plot_adar_adjusted_residual_comparison(adata_ai, residual_before='sv_atoi_residual', residual_after='sv_atoi_residual_deconv_adar', group_key='ground_truth', spot_size=18, figsize=(12, 4)):
    """Compare residual A-to-I maps before and after ADAR-family adjustment.

Parameters
----------
adata_ai : object
    AnnData object containing A-to-I count layers and spot metadata.
residual_before : object
    Column containing residuals before adjustment.
residual_after : object
    Column containing residuals after adjustment.
group_key : object
    Column name containing spatial group, layer, cluster, or domain labels.
spot_size : object
    Marker size for spatial plots.
figsize : object
    Matplotlib figure size.

Returns
-------
object
    Function-specific result. See the function body and returned variables for details."""
    if residual_before not in adata_ai.obs or residual_after not in adata_ai.obs:
        raise ValueError('Both residual columns must be present in adata_ai.obs.')
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    _spatial_axes(axes[0], adata_ai, adata_ai.obs[residual_before].values, 'Residual after deconv', cmap='coolwarm', spot_size=spot_size, center=0)
    _spatial_axes(axes[1], adata_ai, adata_ai.obs[residual_after].values, 'Residual after deconv + ADAR', cmap='coolwarm', spot_size=spot_size, center=0)
    df = adata_ai.obs[[residual_before, residual_after]].copy()
    if group_key in adata_ai.obs:
        df[group_key] = adata_ai.obs[group_key].astype(str)
        plot_df = df.melt(id_vars=group_key, value_vars=[residual_before, residual_after], var_name='model', value_name='residual').dropna()
        sns.boxplot(data=plot_df, x=group_key, y='residual', hue='model', showfliers=False, ax=axes[2])
        axes[2].tick_params(axis='x', rotation=45)
    else:
        plot_df = df.melt(var_name='model', value_name='residual').dropna()
        sns.boxplot(data=plot_df, x='model', y='residual', showfliers=False, ax=axes[2])
        axes[2].tick_params(axis='x', rotation=25)
    axes[2].axhline(0, color='black', linewidth=1)
    axes[2].set_title('Residual distribution')
    axes[2].set_xlabel('')
    plt.tight_layout()
    return (fig, axes)
