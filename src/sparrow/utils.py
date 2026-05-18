import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import pandas as pd
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import ot
from tqdm import tqdm
import seaborn as sns
import random
from torch.backends import cudnn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from collections import Counter
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
import matplotlib.colors as mcolors
from scipy.sparse.csgraph import connected_components
from natsort import natsorted
import shutil
import matplotlib as mpl
from sklearn.metrics import roc_auc_score, roc_curve, auc, average_precision_score, precision_recall_curve
import os
from scipy.sparse import block_diag
import math
import anndata as ad
from scipy.spatial import distance_matrix
from scipy import sparse
import warnings
import statsmodels.api as sm

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """
    Clustering using the mclust algorithm via rpy2.
    """
    np.random.seed(random_seed)
    
    # Activate automatic conversion from numpy to R
    rpy2.robjects.numpy2ri.activate()
    mclust_pkg = importr('mclust')

    # Set random seed in R
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)

    # 1. Extract data and convert to float64 (Crucial! R requires double precision)
    # 2. Ensure memory continuity
    data_np = np.ascontiguousarray(adata.obsm[used_obsm], dtype=np.float64)

    # 3. Check for NaNs
    if np.isnan(data_np).any():
        raise ValueError("Input data contains NaNs, Mclust cannot proceed.")

    # 4. Run Mclust
    # verbose=False to prevent stalling/lag
    try:
        res = mclust_pkg.Mclust(data_np, G=num_cluster, modelNames=modelNames, verbose=False)
    except Exception as e:
        print(f"R execution error: {e}")
        res = None

    # 5. Check if NULL was returned
    if res is None or isinstance(res, robjects.rinterface.NULLType):
        # Attempt retry with automatic model selection (remove modelNames='EEE' restriction)
        print("Mclust returned NULL with modelNames='EEE'. Retrying with auto model selection...")
        try:
            res = mclust_pkg.Mclust(data_np, G=num_cluster, verbose=False)
        except:
            pass
            
        if res is None or isinstance(res, robjects.rinterface.NULLType):
            raise RuntimeError("Mclust failed to converge (returned NULL). Data might be singular or too high-dimensional.")

    # 6. Extract results
    mclust_res = np.array(res.rx2('classification'))

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def search_res(adata, n_clusters, method='leiden', use_rep='emb_pca', start=0.1, end=3.0, increment=0.01):
    '''
    Linear Reverse Search. 
    '''
    print(f'  [Search] Linear Reverse Search for k={n_clusters}...')
    
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    
    found_res = None
    
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
           sc.tl.leiden(adata, random_state=0, resolution=res)
           count_unique = len(adata.obs['leiden'].unique())
        
        if count_unique == n_clusters:
            found_res = res
            break 
    
    if found_res is None:
        raise ValueError(f"Resolution not found for k={n_clusters} in range [{start}, {end}]. Please try bigger range or smaller step!")
        
    return found_res

def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec  = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
        
    return np.array([str(i) for i in list(new_type)])


def clustering_auto(
    adata, 
    n_clusters=7, 
    radius=50, 
    n_components=[20], 
    key='emb',         
    method='leiden', 
    eval_metric='sc',  
    refinement=True,
    start=0.1, end=3.0, increment=0.01 
):
    
    if isinstance(n_components, int):
        n_components_list = [n_components]
    else:
        n_components_list = n_components
    
    if eval_metric in ['ari', 'nmi']:
        if 'ground_truth' not in adata.obs.columns:
            print("Warning: 'ground_truth' not found. Falling back to 'sc'.")
            eval_metric = 'sc'

    best_score = -np.inf
    best_n_comp = None
    best_labels = None
    
    # èŽ·å–åŽŸå§‹ Embedding
    X_raw = adata.obsm[key].copy()

    print(f"Starting Auto-Clustering (Method: {method}, Metric: {eval_metric})")

    for n_comp in n_components_list:
        
        # --- PCA ---
        curr_n = min(n_comp, X_raw.shape[1])
        pca = PCA(n_components=curr_n, random_state=42)
        X_pca = pca.fit_transform(X_raw)
        adata.obsm['emb_pca'] = X_pca
        
        temp_labels = None
        
        try:
            # --- Clustering ---
            if method == 'leiden':
                # è¿™é‡Œå¦‚æžœæŠ›å‡º ValueErrorï¼Œä¼šè¢« except æ•èŽ·
                res = search_res(adata, n_clusters, method='leiden', use_rep='emb_pca', 
                                 start=start, end=end, increment=increment)
                
                sc.tl.leiden(adata, random_state=0, resolution=res)
                temp_labels = adata.obs['leiden'].values.copy()
                
            elif method == 'mclust':
                adata = mclust_R(adata, num_cluster=n_clusters, used_obsm='emb_pca')
                temp_labels = adata.obs['mclust'].values.copy()
            
            # --- Refinement ---
            if refinement:
                adata.obs['temp_refine'] = temp_labels
                temp_labels = refine_label(adata, radius, key='temp_refine')
            
            # --- Evaluation ---
            n_found = len(np.unique(temp_labels))
            if n_found < 2:
                print(f"  [n={n_comp}] Failed: Only 1 cluster.")
            else:
                if eval_metric == 'sc':
                    idx = np.random.choice(X_raw.shape[0], min(10000, X_raw.shape[0]), replace=False)
                    score = silhouette_score(X_raw[idx], temp_labels[idx])
                elif eval_metric == 'ch':
                    score = calinski_harabasz_score(X_raw, temp_labels)
                elif eval_metric in ['ari', 'nmi']:
                    df_eval = pd.DataFrame({'pred': temp_labels, 'gt': adata.obs['ground_truth'].values})
                    df_eval = df_eval[df_eval['gt'].notna() & (df_eval['gt'] != 'Unknown')]
                    
                    if len(df_eval) > 0:
                        if eval_metric == 'ari':
                            score = adjusted_rand_score(df_eval['gt'], df_eval['pred'])
                        else:
                            score = normalized_mutual_info_score(df_eval['gt'], df_eval['pred'])
                    else:
                        score = 0.0
                
                print(f"  [n={n_comp}] Res={res if method=='leiden' else 'N/A'} | K={n_found} | {eval_metric.upper()}={score:.4f}")

                # --- æ‹©ä¼˜ ---
                if score > best_score:
                    best_score = score
                    best_n_comp = n_comp
                    best_labels = temp_labels
                
        except Exception as e:
            print(f"  [n={n_comp}] Skipped: {e}")
            continue

    if best_labels is None:
        raise RuntimeError(f"Auto-clustering failed for all n_components. Please check resolution range or data.")

    print(f"\n>>> Winner: n_components={best_n_comp}, {eval_metric.upper()}={best_score:.4f}")
    adata.obs['domain'] = best_labels
    
    pca = PCA(n_components=min(best_n_comp, X_raw.shape[1]), random_state=42)
    adata.obsm['emb_pca'] = pca.fit_transform(X_raw)
    
    return adata, best_n_comp



def clustering(adata, n_clusters=7, radius=50, n_components=20, key='emb', 
               method='mclust', start=0.1, end=1, increment=0.01, 
               refinement=False, spatial_weight=0.0):
    """
    Unified Spatial Clustering function.
    
    Parameters
    ----------
    adata : anndata
        AnnData object.
    n_clusters : int
        Target number of clusters.
    radius : int
        Radius for spatial refinement (label smoothing).
    n_components : int
        Number of PCA components to use.
    key : str
        Key in adata.obsm to use as input features (default: 'emb').
    method : str
        Clustering method ('mclust', 'leiden').
    spatial_weight : float
        Weight of spatial coordinates injection. 
        - If 0.0: Pure gene expression clustering (Original behavior).
        - If > 0.0: Injects normalized spatial coordinates into embeddings. 
          Recommended 0.5 - 2.0 for DLPFC to force parallel/smooth boundaries.
    """
    
    # 1. Prepare Feature Data
    X_data = adata.obsm[key].copy()
    
    # 2. Spatial Prior Injection Logic
    if spatial_weight > 0:
        print(f"Running clustering with Spatial Prior (weight={spatial_weight})...")
        # Get spatial coordinates
        X_spatial = adata.obsm['spatial'].copy()
        
        # Normalize coordinates (Crucial, otherwise coordinate values might dominate features)
        scaler = StandardScaler()
        X_spatial = scaler.fit_transform(X_spatial)
        
        # Weight and concatenate: [Features, weight * (x, y)]
        X_spatial_weighted = X_spatial * spatial_weight
        X_data = np.concatenate([X_data, X_spatial_weighted], axis=1)
    else:
        print("Running standard clustering ...")

    # 3. Dimensionality Reduction (PCA)
    # Perform PCA on either original or concatenated features to learn 
    # the best combination of gene features and spatial location.
    pca = PCA(n_components=n_components, random_state=42)
    
    # Prevent case where feature count is less than n_components
    if X_data.shape[1] > n_components:
        embedding = pca.fit_transform(X_data)
    else:
        embedding = X_data
        
    adata.obsm['emb_pca'] = embedding

    # 4. Execute Clustering
    if method == 'mclust':
       adata = mclust_R(adata, used_obsm='emb_pca', num_cluster=n_clusters)
       adata.obs['domain'] = adata.obs['mclust']
       
    elif method == 'leiden':
       res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
       sc.tl.leiden(adata, random_state=0, resolution=res)
       adata.obs['domain'] = adata.obs['leiden']
       
    # 5. Post-processing Refinement (Optional)
    if refinement:  
       print(f"Applying refinement with radius={radius}...")
       new_type = refine_label(adata, radius, key='domain')
       adata.obs['domain'] = new_type

    return adata
# ==============================================================================
# 2. è¾…åŠ©å‡½æ•°ï¼šRefinement
# ==============================================================================

def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []

    old_type = adata.obs[key].values
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = [old_type[index[j]] for j in range(1, n_neigh + 1)]
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    return np.array([str(i) for i in new_type])




def extract_top_value(map_matrix, retain_percent = 0.1):
    '''Filter out cells with low mapping probability'''
    top_k  = retain_percent * map_matrix.shape[1]
    output = map_matrix * (np.argsort(np.argsort(map_matrix)) >= map_matrix.shape[1] - top_k)
    return output

def construct_cell_type_matrix(adata_sc):
    label = 'cell_type'
    n_type = len(list(adata_sc.obs[label].unique()))
    zeros = np.zeros([adata_sc.n_obs, n_type])
    cell_type = list(adata_sc.obs[label].unique())
    cell_type = [str(s) for s in cell_type]
    cell_type.sort()
    mat = pd.DataFrame(zeros, index=adata_sc.obs_names, columns=cell_type)
    for cell in list(adata_sc.obs_names):
        ctype = adata_sc.obs.loc[cell, label]
        mat.loc[cell, str(ctype)] = 1
    return mat

    
def extract_top_k_and_renorm(M, k):
    M_new = np.zeros_like(M)
    for i in range(M.shape[0]):  # per spot
        idx = np.argsort(M[i])[-k:]
        M_new[i, idx] = M[i, idx]
        M_new[i] /= (M_new[i].sum() + 1e-8)
    return M_new


def project_cell_to_spot(adata, adata_sc, retain_percent=0.1):
    """
    Project scRNA-derived cell types onto spatial transcriptomics spots
    using learned mapping matrix.
    """

    assert 'map_matrix' in adata.obsm
    assert 'cell_type' in adata_sc.obs

    # --- 1. Mapping matrix (spot x cell) ---
    map_matrix = np.nan_to_num(adata.obsm['map_matrix'])

    # --- 2. Top-k filtering per spot ---
    k = max(1, int(retain_percent * map_matrix.shape[1]))
    map_matrix = extract_top_k_and_renorm(map_matrix, k)

    # --- 3. Cell type one-hot matrix (cell x type) ---
    cell_types = adata_sc.obs['cell_type'].astype(str)
    cell_type_order = sorted(cell_types.unique())
    matrix_cell_type = pd.get_dummies(cell_types)[cell_type_order].values

    # --- 4. Projection ---
    matrix_projection = map_matrix @ matrix_cell_type  # spot x cell_type

    # --- 5. Normalize per spot ---
    df_projection = pd.DataFrame(
        matrix_projection,
        index=adata.obs_names,
        columns=cell_type_order
    )

    df_projection = df_projection.div(
        df_projection.sum(axis=1).clip(lower=1e-6),
        axis=0
    )

    # --- 6. Save ---
    adata.obs[cell_type_order] = df_projection

    return adata


def standardize_labels(labels):
    try:
        labels_int = labels.astype(int)
        if labels_int.min() == 0:
            labels_int = labels_int + 1
        labels_str = labels_int.astype(str)
        unique_cats = sorted(labels_str.unique(), key=int)
        return pd.Categorical(labels_str, categories=unique_cats, ordered=True)
    except:
        return labels.astype(str).astype('category')




def calculate_metrics(adata_obs, pred_df, pred_col, embedding=None):
    """
    Calculates ARI, NMI.
    Optionally calculates SC (Silhouette) and CH (Calinski-Harabasz)
    if embedding is provided.

    Returns a dict of scores.
    """

    # Merge for GT alignment
    combined = pd.merge(
        adata_obs[['ground_truth']],
        pred_df[[pred_col]],
        left_index=True,
        right_index=True,
        how='inner'
    )

    # ------------------
    # ARI / NMI (external, require GT)
    # ------------------
    valid_gt = combined[combined['ground_truth'] != 'Unknown'].dropna()

    if len(valid_gt) > 0:
        ari = adjusted_rand_score(valid_gt['ground_truth'], valid_gt[pred_col])
        nmi = normalized_mutual_info_score(valid_gt['ground_truth'], valid_gt[pred_col])
    else:
        ari, nmi = None, None

    # ------------------
    # SC / CH (internal, optional)
    # ------------------
    sc_score, ch_score = None, None

    if embedding is not None:
        # Align indices
        common_indices = combined.index.intersection(adata_obs.index)

        if len(common_indices) >= 2:
            try:
                valid_indices = [adata_obs.index.get_loc(idx) for idx in common_indices]
                emb_subset = embedding[valid_indices]
                labels_subset = combined.loc[common_indices, pred_col]

                # Silhouette requires >1 cluster
                if labels_subset.nunique() > 1:
                    sc_score = silhouette_score(emb_subset, labels_subset)
                    ch_score = calinski_harabasz_score(emb_subset, labels_subset)
            except Exception:
                sc_score, ch_score = None, None

    return {
        'ARI': ari,
        'NMI': nmi,
        'SC': sc_score,
        'CH': ch_score
    }


def find_best_in_group(file_list, adata, root_dir, sample_id, embedding=None):
    best_ari = -1
    best_name = "None"
    best_labels = None
    best_filename = None
    best_metrics = {'ARI': None, 'NMI': None, 'SC': None, 'CH': None}

    for method_name, filename in file_list:
        path = f"{root_dir}/{sample_id}/results/{filename}"
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, index_col=0)
                col = df.columns[0]

                metrics = calculate_metrics(
                    adata_obs=adata.obs,
                    pred_df=df,
                    pred_col=col,
                    embedding=embedding
                )

                if metrics['ARI'] is not None and metrics['ARI'] > best_ari:
                    best_ari = metrics['ARI']
                    best_name = method_name
                    best_labels = df[col]
                    best_filename = filename
                    best_metrics = metrics

            except Exception:
                pass

    return best_metrics, best_name, best_labels, best_filename


def standardize_labels(labels):
    try:
        labels_int = labels.astype(int)
        if labels_int.min() == 0:
            labels_int = labels_int + 1
        labels_str = labels_int.astype(str)
        unique_cats = sorted(labels_str.unique(), key=int)
        return pd.Categorical(labels_str, categories=unique_cats, ordered=True)
    except:
        return labels.astype(str).astype('category')

def calculate_ari_strict(adata_obs, pred_df, pred_col):
    combined = pd.merge(adata_obs[['ground_truth']], pred_df[[pred_col]], 
                        left_index=True, right_index=True, how='inner')
    combined = combined.replace('Unknown', np.nan).dropna()
    if len(combined) == 0: return -1
    return adjusted_rand_score(combined['ground_truth'], combined[pred_col])



def calculate_ari(adata, pred_col):
    """Calculates ARI against ground_truth, handling Unknown/NaNs"""
    if 'ground_truth' not in adata.obs:
        return 0.0
    
    # Create temp dataframe aligned by index
    df = pd.DataFrame({
        'gt': adata.obs['ground_truth'],
        'pred': adata.obs[pred_col]
    })
    
    # Filter out Unknown/NaN from Ground Truth
    df = df[df['gt'] != 'Unknown']
    df = df.dropna()
    
    if len(df) == 0:
        return 0.0
        
    return adjusted_rand_score(df['gt'], df['pred'])