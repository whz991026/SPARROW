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
warnings.filterwarnings("ignore")

def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def preprocess(adata, n_top_genes=3000):
    """Standard preprocessing for gene expression data"""
    print(f"Preprocessing Gene Expression: Selecting top {n_top_genes} genes...")
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)

def preprocess_atoi(adata_ai, n_top_features=1000, min_cells_with_G=3,alpha=0.5,beta=0.5):
    """
    Robust Preprocessing for A-to-I:
    1. Filter low-coverage sites.
    2. Fixed Beta-Binomial smoothing (alpha=0.5, beta=0.5).
    3. HVG selection (Seurat flavor).
    4. Z-score Scaling (Crucial for Neural Networks).
    """
    import numpy as np
    import scanpy as sc
    import scipy.sparse as sp
    
    print(f"Preprocessing A-to-I data (Robust Mode)...")

    if 'A' not in adata_ai.layers.keys() or 'G' not in adata_ai.layers.keys():
        raise ValueError("adata_ai must contain 'A' and 'G' in layers.")

    A = adata_ai.layers['A']
    G = adata_ai.layers['G']

    if sp.issparse(A): A = A.toarray()
    if sp.issparse(G): G = G.toarray()
    
    # ----------------------
    # 1. Robust Filtering
    # ----------------------
    # è‡³å°‘åœ¨ min_cells ä¸ªç»†èƒžä¸­æœ‰ç¼–è¾‘äº‹ä»¶
    cells_with_G = np.sum(G > 0, axis=0) 
    keep_sites = cells_with_G >= min_cells_with_G
    n_kept = np.sum(keep_sites)
    
    print(f"  Filtering: {adata_ai.n_vars} -> {n_kept} sites (min_cells_with_G={min_cells_with_G})")
    
    if n_kept == 0:
        raise ValueError("No sites passed filtering.")
    
    if n_kept < n_top_features:
        print(f"  Warning: Only {n_kept} sites kept. Using all of them.")
        n_top_features_use = n_kept
    else:
        n_top_features_use = n_top_features
    
    A = A[:, keep_sites]
    G = G[:, keep_sites]
    var_filtered = adata_ai.var.iloc[keep_sites].copy()
    
    # ----------------------
    # 2. Fixed Smoothing (Alpha=0.5, Beta=0.5)
    # ----------------------
    # è¿™æ˜¯æœ€ç¨³å¥çš„æ–¹æ³•ï¼Œä¸ä¼šå› ä¸ºæ–¹å·®å¼‚å¸¸è€ŒæŠ¥é”™
    alpha = alpha
    beta = beta
    coverage = A + G
    # (G + 0.5) / (Depth + 1.0)
    editing_level = (G + alpha) / (coverage + alpha + beta)
    
    # ----------------------
    # 3. Create AnnData
    # ----------------------
    adata_processed = sc.AnnData(X=editing_level)
    adata_processed.obs = adata_ai.obs.copy()
    adata_processed.var = var_filtered
    
    # ----------------------
    # 4. HVG & Scale (å…³é”®ä¼˜åŒ–)
    # ----------------------
    # ä½¿ç”¨ 'seurat' flavorï¼Œå®ƒå¯¹ Ratio/Log æ•°æ®æ¯” 'cell_ranger' æ›´å‹å¥½
    # cell_ranger æœŸæœ›çš„æ˜¯æ•´æ•° Count çš„å‡å€¼æ–¹å·®å…³ç³»
    print(f"  Selecting top {n_top_features_use} HVGs...")
    sc.pp.highly_variable_genes(
        adata_processed, 
        n_top_genes=n_top_features_use, 
        flavor='seurat', # æŽ¨èä¿®æ”¹ä¸º seurat
        subset=True
    )
    
    # === å…³é”®ï¼šæ ‡å‡†åŒ– ===
    # è®© A-to-I çš„åˆ†å¸ƒ(å‡å€¼0æ–¹å·®1)ä¸Ž Gene ä¸€è‡´ï¼Œ
    # è¿™æ · Attention Gate æ¯”è¾ƒä¸¤æ¡çº¿çš„æƒé‡æ—¶æ‰å…¬å¹³ã€‚
    print("  Scaling data (Z-score)...")
    sc.pp.scale(adata_processed, max_value=10)
    
    print(f"Final A-to-I shape: {adata_processed.shape}")
    
    return adata_processed
 
# ==========================================
# 2. Graph Construction and Common Utilities
# ==========================================

def construct_interaction(adata, n_neighbors=3):
    position = adata.obsm['spatial']
    distance_matrix = ot.dist(position, position, metric='euclidean')
    n_spot = distance_matrix.shape[0]
    adata.obsm['distance_matrix'] = distance_matrix
    interaction = np.zeros([n_spot, n_spot])
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1
    adata.obsm['graph_neigh'] = interaction
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj>1, 1, adj)
    adata.obsm['adj'] = adj

def construct_interaction_KNN(adata, n_neighbors=3):
    position = adata.obsm['spatial']
    n_spot = position.shape[0]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(position)
    _ , indices = nbrs.kneighbors(position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    interaction = np.zeros([n_spot, n_spot])
    interaction[x, y] = 1
    adata.obsm['graph_neigh'] = interaction
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj>1, 1, adj)
    adata.obsm['adj'] = adj

def preprocess_adj(adj):
    adj_normalized = normalize_adj(adj)+np.eye(adj.shape[0])
    return adj_normalized

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_adj_sparse(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def permutation(feature):
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]
    return feature_permutated

def add_contrastive_label(adata):
    n_spot = adata.n_obs
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
    adata.obsm['label_CSL'] = label_CSL

def get_feature(adata, deconvolution=False):
    """ Legacy function kept for single-mode compatibility """
    if deconvolution:
        adata_Vars = adata
    else:
        adata_Vars =  adata[:, adata.var['highly_variable']]
    
    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
       feat = adata_Vars.X.toarray()[:, ]
    else:
       feat = adata_Vars.X[:, ] 
    
    feat_a = permutation(feat)
    adata.obsm['feat'] = feat
    adata.obsm['feat_a'] = feat_a

def filter_with_overlap_gene(adata, adata_sc):
    if 'highly_variable' not in adata.var.keys():
       raise ValueError("'highly_variable' are not existed in adata!")
    else:    
       adata = adata[:, adata.var['highly_variable']]
       
    if 'highly_variable' not in adata_sc.var.keys():
       raise ValueError("'highly_variable' are not existed in adata_sc!")
    else:    
       adata_sc = adata_sc[:, adata_sc.var['highly_variable']]   
    
    # Refine `marker_genes` so that they are shared by both adatas
    genes = list(set(adata.var.index) & set(adata_sc.var.index))
    genes.sort()
    print('Number of overlap genes:', len(genes))
    
    adata.uns["overlap_genes"] = genes
    adata_sc.uns["overlap_genes"] = genes
    
    adata = adata[:, genes]
    adata_sc = adata_sc[:, genes]
    
    return adata, adata_sc
    

def sub_clustering(adata, target_class, n_sub_clusters, key='domain', method='mclust', radius=None, refinement=True, resolution=None, n_neighbors=10, n_components=20, spatial_weight=0.0):
    """
    Perform sub-clustering on a specific major cluster (Divide and Conquer) with optional Spatial Injection.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing global clustering results.
    target_class : str or int
        The label of the major cluster to subdivide.
    n_sub_clusters : int
        The target number of sub-clusters.
    key : str
        The column name storing the major cluster labels (default: 'domain').
    method : str
        'mclust' or 'leiden'.
    radius : int or None
        Refinement radius. Defaults to 10 if None.
    resolution : float or None
        Specific resolution for Leiden. If None, auto-search is used.
    n_neighbors : int
        Number of neighbors for Leiden graph construction.
    n_components : int
        Number of PCA components for feature reduction.
    spatial_weight : float
        Weight of spatial coordinates injection. 
        - If 0.0: Pure gene expression clustering.
        - If > 0.0: Injects normalized spatial coordinates into embeddings before PCA.
    """
    print(f"\n>>> Starting sub-clustering on cluster '{target_class}' (Target: {n_sub_clusters} sub-clusters)...")
    
    # 1. Extract subset
    subset_mask = adata.obs[key].astype(str) == str(target_class)
    if np.sum(subset_mask) == 0:
        print(f"Error: Cluster '{target_class}' not found in column '{key}'.")
        return adata
        
    adata_sub = adata[subset_mask].copy()
    print(f"    - Subset contains {adata_sub.n_obs} cells.")
    
    # Check cell count
    if adata_sub.n_obs < n_sub_clusters * 5:
        print(f"Warning: Too few cells, skipping sub-clustering.")
        return adata

    # 2. Prepare Features & Spatial Injection
    # ---------------------------------------------------------
    X_data = adata_sub.obsm['emb'].copy()
    
    if spatial_weight > 0:
        print(f"    - Injecting Spatial Prior (weight={spatial_weight})...")
        # Get spatial coordinates from the subset
        X_spatial = adata_sub.obsm['spatial'].copy()
        
        # Normalize coordinates (Critical for sub-regions to balance variance)
        scaler = StandardScaler()
        X_spatial = scaler.fit_transform(X_spatial)
        
        # Weight and Concatenate
        X_spatial_weighted = X_spatial * spatial_weight
        X_data = np.concatenate([X_data, X_spatial_weighted], axis=1)
    else:
        print("    - Using feature embeddings only.")
    # ---------------------------------------------------------

    # 3. PCA (on combined data)
    # Dynamically adjust PCA components based on available samples/features
    n_comps = min(n_components, X_data.shape[1], X_data.shape[0] - 1)
    pca = PCA(n_components=n_comps, random_state=42)
    adata_sub.obsm['emb_pca_sub'] = pca.fit_transform(X_data)
    
    # 4. Perform clustering
    sub_labels = None
    
    if method == 'mclust':
        try:
            adata_sub = mclust_R(adata_sub, num_cluster=n_sub_clusters, used_obsm='emb_pca_sub')
            sub_labels = adata_sub.obs['mclust'].astype(str).values
        except Exception as e:
            print(f"    - Mclust failed: {e}. Switching to Leiden.")
            method = 'leiden' # Fallback
            
    if method in ['leiden']:
        if resolution is not None:
            print(f"    - Using provided resolution: {resolution}")
            res = resolution
            sc.pp.neighbors(adata_sub, n_neighbors=n_neighbors, use_rep='emb_pca_sub')
        else:
            # Note: ensure search_res is defined in your scope
            res = search_res(adata_sub, n_clusters=n_sub_clusters, method=method, use_rep='emb_pca_sub', start=0.1, end=2.0, increment=0.05)
        
        if method == 'leiden':
            sc.tl.leiden(adata_sub, resolution=res)
            sub_labels = adata_sub.obs['leiden'].astype(str).values

    unique_labels = np.unique(sub_labels)
    if len(unique_labels) < 2:
        print(f"Warning: Sub-clustering failed to split (resulted in {len(unique_labels)} class). Keeping original labels.")
        return adata
    
    print(f"    - Preliminary split results: {unique_labels}")

    # 5. (Optional) Spatial refinement of sub-regions
    if refinement:
        if radius is None: radius = 10 
        print(f"    - Performing Refinement (Radius={radius})...")
        
        adata_sub.obs['temp_sub'] = sub_labels
        # Note: refine_label uses 'spatial' from adata_sub, which is correct for local distance
        refined_labels = refine_label(adata_sub, radius=radius, key='temp_sub')
        
        if len(set(refined_labels)) < len(unique_labels):
            print(f"Warning: Class count decreased after refinement. Radius ({radius}) might be too large.")
        
        sub_labels = np.array(refined_labels)
    
    # 6. Merge results
    target_col = f'{key}_refined'
    if target_col not in adata.obs.columns:
        adata.obs[target_col] = adata.obs[key].astype(str).values
    
    new_labels = adata.obs[target_col].astype(str).values
    
    # Construct new labels: e.g., 4_1, 4_2
    final_sub_labels = [f"{target_class}_{l}" for l in sub_labels]
    
    new_labels[subset_mask] = final_sub_labels
    
    adata.obs[target_col] = new_labels
    adata.obs[target_col] = adata.obs[target_col].astype('category')
    
    # Output statistics
    unique, counts = np.unique(new_labels, return_counts=True)
    stats = dict(zip(unique, counts))
    related_stats = {k: v for k, v in stats.items() if str(k).startswith(f"{target_class}_")}
    
    print(f"    - Final refined split stats for class {target_class}: {related_stats}")
    print(f"Done! Results saved in '{target_col}'.")
    
    return adata

def extract_boundary_as_new_cluster(adata, cluster_A, cluster_B, key='domain', 
                                    width=5, new_label='New_Layer', 
                                    show_plot=True, 
                                    selection_mode='all', 
                                    min_component_size=10):  
    """
    Identifies the physical boundary between two clusters and defines it as a new cluster.
    Supports selecting only the largest boundary or all disjoint boundaries.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    cluster_A, cluster_B : str
        IDs of the two clusters to find the boundary between.
    key : str
        Column name of the clustering labels (default: 'domain').
    width : int
        Boundary width. 
    new_label : str
        Name for the newly created cluster.
    show_plot : bool
        Whether to show the spatial plot.
    selection_mode : str
        - 'largest': Keep only the largest connected boundary component (removes everything else).
        - 'all': Keep all disjoint boundary components that are larger than `min_component_size`.
    min_component_size : int
        Threshold to filter out noise when selection_mode='all'.
        
    Returns
    -------
    adata : AnnData
        Updated adata with a new column.
    """
    print(f"[Advanced] Mining boundary between Cluster {cluster_A} and {cluster_B}...")
    print(f"    - Strategy: Keep {selection_mode} component(s)")
    
    # 1. Prepare data
    spatial = adata.obsm['spatial']
    labels = adata.obs[key].astype(str).values
    
    # 2. Build spatial neighbor graph (KNN)
    nbrs = NearestNeighbors(n_neighbors=width * 3).fit(spatial)
    distances, indices = nbrs.kneighbors(spatial)
    median_spacing = np.median(distances[:, 1])
    
    # 3. Identify boundary points (Preliminary filter)
    boundary_mask = np.zeros(adata.n_obs, dtype=bool)
    cluster_A = str(cluster_A)
    cluster_B = str(cluster_B)
    
    for i in range(adata.n_obs):
        current_label = labels[i]
        if current_label not in [cluster_A, cluster_B]:
            continue
            
        neighbor_indices = indices[i, 1:width+1]
        neighbor_labels = labels[neighbor_indices]
        neighbor_dists = distances[i, 1:width+1]
        
        is_boundary = False
        if current_label == cluster_A and cluster_B in neighbor_labels:
            is_boundary = True
        elif current_label == cluster_B and cluster_A in neighbor_labels:
            is_boundary = True
            
        # Guardrail 1: Distance Restriction
        if is_boundary:
            if np.mean(neighbor_dists) > median_spacing * 3.0: 
                is_boundary = False
        
        if is_boundary:
            boundary_mask[i] = True
            
    # --- Guardrail 2: Connectivity Check (Selection Logic) ---
    n_raw = np.sum(boundary_mask)
    if n_raw > 0:
        sub_idx = np.where(boundary_mask)[0]
        sub_coords = spatial[sub_idx]
        
        k_conn = min(6, len(sub_idx)-1)
        if k_conn > 0:
            nbrs_sub = NearestNeighbors(n_neighbors=k_conn).fit(sub_coords)
            graph = nbrs_sub.kneighbors_graph(sub_coords, mode='connectivity')
            
            # Find connected components
            n_cc, labels_cc = connected_components(graph)
            component_sizes = np.bincount(labels_cc)
            
            print(f"    - Found {n_cc} disconnected boundary segments.")
            
            # --- Logic Branch based on selection_mode ---
            if selection_mode == 'largest':
                # Original behavior: Only keep the single biggest chunk
                largest_cc_label = np.argmax(component_sizes)
                keep_local_mask = (labels_cc == largest_cc_label)
                
            elif selection_mode == 'all':
                # Keep ALL chunks that are big enough (not just noise)
                valid_labels = np.where(component_sizes >= min_component_size)[0]
                keep_local_mask = np.isin(labels_cc, valid_labels)
                
                dropped = n_cc - len(valid_labels)
                if dropped > 0:
                    print(f"    - Dropped {dropped} small noise segments (< {min_component_size} cells).")
            else:
                raise ValueError("selection_mode must be 'largest' or 'all'")
            
            # Apply filter
            noise_indices = sub_idx[~keep_local_mask]
            boundary_mask[noise_indices] = False

    n_found = np.sum(boundary_mask)
    print(f"    - Finalized {n_found} valid boundary cells.")
    
    if n_found == 0:
        print("Warning: No boundary found.")
        return adata

    # 4. Generate new label column
    new_col_name = f"{key}_refined" 
    new_labels = labels.copy()
    new_labels[boundary_mask] = new_label
    
    adata.obs[new_col_name] = new_labels
    adata.obs[new_col_name] = adata.obs[new_col_name].astype('category')
    
    # 5. Plotting
    if show_plot:
        plt.rcParams["figure.figsize"] = (6, 6)
        sc.pl.spatial(
            adata,
            color=new_col_name,
            title=f"Extracted '{new_label}' (Mode: {selection_mode})",
            spot_size=50,
            palette='tab20',
            frameon=False,
            show=True
        )
    
    print(f"Done! Results saved in column '{new_col_name}'.")
    return adata