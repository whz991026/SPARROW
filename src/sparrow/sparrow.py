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


class SPARROW():
    def __init__(self, 
        adata, 
        adata_sc = None, 
        adata_ai = None,
        use_gene = True,
        use_ai = False,
        device= torch.device('cpu'),
        learning_rate=0.001,
        learning_rate_sc = 0.01,
        weight_decay=0.00,
        epochs=600, 
        epochs_sc=600,   
        epochs_map=1000,  
        dim_input=3000,
        dim_output=64, 
        random_seed = 42, 
        alpha = 10, 
        beta = 1, 
        theta = 0.1,
        lamda1 = 10, 
        lamda2 = 1,
        deconvolution = False, 
        datatype = '10X',
        n_top_genes = 3000,
        n_top_ai = 1000,
        dynamic_graph=False,
        dynamic_update_interval=100, 
        dynamic_k=10,
        dynamic_start_epoch=200,    
        dynamic_alpha=0.5         
    ):
        
            
        self.adata = adata.copy()
        self.adata_sc = adata_sc.copy() if adata_sc is not None else None
        self.adata_ai = adata_ai.copy() if adata_ai is not None else None
        self.n_spot = self.adata.n_obs 
        self.deconvolution = deconvolution
        self.device = device
        # Modality switches
        self.use_gene = use_gene
        self.use_ai = use_ai


        if self.deconvolution:
            if not self.use_gene:
                raise ValueError("Config Error: 'use_gene' must be True when 'deconvolution=True'.")
            if self.use_ai:
                raise ValueError("Config Error: 'use_ai' must be False when 'deconvolution=True'.")
        if self.use_ai and self.adata_ai is not None:
            warnings.warn(
                "Config Warning: 'use_ai=True' while 'adata_ai' is provided. "
                "This is allowed but not recommended. Please ensure this is intended.",
                UserWarning
            )
        # Dynamic Graph Parameters
        self.dynamic_graph = dynamic_graph
        self.dynamic_update_interval = dynamic_update_interval
        self.dynamic_k = dynamic_k
        self.dynamic_start_epoch = dynamic_start_epoch
        self.dynamic_alpha = dynamic_alpha
        
        if self.adata_ai is not None and not self.use_ai:
            print("Note: adata_ai provided. Set use_ai=True to use it.")
            
        
        self.learning_rate = learning_rate
        self.learning_rate_sc = learning_rate_sc
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.epochs = epochs
        self.epochs_sc = epochs_sc
        self.epochs_map = epochs_map
        self.random_seed = random_seed
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.datatype = datatype
        self.dim_output = dim_output
        
        self.n_top_genes = n_top_genes
        self.n_top_ai = n_top_ai
        
        fix_seed(self.random_seed)
        
        # 1. Graph
        if 'adj' not in self.adata.obsm.keys():
           if self.datatype in ['Stereo', 'Slide', 'BARISTAseq']: construct_interaction_KNN(self.adata)
           else: construct_interaction(self.adata)
        
        
        self.initial_adj_np = self.adata.obsm['adj'].copy()
        self.adj = self.initial_adj_np 
        
        self.graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0])).to(self.device)
        
        # PyTorch
        if self.datatype in ['Stereo', 'Slide', 'BARISTAseq']:
           print('Building sparse matrix ...')
           self.adj_tensor = preprocess_adj_sparse(self.adj).to(self.device)
        else: 
           self.adj_tensor = preprocess_adj(self.adj)
           self.adj_tensor = torch.FloatTensor(self.adj_tensor).to(self.device)
        
        # 2. Label
        if 'label_CSL' not in self.adata.obsm.keys():    
           add_contrastive_label(self.adata)
        self.label_CSL = torch.FloatTensor(self.adata.obsm['label_CSL']).to(self.device)
    
        # 3. Features
        if self.deconvolution:
            print("Mode: Deconvolution (Gene Only).")
            if 'feat' not in self.adata.obsm.keys(): get_feature(self.adata, deconvolution=True)
            self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
            self.dim_input = self.features.shape[1]
            
            # Setup scRNA
            if isinstance(self.adata.X, csc_matrix) or isinstance(self.adata.X, csr_matrix): self.feat_sp = self.adata.X.toarray()[:, ]
            else: self.feat_sp = self.adata.X[:, ]
            if isinstance(self.adata_sc.X, csc_matrix) or isinstance(self.adata_sc.X, csr_matrix): self.feat_sc = self.adata_sc.X.toarray()[:, ]
            else: self.feat_sc = self.adata_sc.X[:, ]
            self.feat_sc = pd.DataFrame(self.feat_sc).fillna(0).values
            self.feat_sp = pd.DataFrame(self.feat_sp).fillna(0).values
            self.feat_sc = torch.FloatTensor(self.feat_sc).to(self.device)
            self.feat_sp = torch.FloatTensor(self.feat_sp).to(self.device)
            self.n_cell = self.adata_sc.n_obs
            self.features_a = torch.FloatTensor(self.adata.obsm['feat_a'].copy()).to(self.device)
            
        else:
            print("Mode: Clustering/Refinement.")
            self.construct_multimodal_features()
            feat_np = self.features.cpu().numpy()
            feat_a_np = permutation(feat_np)
            self.features_a = torch.FloatTensor(feat_a_np).to(self.device)
    
        # 4. Model Selection
        if self.datatype in ['Stereo', 'Slide', 'BARISTAseq']:
            print("Using Sparse Encoder.")
            self.model = Encoder_sparse(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
        else:
            self.model = Encoder(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
        
        self.loss_CSL = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, weight_decay=self.weight_decay)

    def construct_multimodal_features(self):
        feature_list = []
        desc_list = []
        print("Constructing features...")
        if self.use_gene:
            print(f"- Processing Gene Expression (Top {self.n_top_genes})...")
            if 'highly_variable' not in self.adata.var.keys():
                preprocess(self.adata, n_top_genes=self.n_top_genes)
            adata_Vars = self.adata[:, self.adata.var['highly_variable']]
            if sp.issparse(adata_Vars.X): feat_gene = adata_Vars.X.toarray()
            else: feat_gene = adata_Vars.X
            feature_list.append(feat_gene)
            desc_list.append("Gene")
        if self.use_ai:
            if self.adata_ai is None: raise ValueError("use_ai=True but adata_ai is None!")
            print(f"- Processing A-to-I Editing (Top {self.n_top_ai})...")
            adata_ai_processed = preprocess_atoi(self.adata_ai, n_top_features=self.n_top_ai)
            adata_ai_vars =adata_ai_processed
            #adata_ai_vars = adata_ai_processed[:, adata_ai_processed.var['highly_variable']]
            if sp.issparse(adata_ai_vars.X): feat_ai = adata_ai_vars.X.toarray()
            else: feat_ai = adata_ai_vars.X
            feature_list.append(feat_ai)
            desc_list.append("AI")
        if not feature_list: raise ValueError("No features selected!")
        feat_final = np.concatenate(feature_list, axis=1)
        mode_str = " + ".join(desc_list)
        print(f"Final Input: [{mode_str}] | Dimension: {feat_final.shape}")
        self.dim_input = feat_final.shape[1]
        self.adata.obsm['feat_multimodal'] = feat_final
        self.features = torch.FloatTensor(feat_final).to(self.device)

    def update_graph(self, embedding):
        """
        æ”¹è¿›çš„ Dynamic Graph æ›´æ–°é€»è¾‘ï¼š
        Original Spatial Graph (A_s) + Learned Feature Graph (A_f)
        """
        n_neighbors = self.dynamic_k
        n_spot = embedding.shape[0]
        
        # 1. KNN (Feature Graph)
        emb_norm = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-12)
        
        nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(emb_norm)
        _, indices = nbrs.kneighbors(emb_norm)
        
        x = indices[:, 0].repeat(n_neighbors)
        y = indices[:, 1:].flatten()
        
        # adj
        interaction = np.zeros([n_spot, n_spot])
        interaction[x, y] = 1
        adj_feat = interaction
        adj_feat = adj_feat + adj_feat.T
        adj_feat = np.where(adj_feat > 1, 1, adj_feat) # äºŒå€¼åŒ–
        
        # 2. (Graph Fusion)
        # alpha * Spatial + (1-alpha) * Feature
        # self.initial_adj_np 
        
        if self.dynamic_alpha > 0:
            adj_combined = self.initial_adj_np + adj_feat
            adj_combined = np.where(adj_combined > 0, 1, 0)
            
        else:
            adj_combined = adj_feat 

        # 3. Tensor
        if self.datatype in ['Stereo', 'Slide', 'BARISTAseq']:
            self.adj_tensor = preprocess_adj_sparse(adj_combined).to(self.device)
        else:
            adj_norm = preprocess_adj(adj_combined)
            self.adj_tensor = torch.FloatTensor(adj_norm).to(self.device)

    def train(self, mask_rate=0.0, visible_rate=1):
        mode_str = "Standard" if mask_rate == 0 else f"MAE (Rate={mask_rate})"
        print(f'Begin to train SPARROW [{mode_str}]...')
        if self.dynamic_graph:
            print(f'Dynamic Graph: ENABLED | Start Ep: {self.dynamic_start_epoch} | Interval: {self.dynamic_update_interval}')
            print(f'Strategy: Spatial-Feature Fusion (Preserving Spatial Prior)')
        
        self.model.train()
        
        for epoch in tqdm(range(self.epochs)): 
            self.model.train()
            
            # --- Dynamic Graph Update Logic ---
            if self.dynamic_graph and epoch >= self.dynamic_start_epoch:
                if (epoch - self.dynamic_start_epoch) % self.dynamic_update_interval == 0:
                    self.model.eval()
                    with torch.no_grad():
                        # Embedding 
                        _, z_raw, _, _ = self.model(self.features, self.features, self.adj_tensor)
                        z_np = z_raw.cpu().numpy()
                    
                    
                    self.update_graph(z_np)
                    self.model.train()
            # ------------------------------------------

            # 1. Masking
            if mask_rate > 0:
                mask = torch.rand(self.features.shape).to(self.device)
                mask = (mask > mask_rate).float()
                masked_input = self.features * mask
            else:
                masked_input = self.features
    
            # 2. Forward (self.adj_tensor)
            self.features_a = permutation(self.features)
            self.hiden_feat, self.emb, ret, ret_a = self.model(masked_input, self.features_a, self.adj_tensor)
            
            # 3. Loss
            self.loss_sl_1 = self.loss_CSL(ret, self.label_CSL)
            self.loss_sl_2 = self.loss_CSL(ret_a, self.label_CSL)
            
            if mask_rate > 0:
                recon_error = (self.emb - self.features) ** 2
                mask_indices = 1 - mask
                loss_mask = (recon_error * mask_indices).sum() / (mask_indices.sum() + 1e-8)
                loss_visible = (recon_error * mask).sum() / (mask.sum() + 1e-8)
                self.loss_feat = loss_mask +  visible_rate * loss_visible
            else:
                self.loss_feat = F.mse_loss(self.features, self.emb)
            
            loss =  self.alpha * self.loss_feat + self.beta * (self.loss_sl_1 + self.loss_sl_2)
            
            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
        
    
        print("Optimization finished!")
        
        with torch.no_grad():
             self.model.eval()
             if self.deconvolution:
                self.emb_rec = self.model(self.features, self.features_a, self.adj_tensor)[1]
                return self.emb_rec
             else:  
                if self.datatype in ['Stereo', 'Slide', 'BARISTAseq']:
                   self.emb_rec = self.model(self.features, self.features_a, self.adj_tensor)[1]
                   self.emb_rec = F.normalize(self.emb_rec, p=2, dim=1).detach().cpu().numpy() 
                else:
                   self.emb_rec = self.model(self.features, self.features_a, self.adj_tensor)[1].detach().cpu().numpy()
                self.adata.obsm['emb'] = self.emb_rec
                return self.adata


    def train_sc(self):
        self.model_sc = Encoder_sc(self.dim_input, self.dim_output).to(self.device)
        self.optimizer_sc = torch.optim.Adam(self.model_sc.parameters(), lr=self.learning_rate_sc)  
        for epoch in tqdm(range(self.epochs_sc)):
            self.model_sc.train()
            emb = self.model_sc(self.feat_sc)
            loss = F.mse_loss(emb, self.feat_sc)
            self.optimizer_sc.zero_grad()
            loss.backward()
            self.optimizer_sc.step()
        with torch.no_grad():
            self.model_sc.eval()
            emb_sc = self.model_sc(self.feat_sc)
            return emb_sc
    
    def train_map(self, mask_rate=0.0, visible_rate=1):
        emb_sp = self.train(mask_rate=mask_rate, visible_rate=visible_rate) 
        if isinstance(emb_sp, sc.AnnData): 
            emb_sp = torch.FloatTensor(emb_sp.obsm['emb']).to(self.device)
        emb_sc = self.train_sc()
        emb_sp = F.normalize(emb_sp, p=2, eps=1e-12, dim=1)
        emb_sc = F.normalize(emb_sc, p=2, eps=1e-12, dim=1)
        self.model_map = Encoder_map(self.n_cell, self.n_spot).to(self.device)  
        self.optimizer_map = torch.optim.Adam(self.model_map.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        for epoch in tqdm(range(self.epochs_map)):
            self.model_map.train()
            self.map_matrix = self.model_map()
            loss_recon, loss_NCE = self.loss(emb_sp, emb_sc)
            loss = self.lamda1 * loss_recon + self.lamda2 * loss_NCE
            self.optimizer_map.zero_grad()
            loss.backward()
            self.optimizer_map.step()
        with torch.no_grad():
            self.model_map.eval()
            emb_sp = emb_sp.cpu().numpy()
            emb_sc = emb_sc.cpu().numpy()
            map_matrix = F.softmax(self.map_matrix, dim=1).cpu().numpy() 
            self.adata.obsm['emb_sp'] = emb_sp
            self.adata_sc.obsm['emb_sc'] = emb_sc
            self.adata.obsm['map_matrix'] = map_matrix.T
            return self.adata, self.adata_sc

    def loss(self, emb_sp, emb_sc):
        map_probs = F.softmax(self.map_matrix, dim=1)   
        self.pred_sp = torch.matmul(map_probs.t(), emb_sc)
        loss_recon = F.mse_loss(self.pred_sp, emb_sp, reduction='mean')
        loss_NCE = self.Noise_Cross_Entropy(self.pred_sp, emb_sp)
        return loss_recon, loss_NCE
        
    def Noise_Cross_Entropy(self, pred_sp, emb_sp):
        mat = self.cosine_similarity(pred_sp, emb_sp) 
        k = torch.exp(mat).sum(axis=1) - torch.exp(torch.diag(mat, 0))
        p = torch.exp(mat)
        p = torch.mul(p, self.graph_neigh).sum(axis=1)
        ave = torch.div(p, k)
        loss = - torch.log(ave).mean()
        return loss
    
    def cosine_similarity(self, pred_sp, emb_sp):
        M = torch.matmul(pred_sp, emb_sp.T)
        Norm_c = torch.norm(pred_sp, p=2, dim=1)
        Norm_s = torch.norm(emb_sp, p=2, dim=1)
        Norm = torch.matmul(Norm_c.reshape((pred_sp.shape[0], 1)), Norm_s.reshape((emb_sp.shape[0], 1)).T) + -5e-12
        M = torch.div(M, Norm)
        if torch.any(torch.isnan(M)):
           M = torch.where(torch.isnan(M), torch.full_like(M, 0.4868), M)
        return M

