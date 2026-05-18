"""SPARROW training interface.

This module contains the high-level SPARROW class for spatial clustering,
optional A-to-I multimodal training, and optional deconvolution/mapping.
"""

import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from anndata import AnnData
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

try:
    from .model import Encoder, Encoder_map, Encoder_sc, Encoder_sparse
    from .preprocess import (
        add_contrastive_label,
        construct_interaction,
        construct_interaction_KNN,
        fix_seed,
        get_feature,
        permutation,
        preprocess,
        preprocess_adj,
        preprocess_adj_sparse,
        preprocess_atoi,
    )
except ImportError:
    from model import Encoder, Encoder_map, Encoder_sc, Encoder_sparse
    from preprocess import (
        add_contrastive_label,
        construct_interaction,
        construct_interaction_KNN,
        fix_seed,
        get_feature,
        permutation,
        preprocess,
        preprocess_adj,
        preprocess_adj_sparse,
        preprocess_atoi,
    )


class SPARROW:
    """High-level SPARROW model wrapper.

    Parameters
    ----------
    adata : anndata.AnnData
        Spatial transcriptomics AnnData object. It should contain spatial
        coordinates in ``adata.obsm['spatial']`` unless a graph has already
        been stored in ``adata.obsm['adj']`` and ``adata.obsm['graph_neigh']``.
    adata_sc : anndata.AnnData or None, default=None
        Single-cell reference AnnData object. Required when
        ``deconvolution=True``.
    adata_ai : anndata.AnnData or None, default=None
        A-to-I editing AnnData object. Required when ``use_ai=True``.
    use_gene : bool, default=True
        Whether to use gene expression features for clustering/refinement.
    use_ai : bool, default=False
        Whether to use A-to-I editing features together with gene expression.
    device : torch.device, default=torch.device('cpu')
        Device used for model training.
    learning_rate : float, default=0.001
        Learning rate for the spatial encoder and mapping model.
    learning_rate_sc : float, default=0.01
        Learning rate for the single-cell autoencoder in deconvolution mode.
    weight_decay : float, default=0.0
        Weight decay used by the Adam optimizer.
    epochs : int, default=600
        Number of epochs for training the spatial encoder.
    epochs_sc : int, default=600
        Number of epochs for training the single-cell autoencoder.
    epochs_map : int, default=1000
        Number of epochs for training the cell-to-spot mapping matrix.
    dim_input : int, default=3000
        Initial input feature dimension. This is updated automatically after
        feature construction.
    dim_output : int, default=64
        Latent embedding dimension of the spatial encoder.
    random_seed : int, default=42
        Random seed for reproducibility.
    alpha : float, default=10
        Weight for the feature reconstruction loss.
    beta : float, default=1
        Weight for the contrastive spatial loss.
    theta : float, default=0.1
        Reserved hyperparameter kept for API compatibility.
    lamda1 : float, default=10
        Weight for the mapping reconstruction loss in deconvolution mode.
    lamda2 : float, default=1
        Weight for the mapping contrastive loss in deconvolution mode.
    deconvolution : bool, default=False
        Whether to run the deconvolution workflow using spatial and single-cell
        gene expression data.
    datatype : str, default='10X'
        Spatial technology type. ``'Stereo'``, ``'Slide'``, and
        ``'BARISTAseq'`` use sparse adjacency handling; other values use dense
        adjacency handling.
    n_top_genes : int, default=3000
        Number of highly variable genes selected from spatial gene expression.
    n_top_ai : int, default=1000
        Number of highly variable A-to-I features selected from ``adata_ai``.
    dynamic_graph : bool, default=False
        Whether to update the graph during training using learned embeddings.
    dynamic_update_interval : int, default=100
        Number of epochs between dynamic graph updates.
    dynamic_k : int, default=10
        Number of nearest neighbors used for the learned feature graph.
    dynamic_start_epoch : int, default=200
        Epoch at which dynamic graph updates start.
    dynamic_alpha : float, default=0.5
        Dynamic graph fusion switch. Values greater than 0 preserve the initial
        spatial graph and add learned feature edges; values less than or equal
        to 0 use only the learned feature graph.
    """

    def __init__(
        self,
        adata,
        adata_sc=None,
        adata_ai=None,
        use_gene=True,
        use_ai=False,
        device=torch.device("cpu"),
        learning_rate=0.001,
        learning_rate_sc=0.01,
        weight_decay=0.0,
        epochs=600,
        epochs_sc=600,
        epochs_map=1000,
        dim_input=3000,
        dim_output=64,
        random_seed=42,
        alpha=10,
        beta=1,
        theta=0.1,
        lamda1=10,
        lamda2=1,
        deconvolution=False,
        datatype="10X",
        n_top_genes=3000,
        n_top_ai=1000,
        dynamic_graph=False,
        dynamic_update_interval=100,
        dynamic_k=10,
        dynamic_start_epoch=200,
        dynamic_alpha=0.5,
    ):
        self.adata = adata.copy()
        self.adata_sc = adata_sc.copy() if adata_sc is not None else None
        self.adata_ai = adata_ai.copy() if adata_ai is not None else None
        self.n_spot = self.adata.n_obs
        self.deconvolution = deconvolution
        self.device = device
        self.use_gene = use_gene
        self.use_ai = use_ai

        if self.deconvolution:
            if not self.use_gene:
                raise ValueError("'use_gene' must be True when 'deconvolution=True'.")
            if self.use_ai:
                raise ValueError("'use_ai' must be False when 'deconvolution=True'.")
            if self.adata_sc is None:
                raise ValueError("'adata_sc' is required when 'deconvolution=True'.")

        if self.adata_ai is not None and not self.use_ai:
            print("Note: adata_ai was provided. Set use_ai=True to use it.")

        self.dynamic_graph = dynamic_graph
        self.dynamic_update_interval = dynamic_update_interval
        self.dynamic_k = dynamic_k
        self.dynamic_start_epoch = dynamic_start_epoch
        self.dynamic_alpha = dynamic_alpha

        self.learning_rate = learning_rate
        self.learning_rate_sc = learning_rate_sc
        self.weight_decay = weight_decay
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
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.n_top_genes = n_top_genes
        self.n_top_ai = n_top_ai

        fix_seed(self.random_seed)
        self._initialize_graph()
        self._initialize_labels()
        self._initialize_features()
        self._initialize_model()

    def _initialize_graph(self):
        """Build and store the spatial graph tensors used by the encoder."""
        if "adj" not in self.adata.obsm.keys():
            if self.datatype in ["Stereo", "Slide", "BARISTAseq"]:
                construct_interaction_KNN(self.adata)
            else:
                construct_interaction(self.adata)

        self.initial_adj_np = self.adata.obsm["adj"].copy()
        self.adj = self.initial_adj_np
        self.graph_neigh = torch.FloatTensor(
            self.adata.obsm["graph_neigh"].copy() + np.eye(self.adj.shape[0])
        ).to(self.device)

        if self.datatype in ["Stereo", "Slide", "BARISTAseq"]:
            print("Building sparse matrix ...")
            self.adj_tensor = preprocess_adj_sparse(self.adj).to(self.device)
        else:
            adj_norm = preprocess_adj(self.adj)
            self.adj_tensor = torch.FloatTensor(adj_norm).to(self.device)

    def _initialize_labels(self):
        """Create contrastive labels and move them to the selected device."""
        if "label_CSL" not in self.adata.obsm.keys():
            add_contrastive_label(self.adata)
        self.label_CSL = torch.FloatTensor(self.adata.obsm["label_CSL"]).to(self.device)

    def _initialize_features(self):
        """Prepare model input features for clustering or deconvolution mode."""
        if self.deconvolution:
            print("Mode: Deconvolution (Gene Only).")
            if "feat" not in self.adata.obsm.keys():
                get_feature(self.adata, deconvolution=True)

            self.features = torch.FloatTensor(self.adata.obsm["feat"].copy()).to(self.device)
            self.dim_input = self.features.shape[1]

            self.feat_sp = self._to_dense_array(self.adata.X)
            self.feat_sc = self._to_dense_array(self.adata_sc.X)
            self.feat_sc = pd.DataFrame(self.feat_sc).fillna(0).values
            self.feat_sp = pd.DataFrame(self.feat_sp).fillna(0).values
            self.feat_sc = torch.FloatTensor(self.feat_sc).to(self.device)
            self.feat_sp = torch.FloatTensor(self.feat_sp).to(self.device)
            self.n_cell = self.adata_sc.n_obs
            self.features_a = torch.FloatTensor(self.adata.obsm["feat_a"].copy()).to(self.device)
        else:
            print("Mode: Clustering/Refinement.")
            self.construct_multimodal_features()
            feat_a_np = permutation(self.features.cpu().numpy())
            self.features_a = torch.FloatTensor(feat_a_np).to(self.device)

    def _initialize_model(self):
        """Initialize the encoder, contrastive loss, and optimizer."""
        if self.datatype in ["Stereo", "Slide", "BARISTAseq"]:
            print("Using Sparse Encoder.")
            self.model = Encoder_sparse(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
        else:
            self.model = Encoder(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)

        self.loss_CSL = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), self.learning_rate, weight_decay=self.weight_decay
        )

    @staticmethod
    def _to_dense_array(matrix):
        """Convert a dense or sparse matrix to a NumPy array.

        Parameters
        ----------
        matrix : array-like or scipy.sparse matrix
            Input matrix to convert.

        Returns
        -------
        numpy.ndarray
            Dense matrix representation.
        """
        if isinstance(matrix, (csc_matrix, csr_matrix)) or sp.issparse(matrix):
            return matrix.toarray()[:, ]
        return matrix[:, ]

    def construct_multimodal_features(self):
        """Construct gene and/or A-to-I feature matrices for model training.

        Parameters
        ----------
        None
            This method uses instance attributes including ``adata``,
            ``adata_ai``, ``use_gene``, ``use_ai``, ``n_top_genes``, and
            ``n_top_ai``.

        Returns
        -------
        None
            The combined feature matrix is stored in
            ``self.adata.obsm['feat_multimodal']`` and ``self.features``.
        """
        feature_list = []
        desc_list = []
        print("Constructing features...")

        if self.use_gene:
            print(f"- Processing Gene Expression (Top {self.n_top_genes})...")
            if "highly_variable" not in self.adata.var.keys():
                preprocess(self.adata, n_top_genes=self.n_top_genes)

            adata_vars = self.adata[:, self.adata.var["highly_variable"]]
            feat_gene = adata_vars.X.toarray() if sp.issparse(adata_vars.X) else adata_vars.X
            feature_list.append(feat_gene)
            desc_list.append("Gene")

        if self.use_ai:
            if self.adata_ai is None:
                raise ValueError("use_ai=True but adata_ai is None.")

            print(f"- Processing A-to-I Editing (Top {self.n_top_ai})...")
            adata_ai_processed = preprocess_atoi(self.adata_ai, n_top_features=self.n_top_ai)
            feat_ai = (
                adata_ai_processed.X.toarray()
                if sp.issparse(adata_ai_processed.X)
                else adata_ai_processed.X
            )
            feature_list.append(feat_ai)
            desc_list.append("A-to-I")

        if not feature_list:
            raise ValueError("No features selected. Set use_gene=True and/or use_ai=True.")

        feat_final = np.concatenate(feature_list, axis=1)
        mode_str = " + ".join(desc_list)
        print(f"Final Input: [{mode_str}] | Dimension: {feat_final.shape}")

        self.dim_input = feat_final.shape[1]
        self.adata.obsm["feat_multimodal"] = feat_final
        self.features = torch.FloatTensor(feat_final).to(self.device)

    def update_graph(self, embedding):
        """Update the training graph using learned embeddings.

        Parameters
        ----------
        embedding : numpy.ndarray
            Learned spot embedding matrix with shape ``n_spots x n_features``.
            A KNN graph is built from this matrix and then optionally fused
            with the initial spatial graph.

        Returns
        -------
        None
            The normalized graph tensor is stored in ``self.adj_tensor``.
        """
        n_neighbors = self.dynamic_k
        n_spot = embedding.shape[0]

        emb_norm = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-12)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(emb_norm)
        _, indices = nbrs.kneighbors(emb_norm)

        x = indices[:, 0].repeat(n_neighbors)
        y = indices[:, 1:].flatten()

        interaction = np.zeros([n_spot, n_spot])
        interaction[x, y] = 1
        adj_feat = interaction + interaction.T
        adj_feat = np.where(adj_feat > 1, 1, adj_feat)

        if self.dynamic_alpha > 0:
            adj_combined = self.initial_adj_np + adj_feat
            adj_combined = np.where(adj_combined > 0, 1, 0)
        else:
            adj_combined = adj_feat

        if self.datatype in ["Stereo", "Slide", "BARISTAseq"]:
            self.adj_tensor = preprocess_adj_sparse(adj_combined).to(self.device)
        else:
            adj_norm = preprocess_adj(adj_combined)
            self.adj_tensor = torch.FloatTensor(adj_norm).to(self.device)

    def train(self, mask_rate=0.0, visible_rate=1):
        """Train the spatial encoder.

        Parameters
        ----------
        mask_rate : float, default=0.0
            Fraction of input feature values randomly masked during training.
            ``0.0`` uses standard reconstruction; values greater than 0 enable
            masked autoencoder-style reconstruction.
        visible_rate : float, default=1
            Weight for reconstruction loss on visible entries when
            ``mask_rate > 0``.

        Returns
        -------
        anndata.AnnData or torch.Tensor
            In clustering/refinement mode, returns ``self.adata`` with the
            learned embedding stored in ``adata.obsm['emb']``. In
            deconvolution mode, returns the reconstructed spatial embedding as
            a torch tensor.
        """
        mode_str = "Standard" if mask_rate == 0 else f"MAE (Rate={mask_rate})"
        print(f"Begin to train SPARROW [{mode_str}]...")

        if self.dynamic_graph:
            print(
                "Dynamic Graph: ENABLED | "
                f"Start Ep: {self.dynamic_start_epoch} | "
                f"Interval: {self.dynamic_update_interval}"
            )
            print("Strategy: Spatial-feature fusion with preserved spatial prior")

        self.model.train()

        for epoch in tqdm(range(self.epochs)):
            self.model.train()

            if self.dynamic_graph and epoch >= self.dynamic_start_epoch:
                if (epoch - self.dynamic_start_epoch) % self.dynamic_update_interval == 0:
                    self.model.eval()
                    with torch.no_grad():
                        _, z_raw, _, _ = self.model(self.features, self.features, self.adj_tensor)
                        z_np = z_raw.cpu().numpy()
                    self.update_graph(z_np)
                    self.model.train()

            if mask_rate > 0:
                mask = torch.rand(self.features.shape).to(self.device)
                mask = (mask > mask_rate).float()
                masked_input = self.features * mask
            else:
                mask = None
                masked_input = self.features

            self.features_a = permutation(self.features)
            self.hiden_feat, self.emb, ret, ret_a = self.model(
                masked_input, self.features_a, self.adj_tensor
            )

            self.loss_sl_1 = self.loss_CSL(ret, self.label_CSL)
            self.loss_sl_2 = self.loss_CSL(ret_a, self.label_CSL)

            if mask_rate > 0:
                recon_error = (self.emb - self.features) ** 2
                mask_indices = 1 - mask
                loss_mask = (recon_error * mask_indices).sum() / (mask_indices.sum() + 1e-8)
                loss_visible = (recon_error * mask).sum() / (mask.sum() + 1e-8)
                self.loss_feat = loss_mask + visible_rate * loss_visible
            else:
                self.loss_feat = F.mse_loss(self.features, self.emb)

            loss = self.alpha * self.loss_feat + self.beta * (self.loss_sl_1 + self.loss_sl_2)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print("Optimization finished!")

        with torch.no_grad():
            self.model.eval()
            if self.deconvolution:
                self.emb_rec = self.model(self.features, self.features_a, self.adj_tensor)[1]
                return self.emb_rec

            self.emb_rec = self.model(self.features, self.features_a, self.adj_tensor)[1]
            if self.datatype in ["Stereo", "Slide", "BARISTAseq"]:
                self.emb_rec = F.normalize(self.emb_rec, p=2, dim=1).detach().cpu().numpy()
            else:
                self.emb_rec = self.emb_rec.detach().cpu().numpy()

            self.adata.obsm["emb"] = self.emb_rec
            return self.adata

    def train_sc(self):
        """Train the single-cell autoencoder used in deconvolution mode.

        Parameters
        ----------
        None
            Uses ``self.feat_sc``, ``self.dim_input``, ``self.dim_output``,
            ``self.learning_rate_sc``, and ``self.epochs_sc``.

        Returns
        -------
        torch.Tensor
            Reconstructed single-cell feature embedding.
        """
        self.model_sc = Encoder_sc(self.dim_input, self.dim_output).to(self.device)
        self.optimizer_sc = torch.optim.Adam(self.model_sc.parameters(), lr=self.learning_rate_sc)

        for _ in tqdm(range(self.epochs_sc)):
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
        """Train the cell-to-spot mapping model for deconvolution.

        Parameters
        ----------
        mask_rate : float, default=0.0
            Fraction of spatial input feature values masked when training the
            spatial encoder.
        visible_rate : float, default=1
            Weight for visible-entry reconstruction when ``mask_rate > 0``.

        Returns
        -------
        tuple[anndata.AnnData, anndata.AnnData]
            Spatial and single-cell AnnData objects with embeddings saved in
            ``adata.obsm['emb_sp']`` and ``adata_sc.obsm['emb_sc']``. The
            mapping matrix is stored in ``adata.obsm['map_matrix']``.
        """
        emb_sp = self.train(mask_rate=mask_rate, visible_rate=visible_rate)
        if isinstance(emb_sp, AnnData):
            emb_sp = torch.FloatTensor(emb_sp.obsm["emb"]).to(self.device)

        emb_sc = self.train_sc()
        emb_sp = F.normalize(emb_sp, p=2, eps=1e-12, dim=1)
        emb_sc = F.normalize(emb_sc, p=2, eps=1e-12, dim=1)

        self.model_map = Encoder_map(self.n_cell, self.n_spot).to(self.device)
        self.optimizer_map = torch.optim.Adam(
            self.model_map.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        for _ in tqdm(range(self.epochs_map)):
            self.model_map.train()
            self.map_matrix = self.model_map()
            loss_recon, loss_nce = self.loss(emb_sp, emb_sc)
            loss = self.lamda1 * loss_recon + self.lamda2 * loss_nce
            self.optimizer_map.zero_grad()
            loss.backward()
            self.optimizer_map.step()

        with torch.no_grad():
            self.model_map.eval()
            emb_sp_np = emb_sp.cpu().numpy()
            emb_sc_np = emb_sc.cpu().numpy()
            map_matrix = F.softmax(self.map_matrix, dim=1).cpu().numpy()
            self.adata.obsm["emb_sp"] = emb_sp_np
            self.adata_sc.obsm["emb_sc"] = emb_sc_np
            self.adata.obsm["map_matrix"] = map_matrix.T
            return self.adata, self.adata_sc

    def loss(self, emb_sp, emb_sc):
        """Compute mapping reconstruction and contrastive losses.

        Parameters
        ----------
        emb_sp : torch.Tensor
            Spatial spot embedding matrix with shape ``n_spots x n_features``.
        emb_sc : torch.Tensor
            Single-cell embedding matrix with shape ``n_cells x n_features``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Reconstruction loss and noise-contrastive loss.
        """
        map_probs = F.softmax(self.map_matrix, dim=1)
        self.pred_sp = torch.matmul(map_probs.t(), emb_sc)
        loss_recon = F.mse_loss(self.pred_sp, emb_sp, reduction="mean")
        loss_nce = self.Noise_Cross_Entropy(self.pred_sp, emb_sp)
        return loss_recon, loss_nce

    def Noise_Cross_Entropy(self, pred_sp, emb_sp):
        """Compute graph-aware noise-contrastive loss.

        Parameters
        ----------
        pred_sp : torch.Tensor
            Predicted spatial embedding reconstructed from mapped single cells.
        emb_sp : torch.Tensor
            Target spatial embedding.

        Returns
        -------
        torch.Tensor
            Scalar contrastive loss.
        """
        mat = self.cosine_similarity(pred_sp, emb_sp)
        k = torch.exp(mat).sum(axis=1) - torch.exp(torch.diag(mat, 0))
        p = torch.exp(mat)
        p = torch.mul(p, self.graph_neigh).sum(axis=1)
        ave = torch.div(p, k)
        loss = -torch.log(ave).mean()
        return loss

    def cosine_similarity(self, pred_sp, emb_sp):
        """Compute pairwise cosine similarity between two embedding matrices.

        Parameters
        ----------
        pred_sp : torch.Tensor
            First embedding matrix with shape ``n_spots x n_features``.
        emb_sp : torch.Tensor
            Second embedding matrix with shape ``n_spots x n_features``.

        Returns
        -------
        torch.Tensor
            Pairwise cosine similarity matrix with shape ``n_spots x n_spots``.
        """
        numerator = torch.matmul(pred_sp, emb_sp.T)
        norm_pred = torch.norm(pred_sp, p=2, dim=1)
        norm_sp = torch.norm(emb_sp, p=2, dim=1)
        denominator = (
            torch.matmul(norm_pred.reshape((pred_sp.shape[0], 1)), norm_sp.reshape((emb_sp.shape[0], 1)).T)
            - 5e-12
        )
        similarity = torch.div(numerator, denominator)
        if torch.any(torch.isnan(similarity)):
            similarity = torch.where(torch.isnan(similarity), torch.full_like(similarity, 0.4868), similarity)
        return similarity
