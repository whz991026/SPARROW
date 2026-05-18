"""Neural network modules used by the SPARROW model.

This file keeps only the imports required by the model definitions and preserves
all original model interfaces for backward compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Discriminator(nn.Module):
    """Bilinear discriminator for contrastive learning.

    Parameters
    ----------
    n_h : int
        Hidden embedding dimension. Both input embeddings passed to the bilinear
        layer must have this feature dimension.
    """

    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for module in self.modules():
            self.weights_init(module)

    def weights_init(self, module):
        """Initialize bilinear layer parameters.

        Parameters
        ----------
        module : torch.nn.Module
            Module to initialize. Only ``nn.Bilinear`` modules are modified.

        Returns
        -------
        None
            The module is updated in place.
        """
        if isinstance(module, nn.Bilinear):
            torch.nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi):
        """Compute discriminator logits for positive and negative embeddings.

        Parameters
        ----------
        c : torch.Tensor
            Context embedding tensor. It is expanded to match ``h_pl``.
        h_pl : torch.Tensor
            Positive sample embeddings with shape ``(n_samples, n_h)``.
        h_mi : torch.Tensor
            Negative sample embeddings with shape ``(n_samples, n_h)``.

        Returns
        -------
        torch.Tensor
            Concatenated logits for positive and negative pairs with shape
            ``(n_samples, 2)``.
        """
        c_x = c.expand_as(h_pl)
        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)
        logits = torch.cat((sc_1, sc_2), dim=1)
        return logits


class AvgReadout(nn.Module):
    """Average neighborhood readout followed by L2 normalization."""

    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        """Aggregate embeddings using a neighborhood mask.

        Parameters
        ----------
        emb : torch.Tensor
            Node or spot embeddings with shape ``(n_nodes, n_features)``.
        mask : torch.Tensor
            Neighborhood mask or adjacency-like matrix with shape
            ``(n_nodes, n_nodes)``. Each row defines the nodes used to compute
            the local average for one output embedding.

        Returns
        -------
        torch.Tensor
            L2-normalized readout embeddings with shape
            ``(n_nodes, n_features)``.
        """
        if mask is None:
            raise ValueError("mask must be provided for AvgReadout.")

        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, dim=1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum
        return F.normalize(global_emb, p=2, dim=1)


class Encoder(nn.Module):
    """Dense graph autoencoder with contrastive discrimination.

    This encoder expects a dense adjacency matrix and uses ``torch.mm`` for graph
    propagation.

    Parameters
    ----------
    in_features : int
        Number of input features for each spot or cell.
    out_features : int
        Dimension of the hidden embedding.
    graph_neigh : torch.Tensor
        Dense neighborhood mask used by ``AvgReadout``.
    dropout : float, optional
        Dropout probability applied to the input features. Default is ``0.0``.
    act : callable, optional
        Activation function applied to the hidden embedding. Default is
        ``torch.nn.functional.relu``.
    """

    def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act

        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        self.reset_parameters()

        self.disc = Discriminator(self.out_features)
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()

    def reset_parameters(self):
        """Initialize trainable encoder weights with Xavier uniform values.

        Returns
        -------
        None
            Parameters are updated in place.
        """
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def forward(self, feat, feat_a, adj):
        """Run dense graph encoding and contrastive scoring.

        Parameters
        ----------
        feat : torch.Tensor
            Original feature matrix with shape ``(n_nodes, in_features)``.
        feat_a : torch.Tensor
            Augmented or permuted feature matrix with the same shape as
            ``feat``.
        adj : torch.Tensor
            Dense normalized adjacency matrix with shape ``(n_nodes, n_nodes)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            ``hiden_emb`` : hidden embedding before activation.
            ``h`` : reconstructed feature matrix.
            ``ret`` : discriminator logits using the original context.
            ``ret_a`` : discriminator logits using the augmented context.
        """
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = torch.mm(adj, z)
        hiden_emb = z

        h = torch.mm(z, self.weight2)
        h = torch.mm(adj, h)
        emb = self.act(z)

        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = torch.mm(adj, z_a)
        emb_a = self.act(z_a)

        g = self.read(emb, self.graph_neigh)
        g = self.sigm(g)
        g_a = self.read(emb_a, self.graph_neigh)
        g_a = self.sigm(g_a)

        ret = self.disc(g, emb, emb_a)
        ret_a = self.disc(g_a, emb_a, emb)

        return hiden_emb, h, ret, ret_a


class Encoder_sparse(nn.Module):
    """Sparse graph autoencoder with contrastive discrimination.

    This encoder expects a sparse adjacency matrix and uses ``torch.spmm`` for
    graph propagation.

    Parameters
    ----------
    in_features : int
        Number of input features for each spot or cell.
    out_features : int
        Dimension of the hidden embedding.
    graph_neigh : torch.Tensor
        Dense neighborhood mask used by ``AvgReadout``.
    dropout : float, optional
        Dropout probability applied to the input features. Default is ``0.0``.
    act : callable, optional
        Activation function applied to the hidden embedding. Default is
        ``torch.nn.functional.relu``.
    """

    def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.relu):
        super(Encoder_sparse, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act

        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        self.reset_parameters()

        self.disc = Discriminator(self.out_features)
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()

    def reset_parameters(self):
        """Initialize trainable encoder weights with Xavier uniform values.

        Returns
        -------
        None
            Parameters are updated in place.
        """
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def forward(self, feat, feat_a, adj):
        """Run sparse graph encoding and contrastive scoring.

        Parameters
        ----------
        feat : torch.Tensor
            Original feature matrix with shape ``(n_nodes, in_features)``.
        feat_a : torch.Tensor
            Augmented or permuted feature matrix with the same shape as
            ``feat``.
        adj : torch.Tensor
            Sparse normalized adjacency matrix with shape ``(n_nodes, n_nodes)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            ``hiden_emb`` : hidden embedding before activation.
            ``h`` : reconstructed feature matrix.
            ``ret`` : discriminator logits using the original context.
            ``ret_a`` : discriminator logits using the augmented context.
        """
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = torch.spmm(adj, z)
        hiden_emb = z

        h = torch.mm(z, self.weight2)
        h = torch.spmm(adj, h)
        emb = self.act(z)

        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = torch.spmm(adj, z_a)
        emb_a = self.act(z_a)

        g = self.read(emb, self.graph_neigh)
        g = self.sigm(g)
        g_a = self.read(emb_a, self.graph_neigh)
        g_a = self.sigm(g_a)

        ret = self.disc(g, emb, emb_a)
        ret_a = self.disc(g_a, emb_a, emb)

        return hiden_emb, h, ret, ret_a


class Encoder_sc(nn.Module):
    """Fully connected autoencoder for single-cell expression features.

    Parameters
    ----------
    dim_input : int
        Number of input features for each cell.
    dim_output : int
        Kept for backward compatibility with the original API. The current
        architecture uses a fixed bottleneck size of 32 and does not directly
        use this argument.
    dropout : float, optional
        Dropout probability applied to the input features. Default is ``0.0``.
    act : callable, optional
        Kept for backward compatibility with the original API. The original
        forward pass does not apply a nonlinear activation, so this argument is
        stored but not used.
    """

    def __init__(self, dim_input, dim_output, dropout=0.0, act=F.relu):
        super(Encoder_sc, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim1 = 256
        self.dim2 = 64
        self.dim3 = 32
        self.act = act
        self.dropout = dropout

        self.weight1_en = Parameter(torch.FloatTensor(self.dim_input, self.dim1))
        self.weight2_en = Parameter(torch.FloatTensor(self.dim1, self.dim2))
        self.weight3_en = Parameter(torch.FloatTensor(self.dim2, self.dim3))

        self.weight1_de = Parameter(torch.FloatTensor(self.dim3, self.dim2))
        self.weight2_de = Parameter(torch.FloatTensor(self.dim2, self.dim1))
        self.weight3_de = Parameter(torch.FloatTensor(self.dim1, self.dim_input))
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize all autoencoder weights with Xavier uniform values.

        Returns
        -------
        None
            Parameters are updated in place.
        """
        torch.nn.init.xavier_uniform_(self.weight1_en)
        torch.nn.init.xavier_uniform_(self.weight1_de)
        torch.nn.init.xavier_uniform_(self.weight2_en)
        torch.nn.init.xavier_uniform_(self.weight2_de)
        torch.nn.init.xavier_uniform_(self.weight3_en)
        torch.nn.init.xavier_uniform_(self.weight3_de)

    def forward(self, x):
        """Encode and reconstruct single-cell features.

        Parameters
        ----------
        x : torch.Tensor
            Input feature matrix with shape ``(n_cells, dim_input)``.

        Returns
        -------
        torch.Tensor
            Reconstructed feature matrix with shape ``(n_cells, dim_input)``.
        """
        x = F.dropout(x, self.dropout, self.training)
        x = torch.mm(x, self.weight1_en)
        x = torch.mm(x, self.weight2_en)
        x = torch.mm(x, self.weight3_en)
        x = torch.mm(x, self.weight1_de)
        x = torch.mm(x, self.weight2_de)
        x = torch.mm(x, self.weight3_de)
        return x


class Encoder_map(nn.Module):
    """Trainable cell-to-spot mapping matrix.

    Parameters
    ----------
    n_cell : int
        Number of single cells.
    n_spot : int
        Number of spatial spots.
    """

    def __init__(self, n_cell, n_spot):
        super(Encoder_map, self).__init__()
        self.n_cell = n_cell
        self.n_spot = n_spot
        self.M = Parameter(torch.FloatTensor(self.n_cell, self.n_spot))
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the mapping matrix with Xavier uniform values.

        Returns
        -------
        None
            The mapping matrix is updated in place.
        """
        torch.nn.init.xavier_uniform_(self.M)

    def forward(self):
        """Return the trainable mapping matrix.

        Parameters
        ----------
        None

        Returns
        -------
        torch.Tensor
            Mapping matrix with shape ``(n_cell, n_spot)``.
        """
        return self.M


__all__ = [
    "Discriminator",
    "AvgReadout",
    "Encoder",
    "Encoder_sparse",
    "Encoder_sc",
    "Encoder_map",
]
