import torch
import dgl as dgl
from torch import nn
from dgl import nn as dglnn


class DGLGCNModel(nn.Module):
    def __init__(self,
                 in_features: int,
                 gnn_features: int,
                 out_features: int):
        nn.Module.__init__(self)
        self.i2h = dglnn.GraphConv(
            in_features, gnn_features,
            norm='right', bias=True,
            allow_zero_in_degree=True
        )
        self.h2o = dglnn.GraphConv(
            gnn_features, out_features,
            norm='right', bias=True,
            allow_zero_in_degree=True
        )
        self.activation = nn.Sequential(
            nn.Dropout(0.5),
            nn.ReLU(),
        )

    def forward(self, block, x):
        x = self.i2h(block, x)
        x = self.activation(x)
        x = self.h2o(block, x)
        return x


class DGLGATModel(nn.Module):
    def __init__(self,
                 in_features: int,
                 gnn_features: int,
                 out_features: int,
                 n_heads: int = 8):
        nn.Module.__init__(self)
        self.i2h = dglnn.GATConv(
            in_features, gnn_features,
            n_heads, activation=None, bias=True,
            allow_zero_in_degree=True
        )
        self.h2o = dglnn.GATConv(
            gnn_features, out_features,
            n_heads, activation=None, bias=True,
            allow_zero_in_degree=True
        )
        self.activation = nn.Sequential(
            nn.Dropout(0.5),
            nn.ELU()
        )

    def forward(self, block, x):
        x = torch.mean(
            self.i2h(block, x),
            dim=1
        )
        x = self.activation(x)
        x = torch.mean(
            self.h2o(block, x),
            dim=1
        )
        return x


class DGLREmbedding(nn.Module):
    def __init__(self,
                 g: dgl.DGLHeteroGraph,
                 embedding_dim: int):
        nn.Module.__init__(self)
        self.embeds = nn.ModuleDict({
            nty: nn.Embedding(
                g.num_nodes(nty),
                embedding_dim
            )
            for nty in g.ntypes
        })

    def forward(self, block):
        return {
            nty: self.embeds[
                nty
            ].weight
            for nty in sorted(list(block.ntypes))
        }


class DGLRGCNModel(nn.Module):
    def __init__(self,
                 g: dgl.DGLHeteroGraph,
                 in_features: int,
                 gnn_features: int,
                 out_features: int):
        nn.Module.__init__(self)
        #
        self.embed = DGLREmbedding(
            g, embedding_dim=in_features
        )
        self.i2h = dglnn.HeteroGraphConv(
            {
                ety: dglnn.GraphConv(
                    in_features, gnn_features,
                    activation=None, bias=True,
                    allow_zero_in_degree=True
                )
                for ety in sorted(list(g.etypes))
            }, aggregate='mean'
        )
        self.h2o = dglnn.HeteroGraphConv(
            {
                ety: dglnn.GraphConv(
                    gnn_features, out_features,
                    activation=None, bias=True,
                    allow_zero_in_degree=True
                )
                for ety in sorted(list(g.etypes))
            }, aggregate='mean'
        )
        self.activation = nn.Sequential(
            nn.Dropout(0.5),
            nn.ReLU(),
        )

    def forward(self, block):
        xs = self.embed(block)
        hs = self.i2h(block, xs)
        hs = {
            k: self.activation(h)
            if h.dim() == 2 else
            self.activation(
                torch.mean(h, dim=1)
            )
            for k, h in hs.items()
        }
        hs = self.h2o(block, hs)
        hs = {
            k: h
            if h.dim() == 2 else
            torch.mean(h, dim=1)
            for k, h in hs.items()
        }
        return hs


class DGLRGATModel(nn.Module):
    def __init__(self,
                 g: dgl.DGLHeteroGraph,
                 in_features: int,
                 gnn_features: int,
                 out_features: int,
                 n_heads: int = 8):
        nn.Module.__init__(self)
        #
        self.embed = DGLREmbedding(
            g, embedding_dim=in_features
        )
        self.i2h = dglnn.HeteroGraphConv(
            {
                ety: dglnn.GATConv(
                    in_features, gnn_features,
                    n_heads, activation=None, bias=True,
                    allow_zero_in_degree=True
                )
                for ety in sorted(list(g.etypes))
            }, aggregate='mean'
        )
        self.h2o = dglnn.HeteroGraphConv(
            {
                ety: dglnn.GATConv(
                    gnn_features, out_features,
                    n_heads, activation=None, bias=True,
                    allow_zero_in_degree=True
                )
                for ety in sorted(list(g.etypes))
            }, aggregate='mean'
        )
        self.activation = nn.Sequential(
            nn.Dropout(0.5),
            nn.ReLU(),
        )

    def forward(self, block):
        xs = self.embed(block)
        hs = self.i2h(block, xs)
        hs = {
            k: self.activation(
                torch.mean(h, dim=1)
            )
            for k, h in hs.items()
        }
        hs = self.h2o(block, hs)
        hs = {
            k: torch.mean(h, dim=1)
            for k, h in hs.items()
        }
        return hs
