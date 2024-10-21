from torch import nn
from torch_geometric import nn as pygnn


class PyGGCNModel(nn.Module):
    def __init__(self,
                 in_features: int,
                 gnn_features: int,
                 out_features: int):
        nn.Module.__init__(self)
        self.i2h = pygnn.GCNConv(
            in_features, gnn_features
        )
        self.h2o = pygnn.GCNConv(
            gnn_features, out_features
        )
        self.activation = nn.Sequential(
            nn.Dropout(0.5),
            nn.ReLU(),
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        x = self.i2h(x, edge_index)
        x = self.activation(x)
        x = self.h2o(x, edge_index)
        return x


class PyGGATModel(nn.Module):
    def __init__(self,
                 in_features: int,
                 gnn_features: int,
                 out_features: int,
                 n_heads: int = 8):
        nn.Module.__init__(self)
        self.i2h = pygnn.GATConv(
            in_features, gnn_features,
            n_heads, concat=False
        )
        self.h2o = pygnn.GATConv(
            gnn_features, out_features,
            n_heads, concat=False
        )
        self.activation = nn.Sequential(
            nn.Dropout(0.5),
            nn.ReLU(),
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        x = self.i2h(x, edge_index)
        x = self.activation(x)
        x = self.h2o(x, edge_index)
        return x


class PyGREmbedding(nn.Module):
    def __init__(self, g, embedding_dim: int):
        nn.Module.__init__(self)
        self.embeds = nn.ModuleDict({
            nty: nn.Embedding(
                g[nty].num_nodes,
                embedding_dim
            )
            for nty in g.node_types
        })

    def forward(self, g):
        return {
            nty: self.embeds[
                nty
            ].weight
            for nty in g.node_types
        }


class PyGRGCNModel(nn.Module):
    def __init__(self, graph,
                 in_features: int,
                 gnn_features: int,
                 out_features: int,
                 n_heads: int = 8):
        nn.Module.__init__(self)
        self.em = PyGREmbedding(
            graph, in_features
        )
        self.i2h = pygnn.HeteroConv(
            {
                ety: pygnn.GCNConv(
                    in_features, gnn_features
                ) for ety in graph.metadata()[1]
            }, aggr='mean'
        )
        self.h2o = pygnn.HeteroConv(
            {
                ety: pygnn.GCNConv(
                    gnn_features, out_features
                ) for ety in graph.metadata()[1]
            }, aggr='mean'
        )
        self.activation = nn.Sequential(
            nn.Dropout(0.5),
            nn.ReLU(),
        )

    def forward(self, data):
        x_dict = self.em(data)
        edge_sict = data.edge_index_dict
        x_dict = self.i2h(x_dict, edge_sict)
        x_dict = {
            k: self.activation(v)
            for k, v in x_dict.items()
        }
        x_dict = self.h2o(x_dict, edge_sict)
        return x_dict


class PyGRGATModel(nn.Module):
    def __init__(self, graph,
                 in_features: int,
                 gnn_features: int,
                 out_features: int,
                 n_heads: int = 8):
        nn.Module.__init__(self)
        self.em = PyGREmbedding(
            graph, in_features
        )
        self.i2h = pygnn.HeteroConv(
            {
                ety: pygnn.GATConv(
                    in_features, gnn_features,
                    n_heads, concat=False
                ) for ety in graph.metadata()[1]
            }, aggr='mean'
        )
        self.h2o = pygnn.HeteroConv(
            {
                ety: pygnn.GATConv(
                    gnn_features, out_features,
                    n_heads, concat=False
                ) for ety in graph.metadata()[1]
            }, aggr='mean'
        )
        self.activation = nn.Sequential(
            nn.Dropout(0.5),
            nn.ReLU(),
        )

    def forward(self, data):
        x_dict = self.em(data)
        edge_sict = data.edge_index_dict
        x_dict = self.i2h(x_dict, edge_sict)
        x_dict = {
            k: self.activation(v)
            for k, v in x_dict.items()
        }
        x_dict = self.h2o(x_dict, edge_sict)
        return x_dict
