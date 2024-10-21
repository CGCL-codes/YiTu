import torch
from torch import nn
from xgnn_h import mp
from typing import Union


class GCNLayer(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        nn.Module.__init__(self)
        #
        self.fc = nn.Linear(in_features, out_features)

    def forward(self,
                graph: mp.Graph,
                norm: torch.Tensor,
                x: Union[torch.Tensor, list]):
        if isinstance(x, (list, tuple)):
            assert len(x) == 2
            x = x[0]
        h = self.fc(x)
        h = h.view(size=[
            -1, 1, h.size(-1)
        ])
        graph.src_node['u'] = h
        graph.message_func(mp.Fn.copy_u('u', 'm'))
        graph.reduce_func(mp.Fn.aggregate_sum('m', 'v'))
        h = torch.squeeze(graph.dst_node['v'], dim=1)
        return torch.multiply(norm, h)


class GCNModel(nn.Module):
    def __init__(self,
                 in_features: int,
                 gnn_features: int,
                 out_features: int):
        nn.Module.__init__(self)
        #
        self.i2h = GCNLayer(
            in_features, gnn_features
        )
        self.h2o = GCNLayer(
            gnn_features, out_features
        )
        self.activation = nn.Sequential(
            nn.Dropout(0.5),
            nn.ReLU(),
        )

    def forward(self,
                graph: mp.Graph,
                x: torch.Tensor,
                norm: torch.Tensor):
        h = self.i2h(graph, norm, x)
        h = self.activation(h)
        h = self.h2o(graph, norm, h)
        return h


class GATLayer(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 n_heads: int):
        nn.Module.__init__(self)
        #
        self.n_heads = n_heads
        self.n_features = out_features
        self.linear_q = nn.Linear(n_heads * out_features, n_heads)
        self.linear_k = nn.Linear(n_heads * out_features, n_heads)
        self.linear_v = nn.Linear(in_features, n_heads * out_features)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, graph: mp.Graph, x):
        # different to bert
        if isinstance(x, (list, tuple)):
            assert len(x) == 2
            src_size = x[0].size(0)
            dst_size = x[1].size(0)
            graph.blk.size[0] == src_size
            graph.blk.size[1] == dst_size
            #
            h_src = self.linear_v(x[0])
            h_dst = self.linear_v(x[1])
        elif isinstance(x, torch.Tensor):
            h_src = h_dst = self.linear_v(x)
        else:
            raise TypeError
        q = self.linear_q(h_dst)
        k = self.linear_k(h_src)
        h_src = h_src.view(size=[
            -1, self.n_heads,
            self.n_features
        ])
        graph.src_node['q'] = q
        graph.dst_node['k'] = k
        graph.src_node['u'] = h_src

        # gat attention
        graph.message_func(mp.Fn.u_add_v('k', 'q', 'e'))
        graph.edge['coeff'] = self.leaky_relu(graph.edge['e'])
        graph.message_func(mp.Fn.edge_softmax('coeff', 'attn'))
        graph.message_func(mp.Fn.u_mul_e('u', 'attn', 'm'))
        graph.reduce_func(mp.Fn.aggregate_sum('m', 'v'))
        return torch.mean(graph.dst_node['v'], dim=1)


class GATModel(nn.Module):
    def __init__(self,
                 in_features: int,
                 gnn_features: int,
                 out_features: int,
                 n_heads: int = 8):
        nn.Module.__init__(self)
        #
        self.i2h = GATLayer(
            in_features, gnn_features,
            n_heads=n_heads
        )
        self.h2o = GATLayer(
            gnn_features, out_features,
            n_heads=n_heads
        )
        self.activation = nn.Sequential(
            nn.Dropout(0.5),
            nn.ELU()
        )

    def forward(self, graph: mp.Graph, x: torch.Tensor):
        h = self.i2h(graph, x)
        h = self.activation(h)
        h = self.h2o(graph, h)
        return h


class HeteroGraphConv(nn.Module):
    def __init__(self, convs: dict):
        nn.Module.__init__(self)
        self.convs = nn.ModuleDict(convs)

    def forward(self,
                hgraph: mp.HeteroGraph,
                xs: dict,
                norms: dict = None):
        counts = {}
        outputs = {}
        for (sty, ety, dty), graph \
                in hgraph:
            if norms:
                res = self.convs[ety](
                    graph,
                    norms[str((sty, ety, dty))],
                    (xs[sty], xs[dty])
                )
            else:
                res = self.convs[ety](
                    graph, (xs[sty], xs[dty])
                )
            if dty not in counts:
                counts[dty] = 1
            else:
                counts[dty] += 1
            if dty not in outputs:
                outputs[dty] = res
            else:
                exist = outputs[dty]
                outputs[dty] = exist + res
        for dty, res in outputs.items():
            outputs[dty] = res / counts[dty]
        return outputs


class REmbedding(nn.Module):
    def __init__(self,
                 hgraph: mp.HeteroGraph,
                 embedding_dim: int):
        nn.Module.__init__(self)
        self.embeds = nn.ModuleDict({
            nty: nn.Embedding(
                num, embedding_dim
            )
            for nty, num in hgraph.nty2num.items()
        })

    def forward(self,
                hgraph: mp.HeteroGraph,
                xs: dict):
        return {
            nty: self.embeds[
                nty
            ](xs[nty])
            for nty in hgraph.nty2num
        }


class RGCNModel(nn.Module):
    def __init__(self,
                 hgraph: mp.HeteroGraph,
                 in_features: int,
                 gnn_features: int,
                 out_features: int):
        nn.Module.__init__(self)
        #
        self.em = REmbedding(
            hgraph=hgraph,
            embedding_dim=in_features
        )
        self.i2h = HeteroGraphConv({
            ety: GCNLayer(
                in_features=in_features,
                out_features=gnn_features
            )
            for ety in hgraph.etypes
        })
        self.h2o = HeteroGraphConv({
            ety: GCNLayer(
                in_features=gnn_features,
                out_features=out_features
            )
            for ety in hgraph.etypes
        })
        self.activation = nn.Sequential(
            nn.Dropout(0.5),
            nn.ReLU()
        )

    def forward(self,
                hgraph: mp.HeteroGraph,
                norms: dict, xs: dict):
        xs = self.em(hgraph, xs)
        hs = self.i2h(hgraph, xs, norms)
        hs = {
            k: self.activation(h)
            for k, h in hs.items()
        }
        hs = self.h2o(hgraph, hs, norms)
        return hs


class RGATModel(nn.Module):
    def __init__(self,
                 hgraph: mp.HeteroGraph,
                 in_features: int,
                 gnn_features: int,
                 out_features: int,
                 n_heads: int = 8):
        nn.Module.__init__(self)
        #
        self.em = REmbedding(
            hgraph=hgraph,
            embedding_dim=in_features
        )
        self.i2h = HeteroGraphConv({
            ety: GATLayer(
                in_features=in_features,
                out_features=gnn_features,
                n_heads=n_heads
            )
            for ety in hgraph.etypes
        })
        self.h2o = HeteroGraphConv({
            ety: GATLayer(
                in_features=gnn_features,
                out_features=out_features,
                n_heads=n_heads
            )
            for ety in hgraph.etypes
        })
        self.activation = nn.Sequential(
            nn.Dropout(0.5),
            nn.ELU()
        )

    def forward(self,
                hgraph: mp.HeteroGraph,
                xs: dict):
        xs = self.em(hgraph, xs)
        hs = self.i2h(hgraph, xs)
        hs = {
            k: self.activation(h)
            for k, h in hs.items()
        }
        hs = self.h2o(hgraph, hs)
        return hs
