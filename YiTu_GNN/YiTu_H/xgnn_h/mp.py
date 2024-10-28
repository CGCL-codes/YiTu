import torch
import xgnn_h
from torch import overrides
from xgnn_h import block


def message_wrapper(graph, func, **kwargs):
    n_heads = None
    for v in kwargs.values():
        assert v.dim() in [2, 3]
        if not n_heads:
            n_heads = v.size(1)
        assert n_heads == v.size(1)
    return torch.zeros(
        size=[graph.num_edges(), n_heads],
        device=graph.device()
    )


def reduce_wrapper(graph, func, **kwargs):
    n_heads = None
    for v in kwargs.values():
        assert v.dim() == 2
        if not n_heads:
            n_heads = v.size(1)
        assert n_heads == v.size(1)
    return torch.zeros(
        size=[graph.num_dst_nodes(), n_heads, graph.num_features()],
        device=graph.device())


class Fn:
    @classmethod
    def copy_u(cls, u, e):
        desc = {
            'func': 'copy_u',
            'u': u, 'out': e
        }
        return desc

    @classmethod
    def u_add_v(cls, u, v, e):
        desc = {
            'func': 'u_add_v',
            'u': u, 'v': v, 'out': e
        }
        return desc

    @classmethod
    def u_mul_e(cls, u, e, v):
        desc = {
            'func': 'u_mul_e',
            'u': u, 'e': e, 'out': v
        }
        return desc

    @classmethod
    def edge_softmax(cls, e1, e2):
        desc = {
            'func': 'edge_softmax',
            'e': e1, 'out': e2
        }
        return desc

    @classmethod
    def aggregate_sum(cls, e, v):
        desc = {
            'func': 'aggregate_sum',
            'e': e, 'out': v
        }
        return desc


class Graph:
    def __init__(self, blk: block.Block):
        self.blk = blk
        self.edge = dict()
        self.src_node = dict()
        self.dst_node = dict()

    def device(self):
        for v in self.src_node.values():
            return v.device
        raise RuntimeError

    def num_edges(self):
        return self.blk.num_edges()

    def num_src_nodes(self):
        return self.blk.num_src_nodes()

    def num_dst_nodes(self):
        return self.blk.num_dst_nodes()

    def num_features(self):
        # TODO: fix workaround
        n_features = None
        for v in self.src_node.values():
            if v.dim() != 3:
                continue
            if not n_features:
                n_features = v.size(-1)
            assert n_features == v.size(-1)
        assert n_features
        return n_features

    def right_norm(self):
        return self.blk.right_norm

    def _build_kwargs(self, desc: dict):
        kwargs = {}
        for k, v in desc.items():
            if k in ['func', 'out']:
                continue
            for data in [self.edge,
                         self.src_node,
                         self.dst_node]:
                if v not in data:
                    continue
                kwargs[k] = data[v]
        return kwargs

    def message_func(self, desc: dict):
        kwargs = self._build_kwargs(desc)
        self.edge[desc['out']] = \
            overrides.handle_torch_function(
            message_wrapper, kwargs.values(),
            self, desc['func'], **kwargs
        )

    def reduce_func(self, desc):
        kwargs = self._build_kwargs(desc)
        self.dst_node[desc['out']] = \
            overrides.handle_torch_function(
            reduce_wrapper, kwargs.values(),
            self, desc['func'], **kwargs
        )

    @staticmethod
    def from_dglgraph(graph):
        import dgl
        assert isinstance(
            graph, dgl.DGLGraph
        )
        assert graph.is_homogeneous

        #
        adj = graph.adj(
            transpose=True,
            scipy_fmt='csr'
        )
        blk = block.Block(
            size=[
                graph.num_nodes(),
                graph.num_nodes()
            ],
            adj=[
                torch.IntTensor(
                    adj.indptr
                ).to(graph.device),
                torch.IntTensor(
                    adj.indices
                ).to(graph.device)
            ],
            right_norm=torch.div(
                1.0, graph.in_degrees()
            ).unsqueeze(-1)
        )
        return Graph(blk)


class HeteroGraph:
    def __init__(self):
        self.device = None
        self.etypes = []
        self.nty2num = {}
        self.idx2rel = {}
        self.rel2idx = {}
        self.hetero_graph = {}

    def __iter__(self):
        return iter(self.hetero_graph.items())

    @staticmethod
    def from_dglgraph(graph):
        import dgl
        assert isinstance(
            graph, (
                dgl.DGLGraph,
                dgl.DGLHeteroGraph
            )
        )
        assert not graph.is_homogeneous
        hgraph = HeteroGraph()
        hgraph.device = graph.device
        for nty in graph.ntypes:
            hgraph.nty2num[
                nty
            ] = graph.num_nodes(nty)
        hgraph.etypes = list(
            graph.etypes
        )
        for sty, ety, dty in \
                graph.canonical_etypes:
            adj = graph.adj(
                transpose=True,
                scipy_fmt='csr',
                etype=(sty, ety, dty)
            )
            blk = block.Block(
                size=[
                    graph.num_nodes(dty),
                    graph.num_nodes(sty)
                ],
                adj=[
                    torch.IntTensor(
                        adj.indptr
                    ).to(graph.device),
                    torch.IntTensor(
                        adj.indices
                    ).to(graph.device)
                ],
                right_norm=torch.div(
                    1.0, graph.in_degrees(
                        etype=(sty, ety, dty)
                    )
                ).unsqueeze(-1)
            )
            hgraph.idx2rel[
                len(hgraph.idx2rel)
            ] = [sty, ety, dty]
            hgraph.rel2idx[
                sty, ety, dty
            ] = len(hgraph.rel2idx)
            hgraph.hetero_graph[
                sty, ety, dty
            ] = Graph(blk)
        return hgraph


def from_dglgraph(graph):
    import dgl
    assert isinstance(
        graph, (dgl.DGLGraph,
                dgl.DGLHeteroGraph)
    )
    if graph.is_homogeneous:
        return Graph.from_dglgraph(graph)
    else:
        return HeteroGraph.from_dglgraph(graph)
