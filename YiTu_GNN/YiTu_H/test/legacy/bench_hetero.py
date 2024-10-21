import time
import torch
import dgl as dgl
from torch import nn
from dgl import nn as dglnn
from dgl.data import CitationGraphDataset, CoraGraphDataset, PubmedGraphDataset
from dgl.data.rdf import RDFGraphDataset, AIFBDataset, MUTAGDataset, AMDataset


class RGCNLayer(nn.Module):
    def __init__(self,
                 g: dgl.DGLHeteroGraph,
                 layer, layer_args,
                 activation=None,
                 dropout=None):
        nn.Module.__init__(self)
        #
        self.g = g
        self.convs = dglnn.HeteroGraphConv(
            {
                ety: layer(*layer_args)
                for ety in sorted(list(g.etypes))
            }, aggregate='sum'
        )
        self.activation = activation
        self.dropout = dropout

    def forward(self, features):
        hs = self.convs(self.g, features)
        return hs


def run_homo(dataset, layer, args, d_hidden, n_labels):
    print('----- {} {} -----'.format(
        type(dataset), layer
    ))

    #
    graph = dataset[0].to('cuda')
    print('n_labels:', n_labels)
    print('d_hidden:', d_hidden)
    print('n_nodes:', graph.num_nodes())

    #
    feature = torch.randn(
        size=[graph.num_nodes(), d_hidden]
    ).to('cuda')
    i2h_layer = layer(*args).to('cuda')

    # profile
    i2h_layer(graph, feature)
    with Profiler():
        y = i2h_layer(graph, feature)


def run_hetero(dataset, layer, args, d_hidden, n_labels):
    print('----- {} {} -----'.format(
        type(dataset), layer
    ))

    #
    graph = dataset[0].to('cuda')
    category = dataset.predict_category
    labels = graph.ndata['labels']
    test_idx = graph.ndata['test_idx']
    train_idx = graph.ndata['train_idx']

    #
    print('n_labels:', n_labels)
    print('d_hidden:', d_hidden)
    print('n_entities:', len(graph.ntypes))
    for nty in graph.ntypes:
        print('num_nodes:', nty, graph.number_of_nodes(nty))
    print('category:', category)

    #
    features = {}
    for nty in graph.ntypes:
        n = graph.number_of_nodes(nty)
        features[nty] = torch.randn(
            size=[n, d_hidden]
        ).to('cuda')

    #
    i2h_layer = RGCNLayer(
        graph,
        layer=layer,
        layer_args=args,
        activation=nn.ReLU(),
        dropout=None
    ).to('cuda')

    # profile
    i2h_layer(features)
    with Profiler():
        y = i2h_layer(features)


if __name__ == "__main__":
    # gcn vs rgcn
    # gat vs rgat
    # operator level
    # heterogenous level
    # invocation, allocation, roofline
    for dataset in [
        # homo
        CoraGraphDataset,
        PubmedGraphDataset,
        # hetero
        AIFBDataset, MUTAGDataset,  # AMDataset
    ]:
        d_hidden = 16
        dataset = dataset()
        n_labels = dataset.num_classes
        for layer, args in [
            (
                dglnn.GraphConv,
                [d_hidden, d_hidden, 'right']
            ),
            (
                dglnn.GATConv,
                [d_hidden, d_hidden, 8]
            )
        ]:
            if isinstance(dataset, CitationGraphDataset):
                run_homo(
                    dataset, layer, args, d_hidden, n_labels
                )
            elif isinstance(dataset, RDFGraphDataset):
                run_hetero(
                    dataset, layer, args, d_hidden, n_labels
                )
            else:
                raise NotImplementedError
            time.sleep(2.0)
