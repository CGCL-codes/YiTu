import time
import torch
import dgl as dgl
from dgl.data import rdf
from dgl import nn as dglnn
from common.dglmodel import DGLRGATModel
from xgnn_h import utils


class Benchmark:
    @staticmethod
    def bench_dgl_hetero(dataset, model, n_layers, batch_size):
        # info
        print('[DGL] {}, {}, n_layers={}, batch_size={}'.format(
            dataset.name, model.__name__, n_layers, batch_size
        ))

        # dataset
        graph = dataset[0]
        # print(graph.num_nodes(), graph.num_edges())
        loader = dgl.dataloading.NodeDataLoader(
            graph=graph,
            indices={
                nty: torch.randperm(
                    graph.num_nodes(nty)
                )
                for nty in graph.ntypes
            },
            graph_sampler=dgl.dataloading.MultiLayerNeighborSampler(
                [16] * n_layers
            ),
            batch_size=batch_size
        )
        batch = next(iter(loader))
        src, dst, blk = batch
        #
        subgraph = blk[0].to('cuda')
        num_nodes = sum([
            subgraph.num_nodes(nty)
            for nty in subgraph.ntypes
        ])
        num_edges = sum([
            subgraph.num_edges(rel)
            for rel in graph.canonical_etypes
        ])
        print('num_nodes: {}, num_edges: {}'
              .format(num_nodes, num_edges))

        #
        n_labels = dataset.num_classes
        print('n_labels:', n_labels)
        category = dataset.predict_category
        print('predict_category:', category)

        # inputs
        n_heads = 8
        d_hidden = 32
        features = {
            nty: torch.zeros(
                [subgraph.num_nodes(nty), d_hidden]
            ).to('cuda')
            for nty in subgraph.ntypes
        }
        model = dglnn.HeteroGraphConv(
            {
                ety: dglnn.GATConv(
                    d_hidden, d_hidden,
                    n_heads, activation=None, bias=True,
                    allow_zero_in_degree=True
                )
                for ety in sorted(list(subgraph.etypes))
            }, aggregate='mean'
        ).to('cuda')

        # prewarm
        y = model(subgraph, features)[category]
        torch.sum(y).backward()
        torch.cuda.synchronize()

        # training
        n_epochs = 20
        timing = None
        time.sleep(2.0)
        print('[TRAINING]')
        with utils.Profiler(n_epochs) as prof:
            for _ in range(n_epochs):
                y = model(subgraph, features)[category]
                torch.sum(y).backward()
            torch.cuda.synchronize()
            timing = prof.timing() / n_epochs
        print('throughput: {:.1f}'.format(num_edges / timing))


def main():
    model = DGLRGATModel
    dataset = rdf.AMDataset()
    for n in range(2, 9):
        Benchmark.bench_dgl_hetero(
            dataset, model,
            n_layers=n, batch_size=1024
        )
    #
    for bs in [256, 512, 1024, 2048, 4096, 8192]:
        Benchmark.bench_dgl_hetero(
            dataset, model,
            n_layers=2, batch_size=bs
        )


if __name__ == "__main__":
    main()
