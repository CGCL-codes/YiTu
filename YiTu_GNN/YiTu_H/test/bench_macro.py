import time
import torch
import xgnn_h
import argparse
import dgl as dgl
from xgnn_h import mp, utils
from dgl.data import rdf, reddit
from dgl.data import citation_graph as cit
from dgl.data import gnn_benchmark as bench
from common.model import GCNModel, GATModel, RGCNModel, RGATModel
from common.dglmodel import DGLGCNModel, DGLGATModel, DGLRGCNModel, DGLRGATModel
try:
    import torch_geometric as pyg
    from torch_geometric import loader as pygld
    from torch_geometric import datasets as pygds
    from common.pygmodel import PyGGCNModel, PyGGATModel, PyGRGCNModel, PyGRGATModel
except ImportError as e:
    print('PyG not imported')
    PyGGCNModel = PyGGATModel = PyGRGCNModel = PyGRGATModel = None

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


class BenchMethods:
    @staticmethod
    def _check_dataset(dataset):
        # dataset
        divider = 10 * '-'
        print('{}[{}]{}'.format(
            divider, dataset.name, divider
        ))
        graph: dgl.DGLGraph = dataset[0]
        print('n_nodes:', graph.num_nodes())
        #
        if graph.is_homogeneous:
            avg_degrees = torch.mean(
                graph.in_degrees().type(
                    torch.FloatTensor
                )
            ).item()
            print('n_edges:', graph.num_edges())
            print('avg_degrees:', avg_degrees)
        else:
            n_rows = 0
            n_edges = 0
            avg_degrees = []
            print('node-types:', len(graph.ntypes))
            print(
                'meta-paths:',
                len(graph.canonical_etypes)
            )
            for sty, ety, dty in graph.canonical_etypes:
                avg_degrees.append(
                    torch.mean(
                        graph.in_degrees(
                            etype=(sty, ety, dty)
                        ).type(
                            torch.FloatTensor
                        )
                    ).item()
                )
                n_rows += graph.num_dst_nodes(dty)
                n_edges += graph.num_edges((sty, ety, dty))
            print('n_rows:', n_rows)
            print('n_edges:', n_edges)
            avg_degrees = sum(avg_degrees) / len(avg_degrees)
            print('avg_degrees:', avg_degrees)

    @staticmethod
    def _bench_pyg_homo(dataset, model, d_hidden):
        # info
        if hasattr(dataset, 'name'):
            name = dataset.name
        else:
            name = type(dataset).__name__
        print('[PYG] {}, {}, d_hidden={}'.format(
            name, model.__name__, d_hidden
        ))

        # dataset
        n_epochs = 20
        graph = dataset[0]
        if graph.num_edges > 128 * 1024:
            n_nodes = graph.num_nodes
            nids = torch.randperm(n_nodes // 16)
            sampler = pygld.NeighborLoader(
                graph, num_neighbors=[64],
                input_nodes=nids, batch_size=n_nodes // 16
            )
            graph = next(iter(sampler))
            print('graph is too large, sample 1/16 of its nodes')
        graph = graph.to('cuda')
        n_nodes = graph.num_nodes
        print('n_nodes:', n_nodes)
        n_edges = graph.num_edges
        print('n_edges:', n_edges)
        n_labels = dataset.num_classes
        print('n_labels:', n_labels)
        n_features = graph.num_features
        print('n_features:', n_features)

        # inputs
        gradient = torch.ones(
            [n_nodes, n_labels]
        ).to('cuda')
        model = model(
            in_features=n_features,
            gnn_features=d_hidden,
            out_features=n_labels
        ).to('cuda')

        # prewarm
        y = model(graph)
        y.backward(gradient=gradient)
        torch.cuda.synchronize()

        # training
        timing = None
        time.sleep(2.0)
        print('[TRAINING]')
        with utils.Profiler(n_epochs) as prof:
            for _ in range(n_epochs):
                y = model(graph)
                y.backward(gradient=gradient)
            torch.cuda.synchronize()
            timing = prof.timing() / n_epochs
        print('throughput: {:.1f}'.format(n_edges / timing))

    @staticmethod
    def _bench_pyg_hetero(dataset, model, d_hidden):
        # info
        if hasattr(dataset, 'name'):
            name = dataset.name
        else:
            name = type(dataset).__name__
        print('[PYG] {}, {}, d_hidden={}'.format(
            name, model.__name__, d_hidden
        ))

        # dataset
        n_epochs = 20
        graph = dataset[0]
        assert len(graph.node_types) == 1
        nty = graph.node_types[0]
        if graph.num_edges > 128 * 1024:
            sampler = pygld.NeighborLoader(
                graph,
                num_neighbors={
                    rel: [64]
                    for rel in graph.edge_types
                },
                input_nodes=nty,
                batch_size=graph.num_nodes // 16
            )
            graph = next(iter(sampler))
            print('graph is too large, sample 1/16 of its nodes')
        graph = graph.to('cuda')
        n_edges = graph.num_edges
        print('n_edges:', n_edges)
        n_nodes = graph.num_nodes
        print('n_nodes:', n_nodes)
        n_labels = torch.max(
            graph[nty]['train_y']
        ).item() + 1
        print('n_labels:', n_labels)

        # inputs
        gradient = torch.ones([
            graph[nty].num_nodes,
            n_labels
        ]).to('cuda')
        model = model(
            graph=graph,
            in_features=d_hidden,
            gnn_features=d_hidden,
            out_features=n_labels
        ).to('cuda')

        # prewarm
        y = model(graph)[nty]
        y.backward(gradient=gradient)
        torch.cuda.synchronize()

        # training
        timing = None
        time.sleep(2.0)
        print('[TRAINING]')
        with utils.Profiler(n_epochs) as prof:
            for _ in range(n_epochs):
                y = model(graph)[nty]
                y.backward(gradient=gradient)
            torch.cuda.synchronize()
            timing = prof.timing() / n_epochs
        print('throughput: {:.1f}'.format(n_edges / timing))

    @staticmethod
    def _bench_dgl_homo(dataset, model, d_hidden):
        # info
        print('[DGL] {}, {}, d_hidden={}'.format(
            dataset.name, model.__name__, d_hidden
        ))

        # dataset
        n_epochs = 20
        graph = dataset[0]
        if graph.num_edges() > 128 * 1024:
            n_nodes = graph.num_nodes()
            nids = torch.randperm(n_nodes // 16)
            graph = dgl.sampling.sample_neighbors(
                g=graph, nodes=nids, fanout=64
            )
            print('graph is too large, sample 1/16 of its nodes')
        graph = graph.to('cuda')
        n_nodes = graph.num_nodes()
        print('n_nodes:', n_nodes)
        n_edges = graph.num_edges()
        print('n_edges:', n_edges)
        n_labels = dataset.num_classes
        print('n_labels:', n_labels)
        feature = graph.ndata.pop(
            'feat'
        ).to('cuda')
        n_features = feature.size(-1)
        print('n_features:', n_features)

        # inputs
        gradient = torch.ones(
            [n_nodes, n_labels]
        ).to('cuda')
        model = model(
            in_features=n_features,
            gnn_features=d_hidden,
            out_features=n_labels
        ).to('cuda')

        # prewarm
        y = model(graph, feature)
        y.backward(gradient=gradient)
        torch.cuda.synchronize()

        # training
        timing = None
        time.sleep(2.0)
        print('[TRAINING]')
        with utils.Profiler(n_epochs) as prof:
            for _ in range(n_epochs):
                y = model(graph, feature)
                y.backward(gradient=gradient)
            torch.cuda.synchronize()
            timing = prof.timing() / n_epochs
        print('throughput: {:.1f}'.format(n_edges / timing))

    @staticmethod
    def _bench_dgl_hetero(dataset, model, d_hidden):
        # info
        print('[DGL] {}, {}, d_hidden={}'.format(
            dataset.name, model.__name__, d_hidden
        ))

        # dataset
        n_epochs = 20
        graph = dataset[0]
        if graph.num_edges() > 128 * 1024:
            nids = {
                nty: torch.randperm(
                    graph.num_nodes(nty) // 16
                )
                for nty in graph.ntypes
            }
            graph = dgl.sampling.sample_neighbors(
                g=graph, nodes=nids, fanout=64
            )
            print('graph is too large, sample 1/16 of its nodes')
        graph = graph.to('cuda')
        n_edges = graph.num_edges()
        print('n_edges:', n_edges)
        n_labels = dataset.num_classes
        print('n_labels:', n_labels)
        category = dataset.predict_category
        print('predict_category:', category)

        # inputs
        gradient = torch.ones([
            graph.num_nodes(category),
            n_labels
        ]).to('cuda')
        model = model(
            g=graph,
            in_features=d_hidden,
            gnn_features=d_hidden,
            out_features=n_labels
        ).to('cuda')

        # prewarm
        y = model(graph)[category]
        y.backward(gradient=gradient)
        torch.cuda.synchronize()

        # training
        timing = None
        time.sleep(2.0)
        print('[TRAINING]')
        with utils.Profiler(n_epochs) as prof:
            for _ in range(n_epochs):
                y = model(graph)[category]
                y.backward(gradient=gradient)
            torch.cuda.synchronize()
            timing = prof.timing() / n_epochs
        print('throughput: {:.1f}'.format(n_edges / timing))

    @staticmethod
    def _bench_xgnn_h_homo(dataset, model, d_hidden):
        # info
        print('[xgnn_h] {}, {}, d_hidden={}'.format(
            type(dataset).__name__,
            model.__name__, d_hidden
        ))

        # dataset
        n_epochs = 20
        dglgraph = dataset[0]
        if dglgraph.num_edges() > 128 * 1024:
            n_nodes = dglgraph.num_nodes()
            nids = torch.randperm(n_nodes // 16)
            dglgraph = dgl.sampling.sample_neighbors(
                g=dglgraph, nodes=nids, fanout=64
            )
            print('graph is too large, sample 1/16 of its nodes')
        dglgraph = dglgraph.to('cuda')
        graph = mp.from_dglgraph(dglgraph)
        n_nodes = dglgraph.num_nodes()
        print('n_nodes:', n_nodes)
        n_edges = dglgraph.num_edges()
        print('n_edges:', n_edges)
        n_labels = dataset.num_classes
        print('n_labels:', n_labels)
        feature = dglgraph.ndata.pop('feat')
        n_features = feature.size(-1)
        print('n_features:', n_features)

        # inputs
        feature = torch.randn(
            size=[n_nodes, n_features]
        ).to('cuda')
        gradient = torch.ones(
            [n_nodes, n_labels]
        ).to('cuda')
        model = model(
            in_features=n_features,
            gnn_features=d_hidden,
            out_features=n_labels
        ).to('cuda')
        kwargs = dict({
            'graph': graph, 'x': feature
        })
        if isinstance(model, GCNModel):
            kwargs['norm'] = graph.right_norm()

        # optimizer
        mod2ir = xgnn_h.Module2IR()
        optimizer = xgnn_h.Optimizer()
        dataflow = mod2ir.transform(
            model, kwargs=kwargs
        )
        dataflow = optimizer.lower(
            dataflow, kwargs=kwargs
        )
        executor = xgnn_h.Executor()

        # prewarm
        executor.train()
        y = executor.run(
            dataflow, kwargs=kwargs
        )
        y.backward(gradient=gradient)
        torch.cuda.synchronize()

        # training
        timing = None
        time.sleep(2.0)
        print('[TRAINING]')
        with utils.Profiler(n_epochs) as prof:
            for _ in range(n_epochs):
                y = executor.run(
                    dataflow, kwargs=kwargs
                )
                y.backward(gradient=gradient)
            torch.cuda.synchronize()
            timing = prof.timing() / n_epochs
        print('throughput: {:.1f}'.format(n_edges / timing))

    @staticmethod
    def _bench_xgnn_h_hetero(dataset, model, d_hidden):
        # info
        print('[xgnn_h] {}, {}, d_hidden={}'.format(
            type(dataset).__name__,
            model.__name__, d_hidden
        ))

        # dataset
        n_epochs = 20
        dglgraph = dataset[0]
        if dglgraph.num_edges() > 128 * 1024:
            nids = {
                nty: torch.randperm(
                    dglgraph.num_nodes(nty) // 16
                )
                for nty in dglgraph.ntypes
            }
            dglgraph = dgl.sampling.sample_neighbors(
                g=dglgraph, nodes=nids, fanout=64
            )
            print('graph is too large, sample 1/16 of its nodes')
        dglgraph = dglgraph.to('cuda')
        graph = mp.from_dglgraph(dglgraph)
        n_edges = dglgraph.num_edges()
        print('n_edges:', n_edges)
        n_labels = dataset.num_classes
        print('n_labels:', n_labels)
        category = dataset.predict_category
        print('predict_category:', category)

        # inputs
        gradient = torch.ones([
            dglgraph.num_nodes(category),
            n_labels
        ]).to('cuda')
        model = model(
            hgraph=graph,
            in_features=d_hidden,
            gnn_features=d_hidden,
            out_features=n_labels
        ).to('cuda')
        node_indices = {
            nty: torch.linspace(
                0, num - 1, num,
                dtype=torch.int64
            ).to('cuda')
            for nty, num in graph.nty2num.items()
        }
        kwargs = dict({
            'hgraph': graph,
            'xs': node_indices
        })
        if isinstance(model, RGCNModel):
            kwargs['norms'] = {
                str(rel): g.right_norm()
                for rel, g in graph.hetero_graph.items()
            }

        # optimizer
        mod2ir = xgnn_h.Module2IR()
        optimizer = xgnn_h.Optimizer()
        stitcher = xgnn_h.Stitcher()
        dataflow = mod2ir.transform(
            model, kwargs=kwargs
        )[category]
        dataflow = optimizer.lower(
            dataflow, kwargs=kwargs
        )
        dataflow = stitcher.transform(
            dataflow, kwargs=kwargs
        )
        executor = xgnn_h.Executor()

        # prewarm
        executor.train()
        y = executor.run(
            dataflow, kwargs=kwargs
        )
        y.backward(gradient=gradient)
        torch.cuda.synchronize()

        # training
        timing = None
        time.sleep(2.0)
        print('[TRAINING]')
        with utils.Profiler(n_epochs) as prof:
            for _ in range(n_epochs):
                y = executor.run(
                    dataflow, kwargs=kwargs
                )
                y.backward(gradient=gradient)
            torch.cuda.synchronize()
            timing = prof.timing() / n_epochs
        print('throughput: {:.1f}'.format(n_edges / timing))


class Benchmark(BenchMethods):
    HOM_MODELS = [
        (PyGGCNModel, DGLGCNModel, GCNModel),
        (PyGGATModel, DGLGATModel, GATModel),
    ]
    HET_MODELS = [
        (PyGRGCNModel, DGLRGCNModel, RGCNModel),
        (PyGRGATModel, DGLRGATModel, RGATModel),
    ]
    HOM_DATASETS = [
        'cora_tiny', 'amazon', 'cora_full', 'reddit',
    ]
    HET_DATASETS = [
        'aifb_hetero', 'mutag_hetero', 'bgs_hetero', 'am_hetero'
    ]
    DGL_DATASETS = {
        'cora_tiny': cit.CoraGraphDataset,  # 2.7k
        'amazon': bench.AmazonCoBuyPhotoDataset,  # 7.7k
        'cora_full': bench.CoraFullDataset,  # 19.8k
        'reddit': reddit.RedditDataset,  # 233.0k
        #
        'aifb_hetero': rdf.AIFBDataset,  # 7.3k
        'mutag_hetero': rdf.MUTAGDataset,  # 27.2k
        'bgs_hetero': rdf.BGSDataset,  # 94.8k
        'am_hetero': rdf.AMDataset  # 881.7k
    }
    PYG_DATASETS = {
        'cora_tiny': lambda: pygds.Planetoid(root='.data', name='Cora'),
        'amazon': lambda: pygds.Amazon(root='.data', name='Photo'),
        'cora_full': lambda: pygds.CoraFull(root='.data'),
        'reddit': lambda: pygds.Reddit(root='.data'),
        #
        'aifb_hetero': lambda: pygds.Entities(root='.data', name='AIFB', hetero=True),
        'mutag_hetero': lambda: pygds.Entities(root='.data', name='MUTAG', hetero=True),
        'bgs_hetero': lambda: pygds.Entities(root='.data', name='BGS', hetero=True),
        'am_hetero': lambda: pygds.Entities(root='.data', name='AM', hetero=True),
    }

    def dataset_info(self):
        for dataset in self.HOM_DATASETS:
            dataset = self.DGL_DATASETS[
                dataset
            ](verbose=False)
            self._check_dataset(dataset)
        for dataset in self.HET_DATASETS:
            dataset = self.DGL_DATASETS[
                dataset
            ](verbose=False)
            self._check_dataset(dataset)

    def bench_homogenous(self, lib, model, dataset, d_hidden):
        if lib == 'pyg':
            dataset = self.PYG_DATASETS[
                dataset
            ]()
            if model == 'gcn':
                model = PyGGCNModel
            elif model == 'gat':
                model = PyGGATModel
            else:
                raise RuntimeError
            self._bench_pyg_homo(
                dataset=dataset,
                model=model,
                d_hidden=d_hidden
            )
        elif lib == 'dgl':
            dataset = self.DGL_DATASETS[
                dataset
            ]()
            if model == 'gcn':
                model = DGLGCNModel
            elif model == 'gat':
                model = DGLGATModel
            else:
                raise RuntimeError
            self._bench_dgl_homo(
                dataset=dataset,
                model=model,
                d_hidden=d_hidden
            )
        elif lib == 'xgnn_h':
            dataset = self.DGL_DATASETS[
                dataset
            ]()
            if model == 'gcn':
                model = GCNModel
            elif model == 'gat':
                model = GATModel
            else:
                raise RuntimeError
            self._bench_xgnn_h_homo(
                dataset=dataset,
                model=model,
                d_hidden=d_hidden
            )
        else:
            raise RuntimeError

    def bench_heterogenous(self, lib, model, dataset, d_hidden):
        if lib == 'pyg':
            dataset = self.PYG_DATASETS[
                dataset
            ]()
            if model == 'rgcn':
                model = PyGRGCNModel
            elif model == 'rgat':
                model = PyGRGATModel
            else:
                raise RuntimeError
            self._bench_pyg_hetero(
                dataset=dataset,
                model=model,
                d_hidden=d_hidden
            )
        elif lib == 'dgl':
            dataset = self.DGL_DATASETS[
                dataset
            ]()
            if model == 'rgcn':
                model = DGLRGCNModel
            elif model == 'rgat':
                model = DGLRGATModel
            else:
                raise RuntimeError
            self._bench_dgl_hetero(
                dataset=dataset,
                model=model,
                d_hidden=d_hidden
            )
        elif lib == 'xgnn_h':
            dataset = self.DGL_DATASETS[
                dataset
            ]()
            if model == 'rgcn':
                model = RGCNModel
            elif model == 'rgat':
                model = RGATModel
            else:
                raise RuntimeError
            self._bench_xgnn_h_hetero(
                dataset=dataset,
                model=model,
                d_hidden=d_hidden
            )
        else:
            raise RuntimeError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--info', action='store_true')
    parser.add_argument('--lib', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--d_hidden', type=int)
    args = parser.parse_args()

    #
    benchmark = Benchmark()
    if args.info:
        benchmark.dataset_info()
        return

    if args.model.startswith('r'):
        benchmark.bench_heterogenous(
            args.lib, args.model,
            args.dataset, args.d_hidden
        )
    else:
        benchmark.bench_homogenous(
            args.lib, args.model,
            args.dataset, args.d_hidden
        )


if __name__ == "__main__":
    main()
