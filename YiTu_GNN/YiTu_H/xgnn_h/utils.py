import time
import torch
import pstats
import cProfile
import dgl as dgl
import networkx as nx
# from matplotlib import pyplot as plt


def draw_metapath(graph: dgl.DGLGraph):
    metapath = nx.MultiDiGraph()
    for sty, ety, dty in graph.canonical_etypes:
        n_src = graph.number_of_nodes(sty)
        n_dst = graph.number_of_nodes(dty)
        print('{}-->{}-->{}: [{}, {}]'.format(
            sty, ety, dty, n_dst, n_src
        ))
        metapath.add_edge(
            '{}-{}'.format(sty, n_src),
            '{}-{}'.format(dty, n_dst),
            label=ety
        )
    pos = nx.shell_layout(metapath)
    nx.draw_networkx(
        metapath, pos=pos, node_size=1200
    )
    plt.show()


class Profiler:
    def __init__(self, n_iter: int):
        self._n_iter = n_iter
        self._profiler = cProfile.Profile()

    def __enter__(self):
        self._timing = time.time()
        self._before = torch.cuda.memory_stats()
        self._profiler.enable()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        assert not exc_value
        self._profiler.disable()
        stats = self._cuda_stats(
            self._before, torch.cuda.memory_stats()
        )
        for name, value in stats.items():
            print('{}: {}'.format(name, value))
        pstats.Stats(self._profiler) \
            .strip_dirs() \
            .sort_stats(pstats.SortKey.TIME) \
            .print_stats(10)

    def timing(self):
        return time.time() - self._timing

    def _cuda_stats(self, before: dict, after: dict):
        res = {}
        num_columns = [
            'allocation.all.allocated',
            'allocation.all.freed',
            'allocation.all.current',
            'allocation.small_pool.allocated',
            'allocation.small_pool.freed',
            'allocation.small_pool.current',
            'allocation.large_pool.allocated',
            'allocation.large_pool.freed',
            'allocation.large_pool.current',
        ]
        size_columns = [
            'allocated_bytes.all.allocated',
            'allocated_bytes.all.freed',
            'allocated_bytes.all.current',
            'allocated_bytes.small_pool.allocated',
            'allocated_bytes.small_pool.freed',
            'allocated_bytes.small_pool.current',
            'allocated_bytes.large_pool.allocated',
            'allocated_bytes.large_pool.freed',
            'allocated_bytes.large_pool.current',
        ]
        for name in num_columns:
            res[name] = after[name] - before[name]
        for name in size_columns:
            diff = after[name] - before[name]
            diff /= 1024.0 * 1024.0 * self._n_iter
            res[name] = '{:.2f} MB'.format(diff)
        return res
