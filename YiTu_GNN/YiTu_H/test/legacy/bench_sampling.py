import time
import random
import torch
import dgl
from dgl.nn import GraphConv, GATConv
from dgl.data import CoraGraphDataset, RedditDataset


def check_sampling_neighbor(dataset):
    # attribute
    graph = dataset[0]
    train_mask = graph.ndata['train_mask']
    train_nodes = torch.nonzero(train_mask).squeeze()
    print('num_train: {}'.format(len(train_nodes)))

    # sampling
    # """
    for k in [1, 16, 32]:
        before = time.time()
        dst_nodes = torch.stack(
            random.choices(train_nodes, k=k)
        )
        #
        blocks = []
        for i in range(2):
            frontier = dgl.sampling.sample_neighbors(
                graph,
                nodes=dst_nodes,
                fanout=-1
            )
            # frontier = dgl.in_subgraph(graph, dst_nodes)
            block = dgl.to_block(frontier, dst_nodes)
            dst_nodes = block.srcnodes['_N'].data['_ID']
            # dst_nodes = block.srcnodes()
            blocks.append(block)
        print('in_subgraph({}), uv: {}/{}/{}, timing: {:.3f}'
              .format(k,
                      blocks[0].num_dst_nodes(),
                      blocks[1].num_dst_nodes(),
                      blocks[1].num_src_nodes(),
                      time.time() - before))
    # """

    # dataloader
    # """
    train_nid = torch.nonzero(train_mask).squeeze()
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    for k in [1, 16, 32]:
        dataloader = dgl.dataloading.NodeDataLoader(
            graph, train_nid, sampler,
            batch_size=k, shuffle=True,
            drop_last=False, num_workers=0
        )
        dataiter = iter(dataloader)
        before = time.time()
        a, b, blocks = next(dataiter)
        print('NodeDataLoader({}), uv: {}/{}/{}, timing: {:.3f}'
              .format(k,
                      blocks[1].num_dst_nodes(),
                      blocks[1].num_src_nodes(),
                      blocks[0].num_src_nodes(),
                      time.time() - before))
    # """


def test():
    # dataset
    dataset = RedditDataset(
        verbose=False
    )

    # profile
    import pstats
    import cProfile
    profiler = cProfile.Profile()

    #
    profiler.enable()
    check_sampling_neighbor(dataset)
    profiler.disable()
    pstats.Stats(profiler).strip_dirs() \
        .sort_stats(pstats.SortKey.TIME).print_stats(20)


if __name__ == "__main__":
    test()
