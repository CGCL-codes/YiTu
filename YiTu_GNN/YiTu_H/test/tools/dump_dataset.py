'''
@Author  :   Yuntao GUI
@License :   (C) Copyright 2020-2077, CUHK
@Contact :   bosskwei95@outlook.com
@File    :   utils.py
@Time    :   2020/12/05
@Desc    :
'''
import torch
from tqdm import tqdm
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import RedditDataset, CoraFull, CoauthorCSDataset, CoauthorPhysicsDataset, AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset


def to_dense(n_rows, n_cols, sparse):
    values, indices, indptr = sparse
    dense = torch.zeros(size=[n_rows, n_cols])
    for row in range(n_rows):
        for i in range(indptr[row], indptr[row + 1]):
            col = indices[i]
            val = values[i]
            dense[row, col] = val
    return dense


def to_csr(dense):
    n_rows, n_cols = dense.size()
    values, indices, indptr = [], [], []
    for row in tqdm(range(n_rows)):
        indptr.append(len(indices))
        for col in torch.nonzero(dense[row]):
            assert torch.is_nonzero(dense[row, col])
            indices.append(col.item())
            values.append(dense[row, col].item())
    indptr.append(len(indices))
    #
    values = torch.FloatTensor(values)
    indices, indptr = torch.LongTensor(indices), torch.LongTensor(indptr)
    #
    assert torch.equal(dense, to_dense(
        n_rows, n_cols, [values, indices, indptr]))
    print('compare dense and csr success')
    #
    return values, indices, indptr


def dump_bench_dataset(data):
    graph = data[0]
    # info
    name = data.name
    n_nodes = graph.num_nodes()
    n_edges = graph.num_edges()

    # labels
    labels = graph.ndata['label']
    labels = torch.LongTensor(labels)

    # features
    features = graph.ndata['feat']
    features = torch.FloatTensor(features)
    assert len(features) == n_nodes
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    train_mask = graph.ndata['train_mask']

    #
    print('name: {}'.format(name))
    print('n_nodes: {}'.format(n_nodes))
    print('n_edges: {}'.format(n_edges))
    print('labels: {}'.format(labels.shape))
    print('features: {}'.format(features.shape))

    # adjacency
    adjacency = graph.adjacency_matrix(scipy_fmt='csr', transpose=False)
    rev_adjacency = graph.adjacency_matrix(scipy_fmt='csr', transpose=True)

    # degree
    in_degrees = graph.in_degrees()
    out_degrees = graph.out_degrees()

    #
    adj_values = torch.FloatTensor(adjacency.data)
    adj_indices = torch.IntTensor(adjacency.indices)
    adj_indptr = torch.IntTensor(adjacency.indptr)

    rev_values = torch.FloatTensor(rev_adjacency.data)
    rev_indices = torch.IntTensor(rev_adjacency.indices)
    rev_indptr = torch.IntTensor(rev_adjacency.indptr)

    #
    filename = '.data/{}.pt'.format(name)
    torch.save({
        'labels': labels, 'features': features,
        'val_mask': val_mask, 'test_mask': test_mask, 'train_mask': train_mask,
        'adjacency': to_dense(n_nodes, n_nodes, [adj_values, adj_indices, adj_indptr]),
        'rev_adjacency': to_dense(n_nodes, n_nodes, [rev_values, rev_indices, rev_indptr]),
        'in_degrees': in_degrees, 'out_degrees': out_degrees,
        'adj_values': adj_values, 'adj_indices': adj_indices, 'adj_indptr': adj_indptr,
        'rev_values': rev_values, 'rev_indices': rev_indices, 'rev_indptr': rev_indptr,
    }, filename)
    print('dump {} finish'.format(filename))


def main():
    dataset = CoraGraphDataset()
    dump_bench_dataset(dataset)
    # dataset = CoraFull()
    # dump_bench_dataset(dataset)
    # for dataset in [RedditDataset, CoraFull, CoauthorCSDataset,
    #                 CoauthorPhysicsDataset, AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset]:
    #                 dataset()
    # dump_bench_dataset(dataset())


if __name__ == "__main__":
    main()
