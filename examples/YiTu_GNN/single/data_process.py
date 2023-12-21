import os
import pickle as pkl
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import argparse


def process(root_, dataset):
    """process data"""
    names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
    objects = []

    for name in names:
        try:
            with open("{}/ind.{}.{}".format(root_, dataset, name), "rb") as f:
                objects.append(pkl.load(f, encoding="latin1"))
        except IOError as e:
            raise e

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = _parse_index_file("{}/ind.{}.test.index".format(root_, dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == "citeseer":
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = _normalize_cora_features(features)
    graph = nx.Graph(nx.from_dict_of_lists(graph))
    graph = graph.to_directed()

    onehot_labels = np.vstack((ally, ty))
    onehot_labels[test_idx_reorder, :] = onehot_labels[test_idx_range, :]
    labels = np.argmax(onehot_labels, 1)

    adj_coo_row = []
    adj_coo_col = []
    line_count = 0

    for e in graph.edges:
        adj_coo_row.append(e[0])
        adj_coo_col.append(e[1])
        line_count += 1

    for i in range(len(labels)):
        adj_coo_row.append(i)
        adj_coo_col.append(i)

    num_nodes = len(labels)
    num_edges = len(adj_coo_row)
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = _sample_mask(idx_train, num_nodes)
    val_mask = _sample_mask(idx_val, num_nodes)
    test_mask = _sample_mask(idx_test, num_nodes)

    adj_coo_matrix = coo_matrix(
        (np.ones(len(adj_coo_row), dtype=bool), (adj_coo_row, adj_coo_col)),
        shape=(num_nodes, num_nodes),
    )
    # out_degrees = np.sum(adj_coo_matrix, axis=1)
    # in_degrees = np.sum(adj_coo_matrix, axis=0)
    # adj_csr_matrix = adj_coo_matrix.tocsr()

    path = os.path.join(root_, "{}.npz".format(dataset))
    np.savez(
        path,
        feat=features,
        label=labels,
        test_mask=test_mask,
        train_mask=train_mask,
        val_mask=val_mask,
        src_li=adj_coo_row,
        dst_li=adj_coo_col,
        num_edges=num_edges,
        num_nodes=num_nodes,
        n_classes=onehot_labels.shape[1],
    )


def _parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def _normalize_cora_features(features):
    row_sum = np.array(features.sum(1))
    r_inv = np.power(row_sum * 1.0, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return np.asarray(features.todense(), dtype=np.float32)


def _sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l, dtype=bool)
    mask[idx] = True
    return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="/home/data"
    )
    parser.add_argument("--dataset", type=str, default="citeseer")
    args = parser.parse_args()
    process(args.data_path, args.dataset)
