import argparse
import time
import os

import numpy as np
import dgl
import torch as th

# from YiTu_GNN.distributed import partition_graph
from YiTu_GNN.distributed.partition import partition_graph


def load_np_data(data_path):
    data = np.load(data_path)
    feat = data["feat"]
    label = data["label"]
    train_mask = data["train_mask"]
    val_mask = data["val_mask"]
    test_mask = data["test_mask"]
    coo_row = data["src_li"]
    coo_col = data["dst_li"]
    num_nodes = data["num_nodes"]
    num_edges = data["num_edges"]
    num_classes = data["n_classes"]

    # numpy to torch tensor
    feat = th.from_numpy(feat)
    feat = th.tensor(feat, dtype=th.float32)
    label = th.from_numpy(label)
    train_mask = th.from_numpy(train_mask)
    val_mask = th.from_numpy(val_mask)
    test_mask = th.from_numpy(test_mask)
    coo_row = th.from_numpy(coo_row)
    coo_col = th.from_numpy(coo_col)
    # 转成dgl graph
    graph = dgl.graph((coo_row, coo_col))
    graph.ndata["feat"] = feat
    graph.ndata["labels"] = label
    graph.ndata["train_mask"] = train_mask
    graph.ndata["val_mask"] = val_mask
    graph.ndata["test_mask"] = test_mask
    return graph, num_classes


def load_reddit(raw_dir, self_loop=True):
    from dgl.data import RedditDataset

    # load reddit data
    data = RedditDataset(self_loop=self_loop, raw_dir=raw_dir)
    g = data[0]
    # g.ndata["features"] = g.ndata.pop("feat")
    g.ndata["labels"] = g.ndata.pop("label")
    return g, data.num_classes


def load_ogb(name, root="dataset"):
    from ogb.nodeproppred import DglNodePropPredDataset

    print("load", name)
    data = DglNodePropPredDataset(name=name, root=root)
    print("finish loading", name)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    labels = labels[:, 0]

    # graph.ndata["features"] = graph.ndata.pop("feat")
    graph.ndata["labels"] = labels
    # in_feats = graph.ndata["features"].shape[1]
    num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )
    train_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    train_mask[train_nid] = True
    val_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    val_mask[val_nid] = True
    test_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    test_mask[test_nid] = True
    graph.ndata["train_mask"] = train_mask
    graph.ndata["val_mask"] = val_mask
    graph.ndata["test_mask"] = test_mask
    print("finish constructing", name)
    return graph, num_labels


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Partition builtin graphs")
    argparser.add_argument(
        "--dataset",
        type=str,
        default="reddit",
        help="datasets: reddit, ogbn-products, ogbn-papers100M, soc-LiveJournal1, soc-pokec-relationships",
    )
    argparser.add_argument(
        "--num_parts", type=int, default=4, help="number of partitions"
    )
    argparser.add_argument(
        "--part_method", type=str, default="random", help="the partition method"
    )
    argparser.add_argument(
        "--balance_train",
        action="store_true",
        help="balance the training size in each partition.",
    )
    argparser.add_argument(
        "--undirected",
        action="store_true",
        help="turn the graph into an undirected graph.",
    )
    argparser.add_argument(
        "--balance_edges",
        action="store_true",
        help="balance the number of edges in each partition.",
    )
    argparser.add_argument(
        "--num_trainers_per_machine",
        type=int,
        default=1,
        help="the number of trainers per machine. The trainer ids are stored\
                                in the node feature 'trainer_id'",
    )
    argparser.add_argument(
        "--output",
        type=str,
        default="/home/data/",
        help="Output path of partitioned graph.",
    )
    args = argparser.parse_args()

    start = time.time()
    if args.dataset == "reddit":
        g, _ = load_reddit(args.output)
    elif args.dataset == "ogbn-products":
        g, _ = load_ogb("ogbn-products", args.output)
    elif args.dataset == "ogbn-papers100M":
        g, _ = load_ogb("ogbn-papers100M", args.output)
    else:
        g, _ = load_np_data(os.path.join(args.output, args.dataset + ".npz"))
    print("load {} takes {:.3f} seconds".format(args.dataset, time.time() - start))
    print("|V|={}, |E|={}".format(g.number_of_nodes(), g.number_of_edges()))
    print(
        "train: {}, valid: {}, test: {}".format(
            th.sum(g.ndata["train_mask"]),
            th.sum(g.ndata["val_mask"]),
            th.sum(g.ndata["test_mask"]),
        )
    )
    if args.balance_train:
        balance_ntypes = g.ndata["train_mask"]
    else:
        balance_ntypes = None

    if args.undirected:
        sym_g = dgl.to_bidirected(g, readonly=True)
        for key in g.ndata:
            sym_g.ndata[key] = g.ndata[key]
        g = sym_g

    partition_graph(
        g,
        args.dataset,
        args.num_parts,
        os.path.join(
            args.output,
            "YiTu_GNN/{}/{}/{}part_data".format(
                args.dataset, args.part_method, args.num_parts
            ),
        ),
        part_method=args.part_method,
        balance_ntypes=balance_ntypes,
        balance_edges=args.balance_edges,
        num_trainers_per_machine=args.num_trainers_per_machine,
        feat_name="feat",
    )
