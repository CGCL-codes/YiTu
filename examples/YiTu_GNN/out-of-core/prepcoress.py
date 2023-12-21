import argparse
import time
import os
import numpy as np
import json

import dgl
from ogb.nodeproppred import NodePropPredDataset


def load_reddit(raw_dir, self_loop=True):
    from dgl.data import RedditDataset

    # load reddit data
    data = RedditDataset(self_loop=self_loop, raw_dir=raw_dir)
    g = data[0]
    # g.ndata["features"] = g.ndata.pop("feat")
    g.ndata["labels"] = g.ndata.pop("label")
    return g, data.num_classes


def process_ogbn(name, data_path, output_path):
    print("load", name)
    data = NodePropPredDataset(name=name, root=data_path)
    print("finish loading", name)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    labels = labels[:, 0]
    edge_index = graph["edge_index"]
    node_feat = graph["node_feat"]
    num_nodes = graph["num_nodes"]
    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # edge_index.npy, node_feat.npy, label.npy, train_nodes.npy, val_nodes.npy, test_nodes.npy
    np.save("{}/edge_index.npy".format(output_path), edge_index)
    np.save("{}/node_feat.npy".format(output_path), node_feat)
    np.save("{}/labels.npy".format(output_path), labels)
    np.save("{}/train_nodes.npy".format(output_path), train_nid)
    np.save("{}/test_nodes.npy".format(output_path), test_nid)
    np.save("{}/val_nodes.npy".format(output_path), val_nid)
    num_classes = len(np.unique(labels))
    graph_info = {
        "name": name,
        "num_nodes": num_nodes,
        "num_edges": edge_index.shape[1],
        "num_classes": num_classes,
    }
    with open("{}/graph_info.json".format(output_path), "w") as f:
        json.dump(graph_info, f, sort_keys=True, indent=4)
    print("finish process", name)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Partition builtin graphs")
    argparser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-products",
        help="datasets: reddit, ogbn-products, ogbn-papers100M",
    )
    argparser.add_argument(
        "--data_path",
        type=str,
        default="/home/nx/ningxin/.data/mindspore-gl",
        help="Output path of partitioned graph.",
    )
    argparser.add_argument(
        "--output_path",
        type=str,
        default="/home/nx/ningxin/.data/mmap/ogbn-products",
        help="Output path of partitioned graph.",
    )
    args = argparser.parse_args()

    if args.dataset == "reddit":
        load_reddit(args.data_path)
    elif args.dataset == "ogbn-products":
        process_ogbn("ogbn-products", args.data_path, args.output_path)
    elif args.dataset == "ogbn-papers100M":
        process_ogbn("ogbn-papers100M", args.data_path, args.output_path)
