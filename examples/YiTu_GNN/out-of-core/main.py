import os
import time

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from graphsage import SAGE
from YiTu_GNN.mmap.part_info import PartInfo
from YiTu_GNN.mmap.sampler import mmap_neighbor_sampler


class NeighborSampler:
    def __init__(self, hops, part_info, device) -> None:
        self.hops = hops
        self.part_info = part_info
        self.device = device

    def sampler(self, seeds):
        blocks = mmap_neighbor_sampler(seeds, self.hops, self.part_info)
        all_nodes = blocks[0][1]
        nodes_feat = self.part_info.nodes_feat(all_nodes)

        all_nodes = torch.tensor(all_nodes, dtype=torch.int64).to(self.device)
        nodes_feat = torch.tensor(nodes_feat, dtype=torch.float32).to(self.device)
        edge_indexs = []
        for g, _, _ in blocks:
            edge_indexs.append(torch.tensor(g, dtype=torch.int64).to(self.device))
        return edge_indexs, all_nodes, nodes_feat, len(seeds)


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def main(args):
    if args.gpu == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(args.gpu))
    part_config_path = os.path.join(args.data_path, "part_info.json")
    part_info = PartInfo(part_config_path)
    sampler = NeighborSampler(
        [int(i) for i in args.fanout.split(",")], part_info, device
    )
    train_nodes = part_info.train_nodes
    train_dataloader = DataLoader(
        train_nodes,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=sampler.sampler,
        num_workers=args.num_workers,
    )
    test_nodes = part_info.test_nodes
    val_nodes = part_info.val_nodes
    labels = part_info.labels
    labels = torch.tensor(labels, dtype=torch.int64).to(device)
    model = SAGE(
        part_info.feature_dim, args.num_hidden, args.num_classes, args.num_layers
    )
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    start = time.time()
    for epoch in range(args.epochs):
        for step, blocks in enumerate(train_dataloader):
            edge_indexs, all_nodes, nodes_feat, target_size = blocks
            label = labels[all_nodes[:target_size]]
            model.train()
            pred = model(nodes_feat, edge_indexs)
            loss = loss_fcn(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % args.eval_every == 0:
                acc = compute_acc(pred, label)
                print(
                    "Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f}".format(
                        epoch,
                        step,
                        loss.item(),
                        acc.item(),
                    )
                )
    print(time.time() - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphSage for mmap training")
    parser.add_argument(
        "--dataset", type=str, default="ogbn-products", help="dataset name"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/nx/ningxin/.data/mmap/output/",
        help="path to dataset",
    )
    parser.add_argument("--gpu", type=int, default=0, help="which gpu to use")
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers")
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="number of epochs to train (default: 200)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.003, help="learning rate (default: 0.01)"
    )
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout")
    parser.add_argument("--num_hidden", type=int, default=16, help="num_hidden")
    parser.add_argument("--num_classes", type=int, default=47, help="num_classes")
    parser.add_argument("--batch_size", type=int, default=1000, help="batch size")
    parser.add_argument("--fanout", type=str, default="25,10", help="fanout")
    parser.add_argument("--agg_type", type=str, default="mean", help="aggregation type")
    parser.add_argument("--eval", type=bool, default=True, help="eval")
    parser.add_argument("--eval_every", type=int, default=1, help="eval every")
    parser.add_argument(
        "--num_workers", type=int, default=0, help="number of sampling workers"
    )
    args = parser.parse_args()
    print(args)
    main(args)
