import os
import sys
import argparse, time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import numpy as np
import dgl

from YiTu_GNN.NDP.model.gcn_nssc import GCNSampling
import YiTu_GNN.NDP.data as data


def count_nf_vnum(nf):
  vnum = 0
  for lid in range(nf.num_layers):
    vnum += nf.layer_nid(lid).size(0)
  return vnum


def main(args):
  dataname = os.path.basename(args.dataset)
  g = dgl.contrib.graph_store.create_graph_from_store(dataname, "shared_mem")
  labels = data.get_labels(args.dataset)
  n_classes = len(np.unique(labels))
  train_mask, val_mask, test_mask = data.get_masks(args.dataset)
  train_nid = np.nonzero(train_mask)[0].astype(np.int64)

  num_hops = args.n_layers if args.preprocess else args.n_layers + 1

  for epoch in range(args.n_epochs):
    epoch_load_vnum = 0
    for nf in dgl.contrib.sampling.NeighborSampler(g, args.batch_size,
                                                  args.num_neighbors,
                                                  neighbor_type='in',
                                                  shuffle=True,
                                                  num_workers=16,
                                                  num_hops=num_hops,
                                                  seed_nodes=train_nid,
                                                  prefetch=False):
      epoch_load_vnum += count_nf_vnum(nf)
    print('Epoch loaded vertex#: ', epoch_load_vnum)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Eval')
  parser.add_argument("--dataset", type=str, default=None,
                      help="path to the dataset folder")
  # model arch
  parser.add_argument("--feat-size", type=int, default=600,
                      help='input feature size')
  parser.add_argument("--n-layers", type=int, default=1,
                      help="number of hidden gcn layers")
  parser.add_argument("--preprocess", dest='preprocess', action='store_true')
  parser.set_defaults(preprocess=False)
  # training hyper-params
  parser.add_argument("--n-epochs", type=int, default=10,
                      help="number of training epochs")
  parser.add_argument("--batch-size", type=int, default=6000,
                      help="batch size")
  # sampling hyper-params
  parser.add_argument("--num-neighbors", type=int, default=2,
                      help="number of neighbors to be sampled")
  
  args = parser.parse_args()
  main(args)