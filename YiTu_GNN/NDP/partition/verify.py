import os
import argparse
import numpy as np
import scipy.sparse as spsp
import networkx as nx

from utils import *

def draw_graph(sub_adj, sub2fullid=None, pos=None, colored_nodes=None):
  g = nx.from_scipy_sparse_matrix(sub_adj, create_using=nx.DiGraph())
  if pos is None:
    pos = nx.kamada_kawai_layout(g)
  else:
    pos = pos(g)
  color_map = ['gray']
  if colored_nodes is not None:
    color_map = ['gray'] * sub_adj.shape[0]
    for nid in colored_nodes:
      color_map[nid] = 'orange'
  #pos = nx.spring_layout(g)
  if sub2fullid is None:
    nx.draw(g, pos, with_labels=True, arrows=True, node_color=color_map)
  else:
    labels = {idx: sub2fullid[idx] for idx in range(len(sub2fullid))}
    nx.draw(g, pos, arrows=True, node_color=color_map)
    _ = nx.draw_networkx_labels(g, pos, labels=labels)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='PartitionVerify')
  parser.add_argument("--dataset", type=str, default=None,
                      help="dataset dir")

  train_dataset = os.path.join(dataset, 'train')
  partition_dataset = os.path.join(dataset, 'partition')
  # full graph file
  full_adj_file = os.path.join(dataset, 'adj.npz')
  full_train_mask = os.path.join(dataset, 'train.npy')
  # train graph file
  train_adj_file = os.path.join(train_dataset, 'adj.npz')
  train2fullid_file = os.path.join(train_dataset, 'train2fullid.npy')
  # partition graph file
  pfile = ['wrap_subadj_{}_1hop.npz'.format(str(idx)) for idx in range(2)]
  pfile = [os.path.join(partition_dataset, p) for p in pfile]
  mapfile = ['wrap_sub2trainid_{}_1hop.npy'.format(str(idx)) for idx in range(2)]
  mapfile = [os.path.join(partition_dataset, mp) for mp in mapfile]
  trainfile = ['train_{}_1hop.npy'.format(str(idx)) for idx in range(2)]
  trainfile = [os.path.join(partition_dataset, tf) for tf in trainfile]

  # full graph visualization
  full_adj = spsp.load_npz(full_adj_file)
  train_nid_full = np.arange(full_adj.shape[0])[np.load(full_train_mask).astype(np.bool)]
  print('train nids:', train_nid_full)
  draw_graph(full_adj, colored_nodes=train_nid_full)
  # train graph visualization
  train_adj = spsp.load_npz(train_adj_file)
  train2fullid = np.load(train2fullid_file)
  draw_graph(train_adj, train2fullid, 
             colored_nodes=full2sub_nid(train2fullid, train_nid_full))
  # sub graph visualization
  sub_adjs = [spsp.load_npz(subadj_file) for subadj_file in pfile]
  sub2trainids = [train2fullid[np.load(submap_file)] for submap_file in mapfile]
  trainids = [np.load(tf) for tf in trainfile]

  print(train2fullid[sub2trainids[0][trainids[0]]])
  draw_graph(sub_adjs[0], train2fullid[sub2trainids[0]], pos=nx.spring_layout,
             colored_nodes=trainids[0])
  print(train2fullid[sub2trainids[1][trainids[1]]])
  draw_graph(sub_adjs[1], train2fullid[sub2trainids[1]], pos=nx.spring_layout,
             colored_nodes=trainids[1])

