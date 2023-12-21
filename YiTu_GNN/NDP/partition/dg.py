import os
import sys
import dgl
from dgl._deprecate.graph import DGLGraph
import torch
import numpy as np
import scipy.sparse as spsp
import argparse
import YiTu_GNN.NDP.data as data

import ordering
from utils import get_sub_graph

def in_neighbors(csc_adj, nid):
  return csc_adj.indices[csc_adj.ixgnn.NDPtr[nid]: csc_adj.ixgnn.NDPtr[nid+1]]


def in_neighbors_hop(csc_adj, nid, hops):
  if hops == 1:
    return in_neighbors(csc_adj, nid)
  else:
    nids = []
    for depth in range(hops):
      neighs = nids[-1] if len(nids) != 0 else [nid]
      for n in neighs:
        nids.append(in_neighbors(csc_adj, n))
    return np.unique(np.hstack(nids))


def dg_max_score(score, p_vnum):
  ids = np.argsort(score)[-2:]
  if score[ids[0]] != score[ids[1]]:
    return ids[1]
  else:
    return ids[0] if p_vnum[ids[0]] < p_vnum[ids[1]] else ids[1]


def dg_ind(adj, neighbors, belongs, p_vnum, r_vnum, pnum):
  """
  Params:
    neighbor: in-neighbor vertex set
    belongs: np array, each vertex belongings to which partition
    p_vnum: np array, each partition total vertex w/o. redundancy
    r_vnum: np array, each partition total vertex w/. redundancy
    pnum: partition number
  """
  com_neighbor = np.ones(pnum, dtype=np.int64)
  score = np.zeros(pnum, dtype=np.float32)
  # count belonged vertex
  neighbor_belong = belongs[neighbors]
  belonged = neighbor_belong[np.where(neighbor_belong != -1)]
  pid, freq = np.unique(belonged, return_counts=True)
  com_neighbor[pid] += freq
  avg_num = adj.shape[0] * 0.65 / pnum # need modify to match the train vertex num
  score = com_neighbor * (-p_vnum + avg_num) / (r_vnum + 1)
  return score


def dg(partition_num, adj, train_nids, hops):
  csc_adj = adj.tocsc()
  vnum = adj.shape[0]
  vtrain_num = train_nids.shape[0]
  belongs = -np.ones(vnum, dtype=np.int8)
  r_belongs = [-np.ones(vnum, dtype=np.int8) for _ in range(partition_num)]
  p_vnum = np.zeros(partition_num, dtype=np.int64)
  r_vnum = np.zeros(partition_num, dtype=np.int64)

  progress = 0
  #for nid in range(0, train_nids):
  print('total vertices: {} | train vertices: {}'.format(vnum, vtrain_num))
  for step, nid in enumerate(train_nids):  
    #neighbors = in_neighbors(csc_adj, nid)
    neighbors = in_neighbors_hop(csc_adj, nid, hops)
    score = dg_ind(csc_adj, neighbors, belongs, p_vnum, r_vnum, partition_num)
    ind = dg_max_score(score, p_vnum)
    if belongs[nid] == -1:
      belongs[nid] = ind
      p_vnum[ind] += 1
      neighbors = np.append(neighbors, nid)
      for neigh_nid in neighbors:
        if r_belongs[ind][neigh_nid] == -1:
          r_belongs[ind][neigh_nid] = 1
          r_vnum[ind] += 1
    # progress
    if int(vtrain_num * progress / 100) <= step:
      sys.stdout.write('=>{}%\r'.format(progress))
      sys.stdout.flush()
      progress += 1
  print('')
  
  sub_v = []
  sub_trainv = []
  for pid in range(partition_num):
    p_trainids = np.where(belongs == pid)[0]
    sub_trainv.append(p_trainids)
    p_v = np.where(r_belongs[pid] != -1)[0]
    sub_v.append(p_v)
    assert p_v.shape[0] == r_vnum[pid]
    print('vertex# with self-reliance: ', r_vnum[pid])
    print('vertex# w/o  self-reliance: ', p_vnum[pid])
    #print('orginal vertex: ', np.where(belongs == pid)[0])
    #print('redundancy vertex: ', np.where(r_belongs[pid] != -1)[0])
  return sub_v, sub_trainv



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Partition')
  parser.add_argument("--dataset", type=str, default=None,
                      help="dataset dir")
  parser.add_argument("--partition", type=int, default=2,
                      help="num of partitions")
  parser.add_argument("--num-hops", type=int, default=1,
                      help="num of hop neighbors required for a batch")
  parser.add_argument("--ordering", dest='ordering', action='store_true')
  parser.set_defaults(ordering=False)
  args = parser.parse_args()

  # get data
  adj = spsp.load_npz(os.path.join(args.dataset, 'adj.npz'))
  train_mask, val_mask, test_mask = data.get_masks(args.dataset)
  train_nids = np.nonzero(train_mask)[0].astype(np.int64)
  labels = data.get_labels(args.dataset)
  
  # ordering
  if args.ordering:
    print('re-ordering graphs...')
    adj = adj.tocsc()
    adj, vmap = ordering.reordering(adj, depth=args.num_hop) # vmap: orig -> new
    # save to files
    mapv = np.zeros(vmap.shape, dtype=np.int64)
    mapv[vmap] = np.arange(vmap.shape[0]) # mapv: new -> orig
    train_nids = np.sort(vmap[train_nids])
    spsp.save_npz(os.path.join(args.dataset, 'adj.npz'), adj)
    np.save(os.path.join(args.dataset, 'labels.npy'), labels[mapv])
    np.save(os.path.join(args.dataset, 'train.npy'), train_mask[mapv])
    np.save(os.path.join(args.dataset, 'val.npy'), val_mask[mapv])
    np.save(os.path.join(args.dataset, 'test.npy'), test_mask[mapv])
  
  # partition
  p_v, p_trainv = dg(args.partition, adj, train_nids, args.num_hop)
  
  # save to file
  partition_dataset = os.path.join(args.dataset, '{}naive'.format(args.partition))
  try:
    os.mkdir(partition_dataset)
  except FileExistsError:
    pass
  dgl_g = DGLGraph(adj, readonly=True)
  for pid, (pv, ptrainv) in enumerate(zip(p_v, p_trainv)):
    print('generating subgraph# {}...'.format(pid))
    #subadj, sub2fullid, subtrainid = node2graph(adj, pv, ptrainv)
    subadj, sub2fullid, subtrainid = get_sub_graph(dgl_g, ptrainv, args.num_hop)
    sublabel = labels[sub2fullid[subtrainid]]
    # files
    subadj_file = os.path.join(
      partition_dataset,
      'subadj_{}.npz'.format(str(pid)))
    sub_trainid_file = os.path.join(
      partition_dataset,
      'sub_trainid_{}.npy'.format(str(pid)))
    sub_train2full_file = os.path.join(
      partition_dataset,
      'sub_train2fullid_{}.npy'.format(str(pid)))
    sub_label_file = os.path.join(
      partition_dataset,
      'sub_label_{}.npy'.format(str(pid)))
    spsp.save_npz(subadj_file, subadj)
    np.save(sub_trainid_file, subtrainid)
    np.save(sub_train2full_file, sub2fullid)
    np.save(sub_label_file, sublabel)