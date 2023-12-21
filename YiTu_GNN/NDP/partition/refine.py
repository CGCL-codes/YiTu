import os
import numpy as np
import scipy.sparse as spsp
import networkx as nx

from utils import *

def get_num_hop_in_neighbors(coo_adj, node_ids, num_hop, excluded_nodes=None):
  """
  Get num-hop neighbor idx for the given graph `coo_adj` and `node_ids`
  Return:
    nodes: list of nd arrays, [1-hop neighbors, 2-hop neighbors, ...]
  """
  select_mask = np.vectorize(exclude, excluded=['node_range'])

  neighbors = []
  for _ in range(num_hop):
    node_ids = get_in_neighbors(coo_adj, node_ids)
    neighbors.append(node_ids)
    if excluded_nodes is not None:
      mask = select_mask(nid=neighbors[-1], node_range=excluded_nodes)
      neighbors[-1] = neighbors[-1][mask]
  return neighbors


def build_train_graph(coo_adj, train_nids, num_hop):
  """
  Build training graphs
  Params:
    coo_adj: coo sparse adjacancy matrix for sub graph
    train_nids: np array for training node idx.
    num_hop: num-hop neighbors for each train node
  Returns:
    coo_adj: coo sparse adjacancy matrix for new graph
    train2fullid: new mappings for new training graph idx to full graph node idx
    valid_train_nids: valid train nids under full graph space. 
                      Note some train id may not be valid (i.e. no in-neighbors).
                      These nodes will still be included to sub graph.
  """
  # step 1: get in-neighbors for each train nids
  neighbors = get_num_hop_in_neighbors(coo_adj, train_nids, num_hop)
  # step 2: get edge (src, dst) pair
  #isin_mask_vfunc = np.vectorize(include, excluded=['node_range'])
  src = coo_adj.row
  dst = coo_adj.col
  neighbors = [train_nids] + neighbors
  train_src = []
  train_dst = []
  for hop in range(num_hop):
    hop_dst = neighbors[hop]
    hop_src = neighbors[hop+1]
    #src_mask = isin_mask_vfunc(nid=src, node_range=hop_src)
    #dst_mask = isin_mask_vfunc(nid=dst, node_range=hop_dst)
    src_mask = pinclude(src, hop_src)
    dst_mask = pinclude(dst, hop_dst)
    mask = src_mask * dst_mask
    hop_src = src[mask]
    hop_dst = dst[mask]
    if hop == 0:
      valid_train_nids = np.unique(hop_dst)
    train_src.append(hop_src)
    train_dst.append(hop_dst)
  train_src = np.concatenate(tuple(train_src))
  train_dst = np.concatenate(tuple(train_dst))
  # step 3: translate src, dst node ids to new namespace
  train2fullid = np.unique(np.concatenate((train_src, train_dst)))
  train_sub_src = full2sub_nid(train2fullid, train_src)
  train_sub_dst = full2sub_nid(train2fullid, train_dst)
  # step 4: build graph
  edge = np.ones(len(train_src), dtype=np.int)
  new_coo_adj = spsp.coo_matrix((edge, (train_sub_src, train_sub_dst)),
                                shape=(len(train2fullid), len(train2fullid)))
  return new_coo_adj, train2fullid, valid_train_nids
  

def wrap_neighbor(full_adj, sub_adj, sub2fullid, num_hop, train_nids=None):
  """
  Params:
    full_adj: coo sparse adjacancy matrix for full graph
    sub_adj:  coo sparse adjacancy matrix for sub graph
    sub2fullid: np array mapping sub_adj node idx to full graph node idx
    num_hop: num-hop neighbors for each node will be included in the sub graph
    train_nids: np array for training node idx. If provided, only training node
                neighbors will be included.
  Returns:
    sub_adj: coo sparse adjacancy matrix for wrapped sub graph
    sub2fullid: New mappings for new sub graph idx to full graph node idx
  """
  # step 1: get extra edge tuple (src, dst)
  isin_mask_vfunc = np.vectorize(include, excluded=['node_range'])
  sub_train_nids_infull_mask = isin_mask_vfunc(nid=sub2fullid, node_range=train_nids)
  sub_train_nids_infull = sub2fullid[sub_train_nids_infull_mask]
  nodes = sub_train_nids_infull
  neighbors = get_num_hop_in_neighbors(full_adj, nodes, num_hop,
                                   excluded_nodes=sub2fullid)
  neighbors = [nodes] + neighbors
  extra_src = []
  extra_dst = []
  full_src = full_adj.row
  full_dst = full_adj.col
  for hop in range(num_hop):
    # only in-edge will participate into computations
    dst = neighbors[hop]
    src = neighbors[hop + 1]
    extra_src_mask = isin_mask_vfunc(nid=full_src, node_range=src)
    extra_dst_mask = isin_mask_vfunc(nid=full_dst, node_range=dst)
    extra_mask = extra_src_mask * extra_dst_mask
    extra_src.append(full_src[extra_mask])
    extra_dst.append(full_dst[extra_mask])
  extra_src = np.concatenate(tuple(extra_src))
  extra_dst = np.concatenate(tuple(extra_dst))
  # step 2: translate extra node id, edges into sub graph namespace
  new_src = np.concatenate((sub2fullid[sub_adj.row], extra_src))
  new_dst = np.concatenate((sub2fullid[sub_adj.col], extra_dst))
  sub2fullid = np.unique(np.concatenate((new_src, new_dst)))
  new_sub_src = full2sub_nid(sub2fullid, new_src)
  new_sub_dst = full2sub_nid(sub2fullid, new_dst)
  # step 3: construct new sub graph
  edge = np.ones(len(new_src), dtype=np.int)
  new_coo_adj = spsp.coo_matrix((edge, (new_sub_src, new_sub_dst)),
                                shape=(len(sub2fullid), len(sub2fullid)))
  return new_coo_adj, sub2fullid


