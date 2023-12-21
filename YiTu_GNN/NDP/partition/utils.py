import os
import sys
import dgl
from dgl._deprecate.graph import DGLGraph
import torch
import numpy as np
import scipy.sparse as spsp

def get_sub_graph(dgl_g, train_nid, num_hops):
  nfs = []
  for nf in dgl.contrib.sampling.NeighborSampler(dgl_g, len(train_nid),
                                                 dgl_g.number_of_nodes(),
                                                 neighbor_type='in',
                                                 shuffle=False,
                                                 num_workers=16,
                                                 num_hops=num_hops,
                                                 seed_nodes=train_nid,
                                                 prefetch=False):
    nfs.append(nf)
  
  assert(len(nfs) == 1)
  nf = nfs[0]
  full_edge_src = []
  full_edge_dst = []
  for i in range(nf.num_blocks):
    nf_src_nids, nf_dst_nids, _ = nf.block_edges(i, remap_local=False)
    full_edge_src.append(nf.map_to_parent_nid(nf_src_nids))
    full_edge_dst.append(nf.map_to_parent_nid(nf_dst_nids))
  full_srcs = torch.cat(tuple(full_edge_src)).numpy()
  full_dsts = torch.cat(tuple(full_edge_dst)).numpy()
  # set up mappings
  sub2full = np.unique(np.concatenate((full_srcs, full_dsts)))
  full2sub = np.zeros(np.max(sub2full) + 1, dtype=np.int64)
  full2sub[sub2full] = np.arange(len(sub2full), dtype=np.int64)
  # map to sub graph space
  sub_srcs = full2sub[full_srcs]
  sub_dsts = full2sub[full_dsts]
  vnum = len(sub2full)
  enum = len(sub_srcs)
  data = np.ones(sub_srcs.shape[0], dtype=np.uint8)
  coo_adj = spsp.coo_matrix((data, (sub_srcs, sub_dsts)), shape=(vnum, vnum))
  csr_adj = coo_adj.tocsr() # remove redundant edges
  enum = csr_adj.data.shape[0]
  csr_adj.data = np.ones(enum, dtype=np.uint8)
  print('vertex#: {} edge#: {}'.format(vnum, enum))
  # train nid
  tnid = nf.layer_parent_nid(-1).numpy()
  valid_t_max = np.max(sub2full)
  valid_t_min = np.min(tnid)
  tnid = np.where(tnid <= valid_t_max, tnid, valid_t_min)
  subtrainid = full2sub[np.unique(tnid)]
  return csr_adj, sub2full, subtrainid


def node2graph(fulladj, nodelist, train_nids):
  g = DGLGraph(fulladj)
  subg = g.subgraph(nodelist)
  sub2full = subg.parent_nid.numpy()
  subadj = subg.adjacency_matrix_scipy(transpose=True, return_edge_ids=False)
  # get train vertices under subgraph scope
  subtrain = subg.map_to_subgraph_nid(train_nids).numpy()
  return subadj, sub2full, subtrain