import os
import sys
import dgl
from dgl._deprecate.graph import DGLGraph
import numpy as np
import scipy.sparse as spsp
import argparse
import YiTu_GNN.NDP.data as data

def multi_in_neighbors(csc_adj, nids):
  neighs = []
  for nid in nids:
    in_neighbors = csc_adj.indices[csc_adj.ixgnn.NDPtr[nid]: csc_adj.ixgnn.NDPtr[nid+1]]
    neighs.append(in_neighbors)
  return np.unique(np.hstack(neighs))
    

def num_edges(csc_adj):
  return csc_adj.indices.shape[0]

def reordering(csc_adj, depth=1):
  vnum = csc_adj.shape[0]
  enum = num_edges(csc_adj)
  in_degrees = csc_adj.ixgnn.NDPtr[1:] - csc_adj.ixgnn.NDPtr[:-1]
  nids = np.argsort(-in_degrees) # sort with descending order
  # construct vertex mappings from old graph to new graph
  edge_src = np.zeros(enum, dtype=np.int64)
  edge_dst = np.zeros(enum, dtype=np.int64)
  vmap = -np.ones(vnum, dtype=np.int64)
  maps = 0
  progress = 0 # for informing progress bar
  for step, nid in enumerate(nids):
    if vmap[nid] == -1:
      vmap[nid] = maps
      maps += 1
      in_neighs = np.array([nid], dtype=np.int64)
      for _ in range(depth):
        in_neighs = multi_in_neighbors(csc_adj, in_neighs)
        for vnei in in_neighs:
          if vmap[nid] == -1:
            vmap[vnei] = maps
            maps += 1
    if vnum * progress // 100 <= step:
      sys.stdout.write('=>{}%\r'.format(progress))
      sys.stdout.flush()
      progress += 1
  print('')
  assert maps == vnum
  # construct new graph
  coo_adj = csc_adj.tocoo()
  vsrc = vmap[coo_adj.row]
  vdst = vmap[coo_adj.col]
  new_coo_adj = spsp.coo_matrix((coo_adj.data, (vsrc, vdst)), shape=(vnum, vnum))
  return new_coo_adj, vmap


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Partition')
  parser.add_argument("--dataset", type=str, default=None,
                      help="dataset dir")
  parser.add_argument("--num-hop", type=int, default=1,
                      help="num of hop neighbors required for a batch")
  args = parser.parse_args()

  adj = spsp.load_npz(os.path.join(args.dataset, 'adj.npz'))
  print('orig adj matrix: ')
  print(adj.todense())
  new_adj, maps = reordering(adj.tocsc(), depth=args.num_hop)
  print('vertex mappings from orig graph to new graph:')
  print(maps)
  print('new adj matrix:')
  print(new_adj.todense())