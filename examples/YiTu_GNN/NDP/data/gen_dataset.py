
import networkx as nx
import scipy.sparse as spsp
import os
import argparse

def uniform_random_graph(nodes, edges, is_direct=False):
  nx_graph = nx.generators.random_graphs.gnm_random_graph(
    nodes, edges, directed=is_direct)
  return nx.to_scipy_sparse_matrix(nx_graph, weight=None, format='coo')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='GenDataset')

  parser.add_argument('--dist', type=str, default='uniform',
                      help='distribution of edges: uniform')
  
  parser.add_argument('--nodes', type=int, default=None,
                      help='node nums')
  parser.add_argument('--edges', type=int, default=None,
                      help='edge nums')
  
  parser.add_argument('--output', type=str, default=None,
                      help='output folder')
  
  parser.add_argument('--directed', dest='directed', action='store_true')
  parser.set_defaults(directed=False)
  
  args = parser.parse_args()

  if args.dist == 'uniform':
    adj = uniform_random_graph(args.nodes, args.edges, args.directed)
    spsp.save_npz(args.output, adj)
  else:
    print('Unknown distribution')