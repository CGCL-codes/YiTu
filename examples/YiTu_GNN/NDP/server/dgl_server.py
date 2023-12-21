import os
import argparse, time, math
import numpy as np
from scipy import sparse as spsp
import torch
import dgl
from dgl._deprecate.graph import DGLGraph
from dgl.data import register_data_args, load_data

def main(args):
  data = load_data(args)

  if args.self_loop and not args.dataset.startswith('reddit'):
    data.graph.add_edges_from([(i,i) for i in range(len(data.graph))])

  features = torch.FloatTensor(data.features)
  labels = torch.LongTensor(data.labels)
  train_mask = torch.ByteTensor(data.train_mask)
  val_mask = torch.ByteTensor(data.val_mask)
  test_mask = torch.ByteTensor(data.test_mask)
  in_feats = features.shape[1]
  n_classes = data.num_labels
  n_edges = data.graph.number_of_edges()

  n_train_samples = train_mask.sum().item()
  n_val_samples = val_mask.sum().item()
  n_test_samples = test_mask.sum().item()

  graph_name = args.dataset

  print("""----Data statistics------'
    #Edges %d
    #Classes %d
    #Train samples %d
    #Val samples %d
    #Test samples %d""" %
        (n_edges, n_classes,
            n_train_samples,
            n_val_samples,
            n_test_samples))
  
  g = dgl.contrib.graph_store.create_graph_store_server(
        data.graph, graph_name,
        'shared_mem', args.num_workers, 
        False)
  dgl_g = DGLGraph(data.graph, readonly=True)
  norm = 1. / dgl_g.in_degrees().float().unsqueeze(1)
  del dgl_g
  g.ndata['norm'] = norm
  g.ndata['features'] = features
  g.ndata['labels'] = labels
  g.ndata['train_mask'] = train_mask
  g.ndata['val_mask'] = val_mask
  g.ndata['test_mask'] = test_mask
  print('start running graph server on dataset: {}'.format(graph_name))
  g.run()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='GCN')
  register_data_args(parser)
  parser.add_argument("--graph-file", type=str, default="",
    help="graph file")
  parser.add_argument("--num-feats", type=int, default=600,
    help="the number of features")
  parser.add_argument("--self-loop", action='store_true',
    help="graph self-loop (default=False)")
  parser.add_argument("--num-workers", type=int, default=1,
    help="the number of workers")
  args = parser.parse_args()
  
  main(args)
  print('graph server end')