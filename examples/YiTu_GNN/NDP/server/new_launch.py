import sys
import os
import argparse
import numpy as np
import torch
import dgl
from dgl._deprecate.graph import DGLGraph
from dgl.contrib.sampling import SamplerPool
import dgl.function as fn
import multiprocessing

import YiTu_GNN.NDP.data as data
from YiTu_GNN.NDP.parallel import SampleDeliver

def main(args):
  coo_adj, feat = data.get_graph_data(args.dataset)

  graph = DGLGraph(coo_adj, readonly=True)
  features = torch.FloatTensor(feat)

  graph_name = os.path.basename(args.dataset)
  vnum = graph.number_of_nodes()
  enum = graph.number_of_edges()
  feat_size = feat.shape[1]

  print('=' * 30)
  print("Graph Name: {}\nNodes Num: {}\tEdges Num: {}\nFeature Size: {}"
        .format(graph_name, vnum, enum, feat_size)
  )
  print('=' * 30)

  # create server
  g = dgl.contrib.graph_store.create_graph_store_server(
        graph, graph_name,
        'shared_mem', args.num_workers, 
        False, edge_dir='in')
  
  # calculate norm for gcn
  dgl_g = DGLGraph(graph, readonly=True)

  if args.model == 'gcn':
    dgl_g = DGLGraph(graph, readonly=True)
    norm = 1. / dgl_g.in_degrees().float().unsqueeze(1)
    # preprocess 
    if args.preprocess:
      print('Preprocessing features...')
      dgl_g.ndata['norm'] = norm
      dgl_g.ndata['features'] = features
      dgl_g.update_all(fn.copy_src(src='features', out='m'),
                       fn.sum(msg='m', out='preprocess'),
                       lambda node : {'preprocess': node.data['preprocess'] * node.data['norm']})
      features = dgl_g.ndata['preprocess']
    g.ndata['norm'] = norm
    g.ndata['features'] = features
    del dgl_g

  elif args.model == 'graphsage':
    if args.preprocess: # for simple preprocessing
      print('preprocessing: warning: jusy copy')
      g.ndata['neigh'] = features
    g.ndata['features'] = features

  # remote sampler 
  if args.sample:
    train_mask, val_mask, test_mask = data.get_masks(args.dataset)
    train_nid = np.nonzero(train_mask)[0].astype(np.int64)
    hops = args.gnn_layers - 1 if args.preprocess else args.gnn_layers
    print('Expected trainer#: {}. Start sampling at server end...'.format(args.num_workers))
    deliver = SampleDeliver(graph, train_nid, args.num_neighbors, hops, args.num_workers)
    deliver.async_sample(args.n_epochs, args.batch_size, one2all=args.one2all)
      
  print('start running graph server on dataset: {}'.format(graph_name))
  g.run()




if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='GraphServer')

  parser.add_argument("--dataset", type=str, default=None,
                      help="dataset folder path")
  
  parser.add_argument("--num-workers", type=int, default=1,
                      help="the number of workers")
  
  parser.add_argument("--model", type=str, default="gcn",
                      help="model type for preprocessing")

  # sample options
  parser.add_argument("--sample", dest='sample', action='store_true')
  parser.set_defaults(sample=False)
  parser.add_argument("--num-neighbors", type=int, default=2)
  parser.add_argument("--gnn-layers", type=int, default=2)
  parser.add_argument("--batch-size", type=int, default=6000)
  #parser.add_argument("--num-workers", type=int, default=8)
  parser.add_argument("--n-epochs", type=int, default=10)
  parser.add_argument("--one2all", dest='one2all', action='store_true')
  parser.set_defaults(one2all=False)

  parser.add_argument("--preprocess", dest='preprocess', action='store_true')
  parser.set_defaults(preprocess=False)
  
  args = parser.parse_args()
  main(args)