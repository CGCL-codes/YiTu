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
import YiTu_GNN.NDP.utils

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
    print('Expected trainer#: {}. Start sampling at server end...'.format(args.num_workers))
    if args.one2all:
      sampler_proc = multiprocessing.Process(target=sample_one2all, args=(graph, train_nid, args))
      sampler_proc.start()
    else:
      sampler_proc = sample_one2one(graph, train_nid, args)
  
  print('start running graph server on dataset: {}'.format(graph_name))
  g.run()
  sampler_proc.close()
  sampler_proc.join()


def sample_one2all(graph, train_nid, args):
  # wait all trainers to connect
  sock = YiTu_GNN.NDP.utils.server(args.num_workers)
  start_port = 8760
  n_trainer = args.num_workers
  num_hops = args.gnn_layers - 1 if args.preprocess else args.gnn_layers
  namebook = {tid: '127.0.0.1:'+str(start_port+tid) for tid in range(n_trainer)}
  sender = dgl.contrib.sampling.SamplerSender(namebook, net_type='socket')
  sampler = dgl.contrib.sampling.NeighborSampler(graph, args.batch_size,
              args.num_neighbors, neighbor_type='in',
              shuffle=True, num_workers=n_trainer,
              num_hops=num_hops, seed_nodes=train_nid,
              prefetch=True, add_self_loop=False)
  
  for epoch in range(args.n_epochs):
    tid = 0
    idx = 0
    for nf in sampler:
      # send is a non-blocking func
      sender.send(nf, tid % n_trainer)
      tid += 1
      if tid % n_trainer == 0:
        idx += 1
        if idx % 100 == 0:
          YiTu_GNN.NDP.utils.barrier(sock, role='server')
    # temporary solution: makeup the unbalanced pieces
    print('Epoch {} end. Next tid: {}'.format(epoch+1, tid % n_trainer))
    while tid % n_trainer != 0:
      sender.send(nf, tid % n_trainer)
      print('Epoch {}: Makeup Sending tid: {}'.format(epoch+1, tid % n_trainer))
      tid += 1
    # signal all trainers for the end of one epoch
    for tid in range(n_trainer):
      sender.signal(tid)
    # set barrier for waiting all trainers finish the current epoch
    YiTu_GNN.NDP.utils.barrier(sock, role='server')


def sample_one2one(graph, train_nid, args):
  n_trainer = args.num_workers
  chunk_size = int(train_nid.shape[0] / n_trainer) - 1
  p = multiprocessing.Pool()
  for rank in range(args.num_workers):
    print('Starting child sampler process {}'.format(rank))
    sampler_nid = train_nid[chunk_size * rank:chunk_size * (rank + 1)]
    p.apply_async(single_sampler, args=(graph, sampler_nid, rank, args))
  return p


def single_sampler(graph, train_nid, rank, args):
  sock = YiTu_GNN.NDP.utils.server(1, port=8200+rank)
  n_trainer = args.num_workers
  start_port = 8760
  num_hops = args.gnn_layers - 1 if args.preprocess else args.gnn_layers
  namebook = {0: '127.0.0.1:'+str(start_port+rank)}
  sender = dgl.contrib.sampling.SamplerSender(namebook, net_type='socket')
  sampler = dgl.contrib.sampling.NeighborSampler(graph, args.batch_size,
            args.num_neighbors, neighbor_type='in',
            shuffle=True, num_workers=2,
            num_hops=num_hops, seed_nodes=train_nid,
            prefetch=True, add_self_loop=False)
  for epoch in range(args.n_epochs):
    idx = 0
    for nf in sampler:
      sender.send(nf, 0)
      idx += 1
      if idx % 100 == 0:
        YiTu_GNN.NDP.utils.barrier(sock, role='server')
    sender.signal(0)
    # barrier
    YiTu_GNN.NDP.utils.barrier(sock, role='server')



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