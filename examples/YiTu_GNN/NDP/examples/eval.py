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
import YiTu_GNN.NDP.data as data

def gnneval(args, infer_model, graph, labels, rank, test_nid):
  """
  Evaluation just on a single machine
  """
  ctx = torch.device(rank)

  num_hops = args.n_layers if args.preprocess else args.n_layers + 1
  for nf in dgl.contrib.sampling.NeighborSampler(graph,len(test_nid),
                                                 graph.number_of_nodes(),
                                                 neighbor_type='in',
                                                 num_workers=16,
                                                 num_hops=num_hops,
                                                 seed_nodes=test_nid):
    nf.copy_from_parent(ctx=ctx)
  
  for ckpt in range(args.start,args.end,args.interval):
    nf.copy_from_parent(ctx=ctx)
    train_model = torch.load(
      os.path.join(args.ckpt, args.arch + '_' + str(ckpt))
    )
    for infer_param, param in zip(infer_model.parameters(), train_model.parameters()):    
      infer_param.data.copy_(param.data)
    infer_model.cuda(ctx)

    num_acc = 0.
    infer_model.eval()
    with torch.no_grad():
      nf.copy_from_parent(ctx=ctx)
      pred = infer_model(nf)
      batch_nids = nf.layer_parent_nid(-1)
      batch_labels = labels[batch_nids].cuda(rank)
      num_acc += (pred.argmax(dim=1) == batch_labels).sum().cpu().item()
    
    print("[{}]: Test Accuracy {:.4f}".format(ckpt, num_acc / len(test_nid)))


def main(args):

  dataname = os.path.basename(args.dataset)
  g = dgl.contrib.graph_store.create_graph_from_store(dataname, "shared_mem")
  labels = data.get_labels(args.dataset)
  n_classes = len(np.unique(labels))
  train_mask, val_mask, test_mask = data.get_masks(args.dataset)
  test_nid = np.nonzero(test_mask)[0].astype(np.int64)

  labels = torch.LongTensor(labels)

  if args.arch == 'gcn-nssc':
    from YiTu_GNN.NDP.model.gcn_nssc import GCNSampling, GCNInfer
    infer_model = GCNInfer(args.feat_size,
                           32,
                           n_classes,
                           args.n_layers,
                           F.relu,
                           args.preprocess)
  elif args.arch == 'gs-nssc':
    from YiTu_GNN.NDP.model.graphsage_nssc import GraphSageSampling
    infer_model = GraphSageSampling(args.feat_size,
                            16,
                            n_classes,
                            args.n_layers,
                            F.relu,
                            0,
                            'mean',
                            args.preprocess)
  else:
    print('Unknown arch')
    sys.exit(-1)
  
  gnneval(args, infer_model, g, labels, 0, test_nid)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='GCNInfer')

  parser.add_argument("--gpu", type=int, default=None,
                      help="gpu id. such as 0 or 1 or 2")
  parser.add_argument("--dataset", type=str, default=None,
                      help="path to the dataset folder")
  parser.add_argument("--arch", type=str, default='gcn-nssc',
                      help='model arch')
  # model arch
  parser.add_argument("--feat-size", type=int, default=602,
                      help='input feature size')
  parser.add_argument("--n-layers", type=int, default=1,
                      help="number of hidden gcn layers")
  # training hyper-params
  parser.add_argument("--start", type=int, default=0,
                      help="eval epoch start")
  parser.add_argument("--interval", type=int, default=5,
                      help="eval epoch interval")
  parser.add_argument("--end", type=int, default=60,
                      help='eval epoch end (not include)')
  
  parser.add_argument("--ckpt", type=str, default='checkpoint',
                      help="checkpoint dir")
  
  parser.add_argument("--preprocess", dest='preprocess', action='store_true')
  parser.set_defaults(preprocess=False)
  
  args = parser.parse_args()

  os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
  
  main(args)