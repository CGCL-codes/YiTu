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

from YiTu_GNN.NDP.model.gcn_nssc import GCNSampling
import YiTu_GNN.NDP.data as data

def init_process(rank, world_size, backend):
  os.environ['MASTER_ADDR'] = '127.0.0.1'
  os.environ['MASTER_PORT'] = '29501'
  dist.init_process_group(backend, rank=rank, world_size=world_size)
  torch.cuda.set_device(rank)
  torch.manual_seed(rank)
  print('rank [{}] process successfully launches'.format(rank))

def trainer(rank, world_size, args, backend='nccl'):
  # init multi process
  init_process(rank, world_size, backend)
  
  # load data
  dataname = os.path.basename(args.dataset)
  g = dgl.contrib.graph_store.create_graph_from_store(dataname, "shared_mem")
  labels = data.get_labels(args.dataset)
  n_classes = len(np.unique(labels))
  # masks for semi-supervised learning
  train_mask, val_mask, test_mask = data.get_masks(args.dataset)
  train_nid = np.nonzero(train_mask)[0].astype(np.int64)
  chunk_size = int(train_nid.shape[0] / world_size) - 1
  train_nid = train_nid[chunk_size * rank:chunk_size * (rank + 1)]
  test_nid = np.nonzero(test_mask)[0].astype(np.int64)
  # to torch tensor
  labels = torch.LongTensor(labels)
  train_mask = torch.ByteTensor(train_mask)
  val_mask = torch.ByteTensor(val_mask)
  test_mask = torch.ByteTensor(test_mask)

  # prepare model
  num_hops = args.n_layers if args.preprocess else args.n_layers + 1
  model = GCNSampling(args.feat_size,
                      args.n_hidden,
                      n_classes,
                      args.n_layers,
                      F.relu,
                      args.dropout,
                      args.preprocess)
  loss_fcn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),
                               lr=args.lr,
                               weight_decay=args.weight_decay)
  model.cuda(rank)
  model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
  ctx = torch.device(rank)

  # start training
  epoch_dur = []
  sampler = dgl.contrib.sampling.NeighborSampler(g, args.batch_size,
                                                    args.num_neighbors,
                                                    neighbor_type='in',
                                                    shuffle=True,
                                                    num_workers=args.num_workers,
                                                    num_hops=num_hops,
                                                    seed_nodes=train_nid,
                                                    prefetch=False)
  profile_begin = time.time()
  with torch.autograd.profiler.profile(enabled=(rank==0), use_cuda=True) as prof:
    for epoch in range(args.n_epochs):
      model.train()
      epoch_start_time = time.time()
      step = 0
      for nf in sampler:
        with torch.autograd.profiler.record_function('gpu-load'):
          nf.copy_from_parent(ctx=ctx)
          batch_nids = nf.layer_parent_nid(-1)
          label = labels[batch_nids]
          label = label.cuda(rank, non_blocking=True)
        step += 1
        if rank == 0 and step % 20 == 0:
          print('epoch [{}] step [{}]'.format(epoch + 1, step))
      if rank == 0:
        epoch_dur.append(time.time() - epoch_start_time)
        print('Epoch average time: {:.4f}'.format(np.mean(np.array(epoch_dur[2:]))))
  print('Total Time: {:.4f}s'.format(time.time() - profile_begin))
  if rank == 0:
    print(prof.key_averages().table(sort_by='cuda_time_total'))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='GCN Profiling')

  parser.add_argument("--gpu", type=str, default='cpu',
                      help="gpu ids. such as 0 or 0,1,2")
  parser.add_argument("--dataset", type=str, default=None,
                      help="path to the dataset folder")
  # model arch
  parser.add_argument("--feat-size", type=int, default=600,
                      help='input feature size')
  parser.add_argument("--dropout", type=float, default=0.2,
                      help="dropout probability")
  parser.add_argument("--n-hidden", type=int, default=32,
                      help="number of hidden gcn units")
  parser.add_argument("--n-layers", type=int, default=1,
                      help="number of hidden gcn layers")
  parser.add_argument("--preprocess", dest='preprocess', action='store_true')
  parser.set_defaults(preprocess=False)
  # training hyper-params
  parser.add_argument("--lr", type=float, default=3e-2,
                      help="learning rate")
  parser.add_argument("--n-epochs", type=int, default=10,
                      help="number of training epochs")
  parser.add_argument("--batch-size", type=int, default=6000,
                      help="batch size")
  parser.add_argument("--weight-decay", type=float, default=0,
                      help="Weight for L2 loss")
  # sampling hyper-params
  parser.add_argument("--num-neighbors", type=int, default=2,
                      help="number of neighbors to be sampled")
  parser.add_argument("--num-workers", type=int, default=16)

  args = parser.parse_args()

  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
  gpu_num = len(args.gpu.split(','))

  mp.spawn(trainer, args=(gpu_num, args), nprocs=gpu_num, join=True)
  