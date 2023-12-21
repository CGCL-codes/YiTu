import os
import sys
# set environment
#module_name ='YiTu_GNN.NDP'
#modpath = os.path.abspath('.')
#if module_name in modpath:
#  idx = modpath.find(module_name)
#  modpath = modpath[:idx]
#sys.path.append(modpath)

import dgl
import torch
import numpy as np
import multiprocessing as mp
import socket

barrier_interval = 50

class SampleLoader:
  """ SampleLoader
  sample load pipeline
  """
  def __init__(self, graph, rank, one2all=True):
    # connect to server sampler:
    barrier_rank = 0 if one2all else rank
    self._barrier = SampleBarrier('trainer', rank=barrier_rank)

    self._graph = graph
    self._rank = rank
    self._port = 8760
    # wait for recving samples
    self._recver = dgl.contrib.sampling.SamplerReceiver(
      self._graph, 
      '127.0.0.1:' + str(self._port + rank),
      1, # sender num
      net_type='socket'
    )
    self._batch_num = 0
    self._barrier_interval = barrier_interval
    self._sampler_iter = None


  def __iter__(self):
    self._batch_num = 0
    self._recver_iter = iter(self._recver)
    return self

  
  def __next__(self):
    try:
      nf = next(self._recver_iter)
    except StopIteration:
      # end of an epoch
      self._barrier.barrier()
      self._batch_num = 0
      raise StopIteration
    self._batch_num += 1
    #if self._batch_num % self._barrier_interval == 0:
    if self._batch_num % self._barrier_interval == 0:
      self._barrier.barrier()
    return nf


  def __del__(self):
    del self._recver
  

class SampleDeliver:
  """ Sample Deliver
  deliver sample through network
  """
  def __init__(self, graph,
               train_nid,
               neighbor_num,
               hops,
               trainer_num):
    self._graph = graph
    self._train_nid = train_nid
    self._neighbor_num = neighbor_num
    self._hops = hops
    self._trainer_num = trainer_num
    self._sender_port = 8760
    self._proc = None
    self._one2all = True
    self._barrier_interval = barrier_interval


  def async_sample(self, epoch, batch_size, one2all=True):
    self._one2all = one2all
    if one2all:
      self._proc = mp.Process(target=self.one2all_sample,
                              args=(epoch, batch_size))
      self._proc.start()
    else:
      if not isinstance(self._train_nid, list):
        chunk_size = int(self._train_nid.shape[0] / self._trainer_num) - 1
      #self._proc = mp.Pool()
      self._proc = []
      for rank in range(self._trainer_num):
        print('starting child sampler process {}'.format(rank))
        if isinstance(self._train_nid, list):
          sampler_nid = self._train_nid[rank]
        else:
          sampler_nid = self._train_nid[chunk_size * rank:chunk_size * (rank + 1)]
        #self._proc.apply_async(self.one2one_sample,
        #                       args=(epoch, batch_size, sampler_nid, rank))
        proc = mp.Process(target=self.one2one_sample,
                          args=(epoch, batch_size, sampler_nid, rank))
        proc.start()
        self._proc.append(proc)

  def one2all_sample(self, epoch_num, batch_size):
    # waiting trainers connecting
    barrier = SampleBarrier('server', trainer_num=self._trainer_num)

    namebook = {tid: '127.0.0.1:' + str(self._sender_port + tid)\
                        for tid in range(self._trainer_num)}
    sampler = dgl.contrib.sampling.NeighborSampler(
                self._graph, batch_size,
                self._neighbor_num, neighbor_type='in',
                shuffle=True, num_workers=self._trainer_num * 2,
                num_hops=self._hops, seed_nodes=self._train_nid,
                prefetch=True, add_self_loop=False
    )
    sender = dgl.contrib.sampling.SamplerSender(namebook, net_type='socket')
    for epoch in range(epoch_num):
      tid = 0
      idx = 0
      for nf in sampler:
        # non-blocking send
        sender.send(nf, tid % self._trainer_num)
        tid += 1
        if tid % self._trainer_num == 0:
          idx += 1
          #print('sent batch ', idx)
          if idx % self._barrier_interval == 0:
            barrier.barrier()
      # temporary solution: makeup the unbalanced pieces
      print('Epoch {} end. Next tid: {}'.format(epoch+1, tid % self._trainer_num))
      while tid % self._trainer_num != 0:
        sender.send(nf, tid % self._trainer_num)
        print('Epoch {}: Makeup Sending tid: {}'.format(epoch+1, tid % self._trainer_num))
        tid += 1
      # end of epoch
      for tid in range(self._trainer_num):
        sender.signal(tid)
      barrier.barrier()
  

  def one2one_sample(self, epoch_num, batch_size, train_nid, rank):
    # waiting trainers connecting
    barrier = SampleBarrier('server', rank=rank)
    namebook = {0: '127.0.0.1:'+ str(self._sender_port + rank)}
    sender = dgl.contrib.sampling.SamplerSender(namebook, net_type='socket')
    graph = self._graph[rank] if isinstance(self._graph, list) else self._graph
    sampler = dgl.contrib.sampling.NeighborSampler(
      graph, batch_size,
      self._neighbor_num, neighbor_type='in',
      shuffle=True, num_workers=4,
      num_hops=self._hops, seed_nodes=train_nid,
      prefetch=True, add_self_loop=False
    )
    for epoch in range(epoch_num):
      idx = 0
      for nf in sampler:
        sender.send(nf, 0)
        idx += 1
        if idx % self._barrier_interval == 0:
          barrier.barrier()
      sender.signal(0)
      # barrier
      barrier.barrier()
      

  def __del__(self):
    if not self._proc is None:
      if self._one2all:
        self._proc.join()
      else:
        self._proc.close()
        self._proc.join()


class SampleBarrier:
  
  def __init__(self, role, trainer_num=1, rank=0):
    """
    Params:
      role       :
        'server' or 'trainer'
      trainer_num: 
        for role == 'server'
      rank       : 
        for one2one sampling
    """
    self._ip = '127.0.0.1'
    self._port = 8200 + rank
    self._role = role
    if self._role == 'server':
      print('start listening at: ' + self._ip + ' : ' + str(self._port))
      self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      self._server_sock.bind((self._ip, self._port))
      self._server_sock.listen(8)
      self._socks = []
      while trainer_num != 0:
        clientsocket, addr = self._server_sock.accept()
        print('recv a connection. Waiting for connections of {} trainers'.format(trainer_num-1))
        clientsocket.setblocking(1)
        self._socks.append(clientsocket)
        trainer_num -= 1
    elif self._role == 'trainer':
      print('[{}]: try connecting server at: '.format(rank) + self._ip + ' : ' + str(self._port))
      self._socks = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      self._socks.connect((self._ip, self._port))
      self._socks.setblocking(0) # to non-blocking
      print('connected to a remote sampler.')
    else:
      print('Unknown role')
      sys.exit(-1)
  

  def barrier(self):
    if self._role == 'server':
      for sock in self._socks:
        _ = sock.recv(128)
    else:
      self._socks.send('barrier'.encode('utf-8'))