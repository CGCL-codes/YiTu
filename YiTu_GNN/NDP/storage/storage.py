import os
import sys
# set environment
#module_name ='YiTu_GNN.NDP'
#modpath = os.path.abspath('.')
#if module_name in modpath:
#  idx = modpath.find(module_name)
#  modpath = modpath[:idx]
#sys.path.append(modpath)

import numpy as np
import numba
import torch
from dgl._deprecate.graph import DGLGraph
from dgl._deprecate.frame import Frame, FrameRef
import dgl.utils

class GraphCacheServer:
  """
  Manage graph features
  Automatically fetch the feature tensor from CPU or GPU
  """
  def __init__(self, graph, node_num, nid_map, gpuid):
    """
    Paramters:
      graph:   should be created from `dgl.contrib.graph_store`
      node_num: should be sub graph node num
      nid_map: torch tensor. map from local node id to full graph id.
               used in fetch features from remote
    """
    self.graph = graph
    self.gpuid = gpuid
    self.node_num = node_num
    self.nid_map = nid_map.clone().detach().cuda(self.gpuid)
    self.nid_map.requires_grad_(False)
    
    # masks for manage the feature locations: default in CPU
    self.gpu_flag = torch.zeros(self.node_num).bool().cuda(self.gpuid)
    self.gpu_flag.requires_grad_(False)

    self.cached_num = 0
    self.capability = node_num

    # gpu tensor cache
    self.full_cached = False
    self.dims = {}          # {'field name': dims of the tensor data of a node}
    self.total_dim = 0
    self.gpu_fix_cache = dict() # {'field name': tensor data for cached nodes in gpu}
    with torch.cuda.device(self.gpuid):
      self.localid2cacheid = torch.cuda.LongTensor(node_num).fill_(0)
      self.localid2cacheid.requires_grad_(False)
    
    # logs
    self.log = False
    self.try_num = 0
    self.miss_num = 0

  
  def init_field(self, embed_names):
    with torch.cuda.device(self.gpuid):
      nid = torch.cuda.LongTensor([0])
    feats = self.get_feat_from_server(nid, embed_names)
    self.total_dim = 0
    for name in embed_names:
      self.dims[name] = feats[name].size(1)
      self.total_dim += feats[name].size(1)
    #print('total dims: {}'.format(self.total_dim))


  def auto_cache(self, dgl_g, embed_names):
    """
    Automatically cache the node features
    Params:
      g: DGLGraph for local graphs
      embed_names: field name list, e.g. ['features', 'norm']
    """
    # Step1: get available GPU memory
    peak_allocated_mem = torch.cuda.max_memory_allocated(device=self.gpuid)
    peak_cached_mem = torch.cuda.max_memory_cached(device=self.gpuid)
    total_mem = torch.cuda.get_device_properties(self.gpuid).total_memory
    available = total_mem - peak_allocated_mem - peak_cached_mem \
                - 1024 * 1024 * 1024 # in bytes
    # Stpe2: get capability
    self.capability = int(available / (self.total_dim * 4)) # assume float32 = 4 bytes
    #self.capability = int(6 * 1024 * 1024 * 1024 / (self.total_dim * 4))
    #self.capability = int(self.node_num * 0.8)
    print('Cache Memory: {:.2f}G. Capability: {}'
          .format(available / 1024 / 1024 / 1024, self.capability))
    # Step3: cache
    if self.capability >= self.node_num:
      # fully cache
      print('cache the full graph...')
      full_nids = torch.arange(self.node_num).cuda(self.gpuid)
      data_frame = self.get_feat_from_server(full_nids, embed_names)
      self.cache_fix_data(full_nids, data_frame, is_full=True)
    else:
      # choose top-cap out-degree nodes to cache
      print('cache the part of graph... caching percentage: {:.4f}'
            .format(self.capability / self.node_num))
      out_degrees = dgl_g.out_degrees()
      sort_nid = torch.argsort(out_degrees, descending=True)
      cache_nid = sort_nid[:self.capability]
      data_frame = self.get_feat_from_server(cache_nid, embed_names)
      self.cache_fix_data(cache_nid, data_frame, is_full=False)


  def get_feat_from_server(self, nids, embed_names, to_gpu=False):
    """
    Fetch features of `nids` from remote server in shared CPU
    Params
      g: created from `dgl.contrib.graph_store.create_graph_from_store`
      nids: required node ids in local graph, should be in gpu
      embed_names: field name list, e.g. ['features', 'norm']
    Return:
      feature tensors of these nids (in CPU)
    """
    nids_in_full = self.nid_map[nids]
    #cpu_frame = self.graph._node_frame[dgl.utils.toindex(nids_in_full.cpu())]
    #data_frame = {}
    #for name in embed_names:
    #  if to_gpu:
    #    data_frame[name] = cpu_frame[name].cuda(self.gpuid)
    #  else:
    #    data_frame[name] = cpu_frame[name]
    #return data_frame
    nids = nids_in_full.cpu()
    if to_gpu:
      frame = {name: self.graph._node_frame._frame[name].data[nids].cuda(self.gpuid, non_blocking=True)\
                   for name in embed_names}
    else:
      frame = {name: self.graph._node_frame._frame[name].data[nids] for name in embed_names}
    return frame
  
  
  def cache_fix_data(self, nids, data, is_full=False):
    """
    User should make sure tensor data under every field name should
    have same num (axis 0)
    Params:
      nids: node ids to be cached in local graph.
            should be equal to data rows. should be in gpu
      data: dict: {'field name': tensor data}
    """
    rows = nids.size(0)
    self.localid2cacheid[nids] = torch.arange(rows).cuda(self.gpuid)
    self.cached_num = rows
    for name in data:
      data_rows = data[name].size(0)
      assert (rows == data_rows)
      self.dims[name] = data[name].size(1)
      self.gpu_fix_cache[name] = data[name].cuda(self.gpuid)
    # setup flags
    self.gpu_flag[nids] = True
    self.full_cached = is_full

  
  def fetch_data(self, nodeflow):
    """
    copy feature from local GPU memory or
    remote CPU memory, which depends on feature
    current location.
    --Note: Should be paralleled
    Params:
      nodeflow: DGL nodeflow. all nids in nodeflow should
                under sub-graph space
    """
    if self.full_cached:
      self.fetch_from_cache(nodeflow)
      return
    with torch.autograd.profiler.record_function('cache-idxload'):
      nf_nids = nodeflow._node_mapping.tousertensor().cuda(self.gpuid)
      offsets = nodeflow._layer_offsets
    for i in range(nodeflow.num_layers):
      #with torch.autograd.profiler.record_function('cache-idx-load'):
        #tnid = nodeflow.layer_parent_nid(i).cuda(self.gpuid)
      tnid = nf_nids[offsets[i]:offsets[i+1]]
      # get nids -- overhead ~0.1s
      with torch.autograd.profiler.record_function('cache-index'):
        gpu_mask = self.gpu_flag[tnid]
        nids_in_gpu = tnid[gpu_mask]
        cpu_mask = ~gpu_mask
        nids_in_cpu = tnid[cpu_mask]
      # create frame
      with torch.autograd.profiler.record_function('cache-allocate'):
        with torch.cuda.device(self.gpuid):
          frame = {name: torch.cuda.FloatTensor(tnid.size(0), self.dims[name]) \
                    for name in self.dims}
      # for gpu cached tensors: ##NOTE: Make sure it is in-place update!
      with torch.autograd.profiler.record_function('cache-gpu'):
        if nids_in_gpu.size(0) != 0:
          cacheid = self.localid2cacheid[nids_in_gpu]
          for name in self.dims:
            frame[name][gpu_mask] = self.gpu_fix_cache[name][cacheid]
      # for cpu cached tensors: ##NOTE: Make sure it is in-place update!
      with torch.autograd.profiler.record_function('cache-cpu'):
        if nids_in_cpu.size(0) != 0:
          cpu_data_frame = self.get_feat_from_server(
            nids_in_cpu, list(self.dims), to_gpu=True)
          for name in self.dims:
            frame[name][cpu_mask] = cpu_data_frame[name]
      with torch.autograd.profiler.record_function('cache-asign'):
        nodeflow._node_frames[i] = FrameRef(Frame(frame))
      if self.log:
        self.log_miss_rate(nids_in_cpu.size(0), tnid.size(0))


  def fetch_from_cache(self, nodeflow):
    for i in range(nodeflow.num_layers):
      #nid = dgl.utils.toindex(nodeflow.layer_parent_nid(i))
      with torch.autograd.profiler.record_function('cache-idxload'):
        tnid = nodeflow.layer_parent_nid(i).cuda(self.gpuid)
      with torch.autograd.profiler.record_function('cache-gpu'):
        frame = {}
        for name in self.gpu_fix_cache:
          frame[name] = self.gpu_fix_cache[name][tnid]
      nodeflow._node_frames[i] = FrameRef(Frame(frame))

  
  def log_miss_rate(self, miss_num, total_num):
    self.try_num += total_num
    self.miss_num += miss_num
  
  def get_miss_rate(self):
    miss_rate = float(self.miss_num) / self.try_num
    self.miss_num = 0
    self.try_num = 0
    return miss_rate