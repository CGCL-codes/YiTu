import argparse, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl._deprecate.graph import DGLGraph

class NodeUpdate(nn.Module):
  def __init__(self, layer_id, in_feats, out_feats, dropout, activation=None, test=False, concat=False):
    super(NodeUpdate, self).__init__()
    self.layer_id = layer_id
    self.linear = nn.Linear(in_feats, out_feats)
    self.dropout = None
    if dropout != 0:
        self.dropout = nn.Dropout(p=dropout)
    self.activation = activation
    self.concat = concat
    self.test = test

  def forward(self, node):
    h = node.data['h']
    if self.test:
        norm = node.data['norm']
        h = h * norm
    else:
        agg_history_str = 'agg_h_{}'.format(self.layer_id-1)
        agg_history = node.data[agg_history_str]
        # control variate
        h = h + agg_history
        if self.dropout:
            h = self.dropout(h)
    h = self.linear(h)
    if self.concat:
        h = torch.cat((h, self.activation(h)), dim=1)
    elif self.activation:
        h = self.activation(h)
    return {'activation': h}


class GCNSampling(nn.Module):
  def __init__(self,
               in_feats,
               n_hidden,
               n_classes,
               n_layers,
               activation,
               dropout):
    super(GCNSampling, self).__init__()
    self.n_layers = n_layers
    self.dropout = None
    self.preprocess = False
    if dropout != 0:
        self.dropout = nn.Dropout(p=dropout)
    self.activation = activation
    # input layer
    self.linear = nn.Linear(in_feats, n_hidden)
    self.layers = nn.ModuleList()
    # hidden layers
    for i in range(1, n_layers):
        skip_start = (i == n_layers-1)
        self.layers.append(NodeUpdate(i, n_hidden, n_hidden, dropout, activation, concat=skip_start))
    # output layer
    self.layers.append(NodeUpdate(n_layers, 2*n_hidden, n_classes, dropout))

  def forward(self, nf):

    h = nf.layers[0].data['preprocess']
    if self.dropout:
        h = self.dropout(h)
    h = self.linear(h)

    skip_start = (0 == self.n_layers-1)
    if skip_start:
      h = torch.cat((h, self.activation(h)), dim=1)
    else:
      h = self.activation(h)

    for i, layer in enumerate(self.layers):
      new_history = h.clone().detach()
      history_str = 'h_{}'.format(i)
      history = nf.layers[i].data[history_str]
      h = h - history

      nf.layers[i].data['h'] = h
      nf.block_compute(i,
                       fn.copy_src(src='h', out='m'),
                       fn.mean(msg='m', out='h'),
                       layer)
      h = nf.layers[i+1].data.pop('activation')
      # update history
      if i < nf.num_layers-1:
          nf.layers[i].data[history_str] = new_history

    return h


class GCNInfer(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation):
        super(GCNInfer, self).__init__()
        self.n_layers = n_layers
        self.activation = activation
        # input layer
        self.linear = nn.Linear(in_feats, n_hidden)
        self.layers = nn.ModuleList()
        # hidden layers
        for i in range(1, n_layers):
            skip_start = (i == n_layers-1)
            self.layers.append(NodeUpdate(i, n_hidden, n_hidden, 0, activation, True, concat=skip_start))
        # output layer
        self.layers.append(NodeUpdate(n_layers, 2*n_hidden, n_classes, 0, None, True))

    def forward(self, nf):
        h = nf.layers[0].data['preprocess']
        h = self.linear(h)

        skip_start = (0 == self.n_layers-1)
        if skip_start:
            h = torch.cat((h, self.activation(h)), dim=1)
        else:
            h = self.activation(h)

        for i, layer in enumerate(self.layers):
            nf.layers[i].data['h'] = h
            nf.block_compute(i,
                             fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'),
                             layer)
            h = nf.layers[i+1].data.pop('activation')

        return h
