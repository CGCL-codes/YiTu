import torch
import torch.nn as nn
import dgl
import dgl.function as fn

class NodeUpdate(nn.Module):
  def __init__(self, in_feats, out_feats, activation=None, test=False, concat=False):
    super(NodeUpdate, self).__init__()
    self.linear = nn.Linear(in_feats, out_feats)
    self.activation = activation
    self.concat = concat
    self.test = test

  def forward(self, node):
    h = node.data['h']
    if self.test:
        h = h * node.data['norm']
    h = self.linear(h)
    # skip connection
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
               dropout,
               preprocess=False):
    super(GCNSampling, self).__init__()

    self.preprocess = preprocess
    
    self.n_layers = n_layers
    if dropout != 0:
      self.dropout = nn.Dropout(p=dropout)
    else:
      self.dropout = None
    self.layers = nn.ModuleList()
    # input layer
    if self.preprocess:
      self.linear = nn.Linear(in_feats, n_hidden)
      self.activation = activation
    else:
      skip_start = (0 == n_layers-1)
      self.layers.append(NodeUpdate(in_feats, n_hidden, activation, concat=skip_start))
    # hidden layers
    for i in range(1, n_layers):
      skip_start = (i == n_layers-1)
      self.layers.append(NodeUpdate(n_hidden, n_hidden, activation, concat=skip_start))
    # output layer
    self.layers.append(NodeUpdate(2*n_hidden, n_classes))

  def forward(self, nf):
    if self.preprocess:
      return self.preprocess_forward(nf)
    
    nf.layers[0].data['activation'] = nf.layers[0].data['features']

    for i, layer in enumerate(self.layers):
      h = nf.layers[i].data.pop('activation')
      if self.dropout:
          h = self.dropout(h)
      nf.layers[i].data['h'] = h
      nf.block_compute(i,
                       fn.copy_src(src='h', out='m'),
                       fn.mean(msg='m', out='h'),
                       layer)

    h = nf.layers[-1].data.pop('activation')
    return h


  def preprocess_forward(self, nf):
    h = nf.layers[0].data['features']
    if self.dropout:
      h = self.dropout(h)
    h = self.linear(h)

    skip_start = (0 == self.n_layers - 1)
    if skip_start:
      h = torch.cat((h, self.activation(h)), dim=1)
    else:
      h = self.activation(h)
    
    for i, layer in enumerate(self.layers):
      nf.layers[i].data['h'] = h
      nf.block_compute(i,
                       fn.copy_src(src='h', out='m'),
                       fn.mean(msg='m', out='h'),
                       layer)
      h = nf.layers[i+1].data.pop('activation')

    return h


class GCNInfer(nn.Module):
  def __init__(self,
               in_feats,
               n_hidden,
               n_classes,
               n_layers,
               activation,
               preprocess=False):
    super(GCNInfer, self).__init__()
    self.preprocess = preprocess
    self.n_layers = n_layers
    self.layers = nn.ModuleList()

    # input layer
    if self.preprocess:
      self.linear = nn.Linear(in_feats, n_hidden)
      self.activation = activation
    else:
      skip_start = (0 == n_layers-1)
      self.layers.append(NodeUpdate(in_feats, n_hidden, activation, test=True, concat=skip_start))
    # hidden layers
    for i in range(1, n_layers):
      skip_start = (i == n_layers-1)
      self.layers.append(NodeUpdate(n_hidden, n_hidden, activation, test=True, concat=skip_start))
    # output layer
    self.layers.append(NodeUpdate(2*n_hidden, n_classes, test=True))

  def forward(self, nf):
    if self.preprocess:
      return self.preprocess_forward(nf)

    nf.layers[0].data['activation'] = nf.layers[0].data['features']

    for i, layer in enumerate(self.layers):
      h = nf.layers[i].data.pop('activation')
      nf.layers[i].data['h'] = h
      nf.block_compute(i,
                       fn.copy_src(src='h', out='m'),
                       fn.sum(msg='m', out='h'),
                       layer)

    h = nf.layers[-1].data.pop('activation')
    return h
  
  def preprocess_forward(self, nf):
    h = nf.layers[0].data['features']
    h = self.linear(h)

    skip_start = (0 == self.n_layers - 1)
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