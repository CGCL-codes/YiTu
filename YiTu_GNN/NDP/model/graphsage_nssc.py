import torch
import torch.nn as nn
import dgl
import dgl.function as fn

class NodeUpdate(nn.Module):
  def __init__(self, 
               in_feats,
               out_feats,
               activation=None,
               concat=False):
    super(NodeUpdate, self).__init__()
    self.fc_neigh = nn.Linear(in_feats, out_feats)
    self.fc_self = nn.Linear(in_feats, out_feats)
    self.activation = activation
    self.concat = concat
    gain = nn.init.calculate_gain('relu')
    nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
    nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)

  def forward(self, node):
    for w in node.data:
      print(w)
    h = node.data['h']
    h_neigh = node.data['neigh']
    h = self.fc_self(h) + self.fc_neigh(h_neigh)
    # skip connection
    if self.concat:
        h = torch.cat((h, self.activation(h)), dim=1)
    elif self.activation:
        h = self.activation(h)
    return {'activation': h}


class GraphSageSampling(nn.Module):
  def __init__(self,
               in_feats,
               n_hidden,
               n_classes,
               n_layers, 
               activation=None,
               dropout=0.,
               aggregator_type='pool',
               preprocess=False):
    super(GraphSageSampling, self).__init__()

    self.preprocess = preprocess
    self.n_layers = n_layers
    self.dropout = nn.Dropout(dropout)
    self.activation = activation
    self.aggregator_type = aggregator_type
    self.layers = nn.ModuleList()
    self.reducer = nn.ModuleList() # for lstm

    # inpute layers
    if self.preprocess: # NOTE: preprocess can only be enabled except lstm agg.
      self.fc_self = nn.Linear(in_feats, n_hidden)
      self.fc_neigh = nn.Linear(in_feats, n_hidden)
    else:
      skip_start = (0 == n_layers-1)
      if aggregator_type == 'lstm':
        self.reducer.append(nn.LSTM(in_feats, in_feats, batch_first=True))
      self.layers.append(NodeUpdate(in_feats, n_hidden, activation, concat=skip_start))
    # hidden layers
    for i in range(1, n_layers):
      skip_start = (i == n_layers-1)
      if aggregator_type == 'lstm':
        self.reducer.append(nn.LSTM(n_hidden, n_hidden, batch_first=True))
      self.layers.append(NodeUpdate(n_hidden, n_hidden, activation, concat=skip_start))
    # output layers
    if aggregator_type == 'lstm':
      self.reducer.append(nn.LSTM(2 * n_hidden, 2 * n_hidden, batch_first=True))
    self.layers.append(NodeUpdate(2 * n_hidden, n_classes))


  def forward(self, nf):
    if self.preprocess:
      for i in range(nf.num_layers):
        h = nf.layers[i].data.pop('features')
        neigh = nf.layers[i].data.pop('neigh')
        if self.dropout:
          h = self.dropout(h)
        h = self.fc_self(h) + self.fc_neigh(neigh)
        skip_start = (0 == self.n_layers - 1)
        if skip_start:
          h = torch.cat((h, self.activation(h)), dim=1)
        else:
          h = self.activation(h)
        nf.layers[i].data['h'] = h
    else:
      for lid in range(nf.num_layers):
        nf.layers[lid].data['h'] = nf.layers[lid].data.pop('features')

    for lid, layer in enumerate(self.layers):
      for i in range(lid, nf.num_layers - 1):
        h = nf.layers[i].data.pop('h')
        h = self.dropout(h)
        nf.layers[i].data['h'] = h
        if self.aggregator_type == 'mean':
          nf.block_compute(i,
                           fn.copy_src(src='h', out='m'),
                           fn.mean('m', 'neigh'),
                           layer)
        elif self.aggregator_type == 'gcn':
          nf.block_compute(i,
                           fn.copy_src(src='h', out='m'),
                           fn.sum('m', 'neigh'),
                           layer)
        elif self.aggregator_type == 'pool':
          nf.block_compute(i,
                           fn.copy_src(src='h', out='m'),
                           fn.max('m', 'neigh'),
                           layer)
        elif self.aggregator_type == 'lstm':
          reducer = self.reducer[i]
          def _reducer(self, nodes):
            m = nodes.mailbox['m'] # (B, L, D)
            batch_size = m.shape[0]
            h = (m.new_zeros((1, batch_size, self._in_feats)),
                 m.new_zeros((1, batch_size, self._in_feats)))
            _, (rst, _) = reducer(m, h)
            return {'neigh': rst.squeeze(0)}

          nf.block_compute(i,
                           fn.copy_src(src='h', out='m'),
                           _reducer,
                           layer)
        else:
          raise KeyError('Aggregator type {} not recognized.'.format(self.aggregator_type))
      # set up new feat
      for i in range(lid + 1, nf.num_layers):
        h = nf.layers[i].data.pop('activation')
        nf.layers[i].data['h'] = h

    h = nf.layers[nf.num_layers - 1].data.pop('h')
    return h



