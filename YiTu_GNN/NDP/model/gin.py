import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

class NodeUpdate(nn.Module):
  def __init__(self, 
               in_feats,
               out_feats,
               dropout=0.0, 
               eps=0, 
               aggregator_type='mean'):
    super(NodeUpdate, self).__init__()
    self.mlp = nn.Sequential(
            nn.Linear(in_feats, out_feats),
            nn.BatchNorm1d(out_feats),
            nn.ReLU(),
            nn.Linear(out_feats, out_feats),
            nn.BatchNorm1d(out_feats),
            nn.ReLU()
        )
    self.eps = eps
    self.dropout = nn.Dropout(p=dropout)
    self.aggregator_type = aggregator_type

  def forward(self, node):
    h = node.data['h']
    h_neigh = node.data['neigh']
    out = self.mlp((1 + self.eps) * node.data['h'] + node.data['neigh'])
    h = self.dropout(out)
    return {'activation': h}  

class GINSampling(nn.Module):
    def __init__(self,
               in_feats,
               n_hidden,
               n_classes,
               n_layers, 
               eps, 
               activation=None,
               dropout=0.,
               aggregator_type='mean',
               preprocess=False):

        super(GINSampling, self).__init__()
        
        self.preprocess = preprocess
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.aggregator_type = aggregator_type
        self.layers = nn.ModuleList()

        self.layers.append(NodeUpdate(in_feats, n_hidden, dropout=dropout, eps=0, aggregator_type=aggregator_type))
        for i in range(1, n_layers):
            self.layers.append(NodeUpdate(n_hidden, n_hidden, dropout=dropout, eps=0, aggregator_type=aggregator_type))
        self.layers.append(NodeUpdate(n_hidden, n_classes, dropout=dropout, eps=0, aggregator_type=aggregator_type))
    
    def forward(self,nf):
        for lid in range(nf.num_layers):
            nf.layers[lid].data['h'] = nf.layers[lid].data.pop('features')

        for lid, layer in enumerate(self.layers):
            for i in range(lid, nf.num_layers - 1):
                h = nf.layers[i].data.pop('h')
                h = self.dropout(h)
                nf.layers[i].data['h'] = h
                nf.block_compute(i,
                           fn.copy_src(src='h', out='m'),
                           fn.mean('m', 'neigh'),
                           layer)
            for i in range(lid + 1, nf.num_layers):
                h = nf.layers[i].data.pop('activation')
                nf.layers[i].data['h'] = h
            
        h = nf.layers[nf.num_layers - 1].data.pop('h')
        return h