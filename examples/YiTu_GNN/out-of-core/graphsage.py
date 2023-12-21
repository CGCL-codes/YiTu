import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter


class SAGEConv(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        aggr: str = "mean",
        normalize: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.normalize = normalize
        self.aggr = aggr
        self.linear = nn.Linear(in_channel, out_channel, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x, edge_index) -> Tensor:
        out = self.linear(x)
        # aggr
        src, dst = edge_index[0], edge_index[1]
        src_feat = out[src]

        out = scatter(src_feat, dst, dim=0, reduce=self.aggr)
        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)
        return out


class SAGE(nn.Module):
    def __init__(
        self, in_channel, hidden_channel, num_classes, num_layers, dropout=0.5
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channel, hidden_channel))
        for _ in range(1, num_layers - 1):
            self.convs.append(SAGEConv(hidden_channel, hidden_channel))
        self.convs.append(SAGEConv(hidden_channel, num_classes))

    def forward(self, x, blocks):
        for i, edge_index in enumerate(blocks):
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
