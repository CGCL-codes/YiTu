import torch
import torch.nn.functional as F

from YiTu_GNN.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, in_features, num_hidden, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_features, num_hidden)
        self.conv2 = GCNConv(num_hidden, num_classes)

    def forward(self, x, input_info):
        x = F.relu(self.conv1(x, input_info.set_input()))
        x = self.conv2(x, input_info.set_hidden())
        return F.log_softmax(x, dim=1)
