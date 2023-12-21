import torch
import torch.nn.functional as F

from YiTu_GNN.nn import GINConv


class GIN(torch.nn.Module):
    def __init__(self, in_features, num_hidden, num_classes):
        super(GIN, self).__init__()
        self.conv1 = GINConv(in_features, num_hidden)
        self.conv2 = GINConv(num_hidden, num_hidden)
        self.conv3 = GINConv(num_hidden, num_hidden)
        self.conv4 = GINConv(num_hidden, num_hidden)
        self.conv5 = GINConv(num_hidden, num_classes)

    def forward(self, x, input_info):
        x = F.relu(self.conv1(x, input_info.set_input()))
        x = F.relu(self.conv2(x, input_info.set_hidden()))
        x = F.relu(self.conv3(x, input_info.set_hidden()))
        x = F.relu(self.conv4(x, input_info.set_hidden()))
        x = self.conv5(x, input_info.set_hidden())
        return F.log_softmax(x, dim=1)
