import logging

import torch.nn as nn

from YiTu_GNN.nn import SAGEConv
import YiTu_GNN.distributed as dist


class ModelParallelStage(nn.Module):
    """模型并行阶段
    注意:
        1. in_feats的大小是切分后的特征维度
        2. 模型并行阶段不能执行激活函数
    Args:
        reduce: 聚合方法, mean, max, min, sum
    """

    def __init__(self, in_feats, n_hidden, n_workers, reduce="mean"):
        super().__init__()
        self.n_workers = n_workers
        self.model_layer = SAGEConv(in_feats, n_hidden, reduce)

    def forward(self, blocks, feats):
        output = []
        # 遍历每个机器上的第一层block
        for i in range(self.n_workers):
            assert (
                blocks[i].srcnodes().shape[0] == feats[i].shape[0]
            ), "number of nodes {} is not equal to the number of features {}!".format(
                blocks[i].srcnodes().shape[0], feats[i].shape[0]
            )
            x = self.model_layer(blocks[i], feats[i])
            output.append(x)
        return output


class DataParallelStage(nn.Module):
    """数据并行阶段"""

    def __init__(
        self, n_hidden, n_classes, n_layers, activation, dropout, reduce="mean"
    ):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        self.n_classes = n_classes
        for _ in range(1, n_layers):
            self.layers.append(SAGEConv(n_hidden, n_hidden, reduce))
        self.layers.append(SAGEConv(n_hidden, n_classes, reduce))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        # 数据并行
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            assert (
                block.srcnodes().shape[0] == x.shape[0]
            ), "number of nodes {} is not equal to the number of features {}!".format(
                block.srcnodes().shape[0], x.shape[0]
            )
            x = layer(block, x)
            if l != self.n_layers - 1:
                x = self.activation(x)
                x = self.dropout(x)
        return x


class DistSAGE(dist.StageModule):
    def __init__(
        self,
        in_feats,
        n_hidden,
        n_classes,
        n_layers,
        n_workers,
        rank,
        device,
        activation,
        dropout,
        reduce="mean",
    ):
        logging.debug(
            "rank: {}, in_feats: {}, n_hidden: {}, n_classes: {}".format(
                rank, in_feats, n_hidden, n_classes
            )
        )
        # 注意：有且仅能有两个module：model module和data module
        # 模型并行
        model_stage = ModelParallelStage(in_feats, n_hidden, n_workers, reduce).to(
            device
        )
        # 数据并行
        data_stage = DataParallelStage(
            n_hidden, n_classes, n_layers - 1, activation, dropout, reduce=reduce
        ).to(device)
        super().__init__(model_stage, data_stage, device, activation, dropout)
