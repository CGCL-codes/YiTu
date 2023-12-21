import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


class StageModule(nn.Module):
    r"""分阶段执行模块: 将模型的forward拆分为模型并行和数据并行两个阶段, 其中网络第一层执行模型并行, 接下来的网络执行数据并行.
    为了方便执行需要手动将网络拆分为model_stage和data_stage两个模块. data_stage的实现和普通模型的实现并无差别, 只需要注意model_stage
    的实现即可:
        1. 如何切分模型: 首先已经将特征X按照维度切分成n份, 表示成[x_1, x_2, ..., x_n], 然后我们需要对model_stage这个模型进行切分, 也就是将参数按照行切分成n份[w_1, w_2, ..., w_n]^T, 这样节点 i 上只需完成`x_i w_i`的矩阵运算. 其中参数 w_i的行数等于特征 x_i 的维度, 所以我们只需要正确输入特征 x_i的维度就能够正确切分模型.
        2. model_stage中不能执行激活函数: model_stage之后得到的是部分结果, 需要reduce才能得到`x_1 w_1 + ... + x_i w_i`这个完整结果, 所以需要在数据并行前执行激活函数, 而不能在model_stage中执行.
    """

    def __init__(
        self,
        model_stage,
        data_stage,
        device,
        activation,
        dropout,
    ):
        super().__init__()
        # 模型并行
        self.model_stage = model_stage.to(device)
        # 数据并行
        self.data_stage = data_stage.to(device)
        self.data_stage = DDP(
            self.data_stage, device_ids=[device], output_device=device
        )
        self.activation = activation
        self.dropout = dropout

    def model_forward(self, blocks, feats):
        return self.model_stage(blocks, feats)

    def data_forward(self, blocks, feats):
        """获取模型并行阶段的输出后计算数据并行"""
        # 对reduce后的结果执行非线性激活函数
        feats = self.activation(feats)
        feats = F.dropout(feats, self.dropout)
        x = self.data_stage(blocks, feats)
        return x
