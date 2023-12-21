import torch
import math

from YiTu_GNN import YiTu_GNN_kernel


class GNNAFunction_GIN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight, inputInfo, eplison):
        # print("partSize: {}, dimWorker: {}, warpPerBlock: {}".format(inputInfo.partSize, inputInfo.dimWorker, inputInfo.warpPerBlock))
        X_prime, X_agg = YiTu_GNN_kernel.forward_gin(
            X,
            weight,
            inputInfo.row_pointers,
            inputInfo.column_index,
            eplison,
            inputInfo.partPtr,
            inputInfo.part2Node,
            inputInfo.partSize,
            inputInfo.dimWorker,
            inputInfo.warpPerBlock,
        )

        ctx.save_for_backward(X_agg, weight)
        ctx.inputInfo = inputInfo
        ctx.partSize, ctx.dimWorker, ctx.warpPerBlock, ctx.eplison = (
            inputInfo.partSize,
            inputInfo.dimWorker,
            inputInfo.warpPerBlock,
            eplison,
        )

        return X_prime

    @staticmethod
    def backward(ctx, d_output):

        X, weights = ctx.saved_tensors
        inputInfo = ctx.inputInfo

        d_input, d_weights = YiTu_GNN_kernel.backward_gin(
            d_output,
            X,
            weights,
            inputInfo.row_pointers,
            inputInfo.column_index,
            ctx.eplison,
            inputInfo.partPtr,
            inputInfo.part2Node,
            ctx.partSize,
            ctx.dimWorker,
            ctx.warpPerBlock,
        )

        return d_input, d_weights, None, None


class GINConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GINConv, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        self.eplison = 0.5
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, X, inputInfo):
        """
        @param:
        X:  the input tensor of the graph node embedding, shape: [n_nodes, n_dim].
        A:  the CSR node pointer of the graph, shape: [node, 1].
        edges: the CSR edge list of the graph, shape: [edge, 1].
        partitioin: for the graph with the part-based optimziation.
        """
        return GNNAFunction_GIN.apply(X, self.weights, inputInfo, self.eplison)
