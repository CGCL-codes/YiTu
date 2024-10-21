import torch
import graph_ext
from torch import autograd


class GEMMBundleFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, w_1, w_2, b_1, b_2):
        y_1, y_2 = graph_ext.b2gemm(
            x, w_1, w_2
        )
        y_1.requires_grad = True
        y_2.requires_grad = True
        ctx.save_for_backward(x, w_1, w_2)
        return y_1, y_2

    @staticmethod
    def backward(ctx, grad_1, grad_2):
        x, w_1, w_2 = ctx.saved_tensors
        grad_1 = grad_1.contiguous()
        grad_2 = grad_2.contiguous()
        grad = graph_ext.b2gemm_backward(
            x, w_1, w_2, grad_1, grad_2
        )
        dx, dw_1, dw_2 = grad
        return dx, dw_1, dw_2, None, None
