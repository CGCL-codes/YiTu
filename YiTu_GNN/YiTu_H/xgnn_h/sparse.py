import torch
import graph_ext
from xgnn_h import block
from torch import autograd


class GSPMMFunction(autograd.Function):
    @staticmethod
    def forward(ctx, adj_sparse, adj_values, x):
        indptr = adj_sparse[0]
        indices = adj_sparse[1]
        #
        y = graph_ext.spmm_forward(
            adj_values, indptr, indices,
            x
        )
        ctx.adj_sparse = adj_sparse
        ctx.save_for_backward(adj_values, x)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        indptr = ctx.adj_sparse[0]
        indices = ctx.adj_sparse[1]
        values, x = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        assert len(ctx.needs_input_grad) == 3
        assert ctx.needs_input_grad[0] is False
        #
        grad_a, grad_x = graph_ext.spmm_backward(
            values, indptr, indices,
            x, grad_out
        )
        #
        return None, grad_a, grad_x


def gspmm(block: block.Block,
          edge: torch.Tensor,
          x: torch.Tensor):
    assert x.dim() == 3
    indices = block.adj_sparse[1]
    if edge is None:
        edge = torch.ones(
            size=[indices.size(0),
                  x.size(1)],
            device=x.device
        )
    return GSPMMFunction.apply(
        block.adj_sparse, edge, x
    )


class GSDDMMFunction(autograd.Function):
    @staticmethod
    def forward(ctx, adj_sparse, query, key):
        indptr, indices = adj_sparse
        attn_values = graph_ext.sddmm_forward(
            indptr, indices, query, key
        )
        #
        ctx.adj_sparse = adj_sparse
        ctx.save_for_backward(query, key, attn_values)
        return attn_values

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = grad_out.contiguous()
        indptr, indices = ctx.adj_sparse
        query, key, attn_values = ctx.saved_tensors
        assert len(ctx.needs_input_grad) == 3
        assert ctx.needs_input_grad[0] is False
        assert ctx.needs_input_grad[1] is True
        assert ctx.needs_input_grad[2] is True
        #
        grad_query, grad_key = graph_ext.sddmm_backward(
            indptr, indices, query, key,
            attn_values, grad_out
        )
        #
        return None, grad_query, grad_key


def fused_gsddmm(block: block.Block,
                 query: torch.Tensor,
                 key: torch.Tensor):
    if query.dim() == 2:
        query = query.unsqueeze(0)
    return GSDDMMFunction.apply(
        block.adj_sparse, query, key
    )


class HFUSEDFunction(autograd.Function):
    @staticmethod
    def forward(ctx, spmm_sparse, spmm_values, spmm_x,
                sddmm_sparse, sddmm_query, sddmm_key):
        spmm_out, attn_values = graph_ext.hfused_forward(
            spmm_values, spmm_sparse[0], spmm_sparse[1], spmm_x,
            sddmm_sparse[0], sddmm_sparse[1], sddmm_query, sddmm_key
        )
        #
        ctx.spmm_sparse = spmm_sparse
        ctx.sddmm_sparse = sddmm_sparse
        ctx.save_for_backward(sddmm_query, sddmm_key)
        return spmm_out, attn_values

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError


def hfused_spddmm(spmm_block: block.Block, spmm_values, spmm_x,
                  sddmm_block: block.Block, sddmm_query, sddmm_key):
    if sddmm_query.dim() == 2:
        sddmm_query = sddmm_query.unsqueeze(0)
    return HFUSEDFunction.apply(
        spmm_block.adj_sparse, spmm_values, spmm_x,
        sddmm_block.adj_sparse, sddmm_query, sddmm_key
    )
