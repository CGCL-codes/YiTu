import torch
import random
from torch import nn
from xgnn_h import block, sparse, bundle, convert
from tqdm import tqdm

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def rand_adjacency(n_rows, n_cols, density=0.02):
    adj_adjacency = None
    while True:
        dense_raw = torch.rand(
            size=[n_rows, n_cols]
        )
        adj_adjacency = torch.where(
            dense_raw < density, 1.0, 0.0
        )
        if adj_adjacency.max() != 0.0:
            break
    return adj_adjacency


def check_gspmm():
    n_src = random.randint(1, 512)
    n_dst = random.randint(1, 512)
    n_heads = random.randint(1, 32)
    n_features = random.randint(1, 256)

    #
    adj_adjacency = rand_adjacency(
        n_rows=n_dst, n_cols=n_src
    )
    indptr, indices = convert.to_csr(
        adj_adjacency
    )[0]

    #
    x = torch.randn(
        [n_src, n_heads, n_features]
    ).to('cuda')
    values = torch.randn(
        [indices.size(0), n_heads]
    ).to('cuda')
    values.requires_grad = True
    linear = nn.Linear(
        n_features, n_features
    ).to('cuda')
    blk = block.Block(
        size=[n_dst, n_src],
        adj=[indptr, indices]
    ).to('cuda')
    attn_adjacency = convert.to_dense_mha(
        n_dst, n_src, blk.adj_sparse, values
    ).to('cuda')
    attn_adjacency.requires_grad = True
    adj_adjacency = adj_adjacency.to('cuda')

    #
    y_1 = torch.bmm(
        attn_adjacency,
        linear(x).transpose(0, 1)
    ).transpose(0, 1)
    y_1.sum().backward()
    grad_1 = linear.weight.grad.clone()
    grad_2 = attn_adjacency.grad.clone()
    grad_2 = torch.multiply(
        adj_adjacency.unsqueeze(0), grad_2
    )

    #
    linear.zero_grad()
    y_2 = sparse.gspmm(
        blk, values, linear(x)
    )
    y_2.sum().backward()
    grad_3 = linear.weight.grad.clone()
    grad_4 = convert.to_dense_mha(
        n_dst, n_src, blk.adj_sparse,
        values.grad.clone()
    ).to('cuda')

    #
    assert torch.allclose(
        y_1, y_2, atol=1e-3
    )
    assert torch.allclose(
        grad_1, grad_3, atol=1e-3
    )
    assert torch.allclose(
        grad_2, grad_4, atol=1e-3
    )
    print('[PASS] check_gspmm')


def check_gsddmm():
    n_src = random.randint(1, 512)
    n_dst = random.randint(1, 512)
    n_heads = random.randint(1, 16)
    n_features = random.randint(1, 256)

    #
    adj_adjacency = rand_adjacency(
        n_rows=n_dst, n_cols=n_src
    )
    indptr, indices = convert.to_csr(
        adj_adjacency
    )[0]

    #
    x_dst = torch.randn(
        [n_dst, n_heads, n_features]
    ).to('cuda')
    x_src = torch.randn(
        [n_src, n_heads, n_features]
    ).to('cuda')
    linear_q = nn.Linear(
        n_features, 1
    ).to('cuda')
    linear_k = nn.Linear(
        n_features, 1
    ).to('cuda')
    blk = block.Block(
        size=[n_dst, n_src],
        adj=[indptr, indices]
    ).to('cuda')
    adj_adjacency = adj_adjacency.to('cuda')

    #
    q = linear_q(x_dst).squeeze(-1)
    k = linear_k(x_src).squeeze(-1)
    coeff_r = k.repeat([n_dst, 1])
    coeff_l = q.repeat_interleave(n_src, dim=0)
    coeff_e = (coeff_l + coeff_r).view(
        [n_dst, n_src, -1]
    )
    coeff_e = nn.LeakyReLU(
        negative_slope=0.2
    )(coeff_e)
    negative = -1e38 * torch.ones_like(coeff_e)
    coeff_e = torch.where(
        adj_adjacency.unsqueeze(-1) > 0.0,
        coeff_e, negative
    )
    coeff_e = coeff_e.transpose(0, 2)
    coeff_e = coeff_e.transpose(1, 2)
    attn_1 = torch.softmax(coeff_e, dim=-1)
    attn_1 = torch.multiply(
        adj_adjacency.unsqueeze(0), attn_1
    )
    y_1 = torch.bmm(
        attn_1, x_src.transpose(0, 1)
    ).transpose(0, 1)
    y_1.sum().backward()
    grad_q_1 = linear_q.weight.grad.clone()
    grad_k_1 = linear_k.weight.grad.clone()

    #
    linear_q.zero_grad()
    linear_k.zero_grad()
    q = linear_q(x_dst).squeeze(-1)
    k = linear_k(x_src).squeeze(-1)
    attn_2 = sparse.fused_gsddmm(
        blk, q, k
    )
    y_2 = sparse.gspmm(
        blk, attn_2, x_src
    )
    y_2.sum().backward()
    grad_q_2 = linear_q.weight.grad.clone()
    grad_k_2 = linear_k.weight.grad.clone()

    #
    assert torch.allclose(
        y_1, y_2, atol=1e-3
    )
    assert torch.allclose(
        grad_q_1, grad_q_2, atol=1e-3
    )
    assert torch.allclose(
        grad_k_1, grad_k_2, atol=1e-3
    )
    print('[PASS] check_gsddmm')


def check_bundle():
    n_nodes = random.randint(8, 1024)
    n_features = random.randint(8, 256)
    x = torch.randn(
        [n_nodes, n_features]
    ).to('cuda')
    x.requires_grad = True

    #
    d_hidden = random.randint(1, 32)
    fc_1 = nn.Linear(
        n_features, d_hidden, bias=False
    ).to('cuda')
    fc_2 = nn.Linear(
        n_features, d_hidden, bias=False
    ).to('cuda')
    torch.cuda.synchronize()

    #
    y_1, y_2 = fc_1(x), fc_2(x)
    torch.sum(y_1 + y_2).backward()
    torch.cuda.synchronize()
    grad_x_1 = x.grad.clone()
    grad_fc_1 = fc_1.weight.grad.clone()
    grad_fc_2 = fc_2.weight.grad.clone()

    #
    x.grad.zero_()
    fc_1.zero_grad()
    fc_2.zero_grad()
    y_3, y_4 = bundle.GEMMBundleFunction.apply(
        x, fc_1.weight, fc_2.weight,
        fc_1.bias, fc_2.bias
    )
    torch.sum(y_3 + y_4).backward()
    torch.cuda.synchronize()
    grad_x_2 = x.grad.clone()
    grad_fc_3 = fc_1.weight.grad.clone()
    grad_fc_4 = fc_2.weight.grad.clone()

    #
    assert torch.allclose(y_1, y_3, atol=1e-3)
    assert torch.allclose(y_2, y_4, atol=1e-3)
    assert torch.allclose(grad_fc_1, grad_fc_3, atol=1e-3)
    assert torch.allclose(grad_fc_2, grad_fc_4, atol=1e-3)
    assert torch.allclose(grad_x_1, grad_x_2, atol=1e-3)
    print('[PASS] check_bundle')


def check_stitch():
    n_dst = random.randint(1, 64)
    n_heads = random.randint(1, 4)
    n_stitches = random.randint(2, 8)
    n_features = random.randint(1, 64)

    #
    src_list = []
    dst_list = []
    block_list = []
    adjacency_list = []
    while True:
        density = 0.02
        n_src = random.randint(1, 256)
        #
        dense_raw = torch.rand(
            size=[n_dst, n_src]
        )
        adj_adjacency = torch.where(
            dense_raw < density, 1.0, 0.0
        )
        if adj_adjacency.max() == 0.0:
            continue
        indptr, indices = convert.to_csr(
            adj_adjacency
        )[0]
        assert n_dst == len(indptr) - 1
        src_list.append(
            torch.randn(
                [n_src, n_heads, n_features]
            ).to('cuda')
        )
        dst_list.append(
            torch.randn(
                [n_dst, n_heads, n_features]
            ).to('cuda')
        )
        block_list.append(
            block.Block(
                size=[n_dst, n_src],
                adj=[indptr, indices]
            ).to('cuda')
        )
        adjacency_list.append(
            adj_adjacency.to('cuda')
        )
        if len(block_list) == n_stitches:
            break

    #
    new_src = torch.cat(
        src_list, dim=0
    ).to('cuda')
    new_dst = torch.stack(
        dst_list, dim=0
    ).to('cuda')
    new_block = block.stitch_csr(
        block_list
    ).to('cuda')
    new_adjacency = torch.cat(
        adjacency_list, dim=-1
    ).to('cuda')
    assert new_block.size[1] == new_src.size(0)
    assert new_block.size[0] == new_dst.size(1)
    assert new_block.size[0] == new_adjacency.size(0)
    assert new_block.size[1] == new_adjacency.size(1)
    linear_q = nn.Linear(n_features, 1).to('cuda')
    linear_k = nn.Linear(n_features, 1).to('cuda')
    linear_v = nn.Linear(n_features, n_features).to('cuda')

    #
    y_1 = torch.zeros([
        n_dst, n_heads, n_features
    ], device='cuda')
    for blk, src, dst in \
            zip(block_list, src_list, dst_list):
        q = torch.squeeze(
            linear_q(dst), dim=-1
        )
        k = torch.squeeze(
            linear_k(src), dim=-1
        )
        v = linear_v(src)
        e_1 = sparse.fused_gsddmm(
            blk, q, k
        )
        y_1 += sparse.gspmm(
            blk, e_1, v

        )
    torch.sum(y_1).backward()
    grad_q_1 = linear_q.weight.grad.clone()
    grad_k_1 = linear_k.weight.grad.clone()
    grad_v_1 = linear_v.weight.grad.clone()

    #
    linear_q.zero_grad()
    linear_k.zero_grad()
    linear_v.zero_grad()
    q = torch.squeeze(
        linear_q(new_dst), dim=-1
    )
    k = torch.squeeze(
        linear_k(new_src), dim=-1
    )
    v = linear_v(new_src)
    e_2 = sparse.fused_gsddmm(
        new_block, q, k
    )
    y_2 = sparse.gspmm(
        new_block, e_2, v

    )
    torch.sum(y_2).backward()
    grad_q_2 = linear_q.weight.grad.clone()
    grad_k_2 = linear_k.weight.grad.clone()
    grad_v_2 = linear_v.weight.grad.clone()

    #
    assert torch.allclose(y_1, y_2, atol=1e-3)
    assert torch.allclose(grad_q_1, grad_q_2, atol=1e-3)
    assert torch.allclose(grad_k_1, grad_k_2, atol=1e-3)
    assert torch.allclose(grad_v_1, grad_v_2, atol=1e-3)
    print('[PASS] check_stitch')


def check_hfuse():
    """
    n_src = random.randint(1, 512)
    n_dst = random.randint(1, 512)
    n_heads = random.randint(1, 8)
    n_features = random.randint(1, 64)
    """
    n_dst = 4096
    n_src = 2048
    n_heads = 8
    n_features = 4

    #
    spmm_adjacency = rand_adjacency(
        n_rows=n_dst, n_cols=n_src, density=0.02
    )
    spmm_indptr, spmm_indices = convert.to_csr(
        dense=spmm_adjacency
    )[0]
    spmm_values = torch.randn(
        [spmm_indices.size(0), n_heads]
    ).to('cuda')
    sddmm_adjacency = rand_adjacency(
        n_rows=n_dst, n_cols=n_src
    )
    sddmm_indptr, sddmm_indices = convert.to_csr(
        dense=sddmm_adjacency
    )[0]

    #
    x_dst = torch.randn(
        [n_dst, n_heads, n_features]
    ).to('cuda')
    x_src = torch.randn(
        [n_src, n_heads, n_features]
    ).to('cuda')
    linear_q = nn.Linear(
        n_features, 1
    ).to('cuda')
    linear_k = nn.Linear(
        n_features, 1
    ).to('cuda')
    spmm_blk = block.Block(
        size=[n_dst, n_src],
        adj=[spmm_indptr, spmm_indices]
    ).to('cuda')
    sddmm_blk = block.Block(
        size=[n_dst, n_src],
        adj=[sddmm_indptr, sddmm_indices]
    ).to('cuda')

    #
    q = torch.squeeze(
        linear_q(x_dst), dim=-1
    )
    k = torch.squeeze(
        linear_k(x_src), dim=-1
    )
    y_1 = sparse.gspmm(
        spmm_blk, spmm_values, x_src
    )
    y_2 = sparse.fused_gsddmm(
        sddmm_blk, q, k
    )
    y_3, y_4 = sparse.hfused_spddmm(
        spmm_blk, spmm_values,
        x_src, sddmm_blk, q, k
    )

    #
    assert torch.allclose(y_1, y_3, atol=1e-3)
    assert torch.allclose(y_2, y_4, atol=1e-3)


def test():
    for _ in tqdm(range(16)):
        check_gspmm()
        check_gsddmm()
        check_bundle()
        check_stitch()
        check_hfuse()


if __name__ == "__main__":
    test()
