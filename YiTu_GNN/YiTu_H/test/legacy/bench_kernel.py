import time
import torch
from torch import nn
from xgnn_h import mp, sparse, bundle, convert, block
from dgl.data import CoraGraphDataset, CoraFullDataset, AmazonCoBuyPhotoDataset


def bench_gspmm():
    density = 0.001
    n_src = 2048
    n_dst = 2048
    n_heads = 1
    n_features = 16

    #
    adj_adjacency = None
    while True:
        dense_raw = torch.rand(
            size=[n_dst, n_src]
        )
        adj_adjacency = torch.where(
            dense_raw < density, 1.0, 0.0
        )
        if adj_adjacency.max() != 0.0:
            break
    indptr, indices = convert.to_csr(
        adj_adjacency
    )[0]

    #
    feature = torch.randn(
        size=[n_src, 1, n_features],
        requires_grad=True
    ).to('cuda')
    gradient = torch.ones_like(feature)
    values = torch.randn(
        [indices.size(0), n_heads]
    ).to('cuda')
    values.requires_grad = True
    blk = block.Block(
        size=[n_dst, n_src],
        adj=[indptr, indices]
    ).to('cuda')
    torch.cuda.synchronize()

    #
    forward = []
    backward = []
    for _ in range(100):
        before = time.time()
        y = sparse.gspmm(
            block=blk,
            edge=None, x=feature
        )
        torch.cuda.synchronize()
        forward.append(time.time() - before)
        #
        before = time.time()
        y.backward(gradient=gradient)
        torch.cuda.synchronize()
        backward.append(time.time() - before)
    forward = sorted(forward)[5:-5]
    backward = sorted(backward)[5:-5]
    print('forward: {:.3f}, backward: {:.3f}, slowdown: {:.3f}'.format(
        sum(forward), sum(backward), sum(backward) / sum(forward)
    ))


def bench_bundle():
    n_nodes = 64
    n_features = 64
    x = torch.randn(
        [n_nodes, n_features]
    ).to('cuda')
    x.requires_grad = True

    #
    d_hidden = 16
    fc_1 = nn.Linear(
        n_features, d_hidden, bias=False
    ).to('cuda')
    fc_2 = nn.Linear(
        n_features, d_hidden, bias=False
    ).to('cuda')
    torch.cuda.synchronize()

    #
    origin = []
    bundled = []
    for _ in range(100):
        before = time.time()
        y_1, y_2 = fc_1(x), fc_2(x)
        torch.sum(y_1 + y_2).backward()
        torch.cuda.synchronize()
        origin.append(time.time() - before)
        #
        before = time.time()
        y_3, y_4 = bundle.GEMMBundleFunction.apply(
            x, fc_1.weight, fc_2.weight,
            fc_1.bias, fc_2.bias
        )
        torch.sum(y_3 + y_4).backward()
        torch.cuda.synchronize()
        bundled.append(time.time() - before)
    origin = sorted(origin)[5:-5]
    bundled = sorted(bundled)[5:-5]
    print('origin: {:.3f}, bundled: {:.3f}'.format(
        sum(origin), sum(bundled)
    ))


def bench_hfused():
    density = 0.001
    n_src = 512
    n_dst = 512
    n_heads = 8
    n_features = 16

    #
    adj_adjacency = None
    while True:
        dense_raw = torch.rand(
            size=[n_dst, n_src]
        )
        adj_adjacency = torch.where(
            dense_raw < density, 1.0, 0.0
        )
        if adj_adjacency.max() != 0.0:
            break
    indptr, indices = convert.to_csr(
        adj_adjacency
    )[0]

    #
    a = 0


def main():
    bench_hfused()


if __name__ == "__main__":
    main()
