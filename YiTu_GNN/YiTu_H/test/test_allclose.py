import torch
import xgnn_h
from torch import nn
from xgnn_h import mp
from dgl import nn as dglnn
from dgl.data import CoraGraphDataset
from common.model import GCNLayer
from tqdm import tqdm


def check_gcn():
    dataset = CoraGraphDataset(
        verbose=False
    )
    dglgraph = dataset[0].to('cuda')
    graph = mp.from_dglgraph(dglgraph)

    # inputs
    n_features, d_hidden = 32, 16
    n_nodes = dglgraph.num_nodes()
    feature = torch.randn(
        size=[n_nodes, n_features]
    ).to('cuda')
    dglmodel = dglnn.GraphConv(
        n_features, d_hidden,
        norm='right', bias=False
    ).to('cuda')
    model = GCNLayer(
        n_features, d_hidden
    ).to('cuda')
    kwargs = dict({
        'graph': graph, 'x': feature,
        'norm': graph.right_norm()
    })

    # params
    with torch.no_grad():
        w = model.fc.weight
        model.fc.bias.zero_()
        dglmodel.weight.copy_(w.T)

    # result
    mod2ir = xgnn_h.Module2IR()
    optimizer = xgnn_h.Optimizer()
    executor = xgnn_h.Executor()
    dataflow = mod2ir.transform(
        model, kwargs=kwargs
    )
    dataflow = optimizer.lower(
        dataflow, kwargs=kwargs
    )
    executor.train()
    y_1 = executor.run(
        dataflow, kwargs=kwargs
    )
    torch.sum(y_1).backward()
    grad_1 = model.fc.weight.grad
    y_2 = dglmodel(
        dglgraph, feature
    )
    torch.sum(y_2).backward()
    grad_2 = dglmodel.weight.grad

    #
    assert torch.allclose(
        y_1, y_2, atol=1e-3
    )
    assert torch.allclose(
        grad_1, grad_2.T, atol=1e-3
    )


def test():
    for _ in tqdm(range(16)):
        check_gcn()


if __name__ == "__main__":
    test()
