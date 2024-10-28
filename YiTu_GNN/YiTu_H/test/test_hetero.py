import torch
import xgnn_h
from xgnn_h import mp
from torch import nn, optim
from dgl.data.rdf import MUTAGDataset
from common.model import RGATModel
from tqdm import tqdm


def check_hetero():
    dataset = MUTAGDataset(
        verbose=False
    )
    dglgraph = dataset[0].to('cuda')
    graph = mp.from_dglgraph(dglgraph)
    pred_cat = dataset.predict_category
    label = dglgraph.ndata.pop(
        'labels'
    )[pred_cat].type(
        torch.LongTensor
    ).to('cuda')
    n_labels = dataset.num_classes
    test_mask = dglgraph.ndata.pop(
        'test_mask'
    )[pred_cat].type(
        torch.BoolTensor
    ).to('cuda')
    train_mask = dglgraph.ndata.pop(
        'train_mask'
    )[pred_cat].type(
        torch.BoolTensor
    ).to('cuda')
    print('predict_category:', pred_cat)

    #
    n_heads = 2
    d_hidden = 16
    model = RGATModel(
        hgraph=graph,
        in_features=d_hidden,
        gnn_features=d_hidden,
        out_features=n_labels,
        n_heads=n_heads
    ).to('cuda')
    node_indices = {
        nty: torch.linspace(
            0, num - 1, num,
            dtype=torch.int64
        ).to('cuda')
        for nty, num in graph.nty2num.items()
    }
    kwargs = dict({
        'hgraph': graph,
        'xs': node_indices
    })

    #
    print('===== mod2ir =====')
    mod2ir = xgnn_h.Module2IR()
    dataflow = mod2ir.transform(
        model, kwargs=kwargs
    )[pred_cat]
    xgnn_h.Printer().dump(dataflow)

    #
    print('===== lower =====')
    optimizer = xgnn_h.Optimizer()
    dataflow = optimizer.lower(
        dataflow, kwargs=kwargs
    )
    xgnn_h.Printer().dump(dataflow)

    #
    print('===== stitch =====')
    stitcher = xgnn_h.Stitcher()
    dataflow = stitcher.transform(
        dataflow, kwargs=kwargs
    )
    xgnn_h.Printer().dump(dataflow)

    #
    print('===== executor =====')
    executor = xgnn_h.Executor()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3, weight_decay=1e-5
    )

    #
    def train():
        executor.train()
        logits = executor.run(
            dataflow,
            kwargs=kwargs
        )
        loss = loss_fn(
            logits[train_mask],
            target=label[train_mask]
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def evaluate():
        executor.eval()
        with torch.no_grad():
            logits = executor.run(
                dataflow,
                kwargs=kwargs
            )
            assert logits.dim() == 2
            indices = torch.argmax(
                logits[test_mask], dim=-1
            )
            correct = torch.sum(
                indices == label[test_mask]
            )
            return correct.item() / len(indices)

    #
    for epoch in range(20):
        loss_val = None
        for _ in range(10):
            loss_val = train()
        #
        accuracy = evaluate()
        print('[epoch={}] loss: {:.04f}, accuracy: {:.02f}'.format(
            epoch, loss_val, accuracy
        ))
        if accuracy > 0.6:
            return True
    else:
        return False


def test():
    for _ in tqdm(range(10)):
        if check_hetero():
            break
    else:
        raise RuntimeError("not convergent")


if __name__ == "__main__":
    test()
