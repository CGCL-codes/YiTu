import torch
import xgnn_h
from xgnn_h import mp
from torch import nn, optim
from dgl.data import CoraGraphDataset
from common.model import GATModel
from tqdm import tqdm


def check_homo():
    dataset = CoraGraphDataset(
        verbose=True
    )
    dglgraph = dataset[0].to('cuda')
    graph = mp.from_dglgraph(dglgraph)
    label = dglgraph.ndata.pop(
        'label'
    ).type(torch.LongTensor).to('cuda')
    feature = dglgraph.ndata.pop(
        'feat'
    ).type(torch.FloatTensor).to('cuda')
    test_mask = dglgraph.ndata.pop(
        'test_mask'
    ).type(torch.BoolTensor).to('cuda')
    train_mask = dglgraph.ndata.pop(
        'train_mask'
    ).type(torch.BoolTensor).to('cuda')
    n_labels = dataset.num_classes
    n_features = feature.size(1)

    #
    n_heads = 8
    d_hidden = 32
    model = GATModel(
        in_features=n_features,
        gnn_features=d_hidden,
        out_features=n_labels,
        n_heads=n_heads,
    ).to('cuda')
    kwargs = dict({
        'graph': graph,
        'x': feature
    })

    #
    print('===== mod2ir =====')
    mod2ir = xgnn_h.Module2IR()
    dataflow = mod2ir.transform(
        model, kwargs=kwargs
    )
    xgnn_h.Printer().dump(dataflow)

    #
    print('===== optimizer =====')
    optimizer = xgnn_h.Optimizer()
    dataflow = optimizer.lower(
        dataflow, kwargs=kwargs
    )
    xgnn_h.Printer().dump(dataflow)

    #
    print('===== executor =====')
    executor = xgnn_h.Executor()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=2e-4, weight_decay=1e-5
    )

    #
    def train():
        executor.train()
        logits = executor.run(
            dataflow, kwargs=kwargs
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
                dataflow, kwargs={
                    'graph': graph,
                    'x': feature
                }
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
        for _ in range(50):
            loss_val = train()
        #
        accuracy = evaluate()
        print('[epoch={}] loss: {:.04f}, accuracy: {:.02f}'.format(
            epoch, loss_val, accuracy
        ))
        if accuracy > 0.72:
            return True

    return False


def test():
    for _ in tqdm(range(10)):
        if check_homo():
            break
    else:
        raise RuntimeError("not convergent")


if __name__ == "__main__":
    test()
