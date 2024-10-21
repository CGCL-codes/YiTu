import dgl
import torch
import xgnn_h
from torch import nn, optim
from xgnn_h import block, sparse
from dgl.data.rdf import AIFBDataset


class REmbedding(nn.Module):
    def __init__(self,
                 dglgraph: dgl.DGLHeteroGraph,
                 embedding_dim: int):
        nn.Module.__init__(self)
        self.embeds = nn.ModuleDict({
            nty: nn.Embedding(
                dglgraph.num_nodes(nty),
                embedding_dim
            )
            for nty in dglgraph.ntypes
        })

    def forward(self):
        return {
            nty: self.embeds[nty].weight
            for nty in self.embeds.keys()
        }


class GATLayer(nn.Module):
    def __init__(self,
                 block: block.Block,
                 in_features: int,
                 out_features: int,
                 n_heads: int = 8):
        nn.Module.__init__(self)
        #
        self.block = block
        self.n_heads = n_heads
        self.n_features = out_features
        self.linear_q = nn.Linear(n_heads * out_features, n_heads)
        self.linear_k = nn.Linear(n_heads * out_features, n_heads)
        self.linear_v = nn.Linear(in_features, n_heads * out_features)

    def forward(self, x):
        # some other impl uses different
        # linear layers for h_src and h_dst
        h_src = self.linear_v(x[0])
        h_dst = self.linear_v(x[1])

        # feature transformation
        q = self.linear_q(h_dst)
        k = self.linear_k(h_src)
        h_src = h_src.view(size=[
            -1, self.n_heads,
            self.n_features
        ])

        #
        attn = sparse.fused_gsddmm(
            block=self.block,
            query=q, key=k
        )
        h = sparse.gspmm(
            block=self.block,
            edge=attn, x=h_src
        )
        return torch.mean(h, dim=1)


class HeteroGraphConv(nn.Module):
    def __init__(self,
                 dglgraph: dgl.DGLHeteroGraph,
                 in_features: int,
                 out_features: int):
        nn.Module.__init__(self)
        self.convs = nn.ModuleDict()
        self.rels = dglgraph.canonical_etypes
        for sty, ety, dty in self.rels:
            adj = dglgraph.adj(
                transpose=True,
                scipy_fmt='csr',
                etype=(sty, ety, dty)
            )
            graph = block.Block(
                size=[
                    dglgraph.num_nodes(dty),
                    dglgraph.num_nodes(sty)
                ],
                adj=[
                    torch.IntTensor(
                        adj.indptr
                    ).to(dglgraph.device),
                    torch.IntTensor(
                        adj.indices
                    ).to(dglgraph.device)
                ]
            )
            rel = '{}--{}--{}'.format(
                sty, ety, dty)
            self.convs[rel] = GATLayer(
                block=graph,
                in_features=in_features,
                out_features=out_features
            )

    def forward(self, xs: dict):
        counts = {}
        outputs = {}
        for sty, ety, dty in self.rels:
            # feature
            rel = '{}--{}--{}'.format(
                sty, ety, dty)
            res = self.convs[rel](
                (xs[sty], xs[dty])
            )
            # count by dst
            if dty not in counts:
                counts[dty] = 1
            else:
                counts[dty] += 1
            # concat same dst
            if dty not in outputs:
                outputs[dty] = res
            else:
                exist = outputs[dty]
                outputs[dty] = exist + res
        # mean aggegration
        for dty, res in outputs.items():
            outputs[dty] = res / counts[dty]
        return outputs


class RGATModel(nn.Module):
    def __init__(self,
                 dglgraph: dgl.DGLHeteroGraph,
                 in_features: int,
                 gnn_features: int,
                 out_features: int):
        nn.Module.__init__(self)
        #
        self.em = REmbedding(
            dglgraph=dglgraph,
            embedding_dim=in_features
        )
        self.i2h = HeteroGraphConv(
            dglgraph=dglgraph,
            in_features=in_features,
            out_features=gnn_features
        )
        self.h2o = HeteroGraphConv(
            dglgraph=dglgraph,
            in_features=gnn_features,
            out_features=out_features
        )
        self.activation = nn.Sequential(
            nn.Dropout(0.5),
            nn.ELU()
        )

    def forward(self):
        xs = self.em()
        hs = self.i2h(xs)
        hs = {
            k: self.activation(h)
            for k, h in hs.items()
        }
        hs = self.h2o(hs)
        return hs


def check_minimal():
    # dataset
    dataset = AIFBDataset(
        verbose=False
    )
    dglgraph = dataset[0].to('cuda')
    pred_cat = dataset.predict_category
    label = dglgraph.ndata.pop(
        'labels'
    )[pred_cat].type(
        torch.LongTensor
    ).to('cuda')
    n_labels = dataset.num_classes
    print('num_classes:', n_labels)
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

    # minimal model
    d_hidden = 16
    model = RGATModel(
        dglgraph=dglgraph,
        in_features=d_hidden,
        gnn_features=d_hidden,
        out_features=n_labels,
    ).to('cuda')
    b = model()
    a = 0


def test():
    check_minimal()


if __name__ == "__main__":
    test()
