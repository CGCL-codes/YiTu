import dgl
import torch
import sys
print(sys.path)
from YiTu_GNN.utils import partition_graph
from dgl.data.utils import load_tensors
import os
import json

def load_features(part_config, part_id):
    config_path = os.path.dirname(part_config)
    relative_to_config = lambda path: os.path.join(config_path, path)

    with open(part_config) as conf_f:
        part_metadata = json.load(conf_f)
    assert "part-{}".format(part_id) in part_metadata, "part-{} does not exist".format(
        part_id
    )
    part_files = part_metadata["part-{}".format(part_id)]
    assert "part_feats" in part_files, "the partition does not contain node features."
    part_feats = load_tensors(relative_to_config(part_files["part_feats"]))
    print("part id: {}, feats shape: {}".format(part_id, part_feats["feat"].shape))
    return part_feats["feat"]


if __name__ == "__main__":

    src = torch.LongTensor([0, 0, 1, 2, 3, 4])
    dst = torch.LongTensor([1, 3, 4, 0, 4, 0])
    g = dgl.graph((src, dst))
    g.ndata["feat"] = torch.tensor([[i]*5 for i in range(5)])
    orig_ids, orig_edges = partition_graph(
        g,
        "test",
        2,
        num_hops=1,
        part_method="random",
        out_path="output/",
        reshuffle=True,
        feat_name="feat",
        balance_edges=True,
        return_mapping=True
    )
    print(orig_ids)
    print(orig_edges)
    g0, node_feats, edge_feats, gpb, graph_name, _, _ = dgl.distributed.load_partition(
        "output/test.json", 0
    )
    g1, node_feats, edge_feats, gpb, graph_name, _, _ = dgl.distributed.load_partition(
        "output/test.json", 1
    )
    feat_0 = load_features("output/test.json", 0)
    feat_1 = load_features("output/test.json", 1)
    print("feat 0: {}".format(feat_0))
    print("feat 1: {}".format(feat_1))
    
