import numpy as np

from YiTu_GNN import sample_kernel
from YiTu_GNN.mmap import PartInfo
from YiTu_GNN.mmap.sampler import mmap_neighbor_sampler


def test():
    part_config_path = "/home/nx/ningxin/.data/mmap/output/part_info.json"
    part_info = PartInfo(part_config_path)
    csr = part_info.part_graph(0)
    seeds = np.array([0, 200000], dtype=np.int32)
    res = mmap_neighbor_sampler(seeds, [2, 3], part_info)
    print(res)
    all_nodes = res[0][1]
    feat = part_info.nodes_feat(all_nodes)
    # print(feat)
    # print(res)


test()
