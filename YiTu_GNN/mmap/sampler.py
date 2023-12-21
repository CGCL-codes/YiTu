import numpy as np

from YiTu_GNN import sample_kernel
from YiTu_GNN.mmap import PartInfo


def __mmap_sample_one_hop(seeds, neighbor_num, part_info: PartInfo):
    # 1. 将seeds按照图partition进行切分
    part_ids = part_info.ids2partids(seeds)
    part_seeds = []
    for i in range(part_info.num_parts):
        mask = part_ids == i
        # 转换成local id进行采样
        local_seeds = part_info.global2localid(seeds[mask], i)
        part_seeds.append(local_seeds)
    nids = seeds.tolist()
    all_edge_index = []
    # 2. 读取seed对应的part，将seed转换成local id并执行采样
    for i in range(part_info.num_parts):
        csr = part_info.part_graph(i)
        # nids表示采样得到的所有节点，edge_ids表示在csr中的id(local id), 暂时没使用
        edge_index, edge_ids, nids = sample_kernel.mmap_sample_one_hop_unbias_helper(
            csr.indptr, csr.indices, neighbor_num, part_seeds[i], nids
        )
        # 转置
        # edge_index = np.array([edge_index[1], edge_index[0]])
        all_edge_index.append(edge_index)
    all_nids = np.array(nids, dtype=np.int32)
    # 结果：row是local id，column是global id
    # 3. 将local id转换成global id合并结果
    for i in range(part_info.num_parts):
        all_edge_index[i][0] = part_info.local2globalid(all_edge_index[i][0], i)
    output_edge_index = np.concatenate(all_edge_index, axis=1)
    # 4. reindex, 对结果重新编号
    reindex_map = {all_nids[index]: index for index in range(all_nids.shape[0])}
    output_edge_index = sample_kernel.map_edges(output_edge_index, reindex_map)
    # 转置
    output_edge_index = np.array([output_edge_index[1], output_edge_index[0]])
    return output_edge_index, all_nids


def mmap_neighbor_sampler(seeds: np.ndarray, hops, part_info: PartInfo):
    blocks = []
    for hop in hops:
        seeds = np.array(seeds, dtype=np.int32)
        # 目标节点数
        target_size = seeds.shape[0]
        edge_index, all_nids = __mmap_sample_one_hop(seeds, hop, part_info)
        seeds = all_nids
        blocks.insert(0, (edge_index, all_nids, target_size))
    return blocks
