from typing import List
import os
import shutil
from scipy.sparse import csr_matrix
import numpy as np
import numpy.ma as ma
import json

# MAX_LENGTH = 3


def get_part_id(num_partitions, num_nodes, node_id):
    part_size = num_nodes // num_partitions
    part_id = (
        num_partitions - 1
        if node_id >= part_size * num_partitions
        else node_id // part_size
    )
    assert part_id < num_partitions, "part id: {} >= num_partitions: {}".format(
        part_id, num_partitions
    )
    return part_id


def write_list_to_file(edges: List[str], file_path):
    with open(file_path, "a+") as f:
        for edge in edges:
            f.write(edge)
    edges.clear()


def save_partition_to_npy(file_path, part_path, node_map, num_nodes, split):
    row = []
    col = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip().split(split)
            row.append(int(line[0]))
            col.append(int(line[1]))
    # coo to csr
    row = np.array(row, dtype=np.int32)
    col = np.array(col, dtype=np.int32)
    data = np.zeros(row.shape[0])
    csr = csr_matrix((data, (row, col)), shape=(node_map[1], num_nodes))
    indptr = csr.indptr[node_map[0] :]
    # np.savez(np_path, row_ptr=indptr, col_idx=csr.indices)
    np.save(part_path["row_ptr"], indptr)
    np.save(part_path["col_idx"], csr.indices)


def block_partition(
    num_partitions, dataset_path, output_path=None, split=" ", block_size=10000
):
    """对使用边表格式的大规模图数据进行partition, 采样分块读取的方式将属于相同partition的边写入到同一个
    临时文件中, 然后将临时文件中的数据读入到内存中, 将coo格式的图转换成csr格式.
    partition方法: 将节点ID在 [i * part_size, (i + 1) * part_size) 范围内的节点划分到partition i中.
    """
    if output_path is None:
        base = os.path.dirname(dataset_path)
        output_path = base + "/output"
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)
    parts = [[] for _ in range(num_partitions)]
    num_edges = num_nodes = 0
    part_edge_count = [0 for _ in range(num_partitions)]
    # 1. 创建num_partitions个文件
    with open(dataset_path, "r") as f:
        # 从数据第一行中读取图的节点数和边数
        line = f.readline().strip().split(split)
        num_nodes = int(line[0])
        num_edges = int(line[1])
        while True:
            # 分块读取数据
            line = f.readlines(block_size)
            # 遍历行数据
            for edge in line:
                src = int(edge.split(split)[0])
                part_id = get_part_id(num_partitions, num_nodes, src)
                part_edge_count[part_id] += 1
                parts[part_id].append(edge)
                if len(parts[part_id]) > block_size:
                    file_path = output_path + "/{}.txt".format(part_id)
                    write_list_to_file(parts[part_id], file_path)
            if not line:
                break
    # 2. 将edge写入到临时的partition文件中
    for i, part in enumerate(parts):
        file_path = output_path + "/{}.txt".format(i)
        write_list_to_file(part, file_path)
    node_map = []
    part_size = num_nodes // num_partitions
    for i in range(num_partitions):
        if i != num_partitions - 1:
            node_map.append([part_size * i, part_size * (i + 1)])
        else:
            node_map.append([part_size * i, num_nodes])
    edge_map = []
    count = 0
    for i in range(num_partitions):
        if i == 0:
            edge_map.append([0, part_edge_count[i]])
        else:
            edge_map.append([count, count + part_edge_count[i]])
        count += part_edge_count[i]
    # 分区信息：num_parts, num_edges, num_nodes, num_parts, node_map
    part_info = {
        "num_parts": num_partitions,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "node_map": node_map,
        "edge_map": edge_map,
    }
    # 3. 将partition文件中的数据转换成csr格式并以npy的格式保存到文件中
    for i in range(num_partitions):
        file_path = output_path + "/{}.txt".format(i)
        # 将结果保存到该文件夹中
        os.mkdir("{}/part-{}".format(output_path, i))
        row_ptr_path = "{}/part-{}/rowptr.npy".format(output_path, i)
        col_idx_path = "{}/part-{}/colidx.npy".format(output_path, i)
        part_path = {"row_ptr": row_ptr_path, "col_idx": col_idx_path}
        # np_path = "{}/part-{}.npz".format(output_path, i)
        save_partition_to_npy(file_path, part_path, node_map[i], num_nodes, split)
        part_info["part-{}".format(i)] = {"part-graph": part_path}
    # 4. 将partition信息写入json文件中
    with open("{}/part_info.json".format(output_path), "w") as outfile:
        json.dump(part_info, outfile, sort_keys=True, indent=4)
    # 5. 删除临时的part文件
    for i, part in enumerate(parts):
        file_path = output_path + "/{}.txt".format(i)
        os.remove(file_path)


def mmap_partition(num_partitions, data_path, output_path=None):
    """利用mmap对大规模图数据进行partition"""
    if output_path is None:
        base = os.path.dirname(data_path)
        output_path = base + "/output"
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)
    # 输入图数据包括以下数据
    file_names = [
        "edge_index.npy",
        "node_feat.npy",
        "labels.npy",
        "train_nodes.npy",
        "test_nodes.npy",
        "val_nodes.npy",
    ]
    graph_info = None
    with open("{}/graph_info.json".format(data_path, "r")) as f:
        graph_info = json.load(f)
    num_nodes = int(graph_info["num_nodes"])
    num_edges = int(graph_info["num_edges"])
    edge_index_path = "{}/edge_index.npy".format(data_path)
    edge_index = np.lib.format.open_memmap(edge_index_path, mode="r")
    node_feat = np.lib.format.open_memmap(
        "{}/node_feat.npy".format(data_path), mode="r"
    )
    shutil.copy("{}/labels.npy".format(data_path), output_path)
    shutil.copy("{}/train_nodes.npy".format(data_path), output_path)
    shutil.copy("{}/test_nodes.npy".format(data_path), output_path)
    shutil.copy("{}/val_nodes.npy".format(data_path), output_path)
    part_size = num_nodes // num_partitions
    node_map = []
    edge_map = []
    part_info = {
        "num_parts": num_partitions,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "train_path": "{}/train_nodes.npy".format(output_path),
        "test_path": "{}/test_nodes.npy".format(output_path),
        "val_path": "{}/val_nodes.npy".format(output_path),
        "labels_path": "{}/labels.npy".format(output_path),
    }
    for i in range(num_partitions):
        if i != num_partitions - 1:
            min_node = i * part_size
            max_node = (i + 1) * part_size
        else:
            min_node = i * part_size
            max_node = num_nodes
        node_map.append([min_node, max_node])
        # partition graph
        mask = ma.masked_inside(edge_index[0], min_node, max_node - 1)
        row = edge_index[0][mask.mask]
        col = edge_index[1][mask.mask]
        data = np.zeros(row.shape[0])
        # coo to csr
        csr = csr_matrix((data, (row, col)), shape=(node_map[i][1], num_nodes))
        # 将结果保存到该文件夹中
        os.mkdir("{}/part-{}".format(output_path, i))
        row_ptr_path = "{}/part-{}/rowptr.npy".format(output_path, i)
        col_idx_path = "{}/part-{}/colidx.npy".format(output_path, i)

        indptr = csr.indptr[node_map[i][0] :]
        np.save(row_ptr_path, indptr)
        np.save(col_idx_path, csr.indices)
        # partition feature
        feat = node_feat[min_node:max_node, :]
        feat_path = "{}/part-{}/node_feat.npy".format(output_path, i)
        np.save(feat_path, feat)
        part_path = {
            "row_ptr": row_ptr_path,
            "col_idx": col_idx_path,
            "node_feat": feat_path,
        }
        part_info["part-{}".format(i)] = {"part-graph": part_path}
        if i == 0:
            edge_map.append([0, row.shape[0]])
        else:
            edge_map.append([edge_map[i - 1][1], edge_map[i - 1][1] + row.shape[0]])
    part_info["node_map"] = node_map
    part_info["edge_map"] = edge_map
    # 4. 将partition信息写入json文件中
    with open("{}/part_info.json".format(output_path), "w") as outfile:
        json.dump(part_info, outfile, sort_keys=True, indent=4)
