import numpy as np
import argparse
import os


def compute_mask(nodes_id, num_nodes):
    mask = np.zeros(num_nodes, dtype=bool)
    mask[nodes_id] = True
    return mask


def process_txt_data(root, dataset, num_classes, feat_dims, split="\t", output=None):
    """
    dataset: soc-LiveJournal1, soc-pokec-relationships
    """
    coo_row = []
    coo_col = []
    data_path = os.path.join(root, dataset + ".txt")
    # 读取图拓扑
    with open(data_path, "r") as f:
        for line in f:
            line = line.strip("\n")
            if line[0] != "#":
                edge = line.split(split)
                coo_row.append(int(edge[0]))
                coo_col.append(int(edge[1]))
    coo_row = np.array(coo_row)
    coo_col = np.array(coo_col)
    dst_nodes = np.unique(coo_col)
    max_nodes = 0
    row_max = coo_row.max()
    max_nodes = row_max if row_max > max_nodes else max_nodes
    col_max = coo_col.max()
    max_nodes = col_max if col_max > max_nodes else max_nodes
    num_nodes = max_nodes + 1
    num_edges = coo_row.shape[0]
    print("Graph {}, num_nodes={}, num_edges={}".format(dataset, num_nodes, num_edges))
    # 随机生成图特征
    feats = np.random.randn(num_nodes, feat_dims)
    feats = np.array(feats, dtype=np.float32)
    # 随机生成label
    labels = np.random.randint(low=0, high=num_classes, size=num_nodes)
    # 随机生成训练集、验证集与测试集
    # 6:2:2
    # 打乱所有目标节点
    np.random.shuffle(dst_nodes)
    num_train = len(dst_nodes) * 6 // 10
    num_val = len(dst_nodes) * 2 // 10
    num_test = len(dst_nodes) * 2 // 10
    train_idx = np.arange(0, num_train)
    train_id = dst_nodes[train_idx]
    train_mask = compute_mask(train_id, num_nodes)

    val_idx = np.arange(num_train, num_train + num_val)
    val_id = dst_nodes[val_idx]
    val_mask = compute_mask(val_id, num_nodes)

    test_idx = np.arange(num_train + num_val, num_train + num_val + num_test)
    test_id = dst_nodes[test_idx]
    test_mask = compute_mask(test_id, num_nodes)

    print(
        "Graph {}, #train={}, #val={}, #test={}".format(
            dataset, len(train_id), len(val_id), len(test_id)
        )
    )
    if output is None:
        output = os.path.join(root, dataset + ".npz")
    np.savez(
        output,
        feat=feats,
        label=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        src_li=coo_row,
        dst_li=coo_col,
        num_edges=num_edges,
        num_nodes=num_nodes,
        n_classes=num_classes,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/home/data/")
    parser.add_argument(
        "--dataset",
        type=str,
        default="soc-LiveJournal1",
        help="dataset: soc-LiveJournal1, soc-pokec-relationships",
    )
    parser.add_argument("--num_classes", type=int, default=11)
    parser.add_argument("--feat_dims", type=int, default=512)
    parser.add_argument("--split", type=str, default="\t")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    process_txt_data(
        args.data_path,
        args.dataset,
        args.num_classes,
        args.feat_dims,
        args.split,
        args.output,
    )
