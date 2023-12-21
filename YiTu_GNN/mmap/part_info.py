from collections import namedtuple
import json
import numpy as np

CsrGraph = namedtuple("CsrGraph", ["indptr", "indices"])


class PartInfo:
    def __init__(self, part_config_path) -> None:
        with open(part_config_path, "r") as f:
            self.__part_info = json.load(f)
        self._train_nodes = None
        self._test_nodes = None
        self._val_nodes = None
        self._labels = None
        self.__init_part_graph()

    def __init_part_graph(self):
        self.__part_graph = []
        self.__part_feat = []
        # 加载图拓扑信息和图特征
        for part_id in range(self.num_parts):
            node_map = self.node_map(part_id)
            edge_map = self.edge_map(part_id)
            part_graph_config = self.__part_info["part-{}".format(part_id)][
                "part-graph"
            ]
            row_ptr_path = part_graph_config["row_ptr"]
            row_ptr = np.lib.format.open_memmap(
                row_ptr_path, mode="r", shape=(node_map[1] - node_map[0] + 1,)
            )
            col_idx_path = part_graph_config["col_idx"]
            col_idx = np.lib.format.open_memmap(
                col_idx_path, mode="r", shape=(edge_map[1] - edge_map[0],)
            )
            self.__part_graph.append(CsrGraph(row_ptr, col_idx))
            feat_path = part_graph_config["node_feat"]
            feat = np.lib.format.open_memmap(feat_path, mode="r")
            self.__part_feat.append(feat)
        # 加载训练集、测试集、验证集和标签
        train_path = self.__part_info["train_path"]
        self._train_nodes = np.load(train_path)
        test_path = self.__part_info["test_path"]
        self._test_nodes = np.load(test_path)
        val_path = self.__part_info["val_path"]
        self._val_nodes = np.load(val_path)
        labels_path = self.__part_info["labels_path"]
        self._labels = np.load(labels_path)

    @property
    def num_parts(self):
        return self.__part_info["num_parts"]

    @property
    def num_edges(self):
        return self.__part_info["num_edges"]

    @property
    def num_nodes(self):
        return self.__part_info["num_nodes"]

    @property
    def train_nodes(self):
        return self._train_nodes

    @property
    def test_nodes(self):
        return self._test_nodes

    @property
    def val_nodes(self):
        return self._val_nodes

    @property
    def labels(self):
        return self._labels

    @property
    def feature_dim(self):
        return self.__part_feat[0].shape[1]

    def node_map(self, part_id):
        assert part_id < self.num_parts
        return self.__part_info["node_map"][part_id]

    def edge_map(self, part_id):
        assert part_id < self.num_parts
        return self.__part_info["edge_map"][part_id]

    def global2localid(self, ids, part_id):
        start = self.node_map(part_id)[0]
        globalid = ids - start
        return globalid

    def local2globalid(self, ids, part_id):
        start = self.node_map(part_id)[0]
        localid = ids + start
        return localid

    def ids2partids(self, ids):
        part_size = self.num_nodes // self.num_parts
        part_ids = ids // part_size
        mask = part_ids >= self.num_parts
        part_ids[mask] -= 1
        return part_ids

    def part_graph(self, part_id) -> CsrGraph:
        assert part_id < self.num_parts
        return self.__part_graph[part_id]

    def nodes_feat(self, nodes):
        part_ids = self.ids2partids(nodes)
        dim = self.__part_feat[0].shape[1]
        out = np.zeros((nodes.shape[0], dim))
        for i in range(self.num_parts):
            mask = part_ids == i
            local_nodes = self.global2localid(nodes[mask], i)
            out[mask] = self.__part_feat[i][local_nodes]
        return out
