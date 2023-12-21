from YiTu_GNN.mmap.partition import mmap_partition
from YiTu_GNN.mmap.partition import block_partition


if __name__ == "__main__":
    # block_partition(2, "/home/nx/ningxin/DistGNN/out-of-core-gnn/dataset/graph.txt")
    mmap_partition(2, data_path="/home/nx/ningxin/.data/mmap/ogbn-products")
