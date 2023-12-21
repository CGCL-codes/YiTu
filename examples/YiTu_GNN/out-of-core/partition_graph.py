import argparse
from YiTu_GNN.mmap.partition import mmap_partition


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Partition builtin graphs")
    argparser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-products",
        help="datasets: reddit, ogbn-products, ogbn-papers100M",
    )
    argparser.add_argument(
        "--num_parts", type=int, default=2, help="number of partitions"
    )
    argparser.add_argument(
        "--data_path",
        type=str,
        default="/home/nx/ningxin/.data/mmap/ogbn-products",
        help="Output path of partitioned graph.",
    )

    args = argparser.parse_args()
    # block_partition(2, "/home/nx/ningxin/DistGNN/out-of-core-gnn/dataset/graph.txt")
    mmap_partition(args.num_parts, data_path=args.data_path)
