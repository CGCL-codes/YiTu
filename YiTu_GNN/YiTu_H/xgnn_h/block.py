import torch
from typing import List


class Block:
    def __init__(self, size: list, adj: list, right_norm=None):
        self.size = size
        self.adj_sparse = adj
        self.right_norm = right_norm

    def to(self, device):
        self.adj_sparse = [
            self.adj_sparse[0].to(device),
            self.adj_sparse[1].to(device)
        ]
        if self.right_norm is not None:
            self.right_norm = \
                self.right_norm.to(device)
        return self

    def num_edges(self):
        indices = self.adj_sparse[1]
        assert indices.dim() == 1
        return indices.numel()

    def num_src_nodes(self):
        return self.size[1]

    def num_dst_nodes(self):
        indptr = self.adj_sparse[0]
        assert indptr.dim() == 1
        assert self.size[0] == indptr.size(0) - 1
        return self.size[0]


def stitch_csr(block_list: List[Block]):
    from tqdm import tqdm
    lens = [
        blk.size[0]
        for blk in block_list
    ]
    n_rows = min(lens)
    assert len(lens) <= 16
    assert min(lens) == max(lens)
    new_indptr, new_indices = [], []
    for row in tqdm(range(n_rows)):
        cursor = 0
        new_indptr.append(len(new_indices))
        for bid, blk in enumerate(block_list):
            indptr, indices = blk.adj_sparse
            for i in range(indptr[row], indptr[row + 1]):
                col = cursor + indices[i].item()
                assert col <= 2 ** 24
                new_indices.append((bid << 24) + col)
            cursor += blk.size[1]
    new_indptr.append(len(new_indices))
    n_cols = sum([
        blk.size[1]
        for blk in block_list
    ])
    return Block(
        size=[n_rows, n_cols],
        adj=[
            torch.IntTensor(new_indptr),
            torch.IntTensor(new_indices)
        ]
    )
