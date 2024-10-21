import torch


def to_dense(n_rows, n_cols, sparse, values):
    indptr, indices = sparse
    assert values.dim() == 1
    assert values.size(0) == indices.size(0)
    dense = torch.zeros(size=[n_rows, n_cols])
    for row in range(n_rows):
        for i in range(indptr[row], indptr[row + 1]):
            col = indices[i]
            dense[row, col] = values[i].item()
    return dense


def to_dense_mha(n_rows, n_cols, sparse, values):
    indptr, indices = sparse
    assert values.dim() == 2
    assert values.size(0) == indices.size(0)
    n_heads = values.size(1)
    dense = torch.zeros(
        size=[n_heads, n_rows, n_cols]
    )
    for row in range(n_rows):
        for i in range(indptr[row], indptr[row + 1]):
            col = indices[i]
            for h in range(n_heads):
                dense[h, row, col] = values[i, h].item()
    return dense


def to_csr(dense):
    assert dense.dim() == 2
    n_rows = dense.size(0)
    values = []
    indptr, indices = [], []
    for row in range(n_rows):
        indptr.append(len(indices))
        for col in torch.nonzero(dense[row]):
            values.append(dense[row, col].item())
            indices.append(col.item())
    indptr.append(len(indices))
    #
    indptr = torch.IntTensor(indptr)
    indices = torch.IntTensor(indices)
    values = torch.FloatTensor(values)
    return [indptr, indices], values


def transpose(n_rows, n_cols, sparse, values):
    assert values.dim() == 1
    dense = to_dense(
        n_rows, n_cols, sparse, values
    )
    return to_csr(dense)[-1]


def transpose_mha(n_rows, n_cols, sparse, values):
    assert values.dim() == 2
    rev_values = []
    for h in range(values.size(-1)):
        dense = to_dense(
            n_rows, n_cols, sparse, values[:, h]
        ).T.contiguous()
        rev_values.append(to_csr(dense)[-1])
    rev_values = torch.stack(rev_values)
    return rev_values.T.contiguous()
