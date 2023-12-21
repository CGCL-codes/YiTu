import numpy as np
cimport numpy as np
cimport cython
from libcpp.map cimport map
from libcpp.set cimport set
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libc.stdlib cimport srand, rand, RAND_MAX
from libc.time cimport time
from libcpp cimport bool
from cython.parallel import prange

@cython.boundscheck(False)
@cython.wraparound(False)
def map_edges(np.ndarray[np.int32_t, ndim=2] edges, reindex):
    """Mapping edges by given dictionary
    """
    cdef unordered_map[int, int] m = reindex
    cdef int i = 0
    cdef int h = edges.shape[1]
    cdef int j
    cdef int [:, :] edges_view = edges
    with nogil:
        for i in prange(h, schedule="static"):
            edges_view[0, i] = m[edges_view[0, i]]
            edges_view[1, i] = m[edges_view[1, i]]
    return edges


@cython.boundscheck(False)
@cython.wraparound(False)
def map_nodes(nodes, reindex):
    """Mapping nodes by given dictionary
    """
    cdef np.ndarray[np.int32_t, ndim=1] t_nodes = np.array(nodes, dtype=np.int32)
    cdef unordered_map[int, int] m = reindex
    cdef int i = 0
    cdef int h = len(nodes)
    cdef np.ndarray[np.int32_t, ndim=1] new_nodes = np.zeros([h], dtype=np.int32)
    cdef int j
    with nogil:
        for i in xrange(h):
            j = t_nodes[i]
            new_nodes[i] = m[j]
    return new_nodes


@cython.boundscheck(False)
@cython.wraparound(False)
def sample_one_hop_unbias(np.ndarray[np.int32_t, ndim=1] csr_row, np.ndarray[np.int32_t, ndim=1] csr_col, int neighbor_num,
                          np.ndarray[np.int32_t, ndim=1] seeds, bool replace=False):

    
    cdef int seeds_length = len(seeds)

    cdef int seed_idx
    cdef int node
    cdef int col_start, col_end
    cdef int total_edge_num = 0
    cdef int offset = 0
    cdef int idx
    cdef unordered_set[int] n_id_set
    cdef vector[int] n_ids

    for seed_idx in xrange(seeds_length):
        node = seeds[seed_idx]
        n_id_set.insert(node)
        n_ids.push_back(node)
        col_start, col_end = csr_row[node], csr_row[node + 1]
        if col_end - col_start <= neighbor_num:
            total_edge_num += col_end - col_start
        else:
            total_edge_num += neighbor_num
    #
    cdef np.ndarray[np.int32_t, ndim=1] edge_ids = np.zeros([total_edge_num], dtype=np.int32)

    # define result
    cdef np.ndarray[np.int32_t, ndim=2] res_edge_index = np.zeros([2, total_edge_num], dtype=np.int32)
    for seed_idx in xrange(seeds_length):
        node = seeds[seed_idx]
        col_start, col_end = csr_row[node], csr_row[node + 1]
        if col_end - col_start <= neighbor_num:
            # print("neighbor < count")
            res_edge_index[0][offset: offset + col_end - col_start] = node

            res_edge_index[1][offset: offset + col_end - col_start] = csr_col[col_start: col_end]
            for idx in xrange(col_end - col_start):
                edge_ids[offset + idx] = col_start + idx
                if n_id_set.find(csr_col[col_start + idx]) == n_id_set.end():
                    n_id_set.insert(csr_col[col_start + idx])
                    n_ids.push_back(csr_col[col_start + idx])
            offset += col_end - col_start
        else:

            choose_index(edge_ids, neighbor_num, col_end - col_start, offset, col_start)
            #update
            res_edge_index[0, offset: offset + neighbor_num] = node
            res_edge_index[1, offset: offset + neighbor_num] = csr_col[edge_ids[offset: offset + neighbor_num]]
            for idx in range(offset, offset + neighbor_num):
                if n_id_set.find(csr_col[edge_ids[idx]]) == n_id_set.end():
                    n_id_set.insert(csr_col[edge_ids[idx]])
                    n_ids.push_back(csr_col[edge_ids[idx]])
            offset += neighbor_num
    return res_edge_index, edge_ids, n_ids


@cython.boundscheck(False)
@cython.wraparound(False)
def mmap_sample_one_hop_unbias_helper(np.ndarray[np.int32_t, ndim=1] csr_row, np.ndarray[np.int32_t, ndim=1] csr_col, int neighbor_num,
                          np.ndarray[np.int32_t, ndim=1] seeds, vector[int] n_ids, bool replace=False):
    cdef int seeds_length = len(seeds)

    cdef int seed_idx
    cdef int node
    cdef int col_start, col_end
    cdef int total_edge_num = 0
    cdef int offset = 0
    cdef int idx
    # 利用n_ids初始化set
    cdef unordered_set[int] n_id_set = {i for i in n_ids}
    # cdef vector[int] n_ids

    # 获取结果edge_index的大小
    for seed_idx in xrange(seeds_length):
        node = seeds[seed_idx]
        # n_id_set.insert(node)
        # n_ids.push_back(node)
        col_start, col_end = csr_row[node], csr_row[node + 1]
        if col_end - col_start <= neighbor_num:
            total_edge_num += col_end - col_start
        else:
            total_edge_num += neighbor_num
    
    cdef np.ndarray[np.int32_t, ndim=1] edge_ids = np.zeros([total_edge_num], dtype=np.int32)

    # define result
    cdef np.ndarray[np.int32_t, ndim=2] res_edge_index = np.zeros([2, total_edge_num], dtype=np.int32)
    for seed_idx in xrange(seeds_length):
        node = seeds[seed_idx]
        col_start, col_end = csr_row[node], csr_row[node + 1]
        if col_end - col_start <= neighbor_num:
            # print("neighbor < count")
            res_edge_index[0][offset: offset + col_end - col_start] = node

            res_edge_index[1][offset: offset + col_end - col_start] = csr_col[col_start: col_end]
            for idx in xrange(col_end - col_start):
                edge_ids[offset + idx] = col_start + idx
                if n_id_set.find(csr_col[col_start + idx]) == n_id_set.end():
                    n_id_set.insert(csr_col[col_start + idx])
                    n_ids.push_back(csr_col[col_start + idx])
            offset += col_end - col_start
        else:

            choose_index(edge_ids, neighbor_num, col_end - col_start, offset, col_start)
            #update
            res_edge_index[0, offset: offset + neighbor_num] = node
            res_edge_index[1, offset: offset + neighbor_num] = csr_col[edge_ids[offset: offset + neighbor_num]]
            for idx in range(offset, offset + neighbor_num):
                if n_id_set.find(csr_col[edge_ids[idx]]) == n_id_set.end():
                    n_id_set.insert(csr_col[edge_ids[idx]])
                    n_ids.push_back(csr_col[edge_ids[idx]])
            offset += neighbor_num
    return res_edge_index, edge_ids, n_ids


@cython.boundscheck(False)
@cython.wraparound(False)
def set_node_map_idx(np.ndarray[ndim=1, dtype=np.int32_t] node_map_idx, np.ndarray[ndim=1, dtype=np.int32_t] graph_nodes):
    cdef int [:] node_map_idx_view = node_map_idx
    cdef int [:] graph_nodes_view = graph_nodes
    cdef int node_count = graph_nodes.shape[0]
    cdef int idx
    with nogil:
        for idx in prange(node_count, schedule="static"):
            node_map_idx_view[graph_nodes_view[idx]: graph_nodes_view[idx + 1]] = idx
    return node_map_idx

@cython.boundscheck(False)
@cython.wraparound(False)
def set_edge_map_idx(np.ndarray[ndim=1, dtype=np.int32_t] edge_map_idx, np.ndarray[ndim=1, dtype=np.int32_t] graph_edges):
    return set_node_map_idx(edge_map_idx, graph_edges)

@cython.boundscheck(False)
@cython.wraparound(False)
def choose_index(np.ndarray[ndim=1, dtype=np.int32_t] rnd, int neighbor_count, int total_neighbor_size,
                 int offset, int col_start):
    # Sample without replacement via Robert Floyd algorithm
    # https://www.nowherenearithaca.com/2013/05/robert-floyds-tiny-and-beautiful.html
    cdef unordered_set[int] perm
    cdef int j, t
    cdef idx = 0
    # set rand seed
    srand(time(NULL))
    for j in xrange(total_neighbor_size - neighbor_count, total_neighbor_size):
        t = rand() % j
        # t不存在则采样该邻居
        if perm.find(t) == perm.end():
            perm.insert(t)
            rnd[offset + idx] = t + col_start
            # print("if: {}".format(t + col_start))
        else:
            perm.insert(j)
            rnd[offset + idx] = j + col_start
            # print("else: {}".format(j + col_start))
        idx += 1
