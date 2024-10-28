import torch
from torch import nn
from xgnn_h import mp, ir, trace


class Optimizer:
    def __init__(self):
        pass

    def _lower_spmm(self, root_node: ir.Op):
        # recursive
        for name in root_node.prevs:
            child = root_node.prevs[name]
            child = self._lower_spmm(child)
            if root_node not in child.next:
                child.next.append(root_node)
            root_node.prevs[name] = child

        # transform
        aggre_node = None
        if isinstance(root_node, ir.OpVertFunc):
            if root_node.func_name != 'aggregate_sum':
                raise NotImplementedError
            aggre_node = root_node.prevs['e']
        if isinstance(aggre_node, ir.OpEdgeFunc):
            if aggre_node.func_name == 'copy_u':
                edge_node = None
                x_node = aggre_node.prevs['u']
                graph_node = aggre_node.prevs['g']
            elif aggre_node.func_name == 'u_mul_e':
                x_node = aggre_node.prevs['u']
                edge_node = aggre_node.prevs['e']
                graph_node = edge_node.prevs['g']
                if not isinstance(edge_node,
                                  ir.OpFusedSDDMM):
                    raise NotImplementedError
            else:
                raise NotImplementedError
            root_node = ir.OpFusedSPMM(
                graph=graph_node,
                edge=edge_node, x=x_node
            )

        #
        return root_node

    def _lower_sddmm(self, root_node: ir.Op):
        # recursive
        for name in root_node.prevs:
            child = root_node.prevs[name]
            child = self._lower_sddmm(child)
            root_node.prevs[name] = child

        # transform
        leaky_node = None
        if isinstance(root_node, ir.OpEdgeFunc):
            if root_node.func_name == 'edge_softmax':
                leaky_node = root_node.prevs['e']
        sddmm_node = None
        if isinstance(leaky_node, ir.OpLeakyRelu):
            sddmm_node = leaky_node.prevs['x']
        query_node, key_node = None, None
        if isinstance(sddmm_node, ir.OpEdgeFunc):
            if sddmm_node.func_name == 'u_add_v':
                graph_node = sddmm_node.prevs['g']
                query_node = sddmm_node.prevs['v']
                key_node = sddmm_node.prevs['u']
                root_node = ir.OpFusedSDDMM(
                    size=leaky_node.size,
                    graph=graph_node,
                    query=query_node,
                    key=key_node,
                    fusion_scheme='gat_sddmm'
                )
            else:
                raise NotImplementedError

        #
        return root_node

    def _bundle_gemm(self, root_node: ir.Op):
        # recursive
        for name in root_node.prevs:
            child = root_node.prevs[name]
            child = self._bundle_gemm(child)
            root_node.prevs[name] = child

        # transform
        query_node, key_node = None, None
        if isinstance(root_node, ir.OpFusedSDDMM):
            key_node = root_node.prevs['k']
            query_node = root_node.prevs['q']

        #
        return root_node

    def lower(self, dataflow, kwargs: dict):
        if isinstance(dataflow, dict):
            dataflow = {
                k: self._lower_spmm(
                    self._lower_sddmm(v)
                )
                for k, v in dataflow.items()
            }
        elif isinstance(dataflow, ir.Op):
            dataflow = self._lower_spmm(
                self._lower_sddmm(dataflow)
            )
        else:
            raise NotImplementedError
        return dataflow
