import torch
from torch.nn import functional
from xgnn_h import mp, ir, sparse


class Executor:
    def __init__(self):
        self._cache = dict()
        self._is_training = None

    def eval(self):
        self._is_training = False

    def train(self):
        self._is_training = True

    def _execute(self, root_node: ir.Op, kwargs: dict):
        child_args = {
            k: self._execute(v, kwargs)
            for k, v in root_node.prevs.items()
        }
        if isinstance(root_node, ir.OpFusedSPMM):
            if root_node not in self._cache:
                if 'e' not in child_args:
                    edge = None
                else:
                    edge = child_args['e']
                self._cache[root_node] = sparse.gspmm(
                    block=child_args['g'],
                    edge=edge, x=child_args['x']
                )
            return self._cache[root_node]
        elif isinstance(root_node, ir.OpFusedSDDMM):
            if root_node not in self._cache:
                self._cache[root_node] = sparse.fused_gsddmm(
                    block=child_args['g'],
                    query=child_args['q'],
                    key=child_args['k']
                )
            return self._cache[root_node]
        elif isinstance(root_node, ir.OpLinear):
            if root_node not in self._cache:
                self._cache[root_node] = functional.linear(
                    input=child_args['x'],
                    weight=root_node.ref_params['weight'],
                    bias=root_node.ref_params['bias']
                )
            return self._cache[root_node]
        elif isinstance(root_node, ir.OpDropout):
            if root_node not in self._cache:
                self._cache[root_node] = functional.dropout(
                    child_args['x'],
                    p=root_node.val_params['p'],
                    training=self._is_training
                )
            return self._cache[root_node]
        elif isinstance(root_node, ir.OpSqueeze):
            if root_node not in self._cache:
                self._cache[root_node] = torch.squeeze(
                    child_args['x'],
                    dim=root_node.val_params['dim'],
                )
            return self._cache[root_node]
        elif isinstance(root_node, ir.OpMultiply):
            if root_node not in self._cache:
                self._cache[root_node] = torch.multiply(
                    child_args['a'], child_args['b'],
                )
            return self._cache[root_node]
        elif isinstance(root_node, ir.OpConcat):
            if root_node not in self._cache:
                self._cache[root_node] = torch.cat(
                    [v for _, v in child_args.items()],
                    dim=root_node.val_params['dim'],
                )
            return self._cache[root_node]
        elif isinstance(root_node, ir.OpStack):
            if root_node not in self._cache:
                self._cache[root_node] = torch.stack(
                    [v for _, v in child_args.items()],
                    dim=root_node.val_params['dim'],
                )
            return self._cache[root_node]
        elif isinstance(root_node, ir.OpEmbed):    
            if root_node not in self._cache:
                self._cache[root_node] = functional.embedding(
                    child_args['x'],
                    weight=root_node.ref_params['embed']
                )
            return self._cache[root_node]
        elif isinstance(root_node, ir.OpScale):
            if root_node not in self._cache:
                self._cache[root_node] = torch.multiply(
                    child_args['x'],
                    root_node.val_params['scale'],
                )
            return self._cache[root_node]
        elif isinstance(root_node, ir.OpMean):
            if root_node not in self._cache:
                self._cache[root_node] = torch.mean(
                    child_args['x'],
                    dim=root_node.val_params['dim']
                )
            return self._cache[root_node]
        elif isinstance(root_node, ir.OpView):
            if root_node not in self._cache:
                self._cache[root_node] = child_args['x'].view(
                    size=root_node.val_params['size']
                )
            return self._cache[root_node]
        elif isinstance(root_node, ir.OpRelu):
            if root_node not in self._cache:
                self._cache[root_node] = torch.relu(
                    child_args['x']
                )
            return self._cache[root_node]
        elif isinstance(root_node, ir.OpELU):
            if root_node not in self._cache:
                self._cache[root_node] = functional.elu(
                    child_args['x']
                )
            return self._cache[root_node]
        elif isinstance(root_node, ir.OpAdd):
            if root_node not in self._cache:
                self._cache[root_node] = torch.add(
                    child_args['a'], child_args['b']
                )
            return self._cache[root_node]
        elif isinstance(root_node, ir.OpGraph):
            if root_node.name in kwargs:
                return kwargs[root_node.name]
            if root_node not in self._cache:
                splited = root_node.name.split('.')
                name, idx = splited[0], int(splited[1])
                sty, ety, dty = kwargs[name].idx2rel[idx]
                self._cache[root_node] = kwargs[
                    name
                ].hetero_graph[sty, ety, dty].blk
            return self._cache[root_node]
        elif isinstance(root_node, ir.OpTensor):
            assert root_node.name
            if root_node.name in kwargs:
                return kwargs[root_node.name]
            if root_node not in self._cache:
                splited = root_node.name.split('.')
                name, entity = splited[0], splited[1]
                self._cache[root_node] = kwargs[name][entity]
            return self._cache[root_node]
        else:
            raise NotImplementedError

    def run(self, dataflow: ir.Op, kwargs: dict):
        # replace mp.Graph
        def process(kwargs):
            post_insert = []
            post_removal = []
            for k, v in kwargs.items():
                if isinstance(v, mp.Graph):
                    post_removal.append(k)
                    post_insert.append([k, v.blk])
            for k in post_removal:
                del kwargs[k]
            for k, v in post_insert:
                kwargs[k] = v
            return kwargs
        kwargs = process(kwargs)

        #
        try:
            return self._execute(dataflow, kwargs=kwargs)
        finally:
            self._cache.clear()
