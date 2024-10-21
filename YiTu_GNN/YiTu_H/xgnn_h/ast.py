import torch
import xgnn_h
from torch import nn
from xgnn_h import mp, ir, trace


class Module2IR:
    def __init__(self):
        self._tracer2ir = dict()

    def _visit(self, node, kwargs: dict):
        #
        if node in self._tracer2ir:
            return self._tracer2ir[node]

        #
        if isinstance(node, mp.Graph):
            for k, v in kwargs.items():
                if v != node:
                    continue
                self._tracer2ir[node] = ir.OpGraph(
                    node.blk, name=k
                )
                return self._tracer2ir[node]
            for k, het in kwargs.items():
                if not isinstance(
                    het, mp.HeteroGraph
                ):
                    continue
                for rel, homg in \
                        het.hetero_graph.items():
                    blk = homg.blk
                    if blk != node.blk:
                        continue
                    self._tracer2ir[node] = ir.OpGraph(
                        node.blk,
                        name='{}.{}'.format(
                            k, het.rel2idx[rel]
                        )
                    )
                    return self._tracer2ir[node]
            raise RuntimeError
        elif isinstance(node, trace.Tracer):
            if not node.previous_func:
                for k, v in kwargs.items():
                    if id(v) != id(node):
                        continue
                    self._tracer2ir[node] = ir.OpTensor(
                        size=node.size(),
                        name=k
                    )
                    return self._tracer2ir[node]
                for k, v in kwargs.items():
                    if not isinstance(v, dict):
                        continue
                    for entity, t in v.items():
                        if id(t) != id(node):
                            continue
                        self._tracer2ir[node] = ir.OpTensor(
                            size=node.size(),
                            name='{}.{}'.format(k, entity)
                        )
                        return self._tracer2ir[node]
                raise RuntimeError
            elif node.previous_func == 'add':
                node_a, node_b = node.previous_args
                self._tracer2ir[node] = ir.OpAdd(
                    a=self._visit(node_a, kwargs),
                    b=self._visit(node_b, kwargs)
                )
                return self._tracer2ir[node]
            elif node.previous_func == 'div':
                node_x, node_a = node.previous_args
                if not isinstance(node_a, (int,
                                           float)):
                    raise NotImplementedError
                scale = 1.0 / node_a
                self._tracer2ir[node] = ir.OpScale(
                    scale=scale,
                    x=self._visit(node_x, kwargs)
                )
                return self._tracer2ir[node]
            elif node.previous_func == 'elu':
                node_x, = node.previous_args
                self._tracer2ir[node] = ir.OpELU(
                    x=self._visit(node_x, kwargs)
                )
                return self._tracer2ir[node]
            elif node.previous_func == 'relu':
                node_x, = node.previous_args
                self._tracer2ir[node] = ir.OpRelu(
                    x=self._visit(node_x, kwargs)
                )
                return self._tracer2ir[node]
            elif node.previous_func == 'view':
                node_x, = node.previous_args
                self._tracer2ir[node] = ir.OpView(
                    x=self._visit(node_x, kwargs),
                    size=node.previous_kwargs['size']
                )
                return self._tracer2ir[node]
            elif node.previous_func == 'mean':
                node_x, = node.previous_args
                self._tracer2ir[node] = ir.OpMean(
                    x=self._visit(node_x, kwargs),
                    dim=node.previous_kwargs['dim']
                )
                return self._tracer2ir[node]
            elif node.previous_func == 'linear':
                node_b = None
                if 'bias' in node.previous_kwargs:
                    node_b = node.previous_kwargs['bias']
                node_x, node_w = node.previous_args
                self._tracer2ir[node] = ir.OpLinear(
                    x=self._visit(node_x, kwargs),
                    w=node_w, b=node_b
                )
                return self._tracer2ir[node]
            elif node.previous_func == 'dropout':
                node_x, = node.previous_args
                self._tracer2ir[node] = ir.OpDropout(
                    x=self._visit(node_x, kwargs),
                    p=node.previous_kwargs['p']
                )
                return self._tracer2ir[node]
            elif node.previous_func == 'squeeze':
                node_x, = node.previous_args
                self._tracer2ir[node] = ir.OpSqueeze(
                    x=self._visit(node_x, kwargs),
                    dim=node.previous_kwargs['dim']
                )
                return self._tracer2ir[node]
            elif node.previous_func == 'multiply':
                node_a, node_b = node.previous_args
                self._tracer2ir[node] = ir.OpMultiply(
                    a=self._visit(node_a, kwargs),
                    b=self._visit(node_b, kwargs),
                )
                return self._tracer2ir[node]
            elif node.previous_func == 'embedding':
                node_xs = node.previous_args
                self._tracer2ir[node] = ir.OpEmbed(
                    x=self._visit(node_xs[0], kwargs),
                    embed=node_xs[1]
                )
                return self._tracer2ir[node]
            elif node.previous_func == 'leaky_relu':
                node_x, = node.previous_args
                self._tracer2ir[node] = ir.OpLeakyRelu(
                    x=self._visit(node_x, kwargs)
                )
                return self._tracer2ir[node]
            elif node.previous_func == 'reduce_wrapper':
                block, func_name = node.previous_args
                prevs = {
                    'g': self._visit(block, kwargs)
                }
                prevs.update({
                    k: self._visit(v, kwargs)
                    for k, v in node.previous_kwargs.items()
                })
                self._tracer2ir[node] = ir.OpVertFunc(
                    size=node.size(),
                    prevs=prevs, func_name=func_name
                )
                return self._tracer2ir[node]
            elif node.previous_func == 'message_wrapper':
                block, func_name = node.previous_args
                prevs = {
                    'g': self._visit(block, kwargs)
                }
                prevs.update({
                    k: self._visit(v, kwargs)
                    for k, v in node.previous_kwargs.items()
                })
                self._tracer2ir[node] = ir.OpEdgeFunc(
                    size=node.size(),
                    prevs=prevs, func_name=func_name
                )
                return self._tracer2ir[node]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def transform(self, model: nn.Module, kwargs: dict) -> ir.Op:
        # build tracer
        def process(x):
            if isinstance(x, dict):
                return {
                    k: process(v)
                    for k, v in x.items()
                }
            elif isinstance(x, (list, tuple)):
                return [
                    process(v) for v in x
                ]
            elif isinstance(x, torch.Tensor):
                xty = x.type()
                return trace.Tracer(
                    torch.zeros(x.size(), device='cpu')
                ).type(xty).to(x.device)
            elif isinstance(x, (mp.Graph,
                                mp.HeteroGraph)):
                return x
            else:
                raise NotImplementedError
        kwargs = process(kwargs)

        #
        output = model(**kwargs)

        # transform to ir by tracer
        if isinstance(output, dict):
            root = {
                k: self._visit(
                    v, kwargs
                )
                for k, v in output.items()
            }
        elif isinstance(output, torch.Tensor):
            root = self._visit(
                output, kwargs
            )
        else:
            raise NotImplementedError

        return root
