import torch
from torch import nn
from xgnn_h import block
from typing import List, Dict, Union


class Op:
    def __init__(self, prevs: dict, name: str = ''):
        if type(self) is Op:
            raise RuntimeError('base class should be inherited')
        self.name = name
        self.size = None
        self.next = []
        self.prevs = prevs
        self.ref_params = {}
        self.val_params = {}


class OpGraph(Op):
    def __init__(self, blk: block.Block, name: str):
        Op.__init__(self, {}, name)
        assert blk.size
        self.size = blk.size


class OpTensor(Op):
    def __init__(self, size: List[int], prevs: dict = {}, name: str = ''):
        Op.__init__(self, prevs, name)
        self.size = list(size)
        if not prevs:
            return
        assert isinstance(prevs, dict)
        for n in prevs.values():
            n.next.append(self)


class OpAdd(OpTensor):
    def __init__(self,
                 a: OpTensor,
                 b: OpTensor,
                 name: str = ''):
        assert a.size == b.size
        OpTensor.__init__(
            self,
            size=b.size,
            prevs={'a': a, 'b': b},
            name=name
        )


class OpEmbed(OpTensor):
    def __init__(self,
                 x: torch.Tensor,
                 embed: nn.Parameter,
                 name: str = ''):
        assert len(x.size) == 1
        OpTensor.__init__(
            self,
            size=[x.size[0], embed.size()[1:]],
            prevs={'x': x},
            name=name
        )
        self.ref_params = {'embed': embed}


class OpView(OpTensor):
    def __init__(self,
                 x: OpTensor,
                 size: list,
                 name: str = ''):
        eval_size = []
        for i, s in enumerate(size):
            if s != -1:
                eval_size.append(s)
            else:
                eval_size.append(x.size[i])
        OpTensor.__init__(
            self,
            size=eval_size,
            prevs={'x': x},
            name=name
        )
        self.val_params['size'] = size


class OpMean(OpTensor):
    def __init__(self,
                 x: OpTensor,
                 dim: int,
                 name: str = ''):
        assert dim < len(x.size)
        size = x.size[:dim] + x.size[dim+1:]
        OpTensor.__init__(
            self,
            size=size,
            prevs={'x': x},
            name=name
        )
        self.val_params['dim'] = dim


class OpScale(OpTensor):
    def __init__(self,
                 scale: float,
                 x: OpTensor,
                 name: str = ''):
        OpTensor.__init__(
            self,
            size=x.size,
            prevs={'x': x},
            name=name
        )
        self.val_params['scale'] = scale


class OpStack(OpTensor):
    def __init__(self,
                 xs: List[OpTensor],
                 dim: int,
                 name: str = ''):
        if dim != 0:
            raise NotImplementedError
        OpTensor.__init__(
            self,
            size=[len(xs)] + list(xs[0].size),
            prevs={k: v for k, v in enumerate(xs)},
            name=name
        )
        self.val_params['dim'] = dim


class OpConcat(OpTensor):
    def __init__(self,
                 xs: List[OpTensor],
                 dim: int,
                 name: str = ''):
        if dim != 0:
            raise NotImplementedError
        size = list(xs[0].size)
        for x in xs[1:]:
            size[0] += x.size[0]
            assert size[1:] == x.size[1:]
        OpTensor.__init__(
            self,
            size=size,
            prevs={k: v for k, v in enumerate(xs)},
            name=name
        )
        self.val_params['dim'] = dim


class OpSqueeze(OpTensor):
    def __init__(self,
                 x: OpTensor,
                 dim: int,
                 name: str = ''):
        if dim == -1:
            raise NotImplementedError
        OpTensor.__init__(
            self,
            size=x.size[:dim] + x.size[dim+1:],
            prevs={'x': x},
            name=name
        )
        self.val_params['dim'] = dim


class OpMultiply(OpTensor):
    def __init__(self,
                 a: OpTensor,
                 b: OpTensor,
                 name: str = ''):
        assert len(a.size) == len(b.size)
        OpTensor.__init__(
            self,
            size=[max(a, b)
                  for a, b in
                  zip(a.size, b.size)],
            prevs={'a': a, 'b': b},
            name=name
        )


class OpLinear(OpTensor):
    def __init__(self,
                 x: OpTensor,
                 w: nn.Parameter,
                 b: nn.Parameter,
                 name: str = ''):
        OpTensor.__init__(
            self,
            size=[x.size[0], w.size(0)],
            prevs={'x': x},
            name=name
        )
        self.ref_params = {'weight': w, 'bias': b}


class OpDropout(OpTensor):
    def __init__(self,
                 x: OpTensor,
                 p: float,
                 name: str = ''):
        OpTensor.__init__(
            self,
            size=x.size,
            prevs={'x': x},
            name=name
        )
        self.val_params = {'p': p}


class OpFusedSPMM(OpTensor):
    def __init__(self,
                 graph: OpGraph,
                 edge: OpTensor,
                 x: OpTensor,
                 name: str = ''):
        assert len(x.size) == 3
        assert graph.size[1] == x.size[0]
        if edge is None:
            prevs = {'g': graph, 'x': x}
        else:
            assert len(edge.size) == 2
            prevs = {'g': graph, 'e': edge, 'x': x}
        OpTensor.__init__(
            self,
            size=[graph.size[0]] + x.size[1:],
            prevs=prevs,
            name=name
        )


class OpFusedSDDMM(OpGraph):
    def __init__(self,
                 size: list,
                 graph: OpGraph,
                 query: OpTensor,
                 key: OpTensor,
                 fusion_scheme: str,
                 name: str = ''):
        assert len(graph.size) == 2
        assert len(query.size) in [2, 3]
        assert len(key.size) == 2
        assert query.size[-1] == key.size[-1]
        assert graph.size[0] == query.size[-2]
        assert graph.size[1] == key.size[0]
        OpTensor.__init__(
            self,
            size=size,
            prevs={'g': graph, 'q': query, 'k': key},
            name=name
        )
        self.fusion_scheme = fusion_scheme


class OpELU(OpTensor):
    def __init__(self,
                 x: OpTensor,
                 name: str = ''):
        OpTensor.__init__(
            self,
            size=x.size,
            prevs={'x': x},
            name=name
        )


class OpRelu(OpTensor):
    def __init__(self,
                 x: OpTensor,
                 name: str = ''):
        OpTensor.__init__(
            self,
            size=x.size,
            prevs={'x': x},
            name=name
        )


class OpLeakyRelu(OpTensor):
    def __init__(self,
                 x: OpTensor,
                 name: str = ''):
        OpTensor.__init__(
            self,
            size=x.size,
            prevs={'x': x},
            name=name
        )


class OpVertFunc(OpTensor):
    def __init__(self, size: list, prevs: dict, func_name: str):
        OpTensor.__init__(
            self,
            size=size,
            prevs=prevs
        )
        self.func_name = func_name


class OpEdgeFunc(OpTensor):
    def __init__(self, size: list, prevs: dict, func_name: str):
        OpTensor.__init__(
            self,
            size=size,
            prevs=prevs
        )
        self.func_name = func_name
