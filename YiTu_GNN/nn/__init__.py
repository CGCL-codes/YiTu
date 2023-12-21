import YiTu_GNN.distributed as dist

if dist.in_distributed_mode():
    from .GCNConv import GraphConv as GCNConv
else:
    from .GCNConv import GCNConv
from .GINConv import GINConv
from .SAGEConv import SAGEConv

__all__ = ["GCNConv", "GINConv", "SAGEConv"]
