import os

from .partition import partition_graph
from .stage_module import StageModule

__all__=["partition_graph", "StageModule"]

def in_distributed_mode():
    if "DGL_DIST_MODE" in os.environ:
        return True
    return False
