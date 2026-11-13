"""
Graph readouts: node embeddings of a batch of graphs → one vector per graph.

* ``"mean"`` — :func:`torch_geometric.nn.global_mean_pool`: one average over all nodes, size d.
  Loses where in the grid something happens.
* ``"substation"`` — :func:`substation_readout`: one average per substation, concatenated, size
  n_sub × d. Keeps the location at the level where topology actions act, and has the same size
  for every graph representation of a grid.
"""
from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

ReadoutType = Literal["mean", "substation"]


def substation_readout(h: Tensor, batch: Tensor, node_substations: Tensor, num_graphs: int, n_sub: int) -> Tensor:
    """Mean node embedding per substation, concatenated in substation order per graph.

    A node counts fully for each substation it belongs to (line nodes: both ends). Substations
    without nodes get zeros.

    Args:
        h: Node embeddings [num_nodes, d].
        batch: Graph index of every node [num_nodes].
        node_substations: Substation(s) of every node [num_nodes, 2], -1 = none.
        num_graphs: Graphs in the batch (B).
        n_sub: Substations per graph.

    Returns:
        [B, n_sub * d]; columns d*s ... d*(s+1)-1 hold substation s.
    """
    node_idx, column = torch.nonzero(node_substations >= 0, as_tuple=True)
    group = batch[node_idx] * n_sub + node_substations[node_idx, column].long()
    sums = h.new_zeros(num_graphs * n_sub, h.shape[1]).index_add_(0, group, h[node_idx])
    counts = torch.bincount(group, minlength=num_graphs * n_sub).clamp(min=1).to(h.dtype)
    return (sums / counts.unsqueeze(1)).reshape(num_graphs, n_sub * h.shape[1])
