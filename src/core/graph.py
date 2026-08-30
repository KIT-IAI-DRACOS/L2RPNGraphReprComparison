"""
Graph utility functions used across the RL pipeline.
"""

from typing import Union, Dict, Tuple

import torch
from torch import Tensor
from torch_geometric.utils import dense_to_sparse

# Cache for fully_connected_edge_index — keyed by (num_nodes, self_loops).
# The tensor is always built on CPU and moved to the target device on each call,
# so the cache stores only the CPU copy.
_FC_EDGE_INDEX_CACHE: Dict[Tuple[int, bool], Tensor] = {}


def fully_connected_edge_index(
    num_nodes: int,
    device: Union[str, torch.device] = "cpu",
    self_loops: bool = False,
) -> Tensor:
    """
    Edge index for a fully-connected directed graph with *num_nodes* nodes.

    The CPU result is cached after the first call for a given ``(num_nodes,
    self_loops)`` pair, so repeated calls (e.g. once per training step) are
    essentially free.

    :param num_nodes: Number of nodes.
    :param device: Target device.
    :param self_loops: Include self-loops if True.
    :return: Edge index [2, E].
    """
    key = (num_nodes, self_loops)
    if key not in _FC_EDGE_INDEX_CACHE:
        senders, receivers = torch.meshgrid(
            torch.arange(num_nodes), torch.arange(num_nodes), indexing="ij"
        )
        edge_index = torch.stack([senders.flatten(), receivers.flatten()], dim=0)
        if not self_loops:
            edge_index = edge_index[:, edge_index[0] != edge_index[1]]
        _FC_EDGE_INDEX_CACHE[key] = edge_index  # stored on CPU
    return _FC_EDGE_INDEX_CACHE[key].to(device=device)


def fully_connected_edge_index_per_batch(
    batch: Tensor,
    device: Union[str, torch.device] = "cpu",
    self_loops: bool = False,
) -> Tensor:
    """
    Fully-connected edge index for a *batch* of graphs.

    :param batch: Graph-membership vector per node [N].
    :param device: Target device.
    :param self_loops: Include self-loops if True.
    :return: Edge index [2, sum_g E_g].
    """
    num_graphs = int(batch.max()) + 1
    edge_indices = []
    for g in range(num_graphs):
        node_idx = (batch == g).nonzero(as_tuple=False).view(-1)
        n = node_idx.numel()
        if n == 0:
            continue
        adj = torch.ones((n, n), dtype=torch.bool, device=device)
        if not self_loops:
            adj.fill_diagonal_(False)
        edge_index_local, _ = dense_to_sparse(adj)
        edge_indices.append(node_idx[edge_index_local])
    return torch.cat(edge_indices, dim=1)


def edge_membership_mask(
    super_edge_set: Tensor,
    sub_edge_set: Tensor,
) -> Tensor:
    """
    Boolean mask: which edges in *super_edge_set* also appear in *sub_edge_set*.

    :param super_edge_set: [2, E_super].
    :param sub_edge_set: [2, E_sub].
    :return: Bool mask [E_super].
    """
    num_nodes = super_edge_set.max() + 1
    a = super_edge_set[0] * num_nodes + super_edge_set[1]
    b = sub_edge_set[0] * num_nodes + sub_edge_set[1]
    return torch.isin(a, b)
