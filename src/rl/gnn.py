"""
GNN implementation.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import nn, Tensor
from torch_geometric.nn import GCNConv, GATConv, BatchNorm, global_mean_pool

from .mlp import MLP

ConvType = Literal["gcn", "gin", "gat"]


def _mean_l2_norm(t: Tensor, eps: float = 1e-12) -> Tensor:
    """Mean per-node L2 norm of feature vectors."""
    return torch.linalg.vector_norm(t, dim=-1).mean().clamp_min(eps)


class GNN(nn.Module):
    """
    Standard GNN without edge-type conditioning — baseline for comparison.

    Uses a single conv layer per message-passing step applied to all edges
    equally. Supports two convolution kernels via *conv_type*:

    * ``"gcn"`` (default) — GCNConv with ``improved=True``. Edge attributes
      are projected to scalar edge weights via a small MLP.
    * ``"gat"`` — GATConv with multi-head attention (``concat=False`` so
      output dim stays *hidden_dim*). Edge attributes are fed directly into
      the attention mechanism; no MLP projection is required.

    Does not require edge-type posteriors from a separate encoder.

    :param x_dim: Node feature dimensionality.
    :param hidden_dim: Hidden node dimensionality.
    :param x_out_dim: Output graph-level embedding dimensionality.
    :param num_layers: Number of message-passing layers.
    :param num_edge_types: Number of edge types (default 1). When > 1 edges
        must be annotated with integer type indices in the forward call.
    :param dropout_prob: Dropout probability.
    :param edge_dim: Dimensionality of edge attributes. Auto-detected from the
        observation space by the RLlib model wrapper; pass ``None`` to disable
        edge feature support. For GCN a projection MLP maps
        ``edge_attr [E, edge_dim] → scalar [E]``; for GAT edge attributes are
        passed directly into the attention kernel.
    :param residual: If ``True``, h = h + h_new each layer; else h = h_new.
    :param conv_type: Convolution kernel — ``"gcn"`` or ``"gat"``.
    :param num_heads: Number of attention heads (GAT only; ignored for GCN).
    """

    def __init__(
            self,
            x_dim: int,
            hidden_dim: int,
            x_out_dim: int,
            num_layers: int = 3,
            num_edge_types: int = 1,
            dropout_prob: float = 0.0,
            edge_dim: Optional[int] = None,
            residual: bool = True,
            conv_type: ConvType = "gcn",
            num_heads: int = 4,
    ):
        super().__init__()
        self.residual = residual
        self.edge_dim = edge_dim
        self.conv_type = conv_type

        self.node_proj = MLP(x_dim, hidden_dim, hidden_dim, dropout_prob=dropout_prob, do_batch_norm=False)
        self.bn_node_proj = BatchNorm(hidden_dim)

        if conv_type == "gcn":
            if edge_dim is not None:
                self.edge_proj = MLP(edge_dim, hidden_dim, 1, dropout_prob=dropout_prob, do_batch_norm=False)
            self.layers = nn.ModuleList([nn.ModuleList([
                GCNConv(hidden_dim, hidden_dim, improved=True, add_self_loops=True)
                for _ in range(num_edge_types)])
                for _ in range(num_layers)])
        elif conv_type == "gat":
            # GAT incorporates edge_attr natively via the attention kernel —
            # no MLP projection needed.
            self.layers = nn.ModuleList([nn.ModuleList([
                GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False,
                        edge_dim=edge_dim, add_self_loops=True)
                for _ in range(num_edge_types)])
                for _ in range(num_layers)])
        else:
            raise ValueError(f"Unknown conv_type: {conv_type!r}. Expected 'gcn' or 'gat'.")

        self.bn_mp = nn.ModuleList([BatchNorm(hidden_dim) for _ in range(num_layers)])
        self.activation_function = nn.ELU()
        self.dropout = nn.Dropout(dropout_prob)

        self.final = MLP(hidden_dim, hidden_dim, x_out_dim, dropout_prob=dropout_prob, do_batch_norm=False)
        self.stats: dict = {}

    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            batch: Tensor,
            edge_types: Optional[Tensor] = None,
            edge_weights: Optional[Tensor] = None,
            edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """
        :param x: Node features [N, x_dim].
        :param edge_index: Graph connectivity [2, E].
        :param batch: Batch vector [N].
        :param edge_types: Integer edge-type index per edge [E]. Defaults to
            all-zero (single type) when not provided.
        :param edge_weights: Scalar edge weights [E]. GCN only; mutually
            exclusive with *edge_attr*.
        :param edge_attr: Edge attribute matrix [E, edge_dim]. For GCN,
            projected to scalar weights; for GAT, fed into attention directly.
        :return: Graph-level embeddings [B, x_out_dim].
        """
        assert edge_weights is None or edge_attr is None, \
            "edge_weights and edge_attr are mutually exclusive"

        if edge_types is None:
            edge_types = torch.zeros(edge_index.shape[1], device=x.device, dtype=torch.long)

        # GCN: project edge attributes to scalar weights once before the loop.
        if self.conv_type == "gcn" and edge_attr is not None and edge_weights is None:
            assert self.edge_dim is not None, "edge_dim must be set at construction to use edge_attr"
            edge_weights = self.edge_proj(edge_attr).squeeze(-1)

        h = self.bn_node_proj(self.node_proj(x))

        for i, layer in enumerate(self.layers):
            h_new = torch.zeros_like(h)
            for j, edge_type_conv in enumerate(layer):
                mask = edge_types == j
                ei_masked = edge_index[:, mask]
                if self.conv_type == "gat":
                    ea_masked = edge_attr[mask] if edge_attr is not None else None
                    h_new += edge_type_conv(x=h, edge_index=ei_masked, edge_attr=ea_masked)
                else:
                    h_new += edge_type_conv(
                        x=h,
                        edge_index=ei_masked,
                        edge_weight=edge_weights[mask] if edge_weights is not None else None,
                    )

            h_new = self.bn_mp[i](h_new)
            h_new = self.activation_function(h_new)
            self.stats[f"msg_ratio_layer_{i}"] = _mean_l2_norm(h_new) / _mean_l2_norm(h)
            h = self.dropout(h)
            h = h + h_new if self.residual else h_new

        return global_mean_pool(self.final(h), batch)
