"""
Integrates GNN in rllibs ecosystem
"""

import typing
from typing import List, Tuple

import numpy as np
import torch
from gymnasium.spaces import Dict, Discrete, Box
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import nn, Tensor

from core.utils import getl
from grid2op_env.observation_converter import NODES, EDGE_INDEX, EDGE_MASK, NODE_MASK, EDGE_TYPE, EDGE_WEIGHTS, EDGES
from .gnn import GNN
from .utils import assert_graph_obs_space_and_get_x_dim


class GNNModel(TorchModelV2, nn.Module):
    def __init__(self,
                 obs_space: Dict,
                 action_space: Discrete,
                 num_outputs: int,
                 model_config: ModelConfigDict,
                 name: str,
                 **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        gnn_cfg = kwargs.get('gnn', {})
        self.gnn: GNN = GNN(
            x_dim=assert_graph_obs_space_and_get_x_dim(obs_space),
            hidden_dim=getl(gnn_cfg, 'hidden_dim', 64),
            x_out_dim=getl(gnn_cfg, 'out_dim', 64),
            num_layers=getl(gnn_cfg, 'num_layers', 3),
            dropout_prob=getl(gnn_cfg, 'dropout_prob', 0.0),
            residual=getl(gnn_cfg, 'residual', True),
            num_edge_types=int(obs_space[EDGE_TYPE].high.flat[0]) + 1 if EDGE_TYPE in obs_space.spaces else 1,
            edge_dim=obs_space[EDGES].shape[-1] if EDGES in obs_space.spaces else None,
            conv_type=getl(gnn_cfg, 'conv_type', 'gcn'),
            num_heads=getl(gnn_cfg, 'num_heads', 4),
        )
        gnn_output_space = Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(getl(gnn_cfg, 'out_dim', 64),),
            dtype=np.float32
        )
        self.mlp = FullyConnectedNetwork(
            obs_space=gnn_output_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=model_config,
            name=name + "_fully_connected_network",
        )

    def forward(self, input_dict: typing.Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> Tuple[TensorType, List[TensorType]]:
        node_features_batch = input_dict["obs"][NODES]   # [B, max_N, node_dim]
        edge_index_batch = input_dict["obs"][EDGE_INDEX]  # [B, 2, E_max]
        edge_mask = input_dict["obs"][EDGE_MASK]          # [B, E_max]
        edge_types = input_dict["obs"].get(EDGE_TYPE)
        edge_weights = input_dict["obs"].get(EDGE_WEIGHTS)
        edge_attr = input_dict["obs"].get(EDGES)

        B, max_N, _ = node_features_batch.shape
        device = node_features_batch.device

        # Node mask: [B, max_N] — falls back to all-True when not provided or when
        # all entries are False (RLlib dummy init batch uses zero-filled obs tensors).
        node_mask = input_dict["obs"].get(NODE_MASK, None)
        if node_mask is None or not node_mask.bool().any():
            node_mask = torch.ones(B, max_N, dtype=torch.bool, device=device)
        else:
            node_mask = node_mask.bool()

        # Per-graph real node counts and cumulative global offsets
        n_real = node_mask.sum(dim=1)                               # [B]
        global_offsets = torch.zeros(B, dtype=torch.long, device=device)
        global_offsets[1:] = n_real[:-1].cumsum(0)

        # Flatten only real nodes; build matching batch vector
        x = node_features_batch[node_mask]                          # [sum(n_real), node_dim]
        batch = torch.arange(B, device=device).repeat_interleave(n_real)

        # Remap table: padded node index -> global dense index
        # dense_local[b, p] = 0-based dense index of node p within graph b
        dense_local = node_mask.long().cumsum(dim=1) - 1           # [B, max_N]
        remap = dense_local + global_offsets.unsqueeze(1)           # [B, max_N]

        # Remap edge_index from padded to dense global indices
        ei = edge_index_batch                                        # [B, 2, E_max]
        src_remapped = remap.gather(1, ei[:, 0, :].clamp(0, max_N - 1))  # [B, E_max]
        dst_remapped = remap.gather(1, ei[:, 1, :].clamp(0, max_N - 1))  # [B, E_max]
        ei_remapped = torch.stack([src_remapped, dst_remapped], dim=1)    # [B, 2, E_max]

        # Select valid edges: [2, total_E]
        valid_edges = edge_mask.bool()
        ei_flat = ei_remapped.permute(1, 0, 2)[:, valid_edges]

        # Filter optional per-edge tensors from [B, E_max, ...] to [total_E, ...]
        if edge_attr is not None:
            edge_attr = edge_attr[valid_edges]       # [total_E, e_dim]
        if edge_types is not None:
            edge_types = edge_types[valid_edges]     # [total_E]

        gnn_out: Tensor = self.gnn(
            x=x,
            batch=batch,
            edge_index=ei_flat.to(dtype=torch.long),
            edge_weights=edge_weights,
            edge_attr=edge_attr,
            edge_types=edge_types
        )

        logits, _ = self.mlp({"obs": gnn_out}, state, seq_lens)
        return logits, []

    def value_function(self) -> Tensor:
        return self.mlp.value_function()
