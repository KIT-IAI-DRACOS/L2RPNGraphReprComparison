"""Two-layer ELU MLP"""

import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor


class MLP(nn.Module):
    """
    Two-layer fully-connected ELU network with optional batch normalization.

    Architecture: Linear → ELU → Dropout → Linear → ELU → [BatchNorm]

    :param input_features: Input dimensionality.
    :param hidden_dim: Hidden layer dimensionality.
    :param output_features: Output dimensionality.
    :param dropout_prob: Dropout probability (applied between the two layers).
    :param do_batch_norm: Apply batch normalization on the output if True.
    """

    def __init__(
        self,
        input_features: int,
        hidden_dim: int,
        output_features: int,
        dropout_prob: float = 0.0,
        do_batch_norm: bool = True,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_features)
        self.bn = nn.BatchNorm1d(output_features)
        self.dropout_prob = dropout_prob
        self.do_batch_norm = do_batch_norm
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _apply_bn(self, x: Tensor) -> Tensor:
        # BatchNorm1d expects 2-D input; flatten batch dims, normalize, reshape.
        batch_shape = x.shape[:-1]
        flat = x.view(int(np.prod(batch_shape)), -1)
        return self.bn(flat).view(x.shape)

    def forward(self, x: Tensor) -> Tensor:
        x = F.elu(self.fc1(x))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self._apply_bn(x) if self.do_batch_norm else x
