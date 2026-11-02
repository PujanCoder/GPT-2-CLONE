import torch
import torch.nn as nn

from model.attention import CausalSelfAttention
from model.mlp import MLP


class TransformerBlock(nn.Module):

    def __init__(self, config):

        super().__init__()

        # First Layer Normalization
        self.ln1 = nn.LayerNorm(
            config.embedding_dim,
            eps=1e-5
        )

        # Self Attention
        self.attention = CausalSelfAttention(config)

        # Second Layer Normalization
        self.ln2 = nn.LayerNorm(
            config.embedding_dim,
            eps=1e-5
        )

        # Feed Forward Network
        self.mlp = MLP(config)

    def forward(self, x):

        # Attention sublayer + residual connection
        x = x + self.attention(self.ln1(x))

        # MLP sublayer + residual connection
        x = x + self.mlp(self.ln2(x))

        return x