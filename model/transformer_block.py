import torch
import torch.nn as nn

from model.attention import CausalSelfAttention
from model.mlp import MLP
from model.layer_norm import LayerNormalization


class TransformerBlock(nn.Module):

    def __init__(self, config):

        super().__init__()

        # First Layer Normalization
        self.ln_1 = LayerNormalization(
            embedding_dim=config.embedding_dim
        )

        # Causal Self Attention
        self.attention = CausalSelfAttention(
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            context_length=config.context_length,
            dropout=config.dropout
        )

        # Second Layer Normalization
        self.ln_2 = LayerNormalization(
            embedding_dim=config.embedding_dim
        )

        # Feed Forward Network
        self.mlp = MLP(
            embedding_dim=config.embedding_dim,
            hidden_dim=config.embedding_dim * 4,
            dropout=config.dropout
        )

    def forward(self, x):

        # Attention sublayer
        residual = x

        x = self.ln_1(x)

        x = self.attention(x)

        x = residual + x

        # MLP sublayer
        residual = x

        x = self.ln_2(x)

        x = self.mlp(x)

        x = residual + x

        return x