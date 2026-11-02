
import torch
import torch.nn as nn
import math


class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        context_length,
        dropout=0.1
    ):

        super().__init__()

        assert embedding_dim % num_heads == 0, (
            "Embedding dimension must be divisible by number of heads"
        )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Query, Key and Value projections
        self.qkv = nn.Linear(
            embedding_dim,
            3 * embedding_dim
        )

        # Output projection
        self.out_proj = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        # Dropout layers
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

        # Causal mask
        mask = torch.tril(
            torch.ones(context_length, context_length)
        )

        self.register_buffer(
            "mask",
            mask.view(1, 1, context_length, context_length),
            persistent=False
        )

    def forward(self, x):

        batch_size, sequence_length, embedding_dim = x.shape

        # 1. Generate Query, Key and Value
        qkv = self.qkv(x)

        # 2. Separate Q, K and V
        q, k, v = qkv.chunk(3, dim=-1)

        # 3. Split into multiple attention heads
        q = q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        k = k.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        v = v.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        # 4. Move heads before sequence dimension
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Shape: [batch, heads, sequence, head_dim]

        # 5. Calculate attention scores
        scores = q @ k.transpose(-2, -1)

        # 6. Scale attention scores
        scores = scores / math.sqrt(self.head_dim)

        # 7. Apply causal mask
        causal_mask = self.mask[
            :, :, :sequence_length, :sequence_length
        ]

        scores = scores.masked_fill(
            causal_mask == 0,
            float("-inf")
        )

        # 8. Convert scores into probabilities
        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        # 9. Apply attention dropout
        attention_weights = self.attention_dropout(
            attention_weights
        )

        # 10. Multiply attention weights by values
        output = attention_weights @ v

        # 11. Combine attention heads
        output = output.transpose(1, 2).contiguous()

        output = output.view(
            batch_size,
            sequence_length,
            self.embedding_dim
        )

        # 12. Final output projection
        output = self.out_proj(output)

        output = self.residual_dropout(output)

        return output
