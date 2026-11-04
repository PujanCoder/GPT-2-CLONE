import torch
import torch.nn as nn

from model.embeddings import GPTEmbedding
from model.transformer_block import TransformerBlock
from model.block import LayerNormalization


class GPT2(nn.Module):

    def __init__(
        self,
        vocab_size=50257,
        context_length=1024,
        embedding_dim=768,
        num_heads=12,
        num_layers=12,
        dropout=0.1,
        mlp_hidden_dim=3072
    ):

        super().__init__()

        # Token + Positional Embeddings
        self.embedding = GPTEmbedding(
            vocab_size=vocab_size,
            context_length=context_length,
            embedding_dim=embedding_dim
        )

        self.dropout = nn.Dropout(dropout)

        # Transformer Blocks
        self.transformer_blocks = nn.ModuleList([

            TransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                context_length=context_length,
                dropout=dropout,
                mlp_hidden_dim=mlp_hidden_dim
            )

            for _ in range(num_layers)

        ])

        # Final Layer Normalization
        self.final_norm = LayerNormalization(
            embedding_dim
        )

        # Language Model Head
        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size,
            bias=False
        )

        # Weight tying
        self.lm_head.weight = (
            self.embedding.token_embedding.embedding.weight
        )

        self.context_length = context_length

    def forward(self, input_ids):

        batch_size, sequence_length = input_ids.shape

        if sequence_length > self.context_length:
            raise ValueError(
                "Sequence length exceeds context length"
            )

        # Embeddings
        x = self.embedding(input_ids)

        x = self.dropout(x)

        # Transformer Blocks
        for block in self.transformer_blocks:

            x = block(x)

        # Final normalization
        x = self.final_norm(x)

        # Vocabulary predictions
        logits = self.lm_head(x)

        return logits