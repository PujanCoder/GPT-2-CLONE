
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim):

        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

    def forward(self, input_ids):

        return self.embedding(input_ids)

class PositionalEmbedding(nn.Module):

    def __init__(self, context_length, embedding_dim):

        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=context_length,
            embedding_dim=embedding_dim
        )

    def forward(self, input_ids):

        batch_size, sequence_length = input_ids.shape

        positions = torch.arange(
            sequence_length,
            device=input_ids.device
        )

        return self.embedding(positions)

class GPTEmbedding(nn.Module):

    def __init__(
        self,
        vocab_size,
        context_length,
        embedding_dim
    ):

        super().__init__()

        self.token_embedding = TokenEmbedding(
            vocab_size,
            embedding_dim
        )

        self.position_embedding = PositionalEmbedding(
            context_length,
            embedding_dim
        )

    def forward(self, input_ids):

        token_embeds = self.token_embedding(input_ids)

        position_embeds = self.position_embedding(input_ids)

        return token_embeds + position_embeds
