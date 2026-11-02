import torch
import torch.nn as nn
import math


class MLP(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.fc1 = nn.Linear(
            config.embedding_dim,
            config.embedding_dim * 4,
            bias=config.use_bias
        )

        self.activation = nn.GELU(approximate="tanh")

        self.fc2 = nn.Linear(
            config.embedding_dim * 4,
            config.embedding_dim,
            bias=config.use_bias
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):

        x = self.fc1(x)

        x = self.activation(x)

        x = self.fc2(x)

        x = self.dropout(x)

        return x