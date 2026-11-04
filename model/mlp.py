import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(
        self,
        embedding_dim=768,
        hidden_dim=3072,
        dropout=0.1
    ):

        super().__init__()

        self.fc1 = nn.Linear(
            embedding_dim,
            hidden_dim
        )

        self.activation = nn.GELU(approximate="tanh")

        self.fc2 = nn.Linear(
            hidden_dim,
            embedding_dim
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = self.fc1(x)

        x = self.activation(x)

        x = self.fc2(x)

        x = self.dropout(x)

        return x