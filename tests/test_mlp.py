import torch
from types import SimpleNamespace

from model.mlp import MLP


def test_mlp_output_shape():

    config = SimpleNamespace(
        embedding_dim=768,
        use_bias=True,
        dropout=0.1
    )

    model = MLP(config)

    x = torch.randn(2, 10, 768)

    output = model(x)

    assert output.shape == (2, 10, 768)


def test_mlp_output_is_finite():

    config = SimpleNamespace(
        embedding_dim=768,
        use_bias=True,
        dropout=0.1
    )

    model = MLP(config)

    x = torch.randn(2, 10, 768)

    output = model(x)

    assert torch.isfinite(output).all()