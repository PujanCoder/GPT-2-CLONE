
import torch
from types import SimpleNamespace

from model.transformer_block import TransformerBlock
def get_config():
    return SimpleNamespace(
        embedding_dim=768,
        num_heads=12,
        context_length=128,
        dropout=0.1,
        use_bias=True
    )

def test_transformer_output_shape():
    config = get_config()
    model = TransformerBlock(config)

    x = torch.randn(2, 10, 768)

    output = model(x)

    assert output.shape == (2, 10, 768)

def test_transformer_finite_output():
    config = get_config()
    model = TransformerBlock(config)

    x = torch.randn(2, 10, 768)

    output = model(x)

    assert torch.isfinite(output).all()

def test_transformer_backward():
    config = get_config()
    model = TransformerBlock(config)

    x = torch.randn(2, 10, 768, requires_grad=True)

    output = model(x)

    loss = output.mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()