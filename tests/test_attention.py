
import torch

from model.attention import CausalSelfAttention


def test_attention_output_shape():

    model = CausalSelfAttention(
        embedding_dim=768,
        num_heads=12,
        context_length=128,
        dropout=0.1
    )

    x = torch.randn(2, 10, 768)

    output = model(x)

    assert output.shape == (2, 10, 768)


def test_attention_no_future_information():

    torch.manual_seed(42)

    model = CausalSelfAttention(
        embedding_dim=768,
        num_heads=12,
        context_length=128,
        dropout=0.0
    )

    model.eval()

    x1 = torch.randn(1, 10, 768)

    x2 = x1.clone()

    # Change future tokens
    x2[:, 5:, :] = torch.randn(1, 5, 768)

    with torch.no_grad():

        output1 = model(x1)
        output2 = model(x2)

    # Earlier positions must remain unchanged
    assert torch.allclose(
        output1[:, :5, :],
        output2[:, :5, :],
        atol=1e-6
    )


def test_attention_invalid_heads():

    try:

        CausalSelfAttention(
            embedding_dim=768,
            num_heads=10,
            context_length=128
        )

    except AssertionError:

        return

    assert False, "Expected assertion error"


def test_attention_finite_output():

    model = CausalSelfAttention(
        embedding_dim=768,
        num_heads=12,
        context_length=128
    )

    x = torch.randn(2, 10, 768)

    output = model(x)

    assert torch.isfinite(output).all()
