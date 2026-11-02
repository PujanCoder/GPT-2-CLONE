
import torch

from model.embeddings import GPTEmbedding


def test_gpt_embedding():

    vocab_size = 50257
    context_length = 128
    embedding_dim = 768

    model = GPTEmbedding(
        vocab_size,
        context_length,
        embedding_dim
    )

    input_ids = torch.randint(
        0,
        vocab_size,
        (2, 10)
    )

    output = model(input_ids)

    assert output.shape == (2, 10, 768)
    assert torch.isfinite(output).all()
