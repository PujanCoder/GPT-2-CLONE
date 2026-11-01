
from tokenizer.tokenizer import GPT2Tokenizer


def test_gpt2_tokenizer_roundtrip():
    tokenizer = GPT2Tokenizer(
        "vocab.json",
        "merges.txt",
    )

    text = "Hello world!"

    token_ids = tokenizer.encode(text)
    decoded_text = tokenizer.decode(token_ids)

    assert decoded_text == text
    assert len(token_ids) > 0


def test_gpt2_known_token_ids():
    tokenizer = GPT2Tokenizer(
        "vocab.json",
        "merges.txt",
    )

    # Original GPT-2 vocabulary token IDs.
    assert tokenizer.encode("Hello") == [15496]
    assert tokenizer.encode(" world") == [995]
