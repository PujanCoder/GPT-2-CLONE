
import json
import regex

from tokenizer.byte_encoder import bytes_to_unicode
from tokenizer.bpe import BytePairEncoder


class GPT2Tokenizer:
    def __init__(self, vocab_file, merges_file):
        with open(vocab_file, "r", encoding="utf-8") as f:
            self.encoder = json.load(f)

        self.decoder = {
            token_id: token
            for token, token_id in self.encoder.items()
        }

        with open(merges_file, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

        # Ignore the version header in merges.txt.
        if lines and lines[0].startswith("#"):
            lines = lines[1:]

        merges = [
            tuple(line.split())
            for line in lines
            if line.strip()
        ]

        self.byte_encoder, self.byte_decoder = bytes_to_unicode()
        self.bpe = BytePairEncoder(merges)

        # GPT-2's original pre-tokenization pattern.
        self.pattern = regex.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        self.eot_token = "<|endoftext|>"

    def encode(self, text):
        """Convert text into GPT-2 token IDs."""
        token_ids = []

        for piece in regex.findall(self.pattern, text):
            # Convert the piece into UTF-8 bytes, then map each byte
            # to the reversible Unicode representation used by GPT-2.
            encoded_bytes = piece.encode("utf-8")

            encoded_text = "".join(
                self.byte_encoder[byte]
                for byte in encoded_bytes
            )

            # Apply BPE and convert each resulting token to its ID.
            bpe_tokens = self.bpe.encode(encoded_text)

            for token in bpe_tokens:
                token_ids.append(self.encoder[token])

        return token_ids

    def decode(self, token_ids):
        """Convert GPT-2 token IDs back into text."""
        encoded_text = "".join(
            self.decoder[token_id]
            for token_id in token_ids
        )

        # Convert the byte-level Unicode representation back to bytes.
        decoded_bytes = bytearray(
            self.byte_decoder[character]
            for character in encoded_text
        )

        return decoded_bytes.decode("utf-8", errors="replace")
