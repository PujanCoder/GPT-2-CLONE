from tokenizer import GPT2Tokenizer

tokenizer = GPT2Tokenizer(
    "tokenizer/vocab.json",
    "tokenizer/merges.txt"
)

text = "GPT-2 learns to predict the next token."

# Encode
token_ids = tokenizer.encode(text)

# Decode
decoded_text = tokenizer.decode(token_ids)

print("Original text:")
print(text)

print("\nToken IDs:")
print(token_ids)

print("\nNumber of tokens:")
print(len(token_ids))

print("\nDecoded text:")
print(decoded_text)