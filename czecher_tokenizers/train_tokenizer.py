## COURTESY OF ANDREJ KARPATHY - NANOCHAT

"""
Train a tokenizer using the HuggingFace Tokenizers library.
In the style of GPT-4 tokenizer.
"""
import os
from pathlib import Path
import time
import argparse
import torch
from czecher_tokenizers.hf_tokenizer import HuggingFaceTokenizer

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
parser.add_argument('--max_chars', type=int, default=10_000_000_000, help='Maximum characters to train on (default: 10B)')
parser.add_argument('--doc_cap', type=int, default=10_000, help='Maximum characters per document (default: 10,000)')
parser.add_argument('--vocab_size', type=int, default=65536, help='Vocabulary size (default: 65536 = 2^16)')
args = parser.parse_args()
print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")

# Text iterator

def load_sentences(path="sentences.txt"):
    return [line.rstrip("\n") for line in Path(path).read_text(encoding="utf-8").splitlines()]

def text_iterator(sentence_path: str):
    """
    1) Flatten the batches into a single iterator
    2) Crop every document to args.doc_cap characters
    3) Break when we've seen args.max_chars characters
    """

    sentences = load_sentences(path=sentence_path)

    nchars = 0
    for text in sentences:
        nchars += len(text)
        yield text
        if nchars > args.max_chars:
                return

text_iter = text_iterator('data/sentences_5M.txt')

# -----------------------------------------------------------------------------
# Train the tokenizer
t0 = time.time()
tokenizer = HuggingFaceTokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s")

# -----------------------------------------------------------------------------
# Save the tokenizer to disk
tokenizer_dir = 'czecher_tokenizers'
tokenizer.save(tokenizer_dir, 'tokenizer.json')

# -----------------------------------------------------------------------------
# Quick inline sanity check
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text

# -----------------------------------------------------------------------------
# One more thing: we wish to cache a mapping from token id to number of bytes of that token
# for efficient evaluation of bits per byte. Unlike the typical mean loss, this
# allows us to report a loss that is invariant to the vocab size of the tokenizer.
# The bits per byte on the validation set is then one of the primary metrics we care about.
# vocab_size = tokenizer.get_vocab_size()
# special_set = set(tokenizer.get_special_tokens())
# token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
# token_bytes = []
# for token_id in range(vocab_size):
#     token_str = token_strings[token_id] # the Python string representation of this token
#     if token_str in special_set:
#         token_bytes.append(0) # special characters are not counted
#     else:
#         id_bytes = len(token_str.encode("utf-8")) # number of bytes that make up this token
#         token_bytes.append(id_bytes)
# token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
# token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
# with open(token_bytes_path, "wb") as f:
#     torch.save(token_bytes, f)
# print(f"Saved token_bytes to {token_bytes_path}")
