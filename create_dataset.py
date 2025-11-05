import os
import math
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from typing import Tuple, List

from czecher_tokenizers.bpe_tokenizer import GPTTokenizer
from czecher_tokenizers.tokenizer import Tokenizer

COMMA = ','


def create_tokenizer() -> Tokenizer:
    # e.g. return Tokenizer("path/to/model")
    return GPTTokenizer(json_file='tokenizer.json')


def get_data_pair(sentence: str, tokenizer: Tokenizer, max_tokens: int) -> Tuple[List[int], List[int]]:
    formatted = sentence.replace(COMMA, '')
    in_tokens = tokenizer.tokenize(formatted, max_tokens=max_tokens)

    probabilities = [0] * max_tokens
    commas_found = 0
    chars_decoded = 0

    # Walk token-by-token; detokenize piece-by-piece
    for idx, token_id in enumerate(in_tokens):
        token = tokenizer.detokenize([token_id])   # assumed to return piece without special tokens
        chars_decoded += len(token)
        if chars_decoded + commas_found >= len(sentence):
            break
        original_next_char = sentence[chars_decoded + commas_found]
        if original_next_char == COMMA:
            probabilities[idx] = 1
            commas_found += 1

    return in_tokens, probabilities


def _worker_process(lines: List[str], max_tokens: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    One worker processes a chunk of lines into two numpy arrays of shape (len(lines), max_tokens).
    Initializes its own tokenizer (fork-safe).
    """
    tok = create_tokenizer()
    n = len(lines)
    X = np.zeros((n, max_tokens), dtype=np.uint32)  # safe default; will be cast later
    Y = np.zeros((n, max_tokens), dtype=np.uint8)

    for i, s in enumerate(lines):
        s = s.rstrip('\n')
        if not s:
            continue
        ids, probs = get_data_pair(s, tok, max_tokens)
        X[i, :] = ids
        Y[i, :] = probs
    return X, Y


def count_lines(txt_path: str) -> int:
    cnt = 0
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        for _ in f:
            cnt += 1
    return cnt


def preprocess_to_memmap(
    txt_path: str,
    out_dir: str,
    max_tokens: int = 128,
    batch_lines: int = 200_000,   # tune for RAM/CPU
    num_proc: int = 0,            # 0/1 = single-process; >1 uses multiprocessing
    dtype_inputs: np.dtype = None # None -> infer from tokenizer vocab in a small probe
):
    os.makedirs(out_dir, exist_ok=True)

    # Pass 1: count lines (N)
    print("Counting lines...")
    N = count_lines(txt_path)
    print(f"Found {N:,} sentences.")

    # Probe tokenizer vocab to set dtype if not provided
    if dtype_inputs is None:
        tok = create_tokenizer()
        if tok.vocab_size() <= 65535:
            dtype_inputs = np.uint16
        else:
            dtype_inputs = np.uint32  # safe for any vocab
        del tok

    # Allocate memmaps
    inputs_path = os.path.join(out_dir, "inputs.bin")
    labels_path = os.path.join(out_dir, "labels.bin")

    Xmm = np.memmap(inputs_path, dtype=dtype_inputs, mode='w+', shape=(N, max_tokens))
    Ymm = np.memmap(labels_path, dtype=np.uint8,    mode='w+', shape=(N, max_tokens))

    # Stream the file in batches
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as fin:
        total_batches = math.ceil(N / batch_lines)
        idx = 0

        if num_proc and num_proc > 1:
            # Multiprocessing path: read batches → scatter to workers → gather results
            for _ in tqdm(range(total_batches), desc="Preprocessing (mp)"):
                lines = []
                for _ in range(min(batch_lines, N - idx)):
                    line = fin.readline()
                    if not line:
                        break
                    lines.append(line)

                if not lines:
                    break

                # Split lines into roughly equal shards for workers
                shard_size = max(1, len(lines) // num_proc)
                shards = [lines[i:i + shard_size] for i in range(0, len(lines), shard_size)]

                with Pool(processes=num_proc) as pool:
                    results = pool.map(partial(_worker_process, max_tokens=max_tokens), shards)

                # Concatenate worker outputs in original order
                Xb = np.vstack([r[0] for r in results])
                Yb = np.vstack([r[1] for r in results])

                # Cast inputs to desired dtype once here
                if Xb.dtype != dtype_inputs:
                    Xb = Xb.astype(dtype_inputs, copy=False)

                end = idx + Xb.shape[0]
                Xmm[idx:end, :] = Xb
                Ymm[idx:end, :] = Yb
                idx = end
        else:
            # Single-process path: minimal overhead
            tok = create_tokenizer()
            for _ in tqdm(range(total_batches), desc="Preprocessing"):
                lines = []
                for _ in range(min(batch_lines, N - idx)):
                    line = fin.readline()
                    if not line:
                        break
                    lines.append(line)

                if not lines:
                    break

                n = len(lines)
                Xb = np.zeros((n, max_tokens), dtype=dtype_inputs)
                Yb = np.zeros((n, max_tokens), dtype=np.uint8)
                for i, s in enumerate(lines):
                    s = s.rstrip('\n')
                    if not s:
                        continue
                    ids, probs = get_data_pair(s, tok, max_tokens)
                    # cast in place
                    if dtype_inputs == np.uint16:
                        ids = [int(x) & 0xFFFF for x in ids]
                    Xb[i, :] = ids
                    Yb[i, :] = probs

                end = idx + n
                Xmm[idx:end, :] = Xb
                Ymm[idx:end, :] = Yb
                idx = end

    # Flush to disk
    Xmm.flush()
    Ymm.flush()
    print(f"Wrote:\n  {inputs_path}\n  {labels_path}\nShapes: ({N}, {max_tokens}) each (inputs dtype {dtype_inputs}, labels uint8)")


if __name__ == "__main__":
    preprocess_to_memmap(
        txt_path="data/sentences_5m.txt",
        out_dir="./comma_memmap",
        max_tokens=128,
        batch_lines=100_000,        # ~200k lines per chunk; adjust to your RAM/CPU
        num_proc=8                  # try 0/1 first; if tokenizer is process-safe, use CPU cores
    )
