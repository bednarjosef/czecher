# gpt2_tokenizer.py
from typing import List, Dict, Optional
import unicodedata, csv, os
from transformers import AutoTokenizer

PAD = "[PAD]"
BOS = "[BOS]"
EOS = "[EOS]"
UNK = "[UNK]"

class GPTTokenizer:
    def __init__(self, norm_form: str = "NFC", add_prefix_space: bool = True):
        self.norm_form = norm_form
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=True, add_prefix_space=add_prefix_space)

        special_tokens = {
            "pad_token": PAD,
            "bos_token": BOS,
            "eos_token": EOS,
            "unk_token": UNK,
        }
        self._ensure_specials(special_tokens)

        self.vocab: Dict[str, int] = self.tokenizer.get_vocab()
        self.inv_vocab: Dict[int, str] = {i: tok for tok, i in self.vocab.items()}

    def _ensure_specials(self, specials: Dict[str, str]):
        add_list = []
        for key, tok in specials.items():
            cur = getattr(self.tokenizer, key, None)
            if cur != tok:
                add_list.append(tok)
                setattr(self.tokenizer, key, tok)
        if add_list:
            self.tokenizer.add_special_tokens({"additional_special_tokens": add_list})
        self.tokenizer.pad_token = specials["pad_token"]
        self.tokenizer.bos_token = specials["bos_token"]
        self.tokenizer.eos_token = specials["eos_token"]
        self.tokenizer.unk_token = specials["unk_token"]

    # ---------- CharTokenizer-compatible API ----------
    def save_vocabulary(self, csv_path: str):
        """
        Saves a CSV 'id,token' for compatibility AND saves full HF files
        (vocab/merges/tokenizer.json) in the same directory for proper reload.
        """
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        # 1) CSV (compat)
        print(f"Saving vocabulary CSV to {csv_path}...")
        vocab = self.tokenizer.get_vocab()  # token -> id
        inv = {i: tok for tok, i in vocab.items()}
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["id", "token"])
            for i in range(len(inv)):
                w.writerow([i, inv[i]])
        print("CSV saved.")

        # 2) Full tokenizer files for real loading later
        save_dir = os.path.dirname(csv_path) or "."
        print(f"Saving full tokenizer to {save_dir}...")
        self.tokenizer.save_pretrained(save_dir)
        print("Tokenizer files saved.")

    def load_vocabulary(self, csv_path: str):
        load_dir = os.path.dirname(csv_path) or "."
        print(f"Loading tokenizer from {load_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(load_dir, use_fast=True)
        # ensure specials as we expect them
        self._ensure_specials({
            "pad_token": PAD, "bos_token": BOS, "eos_token": EOS, "unk_token": UNK
        })
        self.vocab = self.tokenizer.get_vocab()
        self.inv_vocab = {i: tok for tok, i in self.vocab.items()}
        print("Tokenizer loaded successfully.")
        return self

    def vocab_size(self) -> int:
        return len(self.tokenizer)

    def normalize(self, sentence: str) -> str:
        return unicodedata.normalize(self.norm_form, sentence)

    def build_vocabulary(self, sentences: list[str], csv_path: str):
        self.save_vocabulary(csv_path)
        return self

    def get_token_id(self, token: str) -> int:
        tid = self.tokenizer.convert_tokens_to_ids(token)
        if tid is None or tid == self.tokenizer.unk_token_id:
            return self.tokenizer.unk_token_id
        return tid

    def tokenize(self, sentence: str, max_len: Optional[int] = None) -> List[int]:
        text = self.normalize(sentence)
        # encode without adding BOS/EOS because we'll add our own
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        ids = [self.tokenizer.bos_token_id] + ids + [self.tokenizer.eos_token_id]
        if max_len is not None and len(ids) < max_len:
            ids = ids + [self.tokenizer.pad_token_id] * (max_len - len(ids))
        return ids

    def detokenize(self, token_ids: List[int], strip_specials: bool = True) -> str:
        if strip_specials:
            ids = [int(t) for t in token_ids
                   if t not in (self.tokenizer.pad_token_id,
                                self.tokenizer.bos_token_id,
                                self.tokenizer.eos_token_id)]
        else:
            ids = [int(t) for t in token_ids]
        # decode will ignore unknown control tokens; specials removed above
        return self.tokenizer.decode(ids, skip_special_tokens=True)
