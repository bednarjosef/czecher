from typing import List, Dict, Optional
import unicodedata, csv

PAD = "[PAD]"
BOS = "[BOS]"
EOS = "[EOS]"
UNK = "[UNK]"

class CharTokenizer():
    def __init__(self, norm_form: str = 'NFC'):
        self.norm_form = norm_form
        self.vocab: Dict[str, int] = {}
        self.inv_vocab: Dict[int, str] = {}
        self.sentences: List[str] = []

    def save_vocabulary(self, csv_path: str):  # FROM CHATGPT
        print(f'Saving vocabulary to {csv_path}...')
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "token"])
            for i in range(len(self.inv_vocab)):
                w.writerow([i, self.inv_vocab[i]])
        print(f'Vocabulary successfully saved to {csv_path}.')

    def load_vocabulary(self, csv_path: str):  # FROM CHATGPT
        print(f'Loading vocabulary from {csv_path}...')
        vocab = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                i = int(row["id"])
                tok = row["token"]
                vocab[tok] = i               # rebuild token->id
        # optional: quick sanity check that ids are contiguous
        ids = sorted(vocab.values())
        assert ids == list(range(len(ids))), "ids must be contiguous 0..V-1"
        self.vocab = vocab
        self.inv_vocab = {i: tok for tok, i in vocab.items()}
        print(f'Dataset loaded successfully.')
        return self

    def vocab_size(self):
        return len(self.vocab)

    def normalize(self, sentence: str) -> str:
        return unicodedata.normalize("NFC", sentence)

    def build_vocabulary(self, sentences: list[str], csv_path: str):
        print(f'Building a new vocabulary from {len(sentences)} sentences...')
        self.sentences = [self.normalize(s) for s in sentences]
        chars = set()
        for s in self.sentences: chars.update(s)
        itos = [PAD, BOS, EOS, UNK] + sorted(chars)
        vocab = {ch:i for i,ch in enumerate(itos)}
        self.vocab = vocab
        self.inv_vocab = {i: tok for tok, i in vocab.items()}
        print(f'Vocabulary built successfully ({len(itos)} tokens).')
        self.save_vocabulary(csv_path)
        return self
    
    def get_token_id(self, token):
        return self.vocab.get(token, self.vocab[UNK])
    
    def tokenize(self, sentence: str, max_len = None):
        sentence = self.normalize(sentence)
        ids = [self.get_token_id(ch) for ch in sentence]
        ids = [self.get_token_id(BOS)] + ids + [self.get_token_id(EOS)]
        num_tokens = len(ids)
        if max_len and num_tokens < max_len:
            diff = max_len - num_tokens
            ids = ids + [self.get_token_id(PAD)] * diff
        return ids

    def detokenize(self, token_ids, strip_specials=True):
        out = []
        for tid in token_ids:
            tok = self.inv_vocab.get(int(tid), UNK)

            if strip_specials:
                if tok == PAD:
                    continue
                if tok == BOS:
                    # just skip it
                    continue
                if tok == EOS:
                    # stop reconstruction at EOS
                    break
            out.append(tok)
        return "".join(out)
