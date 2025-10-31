import csv
from typing import Dict, List
import unicodedata

PAD = "[PAD]"
EOS = "[EOS]"
UNK = "[UNK]"

class Tokenizer:
    def __init__(self, norm_form: str = 'NFC'):
        self.norm_form = norm_form
        self.vocab: Dict[str, int] = {}
        self.inv_vocab: Dict[int, str] = {}
        self.sentences: List[str] = []

    def vocab_size(self):
        return len(self.vocab)

    def normalize(self, sentence: str) -> str:
        return unicodedata.normalize("NFC", sentence)

    def get_token_id(self, token: str):
            return self.vocab.get(token, self.vocab[UNK])
    
    def get_pad_token_id(self) -> int:
        pass

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
                vocab[tok] = i

        self.vocab = vocab
        self.inv_vocab = {i: tok for tok, i in vocab.items()}
        print(f'Vocabulary loaded successfully.')
        return self
    
    def build_vocabulary(self, sentences: list[str], csv_path: str):
        pass
    
    def tokenize(self, text: str, max_tokens: int = None):
        pass

    def detokenize(self, tokens: list[int]):
        pass

