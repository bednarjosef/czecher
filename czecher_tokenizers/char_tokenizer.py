from czecher_tokenizers.tokenizer import Tokenizer

PAD = "[PAD]"
EOS = "[EOS]"
UNK = "[UNK]"

class CharTokenizer(Tokenizer):
    def __init__(self, norm_form = 'NFC'):
        super().__init__(norm_form)

    def build_vocabulary(self, sentences: list[str], csv_path: str):
        print(f'Building a new CHAR vocabulary from {len(sentences)} sentences...')
        self.sentences = [self.normalize(s) for s in sentences]
        chars = set()
        for s in self.sentences: chars.update(s)
        itos = [PAD, EOS, UNK] + sorted(chars)
        vocab = {ch:i for i,ch in enumerate(itos)}
        self.vocab = vocab
        self.inv_vocab = {i: tok for tok, i in vocab.items()}
        print(f'Vocabulary built successfully ({len(itos)} tokens).')
        self.save_vocabulary(csv_path)
        return self

    def tokenize(self, text: str, max_tokens):
        text = self.normalize(text)
        ids = [self.get_token_id(ch) for ch in text]
        ids = ids + [self.get_token_id(EOS)]
        num_tokens = len(ids)
        if num_tokens < max_tokens:
            diff = max_tokens - num_tokens
            ids = ids + [self.get_token_id(PAD)] * diff
        return ids

    def detokenize(self, tokens: list[int]):
        translated = []
        for token_id in tokens:
            token = self.inv_vocab.get(int(token_id), UNK)
            if token == PAD:
                continue
            if token == EOS:
                break
            translated.append(token)
        return "".join(translated)
