from typing import List

from czecher_tokenizers.tokenizer import Tokenizer
from czecher_tokenizers.hf_tokenizer import HuggingFaceTokenizer

class GPTTokenizer(Tokenizer):
    def __init__(self, norm_form = 'NFC', json_file='tokenizer.json'):
        super().__init__(norm_form)
        self.tokenizer = HuggingFaceTokenizer.from_directory('czecher_tokenizers', json_file=json_file)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()
    
    # def get_token_id(self, token: str):
    #     token = self.normalize(token)
    #     return self.tokenizer._encode_one(token, prepend=None, append=None)
    
    def get_pad_token_id(self) -> int:
        return self.tokenizer.get_pad_token_id()
    
    def save_vocabulary(self, csv_path):
        pass

    def build_vocabulary(self, sentences: list[str], csv_path: str):
        pass

    def tokenize(self, text, max_tokens: int) -> List[int]:  # text is str or list[str]
        text = self.normalize(text)
        ids = self.tokenizer.encode(text, prepend=self.tokenizer.get_bos_token_id(), append=self.tokenizer.get_eos_token_id())
        if len(ids) > max_tokens:
            ids = ids[:max_tokens-1] + [self.tokenizer.get_eos_token_id()]
        if len(ids) < max_tokens:
            ids = ids + [self.tokenizer.get_pad_token_id()] * (max_tokens - len(ids))
        return ids

    def detokenize(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
