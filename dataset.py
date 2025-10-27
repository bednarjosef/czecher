import torch
import json, pandas as pd
from torch.utils.data import Dataset, DataLoader
from char_tokenizer import CharTokenizer
import random


class CommaDataset(Dataset):
    def __init__(self):
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        sentence = self.data[index]['sentence']
        commas = self.data[index]['commas']

        sample = { 'sentence': sentence, 'commas': commas }
        return sample
    
    def random_sample(self):
        return self[random.randint(0, len(self))]
    
    def get_data_pair(self, sentence: str, max_len: int):
        formatted = sentence.strip(',')
        in_tokens = self.tokenizer.tokenize(formatted, max_len=max_len)
        probabilities = [0] * max_len
        commas_found = 0
        for idx, char in enumerate(sentence):
            if char != ',':
                continue
            probabilities[idx - commas_found] = 1  # +1 is for BOS token
            commas_found += 1
        return in_tokens, probabilities
    
    def save_dataset(self, csv_path: str):
        print(f'Saving dataset to {csv_path}...')
        rows = []
        for i, item in enumerate(self.data):
            rows.append({
                "id": i,
                "sentence": json.dumps(item["sentence"], separators=(",", ":")),  # compact JSON
                "commas":   json.dumps(item["commas"],   separators=(",", ":")),
            })

        df = pd.DataFrame(rows, columns=["id", "sentence", "commas"])
        df.to_csv(csv_path, index=False)
        print(f'Dataset successfully saved to {csv_path}.')

    def load_dataset(self, csv_path: str):
        print(f'Loading dataset from {csv_path}...')
        df = pd.read_csv(csv_path)
        data = []
        for _, row in df.iterrows():
            data.append({
                "sentence": json.loads(row["sentence"]),
                "commas":   json.loads(row["commas"]),
            })
        self.data = data
        print(f'Dataset loaded successfully.')
        return self

    def create_dataset(self, sentences: list[str], tokenizer: CharTokenizer, csv_path: str, max_len: int = 512):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_len = max_len
        data = []
        num_sentences = len(sentences)

        print(f'Creating a dataset from {num_sentences} sentences...')
        for idx, sentence in enumerate(self.sentences):
            if (len(sentence) + 2) > max_len:  # +2 accounts for BOS and EOS tokens in tokenizer
                continue
            data_in, data_out = self.get_data_pair(sentence, max_len)
            pair = { 'sentence': data_in, 'commas': data_out }
            data.append(pair)
            if (idx+1) % (num_sentences // 10) == 0:
                print(f'{idx+1}/{num_sentences} sentences processed.')
        
        self.data = data
        print(f'Dataset created successfully ({len(data)} samples).')
        self.save_dataset(csv_path)
        return self
