import numpy as np
import torch
from torch.utils.data import Dataset

class CommaMemmapDataset(Dataset):
    def __init__(self, inputs_path: str, labels_path: str, max_tokens: int, pad_id: int = 0):
        self.X = np.memmap(inputs_path, mode='r', dtype=np.uint32)  # or uint32, must match
        self.Y = np.memmap(labels_path, mode='r', dtype=np.uint8)
        self.max_tokens = max_tokens
        self.pad_id = pad_id
        n_tokens = self.X.size
        self.N = n_tokens // max_tokens
        self.X = self.X.reshape(self.N, max_tokens)
        self.Y = self.Y.reshape(self.N, max_tokens)
        print(f'Successfully loaded Memmap Dataset.')

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx].astype(np.int64, copy=False))
        y = torch.from_numpy(self.Y[idx].astype(np.float32, copy=False))
        return {"inputs": x, "labels": y}
