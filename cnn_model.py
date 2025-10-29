import os
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW


class CzecherCNN(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, embedding_dim: int = 128,
                 channels: int = 128, kernel_size: int = 5, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)

        # project embeddings to conv channels if needed
        self.in_proj = nn.Linear(embedding_dim, channels) if embedding_dim != channels else nn.Identity()

        layers = [] 
        pad = (kernel_size - 1) // 2  # "same" padding for odd kernels
        for _ in range(n_layers):
            layers += [
                nn.Conv1d(channels, channels, kernel_size, padding=pad),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
        self.conv = nn.Sequential(*layers)
        self.head = nn.Conv1d(channels, 1, kernel_size=1)

        self._config = dict(
            vocab_size=vocab_size,
            pad_id=pad_id,
            embedding_dim=embedding_dim,
            channels=channels,
            kernel_size=kernel_size,
            n_layers=n_layers,
            dropout=dropout,
        )
    
    def save(self, path: str = 'model.pt') -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "config": self._config,
            "state_dict": self.state_dict(),
            "format": "CommaModel.v1",
        }
        torch.save(payload, path)
        print(f'Model saved to {path}.')

    @classmethod
    def load(cls, path: str = 'model.pt', map_location: Optional[str | torch.device] = None) -> "CommaModel":
        ckpt = torch.load(path, map_location=map_location)
        config = ckpt["config"]
        model = cls(**config)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        print(f'Model loaded from {path}.')
        return model
        

    def forward(self, input_ids):         # input_ids: [B, T] (long)
        x = self.embed(input_ids)         # [B, T, E]
        x = self.in_proj(x)               # [B, T, C]
        x = x.transpose(1, 2)             # [B, C, T] for Conv1d
        x = self.conv(x)                  # [B, C, T]
        # x = x.mean(dim=2)                 # global avg pool over T -> [B, C]
        logits = self.head(x).squeeze(1)             # [B, 512]
        return logits

    
    def train_epoch(self, train_loader, weight = 8, device = 'cpu'):
        self.train()

        optimizer = AdamW(self.parameters(), lr=2e-4)  # 0.0002 ?
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(weight), device=device))

        total = 0.0
        for batch in train_loader:
            # print(batch)
            inputs = batch['inputs'].to(device).long()
            labels = batch['labels'].to(device).float()

            logits = self(inputs)
            mask = (inputs != self.pad_id)
            loss = loss_fn(logits[mask], labels[mask])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += float(loss.item())
        
        return total / max(1, len(train_loader))
    
    @torch.no_grad()
    def evaluate(self, eval_loader, threshold: float = 0.5, pos_weight=None, device=None):
        self.eval()

        if device is None:
            device = next(self.parameters()).device

        # Loss function (no need for reduction='none'; we average per batch then overall)
        if pos_weight is not None:
            if isinstance(pos_weight, (float, int)):
                pos_weight = torch.tensor(float(pos_weight), device=device)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            loss_fn = nn.BCEWithLogitsLoss()

        total_loss = 0.0
        n_batches = 0

        # Running counts for micro metrics
        tp = fp = fn = tn = 0

        for batch in eval_loader:
            inputs = batch['inputs'].to(device).long()
            labels = batch['labels'].to(device).float()

            logits = self(inputs)                  # [B, 512] (or [B, 1])
            mask = (inputs != self.pad_id)

            loss = loss_fn(logits[mask], labels[mask])
            total_loss += float(loss.item())
            n_batches += 1

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold) & mask
            y_true = (labels >= 0.5).bool() & mask

            tp += (preds &  y_true).sum().item()
            fp += (preds & ~y_true).sum().item()
            fn += (~preds &  y_true).sum().item()
            tn += (~preds & ~y_true).sum().item()

        # Safeguards against divide-by-zero
        precision = tp / (tp + fp + 1e-12)
        recall    = tp / (tp + fn + 1e-12)
        f1        = 2 * precision * recall / (precision + recall + 1e-12)
        accuracy  = (tp + tn) / max(1, (tp + tn + fp + fn))

        avg_loss = total_loss / max(1, n_batches)

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'threshold': threshold,
        }
    
    @torch.no_grad()
    def predict_comma_ids(self, input_ids, threshold: float = 0.5, device='cpu'):
        """Get the idxs of chars after which to add a comma."""
        self.eval()
        x = torch.as_tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]
        logits = self(x)                              # [1, T]
        probs = torch.sigmoid(logits)[0]              # [T]
        mask = (x[0] != self.pad_id)

        comma_idxs = torch.nonzero((probs >= threshold) & mask, as_tuple=False).flatten().tolist()

        # Subtract 1 because of BOS token
        for c_idx in comma_idxs:
            c_idx -= 1
        return comma_idxs, probs.detach().cpu().tolist()

    @torch.no_grad()
    def punctuate(self, text: str, tokenizer, threshold: float = 0.5, device='cpu'):
        """Return a corrected sentence."""
        ids = tokenizer.tokenize(text)
        comma_idxs, probs = self.predict_comma_ids(ids, threshold=threshold, device=device)
        out = []
        T = len(text)
        comma_set = set(comma_idxs)

        for i, ch in enumerate(text):
            out.append(ch)
            if i in comma_set:
                out.append(",")

        punctuated = "".join(out)
        return punctuated