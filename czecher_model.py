import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW


class CommaModel(nn.Module):
    def __init__(self, vocab_size: int, unk_id: int, embedding_dim: int = 128,
                 channels: int = 128, kernel_size: int = 5, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=unk_id)

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

        # global average pool over time → [B, C]
        # self.head = nn.Sequential(
        #     nn.Linear(channels, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512)      # final logits per class (no sigmoid)
        # )

        self.head = nn.Conv1d(channels, 1, kernel_size=1)
        

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

            predictions = self(inputs)
            loss = loss_fn(predictions, labels)
            
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
            loss = loss_fn(logits, labels)
            total_loss += float(loss.item())
            n_batches += 1

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold)

            y_true = (labels >= 0.5).bool()
            y_pred = preds.bool()

            tp += (y_pred &  y_true).sum().item()
            fp += (y_pred & ~y_true).sum().item()
            fn += (~y_pred &  y_true).sum().item()
            tn += (~y_pred & ~y_true).sum().item()

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
    
    # def evaluate(self, eval_loader):
    #     self.eval()
    #     tp=fp=fn=0
    #     for batch in eval_loader:
    #         inputs = batch["input_ids"].to(device)
    #         labels = batch["labels"].to(device)
    #         attention_mask = batch["attention_mask"].to(device)

    #         logits = self(inputs, attention_mask)
    #         probs = logits.sigmoid()

    #         pred = (probs >= thresh).float()
    #         y = labels

    #         tp += ((pred==1) & (y==1) & mask).sum().item()
    #         fp += ((pred==1) & (y==0) & mask).sum().item()
    #         fn += ((pred==0) & (y==1) & mask).sum().item()

    #     prec = tp/(tp+fp+1e-9)
    #     rec  = tp/(tp+fn+1e-9)
    #     f1 = 2*prec*rec/(prec+rec+1e-9)
    #     return prec, rec, f1