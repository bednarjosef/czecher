import os, time, torch, torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Optional

from czecher_tokenizers.tokenizer import Tokenizer
from dataset import CommaDataset
from utils.train_logger import log_step_wandb


class CzecherTransformer(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, embedding_dim: int = 256, max_len: int = 512, d_model: int = 256, nhead: int = 4, num_layers: int = 5, dim_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.pos = nn.Embedding(max_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

        self._config = dict(
            vocab_size=vocab_size,
            pad_id=pad_id,
            embedding_dim=embedding_dim,
            max_len=max_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_ff=dim_ff,
            dropout=dropout,
        )
        self.to("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, input_ids):
        B, T = input_ids.shape
        device = input_ids.device
        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)

        x = self.embed(input_ids) + self.pos(pos_ids)
        key_pad = input_ids.eq(self.pad_id)

        h = self.encoder(x, src_key_padding_mask=key_pad)
        logits = self.head(h).squeeze(-1)
        return logits
    
    def train_model(self, dataset: CommaDataset, epochs: int, batch_size: int = 256, lr: float = 2e-4, pos_weight: float = 3, log_every: int = 300, log_fn: function = None) -> tuple[float, list]:
        train_start = time.time()
        progress = []

        print(f'Creating splits...')
        split = int(0.90 * len(dataset))
        train_ds, eval_ds = torch.utils.data.random_split(dataset, [split, len(dataset)-split])

        print(f'Creating DataLoaders...')
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
        eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

        print(f'Loading model and optimizer...')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)

        print(f'Beginning training...')
        global_steps = 0
        for epoch in range(1, epochs+1, 1):
            ts = time.time()
            best_eval, global_steps = self.train_epoch(global_steps, train_loader, eval_loader, lr=lr, pos_weight=pos_weight, device=device, log_every=log_every, log_fn=log_fn)
            best_eval['epoch'] = epoch
            log_fn(best_eval)
            progress.append(best_eval)
            print(f"Epoch {epoch} - {round(time.time() - ts, 2)} seconds: loss={best_eval['train/loss']:.4f} best_threshold={best_eval['eval/best_threshold']:.2f} | P/R/F1={best_eval['eval/precision']:.3f}/{best_eval['eval/recall']:.3f}/{best_eval['f1']:.3f}")

        print(f'Training {epochs} epochs complete in {round(time.time() - train_start)} seconds.')
        return best_eval['eval/best_threshold'], progress
    
    def train_epoch(self, global_steps: int, train_loader: DataLoader, eval_loader: DataLoader, lr: float = 2e-4, pos_weight: float = 1, device = 'cuda', log_every: int = 300, log_fn: function = None) -> tuple[dict, int]:
        self.train()
        self.to(device)

        scaler = torch.amp.GradScaler(device)
        optimizer = AdamW(self.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight), device=device))

        steps = len(train_loader)
        total_loss = 0.0
        ts = time.time()
        for step, batch in enumerate(train_loader, start=1):
            inputs = batch['inputs'].to(device).long()
            labels = batch['labels'].to(device).float()
            optimizer.zero_grad(set_to_none=True)
            mask = (inputs != self.pad_id)

            with torch.amp.autocast(device_type=device, dtype=torch.float16):
                logits = self(inputs)
                loss = loss_fn(logits[mask], labels[mask])
            
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item())
            global_steps += 1
            if log_every and global_steps % log_every == 0:
                best_eval = self.get_best_eval(eval_loader, device=device)
                print(f'Finished {step}/{steps} steps in epoch - {round(step / (time.time() - ts), 2)} steps/second ({global_steps} total steps)')
                log_fn(best_eval, step=global_steps)
                ts = time.time()
        

        best_eval = self.get_best_eval(eval_loader, device=device)
        best_eval['train/loss'] = total_loss / max(1, len(train_loader))

        return best_eval, global_steps

    
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
    def load(cls, path: str = 'model.pt', map_location: Optional[str | torch.device] = None):
        ckpt = torch.load(path, map_location=map_location)
        config = ckpt["config"]
        model = cls(**config)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        print(f'Model loaded from {path}.')
        return model

    @torch.no_grad()
    def predict_logits(self, input_ids, device=None):
        if device is None:
            device = next(self.parameters()).device
        x = torch.as_tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
        logits = self(x)[0]
        mask = (x[0] != self.pad_id)
        return logits, mask

    @torch.no_grad()
    def predict_probs(self, input_ids, device=None):
        logits, mask = self.predict_logits(input_ids, device)
        probs = torch.sigmoid(logits)
        return probs.tolist(), mask

    @torch.no_grad()
    def punctuate(self, text: str, tokenizer: Tokenizer, threshold: float = 0.5, device='cuda'):
        ## TODO: add option if threshold='best' to use from self.get_best_eval()
        """Return a corrected sentence."""
        token_ids = tokenizer.tokenize(text, max_tokens=128)
        probs, _mask = self.predict_probs(token_ids, device)

        punctuated = ''
        for idx, comma_prob in enumerate(probs):
            punctuated = punctuated + tokenizer.detokenize([token_ids[idx]])
            if comma_prob >= threshold:
                punctuated = punctuated + ','
        return punctuated
    
    @torch.no_grad()
    def get_best_eval(self, eval_loader, device="cuda"):
        self.eval()
        probs_all, labels_all, mask_all = [], [], []
        for batch in eval_loader:
            inputs = batch['inputs'].to(device).long()
            labels = batch['labels'].to(device).float()
            logits = self(inputs)
            mask = inputs.ne(self.pad_id)
            probs_all.append(torch.sigmoid(logits)[mask])
            labels_all.append(labels[mask])
        probs = torch.cat(probs_all)   # [N]
        labels = torch.cat(labels_all) # [N] 0/1

        best = { "eval/best_threshold": 0.5, "eval/f1": -1, "eval/precision": 0, "eval/recall": 0 }
        for th in torch.linspace(0.05, 0.95, 19, device=probs.device):
            pred = probs >= th
            tp = (pred & (labels > 0.5)).sum().item()
            fp = (pred & (labels <= 0.5)).sum().item()
            fn = ((~pred) & (labels > 0.5)).sum().item()
            p  = tp / (tp + fp + 1e-12)
            r  = tp / (tp + fn + 1e-12)
            f1 = 2*p*r/(p+r+1e-12)
            if f1 > best["eval/f1"]:
                best = { "eval/best_threshold": round(float(th.item()), 4), "eval/f1": round(f1, 4), "eval/precision": round(p, 4), "eval/recall": round(r, 4) }
        return best
    

def collate(batch):
    inputs = torch.tensor([b['sentence'] for b in batch], dtype=torch.long)
    labels = torch.tensor([b['commas'] for b in batch], dtype=torch.float32)
    return { 'inputs': inputs, 'labels': labels }
