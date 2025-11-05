import os, time, torch, torch.nn as nn
from typing import Optional

from tokenizer import Tokenizer

class CzecherTransformer(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, embedding_dim: int = 256, max_tokens: int = 128, d_model: int = 256, nhead: int = 4, num_layers: int = 5, dim_ff: int = 512, dropout: float = 0.1, max_len=None):  # max_tokens = 512
        super().__init__()
        self.pad_id = pad_id
        if max_len:
            max_tokens = max_len
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.pos = nn.Embedding(max_tokens, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.final_ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

        self._config = dict(
            vocab_size=vocab_size,
            pad_id=pad_id,
            embedding_dim=embedding_dim,
            max_tokens=max_tokens,
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
        h = self.final_ln(h)
        logits = self.head(h).squeeze(-1)
        return logits

    def _checkpoint_payload(self, optimizer=None, global_steps=0, extra: dict | None = None):
        return {
            "format": "CommaModel.v1",
            "config": self._config,
            "state_dict": self.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "global_steps": global_steps,
            "extra": extra or {},
        }

    def save_checkpoint(self, path: str, optimizer=None, global_steps=0, extra: dict | None = None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self._checkpoint_payload(optimizer, global_steps, extra), path)
        print(f"[ckpt] Saved checkpoint → {path}")

    @classmethod
    def load_checkpoint(cls, path: str, map_location: str | torch.device = "cpu"):
        """Returns (model, optimizer_state_dict, meta_dict)"""
        ckpt = torch.load(path, map_location=map_location)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"], strict=True)
        meta = {
            "global_steps": ckpt.get("global_steps", 0),
            "extra": ckpt.get("extra", {}),
        }
        return model, ckpt.get("optimizer"), meta
    

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
        ts = time.time()
        ## TODO: add option if threshold='best' to use from self.get_best_eval()
        """Return a corrected sentence."""
        token_ids = tokenizer.tokenize(text, max_tokens=128)
        probs, _mask = self.predict_probs(token_ids, device)

        pad_stop = len(token_ids)
        if self.pad_id in token_ids:
            pad_stop = token_ids.index(self.pad_id)

        punctuated = ''
        for idx in range(pad_stop):
            punctuated += tokenizer.detokenize([token_ids[idx]])
            if probs[idx] >= threshold:
                punctuated += ','
        print(f'Punctuation finished in {round(time.time()-ts, 3)} seconds.')
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
    
    @torch.no_grad()
    def get_eval(self, eval_loader, threshold=0.5, device='cuda'):
        self.eval()
        probs_all, labels_all, mask_all = [], [], []
        for batch in eval_loader:
            inputs = batch['inputs'].to(device).long()
            labels = batch['labels'].to(device).float()
            logits = self(inputs)
            mask = inputs.ne(self.pad_id)
            probs_all.append(torch.sigmoid(logits)[mask])
            labels_all.append(labels[mask])
        probs = torch.cat(probs_all)
        labels = torch.cat(labels_all)

        pred = probs >= threshold
        tp = (pred & (labels > 0.5)).sum().item()
        fp = (pred & (labels <= 0.5)).sum().item()
        fn = ((~pred) & (labels > 0.5)).sum().item()
        p  = tp / (tp + fp + 1e-12)
        r  = tp / (tp + fn + 1e-12)
        f1 = 2*p*r/(p+r+1e-12)
        return { 'eval/precision': round(p, 4), 'eval/recall': round(r, 4), 'eval/f1': round(f1, 4) }

    @torch.no_grad()
    def get_train_eval(self, train_loader, threshold=0.5, device='cuda'):
        self.eval()
        probs_all, labels_all, mask_all = [], [], []
        for batch in train_loader:
            inputs = batch['inputs'].to(device).long()
            labels = batch['labels'].to(device).float()
            logits = self(inputs)
            mask = inputs.ne(self.pad_id)
            probs_all.append(torch.sigmoid(logits)[mask])
            labels_all.append(labels[mask])
        probs = torch.cat(probs_all)
        labels = torch.cat(labels_all)

        pred = probs >= threshold
        tp = (pred & (labels > 0.5)).sum().item()
        fp = (pred & (labels <= 0.5)).sum().item()
        fn = ((~pred) & (labels > 0.5)).sum().item()
        p  = tp / (tp + fp + 1e-12)
        r  = tp / (tp + fn + 1e-12)
        f1 = 2*p*r/(p+r+1e-12)
        return { 'train/precision': round(p, 4), 'train/recall': round(r, 4), 'train/f1': round(f1, 4) }

    @torch.no_grad()
    def get_full_eval(self, train_loader, eval_loader, threshold=0.5, device='cuda'):
        eval_d = self.get_eval(eval_loader=eval_loader, threshold=threshold, device=device)
        train_d = self.get_train_eval(train_loader=train_loader[:len(eval_loader)], threshold=threshold, device=device)
        # evals = {**eval_d, **train_d}
        evals = eval_d | train_d
        return evals

