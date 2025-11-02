import os, time, torch, torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Optional

from czecher_tokenizers.tokenizer import Tokenizer
from dataset import CommaDataset
from utils.train_logger import log_step_wandb


class CzecherTransformer(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, embedding_dim: int = 256, max_tokens: int = 128, d_model: int = 256, nhead: int = 4, num_layers: int = 5, dim_ff: int = 512, dropout: float = 0.1):  # max_tokens = 512
        super().__init__()
        self.pad_id = pad_id
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
            max_len=max_tokens,
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
    
    # def train_model(self, dataset, epochs: int, batch_size: int = 256, lr: float = 2e-4, pos_weight: float = 3, log_every: int = 300, log_fn = None, save_every_steps: int | None = 5000, resume_from: str | None = None) -> tuple[float, list]:
    #     train_start = time.time()

    #     torch.backends.cuda.matmul.allow_tf32 = True
    #     torch.backends.cudnn.allow_tf32 = True

    #     progress = []
    #     ckpt_dir = 'checkpoints'

    #     print(f'Creating splits...')
    #     split = int(0.90 * len(dataset))
    #     train_ds, eval_ds = torch.utils.data.random_split(dataset, [split, len(dataset)-split])

    #     print(f'Creating DataLoaders...')
    #     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, persistent_workers=True, num_workers=8, prefetch_factor=2)
    #     eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, pin_memory=True, persistent_workers=True, num_workers=8)

    #     print(f'Loading model and optimizer...')
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    #     self.to(device)

    #     scaler = torch.amp.GradScaler(device)
    #     optimizer = AdamW(self.parameters(), lr=lr, weight_decay=0.01, fused=True)
    #     use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    #     amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16 if use_bf16 else torch.float16)

    #     best_f1 = -1.0
    #     start_epoch = 1

    #     if resume_from:
    #         print(f"[ckpt] Resuming from {resume_from}")
    #         model_loaded, opt_state, scaler_state, meta = self.load_checkpoint(resume_from, map_location=device)
    #         self.load_state_dict(model_loaded.state_dict(), strict=True)
    #         if opt_state is not None:
    #             optimizer.load_state_dict(opt_state)
    #         if scaler_state is not None and torch.cuda.is_available():
    #             scaler.load_state_dict(scaler_state)
    #         start_epoch = int(meta.get("epoch", 0)) + 1
    #         global_steps = int(meta.get("global_steps", 0))
    #         best_f1 = float(meta.get("best_f1", -1.0))

    #     print(f'Beginning training...')
    #     global_steps = 0
    #     for epoch in range(start_epoch, epochs + 1):
    #         ts = time.time()
    #         best_eval, global_steps = self.train_epoch(global_steps, train_loader, eval_loader, lr=lr, pos_weight=pos_weight, device=device, log_every=log_every, log_fn=log_fn, optimizer=optimizer, scaler=scaler, amp_ctx=amp_ctx, save_every_steps=save_every_steps, epoch=epoch, best_f1=best_f1)
    #         log_fn(best_eval)
    #         progress.append(best_eval)

    #         curr_f1 = best_eval["eval/f1"]
    #         if curr_f1 > best_f1:
    #             best_f1 = curr_f1
    #             self.save_checkpoint(
    #                 os.path.join(ckpt_dir, "best.pt"),
    #                 optimizer=optimizer, scaler=scaler,
    #                 epoch=epoch, global_steps=global_steps, best_f1=best_f1
    #             )

    #         # Always save "last.pt" at end of epoch
    #         self.save_checkpoint(
    #             os.path.join(ckpt_dir, "last.pt"),
    #             optimizer=optimizer, scaler=scaler,
    #             epoch=epoch, global_steps=global_steps, best_f1=best_f1
    #         )

    #         print(f"Epoch {epoch} - {round(time.time() - ts, 2)} seconds: loss={best_eval['train/loss']:.4f} best_threshold={best_eval['eval/best_threshold']:.2f} | P/R/F1={best_eval['eval/precision']:.3f}/{best_eval['eval/recall']:.3f}/{best_eval['eval/f1']:.3f}")

    #     print(f'Training {epochs} epochs complete in {round(time.time() - train_start)} seconds.')
    #     return best_eval['eval/best_threshold'], progress
    
    # def train_epoch(self, global_steps: int, train_loader: DataLoader, eval_loader: DataLoader, lr: float = 2e-4, pos_weight: float = 1, device = 'cuda', log_every: int = 300, log_fn = None, optimizer=None, scaler=None, amp_ctx=None, save_every_steps=None, epoch=None, best_f1=None) -> tuple[dict, int]:
    #     self.train()
    #     self.to(device)

    #     loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight), device=device))

    #     steps = len(train_loader)
    #     total_loss = 0.0
    #     ts = time.time()
    #     for step, batch in enumerate(train_loader, start=1):
    #         inputs = batch['inputs'].to(device).long()
    #         labels = batch['labels'].to(device).float()
    #         optimizer.zero_grad(set_to_none=True)
    #         mask = (inputs != self.pad_id)

    #         with amp_ctx:
    #             logits = self(inputs)
    #             loss = loss_fn(logits[mask], labels[mask])
            
    #         scaler.scale(loss).backward()
    #         scaler.unscale_(optimizer)
    #         torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
    #         scaler.step(optimizer)
    #         scaler.update()
    #         optimizer.zero_grad(set_to_none=True)
    #         total_loss += float(loss.item())

    #         global_steps += 1
    #         if log_every and global_steps % log_every == 0:
    #             best_eval = self.get_best_eval(eval_loader, device=device)
    #             best_eval['train/loss'] = total_loss / max(1, step)

    #             print(f'Finished {step}/{steps} steps in epoch - {round(step / (time.time() - ts), 2)} steps/second ({global_steps} total steps)')
    #             log_fn(best_eval, step=global_steps)
            
    #         if save_every_steps and (global_steps % save_every_steps == 0):
    #             self.save_checkpoint(
    #                 os.path.join('checkpoints', f"step_{global_steps}.pt"),
    #                 optimizer=optimizer,
    #                 scaler=scaler if device == 'cuda' else None,
    #                 epoch=epoch,
    #                 global_steps=global_steps,
    #                 best_f1=best_f1,
    #             )
        

    #     best_eval = self.get_best_eval(eval_loader, device=device)
    #     best_eval['train/loss'] = total_loss / max(1, steps)
    #     # best_eval['epoch'] = epoch
    #     return best_eval, global_steps

    def _checkpoint_payload(self, optimizer=None, scaler=None, epoch=0, global_steps=0, best_f1=None, extra: dict | None = None):
        return {
            "format": "CommaModel.v1",
            "config": self._config,
            "state_dict": self.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scaler": scaler.state_dict() if (scaler is not None) else None,
            "epoch": epoch,
            "global_steps": global_steps,
            "best_f1": best_f1,
            "extra": extra or {},
            # Optional: RNG states for full determinism
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
        }

    def save_checkpoint(self, path: str, optimizer=None, scaler=None, epoch=0, global_steps=0, best_f1=None, extra: dict | None = None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self._checkpoint_payload(optimizer, scaler, epoch, global_steps, best_f1, extra), path)
        print(f"[ckpt] Saved checkpoint → {path}")

    @classmethod
    def load_checkpoint(cls, path: str, map_location: str | torch.device = "cpu"):
        """Returns (model, optimizer_state_dict, scaler_state_dict, meta_dict)"""
        ckpt = torch.load(path, map_location=map_location)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"], strict=True)
        meta = {
            "epoch": ckpt.get("epoch", 0),
            "global_steps": ckpt.get("global_steps", 0),
            "best_f1": ckpt.get("best_f1", None),
            "extra": ckpt.get("extra", {}),
        }
        return model, ckpt.get("optimizer"), ckpt.get("scaler"), meta
    

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

        pad_stop = len(token_ids)
        if self.pad_id in token_ids:
            pad_stop = token_ids.index(self.pad_id)

        punctuated = ''
        for idx in range(pad_stop):
            punctuated += tokenizer.detokenize([token_ids[idx]])
            if probs[idx] >= threshold:
                punctuated += ','
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
    