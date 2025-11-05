# torchrun --standalone --nproc_per_node=4 train.py

import os, argparse
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb

from tokenizer import Tokenizer
from model import CzecherTransformer
from dataset import MemmapDataset


# parser = argparse.ArgumentParser(description='Start the training loop.')
# parser.add_argument('--tokenizer_path', type=str, help='Direct path to the tokenizer json.')
# args = parser.parse_args()

dataset_path = 'downloads/dataset'
tokenizer_path = 'downloads/dataset/tokenizer.json'
save_name = '11m_12layer_2epoch.pt'

# ----------------------------
# User/configurable settings
# ----------------------------
depth = 12
lr = 1e-4                    # base LR (we'll scale by world_size below if you want)
dropout = 0.05
max_tokens = 128
epochs = 1
eval_every = 1000
batch_size_per_gpu = 512
grad_accum_steps = 2

# Optimization
weight_decay = 0.01
grad_clip = 1.0
warmup_ratio = 0.10          # 10% warmup
warmdown_ratio = 0.20        # last 20% cosine tail to floor
final_lr_frac = 0.20         # floor = 20% of peak

# ----------------------------
# DDP / device init
# ----------------------------
def ddp_init():
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        is_master = rank == 0
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_master = True
    return ddp, rank, local_rank, world_size, device

ddp, rank, local_rank, world_size, device = ddp_init()
is_master = (rank == 0)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)

# ----------------------------
# Tokenizer / model dims
# ----------------------------
tokenizer = Tokenizer.from_json(tokenizer_path)
vocab_size = tokenizer.get_vocab_size()
if is_master:
    print(f"Vocab size: {vocab_size:,}")

num_layers = depth
model_dim = depth * 64
num_heads = max(1, (model_dim + 127) // 128)  # ~128 head dim
dim_ff = 4 * model_dim

# ----------------------------
# Model
# ----------------------------
model = CzecherTransformer(
    vocab_size=vocab_size,
    pad_id=tokenizer.get_pad_token_id(),
    max_tokens=max_tokens,
    num_layers=num_layers,
    d_model=model_dim,
    embedding_dim=model_dim,
    nhead=num_heads,
    dim_ff=dim_ff,
    dropout=dropout,
)

model.to(device)
model = torch.compile(model, dynamic=False)

num_params = sum(p.numel() for p in model.parameters())
if is_master:
    print(f"Number of parameters: {num_params:,}")

# ----------------------------
# Dataset / Loaders
# ----------------------------
dataset = MemmapDataset(dataset_path, max_tokens=max_tokens, pad_id=tokenizer.get_pad_token_id())

if is_master:
    print("Creating splits...")
split = int(0.95 * len(dataset))
train_ds, eval_ds = torch.utils.data.random_split(dataset, [split, len(dataset) - split])

# Samplers
train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True) if ddp else None
# We'll only evaluate on rank 0; use a normal (non-distributed) eval loader there.
# Non-master ranks will set eval_loader=None and skip eval.

if is_master:
    print("Creating DataLoaders...")

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size_per_gpu,
    shuffle=(train_sampler is None),
    sampler=train_sampler,
    pin_memory=True,
    persistent_workers=True,
    num_workers=8,
    prefetch_factor=2,
    drop_last=True,
)

eval_loader = None
if is_master:
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size_per_gpu,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
        num_workers=8,
        drop_last=False,
    )

steps_per_epoch = len(train_loader)
total_steps = epochs * steps_per_epoch
if is_master:
    print(f"[train] world_size={world_size} steps/epoch={steps_per_epoch} total_steps={total_steps} accum={grad_accum_steps}")

# ----------------------------
# LR scheduler (warmup + cosine to floor)
# ----------------------------
def make_cosine_warmup_sched(total_steps: int, warmup_ratio=0.10, floor_frac=0.20):
    warmup_steps = max(1, int(warmup_ratio * total_steps))
    import math
    def lr_mult(step):
        # step is 0-based global optimizer step index
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        # cosine from 1.0 to floor
        return floor_frac + (1.0 - floor_frac) * 0.5 * (1.0 + math.cos(math.pi * t))
    return lr_mult

lr_mult = make_cosine_warmup_sched(total_steps, warmup_ratio, final_lr_frac)

# Optionally scale LR linearly with world size (classic rule-of-thumb).
scaled_lr = lr * world_size

# ----------------------------
# Optimizer
# ----------------------------
optimizer = AdamW(model.parameters(), lr=scaled_lr, weight_decay=weight_decay, fused=True)

# Wrap with DDP (after optimizer created is fine; DDP wraps model only)
if ddp:
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

# ----------------------------
# WANDB (rank 0 only)
# ----------------------------
if is_master:
    run = wandb.init(
        entity="czecher-team",
        project="czecher-commas",
        config={
            "epochs": epochs,
            "total_steps": total_steps,
            "max_tokens": max_tokens,
            "batch_size_per_gpu": batch_size_per_gpu,
            "grad_accum_steps": grad_accum_steps,
            "learning_rate_base": lr,
            "learning_rate_scaled": scaled_lr,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "layers": depth,
            "d_model": model_dim,
            "num_heads": num_heads,
            "dim_ff": dim_ff,
            "architecture": "Transformer",
            "dataset": "comma_memmap_10m",
            "world_size": world_size,
        },
    )
else:
    class _DummyRun:
        def log(self, *a, **k): pass
        def finish(self): pass
    run = _DummyRun()

# ----------------------------
# Loss
# ----------------------------
loss_fn = nn.BCEWithLogitsLoss()

# ----------------------------
# Training
# ----------------------------
global_step = 1
tokens_per_step_global = batch_size_per_gpu * max_tokens * grad_accum_steps * world_size

for epoch in range(1, epochs + 1):
    if ddp:
        # Important for DistributedSampler shuffling
        train_sampler.set_epoch(epoch)

    model.train()

    # (Re-)create iterator per epoch
    dataloader_iter = iter(train_loader)

    # Prefetch first batch
    batch = next(dataloader_iter)
    inputs = batch["inputs"].pin_memory().to(device, non_blocking=True).long()
    labels = batch["labels"].pin_memory().to(device, non_blocking=True).float()

    total_loss = 0.0
    epoch_start = time.time()

    for step_in_epoch in range(1, steps_per_epoch + 1):
        last_global_step = (global_step) == total_steps

        synchronize()
        iter_start = time.time()
        optimizer.zero_grad(set_to_none=True)

        # Gradient accumulation with no_sync to avoid redundant all-reduces
        for micro in range(grad_accum_steps):
            mask = inputs.ne(model.module.pad_id if ddp else model.pad_id)
            sync_ok = (micro == grad_accum_steps - 1)

            no_sync_ctx = model.no_sync() if (ddp and not sync_ok) else nullcontext()
            with no_sync_ctx:
                with autocast_ctx:
                    logits = model(inputs)
                    loss = loss_fn(logits[mask], labels[mask]) / grad_accum_steps

                total_loss += float(loss.item())
                loss.backward()

            # Async prefetch next batch during backward
            try:
                next_b = next(dataloader_iter)
            except StopIteration:
                # re-create for safety; should rarely happen mid-epoch with drop_last=True
                dataloader_iter = iter(train_loader)
                next_b = next(dataloader_iter)

            next_inputs = next_b["inputs"].pin_memory().to(device, non_blocking=True).long()
            next_labels = next_b["labels"].pin_memory().to(device, non_blocking=True).float()

            # move to next after the micro-step
            if sync_ok:
                inputs, labels = next_inputs, next_labels

        # Gradient clipping
        if grad_clip > 0:
            # clip the underlying module params if DDP
            nn.utils.clip_grad_norm_(model.module.parameters() if ddp else model.parameters(), grad_clip)

        # LR schedule (global step 0-based)
        lr_now = scaled_lr * lr_mult(global_step)
        for g in optimizer.param_groups:
            g["lr"] = lr_now

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        synchronize()

        # Logging (master only)
        if is_master and (global_step % 50 == 0):
            dt = time.time() - iter_start
            tok_per_sec = int(tokens_per_step_global / max(dt, 1e-9))
            print(f"[step {global_step}/{total_steps}] lr={lr_now:.2e} loss={loss.item():.4f} tok/s={tok_per_sec:,}")
            run.log({"lr": lr_now, "tok/s": tok_per_sec}, step=global_step)

        # Periodic eval (master only)
        if is_master and (last_global_step or ((global_step) % eval_every == 0)):
            model.eval()
            # unwrap if DDP for evaluation helper
            eval_model = model.module if ddp else model
            eval = eval_model.get_full_eval(train_loader, eval_loader, threshold=0.5, device=str(device))
            eval["train/loss"] = total_loss / max(1, (step_in_epoch))
            run.log(eval, step=global_step)
            print(f"[eval] step={global_step} loss={eval['train/loss']:.4f} f1={eval['eval/f1']:.3f}")
            # model.save_checkpoint(path=os.path.join('checkpoints', "last.pt"), optimizer=optimizer, global_steps=global_step)
            model.train()

        global_step += 1

        if last_global_step:
            break

    if is_master:
        elapsed = time.time() - epoch_start
        print(f"[epoch {epoch}] done in {elapsed:.2f}s")

# Save once (master only)
if is_master:
    # unwrap if DDP
    to_save = model.module if ddp else model
    os.makedirs('models', exist_ok=True)
    to_save.save(f'models/{save_name}')
    # model.save_checkpoint(path=os.path.join('checkpoints', "last.pt"), optimizer=optimizer, global_steps=global_step)
    print("[train] training finished, model saved.")

# Clean up
run.finish()
if ddp:
    dist.barrier()
    dist.destroy_process_group()
