import os

from czecher_tokenizers.bpe_tokenizer import GPTTokenizer
from models.transformer_model import CzecherTransformer
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
from contextlib import nullcontext
from torch.optim import AdamW
from memmap_dataset import CommaMemmapDataset
from torch.utils.data import DataLoader
import torch.nn as nn

import wandb
import torch

device_type = ""
depth = 10
lr = 1e-4
max_tokens = 128
epochs = 1
eval_every = 300
device_batch_size = 128 # per-device batch size
grad_accum_steps = 4
dataset_size = 5_000_000

num_iterations = dataset_size // device_batch_size # explicit number of steps of the optimization (-1 = disable)

# Optimization
weight_decay = 0.01 # weight decay for the embedding/unembedding parameters (Adam)
grad_clip = 1.0 # gradient clipping value (0.0 = disabled)
warmup_ratio = 0.2 # ratio of iterations for LR warmup
warmdown_ratio = 0.2 # ratio of iterations for LR warmdown
final_lr_frac = 0.2 # final LR is this fraction of the initial LR

# Compute init
device_type = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None

# Tokenizer will be useful for evaluation, also we need the vocab size
tokenizer = GPTTokenizer(json_file='tokenizer.json')
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

# Model kwargs are derived from the desired depth of the model
num_layers = depth
model_dim = depth * 64 # aspect ratio 64 (usually this is varied from 64 -> 128 as model size increases)
num_heads = max(1, (model_dim + 127) // 128) # head dim 128 (the division here is ceil div)
dim_ff = 4 * model_dim

# Initialize the Model
model = CzecherTransformer(vocab_size=vocab_size, pad_id=tokenizer.get_pad_token_id(), max_tokens=max_tokens, num_layers=num_layers, d_model=model_dim, embedding_dim=model_dim, nhead=num_heads, dim_ff=dim_ff)
model.to(device_type)
model = torch.compile(model, dynamic=False)
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params:,}")

# Initialize the Optimizer
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, fused=True)
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16 if use_bf16 else torch.float16)
dataset = CommaMemmapDataset("./comma_memmap/inputs.bin", "./comma_memmap/labels.bin", max_tokens=max_tokens, pad_id=tokenizer.get_pad_token_id())

# Initialize the DataLoaders for train/val
print(f'Creating splits...')
split = int(0.90 * len(dataset))
train_ds, eval_ds = torch.utils.data.random_split(dataset, [split, len(dataset)-split])

print(f'Creating DataLoaders...')
train_loader = DataLoader(train_ds, batch_size=device_batch_size, shuffle=True, pin_memory=True, persistent_workers=True, num_workers=8, prefetch_factor=2)
eval_loader = DataLoader(eval_ds, batch_size=device_batch_size, shuffle=False, pin_memory=True, persistent_workers=True, num_workers=8)
total_steps = epochs * len(train_loader)

# Learning rate scheduler
def get_lr_multiplier(step):
    warmup_iters = round(warmup_ratio * total_steps)
    warmdown_iters = round(warmdown_ratio * total_steps)
    if step < warmup_iters:
        return (step + 1) / warmup_iters
    elif step <= total_steps - warmdown_iters:
        return 1.0
    else:
        progress = (total_steps - step) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac

# Training loop
model.train()
loss_fn = nn.BCEWithLogitsLoss()

dataloader_iter = iter(train_loader)
def next_batch():
    try:
        return next(dataloader_iter)
    except StopIteration:
        return next(iter(train_loader))  # restart if exhausted

# prefetch the first batch
batch = next_batch()
inputs = batch['inputs'].pin_memory().to(device_type, non_blocking=True).long()
labels = batch['labels'].pin_memory().to(device_type, non_blocking=True).float()

total_loss = 0.0
t0 = time.time()

run = wandb.init(
    entity="czecher-team",
    project="czecher-commas",
    config={
        "epochs": epochs,
        "max_tokens": max_tokens,
        "batch_size": device_batch_size,
        "grad_accum_steps": grad_accum_steps,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "layers": depth,
        "d_model": model_dim,
        "num_heads": num_heads,
        "dim_ff": dim_ff,
        "architecture": "Transformer",
        "dataset": "comma_memmap_5m"
    }
)
print(f"[train] total_steps={total_steps}, accum={grad_accum_steps}")

# note that we run +1 steps only so that we can eval and save at the end
for step in range(1, total_steps + 1):
    last_step = step == total_steps

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    iter_start = time.time()
    optimizer.zero_grad(set_to_none=True)

    for micro in range(grad_accum_steps):
        mask = inputs.ne(model.pad_id)
        with amp_ctx:
            logits = model(inputs)
            loss = loss_fn(logits[mask], labels[mask]) / grad_accum_steps
        total_loss += float(loss.item())

        loss.backward()

        # async prefetch next batch during backward
        next_b = next_batch()
        next_inputs = next_b['inputs'].pin_memory().to(device_type, non_blocking=True).long()
        next_labels = next_b['labels'].pin_memory().to(device_type, non_blocking=True).float()

        # move to next after backward finishes
        if micro == grad_accum_steps - 1:
            inputs, labels = next_inputs, next_labels

    # gradient clipping
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # cosine decay lr
    lr_mult = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = lr * lr_mult

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    synchronize()
    dt = time.time() - iter_start
    tokens_per_step = train_loader.batch_size * max_tokens * grad_accum_steps
    tok_per_sec = int(tokens_per_step / dt)

    if step % 50 == 0:
        print(f"[step {step}] lr={optimizer.param_groups[0]['lr']:.2e} loss={loss.item():.4f} tok/s={tok_per_sec:,}")
        run.log(data={'lr': optimizer.param_groups[0]['lr'], 'tok/s': tok_per_sec}, step=step)

    if last_step or (step % eval_every == 0 and step > 0):
        model.eval()
        best_eval = model.get_best_eval(eval_loader, device=device_type)
        best_eval["train/loss"] = total_loss / max(1, (step + 1))

        run.log(best_eval, step=step)
        print(f"[eval] step={step} loss={best_eval['train/loss']:.4f} f1={best_eval['eval/f1']:.3f}")
        model.train()

    if last_step:
        model.save("data/trained_models/5m_model2.pt")
        print("[train] training finished, model saved.")
        break


run.finish()
