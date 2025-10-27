import torch
from torch.utils.data import DataLoader

from czecher_model import masked_bce_with_logits

def collate(batch):
    # batch: list of {"sentence": list[int], "commas": list[int]}
    input_ids = torch.tensor([b["sentence"] for b in batch], dtype=torch.long)
    labels    = torch.tensor([b["commas"]   for b in batch], dtype=torch.float32)
    # attention: everything that is not PAD is True
    pad_id = input_ids[0,0].item()  # careful: this only works if PAD==id 0 at position 0; better pass pad_id explicitly
    attention_mask = ~input_ids.eq(pad_id)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

def train_one_epoch(model, loader, optimizer, pad_id, bos_id, eos_id, pos_weight=4.0, device="cuda"):
    model.train()
    tot = 0.0
    cnt = 0  
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        logits = model(input_ids, attention_mask)
        loss, mask = masked_bce_with_logits(
            logits, labels, input_ids, pad_id, bos_id, eos_id, pos_weight=pos_weight
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        tot += loss.item()
        cnt += 1
    return tot / max(1, cnt)

@torch.no_grad()
def evaluate(model, loader, pad_id, bos_id, eos_id, thresh=0.5, device="cuda"):
    model.eval()
    tp=fp=fn=0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        logits = model(input_ids, attention_mask)
        probs = logits.sigmoid()
        mask = ~input_ids.eq(pad_id) & ~input_ids.eq(bos_id) & ~input_ids.eq(eos_id)

        pred = (probs >= thresh).float()
        y = labels

        tp += ((pred==1) & (y==1) & mask).sum().item()
        fp += ((pred==1) & (y==0) & mask).sum().item()
        fn += ((pred==0) & (y==1) & mask).sum().item()

    prec = tp/(tp+fp+1e-9)
    rec  = tp/(tp+fn+1e-9)
    f1 = 2*prec*rec/(prec+rec+1e-9)
    return prec, rec, f1
