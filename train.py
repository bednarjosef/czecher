import torch
from torch.utils.data import DataLoader
from char_tokenizer import CharTokenizer
from cnn_model import CzecherCNN
from dataset import CommaDataset


def collate(batch):
    inputs = torch.tensor([b['sentence'] for b in batch], dtype=torch.long)
    labels = torch.tensor([b['commas'] for b in batch], dtype=torch.float32)
    return { 'inputs': inputs, 'labels': labels }


def train_model(model: CzecherCNN, dataset: CommaDataset, epochs: int, batch_size: int = 64):
    print(f'Creating splits...')
    split = int(0.90 * len(dataset))
    train_ds, dev_ds = torch.utils.data.random_split(dataset, [split, len(dataset)-split])

    print(f'Creating DataLoaders...')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    eval_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    print(f'Loading model and optimizer...')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print(f'Beginning training...')
    for epoch in range(epochs):
        epoch_loss = model.train_epoch(train_loader, eval_loader, 8, device)
        evals = model.evaluate(eval_loader, threshold=0.5, pos_weight=3, device=device)
        print(f"Epoch {epoch+1}: loss={epoch_loss:.4f}  P/R/F1={evals['precision']:.3f}/{evals['recall']:.3f}/{evals['f1']:.3f}")
