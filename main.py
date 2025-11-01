import torch, wandb

from torch.utils.data import DataLoader

from dataset import CommaDataset
from models.transformer_model import CzecherTransformer
from czecher_tokenizers.bpe_tokenizer import GPTTokenizer
from czecher_tokenizers.char_tokenizer import CharTokenizer
from utils.download_wiki_texts import load_sentences


def collate(batch):
    inputs = torch.tensor([b['sentence'] for b in batch], dtype=torch.long)
    labels = torch.tensor([b['commas'] for b in batch], dtype=torch.float32)
    return { 'inputs': inputs, 'labels': labels }

def estimate_pos_weight(tokenizer: CharTokenizer, dataset, device="cpu"):
    pad_id = tokenizer.get_token_id('[PAD]')
    split = int(0.90 * len(dataset))
    train_ds, dev_ds = torch.utils.data.random_split(dataset, [split, len(dataset)-split])

    print(f'Creating DataLoaders...')
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate)
    eval_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)

    pos = tot = 0
    for batch in train_loader:
        y = batch['labels'].to(device).float()
        x = batch['inputs'].to(device).long()
        mask = x.ne(pad_id)
        pos += (y[mask] > 0.5).sum().item()
        tot += mask.sum().item()
    pi = pos / max(1, tot)
    return (1 - pi) / max(1e-6, pi), pi


def main():
    epochs = 5
    batch_size = 256
    lr = 2e-4
    pos_weight = 1

    layers = 7

    run = wandb.init(
        entity="czecher-team",
        project="czecher-commas",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "pos_weight": pos_weight,
            "layers": layers,
            "architecture": "Transformer",
            "dataset": "dataset_500k"
        }
    )

    tokenizer = GPTTokenizer(json_file='tokenizer.json')
    dataset = CommaDataset().load_dataset(csv_path='data/bpe/dataset_500k.csv')
    model = CzecherTransformer(vocab_size=tokenizer.vocab_size(), pad_id=tokenizer.get_pad_token_id(), embedding_dim=256, num_layers=layers).load('data/trained_models/500k_model7.pt')

    best_threshold, progress = model.train_model(dataset, epochs=epochs, batch_size=batch_size, lr=lr, pos_weight=pos_weight, log_every=300, log_fn=run.log)
    model.save('data/trained_models/500k_model5.pt')
    # model = model.load('data/500k_model3.pt')

    text = "V roce 1971 pak Salivarová a Škvorecký založili nakladatelství '68 Publishers kde pak vydávali především české knihy které nemohly vycházet v komunistickém Československu."
    corrected = model.punctuate(text, tokenizer, threshold=best_threshold, device=None)
    print(text)
    print(corrected)

    # sentences = load_sentences(path='data/sentences_500k.txt')
    # dataset = CommaDataset().create_dataset(sentences, tokenizer, 'data/bpe/dataset_500k.csv', max_tokens=128)


if __name__ == '__main__':
    main()
