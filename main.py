from dataset import CommaDataset
from char_tokenizer import CharTokenizer
from download_wiki_texts import load_sentences

from torch.optim import AdamW
from czecher_model import CommaModel
from train import collate, train_one_epoch, evaluate
from torch.utils.data import DataLoader
import torch

def main():
    print('Hello world!')
    # sentences = load_sentences(path='data/sentences.txt')
    # tokenizer = CharTokenizer().build_vocabulary(sentences, 'data/vocabulary.csv')
    # dataset = CommaDataset().create_dataset(sentences, tokenizer, 'data/dataset.csv', max_len=512)
    tokenizer = CharTokenizer().load_vocabulary('data/vocabulary.csv')
    dataset = CommaDataset().load_dataset('data/dataset.csv')
    train_loop(dataset, tokenizer)


def train_loop(dataset, tokenizer: CharTokenizer):
    vocab_size = tokenizer.vocab_size()
    pad_id = tokenizer.get_token_id("[PAD]")
    bos_id = tokenizer.get_token_id("[BOS]")
    eos_id = tokenizer.get_token_id("[EOS]")

    print(f'Creating splits...')
    split = int(0.95 * len(dataset))
    train_ds, dev_ds = torch.utils.data.random_split(dataset, [split, len(dataset)-split])

    print(f'Creating DataLoaders...')
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate, num_workers=2)
    dev_loader   = DataLoader(dev_ds,   batch_size=64, shuffle=False, collate_fn=collate, num_workers=2)

    print(f'Loading model and optimizer...')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CommaModel(vocab_size, pad_id, bos_id, eos_id, d_model=64, nhead=4, num_layers=4, dim_ff=512).to(device)
    opt = AdamW(model.parameters(), lr=2e-4)

    print(f'Beginning training...')
    for epoch in range(5):
        tr_loss = train_one_epoch(model, train_loader, opt, pad_id, bos_id, eos_id, pos_weight=4.0, device=device)
        p,r,f1 = evaluate(model, dev_loader, pad_id, bos_id, eos_id, thresh=0.5, device=device)
        print(f"Epoch {epoch+1}: loss={tr_loss:.4f}  P/R/F1={p:.3f}/{r:.3f}/{f1:.3f}")


if __name__ == '__main__':
    main()
