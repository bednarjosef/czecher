from pathlib import Path
from dataset import CommaDataset
from czecher_tokenizers.bpe_tokenizer import GPTTokenizer
from czecher_tokenizers.char_tokenizer import CharTokenizer
# from download_wiki_texts import load_sentences
from cnn_model import CzecherCNN
from train import train_model
from transformer_model import CzecherTransformer
import torch
from torch.utils.data import DataLoader


def load_sentences(path="sentences.txt"):
    return [line.rstrip("\n") for line in Path(path).read_text(encoding="utf-8").splitlines()]

def correct_sentence(text, model, tokenizer):
    punctuated = model.punctuate(text, tokenizer, threshold=0.5)
    print("INPUT: ", text)
    print("OUTPUT:", punctuated)

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
    print('Hello world!')
    tokenizer = GPTTokenizer(json_file='tokenizer.json')
    sentences = load_sentences(path='data/sentences_500k.txt')
    dataset = CommaDataset().create_dataset(sentences, tokenizer, 'data/bpe/dataset_500k.csv', max_tokens=128)

    # dataset = CommaDataset().load_dataset(csv_path='data/bpe/dataset_500k.csv')
    # point = dataset[394]
    # tokens = point['sentence']
    # commas = point['commas']
    # strs = []
    # for token_id in tokens:
    #     tok_str = tokenizer.detokenize([token_id])
    #     strs.append(tok_str)
    # final = "|".join(strs)
    # print(tokens)
    # print(final)
    # print(commas)
    # commad = ''
    # for idx, cm in enumerate(commas):
    #     commad = commad + tokenizer.detokenize([tokens[idx]])
    #     if cm == 1:
    #         commad = commad + ','
    # print(commad)
            


    # dataset = CommaDataset().create_dataset(sentences, tokenizer, 'data/bpe/dataset_500k.csv', max_tokens=256)

    # tokens = tokenizer.tokenize('Kdyz jsem prisel domu, bylo mi fajn (ale ne moc), a zaroven jsem vubec nevedel, kdy uz prijdou rodice.', max_tokens=256)
    # strs = []
    # for token_id in tokens:
    #     tok_str = tokenizer.detokenize([token_id])
    #     strs.append(tok_str)
    # final = "|".join(strs)
    # print(tokens)
    # print(final)

    # dataset = CommaDataset().load_dataset('data/dataset.csv')
    # pos = estimate_pos_weight(tokenizer, dataset, 'cpu')
    # print(pos)
    # model = CzecherTransformer(vocab_size=tokenizer.vocab_size(), pad_id=tokenizer.get_pad_token_id(), embedding_dim=256)
    # train_model(model, dataset, epochs=5, batch_size=256)
    #model.save('data/500k_model1.pt')
    # model = model.load('data/transformer_model7.pt')
    # text = "V roce 1971 pak Salivarová a Škvorecký založili nakladatelství '68 Publishers kde pak vydávali především české knihy které nemohly vycházet v komunistickém Československu."
    # correct_sentence(text, model, tokenizer)


if __name__ == '__main__':
    main()
