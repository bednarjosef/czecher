from pathlib import Path
from dataset import CommaDataset
from char_tokenizer import CharTokenizer
# from download_wiki_texts import load_sentences
from czecher_model import CommaModel
from train import train_model


def load_sentences(path="sentences.txt"):
    return [line.rstrip("\n") for line in Path(path).read_text(encoding="utf-8").splitlines()]


def main():
    print('Hello world!')
    # sentences = load_sentences(path='data/sentences.txt')
    # tokenizer = CharTokenizer().build_vocabulary(sentences, 'data/vocabulary.csv')
    # dataset = CommaDataset().create_dataset(sentences, tokenizer, 'data/dataset.csv', max_len=512)
    tokenizer = CharTokenizer().load_vocabulary('data/vocabulary.csv')
    dataset = CommaDataset().load_dataset('data/dataset.csv')
    model = CommaModel(embedding_dim=128, vocab_size=tokenizer.vocab_size(), pad_id=tokenizer.get_token_id('[PAD]'))
    train_model(model, dataset, epochs=1)
    model.save('model.pt')


if __name__ == '__main__':
    main()
