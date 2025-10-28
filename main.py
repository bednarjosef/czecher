from dataset import CommaDataset
from char_tokenizer import CharTokenizer
from download_wiki_texts import load_sentences

from czecher_model import CommaModel
from train import train_model


def main():
    print('Hello world!')
    # sentences = load_sentences(path='data/sentences.txt')
    # tokenizer = CharTokenizer().build_vocabulary(sentences, 'data/vocabulary.csv')
    # dataset = CommaDataset().create_dataset(sentences, tokenizer, 'data/dataset.csv', max_len=512)
    tokenizer = CharTokenizer().load_vocabulary('data/vocabulary.csv')
    dataset = CommaDataset().load_dataset('data/dataset.csv')
    model = CommaModel(embedding_dim=128, vocab_size=tokenizer.vocab_size(), unk_id=tokenizer.get_token_id('[UNK]'))
    train_model(model, dataset, epochs=10)


if __name__ == '__main__':
    main()
