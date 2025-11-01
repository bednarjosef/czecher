from datasets import load_dataset
from blingfire import text_to_sentences
from tqdm import tqdm
from itertools import islice
from pathlib import Path

MIN_LEN = 20
MAX_LEN = 512
TARGET_SENTENCES = 5_000_000 + 8


def save_sentences(sentences, path="sentences.txt"):
    print(f'Saving {len(sentences)} sentences to {path}...')
    path = Path(path)
    path.write_text(
        "\n".join(s.replace("\r", " ").replace("\n", " ").strip() for s in sentences),
        encoding="utf-8"
    )
    print(f'Succesfully saved {len(sentences)} to {path}.')


def load_sentences(path="sentences.txt"):
    return [line.rstrip("\n") for line in Path(path).read_text(encoding="utf-8").splitlines()]


def download_wiki_sentences():
    stream = load_dataset("wikimedia/wikipedia", "20231101.cs",
                          split="train", streaming=True)

    sentences = []
    it = iter(stream)  # make ONE iterator

    while len(sentences) < TARGET_SENTENCES:
        try:
            rec = next(it)  # get next page record (a dict)
        except StopIteration:
            # dataset ended before hitting target
            break

        text = rec.get("text") or rec.get("content") or ""
        if not text:
            continue

        for sent in text_to_sentences(text).split("\n"):
            s = sent.strip()
            if MIN_LEN <= len(s) <= MAX_LEN:
                sentences.append(s)
                # (optional) comment out to avoid spammy console:
                print(f'Found {len(sentences)}/{TARGET_SENTENCES} sentences.')
                if len(sentences) >= TARGET_SENTENCES:
                    break

    print(f"Collected {len(sentences)} sentences.")
    return sentences[8:]


if __name__ == '__main__':
    sentences = download_wiki_sentences()
    save_sentences(sentences, 'data/sentences_5M.txt')
