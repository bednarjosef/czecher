import os, argparse

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
    # parser.add_argument('--max_chars', type=int, default=10_000_000_000, help='Maximum characters to train on (default: 10B)')
    # args = parser.parse_args()
    os.makedirs('downloads', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
