import os, argparse
from huggingface_hub import snapshot_download


def download_dataset(hf_repo: str) -> str:
    '''Download an HF dataset into downloads/{repo}'''
    # hf_repo = 'josefbednar/czech-memmap-128'
    print(f'Downloading dataset from {hf_repo}...')
    target_path = 'downloads/' + hf_repo 
    os.makedirs(target_path, exist_ok=True)  
    snapshot_download(repo_id=hf_repo, repo_type="dataset", local_dir=target_path, local_dir_use_symlinks=False)
    print(f'Dataset downloaded to {target_path}.')
    return target_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download necessities for training from HF.')
    parser.add_argument('--repo', type=str, help='HF repo with the dataset and tokenizer.')
    args = parser.parse_args()
    download_dataset(args.repo)
