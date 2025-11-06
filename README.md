# czecher
A lightweight Transformer that learns where to put commas in Czech sentences.
Trained on 11 million sentences (128 tokens), parallel GPU training with torchrun (DDP), and Wandb logging.

# Try it out
You can try out the Czecher model on [czecher.cz](https://czecher-web-pq4z.vercel.app/){:target="_blank"}.


<!-- <p align="left"> <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white"></a> <a href="#"><img alt="License" src="https://img.shields.io/badge/License-MIT-green"></a> <a href="#"><img alt="W&B" src="https://img.shields.io/badge/Weights%20%26%20Biases-Logging-yellow"></a> </p> -->

# Quickstart - Linux
## Get the repository and initialize
```bash
# Get the repository locally
git clone https://github.com/bednarjosef/czecher.git
cd czecher

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install packages from requirements.txt
pip install -r requirements.txt

# Run setup.py (creates directories)
python3 setup.py
```

## Download a dataset
```bash
# Download a specified HF dataset repository into /downloads/
python3 download.py --repo josefbednar/syn2006pub-11m-128-tokens-commas
```

## Single-GPU train
```python3 train.py```

## Multi-GPU (4 GPUs on one node)
```torchrun --standalone --nproc_per_node=4 train.py```


Important: Whatever max_tokens you build the dataset with must match the model max_tokens at train time. If you rebuild with a different length, rebuild both inputs.bin and labels.bin.
