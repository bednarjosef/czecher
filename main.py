import torch, wandb

from torch.utils.data import DataLoader

from dataset import CommaDataset
from memmap_dataset import CommaMemmapDataset
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
    # depth = 20
    # num_layers = 20
    # model_dim = 20 * 64 = 1280
    # num_heads = max(1, (model_dim + 127) // 128) = 1407 // 128 = 10 # head dim 128 (the division here is ceil div)
    # num_kv_heads = num_heads = 10

    epochs = 1
    batch_size = 512
    lr = 2e-4
    pos_weight = 1

    num_layers = 12
    model_dim = num_layers * 64
    num_heads = max(1, (model_dim + 127) // 128)  # ~128 head dim
    dim_ff = 4 * model_dim
    

    # run = wandb.init(
    #     entity="czecher-team",
    #     project="czecher-commas",
    #     config={
    #         "epochs": epochs,
    #         "batch_size": batch_size,
    #         "learning_rate": lr,
    #         "pos_weight": pos_weight,
    #         "layers": layers,
    #         "d_model": d_model,
    #         "num_heads": nhead,
    #         "dim_ff": dim_ff,
    #         "architecture": "Transformer",
    #         "dataset": "comma_memmap_5m"
    #     }
    # )

    tokenizer = GPTTokenizer(json_file='tokenizer.json')
    # dataset = CommaDataset().load_dataset(csv_path='data/bpe/dataset_500k.csv')
    # dataset = CommaMemmapDataset("./comma_memmap/inputs.bin", "./comma_memmap/labels.bin", max_tokens=128, pad_id=tokenizer.get_pad_token_id())
    model = CzecherTransformer(vocab_size=tokenizer.get_vocab_size(), pad_id=tokenizer.get_pad_token_id(), max_tokens=128, num_layers=num_layers, d_model=model_dim, embedding_dim=model_dim, nhead=num_heads, dim_ff=dim_ff)
    model = model.load('data/trained_models/5m_model_4gpu_2.pt', torch.device('cpu'))

    # best_threshold, progress = model.train_model(dataset, epochs=epochs, batch_size=batch_size, lr=lr, pos_weight=pos_weight, log_every=300, log_fn=run.log)
    # model.save('data/trained_models/5m_model1.pt')

    text = "V roce 1971 pak Salivarová a Škvorecký založili nakladatelství '68 Publishers kde pak vydávali především české knihy které nemohly vycházet v komunistickém Československu."
    text = "Teoreticko-lingvistické zkoumání se zakládá na obecné vědecké metodě: jeho výstupem jsou explicitní formálně zpracované teorie a hypotézy jejichž platnost je následně testována s pomocí dat z konkrétních jazyků."
    text = "Historická lingvistika zkoumá jazyky které se užívaly v minulosti (starou češtinu staroslověnštinu apod.) popř. se zaměřuje na výzkum vývoje jazyka v čase (viz níže: diachronní přístup)."
    text = "Jazyk je velmi mnohotvárný jev a v závislosti na tom z jakého úhlu k němu přistupujeme můžeme lingvistiku zařadit do několika širších disciplín."
    text = "Lingvistika z tohoto úhlu pohledu je tedy příbuzná vědám o nelingvistických znacích a prostředcích komunikace jako je neverbální komunikace různé pomocné komunikační systémy např. světelné či kouřové signály nebo znakové systémy uměle vytvářené pro potřebu komunikace s počítači a stroji."
    corrected = model.punctuate(text, tokenizer, threshold=0.45, device='cpu')
    print(text)
    print(corrected)

    # sentences = load_sentences(path='data/sentences_500k.txt')
    # dataset = CommaDataset().create_dataset(sentences, tokenizer, 'data/bpe/dataset_500k.csv', max_tokens=128)


if __name__ == '__main__':
    main()
