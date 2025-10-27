import torch
import torch.nn as nn

class CommaModel(nn.Module):
    def __init__(self, vocab_size, pad_id, bos_id, eos_id,
                 d_model=512, nhead=8, num_layers=6, dim_ff=2048, dropout=0.1):
        super().__init__()
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = nn.Embedding(4096, d_model)  # learned positions (simple & works)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, input_ids, attention_mask=None):
        # input_ids: [B, T], attention_mask: [B, T] (True=keep, False=pad)
        B, T = input_ids.shape
        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.embed(input_ids) + self.pos(pos_ids)

        # Transformer expects src_key_padding_mask=True for PADs
        if attention_mask is None:
            key_pad = input_ids.eq(self.pad_id)
        else:
            key_pad = ~attention_mask  # True where PAD
        h = self.encoder(x, src_key_padding_mask=key_pad)  # [B, T, H]
        logits = self.head(h).squeeze(-1)  # [B, T]
        return logits

def masked_bce_with_logits(logits, labels, input_ids, pad_id, bos_id, eos_id, pos_weight=1.0):
    """
    Compute BCE only on real character positions (not PAD/BOS/EOS).
    logits, labels: [B, T]
    """
    with torch.no_grad():
        mask = ~input_ids.eq(pad_id) & ~input_ids.eq(bos_id) & ~input_ids.eq(eos_id)
    if isinstance(pos_weight, (float, int)):
        pos_weight = torch.tensor(pos_weight, device=logits.device)
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits[mask], labels[mask], pos_weight=pos_weight
    )
    return loss, mask
