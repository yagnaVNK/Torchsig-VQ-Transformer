import torch
import torch.nn as nn
import torch.nn.functional as F
from generative.networks.nets import DecoderOnlyTransformer

class MonaiDecoderOnlyModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers, vocab_size, max_len, block_size,additional_vocab):
        super().__init__()
        print("Initialized monai transformer with extended vocabulary")
        self.additional_vocab= additional_vocab
        self.vocab_size = vocab_size
        self.transformer = DecoderOnlyTransformer(
            num_tokens=vocab_size + additional_vocab,
            max_seq_len=max_len + 2,
            attn_layers_dim=d_model,
            attn_layers_depth=num_layers,
            attn_layers_heads=nhead,
            with_cross_attention=False,
            embedding_dropout_rate=0.1
        )
        self.block_size = block_size + 2
        self.lm_head = nn.Linear(d_model + additional_vocab , vocab_size + additional_vocab)

    def forward(self, idx):
        features = self.transformer(idx)
        logits = self.lm_head(features)
        return logits

    def generate(self, idx, max_new_tokens):
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.block_size:]
                logits = self(idx_cond)
                logits = logits[:, -1, :]
                logits[:, self.vocab_size  :] = float('-inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
        return idx