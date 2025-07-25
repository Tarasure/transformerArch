import torch
import torch.nn as nn
from phase3_transformer import TransformerBlock
from phase1_math_utils import Layer_norm

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, block_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(*[TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        token_emb = self.token_embedding(idx)
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos)
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
