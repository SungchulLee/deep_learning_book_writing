#!/usr/bin/env python3
'''
GPT - Generative Pre-trained Transformer
Paper: "Improving Language Understanding by Generative Pre-Training" (2018)
Key: Unidirectional transformer, autoregressive generation
'''
import torch
import torch.nn as nn

class GPTBlock(nn.Module):
    def __init__(self, d_model=768, n_heads=12):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        attn_mask = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).bool().to(x.device)
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.mlp(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size=50257, d_model=768, n_layers=12, n_heads=12, max_len=1024):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        
        self.blocks = nn.Sequential(*[GPTBlock(d_model, n_heads) for _ in range(n_layers)])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, x):
        b, t = x.size()
        pos = torch.arange(0, t, dtype=torch.long, device=x.device).unsqueeze(0)
        
        tok_emb = self.token_emb(x)
        pos_emb = self.pos_emb(pos)
        x = tok_emb + pos_emb
        
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits

if __name__ == "__main__":
    model = GPT()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
