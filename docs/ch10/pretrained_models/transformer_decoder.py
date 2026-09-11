"""transformer_decoder — transformer_decoder 페이지의 코드를 모듈로 쓸 수 있게 옮겨 놓은 것이다.

이 파일은 ch10/pretrained_models/transformer_decoder.md 에서 자동으로 뽑아낸 것이므로, 고칠 일이 있으면
그 페이지를 고치고 다시 뽑아내는 편이 낫다.
"""

import torch
import torch.nn as nn


class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 가린 자기 주의
        attn_output, _ = self.self_attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 순전파
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers,
                 d_ff, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, d_model))

        self.layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]

        for layer in self.layers:
            x = layer(x, mask)

        return self.fc_out(x)


if __name__ == "__main__":
    pass
