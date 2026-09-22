"""seq2seq_with_attention — seq2seq_with_attention 페이지의 코드를 모듈로 쓸 수 있게 옮겨 놓은 것이다.

이 파일은 ch10/transformer_architecture/seq2seq_with_attention.md 에서 자동으로 뽑아낸 것이므로, 고칠 일이 있으면
그 페이지를 고치고 다시 뽑아내는 편이 낫다.
"""

"""
주의를 쓰는 수열 대 수열 모형
"""
import torch
import torch.nn as nn
from attention_mechanisms import BahdanauAttention

# ========================================================================
# 메인
# ========================================================================

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
    
    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = BahdanauAttention(hidden_size)
        self.rnn = nn.GRU(embed_size + hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden, encoder_outputs):
        embedded = self.embedding(x)
        context, attn_weights = self.attention(hidden.squeeze(0), encoder_outputs)
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        output = self.fc(output.squeeze(1))
        return output, hidden, attn_weights

class Seq2SeqWithAttention(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, embed_size, hidden_size):
        super().__init__()
        self.encoder = Encoder(src_vocab, embed_size, hidden_size)
        self.decoder = Decoder(tgt_vocab, embed_size, hidden_size)
    
    def forward(self, src, tgt):
        encoder_outputs, hidden = self.encoder(src)
        outputs = []
        for t in range(tgt.size(1)):
            output, hidden, _ = self.decoder(tgt[:, t:t+1], hidden, encoder_outputs)
            outputs.append(output)
        return torch.stack(outputs, dim=1)


if __name__ == "__main__":
    pass
