"""
Sequence-to-Sequence Model with Attention
"""
import torch
import torch.nn as nn
from attention_mechanisms import BahdanauAttention

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
