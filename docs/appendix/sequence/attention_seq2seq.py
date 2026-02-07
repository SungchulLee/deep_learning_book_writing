#!/usr/bin/env python3
'''
Attention Mechanism in Seq2Seq
Paper: "Neural Machine Translation by Jointly Learning to Align and Translate" (2014)
Key: Attention mechanism to focus on relevant parts of input
'''
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs):
        # hidden: [1, batch, hidden_size]
        # encoder_outputs: [batch, seq_len, hidden_size]
        
        batch_size = encoder_outputs.shape[0]
        seq_len = encoder_outputs.shape[1]
        
        # Repeat decoder hidden state seq_len times
        hidden = hidden.permute(1, 0, 2)  # [batch, 1, hidden_size]
        hidden = hidden.repeat(1, seq_len, 1)  # [batch, seq_len, hidden_size]
        
        # Calculate attention energies
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)  # [batch, seq_len]
        
        return torch.nn.functional.softmax(attention, dim=1)

class AttentionDecoder(nn.Module):
    def __init__(self, output_size, hidden_size):
        super().__init__()
        self.attention = Attention(hidden_size)
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.GRU(hidden_size * 2, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        
        # Calculate attention weights
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)  # [batch, 1, seq_len]
        
        # Calculate context vector
        context = torch.bmm(a, encoder_outputs)  # [batch, 1, hidden_size]
        
        # Concatenate embedded input and context
        rnn_input = torch.cat((embedded, context), dim=2)
        
        output, hidden = self.rnn(rnn_input, hidden)
        prediction = self.fc(output.squeeze(1))
        
        return prediction, hidden, a.squeeze(1)

class AttentionSeq2Seq(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = AttentionDecoder(output_size, hidden_size)
    
    def forward(self, src, trg):
        encoder_outputs, (hidden, _) = self.encoder(src)
        
        outputs = []
        for t in range(trg.shape[1]):
            output, hidden, _ = self.decoder(trg[:, t:t+1], hidden, encoder_outputs)
            outputs.append(output)
        
        return torch.stack(outputs, dim=1)

if __name__ == "__main__":
    model = AttentionSeq2Seq(input_size=100, output_size=100)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
