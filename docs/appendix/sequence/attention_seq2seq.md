# Attention SEQ2SEQ

Attention SEQ2SEQ was introduced in the 2014 paper "Neural Machine Translation by Jointly Learning to Align and Translate." Attention mechanism to focus on relevant parts of input.

This implementation provides a concise, educational reference for Attention SEQ2SEQ. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## 코드

```python
#!/usr/bin/env python3
'''
Attention Mechanism in Seq2Seq
Paper: "Neural Machine Translation by Jointly Learning to Align and Translate" (2014)
Key: Attention mechanism to focus on relevant parts of input
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

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
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")```

## 논의

이 짜보기는 갈래 3개(`Attention`, `AttentionDecoder`, `AttentionSeq2Seq`)를 매기고, 이들이 어울려 온전한 이음 모형 얼개를 이룬다. 갈래마다 남다른 몫을 담아 코드를 묶음으로 나누고 넓히기 쉽게 한다. `forward` 방법이 PyTorch이 절로 미분하는 데 쓰는 셈 그림을 매긴다.

여기 실린 코드는 본보기 짜보기라 다듬기보다 알아보기 쉬움을 앞세운다. 서비스 얼개라면 흔히 섞인 촘촘함 익히기, 흩은 자료 나란히, 더 정교한 자료 불리기를 더한다. 그래도 여기서 보인 얼개의 고갱이 깨침은 크기와 상관없이 그대로다.

## 익힘 문제

**익힘 1.**
기본 첫자리로 잡은 `Attention`의 배울 수 있는 매개변수를 모두 세어라. 짐과 치우침을 아울러 켜마다 나누어 적어라.

??? success "익힘 1 풀이"
    `nn.Linear(in_features, out_features)`마다 짐 매개변수가 `in_features * out_features`개이고 치우침이 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 짐이 `in_c * out_c * k * k`개, 치우침이 `out_c`개다. `nn.Embedding(num, dim)`은 매개변수가 `num * dim`개다. 켜를 모두 더한다. `sum(p.numel() for p in model.parameters())`으로 따져 볼 수 있다.

---

**익힘 2.**
눈길 짐 뒤(값과 곱하기 앞)에 드롭아웃 켜를 더하여라. 익히는 동안 드롭아웃 비율 0.1을 써라. 눈길 드롭아웃이 다독임에 왜 도움이 되는지 밝혀라.

??? success "익힘 2 풀이"
    `__init__`에 `self.attn_dropout = nn.Dropout(0.1)`을 더하고 소프트맥스 뒤에 건다. `attn_weights = self.attn_dropout(F.softmax(scores, dim=-1))`. 눈길 드롭아웃은 익히는 동안 눈길 짐 얼마를 아무렇게나 0으로 만들어, 모형이 낱말끼리의 어떤 사이에만 기대는 것을 막는다. 그래서 모형이 눈길을 더 고루 나누고 더 든든한 드러냄을 배우게 되는데, 이는 여느 드롭아웃이 신경 낱자리끼리 함께 길드는 것을 막는 것과 같다.

---

**익힘 3.**
스스로 눈길의 셈 번거로움을 이음 길이 $n$과 모형 차수 $d$의 함수로 밝혀라. 이것이 긴 이음에 Longformer이나 Linformer 같은 얼개를 이끄는 까닭은 무엇인가?

??? success "익힘 3 풀이"
    여느 스스로 눈길은 $n \times n$ 눈길 행렬을 셈하므로 때는 $O(n^2 d)$, 눈길 짐의 기억은 $O(n^2)$이다. 이음이 길면($n = 4096$ 따위) 감당할 수 없다. Longformer은 그 자리 미닫이 창 눈길($O(n \cdot w \cdot d)$, $w$은 창 크기)과 고른 낱말에 대한 성긴 두루 눈길을 아울러 쓴다. Linformer은 열쇠와 값을 더 낮은 차수 $k \ll n$으로 되비추어 번거로움을 $O(n \cdot k \cdot d)$으로 줄인다. 둘 다 드러내는 힘을 얼마쯤 내주고 긴 들임에서 잘 들게 한다.

---

**익힘 4.**
`Attention`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 얼개를 짜라. 켜 2개, 4개, 8개로 시험하여라.

??? success "익힘 4 풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch이 매개변수를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = Attention(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.
