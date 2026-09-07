# Seq2Seq

Seq2Seq was introduced in the 2014 paper "Sequence to Sequence Learning with Neural Networks." Encoder-Decoder architecture for sequence transduction.

This implementation provides a concise, educational reference for Seq2Seq. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## 코드

```python
#!/usr/bin/env python3
'''
Seq2Seq - Sequence to Sequence Learning with Neural Networks
Paper: "Sequence to Sequence Learning with Neural Networks" (2014)
Key: Encoder-Decoder architecture for sequence transduction
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
    
    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden, cell):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256, num_layers=2):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.decoder = Decoder(output_size, hidden_size, num_layers)
    
    def forward(self, src, trg):
        hidden, cell = self.encoder(src)
        outputs, _, _ = self.decoder(trg, hidden, cell)
        return outputs

if __name__ == "__main__":
    model = Seq2Seq(input_size=1000, output_size=1000)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")```

## 논의

이 짜보기는 갈래 3개(`Encoder`, `Decoder`, `Seq2Seq`)를 매기고, 이들이 어울려 온전한 이음 모형 얼개를 이룬다. 갈래마다 남다른 몫을 담아 코드를 묶음으로 나누고 넓히기 쉽게 한다. `forward` 방법이 PyTorch이 절로 미분하는 데 쓰는 셈 그림을 매긴다.

여기 실린 코드는 본보기 짜보기라 다듬기보다 알아보기 쉬움을 앞세운다. 서비스 얼개라면 흔히 섞인 촘촘함 익히기, 흩은 자료 나란히, 더 정교한 자료 불리기를 더한다. 그래도 여기서 보인 얼개의 고갱이 깨침은 크기와 상관없이 그대로다.

## 익힘 문제

**익힘 1.**
기본 첫자리로 잡은 `Encoder`의 배울 수 있는 매개변수를 모두 세어라. 짐과 치우침을 아울러 켜마다 나누어 적어라.

??? success "익힘 1 풀이"
    `nn.Linear(in_features, out_features)`마다 짐 매개변수가 `in_features * out_features`개이고 치우침이 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 짐이 `in_c * out_c * k * k`개, 치우침이 `out_c`개다. `nn.Embedding(num, dim)`은 매개변수가 `num * dim`개다. 켜를 모두 더한다. `sum(p.numel() for p in model.parameters())`으로 따져 볼 수 있다.

---

**익힘 2.**
들임의 꼴과 자료 갈래가 바라는 대로인지 살피는 들임 살피기를 으뜸 함수나 갈래에 더하여라. 올바르지 않은 들임에는 알아듣기 쉬운 어긋남 알림을 띄워라.

??? success "익힘 2 풀이"
    `forward` 방법(또는 알맞은 함수)의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 살핌을 더한다. 꼴을 살피려면 종요로운 차수를 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아듣기 쉬운 어긋남 알림은 벌레잡기를 크게 앞당기고 코드를 되쓰기 든든하게 한다.

---

**익힘 3.**
숨은 크기 $h$과 들임 크기 $x$이 같을 때 LSTM 칸과 GRU 칸의 매개변수 수를 견주어라. 어느 쪽이 더 적고 왜 그런가?

??? success "익힘 3 풀이"
    LSTM에는 문이 넷(들임, 잊음, 칸, 날임) 있고 저마다 들임과 숨은 상태의 짐 행렬을 지니므로 매개변수는 $4 \times (x \cdot h + h \cdot h + h) = 4(xh + h^2 + h)$개다. GRU에는 문이 셋(되돌림, 고침, 새것) 있어 $3 \times (x \cdot h + h \cdot h + h) = 3(xh + h^2 + h)$개다. GRU은 문이 넷이 아니라 셋이고 칸 상태와 숨은 상태를 하나로 묶으므로 LSTM의 75%이다. 참으로는 매개변수가 적어도 GRU이 LSTM과 엇비슷하게 잘 듣는 일이 잦다.

---

**익힘 4.**
`Encoder`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 얼개를 짜라. 켜 2개, 4개, 8개로 시험하여라.

??? success "익힘 4 풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch이 매개변수를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = Encoder(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.
