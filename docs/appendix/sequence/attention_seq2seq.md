# 어텐션 Seq2Seq

어텐션 Seq2Seq은 2014년 논문 "Neural Machine Translation by Jointly Learning to Align and Translate"에서 나왔다. 디코더가 출력 토큰을 하나 낼 때마다 인코더 출력 전체에 웨이트를 매겨(어텐션 웨이트) 관련 있는 부분에 집중하게 하는 어텐션 메커니즘을 쓴다. 고정 길이 벡터 하나에 입력 전체를 압축하던 Seq2Seq의 병목을 푸는 장치다.

여기 구현은 어텐션 Seq2Seq을 짧고 트레인하기 좋게 보인 예제다. 코드는 핵심 구조와 순전파(forward pass)에 초점을 두어, 핵심 구성 요소를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
Seq2Seq의 어텐션 메커니즘
논문: "Neural Machine Translation by Jointly Learning to Align and Translate" (2014)
핵심: 입력에서 관련 있는 부분에 집중하게 하는 어텐션 메커니즘
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
        
        # 디코더의 은닉 상태를 seq_len 번 반복한다
        hidden = hidden.permute(1, 0, 2)  # [batch, 1, hidden_size]
        hidden = hidden.repeat(1, seq_len, 1)  # [batch, seq_len, hidden_size]
        
        # 어텐션 에너지(정렬 점수)를 계산한다
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
        
        # 어텐션 웨이트를 계산한다
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)  # [batch, 1, seq_len]
        
        # 문맥 벡터를 계산한다
        context = torch.bmm(a, encoder_outputs)  # [batch, 1, hidden_size]
        
        # 임베딩한 입력과 문맥 벡터를 이어 붙인다
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
```

**출력:**

```
Parameters: 1,140,836
```

## 2. 논의

이 구현은 클래스 3개(`Attention`, `AttentionDecoder`, `AttentionSeq2Seq`)를 정의하고, 이들이 함께 완전한 시퀀스 모델 구조를 이룬다. 클래스마다 고유한 역할을 맡아 코드를 모듈화하고 확장하기 쉽게 한다. `forward` 메서드가 PyTorch의 자동 미분에 쓰이는 계산 그래프를 정의한다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 트레인에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 트레인, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
기본 설정으로 초기화한 `Attention`의 트레인 가능한 파라메터를 모두 세어라. 웨이트와 바이어스를 포함해 레이어마다 나누어 적어라.

</div>

??? success "풀이"
    `nn.Linear(in_features, out_features)`마다 웨이트 파라메터가 `in_features * out_features`개이고 바이어스가 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 웨이트가 `in_c * out_c * k * k`개, 바이어스가 `out_c`개다. `nn.Embedding(num, dim)`은 파라메터가 `num * dim`개다. 모든 레이어의 값을 더한다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
어텐션 웨이트 뒤(값과 곱하기 앞)에 드롭아웃 레이어를 더하여라. 트레인하는 동안 드롭아웃 비율 0.1을 써라. 어텐션 드롭아웃이 정칙화에 왜 도움이 되는지 설명하라.

</div>

??? success "풀이"
    `__init__`에 `self.attn_dropout = nn.Dropout(0.1)`을 더하고 소프트맥스 뒤에 건다. `attn_weights = self.attn_dropout(F.softmax(scores, dim=-1))`. 어텐션 드롭아웃은 트레인하는 동안 어텐션 웨이트의 일부를 무작위로 0으로 만들어, 모델이 특정 토큰 사이의 관계에만 의존하는 것을 막는다. 그래서 모델이 어텐션을 더 고르게 분산하고 더 강건한 표현을 학습하게 되는데, 이는 표준 드롭아웃이 뉴런들의 공적응(co-adaptation)을 막는 것과 같다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
셀프 어텐션의 계산 복잡도를 시퀀스 길이 $n$과 모델 차원 $d$의 함수로 구하라. 이것이 긴 시퀀스에서 Longformer나 Linformer 같은 구조가 나온 동기가 되는 까닭은 무엇인가?

</div>

??? success "풀이"
    표준 셀프 어텐션은 $n \times n$ 어텐션 행렬을 계산하므로 시간 복잡도는 $O(n^2 d)$, 어텐션 웨이트의 메모리는 $O(n^2)$이다. 시퀀스가 길면($n = 4096$ 따위) 감당할 수 없다. Longformer는 국소 슬라이딩 윈도 어텐션($O(n \cdot w \cdot d)$, $w$는 윈도 크기)과 선택한 토큰에 대한 희소 전역 어텐션을 함께 쓴다. Linformer는 키와 값을 더 낮은 차원 $k \ll n$으로 사영해 복잡도를 $O(n \cdot k \cdot d)$로 줄인다. 둘 다 표현력을 얼마쯤 내주고 긴 입력에서의 효율을 얻는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`Attention`을 레이어나 블록의 수를 설정할 수 있게 확장하라. `__init__`에 `num_layers` 인자를 더하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 만들어라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩된 레이어를 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 반복한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`를 써야 PyTorch가 모든 파라메터를 옵티마이저에 등록한다. `for n in [2, 4, 8]: model = Attention(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`로 시험한다.

## 정리하며

**다룬 것** — 어텐션 Seq2Seq

이 구현은 클래스 3개(`Attention`, `AttentionDecoder`, `AttentionSeq2Seq`)를 정의하고, 이들이 함께 완전한 시퀀스 모델 구조를 이룬다.

핵심 클래스는 `Attention`, `AttentionDecoder`, `AttentionSeq2Seq`이며 앞의 연습문제 4개로 스스로 확인해 볼 수 있다.
