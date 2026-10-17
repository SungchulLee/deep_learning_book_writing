# Seq2Seq

입력 토큰열을 LSTM 인코더의 최종 상태로 요약하고, 그 상태를 LSTM 디코더에 전달하는 기본 Seq2Seq 예제다. 본문 실행부는 모델의 파라메터 수만 출력하며 번역 데이터 학습이나 자유 생성은 수행하지 않는다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
Seq2Seq — 뉴럴 넷트웍으로 하는 순차열 대 순차열 학습
논문: "Sequence to Sequence Learning with Neural Networks" (2014)
핵심: 순차열 변환을 위한 인코더-디코더 구조
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
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 2,874,344
```

## 2. 논의

Encoder의 input_size는 연속 피처 수가 아니라 입력 어휘 크기다. 정수 ID 형태 $(B,S)$를 임베딩한 뒤 인코더의 은닉·셀 상태를 반환한다. 인코더와 디코더는 같은 레이어수와 은닉 차원을 사용하므로 상태를 직접 전달할 수 있다.

Decoder는 전달된 타깃 토큰열 전체를 입력받고 각 위치의 로짓을 계산한다. 학습 시 정답을 그대로 같은 위치의 입력과 타깃으로 쓰면 답을 미리 보게 되므로, 시작 토큰을 포함한 이전 토큰열을 입력으로 쓰고 한 칸 이동한 다음 토큰을 정답으로 만들어야 한다. 이 정렬은 현재 forward가 자동으로 수행하지 않는다.

패딩과 실제 길이, 종료 토큰, 로스·옵티마이저와 검증 분할은 구현되어 있지 않다. 자유 생성에는 시작 토큰에서 시작해 예측 토큰을 다음 입력으로 전달하는 별도 루프가 필요하다. 현재 호출은 타깃 길이에 따른 로짓을 반환할 뿐 생성 전략을 구현한 것이 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
입력 어휘 V, 은닉 크기 h, 레이어수 L인 Encoder의 파라메터 수를 구하라.

</div>

??? success "풀이"
    임베딩은 Vh개다. 모든 순환 레이어의 입력·은닉 크기가 h이고 기본 두 바이어스를 사용하므로 레이어마다 $4h(2h+2)$개다. 총 $Vh+L\,4h(2h+2)$개이며 디코더와 출력 레이어는 제외한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
src와 trg의 형태·자료형, 디코더 정답의 정렬을 설명하라.

</div>

??? success "풀이"
    src는 $(B,S)$, trg 입력은 $(B,T)$의 long ID이며 각각 어휘 범위에 있어야 한다. 디코더 입력은 이전 정답 토큰, 로스의 타깃은 그 다음 토큰으로 정렬한다. 출력 로짓은 $(B,T,V_{out})$이고 패딩 위치는 로스에서 제외한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
본문의 Seq2Seq forward만으로 새로운 번역을 생성할 수 있는가?

</div>

??? success "풀이"
    새로운 타깃을 스스로 선택하는 루프가 없으므로 충분하지 않다. 인코더 상태와 시작 토큰에서 시작해 단계별 로짓을 선택하고 상태·토큰을 갱신해야 한다. 종료 토큰과 최대 길이도 정해야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 풀이의 `ModuleList` 코드가 원래 모델에 바로 적용될 수 있는지 검토하라. 입력·출력 차원, 활성화, 생성자와 `forward`에서 필요한 수정을 설명하라.

</div>

??? success "풀이"
    아래 코드는 깊이를 설정하는 아이디어를 위한 조각이다. 원래 모델의 생성자와 텐서 차원을 반영한 완성 구현은 아니다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `ModuleList`는 하위 모듈을 등록하지만 실행 순서나 활성화를 자동으로 정하지 않는다. 입력 레이어와 출력 레이어는 과제의 차원을 유지하고, 은닉 레이어 사이에 필요한 비선형성을 넣어야 한다. 선형 레이어만 반복하면 아핀 함수 하나로 합쳐진다. 원래 클래스에 없는 `num_layers` 인수를 그대로 넘길 수 없으므로 생성자와 `forward`를 함께 수정해야 한다.

## 정리하며

기본 Seq2Seq는 인코더 상태를 디코더로 전달한다. 학습 입력의 한 칸 이동과 패딩 제외는 외부에서 맞춰야 하며, 현재 코드에는 학습과 자유 생성 루프가 없다.
