# 어텐션 Seq2Seq

인코더의 위치별 출력과 디코더 상태로 가산 어텐션을 계산하는 Seq2Seq 구조다. 가중합 문맥을 이전 토큰의 임베딩과 함께 GRU 디코더에 전달한다. 본문 실행부는 파라메터 수만 확인하며 모델을 학습하지 않는다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
Seq2Seq의 어텐션 장치
논문: "Neural Machine Translation by Jointly Learning to Align and Translate" (2014)
핵심: 입력의 쓸모 있는 부분에 집중하는 어텐션 장치
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
        # hidden: [1, 배치, hidden_size]
        # encoder_outputs: [배치, seq_len, hidden_size]
        
        batch_size = encoder_outputs.shape[0]
        seq_len = encoder_outputs.shape[1]
        
        # 디코더의 숨은 상태를 seq_len번 되풀이
        hidden = hidden.permute(1, 0, 2)  # [배치, 1, hidden_size]
        hidden = hidden.repeat(1, seq_len, 1)  # [배치, seq_len, hidden_size]
        
        # 어텐션 에너지 계산
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)  # [배치, seq_len]
        
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
        
        # 어텐션 웨이트 계산
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)  # [배치, 1, seq_len]
        
        # 문맥 벡터 계산
        context = torch.bmm(a, encoder_outputs)  # [배치, 1, hidden_size]
        
        # 임베딩한 입력과 문맥 이어 붙이기
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

이 인코더는 입력 임베딩 없이 LSTM을 사용하므로 src는 $(B,S,F)$의 실수 피처가다. 반면 trg는 $(B,T)$의 정수 토큰 ID다. input_size=100은 인코더 피처 수이고 output_size=100은 출력 어휘 크기이며, 같은 숫자여도 의미가 다르다.

Attention은 단일 레이어 상태 $(1,B,H)$를 입력 길이만큼 반복하고, 인코더 출력과 이어 붙여 점수를 계산한다. 소프트맥스 축은 입력 위치다. AttentionDecoder는 $(B,1)$의 토큰 하나와 상태를 받아 $(B,V)$ 로짓, 새 상태와 $(B,S)$ 웨이트를 반환한다.

인코더는 LSTM이고 디코더는 GRU이므로 인코더의 최종 셀 상태는 버리고 은닉 상태만 초기 디코더 상태로 사용한다. 여러 레이어로 일반화하려면 어텐션 쿼리와 초기 상태의 레이어 축을 따로 처리해야 한다.

패딩 마스크가 없어 패딩 위치도 정규화와 문맥 계산에 포함된다. trg의 한 칸 이동은 호출자가 처리해야 하며 이 코드는 이전 정답 토큰을 입력하는 계산 형태다. 자유 생성, 종료 조건과 실제 번역 평가는 포함되어 있지 않다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
은닉 크기 H인 Attention의 학습 파라메터 수를 구하라.

</div>

??? success "연습문제 1 풀이"
    Linear(2H,H)는 $2H^2+H$개, 편향 없는 Linear(H,1)은 H개다. 합계는 $2H^2+2H$개이고 H=256이면 131584개다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
소프트맥스 뒤에 드롭아웃을 적용하면 웨이트 합은 어떻게 되는가?

</div>

??? success "연습문제 2 풀이"
    학습 중 일부 웨이트가 0이 되고 유지된 값은 1/(1-p)배 되므로 한 번의 호출에서 합이 반드시 1인 것은 아니다. 기대값은 원래 웨이트와 같다. 평가 모드에서는 드롭아웃이 꺼진다. 규제 효과가 있을 수 있지만 더 고른 분포나 성능 향상을 보장하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
입력 길이 S, 출력 길이 T인 이 가산 어텐션의 비용을 설명하라.

</div>

??? success "연습문제 3 풀이"
    각 단계에서 모든 입력 위치의 결합 피처를 H차원으로 선형 변환하므로 현재 직접 구현의 점수 계산은 대략 $O(TSH^2)$다. 문맥 가중합은 $O(TSH)$다. 전체 디코더 단계의 웨이트를 저장하면 $O(TS)$이며, 이는 입력 길이만의 셀프 어텐션 $S^2$와 구분한다. 투영을 미리 계산하면 연산 구성이 달라질 수 있다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 풀이의 `ModuleList` 코드가 원래 모델에 바로 적용될 수 있는지 검토하라. 입력·출력 차원, 활성화, 생성자와 `forward`에서 필요한 수정을 설명하라.

</div>

??? success "연습문제 4 풀이"
    아래 코드는 깊이를 설정하는 아이디어를 위한 조각이다. 원래 모델의 생성자와 텐서 차원을 반영한 완성 구현은 아니다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `ModuleList`는 하위 모듈을 등록하지만 실행 순서나 활성화를 자동으로 정하지 않는다. 입력 레이어와 출력 레이어는 과제의 차원을 유지하고, 은닉 레이어 사이에 필요한 비선형성을 넣어야 한다. 선형 레이어만 반복하면 아핀 함수 하나로 합쳐진다. 원래 클래스에 없는 `num_layers` 인수를 그대로 넘길 수 없으므로 생성자와 `forward`를 함께 수정해야 한다.

## 정리하며

어텐션 Seq2Seq는 현재 디코더 상태로 입력 위치들을 참조한다. 이 구현의 src·trg 자료형, 단일 레이어 상태와 패딩 마스크 부재를 확인해야 한다. 모델 정의만으로 번역 성능이나 생성 동작이 검증된 것은 아니다.
