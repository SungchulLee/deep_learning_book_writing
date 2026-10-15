# 순환 신경망 기준 모델

두 층 LSTM의 마지막 은닉 상태로 실수 시퀀스를 분류하는 기준 모델이다. 시간축의 순차 의존과 고정 크기 상태라는 설계를 살펴본다. 다른 구조보다 항상 느리거나 적은 데이터로 학습된다고 단정할 수는 없다.

## 1. 코드

```python
import torch.nn as nn


class RNNBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, num_classes=10):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (hidden, _) = self.rnn(x)
        return self.classifier(hidden[-1])


if __name__ == "__main__":
    pass
```

## 2. 논의

입력은 $(B,L,64)$이고 batch_first=True로 처리한다. hidden은 $(\mathrm{num\_layers},B,H)$이며 hidden[-1]은 마지막 층의 최종 상태다. 셀 상태는 반환받지만 분류기에 사용하지 않는다. 분류 로짓은 $(B,10)$이다.

각 호출은 초기 상태를 받지 않아 기본 0 상태에서 시작한다. 이전 배치의 문맥을 자동으로 유지하지 않는다. 패딩 길이·packed sequence도 처리하지 않아 PAD가 있으면 최종 상태가 실제 마지막 토큰이 아닌 패딩까지 반영한다.

마지막 상태만 쓰는 방식과 전체 출력의 어텐션 풀링은 다른 요약 규칙이다. LSTM의 고정 상태가 유용한 정보를 얼마나 유지하는지는 데이터·학습에 달려 있으며, 배치·특징 연산은 병렬화할 수 있어 모든 계산이 직렬인 것은 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
기본 RNNBaseline의 두 LSTM층과 분류기 매개변수를 정확히 계산하라.

</div>

??? success "연습문제 1 풀이"
    ```python
    model = RNNBaseline(input_dim=64, hidden_dim=256, num_layers=2)
    params = sum(p.numel() for p in model.parameters())
    print(f"RNN parameters: {params:,}")
    ```
    입력 차원 I, 은닉 H의 단층은 $4H(I+H+2)$개다. 1층은 329,728개, 2층은 526,336개, 분류기는 2,570개로 총 858,634개다. 비교 트랜스포머는 4,783,370개로 구성과 깊이가 다르다. LSTM에 어텐션이 없다는 이유만으로 모든 폭·층 수에서 더 작다고 할 수는 없다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
양방향 변형의 마지막 상태 선택과 스트리밍 한계를 설명하라.

</div>

??? success "연습문제 2 풀이"
    ```python
    class BiRNNBaseline(nn.Module):
        def __init__(self, input_dim, hidden_dim=256, num_layers=2, num_classes=10):
            super().__init__()
            self.rnn = nn.LSTM(
                input_dim, hidden_dim, num_layers,
                batch_first=True, bidirectional=True
            )
            self.classifier = nn.Linear(hidden_dim * 2, num_classes)

        def forward(self, x):
            _, (hidden, _) = self.rnn(x)
            # 앞뒤 방향의 마지막 숨은 상태를 이어 붙인다
            h_fwd = hidden[-2]  # 마지막 층의 앞 방향
            h_bwd = hidden[-1]  # 마지막 층의 뒤 방향
            combined = torch.cat([h_fwd, h_bwd], dim=-1)
            return self.classifier(combined)
    ```
    hidden[-2]와 hidden[-1]은 마지막 층의 정방향·역방향 최종 상태로 각각 (B,256)이다. 둘을 이어 분류기 입력은 (B,512)가 된다. 역방향은 미래 입력을 필요로 하므로 원문 전체가 없는 실시간 인과 예측에 그대로 사용할 수 없다. torch를 직접 import하지 않은 이 페이지 조각은 torch.cat을 위해 추가 import도 필요하다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
어텐션 풀링 변형에 필요한 패딩 처리와 해석의 한계를 설명하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    class AttentiveRNN(nn.Module):
        def __init__(self, input_dim, hidden_dim=256, num_layers=2, num_classes=10):
            super().__init__()
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.attention = nn.Linear(hidden_dim, 1)
            self.classifier = nn.Linear(hidden_dim, num_classes)

        def forward(self, x):
            outputs, _ = self.rnn(x)  # (batch, seq_len, hidden_dim)
            scores = self.attention(outputs).squeeze(-1)  # (batch, seq_len)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            context = (outputs * weights).sum(dim=1)  # (batch, hidden_dim)
            return self.classifier(context)
    ```
    시점별 점수에 PAD 마스크를 적용한 뒤 소프트맥스해야 하고, LSTM 자체도 길이 정보를 처리해야 한다. 현재 AttentiveRNN에는 두 처리 모두 없으며 torch 이름도 별도 import가 필요하다. 모든 출력에 경로를 제공하지만 어텐션 가중치만으로 특정 시점의 인과적 중요도를 확정할 수는 없다.

## 정리하며

LSTM 기준 모델은 각 호출의 마지막 상태로 분류하며 패딩·스트리밍 처리는 별도로 필요하다. 양방향 상태·어텐션 풀링과 단순 최종 상태를 구분하고 계수·과제 성능·시간을 같은 조건에서 비교한다.
