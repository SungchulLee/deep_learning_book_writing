# 순환 신경망 기준선

LSTM을 쓴 순환 신경망 기준선은 트랜스포머와 합성곱 신경망 구조에 견줄, 차례를 따르는 처리의 잣대를 준다. LSTM은 한 번에 한 걸음씩 수열을 처리하며 시간에 따른 의존을 담은 숨은 상태를 지킨다. 트랜스포머보다 학습이 느리지만 LSTM은 여러 수열 과제에서 여전히 튼튼한 기준선이다.

## 코드

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

## 논의

이 모형은 두 층짜리 LSTM을 쓰며 층마다 수열 전체를 차례로 처리하여 숨은 상태를 다음 층으로 넘긴다. `batch_first=True`는 입력의 꼴이 `(batch, seq_len, input_dim)`이라는 뜻이다. LSTM은 시간 단계마다의 출력과 마지막 숨은 상태와 세포 상태를 함께 돌려주는데, 이 모형은 마지막 층의 마지막 숨은 상태(`hidden[-1]`)만 수열의 표현으로 쓴다.

마지막 숨은 상태를 분류기의 입력으로 쓴다는 것은 모형이 수열 전체를 크기가 고정된 벡터 하나로 눌러 담아야 한다는 뜻이다. 그래서 긴 수열에서 정보 병목이 생기는데 이것이 주의 얼개가 나온 큰 이유 가운데 하나이다. 숨은 차원(256)과 층 수(2)가 모형의 그릇을 정한다. 더 깊은 LSTM은 더 복잡한 시간 무늬를 배울 수 있지만 기울기가 사라져 학습하기 더 어렵다.

트랜스포머에 견주면 LSTM에는 근본적인 한계가 둘 있다. 토큰을 차례로 처리하므로 학습 중에 병렬로 할 수 없고, 먼 거리에 정보를 나르는 일을 숨은 상태에 기대므로 잊어버릴 수 있다. 그러나 LSTM은 차례가 있는 데이터에 대한 강한 귀납 편향을 지니고, 잘 학습하는 데 데이터가 덜 들며, $n \times n$ 주의 행렬을 담아 두지 않으므로 기억도 덜 든다.

## 연습문제

**연습문제 1.**
`input_dim=64`, `hidden_dim=256`, `num_layers=2`인 이 `RNNBaseline`의 매개변수 수를 비슷한 차원의 트랜스포머 인코더와 견주어라. 어느 쪽이 매개변수가 더 많고 그 까닭은 무엇인가?

??? success "연습문제 1 풀이"
    ```python
    model = RNNBaseline(input_dim=64, hidden_dim=256, num_layers=2)
    params = sum(p.numel() for p in model.parameters())
    print(f"RNN parameters: {params:,}")
    ```
    LSTM 층은 문 넷에 대해 $4 \times (d_{\text{input}} \times d_{\text{hidden}} + d_{\text{hidden}}^2 + 2 \times d_{\text{hidden}})$개의 매개변수를 가진다. 1층은 $4 \times (64 \times 256 + 256 \times 256 + 512) = 330{,}752$이고 2층은 $4 \times (256 \times 256 + 256 \times 256 + 512) = 526{,}336$이다. 여기에 분류기 $256 \times 10 + 10 = 2{,}570$을 더한다. 모두 약 86만이다. $d_{\text{model}}=256$인 트랜스포머 인코더 층은 대략 80만 개인데, LSTM에는 따로 주의 매개변수가 없다.

---

**연습문제 2.**
양방향 LSTM을 쓰고 앞뒤 방향의 숨은 상태를 이어 붙이도록 모형을 고쳐라. 이 바꿈이 분류기의 입력 차원과 맥락을 잡아내는 능력에 어떤 영향을 주는가?

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
    양방향 LSTM은 수열을 두 방향으로 처리하여 숨은 상태의 크기를 $2 \times 256 = 512$으로 두 배로 만든다. 그래서 자리마다 지난 맥락과 앞으로의 맥락을 함께 잡아낼 수 있는데, BERT의 양방향 주의와 비슷하지만 여전히 차례를 따르는 처리이다.

---

**연습문제 3.**
마지막 숨은 상태를 쓰는 방식을 LSTM의 모든 출력에 대한 주의 기반 풀링으로 바꾸어라. 어느 시간 단계가 분류에 가장 중요한지 배우는 간단한 주의 얼개를 구현하라.

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
    주의 풀링은 분류 과제와 얼마나 관련되는지에 따라 시간 단계마다 가중치를 주는 법을 배워, 마지막 숨은 상태만 쓸 때의 정보 병목을 누그러뜨린다. 이는 나중에 트랜스포머의 온전한 자기 주의로 자라난 주의 얼개의 간단한 꼴이다.
