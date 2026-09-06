# 합성곱 신경망 기준선

1차원 합성곱 신경망은 수열 모형화 과제에서 트랜스포머와 순환 신경망 구조에 견줄 기준선 노릇을 한다. 합성곱 신경망은 가중치를 함께 쓰는 국소 수용 영역으로 수열을 처리하여, 계산이 효율적이고 글 분류나 시계열 분석처럼 국소 무늬가 뜻있는 과제에서 좋은 성능을 낸다.

## 코드

```python
import torch.nn as nn


class CNNBaseline(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, C, L]
        x = self.conv_layers(x)
        x = x.squeeze(-1)
        return self.classifier(x)


if __name__ == "__main__":
    pass
```

## 논의

이 구조는 합성곱 블록 셋으로 이루어지며 블록마다 핵 크기가 3인 1차원 합성곱, ReLU 활성, 최대 풀링을 적용한다. 통로 차원이 입력 차원에서 64, 128, 256으로 차츰 늘어 층마다 점점 더 추상적인 특징을 배우게 한다. `padding=1`이 합성곱 뒤에도 수열 길이를 지켜 주므로 시간 차원을 줄이는 것은 최대 풀링 층뿐이다.

마지막 `AdaptiveAvgPool1d(1)` 층이 입력 길이와 무관하게 수열을 벡터 하나로 줄여 주어 추론 때 수열 길이에 대해 모형이 자유로워진다. 이 전역 평균 풀링은 크기가 고정된 펴기 연산을 대신하며 분류 머리로 넘어가는 256차원 표현을 낸다.

입력 텐서를 파이토치의 Conv1d 관례에 맞추려고 `(batch, length, channels)`에서 `(batch, channels, length)`로 옮겨 놓는다. 본디 순환 신경망이나 트랜스포머에 맞추어 놓은 수열 데이터를 합성곱 신경망에 쓸 때 흔한 방식이다. 트랜스포머에 견주면 합성곱 신경망은 받는 영역이 핵의 크기와 신경망의 깊이에 매여 좁지만 매개변수가 훨씬 적게 들고 짧은 수열에서 더 빠르다.

## 연습문제

**연습문제 1.**
(풀링을 빼고) 합성곱 층 셋을 모두 지난 뒤 이 합성곱 신경망이 받는 영역을 셈하라. 그 영역은 트랜스포머의 전역 주의와 견주어 어떠한가?

??? success "연습문제 1 풀이"
    핵 크기가 3인 `Conv1d`마다 앞 층보다 받는 영역이 2씩 는다. 1층은 3, 2층은 $3 + 2 = 5$, 3층은 $5 + 2 = 7$이다. (저마다 실효 영역을 두 배로 만드는) 최대 풀링 층 둘까지 넣으면 실제로 받는 영역은 $7 \times 4 = 28$자리이다. 첫 층부터 수열 전체를 아우르는 트랜스포머의 전역 주의보다 훨씬 좁다. 합성곱 신경망은 견줄 만한 영역을 얻으려면 층을 많이 쌓아야 한다.

---

**연습문제 2.**
합성곱마다 그 뒤(ReLU 앞)에 배치 정규화를 더하고 있을 때와 없을 때의 학습 수렴을 견주어라. 배치 정규화가 합성곱 신경망 학습에 왜 도움이 되는지 설명하라.

??? success "연습문제 2 풀이"
    ```python
    self.conv_layers = nn.Sequential(
        nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Conv1d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Conv1d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
    )
    ```
    배치 정규화는 배치에 걸쳐 활성을 정규화하여 안쪽 공변량 이동을 줄인다. 그래서 학습률을 더 높일 수 있고 규제 효과가 있으며 대체로 더 빨리 수렴하고 더 잘 일반화된다.

---

**연습문제 3.**
최대 풀링 층을 걸음이 있는 합성곱(`stride=2, kernel_size=3, padding=1`)으로 바꾸고 매개변수 수와 성능을 견주어라. 학습되는 줄이기와 고정된 줄이기의 개념 차이는 무엇인가?

??? success "연습문제 3 풀이"
    ```python
    self.conv_layers = nn.Sequential(
        nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),  # 학습되는 줄이기
        nn.ReLU(),
        nn.Conv1d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv1d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
    )
    ```
    걸음이 있는 합성곱은 줄이는 방법을 배우므로 고정된 최댓값 연산보다 정보를 더 지킬 수 있다. 최대 풀링은 가장 센 활성을 고르는데, 옮김에 대한 불변성을 주지만 다른 정보를 버린다. 걸음이 있는 합성곱은 매개변수가 늘지만 과제에 맞는 줄이기 방식을 배울 수 있다.
