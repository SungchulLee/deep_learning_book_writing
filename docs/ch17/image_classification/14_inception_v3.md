# 인셉션 V3

인셉션 v3은 2015년 논문 "인셉션 구조 다시 보기"에서 나왔으며, 합성곱을 어긋난 짝(예: $n \times 1$과 $1 \times n$)으로 나누고 레이블 스무딩 정칙화를 더하고 보조 분류기를 다듬어 본디 GoogLeNet을 낫게 한다.

## 코드

```python
import torch
import torch.nn as nn


class InceptionV3(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, bias=False)
        self.conv2 = nn.Conv2d(32, 32, 3, bias=False)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.fc(x)


if __name__ == "__main__":
    model = InceptionV3()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## 논의

나눈 합성곱이 인셉션 v3의 고갱이 나아짐이다. $5 \times 5$ 합성곱을 $3 \times 3$ 합성곱 둘로 갈음하고, 큰 합성곱은 다시 어긋난 $1 \times n$과 $n \times 1$ 짝으로 나눈다. 그러면 매개변수와 셈이 줄면서 그물의 깊이와 비선형은 늘어난다.

딱 떨어지는 원핫 목표를 부드러운 분포(보기로 참 갈래에 $0.9$, 나머지에 $0.1/(K-1)$)로 갈음하는 이름표 부드럽게 하기는 지나치게 자신하는 어림을 막고 두루 통함을 낫게 한다. 인셉션 v3은 ImageNet에서 상위 5 어긋남을 4% 아래로 낮춘다.

## 연습문제

**연습문제 1.**
들임 채널 256개, 날임 채널 256개인 $5 \times 5$ 합성곱을 $3 \times 3$ 합성곱 둘로 나눌 때 아끼는 매개변수를 셈하여라.

??? success "연습문제 1 풀이"
    본디 $5 \times 5$: 매개변수 $256 \times 256 \times 25 = 1{,}638{,}400$개. $3 \times 3$ 둘: $256 \times 256 \times 9 \times 2 = 1{,}179{,}648$개. 약 28% 아낀다.

---

**연습문제 2.**
계수 $\epsilon = 0.1$인 레이블 스무딩이 1000갈래 문제의 목표 분포를 어떻게 바꾸는지 밝혀라.

??? success "연습문제 2 풀이"
    참 갈래의 목표가 1.0이 아니라 $1 - \epsilon = 0.9$이 되고 나머지 갈래마다 $\epsilon / (K-1) = 0.1/999 \approx 0.0001$을 받는다. 그래서 모델이 지나치게 자신하지 않게 되고 눈금이 잘 맞는다.

---

**연습문제 3.**
어긋난 합성곱 나누기를 짜라. $7 \times 7$ 합성곱을 $1 \times 7$ 다음 $7 \times 1$ 합성곱으로 갈음하여라.

??? success "연습문제 3 풀이"
    ```python
class AsymmetricConv(nn.Module):
    def __init__(self, in_ch, out_ch, n=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (1, n), padding=(0, n//2), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, (n, 1), padding=(n//2, 0), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)
```

매개변수 아낌: 채널 짝마다 $7 \times 7 = 49$ 대 $7 + 7 = 14$이니 $3.5\times$ 줄어든다.
