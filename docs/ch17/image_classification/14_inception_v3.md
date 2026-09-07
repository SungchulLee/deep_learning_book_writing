# 인셉션 V3

Inception v3, from the 2015 paper "Rethinking the Inception Architecture," improves upon the original GoogLeNet by factorizing convolutions into asymmetric pairs (e.g., $n \times 1$ and $1 \times n$), adding label smoothing regularization, and refining auxiliary classifiers.

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

Factorized convolutions are Inception v3's key improvement. A $5 \times 5$ convolution is replaced by two $3 \times 3$ convolutions, and large convolutions are further factorized into asymmetric $1 \times n$ and $n \times 1$ pairs. This reduces parameters and computation while increasing the depth and non-linearity of the network.

딱 떨어지는 원핫 목표를 부드러운 분포(보기로 참 갈래에 $0.9$, 나머지에 $0.1/(K-1)$)로 갈음하는 이름표 부드럽게 하기는 지나치게 자신하는 어림을 막고 두루 통함을 낫게 한다. 인셉션 v3은 ImageNet에서 상위 5 어긋남을 4% 아래로 낮춘다.

## 연습문제

**연습문제 1.**
Compute the parameter savings when factorizing a $5 \times 5$ convolution with 256 input and 256 output channels into two $3 \times 3$ convolutions.

??? success "연습문제 1 풀이"
    Original $5 \times 5$: $256 \times 256 \times 25 = 1{,}638{,}400$ parameters. Two $3 \times 3$: $256 \times 256 \times 9 \times 2 = 1{,}179{,}648$ parameters. Savings: about 28%.

---

**연습문제 2.**
Explain how label smoothing with factor $\epsilon = 0.1$ changes the target distribution for a 1000-class problem.

??? success "연습문제 2 풀이"
    The true class target becomes $1 - \epsilon = 0.9$ instead of 1.0, and every other class gets $\epsilon / (K-1) = 0.1/999 \approx 0.0001$. This discourages the model from becoming overconfident and improves calibration.

---

**연습문제 3.**
Implement asymmetric convolution factorization: replace a $7 \times 7$ convolution with a $1 \times 7$ followed by $7 \times 1$ convolution.

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

Parameter savings: $7 \times 7 = 49$ vs $7 + 7 = 14$ per channel pair, a $3.5\times$ reduction.
