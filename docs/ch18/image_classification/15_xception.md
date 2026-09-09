# Xception

Chollet의 2017년 논문에서 나온 Xception(극단 인셉션)은 인셉션 단원을 모두 깊이별로 갈라지는 누비기로 갈음해 인셉션 가설을 끝까지 밀어붙인다. 그 결과 더 단순하고 고른 얼개가 되며 ImageNet에서 인셉션 v3을 앞선다.

## 1. 코드

```python
import torch
import torch.nn as nn


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class Xception(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.sep_conv = SeparableConv2d(32, 64)
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.sep_conv(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.fc(x)


if __name__ == "__main__":
    model = Xception()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 2,052,392
```

## 2. 논의

Xception의 고갱이 눈썰미는 채널 사이의 상관과 공간 상관을 아주 따로 다룰 수 있다는 것이다. 깊이별 분리 합성곱은 먼저 채널마다 공간 거르개를 걸고(깊이별) 그다음 $1 \times 1$ 점별 합성곱으로 채널을 섞는다. 이는 채널마다 제 무리를 이루는 인셉션의 "끝까지 간" 갈래다.

Xception은 얼개를 들머리 흐름, 가운데 흐름, 날머리 흐름으로 짠다. 가운데 흐름은 되풀이되는 덩이 8개로 이루어지며 덩이마다 갈라지는 누비기 3개와 잔차 이음이 있다. 이 단순하고 단원별로 나뉜 꾸밈은 들쭉날쭉한 인셉션 단원보다 짜기도 다듬기도 쉽다.

## 연습문제

**연습문제 1.**
들임 채널 256개, 날임 채널 256개일 때 여느 $3 \times 3$ 합성곱과 깊이별 분리 합성곱의 매개변수 수를 견주어라.

??? success "연습문제 1 풀이"
    여느 합성곱: 매개변수 $256 \times 256 \times 9 = 589{,}824$개. 깊이별 분리: $256 \times 9 + 256 \times 256 = 2{,}304 + 65{,}536 = 67{,}840$개. 분리한 쪽이 매개변수가 약 $8.7\times$ 적다.

---

**연습문제 2.**
인셉션 방식의 갈라지는 누비기와 보통의 깊이별로 갈라지는 누비기의 차이를 설명하여라.

??? success "연습문제 2 풀이"
    여느 깊이별 분리 합성곱에서는 점별($1 \times 1$) 합성곱이 깊이별 합성곱 뒤에 온다. 인셉션에서는 채널 사이($1 \times 1$) 합성곱이 먼저 오고 공간 합성곱이 뒤따른다. Xception은 여느 차례(깊이별 다음 점별)를 쓰며 그편이 더 낫다고 보인다.

---

**연습문제 3.**
SeparableConv2d 덩이에 잔차 이음을 더하고, 지름길에 내리쬐기가 필요한 때를 논하여라.

??? success "연습문제 3 풀이"
    ```python
class SepConvWithResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.sep = SeparableConv2d(in_ch, out_ch, stride=stride)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    def forward(self, x):
        return self.sep(x) + self.shortcut(x)
```

들임과 내놓음 사이에 자리 차원(성큼 > 1)이나 채널 수가 바뀔 때마다 내리쬐기가 필요하다.

## 정리하며

**다룬 것** — Xception

Xception의 고갱이 눈썰미는 채널 사이의 상관과 공간 상관을 아주 따로 다룰 수 있다는 것이다.

고갱이 갈래는 `SeparableConv2d`, `Xception`, `SepConvWithResidual`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
