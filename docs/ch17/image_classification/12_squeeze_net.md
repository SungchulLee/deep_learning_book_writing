# SqueezeNet

SqueezeNet은 2016년 논문 "SqueezeNet: 매개변수 50배 적게 AlexNet 수준의 정확도"에서 나왔으며, 파이어 모듈 설계로 매개변수가 아주 적으면서도(~120만 개) 겨룰 만한 정확도를 이룬다. 파이어 모듈은 쥐어짜기 켜($1 \times 1$ 합성곱)로 채널 수를 줄인 뒤 $1 \times 1$과 $3 \times 3$ 합성곱을 나란히 걸어 넓힌다.

## 1. 코드

```python
import torch
import torch.nn as nn


class Fire(nn.Module):
    def __init__(self, in_channels, squeeze_channels,
                 expand1x1_channels, expand3x3_channels):
        super().__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(96, 16, 64, 64),
            Fire(128, 16, 64, 64),
            Fire(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 32, 128, 128),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


if __name__ == "__main__":
    model = SqueezeNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 646,344
```

## 2. 논의

파이어 모듈이 SqueezeNet을 SqueezeNet이게 하는 새로움이다. 쥐어짜기 켜가 채널 수를 작은 값(예: 16)으로 줄이고, 넓히기 켜가 $1 \times 1$과 $3 \times 3$ 합성곱을 나란히 걸어 그 날임을 이어 붙인다. 이 쥐어짜고 넓히는 무늬가 표현력을 지키면서 매개변수를 크게 줄인다.

SqueezeNet은 얼개를 잘 꾸미면 매개변수를 50분의 1만 쓰고도 AlexNet 수준의 정확도를 낼 수 있음을 보여 준다. 깊은 눌러 담기 재주로 줄이면 모델이 0.5 MB 아래로 작아져 자원이 극도로 빠듯한 기기에도 펼칠 수 있다.

## 연습문제

**연습문제 1.**
squeeze=16, expand1x1=64, expand3x3=64, 들임 채널=128인 파이어 단원의 매개변수 개수를 셈하여라.

??? success "연습문제 1 풀이"

    - Squeeze: $128 \times 16 + 16 = 2{,}064$
    - Expand $1 \times 1$: $16 \times 64 + 64 = 1{,}088$
    - Expand $3 \times 3$: $16 \times 64 \times 9 + 64 = 9{,}280$

    모두: 파이어 단원마다 매개변수 $12{,}432$개.

---

**연습문제 2.**
expand1x1=64, expand3x3=64인 파이어 단원이 내놓는 채널 수는 얼마인가?

??? success "연습문제 2 풀이"
    부풀림 가지 둘을 채널 차원으로 이어 붙이므로 내놓는 채널은 `expand1x1 + expand3x3 = 64 + 64 = 128`이다.

---

**연습문제 3.**
파이어 단원에 잔차 이음을 더하여라(에움길을 갖춘 SqueezeNet). 어떤 조건에서 건너뛰는 이음을 더할 수 있는가?

??? success "연습문제 3 풀이"
    `in_channels == expand1x1 + expand3x3`일 때, 곧 들임과 내놓음의 채널 수가 맞을 때 건너뛰는 이음을 더할 수 있다:

    ```python
    class FireWithBypass(Fire):
        def forward(self, x):
            out = super().forward(x)
            if x.shape[1] == out.shape[1]:
                return out + x
            return out
    ```

    에움길 이음은 기울기 흐름을 낫게 하며 매개변수를 늘리지 않고도 정확도를 2~4% 올릴 수 있다.

## 정리하며

**다룬 것** — SqueezeNet

파이어 모듈이 SqueezeNet을 SqueezeNet이게 하는 새로움이다.

고갱이 갈래는 `Fire`, `SqueezeNet`, `FireWithBypass`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
