# YOLOv3

YOLOv3은 2018년 글 "YOLOv3: An Incremental Improvement"에서 나왔다. 한 방에 물체를 알아내며 잣대 셋에서 미루어 본다.

여기 짜보기는 YOLOv3을 짧고 배우기 좋게 보인 본이다. 코드는 고갱이 얼개와 앞으로 걸음에 마음을 두어, 고갱이 꾸밈새를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
YOLOv3 - 한 번만 본다 v3
논문: "YOLOv3: An Incremental Improvement" (2018)
고갱이: 한 방에 끝내는 물체 알아내기로 세 가지 자에서 예측한다
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x):
        return self.leaky(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels // 2, 1)
        self.conv2 = ConvBlock(channels // 2, channels, 3)
    
    def forward(self, x):
        return x + self.conv2(self.conv1(x))

class YOLOv3(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        # Darknet-53 등뼈(단순하게 만듦)
        self.conv1 = ConvBlock(3, 32, 3)
        self.conv2 = ConvBlock(32, 64, 3, stride=2)
        self.res1 = ResidualBlock(64)
        
        self.conv3 = ConvBlock(64, 128, 3, stride=2)
        self.res2 = nn.Sequential(*[ResidualBlock(128) for _ in range(2)])
        
        self.conv4 = ConvBlock(128, 256, 3, stride=2)
        self.res3 = nn.Sequential(*[ResidualBlock(256) for _ in range(8)])
        
        # 서로 다른 잣대의 알아내기 머리
        self.detect1 = nn.Conv2d(256, (5 + num_classes) * 3, 1)  # 3 anchors per scale
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x)
        x = self.conv3(x)
        x = self.res2(x)
        x = self.conv4(x)
        x = self.res3(x)
        
        detections = self.detect1(x)
        return detections

if __name__ == "__main__":
    model = YOLOv3()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## 2. 논의

이 짜보기는 갈래 3개(`ConvBlock`, `ResidualBlock`, `YOLOv3`)를 매기고, 이들이 어울려 온전한 물체 알아내기 얼개를 이룬다. 갈래마다 남다른 몫을 담아 코드를 묶음으로 나누고 넓히기 쉽게 한다. `forward` 방법이 PyTorch가 절로 미분하는 데 쓰는 셈 그림을 매긴다.

여기 실린 코드는 본보기 짜보기라 다듬기보다 알아보기 쉬움을 앞세운다. 서비스 얼개라면 흔히 섞인 촘촘함 익히기, 흩은 자료 나란히, 더 정교한 자료 불리기를 더한다. 그래도 여기서 보인 얼개의 고갱이 깨침은 크기와 상관없이 그대로다.

## 연습문제

**연습문제 1.**
`ConvBlock`의 앞으로 걸음을 따라 텐서의 꼴을 좇아라. 기본 매개변수로 들임 보기 4개를 묶어 넣었을 때, 큰 셈(엮음, 모으기, 선형 켜)마다 꼴이 어떻게 되는지 적어라.

??? success "연습문제 1 풀이"
    들임의 꼴에서 비롯해 켜를 차례로 건다. `Conv2d(in_c, out_c, k)`마다 자리 차수는 $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌고(덧대기 없이) `padding=k//2`이면 그대로다. 알갱이 2로 모으면 자리 차수가 반이 된다. 선형 켜는 마지막 차수를 바꾼다. 묶음 차수는 내내 그대로임을 좇아라. 엮음 켜에서는 $(B, C, H, W)$, 편 뒤에는 $(B, F)$으로 가운데 꼴을 적어라.

---

**연습문제 2.**
얼개를 크기 $64 \times 64$의 RGB 그림(들임 꼴: $3 \times 64 \times 64$)을 받도록 고쳐라. 켜의 차수를 모두 그에 맞게 손보고 모형이 어긋남 없이 도는지 따져라.

??? success "연습문제 2 풀이"
    첫 엮음 켜의 `in_channels`을 이제 값에서 3으로 바꾼다. 엮음과 모으기 켜마다 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$으로 자리 차수를 다시 셈한다. 첫 선형 켜의 `in_features`을 마지막 엮음/모으기 켜의 편 날임에 맞춘다. `model = ConvBlock(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`으로 따진다.

---

**연습문제 3.**
들임과 날임의 차수가 같을 때 여느 엮음과 깊이별로 가른 엮음의 매개변수 수와 뜨는 셈 횟수를 견주어라. 셈을 가장 크게 아끼는 때는 언제인가?

??? success "연습문제 3 풀이"
    여느 `Conv2d(C_in, C_out, k)`의 매개변수는 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개다. 깊이별로 가른 엮음은 이를 (1) 깊이별: $C_{{\text{{in}}}} \times k^2$개(들임 갈래마다 거르개 하나)와 (2) 점별: $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 엮음)로 나눈다. 매개변수의 견줌은 어림잡아 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 매개변수가 약 $8{-}9\times$ 적다. $C_{{\text{{out}}}}$과 $k$이 모두 클 때 가장 크게 아낀다.

---

**연습문제 4.**
`ConvBlock`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 얼개를 짜라. 켜 2개, 4개, 8개로 시험하여라.

??? success "연습문제 4 풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch가 매개변수를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = ConvBlock(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — YOLOv3

이 짜보기는 갈래 3개(`ConvBlock`, `ResidualBlock`, `YOLOv3`)를 매기고, 이들이 어울려 온전한 물체 알아내기 얼개를 이룬다.

고갱이 갈래는 `ConvBlock`, `ResidualBlock`, `YOLOv3`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
