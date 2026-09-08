# ResNeSt

ResNeSt(가른 눈길 그물)은 2020년에 나왔으며, 나머지 덩이마다 여러 결 그림 무리에 갈래별 눈길을 거는 가른 눈길 덩이를 넣어 ResNet을 넓혔다. SE-Net과 SK-Net에서 실마리를 얻어, ResNeSt은 갈래를 넘나드는 주고받음을 더 잘게 담아, 셈 값을 크게 늘리지 않고도 그림 가름, 물체 알아내기, 뜻 나누기 같은 일의 드러냄 배우기를 낫게 한다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
ResNeSt - 나눔 눈길 그물
논문: "ResNeSt: Split-Attention Networks" (2020)
고갱이: 특징 무리를 가로질러 채널마다 눈길을 두는 나눔 눈길 덩이
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class SplitAttention(nn.Module):
    def __init__(self, channels, radix=2, groups=1):
        super().__init__()
        self.radix = radix
        self.groups = groups
        inter_channels = max(channels * radix // 4, 32)
        
        self.fc1 = nn.Conv2d(channels, inter_channels, 1, groups=groups)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(inter_channels, channels * radix, 1, groups=groups)
        self.rsoftmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        batch = x.size(0)
        x = x.view(batch, self.radix, -1, *x.shape[2:])
        gap = x.sum(dim=1)
        gap = nn.functional.adaptive_avg_pool2d(gap, 1)
        
        atten = self.fc1(gap)
        atten = self.bn1(atten)
        atten = self.relu(atten)
        atten = self.fc2(atten)
        atten = atten.view(batch, self.radix, -1)
        atten = self.rsoftmax(atten)
        atten = atten.view(batch, self.radix, -1, 1, 1)
        
        out = (x * atten).sum(dim=1)
        return out

class ResNeStBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, radix=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels * radix, 3, stride, 1, groups=radix, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * radix)
        self.split_attention = SplitAttention(out_channels, radix)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.split_attention(out)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return self.relu(out)

class ResNeSt(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(64, 64, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        for i in range(blocks):
            stride = 2 if i == 0 and in_channels != 64 else 1
            layers.append(ResNeStBlock(in_channels if i == 0 else out_channels * 4, out_channels, stride))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

if __name__ == "__main__":
    model = ResNeSt()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 424,936
```

## 2. 논의

이 짜보기는 갈래 3개(`SplitAttention`, `ResNeStBlock`, `ResNeSt`)를 매기고, 이들이 어울려 온전한 엮음 신경 그물 얼개를 이룬다. 갈래마다 남다른 몫을 담아 코드를 묶음으로 나누고 넓히기 쉽게 한다. `forward` 방법이 PyTorch가 절로 미분하는 데 쓰는 셈 그림을 매긴다.

여기 실린 코드는 본보기 짜보기라 다듬기보다 알아보기 쉬움을 앞세운다. 서비스 얼개라면 흔히 섞인 촘촘함 익히기, 흩은 자료 나란히, 더 정교한 자료 불리기를 더한다. 그래도 여기서 보인 얼개의 고갱이 깨침은 크기와 상관없이 그대로다.

## 연습문제

**연습문제 1.**
`SplitAttention`의 앞으로 걸음을 따라 텐서의 꼴을 좇아라. 기본 매개변수로 들임 보기 4개를 묶어 넣었을 때, 큰 셈(엮음, 모으기, 선형 켜)마다 꼴이 어떻게 되는지 적어라.

??? success "연습문제 1 풀이"
    들임의 꼴에서 비롯해 켜를 차례로 건다. `Conv2d(in_c, out_c, k)`마다 자리 차수는 $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌고(덧대기 없이) `padding=k//2`이면 그대로다. 알갱이 2로 모으면 자리 차수가 반이 된다. 선형 켜는 마지막 차수를 바꾼다. 묶음 차수는 내내 그대로임을 좇아라. 엮음 켜에서는 $(B, C, H, W)$, 편 뒤에는 $(B, F)$으로 가운데 꼴을 적어라.

---

**연습문제 2.**
얼개를 크기 $64 \times 64$의 RGB 그림(들임 꼴: $3 \times 64 \times 64$)을 받도록 고쳐라. 켜의 차수를 모두 그에 맞게 손보고 모형이 어긋남 없이 도는지 따져라.

??? success "연습문제 2 풀이"
    첫 엮음 켜의 `in_channels`을 이제 값에서 3으로 바꾼다. 엮음과 모으기 켜마다 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$으로 자리 차수를 다시 셈한다. 첫 선형 켜의 `in_features`을 마지막 엮음/모으기 켜의 편 날임에 맞춘다. `model = SplitAttention(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`으로 따진다.

---

**연습문제 3.**
스스로 눈길의 셈 번거로움을 이음 길이 $n$과 모형 차수 $d$의 함수로 밝혀라. 이것이 긴 이음에 Longformer이나 Linformer 같은 얼개를 이끄는 까닭은 무엇인가?

??? success "연습문제 3 풀이"
    여느 스스로 눈길은 $n \times n$ 눈길 행렬을 셈하므로 때는 $O(n^2 d)$, 눈길 짐의 기억은 $O(n^2)$이다. 이음이 길면($n = 4096$ 따위) 감당할 수 없다. Longformer는 그 자리 미닫이 창 눈길($O(n \cdot w \cdot d)$, $w$은 창 크기)과 고른 낱말에 대한 성긴 두루 눈길을 아울러 쓴다. Linformer는 열쇠와 값을 더 낮은 차수 $k \ll n$으로 되비추어 번거로움을 $O(n \cdot k \cdot d)$으로 줄인다. 둘 다 드러내는 힘을 얼마쯤 내주고 긴 들임에서 잘 들게 한다.

---

**연습문제 4.**
`SplitAttention`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 얼개를 짜라. 켜 2개, 4개, 8개로 시험하여라.

??? success "연습문제 4 풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch가 매개변수를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = SplitAttention(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — ResNeSt

이 짜보기는 갈래 3개(`SplitAttention`, `ResNeStBlock`, `ResNeSt`)를 매기고, 이들이 어울려 온전한 엮음 신경 그물 얼개를 이룬다.

고갱이 갈래는 `SplitAttention`, `ResNeStBlock`, `ResNeSt`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
