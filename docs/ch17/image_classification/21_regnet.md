# RegNet

Radosavovic 외의 2020년 논문 "Designing Network Design Spaces"에서 나온 RegNet은 그물 너비와 깊이를 양자화한 선형 매개변수로 나타내어, 효율적이면서도 정확한 단순하고 고른 얼개를 꾸민다.

## 코드

```python
import torch
import torch.nn as nn


class RegNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1,
                              groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return self.relu(out)


class RegNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.layer1 = self._make_layer(32, 64, 2)
        self.head = nn.Linear(64, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        for i in range(blocks):
            stride = 2 if i == 0 else 1
            layers.append(RegNetBlock(
                in_channels if i == 0 else out_channels, out_channels, stride))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)


if __name__ == "__main__":
    model = RegNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## 논의

RegNet의 설계 철학은 좋은 그물을 내놓는 설계 공간의 단순한 매개변수화를 찾는 것이다. 단계마다 너비는 양자화한 선형 함수 $w_j = w_0 + w_a \cdot j$을 따르며 무리 너비의 가장 가까운 배수로 양자화한다. 그러면 반듯하고 미리 헤아릴 수 있는 구조가 나온다.

마구잡이로 뽑은 그물 무리를 살펴 RegNet은 한결같이 좋은 모델을 낳는 꾸밈 원리를 가려낸다. 곧 너비는 단계마다 늘어나야 하고, 깊이는 알맞아야 하며, 병목 비를 갖춘 묶음 누비기가 정확도와 효율의 맞바꿈에서 가장 낫다.

## 연습문제

**연습문제 1.**
RegNet 너비의 양자화된 선형 매개변수 나타내기를 설명하여라.

??? success "연습문제 1 풀이"
    RegNet은 단계 너비를 $w_j = \text{quantize}(w_0 + w_a \cdot j, q)$으로 매긴다. 여기서 $w_0$은 첫 너비, $w_a$은 기울기, $j$은 단계 번호, $q$은 양자화 걸음(무리 너비)이다. 이 단순한 식이 매개변수 셋만으로 모든 단계 너비를 만들어 내므로 설계 공간이 낮은 차원이 되고 찾기도 쉽다.

---

**연습문제 2.**
RegNet의 꾸밈 방법론과 신경 얼개 찾기를 견주어라. 맞바꿈은 무엇인가?

??? success "연습문제 2 풀이"
    RegNet은 마구잡이 표집과 통계 살피기로 좋은 꾸밈 원리를 가려내는 반면, 신경 얼개 찾기는 북돋움 배움이나 진화 찾기로 특정 얼개를 찾는다. RegNet은 훨씬 값이 싸고(찾는 동안 익힐 필요가 없다) 읽어 낼 수 있는 꾸밈 규칙을 내놓으며 잣수를 넘나들어 두루 통한다. 신경 얼개 찾기는 더 특화된 얼개를 찾을 수 있지만 값이 비싸고 대리 일에 지나치게 맞춰질 수 있다.

---

**연습문제 3.**
선형 너비 규칙 $w_j = 24 + 24j$과 깊이 $[2, 4, 6, 2]$을 따르는 4단계 RegNet 자리매김을 만들어라.

??? success "연습문제 3 풀이"
    ```python
widths = [24 + 24 * j for j in range(4)]  # [24, 48, 72, 96]
# 가장 가까운 8의 배수로 양자화:
widths = [(w // 8) * 8 for w in widths]  # [24, 48, 72, 96]
depths = [2, 4, 6, 2]

# 단계를 세운다
for i, (w, d) in enumerate(zip(widths, depths)):
    print(f"Stage {i}: width={w}, depth={d}")
# 단계 0: 너비=24, 깊이=2
# 단계 1: 너비=48, 깊이=4
# 단계 2: 너비=72, 깊이=6
# 단계 3: 너비=96, 깊이=2
```
