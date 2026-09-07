# ResNet

ResNet(깊은 잔차 학습)은 허 카이밍 등이 2015년 글 "그림 인식을 위한 깊은 잔차 학습"에서 내놓았다. 고갱이 생각은 잔차 연결이며, 그물이 기본으로 항등 사상을 배우게 하여 기울기 소실 문제를 풀고 수백 켜짜리 그물도 학습할 수 있게 한다. 여기 짠 ResNet-50은 $1 \times 1$, $3 \times 3$, $1 \times 1$ 합성곱으로 이루어진 병목 블록을 쓴다.

## 코드

```python
import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels * 4:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )
        layers = [Bottleneck(in_channels, out_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_channels * 4, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


if __name__ == "__main__":
    model = ResNet50()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## 논의

잔차 연결이 ResNet을 ResNet이게 하는 결이다. 병목 블록마다 잔차 함수 $F(\mathbf{x})$을 셈해 들임에 더한다. $\mathbf{y} = F(\mathbf{x}) + \mathbf{x}$이다. 가장 좋은 변환이 항등에 가까우면 그물은 작은 잔차만 배우면 되므로 온 사상을 맨바닥부터 배우는 것보다 쉽다. 이 눈썰미 덕에 50, 101, 나아가 152켜짜리 그물까지 학습할 수 있게 되었고, 이는 그전 어느 것보다도 훨씬 깊다.

병목 설계는 $1 \times 1$ 합성곱으로 채널 차원을 줄이고, $3 \times 3$ 합성곱으로 공간을 다루고, 다시 $1 \times 1$ 합성곱으로 채널 차원을 되살린다(4배로 넓힌다). 그래서 날임 채널 수를 크게 지키면서도 셈 값을 감당할 만하게 둔다. 단계 사이에서 공간 차원이 바뀔 때는 보폭을 맞춘 학습된 $1 \times 1$ 투영이 지름길 경로를 맞춘다.

ResNet이 깊은 배움에 미친 영향은 아무리 말해도 지나치지 않다. 건너뛰는 이음이라는 원리는 DenseNet에서 변환기까지 요즘 얼개 거의 모두에 나타난다. ResNet-50은 갈래 매기기, 알아내기, 나누기를 아우르는 여러 일에서 옮겨 배우기의 표준 등뼈로 남아 있다.

## 연습문제

**연습문제 1.**
`in_channels=256`, `out_channels=64`, 공간 크기 $56 \times 56$인 병목 블록 하나의 곱셈-덧셈 연산 수(FLOPs)를 셈하여라.

??? success "연습문제 1 풀이"
    합성곱마다 FLOPs $\approx 2 \times C_{\text{in}} \times C_{\text{out}} \times K^2 \times H \times W$이다.

    - Conv1 ($1 \times 1$, $256 \to 64$): $2 \times 256 \times 64 \times 1 \times 56 \times 56 = 102{,}760{,}448$
    - Conv2 ($3 \times 3$, $64 \to 64$): $2 \times 64 \times 64 \times 9 \times 56 \times 56 = 231{,}211{,}008$
    - Conv3 ($1 \times 1$, $64 \to 256$): $2 \times 64 \times 256 \times 1 \times 56 \times 56 = 102{,}760{,}448$

    모두 블록마다 약 $436{,}731{,}904$ FLOPs($\approx 437$ MFLOPs)이다.

---

**연습문제 2.**
`downsample` 길이 왜 단계마다 첫 덩이에만 필요한지 설명하여라. 그것을 없애면 어떻게 되는가?

??? success "연습문제 2 풀이"
    내림 표본 경로(보폭 2인 $1 \times 1$ 합성곱)는 블록의 들임과 날임 사이에서 공간 차원이나 채널 수가 바뀔 때 필요하다. 이는 단계마다 첫 블록에서만 일어나며 그때 보폭이 1에서 2로 바뀌고 날임 채널이 늘어난다. 같은 단계의 뒤 블록은 들임과 날임 차원이 같으므로 항등 지름길이 그대로 동작한다.

    단계 경계에서 내림 표본 경로를 없애면 텐서의 꼴이 달라(공간 해상도와 채널 수가 다르다) $\text{out} + \text{identity}$ 덧셈이 실패하고 실행 오류가 난다.

---

**연습문제 3.**
병목 없이 $3 \times 3$ 합성곱 둘로 된 기본 블록을 써서 켜 구성이 $[2, 2, 2, 2]$인 ResNet-18 갈래를 짜라. 매개변수 수를 ResNet-50과 견주어라.

??? success "연습문제 3 풀이"
    ```python
    class BasicBlock(nn.Module):
        expansion = 1
        def __init__(self, in_channels, out_channels, stride=1, downsample=None):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample

        def forward(self, x):
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            if self.downsample:
                identity = self.downsample(x)
            return self.relu(out + identity)
    ```

    기본 덩이가 $[2, 2, 2, 2]$인 ResNet-18은 매개변수가 약 1170만 개이고, 병목 덩이를 쓰는 ResNet-50은 약 2560만 개로 대략 2.2배 많다.
