# HRNet

HRNet(높은 결 네트워크)은 2019년 글 "보고 알아보기를 위한 깊은 높은 결 리프리젠테이션 학습하기"에서 나왔으며, 네트워크 내내 높은 결의 리프리젠테이션을 지킨다. 결 그림을 차츰 성기게 했다가 다시 촘촘하게 하여 결을 되찾는 여느 구조와 달리, HRNet은 여러 결의 흐름을 나란히 잇고 그 사이에서 소식을 거듭 주고받는다. 이 꾸밈은 더 넉넉하고 자리가 더 또렷한 결을 내어 자세 어림이나 뜻 나누기 같은 일에 잘 듣는다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
HRNet - 높은 해상도 넷트웍
논문: "Deep High-Resolution Representation Learning" (2019)
핵심: 넷트웍을 지나는 동안 높은 해상도 리프리젠테이션을 지킨다
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class HRModule(nn.Module):
    def __init__(self, num_branches, channels):
        super().__init__()
        self.num_branches = num_branches
        self.branches = nn.ModuleList([
            nn.Sequential(*[BasicBlock(channels[i], channels[i]) for _ in range(4)])
            for i in range(num_branches)
        ])
        self.fuse_layers = nn.ModuleList()
        for i in range(num_branches):
            fuse_layer = nn.ModuleList()
            for j in range(num_branches):
                if i == j:
                    fuse_layer.append(nn.Identity())
                elif j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(channels[j], channels[i], 1, bias=False),
                        nn.BatchNorm2d(channels[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                    ))
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(nn.Sequential(
                                nn.Conv2d(channels[j], channels[i], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(channels[i])
                            ))
                        else:
                            conv_downsamples.append(nn.Sequential(
                                nn.Conv2d(channels[j], channels[j], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(channels[j]),
                                nn.ReLU(inplace=True)
                            ))
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            self.fuse_layers.append(fuse_layer)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = [branch(xi) for branch, xi in zip(self.branches, x)]
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse

class HRNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.stage = HRModule(2, [32, 64])
        self.incre_modules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(32, 128, 3, 1, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
        ])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = [x, torch.nn.functional.avg_pool2d(x, 2)]
        x = self.stage(x)
        x = [incre(xi) for incre, xi in zip(self.incre_modules, [x[0]])]
        x = self.avgpool(x[0]).flatten(1)
        return self.fc(x)

if __name__ == "__main__":
    model = HRNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 595,816
```

## 2. 논의

HRNet을 가르는 결은 나란한 여러 결 구조다. 결을 먼저 줄였다가 되찾는 여느 인코더-디코더나 결 두겁 방식 대신, HRNet은 높은 결의 흐름을 내내 지키며 낮은 결의 나란한 흐름을 차츰 더한다. 여러 잣대 녹이기 배치가 모든 결 켜 사이에서 소식을 거듭 주고받아, 높은 결의 흐름이 낮은 결이 지닌 뜻의 넉넉함을 얻으면서도 잔 자리 결을 지킨다.

녹이는 구조는 꼼꼼히 꾸며졌다. 낮은 결에서 높은 결로 녹일 때는 $1 \times 1$ 컨볼루션으로 클래스를 맞춘 뒤 두 겹 곧은 계산이나 가장 가까운 이웃으로 촘촘하게 한다. 높은 결에서 낮은 결로 녹일 때는 걸음 있는 $3 \times 3$ 컨볼루션으로 자리 차수를 차츰 줄인다. 이 어긋남은 서로 다른 쓰임을 비춘다. 촘촘하게 하기에는 클래스 맞추기가, 성기게 하기에는 자리 모으기가 든다.

HRNet의 꾸밈은 촘촘한 추론 일에 몹시 세다. 사람 자세 어림에서는 자리가 또렷한 높은 결의 리프리젠테이션이 곧바로 맞는 핵심 점 히트맵을 준다. 뜻 나누기에서는 넉넉한 여러 잣대 결이 테두리를 잘게 갈라 준다. 이 구조는 나중에 결을 되찾는 길에 기대는 방법보다 한결같이 낫다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
결이 $H \times W$, $H/2 \times W/2$, $H/4 \times W/4$인 가지 셋을 지닌 `HRModule`에는 녹이는 길이 몇 개 있는가? 드는 촘촘하게 하기와 성기게 하기 계산을 늘어놓아라.

</div>

??? success "연습문제 1 풀이"
    가지가 셋이면 녹이는 길은 $3 \times 3 = 9$개다(가지마다 저 자신을 넣어 세 가지 모두에서 받는다). 제 자리 길: 3개(가지 $i$에서 저 자신으로). 촘촘하게 하는 길: 3개(가지 1→0: $2\times$; 가지 2→0: $4\times$; 가지 2→1: $2\times$). 성기게 하는 길: 3개(가지 0→1: 걸음 2 컨볼루션 하나; 가지 0→2: 걸음 2 컨볼루션 둘; 가지 1→2: 걸음 2 컨볼루션 하나). 촘촘하게 하는 길은 저마다 $1 \times 1$ 컨볼루션으로 클래스를 맞춘 뒤 촘촘하게 한다. 성기게 하는 길은 저마다 걸음 2의 $3 \times 3$ 컨볼루션을 잇달아 쓴다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
자리 결을 지키는 HRNet의 길과 건너뛰는 이음을 쓰는 U-Net의 인코더-디코더 길을 비교하여라. 저마다의 좋은 점과 나쁜 점은 무엇인가?

</div>

??? success "연습문제 2 풀이"
    HRNet은 높은 결의 결을 내내 지키므로 공간 정보가 아주 사라지지는 않는다. 여러 잣대 녹이기가 거듭 일어나 차츰 다듬어 간다. 다만 높은 결의 흐름이 네트워크 깊이 내내 온 결로 결을 다루므로 계산이 값비싸다. U-Net은 차츰 성기게 하여 깊은 켜의 계산을 줄이고, 촘촘하게 할 때 건너뛰는 이음으로 자리 결을 되찾는다. 계산은 더 잘 들지만, 목을 가로지르는 소식의 틈을 건너뛰는 이음에 기댄다. U-Net의 건너뛰는 이음은 맞는 결끼리 단순히 이어 붙이는 것이고, HRNet의 녹이기는 도막마다 모든 결 켜를 아우르므로 더 촘촘하다. HRNet은 대체로 자리 추론이 더 맞지만 계산 비용이 더 든다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
가지에는 깊이별로 가른 컨볼루션을, 모든 녹이기 계산에는 $1 \times 1$ 컨볼루션을 쓰는 가벼운 `HRModule`을 꾸며라(걸음 있는 $3 \times 3$ 컨볼루션은 모으기 + $1 \times 1$ 컨볼루션으로 갈음한다).

</div>

??? success "연습문제 3 풀이"
    ```python
    class LightBasicBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.dw = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.pw = nn.Conv2d(channels, channels, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(channels)
            self.relu = nn.ReLU(inplace=True)
        def forward(self, x):
            out = self.relu(self.bn1(self.dw(x)))
            out = self.bn2(self.pw(out))
            return self.relu(out + x)

    class LightHRModule(nn.Module):
        def __init__(self, num_branches, channels):
            super().__init__()
            self.num_branches = num_branches
            self.branches = nn.ModuleList([
                nn.Sequential(*[LightBasicBlock(channels[i]) for _ in range(2)])
                for i in range(num_branches)
            ])
            # 녹이기는 모두 모으기 + 1x1 컨볼루션(내림)이나 1x1 컨볼루션 + 촘촘하게 하기(오름)를 쓴다
            # 구현은 같은 fuse_layers 무늬를 따르되
            # 걸음 있는 3x3 컨볼루션을 avg_pool2d + 1x1 컨볼루션으로 갈음한다.
    ```

## 정리하며

**다룬 것** — HRNet

HRNet을 가르는 결은 나란한 여러 결 구조다.

핵심 클래스는 `BasicBlock`, `HRModule`, `HRNet`, `LightBasicBlock`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
