# HRNet

HRNet(High-Resolution Network, 고해상도 네트워크)은 2019년 논문 "Deep High-Resolution Representation Learning for Visual Recognition"에서 나왔으며, 네트워크 전체에 걸쳐 고해상도 표현을 유지한다. 피처 맵을 차츰 다운샘플링했다가 다시 업샘플링하여 해상도를 복원하는 일반적인 구조와 달리, HRNet은 여러 해상도의 스트림을 병렬로 연결하고 그 사이에서 정보를 반복해서 교환한다. 이 설계는 더 풍부하고 공간적으로 더 정밀한 피처를 만들어 자세 추정(pose estimation)이나 시맨틱 세그멘테이션 같은 과제에 효과적이다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
HRNet - 고해상도 네트워크
논문: "Deep High-Resolution Representation Learning" (2019)
핵심: 네트워크 전체에 걸쳐 고해상도 표현을 유지한다
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

HRNet을 특징짓는 것은 병렬 다중 해상도 구조다. 해상도를 먼저 줄였다가 복원하는 일반적인 인코더-디코더나 피처 피라미드 방식 대신, HRNet은 고해상도 스트림을 끝까지 유지하며 저해상도의 병렬 스트림을 차츰 더한다. 다중 스케일 융합(multi-scale fusion) 모듈이 모든 해상도 수준 사이에서 정보를 반복해서 교환하여, 고해상도 스트림이 저해상도가 지닌 풍부한 의미 정보를 얻으면서도 세밀한 공간 정보를 유지한다.

융합 메커니즘은 세심하게 설계되었다. 저해상도에서 고해상도로 융합할 때는 $1 \times 1$ 컨볼루션으로 채널을 맞춘 뒤 쌍선형 보간이나 최근접 이웃 보간으로 업샘플링한다(이 쪽의 코드는 최근접 이웃 보간을 쓴다). 고해상도에서 저해상도로 융합할 때는 스트라이드가 있는 $3 \times 3$ 컨볼루션으로 공간 차원을 차츰 줄인다. 이 비대칭은 서로 다른 요구를 반영한다. 업샘플링에는 채널 정렬이, 다운샘플링에는 공간 집계가 필요하다.

HRNet의 설계는 밀집 예측(dense prediction, 픽셀마다 출력을 내는 과제)에 특히 강하다. 사람 자세 추정에서는 공간적으로 정밀한 고해상도 표현이 곧바로 정확한 키포인트 히트맵을 만든다. 시맨틱 세그멘테이션에서는 풍부한 다중 스케일 피처가 경계를 세밀하게 구분해 준다. 원 논문은 이 두 과제에서 해상도를 나중에 복원하는 경로에 의존하는 방법보다 나은 결과를 보고했다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
해상도가 $H \times W$, $H/2 \times W/2$, $H/4 \times W/4$인 브랜치 셋을 지닌 `HRModule`에는 융합 경로가 몇 개 있는가? 필요한 업샘플링과 다운샘플링 연산을 나열하여라.

</div>

??? success "풀이"
    브랜치가 셋이면 융합 경로는 $3 \times 3 = 9$개다(브랜치마다 자기 자신을 포함해 세 브랜치 모두에서 받는다). 항등 경로: 3개(브랜치 $i$에서 자기 자신으로). 업샘플링 경로: 3개(브랜치 1→0: $2\times$; 브랜치 2→0: $4\times$; 브랜치 2→1: $2\times$). 다운샘플링 경로: 3개(브랜치 0→1: 스트라이드 2 컨볼루션 하나; 브랜치 0→2: 스트라이드 2 컨볼루션 둘; 브랜치 1→2: 스트라이드 2 컨볼루션 하나). 업샘플링 경로는 저마다 $1 \times 1$ 컨볼루션으로 채널을 맞춘 뒤 업샘플링한다. 다운샘플링 경로는 저마다 스트라이드 2의 $3 \times 3$ 컨볼루션을 잇달아 쓴다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
공간 해상도를 유지하는 HRNet의 방식과 스킵 연결을 쓰는 U-Net의 인코더-디코더 방식을 비교하여라. 각각의 장점과 단점은 무엇인가?

</div>

??? success "풀이"
    HRNet은 고해상도 피처를 끝까지 유지하므로 공간 정보가 완전히 사라지지는 않는다. 다중 스케일 융합이 반복해서 일어나 점진적으로 정제해 간다. 다만 고해상도 스트림이 네트워크 깊이 전체에 걸쳐 전체 해상도로 피처를 처리하므로 계산 비용이 크다. U-Net은 차츰 다운샘플링하여 깊은 레이어의 계산을 줄이고, 업샘플링할 때 스킵 연결로 공간 정보를 복원한다. 계산은 더 효율적이지만, 병목을 가로지르는 정보의 손실을 스킵 연결에 의존해 메운다. U-Net의 스킵 연결은 대응하는 해상도끼리 단순히 이어 붙이는 것이고, HRNet의 융합은 스테이지마다 모든 해상도 수준을 결합하므로 더 조밀하다. HRNet은 대체로 공간 예측이 더 정확하지만 계산 비용이 더 든다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
브랜치에는 깊이별 분리 컨볼루션을, 모든 융합 연산에는 $1 \times 1$ 컨볼루션을 쓰는 경량 `HRModule`을 설계하여라(스트라이드가 있는 $3 \times 3$ 컨볼루션은 풀링 + $1 \times 1$ 컨볼루션으로 대체한다).

</div>

??? success "풀이"
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
            # 융합은 모두 풀링 + 1x1 컨볼루션(다운샘플링)이나 1x1 컨볼루션 + 업샘플링을 쓴다
            # 구현은 같은 fuse_layers 패턴을 따르되
            # 스트라이드가 있는 3x3 컨볼루션을 avg_pool2d + 1x1 컨볼루션으로 대체한다.
    ```

## 정리하며

**다룬 것** — HRNet

HRNet을 특징짓는 것은 병렬 다중 해상도 구조다.

핵심 클래스는 `BasicBlock`, `HRModule`, `HRNet`, `LightBasicBlock`이며 앞의 연습문제 3개로 스스로 확인해 볼 수 있다.
