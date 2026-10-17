# HRNet

HRNet은 비교적 높은 해상도의 피처 경로를 유지하면서 여러 해상도의 경로를 병렬로 처리하고 반복해서 결합한다. 고해상도는 원본 화소 크기를 그대로 유지한다는 뜻이 아니라 네트워크 내부의 상대적인 해상도다. 아래 코드는 두 분기의 융합 골격이며 stem과 분기 사이의 채널 변환이 누락되어 있다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
HRNet — 높은 해상도 넷트웍
논문: "Deep High-Resolution Representation Learning" (2019)
핵심: 넷트웍 전체에서 높은 해상도 나타냄을 지킨다
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

### 병렬 해상도와 융합

각 분기는 자기 해상도에서 잔차 블록을 처리한다. 다른 분기의 피처를 합치려면 채널과 공간 크기를 모두 맞춘다. 저해상도→고해상도는 1×1 채널 변환과 업샘플링, 고해상도→저해상도는 stride2 컨볼루션을 필요한 횟수만큼 적용한다. 같은 해상도의 여러 피처를 원소별로 더하는 융합이며 채널 연결과 구분한다.

두 분기의 목표가 (N,32,H,W), (N,64,H/2,W/2)라면 고해상도 출력에는 두 번째 분기의 64→32 및 2배 업샘플을 더하고, 저해상도 출력에는 첫 분기의 32→64 stride2 변환을 더한다. 채널과 목표 크기를 맞추는 과정이 핵심이다.

### BasicBlock과 입력 리스트 조건

이 `BasicBlock`에는 지름길 투영이 없으므로 입력과 출력 채널이 같아야 덧셈이 가능하다. 인수에 서로 다른 폭을 지정할 수 있다는 사실이 그 경우의 실행을 지원한다는 뜻은 아니다. HRModule은 이 조건을 맞춰 각 분기에 같은 입출력 폭을 사용한다.

분기 입력 리스트의 길이·채널·해상도가 구성과 맞아야 한다. 코드의 고정 배수 업샘플은 홀수 공간 크기의 반복 축소와 정확히 맞지 않을 수 있다. 예를 들어 15를 2배 축소한 7을 2배로 키우면 14여서 15와 더할 수 없다. 임의 크기에 일반화하려면 목표 크기를 명시하는 융합 등이 필요하다.

### 본체의 누락

stem은 두 stride2 컨볼루션으로 64채널을 만든다. 본체가 만드는 리스트 `[x,avg_pool2d(x,2)]`의 두 입력도 모두 64채널이다. 그러나 `HRModule(2,[32,64])`의 첫 분기는 32채널을 요구하므로 바로 채널 불일치가 발생한다. 첫 경로의 64→32 변환 등 유효한 transition이 필요하지만 코드는 그대로 보존했다.

64×64 RGB에서는 stem 뒤 (N,64,16,16), 두 번째 입력은 (N,64,8,8)이다. 필요한 첫 입력은 (N,32,16,16)이므로 공간 크기만 맞는다고 해결되지 않는다. 분류 head도 첫 고해상도 경로만 사용하는 축약형으로, 전체 HRNet의 다단계·다중 해상도 분류 head와 다르다.

구조의 배경은 [2019년 포즈 추정 논문](https://arxiv.org/abs/1902.09212)과 [시각 인식 확장](https://arxiv.org/abs/1908.07919)을 구분해 참고한다. 기존 출력은 파라미터 수만 계산하며 순전파·학습 검증은 아니다. 연습의 ModuleList도 정의되지 않은 골격이므로 실제 채널·융합 규칙을 대체하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
입력 (4,32,16,16)인 BasicBlock(32,32)의 형태와 제한을 설명하라.

</div>

??? success "풀이"
    두 3×3은 stride1·padding1이라 (4,32,16,16)을 유지한다. 잔차와 더한 후 ReLU를 적용한다. 지름길 투영이 없으므로 BasicBlock(32,64)처럼 채널을 바꾸면 주 분기와 입력이 맞지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
64×64 RGB 입력에서 본체의 첫 분기가 실패하는 원인을 설명하라.

</div>

??? success "풀이"
    stem 뒤 64×16×16, 풀링한 두 번째 입력은 64×8×8이다. 첫 분기는 32×16×16을 기대하므로 채널 변환이 누락되었다. 입력 채널을 RGB로 고치거나 fc만 변경하는 문제가 아니다. 두 분기의 transition과 이후 융합의 형태를 맞춰야 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
32채널 고해상도와 64채널 저해상도의 상호 융합 연산을 설명하라.

</div>

??? success "풀이"
    저→고는 1×1로64→32를 만든 뒤 목표 고해상도로 키워 더한다. 고→저는 stride2 3×3으로32→64를 만들고 더한다. 채널 연결이 아니라 형태를 맞춘 합이며, 홀수 크기에서는 고정 배수 업샘플과 축소 결과가 정확히 맞는지 확인해야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 반복 골격으로 HRModule을 확장할 때 필요한 분기별 설계를 설명하라.

</div>

??? success "풀이"

    각 분기의 폭·해상도·반복 수, 새 분기를 만드는 transition과 모든 분기 쌍의 융합을 정의해야 한다. YourBlock·dim 등의 일반 골격만으로 이런 연결이 만들어지지 않는다. BasicBlock을 반복할 때도 지름길 투영이 없다는 제약과 forward 순회를 확인한다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

HRNet의 핵심은 병렬 해상도 경로와 반복 융합이다. 이 예제는 첫 분기의 채널 transition이 빠져 있으며, 고정 배수 업샘플과 블록의 무투영 지름길도 별도 형태 조건을 가진다.
