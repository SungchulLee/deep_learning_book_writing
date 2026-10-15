# PyramidNet

PyramidNet은 잔차 블록을 지날 때마다 채널 폭을 점진적으로 늘리는 CNN이다. Han 외의 CVPR 2017 논문은 단계 경계에서만 폭을 크게 늘리는 구조와 이 방식을 비교했다. 아래 코드는 채널 증가와 zero-padding 지름길의 의도를 보여 주지만, 현재 풀링 조건 때문에 폭이 바뀌는 stride1 블록에서 형태가 맞지 않는다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
PyramidNet — 깊은 피라미드 잔차 그물
논문: "Deep Pyramidal Residual Networks" (2017)
핵심: 특징 지도 차원을 차츰 늘린다
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class PyramidBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(2, 2),
            )
    
    def forward(self, x):
        out = self.conv1(torch.nn.functional.relu(self.bn1(x)))
        out = self.conv2(torch.nn.functional.relu(self.bn2(out)))
        out = self.bn3(out)
        
        # 차원을 맞추려 0으로 덧대기
        shortcut = self.shortcut(x)
        if shortcut.size(1) != out.size(1):
            pad_size = out.size(1) - shortcut.size(1)
            shortcut = torch.nn.functional.pad(shortcut, (0, 0, 0, 0, 0, pad_size))
        
        out += shortcut
        return out

class PyramidNet(nn.Module):
    def __init__(self, num_classes=1000, alpha=48, depth=110):
        super().__init__()
        n = (depth - 2) // 6
        
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        
        add_rate = alpha / (3 * n)
        in_channels = 16
        
        layers = []
        for i in range(3 * n):
            out_channels = round(16 + add_rate * (i + 1))
            stride = 2 if i == n or i == 2 * n else 1
            layers.append(PyramidBasicBlock(in_channels, out_channels, stride))
            in_channels = out_channels
        
        self.layers = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layers(x)
        x = torch.nn.functional.relu(self.bn(x))
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

if __name__ == "__main__":
    model = PyramidNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 1,837,024
```

## 2. 논의

### 채널 폭의 점진적 증가

코드는 n=⌊(depth−2)/6⌋, 총 3n개 basic block을 사용한다. 증가율 Δ=α/(3n)에서 i번째 출력 폭은 round(16+Δ(i+1))이다. 기본 depth=110이면 n=18, 블록은 54개이며 α=48일 때 마지막 폭은 64다. 각 블록에 합성곱 두 개, stem과 분류기를 더하는 6n+2의 깊이 관례다.

블록은 사전 활성화 BN–ReLU를 사용하며 마지막 BN의 출력에 지름길을 더하고 바로 반환한다. 덧셈 뒤 ReLU가 있는 일반 post-activation 블록과 다르다. 부드러운 폭 증가는 설계 선택이지 정보 손실이나 경사 문제를 모든 모델에서 줄인다는 보편적 보장은 아니다.

### 지름길의 채널 패딩과 공간 축소

채널만 늘어나면 기존 입력의 공간 크기를 유지하면서 새 채널을 0으로 채워 지름길을 맞출 수 있다. stride2로 공간을 줄이는 경계에서는 풀링도 필요하다. 학습 가능한 1×1 투영과 달리 추가 채널에 대응하는 지름길의 값은 0이며, 그 새 특징은 주 분기가 만든다.

그러나 이 코드는 **stride≠1 또는 채널 수가 다르면 항상 AvgPool2d(2,2)**를 적용한다. 그래서 채널만 증가하고 stride=1인 블록에서도 지름길 공간이 절반으로 줄어 주 분기와 덧셈할 수 없다. 기본 α=48의 첫 블록도 16→17, stride1이므로 이미 이 문제가 발생한다. 채널을 패딩하는 연산은 공간 불일치를 고치지 못한다.

stride2에서도 홀수 H를 3×3 padding1로 줄이면 ceil(H/2)지만 2×2 평균 풀링은 floor(H/2)이어서 불일치가 생길 수 있다. 채널이 감소하는 설정의 음수 padding을 확대용 zero-padding과 동일하게 해석해서도 안 된다.

### 입력과 매개변수의 조건

입력 RGB 64×64는 stem에서 공간을 유지하고 의도한 두 다운샘플 경계 후 16×16이 된다. 최종 적응형 풀링은 마지막 채널 폭으로 요약하므로 fc 폭을 공간 크기에 맞춰 바꿀 필요는 없다. 다만 이 의도된 흐름은 현재 지름길 오류가 해결되었다는 뜻은 아니다.

유효한 n>0과 depth 관례, 증가 폭과 양의 채널, 경계의 stride를 명시해야 한다. depth가 6n+2가 아니면 정수 나눗셈으로 이름의 깊이와 실제 블록 수가 어긋날 수 있다. 연습의 ModuleList 골격은 폭 증가 일정을 구현하지 않는다. [원 논문](https://openaccess.thecvf.com/content_cvpr_2017/html/Han_Deep_Pyramidal_Residual_CVPR_2017_paper.html)의 결과와 현재 예제의 오류를 구분하며 코드·출력은 실행하지 않고 보존했다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
기본 α=48, depth=110의 첫 블록 형태와 오류를 설명하라.

</div>

??? success "연습문제 1 풀이"
    n=18, 증가율은48/54이고 첫 출력은round(16+48/54)=17이다. stride1 주 분기는 H×W를 유지하지만 채널이 다르다는 이유로 지름길은2×2 풀링되어 절반 크기가 된다. 채널 패딩 후에도 공간은 맞지 않아 덧셈이 실패한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
64×64 RGB의 의도된 경로와 실제 코드 검증을 구분하라.

</div>

??? success "연습문제 2 풀이"
    stem은 16×64×64를 만들고 의도한 두 stride2 경계 뒤 공간은16×16이다. 전역 풀링 후 마지막 채널 폭을 fc에 전달한다. 하지만 그 이전의 stride1 채널 증가에서 지름길이 잘못 풀링되므로 이 형태 계획만으로 현재 모델이 실행된다고 할 수 없다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
채널 zero-padding과 투영 지름길의 차이를 설명하라.

</div>

??? success "연습문제 3 풀이"
    zero-padding은 기존 채널을 유지하고 추가 채널을 0으로 채워 학습 가중치를 늘리지 않는다. 투영은 1×1 가중치로 채널을 혼합한다. 어느 쪽이든 주 분기와 공간 크기를 맞춰야 하고, 공간 풀링은 채널 변화와 별도로 stride에 맞춰 적용해야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 반복 골격으로 PyramidNet을 구성하려면 어떤 값들을 추적해야 하는가?

</div>

??? success "연습문제 4 풀이"

    전체 블록 수, 실수 폭 증가율과 반올림한 입출력 폭, 두 다운샘플 경계, 채널 패딩과 공간 풀링 조건을 추적해야 한다. YourBlock·dim의 일반 목록은 이 일정이나 지름길을 구현하지 않는다. 정수 depth 관례와 양의 채널을 확인하고 forward 순회도 정의해야 한다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

PyramidNet은 채널 폭을 블록마다 점진적으로 늘린다. 채널 패딩과 공간 축소는 다른 연산이며, 이 예제의 조건은 채널만 증가해도 풀링해 덧셈 오류를 만든다. 깊이·반올림·홀수 크기 조건을 함께 확인한다.
