# EfficientNet V2

EfficientNetV2는 2021년 논문 "EfficientNetV2: Smaller Models and Faster Training"에서 나왔으며, Fused-MBConv 레이어와 점진적 학습(progressive learning) 전략으로 원래의 EfficientNet을 개선한다. 이로써 정확도는 유지하거나 올리면서 학습 시간은 크게 준다. 구조는 뉴럴 아키텍처 탐색(NAS)과 스케일링을 결합해 찾아냈으며, 학습 속도와 파라메터 효율을 함께 최적화했다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
EfficientNetV2 - 개선된 효율과 속도
논문: "EfficientNetV2: Smaller Models and Faster Training" (2021)
핵심: Fused-MBConv 레이어, 점진적 학습, 더 빠른 학습
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class FusedMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=4):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        return self.conv(x)

class EfficientNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Conv2d(3, 24, 3, 2, 1, bias=False)
        self.classifier = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

if __name__ == "__main__":
    model = EfficientNetV2()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 1,281,648
```

## 2. 논의

EfficientNetV2의 핵심 구조 혁신은 Fused-MBConv 블록이다. 앞쪽 스테이지에서 깊이별 분리 컨볼루션(depthwise separable convolution)을 일반적인 $3 \times 3$ 컨볼루션으로 대체한다. 깊이별 분리 컨볼루션은 파라메터가 적지만 연산 밀도가 낮아 최신 가속 하드웨어(GPU/TPU)를 충분히 활용하지 못한다. Fused 변형은 일반적인 $3 \times 3$ 컨볼루션으로 채널을 확장한 뒤 $1 \times 1$로 사영하여, 파라메터가 조금 늘지만 하드웨어를 더 잘 활용한다. EfficientNetV2는 피처 맵이 큰 앞쪽 스테이지에는 Fused-MBConv를, 뒤쪽 스테이지에는 일반 MBConv를 쓴다.

점진적 학습 전략도 핵심 기여다. 학습하는 동안 이미지 해상도와 정칙화 강도(드롭아웃, 데이터 어그멘테이션)를 차츰 올린다. 초기 에포크에는 작은 이미지와 약한 어그멘테이션을 써서 거친 피처를 빨리 학습하게 하고, 후기 에포크에는 전체 해상도의 이미지와 강한 정칙화를 쓴다. 이 적응적 방식은 해상도를 고정한 학습보다 학습 시간을 최대 11배까지 줄일 수 있다.

EfficientNetV2는 전체에 걸쳐 SiLU(Swish) 활성화 함수를 쓰는데, 여러 상황에서 ReLU보다 낫다고 알려졌다. SiLU 함수 $f(x) = x \cdot \sigma(x)$은 매끄럽고 단조가 아니어서 학습하는 동안 그래디언트가 더 잘 흐른다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
입력 형태가 $(1, 32, 56, 56)$, 확장 비율 4, 출력 채널 32일 때 일반 MBConv 블록과 Fused-MBConv 블록의 계산 비용(FLOPs, 부동소수점 연산 횟수)을 비교하여라.

</div>

??? success "풀이"
    MBConv에서는 (1) $1 \times 1$ 확장: $32 \times 128 \times 56 \times 56 \approx 12.8M$번. (2) $3 \times 3$ 깊이별: $128 \times 9 \times 56 \times 56 \approx 3.6M$번. (3) $1 \times 1$ 사영: $128 \times 32 \times 56 \times 56 \approx 12.8M$번. 모두 $\approx 29.2M$번이다. Fused-MBConv에서는 (1) $3 \times 3$ 일반 컨볼루션: $32 \times 128 \times 9 \times 56 \times 56 \approx 115.6M$번. (2) $1 \times 1$ 사영: $128 \times 32 \times 56 \times 56 \approx 12.8M$번. 모두 $\approx 128.4M$번이다. Fused-MBConv는 연산이 약 4.4배 많지만 하드웨어를 더 잘 활용해서 GPU에서는 실제 소요 시간이 더 짧은 일이 잦다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
점진적 학습(학습하는 동안 해상도를 올리기)이 ResNet처럼 해상도가 고정된 모델보다 EfficientNet 계열 모델에 더 이로운 까닭을 설명하여라.

</div>

??? success "풀이"
    EfficientNet 모델은 깊이, 너비, 해상도를 함께 조정하는 복합 스케일링을 쓴다. 그 구조는 여러 해상도에서 두루 잘 동작하도록 설계되었다. 점진적 학습은 이를 이용해 낮은 해상도(처리할 픽셀이 적어 반복이 빠름)에서 출발해 차츰 전체 해상도로 올린다. 네트워크 구조가 해상도에 구애받지 않으므로 낮은 해상도에서 학습한 피처가 자연스럽게 전이된다. ResNet은 구조의 가정이 고정되어 있어 해상도가 바뀌면 성능이 더 흔들린다. 게다가 점진적 학습의 정칙화 스케줄(낮은 해상도에는 약한 어그멘테이션, 높은 해상도에는 강한 어그멘테이션)은 모델이 어그멘테이션된 작은 이미지에 과적합하는 것을 막는데, 해상도를 고정하고 학습하는 ResNet에는 덜 해당하는 문제다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
SE(Squeeze-and-Excitation) 어텐션과 스킵 연결을 갖춘 완전한 `FusedMBConv` 블록을 구현하여라.

</div>

??? success "풀이"
    ```python
    class FusedMBConvSE(nn.Module):
        def __init__(self, in_ch, out_ch, expand_ratio=4, se_ratio=0.25):
            super().__init__()
            hidden = in_ch * expand_ratio
            self.use_skip = (in_ch == out_ch)
            self.expand = nn.Sequential(
                nn.Conv2d(in_ch, hidden, 3, 1, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.SiLU(inplace=True),
            )
            se_ch = max(1, int(in_ch * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden, se_ch, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(se_ch, hidden, 1),
                nn.Sigmoid(),
            )
            self.project = nn.Sequential(
                nn.Conv2d(hidden, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )

        def forward(self, x):
            out = self.expand(x)
            out = out * self.se(out)
            out = self.project(out)
            if self.use_skip:
                out = out + x
            return out
    ```

## 정리하며

**다룬 것** — EfficientNet V2

EfficientNetV2의 핵심 구조 혁신은 Fused-MBConv 블록이다.

핵심 클래스는 `FusedMBConv`, `EfficientNetV2`, `FusedMBConvSE`이며 앞의 연습문제 3개로 스스로 확인해 볼 수 있다.
