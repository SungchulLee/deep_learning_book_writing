# ConvNeXt V2

ConvNeXt V2는 2023년 논문 "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders"에서 나왔으며, 원래의 ConvNeXt에 두 가지 핵심 혁신을 더한다. 완전 컨볼루션 마스크 오토인코더(FCMAE)를 쓰는 자기 지도 사전 학습 전략과, 전역 반응 정규화(Global Response Normalization, GRN)라는 새 정규화 레이어다. 이 둘 덕에 ConvNeXt V2는 스케일링이 더 잘 되고 여러 모델 크기에서 더 나은 성능을 보인다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
ConvNeXt V2 - 설계를 다듬은 현대적 컨볼루션 네트워크
논문: "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders" (2023)
핵심: 전역 반응 정규화(GRN)와 개선한 학습 전략
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
    
    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return input + x

class ConvNeXtV2(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Conv2d(3, 96, kernel_size=4, stride=4)
        self.blocks = nn.Sequential(*[ConvNeXtV2Block(96) for _ in range(3)])
        self.head = nn.Linear(96, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = x.mean([2, 3])
        return self.head(x)

if __name__ == "__main__":
    model = ConvNeXtV2()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 341,608
```

## 2. 논의

전역 반응 정규화(GRN) 레이어가 ConvNeXt V2 구조의 핵심이다. 마스크 오토인코더로 컨볼루션 네트워크를 사전 학습할 때 생기는 피처 붕괴(feature collapse)를 다룬다. GRN은 먼저 채널마다 공간 반응의 $L^2$ 노름을 계산하고, 그 노름을 채널에 걸친 평균으로 나누어 정규화한 뒤, 정규화한 값으로 원래 피처의 스케일을 다시 조정한다. 학습 가능한 파라메터 $\gamma$와 $\beta$가 유연성을 주고, 잔차 연결 덕에 처음에는 항등 함수처럼 동작할 수 있다.

GRN은 비전 트랜스포머에는 잘 통하는 마스크 오토인코더 사전 학습이 컨볼루션 네트워크에서는 피처의 중복을 낳는다는 관찰에서 출발했다. GRN이 없으면 많은 채널이 비슷한 피처를 학습하여 모델의 실질적인 용량이 줄어든다. 전역 정규화로 채널끼리 경쟁하게 하면 GRN이 피처의 다양성을 높이고 채널마다의 기여를 더 뚜렷하게 한다.

원래의 ConvNeXt와 견주면 V2 블록은 레이어 스케일을 없애고 역 병목의 GELU 활성화 함수 뒤에 GRN을 둔다. 이 변경과 완전 컨볼루션 마스크 오토인코더(FCMAE) 사전 학습을 함께 쓰면, ConvNeXt V2는 아주 작은 것(파라메터 400만)에서 아주 큰 것(6억 넘음)에 이르는 여러 크기에서 트랜스포머 기반 모델에 맞먹거나 앞선다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
GRN 레이어에 형태가 $(B, H, W, D)$인 입력 텐서가 들어올 때 단계마다 $G_x$와 $N_x$의 형태를 설명하여라.

</div>

??? success "풀이"
    형태가 $(B, H, W, D)$인 입력 $x$에서 출발한다. (1) 차원 $(1, 2)$(높이와 너비)에 걸쳐 keepdim으로 계산한 $G_x = \|x\|_2$의 형태는 $(B, 1, 1, D)$이다. 이는 채널마다 공간 반응의 $L^2$ 노름이다. (2) $G_x.\text{mean}(\text{dim}=-1, \text{keepdim}=\text{True})$은 $D$ 차원에 걸쳐 평균을 내어 형태가 $(B, 1, 1, 1)$이다. (3) $N_x = G_x / (\text{mean} + \epsilon)$의 형태는 $(B, 1, 1, D)$이며, 채널마다의 반응을 채널 평균에 견준 값이다. 마지막 출력 $\gamma \cdot (x \cdot N_x) + \beta + x$의 형태는 $(B, H, W, D)$이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
GRN과 SE(Squeeze-and-Excitation) 블록을 비교하여라. 채널 반응을 조절하는 방식에서 무엇이 닮고 무엇이 다른가?

</div>

??? success "풀이"
    GRN과 SE 블록은 둘 다 전역 공간 정보로 채널마다 피처의 스케일을 다시 조정한다. SE 블록은 전역 평균 풀링으로 공간 차원을 압축하고 시그모이드를 쓰는 병목 MLP로 채널 웨이트를 학습하여 $[0, 1]$의 웨이트를 낸다. GRN은 그 대신 공간 차원에 걸쳐 $L^2$ 노름을 계산하고 채널에 걸친 평균으로 정규화하여 유계가 아닌 스케일 값을 낸다. 핵심 차이는 이렇다. (1) SE는 학습된 비선형 변환(MLP)을 쓰고 GRN은 고정된 정규화 식에 학습된 아핀 파라메터를 곁들인다. (2) SE의 웨이트는 시그모이드로 유계이고 GRN의 스케일은 유계가 아니다. (3) GRN에는 잔차 연결이 처음부터 들어 있다. (4) GRN의 파라메터는 $2D$개뿐이라 SE($r$이 축소 비율일 때 $2D^2/r$개)보다 훨씬 적다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
NHWC로 바꾸지 않고 NCHW 형태에서 동작하여 일반적인 컨볼루션 레이어와 호환되는 GRN 클래스를 구현하여라.

</div>

??? success "풀이"
    ```python
    class GRN_NCHW(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

        def forward(self, x):
            # x: (B, C, H, W)
            Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
            Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)    # (B, C, 1, 1)
            return self.gamma * (x * Nx) + self.beta + x
    ```
    핵심 변경은 노름의 차원을 $(1, 2)$에서 $(2, 3)$으로, 평균의 차원을 $-1$에서 $1$로 바꾸고, 학습 가능한 파라메터의 형태를 $(1, D, 1, 1)$로 바꾸어 NCHW 텐서와 브로드캐스팅이 되게 하는 것이다.

## 정리하며

**다룬 것** — ConvNeXt V2

전역 반응 정규화(GRN) 레이어가 ConvNeXt V2 구조의 핵심이다.

핵심 클래스는 `GRN`, `ConvNeXtV2Block`, `ConvNeXtV2`, `GRN_NCHW`이며 앞의 연습문제 3개로 스스로 확인해 볼 수 있다.
