# 기본 잔차 블록

기본 잔차 블록 구현. 신경망의 잔차 연결(건너뛰기 연결)을 소개한다.

합성곱 구조는 요즘 컴퓨터 비전 시스템의 뼈대를 이룬다. 이 구현은 PyTorch로 잔차 신경망 설계의 핵심 개념을 보이며, 이미지 데이터에서 공간적인 특징의 위계가 어떻게 학습되는지 드러낸다.

## 1. 코드

```python
"""
기본 잔차 블록 구현
====================================
신경망의 잔차 연결(건너뛰기 연결) 소개.

핵심 개념:
H(x)를 배우는 대신 F(x) = H(x) - x를 배우므로 출력은 F(x) + x이다
덕분에 기울기가 건너뛰기 연결을 타고 신경망을 곧바로 흐른다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class BasicBlock(nn.Module):
    """
    ResNet을 위한 기본 잔차 블록
    
    구조:
    입력 -> Conv -> BN -> ReLU -> Conv -> BN -> (+) -> ReLU
              |__________________________________|
                    (건너뛰기 연결)
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # 첫 합성곱 층
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 둘째 합성곱 층
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 건너뛰기 연결 (지름길)
        self.shortcut = nn.Sequential()
        
        # 차원이 바뀌면 1x1 합성곱으로 차원을 맞춘다
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        # 주 경로
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # 건너뛰기 연결 더하기
        out += self.shortcut(x)
        
        # 마지막 활성화
        out = F.relu(out)
        
        return out


class PlainBlock(nn.Module):
    """
    견주기 위한 평범한 블록 (건너뛰기 연결 없음)
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(PlainBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


def demonstrate_gradient_flow():
    """
    잔차 연결을 지나는 기울기의 흐름을 보인다
    """
    print("=" * 60)
    print("Gradient Flow Demonstration")
    print("=" * 60)
    
    # 간단한 입력 만들기
    x = torch.randn(1, 64, 32, 32, requires_grad=True)
    
    # 잔차 블록
    res_block = BasicBlock(64, 64)
    res_output = res_block(x)
    loss_res = res_output.sum()
    loss_res.backward()
    res_grad_norm = x.grad.norm().item()
    
    # 평범한 블록
    x.grad = None  # 기울기 초기화
    plain_block = PlainBlock(64, 64)
    plain_output = plain_block(x)
    loss_plain = plain_output.sum()
    loss_plain.backward()
    plain_grad_norm = x.grad.norm().item()
    
    print(f"\nGradient norm for Residual Block: {res_grad_norm:.4f}")
    print(f"Gradient norm for Plain Block: {plain_grad_norm:.4f}")
    print(f"\nResidual connections help maintain gradient magnitude!")
    print("=" * 60)


def test_blocks():
    """
    기본 잔차 블록이 제대로 움직이는지 시험한다
    """
    print("\n" + "=" * 60)
    print("Testing Residual Blocks")
    print("=" * 60)
    
    # 차원이 같은 경우 시험
    print("\n1. Same dimensions (64 -> 64)")
    block1 = BasicBlock(64, 64)
    x1 = torch.randn(2, 64, 32, 32)
    out1 = block1(x1)
    print(f"   Input shape:  {x1.shape}")
    print(f"   Output shape: {out1.shape}")
    
    # 차원이 다른 경우 시험
    print("\n2. Different dimensions (64 -> 128, stride=2)")
    block2 = BasicBlock(64, 128, stride=2)
    x2 = torch.randn(2, 64, 32, 32)
    out2 = block2(x2)
    print(f"   Input shape:  {x2.shape}")
    print(f"   Output shape: {out2.shape}")
    print(f"   Notice: Spatial dimensions halved, channels doubled")
    
    # 매개변수 개수 세기
    print("\n3. Parameter count")
    total_params = sum(p.numel() for p in block1.parameters())
    print(f"   Total parameters in BasicBlock(64, 64): {total_params:,}")
    
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RESIDUAL CONNECTIONS - BASIC CONCEPTS")
    print("=" * 60)
    
    print("\nKey Benefits of Residual Connections:")
    print("1. Easier gradient flow (addresses vanishing gradient)")
    print("2. Enables training of very deep networks (100+ layers)")
    print("3. Learning identity function is easy (F(x) = 0)")
    print("4. Better optimization landscape")
    
    # 시험 실행
    test_blocks()
    
    # 기울기의 흐름 보이기
    demonstrate_gradient_flow()
    
    print("\n" + "=" * 60)
    print("Next: See 02_resnet_implementation.py for full ResNet architecture")
    print("=" * 60 + "\n")
```

## 2. 논의

이 구현은 클래스 두 개(`BasicBlock`, `PlainBlock`)를 정의하며, 이들이 어우러져 완전한 잔차 신경망 구조를 이룬다. 클래스마다 별개의 부품을 감싸므로 코드가 모듈식이고 넓히기 쉽다. `forward` 메서드가 PyTorch의 자동 미분이 쓰는 계산 그래프를 정의한다.

손실 계산은 모델의 출력을 최적화 목표와 이어 준다. 알맞은 손실 함수를 고르는 일은 결정적으로 중요하다. 손실 함수가 모델이 무엇을 최적화하도록 배울지를 정하며, 학습된 표현과 결정 경계를 직접 빚어내기 때문이다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 딥러닝 구조 설계에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
`BasicBlock`의 순전파를 따라가며 텐서의 모양을 추적하라. 기본 매개변수로 입력 표본 4개의 배치에 대해 주요 연산(합성곱, 풀링, 선형층)마다의 모양을 적어라.

??? success "연습문제 1 풀이"
    입력 모양에서 출발하여 각 층을 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 (덧대기가 없으면) $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌거나 (`padding=k//2`이면) 그대로 유지된다. 커널이 2인 풀링은 공간 차원을 절반으로 만든다. 선형 층은 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임에 유의하며 추적한다. 중간 모양을 합성곱 층에서는 $(B, C, H, W)$로, 평탄화 후에는 $(B, F)$로 적는다.

---

**연습문제 2.**
$64 \times 64$ 크기의 RGB 이미지(입력 모양 $3 \times 64 \times 64$)를 받도록 구조를 수정하라. 모든 층의 차원을 그에 맞게 고치고 모델이 오류 없이 실행되는지 확인하라.

??? success "연습문제 2 풀이"
    첫 합성곱 층의 `in_channels`을 지금 값에서 3으로 바꾸어라. 식 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$으로 합성곱과 풀링 층마다의 공간 차원을 다시 계산하라. 마지막 합성곱/풀링 층의 펼친 출력에 맞게 첫 선형층의 `in_features`을 고쳐라. `model = BasicBlock(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`으로 확인하라.

---

**연습문제 3.**
같은 입출력 차원에서 표준 합성곱과 깊이별 분리 합성곱의 매개변수 개수와 FLOPs를 비교하라. 계산 절감이 가장 큰 것은 언제인가?

??? success "연습문제 3 풀이"
    표준 `Conv2d(C_in, C_out, k)`은 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개의 매개변수를 갖는다. 깊이별 분리 합성곱은 이를 둘로 나눈다. (1) 깊이별: $C_{{\text{{in}}}} \times k^2$개(입력 채널마다 필터 하나), (2) 점별: $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 합성곱)이다. 매개변수의 비는 대략 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 매개변수가 약 $8{-}9\times$ 적어진다. 절감은 $C_{{\text{{out}}}}$과 $k$가 모두 클 때 가장 크다.

---

**연습문제 4.**
층이나 블록의 수를 설정할 수 있도록 `BasicBlock`을 확장하라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`으로 훑는다. (보통의 파이썬 리스트가 아니라) `nn.ModuleList`을 써야 PyTorch가 최적화를 위해 모든 매개변수를 등록한다. `for n in [2, 4, 8]: model = BasicBlock(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — 기본 잔차 블록

이 구현은 클래스 두 개(`BasicBlock`, `PlainBlock`)를 정의하며, 이들이 어우러져 완전한 잔차 신경망 구조를 이룬다.

핵심 클래스는 `BasicBlock`, `PlainBlock`이며 앞의 연습문제 4개로 직접 확인할 수 있다.
