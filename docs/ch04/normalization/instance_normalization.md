# 인스턴스 정규화

인스턴스 정규화의 구현과 예제. 인스턴스 정규화는 표본마다, 채널마다 독립적으로 정규화한다.

정규화 기법을 이해하는 것은 깊은 신경망을 효과적으로 만들고 학습시키는 데 필수적이다. 이 구현은 그 핵심 개념을 PyTorch로 보여주며, 현대적인 구조의 구성 요소를 직접 다뤄 볼 기회를 준다.

## 1. 코드

```python
"""
사례 정규화의 구현과 예제
===================================================

사례 정규화는 표본마다, 채널마다 따로 정규화한다.
배치 통계가 섞이면 안 되는 양식 전이와 GAN에서 널리 쓰인다.

논문: "Instance Normalization: The Missing Ingredient for Fast Stylization"
       (Ulyanov et al., 2016)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class InstanceNorm2dNumPy:
    """
    NumPy로 바닥부터 구현한 사례 정규화.
    표본마다, 채널마다 따로 정규화한다.
    """
    
    def __init__(self, num_features, eps=1e-5, affine=True):
        """
        인수:
            num_features: 채널의 수 (C)
            eps: 수치 안정성을 위한 작은 상수
            affine: True이면 gamma와 beta 매개변수를 배운다
        """
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            # 채널마다의 학습 가능한 매개변수
            self.gamma = np.ones((1, num_features, 1, 1))
            self.beta = np.zeros((1, num_features, 1, 1))
        
    def forward(self, x):
        """
        사례 정규화의 순전파.
        
        인수:
            x: 모양이 (N, C, H, W)인 입력
            
        반환값:
            같은 모양의 정규화된 출력
        """
        # 사례별, 채널별로 평균과 분산 계산
        # 각 (N, C)에 대해 공간 차원 (H, W)으로 평균
        mean = np.mean(x, axis=(2, 3), keepdims=True)
        var = np.var(x, axis=(2, 3), keepdims=True)
        
        # 정규화
        x_normalized = (x - mean) / np.sqrt(var + self.eps)
        
        # 켜져 있으면 아핀 변환 적용
        if self.affine:
            x_normalized = self.gamma * x_normalized + self.beta
        
        return x_normalized


class StyleTransferNetwork(nn.Module):
    """
    사례 정규화를 쓰는 양식 전이 신경망.
    사례 정규화는 사례별 대비 정보를 없애 양식 전이를 더 효과적으로 만들므로
    양식 전이에서 매우 중요하다.
    """
    
    def __init__(self):
        super(StyleTransferNetwork, self).__init__()
        
        # 부호기
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
        )
        
        # 잔차 블록
        self.residual_blocks = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
        )
        
        # 복호기
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.residual_blocks(x)
        x = self.decoder(x)
        return x


class ResidualBlock(nn.Module):
    """
    사례 정규화를 갖춘 잔차 블록.
    """
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
        )
    
    def forward(self, x):
        return x + self.conv_block(x)


class GeneratorWithInstanceNorm(nn.Module):
    """
    사례 정규화를 쓰는 GAN 생성기.
    이미지 대 이미지 변환에서 흔하다 (예: CycleGAN, Pix2Pix).
    """
    
    def __init__(self, input_channels=3, output_channels=3, ngf=64):
        super(GeneratorWithInstanceNorm, self).__init__()
        
        # 첫 합성곱
        model = [
            nn.Conv2d(input_channels, ngf, kernel_size=7, padding=3),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]
        
        # 하향 표본화
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(inplace=True)
            ]
        
        # 잔차 블록
        mult = 2 ** n_downsampling
        for i in range(9):
            model += [ResidualBlock(ngf * mult)]
        
        # 상향 표본화
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                  kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(inplace=True)
            ]
        
        # 출력층
        model += [
            nn.Conv2d(ngf, output_channels, kernel_size=7, padding=3),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)


def demonstrate_instance_norm():
    """
    사례 정규화가 어떻게 작동하는지 보인다.
    """
    print("=" * 60)
    print("Instance Normalization Demonstration")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 예시 데이터 만들기: 이미지 2장, 채널 3개, 공간 4x4
    batch_size, channels, height, width = 2, 3, 4, 4
    
    # 이미지별, 채널별로 통계가 다른 데이터 만들기
    x = np.random.randn(batch_size, channels, height, width)
    
    # 채널마다 척도를 다르게 하기
    x[0, 0] *= 10   # 이미지 1, 채널 1: 큰 값
    x[0, 1] *= 1    # 이미지 1, 채널 2: 보통 값
    x[0, 2] *= 0.1  # 이미지 1, 채널 3: 작은 값
    
    x[1, 0] *= 5    # 이미지 2, 채널 1: 중간 값
    x[1, 1] *= 15   # 이미지 2, 채널 2: 아주 큰 값
    x[1, 2] *= 2    # 이미지 2, 채널 3: 보통 값
    
    print("\nOriginal data statistics:")
    for n in range(batch_size):
        print(f"\nImage {n}:")
        for c in range(channels):
            mean = np.mean(x[n, c])
            std = np.std(x[n, c])
            print(f"  Channel {c}: mean={mean:6.2f}, std={std:6.2f}")
    
    # 사례 정규화 적용
    instance_norm = InstanceNorm2dNumPy(channels)
    x_normalized = instance_norm.forward(x)
    
    print("\nAfter Instance Normalization:")
    for n in range(batch_size):
        print(f"\nImage {n}:")
        for c in range(channels):
            mean = np.mean(x_normalized[n, c])
            std = np.std(x_normalized[n, c])
            print(f"  Channel {c}: mean={mean:6.2f}, std={std:6.2f}")
    
    print("\nKey observations:")
    print("- Each (image, channel) pair is normalized independently")
    print("- Mean ≈ 0 and Std ≈ 1 for EACH channel of EACH image")
    print("- No mixing of statistics across samples or channels")


def compare_all_normalizations():
    """
    배치 정규화, 층 정규화, 사례 정규화를 나란히 비교한다.
    """
    print("\n" + "=" * 60)
    print("Comparing All Normalization Methods")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # 예시 데이터 만들기: (N=2, C=3, H=4, W=4)
    x = torch.randn(2, 3, 4, 4) * 10
    
    print("\nInput shape: (N=2, C=3, H=4, W=4)")
    print("N=batch, C=channels, H=height, W=width")
    
    # 배치 정규화 (각 C에 대해 N, H, W로 정규화)
    bn = nn.BatchNorm2d(3)
    bn.eval()
    x_bn = bn(x)
    
    # 층 정규화 (각 N에 대해 C, H, W로 정규화)
    ln = nn.LayerNorm([3, 4, 4])
    x_ln = ln(x)
    
    # 사례 정규화 (각 N, C에 대해 H, W로 정규화)
    instance_norm = nn.InstanceNorm2d(3, affine=False)
    x_in = instance_norm(x)
    
    # G=3인 그룹 정규화 (채널마다 자기 그룹)
    gn = nn.GroupNorm(3, 3)  # 채널마다 한 그룹이면 사례 정규화와 비슷하다
    x_gn = gn(x)
    
    print("\n" + "-" * 60)
    print("Statistics after normalization:")
    print("-" * 60)
    
    print("\nBatch Norm:")
    print(f"  Normalizes over: (N, H, W) for each C")
    print(f"  Mean per channel: {x_bn.mean(dim=(0, 2, 3))}")
    print(f"  Std per channel:  {x_bn.std(dim=(0, 2, 3))}")
    
    print("\nLayer Norm:")
    print(f"  Normalizes over: (C, H, W) for each N")
    print(f"  Mean per sample: {x_ln.mean(dim=(1, 2, 3))}")
    print(f"  Std per sample:  {x_ln.std(dim=(1, 2, 3))}")
    
    print("\nInstance Norm:")
    print(f"  Normalizes over: (H, W) for each N and C")
    for n in range(2):
        print(f"  Sample {n}:")
        for c in range(3):
            mean = x_in[n, c].mean()
            std = x_in[n, c].std()
            print(f"    Channel {c}: mean={mean:.4f}, std={std:.4f}")
    
    print("\n" + "=" * 60)
    print("Summary of Normalization Methods")
    print("=" * 60)
    
    comparison = """
    방법            | 정규화 대상 축  | 쓰임새
    ----------------|-----------------|----------------------------------
    배치 정규화     | N, H, W         | CNN, 큰 배치
    층 정규화       | C, H, W         | RNN, 트랜스포머, 작은 배치
    사례 정규화     | H, W            | 양식 전이, GAN
    그룹 정규화     | (H, W, C/G)     | 작은 배치, 배치 정규화가 안 통할 때
    """
    print(comparison)


def demonstrate_style_transfer_example():
    """
    왜 양식 전이에서 사례 정규화가 중요한지 보인다.
    """
    print("\n" + "=" * 60)
    print("Why Instance Norm for Style Transfer?")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # 내용 이미지 모의실험 (밝음)
    content = torch.randn(1, 3, 32, 32) + 2.0
    
    # 양식 이미지 모의실험 (어두움)
    style = torch.randn(1, 3, 32, 32) - 2.0
    
    print("\nOriginal statistics:")
    print(f"Content image mean: {content.mean():.4f}, std: {content.std():.4f}")
    print(f"Style image mean:   {style.mean():.4f}, std: {style.std():.4f}")
    
    # 배치 정규화를 쓸 때 (이미지 사이에서 통계가 섞인다)
    bn = nn.BatchNorm2d(3)
    bn.eval()
    combined_bn = torch.cat([content, style], dim=0)
    normalized_bn = bn(combined_bn)
    
    print("\nWith Batch Normalization (not ideal):")
    print(f"Normalized content mean: {normalized_bn[0].mean():.4f}")
    print(f"Normalized style mean:   {normalized_bn[1].mean():.4f}")
    print("→ Statistics are mixed across images!")
    
    # 사례 정규화를 쓸 때 (서로 독립)
    instance_norm = nn.InstanceNorm2d(3, affine=False)
    content_in = instance_norm(content)
    style_in = instance_norm(style)
    
    print("\nWith Instance Normalization (ideal):")
    print(f"Normalized content mean: {content_in.mean():.4f}")
    print(f"Normalized style mean:   {style_in.mean():.4f}")
    print("→ Each image normalized independently!")
    
    print("\nKey insight:")
    print("Instance Norm removes instance-specific contrast information,")
    print("allowing the network to focus on transferring style features")
    print("without being influenced by the original image's brightness/contrast.")


if __name__ == "__main__":
    demonstrate_instance_norm()
    compare_all_normalizations()
    demonstrate_style_transfer_example()
    
    print("\n" + "=" * 60)
    print("When to use Instance Normalization:")
    print("=" * 60)
    print("✓ Style transfer networks")
    print("✓ GANs (especially image-to-image translation)")
    print("✓ When each sample should be processed independently")
    print("✓ When batch statistics shouldn't mix")
    print("✓ Real-time applications (no running statistics needed)")
```

## 2. 논의

이 구현은 4개의 클래스(`InstanceNorm2dNumPy`, `StyleTransferNetwork`, `ResidualBlock`, `GeneratorWithInstanceNorm`)를 정의하며, 이들이 함께 작동하여 완전한 정규화 기법 구조를 이룬다. 각 클래스가 서로 다른 구성 요소를 감싸므로 코드가 모듈식이 되고 확장하기 쉬워진다. `forward` 메서드들이 PyTorch가 자동 미분에 사용하는 계산 그래프를 정의한다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 학습 최적화 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
`InstanceNorm2dNumPy`의 순전파를 따라가며 텐서의 모양을 추적하라. 기본 매개변수로 입력 표본 4개의 배치를 넣었을 때, 주요 연산(합성곱, 풀링, 선형 층)마다 그 뒤의 모양을 적어라.

??? success "연습문제 1 풀이"
    입력 모양에서 출발하여 각 층을 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 (덧대기가 없으면) $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌거나 (`padding=k//2`이면) 그대로 유지된다. 커널이 2인 풀링은 공간 차원을 절반으로 만든다. 선형 층은 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임에 유의하며 추적한다. 중간 모양을 합성곱 층에서는 $(B, C, H, W)$로, 평탄화 후에는 $(B, F)$로 적는다.

---

**연습문제 2.**
$64 \times 64$ 크기의 RGB 이미지(입력 모양 $3 \times 64 \times 64$)를 받도록 구조를 수정하라. 모든 층의 차원을 그에 맞게 고치고 모델이 오류 없이 실행되는지 확인하라.

??? success "연습문제 2 풀이"
    첫 번째 합성곱 층의 `in_channels`를 현재 값에서 3으로 바꾼다. 공식 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$을 써서 합성곱 층과 풀링 층마다 공간 차원을 다시 계산한다. 마지막 합성곱/풀링 층의 평탄화된 출력에 맞도록 첫 번째 선형 층의 `in_features`를 고친다. 다음으로 확인한다. `model = InstanceNorm2dNumPy(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`.

---

**연습문제 3.**
같은 입출력 차원에서 표준 합성곱과 깊이별 분리 합성곱의 매개변수 개수와 FLOPs를 비교하라. 계산 절감이 가장 큰 것은 언제인가?

??? success "연습문제 3 풀이"
    표준 `Conv2d(C_in, C_out, k)`은 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개의 매개변수를 갖는다. 깊이별 분리 합성곱은 이를 둘로 나눈다. (1) 깊이별: $C_{{\text{{in}}}} \times k^2$개(입력 채널마다 필터 하나), (2) 점별: $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 합성곱)이다. 매개변수의 비는 대략 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 매개변수가 약 $8{-}9\times$ 적어진다. 절감은 $C_{{\text{{out}}}}$과 $k$가 모두 클 때 가장 크다.

---

**연습문제 4.**
층이나 블록의 개수를 설정할 수 있도록 `InstanceNorm2dNumPy`를 확장하라. `__init__`에 `num_layers` 매개변수를 추가하고 `nn.ModuleList`로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 순회한다. (평범한 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 PyTorch가 모든 매개변수를 최적화 대상으로 등록한다. 시험은 다음과 같이 한다. `for n in [2, 4, 8]: model = InstanceNorm2dNumPy(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.

## 정리하며

**다룬 것** — 인스턴스 정규화

이 구현은 4개의 클래스(`InstanceNorm2dNumPy`, `StyleTransferNetwork`, `ResidualBlock`, `GeneratorWithInstanceNorm`)를 정의하며, 이들이 함께 작동하여 완전한 정규화 기법 구조를 이룬다.

핵심 클래스는 `InstanceNorm2dNumPy`, `StyleTransferNetwork`, `ResidualBlock`, `GeneratorWithInstanceNorm`이며 앞의 연습문제 4개로 직접 확인할 수 있다.
