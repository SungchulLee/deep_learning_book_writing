# 배치 정규화

배치 정규화의 구현과 예제. 배치 정규화는 배치 차원에 걸쳐 입력을 정규화한다.

정규화 기법을 이해하는 것은 깊은 신경망을 효과적으로 만들고 학습시키는 데 필수적이다. 이 구현은 그 핵심 개념을 PyTorch로 보여주며, 현대적인 구조의 구성 요소를 직접 다뤄 볼 기회를 준다.

## 코드

```python
"""
배치 정규화의 구현과 예제
================================================

배치 정규화는 배치 차원에 걸쳐 입력을 정규화한다.
학습을 빠르게 하고 더 높은 학습률을 쓸 수 있게 해 준다.

논문: "Batch Normalization: Accelerating Deep Network Training by
       내부 공변량 이동 줄이기"(Ioffe & Szegedy, 2015)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class BatchNorm1dNumPy:
    """
    NumPy로 바닥부터 구현한 배치 정규화.
    배치 차원에 걸쳐 정규화한다.
    """
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        """
        인수:
            num_features: 특징/채널의 수
            eps: 수치 안정성을 위한 작은 상수
            momentum: 이동 통계를 위한 모멘텀
        """
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # 학습 가능한 매개변수
        self.gamma = np.ones(num_features)  # 배율 매개변수
        self.beta = np.zeros(num_features)  # 이동 매개변수
        
        # 이동 통계 (추론용)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        self.training = True
        
    def forward(self, x):
        """
        배치 정규화의 순전파.
        
        인수:
            x: 모양이 (batch_size, num_features)인 입력
            
        반환값:
            입력과 같은 모양의 정규화된 출력
        """
        if self.training:
            # 배치 통계 계산
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            
            # 이동 통계 갱신
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            # 정규화
            x_normalized = (x - batch_mean) / np.sqrt(batch_var + self.eps)
        else:
            # 추론 중에는 이동 통계를 쓴다
            x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        
        # 배율 조정과 이동
        out = self.gamma * x_normalized + self.beta
        
        return out


class SimpleNetworkWithBatchNorm(nn.Module):
    """
    PyTorch의 BatchNorm 층을 쓰는 신경망 예제.
    """
    
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(SimpleNetworkWithBatchNorm, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # 필요하면 펼치기
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        
        return x


class ConvNetWithBatchNorm(nn.Module):
    """
    배치 정규화를 갖춘 합성곱 신경망.
    합성곱 층에 BatchNorm2d를 쓴다.
    """
    
    def __init__(self, num_classes=10):
        super(ConvNetWithBatchNorm, self).__init__()
        
        # 합성곱 블록 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 합성곱 블록 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # 합성곱 블록 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(256 * 4 * 4, num_classes)
        
    def forward(self, x):
        # 블록 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # 블록 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # 블록 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # 펼치고 분류하기
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


def demonstrate_batch_norm():
    """
    배치 정규화의 효과를 보인다.
    """
    print("=" * 60)
    print("Batch Normalization Demonstration")
    print("=" * 60)
    
    # 척도가 다른 예시 데이터 만들기
    np.random.seed(42)
    batch_size = 32
    num_features = 5
    
    # 특징 1: 작은 값
    feature1 = np.random.randn(batch_size, 1) * 0.1
    # 특징 2: 큰 값
    feature2 = np.random.randn(batch_size, 1) * 100
    # 특징 3: 중간 값
    feature3 = np.random.randn(batch_size, 1) * 10
    # 특징 4~5: 보통 값
    feature4 = np.random.randn(batch_size, 1)
    feature5 = np.random.randn(batch_size, 1)
    
    x = np.concatenate([feature1, feature2, feature3, feature4, feature5], axis=1)
    
    print("\nOriginal data statistics:")
    print(f"Mean per feature: {np.mean(x, axis=0)}")
    print(f"Std per feature:  {np.std(x, axis=0)}")
    
    # 배치 정규화 적용
    bn = BatchNorm1dNumPy(num_features)
    x_normalized = bn.forward(x)
    
    print("\nAfter Batch Normalization:")
    print(f"Mean per feature: {np.mean(x_normalized, axis=0)}")
    print(f"Std per feature:  {np.std(x_normalized, axis=0)}")
    
    print("\nKey observations:")
    print("- All features now have mean ≈ 0 and std ≈ 1")
    print("- Features are on the same scale")
    print("- Gradients can flow more easily")


def compare_with_without_batchnorm():
    """
    배치 정규화가 있을 때와 없을 때의 학습 거동을 비교한다.
    """
    print("\n" + "=" * 60)
    print("Comparison: With vs Without BatchNorm")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # 임시 데이터 만들기
    x = torch.randn(64, 784)
    
    # 배치 정규화가 없는 신경망
    net_without = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    # 배치 정규화가 있는 신경망
    net_with = nn.Sequential(
        nn.Linear(784, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    # 순전파
    with torch.no_grad():
        out_without = net_without(x)
        out_with = net_with(x)
    
    print("\nOutput statistics:")
    print(f"Without BatchNorm - Mean: {out_without.mean():.4f}, Std: {out_without.std():.4f}")
    print(f"With BatchNorm    - Mean: {out_with.mean():.4f}, Std: {out_with.std():.4f}")
    
    print("\nBenefits of Batch Normalization:")
    print("1. Faster convergence during training")
    print("2. Allows higher learning rates")
    print("3. Reduces sensitivity to initialization")
    print("4. Acts as a form of regularization")
    print("5. Reduces internal covariate shift")


if __name__ == "__main__":
    demonstrate_batch_norm()
    compare_with_without_batchnorm()
    
    print("\n" + "=" * 60)
    print("Additional Notes:")
    print("=" * 60)
    print("- BatchNorm normalizes over the batch dimension")
    print("- For Conv layers, use BatchNorm2d (normalizes per channel)")
    print("- For fully connected layers, use BatchNorm1d")
    print("- Remember to call model.eval() during inference!")```

## 논의

이 구현은 3개의 클래스(`BatchNorm1dNumPy`, `SimpleNetworkWithBatchNorm`, `ConvNetWithBatchNorm`)를 정의하며, 이들이 함께 작동하여 완전한 정규화 기법 구조를 이룬다. 각 클래스가 서로 다른 구성 요소를 감싸므로 코드가 모듈식이 되고 확장하기 쉬워진다. `forward` 메서드들이 PyTorch가 자동 미분에 사용하는 계산 그래프를 정의한다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 학습 최적화 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
`BatchNorm1dNumPy`의 순전파를 따라가며 텐서의 모양을 추적하라. 기본 매개변수로 입력 표본 4개의 배치를 넣었을 때, 주요 연산(합성곱, 풀링, 선형 층)마다 그 뒤의 모양을 적어라.

??? success "연습문제 1 풀이"
    입력 모양에서 출발하여 각 층을 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 (덧대기가 없으면) $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌거나 (`padding=k//2`이면) 그대로 유지된다. 커널이 2인 풀링은 공간 차원을 절반으로 만든다. 선형 층은 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임에 유의하며 추적한다. 중간 모양을 합성곱 층에서는 $(B, C, H, W)$로, 평탄화 후에는 $(B, F)$로 적는다.

---

**연습문제 2.**
$64 \times 64$ 크기의 RGB 이미지(입력 모양 $3 \times 64 \times 64$)를 받도록 구조를 수정하라. 모든 층의 차원을 그에 맞게 고치고 모델이 오류 없이 실행되는지 확인하라.

??? success "연습문제 2 풀이"
    첫 번째 합성곱 층의 `in_channels`를 현재 값에서 3으로 바꾼다. 공식 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$을 써서 합성곱 층과 풀링 층마다 공간 차원을 다시 계산한다. 마지막 합성곱/풀링 층의 평탄화된 출력에 맞도록 첫 번째 선형 층의 `in_features`를 고친다. 다음으로 확인한다. `model = BatchNorm1dNumPy(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`.

---

**연습문제 3.**
같은 입출력 차원에서 표준 합성곱과 깊이별 분리 합성곱의 매개변수 개수와 FLOPs를 비교하라. 계산 절감이 가장 큰 것은 언제인가?

??? success "연습문제 3 풀이"
    표준 `Conv2d(C_in, C_out, k)`은 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개의 매개변수를 갖는다. 깊이별 분리 합성곱은 이를 둘로 나눈다. (1) 깊이별: $C_{{\text{{in}}}} \times k^2$개(입력 채널마다 필터 하나), (2) 점별: $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 합성곱)이다. 매개변수의 비는 대략 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 매개변수가 약 $8{-}9\times$ 적어진다. 절감은 $C_{{\text{{out}}}}$과 $k$가 모두 클 때 가장 크다.

---

**연습문제 4.**
층이나 블록의 개수를 설정할 수 있도록 `BatchNorm1dNumPy`를 확장하라. `__init__`에 `num_layers` 매개변수를 추가하고 `nn.ModuleList`로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 순회한다. (평범한 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 PyTorch가 모든 매개변수를 최적화 대상으로 등록한다. 시험은 다음과 같이 한다. `for n in [2, 4, 8]: model = BatchNorm1dNumPy(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
