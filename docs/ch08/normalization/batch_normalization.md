# 배치 정규화

배치 정규화의 NumPy 순전파와 PyTorch MLP·CNN 예제를 살펴본다. 벡터와 이미지에서 통계량을 구하는 축, 학습·평가 모드와 이동 통계량을 구분한다.

## 1. 코드

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
    print("- Remember to call model.eval() during inference!")
```

**출력:**

```
============================================================
Batch Normalization Demonstration
============================================================

Original data statistics:
Mean per feature: [-1.37307353e-02 -2.45400640e+01  4.79469433e-01  7.23479191e-02
 -7.55956637e-02]
Std per feature:  [ 0.09310947 84.37433366  9.58398634  0.99662077  0.89315136]

After Batch Normalization:
Mean per feature: [ 0.00000000e+00  2.77555756e-17  4.16333634e-17  1.73472348e-17
 -1.73472348e-17]
Std per feature:  [0.99942376 1.         0.99999995 0.99999497 0.99999373]

Key observations:
- All features now have mean ≈ 0 and std ≈ 1
- Features are on the same scale
- Gradients can flow more easily

============================================================
Comparison: With vs Without BatchNorm
============================================================

Output statistics:
Without BatchNorm - Mean: -0.0016, Std: 0.1332
With BatchNorm    - Mean: -0.0190, Std: 0.3812

Benefits of Batch Normalization:
1. Faster convergence during training
2. Allows higher learning rates
3. Reduces sensitivity to initialization
4. Acts as a form of regularization
5. Reduces internal covariate shift

============================================================
Additional Notes:
============================================================
- BatchNorm normalizes over the batch dimension
- For Conv layers, use BatchNorm2d (normalizes per channel)
- For fully connected layers, use BatchNorm1d
- Remember to call model.eval() during inference!
```

## 2. 논의

NumPy 구현은 `(B, C)`의 배치 축에서 평균과 분산을 구하고 특징별 gamma와 beta를 적용한다. 이 클래스는 nn.Module이 아니며, 자동 미분이나 매개변수 갱신을 제공하지 않는 순전파 시연이다.

PyTorch BatchNorm1d는 벡터에서 배치 축을, BatchNorm2d는 채널별 배치·공간 축을 함께 사용한다. 기본 설정의 평가에서는 누적된 이동 통계량을 사용하므로 학습과 동작이 다르다.

NumPy 코드의 running_var는 분모 B인 배치 분산을 그대로 누적한다. PyTorch는 학습 출력에는 편향 분산 추정량을 쓰지만 이동 분산 갱신에는 불편추정량을 사용하므로 두 구현의 평가 출력이 항상 일치하지 않는다. [공식 문서](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)의 규약을 구분한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
NumPy 구현의 `(4, C)` 입력에서 통계량과 출력 모양을 구하라.

</div>

??? success "연습문제 1 풀이"
    평균과 분산, gamma와 beta는 `(C,)`이고 배치에 브로드캐스팅된다. 출력은 `(4, C)`이며 합성곱이나 풀링은 수행하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
ConvNetWithBatchNorm의 입력을 32×32에서 64×64로 바꾸면 마지막 선형층에 어떤 수정이 필요한가?

</div>

??? success "연습문제 2 풀이"
    세 합성곱은 크기를 유지하고 세 번의 풀링이 각각 절반으로 줄이므로 32 입력은 4×4, 64 입력은 8×8이 된다. 마지막 선형층 입력은 256×8×8로 바꾸어야 한다. RGB 채널 수는 이미 3이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
NumPy 구현과 PyTorch의 이동 분산이 다른 이유를 설명하라.

</div>

??? success "연습문제 3 풀이"
    NumPy 구현은 배치 분산을 불편 보정하지 않고 누적한다. PyTorch의 이동 분산에는 불편 보정이 들어간다. 학습 시 정규화 식과 이동 통계량 갱신식을 별도로 비교해야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 풀이의 `ModuleList` 코드가 원래 모델에 바로 적용될 수 있는지 검토하라. 입력·출력 차원, 활성화, 생성자와 `forward`에서 필요한 수정을 설명하라.

</div>

??? success "연습문제 4 풀이"
    아래 코드는 PyTorch 모듈의 깊이 설정을 위한 조각이다. NumPy 클래스는 nn.Module이 아니며 `parameters()`나 PyTorch식 호출 인터페이스를 제공하지 않으므로 그대로 적용할 수 없다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `ModuleList`는 하위 모듈을 등록하지만 실행 순서나 활성화를 자동으로 정하지 않는다. 입력층과 출력층은 과제의 차원을 유지하고, 은닉층 사이에 필요한 비선형성을 넣어야 한다. 선형층만 반복하면 아핀 함수 하나로 합쳐진다. 원래 클래스에 없는 `num_layers` 인수를 그대로 넘길 수 없으므로 생성자와 `forward`를 함께 수정해야 한다.

## 정리하며

배치 정규화에서는 통계량을 구하는 축과 이동 통계량의 규약이 중요하다. NumPy 순전파 예제와 PyTorch의 자동 미분 모듈을 구분하고 평가 모드의 차이를 확인한다.
