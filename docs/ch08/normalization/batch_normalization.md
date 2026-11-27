# 배치 정규화

배치 정규화의 NumPy 순전파와 PyTorch MLP·CNN 예제를 살펴본다. 벡터와 이미지에서 통계량을 구하는 축, 트레인·평가 모드와 이동 통계량을 구분한다.

배치 정규화는 피처(채널) 하나하나에 대해, 미니배치 안의 값들에서 평균 $\mu_B$와 분산 $\sigma_B^2$를 구해 표준화한 뒤 트레인되는 배율 $\gamma$와 이동 $\beta$를 적용한다.

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \qquad y = \gamma \hat{x} + \beta$$

추론 때는 배치가 하나뿐일 수도 있어 배치 통계량을 쓸 수 없다. 그래서 트레인하는 동안 배치 평균과 분산을 지수 이동 평균으로 누적해 두는데, 이것이 **이동 통계량**(running statistics)이다. 코드의 `momentum`은 새 배치의 통계량을 그 누적값에 섞는 비율이다.

## 1. 코드

```python
"""
배치 정규화의 구현과 예제
================================================

배치 정규화는 배치 차원에 걸쳐 입력을 정규화한다.
트레인을 빠르게 하고 더 높은 러닝레이트를 쓸 수 있게 해 준다.

논문: "Batch Normalization: Accelerating Deep Network Training by
       Reducing Internal Covariate Shift"(Ioffe & Szegedy, 2015)
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
            num_features: 피처/채널의 수
            eps: 수치 안정성을 위한 작은 상수
            momentum: 이동 통계를 위한 모멘텀
        """
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # 트레인 가능한 파라메터
        self.gamma = np.ones(num_features)  # 배율 파라메터
        self.beta = np.zeros(num_features)  # 이동 파라메터
        
        # 이동 통계 (추론용)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        self.training = True
        
    def forward(self, x):
        """
        배치 정규화의 순전파.
        
        인수:
            x: 형태가 (batch_size, num_features)인 입력
            
        반환값:
            입력과 같은 형태의 정규화된 출력
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
    PyTorch의 BatchNorm 레이어를 쓰는 뉴럴 네트워크 예제.
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
    배치 정규화를 갖춘 컨볼루션 뉴럴 네트워크.
    컨볼루션 레이어에 BatchNorm2d를 쓴다.
    """
    
    def __init__(self, num_classes=10):
        super(ConvNetWithBatchNorm, self).__init__()
        
        # 컨볼루션 블록 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 컨볼루션 블록 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # 컨볼루션 블록 3
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
    
    # 스케일이 다른 예시 데이터 만들기
    np.random.seed(42)
    batch_size = 32
    num_features = 5
    
    # 피처 1: 작은 값
    feature1 = np.random.randn(batch_size, 1) * 0.1
    # 피처 2: 큰 값
    feature2 = np.random.randn(batch_size, 1) * 100
    # 피처 3: 중간 값
    feature3 = np.random.randn(batch_size, 1) * 10
    # 피처 4~5: 보통 값
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
    배치 정규화가 있을 때와 없을 때의 트레인 거동을 비교한다.
    """
    print("\n" + "=" * 60)
    print("Comparison: With vs Without BatchNorm")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # 임시 데이터 만들기
    x = torch.randn(64, 784)
    
    # 배치 정규화가 없는 뉴럴 네트워크
    net_without = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    # 배치 정규화가 있는 뉴럴 네트워크
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

NumPy 구현은 `(B, C)`의 배치 축에서 평균과 분산을 구하고 피처별 gamma와 beta를 적용한다. 이 클래스는 nn.Module이 아니며, 자동 미분이나 파라메터 갱신을 제공하지 않는 순전파 시연이다.

출력에서 표준편차가 84.37이던 피처와 0.093이던 피처가 모두 평균 0, 표준편차 1 근처로 맞춰졌다. 첫째 피처만 0.9994로 조금 작은데, 분산이 약 0.0087로 작아 분모에 더한 $\epsilon = 10^{-5}$가 드러나기 때문이다($1/\sqrt{1 + 10^{-5}/0.0087} \approx 0.9994$).

둘째 실험은 트레인하지 않은 두 네트워크의 출력 통계량(표준편차 0.1332와 0.3812)을 나란히 찍을 뿐이다. 그 아래 출력된 다섯 가지 이점은 문헌에서 알려진 내용을 적은 것이지 이 코드가 잰 결과가 아니다. 특히 다섯째인 내부 공변량 변화(internal covariate shift, 앞 레이어의 파라메터가 바뀌면서 뒤 레이어가 받는 입력의 분포가 계속 달라지는 현상)의 감소는 원 논문의 설명이며, 이후 연구(Santurkar et al., 2018)는 배치 정규화의 효과를 로스 지형이 매끄러워지는 데서 찾는다.

PyTorch BatchNorm1d는 벡터에서 배치 축을, BatchNorm2d는 채널별 배치·공간 축을 함께 사용한다. 기본 설정의 평가에서는 누적된 이동 통계량을 사용하므로 트레인과 동작이 다르다.

NumPy 코드의 running_var는 분모 B인 배치 분산을 그대로 누적한다. PyTorch는 트레인 출력에는 편향 분산 추정량(분모가 B인 추정량)을 쓰지만 이동 분산 갱신에는 불편추정량(분모가 B−1인 추정량)을 사용하므로 두 구현의 평가 출력이 항상 일치하지 않는다. [공식 문서](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)의 규약을 구분한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
NumPy 구현의 `(4, C)` 입력에서 통계량과 출력의 형태(shape)를 구하라.

</div>

??? success "풀이"
    평균과 분산, gamma와 beta는 `(C,)`이고 배치에 브로드캐스팅된다. 출력은 `(4, C)`이며 컨볼루션이나 풀링은 수행하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
ConvNetWithBatchNorm의 입력을 32×32에서 64×64로 바꾸면 마지막 선형 레이어에 어떤 수정이 필요한가?

</div>

??? success "풀이"
    세 컨볼루션은 크기를 유지하고 세 번의 풀링이 각각 절반으로 줄이므로 32 입력은 4×4, 64 입력은 8×8이 된다. 마지막 선형 레이어 입력은 256×8×8로 바꾸어야 한다. RGB 채널 수는 이미 3이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
NumPy 구현과 PyTorch의 이동 분산이 다른 이유를 설명하라.

</div>

??? success "풀이"
    NumPy 구현은 배치 분산을 불편 보정하지 않고 누적한다. PyTorch의 이동 분산에는 불편 보정이 들어간다. 트레인 시 정규화 식과 이동 통계량 갱신식을 별도로 비교해야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 풀이의 `ModuleList` 코드가 원래 모델에 바로 적용될 수 있는지 검토하라. 입력·출력 차원, 엑티베이션 함수, 생성자와 `forward`에서 필요한 수정을 설명하라.

</div>

??? success "풀이"
    아래 코드는 PyTorch 모듈의 깊이 설정을 위한 조각이다. NumPy 클래스는 nn.Module이 아니며 `parameters()`나 PyTorch식 호출 인터페이스를 제공하지 않으므로 그대로 적용할 수 없다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `ModuleList`는 하위 모듈을 등록하지만 실행 순서나 엑티베이션 함수를 자동으로 정하지 않는다. 입력 레이어와 출력 레이어는 과제의 차원을 유지하고, 은닉 레이어 사이에 필요한 비선형성을 넣어야 한다. 선형 레이어만 반복하면 아핀 함수 하나로 합쳐진다. 원래 클래스에 없는 `num_layers` 인수를 그대로 넘길 수 없으므로 생성자와 `forward`를 함께 수정해야 한다.

## 정리하며

배치 정규화에서는 통계량을 구하는 축과 이동 통계량의 규약이 중요하다. NumPy 순전파 예제와 PyTorch의 자동 미분 모듈을 구분하고 평가 모드의 차이를 확인한다.
