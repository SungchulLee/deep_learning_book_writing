# 배치, 미니배치, 확률적 경사 하강법
## 들어가며

표본이 수천 개 또는 수백만 개인 데이터셋으로 학습할 때, 매 반복마다 데이터셋 전체에 대한 경사를 계산하는 것은 계산 비용이 크다. 이 절에서는 **각 경사 갱신에 데이터를 얼마나 쓰는가** 에서 차이가 나는 경사 하강법의 세 가지 근본적인 변형을 살펴본다.

미니배치 경사 하강법이 신경망 학습의 사실상 표준이 되었으므로, 이 변형들을 이해하는 것은 실용적인 기계학습에 필수적이다.

## 세 가지 변형

### 개요

| 변형 | 배치 크기 | 경사 계산 |
|---------|------------|---------------------|
| **배치 경사 하강법** | $N$(데이터셋 전체) | 정확한 경사 |
| **확률적 경사 하강법(SGD)** | 1(표본 하나) | 잡음이 매우 많은 추정 |
| **미니배치 경사 하강법** | $B$(보통 16-256) | 균형 잡힌 추정 |

### 수학적 정식화

표본 $N$개에 대한 손실 함수는 다음과 같다.

$$L(\theta) = \frac{1}{N}\sum_{i=1}^{N} \ell(\theta; x_i, y_i)$$

**배치 경사 하강법**(전체 경사):

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{N}\sum_{i=1}^{N} \nabla_\theta \ell(\theta_t; x_i, y_i)$$

**확률적 경사 하강법**(표본 하나):

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \ell(\theta_t; x_{i_t}, y_{i_t})$$

여기서 $i_t$는 무작위로 추출된다.

**미니배치 경사 하강법**(크기 $B$인 배치):

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{B}\sum_{j \in \mathcal{B}_t} \nabla_\theta \ell(\theta_t; x_j, y_j)$$

여기서 $\mathcal{B}_t$는 무작위로 추출된 미니배치이다.

## 배치 경사 하강법

### 알고리즘

```python
def batch_gradient_descent(X, y, model, criterion, 
                           learning_rate, n_epochs):
    """
    Batch Gradient Descent: Use ALL data in each iteration
    """
    for epoch in range(n_epochs):
        # 전체 데이터셋에 대한 순전파
        y_pred = model(X)
        
        # 모든 표본에 대한 손실 계산
        loss = criterion(y_pred, y)
        
        # 모든 표본에 대한 경사 계산
        loss.backward()
        
        # 매개변수 갱신
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad
        
        # 경사 초기화
        model.zero_grad()
    
    return model
```

### 특징

**장점:**

- 손실의 **정확한 경사** 를 계산한다
- 수렴 궤적이 매끄럽고 안정적이다
- (볼록 문제에서는) 최솟값으로의 수렴이 보장된다

**단점:**

- 큰 데이터셋에서는 계산 비용이 크다
- 메모리를 많이 쓴다(데이터 전체를 올려야 한다)
- 얕은 국소 최솟값을 벗어날 수 없다
- 갱신이 느리다(에폭당 한 번)

### 쓰기 좋은 경우

- 작은 데이터셋($N < 1000$)
- 정확한 경사가 중요한 문제
- 볼록 최적화 문제
- 메모리가 제약이 되지 않을 때

## 확률적 경사 하강법(SGD)

### 알고리즘

```python
def stochastic_gradient_descent(X, y, model, criterion,
                                 learning_rate, n_epochs):
    """
    Stochastic Gradient Descent: Use ONE sample per iteration
    """
    n_samples = X.shape[0]
    
    for epoch in range(n_epochs):
        # 각 에폭 시작 때 데이터 섞기
        indices = torch.randperm(n_samples)
        
        for i in indices:
            # 표본 하나 선택
            X_sample = X[i:i+1]
            y_sample = y[i:i+1]
            
            # 표본 하나에 대한 순전파
            y_pred = model(X_sample)
            
            # 표본 하나에 대한 손실 계산
            loss = criterion(y_pred, y_sample)
            
            # 역전파
            loss.backward()
            
            # 매개변수 갱신
            with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param.grad
            
            model.zero_grad()
    
    return model
```

### 특징

**장점:**

- 반복 하나가 매우 빠르다
- 국소 최솟값을 벗어날 수 있다(잡음이 탐색을 돕는다)
- 온라인 학습이 가능하다(데이터가 도착하는 대로 처리)
- 메모리 사용량이 적다

**단점:**

- 경사 추정에 잡음이 매우 많다
- 수렴이 불규칙하다(분산이 크다)
- 정확한 최솟값에 수렴하지 않을 수 있다
- 벡터화와 GPU 병렬성을 활용할 수 없다

### 잡음과 탐색의 절충

SGD의 경사 잡음은 흔히 **이롭다.**

$$\nabla_\theta \ell(\theta; x_i, y_i) = \nabla_\theta L(\theta) + \epsilon_i$$

여기서 $\epsilon_i$는 표본 하나만 쓰는 데서 오는 "잡음"이다.

이 잡음은 다음을 돕는다.

- 얕은 국소 최솟값에서 벗어나기
- 손실 지형을 더 넓게 탐색하기
- 일반화가 더 잘되는 해(더 평평한 최솟값)로 이끌기

## 미니배치 경사 하강법

### 알고리즘

```python
def minibatch_gradient_descent(X, y, model, criterion,
                                learning_rate, n_epochs, batch_size):
    """
    Mini-Batch Gradient Descent: Best of both worlds
    """
    # 자동 배치 구성을 위한 DataLoader 생성
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(n_epochs):
        for X_batch, y_batch in dataloader:
            # 미니배치에 대한 순전파
            y_pred = model(X_batch)
            
            # 미니배치에 대한 손실 계산
            loss = criterion(y_pred, y_batch)
            
            # 역전파
            loss.backward()
            
            # 매개변수 갱신
            with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param.grad
            
            model.zero_grad()
    
    return model
```

### PyTorch 자료 불러오개

```python
from torch.utils.data import TensorDataset, DataLoader

# 데이터셋 생성
dataset = TensorDataset(X, y)

# 미니배치를 쓰는 DataLoader 생성
train_loader = DataLoader(
    dataset,
    batch_size=64,        # Mini-batch size
    shuffle=True,         # Shuffle each epoch
    num_workers=4,        # Parallel data loading
    pin_memory=True       # Faster GPU transfer
)

# 학습 루프
for epoch in range(n_epochs):
    for X_batch, y_batch in train_loader:
        # 미니배치에 대한 학습 단계
        ...
```

### 특징

**장점:**

- 경사의 정확도와 계산량 사이의 균형을 잡는다
- GPU 병렬성을 효율적으로 활용한다
- 적당한 잡음(탐색에 충분하되 지나치지 않다)
- 실제 문제에 가장 실용적이다

**단점:**

- batch_size라는 하이퍼파라미터가 생긴다
- 경사가 여전히 (정확한 값이 아닌) 추정값이다
- 배치 통계가 달라질 수 있다(BatchNorm에 영향을 준다)

## 비교

### 수렴 양상

```
Loss
  │
  │╲  Batch GD (smooth)
  │ ╲__________
  │
  │╱╲
  │  ╲╱╲╱╲_____  Mini-batch GD (moderate noise)
  │
  │╱╲╱╲
  │   ╲╱╲╱╲╱╲_  SGD (noisy)
  │
  └──────────────────→ Iterations
```

### 계산량 비교

크기 $N = 10,000$인 데이터셋에 대해 다음과 같다.

| 지표 | 배치 GD | SGD | 미니배치($B=64$) |
|--------|----------|-----|-------------------|
| 갱신당 표본 수 | 10,000 | 1 | 64 |
| 에폭당 갱신 횟수 | 1 | 10,000 | 156 |
| 단계당 메모리 | 많음 | 적음 | 중간 |
| GPU 활용도 | 좋음 | 나쁨 | 매우 좋음 |
| 경사의 분산 | 0 | 매우 큼 | 중간 |

### 경험적 예제

```python
import time

# 준비
n_samples = 10000
X = torch.randn(n_samples, 10)
y = torch.randn(n_samples, 1)

# 10 에폭에 대한 시간 비교
# 결과는 하드웨어에 따라 달라진다

# 배치 경사 하강법
# 시간: 약 0.3초, 최종 손실: 1.0012

# SGD(표본 1개)
# 시간: 약 45초, 최종 손실: 1.0089

# 미니배치(batch_size=64)
# 시간: 약 0.8초, 최종 손실: 1.0015
```

## 배치 크기 선택

### 일반적인 지침

| 배치 크기 | 용도 |
|------------|----------|
| 1 | 순수 SGD(실무에서는 거의 쓰지 않는다) |
| 8-32 | 메모리가 제한적이고 일반화가 좋다 |
| 64-256 | 흔한 기본 범위 |
| 512-2048 | 대규모 학습. 학습률 조정이 필요하다 |
| 4096+ | 분산 학습(세심한 조정 필요) |

### 선택에 영향을 주는 요인

1. **GPU 메모리**: 배치가 클수록 메모리가 더 필요하다
2. **일반화**: 배치가 작을수록 일반화가 잘되는 경우가 많다
3. **수렴 속도**: 배치가 크면 더 큰 학습률을 쓸 수 있다
4. **하드웨어 효율**: 배치 크기는 2의 거듭제곱이 좋다

### 2의 거듭제곱

```python
# 좋은 예: 하드웨어 효율을 위한 2의 거듭제곱
batch_sizes = [16, 32, 64, 128, 256, 512]

# 피할 것: 임의의 크기
# batch_size = 100  # 덜 효율적이다
```

## 경사 분산 분석

### 분산 감소

미니배치 경사 추정의 분산은 다음과 같다.

$$\text{Var}[\nabla_\theta L_B] = \frac{\sigma^2}{B}$$

여기서 $\sigma^2$은 표본 하나에 대한 경사의 분산이다.

**함의:**

- 배치 크기를 두 배로 하면 경사의 분산이 절반이 된다
- $B \to N$(배치 경사 하강법)일 때 분산이 0에 가까워진다
- 수확 체감이 있다. $B=64$ 대 $B=128$의 차이는 $B=1$ 대 $B=64$의 차이보다 작다

### 잡음과 일반화

연구에 따르면 작은 배치에서 오는 경사 잡음은 다음으로 이어질 수 있다.

- 더 평평한 최솟값(더 나은 일반화)
- 암묵적 정칙화
- 뾰족한 국소 최솟값에서의 탈출

> "SGD의 잡음은 정칙화 역할을 한다." — Keskar et al., 2017

## 구현 세부 사항

### 섞기

**왜 섞는가?**

- 순서에 의존하는 패턴을 학습하는 것을 막는다
- 각 미니배치가 대표성을 갖도록 한다
- 일반화를 개선한다

```python
# PyTorch가 섞기를 자동으로 처리한다
DataLoader(dataset, shuffle=True)

# 직접 섞기
indices = torch.randperm(len(dataset))
shuffled_data = data[indices]
```

### 마지막 배치 처리하기

$N$이 $B$로 나누어떨어지지 않을 때 쓴다.

```python
# 선택 1: 채워지지 않은 배치를 버린다
DataLoader(dataset, batch_size=64, drop_last=True)

# 선택 2: 채워지지 않은 배치를 유지한다(기본값)
DataLoader(dataset, batch_size=64, drop_last=False)
```

### 경사 누적

제한된 메모리로 더 큰 배치 크기를 흉내 낸다.

```python
accumulation_steps = 4
effective_batch_size = batch_size * accumulation_steps

for i, (X_batch, y_batch) in enumerate(train_loader):
    y_pred = model(X_batch)
    loss = criterion(y_pred, y_batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 실용적인 학습 루프

### 완전한 예제

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 모델
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# 데이터
X_train = torch.randn(1000, 10)
y_train = torch.randn(1000, 1)
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 학습 준비
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 학습 루프
n_epochs = 100
for epoch in range(n_epochs):
    epoch_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        # 순전파
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        
        # 역전파
        optimizer.zero_grad()
        loss.backward()
        
        # 갱신
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
```

## 핵심 요점

1. **배치 경사 하강법**: 정확한 경사, 느림, 안정적인 수렴
2. **SGD**: 빠른 반복, 잡음이 많음, 좋은 탐색
3. **미니배치**: 최선의 절충이며 실무의 표준
4. **배치 크기가 중요하다**: 속도, 메모리, 일반화에 영향을 준다
5. **DataLoader를 쓴다**: 배치 구성, 섞기, 병렬화를 처리해 준다
6. **32-64로 시작한다**: 대부분의 문제에 합리적인 기본값
7. **경사 잡음이 도움이 된다**: 어느 정도의 잡음은 일반화에 이롭다

## 다른 주제와의 연결

- **학습률**: 배치 크기와 상호작용한다. [학습률](learning_rate.md) 참고
- **모멘텀**: 실효 잡음을 줄인다. [고전적 모멘텀](../../ch05/optimizers/momentum.md) 참고
- **배치 정규화**: 통계가 배치 크기에 의존한다. 배치 정규화 참고
- **분산 학습**: 배치 크기를 세심하게 조정해야 한다

## 참고 문헌

- Bottou, L. (2010). Large-scale machine learning with stochastic gradient descent. COMPSTAT.
- Keskar, N. S., et al. (2017). On large-batch training for deep learning: Generalization gap and sharp minima. ICLR.
- Smith, S. L., et al. (2018). Don't decay the learning rate, increase the batch size. ICLR.
- Goyal, P., et al. (2017). Accurate, large minibatch SGD: Training ImageNet in 1 hour.

## 연습문제

**연습문제 1.**
표본이 $N = 1000$개인 데이터셋에 대해 배치 경사 하강법($B = N$), 미니배치 경사 하강법($B = 32$), SGD($B = 1$)의 에폭당 경사 계산 횟수를 비교하라. 어느 쪽이 에폭당 매개변수 갱신을 더 많이 하는가?

??? success "연습문제 1 풀이"
    배치 경사 하강법: 1000개 표본 전체로 경사 계산 1회, 매개변수 갱신 1회.

    미니배치 경사 하강법($B=32$): $\lceil 1000/32 \rceil = 32$회 경사 계산, 32회 매개변수 갱신.

    SGD($B=1$): 1000회 경사 계산, 1000회 매개변수 갱신.

    SGD가 에폭당 가장 많이 갱신하지만(1000회) 각 갱신이 잡음이 매우 많은 경사 추정을 쓴다. 미니배치 경사 하강법이 균형을 잡는다. 에폭당 32회 갱신하며 각각 표본 32개로부터 얻은 꽤 정확한 경사를 쓴다.

---

**연습문제 2.**
미니배치 경사의 기댓값이 전체 배치 경사와 같음을 증명하라. 그런 다음 배치 크기를 $B$라 할 때 분산이 $O(1/B)$로 감소함을 보여라.

??? success "연습문제 2 풀이"
    표본 $i$에 대해 $g_i = \nabla_\theta \ell(x_i, \theta)$라 하자. 미니배치 경사는 $i_j$가 균등하게 추출될 때 $\hat{g} = \frac{1}{B}\sum_{j=1}^B g_{i_j}$이다.

    $\mathbb{E}[\hat{g}] = \frac{1}{B}\sum_{j=1}^B \mathbb{E}[g_{i_j}] = \frac{1}{B} \cdot B \cdot \frac{1}{N}\sum_{i=1}^N g_i = \bar{g}$이며 이것이 전체 배치 경사이다.

    분산은 (복원 추출을 가정하면) $\sigma^2 = \text{Var}(g_i)$일 때 $\text{Var}(\hat{g}) = \frac{1}{B^2}\sum_{j=1}^B \text{Var}(g_{i_j}) = \frac{\sigma^2}{B}$이다. 이로써 $O(1/B)$의 분산 감소가 확인된다. $\square$

---

**연습문제 3.**
물리적 배치 크기 32로 실효 배치 크기 256을 흉내 내는 경사 누적을 구현하라. 매개변수 갱신이 일치함을 확인하라.

??? success "연습문제 3 풀이"
    ```python
    import torch
    import torch.nn as nn

    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    accumulation_steps = 8  # 32 * 8 = 256

    for i, (x, y) in enumerate(dataloader):  # dataloader has batch_size=32
        loss = nn.MSELoss()(model(x), y) / accumulation_steps
        loss.backward()  # gradients accumulate
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    ```
    손실을 `accumulation_steps`로 나누면 누적된 경사가 크기 256인 배치 하나의 경사와 같아진다.

---

**연습문제 4.**
배치 경사 하강법이 갇히는 뾰족한 국소 최솟값을 미니배치 SGD가 벗어날 수 있는 이유를 설명하라. 답을 경사 추정의 잡음과 연결하라.

??? success "연습문제 4 풀이"
    미니배치 SGD는 경사에 암묵적인 잡음을 더한다. $\hat{g} = \bar{g} + \epsilon$이며 $\epsilon$의 분산은 $\sigma^2/B$이다. 이 잡음이 정칙화 역할을 하여, (곡률이 크고 교란에 민감한) 뾰족한 최솟값에서는 벗어나게 하면서도 (잡음에 강건한) 평평한 최솟값에는 머무르게 한다. 배치 경사 하강법은 정확한 경사를 계산하고 결정적으로 그것을 따르므로, 뾰족한 최솟값의 흡인 영역에 들어가면 갇힌 채로 남는다. SGD 잡음에서 오는 이 암묵적 정칙화가 배치 크기가 작을수록 일반화가 잘되는 경우가 많은 이유 중 하나이다.
