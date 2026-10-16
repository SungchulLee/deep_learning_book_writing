# 배치, 미니배치, 확률적 경사 하강법

표본이 수천 개 또는 수백만 개인 데이터셋으로 학습할 때, 매 반복마다 데이터셋 전체에 대한 경사를 계산하는 것은 계산 비용이 크다. 이 절에서는 **각 경사 갱신에 데이터를 얼마나 쓰는가** 에서 차이가 나는 경사 하강법의 세 가지 근본적인 변형을 살펴본다.

미니배치 경사 하강법이 뉴럴 넷트웍 학습의 사실상 표준이 되었으므로, 이 변형들을 이해하는 것은 실용적인 기계학습에 필수적이다.

---

## 1. 세 가지 변형

### 개요

| 변형 | 배치 크기 | 경사 계산 |
|---------|------------|---------------------|
| **배치 경사 하강법** | $N$(데이터셋 전체) | 정확한 경사 |
| **확률적 경사 하강법(SGD)** | 1(표본 하나) | 잡음이 매우 많은 추정 |
| **미니배치 경사 하강법** | $B$(보통 16-256) | 균형 잡힌 추정 |

### 수학적 정식화

표본 $N$개에 대한 로스 함수는 다음과 같다.

$$L(\theta) = \frac{1}{N}\sum_{i=1}^{N} \ell(\theta; x_i, y_i)$$

**배치 경사 하강법**(전체 경사):

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{N}\sum_{i=1}^{N} \nabla_\theta \ell(\theta_t; x_i, y_i)$$

**확률적 경사 하강법**(표본 하나):

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \ell(\theta_t; x_{i_t}, y_{i_t})$$

여기서 $i_t$는 무작위로 추출된다.

**미니배치 경사 하강법**(크기 $B$인 배치):

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{B}\sum_{j \in \mathcal{B}_t} \nabla_\theta \ell(\theta_t; x_j, y_j)$$

여기서 $\mathcal{B}_t$는 무작위로 추출된 미니배치다.

---

## 2. 배치 경사 하강법

### 알고리즘

```python
def batch_gradient_descent(X, y, model, criterion, 
                           learning_rate, n_epochs):
    """
    배치 경사 하강법: 루프마다 데이터를 모두 쓴다
    """
    for epoch in range(n_epochs):
        # 전체 데이터셋에 대한 순전파
        y_pred = model(X)
        
        # 모든 표본에 대한 로스 계산
        loss = criterion(y_pred, y)
        
        # 모든 표본에 대한 경사 계산
        loss.backward()
        
        # 파라메터 갱신
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad
        
        # 경사 초기화
        model.zero_grad()
    
    return model
```

### 피처

**장점:**

- 로스의 **정확한 경사** 를 계산한다
- 경사 추출에 따른 잡음이 없다. 안정성은 러닝레이트와 곡률에 달려 있다
- 적절한 러닝레이트 등 조건을 만족하는 볼록 문제에서는 수렴 보장을 활용할 수 있다

**단점:**

- 큰 데이터셋에서는 계산 비용이 크다
- 한 번에 처리하면 메모리가 많이 든다. 나누어 경사를 누적할 수도 있다
- 얕은 국소 최솟값을 벗어날 수 없다
- 한 업데이트에 전체 데이터를 처리한다(에포크당 한 번)

### 쓰기 좋은 경우

- 작은 데이터셋($N < 1000$)
- 정확한 경사가 중요한 문제
- 볼록 최적화 문제
- 메모리가 제약이 되지 않을 때

---

## 3. 확률적 경사 하강법(SGD)

### 알고리즘

```python
def stochastic_gradient_descent(X, y, model, criterion,
                                 learning_rate, n_epochs):
    """
    확률 경사 하강법: 루프마다 표본 하나를 쓴다
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
            
            # 표본 하나에 대한 로스 계산
            loss = criterion(y_pred, y_sample)
            
            # 역전파
            loss.backward()
            
            # 파라메터 갱신
            with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param.grad
            
            model.zero_grad()
    
    return model
```

### 피처

**장점:**

- 반복 하나가 매우 빠르다
- 국소 최솟값을 벗어날 수 있다(잡음이 탐색을 돕는다)
- 온라인 학습이 가능하다(데이터가 도착하는 대로 처리)
- 메모리 사용량이 적다

**단점:**

- 경사 추정에 잡음이 매우 많다
- 수렴이 불규칙하다(분산이 크다)
- 정확한 최솟값에 수렴하지 않을 수 있다
- 배치 축의 병렬성을 활용하기 어렵다. 표본 내부 연산의 병렬성은 사용할 수 있다

### 잡음과 탐색의 절충

SGD의 표본 경사는 전체 학습 로스의 경사와 차이가 있다. 고정한 파라미터에서 표본을 균등하게 추출하면 이 차이의 기댓값은 0이다.

$$\nabla_\theta \ell(\theta; x_i, y_i) = \nabla_\theta L(\theta) + \epsilon_i$$

여기서 $\epsilon_i$는 표본 하나만 쓰는 데서 오는 "잡음"이다.

경사 잡음은 정류점 주변을 탐색하거나 해의 선택에 영향을 줄 수 있다. 그러나 탈출이나 더 나은 일반화를 항상 보장하지 않으며, 러닝레이트와 잡음의 방향·크기에 따라 수렴을 방해할 수도 있다.

이 효과는 표본 단위 SGD뿐 아니라 미니배치 경사에도 나타난다. 실무에서 “SGD”는 미니배치 방식까지 포함하는 이름으로 자주 사용한다.

---

## 4. 미니배치 경사 하강법

### 알고리즘

```python
def minibatch_gradient_descent(X, y, model, criterion,
                                learning_rate, n_epochs, batch_size):
    """
    작은 배치 경사 하강법: 둘의 좋은 점을 다 가진다
    """
    # 자동 배치 구성을 위한 DataLoader 생성
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(n_epochs):
        for X_batch, y_batch in dataloader:
            # 미니배치에 대한 순전파
            y_pred = model(X_batch)
            
            # 미니배치에 대한 로스 계산
            loss = criterion(y_pred, y_batch)
            
            # 역전파
            loss.backward()
            
            # 파라메터 갱신
            with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param.grad
            
            model.zero_grad()
    
    return model
```

### PyTorch DataLoader

```python
from torch.utils.data import TensorDataset, DataLoader

# 데이터셋 생성
dataset = TensorDataset(X, y)

# 미니배치를 쓰는 DataLoader 생성
train_loader = DataLoader(
    dataset,
    batch_size=64,        # Mini-batch size
    shuffle=True,         # Shuffle each epoch
    # num_workers 를 0으로 둔다. 1 이상이면 맥과 윈도가 일꾼 프로세스를
    # **새로 띄우며**(spawn) 이 각본을 처음부터 다시 읽는다. 맨 바깥에서 자료를
    # 돌리고 있으면 일꾼이 또 일꾼을 띄워 끝내 죽는다. 1 이상을 쓰려면 돌리는
    # 부분을 모두 `if __name__ == "__main__":` 안으로 넣어야 한다.
    num_workers=0,        # Parallel data loading
    pin_memory=True       # Faster GPU transfer
)

# 학습 루프
for epoch in range(n_epochs):
    for X_batch, y_batch in train_loader:
        # 미니배치에 대한 학습 단계
        ...
```

### 피처

**장점:**

- 경사의 정확도와 계산량 사이의 균형을 잡는다
- GPU 병렬성을 효율적으로 활용한다
- 적당한 잡음(탐색에 충분하되 지나치지 않다)
- 실제 문제에 가장 실용적이다

**단점:**

- batch_size라는 하이퍼파라미터가 생긴다
- 경사가 여전히 (정확한 값이 아닌) 추정값이다
- 배치 통계가 달라질 수 있다(BatchNorm에 영향을 준다)

---

## 5. 비교

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

표본 $N=10{,}000$개를 한 번씩 처리하고 마지막 짧은 배치도 포함한다고 하자. 미니배치는 $\lceil N/64\rceil=157$번 업데이트하며 마지막 배치에는 표본 16개가 있다. 아래 하드웨어 활용도는 전형적인 경향이며 작업에 따라 다르다.

| 지표 | 배치 GD | SGD | 미니배치($B=64$) |
|--------|----------|-----|-------------------|
| 갱신당 표본 수 | 10,000 | 1 | 64 |
| 에포크당 업데이트 횟수 | 1 | 10,000 | 157 |
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
# 시간: 약 0.3초, 최종 로스: 1.0012

# SGD(표본 1개)
# 시간: 약 45초, 최종 로스: 1.0089

# 미니배치(batch_size=64)
# 시간: 약 0.8초, 최종 로스: 1.0015
```

---

## 6. 배치 크기 선택

### 일반적인 지침

| 배치 크기 | 용도 |
|------------|----------|
| 1 | 순수 SGD(실무에서는 거의 쓰지 않는다) |
| 8-32 | 메모리가 제한적이고 일반화가 좋다 |
| 64-256 | 흔한 기본 범위 |
| 512-2048 | 대규모 학습. 러닝레이트 조정이 필요하다 |
| 4096+ | 분산 학습(세심한 조정 필요) |

### 선택에 영향을 주는 요인

1. **GPU 메모리**: 배치가 클수록 메모리가 더 필요하다
2. **일반화**: 배치가 작을수록 일반화가 잘되는 경우가 많다
3. **수렴 속도**: 배치가 크면 더 큰 러닝레이트를 쓸 수 있다
4. **하드웨어 효율**: 배치 크기와 텐서 크기에 따라 커널 효율이 달라지므로 실제 처리량을 측정한다

### 배치 크기 후보

```python
# 좋은 예: 하드웨어 효율을 위한 2의 거듭제곱
batch_sizes = [16, 32, 64, 128, 256, 512]

# 피할 것: 임의의 크기
# batch_size = 100  # 덜 효율적이다
```

---

## 7. 경사 분산 분석

### 분산 감소

고정된 파라미터에서 독립적인 복원 추출을 가정하면 미니배치 평균 경사의 성분별 분산(또는 공분산 행렬)은 다음과 같다.

$$\text{Var}[\nabla_\theta L_B] = \frac{\sigma^2}{B}$$

여기서 $\sigma^2$은 표본 하나에 대한 경사의 분산이다.

**함의:**

- 복원 추출에서는 배치 크기를 두 배로 하면 분산이 절반이 된다.
- 유한한 데이터셋에서 비복원 추출하면 분산에 $(N-B)/(N-1)$의 보정 인자가 붙는다. $B=N$이면 전체 경사여서 추출 분산이 0이다.
- $1/B$에 따른 감소는 큰 배치로 갈수록 추가 표본당 감소량이 작아지는 관계를 나타낸다.

### 잡음과 일반화

연구에 따르면 작은 배치에서 오는 경사 잡음은 다음으로 이어질 수 있다.

- 더 평평한 최솟값(더 나은 일반화)
- 암묵적 정규화
- 뾰족한 국소 최솟값에서의 탈출

Keskar 등의 연구는 특정 실험에서 배치 크기, 해의 날카로움, 일반화 사이의 관계를 분석한다. 이를 모든 모델에서 작은 배치가 우월하다는 보장으로 해석해서는 안 된다.

---

## 8. 구현 세부 사항

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

---

## 9. 실용적인 학습 루프

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

**출력:**

```
Epoch 10: Loss = 0.9586
Epoch 20: Loss = 0.9700
Epoch 30: Loss = 0.9207
Epoch 40: Loss = 0.9245
Epoch 50: Loss = 0.8965
Epoch 60: Loss = 0.8759
Epoch 70: Loss = 0.8802
Epoch 80: Loss = 0.8812
Epoch 90: Loss = 0.8559
Epoch 100: Loss = 0.8418
```

---

## 10. 다른 주제와의 연결

- **러닝레이트**: 배치 크기와 상호작용한다. [러닝레이트](learning_rate.md) 참고
- **모멘텀**: 실효 잡음을 줄인다. [고전적 모멘텀](../../ch09/optimizers/momentum.md) 참고
- **배치 정규화**: 통계가 배치 크기에 의존한다. 배치 정규화 참고
- **분산 학습**: 배치 크기를 세심하게 조정해야 한다

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
표본이 $N = 1000$개인 데이터셋에 대해 배치 경사 하강법($B = N$), 미니배치 경사 하강법($B = 32$), SGD($B = 1$)의 에포크당 경사 계산 횟수를 비교하라. 어느 쪽이 에포크당 파라메터 갱신을 더 많이 하는가?

</div>

??? success "연습문제 1 풀이"
    배치 경사 하강법: 1000개 표본 전체로 경사 계산 1회, 파라메터 갱신 1회.

    미니배치 경사 하강법($B=32$): $\lceil 1000/32 \rceil = 32$회 경사 계산, 32회 파라메터 갱신.

    SGD($B=1$): 1000회 경사 계산, 1000회 파라메터 갱신.

    SGD가 에포크당 가장 많이 갱신하지만(1000회) 각 갱신이 잡음이 매우 많은 경사 추정을 쓴다. 미니배치 경사 하강법이 균형을 잡는다. 에포크당 32회 갱신하며 앞의 31개 배치는 표본 32개, 마지막 배치는 표본 8개의 경사를 사용한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
미니배치 경사의 기댓값이 전체 배치 경사와 같음을 증명하라. 그런 다음 배치 크기를 $B$라 할 때 분산이 $O(1/B)$로 감소함을 보여라.

</div>

??? success "연습문제 2 풀이"
    표본 $i$에 대해 $g_i = \nabla_\theta \ell(x_i, \theta)$라 하자. 미니배치 경사는 $i_j$가 균등하게 추출될 때 $\hat{g} = \frac{1}{B}\sum_{j=1}^B g_{i_j}$이다.

    $\mathbb{E}[\hat{g}] = \frac{1}{B}\sum_{j=1}^B \mathbb{E}[g_{i_j}] = \frac{1}{B} \cdot B \cdot \frac{1}{N}\sum_{i=1}^N g_i = \bar{g}$이며 이것이 전체 배치 경사이다.

    분산은 (복원 추출을 가정하면) $\sigma^2 = \text{Var}(g_i)$일 때 $\text{Var}(\hat{g}) = \frac{1}{B^2}\sum_{j=1}^B \text{Var}(g_{i_j}) = \frac{\sigma^2}{B}$이다. 이로써 $O(1/B)$의 분산 감소가 확인된다. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
물리적 배치 크기 32로 실효 배치 크기 256을 흉내 내는 경사 누적을 구현하라. 파라메터 갱신이 일치함을 확인하라.

</div>

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
    크기가 같은 마이크로배치 8개를 같은 파라미터에서 계산하고 각 평균 로스를 8로 나누면 전체 평균 경사를 얻는다. BatchNorm이나 확률적 연산이 있으면 큰 배치 실행과 차이가 생길 수 있다. 마지막 누적 구간이 짧으면 남은 경사 처리와 스케일도 조정해야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
배치 경사 하강법이 갇히는 뾰족한 국소 최솟값을 미니배치 SGD가 벗어날 수 있는 이유를 설명하라. 답을 경사 추정의 잡음과 연결하라.

</div>

??? success "연습문제 4 풀이"
    고정한 파라미터에서 미니배치 경사는 $\hat g=\bar g+\epsilon$으로 표현할 수 있다. 전체 경사가 0이어도 개별 배치 경사가 0이 아니면 업데이트가 발생해 주변으로 이동할 수 있다. 다만 잡음의 방향과 크기, 러닝레이트, 로스 장벽에 따라 탈출 여부가 달라진다. 평평한 해가 항상 일반화에 더 좋다거나, 전체 배치 방법이 모든 뾰족한 최솟값에 갇힌다고 단정할 수는 없다.

## 정리하며

1. **배치 경사 하강법**: 정확한 경사, 느림, 안정적인 수렴
2. **SGD**: 빠른 반복, 잡음이 많음, 좋은 탐색
3. **미니배치**: 최선의 절충이며 실무의 표준
4. **배치 크기가 중요하다**: 속도, 메모리, 일반화에 영향을 준다
5. **DataLoader를 쓴다**: 배치 구성, 섞기, 병렬화를 처리해 준다
6. **32-64로 시작한다**: 대부분의 문제에 합리적인 기본값
7. **경사 잡음의 효과를 확인한다**: 탐색과 일반화에 영향을 주지만 이점은 문제에 따라 다르다

**참고 문헌**

- Bottou, L. (2010). Large-scale machine learning with stochastic gradient descent. COMPSTAT.
- Keskar, N. S., et al. (2017). On large-batch training for deep learning: Generalization gap and sharp minima. ICLR.
- Smith, S. L., et al. (2018). Don't decay the learning rate, increase the batch size. ICLR.
- Goyal, P., et al. (2017). Accurate, large minibatch SGD: Training ImageNet in 1 hour.
