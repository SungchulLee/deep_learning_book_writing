# 다중 클래스 분류
## 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

- 베르누이 분포의 일반화로서 범주형 분포 이해하기
- 다중 클래스 분류의 확률질량함수 유도하기
- 원-핫 부호화를 구현하고 그 수학적 성질 이해하기
- 범주형 분포를 소프트맥스 회귀의 틀과 연결하기
- 이 개념들을 PyTorch에서 다중 클래스 분류 과제에 적용하기

---

## 이진 분류에서 다중 클래스 분류로

### 베르누이 분포 다시 보기

이진 분류에서는 베르누이 분포로 결과를 모형화한다. 확률변수 $Y \in \{0, 1\}$에 대해 다음과 같다.

$$P(Y = y) = p^y (1-p)^{1-y}$$

여기서 $p = P(Y=1)$은 양성 클래스의 확률이다. 이 간결한 표기는 두 경우를 우아하게 아우른다.

- $y = 1$일 때: $P(Y=1) = p^1 (1-p)^0 = p$
- $y = 0$일 때: $P(Y=0) = p^0 (1-p)^1 = 1-p$

### 여러 클래스로 일반화하기

클래스가 $K$개인 다중 클래스 분류에는 **범주형 분포**(**멀티눌리** 또는 **일반화 베르누이** 분포라고도 한다)가 필요하다. $Y \in \{1, 2, \ldots, K\}$을 클래스 이름표라 하고, $\boldsymbol{\pi} = (\pi_1, \pi_2, \ldots, \pi_K)$을 다음을 만족하는 확률 벡터라 하자.

$$\pi_k = P(Y = k), \quad \sum_{k=1}^{K} \pi_k = 1, \quad \pi_k \geq 0$$

확률질량함수(PMF)는 다음과 같다.

$$P(Y = k) = \pi_k$$

---

## 원-핫 부호화: 수학적 토대

### 정의와 동기

**원-핫 부호화**는 범주형 이름표 $y \in \{1, 2, \ldots, K\}$을, 정확히 한 원소만 1이고 나머지는 모두 0인 이진 벡터 $\mathbf{y} \in \{0, 1\}^K$으로 바꾼다.

$$\mathbf{y} = \mathbf{e}_k = (0, \ldots, 0, \underbrace{1}_{k\text{-th position}}, 0, \ldots, 0)^T$$

**예.** 클래스가 $K = 3$개일 때(예: 고양이, 개, 새) 다음과 같다.

- 클래스 1 (고양이): $\mathbf{y} = (1, 0, 0)^T$
- 클래스 2 (개): $\mathbf{y} = (0, 1, 0)^T$
- 클래스 3 (새): $\mathbf{y} = (0, 0, 1)^T$

### 수학적 성질

원-핫 벡터는 여러 중요한 성질을 만족한다.

**성질 1 (합이 1).**

$$\sum_{k=1}^{K} y_k = 1$$

**성질 2 (상호 배타성).**

$$y_i \cdot y_j = 0 \quad \forall\, i \neq j$$

**성질 3 (지시 함수).**

$$y_k = \mathbb{1}[Y = k] = \begin{cases} 1 & \text{if } Y = k \\ 0 & \text{otherwise} \end{cases}$$

**성질 4 (확률 벡터와의 내적).**

$$\mathbf{y}^T \boldsymbol{\pi} = \pi_k \quad \text{when } Y = k$$

마지막 성질이 특히 중요하다. 단순한 내적 하나로 참 클래스의 확률을 꺼낼 수 있게 해 주기 때문이다.

---

## 원-핫 부호화로 나타낸 범주형 분포

### 간결한 PMF 표현

원-핫 부호화를 쓰면 범주형 PMF를 우아한 곱의 형태로 쓸 수 있다.

$$P(\mathbf{y} \mid \boldsymbol{\pi}) = \prod_{k=1}^{K} \pi_k^{y_k}$$

이는 베르누이의 정식화를 그대로 따르면서 $K$개 클래스로 자연스럽게 확장된다. 잘 작동하는지 확인해 보자.

**확인.** 참 클래스가 $c$이면($y_c = 1$이고 $k \neq c$에 대해 $y_k = 0$) 다음과 같다.

$$P(\mathbf{y} \mid \boldsymbol{\pi}) = \pi_1^0 \cdot \pi_2^0 \cdots \pi_c^1 \cdots \pi_K^0 = \pi_c \checkmark$$

### 로그가능도

PMF에 로그를 취하면 다음과 같다.

$$\log P(\mathbf{y} \mid \boldsymbol{\pi}) = \sum_{k=1}^{K} y_k \log \pi_k$$

독립인 표본 $N$개로 이루어진 데이터셋 $\{(\mathbf{x}^{(i)}, \mathbf{y}^{(i)})\}_{i=1}^{N}$에 대해 로그가능도는 다음이 된다.

$$\mathcal{L}(\boldsymbol{\theta}) = \sum_{i=1}^{N} \sum_{k=1}^{K} y_k^{(i)} \log \pi_k^{(i)}$$

여기서 $\pi_k^{(i)} = P(Y = k \mid \mathbf{x}^{(i)};\, \boldsymbol{\theta})$은 모델 매개변수 $\boldsymbol{\theta}$을 통해 입력에 의존한다.

---

## 소프트맥스 회귀와의 관계

### 모델의 구조

소프트맥스 회귀에서는 클래스 확률을 다음과 같이 모형화한다.

$$\pi_k = P(Y = k \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_k^T \mathbf{x} + b_k)}{\sum_{j=1}^{K} \exp(\mathbf{w}_j^T \mathbf{x} + b_j)}$$

이는 **로짓** $z_k = \mathbf{w}_k^T \mathbf{x} + b_k$에 적용한 **소프트맥스 함수**이다.

### 왜 이런 매개화인가?

소프트맥스 함수는 다음을 보장한다.

1. **음이 아님:** 모든 $k$에 대해 $\pi_k > 0$이다 (지수함수는 언제나 양수이다)
2. **정규화:** $\sum_{k=1}^{K} \pi_k = 1$이다 (구성상 그렇다)
3. **단조성:** 로짓이 클수록 확률이 크다
4. **미분 가능성:** 최적화에 쓸 매끄러운 경사를 준다

### 소프트맥스 회귀의 파이프라인

소프트맥스 회귀의 전체 계산 그래프는 다음과 같이 요약된다.

$$\mathbf{x} \;\xrightarrow{\;\mathbf{W},\, \mathbf{b}\;}\; \mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b} \;\xrightarrow{\;\text{softmax}\;}\; \hat{\boldsymbol{\pi}} \;\xrightarrow{\;-\log(\cdot)\;}\; \mathcal{L}$$

여기서 각 기호는 다음과 같다.

- $\mathbf{W} \in \mathbb{R}^{K \times D}$은 가중치 행렬이다
- $\mathbf{b} \in \mathbb{R}^K$은 편향 벡터이다
- $\mathbf{z} \in \mathbb{R}^K$은 로짓(날것의 점수)이다
- $\hat{\boldsymbol{\pi}} \in \Delta^{K-1}$은 확률 단체 위의 예측 확률이다
- $\mathcal{L}$은 교차 엔트로피 손실이다

---

## PyTorch 구현

### 원-핫 부호 만들기

```python
import torch
import torch.nn.functional as F

# 방법 1: torch.nn.functional.one_hot 사용하기
def create_one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    클래스 번호를 원핫 벡터로 바꾼다.

    Args:
        labels: 클래스 번호 [0, num_classes-1]을 담은 모양 (batch_size,)인 텐서
        num_classes: 클래스의 전체 개수 K

    Returns:
        모양이 (batch_size, num_classes)인 원핫 텐서
    """
    return F.one_hot(labels, num_classes=num_classes).float()

# 사용 예
labels = torch.tensor([0, 2, 1, 0])  # 4 samples with classes 0, 2, 1, 0
one_hot = create_one_hot(labels, num_classes=3)
print("Labels:", labels)
print("One-hot encoding:")
print(one_hot)
```

출력:
```
Labels: tensor([0, 2, 1, 0])
One-hot encoding:
tensor([[1., 0., 0.],
        [0., 0., 1.],
        [0., 1., 0.],
        [1., 0., 0.]])
```

### 원-핫을 직접 구현하기

```python
def one_hot_manual(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    이해를 돕고자 직접 구현한 원핫 매김.

    영 텐서를 만들고 scatter_으로 알맞은 자리에 1을
    놓는다.
    """
    batch_size = labels.size(0)
    one_hot = torch.zeros(batch_size, num_classes, device=labels.device)

    # scatter_(dim, index, src)은 src의 값을 index가 지정하는 위치에 넣는다.
    # 여기서는 labels가 주는 위치의 1번 차원을 따라
    # 1을 흩뿌린다.
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

    return one_hot

# 동등함을 확인한다
labels = torch.tensor([0, 2, 1])
assert torch.allclose(one_hot_manual(labels, 3), F.one_hot(labels, 3).float())
print("Manual implementation matches F.one_hot!")
```

### 참 클래스의 확률 꺼내기

```python
def get_true_class_probs(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    참 클래스의 예측 확률을 뽑아낸다.

    이는 (one_hot * probs).sum(dim=1)과 같다.
    다만 gather를 쓰면 더 효율적이다.

    Args:
        probs: 모양이 (batch_size, num_classes)인 예측 확률
        labels: 모양이 (batch_size,)인 참 클래스 번호

    Returns:
        모양이 (batch_size,)인 참 클래스의 확률
    """
    # gather(dim, index)은 인덱스를 써서 dim을 따라 원소를 고른다
    return probs.gather(1, labels.unsqueeze(1)).squeeze(1)

# 예
probs = torch.tensor([
    [0.7, 0.2, 0.1],  # Sample 0: class 0 has prob 0.7
    [0.1, 0.3, 0.6],  # Sample 1: class 2 has prob 0.6
    [0.2, 0.5, 0.3],  # Sample 2: class 1 has prob 0.5
])
labels = torch.tensor([0, 2, 1])  # True classes

true_probs = get_true_class_probs(probs, labels)
print(f"True class probabilities: {true_probs}")  # tensor([0.7, 0.6, 0.5])
```

---

## PyTorch의 CrossEntropyLoss에서 범주형 대 원-핫

### 결정적으로 중요한 구현 세부 사항

PyTorch의 `nn.CrossEntropyLoss`은 원-핫 벡터가 아니라 **클래스 인덱스**를 받는다.

```python
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

# 모델이 낸 로짓 (확률이 아니다!)
logits = torch.tensor([
    [2.0, 1.0, 0.5],
    [0.5, 2.5, 1.0],
])

# 옳은 방법: 클래스 인덱스를 쓴다
labels_indices = torch.tensor([0, 1])
loss_correct = criterion(logits, labels_indices)
print(f"Loss with indices: {loss_correct.item():.4f}")

# 틀린 방법: 표준 CrossEntropyLoss에 원-핫 부호를 쓴다
labels_onehot = F.one_hot(labels_indices, num_classes=3).float()
# loss_wrong = criterion(logits, labels_onehot)  # 이러면 오류가 난다!
```

### 원-핫 부호화가 필요할 때

다음 경우에는 원-핫 부호화를 명시적으로 쓴다.

1. **이름표 평활화:** 딱딱한 목표 대신 부드러운 목표를 쓸 때
2. **지식 증류:** 교사 모델의 확률 분포를 맞출 때
3. **사용자 정의 손실 함수:** 명시적인 확률 목표가 필요할 때
4. **시각화:** 모델의 예측을 살펴볼 때

```python
class LabelSmoothingLoss(nn.Module):
    """
    레이블 스무딩을 곁들인 교차 엔트로피 손실.
    속으로는 원핫으로 매긴 과녁이 있어야 한다.
    """
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # 평활화된 원-핫 목표를 만든다
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
            smooth_targets.scatter_(1, labels.unsqueeze(1), self.confidence)

        # 부드러운 목표로 교차 엔트로피를 계산한다
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(smooth_targets * log_probs).sum(dim=1).mean()

        return loss
```

---

## 완전한 예제: 범주형 분포에서 표본 뽑기

```python
import torch

def sample_categorical(probs: torch.Tensor, num_samples: int = 1000) -> torch.Tensor:
    """
    범주 분포에서 표본을 뽑는다.

    Args:
        probs: 모양이 (num_classes,)인 확률 벡터
        num_samples: 뽑을 표본의 수

    Returns:
        뽑은 클래스 번호를 담은 텐서
    """
    # torch.multinomial은 범주형 분포에서 표본을 뽑는다
    return torch.multinomial(probs, num_samples, replacement=True)

# 범주형 분포를 정의한다
class_probs = torch.tensor([0.5, 0.3, 0.2])  # Cat, Dog, Bird
class_names = ['Cat', 'Dog', 'Bird']

# 분포에서 표본을 뽑는다
samples = sample_categorical(class_probs, num_samples=10000)

# 나온 횟수를 센다
counts = torch.bincount(samples, minlength=3).float()
empirical_probs = counts / counts.sum()

print("True probabilities:", class_probs.numpy())
print("Empirical probabilities:", empirical_probs.numpy())
print("Class counts:", counts.numpy().astype(int))
```

---

## 요약

### 핵심 개념

| 개념 | 공식 / 설명 |
|---------|----------------------|
| 범주형 PMF | $P(Y=k) = \pi_k$ |
| 원-핫 부호화 | $y_k = \mathbb{1}[Y=k]$ |
| 곱 형태의 PMF | $P(\mathbf{y}\mid\boldsymbol{\pi}) = \prod_k \pi_k^{y_k}$ |
| 로그가능도 | $\log P(\mathbf{y}\mid\boldsymbol{\pi}) = \sum_k y_k \log \pi_k$ |
| 확률 꺼내기 | $\mathbf{y}^T \boldsymbol{\pi} = \pi_{\text{true class}}$ |

### PyTorch 함수

| 할 일 | PyTorch 함수 |
|------|------------------|
| 원-핫 만들기 | `F.one_hot(labels, num_classes)` |
| 확률 꺼내기 | `probs.gather(1, labels.unsqueeze(1))` |
| 범주형에서 표본 뽑기 | `torch.multinomial(probs, n)` |
| 교차 엔트로피 손실 | `nn.CrossEntropyLoss()` (인덱스를 받는다!) |

### 흔히 빠지는 함정

!!! warning "반드시 피해야 할 실수"

    1. **`nn.CrossEntropyLoss`에 원-핫 벡터를 넘기지 마라** — 클래스 인덱스를 받는다
    2. **`nn.CrossEntropyLoss` 앞에서 소프트맥스를 적용하지 마라** — 내부에서 적용된다
    3. **차원의 순서에 유의하라** — PyTorch는 `(classes, batch)`가 아니라 `(batch, classes)`를 쓴다

---

## 참고 문헌

1. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 4.
2. Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*, Chapter 3.
3. PyTorch Documentation: [torch.nn.functional.one_hot](https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html)

## 연습문제

**연습문제 1.**
클래스 확률이 로그 선형인 범주형 분포라는 가정에서 소프트맥스 함수를 유도하라.

??? success "연습문제 1 풀이"
    어떤 정규화 상수에 대해 $\log p(y=k|\mathbf{x}) = \mathbf{w}_k^\top\mathbf{x} + c$이라 가정하자. 그러면 $p(y=k|\mathbf{x}) = \frac{e^{\mathbf{w}_k^\top\mathbf{x}}}{\sum_{j=1}^K e^{\mathbf{w}_j^\top\mathbf{x}}}$이며, 이것이 소프트맥스 함수이다. 이는 자연 매개변수가 $\eta_k = \mathbf{w}_k^\top\mathbf{x}$인 범주형 분포의 지수족 형태에서 따라 나온다.

---

**연습문제 2.**
소프트맥스가 상수를 더해도 변하지 않음을 보여라. 즉 $\text{softmax}(\mathbf{z} + c\mathbf{1}) = \text{softmax}(\mathbf{z})$임을 보여라.

??? success "연습문제 2 풀이"
    $$
    \text{softmax}(\mathbf{z}+c)_k = \frac{e^{z_k+c}}{\sum_j e^{z_j+c}} = \frac{e^c e^{z_k}}{e^c \sum_j e^{z_j}} = \frac{e^{z_k}}{\sum_j e^{z_j}} = \text{softmax}(\mathbf{z})_k
    $$

    이 성질은 수치적 안정성에 활용된다. 소프트맥스를 계산하기 전에 $\max_j z_j$을 빼면 된다. $\square$

---

**연습문제 3.**
소프트맥스 함수의 야코비 행렬 $\frac{\partial p_i}{\partial z_j}$을 계산하라.

??? success "연습문제 3 풀이"
    $$
    \frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)
    $$

    여기서 $\delta_{ij}$은 크로네커 델타이다. 행렬 형태로는 $J = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$이다.

    유도: $i = j$일 때 $\frac{\partial}{\partial z_j}\frac{e^{z_j}}{S} = \frac{e^{z_j}S - e^{z_j}e^{z_j}}{S^2} = p_j - p_j^2 = p_j(1 - p_j)$이다. $i \neq j$일 때 $\frac{\partial}{\partial z_j}\frac{e^{z_i}}{S} = -\frac{e^{z_i}e^{z_j}}{S^2} = -p_i p_j$이다.

---

**연습문제 4.**
온도로 배율을 조절한 소프트맥스 $\text{softmax}(\mathbf{z}/T)$을 구현하고, $T \in \{0.1, 1.0, 5.0\}$에 대해 출력 분포에 미치는 효과를 보여라.

??? success "연습문제 4 풀이"
    ```python
    import torch
    z = torch.tensor([2.0, 1.0, 0.1])
    for T in [0.1, 1.0, 5.0]:
        p = torch.softmax(z / T, dim=0)
        print(f"T={T}: {p}")
    # T=0.1: 거의 원-핫 [1.0, 0.0, 0.0]
    # T=1.0: 표준 소프트맥스 [0.66, 0.24, 0.10]
    # T=5.0: 거의 균등 [0.38, 0.34, 0.28]
    ```
    온도가 낮으면 분포가 날카로워지고(확신), 온도가 높으면 평평해진다(불확실).
