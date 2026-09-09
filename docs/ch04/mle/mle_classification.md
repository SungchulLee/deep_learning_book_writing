# 분류에서의 MLE

분류의 손실 함수는 이산 확률분포의 음의 로그가능도이다. **이진 교차 엔트로피는 베르누이 모델의 NLL**이고 **범주형 교차 엔트로피는 범주형 모델의 NLL**이다. 이 절은 이 관계를 정확히 밝히고, 소프트맥스와 시그모이드 함수가 MLE의 틀에서 어떻게 자연스럽게 나오는지 보여준다.

!!! success "핵심 통찰"

    $$\text{Cross-Entropy Loss} = \text{Negative Log-Likelihood of Categorical Distribution}$$
    
    교차 엔트로피 손실로 분류기를 학습시킬 때 당신은 범주형 가능도 아래에서 최대가능도 추정을 하고 있는 것이다.

---

## 1. 이진 분류: 베르누이 MLE에서 나오는 BCE

### 확률 모델

이진 분류에서는 이름표를 베르누이 확률변수로 모형화한다.

$$
y | x \sim \text{Bernoulli}(\sigma(f_\theta(x)))
$$

여기서 $f_\theta(x)$은 모델의 로짓 출력이고 $\sigma(z) = 1/(1+e^{-z})$은 로짓을 확률로 보내는 시그모이드 함수이다.

### 유도

$\hat{p} = \sigma(f_\theta(x))$일 때 관측 하나의 **가능도**는 다음과 같다.

$$
p(y | x, \theta) = \hat{p}^{\,y} \cdot (1 - \hat{p})^{1-y}
$$

**음의 로그가능도**는 다음과 같다.

$$
-\log p(y | x, \theta) = -y \log \hat{p} - (1-y) \log(1 - \hat{p})
$$

데이터셋에 걸쳐 평균하면 **이진 교차 엔트로피(BCE)** 손실을 얻는다.

$$
\boxed{\mathcal{L}_{\text{BCE}} = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\right]}
$$

### 왜 시그모이드인가?

시그모이드 함수는 로그 승산(로짓) 매개화에서 자연스럽게 나온다. 로그 승산을 선형 함수로 모형화하면 다음과 같다.

$$
\log \frac{p}{1-p} = f_\theta(x) \implies p = \sigma(f_\theta(x))
$$

이는 실숫값 $f_\theta(x)$이 무엇이든 $p \in (0, 1)$임을 보장한다. 로지스틱 회귀가 시그모이드를 쓰는 이유가 여기 있다. 시그모이드는 베르누이 분포의 정준 연결 함수이다.

### 수치적 안정성

$\sigma(z)$을 계산한 뒤 $\log(\sigma(z))$을 취해 BCE를 구하는 것은 수치적으로 불안정하다. **log-sum-exp** 기법이 안정한 정식화를 준다.

$$
\text{BCE}(y, z) = \max(z, 0) - yz + \log(1 + e^{-|z|})
$$

여기서 $z = f_\theta(x)$은 로짓이다. 이것이 바로 `nn.BCEWithLogitsLoss`이 구현하는 것이다.

---

## 2. 다중 클래스 분류: 범주형 MLE에서 나오는 교차 엔트로피

### 확률 모델

$K$개 클래스 분류에서는 다음과 같다.

$$
y | x \sim \text{Categorical}(\text{softmax}(f_\theta(x)))
$$

**소프트맥스 함수**는 다음과 같다.

$$
\text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}, \quad k = 1, \ldots, K
$$

### 유도

정답 클래스에서 $y_k = 1$인 원-핫 이름표 $\mathbf{y} = (y_1, \ldots, y_K)$에 대해 다음이 성립한다.

$$
p(\mathbf{y} | x, \theta) = \prod_{k=1}^{K} \hat{p}_k^{y_k}
$$

**음의 로그가능도**는 다음과 같다.

$$
-\log p(\mathbf{y} | x, \theta) = -\sum_{k=1}^{K} y_k \log \hat{p}_k
$$

$c_i$이 정답 클래스인 딱딱한 이름표에서는 이것이 $-\log \hat{p}_{c_i}$으로 간단해진다. **범주형 교차 엔트로피** 손실은 다음과 같다.

$$
\boxed{\mathcal{L}_{\text{CE}} = -\frac{1}{n}\sum_{i=1}^{n} \log \hat{p}_{i, c_i}}
$$

### 왜 소프트맥스인가?

시그모이드가 베르누이의 정준 연결이듯 소프트맥스는 범주형 분포의 정준 연결이다. 이는 다항 로짓 모델에서 나온다.

$$
\log \frac{p_k}{p_K} = z_k \quad (k = 1, \ldots, K-1)
$$

제약 $\sum p_k = 1$ 아래에서 $p_k$에 대해 풀면 정확히 소프트맥스 함수를 얻는다.

### 수치적 안정성

$\exp(z_k) / \sum \exp(z_j)$을 곧바로 계산하면 로짓이 클 때 넘침이 생긴다. **로그 소프트맥스** 기법은 최댓값을 빼 준다.

$$
\log \text{softmax}(z)_k = z_k - \log\sum_{j=1}^{K} e^{z_j} = z_k - \left(m + \log\sum_{j=1}^{K} e^{z_j - m}\right)
$$

여기서 $m = \max_j z_j$이다. PyTorch의 `nn.CrossEntropyLoss`은 로그 소프트맥스와 NLL 손실을 수치적으로 안정한 하나의 연산으로 합친다.

---

## 3. 이름표 평활화

### 동기

딱딱한 원-핫 이름표는 모델이 극단적인 로짓을 내도록(소프트맥스 출력을 0이나 1 쪽으로 밀도록) 부추긴다. **이름표 평활화**는 딱딱한 이름표를 혼합으로 대체한다.

$$
y_k^{\text{smooth}} = (1 - \epsilon) \cdot y_k + \frac{\epsilon}{K}
$$

여기서 $\epsilon$은 평활화 매개변수이다(보통 0.1).

### MLE의 관점에서의 해석

이름표 평활화는 혼합 분포 아래의 MLE와 같다. 확률 $(1 - \epsilon)$으로 이름표가 정확하고, 확률 $\epsilon$으로 이름표가 모든 클래스에서 균등하게 뽑힌다. 이는 모델이 덜 확신하는 예측을 하도록 정칙화한다.

---

## 4. 다중 이름표 분류

각 표본이 여러 클래스에 독립적으로 속할 수 있으면 모델은 독립인 베르누이 분포들의 곱이 된다.

$$
p(\mathbf{y} | x, \theta) = \prod_{k=1}^{K} \hat{p}_k^{y_k}(1 - \hat{p}_k)^{1-y_k}
$$

손실은 독립인 BCE 손실 $K$개의 합이다.

$$
\mathcal{L}_{\text{multi-label}} = -\frac{1}{nK}\sum_{i=1}^{n}\sum_{k=1}^{K}\left[y_{ik}\log\hat{p}_{ik} + (1-y_{ik})\log(1-\hat{p}_{ik})\right]
$$

확률들이 서로 독립이고 합이 1일 필요가 없으므로 각 출력은 (소프트맥스가 아니라) 시그모이드를 쓴다.

---

## 5. PyTorch 구현

### 이진 교차 엔트로피 확인하기

```python
import torch
import torch.nn as nn
import numpy as np

def bce_from_nll():
    """직접 계산한 BCE가 PyTorch 구현과 일치하는지 확인한다."""
    torch.manual_seed(42)
    
    n = 100
    x = torch.randn(n, 2)
    true_w = torch.tensor([2.0, -1.0])
    y = (x @ true_w > 0).float()
    
    # 모델의 예측
    w = torch.randn(2, requires_grad=True)
    logits = x @ w
    probs = torch.sigmoid(logits)
    
    # 직접 계산한 BCE
    eps = 1e-7
    manual_bce = -torch.mean(y * torch.log(probs + eps) + 
                              (1 - y) * torch.log(1 - probs + eps))
    
    # PyTorch의 BCE (확률로부터)
    pytorch_bce = nn.BCELoss()(probs, y)
    
    # 로짓을 쓰는 PyTorch의 BCE (수치적으로 안정)
    pytorch_bce_logits = nn.BCEWithLogitsLoss()(logits, y)
    
    print("Binary Cross-Entropy Verification")
    print(f"Manual BCE:       {manual_bce.item():.6f}")
    print(f"PyTorch BCE:      {pytorch_bce.item():.6f}")
    print(f"BCE with Logits:  {pytorch_bce_logits.item():.6f}")
```

### 범주형 교차 엔트로피 확인하기

```python
def cross_entropy_from_nll():
    """직접 계산한 교차 엔트로피가 PyTorch와 일치하는지 확인한다."""
    torch.manual_seed(42)
    
    n = 100
    K = 5
    x = torch.randn(n, 10)
    y = torch.randint(0, K, (n,))
    
    # 모델 (단순한 선형 모델)
    W = torch.randn(10, K, requires_grad=True)
    logits = x @ W  # Shape: (n, K)
    
    # 직접 계산한 소프트맥스와 교차 엔트로피
    exp_logits = torch.exp(logits - logits.max(dim=1, keepdim=True)[0])
    probs = exp_logits / exp_logits.sum(dim=1, keepdim=True)
    manual_ce = -torch.mean(torch.log(probs[range(n), y] + 1e-7))
    
    # PyTorch의 CrossEntropyLoss (LogSoftmax와 NLLLoss를 합친 것)
    pytorch_ce = nn.CrossEntropyLoss()(logits, y)
    
    print("Cross-Entropy Verification")
    print(f"Manual CE:  {manual_ce.item():.6f}")
    print(f"PyTorch CE: {pytorch_ce.item():.6f}")
```

### 완전한 분류 학습

```python
def train_classification_mle_perspective():
    """분류 학습의 MLE 해석을 보여주는 완전한 예제."""
    torch.manual_seed(42)
    
    # 3개 클래스 데이터를 생성한다
    n_per_class = 100
    X0 = torch.randn(n_per_class, 2) + torch.tensor([-2.0, 0.0])
    X1 = torch.randn(n_per_class, 2) + torch.tensor([2.0, 0.0])
    X2 = torch.randn(n_per_class, 2) + torch.tensor([0.0, 3.0])
    
    X = torch.cat([X0, X1, X2])
    y = torch.cat([torch.zeros(n_per_class), 
                   torch.ones(n_per_class), 
                   2*torch.ones(n_per_class)]).long()
    
    # 뒤섞는다
    perm = torch.randperm(len(X))
    X, y = X[perm], y[perm]
    
    # 모델
    model = nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 3)  # Output: logits for 3 classes
    )
    
    # 교차 엔트로피 = 범주형 NLL
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("Training Classifier (MLE with Categorical likelihood)")
    print("-" * 50)
    
    for epoch in range(300):
        logits = model(X)
        loss = criterion(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 60 == 0:
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                accuracy = (pred == y).float().mean().item()
            
            nll_bits = loss.item() / np.log(2)
            print(f"Epoch {epoch+1}: NLL = {loss.item():.4f} nats "
                  f"({nll_bits:.4f} bits), Accuracy = {accuracy:.2%}")
```

---

## 연습문제

**연습문제 1.**
초점 손실 $\text{FL}(p_t) = -\alpha_t(1-p_t)^\gamma \log(p_t)$을 유도하고, $\gamma = 0$일 때 표준 교차 엔트로피로 환원됨을 보여라.

??? success "연습문제 1 풀이"
    정답 클래스 확률 $p_t$에 대한 표준 교차 엔트로피는 $\text{CE}(p_t) = -\log(p_t)$이다. 초점 손실은 여기에 조절 인수를 더한다.

    $$
    \text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
    $$

    $\gamma = 0$이면 $(1-p_t)^0 = 1$이므로 $\text{FL} = -\alpha_t \log(p_t)$이 되어 가중 교차 엔트로피가 된다. $\alpha_t = 1$이면 정확히 표준 교차 엔트로피이다.

    인수 $(1-p_t)^\gamma$은 잘 분류된 예($p_t$이 1에 가까운 경우)의 비중을 낮추어, 손실이 어렵고 잘못 분류된 예에 집중하게 만든다.

---

**연습문제 2.**
10개 클래스 문제에서 매개변수 $\epsilon = 0.1$인 이름표 평활화를 구현하고, 신뢰도 그림으로 딱딱한 이름표와 보정 정도를 비교하라.

??? success "연습문제 2 풀이"
    이름표 평활화를 쓰면 목표 분포가 $y_k' = (1-\epsilon)y_k + \epsilon/K$이 된다. 여기서 $K$은 클래스의 개수이다.

    ```python
    import torch
    import torch.nn.functional as F

    def label_smoothing_loss(logits, targets, epsilon=0.1, n_classes=10):
        log_probs = F.log_softmax(logits, dim=-1)
        # 딱딱한 목표 성분
        nll = F.nll_loss(log_probs, targets, reduction='none')
        # 균등 성분
        smooth = -log_probs.mean(dim=-1)
        return ((1 - epsilon) * nll + epsilon * smooth).mean()
    ```
    이름표 평활화는 지나치게 확신에 찬 예측을 막고 보통 보정을 개선한다. 신뢰도 그림에서 예측 확률이 관측 빈도와 더 잘 맞는 것을 볼 수 있다.

---

**연습문제 3.**
교차 엔트로피를 최소화하는 것이 경험적 분포와 모델 분포 사이의 KL 발산을 최소화하는 것과 같음을 보여라.

??? success "연습문제 3 풀이"
    경험적 분포 $q$에서 모델 $p_\theta$으로의 KL 발산은 다음과 같다.

    $$
    D_{\text{KL}}(q \| p_\theta) = \sum_k q_k \log \frac{q_k}{p_{\theta,k}} = \underbrace{-H(q)}_{\text{constant}} + \underbrace{\left(-\sum_k q_k \log p_{\theta,k}\right)}_{\text{cross-entropy}}
    $$

    $H(q)$은 $\theta$에 의존하지 않으므로 $D_{\text{KL}}(q\|p_\theta)$을 최소화하는 것은 교차 엔트로피 $H(q, p_\theta) = -\sum_k q_k \log p_{\theta,k}$을 최소화하는 것과 같다. $\square$

---

**연습문제 4.**
누적 연결 확률을 사용하여 순서를 존중하는 순서형 분류(평점 1--5)의 손실 함수를 설계하라.

??? success "연습문제 4 풀이"
    누적 확률을 모형화한다. 문턱값 $\theta_1 < \theta_2 < \cdots < \theta_{K-1}$에 대해 $P(Y \leq k | x) = \sigma(\theta_k - f(x))$이다.

    클래스 확률: $P(Y = k) = P(Y \leq k) - P(Y \leq k-1)$이다.

    NLL 손실은 다음과 같다.

    ```python
    def ordinal_loss(logit, thresholds, target):
        # logit: 스칼라 출력 f(x), thresholds: K-1개의 값
        cumprobs = torch.sigmoid(thresholds - logit)
        # P(Y=k) = cumprob[k] - cumprob[k-1]
        probs = torch.cat([cumprobs[:1], cumprobs[1:] - cumprobs[:-1],
                          1 - cumprobs[-1:]])
        return -torch.log(probs[target] + 1e-8)
    ```
    이웃한 클래스가 문턱값 경계를 공유하므로 순서성이 존중된다.

## 정리하며

이 마당은 이진 분류: 베르누이 MLE에서 나오는 BCE、다중 클래스 분류: 범주형 MLE에서 나오는 교차 엔트로피、이름표 평활화、다중 이름표 분류을 차례로 짚었다.

**참고 문헌**

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Chapter 4
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. Chapter 6
- Lin, T.-Y. et al. (2017). "Focal Loss for Dense Object Detection." *ICCV*
- Szegedy, C. et al. (2016). "Rethinking the Inception Architecture for Computer Vision." *CVPR*
