# 학습률과 이동 폭
## 들어가며

**학습률**(흔히 $\eta$, $\alpha$, `lr`로 표기)은 경사 기반 최적화에서 아마도 가장 중요한 하이퍼파라미터일 것이다. 매개변수 갱신의 크기를 조절하며, 수렴 속도와 애초에 수렴하는지 여부 모두에 깊이 영향을 준다.

이 절에서는 학습률의 역할, 최적화 동역학에 미치는 영향, 그리고 선택과 조정을 위한 실용적인 전략을 살펴본다.

## 학습률의 역할

### 갱신 규칙 다시 보기

경사 하강법의 갱신 규칙은 다음과 같다.

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)$$

학습률 $\eta$는 경사의 **크기를 조정한다.**

- **큰 $\eta$**: 매개변수 공간에서 큰 걸음
- **작은 $\eta$**: 작고 조심스러운 걸음

### 물리적 비유

안개 속에서 산을 내려온다고 상상해 보자.

- **큰 학습률**: 뛰어 내려가기 — 빠르지만 위험하다(골짜기를 지나치거나 절벽에서 떨어질 수 있다)
- **작은 학습률**: 발을 끌며 천천히 가기 — 안전하지만 지루하다(해 지기 전에 베이스캠프에 못 갈 수 있다)
- **최적의 학습률**: 빠르게 걷기 — 통제를 유지하면서 효율적으로 나아간다

## 학습률에 따른 효과

### 너무 작을 때 (eta << 1)

**증상:**

- 수렴이 매우 느리다
- 많은 반복이 필요하다
- 얕은 국소 최솟값에 갇힐 수 있다
- 학습이 지나치게 오래 걸린다

**궤적 예시**(1차원):
```
Loss ↑
     │╲
     │ ╲
     │  ╲
     │   ╲
     │    ╲
     │     ╲
     │      ╲
     │       ╲______________  (very slow descent)
     └──────────────────────→ Iterations
```

### 너무 클 때 (eta >> 1)

**증상:**

- 최솟값 주위에서 진동한다
- 최적값을 지나친다
- 완전히 발산할 수 있다(손실이 무한대로 커진다)
- 학습이 불안정해진다

**궤적 예시**(1차원):
```
Parameter
     │    ╱╲    ╱╲
     │   ╱  ╲  ╱  ╲
     │  ╱    ╲╱    ╲   (oscillating)
     │ ╱            ╲
─────┼─────────────────── Optimal
     │
     └──────────────────→ Iterations
```

### 적당할 때

**증상:**

- 손실이 꾸준히 감소한다
- 수렴이 매끄럽다
- 매개변수가 좋은 값에서 안정된다
- 학습 시간이 효율적이다

**궤적 예시**(1차원):
```
Loss ↑
     │╲
     │ ╲
     │  ╲
     │   ╲____
     │        ╲___________  (smooth descent)
     └──────────────────────→ Iterations
```

## 수학적 분석

### 수렴 조건

**이차 손실** $L(\theta) = \frac{1}{2}a(\theta - \theta^*)^2$에 대해 갱신은 다음과 같이 된다.

$$\theta_{t+1} = \theta_t - \eta \cdot a(\theta_t - \theta^*)$$

정리하면 다음과 같다.

$$\theta_{t+1} - \theta^* = (1 - \eta a)(\theta_t - \theta^*)$$

수렴하려면 $|1 - \eta a| < 1$이어야 하며, 이는 다음을 요구한다.

$$0 < \eta < \frac{2}{a}$$

### 최적 학습률

$1 - \eta a = 0$일 때 가장 빠르게 수렴하며, 이때 다음을 얻는다.

$$\eta^* = \frac{1}{a}$$

이 이상적인 경우에는 **한 단계** 만에 수렴한다!

### 일반적인 경우: 립시츠 경사

**$L$-립시츠 연속인 경사** 를 가진 함수에 대해 다음이 성립한다.

$$\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\| \leq L\|\mathbf{x} - \mathbf{y}\|$$

다음 조건에서 수렴이 보장된다.

$$\eta \leq \frac{1}{L}$$

## 실무에서의 학습률 선택

### 어림 규칙 값

| 문제 유형 | 시작 학습률 |
|--------------|----------------------|
| 선형 회귀 | 0.01 - 0.1 |
| 로지스틱 회귀 | 0.01 - 0.1 |
| 간단한 신경망 | 0.001 - 0.01 |
| 심층 신경망(SGD) | 0.01 - 0.1 |
| 심층 신경망(Adam) | 0.0001 - 0.001 |
| 트랜스포머 | 0.00001 - 0.0001 |
| 사전학습 모델 미세 조정 | 0.00001 - 0.00005 |

### 학습률 탐색기

좋은 학습률을 찾는 체계적인 방법이다.

```python
def learning_rate_finder(model, train_loader, criterion, 
                         lr_min=1e-7, lr_max=1, num_iter=100):
    """
    Find optimal learning rate by gradually increasing it
    and monitoring loss.
    """
    # 원래 상태 저장
    model_state = copy.deepcopy(model.state_dict())
    
    # 학습률을 지수적으로 늘린다
    lr_schedule = np.logspace(np.log10(lr_min), np.log10(lr_max), num_iter)
    
    losses = []
    lrs = []
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_min)
    
    for i, (data, target) in enumerate(train_loader):
        if i >= num_iter:
            break
            
        # 학습률 설정
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_schedule[i]
        
        # 학습 단계
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        lrs.append(lr_schedule[i])
        
        # 손실이 폭발하면 멈춘다
        if loss.item() > 4 * losses[0]:
            break
    
    # 원래 모델 복원
    model.load_state_dict(model_state)
    
    # 손실이 가장 빠르게 줄어드는 학습률 찾기
    # (로그-로그 그래프에서 기울기가 가장 가파르게 음수인 지점)
    return lrs, losses
```

**사용법:**
```python
lrs, losses = learning_rate_finder(model, train_loader, criterion)
plt.semilogx(lrs, losses)
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title('Learning Rate Finder')
# 손실이 아직 줄어드는 구간(폭발하기 전)의 학습률을 고른다
```

### 격자 탐색

단순하지만 효과적이다.

```python
learning_rates = [0.0001, 0.001, 0.01, 0.1]

results = {}
for lr in learning_rates:
    model = create_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    final_loss = train(model, optimizer, epochs=50)
    results[lr] = final_loss
    
best_lr = min(results, key=results.get)
```

## 시각화: 학습률의 효과

### 1차원 손실 지형

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

def gradient_fn(w):
    return 2 * (w - 3)  # Gradient of L(w) = (w - 3)²

def run_gd(w_init, lr, n_steps):
    w = w_init
    trajectory = [w]
    for _ in range(n_steps):
        w = w - lr * gradient_fn(w)
        trajectory.append(w)
    return trajectory

# 서로 다른 학습률
learning_rates = [0.1, 0.5, 0.9, 1.1]
w_init = 7.0
n_steps = 15

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 손실 지형
w_vals = np.linspace(-2, 8, 100)
loss_vals = (w_vals - 3) ** 2

for idx, lr in enumerate(learning_rates):
    ax = axes[idx // 2, idx % 2]
    trajectory = run_gd(w_init, lr, n_steps)
    loss_trajectory = [(w - 3)**2 for w in trajectory]
    
    ax.plot(w_vals, loss_vals, 'gray', alpha=0.5)
    ax.plot(trajectory, loss_trajectory, 'ro-', markersize=5)
    ax.axvline(x=3, color='blue', linestyle='--', alpha=0.5)
    ax.set_xlabel('Weight w')
    ax.set_ylabel('Loss')
    ax.set_title(f'Learning Rate = {lr}')
    ax.set_ylim(-1, 20)

plt.tight_layout()
plt.show()
```

### 2차원 손실 지형

```python
# 2차원 손실 곡면 위의 최적화 경로 시각화
def compute_loss_2d(w, b, X, y):
    y_pred = w * X + b
    return torch.mean((y_pred - y) ** 2).item()

# 등고선 그림 생성
w_range = np.linspace(1, 5, 50)
b_range = np.linspace(0, 4, 50)
W, B = np.meshgrid(w_range, b_range)
Z = np.array([[compute_loss_2d(w, b, X, y) for w, b in zip(row_w, row_b)] 
              for row_w, row_b in zip(W, B)])

plt.contour(W, B, Z, levels=20)
plt.colorbar(label='Loss')
# 최적화 궤적을 겹쳐 그리기
plt.plot(w_history, b_history, 'r.-', label='GD path')
plt.xlabel('Weight w')
plt.ylabel('Bias b')
```

## 학습률 스케줄

### 왜 학습률을 줄이는가?

- **학습 초기**: 큰 학습률이 매개변수 공간을 빠르게 탐색한다
- **학습 후기**: 작은 학습률이 최솟값 근처에서 미세 조정을 가능하게 한다

### 흔히 쓰는 스케줄

**계단식 감쇠:**

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t/s \rfloor}$$

```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=30, gamma=0.1
)
```

**지수 감쇠:**

$$\eta_t = \eta_0 \cdot \gamma^t$$

```python
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, gamma=0.95
)
```

**코사인 어닐링:**

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t\pi}{T}))$$

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100
)
```

**ReduceLROnPlateau**(적응형):
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
# 학습 루프 안에서:
scheduler.step(validation_loss)
```

### 워밍업

작은 학습률로 시작해 점점 늘린다.

```python
def warmup_lambda(epoch):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    return 1.0

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=warmup_lambda
)
```

## 학습률과 배치 크기

### 선형 비례 규칙

배치 크기를 $k$배로 늘릴 때 학습률도 $k$배로 조정한다.

$$\eta_{new} = k \cdot \eta_{original}$$

**직관**: 배치가 클수록 더 정확한 경사 추정을 주므로 더 큰 걸음이 가능하다.

### 실무적 한계

- 배치 크기가 아주 크면 워밍업이 필요할 수 있다
- 어떤 배치 크기를 넘어서면 일반화 성능이 나빠질 수 있다
- 메모리 제약이 배치 크기를 제한하는 경우가 많다

## 적응적 학습률

### 매개변수별 학습률

**Adam**, **RMSprop**, **AdaGrad** 같은 알고리즘은 매개변수마다 별도의 학습률을 유지한다.

$$\theta_{j,t+1} = \theta_{j,t} - \frac{\eta}{\sqrt{v_{j,t}} + \epsilon} \cdot m_{j,t}$$

**장점:**

- 서로 다른 경사 크기에 자동으로 적응한다
- 희소한 경사에서도 잘 동작한다
- 초기 학습률 선택에 덜 민감하다

**참고**: [Adam 최적화기](../../ch05/optimizers/adam.md), [RMSprop](../../ch05/optimizers/rmsprop.md)

## 학습률 문제 진단하기

### 학습률이 너무 클 때의 징후

- 손실이 증가하거나 심하게 진동한다
- `nan`이나 `inf` 값이 나타난다
- 경사가 폭발한다

**해결**: 학습률을 10분의 1로 줄인다

### 학습률이 너무 작을 때의 징후

- 손실이 지나치게 느리게 감소한다
- 검증 손실이 일찍 정체된다
- 학습이 비합리적으로 오래 걸린다

**해결**: 학습률을 2~10배로 늘린다

### 진단 코드

```python
def diagnose_lr(train_losses, val_losses):
    # 발산 여부 확인
    if any(np.isnan(train_losses)) or any(np.isinf(train_losses)):
        return "LR too high: NaN/Inf detected"
    
    # 진동 여부 확인
    if len(train_losses) > 10:
        recent = train_losses[-10:]
        if max(recent) > 2 * min(recent):
            return "LR too high: Significant oscillation"
    
    # 느린 수렴 여부 확인
    if len(train_losses) > 50:
        improvement = (train_losses[0] - train_losses[-1]) / train_losses[0]
        if improvement < 0.1:
            return "LR may be too low: Slow progress"
    
    return "LR appears reasonable"
```

## 핵심 요점

1. **학습률은 경사 갱신의 크기를 조정한다**: 이동 폭을 조절한다
2. **너무 크면**: 진동, 발산, 불안정
3. **너무 작으면**: 느린 수렴, 낭비되는 계산
4. **학습률 탐색기를 쓴다**: 체계적인 선택 방법
5. **학습률 스케줄**: 학습이 진행됨에 따라 줄이면 결과가 가장 좋다
6. **적응적 방법**: Adam 등이 학습률 민감도를 낮춘다
7. **배치 크기에 맞춰 조정한다**: 배치가 크면 더 큰 학습률을 쓸 수 있다

## 다른 주제와의 연결

- **최적화기**: 최적화기의 기초 참고
- **스케줄러**: 학습률 스케줄러에서 자세히 다룬다
- **Adam**: [Adam 최적화기](../../ch05/optimizers/adam.md)의 매개변수별 학습률
- **배치 크기**: [배치, 미니배치, SGD](batch_minibatch_sgd.md)와 관련된다

## 참고 문헌

- Smith, L. N. (2017). Cyclical learning rates for training neural networks. WACV.
- Goyal, P., et al. (2017). Accurate, large minibatch SGD: Training ImageNet in 1 hour. arXiv:1706.02677.
- You, Y., et al. (2019). Large batch optimization for deep learning: Training BERT in 76 minutes. arXiv:1904.00962.

## 연습문제

**연습문제 1.**
$L(w) = (w-5)^2$에 대해 $w_0 = 0$에서 시작하여 $\eta = 0.1$일 때와 $\eta = 1.0$일 때 처음 5회 반복을 계산하라. 어느 쪽이 수렴하는가?

??? success "연습문제 1 풀이"
    경사는 $\nabla L = 2(w - 5)$이므로 갱신 규칙은 $w_{t+1} = w_t - 2\eta(w_t - 5)$이다.

    $\eta = 0.1$일 때: $w_0=0, w_1=1, w_2=1.8, w_3=2.44, w_4=2.952, w_5=3.362$. $w^*=5$를 향해 매끄럽게 수렴한다.

    $\eta = 1.0$일 때: $w_0=0, w_1=10, w_2=0, w_3=10, w_4=0, w_5=10$. 진동하며 결코 수렴하지 않는다. 임계 문턱은 경사의 립시츠 상수가 $L=2$일 때 $\eta < 1/L$이므로 $\eta < 0.5$이다.

---

**연습문제 2.**
이차 함수 $f(x) = \frac{1}{2}x^\top A x - b^\top x$에 대한 경사 하강법이 수렴할 필요충분조건이 $\eta < \frac{2}{\lambda_{\max}(A)}$임을 증명하라. 여기서 $\lambda_{\max}$는 가장 큰 고윳값이다.

??? success "연습문제 2 풀이"
    경사는 $\nabla f = Ax - b$이다. 갱신은 $x_{t+1} = x_t - \eta(Ax_t - b) = (I - \eta A)x_t + \eta b$가 된다. $x^* = A^{-1}b$일 때 $e_t = x_t - x^*$라 두면 $e_{t+1} = (I - \eta A)e_t$이다.

    $A$의 고유기저에서 각 성분은 $e_t^{(i)} = (1 - \eta \lambda_i)^t e_0^{(i)}$로 변화한다. 수렴하려면 모든 $i$에 대해 $|1 - \eta \lambda_i| < 1$이어야 하며 이는 $0 < \eta < 2/\lambda_i$를 준다. 구속력을 갖는 조건은 가장 큰 고윳값에서 나온다. 즉 $\eta < 2/\lambda_{\max}$이다. $\square$

---

**연습문제 3.**
PyTorch로 코사인 어닐링 스케줄을 구현하고 100 에폭 동안의 학습률을 그려라. 2층 MLP로 MNIST를 학습하며 상수 학습률과 손실 곡선을 비교하라.

??? success "연습문제 3 풀이"
    ```python
    import torch
    import torch.nn as nn
    from torch.optim.lr_scheduler import CosineAnnealingLR

    model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)

    lrs = []
    for epoch in range(100):
        lrs.append(scheduler.get_last_lr()[0])
        # ... 학습 단계 ...
        scheduler.step()
    # lrs는 다음을 따른다: lr_t = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(pi*t/T_max))
    ```
    코사인 스케줄은 수렴 근처에서 작은 학습률로 더 많은 에폭을 보내며 세밀한 최적화를 가능하게 하므로 대개 더 낮은 최종 손실을 달성한다.

---

**연습문제 4.**
(학습률을 배치 크기에 비례하여 조정하는) 선형 비례 규칙이 미니배치 간 경사가 비슷할 때 성립하는 근사임을 보여라. 이 근사는 언제 무너지는가?

??? success "연습문제 4 풀이"
    학습률 $\eta$, 배치 크기 $B$로 미니배치 단계를 $k$번 밟는다고 하자. 전체 매개변수 변화는 $\Delta \theta = -\eta \sum_{i=1}^{k} g_i$이며 $g_i$는 각 미니배치의 경사이다. 배치 크기 $kB$, 학습률 $k\eta$로 한 단계를 밟으면 $\bar{g} = \frac{1}{k}\sum g_i$일 때 $\Delta \theta' = -k\eta \bar{g}$이다.

    이 둘은 같다. $\Delta \theta' = -\eta \sum g_i = \Delta \theta$이다. 다만 이는 결합된 갱신에 걸쳐 손실 지형이 선형이라고 가정한 것이며, (1) 곡률에 비해 학습률이 크거나, (2) 배치 정규화 통계가 배치 크기에 따라 달라지거나, (3) 손실 곡면이 매우 비이차적일 때 무너진다.
