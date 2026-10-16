# 러닝레이트와 이동 폭

**러닝레이트**(보통 $\eta$, $\alpha$, `lr`로 표기)은 경사 기반 업데이트의 크기를 조절하는 하이퍼파라미터다. 적절한 크기는 로스의 척도, 입력 정규화, 모델과 옵티마이저에 따라 달라진다. 1보다 작다는 이유만으로 작은 러닝레이트라고 판단할 수는 없다.

이 절에서는 러닝레이트의 역할, 최적화 동역학에 미치는 영향, 그리고 선택과 조정을 위한 실용적인 전략을 살펴본다.

---

## 1. 러닝레이트의 역할

### 갱신 규칙 다시 보기

경사 하강법의 갱신 규칙은 다음과 같다.

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)$$

러닝레이트 $\eta$는 경사의 **크기를 조정한다.**

- **큰 $\eta$**: 파라메터 공간에서 큰 스텝
- **작은 $\eta$**: 작고 조심스러운 스텝

### 물리적 비유

안개 속에서 산을 내려온다고 상상해 보자.

- **큰 러닝레이트**: 뛰어 내려가기 — 빠르지만 위험하다(골짜기를 지나치거나 절벽에서 떨어질 수 있다)
- **작은 러닝레이트**: 발을 끌며 천천히 가기 — 안전하지만 지루하다(해 지기 전에 베이스캠프에 못 갈 수 있다)
- **최적의 러닝레이트**: 빠르게 걷기 — 통제를 유지하면서 효율적으로 나아간다

---

## 2. 러닝레이트에 따른 효과

### 문제의 척도에 비해 너무 작을 때

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

### 문제의 척도에 비해 너무 클 때

**증상:**

- 최솟값 주위에서 진동한다
- 최적값을 지나친다
- 완전히 발산할 수 있다(로스가 무한대로 커진다)
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

- 로스가 꾸준히 감소한다
- 수렴이 매끄럽다
- 파라메터가 좋은 값에서 안정된다
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

---

## 3. 수학적 분석

### 수렴 조건

**$a>0$인 이차 로스** $L(\theta) = \frac{1}{2}a(\theta - \theta^*)^2$에 대해 갱신은 다음과 같이 된다.

$$\theta_{t+1} = \theta_t - \eta \cdot a(\theta_t - \theta^*)$$

정리하면 다음과 같다.

$$\theta_{t+1} - \theta^* = (1 - \eta a)(\theta_t - \theta^*)$$

수렴하려면 $|1 - \eta a| < 1$이어야 하며, 이는 다음을 요구한다.

$$0 < \eta < \frac{2}{a}$$

### 최적 러닝레이트

$1 - \eta a = 0$일 때 가장 빠르게 수렴하며, 이때 다음을 얻는다.

$$\eta^* = \frac{1}{a}$$

이 이상적인 경우에는 **한 단계** 만에 수렴한다!

### 일반적인 경우: 립시츠 경사

**$L$-립시츠 연속인 경사** 를 가진 함수에 대해 다음이 성립한다.

$$\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\| \leq L\|\mathbf{x} - \mathbf{y}\|$$

최솟값이 존재하는 볼록 함수라면, 다음과 같은 양의 고정 러닝레이트에서 경사 하강법의 목적함수 값이 최솟값으로 수렴한다.

$$0 < \eta \leq \frac{1}{L}$$

여기서 $L$은 로스 이름이 아니라 경사의 립시츠 상수다. 비볼록 함수에서도 매끄러움 조건 아래 충분히 작은 러닝레이트는 하강을 보장할 수 있지만, 전역 최솟값으로의 수렴을 보장하지는 않는다.

---

## 4. 실무에서의 러닝레이트 선택

### 탐색 범위의 예

아래 값은 넓은 탐색을 시작할 때의 예시다. 문제 유형만으로 적절한 러닝레이트를 정할 수는 없으며, 사용한 로스의 합·평균 방식과 파라미터화도 함께 고려한다.

| 문제 유형 | 시작 러닝레이트 |
|--------------|----------------------|
| 선형 회귀 | 0.01 - 0.1 |
| 로지스틱 회귀 | 0.01 - 0.1 |
| 간단한 뉴럴 넷트웍 | 0.001 - 0.01 |
| 심층 뉴럴 넷트웍(SGD) | 0.01 - 0.1 |
| 심층 뉴럴 넷트웍(Adam) | 0.0001 - 0.001 |
| 트랜스포머 | 0.00001 - 0.0001 |
| 사전학습 모델 미세 조정 | 0.00001 - 0.00005 |

### 러닝레이트 탐색기

러닝레이트를 점차 늘리면서 로스가 줄어드는 구간과 불안정해지는 구간을 관찰하는 방법이다. 탐색 자체가 모델과 옵티마이저 상태를 바꾸므로 본 학습 전에는 초기 상태를 복원해야 한다.

```python
def learning_rate_finder(model, train_loader, criterion, 
                         lr_min=1e-7, lr_max=1, num_iter=100):
    """
    러닝레이트를 조금씩 키우고 로스를 지켜보며
    가장 좋은 값을 찾는다.
    """
    # 원래 상태 저장
    model_state = copy.deepcopy(model.state_dict())
    
    # 러닝레이트를 지수적으로 늘린다
    lr_schedule = np.logspace(np.log10(lr_min), np.log10(lr_max), num_iter)
    
    losses = []
    lrs = []
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_min)
    
    for i, (data, target) in enumerate(train_loader):
        if i >= num_iter:
            break
            
        # 러닝레이트 설정
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
        
        # 로스가 폭발하면 멈춘다
        if loss.item() > 4 * losses[0]:
            break
    
    # 원래 모델 복원
    model.load_state_dict(model_state)
    
    # 로스가 가장 빠르게 줄어드는 러닝레이트 찾기
    # (로그-로그 그래프에서 그래디언트가 가장 가파르게 음수인 지점)
    return lrs, losses
```

**사용법:**
```python
lrs, losses = learning_rate_finder(model, train_loader, criterion)
plt.semilogx(lrs, losses)
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title('Learning Rate Finder')
# 로스가 아직 줄어드는 구간(폭발하기 전)의 러닝레이트를 고른다
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

---

## 5. 시각화: 러닝레이트의 효과

### 1차원 로스 지형

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

# 서로 다른 러닝레이트
learning_rates = [0.1, 0.5, 0.9, 1.1]
w_init = 7.0
n_steps = 15

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 로스 지형
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

### 2차원 로스 지형

```python
# 2차원 로스 곡면 위의 최적화 경로 시각화
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

---

## 6. 러닝레이트 스케줄

### 왜 러닝레이트를 줄이는가?

- **학습 초기**: 큰 러닝레이트가 파라메터 공간을 빠르게 탐색한다
- **학습 후기**: 작은 러닝레이트가 최솟값 근처에서 미세 조정을 가능하게 한다

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

작은 러닝레이트로 시작해 점점 늘린다.

```python
def warmup_lambda(epoch):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    return 1.0

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=warmup_lambda
)
```

---

## 7. 러닝레이트와 배치 크기

### 선형 비례 규칙

배치 크기를 $k$배로 늘릴 때 러닝레이트도 $k$배로 늘려 보는 경험적 규칙이다. 모든 모델과 옵티마이저에서 성립하는 법칙은 아니다.

$$\eta_{new} = k \cdot \eta_{original}$$

**직관**: 작은 배치로 여러 번 업데이트할 때 경사가 거의 변하지 않는다면, 이를 큰 배치의 한 번 업데이트로 근사할 수 있다. 경사 추정의 분산 감소만으로 큰 러닝레이트의 안정성이 보장되지는 않는다.

### 실무적 한계

- 배치 크기가 아주 크면 워밍업이 필요할 수 있다
- 어떤 배치 크기를 넘어서면 일반화 성능이 나빠질 수 있다
- 메모리 제약이 배치 크기를 제한하는 경우가 많다

---

## 8. 적응적 러닝레이트

### 파라메터별 러닝레이트

**Adam**, **RMSprop**, **AdaGrad**는 경사 통계를 이용해 파라미터 성분별 업데이트 크기를 조정한다. 다음 식은 Adam 형태의 개략적인 표현이며, 정확한 규칙에는 이동 평균의 바이어스 보정 등이 포함된다.

$$\theta_{j,t+1} = \theta_{j,t} - \frac{\eta}{\sqrt{v_{j,t}} + \epsilon} \cdot m_{j,t}$$

**장점:**

- 서로 다른 경사 크기에 자동으로 적응한다
- 희소한 경사에서도 잘 동작한다
- 초기 러닝레이트 선택에 덜 민감하다

**참고**: [Adam 옵티마이저](../../ch09/optimizers/adam.md), [RMSprop](../../ch09/optimizers/rmsprop.md)

---

## 9. 러닝레이트 문제 진단하기

### 러닝레이트가 너무 클 때의 징후

- 로스가 증가하거나 심하게 진동한다
- `nan`이나 `inf` 값이 나타난다
- 경사가 폭발한다

**점검**: 더 작은 러닝레이트로 비교하되, 데이터의 비정상 값, 로스 구현, 정규화와 수치 정밀도도 함께 확인한다.

### 러닝레이트가 너무 작을 때의 징후

- 로스가 지나치게 느리게 감소한다
- 검증 로스가 일찍 정체된다
- 학습이 비합리적으로 오래 걸린다

**점검**: 러닝레이트를 단계적으로 늘려 비교한다. 경사 단절, 모델 용량 부족, 데이터 문제나 과적합도 정체를 만들 수 있으므로 검증 로스만으로 판단하지 않는다.

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

---

## 10. 다른 주제와의 연결

- **옵티마이저**: 옵티마이저의 기초 참고
- **스케줄러**: 러닝레이트 스케줄러에서 자세히 다룬다
- **Adam**: [Adam 옵티마이저](../../ch09/optimizers/adam.md)의 파라메터별 러닝레이트
- **배치 크기**: [배치, 미니배치, SGD](batch_minibatch_sgd.md)와 관련된다

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
$L(w) = (w-5)^2$에 대해 $w_0 = 0$에서 시작하여 $\eta = 0.1$일 때와 $\eta = 1.0$일 때 처음 5회 반복을 계산하라. 어느 쪽이 수렴하는가?

</div>

??? success "연습문제 1 풀이"
    경사는 $\nabla L = 2(w - 5)$이므로 갱신 규칙은 $w_{t+1} = w_t - 2\eta(w_t - 5)$이다.

    $\eta = 0.1$일 때: $w_0=0, w_1=1, w_2=1.8, w_3=2.44, w_4=2.952, w_5=3.362$. $w^*=5$를 향해 매끄럽게 수렴한다.

    $\eta = 1.0$일 때: $w_0=0, w_1=10, w_2=0, w_3=10, w_4=0, w_5=10$. 진동하며 결코 수렴하지 않는다. 이 이차 함수의 정확한 수렴 범위는 $0<\eta<2/L=1$이다. $\eta=0.5$에서는 한 번에 최솟값에 도달하고, $0.5<\eta<1$에서는 진동 폭이 줄어들며 수렴한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
대칭 양의 정부호 행렬 $A$에 대한 이차 함수 $f(x) = \frac{1}{2}x^\top A x - b^\top x$에 대한 경사 하강법이 수렴할 모든 초기값에서 최적해로 수렴할 필요충분조건이 $0<\eta < \frac{2}{\lambda_{\max}(A)}$임을 증명하라. 여기서 $\lambda_{\max}$는 가장 큰 고윳값이다.

</div>

??? success "연습문제 2 풀이"
    경사는 $\nabla f = Ax - b$이다. 갱신은 $x_{t+1} = x_t - \eta(Ax_t - b) = (I - \eta A)x_t + \eta b$가 된다. $x^* = A^{-1}b$일 때 $e_t = x_t - x^*$라 두면 $e_{t+1} = (I - \eta A)e_t$이다.

    $A$의 고유기저에서 각 성분은 $e_t^{(i)} = (1 - \eta \lambda_i)^t e_0^{(i)}$로 변화한다. 수렴하려면 모든 $i$에 대해 $|1 - \eta \lambda_i| < 1$이어야 하며 이는 $0 < \eta < 2/\lambda_i$를 준다. 구속력을 갖는 조건은 가장 큰 고윳값에서 나온다. 즉 $\eta < 2/\lambda_{\max}$이다. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
PyTorch로 코사인 어닐링 스케줄을 구현하고 100 에포크 동안의 러닝레이트를 그려라. 2레이어 MLP로 MNIST를 학습하며 상수 러닝레이트와 로스 곡선을 비교하라.

</div>

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
    제시한 풀이는 러닝레이트 곡선을 그리는 부분이다. MNIST 비교에는 같은 초기값, 데이터 순서, 학습 예산으로 두 학습을 추가해야 한다. 코사인 스케줄이 항상 더 낮은 로스를 주는 것은 아니므로 실제 검증 결과로 판단한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
(러닝레이트를 배치 크기에 비례하여 조정하는) 선형 비례 규칙이 미니배치 간 경사가 비슷할 때 성립하는 근사임을 보여라. 이 근사는 언제 무너지는가?

</div>

??? success "연습문제 4 풀이"
    러닝레이트 $\eta$, 배치 크기 $B$로 미니배치 단계를 $k$번 밟는다고 하자. 전체 파라메터 변화는 $\Delta \theta = -\eta \sum_{i=1}^{k} g_i$이며 $g_i$는 각 미니배치의 경사이다. 배치 크기 $kB$, 러닝레이트 $k\eta$로 한 단계를 밟으면 $\bar{g} = \frac{1}{k}\sum g_i$일 때 $\Delta \theta' = -k\eta \bar{g}$이다.

    모든 $g_i$를 같은 파라미터에서 계산한다면 두 식은 정확히 같다. 실제 작은 배치 학습에서는 매번 파라미터가 바뀌므로, 그 구간에서 경사가 거의 변하지 않을 때만 근사가 성립한다. 큰 곡률, 큰 이동, 배치 통계의 변화, 모멘텀·적응적 상태의 여러 번 갱신 등은 차이를 키울 수 있다.

## 정리하며

1. **러닝레이트는 경사 갱신의 크기를 조정한다**: 이동 폭을 조절한다
2. **너무 크면**: 진동, 발산, 불안정
3. **너무 작으면**: 느린 수렴, 낭비되는 계산
4. **러닝레이트 탐색기를 쓴다**: 체계적인 선택 방법
5. **러닝레이트 스케줄**: 학습 단계에 맞춰 조정하고 검증 결과로 비교한다
6. **적응적 방법**: Adam 등이 러닝레이트 민감도를 낮춘다
7. **배치 크기에 맞춰 조정한다**: 선형 비례 규칙을 초기 가설로 삼되 안정성을 확인한다

**참고 문헌**

- Smith, L. N. (2017). Cyclical learning rates for training neural networks. WACV.
- Goyal, P., et al. (2017). Accurate, large minibatch SGD: Training ImageNet in 1 hour. arXiv:1706.02677.
- You, Y., et al. (2019). Large batch optimization for deep learning: Training BERT in 76 minutes. arXiv:1904.00962.
