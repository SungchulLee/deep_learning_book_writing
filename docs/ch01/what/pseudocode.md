# 의사코드 규약

의사코드(pseudocode)는 알고리즘을 특정 언어에 얽매이지 않는 방식으로 기술한다. 딥러닝 문헌에서 의사코드는 학습 루프, 모델 구조, 최적화 절차를 특정 프레임워크에 묶지 않고 전달하는 수단이다.

## 정의

의사코드는 표준적인 수학 및 프로그래밍 관례를 사용하는, 구조화되었지만 형식에 얽매이지 않는 알고리즘 표기법이다.

$$
\begin{array}{ll}
\textbf{for } i = 1 \textbf{ to } n & \text{범위에 대한 반복} \\
\textbf{while } \text{condition} & \text{참인 동안 반복} \\
\textbf{return } \text{value} & \text{결과 출력} \\
x \leftarrow f(x) & \text{대입}
\end{array}
$$

## 설명

딥러닝 논문은 학습 알고리즘을 정확히 명시하기 위해 의사코드를 사용한다. 표준적인 학습 루프를 의사코드로 쓰면 다음과 같다.

```
TRAIN(model, data, lr, epochs)
  for epoch = 1 to epochs
    for (x, y) in data
      y_hat = model(x)          // 순전파
      L = loss(y_hat, y)        // 손실 계산
      g = gradient(L, params)   // 역전파
      params = params - lr * g  // 매개변수 갱신
  return params
```

이 의사코드는 프레임워크 세부사항(PyTorch냐 TensorFlow냐, GPU 배치, 혼합 정밀도)을 추상화하면서 본질적인 논리를 포착한다. 논문을 읽을 때 의사코드는 제안된 방법에 대한 가장 정확한 기술인 경우가 많다.

딥러닝 의사코드의 주요 규약: $\nabla_\theta$는 매개변수에 대한 경사를, $\leftarrow$는 (동등을 뜻하는 $=$와 달리) 대입을, $\sim$은 분포로부터의 표본 추출을 나타낸다.

## 예제

```python
import torch
import torch.nn as nn

# 의사코드를 PyTorch로 옮기기
# 의사코드: ADAM(params, lr, beta1, beta2, eps)
#   m = 0, v = 0, t = 0
#   repeat:
#     t = t + 1
#     g = gradient(loss, params)
#     m = beta1 * m + (1 - beta1) * g
#     v = beta2 * v + (1 - beta2) * g^2
#     m_hat = m / (1 - beta1^t)
#     v_hat = v / (1 - beta2^t)
#     params = params - lr * m_hat / (sqrt(v_hat) + eps)

# 위 의사코드의 PyTorch 구현
model = nn.Linear(5, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

x = torch.randn(32, 5)
y = torch.randn(32, 1)

for step in range(100):
    y_hat = model(x)                           # 순전파
    loss = nn.functional.mse_loss(y_hat, y)    # 손실 계산
    optimizer.zero_grad()                       # 경사 초기화
    loss.backward()                            # 역전파(g 계산)
    optimizer.step()                           # 매개변수 갱신

print(f"Final loss: {loss.item():.6f}")
print(f"Parameters: {list(model.parameters())[0].shape}")
```

## 연습문제

**연습문제 1.**
다음 의사코드를 PyTorch로 옮겨라. 의사코드에 존재하는 모호한 점을 찾아라.
```
SGD_WITH_MOMENTUM(params, lr, beta)
  v = 0
  for each mini-batch (x, y):
    g = gradient(loss(model(x), y), params)
    v = beta * v + g
    params = params - lr * v
```

??? success "연습문제 1 풀이"
    ```python
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=beta)
    for x, y in dataloader:
        loss = loss_fn(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    ```
    모호한 점: (1) 경사에 정칙화 항이 포함되는지 명시되어 있지 않다. (2) `v = 0`이 모양을 명시하지 않는다(`params`와 같아야 한다). (3) 정지 조건이 명시되어 있지 않다. (4) 모멘텀의 관례가 서로 다르다. PyTorch는 $v = \beta v + g$ 다음 $\theta = \theta - \text{lr} \cdot v$를 사용하지만(의사코드와 일치), 어떤 정식화는 $v = \beta v + \text{lr} \cdot g$를 사용한다.

---

**연습문제 2.**
Adam 최적화기의 의사코드를 작성하되, 대입($\leftarrow$), 동등 판정($=$), 표본 추출($\sim$)을 분명히 구분하라.

??? success "연습문제 2 풀이"
    ```
    ADAM(params, lr, beta1=0.9, beta2=0.999, eps=1e-8)
      m <- 0, v <- 0, t <- 0
      repeat until convergence:
        t <- t + 1
        (x, y) ~ DataLoader              // 미니배치 표본 추출
        g <- gradient(loss(model(x), y), params)
        m <- beta1 * m + (1 - beta1) * g       // 1차 모멘트
        v <- beta2 * v + (1 - beta2) * g^2     // 2차 모멘트
        m_hat <- m / (1 - beta1^t)             // 편향 보정
        v_hat <- v / (1 - beta2^t)             // 편향 보정
        params <- params - lr * m_hat / (sqrt(v_hat) + eps)
      return params
    ```
    핵심 표기: 대입은 $\leftarrow$, 표본 추출은 $\sim$이며, $g^2$과 $\sqrt{v}$는 원소별 연산이다.

---

**연습문제 3.**
다음 의사코드에는 버그가 있다. 찾아서 고쳐라.
```
TRAIN(model, data, lr, epochs)
  for epoch = 1 to epochs:
    for (x, y) in data:
      y_hat = model(x)
      L = loss(y_hat, y)
      g = gradient(L, params)
      params = params - lr * g
  return L
```

??? success "연습문제 3 풀이"
    버그: 이 함수는 `L`을 반환하는데, 이는 마지막 에폭의 마지막 미니배치 손실일 뿐 의미 있는 요약값이 아니다. 또한 미묘한 문제가 하나 더 있다. `gradient(L, params)`는 매개변수 갱신 전에 계산되어야 하며, 의사코드는 단계 사이에 누적된 경사를 지우지 않는다. 수정한 버전은 다음과 같다.
    ```
    TRAIN(model, data, lr, epochs)
      for epoch = 1 to epochs:
        for (x, y) in data:
          y_hat = model(x)
          L = loss(y_hat, y)
          g = gradient(L, params)
          params = params - lr * g
          clear_gradients(params)
      return params
    ```
    마지막 손실값이 아니라 `params`(학습된 모델)를 반환해야 한다.

---

**연습문제 4.**
드롭아웃 의사코드를 수식으로 변환하라. 입력 $\mathbf{h}$와 드롭 확률 $p$가 주어질 때 출력 $\mathbf{h}'$의 기댓값은 무엇인가?

??? success "연습문제 4 풀이"
    의사코드: 각 원소 $h_i$에 대해 $m_i \sim \text{Bernoulli}(1-p)$를 추출하고 $h_i' = m_i \cdot h_i / (1-p)$로 둔다. 수식: $\mathbf{h}' = \frac{\mathbf{m} \odot \mathbf{h}}{1-p}$이며 $\mathbf{m} \sim \text{Bernoulli}(1-p)^d$이다. 기댓값: $\mathbb{E}[\mathbf{h}'] = \frac{\mathbb{E}[\mathbf{m}] \odot \mathbf{h}}{1-p} = \frac{(1-p)\mathbf{h}}{1-p} = \mathbf{h}$. $1/(1-p)$로 크기를 조정하면 $\mathbb{E}[\mathbf{h}'] = \mathbf{h}$가 보장되므로, 드롭아웃은 기댓값의 의미에서 항등 연산이 되고 테스트 시점에는 드롭아웃 없이 같은 신경망을 쓸 수 있다.

---

**연습문제 5.**
학습률 워밍업 후 코사인 감쇠를 수행하는 의사코드를 작성하라. 적절한 표기($\leftarrow$, $\min$, $\cos$)를 사용하고 모든 매개변수를 명시하라.

??? success "연습문제 5 풀이"
    ```
    COSINE_WITH_WARMUP(T_warmup, T_total, eta_max, eta_min)
      for t = 1 to T_total:
        if t <= T_warmup:
          eta_t <- eta_max * t / T_warmup          // 선형 워밍업
        else:
          progress <- (t - T_warmup) / (T_total - T_warmup)
          eta_t <- eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(pi * progress))
        // 이 학습 단계에 eta_t를 사용한다
    ```
    매개변수: $T_{\text{warmup}}$(워밍업 단계 수), $T_{\text{total}}$(전체 단계 수), $\eta_{\max}$(최고 학습률), $\eta_{\min}$(최종 학습률). 워밍업은 $T_{\text{warmup}}$ 단계에 걸쳐 0에서 $\eta_{\max}$까지 선형으로 증가시키고, 코사인 감쇠는 남은 단계에 걸쳐 $\eta_{\max}$에서 $\eta_{\min}$까지 부드럽게 감소시킨다.
