# 시간을 거슬러 가는 역전파

순환 신경망을 학습시키려면 펼친 계산 그래프를 지나 기울기를 계산해야 하는데, 이 절차를 **시간을 거슬러 가는 역전파(BPTT)**라 한다. BPTT를 이해하면 RNN이 어떻게 배우는지, 그리고 왜 긴 순차열에서 애를 먹는지 알 수 있다.

---

## 1. 펼친 계산 그래프

길이가 $T$인 순차열을 처리하는 RNN을 생각해 보자.

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$$

$$y_t = W_{hy} h_t$$

펼치면 깊이가 순차열의 길이 $T$과 같고 모든 층이 가중치를 나누어 쓰며 "층" 하나가 시각 하나에 대응하는 깊은 순방향 신경망이 된다.

```
Loss = L₁ + L₂ + L₃ + L₄
        ↑    ↑    ↑    ↑
       y₁   y₂   y₃   y₄
        ↑    ↑    ↑    ↑
h₀ → h₁ → h₂ → h₃ → h₄
      ↑    ↑    ↑    ↑
     x₁   x₂   x₃   x₄
```

---

## 2. 손실 계산

전체 손실은 시각마다의 몫을 더한 것이다.

$$\mathcal{L} = \sum_{t=1}^{T} L_t(y_t, \hat{y}_t)$$

시각마다 교차 엔트로피로 분류할 때는 다음과 같다.

$$L_t = -\sum_{c} \hat{y}_{t,c} \log(y_{t,c})$$

마지막 출력만 써서 순차열을 분류할 때는 다음과 같다.

$$\mathcal{L} = L_T(y_T, \hat{y})$$

---

## 3. 기울기 유도

### 출력 가중치

기울기 $\frac{\partial \mathcal{L}}{\partial W_{hy}}$은 모든 시각의 몫을 모은 것이다.

$$\frac{\partial \mathcal{L}}{\partial W_{hy}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial y_t} \cdot \frac{\partial y_t}{\partial W_{hy}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial y_t} \cdot h_t^T$$

$y_t = W_{hy} h_t$이므로 미분이 간단하다. $W_{hy}$은 시각마다 국소적으로 작용하므로 시간에 걸친 연쇄 법칙이 필요 없다.

### 숨은 상태의 기울기

숨은 상태 $h_t$의 기울기는 두 곳에서 온다. 하나는 시각 $t$의 지역 손실이고, 다른 하나는 $h_{t+1}$을 지나 미래 시각에서 거슬러 오는 기울기이다.

$$\frac{\partial \mathcal{L}}{\partial h_t} = \frac{\partial L_t}{\partial h_t} + \frac{\partial \mathcal{L}}{\partial h_{t+1}} \cdot \frac{\partial h_{t+1}}{\partial h_t}$$

이 재귀식이 BPTT의 핵심이다. 기울기를 시간을 거슬러 전파한다.

### 순환 가중치

$h_t$이 $W_{hh}$에 곧바로도 기대고 이전의 모든 숨은 상태를 통해서도 기대므로, 기울기 $\frac{\partial \mathcal{L}}{\partial W_{hh}}$을 구하려면 시간에 걸친 온전한 연쇄 법칙이 필요하다.

$$\frac{\partial L_T}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L_T}{\partial h_T} \cdot \frac{\partial h_T}{\partial h_t} \cdot \frac{\partial^+ h_t}{\partial W_{hh}}$$

여기서 $\frac{\partial^+ h_t}{\partial W_{hh}}$은 직접적인(재귀가 아닌) 편미분, 곧 시각 $t$에서 $W_{hh}$을 바로 쓴 몫만을 뜻한다.

---

## 4. 야코비 행렬의 곱

핵심 항 $\frac{\partial h_T}{\partial h_t}$은 야코비 행렬의 곱으로 펼쳐진다.

$$\frac{\partial h_T}{\partial h_t} = \prod_{k=t}^{T-1} \frac{\partial h_{k+1}}{\partial h_k}$$

야코비 행렬 하나하나는 다음과 같다.

$$\frac{\partial h_{k+1}}{\partial h_k} = \text{diag}(1 - h_{k+1}^2) \cdot W_{hh}$$

여기서 $\text{diag}(1 - h_{k+1}^2)$은 활성화 전 값에서 잰 $\tanh$의 도함수이다. 시각 $T - t$개에 걸친 이 행렬 곱이 기울기의 흐름을 살리기도 하고 죽이기도 한다. $T - t$이 크면 이 곱이 사라지거나 폭발하기 쉽다.

---

## 5. PyTorch의 자동 BPTT

PyTorch는 자동 미분 엔진으로 BPTT를 알아서 처리한다. 순전파가 계산 그래프를 세우고 `.backward()`이 BPTT를 수행한다.

```python
import torch
import torch.nn as nn

# 모델
rnn = nn.RNN(input_size=10, hidden_size=20, batch_first=True)
fc = nn.Linear(20, 5)

# 데이터
x = torch.randn(32, 15, 10, requires_grad=True)
targets = torch.randint(0, 5, (32,))

# 순전파 (계산 그래프를 세운다)
outputs, h_n = rnn(x)
logits = fc(h_n.squeeze(0))

# 손실과 역전파 (BPTT가 저절로 일어난다)
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, targets)
loss.backward()

# 이제 기울기를 쓸 수 있다
print(f"Input grad shape: {x.grad.shape}")
print(f"RNN weight grad: {rnn.weight_ih_l0.grad.shape}")
```

**출력:**

```
Input grad shape: torch.Size([32, 15, 10])
RNN weight grad: torch.Size([20, 10])
```

---

## 6. 손으로 하는 BPTT 구현

이해를 돕기 위해 BPTT를 명시적으로 계산해 보자.

```python
def bptt_manual(xs, hs, ys, targets, W_xh, W_hh, W_hy):
    """
    손으로 하는 BPTT 계산.
    
    인수:
        xs: 입력의 목록 [x_1, …, x_T]
        hs: 숨은 상태의 목록 [h_0, h_1, …, h_T]
        ys: 출력의 목록 [y_1, …, y_T]
        targets: 표적의 목록
        W_xh, W_hh, W_hy: 가중치 행렬
    
    반환값:
        모든 매개변수의 기울기
    """
    T = len(xs)
    
    dW_xh = torch.zeros_like(W_xh)
    dW_hh = torch.zeros_like(W_hh)
    dW_hy = torch.zeros_like(W_hy)
    
    # 시간을 거슬러 가는 역전파
    dh_next = torch.zeros_like(hs[0])
    
    for t in reversed(range(T)):
        # 출력 손실에서 오는 기울기
        dy = ys[t] - targets[t]  # 소프트맥스와 교차 엔트로피의 도함수
        
        # 출력 가중치의 기울기
        dW_hy += torch.outer(dy, hs[t + 1])
        
        # 숨은 상태로 흘러드는 기울기
        dh = W_hy.T @ dy + dh_next
        
        # tanh 비선형을 지나는 기울기
        dh_raw = dh * (1 - hs[t + 1] ** 2)
        
        # 입력 가중치와 순환 가중치의 기울기
        dW_xh += torch.outer(dh_raw, xs[t])
        dW_hh += torch.outer(dh_raw, hs[t])
        
        # 이전 시각으로 전파
        dh_next = W_hh.T @ dh_raw
    
    return dW_xh, dW_hh, dW_hy
```

반복문은 시간을 거슬러 나아간다. 단계마다 기울기 `dh_next`이 미래 모든 시각의 정보를 순환을 타고 거슬러 나른다. 이것이 소실과 폭발 문제를 겪는 시간 방향 기울기 흐름이다.

---

## 7. 잘라 낸 BPTT

아주 긴 순차열에서는 (숨은 상태를 모두 저장해야 하므로) 메모리가 순차열 길이에 비례해 늘어 온전한 BPTT를 감당할 수 없다. **잘라 낸 BPTT**는 역전파를 시각 $k$개로 제한한다.

```python
def truncated_bptt(model, sequence, chunk_size, optimizer, criterion):
    """
    잘라 낸 BPTT: 순차열을 덩이로 나누어 처리하고 덩이 안에서만
    역전파한다.
    """
    hidden = None
    total_loss = 0
    
    for i in range(0, len(sequence) - 1, chunk_size):
        inputs = sequence[i:i + chunk_size]
        targets = sequence[i + 1:i + chunk_size + 1]
        
        # 기울기의 흐름을 자르려고 숨은 상태를 떼어 냄
        if hidden is not None:
            hidden = hidden.detach()
        
        # 덩이를 지나는 순전파
        outputs, hidden = model(inputs.unsqueeze(0), hidden)
        loss = criterion(outputs.squeeze(0), targets)
        
        # 역전파 (이 덩이 안에서만)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss
```

`.detach()` 호출이 기울기의 흐름을 자르는 장치이다. 숨은 상태 값은 덩이에서 덩이로 이어지지만 기울기는 떼어 낸 지점을 거슬러 가지 못한다. 기울기의 정확도를 계산 효율과 맞바꾸는 셈이며, $k$단계보다 긴 의존은 배울 수 없다.

---

## 8. 계산 복잡도

길이가 $T$이고 숨은 차원이 $H$인 순차열에 대해 다음과 같다.

| 연산 | 시간 | 공간 |
|-----------|------|-------|
| 순전파 | $O(T \cdot H^2)$ | $O(T \cdot H)$ |
| 역전파 | $O(T \cdot H^2)$ | $O(T \cdot H)$ |
| 잘라 낸 BPTT (덩이 $k$) | $O(T \cdot H^2)$ | $O(k \cdot H)$ |

공간 복잡도는 기울기 계산을 위해 시각마다 숨은 상태를 저장하는 데서 대부분 나온다. 잘라 낸 BPTT는 필요한 공간을 $O(T \cdot H)$에서 $O(k \cdot H)$으로 줄인다.

---

## 9. 기울기 흐름 시각화

```python
import matplotlib.pyplot as plt

def visualize_gradient_flow(model, x):
    """
    마지막 출력에서 입력 자리마다의 기울기 크기를 그려 본다.
    """
    model.train()
    x = x.requires_grad_(True)
    
    outputs, _ = model(x)
    loss = outputs[0, -1, :].sum()
    loss.backward()
    
    grad_norms = x.grad[0].norm(dim=-1).detach().numpy()
    
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(grad_norms)), grad_norms)
    plt.xlabel('Timestep')
    plt.ylabel('Gradient Magnitude')
    plt.title('Gradient Flow Through Time')
    plt.yscale('log')
    plt.show()
```

---

## 연습문제

**연습문제 1.**
온전한 BPTT와 잘라 낸 BPTT의 차이를 설명하라.

??? success "연습문제 1 풀이"
    온전한 BPTT는 순차열 전체를 펼쳐 모든 시각을 거슬러 역전파한다. 잘라 낸 BPTT는 순차열을 길이 $k$의 덩이로 나누어 덩이 안에서만 역전파한다. 잘라 낸 BPTT는 먼 거리 기울기의 정확도를 메모리 효율($O(T)$ 대신 $O(k)$)과 맞바꾼다.

---

**연습문제 2.**
기본 RNN에서 $W_h$에 대한 BPTT 기울기를 유도하라.

??? success "연습문제 2 풀이"
    $\frac{\partial L}{\partial W_h} = \sum_{t=1}^T \frac{\partial L}{\partial h_t} \frac{\partial h_t}{\partial W_h} = \sum_{t=1}^T \delta_t h_{t-1}^\top$이며, 여기서 $\delta_t = \frac{\partial L}{\partial h_t} \odot \tanh'(z_t)$이고 연쇄 법칙이 $\delta_T$을 시간을 거슬러 전파한다.

---

**연습문제 3.**
순차열 길이가 $T$일 때 BPTT의 메모리 복잡도는 얼마인가?

??? success "연습문제 3 풀이"
    역전파를 위해 숨은 상태를 모두 저장하는 데 $O(T \cdot d_h)$이 든다. 긴 순차열에서는 감당하기 어렵다. 해법으로는 잘라 낸 BPTT($T$을 줄인다), 기울기 검사점 두기(저장 대신 다시 계산한다), 되돌릴 수 있는 RNN이 있다.

---

**연습문제 4.**
`torch.autograd`으로 간단한 RNN의 BPTT를 구현하고 기울기를 수치적으로 확인하라.

??? success "연습문제 4 풀이"
    ```python
    # PyTorch는 동적 그래프로 BPTT를 알아서 처리한다
    rnn = nn.RNN(input_size, hidden_size, batch_first=True)
    loss = criterion(rnn(x)[0], targets)
    loss.backward()  # 모든 시각을 지나는 온전한 BPTT
    # 잘라 낸 BPTT: k걸음마다 숨은 상태를 떼어 낸다
    ```

## 정리하며

BPTT는 RNN을 깊은 순방향 신경망으로 펼치고, 시각에 걸친 연쇄 법칙으로 기울기를 계산하고, 공유된 가중치의 기울기를 모아서 역전파를 순환 신경망으로 넓힌다. 핵심은 기울기가 앞쪽 입력에 닿으려면 시각 $T$개를 거슬러야 하고 단계마다 $W_{hh}$과 $\tanh'$을 곱하게 된다는 것이다. 이 곱셈의 사슬이 다음 절에서 살펴볼 기울기 소실과 폭발 문제를 낳는다.
