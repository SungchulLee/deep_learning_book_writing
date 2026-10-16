# 시간 역전파(BPTT)

순환 뉴럴 넷트웍을 학습시키려면 펼친 계산 그래프를 지나 그래디언트를 계산해야 하는데, 이 절차를 **시간 역전파(BPTT)**라 한다. BPTT를 이해하면 RNN이 어떻게 배우는지, 그리고 왜 긴 시퀀스에서 애를 먹는지 알 수 있다.

---

## 1. 펼친 계산 그래프

길이가 $T$인 시퀀스를 처리하는 RNN을 생각해 보자.

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$$

$$y_t = W_{hy} h_t$$

펼치면 깊이가 시퀀스의 길이 $T$과 같고 모든 레이어가 웨이트를 나누어 쓰며 "레이어" 하나가 시각 하나에 대응하는 깊은 순방향 뉴럴 넷트웍이 된다.

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

## 2. 로스 계산

전체 로스는 시각마다의 몫을 더한 것이다.

$$\mathcal{L} = \sum_{t=1}^{T} L_t(y_t, \hat{y}_t)$$

위 식의 $y_t$는 로짓이다. 분류에서는 $p_t=\operatorname{softmax}(y_t)$를 정답 분포 $\hat y_t$와 비교한다.

$$L_t = -\sum_{c} \hat{y}_{t,c} \log(p_{t,c})$$

마지막 출력만 써서 시퀀스를 분류할 때는 다음과 같다.

$$\mathcal{L} = L_T(y_T, \hat{y})$$

---

## 3. 그래디언트 유도

열벡터 기준으로 $z_t=W_{hh}h_{t-1}+W_{xh}x_t+b$, $g_t=\nabla_{h_t}\mathcal L$라 하자. 현재 시점 로스에서 오는 국소 그래디언트를 $q_t=\nabla_{h_t}L_t$라 쓰면 역방향 재귀식은 다음과 같다.

$$g_t=q_t+J_{t+1}^{\top}g_{t+1},\qquad J_{t+1}=\frac{\partial h_{t+1}}{\partial h_t}$$

마지막 시점에서는 미래 항이 없고, 중간 시점에 로스가 없으면 해당 $q_t$는 0이다. 활성화 전 값에 대한 그래디언트는 다음과 같다.

$$\delta_t=g_t\odot(1-h_t^2)$$

공유된 웨이트의 그래디언트는 각 사용 위치의 직접적인 기여를 합한다.

$$\nabla_{W_{hh}}\mathcal L=\sum_{t=1}^{T}\delta_t h_{t-1}^{\top},\qquad \nabla_{W_{xh}}\mathcal L=\sum_{t=1}^{T}\delta_t x_t^{\top}$$

출력 로짓에 대한 그래디언트를 $r_t=\nabla_{y_t}L_t$라 하면 다음과 같다.

$$\nabla_{W_{hy}}\mathcal L=\sum_{t=1}^{T}r_t h_t^{\top}$$

소프트맥스·교차 엔트로피의 경우 $r_t=p_t-\hat y_t$다. 합계 로스가 아닌 평균 로스를 사용하면 해당 집계의 배율을 그래디언트에도 반영한다.

## 4. 야코비안 행렬의 곱

각 시점의 야코비안과 전체 상태 변화의 야코비안은 다음과 같다.

$$J_k=\operatorname{diag}(1-h_k^2)W_{hh}$$

$$\frac{\partial h_T}{\partial h_t}=J_TJ_{T-1}\cdots J_{t+1}$$

행렬은 일반적으로 교환되지 않으므로 곱의 순서를 명시해야 한다. 마지막 로스의 그래디언트는 이 곱의 전치로 앞쪽 상태에 전달된다. 여러 시점에 로스가 있으면 각 로스에서 오는 경로를 더한다. 긴 시간 간격에서 이 곱의 노름이 작아지거나 커지는 현상이 소실과 폭발의 핵심이다.

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

# 로스와 역전파 (BPTT가 저절로 일어난다)
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, targets)
loss.backward()

# 이제 그래디언트를 쓸 수 있다
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
        W_xh, W_hh, W_hy: 웨이트 행렬
    
    반환값:
        모든 파라메터의 그래디언트
    """
    T = len(xs)
    
    dW_xh = torch.zeros_like(W_xh)
    dW_hh = torch.zeros_like(W_hh)
    dW_hy = torch.zeros_like(W_hy)
    
    # 시간을 거슬러 가는 역전파
    dh_next = torch.zeros_like(hs[0])
    
    for t in reversed(range(T)):
        # 출력 로스에서 오는 그래디언트
        dy = ys[t] - targets[t]  # 소프트맥스와 교차 엔트로피의 도함수
        
        # 출력 웨이트의 그래디언트
        dW_hy += torch.outer(dy, hs[t + 1])
        
        # 숨은 상태로 흘러드는 그래디언트
        dh = W_hy.T @ dy + dh_next
        
        # tanh 비선형을 지나는 그래디언트
        dh_raw = dh * (1 - hs[t + 1] ** 2)
        
        # 입력 웨이트와 순환 웨이트의 그래디언트
        dW_xh += torch.outer(dh_raw, xs[t])
        dW_hh += torch.outer(dh_raw, hs[t])
        
        # 이전 시각으로 전파
        dh_next = W_hh.T @ dh_raw
    
    return dW_xh, dW_hh, dW_hy
```

수동 구현의 `ys`에는 로짓이 아니라 소프트맥스 확률이, `targets`에는 원핫 정답 벡터가 들어가야 `dy=ys-targets`가 의도한 도함수다. 각 시점의 입력과 상태는 1차원 벡터라고 가정하며, 반환값은 세 웨이트의 그래디언트뿐이다. 바이어스 그래디언트와 배치·평균 로스의 배율 처리는 구현되어 있지 않다. `dh_next`는 미래 로스에서 이전 상태로 전파되는 그래디언트를 담는다.

---

## 7. 절단 BPTT

전체 BPTT는 역전파를 위해 여러 시점의 중간값을 저장하므로 긴 시퀀스에서 메모리 부담이 커질 수 있다. 절단 BPTT는 상태 값은 이어 가되 청크 경계에서 계산 그래프를 끊어 역전파 범위를 제한한다.

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
        
        # 그래디언트의 흐름을 자르려고 숨은 상태를 떼어 냄
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

`detach()`는 상태 값의 전달을 유지하면서 청크 경계를 넘는 그래디언트를 차단한다. 긴 과거의 정보가 상태에 남을 수 있으므로 “청크 길이보다 긴 의존을 전혀 학습할 수 없다”는 설명은 지나치게 강하다. 다만 먼 시점의 로스로 이전 청크의 계산을 직접 조정하는 경로는 사라진다.

이 함수는 텐서 하나인 은닉 상태를 가정하며 LSTM의 `(h,c)` 튜플에는 각각 분리가 필요하다. 또 마지막 청크가 시퀀스의 마지막 원소까지 입력에 포함하면 입력과 한 칸 이동한 정답의 길이가 달라질 수 있다. 입력 구간을 마지막 예측 가능한 위치까지 제한하고 정답 길이를 맞춰야 한다. 반환 로스는 청크 로스의 단순 합이므로 전체 토큰 평균과도 구분한다.

---

## 8. 계산 복잡도

길이가 $T$이고 은닉 차원이 $H$인 시퀀스에 대해 다음과 같다.

| 연산 | 시간 | 공간 |
|-----------|------|-------|
| 순전파 | $O(T \cdot H^2)$ | $O(T \cdot H)$ |
| 역전파 | $O(T \cdot H^2)$ | $O(T \cdot H)$ |
| 절단 BPTT (덩이 $k$) | $O(T \cdot H^2)$ | $O(k \cdot H)$ |

표는 배치와 입력·출력 차원을 생략하고 은닉 계산이 지배적이라고 가정한 근사다. 전체 메모리에는 파라메터와 그래디언트, 입력·출력과 연산 버퍼도 포함된다. 청크별 역전파 후 그래프를 해제하면 저장할 중간 상태의 길이 의존성을 T에서 k로 줄일 수 있다.

---

## 9. 그래디언트 흐름 시각화

```python
import matplotlib.pyplot as plt

def visualize_gradient_flow(model, x):
    """
    마지막 출력에서 입력 자리마다의 그래디언트 크기를 그려 본다.
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

이 함수는 첫 번째 시퀀스의 마지막 은닉 출력 합이 입력 위치별로 얼마나 민감한지 그린다. 특정 과제 로스의 그래디언트나 모든 배치의 통계는 아니다. 입력은 그래디언트를 보관할 수 있는 실수 리프 텐서여야 하고, 기존 입력·모델 그래디언트는 따로 초기화해야 한다. GPU 텐서를 NumPy로 바꾸려면 CPU로 옮기는 과정도 필요하다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
전체 BPTT와 절단 BPTT의 상태 전달·그래디언트 전달을 비교하라.

</div>

??? success "연습문제 1 풀이"
    전체 BPTT는 처리한 전체 시퀀스의 계산 그래프를 통해 역전파한다. 절단 BPTT는 상태 값은 다음 청크로 전달하되 경계에서 그래프를 끊는다. 메모리를 줄이는 대신 경계를 넘는 로스 기여를 생략하므로 전체 BPTT와 같은 그래디언트는 아니다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
기본 RNN에서 순환 웨이트의 그래디언트를 직접 기여와 미래 로스를 구분해 설명하라.

</div>

??? success "연습문제 2 풀이"
    $g_t=q_t+J_{t+1}^{\top}g_{t+1}$를 역순으로 계산하고 $\delta_t=g_t\odot(1-h_t^2)$를 구한다. 웨이트 기울기는 $\sum_t\delta_t h_{t-1}^{\top}$다. 여기서 각 항은 해당 시점에서 웨이트를 사용한 직접 기여이고, 미래 로스의 영향은 이미 $g_t$에 포함되어 있다. 상태의 전체 미분을 다시 합하면 중복 계산할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
길이 T의 전체 BPTT와 길이 k의 청크 처리에서 중간 상태 저장량을 비교하라.

</div>

??? success "연습문제 3 풀이"
    배치 크기와 다른 차원을 고정하면 은닉 상태 저장량은 각각 $O(TH)$와 $O(kH)$다. 청크 처리도 전체 시퀀스 길이 T를 줄이는 것은 아니며 역전파를 연결하는 구간을 줄인다. 체크포인팅은 중간값을 다시 계산하여 다른 메모리·계산 절충을 만든다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 코드가 자동 미분을 수행하는 단계와 실제로 검증하지 않는 항목을 설명하라. 그래디언트의 정확성을 수치적으로 확인하려면 무엇이 추가로 필요한가?

</div>

??? success "연습문제 4 풀이"
    ```python
    # PyTorch는 동적 그래프로 BPTT를 알아서 처리한다
    rnn = nn.RNN(input_size, hidden_size, batch_first=True)
    loss = criterion(rnn(x)[0], targets)
    loss.backward()  # 모든 시각을 지나는 온전한 BPTT
    # 잘라 낸 BPTT: k걸음마다 숨은 상태를 떼어 낸다
    ```

    아래 코드는 backward 호출만 보여 주며 수치 검증은 수행하지 않는다. 정의된 입력과 로스를 사용하고, 작은 배정밀도 텐서에서 자동 미분 결과를 중앙 유한 차분이나 gradcheck와 비교해야 한다. 각 사용 위치가 공유 파라메터 하나에 기여하도록 구성하고 로스의 합·평균 배율도 맞춘다.

## 정리하며

BPTT는 펼친 RNN의 각 사용 위치에서 공유 웨이트로 오는 기여를 합한다. 야코비안 곱의 순서와 그래디언트 벡터의 방향을 일관되게 써야 한다. 절단 BPTT는 상태 값과 역전파 범위를 분리하는 방법이며, 예제의 청크 길이와 반환 로스에도 주의가 필요하다.
