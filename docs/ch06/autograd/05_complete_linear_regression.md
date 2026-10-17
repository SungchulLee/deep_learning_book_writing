# 선형 회귀 직접 학습하기

잡음이 섞인 선형 데이터를 만들고, 텐서 파라미터를 직접 업데이트해 회귀 모델을 학습한다. 로스 곡선과 적합한 직선을 함께 해석한다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
==============================================================================
첫스텝 튜토리얼 05: 온전한 선형 회귀 학습 루프
==============================================================================

학습 목표:
-------------------
1. 온전한 학습 루프를 밑바닥부터 짓는다
2. 여느 학습 돌이를 이해한다: 앞으로 → 로스 → 뒤로 → 고치기
3. 익힐 때 그래디언트를 제대로 다루는 법을 배운다
4. 학습이 나아가는 모습을 그림으로 본다
5. 자동 미분 없이 파라메터를 고치는 법을 이해한다

핵심 개념:
------------
- 학습 루프: 앞으로, 로스, 뒤로, 고치기(반복)
- 기울기 0으로 만들기: 역전파 앞마다 꼭 해야 한다
- torch.no_grad(): 파라메터를 고칠 때 쓰는 자리
- 로스 좇기: 모여드는지 지켜본다
- 인공 데이터: 깨끗한 시험 데이터를 만든다

참 세상에서의 쓰임:
-----------------------
이것이 모든 깊은 학습 학습 루프의 바탕이다!
이 단순한 보기를 이해하면 복잡한 뉴럴 넷트웍을 다룰 채비가 된다.

==============================================================================
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

# ========================================================================
# 메인
# ========================================================================


def generate_synthetic_data(n_samples=100, true_w=2.0, true_b=1.0, noise_std=0.2):
    """
    인공 선형 회귀 데이터를 만든다: y = w*x + b + 잡음
    
    Args:
        n_samples: 데이터 점의 수
        true_w: 참 그래디언트(웨이트)
        true_b: 참 절편(바이어스)
        noise_std: 가우스 잡음의 표준편차
    
    Returns:
        x, y: 입력 텐서와 타깃 텐서
    """
    x = torch.randn(n_samples, 1)  # Random inputs from N(0,1)
    noise = noise_std * torch.randn(n_samples, 1)  # Gaussian noise
    y = true_w * x + true_b + noise  # True relationship + noise
    return x, y


def main():
    """
    선형 회귀 모델을 밑바닥부터 익히는 온전한 보기.
    x에서 y을 예측하며 파라메터 w(웨이트)과 b(바이어스)을 배운다.
    """
    
    print("="*70)
    print("LINEAR REGRESSION TRAINING FROM SCRATCH")
    print("="*70)
    
    # 재현성을 위한 난수 시드 설정
    torch.manual_seed(42)
    np.random.seed(42)
    
    # ==========================
    # 1단계: 데이터 생성
    # ==========================
    print("\nSTEP 1: Generating Synthetic Data")
    print("-" * 40)
    
    # 우리가 학습하려는 참 파라메터
    TRUE_W = 2.0  # True slope
    TRUE_B = 1.0  # True intercept
    N_SAMPLES = 100
    
    x, y = generate_synthetic_data(N_SAMPLES, TRUE_W, TRUE_B, noise_std=0.2)
    
    print(f"Generated {N_SAMPLES} samples")
    print(f"True model: y = {TRUE_W}*x + {TRUE_B} + noise")
    print(f"x shape: {x.shape}, y shape: {y.shape}")
    print(f"x range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"y range: [{y.min():.2f}, {y.max():.2f}]")
    
    # ==========================
    # 2단계: 파라메터 초기화
    # ==========================
    print("\nSTEP 2: Initializing Model Parameters")
    print("-" * 40)
    
    # 파라메터를 무작위로 초기화
    # 이들은 경사 추적이 켜진 잎 텐서이다
    w = torch.randn(1, requires_grad=True)  # Weight (slope)
    b = torch.zeros(1, requires_grad=True)  # Bias (intercept)
    
    print(f"Initial w: {w.item():.4f} (target: {TRUE_W})")
    print(f"Initial b: {b.item():.4f} (target: {TRUE_B})")
    
    # ==========================
    # 3단계: 하이퍼파라미터 설정
    # ==========================
    print("\nSTEP 3: Setting Hyperparameters")
    print("-" * 40)
    
    learning_rate = 0.1
    num_epochs = 200
    
    print(f"Learning rate: {learning_rate}")
    print(f"Number of epochs: {num_epochs}")
    
    # ==========================
    # 4단계: 학습 루프
    # ==========================
    print("\nSTEP 4: Training Loop")
    print("-" * 40)
    print("Starting training...\n")
    
    # 시각화를 위해 로스를 추적
    loss_history = []
    
    for epoch in range(num_epochs):
        # ========================================
        # 순전파: 예측을 계산한다
        # ========================================
        # 우리 모델: y_hat = w*x + b
        y_pred = x * w + b  # Broadcasting handles the shapes
        
        # 로스 계산: 평균제곱오차(MSE)
        # MSE = mean((예측 - 목표)^2)
        loss = torch.mean((y_pred - y) ** 2)
        
        # 그림을 그리기 위해 로스를 저장
        loss_history.append(loss.item())
        
        # ========================================
        # 역전파: 경사를 계산한다
        # ========================================
        # 매우 중요: 역전파 전에 경사를 0으로!
        # 경사는 기본적으로 누적되므로 반드시 지워야 한다
        if w.grad is not None:
            w.grad.zero_()
        if b.grad is not None:
            b.grad.zero_()
        
        # 역전파로 경사를 계산한다
        # 이것이 w.grad와 b.grad를 채운다
        loss.backward()
        
        # ========================================
        # 갱신 단계: 경사 하강법
        # ========================================
        # 갱신 규칙: θ_new = θ_old - learning_rate * gradient
        # torch.no_grad()를 쓰는 이유:
        # 1. 이 연산들이 계산 그래프에 들어가기를 원하지 않는다
        # 2. 더 빠르고 메모리를 덜 쓴다
        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad
        
        # ========================================
        # 기록: 진행 상황 출력
        # ========================================
        if epoch % 20 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:3d}/{num_epochs}: "
                  f"Loss = {loss.item():.6f} | "
                  f"w = {w.item():.4f} | "
                  f"b = {b.item():.4f}")
    
    # ==========================
    # 5단계: 최종 결과
    # ==========================
    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)
    
    print(f"\nTrue parameters:     w = {TRUE_W:.4f}, b = {TRUE_B:.4f}")
    print(f"Learned parameters:  w = {w.item():.4f}, b = {b.item():.4f}")
    print(f"Error in w: {abs(w.item() - TRUE_W):.4f}")
    print(f"Error in b: {abs(b.item() - TRUE_B):.4f}")
    print(f"Final loss: {loss_history[-1]:.6f}")
    
    # ==========================
    # 6단계: 시각화
    # ==========================
    print("\nGenerating visualization...")
    
    # 그림을 그리기 위해 numpy로 변환
    x_np = x.detach().numpy().flatten()
    y_np = y.detach().numpy().flatten()
    
    # 적합된 직선 생성
    x_sorted = np.sort(x_np)
    y_fitted = w.item() * x_sorted + b.item()
    y_true = TRUE_W * x_sorted + TRUE_B
    
    # 부분 그림 2개를 가진 도표를 만든다
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ----- 그림 1: 데이터와 적합된 직선 -----
    ax1 = axes[0]
    ax1.scatter(x_np, y_np, alpha=0.6, s=30, label='Training Data', color='blue')
    ax1.plot(x_sorted, y_fitted, 'r-', linewidth=3, 
             label=f'Learned: y={w.item():.2f}x+{b.item():.2f}')
    ax1.plot(x_sorted, y_true, 'g--', linewidth=2, 
             label=f'True: y={TRUE_W}x+{TRUE_B}')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Linear Regression: Data and Fitted Line', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ----- 그림 2: 학습 로스 곡선 -----
    ax2 = axes[1]
    ax2.plot(range(num_epochs), loss_history, linewidth=2, color='purple')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss (MSE)', fontsize=12)
    ax2.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale shows convergence better
    
    plt.tight_layout()
    plt.savefig('linear_regression_training.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'linear_regression_training.png'")
    
    # ==========================
    # 이해를 돕는 부분
    # ==========================
    print("\n" + "="*70)
    print("UNDERSTANDING THE TRAINING LOOP")
    print("="*70)
    print("""
    학습 돌이(에폭마다 반복한다):
    
    1. 순전파
       └─ 예측을 계산한다: y_pred = w*x + b
       └─ 로스를 계산한다: MSE = mean((y_pred - y)^2)
    
    2. 역전파
       └─ 묵은 그래디언트를 0으로 만든다: w.grad.zero_(), b.grad.zero_()
       └─ 새 그래디언트를 계산한다: loss.backward()
    
    3. 파라메터 고치기
       └─ 경사 하강법: w = w - lr * w.grad
       └─                   b = b - lr * b.grad
    
    고칠 때 torch.no_grad()을 쓰는 까닭은?
    • 파라메터 고치기는 셈 그래프에 들어가면 안 된다
    • 기억 자리와 셈의 덤을 아낀다
    • 고치기는 파라메터 값에 제자리로 하는 셈이다
    
    그래디언트를 0으로 만드는 까닭은?
    • PyTorch에서는 그래디언트가 기본으로 쌓인다
    • 새 그래디언트를 계산 앞에 묵은 것을 지워야 한다
    • 이를 잊으면 파라메터를 잘못 고치게 된다!
    
    러닝레이트:
    • 너무 크면 학습이 흔들리고 퍼져 나갈 수 있다
    • 너무 작으면 더디게 모여든다
    • 이 단순한 문제에는 0.1이 잘 듣는다
    """)
    
    print("="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("""
    ✓ 학습 루프의 짜임: 앞으로 → 로스 → 뒤로 → 고치기
    ✓ 역전파 앞에는 늘 그래디언트를 0으로 만들어라
    ✓ 파라메터를 고칠 때는 torch.no_grad()을 써라
    ✓ 로스를 좇아 모여드는지 지켜보아라
    ✓ 그림으로 보면 모델이 얼마나 좋은지 알기 쉽다
    ✓ 이 무늬는 모든 뉴럴 넷트웍 학습으로 넓혀진다!
    """)


if __name__ == "__main__":
    main()
```

??? note "전체 출력 (95줄)"

    ```
    ======================================================================
    LINEAR REGRESSION TRAINING FROM SCRATCH
    ======================================================================

    STEP 1: Generating Synthetic Data
    ----------------------------------------
    Generated 100 samples
    True model: y = 2.0*x + 1.0 + noise
    x shape: torch.Size([100, 1]), y shape: torch.Size([100, 1])
    x range: [-2.21, 2.22]
    y range: [-3.32, 5.21]

    STEP 2: Initializing Model Parameters
    ----------------------------------------
    Initial w: 0.3932 (target: 2.0)
    Initial b: 0.0000 (target: 1.0)

    STEP 3: Setting Hyperparameters
    ----------------------------------------
    Learning rate: 0.1
    Number of epochs: 200

    STEP 4: Training Loop
    ----------------------------------------
    Starting training...

    Epoch   0/200: Loss = 3.742756 | w = 0.7163 | b = 0.2207
    Epoch  20/200: Loss = 0.031612 | w = 1.9871 | b = 1.0025
    Epoch  40/200: Loss = 0.031214 | w = 2.0022 | b = 1.0071
    Epoch  60/200: Loss = 0.031214 | w = 2.0024 | b = 1.0071
    Epoch  80/200: Loss = 0.031214 | w = 2.0024 | b = 1.0071
    Epoch 100/200: Loss = 0.031214 | w = 2.0024 | b = 1.0071
    Epoch 120/200: Loss = 0.031214 | w = 2.0024 | b = 1.0071
    Epoch 140/200: Loss = 0.031214 | w = 2.0024 | b = 1.0071
    Epoch 160/200: Loss = 0.031214 | w = 2.0024 | b = 1.0071
    Epoch 180/200: Loss = 0.031214 | w = 2.0024 | b = 1.0071
    Epoch 199/200: Loss = 0.031214 | w = 2.0024 | b = 1.0071

    ======================================================================
    TRAINING COMPLETED!
    ======================================================================

    True parameters:     w = 2.0000, b = 1.0000
    Learned parameters:  w = 2.0024, b = 1.0071
    Error in w: 0.0024
    Error in b: 0.0071
    Final loss: 0.031214

    Generating visualization...
    Plot saved as 'linear_regression_training.png'

    ======================================================================
    UNDERSTANDING THE TRAINING LOOP
    ======================================================================

        학습 돌이(에폭마다 반복한다):
        
        1. 순전파
           └─ 예측을 계산한다: y_pred = w*x + b
           └─ 로스를 계산한다: MSE = mean((y_pred - y)^2)
        
        2. 역전파
           └─ 묵은 그래디언트를 0으로 만든다: w.grad.zero_(), b.grad.zero_()
           └─ 새 그래디언트를 계산한다: loss.backward()
        
        3. 파라메터 고치기
           └─ 경사 하강법: w = w - lr * w.grad
           └─                   b = b - lr * b.grad
        
        고칠 때 torch.no_grad()을 쓰는 까닭은?
        • 파라메터 고치기는 셈 그래프에 들어가면 안 된다
        • 기억 자리와 셈의 덤을 아낀다
        • 고치기는 파라메터 값에 제자리로 하는 셈이다
        
        그래디언트를 0으로 만드는 까닭은?
        • PyTorch에서는 그래디언트가 기본으로 쌓인다
        • 새 그래디언트를 계산 앞에 묵은 것을 지워야 한다
        • 이를 잊으면 파라메터를 잘못 고치게 된다!
        
        러닝레이트:
        • 너무 크면 학습이 흔들리고 퍼져 나갈 수 있다
        • 너무 작으면 더디게 모여든다
        • 이 단순한 문제에는 0.1이 잘 듣는다
        
    ======================================================================
    KEY TAKEAWAYS
    ======================================================================

        ✓ 학습 루프의 짜임: 앞으로 → 로스 → 뒤로 → 고치기
        ✓ 역전파 앞에는 늘 그래디언트를 0으로 만들어라
        ✓ 파라메터를 고칠 때는 torch.no_grad()을 써라
        ✓ 로스를 좇아 모여드는지 지켜보아라
        ✓ 그림으로 보면 모델이 얼마나 좋은지 알기 쉽다
        ✓ 이 무늬는 모든 뉴럴 넷트웍 학습으로 넓혀진다!
        
    ```


## 2. 논의

### 데이터 생성 규칙과 추정값

데이터는 $y=2x+1+\epsilon$으로 생성한다. 모델 $\hat y=wx+b$는 평균제곱오차를 줄이도록 학습한다. 유한한 표본과 잡음 때문에 학습된 값이 생성 규칙의 $w=2$, $b=1$과 정확히 일치할 필요는 없다. 로스 역시 잡음까지 완벽하게 제거해 0이 되는 것을 목표로 해석하면 안 된다.

### 전체 데이터를 사용하는 학습

각 에포크에서 같은 전체 데이터로 로스를 계산하고 한 번 업데이트한다. 이 예제에서는 에포크 수와 업데이트 횟수가 같지만, 미니배치 학습에서는 한 에포크에 여러 업데이트가 이루어진다. 매번 새 순전파를 수행하고 경사를 초기화하므로 `retain_graph=True`는 필요하지 않다.

### 그래프 읽기

산점도와 회귀선은 데이터의 추세를 얼마나 잘 따라가는지 보여준다. 로스 그래프의 로그 축에서는 같은 세로 간격이 같은 비율의 변화를 나타낸다. 학습 로스가 줄었다는 사실만으로 새 데이터에 대한 성능을 판단할 수는 없다. 또한 마지막에 저장한 로스는 마지막 업데이트 전 값이고, 그린 회귀선은 업데이트 후 파라미터를 사용한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
함수 $f(x) = x^3 - 2x^2 + x$를 생각하자. PyTorch autograd를 사용하여 $f'(3)$을 계산하라.

</div>

??? success "풀이"
    ```python
    import torch

    x = torch.tensor(3.0, requires_grad=True)
    f = x**3 - 2*x**2 + x
    f.backward()
    print(x.grad)  # f'(x) = 3x^2 - 4x + 1 = 27 - 12 + 1 = 16.0
    ```

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
잡음이 있는 선형 데이터를 학습할 때 로스가 0이 되거나 학습한 계수가 생성 규칙과 정확히 같아야 하는가?

</div>

??? success "풀이"
    그럴 필요는 없다. 유한 표본의 잡음 때문에 표본 MSE를 최소화하는 계수는 생성 규칙의 계수와 다를 수 있다. 잔차와 별도 데이터의 성능을 함께 살펴야 하며 학습 로스 0을 수렴의 필수 조건으로 삼지 않는다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
잎 텐서 `w`를 만들고, 반복마다 같은 식으로 로스를 새로 계산하라. 경사를 초기화하지 않고 `.backward()`를 세 번 호출하며 매번 `w.grad`를 출력하라. 관찰된 값을 설명하라.

</div>

??? success "풀이"
    ```python
    import torch

    w = torch.tensor(2.0, requires_grad=True)
    for i in range(3):
        loss = (w ** 2).sum()
        loss.backward()
        print(f'After backward {i+1}: w.grad = {w.grad}')
    # 출력: 4.0, 8.0, 12.0
    # 경사가 누적된다. 매 backward가 기존 경사에 2*w = 4.0을 더한다.
    ```

## 정리하며

직접 구현한 선형 회귀에서도 학습의 핵심 순서는 같다. 생성 규칙과 추정값의 차이, 학습 로스와 일반화 성능의 차이를 구분해 결과를 읽는다.
