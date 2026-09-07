# 완전한 선형 회귀

이 스크립트는 완전한 선형 회귀을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
#!/usr/bin/env python3
"""
==============================================================================
첫걸음 익힘 05: 온전한 선형 회귀 익힘 되돌이
==============================================================================

배움 목표:
-------------------
1. 온전한 익힘 되돌이를 맨바닥부터 짓는다
2. 여느 익힘 돌이를 이해한다: 앞으로 → 잃음 → 뒤로 → 고치기
3. 익힐 때 기울기를 제대로 다루는 법을 배운다
4. 익힘이 나아가는 모습을 그림으로 본다
5. 자동 미분 없이 매개변수를 고치는 법을 이해한다

고갱이 개념:
------------
- 익힘 되돌이: 앞으로, 잃음, 뒤로, 고치기(되풀이)
- 기울기 0으로 만들기: 뒤로 걸음 앞마다 꼭 해야 한다
- torch.no_grad(): 매개변수를 고칠 때 쓰는 자리
- 잃음 좇기: 모여드는지 지켜본다
- 인공 자료: 깨끗한 시험 자료를 만든다

참 세상에서의 쓰임:
-----------------------
이것이 모든 깊은 배움 익힘 되돌이의 바탕이다!
이 단순한 보기를 이해하면 복잡한 신경망을 다룰 채비가 된다.

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
    인공 선형 회귀 자료를 만든다: y = w*x + b + 잡음
    
    Args:
        n_samples: 자료 점의 수
        true_w: 참 기울기(무게)
        true_b: 참 절편(치우침)
        noise_std: 가우스 잡음의 표준편차
    
    Returns:
        x, y: 들임 텐서와 과녁 텐서
    """
    x = torch.randn(n_samples, 1)  # Random inputs from N(0,1)
    noise = noise_std * torch.randn(n_samples, 1)  # Gaussian noise
    y = true_w * x + true_b + noise  # True relationship + noise
    return x, y


def main():
    """
    선형 회귀 모형을 맨바닥부터 익히는 온전한 보기.
    x에서 y을 예측하며 매개변수 w(무게)과 b(치우침)을 배운다.
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
    
    # 우리가 학습하려는 참 매개변수
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
    # 2단계: 매개변수 초기화
    # ==========================
    print("\nSTEP 2: Initializing Model Parameters")
    print("-" * 40)
    
    # 매개변수를 무작위로 초기화
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
    
    # 시각화를 위해 손실을 추적
    loss_history = []
    
    for epoch in range(num_epochs):
        # ========================================
        # 순전파: 예측을 계산한다
        # ========================================
        # 우리 모델: y_hat = w*x + b
        y_pred = x * w + b  # Broadcasting handles the shapes
        
        # 손실 계산: 평균제곱오차(MSE)
        # MSE = mean((예측 - 목표)^2)
        loss = torch.mean((y_pred - y) ** 2)
        
        # 그림을 그리기 위해 손실을 저장
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
    
    # ----- 그림 2: 학습 손실 곡선 -----
    ax2 = axes[1]
    ax2.plot(range(num_epochs), loss_history, linewidth=2, color='purple')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss (MSE)', fontsize=12)
    ax2.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale shows convergence better
    
    plt.tight_layout()
    plt.savefig('/home/claude/linear_regression_training.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'linear_regression_training.png'")
    
    # ==========================
    # 이해를 돕는 부분
    # ==========================
    print("\n" + "="*70)
    print("UNDERSTANDING THE TRAINING LOOP")
    print("="*70)
    print("""
    익힘 돌이(판마다 되풀이한다):
    
    1. 앞으로 걸음
       └─ 예측을 셈한다: y_pred = w*x + b
       └─ 잃음을 셈한다: MSE = mean((y_pred - y)^2)
    
    2. 뒤로 걸음
       └─ 묵은 기울기를 0으로 만든다: w.grad.zero_(), b.grad.zero_()
       └─ 새 기울기를 셈한다: loss.backward()
    
    3. 매개변수 고치기
       └─ 기울기 내림: w = w - lr * w.grad
       └─                   b = b - lr * b.grad
    
    고칠 때 torch.no_grad()을 쓰는 까닭은?
    • 매개변수 고치기는 셈 그래프에 들어가면 안 된다
    • 기억 자리와 셈의 덤을 아낀다
    • 고치기는 매개변수 값에 제자리로 하는 셈이다
    
    기울기를 0으로 만드는 까닭은?
    • PyTorch에서는 기울기가 기본으로 쌓인다
    • 새 기울기를 셈하기 앞에 묵은 것을 지워야 한다
    • 이를 잊으면 매개변수를 잘못 고치게 된다!
    
    배움 빠르기:
    • 너무 크면 익힘이 흔들리고 퍼져 나갈 수 있다
    • 너무 작으면 더디게 모여든다
    • 이 단순한 문제에는 0.1이 잘 듣는다
    """)
    
    print("="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("""
    ✓ 익힘 되돌이의 짜임: 앞으로 → 잃음 → 뒤로 → 고치기
    ✓ 뒤로 걸음 앞에는 늘 기울기를 0으로 만들어라
    ✓ 매개변수를 고칠 때는 torch.no_grad()을 써라
    ✓ 잃음을 좇아 모여드는지 지켜보아라
    ✓ 그림으로 보면 모형이 얼마나 좋은지 알기 쉽다
    ✓ 이 무늬는 모든 신경망 익힘으로 넓혀진다!
    """)


if __name__ == "__main__":
    main()```

## 논의

이 코드는 `requires_grad=True`인 텐서에 대한 연산을 자동으로 추적하는 PyTorch의 autograd 체계를 보여준다. 스칼라 손실에 `.backward()`를 호출하면 autograd가 계산 그래프를 역방향으로 훑으며 연쇄 법칙을 적용해 모든 잎 텐서의 경사를 계산한다. 이 구조가 PyTorch의 모든 신경망 학습을 떠받친다.

경사 추적을 제어하는 것은 정확성과 성능 모두에 필수적이다. `torch.no_grad()` 컨텍스트 관리자는 매개변수 갱신이나 추론처럼 계산 그래프에 포함되어서는 안 되는 연산에 대해 autograd를 끈다. `.detach()` 메서드는 저장소는 공유하지만 그래프와는 분리된 텐서를 만들며, 값을 기록하거나 NumPy로 변환할 때 유용하다.

브로드캐스팅은 작은 텐서를 가상으로 확장하여 모양이 다른 텐서 사이의 원소별 연산을 가능하게 한다. PyTorch는 차원을 오른쪽부터 맞추며, 각 차원 쌍이 서로 같거나, 둘 중 하나가 1이거나, 아예 없을 것을 요구한다. 이로써 데이터를 명시적으로 복제하지 않아도 되어 메모리 효율이 좋고 빠르다.

## 연습문제

**연습문제 1.**
함수 $f(x) = x^3 - 2x^2 + x$를 생각하자. PyTorch autograd를 사용하여 $f'(3)$을 계산하라.

??? success "연습문제 1 풀이"
    ```python
    import torch

    x = torch.tensor(3.0, requires_grad=True)
    f = x**3 - 2*x**2 + x
    f.backward()
    print(x.grad)  # f'(x) = 3x^2 - 4x + 1 = 27 - 12 + 1 = 16.0
    ```

---


**연습문제 2.**
`retain_graph=True` 없이 같은 계산 그래프에 `.backward()`를 두 번 호출하면 오류가 나는 이유를 설명하라. `retain_graph=True`는 메모리 사용량에 어떤 영향을 주는가?

??? success "연습문제 2 풀이"
    기본적으로 PyTorch는 메모리를 아끼기 위해 `.backward()` 후에 계산 그래프를 해제한다. `.backward()`를 두 번째로 호출하면 더 이상 존재하지 않는 그래프를 훑으려 하므로 `RuntimeError`가 발생한다. `retain_graph=True`로 두면 그래프가 메모리에 남아 재사용할 수 있지만, 모든 중간 텐서가 할당된 채로 남으므로 메모리 소비가 늘어난다.

---


**연습문제 3.**
잎 텐서 `w`를 만들고 손실을 계산한 뒤, 경사를 초기화하지 않고 `.backward()`를 세 번 호출하며 매번 `w.grad`를 출력하는 코드를 작성하라. 관찰된 값을 설명하라.

??? success "연습문제 3 풀이"
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
