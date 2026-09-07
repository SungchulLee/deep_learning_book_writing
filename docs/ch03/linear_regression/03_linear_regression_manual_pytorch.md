# PyTorch로 직접 계산하는 선형 회귀

이 튜토리얼은 PyTorch 텐서로 선형 회귀를 구현하되 경사는 여전히 직접 계산함으로써 순수 NumPy와 완전한 PyTorch 사이를 잇는다. 이 중간 단계에서 작업해 보면 텐서 연산이 NumPy의 대응물과 어떻게 이어지는지 분명해지고, 다음 단계에서 autograd가 무엇을 자동화해 줄지가 뚜렷해진다. 코드 구조가 NumPy 버전과 동일하므로 비교하기가 쉽다.

## 코드

```python
"""
==============================================================================
03_linear_regression_manual_pytorch.py
==============================================================================
어려움: ⭐⭐ (첫걸음~가운데)

DESCRIPTION:
    PyTorch 텐서를 쓰되 기울기는 손수 셈하는 선형 회귀.
    넘파이와 온전한 PyTorch 사이를 잇는다. 텐서가 어떻게 도는지 보이면서도
    기울기 셈은 우리가 다스린다.

다루는 것:
    - 넘파이 코드를 PyTorch 텐서로 옮기기
    - 텐서로 손수 기울기 셈하기
    - 텐서 셈 이해하기
    - 넘파이와 PyTorch의 길 견주기

PREREQUISITES:
    - 익힘 01(PyTorch 기초)
    - 익힘 02(넘파이로 하는 선형 회귀)

배움 목표:
    - 셈에 PyTorch 텐서를 쓴다
    - 기울기를 손수 셈한다(아직 autograd는 쓰지 않는다)
    - 텐서 셈과 넘파이 셈의 다름을 이해한다
    - autograd가 무엇을 대신해 줄지 헤아린다

걸리는 때: 20분쯤
==============================================================================
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("LINEAR REGRESSION WITH PYTORCH TENSORS (MANUAL GRADIENTS)")
print("=" * 70)

# ============================================================================
# 1부: 합성 데이터 생성
# ============================================================================
print("\n" + "=" * 70)
print("PART 1: GENERATE SYNTHETIC DATA")
print("=" * 70)

# 재현성을 위한 난수 시드 설정
torch.manual_seed(42)
np.random.seed(42)

# 참된 매개변수
TRUE_W = 2.0
TRUE_B = 1.0

# 먼저 NumPy로 데이터를 생성한다
n_samples = 100
X_numpy = np.random.uniform(-10, 10, n_samples)
noise = np.random.normal(0, 2, n_samples)
y_numpy = TRUE_W * X_numpy + TRUE_B + noise

# PyTorch 텐서로 변환
# 대부분의 PyTorch 연산에서 dtype=torch.float32가 표준이다
X = torch.from_numpy(X_numpy).float()  # Convert to float32 tensor
y = torch.from_numpy(y_numpy).float()

print(f"Generated {n_samples} data points")
print(f"True parameters: w={TRUE_W}, b={TRUE_B}")
print(f"\nData types:")
print(f"  X: {X.dtype}, shape: {X.shape}")
print(f"  y: {y.dtype}, shape: {y.shape}")
print(f"\nFirst 5 samples:")
print(f"X[:5] = {X[:5]}")
print(f"y[:5] = {y[:5]}")

# ============================================================================
# 2부: 모델 정의
# ============================================================================
print("\n" + "=" * 70)
print("PART 2: DEFINE THE MODEL")
print("=" * 70)

def predict(X, w, b):
    """
    선형 모형: y_pred = w * X + b
    
    Args:
        X: 들임 특징(텐서)
        w: 무게 매개변수(텐서)
        b: 치우침 매개변수(텐서)
    
    Returns:
        y_pred: 예측(텐서)
    """
    return w * X + b

def compute_loss(y_true, y_pred):
    """
    평균 제곱 어긋남(MSE) 잃음
    
    Args:
        y_true: 참 값(텐서)
        y_pred: 예측(텐서)
    
    Returns:
        loss: MSE 값(홑값 텐서)
    """
    n = y_true.shape[0]  # Number of samples
    loss = (1 / n) * torch.sum((y_true - y_pred) ** 2)
    return loss

# 매개변수를 텐서로 초기화한다
w = torch.tensor([0.0], dtype=torch.float32)
b = torch.tensor([0.0], dtype=torch.float32)

print(f"Initialized parameters:")
print(f"  w: {w}, shape: {w.shape}, dtype: {w.dtype}")
print(f"  b: {b}, shape: {b.shape}, dtype: {b.dtype}")

# 초기 예측을 시험한다
y_pred_init = predict(X, w, b)
initial_loss = compute_loss(y, y_pred_init)

print(f"\nInitial predictions:")
print(f"  y_pred[:5]: {y_pred_init[:5]}")
print(f"  Initial loss: {initial_loss.item():.4f}")

# ============================================================================
# 3부: 경사 직접 계산하기
# ============================================================================
print("\n" + "=" * 70)
print("PART 3: MANUAL GRADIENT COMPUTATION")
print("=" * 70)

def compute_gradients(X, y_true, y_pred):
    """
    w과 b에 대한 MSE의 기울기를 손수 셈한다
    
    넘파이 갈래와 똑같되 PyTorch 텐서를 쓴다
    
    ∂Loss/∂w = (2/n) * sum((y_pred - y_true) * X)
    ∂Loss/∂b = (2/n) * sum(y_pred - y_true)
    
    Args:
        X: 들임 특징(텐서)
        y_true: 참 값(텐서)
        y_pred: 예측(텐서)
    
    Returns:
        grad_w: 무게에 대한 기울기(텐서)
        grad_b: 치우침에 대한 기울기(텐서)
    """
    n = X.shape[0]
    error = y_pred - y_true
    
    # PyTorch 연산으로 경사를 계산한다
    grad_w = (2.0 / n) * torch.sum(error * X)
    grad_b = (2.0 / n) * torch.sum(error)
    
    return grad_w, grad_b

# 경사 계산을 시험한다
grad_w, grad_b = compute_gradients(X, y, y_pred_init)
print(f"Initial gradients:")
print(f"  grad_w: {grad_w.item():.4f}")
print(f"  grad_b: {grad_b.item():.4f}")

# ============================================================================
# 4부: 학습 루프
# ============================================================================
print("\n" + "=" * 70)
print("PART 4: TRAINING LOOP")
print("=" * 70)

# 매개변수를 초기화한다
w = torch.tensor([0.0], dtype=torch.float32)
b = torch.tensor([0.0], dtype=torch.float32)

# 초매개변수
learning_rate = 0.01
n_epochs = 100

# 추적
loss_history = []
w_history = [w.item()]
b_history = [b.item()]

print(f"Training Configuration:")
print(f"  Learning rate: {learning_rate}")
print(f"  Number of epochs: {n_epochs}")
print(f"\n{'Epoch':<8} {'Loss':<12} {'w':<12} {'b':<12}")
print("-" * 50)

for epoch in range(n_epochs):
    # 1. 순전파
    y_pred = predict(X, w, b)
    
    # 2. 손실 계산
    loss = compute_loss(y, y_pred)
    loss_history.append(loss.item())  # .item() converts tensor to Python number
    
    # 3. 경사를 직접 계산한다
    grad_w, grad_b = compute_gradients(X, y, y_pred)
    
    # 4. 매개변수를 갱신한다
    # 참고: 혹시 있을 계산 그래프에서 떼어내야 한다
    # 아직 autograd를 쓰지 않지만 좋은 습관이다
    w = w - learning_rate * grad_w
    b = b - learning_rate * grad_b
    
    # 이력 저장
    w_history.append(w.item())
    b_history.append(b.item())
    
    # 진행 상황 출력
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"{epoch+1:<8} {loss.item():<12.4f} {w.item():<12.4f} {b.item():<12.4f}")

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)
print(f"\nFinal Results:")
print(f"  Learned w: {w.item():.4f} (True: {TRUE_W})")
print(f"  Learned b: {b.item():.4f} (True: {TRUE_B})")
print(f"  Final loss: {loss_history[-1]:.4f}")
print(f"  Initial loss: {loss_history[0]:.4f}")

# ============================================================================
# 5부: 비교 - PYTORCH 연산과 NUMPY 연산
# ============================================================================
print("\n" + "=" * 70)
print("PART 5: PYTORCH OPERATIONS - KEY DIFFERENCES FROM NUMPY")
print("=" * 70)

print("""
PyTorch 텐서와 넘파이 배열 견주기:

Similarities:
1. 낯이 비슷하다: 셈의 이름이 대개 같다
2. 수학 셈이 같다
3. 자리 잡기와 자르기가 비슷하게 움직인다

고갱이 다름:

1. 기기 받침:
   - PyTorch: CPU에서도 GPU에서도 돈다
   - 넘파이: CPU에서만 돈다
   Example:
     tensor_cpu = torch.tensor([1, 2, 3])
     tensor_gpu = tensor_cpu.to('cuda')  # 쓸 수 있으면 GPU으로 옮긴다

2. 자동 미분:
   - PyTorch: 붙박이 autograd(다음 익힘에서 쓴다)
   - 넘파이: 기울기를 손수 셈해야 한다
   
3. 자료 갈래:
   - PyTorch: 기본 실수가 float32이다
   - 넘파이: 기본 실수가 float64이다
   
4. 제자리 셈:
   - PyTorch: _으로 끝나는 셈이 제자리 셈이다
   - 넘파이: 대개 드러내 놓고 대입해야 한다
   Example:
     x.add_(1)  # 제자리에서 1을 더한다(PyTorch)
     x = x + 1  # 새 배열을 만든다(넘파이)

5. 값 뽑아내기:
   - PyTorch: 파이썬 홑값을 얻으려면 .item()을 쓴다
   - 넘파이: 곧바로 꺼내거나 .item()을 쓴다
   Example:
     scalar = tensor.item()  # PyTorch
     scalar = array[0]       # 넘파이
""")

# ============================================================================
# 6부: 결과 시각화
# ============================================================================
print("\n" + "=" * 70)
print("PART 6: VISUALIZE RESULTS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 그림 1: 손실 곡선
axes[0, 0].plot(loss_history, linewidth=2, color='blue')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss (MSE)')
axes[0, 0].set_title('Training Loss (PyTorch Implementation)')
axes[0, 0].grid(True, alpha=0.3)

# 그림 2: 매개변수의 변화
axes[0, 1].plot(w_history, label='w (slope)', linewidth=2)
axes[0, 1].axhline(y=TRUE_W, color='r', linestyle='--', label=f'True w={TRUE_W}')
axes[0, 1].plot(b_history, label='b (intercept)', linewidth=2)
axes[0, 1].axhline(y=TRUE_B, color='g', linestyle='--', label=f'True b={TRUE_B}')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Parameter Value')
axes[0, 1].set_title('Parameter Convergence')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 그림 3: 최종 적합
X_sorted, indices = torch.sort(X)
y_sorted = y[indices]
y_pred_final = predict(X_sorted, w, b)

# 그림을 그리기 위해 numpy로 변환
X_sorted_np = X_sorted.numpy()
y_sorted_np = y_sorted.numpy()
y_pred_final_np = y_pred_final.detach().numpy()

axes[1, 0].scatter(X.numpy(), y.numpy(), alpha=0.5, label='Data')
axes[1, 0].plot(X_sorted_np, TRUE_W * X_sorted_np + TRUE_B, 'r-', 
                linewidth=2, label=f'True: y={TRUE_W}x+{TRUE_B}')
axes[1, 0].plot(X_sorted_np, y_pred_final_np, 'g-', 
                linewidth=2, label=f'Learned: y={w.item():.2f}x+{b.item():.2f}')
axes[1, 0].set_xlabel('X')
axes[1, 0].set_ylabel('y')
axes[1, 0].set_title('Data with Learned Model')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 그림 4: 학습률 민감도
axes[1, 1].text(0.5, 0.6, 'Key Insights:', ha='center', fontsize=14, weight='bold')
axes[1, 1].text(0.5, 0.5, f'Final w: {w.item():.4f} (error: {abs(w.item()-TRUE_W):.4f})', 
                ha='center', fontsize=11)
axes[1, 1].text(0.5, 0.4, f'Final b: {b.item():.4f} (error: {abs(b.item()-TRUE_B):.4f})', 
                ha='center', fontsize=11)
axes[1, 1].text(0.5, 0.3, f'Loss reduction: {((loss_history[0]-loss_history[-1])/loss_history[0]*100):.1f}%', 
                ha='center', fontsize=11)
axes[1, 1].text(0.5, 0.1, 'Next: Autograd will compute\ngradients automatically!', 
                ha='center', fontsize=10, style='italic')
axes[1, 1].set_xlim(0, 1)
axes[1, 1].set_ylim(0, 1)
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('/home/claude/pytorch_linear_regression_tutorial/03_pytorch_manual.png', dpi=100)
print("Saved visualization to: 03_pytorch_manual.png")
plt.show()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
배운 것:
1. PyTorch 텐서는 넘파이 배열과 비슷하되 덧붙은 결이 있다
2. 같은 수학 셈이 텐서에도 통한다
3. 기울기 셈은 아직 손수 한다(지금까지는!)
4. 코드 짜임은 넘파이 갈래와 똑같다

PyTorch 텐서의 이점:
✓ GPU 빠르게 하기(여기서는 안 썼지만 쓸 수 있다)
✓ 자동 미분(다음 익힘에서 다룬다!)
✓ 신경망을 위한 더 큰 생태계의 한 조각이다
✓ 잘 다듬은 C++/CUDA 뒷단

이 길을 고른 까닭:
- 수식을 이해하려고 기울기를 손수 셈했다
- 다음 익힘에서는 autograd가 대신해 준다
- 손수 셈해 보면 모형의 탈을 잡는 데 도움이 된다
- PyTorch가 밑바탕 수식을 바꾸지 않음을 보여 준다

다음 걸음:
- 익힘 04: autograd로 기울기를 절로 셈한다
- 이제 손수 기울기 식을 쓸 일이 없다!
- 모형 얼개와 익힘에 마음을 둔다
""")


if __name__ == "__main__":
    pass
```

## 논의

NumPy로 만든 선형 회귀 구현을 PyTorch 텐서로 옮기는 데는 놀랄 만큼 적은 수정만 필요하다. 원소별 곱, 합, 스칼라 나눗셈 같은 수학 연산의 문법이 거의 같다. 주된 차이는 겉모습에 있다. 텐서는 파이썬 스칼라를 얻으려면 `.item()`을, 차원을 얻으려면 `len()` 대신 `.shape[0]`을, 형 변환에는 `.float()`이나 `torch.from_numpy(...).float()`을 쓴다. 이런 유사성은 의도된 것이다. PyTorch는 NumPy 사용자에게 자연스럽게 느껴지도록 만들어졌다.

여기서 autograd를 쓰지 않는데도 이 코드는 이미 PyTorch의 기반 시설에서 이득을 본다. 텐서는 `.to('cuda')` 호출 한 번으로 GPU로 옮길 수 있으며, 곧바로 행렬 연산이 수천 개의 코어에 걸쳐 병렬화된다. 경사 공식의 계산 비용은 데이터셋 크기에 선형으로 비례하므로, 큰 데이터셋에서는 autograd를 활용하기 전이라도 GPU 가속만으로 상당한 속도 향상을 얻을 수 있다.

경사를 직접 계산하는 것은 중요한 교육적 목적을 가진다. 공식이 수학적 유도와 일치하는지 스스로 확인하게 만들기 때문이다. 사용자 정의 학습 루프에서 버그가 생기는 흔한 원인이 잘못된 경사 식인데, 한 번이라도 직접 구현해 보면 그런 오류를 알아채는 직관이 생긴다. 다음 튜토리얼에서는 autograd가 미분 가능한 모든 계산에 대해 정확한 경사를 자동으로 계산해 주므로 이런 실수의 원인이 완전히 사라진다.

## 연습문제

**익힘 1.**
`w`와 `b`를 하나의 매개변수 벡터 $\theta = [w, b]^T$로 저장하고 입력에 1로 채운 열을 덧붙이도록 학습 루프를 다시 작성하라. 결과가 원래의 두 매개변수 버전과 일치함을 확인하라.

??? success "익힘 1 풀이"
    ```python
    import torch
    import numpy as np
    
    np.random.seed(42)
    n = 100
    X_np = np.random.uniform(-10, 10, n)
    noise = np.random.normal(0, 2, n)
    y_np = 2.0 * X_np + 1.0 + noise
    
    # 확장된 입력: [X, 1]
    X_aug = torch.from_numpy(np.column_stack([X_np, np.ones(n)])).float()
    y = torch.from_numpy(y_np).float()
    
    theta = torch.zeros(2)  # [w, b]
    lr = 0.01
    
    for epoch in range(100):
        y_pred = X_aug @ theta
        error = y_pred - y
        loss = (error ** 2).mean()
        grad = (2.0 / n) * (X_aug.T @ error)
        theta -= lr * grad
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}: w={theta[0].item():.4f}, b={theta[1].item():.4f}, loss={loss.item():.4f}')
    ```

---

**익힘 2.**
autograd를 쓸 때 매개변수 갱신 중에 `torch.no_grad()`를 쓰는 핵심 이점은 무엇이며, 경사를 직접 계산하는 이 튜토리얼에서는 왜 필요하지 않은가?

??? success "익힘 2 풀이"
    autograd를 쓸 때(`requires_grad=True`)는 추적되는 텐서에 대한 모든 산술 연산이 계산 그래프에 기록된다. `torch.no_grad()`가 없으면 매개변수 갱신 `w -= lr * grad_w`가 그래프의 일부로 기록되어 오류가 나거나 의도치 않게 메모리가 늘어난다. 경사를 직접 계산하는 이 튜토리얼에서는 텐서에 `requires_grad=True`가 없으므로 그래프가 만들어지지 않고 컨텍스트 관리자도 필요하지 않다. 다음 튜토리얼에서 autograd를 도입하면 이 구분이 사라진다.

---

**익힘 3.**
학습 루프에 검증 분할(80/20)을 추가하라. 매 에폭 후에 학습 손실과 검증 손실을 모두 계산하여 출력하라. 검증 손실이 학습 손실을 가깝게 따라가는가, 아니면 과적합의 징후가 보이는가?

??? success "익힘 3 풀이"
    ```python
    import torch, numpy as np
    np.random.seed(42)
    n = 100
    X_np = np.random.uniform(-10, 10, n)
    y_np = 2.0 * X_np + 1.0 + np.random.normal(0, 2, n)
    
    split = int(0.8 * n)
    X_train = torch.from_numpy(X_np[:split]).float()
    y_train = torch.from_numpy(y_np[:split]).float()
    X_val = torch.from_numpy(X_np[split:]).float()
    y_val = torch.from_numpy(y_np[split:]).float()
    
    w, b = torch.tensor([0.0]), torch.tensor([0.0])
    lr = 0.01
    
    for epoch in range(100):
        y_pred = w * X_train + b
        train_loss = ((y_train - y_pred) ** 2).mean()
        grad_w = (2.0 / len(X_train)) * ((y_pred - y_train) * X_train).sum()
        grad_b = (2.0 / len(X_train)) * (y_pred - y_train).sum()
        w = w - lr * grad_w
        b = b - lr * grad_b
        val_pred = w * X_val + b
        val_loss = ((y_val - val_pred) ** 2).mean()
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}: train_loss={train_loss.item():.4f}, val_loss={val_loss.item():.4f}')
    ```
    표본이 100개인 단순한 선형 모델에서는 매개변수가 둘뿐이고 과적합될 가능성이 낮으므로 검증 손실이 학습 손실을 가깝게 따라가야 한다.
