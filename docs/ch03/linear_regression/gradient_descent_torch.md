# PyTorch로 만드는 경사 하강법

이 스크립트는 `nn.Module`도 autograd도 쓰지 않고 PyTorch 텐서로 경사를 직접 계산하며 미니배치 경사 하강법을 구현한다. 경사 공식 $g = \frac{2}{B} X^T(X w + b - y)$을 텐서 연산으로 곧바로 계산하고 배치에는 `TensorDataset` / `DataLoader`를 씀으로써, 순수 NumPy와 완전한 PyTorch 추상화 사이의 정확한 중간 지점을 보여준다.

## 코드

```python
"""
Gradient Descent for Linear Regression — PyTorch (Manual)
==========================================================

Manual gradient computation with PyTorch tensors and DataLoader.
No nn.Module, no autograd — pure tensor operations.

Demonstrates:
- TensorDataset / DataLoader for batching
- Manual gradient: g = (2/B) X^T (Xw + b - y)
- In-place parameter updates
- Convergence tracking

Author: Deep Learning Foundations Curriculum
"""

import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ============================================================================
# 설정
# ============================================================================

parser = argparse.ArgumentParser(description="manual PyTorch gradient descent")
parser.add_argument("--n-samples", type=int, default=500)
parser.add_argument("--n-features", type=int, default=3)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--seed", type=int, default=42)
ARGS = parser.parse_args()

torch.manual_seed(ARGS.seed)

# ============================================================================
# 합성 데이터
# ============================================================================

n, p = ARGS.n_samples, ARGS.n_features
X = torch.randn(n, p)
w_true = torch.randn(p)
b_true = 3.0
y = X @ w_true + b_true + 0.3 * torch.randn(n)

# 분할
n_train = int(0.8 * n)
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=ARGS.batch_size, shuffle=True)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"True w: {w_true.numpy()}")
print(f"True b: {b_true}")

# ============================================================================
# 직접 만든 학습 루프
# ============================================================================

w = torch.zeros(p)
b = torch.zeros(1)

history = []
for epoch in range(ARGS.epochs):
    epoch_loss = 0.0
    for X_batch, y_batch in loader:
        # 순전파
        y_pred = X_batch @ w + b
        residual = y_pred - y_batch
        loss = (residual ** 2).mean()

        # 직접 계산한 경사
        B = len(y_batch)
        grad_w = (2.0 / B) * (X_batch.T @ residual)
        grad_b = (2.0 / B) * residual.sum()

        # 갱신
        w -= ARGS.lr * grad_w
        b -= ARGS.lr * grad_b

        epoch_loss += loss.item() * B

    avg_loss = epoch_loss / n_train
    history.append(avg_loss)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d}  MSE = {avg_loss:.6f}")

# ============================================================================
# 평가
# ============================================================================

with torch.no_grad():
    y_pred_test = X_test @ w + b
    test_mse = ((y_pred_test - y_test) ** 2).mean().item()
    ss_res = ((y_test - y_pred_test) ** 2).sum().item()
    ss_tot = ((y_test - y_test.mean()) ** 2).sum().item()
    test_r2 = 1.0 - ss_res / ss_tot

print(f"\nTest MSE: {test_mse:.6f}")
print(f"Test R²:  {test_r2:.6f}")
print(f"\nLearned w: {w.numpy()}")
print(f"True    w: {w_true.numpy()}")
print(f"Learned b: {b.item():.4f}  (true: {b_true})")

# ============================================================================
# 수렴 그림
# ============================================================================

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(history, lw=1.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE")
ax.set_title("Manual PyTorch GD — Training Loss")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("gradient_descent_torch.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: gradient_descent_torch.png")


if __name__ == "__main__":
    pass
```

## 논의

직접 계산하는 경사는 NumPy 버전과 같은 벡터화된 공식을 쓰되 PyTorch 텐서 연산으로 표현한다. `grad_w = (2.0 / B) * (X_batch.T @ residual)`과 `grad_b = (2.0 / B) * residual.sum()`이다. 이 연산들은 NumPy와 같은 결과를 내면서도 `.to('cuda')` 호출 한 번으로 GPU에서 실행할 수 있다. autograd 그래프를 만들지 않으므로 매개변수 갱신 `w -= lr * grad_w`은 단순한 제자리 뺄셈이다.

`TensorDataset`과 `DataLoader`의 조합이 배치 묶기와 뒤섞기의 모든 장부 관리를 처리한다. `TensorDataset`은 특징 텐서와 목표 텐서를 감싸서 `dataset[i]`가 $i$번째 `(x, y)` 쌍을 반환하게 하고, `DataLoader`는 이들을 지정한 크기의 배치로 묶으며 매 에폭마다 순서를 무작위로 뒤섞는다. 이 기반 구조는 경사를 직접 계산하든, autograd를 쓰든, 완전한 `nn.Module`을 쓰든 똑같다. 바뀌는 것은 안쪽 반복문의 본문뿐이다.

학습된 매개변수를 참된 데이터 생성 가중치와 비교하면 직접 계산한 경사가 옳은지 확인할 수 있다. 따로 떼어 둔 시험 데이터의 $R^2$ 점수가 추가 점검이 된다. 잡음이 없는 이 합성 데이터라면 $R^2$이 1.0에 매우 가까워야 한다. 경사가 틀렸다면 매개변수가 수렴하지 않고 $R^2$이 낮게 머무를 것이다. 이런 종류의 온전성 점검은 새로운 손실 함수에 대해 경사 계산을 직접 구현할 때 대단히 유용하다.

## 연습문제

**익힘 1.**
`w`와 `b`에 `requires_grad=True`를 붙여 직접 계산하던 경사를 `loss.backward()`로 대체하라. 결과가 동일한지 확인하라.

??? success "익힘 1 풀이"
    ```python
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    
    torch.manual_seed(42)
    n, p = 500, 3
    X = torch.randn(n, p)
    w_true = torch.randn(p)
    y = X @ w_true + 3.0 + 0.3 * torch.randn(n)
    
    w = torch.zeros(p, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    
    loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
    for epoch in range(100):
        for Xb, yb in loader:
            y_pred = Xb @ w + b
            loss = ((y_pred - yb) ** 2).mean()
            loss.backward()
            with torch.no_grad():
                w -= 0.01 * w.grad
                b -= 0.01 * b.grad
            w.grad.zero_()
            b.grad.zero_()
    
    print(f'Learned w: {w.detach().numpy()}')
    print(f'True    w: {w_true.numpy()}')
    ```

---

**익힘 2.**
경사를 직접 계산하는 학습 루프에서 경사를 0으로 만드는 것을 잊으면 어떻게 되는가? 경사 초기화를 주석 처리하여 흉내 내 보고 학습된 매개변수에 미치는 영향을 관찰하라.

??? success "익힘 2 풀이"
    초기화하지 않으면 이전 배치의 경사가 지워지지 않는다. 다만 경사 변수 `grad_w`와 `grad_b`는 반복마다 다시 계산되는 (누적되는 `.grad` 속성이 아니라 지역 변수인) 값이므로, 직접 계산하는 이 루프는 이 문제의 영향을 받지 않는다. 그러나 `.grad` 속성을 쓰는 autograd에서는 0으로 만드는 것을 잊으면 경사가 배치에 걸쳐 누적되어 실효 경사가 배치마다 커지고 매개변수가 빠르게 발산한다.

---

**익힘 3.**
모든 텐서를 `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`로 옮겨 GPU를 지원하게 하고, 큰 데이터셋(표본 100000개, 특징 100개)에서 속도 향상을 측정하라.

??? success "익힘 3 풀이"
    ```python
    import torch, time
    from torch.utils.data import TensorDataset, DataLoader
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using: {device}')
    
    torch.manual_seed(42)
    n, p = 100000, 100
    X = torch.randn(n, p, device=device)
    w_true = torch.randn(p, device=device)
    y = X @ w_true + 3.0 + 0.3 * torch.randn(n, device=device)
    
    w = torch.zeros(p, device=device)
    b = torch.zeros(1, device=device)
    
    loader = DataLoader(TensorDataset(X.cpu(), y.cpu()), batch_size=256, shuffle=True)
    start = time.time()
    for epoch in range(10):
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            y_pred = Xb @ w + b
            residual = y_pred - yb
            B = len(yb)
            w -= 0.001 * (2.0/B) * (Xb.T @ residual)
            b -= 0.001 * (2.0/B) * residual.sum()
    print(f'Time: {time.time()-start:.2f}s')
    # n과 p가 크면 GPU가 상당한 속도 향상을 준다.
    ```
