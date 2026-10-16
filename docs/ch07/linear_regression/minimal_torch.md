# 최소한의 PyTorch

이 스크립트는 가능한 한 가장 간소한 PyTorch autograd 학습 루프를 보여준다. `requires_grad=True`인 날것의 파라메터 텐서, 손으로 쓴 순전파, 경사 계산을 위한 `loss.backward()`, `torch.no_grad()` 안에서의 직접 파라메터 갱신, 그리고 명시적인 경사 초기화이다. `nn.Module`도, 옵티마이저 객체도, DataLoader도 없다. 오직 선형 회귀에 적용된 자동 미분의 커널뿐이다.

## 1. 코드

```python
"""
가장 단출한 PyTorch 선형 회귀 — 자동 미분
==============================================

날 텐서에 requires_grad=True를 두고 loss.backward()으로 그래디언트를 절로
계산한다.  nn.Module도 최적화기 객체도 쓰지 않는다.

Demonstrates:
- requires_grad=True
- 자동 미분을 위한 loss.backward()
- 파라메터를 고칠 때 쓰는 torch.no_grad() 자리
- 쌓인 그래디언트를 지우는 .grad.zero_()
- 셈 그래프에서 값을 뽑아내는 .detach()

지은이: 깊은 학습 바탕 학습 차례
"""

import argparse
import torch
import matplotlib.pyplot as plt

# ============================================================================
# 설정
# ============================================================================

parser = argparse.ArgumentParser(description="minimal PyTorch autograd")
parser.add_argument("--n-samples", type=int, default=500)
parser.add_argument("--n-features", type=int, default=3)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--seed", type=int, default=42)
ARGS = parser.parse_args()

torch.manual_seed(ARGS.seed)

# ============================================================================
# 데이터
# ============================================================================

n, p = ARGS.n_samples, ARGS.n_features
X = torch.randn(n, p)
w_true = torch.randn(p)
b_true = 3.0
y = X @ w_true + b_true + 0.3 * torch.randn(n)

n_train = int(0.8 * n)
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

# ============================================================================
# 경사를 추적하는 파라메터
# ============================================================================

w = torch.zeros(p, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# ============================================================================
# 학습 루프
# ============================================================================

history = []
for epoch in range(ARGS.epochs):
    # 순전파 (전체 배치)
    y_pred = X_train @ w + b
    loss = ((y_pred - y_train) ** 2).mean()

    # 역전파 — autograd가 w.grad와 b.grad를 채운다
    loss.backward()

    # 파라메터 갱신 — autograd가 추적해서는 안 된다
    with torch.no_grad():
        w -= ARGS.lr * w.grad
        b -= ARGS.lr * b.grad

    # 매우 중요: 경사를 0으로 만든다 (PyTorch는 기본적으로 누적한다)
    w.grad.zero_()
    b.grad.zero_()

    history.append(loss.item())

    if (epoch + 1) % 40 == 0:
        print(f"Epoch {epoch+1:3d}  MSE = {loss.item():.6f}")

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
print(f"\nLearned w: {w.detach().numpy()}")
print(f"True    w: {w_true.numpy()}")
print(f"Learned b: {b.detach().item():.4f}  (true: {b_true})")

# ============================================================================
# 수렴 그림
# ============================================================================

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(history, lw=1.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE")
ax.set_title("Minimal PyTorch (Autograd) — Training Loss")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("minimal_torch.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: minimal_torch.png")


if __name__ == "__main__":
    pass
```

**출력:**

```
Epoch  40  MSE = 2.114635
Epoch  80  MSE = 0.507749
Epoch 120  MSE = 0.175647
Epoch 160  MSE = 0.106049
Epoch 200  MSE = 0.091250

Test MSE: 0.085761
Test R²:  0.834527

Learned w: [-0.1025678  -0.3052043  -0.69520104]
True    w: [-0.11067407 -0.29015508 -0.69889843]
Learned b: 2.9564  (true: 3.0)
Saved: minimal_torch.png
```

## 2. 논의

최소한의 autograd 루프는 학습 과정의 모든 단계를 겉으로 드러낸다. 파라메터 `w`와 `b`는 `requires_grad=True`인 평범한 텐서이다. 순전파 `y_pred = X_train @ w + b`가 계산 그래프를 만든다. 로스 `((y_pred - y_train) ** 2).mean()`이 그 그래프를 잇는다. `loss.backward()`를 호출하면 그래프를 역방향으로 훑으며 각 파라메터에 대한 로스의 정확한 경사로 `w.grad`와 `b.grad`를 채운다.

파라메터 갱신은 두 가지 이유로 `torch.no_grad()` 블록 안에서 이루어져야 한다. 첫째, 뺄셈 `w -= lr * w.grad`는 잎 텐서에 대한 제자리 연산이라 경사 추적이 켜져 있으면 오류가 난다. 둘째, 갱신은 미분 대상 함수의 일부가 아니므로 기록하면 메모리를 낭비한다. 갱신 후에는 `w.grad.zero_()`와 `b.grad.zero_()`로 누적된 경사를 지워 다음 반복을 준비한다.

모델을 평가할 때는 추론 중에 역전파가 필요 없으므로 `torch.no_grad()`로 경사 추적을 완전히 끈다. $R^2$ 점수는 잔차제곱합과 총제곱합으로부터 계산한다. 학습된 웨이트를 참된 데이터 생성 파라메터와 비교하여 수렴을 확인한다. `.numpy()`는 경사를 추적하지 않는 텐서를 요구하므로, 출력하기 위해 계산 그래프에서 파라메터 값을 꺼낼 때 `.detach()` 메서드를 쓴다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
파라메터 갱신 단계에 `torch.optim.SGD`를 쓰도록 최소 루프를 다시 작성하라. 어떤 줄이 바뀌고 어떤 줄이 그대로인가?

</div>

??? success "연습문제 1 풀이"
    ```python
    import torch
    torch.manual_seed(42)
    n, p = 500, 3
    X = torch.randn(n, p)
    w_true = torch.randn(p)
    y = X @ w_true + 3.0 + 0.3 * torch.randn(n)
    
    w = torch.zeros(p, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.SGD([w, b], lr=0.01)  # NEW: optimizer
    
    for epoch in range(200):
        y_pred = X @ w + b                        # SAME: forward pass
        loss = ((y_pred - y) ** 2).mean()          # SAME: loss
        loss.backward()                             # SAME: backward
        optimizer.step()                            # CHANGED: replaces manual update
        optimizer.zero_grad()                       # CHANGED: replaces manual zero
    ```
    순전파, 로스 계산, 역전파는 그대로다. 갱신과 경사 초기화만 옵티마이저 메서드로 바뀐다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
선형 회귀의 MSE가 볼록하다는 사실만으로 최적 파라미터가 유일하고 초기화와 무관하다고 말할 수 있는가?

</div>

??? success "연습문제 2 풀이"
    아니다. 바이어스 열을 포함한 설계 행렬이 열 전체 랭크여야 최소제곱 파라미터가 유일하다. 랭크가 부족하면 같은 예측을 주는 여러 최적 파라미터가 있어 초기값의 영공간 성분이 남을 수 있다. 적절한 고정 러닝레이트에서 로스와 파라미터 수렴의 조건을 구분한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
러닝레이트 예열을 추가하라. 처음 20 에포크 동안 `lr=0.001`로 시작한 뒤 `lr=0.01`로 바꾼다. 처음부터 끝까지 `lr=0.01`을 쓴 경우와 로스 곡선을 비교하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    import torch
    torch.manual_seed(42)
    n, p = 500, 3
    X = torch.randn(n, p)
    w_true = torch.randn(p)
    y = X @ w_true + 3.0 + 0.3 * torch.randn(n)
    
    w = torch.zeros(p, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    
    for epoch in range(200):
        lr = 0.001 if epoch < 20 else 0.01
        y_pred = X @ w + b
        loss = ((y_pred - y) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
        if (epoch+1) % 40 == 0:
            print(f'Epoch {epoch+1}: loss={loss.item():.6f}, lr={lr}')
    # 예열은 초기 로스 곡선을 더 매끄럽게 만들며, 초기 경사가 아주 클 때
    # 도움이 될 수 있다.
    ```

## 정리하며

간단한 학습 루프에서도 로스 계산, 경사 누적, 업데이트와 평가를 구분한다. 파라미터 유일성과 수렴은 설계 행렬과 러닝레이트 조건을 따른다.
