# 다중 출력

다중 목표 회귀는 선형 회귀를 값 하나를 예측하는 것에서 값들의 벡터를 동시에 예측하는 것으로 확장한다. `nn.Linear(p, q)` 층은 $p$개의 입력 특징을 $q$개의 출력으로 대응시키며, 가중치 행렬 $W \in \mathbb{R}^{q \times p}$과 편향 벡터 $b \in \mathbb{R}^q$을 학습한다. 이 튜토리얼은 입력 3개, 출력 2개인 다중 출력 경우를 보이고 학습된 매개변수를 정규 방정식의 해와 대조하여 확인한다.

## 코드

```python
"""
출력이 여럿인 선형 회귀
===================================

과녁이 여럿인 회귀: R^3 → R^2으로 옮기는 nn.Linear(3, 2).

Demonstrates:
- q > 1인 nn.Linear(p, q)
- 가중치 행렬의 꼴: (q, p)으로 담고 x @ W^T + b으로 계산한다
- 출력마다의 R² 평가
- 출력이 여럿일 때 정규 방정식으로 따져 보기

지은이: 깊은 학습 바탕 학습 차례
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ============================================================================
# 설정
# ============================================================================

parser = argparse.ArgumentParser(description="multi-output linear regression")
parser.add_argument("--n-samples", type=int, default=500)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--seed", type=int, default=42)
ARGS = parser.parse_args()

torch.manual_seed(ARGS.seed)
np.random.seed(ARGS.seed)

# ============================================================================
# 합성 데이터: 입력 3개 → 출력 2개
# ============================================================================

n, p, q = ARGS.n_samples, 3, 2
X = torch.randn(n, p)
W_true = torch.tensor([[2.0, -1.0], [0.5, 1.5], [-0.3, 0.8]])  # (p, q) = (3, 2)
b_true = torch.tensor([1.0, -2.0])                                # (q,)
Y = X @ W_true + b_true + 0.2 * torch.randn(n, q)

n_train = int(0.8 * n)
X_train, X_test = X[:n_train], X[n_train:]
Y_train, Y_test = Y[:n_train], Y[n_train:]

loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=ARGS.batch_size, shuffle=True)

print(f"Data shapes: X={X.shape}, Y={Y.shape}")
print(f"W_true (p×q):\n{W_true}")
print(f"b_true: {b_true}")

# ============================================================================
# 모델: nn.Linear(3, 2)
# ============================================================================

model = nn.Linear(p, q)
optimizer = torch.optim.SGD(model.parameters(), lr=ARGS.lr)

print(f"\nModel: {model}")
print(f"Weight shape: {model.weight.shape}  (stored as q×p = {q}×{p})")
print(f"Bias shape:   {model.bias.shape}")

# ============================================================================
# 학습
# ============================================================================

history = []
for epoch in range(ARGS.epochs):
    epoch_loss = 0.0
    for X_b, Y_b in loader:
        Y_pred = model(X_b)                 # (B, q)
        loss = F.mse_loss(Y_pred, Y_b)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(Y_b)

    avg_loss = epoch_loss / n_train
    history.append(avg_loss)

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:3d}  MSE = {avg_loss:.6f}")

# ============================================================================
# 평가
# ============================================================================

with torch.no_grad():
    Y_pred_test = model(X_test)
    overall_mse = F.mse_loss(Y_pred_test, Y_test).item()

    print(f"\nOverall Test MSE: {overall_mse:.6f}")
    for j in range(q):
        ss_res = ((Y_test[:, j] - Y_pred_test[:, j]) ** 2).sum().item()
        ss_tot = ((Y_test[:, j] - Y_test[:, j].mean()) ** 2).sum().item()
        r2 = 1.0 - ss_res / ss_tot
        print(f"Output {j}: R² = {r2:.4f}")

# ============================================================================
# 참된 매개변수와 비교
# ============================================================================

W_learned = model.weight.detach()  # (q, p)
b_learned = model.bias.detach()    # (q,)

print(f"\nLearned W^T (q×p):\n{W_learned}")
print(f"True    W^T (q×p):\n{W_true.T}")
print(f"\nLearned b: {b_learned}")
print(f"True    b: {b_true}")

# ============================================================================
# 정규 방정식으로 확인하기
# ============================================================================

print("\n--- Normal Equation Verification ---")
X_np = np.column_stack([np.ones(n_train), X_train.numpy()])
Y_np = Y_train.numpy()
B_star = np.linalg.solve(X_np.T @ X_np, X_np.T @ Y_np)
print(f"Normal eq bias: {B_star[0]}")
print(f"Normal eq W:\n{B_star[1:]}")

# ============================================================================
# 수렴 그림
# ============================================================================

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(history, lw=1.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE")
ax.set_title(f"Multi-Output Regression ({p}→{q}) — Training Loss")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("multiple_outputs.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: multiple_outputs.png")


if __name__ == "__main__":
    pass
```

## 논의

단일 출력 회귀와의 핵심 차이는 가중치 행렬의 모양이다. `nn.Linear(3, 2)`는 가중치를 $(2, 3)$ 행렬로 저장하므로(PyTorch에서는 `(out_features, in_features)` 순서로 저장한다), 순전파는 $X$가 $(n, 3)$이고 $\hat{Y}$가 $(n, 2)$일 때 $\hat{Y} = X W^T + b$를 계산한다. MSE 손실 `F.mse_loss`는 출력의 모든 원소에 대해 평균을 내어 각 출력 차원을 동등하게 취급한다. 학습 루프는 바꿀 필요가 없다. 출력 차원과 무관하게 똑같은 `optimizer.zero_grad(); loss.backward(); optimizer.step()` 패턴이 통한다.

다중 출력 모델을 평가하려면 출력 차원마다 지표를 계산해야 한다. 각 출력 열에 대한 $R^2$ 점수는 모델이 그 특정 목표를 얼마나 잘 예측하는지를 잰다. 한 출력의 $R^2$이 다른 것들보다 훨씬 낮다면, 그 목표가 주어진 특징으로부터 예측하기 더 어렵다는 뜻일 수 있다. 전체 MSE는 출력에 걸쳐 평균을 내므로 출력별 차이를 가릴 수 있다.

다중 출력 회귀의 정규 방정식은 $B^* = (X^T X)^{-1} X^T Y$이며, 여기서 $Y$는 $(n, q)$ 목표 행렬이고 $B^*$는 (확장된 설계 행렬을 통해 편향을 포함한) $(d+1, q)$ 매개변수 행렬이다. 경사 하강법의 해를 이 닫힌 형태의 답과 비교하면 최적화기가 제대로 수렴했는지 확인할 수 있다. 여기서 쓰는 합성 데이터라면 차이가 무시할 만큼 작아야 한다.

## 연습문제

**익힘 1.**
출력을 2개 대신 5개 예측하도록 모델을 수정하라. 가중치 행렬의 모양이 $(5, 3)$이고 출력별 $R^2$ 점수가 모두 1.0에 가까운지 확인하라.

??? success "익힘 1 풀이"
    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    torch.manual_seed(42)
    n, p, q = 500, 3, 5
    X = torch.randn(n, p)
    W_true = torch.randn(p, q)
    b_true = torch.randn(q)
    Y = X @ W_true + b_true + 0.2 * torch.randn(n, q)
    
    model = nn.Linear(p, q)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(200):
        loss = F.mse_loss(model(X), Y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    
    print(f'Weight shape: {model.weight.shape}')  # (5, 3)
    with torch.no_grad():
        Y_pred = model(X)
        for j in range(q):
            ss_res = ((Y[:, j] - Y_pred[:, j])**2).sum().item()
            ss_tot = ((Y[:, j] - Y[:, j].mean())**2).sum().item()
            print(f'Output {j}: R^2 = {1 - ss_res/ss_tot:.4f}')
    ```

---

**익힘 2.**
`nn.Linear`가 가중치를 $(q, p)$로 저장하는 것과 $W$가 $(p, q)$인 수학적 관례 $Y = XW + b$의 차이를 설명하라. PyTorch는 이를 어떻게 해결하는가?

??? success "익힘 2 풀이"
    PyTorch는 가중치 행렬을 `(out_features, in_features)` = $(q, p)$로 저장하는데, 이는 수학적 관례의 전치이다. 순전파는 `x @ weight.T + bias`를 계산하며, 이는 $W = \text{weight}^T$가 $(p, q)$일 때 $XW + b$와 같다. 이런 저장 방식에서는 가중치 행렬의 각 행이 출력 뉴런 하나의 가중치를 나타내므로 가중치 초기화와 점검에 편리하다. `nn.Linear`가 전치를 내부에서 처리하므로 사용자가 이를 신경 쓸 일은 대개 없다.

---

**익힘 3.**
L2 정칙화(`weight_decay=0.01`)로 다중 출력 모델을 학습시키고 학습된 가중치의 크기를 정칙화하지 않은 버전과 비교하라. 정칙화가 모든 출력에 똑같이 영향을 주는가?

??? success "익힘 3 풀이"
    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    torch.manual_seed(42)
    n, p, q = 500, 3, 2
    X = torch.randn(n, p)
    W_true = torch.tensor([[2.0, -1.0], [0.5, 1.5], [-0.3, 0.8]])
    b_true = torch.tensor([1.0, -2.0])
    Y = X @ W_true + b_true + 0.2 * torch.randn(n, q)
    
    for wd in [0.0, 0.01]:
        model = nn.Linear(p, q)
        opt = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=wd)
        for _ in range(200):
            loss = F.mse_loss(model(X), Y)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f'weight_decay={wd}: weights=\n{model.weight.detach()}')
    # 정칙화는 출력과 무관하게 모든 가중치를 비례해서 줄인다.
    ```
