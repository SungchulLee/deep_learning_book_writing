# 텐서로 만드는 데이터셋

이 스크립트는 선형 회귀에 대한 PyTorch의 정석적인 패턴을 보여준다. `nn.Linear`를 감싼 `nn.Module` 하위 클래스, 배치를 위한 `TensorDataset`과 `DataLoader` 파이프라인, CPU와 GPU 양쪽에서 동작하는 장치 독립적 학습, 그리고 모델을 보존하기 위한 `state_dict` 직렬화이다. 이 패턴은 뒤 장에서 다루는 모든 모델 구조에 그대로 적용되므로 반드시 익혀 두어야 한다.

## 1. 코드

```python
"""
nn.Linear으로 하는 선형 회귀 — 텐서에서 만든 데이터셋
========================================================

nn.Linear, F.mse_loss, optim.SGD, DataLoader, 기기 다루기,
state_dict 저장와 불러오기를 쓰는 PyTorch다운 선형 회귀.

이것이 뒤의 모든 장으로 이어지는 표준 무늬다.

Demonstrates:
- nn.Module 아래 클래스
- TensorDataset / DataLoader 흐름
- 기기를 가리지 않는 학습(CPU / GPU)
- optimizer.zero_grad() → loss.backward() → optimizer.step()
- 추론을 위한 model.eval() + torch.no_grad()
- 모델을 남기기 위한 torch.save / torch.load

지은이: 깊은 학습 바탕 학습 차례
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ============================================================================
# 설정
# ============================================================================

parser = argparse.ArgumentParser(description="nn.Linear linear regression")
parser.add_argument("--n-samples", type=int, default=500)
parser.add_argument("--n-features", type=int, default=3)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--save-path", type=str, default="linear_model.pt")
ARGS = parser.parse_args()

torch.manual_seed(ARGS.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# 데이터
# ============================================================================

n, p = ARGS.n_samples, ARGS.n_features
X = torch.randn(n, p)
w_true = torch.randn(p)
b_true = 3.0
y = X @ w_true + b_true + 0.3 * torch.randn(n)

n_train = int(0.8 * n)
X_train = X[:n_train].to(device)
X_test = X[n_train:].to(device)
y_train = y[:n_train].to(device)
y_test = y[n_train:].to(device)

train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=ARGS.batch_size, shuffle=True)

# ============================================================================
# 모델
# ============================================================================


class LinearRegression(nn.Module):
    def __init__(self, in_features: int, out_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


model = LinearRegression(in_features=p).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=ARGS.lr)

print(model)
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

# ============================================================================
# 학습
# ============================================================================

history = []
for epoch in range(ARGS.epochs):
    model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(y_batch)

    avg_loss = epoch_loss / len(train_ds)
    history.append(avg_loss)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d}  MSE = {avg_loss:.6f}")

# ============================================================================
# 평가
# ============================================================================


@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    y_pred = model(X)
    mse = F.mse_loss(y_pred, y).item()
    ss_res = ((y - y_pred) ** 2).sum().item()
    ss_tot = ((y - y.mean()) ** 2).sum().item()
    r2 = 1.0 - ss_res / ss_tot
    return {"mse": mse, "r2": r2}


train_metrics = evaluate(model, X_train, y_train)
test_metrics = evaluate(model, X_test, y_test)
print(f"\nTrain MSE: {train_metrics['mse']:.6f}  R²: {train_metrics['r2']:.4f}")
print(f"Test  MSE: {test_metrics['mse']:.6f}  R²: {test_metrics['r2']:.4f}")

w_learned = model.linear.weight.detach().cpu().squeeze().numpy()
b_learned = model.linear.bias.detach().cpu().item()
print(f"\nLearned w: {w_learned}")
print(f"True    w: {w_true.numpy()}")
print(f"Learned b: {b_learned:.4f}  (true: {b_true})")

# ============================================================================
# 저장 / 불러오기
# ============================================================================

torch.save(model.state_dict(), ARGS.save_path)
print(f"\nModel saved to {ARGS.save_path}")

model2 = LinearRegression(in_features=p).to(device)
model2.load_state_dict(torch.load(ARGS.save_path, map_location=device))
model2.eval()

test_metrics2 = evaluate(model2, X_test, y_test)
assert abs(test_metrics["mse"] - test_metrics2["mse"]) < 1e-6
print("Model loaded and verified — predictions match.")

# ============================================================================
# 수렴 그림
# ============================================================================

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(history, lw=1.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE")
ax.set_title("nn.Linear — Training Loss")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("dataset_from_tensor.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: dataset_from_tensor.png")


if __name__ == "__main__":
    pass
```

## 2. 논의

`LinearRegression` 모델 하위 클래스는 `nn.Linear(in_features, out_features)`로 완전 연결 층 하나를 정의한다. forward 메서드의 `squeeze(-1)`은 마지막 차원을 없애 출력 모양을 1차원 목표 텐서와 맞춘다. 이는 `nn.Linear`의 출력이 `(batch, 1)` 모양인데 목표는 `(batch,)` 모양일 때 흔히 쓰는 방식이다. 모델은 `.to(device)`로 알맞은 장치로 옮기며, 모든 데이터 텐서도 같은 장치에 있어야 한다.

학습 루프는 표준 패턴을 따른다. DataLoader의 배치를 순회하며 예측과 손실을 계산하고, `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`을 차례로 호출한다. 에폭 평균을 올바르게 가중하기 위해 손실은 `loss.item() * len(y_batch)`로 배치에 걸쳐 누적한다. 마지막 배치의 표본 수가 배치 크기보다 적을 수 있으므로 이런 가중 평균이 중요하다.

모델 평가에는 `@torch.no_grad()` 데코레이터(`with torch.no_grad():` 블록과 같다)를 써서 경사 추적을 꺼 추론 중 메모리와 계산을 아낀다. $R^2$ 점수는 잔차제곱합과 총제곱합으로부터 직접 계산한다. 평가 후에는 모델 가중치를 참된 데이터 생성 매개변수와 비교하여 경사 하강법이 그것을 되찾았는지 확인한다. 마지막으로 `state_dict`와 함께 쓰는 `torch.save`와 `torch.load`가 표준 직렬화 방식을 보여준다.

## 연습문제

**연습문제 1.**
기본 `'mean'` 축약 대신 `nn.MSELoss(reduction='sum')`을 쓰도록 스크립트를 수정하라. 학습을 안정적으로 유지하려면 학습률을 어떻게 바꿔야 하는가?

??? success "연습문제 1 풀이"
    `reduction='sum'`을 쓰면 손실이 배치 크기에 비례하므로 경사가 배치 크기만큼 커진다. 이를 보정하려면 학습률을 배치 크기로 나누어야 한다. 예를 들어 `'mean'`에서 `lr=0.01`이 잘 동작했다면, 배치 크기 32인 `'sum'`에서는 `lr=0.01/32`(약 0.0003)가 필요하다. 반대로 `'sum'` 축약에서 원래 학습률을 그대로 쓰면 최적화기가 너무 큰 걸음을 내디뎌 발산한다.

---

**연습문제 2.**
학습 루프에 검증 분할을 추가하라. 80%는 학습에, 20%는 검증에 쓴다. 매 에폭마다 학습 MSE와 검증 MSE를 모두 출력하라.

??? success "연습문제 2 풀이"
    ```python
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    
    torch.manual_seed(42)
    n, p = 500, 3
    X = torch.randn(n, p)
    w_true = torch.randn(p)
    y = X @ w_true + 3.0 + 0.3 * torch.randn(n)
    
    n_train = int(0.8 * n)
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    
    model = nn.Linear(p, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    
    for epoch in range(100):
        model.train()
        pred = model(X_train).squeeze()
        loss = nn.MSELoss()(pred, y_train)
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val).squeeze()
            val_loss = nn.MSELoss()(val_pred, y_val)
        if (epoch+1) % 20 == 0:
            print(f'Epoch {epoch+1}: train_MSE={loss.item():.6f}, val_MSE={val_loss.item():.6f}')
    ```

---

**연습문제 3.**
입력 특징을 10개 쓰도록 모델을 확장하고, 학습된 가중치가 참된 가중치로 수렴하는지 확인하라. 나란히 비교하여 출력하라.

??? success "연습문제 3 풀이"
    ```python
    import torch
    import torch.nn as nn
    
    torch.manual_seed(42)
    n, p = 500, 10
    X = torch.randn(n, p)
    w_true = torch.randn(p)
    b_true = 3.0
    y = X @ w_true + b_true + 0.3 * torch.randn(n)
    
    model = nn.Linear(p, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(200):
        loss = nn.MSELoss()(model(X).squeeze(), y)
        opt.zero_grad(); loss.backward(); opt.step()
    
    w_learned = model.weight.detach().squeeze().numpy()
    print('Feature | True w | Learned w')
    for i in range(p):
        print(f'  {i:5d} | {w_true[i].item():7.4f} | {w_learned[i]:7.4f}')
    print(f'  bias  | {b_true:7.4f} | {model.bias.item():7.4f}')
    ```

## 정리하며

**다룬 것** — 텐서로 만드는 데이터셋

`LinearRegression` 모델 하위 클래스는 `nn.Linear(in_features, out_features)`로 완전 연결 층 하나를 정의한다.

핵심 클래스는 `LinearRegression`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
