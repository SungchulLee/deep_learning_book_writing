# 정칙화된 로지스틱 회귀
## 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

- 로지스틱 회귀에 정칙화가 왜 필요한지 이해하기
- 정칙화된 손실 함수(L2, L1, 엘라스틱 넷)와 그 경사 유도하기
- 정칙화의 기하학적 해석과 베이즈적 해석 설명하기
- PyTorch로 실전에 쓸 수 있는 완전한 로지스틱 회귀 파이프라인 구현하기
- 정칙화된 로지스틱 회귀를 실제 데이터셋에 적용하고 제대로 평가하기

---

## 왜 정칙화하는가?

### 완전 분리 문제

학습 데이터가 **선형 분리 가능**하면 로지스틱 회귀의 최대가능도 추정값이 존재하지 않는다. 모델이 예측을 더 확신에 차게 만들어 언제나 가능도를 높일 수 있으므로 최적 매개변수가 $\|\boldsymbol{\beta}\| \to \infty$으로 발산한다.

수학적으로, ($y_i \in \{-1, +1\}$ 부호화를 쓸 때) 모든 $i$에 대해 $y_i(\mathbf{x}_i^\top \boldsymbol{\beta}^*) > 0$인 $\boldsymbol{\beta}^*$이 존재한다면, $c \to \infty$으로 $\boldsymbol{\beta}^* \to c\boldsymbol{\beta}^*$처럼 배율을 키울 때 모든 예측이 확실성 쪽으로 몰리고 로그가능도가 (상한인) 0에 다가가지만 결코 도달하지는 못한다.

### 고차원에서의 과적합

완전 분리가 아니어도 다음 경우에 로지스틱 회귀는 과적합할 수 있다.

- 특징의 수 $d$이 $n$에 비해 클 때
- 특징들이 강하게 상관되어 있을 때 (다중공선성)
- 모델이 학습 데이터의 잡음을 외울 때

정칙화는 $\|\boldsymbol{\beta}\|$을 제약하여 지나치게 확신에 찬 예측을 막고 일반화를 개선한다.

---

## L2 정칙화 (릿지)

### 정칙화된 목적 함수

L2 정칙화는 가중치의 L2 노름 제곱에 벌점을 더한다.

$$
\mathcal{L}_{\text{ridge}}(\boldsymbol{\beta}) = -\frac{1}{n}\sum_{i=1}^{n} \left[ y_i \log p_i + (1-y_i) \log(1-p_i) \right] + \frac{\lambda}{2}\|\boldsymbol{\beta}_{1:d}\|_2^2
$$

여기서 $\lambda > 0$은 정칙화 강도이고 $\boldsymbol{\beta}_{1:d}$은 절편 $\beta_0$을 뺀 것이다(절편에는 보통 벌점을 주지 않는다).

### L2 정칙화가 있을 때의 경사

$$
\nabla_{\boldsymbol{\beta}} \mathcal{L}_{\text{ridge}} = \frac{1}{n}\mathbf{X}^\top(\mathbf{p} - \mathbf{y}) + \lambda \boldsymbol{\beta}
$$

(절편 성분에는 $\lambda \beta_0 = 0$이다.) 정칙화 항은 매 경사 단계마다 $\boldsymbol{\beta}$을 0 쪽으로 당기는 힘을 더한다.

### 결정 경계에 미치는 영향

[결정 경계](decision_boundary.md) 절에서 보았듯이 $\|\boldsymbol{\beta}\|$이 확률 전이의 가파름을 조절한다. L2 정칙화는 $\|\boldsymbol{\beta}\|$을 줄여 경계 근처에서 확률이 더 부드럽게 전이하게 만들고, 예측의 확신을 낮추며(확률이 0.5에 더 가까워진다), 실무에서 더 잘 보정된 모델을 낳는다.

### 베이즈적 해석

L2 정칙화는 가중치에 **가우스 사전분포**를 주는 것과 같다.

$$
\boldsymbol{\beta} \sim \mathcal{N}(\mathbf{0}, \sigma_\beta^2 \mathbf{I}), \quad \text{where } \lambda = \frac{1}{\sigma_\beta^2}
$$

정칙화된 목적 함수는 **최대 사후확률(MAP)** 추정에 대응한다.

$$
\boldsymbol{\beta}_{\text{MAP}} = \arg\max_{\boldsymbol{\beta}} \left[ \log P(\mathcal{D}|\boldsymbol{\beta}) + \log P(\boldsymbol{\beta}) \right]
$$

$\log P(\boldsymbol{\beta}) = -\frac{\lambda}{2}\|\boldsymbol{\beta}\|_2^2 + \text{const}$이므로 사후확률을 최대화하는 것은 릿지 벌점을 준 손실을 최소화하는 것과 같다.

---

## L1 정칙화 (라쏘)

### 정칙화된 목적 함수

L1 정칙화는 절댓값의 합에 벌점을 준다.

$$
\mathcal{L}_{\text{lasso}}(\boldsymbol{\beta}) = -\frac{1}{n}\sum_{i=1}^{n} \left[ y_i \log p_i + (1-y_i) \log(1-p_i) \right] + \lambda \|\boldsymbol{\beta}_{1:d}\|_1
$$

### 희소성을 유도하는 성질

L2와의 핵심 차이는 L1 정칙화가 **희소한** 해를 낸다는 점이다. 많은 계수가 정확히 0으로 밀려난다. 그래서 L1 정칙화는 일종의 **자동 특징 선택**이 된다.

기하학적으로 L1 제약 집합 $\|\boldsymbol{\beta}\|_1 \leq t$은 마름모(교차다포체)이다. 손실 등고선은 좌표 하나 이상이 정확히 0인 꼭짓점에서 이 마름모와 만날 가능성이 더 높다.

### 베이즈적 해석

L1 정칙화는 가중치에 대한 **라플라스 사전분포**에 대응한다.

$$
P(\beta_j) = \frac{\lambda}{2}\exp(-\lambda|\beta_j|)
$$

라플라스 분포의 두꺼운 꼬리는 정말 중요한 특징에 큰 계수를 허용하고, 0에서의 날카로운 봉우리는 희소성을 북돋운다.

---

## 엘라스틱 넷

### L1과 L2 결합하기

엘라스틱 넷은 두 벌점을 결합한다.

$$
\mathcal{L}_{\text{elastic}}(\boldsymbol{\beta}) = \text{BCE} + \lambda_1 \|\boldsymbol{\beta}_{1:d}\|_1 + \frac{\lambda_2}{2}\|\boldsymbol{\beta}_{1:d}\|_2^2
$$

또는 동등하게, 혼합 매개변수 $\alpha \in [0, 1]$을 써서 다음과 같이 쓴다.

$$
\mathcal{L}_{\text{elastic}}(\boldsymbol{\beta}) = \text{BCE} + \lambda \left[\alpha \|\boldsymbol{\beta}_{1:d}\|_1 + \frac{1-\alpha}{2}\|\boldsymbol{\beta}_{1:d}\|_2^2\right]
$$

### 무엇을 언제 쓸 것인가

| 방법 | 쓰는 때 |
|--------|-------------|
| L2 (릿지) | 적당히 중요한 특징이 많을 때, 상관된 특징이 있을 때 |
| L1 (라쏘) | 정말 중요한 특징이 적을 때, 특징 선택이 필요할 때 |
| 엘라스틱 넷 | 상관된 특징이 있으면서 희소성도 필요할 때, 상관된 특징의 무리가 있을 때 |

### 정칙화 요약

| 정칙화 항 | 벌점 | 사전분포 | 희소성 | 경사 |
|-------------|---------|-------|----------|----------|
| L2 (릿지) | $\frac{\lambda}{2}\|\boldsymbol{\beta}\|_2^2$ | 가우스 | 없음 | $\lambda \boldsymbol{\beta}$ |
| L1 (라쏘) | $\lambda\|\boldsymbol{\beta}\|_1$ | 라플라스 | 있음 | $\lambda \operatorname{sign}(\boldsymbol{\beta})$ |
| 엘라스틱 넷 | $\lambda[\alpha\|\boldsymbol{\beta}\|_1 + \frac{1-\alpha}{2}\|\boldsymbol{\beta}\|_2^2]$ | 혼합 | 부분적 | $\lambda[\alpha\operatorname{sign}(\boldsymbol{\beta}) + (1-\alpha)\boldsymbol{\beta}]$ |

---

## PyTorch 구현

```python
"""
Regularized Logistic Regression — Complete Implementation
==========================================================

A production-ready implementation covering:
- L2 regularization via weight_decay
- L1 regularization via manual penalty
- Elastic Net regularization
- Data handling with train/val/test splits
- Training with early stopping
- Comprehensive evaluation metrics
- Regularization strength tuning

Author: Deep Learning Foundations
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional

torch.manual_seed(42)
np.random.seed(42)

print("=" * 70)
print("REGULARIZED LOGISTIC REGRESSION — COMPLETE PIPELINE")
print("=" * 70)

# ============================================================================
# 1부: 데이터 준비
# ============================================================================

class BinaryClassificationDataset(Dataset):
    """
    Custom Dataset for binary classification.

    Handles data preprocessing including standardization.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Labels of shape (n_samples,) with values in {0, 1}
        scaler: Optional pre-fitted StandardScaler
        fit_scaler: Whether to fit the scaler (True for training data)
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = False,
    ):
        if scaler is None and fit_scaler:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        elif scaler is not None:
            self.scaler = scaler
            X = self.scaler.transform(X)
        else:
            self.scaler = None

        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    def get_scaler(self) -> Optional[StandardScaler]:
        return self.scaler

def prepare_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.2,
    batch_size: int = 32,
    random_state: int = 42,
) -> Dict:
    """층화 분할로 학습/검증/시험 DataLoader를 준비한다."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted,
        random_state=random_state, stratify=y_temp,
    )

    train_dataset = BinaryClassificationDataset(X_train, y_train, fit_scaler=True)
    scaler = train_dataset.get_scaler()
    val_dataset = BinaryClassificationDataset(X_val, y_val, scaler=scaler)
    test_dataset = BinaryClassificationDataset(X_test, y_test, scaler=scaler)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"\nData prepared:")
    print(f"  Training:   {len(train_dataset):,} samples")
    print(f"  Validation: {len(val_dataset):,} samples")
    print(f"  Test:       {len(test_dataset):,} samples")
    print(f"  Features:   {X.shape[1]}")

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "scaler": scaler,
        "n_features": X.shape[1],
    }

# ============================================================================
# 2부: 모델 정의
# ============================================================================

class LogisticRegression(nn.Module):
    """
    Logistic Regression with optional regularization.

    Architecture: Linear → Sigmoid

    For numerically stable training, use BCEWithLogitsLoss
    and call logits() instead of forward().

    Args:
        n_features: Number of input features
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[0, 1] 범위의 확률을 반환한다."""
        return torch.sigmoid(self.linear(x))

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """날것의 로짓을 반환한다 (BCEWithLogitsLoss용)."""
        return self.linear(x)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """이진 예측을 반환한다."""
        self.eval()
        with torch.no_grad():
            return (self.forward(x) >= threshold).float()

    def l1_penalty(self) -> torch.Tensor:
        """가중치에 대한 L1 벌점을 계산한다 (편향은 제외)."""
        return self.linear.weight.abs().sum()

    def l2_penalty(self) -> torch.Tensor:
        """가중치에 대한 L2 벌점을 계산한다 (편향은 제외)."""
        return (self.linear.weight ** 2).sum()

# ============================================================================
# 3부: 정칙화를 적용한 학습
# ============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 0.01,
    l1_lambda: float = 0.0,
    l2_lambda: float = 0.0,
    patience: int = 10,
    verbose: bool = True,
) -> Dict:
    """
    Train logistic regression with optional L1/L2 regularization.

    L2 regularization is applied via optimizer weight_decay.
    L1 regularization is applied as a manual penalty term in the loss.
    For Elastic Net, set both l1_lambda > 0 and l2_lambda > 0.

    Args:
        model: LogisticRegression model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        num_epochs: Maximum training epochs
        learning_rate: Learning rate
        l1_lambda: L1 regularization strength
        l2_lambda: L2 regularization strength (weight_decay)
        patience: Early stopping patience
        verbose: Print progress

    Returns:
        Training history dictionary
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=l2_lambda
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    if verbose:
        reg_desc = []
        if l2_lambda > 0:
            reg_desc.append(f"L2(λ={l2_lambda})")
        if l1_lambda > 0:
            reg_desc.append(f"L1(λ={l1_lambda})")
        reg_str = " + ".join(reg_desc) if reg_desc else "None"
        print(f"\nRegularization: {reg_str}")
        print(f"Training for up to {num_epochs} epochs (patience={patience})...")
        print("-" * 60)

    for epoch in range(num_epochs):
        # --- 학습 ---
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch_X, batch_y in train_loader:
            logits = model.logits(batch_X)
            loss = criterion(logits, batch_y)

            # L1 벌점은 직접 더한다 (L2는 weight_decay가 처리한다)
            if l1_lambda > 0:
                loss = loss + l1_lambda * model.l1_penalty()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_X)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += len(batch_X)

        train_loss = total_loss / total
        train_acc = correct / total

        # --- 검증 ---
        model.eval()
        val_loss_total, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                logits = model.logits(batch_X)
                loss = criterion(logits, batch_y)
                val_loss_total += loss.item() * len(batch_X)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == batch_y).sum().item()
                val_total += len(batch_X)

        val_loss = val_loss_total / val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # 조기 종료
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, "
                f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
            )

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    history["best_epoch"] = epoch - patience_counter + 1
    return history

# ============================================================================
# 4부: 평가
# ============================================================================

def evaluate_model(model: nn.Module, test_loader: DataLoader) -> Dict:
    """종합적인 모델 평가."""
    model.eval()
    all_probs, all_preds, all_targets = [], [], []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            probs = model(batch_X)
            preds = (probs >= 0.5).float()
            all_probs.extend(probs.numpy().flatten())
            all_preds.extend(preds.numpy().flatten())
            all_targets.extend(batch_y.numpy().flatten())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    return {
        "accuracy": accuracy_score(all_targets, all_preds),
        "precision": precision_score(all_targets, all_preds, zero_division=0),
        "recall": recall_score(all_targets, all_preds, zero_division=0),
        "f1": f1_score(all_targets, all_preds, zero_division=0),
        "auc": roc_auc_score(all_targets, all_probs),
        "confusion_matrix": confusion_matrix(all_targets, all_preds),
    }

def print_evaluation_report(metrics: Dict):
    """형식을 갖춘 평가 보고서를 출력한다."""
    print(f"\n  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc']:.4f}")

    cm = metrics["confusion_matrix"]
    print(f"\n  Confusion Matrix:")
    print(f"             Predicted")
    print(f"               0     1")
    print(f"  Actual 0   {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"         1   {cm[1,0]:4d}  {cm[1,1]:4d}")

# ============================================================================
# 5부: 정칙화 비교
# ============================================================================

print("\n" + "=" * 70)
print("REGULARIZATION COMPARISON ON BREAST CANCER DATASET")
print("=" * 70)

# 데이터를 불러온다
data = load_breast_cancer()
X, y = data.data, data.target
data_dict = prepare_data(X, y, batch_size=32)

configs = [
    {"name": "No Regularization", "l1": 0.0, "l2": 0.0},
    {"name": "L2 (Ridge, λ=0.01)", "l1": 0.0, "l2": 0.01},
    {"name": "L1 (Lasso, λ=0.001)", "l1": 0.001, "l2": 0.0},
    {"name": "Elastic Net", "l1": 0.0005, "l2": 0.005},
]

results = {}

for cfg in configs:
    print(f"\n--- {cfg['name']} ---")
    torch.manual_seed(42)
    model = LogisticRegression(data_dict["n_features"])
    history = train_model(
        model,
        data_dict["train_loader"],
        data_dict["val_loader"],
        num_epochs=200,
        learning_rate=0.01,
        l1_lambda=cfg["l1"],
        l2_lambda=cfg["l2"],
        patience=15,
        verbose=False,
    )
    metrics = evaluate_model(model, data_dict["test_loader"])
    print_evaluation_report(metrics)

    # 가중치 통계량을 저장한다
    weights = model.linear.weight.data.numpy().flatten()
    results[cfg["name"]] = {
        "metrics": metrics,
        "history": history,
        "weights": weights,
        "weight_norm": np.linalg.norm(weights),
        "n_nonzero": np.sum(np.abs(weights) > 1e-3),
    }

print(f"\n\nWeight Statistics Comparison:")
print("-" * 70)
print(f"{'Method':<30} {'||β||₂':>8} {'Nonzero':>10} {'AUC':>8}")
print("-" * 70)
for name, r in results.items():
    print(
        f"{name:<30} {r['weight_norm']:>8.4f} "
        f"{r['n_nonzero']:>10d}/{len(r['weights'])} "
        f"{r['metrics']['auc']:>8.4f}"
    )

# ============================================================================
# 6부: 정칙화 강도 조율
# ============================================================================

print("\n" + "=" * 70)
print("L2 REGULARIZATION STRENGTH TUNING")
print("=" * 70)

lambdas = [0, 0.0001, 0.001, 0.01, 0.1, 1.0]
tuning_results = []

for lam in lambdas:
    torch.manual_seed(42)
    model = LogisticRegression(data_dict["n_features"])
    history = train_model(
        model,
        data_dict["train_loader"],
        data_dict["val_loader"],
        num_epochs=200,
        learning_rate=0.01,
        l2_lambda=lam,
        patience=15,
        verbose=False,
    )
    metrics = evaluate_model(model, data_dict["test_loader"])
    weights = model.linear.weight.data.numpy().flatten()
    tuning_results.append({
        "lambda": lam,
        "val_loss": min(history["val_loss"]),
        "test_auc": metrics["auc"],
        "test_acc": metrics["accuracy"],
        "weight_norm": np.linalg.norm(weights),
    })
    print(
        f"λ={lam:<8.4f}: Val Loss={tuning_results[-1]['val_loss']:.4f}, "
        f"Test AUC={metrics['auc']:.4f}, ||β||={tuning_results[-1]['weight_norm']:.4f}"
    )

# ============================================================================
# 7부: 시각화
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# 그림 1: 정칙화 종류별 학습 곡선
ax = axes[0, 0]
for name, r in results.items():
    ax.plot(r["history"]["val_loss"], label=name)
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Loss")
ax.set_title("Validation Loss Across Regularization Types")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 그림 2: 가중치 분포
ax = axes[0, 1]
for i, (name, r) in enumerate(results.items()):
    ax.hist(r["weights"], bins=20, alpha=0.5, label=name)
ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Weight Value")
ax.set_ylabel("Count")
ax.set_title("Weight Distributions by Regularization")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 그림 3: λ 조율 — AUC와 가중치 노름
ax = axes[1, 0]
lam_vals = [r["lambda"] for r in tuning_results]
aucs = [r["test_auc"] for r in tuning_results]
norms = [r["weight_norm"] for r in tuning_results]

ax.semilogx([max(l, 1e-5) for l in lam_vals], aucs, "b-o", label="Test AUC")
ax.set_xlabel("λ (L2 regularization)")
ax.set_ylabel("Test AUC", color="b")
ax.tick_params(axis="y", labelcolor="b")
ax.grid(True, alpha=0.3)

ax2 = ax.twinx()
ax2.semilogx([max(l, 1e-5) for l in lam_vals], norms, "r--s", label="||β||₂")
ax2.set_ylabel("Weight Norm ||β||₂", color="r")
ax2.tick_params(axis="y", labelcolor="r")
ax.set_title("Effect of L2 Regularization Strength")

# 그림 4: 희소성 비교를 위해 정렬한 가중치 절댓값
ax = axes[1, 1]
for name, r in results.items():
    sorted_abs = np.sort(np.abs(r["weights"]))[::-1]
    ax.plot(sorted_abs, label=name)
ax.axhline(y=1e-3, color="gray", linestyle=":", label="Sparsity threshold")
ax.set_xlabel("Feature Index (sorted by |β|)")
ax.set_ylabel("|β|")
ax.set_title("Weight Magnitude Profiles (Sparsity)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("regularized_logistic_regression.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n✓ Results visualization saved!")
```

---

## 모범 사례

### 데이터 다루기

| 사례 | 이유 |
|----------|--------|
| `DataLoader` 사용 | 효율적인 배치 묶기, 뒤섞기, 다중 처리 |
| 특징 표준화 | 더 빠른 수렴, 특정 특징의 지배 방지 |
| 층화 분할 | 클래스 분포를 보존한다 |
| 세 갈래 분할 | 검증 집합에 과적합하는 것을 막는다 |

### 모델 설계

| 사례 | 이유 |
|----------|--------|
| `nn.Module` 하위 클래스 | 깔끔하고 재사용 가능하며 PyTorch 생태계와 잘 맞는다 |
| `BCEWithLogitsLoss` | 수치적 안정성 |
| Xavier 초기화 | 학습의 움직임에 도움이 된다 |

### 학습

| 사례 | 이유 |
|----------|--------|
| 조기 종료 | 과적합을 막는다 |
| L2에는 `weight_decay` | 효율적이며 최적화기에 내장되어 있다 |
| L1 벌점은 직접 구현 | PyTorch에는 L1 정칙화가 내장되어 있지 않다 |
| 최고 모델 체크포인트 | 최고 성능을 잃지 않는다 |

### 평가

| 사례 | 이유 |
|----------|--------|
| 여러 지표 | 불균형 데이터에서는 정확도만으로는 부족하다 |
| 혼동 행렬 | 오류의 종류를 파악한다 |
| AUC-ROC | 문턱값과 무관한 성능 척도 |

---

## 요약

| 개념 | 공식 | 핵심 통찰 |
|---------|---------|-------------|
| L2 벌점 | $\frac{\lambda}{2}\|\boldsymbol{\beta}\|_2^2$ | 모든 가중치를 줄인다; 가우스 사전분포 |
| L1 벌점 | $\lambda\|\boldsymbol{\beta}\|_1$ | 희소성을 만든다; 라플라스 사전분포 |
| 엘라스틱 넷 | $\lambda[\alpha\|\boldsymbol{\beta}\|_1 + \frac{1-\alpha}{2}\|\boldsymbol{\beta}\|_2^2]$ | 무리 짓기 + 희소성 |
| L2 경사 | $\lambda\boldsymbol{\beta}$ | 가중치 감쇠 |
| L1 열경사 | $\lambda\operatorname{sign}(\boldsymbol{\beta})$ | 0 쪽으로 일정하게 민다 |
| PyTorch의 L2 | 최적화기의 `weight_decay` | 내장되어 있고 효율적이다 |
| PyTorch의 L1 | 손실에 벌점을 직접 더한다 | 명시적으로 추가해야 한다 |

정칙화는 분리 가능한 데이터에서 실패할 수 있는 방법이던 로지스틱 회귀를 수렴이 보장되는 견고하고 잘 정의된 알고리즘으로 바꾼다. L1, L2, 엘라스틱 넷 중 무엇을 고를지는 문제의 구조에 달려 있다. 즉 희소한 해를 기대하는지 조밀한 해를 기대하는지, 그리고 특징들이 어떻게 상관되어 있는지에 달려 있다.

## 연습문제

**연습문제 1.**
L1 정칙화의 근위 연산자(연성 문턱값 처리)를 유도하고 그것이 왜 희소성을 만드는지 설명하라.

??? success "연습문제 1 풀이"
    $\lambda\|w\|_1$에 대한 근위 연산자는 다음과 같다.

    $$
    \text{prox}_{\lambda\|\cdot\|_1}(v) = \text{sign}(v) \max(|v| - \lambda, 0)
    $$

    이것이 연성 문턱값 처리이다. $|v| \leq \lambda$인 값은 정확히 0이 되고, 더 큰 값은 0 쪽으로 $\lambda$만큼 줄어든다. 연성 문턱값이 작은 계수를 0으로 만드는 "죽은 구간"을 만들기 때문에 L1은 희소성을 낳는다.

    L2의 근위 연산자는 $\text{prox}_{\lambda\|\cdot\|_2^2}(v) = v/(1+2\lambda)$으로, 계수를 줄이기만 할 뿐 결코 0으로 만들지 않는다. 이것이 L2가 희소성을 만들지 못하는 이유이다.

---

**연습문제 2.**
$\lambda > 0$일 때 정칙화된 헤세 행렬 $\mathbf{H}_{\text{ridge}} = \mathbf{X}^\top\mathbf{B}\mathbf{X} + n\lambda\mathbf{I}$이 언제나 양의 정부호임을 보여라.

??? success "연습문제 2 풀이"
    $\mathbf{B} = \text{diag}(p_i(1-p_i))$의 대각 성분이 음이 아니므로 $\mathbf{X}^\top\mathbf{B}\mathbf{X}$은 양의 준정부호이다. $\mathbf{v}^\top\mathbf{X}^\top\mathbf{B}\mathbf{X}\mathbf{v} = \|\mathbf{B}^{1/2}\mathbf{X}\mathbf{v}\|^2 \geq 0$이다.

    $n\lambda\mathbf{I}$을 더하면 모든 고윳값이 $n\lambda > 0$만큼 옮겨진다.

    $$
    \mathbf{v}^\top\mathbf{H}_{\text{ridge}}\mathbf{v} = \mathbf{v}^\top\mathbf{X}^\top\mathbf{B}\mathbf{X}\mathbf{v} + n\lambda\|\mathbf{v}\|^2 \geq n\lambda\|\mathbf{v}\|^2 > 0
    $$

    이는 모든 $\mathbf{v} \neq 0$에 대해 성립한다. 따라서 $\mathbf{H}_{\text{ridge}}$은 양의 정부호이며 유일한 전역 최적점이 보장된다. $\square$

---

**연습문제 3.**
완전히 분리 가능한 데이터에서도 L2 정칙화된 로지스틱 회귀에는 언제나 유한한 MLE이 존재함을 증명하라.

??? success "연습문제 3 풀이"
    정칙화된 목적 함수는 $\mathcal{L}(\boldsymbol{\beta}) = -\ell(\boldsymbol{\beta}) + \frac{\lambda}{2}\|\boldsymbol{\beta}\|^2$이다.

    $\|\boldsymbol{\beta}\| \to \infty$일 때 NLL $-\ell(\boldsymbol{\beta}) \geq 0$은 아래로 유계인 반면 $\frac{\lambda}{2}\|\boldsymbol{\beta}\|^2 \to \infty$이다. 따라서 $\mathcal{L} \to \infty$이다.

    $\mathcal{L}$은 연속이고 (0으로) 아래로 유계이며 모든 방향에서 무한대로 가므로, 어떤 유한한 $\boldsymbol{\beta}^*$에서 최솟값을 갖는다. (위에서 증명한 양의 정부호 헤세 행렬로부터 오는) 엄격한 볼록성이 유일성을 보장한다. $\square$

---

**연습문제 4.**
최적의 정칙화 강도 $\lambda$을 고르기 위한 k-겹 교차 검증을 구현하고 검증 곡선을 그려라.

??? success "연습문제 4 풀이"
    ```python
    import torch
    import torch.nn as nn
    from sklearn.model_selection import KFold
    import numpy as np

    lambdas = np.logspace(-4, 2, 20)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for lam in lambdas:
        fold_accs = []
        for train_idx, val_idx in kf.split(X):
            model = nn.Linear(X.shape[1], 1)
            opt = torch.optim.Adam(model.parameters(), lr=0.01,
                                   weight_decay=lam)
            for _ in range(500):
                opt.zero_grad()
                loss = nn.BCEWithLogitsLoss()(model(X[train_idx]).squeeze(),
                                              y[train_idx])
                loss.backward(); opt.step()
            with torch.no_grad():
                preds = (model(X[val_idx]).squeeze() > 0).float()
                fold_accs.append((preds == y[val_idx]).float().mean().item())
        results.append(np.mean(fold_accs))
    # lambda 대 결과를 그리고, 평균 정확도가 가장 높은 lambda를 고른다
    ```
