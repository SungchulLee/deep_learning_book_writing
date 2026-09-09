# 조기 종료 예제

조기 종료 예제. 학습을 멈춤으로써 조기 종료가 과적합을 어떻게 막는지 보여준다

정칙화 기법을 이해하는 것은 깊은 신경망을 효과적으로 만들고 학습시키는 데 필수적이다. 이 구현은 그 핵심 개념을 PyTorch로 보여주며, 현대적인 구조의 구성 요소를 직접 다뤄 볼 기회를 준다.

## 1. 코드

```python
"""
조기 종료 예제
=========================
검증 성능이 나빠지기 시작할 때 학습을 멈춰 과적합을 막는 방식을 보인다.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# ========================================================================
# 메인
# ========================================================================


def create_model(input_dim):
    """신경망 모델을 만든다."""
    # 마지막에 시그모이드를 붙이지 않는다. 아래에서 쓰는
    # BCEWithLogitsLoss가 시그모이드를 안에 품고 있기 때문이다
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )


class EarlyStopping:
    """검증 손실이 나아지지 않으면 학습을 멈춘다.

    학습 루프에서 에포크마다 이 객체에 물어보고, True가 돌아오면 멈춘다.
    """

    def __init__(self, patience=10, min_delta=1e-4, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.best_weights = None
        self.counter = 0

    def __call__(self, score, model):
        """멈춰야 하면 True를 돌려준다."""
        if self.best_score is None or score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                # deepcopy가 필요하다. state_dict()는 살아 있는 파라미터
                # 텐서를 그대로 가리키므로, 그냥 담아 두면 다음 step()이
                # 그 텐서를 덮어써 "가장 좋았던 가중치"가 현재 가중치를
                # 따라 계속 바뀌어 버린다
                self.best_weights = copy.deepcopy(model.state_dict())
            return False

        # 나아지지 않은 에포크를 센다. patience는 "학습을 시작한 뒤"가
        # 아니라 "마지막 개선 뒤" 몇 에포크인지를 뜻한다
        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        """가장 좋았던 가중치로 되돌린다."""
        # 학습이 인내를 다 채우지 못하고 에포크 수를 소진해 끝났더라도
        # 부르는 쪽에서 이것을 불러 주어야 최선의 가중치를 되찾는다
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)


def run_epoch(model, loader, criterion, optimizer=None):
    """한 에포크를 돌고 (손실, 정확도)를 돌려준다."""
    # optimizer를 주었는지로 학습과 평가를 가른다. 모드 전환을
    # 빠뜨리면 드롭아웃이나 배치 정규화가 있는 모델에서 평가가 틀어진다
    model.train() if optimizer is not None else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for xb, yb in loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * xb.size(0)
            correct += ((logits > 0).float() == yb).sum().item()
            total += xb.size(0)
    return total_loss / total, correct / total


def make_loaders(X_train, y_train, X_val, y_val, batch_size=32):
    """학습과 검증 데이터로더를 만든다."""
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size))


def train_without_early_stopping(X_train, y_train, X_val, y_val, epochs=200):
    """조기 종료 없이 모델을 학습시킨다."""
    print("Training WITHOUT early stopping...")
    torch.manual_seed(42)
    model = create_model(X_train.shape[1])
    train_loader, val_loader = make_loaders(X_train, y_train, X_val, y_val)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())

    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    # 200 에포크를 끝까지 돈다. 검증 손실이 돌아선 뒤에도 계속 학습하므로
    # 아래 조기 종료 판본과 견주면 과적합이 눈에 들어온다
    for _ in range(epochs):
        tl, ta = run_epoch(model, train_loader, criterion, optimizer)
        vl, va = run_epoch(model, val_loader, criterion)
        history['loss'].append(tl)
        history['accuracy'].append(ta)
        history['val_loss'].append(vl)
        history['val_accuracy'].append(va)
    return model, history


def train_with_early_stopping(X_train, y_train, X_val, y_val,
                              patience=10, epochs=200):
    """조기 종료와 함께 모델을 학습시킨다."""
    print(f"Training WITH early stopping (patience={patience})...")
    torch.manual_seed(42)
    model = create_model(X_train.shape[1])
    train_loader, val_loader = make_loaders(X_train, y_train, X_val, y_val)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    early_stop = EarlyStopping(patience=patience, restore_best_weights=True)

    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    for epoch in range(epochs):
        tl, ta = run_epoch(model, train_loader, criterion, optimizer)
        vl, va = run_epoch(model, val_loader, criterion)
        history['loss'].append(tl)
        history['accuracy'].append(ta)
        history['val_loss'].append(vl)
        history['val_accuracy'].append(va)

        # 이력을 기록한 "뒤"에 묻는다. 앞에서 물으면 멈추는 에포크의
        # 손실이 history에서 빠진다.
        # 감시하는 것은 학습 손실이 아니라 검증 손실이다. 학습 손실은
        # 과적합이 시작된 뒤에도 계속 내려가 멈출 때를 알려 주지 못한다
        if early_stop(vl, model):
            print(f"  Stopped early at epoch {epoch + 1}")
            break
    early_stop.restore(model)
    return model, history


def train_with_advanced_early_stopping(X_train, y_train, X_val, y_val, epochs=200):
    """학습률 감소와 검사점 저장을 함께 쓰는 학습."""
    print("Training with ADVANCED early stopping (multiple callbacks)...")
    torch.manual_seed(42)
    model = create_model(X_train.shape[1])
    train_loader, val_loader = make_loaders(X_train, y_train, X_val, y_val)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())

    # 인내 값이 15와 5로 다른 것이 이 조합의 핵심이다. 학습률 감소가
    # 5에서 먼저 걸려 모델에게 한 번 더 기회를 주고, 그래도 나아지지
    # 않으면 15에서 조기 종료가 걸린다. 두 값을 거꾸로 두면 학습률을
    # 낮춰 보기도 전에 학습이 끝나 버린다
    early_stop = EarlyStopping(patience=15, restore_best_weights=True)
    # ReduceLROnPlateau는 PyTorch에 내장된 스케줄러다. 코사인 감소처럼
    # 미리 짜 둔 일정과 달리, 실제로 나아지지 않을 때만 반응한다
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
    )

    best_acc = 0.0
    best_state = None
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    for epoch in range(epochs):
        tl, ta = run_epoch(model, train_loader, criterion, optimizer)
        vl, va = run_epoch(model, val_loader, criterion)
        history['loss'].append(tl)
        history['accuracy'].append(ta)
        history['val_loss'].append(vl)
        history['val_accuracy'].append(va)

        # 스케줄러는 검증 손실을 보고 판단하므로 그 값을 넘겨 준다
        scheduler.step(vl)

        # 검사점 저장. 정확도가 가장 높았던 순간의 가중치를 따로 담아 둔다.
        # 주의: 이쪽은 정확도를 보는데 위의 조기 종료는 손실을 본다.
        # 두 지표의 최고점이 같은 에포크라는 보장이 없으므로, 여기 담긴
        # best_state와 아래에서 복원되는 가중치가 서로 다를 수 있다.
        # 이 함수는 조기 종료가 고른 쪽을 돌려주며, best_state는 견주어
        # 볼 수 있도록 남겨 둔 것이다. 둘 중 어느 것을 쓸지는 정해 두어야
        # 하며, 정확도로 고르고 싶다면 마지막에 이것을 실어야 한다
        if va > best_acc:
            best_acc = va
            best_state = copy.deepcopy(model.state_dict())

        if early_stop(vl, model):
            print(f"  Stopped early at epoch {epoch + 1}")
            break

    early_stop.restore(model)
    return model, history


def evaluate(model, X, y):
    """(손실, 정확도, AUC)를 돌려준다."""
    model.eval()
    with torch.no_grad():
        logits = model(X)
        loss = nn.BCEWithLogitsLoss()(logits, y).item()
        probs = torch.sigmoid(logits)
        accuracy = ((logits > 0).float() == y).float().mean().item()
    # AUC는 문턱값을 하나로 정하지 않고 모든 문턱값에 걸친 성능을 재므로,
    # 확률이 필요하다. 그래서 여기서만 시그모이드를 통과시킨다
    auc = roc_auc_score(y.numpy().ravel(), probs.numpy().ravel())
    return loss, accuracy, auc


def plot_comparison(histories, labels):
    """비교를 위해 학습 이력을 그린다."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    metrics = [
        ('loss', 'Loss'),
        ('accuracy', 'Accuracy'),
        ('val_loss', 'Validation Loss'),
        ('val_accuracy', 'Validation Accuracy')
    ]

    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        # 조기 종료가 걸린 실험은 곡선이 짧다. 그 길이 차이 자체가
        # 몇 에포크를 아꼈는지를 보여 준다
        for history, label in zip(histories, labels):
            if metric in history:
                epochs = range(1, len(history[metric]) + 1)
                ax.plot(epochs, history[metric], label=label, linewidth=2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('early_stopping_comparison.png', dpi=150)
    print("\nPlot saved as 'early_stopping_comparison.png'")


def main():
    # 합성 데이터셋 생성
    print("Generating dataset...")
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )

    # 데이터를 셋으로 나눈다. 검증 집합은 언제 멈출지 정하는 데 쓰고,
    # 시험 집합은 그렇게 고른 모델을 마지막에 한 번 재는 데만 쓴다.
    # 검증 집합으로 멈출 때를 골랐으므로 그 위의 성능은 낙관적이며,
    # 정직한 값을 얻으려면 손대지 않은 시험 집합이 따로 있어야 한다
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # 표준화
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}\n")

    # 텐서로 옮긴다. 목표는 (N, 1)이어야 (N, 1)인 출력과 방송되지 않는다
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).reshape(-1, 1)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val).reshape(-1, 1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).reshape(-1, 1)

    # 모델 학습
    model_no_es, history_no_es = train_without_early_stopping(
        X_train, y_train, X_val, y_val, epochs=200
    )

    model_es_10, history_es_10 = train_with_early_stopping(
        X_train, y_train, X_val, y_val, patience=10, epochs=200
    )

    model_es_20, history_es_20 = train_with_early_stopping(
        X_train, y_train, X_val, y_val, patience=20, epochs=200
    )

    model_advanced, history_advanced = train_with_advanced_early_stopping(
        X_train, y_train, X_val, y_val, epochs=200
    )

    # 시험 집합에서 평가
    print("\n" + "="*60)
    print("Test Set Performance")
    print("="*60)

    models = {
        'No Early Stopping': model_no_es,
        'Early Stop (patience=10)': model_es_10,
        'Early Stop (patience=20)': model_es_20,
        'Advanced Early Stop': model_advanced
    }

    for name, model in models.items():
        loss, accuracy, auc = evaluate(model, X_test, y_test)
        print(f"{name}:")
        print(f"  Test Loss: {loss:.4f}")
        print(f"  Test Accuracy: {accuracy:.4f}")
        print(f"  Test AUC: {auc:.4f}\n")

    # 비교 그림
    histories = [history_no_es, history_es_10, history_es_20, history_advanced]
    labels = [
        'No Early Stopping',
        'Early Stop (p=10)',
        'Early Stop (p=20)',
        'Advanced Early Stop'
    ]
    plot_comparison(histories, labels)

    # 요약 출력
    print("\n" + "="*60)
    print("Key Insights")
    print("="*60)
    print(f"• Without early stopping: Trained for {len(history_no_es['loss'])} epochs")
    print(f"• With early stopping (p=10): Stopped at epoch {len(history_es_10['loss'])}")
    print(f"• With early stopping (p=20): Stopped at epoch {len(history_es_20['loss'])}")
    print(f"• Advanced callbacks: Stopped at epoch {len(history_advanced['loss'])}")
    print("\nBenefits of Early Stopping:")
    print("  1. Prevents overfitting by stopping before validation loss increases")
    print("  2. Saves computational time by not training unnecessary epochs")
    print("  3. Automatically finds optimal number of training epochs")
    print("  4. Can be combined with other techniques (learning rate reduction, etc.)")


if __name__ == "__main__":
    main()
```

## 2. 논의

손실 계산은 모델의 출력을 최적화 목표와 이어 준다. 알맞은 손실 함수를 고르는 일은 결정적으로 중요하다. 손실 함수가 모델이 무엇을 최적화하도록 배울지를 정하며, 학습된 표현과 결정 경계를 직접 빚어내기 때문이다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 학습 최적화 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
코드를 끝까지 읽고 핵심적인 설계 결정을 찾아내라. 구체적인 구현 선택 세 가지를 나열하고, 각각이 정칙화 기법에 왜 적절한지 설명하라.

??? success "연습문제 1 풀이"
    설계 결정은 구현마다 다르지만 흔히 다음이 포함된다. (1) 활성화 함수의 선택 — ReLU 계열은 포화되지 않는 경사를 주어 학습을 빠르게 한다. (2) 정규화 전략 — 배치 정규화는 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 — 있을 경우 건너뛰는 경로를 제공하여 깊은 신경망에서도 경사가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 절충을 반영한다.

---

**연습문제 2.**
입력이 기대하는 모양과 자료형을 갖는지 확인하도록 주 함수나 클래스에 입력 검증을 추가하라. 잘못된 입력에는 유익한 오류 메시지를 내라.

??? success "연습문제 2 풀이"
    `forward` 메서드(또는 해당 함수)의 첫머리에 다음과 같은 검사를 추가한다. `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`와 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. 모양을 검증할 때는 중요한 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 유익한 오류 메시지는 디버깅 속도를 크게 높이고 코드를 재사용하기에도 더 견고하게 만든다.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
조기 종료 예제 구현을 검증하는 종합적인 시험 함수를 작성하라. 빈 입력, 원소가 하나뿐인 입력, 아주 큰 입력, 그리고 극단적인 값(0, 아주 큰 수)을 가진 입력 등 경계 사례를 시험하라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_early stopping example():
        model = Early Stopping Example(...)
        # 보통의 입력
        assert model(normal_input).shape == expected_shape
        # 원소가 하나인 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값 (넘침을 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 경사의 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    경사의 흐름을 시험하는 것은 그 구조가 처음부터 끝까지 이어지는 학습을 지원하는지 확인하는 데 특히 중요하다.

## 정리하며

**다룬 것** — 조기 종료 예제

손실 계산은 모델의 출력을 최적화 목표와 이어 준다.

앞의 연습문제 4개로 직접 확인할 수 있다.
