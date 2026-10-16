# 조기 종료 예제

고정 횟수 학습과 검증 로스 기반 조기 종료를 비교한다. 검증으로 종료 시점을 선택하고 별도 테스트셋에서 평가한다.

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
    """뉴럴 넷트웍 모델을 만든다."""
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
    """검증 로스가 나아지지 않으면 학습을 멈춘다.

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
                # 그 텐서를 덮어써 "가장 좋았던 웨이트"가 현재 웨이트를
                # 따라 계속 바뀌어 버린다
                self.best_weights = copy.deepcopy(model.state_dict())
            return False

        # 나아지지 않은 에포크를 센다. patience는 "학습을 시작한 뒤"가
        # 아니라 "마지막 개선 뒤" 몇 에포크인지를 뜻한다
        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        """가장 좋았던 웨이트로 되돌린다."""
        # 학습이 인내를 다 채우지 못하고 에포크 수를 소진해 끝났더라도
        # 부르는 쪽에서 이것을 불러 주어야 최선의 웨이트를 되찾는다
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)


def run_epoch(model, loader, criterion, optimizer=None):
    """한 에포크를 돌고 (로스, 정확도)를 돌려준다."""
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
    # 200 에포크를 끝까지 돈다. 검증 로스가 돌아선 뒤에도 계속 학습하므로
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
        # 로스가 history에서 빠진다.
        # 감시하는 것은 학습 로스가 아니라 검증 로스다. 학습 로스는
        # 과적합이 시작된 뒤에도 계속 내려가 멈출 때를 알려 주지 못한다
        if early_stop(vl, model):
            print(f"  Stopped early at epoch {epoch + 1}")
            break
    early_stop.restore(model)
    return model, history


def train_with_advanced_early_stopping(X_train, y_train, X_val, y_val, epochs=200):
    """러닝레이트 감소와 검사점 저장을 함께 쓰는 학습."""
    print("Training with ADVANCED early stopping (multiple callbacks)...")
    torch.manual_seed(42)
    model = create_model(X_train.shape[1])
    train_loader, val_loader = make_loaders(X_train, y_train, X_val, y_val)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())

    # 인내 값이 15와 5로 다른 것이 이 조합의 핵심이다. 러닝레이트 감소가
    # 5에서 먼저 걸려 모델에게 한 번 더 기회를 주고, 그래도 나아지지
    # 않으면 15에서 조기 종료가 걸린다. 두 값을 거꾸로 두면 러닝레이트를
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

        # 스케줄러는 검증 로스를 보고 판단하므로 그 값을 넘겨 준다
        scheduler.step(vl)

        # 검사점 저장. 정확도가 가장 높았던 순간의 웨이트를 따로 담아 둔다.
        # 주의: 이쪽은 정확도를 보는데 위의 조기 종료는 로스를 본다.
        # 두 지표의 최고점이 같은 에포크라는 보장이 없으므로, 여기 담긴
        # best_state와 아래에서 복원되는 웨이트가 서로 다를 수 있다.
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
    """(로스, 정확도, AUC)를 돌려준다."""
    model.eval()
    with torch.no_grad():
        logits = model(X)
        loss = nn.BCEWithLogitsLoss()(logits, y).item()
        probs = torch.sigmoid(logits)
        accuracy = ((logits > 0).float() == y).float().mean().item()
    # AUC는 임계값을 하나로 정하지 않고 모든 임계값에 걸친 성능을 재므로,
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

**출력:**

```
Generating dataset...
Training set: (1400, 20)
Validation set: (300, 20)
Test set: (300, 20)

Training WITHOUT early stopping...
Training WITH early stopping (patience=10)...
  Stopped early at epoch 20
Training WITH early stopping (patience=20)...
  Stopped early at epoch 30
Training with ADVANCED early stopping (multiple callbacks)...
  Stopped early at epoch 25

============================================================
Test Set Performance
============================================================
No Early Stopping:
  Test Loss: 0.5368
  Test Accuracy: 0.9700
  Test AUC: 0.9900

Early Stop (patience=10):
  Test Loss: 0.1034
  Test Accuracy: 0.9767
  Test AUC: 0.9904

Early Stop (patience=20):
  Test Loss: 0.1034
  Test Accuracy: 0.9767
  Test AUC: 0.9904

Advanced Early Stop:
  Test Loss: 0.1034
  Test Accuracy: 0.9767
  Test AUC: 0.9904


Plot saved as 'early_stopping_comparison.png'

============================================================
Key Insights
============================================================
• Without early stopping: Trained for 200 epochs
• With early stopping (p=10): Stopped at epoch 20
• With early stopping (p=20): Stopped at epoch 30
• Advanced callbacks: Stopped at epoch 25

Benefits of Early Stopping:
  1. Prevents overfitting by stopping before validation loss increases
  2. Saves computational time by not training unnecessary epochs
  3. Automatically finds optimal number of training epochs
  4. Can be combined with other techniques (learning rate reduction, etc.)
```

## 2. 논의

EarlyStopping은 이전 에포크보다 좋아졌는지가 아니라 저장한 best_score에서 min_delta보다 많이 개선되었는지를 본다. 그 조건이 충족되지 않은 횟수가 patience에 이르면 멈춘다.

최선 상태는 deepcopy로 별도 저장한다. restore를 호출해야 그 상태로 돌아가며, 단순히 루프에서 break하는 것만으로 복원이 되지는 않는다. min_delta를 넘지 않은 작은 개선은 저장 기준상 개선으로 취급되지 않을 수 있다.

데이터는 학습·검증·테스트로 분리하고 scaler는 학습 입력으로만 맞춘다. 검증 지표는 선택에 사용되므로 마지막 테스트 결과와 역할이 다르다. 스케줄러와 조합할 때는 러닝레이트 변경 후 회복할 시간을 patience가 허용하는지도 살핀다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
best_score=0.50, min_delta=0.01일 때 0.495는 개선으로 인정되는가?

</div>

??? success "연습문제 1 풀이"
    이 구현은 score < best_score - min_delta를 쓰므로 0.495는 인정되지 않는다. 카운터가 증가하고 저장 상태도 바뀌지 않는다. 수치가 작아졌다는 사실과 판정 기준의 개선을 구분한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
깊은 복사와 restore가 각각 필요한 이유를 설명하라.

</div>

??? success "연습문제 2 풀이"
    state_dict의 텐서를 얕게 참조하면 이후 갱신으로 값이 달라질 수 있어 깊은 복사를 사용한다. 중단 시점은 선택된 저장 시점과 다를 수 있으므로 restore로 저장 상태를 다시 불러온다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
검증셋을 테스트셋 대신 사용해야 하는 이유를 설명하라.

</div>

??? success "연습문제 3 풀이"
    종료 시점은 학습된 모델을 선택하는 하이퍼파라미터이다. 같은 데이터에서 선택과 최종 평가를 반복하면 선택 효과가 성능 추정에 들어간다. 별도 테스트는 선택을 마친 모델에 사용한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래의 테스트 코드 조각을 검토하라. 정의되지 않은 클래스와 잘못된 이름, 입력 모양 가정을 찾아 이 페이지의 실제 함수에 맞는 점검 항목을 제안하라.

</div>

??? success "연습문제 4 풀이"
    아래 조각은 실제 실행 가능한 테스트가 아니다. 함수 이름과 모델 이름부터 이 예제에 맞게 작성해야 한다.
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
    이 페이지는 완성된 사용자 정의 모델 클래스를 제공하지 않는다. 실제 함수의 입력·출력 모양, 허용 자료형, 유한한 값과 경사의 유무를 기준으로 검사한다. 빈 입력을 처리하는 방식은 연산마다 다르며, 무조건 정상적인 로스나 경사가 나와야 한다고 가정하지 않는다.

## 정리하며

조기 종료의 결과는 개선 판정, 대기 횟수와 복원 상태에 달려 있다. 이 구현의 깊은 복사와 명시적인 복원을 확인하고 학습·검증·테스트의 역할을 유지한다.
