# 드롭아웃 예제

같은 이진 분류 MLP에 드롭아웃을 추가해 학습·검증 지표를 비교한다. 과적합 완화 여부는 마스킹 유무만이 아니라 같은 평가 모드의 성능에서 판단한다.

## 1. 코드

```python
"""
드롭아웃 정칙화 예제
===============================
학습 중에 뉴런을 무작위로 떨어뜨려 과적합을 막는 방식을 보인다.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ========================================================================
# 메인
# ========================================================================


def create_model_without_dropout(input_dim):
    """드롭아웃이 없는 신경망을 만든다."""
    # 마지막 층에 시그모이드를 붙이지 않는다. 아래에서 쓰는
    # BCEWithLogitsLoss가 시그모이드를 안에 품고 있어, 여기서 또 걸면
    # 두 번 적용될뿐더러 수치적으로도 불안정해진다
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )


def create_model_with_dropout(input_dim, dropout_rate=0.5):
    """드롭아웃 층이 있는 신경망을 만든다."""
    return nn.Sequential(
        # 선형 → 활성화 → 드롭아웃 차례다. 활성화 뒤에 두어야
        # 활성값을 끄는 것이 된다
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(64, 32),
        nn.ReLU(),
        # 세 층 모두 0.5로 같다. 폭이 32인 층에서는 절반을 끄면 남는
        # 뉴런이 16개뿐이라 꽤 센 편이며, 보통은 뒤로 갈수록 비율을
        # 낮춘다. 여기서는 효과를 뚜렷이 보이려고 일부러 세게 걸었다
        nn.Dropout(dropout_rate),
        # 출력층 뒤에는 드롭아웃이 없다. 예측 자체를 지울 수는 없다
        nn.Linear(32, 1),
    )


def run_epoch(model, loader, criterion, optimizer=None):
    """한 에포크를 돌고 (평균 손실, 정확도)를 돌려준다.

    optimizer가 있으면 학습 모드로, 없으면 평가 모드로 돈다.
    """
    # 드롭아웃을 켜고 끄는 것은 이 한 줄이다. train()에서는 뉴런을
    # 무작위로 끄고 eval()에서는 끄지 않는다. 이것을 빠뜨리면 평가할
    # 때도 뉴런이 꺼져 같은 입력에 매번 다른 답이 나온다
    model.train() if optimizer is not None else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    # 평가할 때는 계산 그래프를 만들지 않아 메모리와 시간을 아낀다
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for xb, yb in loader:
            logits = model(xb)
            loss = criterion(logits, yb)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 배치 크기를 곱해 두었다가 표본 수로 나눈다. 마지막 배치가
            # 짧아도 표본 하나하나가 같은 무게를 갖게 하려는 것이다
            total_loss += loss.item() * xb.size(0)
            # 로짓이 0보다 크면 확률이 0.5보다 크다는 뜻이라,
            # 시그모이드를 거치지 않고 바로 견주어도 된다
            correct += ((logits > 0).float() == yb).sum().item()
            total += xb.size(0)
    return total_loss / total, correct / total


def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3):
    """모델을 학습시키고 에포크별 이력을 돌려준다."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    for _ in range(epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)
        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
    return history


def main():
    # 합성 데이터셋 생성
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                                n_redundant=5, random_state=42)
    
    # 데이터 나누고 척도 맞추기
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    # 학습 집합에는 fit_transform, 시험 집합에는 transform이다. 시험
    # 집합에 fit을 부르면 그 평균과 분산이 학습에 새어 들어간다.
    # 이 두 메서드를 바꿔 쓰는 것이 데이터 누출의 가장 흔한 통로다
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 텐서로 옮긴다. 목표는 (N, 1) 모양이어야 한다. (N,)으로 두면
    # BCEWithLogitsLoss가 (N, 1)인 출력과 방송되어 조용히 틀린 값을 낸다
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).reshape(-1, 1)

    # 학습 집합의 뒤쪽 20%를 검증용으로 떼어 둔다. 앞서
    # train_test_split이 이미 뒤섞어 두었으므로 뒤에서 자르더라도
    # 한쪽으로 치우치지 않는다
    n_val = int(0.2 * len(X_train_t))
    train_ds = TensorDataset(X_train_t[:-n_val], y_train_t[:-n_val])
    val_ds = TensorDataset(X_train_t[-n_val:], y_train_t[-n_val:])
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32)

    # 모델을 만들기 직전마다 씨앗을 다시 심는다. 이렇게 해야 두 모델이
    # 같은 초기 가중치에서 출발해, 결과의 차이를 드롭아웃 하나로
    # 돌릴 수 있다. 씨앗을 한 번만 심으면 초기화 운이 섞여 든다
    print("Training model WITHOUT dropout...")
    torch.manual_seed(42)
    model_no_dropout = create_model_without_dropout(X_train.shape[1])
    history_no_dropout = train_model(model_no_dropout, train_loader, val_loader)

    # 드롭아웃과 함께 모델 학습
    print("Training model WITH dropout...")
    torch.manual_seed(42)
    model_with_dropout = create_model_with_dropout(X_train.shape[1], dropout_rate=0.5)
    history_with_dropout = train_model(model_with_dropout, train_loader, val_loader)

    # 모델 평가.
    # run_epoch에 optimizer를 넘기지 않으므로 평가 모드로 돌고,
    # 드롭아웃이 꺼진 채 재어진다
    criterion = nn.BCEWithLogitsLoss()
    _, test_acc_no_dropout = run_epoch(model_no_dropout, test_loader, criterion)
    _, test_acc_with_dropout = run_epoch(model_with_dropout, test_loader, criterion)
    
    print(f"\nTest Accuracy without Dropout: {test_acc_no_dropout:.4f}")
    print(f"Test Accuracy with Dropout: {test_acc_with_dropout:.4f}")
    
    # 학습 이력 그리기
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    # 이 그림에서 볼 것은 최종 손실의 높낮이가 아니라 두 곡선의 벌어짐이다.
    # 드롭아웃이 없는 쪽은 학습 손실이 계속 내려가는데 검증 손실이
    # 돌아서며 간격이 벌어진다. 그 벌어짐이 곧 과적합이고,
    # 드롭아웃을 건 쪽에서 그 간격이 좁아지는 것이 이 예제의 요점이다.
    # 드롭아웃 쪽 학습 손실이 더 높게 나오는 것도 정상이다.
    # 뉴런을 꺼 둔 채 잰 값이기 때문이다
    plt.plot(history_no_dropout['loss'], label='Train Loss (No Dropout)')
    plt.plot(history_no_dropout['val_loss'], label='Val Loss (No Dropout)')
    plt.plot(history_with_dropout['loss'], label='Train Loss (With Dropout)', linestyle='--')
    plt.plot(history_with_dropout['val_loss'], label='Val Loss (With Dropout)', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history_no_dropout['accuracy'], label='Train Acc (No Dropout)')
    plt.plot(history_no_dropout['val_accuracy'], label='Val Acc (No Dropout)')
    plt.plot(history_with_dropout['accuracy'], label='Train Acc (With Dropout)', linestyle='--')
    plt.plot(history_with_dropout['val_accuracy'], label='Val Acc (With Dropout)', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('dropout_comparison.png')
    print("\nPlot saved as 'dropout_comparison.png'")


if __name__ == "__main__":
    main()
```

**출력:**

```
Training model WITHOUT dropout...
Training model WITH dropout...

Test Accuracy without Dropout: 0.9550
Test Accuracy with Dropout: 0.9550

Plot saved as 'dropout_comparison.png'
```

## 2. 논의

두 모델은 로짓을 내며 BCEWithLogitsLoss를 사용한다. 목표는 출력과 같은 `(N, 1)` 부동소수점 텐서로 맞춘다. 이 손실은 `(N,)` 목표를 조용히 브로드캐스팅해 처리하는 대신 모양 불일치 오류를 낼 수 있다.

코드는 외부 학습·테스트를 나누고 학습 부분을 표준화한 다음 내부 검증을 떼어 둔다. 따라서 최종 테스트 통계는 쓰지 않았지만 내부 검증의 입력 통계가 scaler 학습에 포함된다. 엄격한 검증을 하려면 내부 분할도 먼저 하고 실제 학습 부분에서만 scaler를 fit해야 한다.

학습 중 드롭아웃의 무작위 마스크가 지표를 바꾸므로 검증과 직접 비교할 때 모드 차이를 반영한다. 같은 시드로 모델을 생성해도 데이터 순서와 마스크가 모든 실행에서 같아지는 것은 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
BCEWithLogitsLoss의 입력과 목표 모양을 설명하라.

</div>

??? success "연습문제 1 풀이"
    이 예제는 `(N, 1)` 로짓과 같은 모양의 부동소수점 0/1 목표를 사용한다. 손실 앞에 시그모이드를 따로 붙이지 않는다. 평가에서 확률이 필요하면 시그모이드로 변환한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
내부 검증 분할과 scaler 학습 순서에서 개선할 점을 설명하라.

</div>

??? success "연습문제 2 풀이"
    외부 테스트를 나눈 뒤 학습 부분을 다시 실제 학습·검증으로 나누고, 실제 학습 입력에만 scaler를 fit한다. 같은 변환을 검증과 테스트에 적용한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
드롭아웃 모델의 학습 정확도보다 검증 정확도가 높은 이유를 하나 제시하라.

</div>

??? success "연습문제 3 풀이"
    학습 중에는 마스크가 활성화되어 예측이 흔들리고 평가에서는 꺼지기 때문이다. 분할 난이도 차이도 가능하다. 이 현상만으로 데이터 누출이나 더 나은 일반화를 확정하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래의 테스트 코드 조각을 검토하라. 정의되지 않은 클래스와 잘못된 이름, 입력 모양 가정을 찾아 이 페이지의 실제 함수에 맞는 점검 항목을 제안하라.

</div>

??? success "연습문제 4 풀이"
    아래 조각은 실제 실행 가능한 테스트가 아니다. 함수 이름과 모델 이름부터 이 예제에 맞게 작성해야 한다.
    ```python
    def test_dropout_example():
        model = create_model_with_dropout(input_dim=20, dropout_rate=0.5)
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
    이 페이지는 완성된 사용자 정의 모델 클래스를 제공하지 않는다. 실제 함수의 입력·출력 모양, 허용 자료형, 유한한 값과 경사의 유무를 기준으로 검사한다. 빈 입력을 처리하는 방식은 연산마다 다르며, 무조건 정상적인 손실이나 경사가 나와야 한다고 가정하지 않는다.

## 정리하며

드롭아웃 효과를 비교할 때 출력·손실 규약, 전처리의 분할 순서와 모드 차이를 함께 확인한다. 최종 테스트 지표로 반복해서 설정을 선택하지 않는다.
