# 드롭아웃 예제

드롭아웃은 트레인 중에 은닉 유닛의 출력을 확률 $p$로 0으로 만드는 정칙화 기법이다. 같은 이진 분류 MLP에 드롭아웃을 추가해 트레인·검증 지표를 비교한다. 과적합 완화 여부는 마스킹 유무만이 아니라 같은 평가 모드의 성능에서 판단한다.

## 1. 코드

```python
"""
드롭아웃 정칙화 예제
===============================
트레인 중에 뉴런을 무작위로 떨어뜨려 과적합을 막는 방식을 보인다.
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
    """드롭아웃이 없는 뉴럴 네트워크를 만든다."""
    # 마지막 레이어에 시그모이드를 붙이지 않는다. 아래에서 쓰는
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
    """드롭아웃 레이어가 있는 뉴럴 네트워크를 만든다."""
    return nn.Sequential(
        # 선형 → 엑티베이션 → 드롭아웃 차례다. 엑티베이션 뒤에 두어야
        # 엑티베이션 값을 끄는 것이 된다
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(64, 32),
        nn.ReLU(),
        # 세 레이어 모두 0.5로 같다. 너비가 32인 레이어에서는 절반을 끄면 남는
        # 뉴런이 16개뿐이라 꽤 센 편이며, 보통은 뒤로 갈수록 비율을
        # 낮춘다. 여기서는 효과를 뚜렷이 보이려고 일부러 세게 걸었다
        nn.Dropout(dropout_rate),
        # 출력 레이어 뒤에는 드롭아웃이 없다. 예측 자체를 지울 수는 없다
        nn.Linear(32, 1),
    )


def run_epoch(model, loader, criterion, optimizer=None):
    """한 에포크를 돌고 (평균 로스, 정확도)를 돌려준다.

    optimizer가 있으면 트레인 모드로, 없으면 평가 모드로 돈다.
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
            # 짧아도 표본 하나하나가 같은 비중으로 반영되게 하려는 것이다
            total_loss += loss.item() * xb.size(0)
            # 로짓이 0보다 크면 확률이 0.5보다 크다는 뜻이라,
            # 시그모이드를 거치지 않고 로짓을 바로 0과 비교해도 된다
            correct += ((logits > 0).float() == yb).sum().item()
            total += xb.size(0)
    return total_loss / total, correct / total


def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3):
    """모델을 트레인시키고 에포크별 이력을 돌려준다."""
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
    
    # 데이터 나누고 표준화하기
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    # 트레인 집합에는 fit_transform, 시험 집합에는 transform이다. 시험
    # 집합에 fit을 부르면 그 평균과 분산이 트레인에 새어 들어간다.
    # 이 두 메서드를 바꿔 쓰는 것이 데이터 누출의 가장 흔한 통로다
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 텐서로 옮긴다. 목표는 (N, 1) 형태여야 한다. (N,)으로 두면
    # BCEWithLogitsLoss가 (N, 1)인 출력과 형태가 다르다는 오류를 낸다
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).reshape(-1, 1)

    # 트레인 집합의 뒤쪽 20%를 검증용으로 떼어 둔다. 앞서
    # train_test_split이 이미 뒤섞어 두었으므로 뒤에서 자르더라도
    # 한쪽으로 치우치지 않는다
    n_val = int(0.2 * len(X_train_t))
    train_ds = TensorDataset(X_train_t[:-n_val], y_train_t[:-n_val])
    val_ds = TensorDataset(X_train_t[-n_val:], y_train_t[-n_val:])
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32)

    # 모델을 만들기 직전마다 시드를 다시 설정한다. 이렇게 해야 두 모델이
    # 같은 초기 웨이트에서 출발해, 결과의 차이를 드롭아웃 하나로
    # 돌릴 수 있다. 시드를 한 번만 설정하면 초기화 운이 섞여 든다
    print("Training model WITHOUT dropout...")
    torch.manual_seed(42)
    model_no_dropout = create_model_without_dropout(X_train.shape[1])
    history_no_dropout = train_model(model_no_dropout, train_loader, val_loader)

    # 드롭아웃과 함께 모델 트레인
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
    
    # 트레인 이력 그리기
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    # 이 그림에서 볼 것은 최종 로스의 높낮이가 아니라 트레인 곡선과 검증
    # 곡선 사이의 격차다. 트레인 로스가 계속 내려가는데 검증 로스가
    # 돌아서 올라가면 그 격차가 곧 과적합이다. 드롭아웃을 건 쪽에서
    # 그 격차가 좁아지는지를 이 그림에서 확인한다.
    # 드롭아웃 쪽 트레인 로스가 더 높게 나오는 것도 정상이다.
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

두 모델은 로짓을 내며 BCEWithLogitsLoss를 사용한다. 목표는 출력과 같은 `(N, 1)` 부동소수점 텐서로 맞춘다. 이 로스는 `(N,)` 목표를 조용히 브로드캐스팅해 처리하지 않고 형태 불일치 오류를 낸다.

코드는 외부 트레인·시험을 나누고 트레인 부분을 표준화한 다음 내부 검증을 떼어 둔다. 따라서 최종 시험 통계는 쓰지 않았지만 내부 검증의 입력 통계가 scaler를 맞추는(fit) 데 포함된다. 엄격한 검증을 하려면 내부 분할도 먼저 하고 실제 트레인 부분에서만 scaler를 fit해야 한다.

트레인 중 드롭아웃의 무작위 마스크가 지표를 바꾸므로 검증과 직접 비교할 때 모드 차이를 반영한다. 같은 시드로 두 모델을 생성해 초기 웨이트는 같지만, 두 트레인의 배치 순서까지 같아지는 것은 아니다. 드롭아웃 마스크와 `DataLoader`의 셔플이 같은 전역 난수를 쓰므로, 드롭아웃이 있는 쪽은 마스크를 샘플링한 만큼 셔플 순서가 달라진다.

출력에서 두 모델의 시험 정확도는 0.9550으로 같다. 시험 표본이 200개이므로 둘 다 191개를 맞게 분류한 것이다. 곧 이 한 번의 실행은 시험 정확도에서 드롭아웃의 효과를 보여 주지 못한다. 시드 하나로 잰 값이기도 하므로, 효과를 말하려면 시드를 바꿔 여러 번 돌려 표준편차와 함께 보고, 코드가 저장하는 그림(`dropout_comparison.png`, 이 쪽에는 싣지 않았다)에서 트레인 로스와 검증 로스의 격차를 비교해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
BCEWithLogitsLoss의 입력과 목표 형태를 설명하라.

</div>

??? success "풀이"
    이 예제는 `(N, 1)` 로짓과 같은 형태의 부동소수점 0/1 목표를 사용한다. 로스 앞에 시그모이드를 따로 붙이지 않는다. 평가에서 확률이 필요하면 시그모이드로 변환한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
내부 검증 분할과 scaler를 맞추는 순서에서 개선할 점을 설명하라.

</div>

??? success "풀이"
    외부 시험 집합을 나눈 뒤 트레인 부분을 다시 실제 트레인·검증으로 나누고, 실제 트레인 입력에만 scaler를 fit한다. 같은 변환을 검증과 시험에 적용한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
드롭아웃 모델의 트레인 정확도보다 검증 정확도가 높은 이유를 하나 제시하라.

</div>

??? success "풀이"
    트레인 정확도는 드롭아웃 마스크가 켜진 채로 재어 예측이 흔들리고, 검증 정확도는 마스크를 끈 평가 모드로 재기 때문이다. 분할 난이도 차이도 가능하다. 이 현상만으로 데이터 누출이나 더 나은 일반화를 확정하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래의 테스트 코드 조각을 검토하라. 정의되지 않은 클래스와 잘못된 이름, 입력 형태 가정을 찾아 이 페이지의 실제 함수에 맞는 점검 항목을 제안하라.

</div>

??? success "풀이"
    아래 조각은 실제 실행 가능한 테스트가 아니다. 함수 이름과 모델 이름부터 이 예제에 맞게 작성해야 한다.
    ```python
    def test_dropout_example():
        model = create_model_with_dropout(input_dim=20, dropout_rate=0.5)
        # 보통의 입력
        assert model(normal_input).shape == expected_shape
        # 원소가 하나인 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값 (오버플로를 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 그래디언트의 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    이 페이지는 완성된 사용자 정의 모델 클래스를 제공하지 않는다. 실제 함수의 입력·출력 형태, 허용 자료형, 유한한 값과 그래디언트의 유무를 기준으로 검사한다. 빈 입력을 처리하는 방식은 연산마다 다르며, 무조건 정상적인 로스나 그래디언트가 나와야 한다고 가정하지 않는다.

## 정리하며

드롭아웃 효과를 비교할 때 출력·로스 규약, 전처리의 분할 순서와 모드 차이를 함께 확인한다. 최종 시험 지표로 반복해서 설정을 선택하지 않는다.
