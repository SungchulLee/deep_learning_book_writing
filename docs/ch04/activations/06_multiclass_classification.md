# 활성화 함수를 쓰는 다중 클래스 분류

다중 클래스 분류는 이진 분류를 범주가 셋 이상인 문제로 확장한 것이다. 구조상의 핵심 차이는 출력층이 클래스마다 로짓 하나씩을 내고, 모델이 내부에서 로그 소프트맥스를 적용하는 `CrossEntropyLoss`를 쓴다는 점이다. 이 설계는 수치적으로도 안정하고 계산도 효율적이어서 PyTorch에서 다중 클래스 문제를 다루는 표준 방식이다.

## 1. 코드

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

torch.manual_seed(42)
np.random.seed(42)

class MulticlassClassifier(nn.Module):
    """
    다중 클래스 분류 신경망.
    출력: num_classes개의 로짓 (소프트맥스 없음)
    CrossEntropyLoss와 함께 쓴다
    """
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)  # 로짓을 돌려준다 (소프트맥스를 쓰지 마라!)
        return x

def generate_multiclass_data(n_samples=600, n_classes=3):
    X, y = make_blobs(n_samples=n_samples, centers=n_classes,
                      n_features=2, cluster_std=1.0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    return X_train, y_train, X_test, y_test

def train_multiclass(model, X_train, y_train, X_test, y_test, epochs=200, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            predictions = logits.argmax(dim=1)
            train_acc = (predictions == y_train).float().mean()

        model.eval()
        with torch.no_grad():
            test_logits = model(X_test)
            test_loss = criterion(test_logits, y_test)
            test_predictions = test_logits.argmax(dim=1)
            test_acc = (test_predictions == y_test).float().mean()

        train_losses.append(loss.item())
        test_losses.append(test_loss.item())
        train_accs.append(train_acc.item())
        test_accs.append(test_acc.item())

    return train_losses, test_losses, train_accs, test_accs

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = generate_multiclass_data()
    model = MulticlassClassifier(input_size=2, hidden_size=32, num_classes=3)
    train_losses, test_losses, train_accs, test_accs = train_multiclass(
        model, X_train, y_train, X_test, y_test, epochs=200
    )
    print(f"Final Test Accuracy: {test_accs[-1]:.4f}")
```

## 2. 논의

다중 클래스 분류기는 순전파에서 소프트맥스를 적용하지 않고 클래스마다 하나씩 날것의 로짓을 낸다. 이는 PyTorch의 의도적인 설계이다. `CrossEntropyLoss`이 내부에서 `LogSoftmax`와 `NLLLoss`을 합쳐 계산하는데, 이것이 소프트맥스를 따로 적용하는 것보다 수치적으로 안정하다. 목표 이름표는 원-핫 벡터가 아니라 클래스 인덱스(예: 0, 1, 2)를 담은 `LongTensor`로 주어야 한다.

이 구조는 ReLU 활성화를 쓰는 은닉층 두 개를 두는데, 군집이 잘 분리된 블롭 데이터셋에는 이것으로 충분하다. 추론할 때는 로짓에 `argmax(dim=1)`을 취해 점수가 가장 높은 클래스의 인덱스를 예측으로 얻는다. 학습 루프는 과적합을 살피기 위해 학습 집합과 시험 집합 모두에서 손실과 정확도를 추적한다.

흔한 실수는 `CrossEntropyLoss` 앞에서 소프트맥스를 적용하는 것인데, 이러면 소프트맥스가 두 번 적용되어 성능이 나빠진다. 추론에서 (예컨대 확신도 문턱값을 쓰려고) 실제 확률이 필요하다면 예측 시점에만 `torch.softmax(logits, dim=1)`을 적용하고, `CrossEntropyLoss`으로 학습하는 동안에는 결코 적용하지 마라.

## 연습문제

**연습문제 1.**
클래스를 3개 대신 5개 다루도록 코드를 수정하라. 표본 수를 1000개로 늘리고 모델을 학습시켜라. 최종 시험 정확도와 5개 클래스 각각의 클래스별 정확도를 보고하라.

??? success "연습문제 1 풀이"
    데이터 생성 호출과 모델 인스턴스 생성을 다음으로 바꾼다.
    ```python
    X_train, y_train, X_test, y_test = generate_multiclass_data(
        n_samples=1000, n_classes=5
    )
    model = MulticlassClassifier(input_size=2, hidden_size=64, num_classes=5)
    ```
    학습 후 클래스별 정확도를 계산하려면 다음과 같이 한다.
    ```python
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        preds = logits.argmax(dim=1)
        for c in range(5):
            mask = y_test == c
            acc = (preds[mask] == c).float().mean().item()
            print(f"Class {c}: {acc:.4f}")
    ```
    클래스가 많고 특징 공간이 2차원일 때는 `hidden_size`를 64로 키우면 모델이 더 복잡한 결정 경계를 배우는 데 도움이 된다.

---

**연습문제 2.**
`CrossEntropyLoss`이 원-핫 목표가 아니라 `LongTensor` 형태의 클래스 인덱스를 받는 이유를 설명하라. 실수로 원-핫 벡터를 넘기면 어떻게 되는가?

??? success "연습문제 2 풀이"
    PyTorch의 `CrossEntropyLoss`은 `LogSoftmax`와 `NLLLoss`으로 구현되어 있으며, 클래스 인덱스를 써서 정답 클래스의 로그 확률을 효율적으로 골라낸다. 이 함수는 `(N, C)` 모양의 2차원 텐서가 아니라 정수 클래스 이름표를 담은 `(N,)` 모양의 1차원 목표를 기대하므로, 원-핫 벡터를 넘기면 모양이 어긋나거나 손실이 잘못 계산된다. 부드러운 이름표나 원-핫 목표를 써야 한다면 손실을 직접 계산할 수 있다.
    ```python
    log_probs = torch.log_softmax(logits, dim=1)
    loss = -(one_hot * log_probs).sum(dim=1).mean()
    ```

---

**연습문제 3.**
모델의 순전파에 (출력 앞에) 소프트맥스 층을 추가하고 `CrossEntropyLoss`으로 학습시켜라. 최종 시험 정확도를 원래 모델과 비교하고 관찰된 차이를 설명하라.

??? success "연습문제 3 풀이"
    `CrossEntropyLoss` 앞에 소프트맥스를 추가하면 소프트맥스가 두 번(한 번은 명시적으로, 한 번은 손실 함수 안에서) 적용된다. forward 메서드를 다음과 같이 고친다.
    ```python
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)  # 나쁨: 소프트맥스가 두 번
        return x
    ```
    소프트맥스를 두 번 적용하면 출력 분포가 압축되어 경사가 작아지고 정보량이 줄어들므로, 모델은 더 느리게 그리고 더 나쁜 해로 수렴하기 쉽다. 올바른 방법은 날것의 로짓을 반환하고 소프트맥스는 `CrossEntropyLoss`이 내부에서 처리하게 두는 것이다.

## 정리하며

**다룬 것** — 활성화 함수를 쓰는 다중 클래스 분류

다중 클래스 분류기는 순전파에서 소프트맥스를 적용하지 않고 클래스마다 하나씩 날것의 로짓을 낸다.

핵심 클래스는 `MulticlassClassifier`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
