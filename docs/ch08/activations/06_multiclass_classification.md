# 엑티베이션 함수를 쓰는 다중 클래스 분류

다중 클래스 분류는 이진 분류를 범주가 셋 이상인 문제로 확장한 것이다. 구조상의 핵심 차이는 출력 레이어가 클래스마다 로짓 하나씩을 내고, 모델이 내부에서 로그 소프트맥스를 적용하는 `CrossEntropyLoss`를 쓴다는 점이다. 이 설계는 수치적으로도 안정하고 계산도 효율적이어서 PyTorch에서 다중 클래스 문제를 다루는 표준 방식이다.

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
    다중 클래스 분류 뉴럴 넷트웍.
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
    # 피처가 2개뿐이라 산점도로 그려 볼 수 있고, cluster_std=1.0인
    # 블롭 세 덩이는 거의 겹치지 않는다. 아래에서 시험 정확도가
    # 1.0000이 나오는 것은 모델이 뛰어나서가 아니라 과제가 쉬워서다
    X, y = make_blobs(n_samples=n_samples, centers=n_classes,
                      n_features=2, cluster_std=1.0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # 입력은 실수라 FloatTensor, 이름표는 클래스 번호라 LongTensor다.
    # CrossEntropyLoss는 원-핫 벡터가 아니라 0, 1, 2 같은 정수 인덱스를
    # 요구하며, 이름표를 FloatTensor로 넘기면 형 오류가 난다
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    return X_train, y_train, X_test, y_test

def train_multiclass(model, X_train, y_train, X_test, y_test, epochs=200, lr=0.01):
    # CrossEntropyLoss는 LogSoftmax와 NLLLoss를 하나로 묶은 것이다.
    # 그래서 모델이 소프트맥스를 거치지 않은 날것의 로짓을 내야 한다
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    for epoch in range(epochs):
        # 여기서도 미니배치를 나누지 않고 480개를 통째로 넣는다.
        # 에포크 하나가 경사 하강 한 스텝이다
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # argmax(dim=1)은 클래스 축에서 점수가 가장 큰 자리를 고른다.
            # 소프트맥스는 순서를 바꾸지 않으므로 예측을 얻는 데는
            # 로짓의 argmax만으로 충분하다.
            # 다만 이 logits는 step() 앞에서 계산한 값이라, train_acc는
            # 갱신 전 웨이트의 정확도이고 아래 test_acc는 갱신 후 값이다
            predictions = logits.argmax(dim=1)
            train_acc = (predictions == y_train).float().mean()

        # 이 모델에는 드롭아웃도 배치 정규화도 없어서 train()과 eval()이
        # 실제로 하는 일은 없다. 그래도 습관으로 남겨 두면, 나중에
        # 드롭아웃을 넣었을 때 평가가 조용히 틀어지는 일을 막는다
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
    # input_size는 피처 수, num_classes는 출력 로짓 수다. 이 둘은
    # 데이터가 정하는 값이고, hidden_size만 우리가 고르는 값이다
    model = MulticlassClassifier(input_size=2, hidden_size=32, num_classes=3)
    train_losses, test_losses, train_accs, test_accs = train_multiclass(
        model, X_train, y_train, X_test, y_test, epochs=200
    )
    print(f"Final Test Accuracy: {test_accs[-1]:.4f}")
```

**출력:**

```
Final Test Accuracy: 1.0000
```

## 2. 논의

모델은 클래스마다 로짓 하나를 출력하고 CrossEntropyLoss로 학습한다. 이 예제는 정답 인덱스가 담긴 long 텐서를 사용한다. PyTorch는 로짓과 같은 모양의 부동소수점 확률 목표도 지원하므로 원핫 목표를 항상 금지하는 것은 아니다. 정수 원핫과 유효한 확률 텐서를 구분해야 한다.

ReLU 은닉 레이어는 비선형 표현을 추가한다. 예측 클래스만 필요하면 로짓의 argmax를 사용하고, 확률이 필요하면 소프트맥스를 적용한다. 매 에포크 확인하는 테스트 지표로 설정을 조정하면 테스트셋도 선택에 사용되므로 별도의 검증 분할이 필요하다.

흔한 실수는 `CrossEntropyLoss` 앞에서 소프트맥스를 적용하는 것인데, 이러면 소프트맥스가 두 번 적용되어 성능이 나빠진다. 추론에서 (예컨대 확신도 임계값을 쓰려고) 실제 확률이 필요하다면 예측 시점에만 `torch.softmax(logits, dim=1)`을 적용하고, `CrossEntropyLoss`으로 학습하는 동안에는 결코 적용하지 마라.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
클래스를 3개 대신 5개 다루도록 코드를 수정하라. 표본 수를 1000개로 늘리고 모델을 학습시켜라. 최종 시험 정확도와 5개 클래스 각각의 클래스별 정확도를 보고하라.

</div>

??? success "풀이"
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
    클래스가 많고 피처 공간이 2차원일 때는 `hidden_size`를 64로 키우면 모델이 더 복잡한 결정 경계를 배우는 데 도움이 된다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
CrossEntropyLoss가 클래스 인덱스와 확률 목표를 받는 두 모드를 설명하라. 원핫 목표의 자료형이 왜 중요한가?

</div>

??? success "풀이"
    `(N, C)` 로짓에 인덱스 목표를 사용하면 `(N,)` long 텐서를 전달한다. 확률 목표는 로짓과 같은 `(N, C)` 모양의 부동소수점 텐서로 전달한다. 각 행이 유효한 분포여야 하며 정수 원핫 텐서를 그대로 넣는 방식과 다르다. 아래 코드는 확률 목표의 교차 엔트로피를 직접 계산하는 표현이다.
    ```python
    log_probs = torch.log_softmax(logits, dim=1)
    loss = -(one_hot * log_probs).sum(dim=1).mean()
    ```

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
모델의 순전파에 (출력 앞에) 소프트맥스 레이어를 추가하고 `CrossEntropyLoss`으로 학습시켜라. 최종 시험 정확도를 원래 모델과 비교하고 관찰된 차이를 설명하라.

</div>

??? success "풀이"
    `CrossEntropyLoss` 앞에 소프트맥스를 추가하면 소프트맥스가 두 번(한 번은 명시적으로, 한 번은 로스 함수 안에서) 적용된다. forward 메서드를 다음과 같이 고친다.
    ```python
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)  # 나쁨: 소프트맥스가 두 번
        return x
    ```
    소프트맥스 확률을 로짓으로 해석하면 목적함수와 경사가 달라진다. 변환 후의 점수가 0~1로 제한되어 표현 가능한 로스 범위도 달라진다. 실제 성능 차이는 실험 조건에 달려 있으며, 원래 의도한 학습에는 날것의 로짓을 전달한다.

## 정리하며

다중 클래스 분류는 클래스별 로짓과 CrossEntropyLoss를 연결한다. 정답 인덱스 모드와 확률 목표 모드의 모양·자료형을 구분하고, 소프트맥스 확률을 다시 로짓 로스에 전달하지 않는다. 검증과 테스트의 역할도 분리한다.
