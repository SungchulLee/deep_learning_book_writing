# scikit-learn과의 비교

scikit-learn과의 비교.

이 튜토리얼은 PyTorch에서 로지스틱 회귀에 대한 기초적인 이해를 쌓는다. 코드를 따라가 보면 모델을 세우고, 손실 함수를 정의하고, 경사 하강법으로 학습시키고, 분류 과제에서 성능을 평가하는 법을 알 수 있다.

## 1. 코드

```python
"""scikit-learn과의 비교."""
# ========================================================
# logistic_regression/main.py
# ========================================================
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. 데이터 불러오기
X = pd.read_csv("https://raw.githubusercontent.com/SungchulLee/data/refs/heads/main/gene_expression.csv")
Y = pd.read_csv("https://raw.githubusercontent.com/SungchulLee/data/refs/heads/main/drug_response.csv")

# print(f"{X.shape = }") # (288, 35) 288 - number of data, 35 - number of id + inputs
# print(f"{Y.shape = }") # (288, 25) 25 - number of id + outputs

# 2. 공통 ID 열 이름 (예: 'Unnamed: 0') 기준으로 병합
common_id = X.columns[0]  # 첫 번째 열이 ID라고 가정

# 3. 설명 변수 X, 반응 변수 y 분리
#    - X: intercept 포함 (ID 열만 제외)
#    - y: drug_response의 첫 번째 열 (보통은 'response')
X_data = X.drop(columns=[common_id])        # ID 열 제거 → intercept 포함됨
Y_data = Y.drop(columns=[common_id])

print(f"{X_data.shape = }") # (288, 34) 288 - number of data, 34 - number of inputs
print(f"{Y_data.shape = }") # (288, 24) 24 - number of outputs

print("-"*50)

records = []
for i in range(Y_data.shape[1]):
    y_data = Y_data.iloc[:,i]             # 병합된 데이터에서 response 열 추출

    # print(f"{X_data.shape = }")
    # print(f"{y_data.shape = }")

    # 4. 학습/테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.3, random_state=0
    )

    # 5. 로지스틱 회귀 모델 학습
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 6. 예측 및 평가
    y_pred = model.predict(X_test)
    records.append(accuracy_score(y_test, y_pred))
    print(f"Accuracy of {i:2}-th response: {records[-1]:.2f}")
    # print("-"*50)
    # print("Classification Report:")
    # print(classification_report(y_test, y_pred))

print("-"*50)
records = np.array(records)
print(f"{records.max() = :.2f}")
print(f"{records.min() = :.2f}")


if __name__ == "__main__":
    pass
```

## 2. 논의

이 구현은 깔끔하고 읽기 쉬운 PyTorch 코드로 로지스틱 회귀의 핵심 개념을 보여준다. 모듈식 구조 덕분에 개별 구성 요소를 따로 살펴보고 다른 과제나 데이터셋에 맞게 고치기가 쉽다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 분류 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
코드를 끝까지 읽고 핵심적인 설계 결정을 찾아내라. 구체적인 구현 선택 세 가지를 나열하고, 각각이 로지스틱 회귀에 왜 적절한지 설명하라.

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
scikit-learn과의 비교 구현을 검증하는 종합적인 시험 함수를 작성하라. 빈 입력, 원소가 하나뿐인 입력, 아주 큰 입력, 그리고 극단적인 값(0, 아주 큰 수)을 가진 입력 등 경계 사례를 시험하라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_sklearn comparison():
        model = Sklearn Comparison(...)
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

**다룬 것** — scikit-learn과의 비교

이 구현은 깔끔하고 읽기 쉬운 PyTorch 코드로 로지스틱 회귀의 핵심 개념을 보여준다.

앞의 연습문제 4개로 직접 확인할 수 있다.
