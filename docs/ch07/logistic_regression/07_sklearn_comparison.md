# scikit-learn과의 비교

제공 코드는 유전자 발현 피처로 여러 약물 반응 라벨을 각각 분류하는 scikit-learn 예제다. PyTorch와의 직접 비교 학습은 포함되어 있지 않으므로 제목의 비교 범위를 구분해 읽는다.

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

**출력:**

```
X_data.shape = (288, 34)
Y_data.shape = (288, 24)
--------------------------------------------------
Accuracy of  0-th response: 0.78
Accuracy of  1-th response: 0.94
Accuracy of  2-th response: 0.94
Accuracy of  3-th response: 0.84
Accuracy of  4-th response: 1.00
Accuracy of  5-th response: 0.62
Accuracy of  6-th response: 0.93
Accuracy of  7-th response: 0.99
Accuracy of  8-th response: 0.97
Accuracy of  9-th response: 0.95
Accuracy of 10-th response: 1.00
Accuracy of 11-th response: 0.93
Accuracy of 12-th response: 1.00
Accuracy of 13-th response: 0.83
Accuracy of 14-th response: 0.99
Accuracy of 15-th response: 0.97
Accuracy of 16-th response: 1.00
Accuracy of 17-th response: 1.00
Accuracy of 18-th response: 0.79
Accuracy of 19-th response: 0.95
Accuracy of 20-th response: 0.92
Accuracy of 21-th response: 0.99
Accuracy of 22-th response: 0.55
Accuracy of 23-th response: 0.98
--------------------------------------------------
records.max() = 1.00
records.min() = 0.55
```

## 2. 논의

### 표본 ID와 정답 열

피처와 정답 파일의 행이 같은 표본 순서라고 가정하고 ID 열을 제거한다. 실제 데이터에서는 ID의 유일성과 일치 여부를 확인하고 ID로 정렬·결합해야 한다. 행 수가 같다는 사실만으로 같은 표본이 대응된다고 볼 수 없다.

### 약물별 독립 분류

각 정답 열에 대해 따로 `LogisticRegression`을 적합한다. 정답이 연속 반응값인지 범주 라벨인지 먼저 확인하고, 결측·클래스 수와 분할별 분포를 점검한다. 코드가 출력하는 지표는 해당 분할의 분류 성능이며 임상 효능이나 치료 선택의 검증 결과로 해석하지 않는다.

### 프레임워크 비교를 하려면

PyTorch 구현을 추가할 때는 같은 표본 분할, 피처 스케일, 절편, 클래스 인코딩, 로스의 합·평균과 정규화 강도를 맞춰야 한다. scikit-learn의 기본 L2 정규화와 PyTorch의 비정규화 BCE를 그대로 비교하면 목적부터 다르다. 같은 정확도만으로 두 최적화 해가 같다고 판단할 수 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
제공 코드가 비교하는 대상과 실제로 포함하지 않는 비교를 설명하라.

</div>

??? success "연습문제 1 풀이"
    코드는 scikit-learn 모델을 정답 열마다 적합해 분류 지표를 모은다. PyTorch 분류기를 같은 조건에서 학습하는 부분은 없다. 프레임워크 비교를 하려면 별도 구현과 같은 목적·분할 조건이 필요하다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
유전자 피처와 약물 반응 정답에서 어떤 입력 검증이 필요한가?

</div>

??? success "연습문제 2 풀이"
    ID 정렬과 유일성, 피처의 유한값·결측, 정답의 실제 범주와 클래스 수를 확인한다. 연속 반응값을 임의의 범주로 취급하지 않는다. 각 분할에 학습할 클래스가 포함되는지도 점검한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
피처 파일과 정답 파일의 행 수가 같으면 ID 열을 삭제해도 되는가?

</div>

??? success "연습문제 3 풀이"
    충분하지 않다. 같은 ID 집합, 중복 여부와 행 순서를 확인하거나 ID로 결합해야 한다. 잘못 정렬된 표본은 오류 없이도 다른 사람의 피처와 정답을 연결할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
scikit-learn과의 비교 구현을 검증하는 종합적인 시험 함수를 작성하라. 빈 입력, 원소가 하나뿐인 입력, 아주 큰 입력, 그리고 극단적인 값(0, 아주 큰 수)을 가진 입력 등 경계 사례를 시험하라.

</div>

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

모델 출력과 로스의 입력 조건을 일치시키고 데이터 분할, 전처리와 지표의 의미를 함께 확인한다. 기록된 실행 결과는 해당 설정의 예시이며 모든 환경의 수렴이나 일반화를 보장하지 않는다.
