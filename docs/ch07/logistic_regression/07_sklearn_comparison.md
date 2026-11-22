# scikit-learn과의 비교

이 쪽의 코드는 유전자 발현 피처로 약물 반응 라벨 24개를 하나씩 분류하는 scikit-learn 예제다. PyTorch 모델을 같은 조건에서 트레인해 견주는 부분은 없으므로, 제목의 「비교」는 논의의 「프레임워크 비교를 하려면」에 적은 조건까지로 읽는다.

## 1. 코드

```python
"""scikit-learn과의 비교."""
# ========================================================
# 로지스틱 회귀: 약물 반응 열마다 분류
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

    # 4. 트레인/시험 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.3, random_state=0
    )

    # 5. 로지스틱 회귀 모델 트레인
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

PyTorch 구현을 추가할 때는 같은 표본 분할, 피처 스케일, 절편, 클래스 인코딩, 로스의 합·평균과 정칙화 강도를 맞춰야 한다. scikit-learn의 기본 L2 정칙화와 PyTorch의 정칙화하지 않은 BCE를 그대로 비교하면 목적부터 다르다. 같은 정확도만으로 두 최적화 해가 같다고 판단할 수 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 쪽의 코드가 비교하는 대상과 실제로 포함하지 않는 비교를 설명하라.

</div>

??? success "풀이"
    코드는 scikit-learn 모델을 정답 열마다 적합해 분류 지표를 모은다. PyTorch 분류기를 같은 조건에서 트레인하는 부분은 없다. 프레임워크 비교를 하려면 별도 구현과 같은 목적·분할 조건이 필요하다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
유전자 피처와 약물 반응 정답에서 어떤 입력 검증이 필요한가?

</div>

??? success "풀이"
    ID 정렬과 유일성, 피처의 유한값·결측, 정답의 실제 범주와 클래스 수를 확인한다. 연속 반응값을 임의의 범주로 취급하지 않는다. 각 분할에 트레인할 클래스가 포함되는지도 점검한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
피처 파일과 정답 파일의 행 수가 같으면 ID 열을 삭제해도 되는가?

</div>

??? success "풀이"
    충분하지 않다. 같은 ID 집합, 중복 여부와 행 순서를 확인하거나 ID로 결합해야 한다. 잘못 정렬된 표본은 오류 없이도 다른 사람의 피처와 정답을 연결할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
출력에는 정확도 1.00인 열이 여럿 있다. 이 수만으로 `LogisticRegression`이 그 약물 반응을 잘 예측한다고 말할 수 있는가? 각 정답 열에 대해 **트레인 집합의 다수 클래스만 찍는 베이스라인**의 시험 정확도를 같은 분할(`test_size=0.3, random_state=0`)에서 계산하고 `records`와 견주어라.

</div>

??? success "풀이"
    말할 수 없다. 정답 열은 0과 1뿐인 이진 라벨인데, 한 클래스가 거의 전부인 열에서는 피처를 전혀 보지 않고 그 클래스만 찍어도 정확도가 높게 나온다. 정확도는 같은 분할에서 잰 베이스라인과 견줄 때에만 뜻을 가진다. 이 쪽의 반복문에 두 줄을 더한다.
    ```python
    records, baseline = [], []
    for i in range(Y_data.shape[1]):
        y_data = Y_data.iloc[:, i]
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=0.3, random_state=0
        )
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        records.append(accuracy_score(y_test, model.predict(X_test)))

        majority = y_train.mode()[0]                    # 트레인 집합의 다수 클래스
        baseline.append((y_test == majority).mean())    # 그것만 찍은 시험 정확도

    records, baseline = np.array(records), np.array(baseline)
    for i in range(len(records)):
        print(f"{i:2}-th response: model {records[i]:.2f}  baseline {baseline[i]:.2f}")
    print((records > baseline).sum(), (records == baseline).sum(), (records < baseline).sum())
    ```
    돌려 보면 24개 열 가운데 모델이 베이스라인을 넘는 것은 4개(5 · 9 · 13 · 22번)뿐이고, 13개는 같으며 7개는 오히려 낮다. 정확도 1.00인 다섯 열(4 · 10 · 12 · 16 · 17번)은 베이스라인도 1.00이다 — 시험 집합에 한 클래스만 들어 있었다는 뜻이지 모델이 무엇을 배웠다는 뜻이 아니다. 차이가 가장 큰 열은 13번(0.83 대 0.72)과 22번(0.55 대 0.48)이다. 시험 표본이 87개이고 분할이 하나이므로 0.01~0.03의 차이는 표본 한두 개에 해당한다. 클래스가 치우친 열은 `classification_report`의 클래스별 재현율로 읽어야 한다.

## 정리하며

모델 출력과 로스의 입력 조건을 일치시키고 데이터 분할, 전처리와 지표의 의미를 함께 확인한다. 기록된 실행 결과는 해당 설정의 예시이며 모든 환경의 수렴이나 일반화를 보장하지 않는다.
