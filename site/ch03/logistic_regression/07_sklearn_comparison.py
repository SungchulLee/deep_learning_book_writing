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