# 종합 예제

처음부터 끝까지 이어지는 모델 평가 작업 흐름은 데이터 분할, 교차 검증, 지표 계산, 결과 해석을 하나의 파이프라인으로 엮는다. 이 스크립트는 이진 분류(사기 탐지), 다중 클래스 분류(붓꽃 품종), 회귀(집값 예측)에 대한 완전한 평가를 보인다.

## 코드

```python
"""
종합 예제 - 처음부터 끝까지의 모델 평가
=====================================================

모델 평가의 작업 흐름을 처음부터 끝까지 보이는 완전한 예제.

예제:
1. 이진 분류: 신용카드 사기 탐지
2. 다중 클래스 분류: 붓꽃 품종
3. 회귀: 집값 예측
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (

# ========================================================================
# 메인
# ========================================================================
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)


class Example1_BinaryClassification:
    """
    예제: 신용카드 사기 탐지 (이진 분류)
    
    사업적 맥락:
    - 사기성 신용카드 거래 탐지
    - 거짓 음성(사기를 놓치는 것)의 비용이 크다
    - 거짓 양성(정상 거래를 막는 것)은 고객 경험을 해친다
    - 사기를 잡으려면 재현율이 높아야 하되 정밀도도 알맞게 지켜야 한다
    """
    
    @staticmethod
    def run():
        print("=" * 80)
        print("EXAMPLE 1: BINARY CLASSIFICATION - CREDIT CARD FRAUD DETECTION")
        print("=" * 80)
        
        # 불균형한 합성 데이터셋 생성
        print("\n1. GENERATING DATA")
        print("-" * 80)
        X, y = make_classification(
            n_samples=10000, n_features=20, n_informative=15,
            n_redundant=5, n_classes=2, weights=[0.95, 0.05],  # 사기 5%
            random_state=42
        )
        
        print(f"Total samples: {len(X)}")
        print(f"Fraudulent transactions: {sum(y)} ({sum(y)/len(y)*100:.2f}%)")
        print(f"Legitimate transactions: {len(y)-sum(y)} ({(len(y)-sum(y))/len(y)*100:.2f}%)")
        print("⚠ HIGHLY IMBALANCED DATASET")
        
        # 데이터 나누기
        print("\n2. SPLITTING DATA")
        print("-" * 80)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # 특징의 척도 조정
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 모델 학습
        print("\n3. TRAINING MODELS")
        print("-" * 80)
        
        # 모델 1: 로지스틱 회귀
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train_scaled, y_train)
        print("✓ Logistic Regression trained")
        
        # 모델 2: 랜덤 포리스트
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        print("✓ Random Forest trained")
        
        # 교차 검증
        print("\n4. CROSS-VALIDATION")
        print("-" * 80)
        skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        lr_scores = cross_val_score(lr_model, X_train_scaled, y_train, 
                                   cv=skfold, scoring='f1')
        rf_scores = cross_val_score(rf_model, X_train_scaled, y_train,
                                   cv=skfold, scoring='f1')
        
        print(f"Logistic Regression F1 Score: {lr_scores.mean():.4f} (+/- {lr_scores.std()*2:.4f})")
        print(f"Random Forest F1 Score: {rf_scores.mean():.4f} (+/- {rf_scores.std()*2:.4f})")
        
        # 예측
        print("\n5. TEST SET EVALUATION")
        print("-" * 80)
        
        y_pred_lr = lr_model.predict(X_test_scaled)
        y_pred_rf = rf_model.predict(X_test_scaled)
        
        y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
        y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]
        
        # 평가
        print("\nLOGISTIC REGRESSION:")
        print(f"  Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
        print(f"  Precision: {precision_score(y_test, y_pred_lr):.4f}")
        print(f"  Recall: {recall_score(y_test, y_pred_lr):.4f}")
        print(f"  F1-Score: {f1_score(y_test, y_pred_lr):.4f}")
        print(f"  ROC-AUC: {roc_auc_score(y_test, y_pred_proba_lr):.4f}")
        
        print("\nRANDOM FOREST:")
        print(f"  Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
        print(f"  Precision: {precision_score(y_test, y_pred_rf):.4f}")
        print(f"  Recall: {recall_score(y_test, y_pred_rf):.4f}")
        print(f"  F1-Score: {f1_score(y_test, y_pred_rf):.4f}")
        print(f"  ROC-AUC: {roc_auc_score(y_test, y_pred_proba_rf):.4f}")
        
        # 혼동 행렬
        print("\n6. CONFUSION MATRIX (Random Forest)")
        print("-" * 80)
        cm = confusion_matrix(y_test, y_pred_rf)
        print(cm)
        
        tn, fp, fn, tp = cm.ravel()
        print(f"\nTrue Negatives (Legitimate correctly identified): {tn}")
        print(f"False Positives (Legitimate flagged as fraud): {fp}")
        print(f"False Negatives (Fraud missed): {fn}")
        print(f"True Positives (Fraud caught): {tp}")
        
        # 사업적 해석
        print("\n7. BUSINESS INTERPRETATION")
        print("-" * 80)
        print(f"Out of {sum(y_test)} fraudulent transactions:")
        print(f"  ✓ Caught: {tp} ({tp/sum(y_test)*100:.1f}%)")
        print(f"  ✗ Missed: {fn} ({fn/sum(y_test)*100:.1f}%)")
        print(f"\nOut of {len(y_test)-sum(y_test)} legitimate transactions:")
        print(f"  ✓ Approved: {tn} ({tn/(len(y_test)-sum(y_test))*100:.1f}%)")
        print(f"  ✗ Incorrectly flagged: {fp} ({fp/(len(y_test)-sum(y_test))*100:.1f}%)")
        
        print("\n" + "=" * 80)


class Example2_MulticlassClassification:
    """
    예제: 붓꽃 품종 분류 (다중 클래스)
    
    문제: 측정값으로 붓꽃을 세 품종으로 분류하기
    """
    
    @staticmethod
    def run():
        print("\n\n" + "=" * 80)
        print("EXAMPLE 2: MULTI-CLASS CLASSIFICATION - IRIS SPECIES")
        print("=" * 80)
        
        # 데이터를 불러온다
        print("\n1. LOADING DATA")
        print("-" * 80)
        iris = load_iris()
        X, y = iris.data, iris.target
        class_names = iris.target_names
        
        print(f"Total samples: {len(X)}")
        print(f"Features: {iris.feature_names}")
        print(f"Classes: {class_names}")
        
        # 데이터 나누기
        print("\n2. SPLITTING DATA")
        print("-" * 80)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
        
        # 모델을 학습시킨다
        print("\n3. TRAINING MODEL")
        print("-" * 80)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        print("✓ Random Forest trained")
        
        # 교차 검증
        print("\n4. STRATIFIED CROSS-VALIDATION")
        print("-" * 80)
        skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=skfold, scoring='accuracy')
        print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
        
        # 예측
        y_pred = model.predict(X_test)
        
        # 평가
        print("\n5. TEST SET EVALUATION")
        print("-" * 80)
        print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        
        print("\nPer-Class Metrics:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # 혼동 행렬
        print("\n6. CONFUSION MATRIX")
        print("-" * 80)
        cm = confusion_matrix(y_test, y_pred)
        
        print("             Predicted")
        print("             ", "  ".join([f"{name[:4]:>5}" for name in class_names]))
        print("Actual")
        for i, row in enumerate(cm):
            print(f"{class_names[i][:10]:10}", "  ".join([f"{val:5}" for val in row]))
        
        print("\n" + "=" * 80)


class Example3_Regression:
    """
    예제: 집값 예측 (회귀)
    
    문제: 특징으로 집값 예측하기
    """
    
    @staticmethod
    def run():
        print("\n\n" + "=" * 80)
        print("EXAMPLE 3: REGRESSION - HOUSE PRICE PREDICTION")
        print("=" * 80)
        
        # 합성 데이터 생성
        print("\n1. GENERATING DATA")
        print("-" * 80)
        from sklearn.datasets import make_regression
        X, y = make_regression(
            n_samples=1000, n_features=10, n_informative=8,
            noise=10, random_state=42
        )
        
        # 현실적인 집값으로 척도 맞추기
        y = (y - y.min()) / (y.max() - y.min()) * 500000 + 200000
        
        print(f"Total samples: {len(X)}")
        print(f"Features: 10 (square feet, bedrooms, location score, etc.)")
        print(f"Price range: ${y.min():,.0f} - ${y.max():,.0f}")
        print(f"Mean price: ${y.mean():,.0f}")
        
        # 데이터 나누기
        print("\n2. SPLITTING DATA")
        print("-" * 80)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 특징의 척도 조정
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 모델 학습
        print("\n3. TRAINING MODELS")
        print("-" * 80)
        
        ridge_model = Ridge(alpha=1.0, random_state=42)
        ridge_model.fit(X_train_scaled, y_train)
        print("✓ Ridge Regression trained")
        
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        print("✓ Random Forest trained")
        
        # 교차 검증
        print("\n4. CROSS-VALIDATION")
        print("-" * 80)
        
        ridge_scores = cross_val_score(ridge_model, X_train_scaled, y_train,
                                      cv=5, scoring='r2')
        rf_scores = cross_val_score(rf_model, X_train_scaled, y_train,
                                   cv=5, scoring='r2')
        
        print(f"Ridge R² Score: {ridge_scores.mean():.4f} (+/- {ridge_scores.std()*2:.4f})")
        print(f"Random Forest R² Score: {rf_scores.mean():.4f} (+/- {rf_scores.std()*2:.4f})")
        
        # 예측
        y_pred_ridge = ridge_model.predict(X_test_scaled)
        y_pred_rf = rf_model.predict(X_test_scaled)
        
        # 평가
        print("\n5. TEST SET EVALUATION")
        print("-" * 80)
        
        print("\nRIDGE REGRESSION:")
        print(f"  MAE: ${mean_absolute_error(y_test, y_pred_ridge):,.2f}")
        print(f"  RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred_ridge)):,.2f}")
        print(f"  R² Score: {r2_score(y_test, y_pred_ridge):.4f}")
        
        print("\nRANDOM FOREST:")
        print(f"  MAE: ${mean_absolute_error(y_test, y_pred_rf):,.2f}")
        print(f"  RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred_rf)):,.2f}")
        print(f"  R² Score: {r2_score(y_test, y_pred_rf):.4f}")
        
        # 예측 표본
        print("\n6. SAMPLE PREDICTIONS (Random Forest)")
        print("-" * 80)
        print(f"{'Actual Price':>15} {'Predicted Price':>17} {'Error':>12}")
        print("-" * 50)
        
        for i in range(min(10, len(y_test))):
            actual = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
            predicted = y_pred_rf[i]
            error = actual - predicted
            print(f"${actual:>14,.0f} ${predicted:>16,.0f} ${error:>11,.0f}")
        
        print("\n" + "=" * 80)


def run_all_examples():
    """
    모든 종합 예제 실행
    """
    print("\n" + "#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + " " * 20 + "COMPREHENSIVE EXAMPLES" + " " * 37 + "#")
    print("#" + " " * 15 + "End-to-End Model Evaluation Workflows" + " " * 28 + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)
    
    # 모든 예제 실행
    Example1_BinaryClassification.run()
    Example2_MulticlassClassification.run()
    Example3_Regression.run()
    
    print("\n\n" + "#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + " " * 25 + "ALL EXAMPLES COMPLETED" + " " * 31 + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)
    
    print("\n📚 KEY TAKEAWAYS:")
    print("=" * 80)
    print("1. Always use appropriate metrics for your problem type and business context")
    print("2. Use stratified splitting for classification, especially with imbalanced data")
    print("3. Cross-validation provides more reliable performance estimates than single split")
    print("4. Consider both model performance AND business implications")
    print("5. Confusion matrices reveal which errors your model makes")
    print("6. Compare multiple models before settling on one")
    print("7. Document your evaluation methodology for reproducibility")
    print("=" * 80)


if __name__ == "__main__":
    run_all_examples()```

## 논의

이진 분류 예제(사기 탐지)는 불균형한 데이터에서 지표 선택이 왜 중요한지 보여 준다. 정확도 95%를 내는 모델이 모든 거래를 "정상"이라 예측하여 사기를 하나도 못 잡는 것일 수도 있다. 사기를 잡는 것(거짓 음성을 줄이는 것)이 주된 목표이므로 F1 점수와 재현율이 훨씬 많은 것을 알려 준다.

다중 클래스 예제(붓꽃 품종)는 겹마다 세 품종을 고르게 담도록 층화 교차 검증을 쓴다. 클래스별 분류 보고서는 모델이 특정 품종에서 조직적으로 어려움을 겪는지 드러내며, 이는 특징이 모자라거나 클래스의 경계가 겹친다는 뜻일 수 있다.

회귀 예제(집값)는 RMSE와 $R^2$이 서로 보완하는 정보를 준다는 것을 보인다. RMSE는 오차의 크기를 달러 단위로 주고, $R^2$은 특징이 설명하는 가격 분산의 비율을 알려 준다. $R^2$이 높은데 RMSE도 크다면 목표 자체의 변동이 크다는 뜻이다.

## 연습문제

**연습문제 1.**
코드를 따라가며 쓰인 주요 자료 구조를 찾아라. 각각에 대해 자료형, (해당한다면) 모양, 파이프라인에서의 구실을 적어라.

??? success "연습문제 1 풀이"
    코드를 꼼꼼히 읽으며 변수 대입마다 살펴본다. 텐서는 `.shape`과 `.dtype`을 확인하고, 클래스는 `__init__`의 매개변수와 `forward`/`__call__`의 서명을 확인한다. 이름, 자료형, 모양, 구실을 열로 하는 표에 정리한다.

---


**연습문제 2.**
오류 처리와 입력 검증을 넣도록 코드를 고쳐라. 이 코드를 실전에 쓸 수 있게 하려면 어떤 검사를 더하겠는가?

??? success "연습문제 2 풀이"
    입력에 자료형 검사(`isinstance`), 모양 검증(`assert tensor.dim() == expected`), 값 범위 검사(예: 확률이 [0,1] 안인지)를 넣고, 입출력 연산은 try-except로 감싼다. 빈 배치나 NaN 같은 경계 상황에는 경고를 남긴다. 매개변수와 반환값의 자료형을 적은 독스트링을 붙인다.

---


**연습문제 3.**
직접 고른 새로운 쓰임새를 지원하도록 코드를 확장하라. 무엇을 왜 바꿀지 설명하라.

??? success "연습문제 3 풀이"
    알맞은 확장을 하나 고른다(예: 다른 데이터셋, 지표 추가, 새 모델 변형). 필요한 변경을 설명한다. 새 임포트, 클래스 정의 수정, 초매개변수 갱신, 새로운 시각화나 기록 등이다. 핵심 변경을 구현하고 간단한 시험으로 올바름을 확인한다.

