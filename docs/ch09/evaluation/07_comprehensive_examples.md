# 종합 예제

합성 이진 분류, 붓꽃 다중 클래스 분류와 합성 회귀의 평가를 묶은 예제이다. 제목이나 출력의 응용 이름과 실제 데이터 생성·평가 절차를 구분한다.

## 1. 코드

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
        
        # 피처의 척도 조정
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
    
    문제: 피처로 집값 예측하기
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
        
        # 피처의 척도 조정
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
    run_all_examples()
```

??? note "전체 출력 (193줄)"

    ```

    ################################################################################
    #                                                                              #
    #                    COMPREHENSIVE EXAMPLES                                     #
    #               End-to-End Model Evaluation Workflows                            #
    #                                                                              #
    ################################################################################
    ================================================================================
    EXAMPLE 1: BINARY CLASSIFICATION - CREDIT CARD FRAUD DETECTION
    ================================================================================

    1. GENERATING DATA
    --------------------------------------------------------------------------------
    Total samples: 10000
    Fraudulent transactions: 539 (5.39%)
    Legitimate transactions: 9461 (94.61%)
    ⚠ HIGHLY IMBALANCED DATASET

    2. SPLITTING DATA
    --------------------------------------------------------------------------------
    Training set: 8000 samples
    Test set: 2000 samples

    3. TRAINING MODELS
    --------------------------------------------------------------------------------
    ✓ Logistic Regression trained
    ✓ Random Forest trained

    4. CROSS-VALIDATION
    --------------------------------------------------------------------------------
    Logistic Regression F1 Score: 0.5607 (+/- 0.0952)
    Random Forest F1 Score: 0.5026 (+/- 0.1425)

    5. TEST SET EVALUATION
    --------------------------------------------------------------------------------

    LOGISTIC REGRESSION:
      Accuracy: 0.9625
      Precision: 0.9714
      Recall: 0.3148
      F1-Score: 0.4755
      ROC-AUC: 0.8211

    RANDOM FOREST:
      Accuracy: 0.9615
      Precision: 1.0000
      Recall: 0.2870
      F1-Score: 0.4460
      ROC-AUC: 0.9192

    6. CONFUSION MATRIX (Random Forest)
    --------------------------------------------------------------------------------
    [[1892    0]
     [  77   31]]

    True Negatives (Legitimate correctly identified): 1892
    False Positives (Legitimate flagged as fraud): 0
    False Negatives (Fraud missed): 77
    True Positives (Fraud caught): 31

    7. BUSINESS INTERPRETATION
    --------------------------------------------------------------------------------
    Out of 108 fraudulent transactions:
      ✓ Caught: 31 (28.7%)
      ✗ Missed: 77 (71.3%)

    Out of 1892 legitimate transactions:
      ✓ Approved: 1892 (100.0%)
      ✗ Incorrectly flagged: 0 (0.0%)

    ================================================================================


    ================================================================================
    EXAMPLE 2: MULTI-CLASS CLASSIFICATION - IRIS SPECIES
    ================================================================================

    1. LOADING DATA
    --------------------------------------------------------------------------------
    Total samples: 150
    Features: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    Classes: ['setosa' 'versicolor' 'virginica']

    2. SPLITTING DATA
    --------------------------------------------------------------------------------

    3. TRAINING MODEL
    --------------------------------------------------------------------------------
    ✓ Random Forest trained

    4. STRATIFIED CROSS-VALIDATION
    --------------------------------------------------------------------------------
    CV Accuracy: 0.9524 (+/- 0.0852)

    5. TEST SET EVALUATION
    --------------------------------------------------------------------------------
    Overall Accuracy: 0.8889

    Per-Class Metrics:
                  precision    recall  f1-score   support

          setosa       1.00      1.00      1.00        15
      versicolor       0.78      0.93      0.85        15
       virginica       0.92      0.73      0.81        15

        accuracy                           0.89        45
       macro avg       0.90      0.89      0.89        45
    weighted avg       0.90      0.89      0.89        45


    6. CONFUSION MATRIX
    --------------------------------------------------------------------------------
                 Predicted
                   seto   vers   virg
    Actual
    setosa        15      0      0
    versicolor     0     14      1
    virginica      0      4     11

    ================================================================================


    ================================================================================
    EXAMPLE 3: REGRESSION - HOUSE PRICE PREDICTION
    ================================================================================

    1. GENERATING DATA
    --------------------------------------------------------------------------------
    Total samples: 1000
    Features: 10 (square feet, bedrooms, location score, etc.)
    Price range: $200,000 - $700,000
    Mean price: $455,824

    2. SPLITTING DATA
    --------------------------------------------------------------------------------

    3. TRAINING MODELS
    --------------------------------------------------------------------------------
    ✓ Ridge Regression trained
    ✓ Random Forest trained

    4. CROSS-VALIDATION
    --------------------------------------------------------------------------------
    Ridge R² Score: 0.9918 (+/- 0.0019)
    Random Forest R² Score: 0.8785 (+/- 0.0496)

    5. TEST SET EVALUATION
    --------------------------------------------------------------------------------

    RIDGE REGRESSION:
      MAE: $6,098.17
      RMSE: $7,517.15
      R² Score: 0.9925

    RANDOM FOREST:
      MAE: $22,582.57
      RMSE: $29,342.15
      R² Score: 0.8853

    6. SAMPLE PREDICTIONS (Random Forest)
    --------------------------------------------------------------------------------
       Actual Price   Predicted Price        Error
    --------------------------------------------------
    $       562,880 $         530,842 $     32,038
    $       480,307 $         472,571 $      7,736
    $       408,467 $         395,849 $     12,618
    $       471,523 $         482,055 $    -10,532
    $       441,151 $         497,155 $    -56,004
    $       604,026 $         530,244 $     73,782
    $       424,141 $         464,710 $    -40,569
    $       461,645 $         450,219 $     11,426
    $       485,020 $         480,934 $      4,086
    $       539,037 $         546,958 $     -7,921

    ================================================================================


    ################################################################################
    #                                                                              #
    #                         ALL EXAMPLES COMPLETED                               #
    #                                                                              #
    ################################################################################

    📚 KEY TAKEAWAYS:
    ================================================================================
    1. Always use appropriate metrics for your problem type and business context
    2. Use stratified splitting for classification, especially with imbalanced data
    3. Cross-validation provides more reliable performance estimates than single split
    4. Consider both model performance AND business implications
    5. Confusion matrices reveal which errors your model makes
    6. Compare multiple models before settling on one
    7. Document your evaluation methodology for reproducibility
    ================================================================================
    ```


## 2. 논의

이진 분류는 make_classification으로 만든 불균형 합성 데이터이며 실제 거래 데이터가 아니다. 회귀도 make_regression으로 생성하므로 RMSE를 실제 주택 가격이나 달러 오차로 해석할 수 없다.

이진 분류와 회귀는 외부 학습 부분 전체를 표준화한 뒤 내부 교차 검증을 수행한다. 외부 테스트를 scaler에 fit하지는 않았지만 내부 검증 입력의 통계가 전처리에 반영된다. 엄격한 내부 평가에는 scaler와 모델을 Pipeline으로 함께 검증해야 한다.

여러 모델의 최종 테스트 지표를 비교해 가장 좋은 모델을 고르면 테스트가 선택에 쓰인다. 선택 기준은 검증에서 정하고 마지막 테스트를 독립적으로 해석한다. 불균형 분류의 F1, 재현율과 확률 지표는 각각 다른 질문에 답한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
합성 회귀 RMSE를 달러로 읽을 수 없는 이유를 설명하라.

</div>

??? success "풀이"
    데이터 생성 함수의 목표에는 실제 금액 단위가 주어지지 않았다. 출력에 응용 이름을 붙여도 단위가 생기는 것은 아니다. RMSE는 생성한 목표와 같은 임의 단위이다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
외부 테스트를 분리했는데도 내부 교차 검증에 전처리 누출이 생기는 이유는 무엇인가?

</div>

??? success "풀이"
    내부 폴드의 검증 부분까지 포함해 scaler를 미리 fit했기 때문이다. 외부 분할의 독립성과 내부 모델 선택의 타당성은 별도로 확인해야 한다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
높은 R²와 큰 RMSE가 동시에 나타나는 상황을 설명하라.

</div>

??? success "풀이"
    목표 분산이 크면 기준 평균보다 상당히 나은 예측이어도 단위상의 절대 오차가 클 수 있다. 같은 평가셋에서 R²와 RMSE는 상대·절대 성능을 다르게 표현한다.

## 정리하며

종합 평가에서는 데이터의 실제 출처와 단위, 내부 전처리·검증 절차를 먼저 확인한다. 합성 예제의 지표를 실제 응용의 성과로 읽지 않고 모델 선택과 최종 평가를 분리한다.
