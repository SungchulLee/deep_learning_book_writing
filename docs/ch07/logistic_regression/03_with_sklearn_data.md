# scikit-learn 데이터로 실습하기

scikit-learn의 유방암 데이터로 이진 로지스틱 회귀의 분할, 표준화와 평가를 살펴본다. 정답 인코딩을 먼저 확인해야 지표를 올바르게 해석할 수 있다.

## 1. 코드

??? note "코드 (408줄)"

    ```python
    """
    ==============================================================================
    03_with_sklearn_data.py - 참 데이터셋 다루기
    ================================================================================

    학습 목표:
    - 참 세상 데이터셋을 불러와 다룬다
    - 데이터 미리 다듬기(표준화)를 이해한다
    - 학습/검증/시험 나누기를 제대로 다룬다
    - 여러 자로 모델을 평가한다

    PREREQUISITES:
    - 02_simple_binary_classification.py을 마쳤을 것
    - 평균과 표준편차 이해
    - 기본 통계 지식

    소요 시간: 1시간쯤

    어려움: ⭐⭐☆☆☆ (쉬움~보통)
    ================================================================================
    """

    import torch
    import torch.nn as nn
    import numpy as np
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    # 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
    # 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
    torch.manual_seed(0)

    print("="*80)
    print("BREAST CANCER CLASSIFICATION - A REAL-WORLD EXAMPLE")
    print("="*80)

    # =============================================================================
    # 1부: 데이터셋 불러오기
    # =============================================================================
    print("\n" + "="*80)
    print("PART 1: LOADING AND EXPLORING THE DATASET")
    print("="*80)

    print("\n1.1: About the Wisconsin Breast Cancer Dataset")
    print("-" * 40)
    print("""
    데이터셋: 위스콘신 유방암 진단 데이터셋
    밑동: UCI 기계 학습 저장소
    표본: 환자 569명
    피처: 디지털 그림에서 셈한 수치 피처 30개
    타깃: 악성(1)인가 양성(0)인가

    피처에는 다음이 있다.
      - 반지름(가운데에서 둘레 위 점까지 거리의 평균)
      - 결(잿빛 값의 표준편차)
      - 둘레, 넓이, 매끄러움, 옹골참 따위

    목표: 이 피처로 종양이 악성인지 양성인지 예측한다
    """)

    # 데이터셋을 불러온다
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    print(f"\nDataset loaded successfully!")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Feature names (first 5): {bc.feature_names[:5]}")
    print(f"Target names: {bc.target_names}")  # ['malignant' 'benign']
    print(f"\nClass distribution:")
    print(f"  Malignant (0): {(y==0).sum()} ({100*(y==0).sum()/len(y):.1f}%)")
    print(f"  Benign (1): {(y==1).sum()} ({100*(y==1).sum()/len(y):.1f}%)")

    # =============================================================================
    # 2부: 데이터 전처리
    # =============================================================================
    print("\n" + "="*80)
    print("PART 2: DATA PREPROCESSING")
    print("="*80)

    print("\n2.1: Why Standardization?")
    print("-" * 40)
    print("""
    피처 표준화: 피처를 평균 0, 표준편차 1이 되도록 바꾼다

    왜 필요한가:
      1. 피처마다 잣대가 다르다(보기: 반지름과 넓이)
      2. 표준화한 피처에서 경사 하강법이 더 빨리 모여든다
      3. 잣대가 큰 피처가 휘어잡는 것을 막는다

    식: z = (x - mean) / std

    IMPORTANT: 
      - 스케일러는 학습 데이터에만 맞춘다
      - 시험 데이터에는 같은 바꾸기를 건다
      - 시험 데이터에는 결코 맞추지 마라(정보가 새어 나간다!)
    """)

    # 표준화 전에 데이터를 나눈다
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nData split:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")

    # 피처를 표준화한다
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # Fit on train, transform train
    X_test = scaler.transform(X_test)        # Only transform test (don't fit!)

    print(f"\nAfter standardization:")
    print(f"  Training mean: {X_train.mean():.6f} (should be ≈0)")
    print(f"  Training std: {X_train.std():.6f} (should be ≈1)")

    # PyTorch 텐서로 변환
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train).reshape(-1, 1)
    y_test = torch.FloatTensor(y_test).reshape(-1, 1)

    print(f"\nTensor shapes:")
    print(f"  X_train: {X_train.shape}")  # (455, 30)
    print(f"  y_train: {y_train.shape}")  # (455, 1)
    print(f"  X_test: {X_test.shape}")    # (114, 30)
    print(f"  y_test: {y_test.shape}")    # (114, 1)

    # =============================================================================
    # 3부: 모델 구성하기
    # =============================================================================
    print("\n" + "="*80)
    print("PART 3: BUILDING THE MODEL")
    print("="*80)

    class LogisticRegressionModel(nn.Module):
        """
        유방암 분류를 위한 로지스틱 회귀

        입력: 피처 30개
        출력: 확률 1개(양성)
        """
        def __init__(self, n_features):
            super(LogisticRegressionModel, self).__init__()
            self.linear = nn.Linear(n_features, 1)

        def forward(self, x):
            return torch.sigmoid(self.linear(x))

    n_features = X_train.shape[1]  # 30 features
    model = LogisticRegressionModel(n_features)

    print(f"Model created with {n_features} input features")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # =============================================================================
    # 4부: 학습
    # =============================================================================
    print("\n" + "="*80)
    print("PART 4: TRAINING THE MODEL")
    print("="*80)

    # 준비
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    num_epochs = 500

    # 학습 기록
    history = {
        'loss': [],
        'accuracy': []
    }

    print(f"\nTraining for {num_epochs} epochs...")
    print("-" * 40)

    for epoch in range(num_epochs):
        # 순전파
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 정확도를 계산한다
        with torch.no_grad():
            predicted_classes = (y_pred >= 0.5).float()
            accuracy = (predicted_classes == y_train).float().mean()

        # 이력 저장
        history['loss'].append(loss.item())
        history['accuracy'].append(accuracy.item())

        # 진행 상황 출력
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] "
                  f"Loss: {loss.item():.4f} "
                  f"Accuracy: {accuracy.item():.4f}")

    print("\nTraining completed!")

    # =============================================================================
    # 5부: 여러 지표를 쓰는 평가
    # =============================================================================
    print("\n" + "="*80)
    print("PART 5: COMPREHENSIVE EVALUATION")
    print("="*80)

    model.eval()
    with torch.no_grad():
        # 시험 집합에 대한 예측
        y_pred_proba = model(X_test)
        y_pred_class = (y_pred_proba >= 0.5).float()

        # sklearn 지표를 쓰기 위해 numpy로 바꾼다
        y_test_np = y_test.numpy().flatten()
        y_pred_np = y_pred_class.numpy().flatten()
        y_pred_proba_np = y_pred_proba.numpy().flatten()

        # 지표를 계산한다
        accuracy = accuracy_score(y_test_np, y_pred_np)
        precision = precision_score(y_test_np, y_pred_np)
        recall = recall_score(y_test_np, y_pred_np)
        f1 = f1_score(y_test_np, y_pred_np)
        conf_matrix = confusion_matrix(y_test_np, y_pred_np)

    print("\n5.1: Classification Metrics")
    print("-" * 40)
    print(f"Accuracy:  {accuracy:.4f}  - Overall correctness")
    print(f"Precision: {precision:.4f}  - Of predicted benign, how many were correct?")
    print(f"Recall:    {recall:.4f}  - Of actual benign, how many did we find?")
    print(f"F1-Score:  {f1:.4f}  - Harmonic mean of precision and recall")

    print("\n5.2: Confusion Matrix")
    print("-" * 40)
    print("                Predicted")
    print("              Malig  Benign")
    print(f"Actual Malig    {conf_matrix[0,0]:3d}    {conf_matrix[0,1]:3d}")
    print(f"       Benign   {conf_matrix[1,0]:3d}    {conf_matrix[1,1]:3d}")

    # 혼동 행렬에서 개별 지표를 계산한다
    tn, fp, fn, tp = conf_matrix.ravel()
    print(f"\nTrue Negatives (TN):  {tn} - Correctly identified malignant")
    print(f"False Positives (FP): {fp} - Incorrectly predicted benign (BAD!)")
    print(f"False Negatives (FN): {fn} - Incorrectly predicted malignant")
    print(f"True Positives (TP):  {tp} - Correctly identified benign")

    # =============================================================================
    # 6부: 시각화
    # =============================================================================
    print("\n" + "="*80)
    print("PART 6: CREATING VISUALIZATIONS")
    print("="*80)

    fig = plt.figure(figsize=(15, 10))

    # 그림 1: 학습 로스
    plt.subplot(2, 3, 1)
    plt.plot(history['loss'], 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss', fontweight='bold')
    plt.grid(True, alpha=0.3)

    # 그림 2: 학습 정확도
    plt.subplot(2, 3, 2)
    plt.plot(history['accuracy'], 'g-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy', fontweight='bold')
    plt.ylim([0.5, 1.0])
    plt.grid(True, alpha=0.3)

    # 그림 3: 혼동 행렬
    plt.subplot(2, 3, 3)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Malignant', 'Benign'],
                yticklabels=['Malignant', 'Benign'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix', fontweight='bold')

    # 그림 4: 지표 비교
    plt.subplot(2, 3, 4)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy, precision, recall, f1]
    bars = plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    plt.ylim([0, 1])
    plt.ylabel('Score')
    plt.title('Performance Metrics', fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}', ha='center', va='bottom')

    # 그림 5: 예측 분포
    plt.subplot(2, 3, 5)
    plt.hist(y_pred_proba_np[y_test_np==0], bins=30, alpha=0.6, label='Malignant', color='red')
    plt.hist(y_pred_proba_np[y_test_np==1], bins=30, alpha=0.6, label='Benign', color='blue')
    plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title('Prediction Distribution', fontweight='bold')
    plt.legend()

    # 그림 6: 요약
    plt.subplot(2, 3, 6)
    summary = f"""
    모델 간추림
    {'='*40}

    데이터셋: 위스콘신 유방암
    표본: 모두 {len(X)}개
      - 학습: {len(X_train)}
      - 시험: {len(X_test)}

    Features: {n_features}

    Training:
      - 에폭 수: {num_epochs}
      - 마지막 로스: {history['loss'][-1]:.4f}
      - 마지막 학습 정확도: {history['accuracy'][-1]:.4f}

    시험 성능:
      - Accuracy: {accuracy:.4f}
      - Precision: {precision:.4f}
      - Recall: {recall:.4f}
      - F1-Score: {f1:.4f}

    임상으로 읽기:
      - Missed cancers (FP): {fp}
      - False alarms (FN): {fn}
    """
    plt.text(0.1, 0.5, summary, fontsize=9, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('breast_cancer_results.png',
                dpi=150, bbox_inches='tight')
    print("Visualization saved!")

    # =============================================================================
    # 핵심 요점
    # =============================================================================
    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print("""
    1. 데이터 미리 다듬기
       - 피처는 늘 표준화하라
       - 스케일러는 학습 데이터에만 맞춰라
       - 시험 데이터에는 같은 바꾸기를 건다

    2. 따짐 자
       - 정확도: 고른 데이터셋에 좋다
       - 정밀도: 헛정확도의 값이 클 때 종요롭다
       - 재현율: 놓침의 값이 클 때 종요롭다
       - F1 점수: 정밀도와 재현율의 고른 자리

    3. 의료에서의 쓰임
       - 헛정확도: 쓸데없는 걱정과 시술
       - 놓침: 병을 놓친다(아주 위험하다!)
       - 흔히 높은 재현율을 앞세운다(병을 모두 잡아낸다)

    4. 좋은 버릇
       - 정확도만이 아니라 여러 자를 써라
       - 혼동 행렬을 이해하라
       - 그 분야에서 오차이 치르는 값을 헤아려라
    """)

    print("\n" + "="*80)
    print("EXERCISES")
    print("="*80)
    print("""
    1. 쉬움: test_size 값을 바꾸어 보아라(0.1, 0.3, 0.5)
       성능에 어떤 영향을 주는가?

    2. 보통: 분류 임계를 0.5에서 0.3으로 바꾸어라
       정밀도와 재현율에 어떤 영향을 주는가?

    3. 보통: 여러 러닝레이트를 써 보아라
       lr=0.01, 0.1, 1.0의 학습 굽이를 그려라

    4. 어려움: 웨이트 실은 로스 함수를 짜라
       놓침에 더 큰 벌을 주어라
       실마리: BCELoss의 pos_weight 파라메터를 써라

    5. 어려움: 피처의 종요로움 살피기
       어떤 피처가 가장 종요로운가?
       model.linear.weight 값을 보아라
    """)

    print("\n" + "="*80)
    print("NEXT: 04_bce_vs_bcewithlogits.py")
    print("Learn about numerical stability and better loss functions!")
    print("="*80)


    if __name__ == "__main__":
        pass
    ```


??? note "전체 출력 (172줄)"

    ```
    ================================================================================
    BREAST CANCER CLASSIFICATION - A REAL-WORLD EXAMPLE
    ================================================================================

    ================================================================================
    PART 1: LOADING AND EXPLORING THE DATASET
    ================================================================================

    1.1: About the Wisconsin Breast Cancer Dataset
    ----------------------------------------

    데이터셋: 위스콘신 유방암 진단 데이터셋
    밑동: UCI 기계 학습 저장소
    표본: 환자 569명
    피처: 디지털 그림에서 셈한 수치 피처 30개
    타깃: 악성(1)인가 양성(0)인가

    피처에는 다음이 있다.
      - 반지름(가운데에서 둘레 위 점까지 거리의 평균)
      - 결(잿빛 값의 표준편차)
      - 둘레, 넓이, 매끄러움, 옹골참 따위
      
    목표: 이 피처로 종양이 악성인지 양성인지 예측한다


    Dataset loaded successfully!
    Number of samples: 569
    Number of features: 30
    Feature names (first 5): ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
     'mean smoothness']
    Target names: ['malignant' 'benign']

    Class distribution:
      Malignant (0): 212 (37.3%)
      Benign (1): 357 (62.7%)

    ================================================================================
    PART 2: DATA PREPROCESSING
    ================================================================================

    2.1: Why Standardization?
    ----------------------------------------

    피처 표준화: 피처를 평균 0, 표준편차 1이 되도록 바꾼다

    왜 필요한가:
      1. 피처마다 잣대가 다르다(보기: 반지름과 넓이)
      2. 표준화한 피처에서 경사 하강법이 더 빨리 모여든다
      3. 잣대가 큰 피처가 휘어잡는 것을 막는다
      
    식: z = (x - mean) / std

    IMPORTANT: 
      - 스케일러는 학습 데이터에만 맞춘다
      - 시험 데이터에는 같은 바꾸기를 건다
      - 시험 데이터에는 결코 맞추지 마라(정보가 새어 나간다!)


    Data split:
      Training: 455 samples
      Test: 114 samples

    After standardization:
      Training mean: -0.000000 (should be ≈0)
      Training std: 1.000000 (should be ≈1)

    Tensor shapes:
      X_train: torch.Size([455, 30])
      y_train: torch.Size([455, 1])
      X_test: torch.Size([114, 30])
      y_test: torch.Size([114, 1])

    ================================================================================
    PART 3: BUILDING THE MODEL
    ================================================================================
    Model created with 30 input features
    Total parameters: 31

    ================================================================================
    PART 4: TRAINING THE MODEL
    ================================================================================

    Training for 500 epochs...
    ----------------------------------------
    Epoch [100/500] Loss: 0.0987 Accuracy: 0.9802
    Epoch [200/500] Loss: 0.0802 Accuracy: 0.9846
    Epoch [300/500] Loss: 0.0723 Accuracy: 0.9846
    Epoch [400/500] Loss: 0.0677 Accuracy: 0.9846
    Epoch [500/500] Loss: 0.0646 Accuracy: 0.9868

    Training completed!

    ================================================================================
    PART 5: COMPREHENSIVE EVALUATION
    ================================================================================

    5.1: Classification Metrics
    ----------------------------------------
    Accuracy:  0.9649  - Overall correctness
    Precision: 0.9857  - Of predicted benign, how many were correct?
    Recall:    0.9583  - Of actual benign, how many did we find?
    F1-Score:  0.9718  - Harmonic mean of precision and recall

    5.2: Confusion Matrix
    ----------------------------------------
                    Predicted
                  Malig  Benign
    Actual Malig     41      1
           Benign     3     69

    True Negatives (TN):  41 - Correctly identified malignant
    False Positives (FP): 1 - Incorrectly predicted benign (BAD!)
    False Negatives (FN): 3 - Incorrectly predicted malignant
    True Positives (TP):  69 - Correctly identified benign

    ================================================================================
    PART 6: CREATING VISUALIZATIONS
    ================================================================================
    Visualization saved!

    ================================================================================
    KEY TAKEAWAYS
    ================================================================================

    1. 데이터 미리 다듬기
       - 피처는 늘 표준화하라
       - 스케일러는 학습 데이터에만 맞춰라
       - 시험 데이터에는 같은 바꾸기를 건다

    2. 따짐 자
       - 정확도: 고른 데이터셋에 좋다
       - 정밀도: 헛정확도의 값이 클 때 종요롭다
       - 재현율: 놓침의 값이 클 때 종요롭다
       - F1 점수: 정밀도와 재현율의 고른 자리

    3. 의료에서의 쓰임
       - 헛정확도: 쓸데없는 걱정과 시술
       - 놓침: 병을 놓친다(아주 위험하다!)
       - 흔히 높은 재현율을 앞세운다(병을 모두 잡아낸다)

    4. 좋은 버릇
       - 정확도만이 아니라 여러 자를 써라
       - 혼동 행렬을 이해하라
       - 그 분야에서 오차이 치르는 값을 헤아려라


    ================================================================================
    EXERCISES
    ================================================================================

    1. 쉬움: test_size 값을 바꾸어 보아라(0.1, 0.3, 0.5)
       성능에 어떤 영향을 주는가?

    2. 보통: 분류 임계를 0.5에서 0.3으로 바꾸어라
       정밀도와 재현율에 어떤 영향을 주는가?

    3. 보통: 여러 러닝레이트를 써 보아라
       lr=0.01, 0.1, 1.0의 학습 굽이를 그려라

    4. 어려움: 웨이트 실은 로스 함수를 짜라
       놓침에 더 큰 벌을 주어라
       실마리: BCELoss의 pos_weight 파라메터를 써라

    5. 어려움: 피처의 종요로움 살피기
       어떤 피처가 가장 종요로운가?
       model.linear.weight 값을 보아라


    ================================================================================
    NEXT: 04_bce_vs_bcewithlogits.py
    Learn about numerical stability and better loss functions!
    ================================================================================
    ```


## 2. 논의

### 데이터의 라벨 의미

이 데이터의 기본 인코딩은 0이 악성, 1이 양성이다. 따라서 시그모이드 출력의 클래스 1 확률은 이 인코딩에서 **양성 종양(benign)** 확률이다. 통계 지표의 positive class와 의학적 악성 판정을 같은 단어로 혼동하지 않는다. 이 예제는 분류 구현을 위한 교육용 분석이다.

### 학습 데이터로 정한 변환

피처 30개의 단위가 다르므로 StandardScaler를 학습셋에 적합한 뒤 테스트셋에 적용한다. 계수 크기를 비교할 때도 이 표준화 기준을 명시한다. 테스트 데이터의 통계를 미리 사용하면 평가가 낙관적으로 바뀔 수 있다.

### 지표의 기준 클래스

정밀도와 재현율, ROC 곡선을 계산할 때 어느 클래스를 양성으로 정했는지 확인한다. 높은 전체 정확도만으로 악성 사례의 놓침이 적다고 판단할 수 없다. 분할과 초기값에 따른 변동을 고려하고 독립적인 최종 평가와 설정 선택을 구분한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
입력 피처 수가 $d$인 단일 선형 이진 분류기의 파라미터 수와 출력 모양을 설명하라.

</div>

??? success "연습문제 1 풀이"
    웨이트 $d$개와 바이어스 1개로 총 $d+1$개다. 배치 입력 $(B,d)$에 대한 출력은 $(B,1)$이다. 시그모이드에는 학습 파라미터가 없다. 유방암 데이터의 피처 30개를 쓰는 경우에는 31개다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
옵티마이저를 Adam으로 바꾸고(`torch.optim.Adam`에 `lr=0.001`을 쓴다) 원래 옵티마이저와 학습 수렴을 비교하라. 두 로스 곡선을 같은 그래프에 그려라.

</div>

??? success "연습문제 2 풀이"
    옵티마이저를 만드는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 바꾼다. Adam은 경사와 경사 제곱의 이동 평균으로 성분별 업데이트를 조정한다. 어느 옵티마이저가 더 빨리 수렴하는지는 문제와 러닝레이트에 달려 있다. 같은 초기 웨이트와 데이터 순서, 평가 기준을 사용하고 각 러닝레이트를 조정해 비교한다. 여러 시드의 변동과 검증 성능도 확인한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
클래스 1의 재현율을 악성 종양 검출률이라고 읽어도 되는가?

</div>

??? success "연습문제 3 풀이"
    기본 라벨에서 1은 양성 종양이고 0은 악성이다. 따라서 기본 positive class 1의 재현율은 양성 종양에 대한 값이다. 악성 검출을 해석하려면 기준 클래스와 확률 방향을 명시적으로 바꿔야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 일반적인 계층 확장 코드가 이 페이지의 원래 클래스에 그대로 적용되는지 검토하라. 모델과 데이터셋의 역할, 생성자 인수, 입력·출력 모양과 비선형성에서 필요한 수정 사항을 설명하라.

</div>

??? success "연습문제 4 풀이"
    아래 코드는 일반적인 ModuleList 구성 패턴의 일부이며 이 페이지 클래스의 완성된 대체 구현은 아니다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    적용하려면 생성자와 `forward`를 함께 정의하고 마지막 출력 모양과 로스의 입력 조건을 유지해야 한다. 선형 계층만 여러 개 합성하면 여전히 하나의 선형 사상이므로 표현력을 늘리려면 적절한 비선형성이 필요하다. Dataset은 표본 접근 인터페이스이므로 계층을 등록할 대상이 아니며 별도 모델 클래스에 이 구성을 적용한다.

## 정리하며

모델 출력과 로스의 입력 조건을 일치시키고 데이터 분할, 전처리와 지표의 의미를 함께 확인한다. 기록된 실행 결과는 해당 설정의 예시이며 모든 환경의 수렴이나 일반화를 보장하지 않는다.
