# scikit-learn 데이터로 실습하기

03_with_sklearn_data.py - 실제 데이터셋 다루기

이 튜토리얼은 PyTorch에서 로지스틱 회귀에 대한 기초적인 이해를 쌓는다. 코드를 따라가 보면 모델을 세우고, 손실 함수를 정의하고, 경사 하강법으로 학습시키고, 분류 과제에서 성능을 평가하는 법을 알 수 있다.

## 코드

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
특징: 디지털 그림에서 셈한 수치 특징 30개
과녁: 악성(1)인가 양성(0)인가

특징에는 다음이 있다.
  - 반지름(가운데에서 둘레 위 점까지 거리의 평균)
  - 결(잿빛 값의 표준편차)
  - 둘레, 넓이, 매끄러움, 옹골참 따위
  
목표: 이 특징으로 종양이 악성인지 양성인지 예측한다
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
특징 표준화: 특징을 평균 0, 표준편차 1이 되도록 바꾼다

왜 필요한가:
  1. 특징마다 잣대가 다르다(보기: 반지름과 넓이)
  2. 표준화한 특징에서 경사 하강법이 더 빨리 모여든다
  3. 잣대가 큰 특징이 휘어잡는 것을 막는다
  
식: z = (x - mean) / std

IMPORTANT: 
  - 잣대 잡개는 학습 데이터에만 맞춘다
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

# 특징을 표준화한다
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
    유방암 분류을 위한 로지스틱 회귀
    
    입력: 특징 30개
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

# 그림 1: 학습 손실
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
  - 마지막 손실: {history['loss'][-1]:.4f}
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
plt.savefig('/home/claude/pytorch_logistic_regression_tutorial/01_basics/breast_cancer_results.png',
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
   - 특징은 늘 표준화하라
   - 잣대 잡개는 학습 데이터에만 맞춰라
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

2. 보통: 분류 문턱을 0.5에서 0.3으로 바꾸어라
   정밀도와 재현율에 어떤 영향을 주는가?

3. 보통: 여러 학습률를 써 보아라
   lr=0.01, 0.1, 1.0의 학습 굽이를 그려라

4. 어려움: 가중치 실은 손실 함수를 짜라
   놓침에 더 큰 벌을 주어라
   실마리: BCELoss의 pos_weight 매개변수를 써라

5. 어려움: 특징의 종요로움 살피기
   어떤 특징이 가장 종요로운가?
   model.linear.weight 값을 보아라
""")

print("\n" + "="*80)
print("NEXT: 04_bce_vs_bcewithlogits.py")
print("Learn about numerical stability and better loss functions!")
print("="*80)


if __name__ == "__main__":
    pass
```

## 논의

`LogisticRegressionModel` 클래스는 PyTorch의 `nn.Module` 인터페이스를 사용하여 모델 구조를 감싼다. `forward` 메서드가 계산 그래프를 정의하므로, 학습 중에 PyTorch의 autograd 체계가 경사 계산을 자동으로 처리한다. 이런 모듈식 설계 덕분에 개별 구성 요소를 고치거나 모델을 더 큰 파이프라인에 넣기가 쉬워진다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 분류 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
기본 초기화 상태의 `LogisticRegressionModel`에서 학습 가능한 매개변수의 총 개수를 계산하라. 가중치와 편향을 모두 포함하여 층별로 나누어 세어라.

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)` 각각에는 `in_features * out_features`개의 가중치 매개변수와 (`bias=False`가 아닌 한) `out_features`개의 편향 매개변수가 있다. `nn.Conv2d(in_c, out_c, k)`에는 `in_c * out_c * k * k`개의 가중치와 `out_c`개의 편향이 있다. `nn.Embedding(num, dim)`에는 `num * dim`개의 매개변수가 있다. 모든 층에 대해 더하면 된다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

---

**연습문제 2.**
최적화기를 Adam으로 바꾸고(`torch.optim.Adam`에 `lr=0.001`을 쓴다) 원래 최적화기와 학습 수렴을 비교하라. 두 손실 곡선을 같은 그래프에 그려라.

??? success "연습문제 2 풀이"
    최적화기를 만드는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 바꾼다. Adam은 매개변수마다 적응적인 학습률과 운동량 추정값을 유지하므로 초반 에폭에서 대체로 더 빠르게 수렴한다. Adam의 손실 곡선은 보통 처음 몇 에폭에서 더 가파르게 떨어지지만, 최적점 근처에서는 운동량을 쓴 SGD보다 조금 더 흔들릴 수 있다. 공정한 비교를 위해 둘을 같은 난수 씨앗과 같은 에폭 수로 실행하라.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
층이나 블록의 개수를 설정할 수 있도록 `LogisticRegressionModel`를 확장하라. `__init__`에 `num_layers` 매개변수를 추가하고 `nn.ModuleList`로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 순회한다. (평범한 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 PyTorch가 모든 매개변수를 최적화 대상으로 등록한다. 시험은 다음과 같이 한다. `for n in [2, 4, 8]: model = LogisticRegressionModel(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
