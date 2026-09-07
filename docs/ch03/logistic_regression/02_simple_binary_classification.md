# 간단한 이진 분류

02_simple_binary_classification.py - 첫 번째 로지스틱 회귀 모델

이 튜토리얼은 PyTorch에서 로지스틱 회귀에 대한 기초적인 이해를 쌓는다. 코드를 따라가 보면 모델을 세우고, 손실 함수를 정의하고, 경사 하강법으로 학습시키고, 분류 과제에서 성능을 평가하는 법을 알 수 있다.

## 코드

```python
"""
================================================================================
02_simple_binary_classification.py - 첫 로지스틱 회귀 모형
================================================================================

배움 목표:
- 둘 가름 문제를 이해한다
- PyTorch으로 로지스틱 회귀를 맨바닥부터 짠다
- 시그모이드 함수와 그 결을 배운다
- 기울기 내림으로 모형을 익힌다
- 모형의 성능을 따진다

PREREQUISITES:
- 01_introduction.py을 마쳤을 것
- 선형 모형(y = mx + b) 이해
- 기본 확률 개념

마치는 데 드는 때: 45분쯤

어려움: ⭐⭐☆☆☆ (쉬움)
================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

print("="*80)
print("PART 1: UNDERSTANDING THE PROBLEM")
print("="*80)

# ============================================================================
# 1.1: 이진 분류
# ============================================================================
print("\n1.1: What is Binary Classification?")
print("-" * 40)

print("""
둘 가름: 두 갈래(0 또는 1) 가운데 하나를 예측한다

Examples:
  - 전자우편: 광고(1)인가 아닌가(0)
  - 의료: 병(1)인가 건강(0)인가
  - 손님: 살 것(1)인가 안 살 것(0)인가
  - 그림: 고양이(1)인가 개(0)인가

이 익힘에서는
  - 특징 2개(x1, x2)를 지닌 인공 자료를 만든다
  - 표본마다 갈래 0이나 갈래 1에 든다
  - 목표: 특징에서 갈래를 예측하는 법을 배운다
""")

# ============================================================================
# 1.2: 합성 데이터 생성
# ============================================================================
print("\n1.2: Generating Dataset")
print("-" * 40)

# 재현성을 위한 난수 시드 설정
torch.manual_seed(42)
np.random.seed(42)

# 간단한 2차원 이진 분류 데이터셋을 생성한다
# n_samples: 데이터 점의 개수
# n_features: 입력 특징의 개수 (시각화하기 쉽게 2로 잡는다)
# n_classes: 2 (이진 분류)
# n_clusters_per_class: 클래스들이 얼마나 "떨어져" 있는지
X, y = make_classification(
    n_samples=200,           # Total number of examples
    n_features=2,            # 2D data (x1, x2) for easy plotting
    n_redundant=0,           # No redundant features
    n_informative=2,         # Both features are informative
    n_clusters_per_class=1,  # Single cluster per class
    random_state=42,
    flip_y=0.1              # Add 10% noise (some mislabeled examples)
)

print(f"Dataset shape: X={X.shape}, y={y.shape}")
print(f"X (features): {X.shape[0]} samples × {X.shape[1]} features")
print(f"y (labels): {y.shape[0]} labels")
print(f"Class distribution: Class 0: {(y==0).sum()}, Class 1: {(y==1).sum()}")

# 학습 집합과 시험 집합으로 나눈다
# 학습: 80%, 시험: 20%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nAfter split:")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# PyTorch 텐서로 변환
# 중요: float32(PyTorch의 기본값)로 바꾼다
X_train = torch.FloatTensor(X_train)  # Shape: (160, 2)
X_test = torch.FloatTensor(X_test)    # Shape: (40, 2)
y_train = torch.FloatTensor(y_train).reshape(-1, 1)  # Shape: (160, 1)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)    # Shape: (40, 1)

print(f"\nTensor shapes:")
print(f"X_train: {X_train.shape} (160 samples, 2 features)")
print(f"y_train: {y_train.shape} (160 labels, reshaped to column vector)")
print(f"X_test: {X_test.shape}")
print(f"y_test: {y_test.shape}")


print("\n" + "="*80)
print("PART 2: THE LOGISTIC REGRESSION MODEL")
print("="*80)

# ============================================================================
# 2.1: 모델 이해하기
# ============================================================================
print("\n2.1: Model Architecture")
print("-" * 40)

print("""
로지스틱 회귀 모형:

걸음 1: 선형 아우르기
    z = w1*x1 + w2*x2 + b
    여기서 w1, w2은 무게이고 b은 치우침이다

걸음 2: 시그모이드 살림
    확률 = sigmoid(z) = 1 / (1 + e^(-z))
    
시그모이드의 결:
    - 어떤 값이든 (0, 1) 범위로 옮긴다
    - sigmoid(0) = 0.5(가름 테두리)
    - sigmoid(큰 양수) ≈ 1
    - sigmoid(큰 음수) ≈ 0
    
걸음 3: 가름
    확률 >= 0.5이면 갈래 1으로 예측한다
    확률 < 0.5이면 갈래 0으로 예측한다
""")

# ============================================================================
# 2.2: 모델 구현하기
# ============================================================================
print("\n2.2: Implementing in PyTorch")
print("-" * 40)

class LogisticRegression(nn.Module):
    """
    단순한 로지스틱 회귀 모형
    
    Architecture:
        들임(n_features) → 선형 층 → 시그모이드 → 내놓음(확률)
    
    Parameters:
        n_features (int): 들임 특징의 수
    """
    
    def __init__(self, n_features):
        super(LogisticRegression, self).__init__()
        
        # 선형 층: y = xW^T + b
        # in_features: 입력 특징의 개수 (여기서는 2)
        # out_features: 출력의 개수 (이진 분류에서는 1)
        self.linear = nn.Linear(n_features, 1)
        
        # 선형 층이 만드는 것:
        # - self.linear.weight: 모양 (1, n_features) - 가중치
        # - self.linear.bias: 모양 (1,) - 편향 항
    
    def forward(self, x):
        """
        그물을 지나는 앞으로 걸음
        
        Args:
            x: 꼴이 (batch_size, n_features)인 들임 텐서
            
        Returns:
            probability: 꼴이 (batch_size, 1)인 내놓음 텐서
                        값은 (0, 1) 범위에 있다
        """
        # 1단계: 선형 변환
        # x 꼴: (batch_size, n_features)
        # 내놓음 꼴: (batch_size, 1)
        z = self.linear(x)  # z = w*x + b
        
        # 2단계: 시그모이드 활성화 적용
        # 시그모이드는 z를 (0, 1) 범위의 확률로 보낸다
        probability = torch.sigmoid(z)
        
        return probability

# 모델 인스턴스 생성
n_features = 2  # We have 2 features (x1, x2)
model = LogisticRegression(n_features)

print("Model created!")
print(f"Model structure:\n{model}")
print(f"\nInitial weights: {model.linear.weight.data}")
print(f"Initial bias: {model.linear.bias.data}")


print("\n" + "="*80)
print("PART 3: LOSS FUNCTION AND OPTIMIZER")
print("="*80)

# ============================================================================
# 3.1: 이진 교차 엔트로피 손실
# ============================================================================
print("\n3.1: Understanding the Loss Function")
print("-" * 40)

print("""
둘 엇갈린 엔트로피(BCE) 잃음:

보기 하나에 대해
    참 이름표가 y = 1이면
        잃음 = -log(예측 확률)
        → 갈래 1의 확률을 낮게 예측하면 모형이 벌을 받는다
    
    참 이름표가 y = 0이면
        잃음 = -log(1 - 예측 확률)
        → 갈래 1의 확률을 높게 예측하면 모형이 벌을 받는다

온 식:
    잃음 = -[y*log(p) + (1-y)*log(1-p)]

Properties:
    - 늘 양수다
    - 작을수록 좋다
    - 자신 있게 틀린 예측을 크게 벌한다
""")

# 손실 함수를 만든다
# BCELoss: 이진 교차 엔트로피 손실
# 예상: 예측과 목표가 모두 [0, 1] 범위의 확률
criterion = nn.BCELoss()

print("Loss function: Binary Cross-Entropy (BCE)")

# ============================================================================
# 3.2: 최적화기
# ============================================================================
print("\n3.2: Choosing an Optimizer")
print("-" * 40)

print("""
가장 좋게 하개: 모형 매개변수(무게와 치우침)를 고치는 알고리즘

흔한 가장 좋게 하개:
    - SGD(확률 기울기 내림): 기본이지만 믿을 만하다
    - Adam: 맞추어 가는 배움 빠르기. 대개 더 빨리 모여든다
    - RMSprop: 되도는 신경망에 좋다
    
여기서는 단순하고 이해하기 쉽도록 SGD을 쓴다.
""")

learning_rate = 0.1  # How big each update step is
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

print(f"Optimizer: SGD with learning rate = {learning_rate}")


print("\n" + "="*80)
print("PART 4: TRAINING THE MODEL")
print("="*80)

# ============================================================================
# 4.1: 학습 루프
# ============================================================================
print("\n4.1: Training Loop")
print("-" * 40)

num_epochs = 1000  # Number of times to see the entire dataset
print_every = 100  # Print progress every N epochs

# 학습 기록을 담을 리스트들
train_losses = []
train_accuracies = []

print(f"Starting training for {num_epochs} epochs...")
print("-" * 40)

for epoch in range(num_epochs):
    # ====================
    # 학습 단계
    # ====================
    
    # 1. 순전파: 예측을 계산한다
    # X_train 꼴: (160, 2)
    # predictions 꼴: (160, 1)
    predictions = model(X_train)  # Get model's predicted probabilities
    
    # 2. 손실 계산
    # 예측을 참 이름표와 비교한다
    loss = criterion(predictions, y_train)
    
    # 3. 역전파: 경사를 계산한다
    optimizer.zero_grad()  # Clear old gradients (important!)
    loss.backward()        # Compute new gradients
    
    # 4. 매개변수 갱신
    optimizer.step()       # Update weights and bias using gradients
    
    # ====================
    # 진행 상황 추적
    # ====================
    
    # 학습 정확도를 계산한다
    with torch.no_grad():  # Don't compute gradients for evaluation
        # 확률을 클래스 예측(0 또는 1)으로 바꾼다
        # 확률이 0.5 이상이면 1로, 아니면 0으로 예측한다
        predicted_classes = (predictions >= 0.5).float()
        
        # 정확도를 계산한다: 맞힌 예측의 비율
        correct = (predicted_classes == y_train).sum()
        accuracy = (correct / y_train.shape[0]).item()
    
    # 이력 저장
    train_losses.append(loss.item())
    train_accuracies.append(accuracy)
    
    # 진행 상황 출력
    if (epoch + 1) % print_every == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Loss: {loss.item():.4f} "
              f"Accuracy: {accuracy:.4f}")

print("\nTraining completed!")
print(f"Final loss: {train_losses[-1]:.4f}")
print(f"Final training accuracy: {train_accuracies[-1]:.4f}")


print("\n" + "="*80)
print("PART 5: EVALUATING THE MODEL")
print("="*80)

# ============================================================================
# 5.1: 시험 집합 평가
# ============================================================================
print("\n5.1: Performance on Test Set")
print("-" * 40)

# 시험 집합(보지 않은 데이터)에서 평가한다
model.eval()  # Set model to evaluation mode

with torch.no_grad():  # Don't compute gradients during evaluation
    # 시험 집합의 예측을 얻는다
    test_predictions = model(X_test)
    
    # 확률을 클래스 예측으로 바꾼다
    test_predicted_classes = (test_predictions >= 0.5).float()
    
    # 시험 정확도를 계산한다
    test_correct = (test_predicted_classes == y_test).sum()
    test_accuracy = (test_correct / y_test.shape[0]).item()
    
    # 시험 손실을 계산한다
    test_loss = criterion(test_predictions, y_test)

print(f"Test Loss: {test_loss.item():.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Correct predictions: {int(test_correct)}/{len(y_test)}")


print("\n" + "="*80)
print("PART 6: VISUALIZATION")
print("="*80)

# ============================================================================
# 6.1: 학습 기록
# ============================================================================
print("\n6.1: Creating Visualizations...")

fig = plt.figure(figsize=(15, 10))

# 그림 1: 손실 곡선
plt.subplot(2, 3, 1)
plt.plot(train_losses, 'b-', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# 그림 2: 정확도 곡선
plt.subplot(2, 3, 2)
plt.plot(train_accuracies, 'g-', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Training Accuracy Over Time', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])

# 그림 3: 데이터 분포
plt.subplot(2, 3, 3)
X_train_np = X_train.numpy()
y_train_np = y_train.numpy().flatten()
plt.scatter(X_train_np[y_train_np==0, 0], X_train_np[y_train_np==0, 1], 
           c='blue', label='Class 0', alpha=0.6, edgecolors='k')
plt.scatter(X_train_np[y_train_np==1, 0], X_train_np[y_train_np==1, 1], 
           c='red', label='Class 1', alpha=0.6, edgecolors='k')
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.title('Training Data Distribution', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 그림 4: 결정 경계
plt.subplot(2, 3, 4)
# 결정 경계를 그리기 위한 격자를 만든다
x_min, x_max = X_train_np[:, 0].min() - 1, X_train_np[:, 0].max() + 1
y_min, y_max = X_train_np[:, 1].min() - 1, X_train_np[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# 격자에서 예측한다
with torch.no_grad():
    Z = model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape).numpy()

plt.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.6)
plt.colorbar(label='Probability')
plt.scatter(X_train_np[y_train_np==0, 0], X_train_np[y_train_np==0, 1], 
           c='blue', label='Class 0', edgecolors='k', s=50)
plt.scatter(X_train_np[y_train_np==1, 0], X_train_np[y_train_np==1, 1], 
           c='red', label='Class 1', edgecolors='k', s=50)
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.title('Decision Boundary', fontsize=14, fontweight='bold')
plt.legend()

# 그림 5: 모델 매개변수
plt.subplot(2, 3, 5)
weights = model.linear.weight.data.numpy().flatten()
bias = model.linear.bias.data.numpy()[0]
params_text = f"Learned Parameters:\n\n"
params_text += f"Weight 1 (w1): {weights[0]:.3f}\n"
params_text += f"Weight 2 (w2): {weights[1]:.3f}\n"
params_text += f"Bias (b): {bias:.3f}\n\n"
params_text += f"Decision boundary:\n"
params_text += f"{weights[0]:.3f}*x1 + {weights[1]:.3f}*x2 + {bias:.3f} = 0"
plt.text(0.1, 0.5, params_text, fontsize=12, verticalalignment='center',
         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.axis('off')
plt.title('Model Parameters', fontsize=14, fontweight='bold')

# 그림 6: 성능 요약
plt.subplot(2, 3, 6)
summary_text = f"Performance Summary\n\n"
summary_text += f"Training:\n"
summary_text += f"  Loss: {train_losses[-1]:.4f}\n"
summary_text += f"  Accuracy: {train_accuracies[-1]:.4f}\n\n"
summary_text += f"Testing:\n"
summary_text += f"  Loss: {test_loss.item():.4f}\n"
summary_text += f"  Accuracy: {test_accuracy:.4f}\n\n"
summary_text += f"Dataset:\n"
summary_text += f"  Training samples: {len(X_train)}\n"
summary_text += f"  Test samples: {len(X_test)}\n"
summary_text += f"  Features: {n_features}\n\n"
summary_text += f"Training:\n"
summary_text += f"  Epochs: {num_epochs}\n"
summary_text += f"  Learning rate: {learning_rate}"

plt.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
         family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
plt.axis('off')
plt.title('Summary', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/claude/pytorch_logistic_regression_tutorial/01_basics/simple_classification_results.png',
            dpi=150, bbox_inches='tight')
print("Visualization saved as: simple_classification_results.png")


print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)

print("""
1. 로지스틱 회귀
   - 선형 모형 + 시그모이드 살림
   - 0과 1 사이의 확률을 내놓는다
   - 둘 가름에서는 0.5을 문턱으로 삼는다

2. 익힘 과정
   - 앞으로 걸음: 예측을 셈한다
   - 잃음 셈하기: 어긋남을 잰다
   - 뒤로 걸음: 기울기를 셈한다
   - 매개변수 고치기: 모형을 낫게 한다

3. 종요로운 개념
   - backward() 앞에는 늘 optimizer.zero_grad()을 불러라
   - 따질 때는 model.eval()을 써라
   - 익히지 않을 때는 torch.no_grad()을 써라

4. EVALUATION
   - 익힘 묶음: 모형이 배운 자료
   - 시험 묶음: 두루 얼마나 잘 미치는지
   - 맞음: 옳게 예측한 비율
""")


print("\n" + "="*80)
print("EXERCISES")
print("="*80)

print("""
1. 쉬움: learning_rate을 0.01과 1.0으로 바꾸어라
   - 모여듦에 어떤 영향을 주는가?
   - 어느 쪽이 더 빨리 배우는가?

2. 보통: num_epochs을 100과 5000으로 바꾸어라
   - 더 익히면 늘 나아지는가?
   - 지나치게 맞춰진 낌새를 살펴라

3. 보통: 익힘/시험 나누기를 바꾸어 보아라
   - test_size을 0.1과 0.5으로 바꾸어라
   - How does it affect results?

4. HARD: Implement a function to predict new data:
   def predict(model, x1, x2):
       # 여기에 코드를 작성한다
       pass
   
   Test with: predict(model, 0.5, 0.5)

5. HARD: Add more features to the dataset:
   - Use n_features=4 or 10
   - Modify the model accordingly
   - Compare performance
""")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("""
Great job! You've built your first logistic regression model!

Next tutorial: 03_with_sklearn_data.py
- Work with real-world datasets
- Learn data preprocessing techniques
- Handle different data types

Ready? Run: python 03_with_sklearn_data.py
""")
print("="*80)


if __name__ == "__main__":
    pass
```

## 논의

`LogisticRegression` 클래스는 PyTorch의 `nn.Module` 인터페이스를 사용하여 모델 구조를 감싼다. `forward` 메서드가 계산 그래프를 정의하므로, 학습 중에 PyTorch의 autograd 체계가 경사 계산을 자동으로 처리한다. 이런 모듈식 설계 덕분에 개별 구성 요소를 고치거나 모델을 더 큰 파이프라인에 넣기가 쉬워진다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 분류 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
기본 초기화 상태의 `LogisticRegression`에서 학습 가능한 매개변수의 총 개수를 계산하라. 가중치와 편향을 모두 포함하여 층별로 나누어 세어라.

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
층이나 블록의 개수를 설정할 수 있도록 `LogisticRegression`를 확장하라. `__init__`에 `num_layers` 매개변수를 추가하고 `nn.ModuleList`로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 순회한다. (평범한 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 PyTorch가 모든 매개변수를 최적화 대상으로 등록한다. 시험은 다음과 같이 한다. `for n in [2, 4, 8]: model = LogisticRegression(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
