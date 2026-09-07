# 간단한 이진 분류

02_simple_binary_classification.py - 첫 번째 로지스틱 회귀 모델

이 튜토리얼은 PyTorch에서 로지스틱 회귀에 대한 기초적인 이해를 쌓는다. 코드를 따라가 보면 모델을 세우고, 손실 함수를 정의하고, 경사 하강법으로 학습시키고, 분류 과제에서 성능을 평가하는 법을 알 수 있다.

## 코드

```python
"""
================================================================================
02_simple_binary_classification.py - Your First Logistic Regression Model
================================================================================

배움 목표:
- Understand binary classification problems
- Implement logistic regression from scratch using PyTorch
- Learn the sigmoid function and its properties
- Train a model using gradient descent
- Evaluate model performance

PREREQUISITES:
- Completed 01_introduction.py
- Understanding of linear models (y = mx + b)
- Basic probability concepts

TIME TO COMPLETE: ~45 minutes

DIFFICULTY: ⭐⭐☆☆☆ (Easy)
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
Binary Classification: Predict one of two classes (0 or 1)

Examples:
  - Email: Spam (1) or Not Spam (0)
  - Medical: Disease (1) or Healthy (0)  
  - Customer: Will Buy (1) or Won't Buy (0)
  - Image: Cat (1) or Dog (0)

In this tutorial:
  - We'll create synthetic data with 2 features (x1, x2)
  - Each sample belongs to class 0 or class 1
  - Goal: Learn to predict the class from features
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
Logistic Regression Model:

Step 1: Linear Combination
    z = w1*x1 + w2*x2 + b
    where w1, w2 are weights, b is bias

Step 2: Sigmoid Activation
    probability = sigmoid(z) = 1 / (1 + e^(-z))
    
Properties of Sigmoid:
    - Maps any value to range (0, 1)
    - sigmoid(0) = 0.5 (decision boundary)
    - sigmoid(large positive) ≈ 1
    - sigmoid(large negative) ≈ 0
    
Step 3: Classification
    If probability >= 0.5: predict class 1
    If probability < 0.5: predict class 0
""")

# ============================================================================
# 2.2: 모델 구현하기
# ============================================================================
print("\n2.2: Implementing in PyTorch")
print("-" * 40)

class LogisticRegression(nn.Module):
    """
    Simple Logistic Regression Model
    
    Architecture:
        Input (n_features) → Linear Layer → Sigmoid → Output (probability)
    
    Parameters:
        n_features (int): Number of input features
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
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, n_features)
            
        Returns:
            probability: Output tensor of shape (batch_size, 1)
                        Values in range (0, 1)
        """
        # 1단계: 선형 변환
        # x shape: (batch_size, n_features)
        # output shape: (batch_size, 1)
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
Binary Cross-Entropy (BCE) Loss:

For a single example:
    If actual label y = 1:
        loss = -log(predicted_probability)
        → Model is punished if it predicts low probability for class 1
    
    If actual label y = 0:
        loss = -log(1 - predicted_probability)
        → Model is punished if it predicts high probability for class 1

Full formula:
    loss = -[y*log(p) + (1-y)*log(1-p)]

Properties:
    - Always positive
    - Smaller is better
    - Heavily penalizes confident wrong predictions
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
Optimizer: Algorithm that updates model parameters (weights and biases)

Common optimizers:
    - SGD (Stochastic Gradient Descent): Basic but reliable
    - Adam: Adaptive learning rate, usually converges faster
    - RMSprop: Good for RNNs
    
We'll use SGD for simplicity and understanding.
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
    # X_train shape: (160, 2)
    # predictions shape: (160, 1)
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
1. LOGISTIC REGRESSION
   - Linear model + Sigmoid activation
   - Outputs probability between 0 and 1
   - Threshold at 0.5 for binary classification

2. TRAINING PROCESS
   - Forward pass: compute predictions
   - Calculate loss: measure error
   - Backward pass: compute gradients
   - Update parameters: improve model

3. IMPORTANT CONCEPTS
   - Always call optimizer.zero_grad() before backward()
   - Use model.eval() during evaluation
   - Use torch.no_grad() when not training

4. EVALUATION
   - Train set: what model learned from
   - Test set: how well it generalizes
   - Accuracy: percentage of correct predictions
""")


print("\n" + "="*80)
print("EXERCISES")
print("="*80)

print("""
1. EASY: Change learning_rate to 0.01 and 1.0
   - How does it affect convergence?
   - Which learns faster?

2. MEDIUM: Change num_epochs to 100 and 5000
   - Does more training always help?
   - Look for overfitting signs

3. MEDIUM: Try different train/test splits
   - Change test_size to 0.1 and 0.5
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
