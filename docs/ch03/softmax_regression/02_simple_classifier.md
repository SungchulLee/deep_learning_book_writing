# 간단한 분류기

2단계: 첫 번째 소프트맥스 분류기 만들기

이 튜토리얼은 PyTorch에서 소프트맥스 회귀에 대한 기초적인 이해를 쌓는다. 코드를 따라가 보면 모델을 세우고, 손실 함수를 정의하고, 경사 하강법으로 학습시키고, 분류 과제에서 성능을 평가하는 법을 알 수 있다.

## 코드

```python
"""
===============================================================================
2단계: 첫 소프트맥스 분류기 짓기
===============================================================================
어려움: 첫걸음~가운데
미리 알아 둘 것: 1단계, 기본 PyTorch
학습 목표:
  - 여러 클래스 분류을 위한 단순한 신경망을 짓는다
  - 학습 루프의 짜임을 이해한다
  - 분류 테두리를 그림으로 본다
  - 모델의 성능을 평가한다

소요 시간: 30~45분
===============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.model_selection import train_test_split

# 재현성을 위한 난수 시드 설정
torch.manual_seed(42)
np.random.seed(42)

print("=" * 80)
print("LEVEL 2: BUILDING YOUR FIRST SOFTMAX CLASSIFIER")
print("=" * 80)


# =============================================================================
# 1부: 합성 데이터셋 생성
# =============================================================================
print("\n" + "=" * 80)
print("PART 1: Creating Synthetic Data")
print("=" * 80)

def create_dataset(dataset_type='blobs', n_samples=1000):
    """
    분류을 위한 인공 2차원 데이터셋을 만든다.
    
    Args:
        dataset_type (str): 'blobs', 'moons', 'circles' 가운데 하나
        n_samples (int): 만들 표본의 수
    
    Returns:
        X (np.array): 모양이 (n_samples, 2)인 특징
        y (np.array): 모양이 (n_samples,)인 레이블
    """
    if dataset_type == 'blobs':
        # 잘 분리된 군집 (가장 쉽다)
        X, y = make_blobs(n_samples=n_samples, centers=3, n_features=2,
                         center_box=(-5, 5), random_state=42)
    elif dataset_type == 'moons':
        # 서로 맞물린 반원 두 개 (중간 난이도)
        X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=42)
    elif dataset_type == 'circles':
        # 동심원 (더 어렵다 - 비선형성이 필요하다)
        X, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.5,
                           random_state=42)
    else:
        raise ValueError("dataset_type must be 'blobs', 'moons', or 'circles'")
    
    return X, y


# 데이터셋을 만든다
X, y = create_dataset('blobs', n_samples=1000)

print(f"Dataset shape: X = {X.shape}, y = {y.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Class distribution: {np.bincount(y)}")
print(f"Feature range: X_min = {X.min():.2f}, X_max = {X.max():.2f}")


# 데이터셋을 시각화한다
def plot_data(X, y, title="Dataset Visualization"):
    """클래스별로 색을 입힌 2차원 데이터 점을 그린다."""
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis',
                         alpha=0.6, edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter, label='Class')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    return plt


# 그림을 보려면 주석을 푼다
# plot_data(X, y, "Original Dataset")
# plt.show()


# 학습 집합과 시험 집합으로 나눈다
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")


# PyTorch 텐서로 변환
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

print(f"\nTensor shapes:")
print(f"  X_train: {X_train_tensor.shape}")
print(f"  y_train: {y_train_tensor.shape}")


# =============================================================================
# 2부: 신경망 모델 정의
# =============================================================================
print("\n" + "=" * 80)
print("PART 2: Building the Neural Network")
print("=" * 80)

class SoftmaxClassifier(nn.Module):
    """
    여러 클래스 분류을 위한 단순한 앞먹임 신경망.
    
    Architecture:
        입력 층(특징 2개)
          ↓
        은닉층(뉴런 64개) + ReLU
          ↓
        은닉층(뉴런 32개) + ReLU
          ↓
        출력 층(클래스 3개) → 로짓
    
    눈여겨볼 것: 수치가 든든하도록 CrossEntropyLoss이 안에서 다루므로
          forward()에서는 소프트맥스를 걸지 않는다.
    """
    
    def __init__(self, input_size=2, hidden_size1=64, hidden_size2=32, num_classes=3):
        """
        망의 층을 초기값 잡는다.
        
        Args:
            input_size (int): 입력 특징의 수
            hidden_size1 (int): 첫 은닉층의 뉴런 수
            hidden_size2 (int): 둘째 은닉층의 뉴런 수
            num_classes (int): 출력 클래스의 수
        """
        super(SoftmaxClassifier, self).__init__()
        
        # 층을 정의한다
        self.fc1 = nn.Linear(input_size, hidden_size1)      # First hidden layer
        self.relu1 = nn.ReLU()                              # Activation function
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)    # Second hidden layer
        self.relu2 = nn.ReLU()                              # Activation function
        self.fc3 = nn.Linear(hidden_size2, num_classes)     # Output layer (logits)
        
    def forward(self, x):
        """
        망을 지나는 순전파.
        
        Args:
            x (torch.Tensor): 모양이 (batch_size, input_size)인 입력 텐서
        
        Returns:
            torch.Tensor: 모양이 (batch_size, num_classes)인 로짓
        """
        out = self.fc1(x)      # Apply first linear transformation
        out = self.relu1(out)  # Apply ReLU activation
        out = self.fc2(out)    # Apply second linear transformation
        out = self.relu2(out)  # Apply ReLU activation
        out = self.fc3(out)    # Get final logits (no activation!)
        return out


# 모델을 만든다
model = SoftmaxClassifier(input_size=2, hidden_size1=64, hidden_size2=32, num_classes=3)

print("Model Architecture:")
print(model)
print("\n" + "-" * 80)

# 매개변수 개수 세기
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")


# =============================================================================
# 3부: 학습 구성 요소 준비
# =============================================================================
print("\n" + "=" * 80)
print("PART 3: Setting Up Training")
print("=" * 80)

# 손실 함수
criterion = nn.CrossEntropyLoss()
print("Loss function: CrossEntropyLoss")
print("  - Combines softmax + log + negative log likelihood")
print("  - Takes logits (raw scores) as input")
print("  - Takes class indices as targets")

# 최적화기
learning_rate = 0.01
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print(f"\nOptimizer: Adam")
print(f"  - Learning rate: {learning_rate}")
print("  - Adaptive learning rate for each parameter")
print("  - Good default choice for most problems")

# 학습 에폭 수
num_epochs = 100
print(f"\nTraining for {num_epochs} epochs")


# =============================================================================
# 4부: 학습 루프
# =============================================================================
print("\n" + "=" * 80)
print("PART 4: Training the Model")
print("=" * 80)

def train_model(model, X_train, y_train, X_test, y_test, 
                criterion, optimizer, num_epochs=100, verbose=True):
    """
    모델을 익히고 자를 좇는다.
    
    Args:
        model: 익힐 PyTorch 모델
        X_train, y_train: 학습 데이터
        X_test, y_test: 시험 데이터
        criterion: 손실 함수
        optimizer: 가장 좋게 하는 알고리즘
        num_epochs: 학습 루프 수
        verbose: 나아가는 모습을 찍을지 여부
    
    Returns:
        dict: 학습 자취(손실과 정확도)
    """
    # 학습 기록을 저장한다
    history = {
        'train_loss': [],
        'test_loss': [],
        'train_acc': [],
        'test_acc': []
    }
    
    for epoch in range(num_epochs):
        # =====================================================================
        # 학습 단계
        # =====================================================================
        model.train()  # Set model to training mode
        
        # 순전파
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # 역전파와 최적화
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update parameters
        
        # 학습 정확도를 계산한다
        _, predicted = torch.max(outputs.data, 1)
        train_correct = (predicted == y_train).sum().item()
        train_acc = train_correct / len(y_train)
        
        # =====================================================================
        # 평가 단계
        # =====================================================================
        model.eval()  # Set model to evaluation mode
        
        with torch.no_grad():  # Disable gradient computation
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            
            # 시험 정확도를 계산한다
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_correct = (test_predicted == y_test).sum().item()
            test_acc = test_correct / len(y_test)
        
        # 지표를 저장한다
        history['train_loss'].append(loss.item())
        history['test_loss'].append(test_loss.item())
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        # 진행 상황 출력
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                  f"Train Loss: {loss.item():.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss.item():.4f} | "
                  f"Test Acc: {test_acc:.4f}")
    
    return history


# 모델을 학습시킨다
print("Starting training...\n")
history = train_model(
    model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor,
    criterion, optimizer, num_epochs=num_epochs, verbose=True
)

print("\n✅ Training complete!")


# =============================================================================
# 5부: 학습 진행 상황 시각화
# =============================================================================
print("\n" + "=" * 80)
print("PART 5: Analyzing Training Results")
print("=" * 80)

def plot_training_history(history):
    """학습과 시험의 손실 및 정확도를 그린다."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 손실을 그린다
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['test_loss'], label='Test Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 정확도를 그린다
    axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['test_acc'], label='Test Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# 그림들을 보려면 주석을 푼다
# plot_training_history(history)
# plt.show()

# 최종 지표를 출력한다
final_train_acc = history['train_acc'][-1]
final_test_acc = history['test_acc'][-1]
final_train_loss = history['train_loss'][-1]
final_test_loss = history['test_loss'][-1]

print(f"\nFinal Results:")
print(f"  Train Accuracy: {final_train_acc:.2%}")
print(f"  Test Accuracy:  {final_test_acc:.2%}")
print(f"  Train Loss:     {final_train_loss:.4f}")
print(f"  Test Loss:      {final_test_loss:.4f}")


# =============================================================================
# 6부: 결정 경계 시각화
# =============================================================================
print("\n" + "=" * 80)
print("PART 6: Visualizing Decision Boundaries")
print("=" * 80)

def plot_decision_boundaries(model, X, y, title="Decision Boundaries"):
    """
    모델이 배운 분류 테두리를 그림으로 본다.
    
    Args:
        model: 익힌 PyTorch 모델
        X (np.array): 입력 특징
        y (np.array): 참 레이블
        title (str): 그림 제목
    """
    # 격자를 만든다
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # 격자의 각 점에 대해 예측한다
    model.eval()
    with torch.no_grad():
        mesh_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
        Z = model(mesh_tensor)
        Z = torch.argmax(Z, dim=1).numpy()
    
    Z = Z.reshape(xx.shape)
    
    # 그림
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis',
                         edgecolors='black', linewidth=1, s=50)
    plt.colorbar(scatter, label='Class')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    return plt


# 그림을 보려면 주석을 푼다
# plot_decision_boundaries(model, X_test, y_test, 
#                         "Decision Boundaries on Test Set")
# plt.show()


# =============================================================================
# 7부: 예측하기
# =============================================================================
print("\n" + "=" * 80)
print("PART 7: Making Predictions on New Data")
print("=" * 80)

def predict_with_probabilities(model, X_new):
    """
    예측하고 클래스마다의 확률을 보인다.
    
    Args:
        model: 익힌 모델
        X_new (torch.Tensor): 새 입력 표본
    
    Returns:
        predictions (torch.Tensor): 예측한 클래스 번호
        probabilities (torch.Tensor): 클래스마다의 확률
    """
    model.eval()
    with torch.no_grad():
        logits = model(X_new)
        probabilities = torch.softmax(logits, dim=1)  # Convert logits to probs
        predictions = torch.argmax(probabilities, dim=1)
    return predictions, probabilities


# 시험 표본 몇 개에 대해 예측한다
n_samples_to_show = 5
X_sample = X_test_tensor[:n_samples_to_show]
y_sample = y_test_tensor[:n_samples_to_show]

predictions, probabilities = predict_with_probabilities(model, X_sample)

print(f"\nPredictions on {n_samples_to_show} test samples:")
print("-" * 80)
for i in range(n_samples_to_show):
    print(f"Sample {i+1}:")
    print(f"  Features: [{X_sample[i, 0].item():.2f}, {X_sample[i, 1].item():.2f}]")
    print(f"  True class: {y_sample[i].item()}")
    print(f"  Predicted class: {predictions[i].item()}")
    print(f"  Probabilities: {probabilities[i].numpy()}")
    correct = "✓" if predictions[i].item() == y_sample[i].item() else "✗"
    print(f"  Correct? {correct}")
    print()


# =============================================================================
# 8부: 모델 평가
# =============================================================================
print("\n" + "=" * 80)
print("PART 8: Detailed Model Evaluation")
print("=" * 80)

from sklearn.metrics import classification_report, confusion_matrix

# 시험 집합의 모든 예측을 얻는다
model.eval()
with torch.no_grad():
    all_outputs = model(X_test_tensor)
    all_predictions = torch.argmax(all_outputs, dim=1).numpy()

# 분류 보고서
print("\nClassification Report:")
print("-" * 80)
print(classification_report(y_test, all_predictions))

# 혼동 행렬
cm = confusion_matrix(y_test, all_predictions)
print("\nConfusion Matrix:")
print("-" * 80)
print(cm)
print("\nRow = True class, Column = Predicted class")


# =============================================================================
# 9부: 모델 저장하고 불러오기
# =============================================================================
print("\n" + "=" * 80)
print("PART 9: Saving and Loading the Model")
print("=" * 80)

# 모델을 저장한다
model_path = '/home/claude/softmax_regression_tutorial/level_02_model.pth'
torch.save(model.state_dict(), model_path)
print(f"✅ Model saved to: {model_path}")

# 모델을 불러온다
loaded_model = SoftmaxClassifier(input_size=2, hidden_size1=64, 
                                 hidden_size2=32, num_classes=3)
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.eval()
print(f"✅ Model loaded successfully")

# 불러온 모델이 동작하는지 확인한다
with torch.no_grad():
    test_output = loaded_model(X_test_tensor[:5])
    print(f"\nTest prediction from loaded model: {torch.argmax(test_output, dim=1).numpy()}")


# =============================================================================
# 요약
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY - What You Learned")
print("=" * 80)

print("""
✅ 분류을 위한 인공 데이터셋을 만들었다
✅ PyTorch로 여러 층 신경망을 지었다
✅ 온전한 학습 루프를 짰다
✅ 학습과 시험의 자를 좇았다
✅ 분류 테두리를 그림으로 보았다
✅ 확률 어림을 곁들여 예측했다
✅ 자로 모델의 성능을 따졌다
✅ 익힌 모델을 저장하고 불러왔다

학습 루프의 핵심 조각:
------------------------------
1. 순전파: model(X) → 로짓
2. 손실 계산: criterion(logits, y)
3. 기울기 지우기: optimizer.zero_grad()
4. 역전파: loss.backward()
5. 가중치 고치기: optimizer.step()

다음 걸음:
-----------
→ 3단계: 참 데이터셋으로 익히기(MNIST, 패션 MNIST)
→ 4단계: 밑바닥부터 짜기(맞춤 학습)
→ 5단계: 앞선 기법(정칙화, 데이터 불리기)

🎉 잘했다! 첫 분류기를 짓고 익혔다!
""")


if __name__ == "__main__":
    pass
```

## 논의

`SoftmaxClassifier` 클래스는 PyTorch의 `nn.Module` 인터페이스를 사용하여 모델 구조를 감싼다. `forward` 메서드가 계산 그래프를 정의하므로, 학습 중에 PyTorch의 autograd 체계가 경사 계산을 자동으로 처리한다. 이런 모듈식 설계 덕분에 개별 구성 요소를 고치거나 모델을 더 큰 파이프라인에 넣기가 쉬워진다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 다중 클래스 분류 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
기본 초기화 상태의 `SoftmaxClassifier`에서 학습 가능한 매개변수의 총 개수를 계산하라. 가중치와 편향을 모두 포함하여 층별로 나누어 세어라.

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
층이나 블록의 개수를 설정할 수 있도록 `SoftmaxClassifier`를 확장하라. `__init__`에 `num_layers` 매개변수를 추가하고 `nn.ModuleList`로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 순회한다. (평범한 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 PyTorch가 모든 매개변수를 최적화 대상으로 등록한다. 시험은 다음과 같이 한다. `for n in [2, 4, 8]: model = SoftmaxClassifier(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
