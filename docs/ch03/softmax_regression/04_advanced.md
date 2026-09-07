# 심화

4단계: 소프트맥스 회귀의 심화 기법

이 튜토리얼은 PyTorch에서 소프트맥스 회귀에 대한 기초적인 이해를 쌓는다. 코드를 따라가 보면 모델을 세우고, 손실 함수를 정의하고, 경사 하강법으로 학습시키고, 분류 과제에서 성능을 평가하는 법을 알 수 있다.

## 코드

```python
"""
===============================================================================
4단계: 앞선 소프트맥스 회귀 솜씨
===============================================================================
어려움: 가운데~앞선
미리 알아 둘 것: 1, 2, 3단계
배움 목표:
  - 소프트맥스 회귀를 맨바닥부터 짠다(넘파이)
  - 앞선 정칙화 솜씨(L2, 드롭아웃, 묶음 정규화)
  - 배움 빠르기 짜기
  - 일찍 멈추기
  - 기울기 자르기
  - 맞춤 잃음 함수와 자

마치는 데 드는 때: 60~90분
===============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import time

# 난수 씨앗을 설정한다
torch.manual_seed(42)
np.random.seed(42)

print("=" * 80)
print("LEVEL 4: ADVANCED SOFTMAX REGRESSION TECHNIQUES")
print("=" * 80)


# =============================================================================
# 1부: 바닥부터 만드는 소프트맥스 회귀 (NumPy)
# =============================================================================
print("\n" + "=" * 80)
print("PART 1: Implementing Softmax Regression from Scratch")
print("=" * 80)

class SoftmaxRegressionNumPy:
    """
    넘파이만으로 맨바닥부터 짠 소프트맥스 회귀.
    PyTorch이 속에서 무엇을 하는지 이해하는 데 도움이 된다.
    """
    
    def __init__(self, input_dim, num_classes, lr=0.01, reg_lambda=0.01):
        """
        소프트맥스 회귀 모형의 첫자리를 잡는다.
        
        Args:
            input_dim: 들임 특징의 수
            num_classes: 내놓음 갈래의 수
            lr: 배움 빠르기
            reg_lambda: L2 정칙화 매개변수
        """
        # 가중치와 편향을 작은 무작위 값으로 초기화한다
        self.W = np.random.randn(input_dim, num_classes) * 0.01
        self.b = np.zeros((1, num_classes))
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.loss_history = []
        
    def softmax(self, z):
        """
        z의 줄마다 소프트맥스 값을 셈한다.
        
        Args:
            z: 꼴이 (batch_size, num_classes)인 로짓
        
        Returns:
            꼴이 (batch_size, num_classes)인 확률
        """
        # 수치적 안정성을 위해 최댓값을 뺀다
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        앞으로 걸음: 갈래 확률을 셈한다.
        
        Args:
            X: 꼴이 (batch_size, input_dim)인 들임 특징
        
        Returns:
            꼴이 (batch_size, num_classes)인 확률
        """
        logits = np.dot(X, self.W) + self.b
        probs = self.softmax(logits)
        return probs
    
    def compute_loss(self, X, y):
        """
        L2 정칙화를 곁들인 엇갈린 엔트로피 잃음을 셈한다.
        
        Args:
            X: 들임 특징
            y: 참 이름표(갈래 번호)
        
        Returns:
            평균 잃음 값
        """
        m = X.shape[0]  # Number of samples
        probs = self.forward(X)
        
        # 교차 엔트로피 손실
        # 표본마다 참 클래스의 확률을 얻는다
        correct_logprobs = -np.log(probs[range(m), y] + 1e-10)
        data_loss = np.sum(correct_logprobs) / m
        
        # L2 정칙화 손실
        reg_loss = 0.5 * self.reg_lambda * np.sum(self.W * self.W)
        
        total_loss = data_loss + reg_loss
        return total_loss
    
    def backward(self, X, y):
        """
        뒤로 걸음: 기울기를 셈한다.
        
        Args:
            X: 들임 특징
            y: 참 이름표
        """
        m = X.shape[0]
        probs = self.forward(X)
        
        # 로짓에 대한 손실의 경사를 계산한다
        # 핵심 통찰이다: d(loss)/d(logits) = (probs - one_hot_labels) / m
        dlogits = probs.copy()
        dlogits[range(m), y] -= 1  # Subtract 1 from true class probabilities
        dlogits /= m
        
        # 가중치와 편향에 대한 경사를 계산한다
        dW = np.dot(X.T, dlogits) + self.reg_lambda * self.W  # Add L2 gradient
        db = np.sum(dlogits, axis=0, keepdims=True)
        
        return dW, db
    
    def train_step(self, X, y):
        """
        기울기 내림을 한 걸음 한다.
        
        Args:
            X: 익힘 특징
            y: 익힘 이름표
        """
        # 경사를 계산한다
        dW, db = self.backward(X, y)
        
        # 가중치를 갱신한다
        self.W -= self.lr * dW
        self.b -= self.lr * db
        
        # 손실을 계산하여 저장한다
        loss = self.compute_loss(X, y)
        self.loss_history.append(loss)
        
        return loss
    
    def predict(self, X):
        """
        갈래 이름표를 예측한다.
        
        Args:
            X: 들임 특징
        
        Returns:
            예측한 갈래 번호
        """
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
    
    def accuracy(self, X, y):
        """
        맞음을 셈한다.
        
        Args:
            X: 들임 특징
            y: 참 이름표
        
        Returns:
            맞음 값
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


# 시험용 합성 데이터를 생성한다
print("Generating synthetic dataset...")
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                          n_redundant=5, n_classes=3, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Features: {X_train.shape[1]}")
print(f"Classes: {len(np.unique(y))}")

# NumPy 모델을 학습시킨다
print("\nTraining NumPy implementation...")
model_numpy = SoftmaxRegressionNumPy(
    input_dim=X_train.shape[1],
    num_classes=3,
    lr=0.1,
    reg_lambda=0.01
)

num_epochs = 200
for epoch in range(num_epochs):
    loss = model_numpy.train_step(X_train, y_train)
    
    if (epoch + 1) % 50 == 0:
        train_acc = model_numpy.accuracy(X_train, y_train)
        test_acc = model_numpy.accuracy(X_test, y_test)
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}, "
              f"Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")

final_test_acc = model_numpy.accuracy(X_test, y_test)
print(f"\n✅ NumPy implementation final test accuracy: {final_test_acc:.2%}")


# =============================================================================
# 2부: 배치 정규화를 넣은 고급 모델
# =============================================================================
print("\n" + "=" * 80)
print("PART 2: Batch Normalization")
print("=" * 80)

"""
묶음 정규화:
-------------------
- 층마다 들임의 잣대를 맞춘다
- 익힘을 든든하게 한다
- 배움 빠르기를 더 크게 쓸 수 있다
- 정칙화 노릇을 한다
- 안쪽 함께 바뀜의 옮겨감을 줄인다
"""

class AdvancedClassifier(nn.Module):
    """
    묶음 정규화를 곁들인 신경망.
    
    묶음 정규화는 층의 들임을 맞추어 다음을 이룬다.
    1. 익힘을 빠르게 한다
    2. 첫자리 잡기에 덜 흔들린다
    3. 정칙화 노릇을 한다
    """
    
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes,
                 use_batchnorm=True, dropout_rate=0.3):
        super(AdvancedClassifier, self).__init__()
        
        self.use_batchnorm = use_batchnorm
        
        # 첫 번째 층
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1) if use_batchnorm else nn.Identity()
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # 두 번째 층
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2) if use_batchnorm else nn.Identity()
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # 출력층
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)      # Batch normalization here
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)      # Batch normalization here
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x


# 데이터셋들을 만든다
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 배치 정규화가 있는 모델과 없는 모델을 비교한다
print("\nComparing models with/without batch normalization...")

models_config = [
    ("Without BatchNorm", False),
    ("With BatchNorm", True)
]

for name, use_bn in models_config:
    print(f"\nTraining: {name}")
    print("-" * 40)
    
    model = AdvancedClassifier(
        input_size=20,
        hidden_size1=64,
        hidden_size2=32,
        num_classes=3,
        use_batchnorm=use_bn,
        dropout_rate=0.3
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 빠른 학습
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    
    # 평가한다
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_t).float().mean()
    
    print(f"Final test accuracy: {accuracy:.4f}")


# =============================================================================
# 3부: 학습률 일정 조절
# =============================================================================
print("\n" + "=" * 80)
print("PART 3: Learning Rate Scheduling")
print("=" * 80)

"""
배움 빠르기 짜기:
------------------------
붙박인 배움 빠르기를 쓰는 대신 다음을 할 수 있다.
- 처음에 빠르게 배우도록 큰 배움 빠르기로 비롯한다
- 곱게 다듬으려고 차츰 줄인다
- 여러 꾀를 쓴다: 계단 잦아들기, 지수, 코사인 따위
"""

class LRSchedulerDemo:
    """여러 학습률 일정을 보여준다."""
    
    @staticmethod
    def step_lr_schedule(initial_lr, epoch, step_size=30, gamma=0.1):
        """
        계단 잦아들기: step_size 판마다 배움 빠르기를 gamma배로 줄인다.
        
        보기: 배움 빠르기 = 0.1 → 0.01 → 0.001(gamma=0.1, step_size=30이면)
        """
        return initial_lr * (gamma ** (epoch // step_size))
    
    @staticmethod
    def exponential_schedule(initial_lr, epoch, gamma=0.95):
        """
        지수 잦아들기: 배움 빠르기 = initial_lr * gamma^epoch
        
        매끄럽고 이어지게 잦아든다.
        """
        return initial_lr * (gamma ** epoch)
    
    @staticmethod
    def cosine_schedule(initial_lr, epoch, total_epochs):
        """
        코사인 달구기: 배움 빠르기가 코사인 굽이를 따른다.
        
        크게 비롯해 매끄럽게 줄어든다. 요즘 익힘에서 널리 쓴다.
        """
        return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))


# 여러 일정을 시각화한다
print("Visualizing learning rate schedules...")
epochs = range(100)
initial_lr = 0.1

schedules = {
    'Constant': [initial_lr] * 100,
    'Step Decay': [LRSchedulerDemo.step_lr_schedule(initial_lr, e) for e in epochs],
    'Exponential': [LRSchedulerDemo.exponential_schedule(initial_lr, e) for e in epochs],
    'Cosine': [LRSchedulerDemo.cosine_schedule(initial_lr, e, 100) for e in epochs]
}

# 그리려면 주석을 푼다
# plt.figure(figsize=(10, 6))
# for name, lrs in schedules.items():
#     plt.plot(epochs, lrs, label=name, linewidth=2)
# plt.xlabel('Epoch')
# plt.ylabel('Learning Rate')
# plt.title('Learning Rate Schedules')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()


# 학습률 일정을 적용해 학습한다
print("\nTraining with StepLR scheduler...")
model_scheduled = AdvancedClassifier(20, 64, 32, 3, use_batchnorm=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_scheduled.parameters(), lr=0.01)

# 스케줄러를 만든다: 20 에폭마다 학습률에 0.1을 곱한다
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

num_epochs = 60
lr_history = []

for epoch in range(num_epochs):
    model_scheduled.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model_scheduled(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    
    # 스케줄러를 한 단계 진행시킨다
    scheduler.step()
    
    # 학습률을 추적한다
    current_lr = optimizer.param_groups[0]['lr']
    lr_history.append(current_lr)
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: Learning Rate = {current_lr:.6f}")

print("✅ Training with LR scheduling complete")


# =============================================================================
# 4부: 조기 종료
# =============================================================================
print("\n" + "=" * 80)
print("PART 4: Early Stopping")
print("=" * 80)

"""
일찍 멈추기:
--------------
다짐 잃음이 더 나아지지 않으면 익힘을 멈춘다.
지나치게 맞춰짐을 막고 때를 아낀다.
"""

class EarlyStopping:
    """
    다짐 잃음이 더 나아지지 않으면 익힘을 멈추는 일찍 멈추기.
    """
    
    def __init__(self, patience=5, min_delta=0.001, verbose=True):
        """
        Args:
            patience: 멈추기 앞에 몇 판을 기다릴지
            min_delta: 나아졌다고 볼 가장 작은 바뀜
            verbose: 알림을 찍는다
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
        
    def __call__(self, val_loss, model):
        """
        익힘을 멈춰야 할지 살핀다.
        
        Args:
            val_loss: 이제 다짐 잃음
            model: 이제 모형(가장 좋은 판을 갈무리하려고)
        
        Returns:
            멈춰야 하면 True, 아니면 False
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print(f"  Validation loss improved: {self.best_loss:.4f} → {val_loss:.4f}")
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
            self.counter = 0
        
        return self.early_stop


# 조기 종료를 보여준다
print("Training with early stopping...")

# 데이터를 학습과 검증으로 나눈다
val_size = int(0.2 * len(train_dataset))
train_size = len(train_dataset) - val_size
train_subset, val_subset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size]
)

train_loader_es = DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader_es = DataLoader(val_subset, batch_size=32, shuffle=False)

model_es = AdvancedClassifier(20, 64, 32, 3, use_batchnorm=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_es.parameters(), lr=0.001)

early_stopping = EarlyStopping(patience=10, min_delta=0.001, verbose=True)

max_epochs = 100
for epoch in range(max_epochs):
    # 학습
    model_es.train()
    train_loss = 0
    for X_batch, y_batch in train_loader_es:
        optimizer.zero_grad()
        outputs = model_es(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # 검증
    model_es.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader_es:
            outputs = model_es(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader_es)
    avg_val_loss = val_loss / len(val_loader_es)
    
    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, "
          f"Val Loss = {avg_val_loss:.4f}")
    
    # 조기 종료 여부를 확인한다
    if early_stopping(avg_val_loss, model_es):
        print(f"\n✅ Early stopping triggered at epoch {epoch+1}")
        # 가장 좋은 모델을 불러온다
        model_es.load_state_dict(early_stopping.best_model)
        break

if epoch == max_epochs - 1:
    print(f"\n✅ Training completed all {max_epochs} epochs")


# =============================================================================
# 5부: 경사 자르기
# =============================================================================
print("\n" + "=" * 80)
print("PART 5: Gradient Clipping")
print("=" * 80)

"""
기울기 자르기:
-----------------
기울기의 크기를 옭아매어 터지는 것을 막는다.
익힘을 든든하게 하며 특히 되도는 신경망과 LSTM에 쓸모 있다.
"""

def train_with_gradient_clipping(model, train_loader, criterion, optimizer,
                                max_norm=1.0, num_epochs=10):
    """
    기울기 자르기를 곁들여 모형을 익힌다.
    
    Args:
        max_norm: 기울기의 가장 큰 노름
    """
    print(f"Training with gradient clipping (max_norm={max_norm})...")
    
    for epoch in range(num_epochs):
        model.train()
        total_grad_norm = 0
        num_batches = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # 경사를 자른다
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            total_grad_norm += grad_norm
            num_batches += 1
            
            optimizer.step()
        
        avg_grad_norm = total_grad_norm / num_batches
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: Avg gradient norm = {avg_grad_norm:.4f}")
    
    print("✅ Training with gradient clipping complete")


model_clip = AdvancedClassifier(20, 64, 32, 3, use_batchnorm=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_clip.parameters(), lr=0.001)

train_with_gradient_clipping(model_clip, train_loader, criterion, optimizer,
                            max_norm=1.0, num_epochs=20)


# =============================================================================
# 6부: 사용자 정의 손실 함수 - 이름표 평활화
# =============================================================================
print("\n" + "=" * 80)
print("PART 6: Label Smoothing")
print("=" * 80)

"""
레이블 스무딩:
---------------
딱딱한 과녁(0이나 1) 대신 부드러운 과녁을 쓴다.
보기: [0, 1, 0] 대신 [0.05, 0.9, 0.05]을 쓴다

Benefits:
- 지나친 자신을 막는다
- 정칙화 노릇을 한다
- 두루 미침이 나아지는 일이 잦다
"""

class LabelSmoothingCrossEntropy(nn.Module):
    """
    레이블 스무딩을 곁들인 엇갈린 엔트로피 잃음.
    """
    
    def __init__(self, num_classes, smoothing=0.1):
        """
        Args:
            num_classes: 갈래의 수
            smoothing: 매끄럽게 하기 매개변수(0이면 안 함, 1이면 고르게)
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, predictions, targets):
        """
        레이블 스무딩 잃음을 셈한다.
        
        Args:
            predictions: 모형 로짓 (batch_size, num_classes)
            targets: 참 갈래 번호 (batch_size,)
        """
        # 로그 소프트맥스를 적용한다
        log_probs = torch.nn.functional.log_softmax(predictions, dim=1)
        
        # 평활화된 목표를 만든다
        with torch.no_grad():
            # 균등분포로 시작한다
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
            # 참 클래스에 더 높은 확률을 준다
            smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        # 손실을 계산한다
        loss = torch.sum(-smooth_targets * log_probs, dim=1)
        return loss.mean()


# 표준 교차 엔트로피와 이름표 평활화를 비교한다
print("Comparing standard CrossEntropy vs Label Smoothing...")

configs = [
    ("Standard CrossEntropy", nn.CrossEntropyLoss(), 0.0),
    ("Label Smoothing (0.1)", LabelSmoothingCrossEntropy(3, smoothing=0.1), 0.1)
]

for name, criterion, smoothing in configs:
    print(f"\n{name}:")
    print("-" * 40)
    
    model = AdvancedClassifier(20, 64, 32, 3, use_batchnorm=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 몇 에폭 동안 학습한다
    for epoch in range(30):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    
    # 평가한다
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_t).float().mean()
    
    print(f"Test accuracy: {accuracy:.4f}")


# =============================================================================
# 요약
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY - What You Learned")
print("=" * 80)

print("""
✅ 넘파이로 소프트맥스 회귀를 맨바닥부터 짰다
✅ 익힘을 든든하게 하려고 묶음 정규화를 썼다
✅ 배움 빠르기 짜기 꾀를 걸었다
✅ 지나치게 맞춰짐을 막으려고 일찍 멈추기를 짰다
✅ 익힘을 든든하게 하려고 기울기 자르기를 썼다
✅ 정칙화 솜씨로 레이블 스무딩을 살펴보았다

앞선 솜씨 간추림:
---------------------------
1. 묶음 정규화
   - 층의 들임을 맞춘다
   - 익힘을 빠르게 한다
   - 정칙화 노릇을 한다

2. 배움 빠르기 짜기
   - 계단 잦아들기: 사이를 두고 배움 빠르기를 떨어뜨린다
   - 지수: 매끄럽게 이어지는 잦아들기
   - 코사인: 코사인 굽이를 따른다

3. 일찍 멈추기
   - 다짐 잃음을 지켜보아라
   - 나아짐이 없으면 멈춘다
   - 가장 좋은 모형을 갈무리한다

4. 기울기 자르기
   - 기울기가 터지는 것을 막는다
   - 기울기의 크기를 옭아맨다
   - 든든함을 높인다

5. 레이블 스무딩
   - 딱딱한 과녁 대신 부드러운 과녁
   - 지나친 자신을 막는다
   - 두루 미침이 더 낫다

가장 좋은 버릇:
--------------
• 더 깊은 그물에는 묶음 정규화를 쓴다
• 배움 빠르기를 크게 비롯해 차츰 낮춘다
• 다짐 묶음과 함께 늘 일찍 멈추기를 쓴다
• 되도는 신경망이나 흔들리는 익힘에서는 기울기를 자른다
• 두루 미침을 높이려면 레이블 스무딩을 살펴본다

다음 걸음:
-----------
→ 5단계: 여러 자료 묶음과 얼개를 견주기
→ 여러 솜씨를 뒤섞어 실험해 보기
→ 제 자료 묶음에 써 보기

🎉 잘했다! 앞선 솜씨를 익혔다!
""")


if __name__ == "__main__":
    pass
```

## 논의

이 구현은 5개의 클래스(`SoftmaxRegressionNumPy`, `AdvancedClassifier`, `LRSchedulerDemo`, `EarlyStopping`, 그리고 하나 더)를 정의하며, 이들이 함께 작동하여 완전한 소프트맥스 회귀 구조를 이룬다. 각 클래스가 서로 다른 구성 요소를 감싸므로 코드가 모듈식이 되고 확장하기 쉬워진다. `forward` 메서드들이 PyTorch가 자동 미분에 사용하는 계산 그래프를 정의한다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 다중 클래스 분류 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
기본 초기화 상태의 `SoftmaxRegressionNumPy`에서 학습 가능한 매개변수의 총 개수를 계산하라. 가중치와 편향을 모두 포함하여 층별로 나누어 세어라.

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)` 각각에는 `in_features * out_features`개의 가중치 매개변수와 (`bias=False`가 아닌 한) `out_features`개의 편향 매개변수가 있다. `nn.Conv2d(in_c, out_c, k)`에는 `in_c * out_c * k * k`개의 가중치와 `out_c`개의 편향이 있다. `nn.Embedding(num, dim)`에는 `num * dim`개의 매개변수가 있다. 모든 층에 대해 더하면 된다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

---

**연습문제 2.**
최적화기를 Adam으로 바꾸고(`torch.optim.Adam`에 `lr=0.001`을 쓴다) 원래 최적화기와 학습 수렴을 비교하라. 두 손실 곡선을 같은 그래프에 그려라.

??? success "연습문제 2 풀이"
    최적화기를 만드는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 바꾼다. Adam은 매개변수마다 적응적인 학습률과 운동량 추정값을 유지하므로 초반 에폭에서 대체로 더 빠르게 수렴한다. Adam의 손실 곡선은 보통 처음 몇 에폭에서 더 가파르게 떨어지지만, 최적점 근처에서는 운동량을 쓴 SGD보다 조금 더 흔들릴 수 있다. 공정한 비교를 위해 둘을 같은 난수 씨앗과 같은 에폭 수로 실행하라.

---

**연습문제 3.**
은닉 크기가 $h$이고 입력 크기가 $x$로 같을 때 LSTM 셀과 GRU 셀의 매개변수 개수를 비교하라. 어느 쪽이 더 적으며 그 이유는 무엇인가?

??? success "연습문제 3 풀이"
    LSTM에는 4개의 게이트(입력, 망각, 셀, 출력)가 있고 각 게이트가 입력과 은닉 상태 양쪽에 대한 가중치 행렬을 가지므로 $4 \times (x \cdot h + h \cdot h + h) = 4(xh + h^2 + h)$개의 매개변수를 갖는다. GRU에는 3개의 게이트(재설정, 갱신, 새 상태)가 있어 $3 \times (x \cdot h + h \cdot h + h) = 3(xh + h^2 + h)$개이다. GRU는 게이트를 4개 대신 3개 쓰고 셀 상태와 은닉 상태를 합치므로 LSTM의 75%에 해당하는 매개변수를 갖는다. 실무에서 GRU는 매개변수가 적은데도 LSTM에 견줄 만한 성능을 내는 경우가 많다.

---

**연습문제 4.**
층이나 블록의 개수를 설정할 수 있도록 `SoftmaxRegressionNumPy`를 확장하라. `__init__`에 `num_layers` 매개변수를 추가하고 `nn.ModuleList`로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 순회한다. (평범한 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 PyTorch가 모든 매개변수를 최적화 대상으로 등록한다. 시험은 다음과 같이 한다. `for n in [2, 4, 8]: model = SoftmaxRegressionNumPy(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
