# MNIST 숫자 분류를 위한 신경망

이 스크립트는 MNIST 숫자 분류를 위한 신경망을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
"""
================================================================================
4단계 - 과제 1: MNIST 숫자 가름을 위한 신경망
================================================================================

배움 목표:
- 온전한 신경망을 맨바닥부터 짓는다
- 참 세상의 자료 묶음(MNIST)으로 익힌다
- 익힘/다짐/시험 나누기를 제대로 짠다
- 배운 기울기 내림 개념을 모두 쓴다
- 맞음 95%를 넘긴다

어려움: ⭐⭐⭐⭐ 과제

걸리는 때: 60~90분

PREREQUISITES:
- 1~3단계를 마쳤을 것
- 신경망을 이해하고 있을 것
- PyTorch에 익숙할 것

과제 밝힘:
--------------------
MNIST은 손글씨 숫자(0~9) 70,000개의 자료 묶음이다.
- 익힘 묶음: 그림 60,000장
- 시험 묶음: 그림 10,000장
- 그림 크기: 28x28 잿빛
- 일: 그림마다 10갈래(0~9) 가운데 하나로 가른다

기계 배움의 대표 잣대 문제다!

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import time

print("="*80)
print("MNIST DIGIT CLASSIFICATION WITH NEURAL NETWORKS")
print("="*80)

# 재현성을 위한 난수 시드 설정
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# 1부: 데이터 적재와 전처리
# ============================================================================
print("\n" + "="*80)
print("PART 1: LOADING MNIST DATASET")
print("="*80)

# 변환 정의
# mean=0.1307, std=0.3081로 정규화(MNIST 통계)
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize
])

# 학습 데이터 내려받아 불러오기
print("Downloading MNIST dataset (this may take a minute)...")
train_dataset = datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

# 테스트 데이터 내려받아 불러오기
test_dataset = datasets.MNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)

# 학습 데이터에서 검증 집합 분리
train_size = int(0.8 * len(train_dataset))  # 80% for training
val_size = len(train_dataset) - train_size  # 20% for validation

train_dataset, val_dataset = random_split(
    train_dataset, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"\n✓ Dataset loaded successfully!")
print(f"  Training samples:   {len(train_dataset):,}")
print(f"  Validation samples: {len(val_dataset):,}")
print(f"  Test samples:       {len(test_dataset):,}")

# 데이터 로더 생성
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"\nBatch size: {batch_size}")
print(f"Batches per epoch: {len(train_loader)}")

# ============================================================================
# 2부: 예시 데이터 시각화
# ============================================================================
print("\n" + "="*80)
print("PART 2: VISUALIZING SAMPLE DATA")
print("="*80)

# 학습 데이터 한 배치 가져오기
examples = iter(train_loader)
example_data, example_targets = next(examples)

print(f"\nBatch shape: {example_data.shape}")  # (batch_size, 1, 28, 28)
print(f"Labels shape: {example_targets.shape}")  # (batch_size,)

# 처음 이미지 10개 그리기
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.flatten()

for i in range(10):
    img = example_data[i].squeeze()  # Remove channel dimension
    label = example_targets[i].item()
    
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f'Label: {label}')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('/home/claude/pytorch_gradient_descent_tutorial/level_4_projects/mnist_samples.png', dpi=150)
print("\n✓ Sample images saved as 'mnist_samples.png'")

# ============================================================================
# 3부: 신경망 구조 정의
# ============================================================================
print("\n" + "="*80)
print("PART 3: NEURAL NETWORK ARCHITECTURE")
print("="*80)

class MNISTNet(nn.Module):
    """
    MNIST 가름을 위한 신경망
    
    Architecture:
    - 들임: 28x28 = 특징 784개
    - 숨은 층 1: 뉴런 128개 + ReLU
    - 숨은 층 2: 뉴런 64개 + ReLU
    - 내놓음 층: 뉴런 10개(숫자마다 하나)
    """
    
    def __init__(self):
        super(MNISTNet, self).__init__()
        
        # 입력층: 784(28x28) → 128
        self.fc1 = nn.Linear(28 * 28, 128)
        
        # 은닉층: 128 → 64
        self.fc2 = nn.Linear(128, 64)
        
        # 출력층: 64 → 10(클래스 0-9)
        self.fc3 = nn.Linear(64, 10)
        
        # 정칙화를 위한 드롭아웃(과적합을 막는다)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        """
        그물을 지나는 앞으로 걸음
        
        Args:
            x: 꼴이 (batch_size, 1, 28, 28)인 들임 텐서
        
        Returns:
            output: 꼴이 (batch_size, 10)인 로짓
        """
        # 이미지 펼치기: (배치, 1, 28, 28) → (배치, 784)
        x = x.view(-1, 28 * 28)
        
        # ReLU 활성화를 쓰는 은닉층 1
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # ReLU 활성화를 쓰는 은닉층 2
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 출력층(활성화 없음 - CrossEntropyLoss를 쓸 것이다)
        x = self.fc3(x)
        
        return x

# 모델 생성
model = MNISTNet()

# 매개변수 개수 세기
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\nModel Architecture:")
print(model)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ============================================================================
# 4부: 학습 준비
# ============================================================================
print("\n" + "="*80)
print("PART 4: TRAINING CONFIGURATION")
print("="*80)

# 손실 함수: 분류를 위한 CrossEntropyLoss
# LogSoftmax와 NLLLoss를 결합한다
criterion = nn.CrossEntropyLoss()

# 최적화기: Adam(3단계에서 즐겨 쓰던 것!)
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 학습률 스케줄러: 검증 손실이 정체되면 학습률을 줄인다
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=3, 
    verbose=True
)

print(f"Loss function: CrossEntropyLoss")
print(f"Optimizer: Adam")
print(f"Learning rate: {learning_rate}")
print(f"Scheduler: ReduceLROnPlateau")

# ============================================================================
# 5부: 학습 함수와 검증 함수
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer):
    """한 판 익힌다"""
    model.train()  # Set model to training mode
    
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # 순전파
        output = model(data)
        loss = criterion(output, target)
        
        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 통계
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion):
    """모형을 다진다"""
    model.eval()  # Set model to evaluation mode
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():  # No gradients needed
        for data, target in val_loader:
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

# ============================================================================
# 6부: 학습 루프
# ============================================================================
print("\n" + "="*80)
print("PART 6: TRAINING")
print("="*80)

n_epochs = 15

# 이력 추적
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

print(f"\nTraining for {n_epochs} epochs...")
print("-" * 80)
print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>10} | {'Val Loss':>10} | {'Val Acc':>10} | {'Time':>7}")
print("-" * 80)

start_time = time.time()

for epoch in range(n_epochs):
    epoch_start = time.time()
    
    # 학습
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    
    # 검증
    val_loss, val_acc = validate(model, val_loader, criterion)
    
    # 학습률 스케줄링
    scheduler.step(val_loss)
    
    # 이력 저장
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    epoch_time = time.time() - epoch_start
    
    print(f"{epoch+1:5d} | {train_loss:10.4f} | {train_acc:9.2f}% | {val_loss:10.4f} | {val_acc:9.2f}% | {epoch_time:6.1f}s")

total_time = time.time() - start_time
print("-" * 80)
print(f"Training completed in {total_time:.1f}s ({total_time/n_epochs:.1f}s per epoch)")

# ============================================================================
# 7부: 테스트 집합 평가
# ============================================================================
print("\n" + "="*80)
print("PART 7: FINAL EVALUATION ON TEST SET")
print("="*80)

test_loss, test_acc = validate(model, test_loader, criterion)

print(f"\nTest Set Results:")
print(f"  Loss: {test_loss:.4f}")
print(f"  Accuracy: {test_acc:.2f}%")

if test_acc > 95:
    print("\n🎉 Congratulations! You achieved >95% accuracy!")
elif test_acc > 90:
    print("\n✓ Good job! Try tuning hyperparameters to reach 95%")
else:
    print("\n→ Try training longer or adjusting the architecture")

# ============================================================================
# 8부: 시각화
# ============================================================================
print("\n" + "="*80)
print("VISUALIZATION")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 그림 1: 손실 곡선
axes[0, 0].plot(train_losses, label='Train', linewidth=2)
axes[0, 0].plot(val_losses, label='Validation', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training and Validation Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 그림 2: 정확도 곡선
axes[0, 1].plot(train_accuracies, label='Train', linewidth=2)
axes[0, 1].plot(val_accuracies, label='Validation', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy (%)')
axes[0, 1].set_title('Training and Validation Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 그림 3: 예측 표본
model.eval()
examples = iter(test_loader)
example_data, example_targets = next(examples)

with torch.no_grad():
    output = model(example_data)
    _, predictions = torch.max(output, 1)

# 처음 테스트 예제 6개 보여주기
for i in range(6):
    ax = axes[1, i//3]
    
    if i < 3:
        idx = i
    else:
        idx = i + 3
    
    img = example_data[idx].squeeze()
    true_label = example_targets[idx].item()
    pred_label = predictions[idx].item()
    
    color = 'green' if true_label == pred_label else 'red'
    
    if i < 3:
        ax.imshow(img, cmap='gray')
        ax.set_title(f'True: {true_label}, Pred: {pred_label}', color=color)
        ax.axis('off')

# 그림 4: 혼동 시각화
axes[1, 1].text(0.5, 0.5, 
                f'Test Accuracy\n{test_acc:.2f}%\n\nTest Loss\n{test_loss:.4f}',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=20,
                transform=axes[1, 1].transAxes)
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('/home/claude/pytorch_gradient_descent_tutorial/level_4_projects/mnist_results.png', dpi=150)
print("\n✓ Results saved as 'mnist_results.png'")
print("\nClose the plot window to continue...")
plt.show()

# ============================================================================
# 9부: 모델 저장
# ============================================================================
print("\n" + "="*80)
print("SAVING MODEL")
print("="*80)

# 모델 가중치 저장
torch.save(model.state_dict(), '/home/claude/pytorch_gradient_descent_tutorial/level_4_projects/mnist_model.pth')
print("\n✓ Model saved as 'mnist_model.pth'")

print("\nTo load the model later:")
print("  model = MNISTNet()")
print("  model.load_state_dict(torch.load('mnist_model.pth'))")
print("  model.eval()")

# ============================================================================
# 10부: 핵심 요점
# ============================================================================
print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print("""
1. 온전한 기계 배움 흐름:
   ✓ 자료 불러오기와 미리 다듬기
   ✓ 익힘/다짐/시험 나누기
   ✓ 모형 매기기
   ✓ 제대로 따지는 익힘 되돌이
   ✓ 그림으로 보기와 모형 갈무리

2. 좋은 버릇:
   • 웃매개변수를 손볼 때는 늘 다짐 묶음을 써라
   • 마지막으로 따질 때까지 시험 묶음에 손대지 마라
   • 정칙화에는 드롭아웃을 써라
   • 배움 빠르기 짜기를 넣어라
   • 나중에 쓰도록 모형을 갈무리하라

3. 움직이는 기울기 내림:
   • Adam 가장 좋게 하개는 빨리 모여든다
   • DataLoader로 하는 묶음 익힘
   • Automatic differentiation handles complex network
   • All concepts from Levels 1-3 applied here!

4. ACHIEVING GOOD PERFORMANCE:
   • >95% accuracy on MNIST is achievable
   • Proper architecture design matters
   • Hyperparameter tuning improves results
   • More training typically helps (to a point)

5. NEXT STEPS:
   • Try different architectures (more/fewer layers)
   • Experiment with hyperparameters
   • Add convolutional layers for better performance
   • Try other datasets (Fashion-MNIST, CIFAR-10)
""")

print("="*80)
print("🎉 CONGRATULATIONS!")
print("="*80)
print("""
You've completed a full neural network project!

You now know how to:
✓ Load and preprocess real datasets
✓ Build neural networks with PyTorch
✓ Train models using gradient descent
✓ Evaluate and visualize results
✓ Save and load trained models

This is the foundation for deep learning!
""")
print("="*80)


if __name__ == "__main__":
    pass
```

## 논의

이 코드는 `requires_grad=True`인 텐서에 대한 연산을 자동으로 추적하는 PyTorch의 autograd 체계를 보여준다. 스칼라 손실에 `.backward()`를 호출하면 autograd가 계산 그래프를 역방향으로 훑으며 연쇄 법칙을 적용해 모든 잎 텐서의 경사를 계산한다. 이 구조가 PyTorch의 모든 신경망 학습을 떠받친다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사 초기화, 역전파, 매개변수 갱신이다. 각 구성 요소가 결정적인 역할을 한다. 최적화기는 갱신 규칙(SGD, Adam 등)을 캡슐화하고 학습률과 모멘텀 상태를 내부에서 관리한다.

경사 추적을 제어하는 것은 정확성과 성능 모두에 필수적이다. `torch.no_grad()` 컨텍스트 관리자는 매개변수 갱신이나 추론처럼 계산 그래프에 포함되어서는 안 되는 연산에 대해 autograd를 끈다. `.detach()` 메서드는 저장소는 공유하지만 그래프와는 분리된 텐서를 만들며, 값을 기록하거나 NumPy로 변환할 때 유용하다.

## 연습문제

**연습문제 1.**
함수 $f(x) = x^3 - 2x^2 + x$를 생각하자. PyTorch autograd를 사용하여 $f'(3)$을 계산하라.

??? success "연습문제 1 풀이"
    ```python
    import torch

    x = torch.tensor(3.0, requires_grad=True)
    f = x**3 - 2*x**2 + x
    f.backward()
    print(x.grad)  # f'(x) = 3x^2 - 4x + 1 = 27 - 12 + 1 = 16.0
    ```

---


**연습문제 2.**
`retain_graph=True` 없이 같은 계산 그래프에 `.backward()`를 두 번 호출하면 오류가 나는 이유를 설명하라. `retain_graph=True`는 메모리 사용량에 어떤 영향을 주는가?

??? success "연습문제 2 풀이"
    기본적으로 PyTorch는 메모리를 아끼기 위해 `.backward()` 후에 계산 그래프를 해제한다. `.backward()`를 두 번째로 호출하면 더 이상 존재하지 않는 그래프를 훑으려 하므로 `RuntimeError`가 발생한다. `retain_graph=True`로 두면 그래프가 메모리에 남아 재사용할 수 있지만, 모든 중간 텐서가 할당된 채로 남으므로 메모리 소비가 늘어난다.

---


**연습문제 3.**
잎 텐서 `w`를 만들고 손실을 계산한 뒤, 경사를 초기화하지 않고 `.backward()`를 세 번 호출하며 매번 `w.grad`를 출력하는 코드를 작성하라. 관찰된 값을 설명하라.

??? success "연습문제 3 풀이"
    ```python
    import torch

    w = torch.tensor(2.0, requires_grad=True)
    for i in range(3):
        loss = (w ** 2).sum()
        loss.backward()
        print(f'After backward {i+1}: w.grad = {w.grad}')
    # 출력: 4.0, 8.0, 12.0
    # 경사가 누적된다. 매 backward가 기존 경사에 2*w = 4.0을 더한다.
    ```
