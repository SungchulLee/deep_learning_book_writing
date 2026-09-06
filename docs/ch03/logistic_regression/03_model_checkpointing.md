# 모델 체크포인트

03_model_checkpointing.py - 모델 저장하고 불러오기

이 튜토리얼은 PyTorch에서 로지스틱 회귀에 대한 기초적인 이해를 쌓는다. 코드를 따라가 보면 모델을 세우고, 손실 함수를 정의하고, 경사 하강법으로 학습시키고, 분류 과제에서 성능을 평가하는 법을 알 수 있다.

## 코드

```python
"""
================================================================================
03_model_checkpointing.py - Saving and Loading Models
================================================================================

LEARNING OBJECTIVES:
- Save and load model weights
- Implement checkpoint system
- Resume training from checkpoints
- Save complete training state
- Best practices for model persistence

TIME TO COMPLETE: ~1 hour
DIFFICULTY: ⭐⭐⭐☆☆ (Intermediate)
================================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json

print("="*80)
print("MODEL CHECKPOINTING AND PERSISTENCE")
print("="*80)

# =============================================================================
# 준비
# =============================================================================

# 체크포인트 디렉터리를 만든다
checkpoint_dir = Path("/home/claude/pytorch_logistic_regression_tutorial/02_intermediate/checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

# 데이터를 준비한다
torch.manual_seed(42)
np.random.seed(42)

X, y = make_classification(n_samples=2000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = torch.FloatTensor(scaler.fit_transform(X_train))
X_test = torch.FloatTensor(scaler.transform(X_test))
y_train = torch.FloatTensor(y_train).reshape(-1, 1)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

# =============================================================================
# 모델
# =============================================================================

class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# =============================================================================
# 체크포인트 함수들
# =============================================================================

def save_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """
    Save complete training state
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        epoch: Current epoch number
        loss: Current loss value
        accuracy: Current accuracy
        filepath: Where to save the checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"✓ Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load checkpoint and restore state
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to restore state
        
    Returns:
        Dictionary with training info
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✓ Checkpoint loaded from {filepath}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Loss: {checkpoint['loss']:.4f}")
    print(f"  Accuracy: {checkpoint['accuracy']:.4f}")
    
    return checkpoint


def save_best_model(model, filepath):
    """모델 가중치만 저장한다 (배포용)"""
    torch.save(model.state_dict(), filepath)
    print(f"✓ Best model saved to {filepath}")


# =============================================================================
# 체크포인트를 쓰는 학습
# =============================================================================

print("\n" + "="*80)
print("TRAINING WITH AUTOMATIC CHECKPOINTING")
print("="*80)

model = LogisticRegression(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
save_every = 10  # Save checkpoint every N epochs
best_accuracy = 0.0

print(f"Training for {num_epochs} epochs")
print(f"Saving checkpoints every {save_every} epochs")
print("-" * 60)

for epoch in range(num_epochs):
    # 학습
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_y in train_loader:
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * len(batch_X)
        predicted_classes = (predictions >= 0.5).float()
        correct += (predicted_classes == batch_y).sum().item()
        total += len(batch_X)
    
    avg_loss = train_loss / total
    accuracy = correct / total
    
    # 검증
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            predictions = model(batch_X)
            predicted_classes = (predictions >= 0.5).float()
            test_correct += (predicted_classes == batch_y).sum().item()
            test_total += len(batch_X)
    
    test_accuracy = test_correct / test_total
    
    # 진행 상황 출력
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1:2d}/{num_epochs}] "
              f"Loss: {avg_loss:.4f} "
              f"Train Acc: {accuracy:.4f} "
              f"Test Acc: {test_accuracy:.4f}")
    
    # N 에폭마다 체크포인트를 저장한다
    if (epoch + 1) % save_every == 0:
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
        save_checkpoint(model, optimizer, epoch, avg_loss, test_accuracy, checkpoint_path)
    
    # 최고 성능 모델 저장
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_model_path = checkpoint_dir / "best_model.pt"
        save_best_model(model, best_model_path)
        print(f"  → New best accuracy: {best_accuracy:.4f}")

print("\nTraining completed!")
print(f"Best test accuracy: {best_accuracy:.4f}")

# =============================================================================
# 학습 이어 하기
# =============================================================================

print("\n" + "="*80)
print("DEMONSTRATING RESUME FROM CHECKPOINT")
print("="*80)

# 새 모델과 최적화기를 만든다
new_model = LogisticRegression(X_train.shape[1])
new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)

# 체크포인트를 불러온다
checkpoint_path = checkpoint_dir / "checkpoint_epoch_20.pt"
checkpoint_info = load_checkpoint(checkpoint_path, new_model, new_optimizer)

# 이 지점부터 학습을 이어 간다
print(f"\nResuming training from epoch {checkpoint_info['epoch']+1}...")
resume_epochs = 10

for epoch in range(checkpoint_info['epoch'], checkpoint_info['epoch'] + resume_epochs):
    # 학습 루프 (앞과 같다)
    new_model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_y in train_loader:
        predictions = new_model(batch_X)
        loss = criterion(predictions, batch_y)
        
        new_optimizer.zero_grad()
        loss.backward()
        new_optimizer.step()
        
        train_loss += loss.item() * len(batch_X)
        predicted_classes = (predictions >= 0.5).float()
        correct += (predicted_classes == batch_y).sum().item()
        total += len(batch_X)
    
    avg_loss = train_loss / total
    accuracy = correct / total
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}] Loss: {avg_loss:.4f} Accuracy: {accuracy:.4f}")

print("\n✓ Successfully resumed and continued training!")

# =============================================================================
# 추론을 위한 불러오기
# =============================================================================

print("\n" + "="*80)
print("LOADING MODEL FOR INFERENCE (DEPLOYMENT)")
print("="*80)

# 배포용으로 깨끗한 모델을 만든다
deployment_model = LogisticRegression(X_train.shape[1])

# 가중치만 불러온다 (최적화기 상태는 제외)
best_model_path = checkpoint_dir / "best_model.pt"
deployment_model.load_state_dict(torch.load(best_model_path))
deployment_model.eval()

print("✓ Model loaded for inference")

# 추론을 시험한다
with torch.no_grad():
    sample_input = X_test[:5]  # 5 test samples
    predictions = deployment_model(sample_input)
    predicted_classes = (predictions >= 0.5).float()
    
    print("\nInference test on 5 samples:")
    for i in range(5):
        print(f"  Sample {i+1}: Predicted={int(predicted_classes[i].item())}, "
              f"Probability={predictions[i].item():.4f}, "
              f"Actual={int(y_test[i].item())}")

# =============================================================================
# 체크포인트 관리
# =============================================================================

print("\n" + "="*80)
print("CHECKPOINT MANAGEMENT")
print("="*80)

# 모든 체크포인트를 나열한다
checkpoints = sorted(checkpoint_dir.glob("*.pt"))
print(f"Found {len(checkpoints)} checkpoint files:")
for cp in checkpoints:
    size_kb = cp.stat().st_size / 1024
    print(f"  {cp.name:30s} ({size_kb:.1f} KB)")

# 오래된 체크포인트를 정리한다 (최근 3개만 남긴다)
def cleanup_old_checkpoints(checkpoint_dir, keep_last=3):
    """가장 최근의 체크포인트 N개만 남긴다"""
    checkpoints = sorted(
        [f for f in checkpoint_dir.glob("checkpoint_epoch_*.pt")],
        key=lambda x: int(x.stem.split('_')[-1])
    )
    
    if len(checkpoints) > keep_last:
        to_delete = checkpoints[:-keep_last]
        print(f"\nCleaning up {len(to_delete)} old checkpoints...")
        for cp in to_delete:
            cp.unlink()
            print(f"  Deleted: {cp.name}")
    
    print(f"✓ Kept {min(keep_last, len(checkpoints))} most recent checkpoints")

# 실제로 정리하려면 주석을 푼다:
# cleanup_old_checkpoints(checkpoint_dir, keep_last=3)

# =============================================================================
# 핵심 요점
# =============================================================================

print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print("""
1. TYPES OF SAVING
   ✓ Full checkpoint: model + optimizer + training state
   ✓ Best model: only model weights (for deployment)
   ✓ Periodic: save every N epochs

2. CHECKPOINT CONTENTS
   ✓ model_state_dict: Model weights
   ✓ optimizer_state_dict: Optimizer state (momentum, etc.)
   ✓ epoch: Current epoch number
   ✓ loss/accuracy: Performance metrics
   ✓ Additional: learning rate, random state, etc.

3. BEST PRACTICES
   ✓ Save checkpoints regularly
   ✓ Save best model separately
   ✓ Include training state for resuming
   ✓ Clean up old checkpoints
   ✓ Use meaningful filenames
   ✓ Save metadata (config, date, etc.)

4. WHEN TO CHECKPOINT
   ✓ Every N epochs (e.g., every 10)
   ✓ When validation improves
   ✓ Before long training runs
   ✓ Before hyperparameter changes

5. FILE ORGANIZATION
   checkpoints/
   ├── best_model.pt          # Best performing model
   ├── checkpoint_epoch_10.pt # Periodic checkpoints
   ├── checkpoint_epoch_20.pt
   └── last_checkpoint.pt     # Most recent state
""")

print("\n" + "="*80)
print("EXERCISES")
print("="*80)
print("""
1. EASY: Save training history (losses, accuracies) to JSON

2. MEDIUM: Implement early stopping with checkpoint loading:
   - Save when validation improves
   - If no improvement for N epochs, stop and load best

3. MEDIUM: Add metadata to checkpoints:
   - Timestamp
   - Hyperparameters
   - Model architecture details

4. HARD: Implement checkpoint versioning:
   - Keep different versions of model
   - Compare performance across versions
   - Rollback to previous version if needed

5. HARD: Create deployment package:
   - Save model + preprocessing (scaler)
   - Add inference function
   - Create simple API
""")


if __name__ == "__main__":
    pass
```

## 논의

`LogisticRegression` 클래스는 PyTorch의 `nn.Module` 인터페이스를 사용하여 모델 구조를 감싼다. `forward` 메서드가 계산 그래프를 정의하므로, 학습 중에 PyTorch의 autograd 체계가 경사 계산을 자동으로 처리한다. 이런 모듈식 설계 덕분에 개별 구성 요소를 고치거나 모델을 더 큰 파이프라인에 넣기가 쉬워진다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

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
