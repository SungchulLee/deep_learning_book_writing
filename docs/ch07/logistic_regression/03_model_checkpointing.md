# 모델 체크포인트

모델의 추론 상태를 저장하는 경우와 중단한 학습을 이어가는 경우를 비교한다. 웨이트 외에 함께 보존해야 할 상태를 확인한다.

## 1. 코드

```python
"""
================================================================================
03_model_checkpointing.py - 모델 저장하고 불러오기
================================================================================

학습 목표:
- 모델 웨이트를 저장하고 불러온다
- 되짚음 저장 구조를 짠다
- 되짚음 저장에서 학습을 이어 간다
- 온전한 학습 상태를 저장한다
- 모델을 남기는 좋은 버릇

소요 시간: 1시간쯤
어려움: ⭐⭐⭐☆☆ (가운데)
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
checkpoint_dir = Path("checkpoints")
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
    온전한 학습 상태를 저장한다
    
    Args:
        model: 저장할 모델
        optimizer: 저장할 최적화기
        epoch: 이제 판 번호
        loss: 이제 로스 값
        accuracy: 이제 정확도
        filepath: 되짚음 저장을 둘 자리
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
    되짚음 저장을 불러와 상태를 되살린다
    
    Args:
        filepath: 되짚음 저장 두루마리의 길
        model: 웨이트를 불러 넣을 모델
        optimizer: 상태를 되살릴 최적화기(골라 쓴다)
        
    Returns:
        학습 정보를 담은 사전
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
    """모델 웨이트만 저장한다 (배포용)"""
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

# 웨이트만 불러온다 (최적화기 상태는 제외)
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
1. 저장의 클래스
   ✓ 온 되짚음 저장: 모델 + 최적화기 + 학습 상태
   ✓ 가장 좋은 모델: 모델 웨이트만(내놓기용)
   ✓ 때맞춰: N 에폭마다 저장

2. 되짚음 저장에 담기는 것
   ✓ model_state_dict: 모델 웨이트
   ✓ optimizer_state_dict: 최적화기 상태(여세 따위)
   ✓ epoch: 이제 판 번호
   ✓ loss/accuracy: 성능 자
   ✓ 덧붙임: 러닝레이트, 마구잡이 상태 따위

3. 좋은 버릇
   ✓ 되짚음 저장을 자주 하라
   ✓ 가장 좋은 모델은 따로 저장하라
   ✓ 이어 가려면 학습 상태도 넣어라
   ✓ 묵은 되짚음 저장은 치워라
   ✓ 뜻이 담긴 두루마리 이름을 써라
   ✓ 곁들인 정보(설정, 날짜 따위)도 저장하라

4. 언제 되짚음 저장을 할까
   ✓ N 에폭마다(보기: 10 에폭마다)
   ✓ 검증이 나아질 때
   ✓ 오래 익히기 앞에
   ✓ 초파라메터를 바꾸기 앞에

5. 두루마리 저장 짜임
   checkpoints/
   ├── best_model.pt          # 가장 성능이 좋은 모델
   ├── checkpoint_epoch_10.pt # 때맞춘 되짚음 저장
   ├── checkpoint_epoch_20.pt
   └── last_checkpoint.pt     # 가장 마지막 상태
""")

print("\n" + "="*80)
print("EXERCISES")
print("="*80)
print("""
1. 쉬움: 학습 자취(로스, 정확도)를 JSON으로 저장하라

2. 보통: 되짚음 저장 불러오기를 곁들인 조기 종료를 짜라.
   - 검증이 나아지면 저장한다
   - N 판 동안 나아지지 않으면 멈추고 가장 좋은 것을 불러온다

3. 보통: 되짚음 저장에 곁들인 정보를 더하여라.
   - Timestamp
   - Hyperparameters
   - 모델 구조의 속내

4. 어려움: 되짚음 저장에 판 번호를 매겨라.
   - 모델의 여러 판을 남긴다
   - 에폭마다 성능을 견준다
   - 필요하면 앞선 판으로 되돌린다

5. 어려움: 내놓기 꾸러미를 만들어라.
   - 모델과 미리 다듬기(스케일러)를 저장한다
   - 추론 함수를 더한다
   - 단순한 낯을 만든다
""")


if __name__ == "__main__":
    pass
```

??? note "전체 출력 (178줄)"

    ```
    ================================================================================
    MODEL CHECKPOINTING AND PERSISTENCE
    ================================================================================

    ================================================================================
    TRAINING WITH AUTOMATIC CHECKPOINTING
    ================================================================================
    Training for 50 epochs
    Saving checkpoints every 10 epochs
    ------------------------------------------------------------
    ✓ Best model saved to checkpoints/best_model.pt
      → New best accuracy: 0.3525
    ✓ Best model saved to checkpoints/best_model.pt
      → New best accuracy: 0.3675
    ✓ Best model saved to checkpoints/best_model.pt
      → New best accuracy: 0.4175
    ✓ Best model saved to checkpoints/best_model.pt
      → New best accuracy: 0.4775
    ✓ Best model saved to checkpoints/best_model.pt
      → New best accuracy: 0.5150
    ✓ Best model saved to checkpoints/best_model.pt
      → New best accuracy: 0.5575
    ✓ Best model saved to checkpoints/best_model.pt
      → New best accuracy: 0.6250
    ✓ Best model saved to checkpoints/best_model.pt
      → New best accuracy: 0.6600
    ✓ Best model saved to checkpoints/best_model.pt
      → New best accuracy: 0.7025
    Epoch [10/50] Loss: 0.5961 Train Acc: 0.7431 Test Acc: 0.7300
    ✓ Checkpoint saved to checkpoints/checkpoint_epoch_10.pt
    ✓ Best model saved to checkpoints/best_model.pt
      → New best accuracy: 0.7300
    ✓ Best model saved to checkpoints/best_model.pt
      → New best accuracy: 0.7725
    ✓ Best model saved to checkpoints/best_model.pt
      → New best accuracy: 0.8000
    ✓ Best model saved to checkpoints/best_model.pt
      → New best accuracy: 0.8250
    ✓ Best model saved to checkpoints/best_model.pt
      → New best accuracy: 0.8350
    ✓ Best model saved to checkpoints/best_model.pt
      → New best accuracy: 0.8500
    ✓ Best model saved to checkpoints/best_model.pt
      → New best accuracy: 0.8575
    ✓ Best model saved to checkpoints/best_model.pt
      → New best accuracy: 0.8650
    Epoch [20/50] Loss: 0.4639 Train Acc: 0.8762 Test Acc: 0.8700
    ✓ Checkpoint saved to checkpoints/checkpoint_epoch_20.pt
    ✓ Best model saved to checkpoints/best_model.pt
      → New best accuracy: 0.8700
    ✓ Best model saved to checkpoints/best_model.pt
      → New best accuracy: 0.8725
    ✓ Best model saved to checkpoints/best_model.pt
      → New best accuracy: 0.8750
    ✓ Best model saved to checkpoints/best_model.pt
      → New best accuracy: 0.8800
    ✓ Best model saved to checkpoints/best_model.pt
      → New best accuracy: 0.8875
    ✓ Best model saved to checkpoints/best_model.pt
      → New best accuracy: 0.8900
    Epoch [30/50] Loss: 0.3945 Train Acc: 0.8869 Test Acc: 0.8875
    ✓ Checkpoint saved to checkpoints/checkpoint_epoch_30.pt
    ✓ Best model saved to checkpoints/best_model.pt
      → New best accuracy: 0.8925
    ✓ Best model saved to checkpoints/best_model.pt
      → New best accuracy: 0.8950
    Epoch [40/50] Loss: 0.3541 Train Acc: 0.8894 Test Acc: 0.8925
    ✓ Checkpoint saved to checkpoints/checkpoint_epoch_40.pt
    ✓ Best model saved to checkpoints/best_model.pt
      → New best accuracy: 0.8975
    ✓ Best model saved to checkpoints/best_model.pt
      → New best accuracy: 0.9000
    Epoch [50/50] Loss: 0.3289 Train Acc: 0.8888 Test Acc: 0.9000
    ✓ Checkpoint saved to checkpoints/checkpoint_epoch_50.pt

    Training completed!
    Best test accuracy: 0.9000

    ================================================================================
    DEMONSTRATING RESUME FROM CHECKPOINT
    ================================================================================
    ✓ Checkpoint loaded from checkpoints/checkpoint_epoch_20.pt
      Epoch: 19
      Loss: 0.4639
      Accuracy: 0.8700

    Resuming training from epoch 20...
    Epoch [20] Loss: 0.4549 Accuracy: 0.8775
    Epoch [25] Loss: 0.4175 Accuracy: 0.8844

    ✓ Successfully resumed and continued training!

    ================================================================================
    LOADING MODEL FOR INFERENCE (DEPLOYMENT)
    ================================================================================
    ✓ Model loaded for inference

    Inference test on 5 samples:
      Sample 1: Predicted=0, Probability=0.4676, Actual=1
      Sample 2: Predicted=1, Probability=0.8888, Actual=1
      Sample 3: Predicted=1, Probability=0.6175, Actual=1
      Sample 4: Predicted=1, Probability=0.6468, Actual=0
      Sample 5: Predicted=0, Probability=0.1400, Actual=0

    ================================================================================
    CHECKPOINT MANAGEMENT
    ================================================================================
    Found 6 checkpoint files:
      best_model.pt                  (1.6 KB)
      checkpoint_epoch_10.pt         (3.6 KB)
      checkpoint_epoch_20.pt         (3.6 KB)
      checkpoint_epoch_30.pt         (3.6 KB)
      checkpoint_epoch_40.pt         (3.6 KB)
      checkpoint_epoch_50.pt         (3.6 KB)

    ================================================================================
    KEY TAKEAWAYS
    ================================================================================

    1. 저장의 클래스
       ✓ 온 되짚음 저장: 모델 + 최적화기 + 학습 상태
       ✓ 가장 좋은 모델: 모델 웨이트만(내놓기용)
       ✓ 때맞춰: N 에폭마다 저장

    2. 되짚음 저장에 담기는 것
       ✓ model_state_dict: 모델 웨이트
       ✓ optimizer_state_dict: 최적화기 상태(여세 따위)
       ✓ epoch: 이제 판 번호
       ✓ loss/accuracy: 성능 자
       ✓ 덧붙임: 러닝레이트, 마구잡이 상태 따위

    3. 좋은 버릇
       ✓ 되짚음 저장을 자주 하라
       ✓ 가장 좋은 모델은 따로 저장하라
       ✓ 이어 가려면 학습 상태도 넣어라
       ✓ 묵은 되짚음 저장은 치워라
       ✓ 뜻이 담긴 두루마리 이름을 써라
       ✓ 곁들인 정보(설정, 날짜 따위)도 저장하라

    4. 언제 되짚음 저장을 할까
       ✓ N 에폭마다(보기: 10 에폭마다)
       ✓ 검증이 나아질 때
       ✓ 오래 익히기 앞에
       ✓ 초파라메터를 바꾸기 앞에

    5. 두루마리 저장 짜임
       checkpoints/
       ├── best_model.pt          # 가장 성능이 좋은 모델
       ├── checkpoint_epoch_10.pt # 때맞춘 되짚음 저장
       ├── checkpoint_epoch_20.pt
       └── last_checkpoint.pt     # 가장 마지막 상태


    ================================================================================
    EXERCISES
    ================================================================================

    1. 쉬움: 학습 자취(로스, 정확도)를 JSON으로 저장하라

    2. 보통: 되짚음 저장 불러오기를 곁들인 조기 종료를 짜라.
       - 검증이 나아지면 저장한다
       - N 판 동안 나아지지 않으면 멈추고 가장 좋은 것을 불러온다

    3. 보통: 되짚음 저장에 곁들인 정보를 더하여라.
       - Timestamp
       - Hyperparameters
       - 모델 구조의 속내

    4. 어려움: 되짚음 저장에 판 번호를 매겨라.
       - 모델의 여러 판을 남긴다
       - 에폭마다 성능을 견준다
       - 필요하면 앞선 판으로 되돌린다

    5. 어려움: 내놓기 꾸러미를 만들어라.
       - 모델과 미리 다듬기(스케일러)를 저장한다
       - 추론 함수를 더한다
       - 단순한 낯을 만든다

    ```


## 2. 논의

### 상태 사전과 모델 구조

`state_dict()`는 등록된 파라미터와 영속 버퍼를 담는다. 복원할 때는 호환되는 모델 구조를 만들고 상태를 불러온다. 전처리 통계와 피처 순서, 클래스 인코딩도 같아야 같은 예측을 얻을 수 있다.

### 학습 재개와 최적 체크포인트

옵티마이저의 이동 평균과 단계 수는 파라미터 값과 별개다. 학습 재개에는 이 상태와 에포크, 필요한 스케줄러·정밀도 스케일러·난수 상태도 고려한다. 검증 지표가 가장 좋았던 웨이트를 별도로 저장하는 목적과 마지막 실행을 이어가는 목적을 구분한다.

### 저장과 평가의 경계

복원 후 추론에서는 모델의 평가 모드와 경사 기록을 각각 관리한다. 체크포인트가 복원된다는 사실만으로 전처리나 데이터 분할의 일치가 확인되는 것은 아니다. 외부 체크포인트는 신뢰할 수 있는 출처의 파일을 사용하고 라이브러리 버전의 로드 규칙을 확인한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
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
웨이트를 복원했는데 예측이 다르거나 학습 재개가 달라질 수 있는 이유를 설명하라.

</div>

??? success "연습문제 3 풀이"
    전처리와 피처 순서, 모델 모드가 다르면 같은 웨이트에서도 예측이 달라진다. 옵티마이저·스케줄러·난수 상태가 빠지면 다음 학습 경로도 달라질 수 있다. 복원의 목적에 맞는 상태를 저장한다.

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
