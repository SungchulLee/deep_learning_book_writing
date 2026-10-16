# 실전 예제 — 합성 데이터 분류 파이프라인

이 예시는 CNN·DataLoader·학습·검증을 하나의 흐름으로 묶는다. 이름과 달리 실제 MNIST를 읽지 않고 `SyntheticMNIST`가 만든 임의 영상과 라벨을 사용하므로, 숫자 인식 성능을 보여 주는 실험이 아니라 파이프라인의 동작 예시다.

## 1. 코드

```python
"""튜토리얼 28: 참 세상 보기 - 온전한 MNIST 숫자 분류기"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

# ========================================================================
# 메인
# ========================================================================

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

# MNIST를 흉내 낸 합성 데이터셋(실제로는 진짜 MNIST로 바꾼다)
class SyntheticMNIST(Dataset):
    def __init__(self, size=1000):
        self.data = torch.randn(size, 1, 28, 28)
        self.labels = torch.randint(0, 10, (size,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class ConvNet(nn.Module):
    """숫자 분류를 위한 컨볼루션 뉴럴 넷트웍."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_epoch(model, dataloader, criterion, optimizer, device):
    """한 에폭 익힌다."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, labels in dataloader:
        data, labels = data.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    """모델을 다진다."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def main():
    header("Complete MNIST Classifier Pipeline")
    
    # 1. 준비
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. 데이터 준비
    print("\n1. Preparing Data...")
    dataset = SyntheticMNIST(size=1000)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # 3. 모델 생성
    print("\n2. Creating Model...")
    model = ConvNet().to(device)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # 4. 로스와 최적화기
    print("\n3. Setting up Training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # 5. 학습 루프
    print("\n4. Training...")
    num_epochs = 5
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # 최고 성능 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  → New best model! Acc: {best_val_acc:.2f}%")
    
    # 6. 최종 평가
    print("\n5. Final Evaluation...")
    final_loss, final_acc = validate(model, val_loader, criterion, device)
    print(f"Final Validation Accuracy: {final_acc:.2f}%")
    
    # 7. 추론 예제
    print("\n6. Inference Example...")
    model.eval()
    with torch.no_grad():
        sample_data, sample_label = dataset[0]
        sample_data = sample_data.unsqueeze(0).to(device)
        output = model(sample_data)
        _, predicted = torch.max(output, 1)
        print(f"True label: {sample_label.item()}")
        print(f"Predicted: {predicted.item()}")
        print(f"Confidence: {torch.softmax(output, 1).max().item():.4f}")
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print("""
    다음 스텝:
    1. 참 MNIST 데이터셋으로 해 보아라(torchvision.datasets.MNIST)
    2. 여러 구조를 이리저리 바꾸어 보아라
    3. 데이터 불리기를 더하여라
    4. 여러 최적화기와 러닝레이트를 써 보아라
    5. 조기 종료를 짜라
    6. 텐서보드로 적바림을 더하여라
    7. 모델을 참으로 굴릴 수 있게 내놓아라
    """)

if __name__ == "__main__":
    main()
```

**출력:**

```

======================================================================
Complete MNIST Classifier Pipeline
======================================================================
Using device: cpu

1. Preparing Data...
Train samples: 800
Val samples: 200

2. Creating Model...
ConvNet(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
  (fc1): Linear(in_features=1600, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=10, bias=True)
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (dropout): Dropout(p=0.5, inplace=False)
)

Total parameters: 225,034

3. Setting up Training...

4. Training...
Epoch [1/5]
  Train Loss: 2.3190, Train Acc: 9.12%
  Val Loss: 2.3063, Val Acc: 8.50%
  → New best model! Acc: 8.50%
Epoch [2/5]
  Train Loss: 2.3013, Train Acc: 13.25%
  Val Loss: 2.3075, Val Acc: 8.50%
Epoch [3/5]
  Train Loss: 2.3014, Train Acc: 9.50%
  Val Loss: 2.3041, Val Acc: 11.00%
  → New best model! Acc: 11.00%
Epoch [4/5]
  Train Loss: 2.2934, Train Acc: 12.50%
  Val Loss: 2.3084, Val Acc: 9.00%
Epoch [5/5]
  Train Loss: 2.2959, Train Acc: 12.75%
  Val Loss: 2.3078, Val Acc: 8.00%

5. Final Evaluation...
Final Validation Accuracy: 8.00%

6. Inference Example...
True label: 4
Predicted: 0
Confidence: 0.1115

======================================================================
Training Complete!
======================================================================

    다음 스텝:
    1. 참 MNIST 데이터셋으로 해 보아라(torchvision.datasets.MNIST)
    2. 여러 구조를 이리저리 바꾸어 보아라
    3. 데이터 불리기를 더하여라
    4. 여러 최적화기와 러닝레이트를 써 보아라
    5. 조기 종료를 짜라
    6. 텐서보드로 적바림을 더하여라
    7. 모델을 참으로 굴릴 수 있게 내놓아라
    
```

## 2. 논의

입력과 라벨을 같은 장치로 옮기고, 순전파·로스·역전파·옵티마이저 갱신을 반복한다. 검증에서는 `eval()`로 레이어의 동작을 바꾸고 `no_grad()`로 역전파 기록을 끈다. 두 설정은 서로 다른 역할을 한다.

임의 영상과 독립적인 임의 라벨 사이에는 학습할 일반화 규칙이 없다. 열 클래스가 균등하다면 새 표본의 기대 정확도는 약 10%다. 학습 로스가 내려가도 실제 손글씨를 인식한다고 해석해서는 안 된다.

스케줄러의 호출 시점, 배치별 로스를 데이터 전체로 평균내는 방식, 학습·검증 데이터의 분리를 함께 확인하면 실제 데이터로 바꿀 때 필요한 구성 요소를 점검할 수 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
SGD 대신 Adam 옵티마이저를 쓰도록 코드를 수정하라. 100 에포크에 걸친 수렴 속도를 비교하라.

</div>

??? success "연습문제 1 풀이"
    ```python
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Adam은 적응적 러닝레이트와 모멘텀 덕분에 보통 SGD보다
    # 빠르게 수렴한다. 다만 Adam의 최적 러닝레이트는
    # 보통 SGD보다 작다.
    ```

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
학습 루프에서 `optimizer.zero_grad()`를 없애면 어떤 일이 생기는가? 실험해 보고 학습 로스에 미치는 영향을 설명하라.

</div>

??? success "연습문제 2 풀이"
    업데이트 사이에 경사를 초기화하지 않으면 서로 다른 파라미터에서 계산한 이전 경사까지 누적된다. 따라서 현재 로스의 경사만 사용하는 학습 규칙과 달라지고, 진동이나 발산을 일으킬 수 있다. 경사끼리 상쇄될 수도 있으므로 크기가 항상 증가하거나 반드시 발산하는 것은 아니다. 의도적인 경사 누적에서는 정해진 여러 마이크로배치를 처리한 뒤 한 번 업데이트하고 경사를 초기화한다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
옵티마이저에 L2 정규화(웨이트 감쇠)를 추가하고 그것이 최종 파라메터 값에 어떤 영향을 주는지 관찰하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
    # weight_decay는 로스에 L2 벌점항 lambda * ||w||^2을 더한다.
    # 이는 웨이트를 작게 유도하여 과적합을 막을 수 있다.
    # 최종 웨이트의 크기가 조금 더 작아진다.
    ```

## 정리하며

이 예시는 CNN·DataLoader·학습·검증을 하나의 흐름으로 묶는다. 이름과 달리 실제 MNIST를 읽지 않고 `SyntheticMNIST`가 만든 임의 영상과 라벨을 사용하므로, 숫자 인식 성능을 보여 주는 실험이 아니라 파이프라인의 동작 예시다.
