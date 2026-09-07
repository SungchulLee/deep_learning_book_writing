# 완전한 MNIST 학습

완전한 학습 파이프라인은 데이터 적재, 모델 정의, 손실과 최적화기 설정, 검증을 포함한 학습 루프, 모델 검사점 저장, 추론을 아우른다. 이 스크립트는 MNIST 숫자 분류에서 이 모든 요소를 보이며, 재현성, 장치 처리, 지표 기록 같은 실전의 좋은 관행을 따른다.

## 코드

```python
"""
================================================================================
실전 예제: 완전한 MNIST 숫자 분류
================================================================================

배울 내용:
- 처음부터 끝까지의 완전한 학습 파이프라인
- 데이터 적재와 전처리
- 모델 정의
- 손실과 최적화기를 쓰는 학습
- 검증과 시험
- 모델의 저장과 불러오기
- 실전 코드의 좋은 관행

선수 지식:
- 입문자용 튜토리얼을 모두 마친다
- CNN에 대한 기본 이해

소요 시간: 약 30분
================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

print("=" * 80)
print("COMPLETE MNIST DIGIT CLASSIFICATION")
print("=" * 80)

# ============================================================================
# 1절: 설정과 준비
# ============================================================================
print("\n" + "-" * 80)
print("CONFIGURATION")
print("-" * 80)

# 재현성을 위해 난수 씨앗 고정
torch.manual_seed(42)

# 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 초매개변수
config = {
    'batch_size': 64,
    'test_batch_size': 1000,
    'epochs': 5,
    'learning_rate': 0.01,
    'momentum': 0.9,
    'log_interval': 100,  # N 배치마다 출력
    'save_model': True,
    'model_path': '/home/claude/mnist_model.pt'
}

print("\nHyperparameters:")
for key, value in config.items():
    print(f"  {key}: {value}")

# ============================================================================
# 2절: 데이터 적재와 전처리
# ============================================================================
print("\n" + "-" * 80)
print("DATA LOADING")
print("-" * 80)

# 데이터 변환 정의
transform = transforms.Compose([
    transforms.ToTensor(),  # PIL 이미지를 텐서로 바꾸기
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST의 평균과 표준편차로 정규화
])

print("Downloading MNIST dataset...")

# 학습 데이터 내려받아 불러오기
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# 시험 데이터 내려받아 불러오기
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# 데이터 로더 만들기
train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,  # 학습 데이터 섞기
    num_workers=0  # 데이터 적재에 쓸 프로세스의 수
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config['test_batch_size'],
    shuffle=False,  # 시험 데이터는 섞지 않는다
    num_workers=0
)

print(f"\nDataset Statistics:")
print(f"  Training samples: {len(train_dataset)}")
print(f"  Test samples: {len(test_dataset)}")
print(f"  Number of classes: 10 (digits 0-9)")
print(f"  Image size: 28x28 pixels")
print(f"  Training batches: {len(train_loader)}")
print(f"  Test batches: {len(test_loader)}")

# ============================================================================
# 3절: 모델 정의
# ============================================================================
print("\n" + "-" * 80)
print("MODEL ARCHITECTURE")
print("-" * 80)

class ConvNet(nn.Module):
    """
    MNIST를 위한 합성곱 신경망
    
    구조:
    - 합성곱 층 1: 채널 1 → 32, 3x3 핵
    - 합성곱 층 2: 채널 32 → 64, 3x3 핵
    - 최댓값 풀링: 2x2
    - 완전 연결 1: 9216 → 128
    - 완전 연결 2: 128 → 10 (클래스)
    """
    def __init__(self):
        super(ConvNet, self).__init__()
        # 합성곱 층
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # 풀링 층
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 정칙화를 위한 드롭아웃
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # 완전 연결층
        # conv1, conv2와 풀링 2번을 거치면: 28 → 14 → 7
        # 특징 맵의 크기: 7 * 7 * 64 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # 합성곱 블록 1
        x = F.relu(self.conv1(x))  # 28x28x32
        x = self.pool(x)            # 14x14x32
        
        # 합성곱 블록 2
        x = F.relu(self.conv2(x))  # 14x14x64
        x = self.pool(x)            # 7x7x64
        x = self.dropout1(x)
        
        # 펼치기
        x = x.view(-1, 64 * 7 * 7)  # 벡터로 펼치기
        
        # 완전 연결층
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)  # 활성화 없음 (CrossEntropyLoss가 소프트맥스를 적용한다)
        
        return x

# 모델을 만들어 장치로 옮기기
model = ConvNet().to(device)

# 매개변수 세기
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Model Architecture:")
print(model)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ============================================================================
# 4절: 손실 함수와 최적화기
# ============================================================================
print("\n" + "-" * 80)
print("LOSS FUNCTION AND OPTIMIZER")
print("-" * 80)

# 손실 함수
criterion = nn.CrossEntropyLoss()
print(f"Loss function: {criterion}")

# 최적화기
optimizer = optim.SGD(
    model.parameters(),
    lr=config['learning_rate'],
    momentum=config['momentum']
)
print(f"Optimizer: {optimizer}")

# 학습률 스케줄러
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=1,  # 에포크마다 감쇠
    gamma=0.7     # 학습률에 0.7을 곱한다
)
print(f"LR Scheduler: StepLR(step_size=1, gamma=0.7)")

# ============================================================================
# 5절: 학습 함수
# ============================================================================

def train(model, device, train_loader, optimizer, criterion, epoch):
    """
    모델을 한 에포크 동안 학습시킨다
    
    인수:
        model: 신경망 모델
        device: 학습에 쓸 장치 (CPU/GPU)
        train_loader: 학습 데이터의 DataLoader
        optimizer: 매개변수 갱신에 쓸 최적화기
        criterion: 손실 함수
        epoch: 현재 에포크 번호
    """
    model.train()  # 모델을 학습 모드로
    
    total_loss = 0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # 데이터를 장치로 옮기기
        data, target = data.to(device), target.to(device)
        
        # 기울기 초기화
        optimizer.zero_grad()
        
        # 순전파
        output = model(data)
        
        # 손실 계산
        loss = criterion(output, target)
        
        # 역전파
        loss.backward()
        
        # 매개변수 갱신
        optimizer.step()
        
        # 통계
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # 진행 상황 출력
        if batch_idx % config['log_interval'] == 0:
            print(f'  Batch [{batch_idx}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f} '
                  f'Acc: {100. * correct / total:.2f}%')
    
    # 에포크 통계
    epoch_time = time.time() - start_time
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    print(f'\n  Epoch {epoch} Summary:')
    print(f'    Avg Loss: {avg_loss:.4f}')
    print(f'    Accuracy: {accuracy:.2f}%')
    print(f'    Time: {epoch_time:.2f}s')
    
    return avg_loss, accuracy

# ============================================================================
# 6절: 검증/시험 함수
# ============================================================================

def test(model, device, test_loader, criterion):
    """
    시험 데이터에서 모델을 평가한다
    
    인수:
        model: 신경망 모델
        device: 시험에 쓸 장치 (CPU/GPU)
        test_loader: 시험 데이터의 DataLoader
        criterion: 손실 함수
    
    반환값:
        평균 손실과 정확도
    """
    model.eval()  # 모델을 평가 모드로
    
    test_loss = 0
    correct = 0
    total = 0
    
    # 시험 중에는 기울기를 계산하지 않는다
    with torch.no_grad():
        for data, target in test_loader:
            # 데이터를 장치로 옮기기
            data, target = data.to(device), target.to(device)
            
            # 순전파
            output = model(data)
            
            # 손실 계산
            test_loss += criterion(output, target).item()
            
            # 예측을 얻는다
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    # 통계 계산
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    print(f'\n  Test Results:')
    print(f'    Avg Loss: {avg_loss:.4f}')
    print(f'    Accuracy: {accuracy:.2f}% ({correct}/{total})')
    
    return avg_loss, accuracy

# ============================================================================
# 7절: 학습 루프
# ============================================================================
print("\n" + "-" * 80)
print("TRAINING")
print("-" * 80)

# 이력 기록
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

print(f"\nTraining for {config['epochs']} epochs...\n")

for epoch in range(1, config['epochs'] + 1):
    print(f"{'=' * 80}")
    print(f"Epoch {epoch}/{config['epochs']}")
    print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
    print(f"{'=' * 80}")
    
    # 학습
    train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, epoch)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    
    # 시험
    test_loss, test_acc = test(model, device, test_loader, criterion)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    
    # 학습률을 갱신한다
    scheduler.step()
    
    print()

# ============================================================================
# 8절: 모델 저장
# ============================================================================
print("\n" + "-" * 80)
print("SAVING MODEL")
print("-" * 80)

if config['save_model']:
    # 완전한 모델 저장
    torch.save({
        'epoch': config['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'test_loss': test_losses[-1],
        'test_accuracy': test_accuracies[-1]
    }, config['model_path'])
    
    print(f"Model saved to: {config['model_path']}")
    
    # 가중치만 따로 저장하기도 한다 (파일이 더 작다)
    weights_path = config['model_path'].replace('.pt', '_weights.pt')
    torch.save(model.state_dict(), weights_path)
    print(f"Model weights saved to: {weights_path}")

# ============================================================================
# 9절: 모델 불러와 시험하기
# ============================================================================
print("\n" + "-" * 80)
print("LOADING MODEL")
print("-" * 80)

# 새 모델 인스턴스 만들기
loaded_model = ConvNet().to(device)

# 체크포인트를 불러온다
checkpoint = torch.load(config['model_path'])
loaded_model.load_state_dict(checkpoint['model_state_dict'])

print("Model loaded successfully!")
print(f"  Trained for: {checkpoint['epoch']} epochs")
print(f"  Final test accuracy: {checkpoint['test_accuracy']:.2f}%")

# 불러온 모델 시험
print("\nVerifying loaded model:")
test_loss, test_acc = test(loaded_model, device, test_loader, criterion)

# ============================================================================
# 10절: 추론 예제
# ============================================================================
print("\n" + "-" * 80)
print("INFERENCE EXAMPLE")
print("-" * 80)

# 시험 이미지 배치 하나 가져오기
data_iter = iter(test_loader)
images, labels = next(data_iter)

# 처음 이미지 5장 가져오기
images = images[:5].to(device)
labels = labels[:5]

# 예측한다
model.eval()
with torch.no_grad():
    outputs = model(images)
    probabilities = F.softmax(outputs, dim=1)
    _, predictions = torch.max(outputs, 1)

print("Sample Predictions:")
print(f"{'True':^6} {'Pred':^6} {'Confidence':^12}")
print("-" * 26)

for i in range(5):
    true_label = labels[i].item()
    pred_label = predictions[i].item()
    confidence = probabilities[i, pred_label].item()
    status = "✓" if true_label == pred_label else "✗"
    
    print(f"  {true_label}      {pred_label}      {confidence*100:5.1f}%    {status}")

# ============================================================================
# 11절: 학습 요약
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING SUMMARY")
print("=" * 80)

print(f"\nFinal Results:")
print(f"  Training Accuracy: {train_accuracies[-1]:.2f}%")
print(f"  Test Accuracy: {test_accuracies[-1]:.2f}%")
print(f"  Training Loss: {train_losses[-1]:.4f}")
print(f"  Test Loss: {test_losses[-1]:.4f}")

print(f"\nTraining Progress:")
for epoch in range(config['epochs']):
    print(f"  Epoch {epoch+1}: "
          f"Train Acc={train_accuracies[epoch]:.2f}%, "
          f"Test Acc={test_accuracies[epoch]:.2f}%, "
          f"Test Loss={test_losses[epoch]:.4f}")

# ============================================================================
# 요약
# ============================================================================
print("\n" + "=" * 80)
print("KEY TAKEAWAYS")
print("=" * 80)
print("""
1. 완전한 파이프라인:
   ✓ 자료 불러오기와 미리 다듬기
   ✓ 모형 매기기
   ✓ 손실 함수와 최적화기 설정
   ✓ 검증을 곁들인 학습 루프
   ✓ 모델 저장과 불러오기
   ✓ Inference

2. 보여 준 모범 사례:
   ✓ 효율적인 배치 처리를 위해 DataLoader를 쓴다
   ✓ Set model.train() / model.eval() appropriately
   ✓ Use torch.no_grad() for inference
   ✓ 학습 중 지표를 추적한다
   ✓ 메타데이터와 함께 체크포인트를 저장한다
   ✓ 학습률 스케줄링을 쓴다
   ✓ 정칙화를 위해 드롭아웃을 더한다

3. 실서비스에서 살필 점:
   ✓ 설정 다루기
   ✓ 되풀이할 수 있음(마구잡이 씨앗)
   ✓ Device handling (CPU/GPU)
   ✓ 진행 상황 기록
   ✓ Error handling (not shown but important)
   ✓ 모델 버전 관리

4. 최적화 선택:
   • SGD with momentum (reliable, well-tested)
   • CrossEntropyLoss (standard for classification)
   • StepLR scheduler (gradual learning rate decay)
   • Dropout (prevents overfitting)

다음 단계:
→ 여러 구조로 실험해 보라
→ Try other optimizers (Adam, AdamW)
→ 데이터 증강을 더해 보라
→ 조기 종료를 구현해 보라
→ 시각화에는 텐서보드를 쓰라
→ 자신의 데이터셋에 써 보라!
""")
print("=" * 80)


if __name__ == "__main__":
    pass
```

## 논의

학습 함수는 표준 루프를 구현한다. 모델을 학습 모드로 두고, 배치를 훑으며, 순전파와 손실을 계산하고, 기울기를 0으로 만들고, 역전파하고, 매개변수를 갱신하고, 지표를 누적한다. `model.train()`은 드롭아웃과 배치 정규화가 학습 모드로 동작하게 한다.

시험 함수는 세 가지가 다르다. `model.eval()`이 드롭아웃을 끄고 배치 정규화가 이동 통계를 쓰게 하며, `torch.no_grad()`가 기울기 계산을 꺼서 메모리와 시간을 아끼고, 매개변수를 갱신하지 않는다. `model.eval()`을 잊는 것은 평가 결과를 들쭉날쭉하게 만드는 흔한 실수이다.

검사점에는 모델의 가중치뿐 아니라 최적화기의 상태와 메타데이터(에포크, 손실, 정확도)도 담는다. 이렇게 갖춘 검사점이 있으면 학습을 완전히 이어 갈 수 있고 모델의 내력도 남는다. 가중치만 따로 저장한 파일은 더 작아서 추론 전용 배포에 알맞다.

## 연습문제

**연습문제 1.**
코드를 따라가며 쓰인 주요 자료 구조를 찾아라. 각각에 대해 자료형, (해당한다면) 모양, 파이프라인에서의 구실을 적어라.

??? success "연습문제 1 풀이"
    코드를 꼼꼼히 읽으며 변수 대입마다 살펴본다. 텐서는 `.shape`과 `.dtype`을 확인하고, 클래스는 `__init__`의 매개변수와 `forward`/`__call__`의 서명을 확인한다. 이름, 자료형, 모양, 구실을 열로 하는 표에 정리한다.

---


**연습문제 2.**
오류 처리와 입력 검증을 넣도록 코드를 고쳐라. 이 코드를 실전에 쓸 수 있게 하려면 어떤 검사를 더하겠는가?

??? success "연습문제 2 풀이"
    입력에 자료형 검사(`isinstance`), 모양 검증(`assert tensor.dim() == expected`), 값 범위 검사(예: 확률이 [0,1] 안인지)를 넣고, 입출력 연산은 try-except로 감싼다. 빈 배치나 NaN 같은 경계 상황에는 경고를 남긴다. 매개변수와 반환값의 자료형을 적은 독스트링을 붙인다.

---


**연습문제 3.**
직접 고른 새로운 쓰임새를 지원하도록 코드를 확장하라. 무엇을 왜 바꿀지 설명하라.

??? success "연습문제 3 풀이"
    알맞은 확장을 하나 고른다(예: 다른 데이터셋, 지표 추가, 새 모델 변형). 필요한 변경을 설명한다. 새 임포트, 클래스 정의 수정, 초매개변수 갱신, 새로운 시각화나 기록 등이다. 핵심 변경을 구현하고 간단한 시험으로 올바름을 확인한다.

