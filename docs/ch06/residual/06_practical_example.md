# 실전 예제

실전 예제: CIFAR-10에서 ResNet 학습시키기. 실제 데이터셋으로 ResNet 모델을 학습시키는 완전한 예제이다.

합성곱 구조는 요즘 컴퓨터 비전 시스템의 뼈대를 이룬다. 이 구현은 PyTorch로 잔차 신경망 설계의 핵심 개념을 보이며, 이미지 데이터에서 공간적인 특징의 위계가 어떻게 학습되는지 드러낸다.

## 코드

```python
"""
실전 예제: CIFAR-10에서 ResNet 학습시키기
==============================================
실제 데이터셋으로 ResNet 모델을 학습시키는 완전한 예제.
데이터 적재, 학습 반복문, 평가, 모범 관행을 담고 있다.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import time
import os

# ========================================================================
# 메인
# ========================================================================


# 우리 구현에서 ResNet 가져오기
# 참고: 실제로는 residual_connections 디렉터리에서 실행하거나
# 그에 맞게 가져오기 경로를 고쳐야 한다
try:
    from residual_connections_02_resnet_implementation import resnet18, resnet34, resnet50
except ImportError:
    # 대안: 여기서 최소한의 ResNet을 만들거나 torchvision에서 가져온다
    print("Note: Could not import from 02_resnet_implementation.py")
    print("Make sure to run from the correct directory or adjust imports")
    import sys
    sys.exit(1)


def get_cifar10_dataloaders(batch_size=128, num_workers=2):
    """
    알맞은 증강과 함께 CIFAR-10 데이터셋을 불러와 준비한다
    """
    print("Loading CIFAR-10 dataset...")
    
    # 학습용 데이터 증강
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 검증에는 증강 없음
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 학습 데이터 내려받아 불러오기
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    # 테스트 데이터 내려받아 불러오기
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    print(f"Training samples: {len(trainset)}")
    print(f"Test samples: {len(testset)}")
    print(f"Classes: {classes}")
    
    return trainloader, testloader, classes


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    한 세대 학습시킨다
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 순전파
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 역전파
        loss.backward()
        optimizer.step()
        
        # 통계
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 진행 상황 출력
        if batch_idx % 100 == 0:
            print(f'  Batch [{batch_idx}/{len(dataloader)}] '
                  f'Loss: {loss.item():.3f} | Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """
    시험 집합에서 모델을 평가한다
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = running_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def train_resnet_cifar10(
    model_name='resnet18',
    num_epochs=100,
    batch_size=128,
    learning_rate=0.1,
    weight_decay=5e-4,
    device=None
):
    """
    CIFAR-10에서 ResNet을 학습시키는 완전한 파이프라인
    
    인수:
        model_name: 'resnet18', 'resnet34', 또는 'resnet50'
        num_epochs: 학습 에포크 수
        batch_size: 학습에 쓸 배치 크기
        learning_rate: 처음 학습률
        weight_decay: L2 규제의 세기
        device: 학습에 쓸 장치 (None이면 저절로 찾는다)
    """
    print("=" * 80)
    print(f"Training {model_name.upper()} on CIFAR-10")
    print("=" * 80)
    
    # 장치 준비
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # 데이터를 불러온다
    trainloader, testloader, classes = get_cifar10_dataloaders(batch_size)
    
    # 모델 생성
    print(f"\nInitializing {model_name}...")
    if model_name == 'resnet18':
        model = resnet18(num_classes=10)
    elif model_name == 'resnet34':
        model = resnet34(num_classes=10)
    elif model_name == 'resnet50':
        model = resnet50(num_classes=10)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device)
    
    # 모델 정보 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 손실 함수와 최적화기
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                         momentum=0.9, weight_decay=weight_decay)
    
    # 학습률 스케줄러 (코사인 담금질)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 학습 기록
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': []
    }
    
    best_acc = 0
    start_time = time.time()
    
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # 학습
        train_loss, train_acc = train_epoch(
            model, trainloader, criterion, optimizer, device, epoch)
        
        # 평가한다
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        
        # 학습률을 갱신한다
        scheduler.step()
        
        # 이력 기록
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # 최고 성능 모델 저장
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
            }, f'{model_name}_cifar10_best.pth')
            print(f"  ✓ New best model saved! (Test Acc: {best_acc:.2f}%)")
        
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        print(f"  Time: {epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"Final test accuracy: {history['test_acc'][-1]:.2f}%")
    
    return model, history


def evaluate_per_class(model, dataloader, classes, device):
    """
    부류별 정확도를 평가한다
    """
    print("\n" + "=" * 80)
    print("Per-Class Accuracy Analysis")
    print("=" * 80)
    
    model.eval()
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            for i in range(len(targets)):
                label = targets[i].item()
                class_correct[label] += (predicted[i] == targets[i]).item()
                class_total[label] += 1
    
    print(f"\n{'Class':15} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 50)
    for i, class_name in enumerate(classes):
        acc = 100 * class_correct[i] / class_total[i]
        print(f"{class_name:15} {class_correct[i]:8d} {class_total[i]:8d} {acc:9.2f}%")
    
    overall_acc = 100 * sum(class_correct) / sum(class_total)
    print("-" * 50)
    print(f"{'Overall':15} {sum(class_correct):8d} {sum(class_total):8d} {overall_acc:9.2f}%")
    print("=" * 80)


def quick_demo(epochs=2):
    """
    시험 삼아 몇 세대만 돌리는 빠른 시연
    """
    print("\n" + "=" * 80)
    print("QUICK DEMO: Training ResNet-18 on CIFAR-10")
    print("=" * 80)
    print("\nNote: This is a quick demo with only 2 epochs.")
    print("For real training, use 100+ epochs to achieve ~93% accuracy.")
    
    model, history = train_resnet_cifar10(
        model_name='resnet18',
        num_epochs=epochs,
        batch_size=128,
        learning_rate=0.1
    )
    
    return model, history


if __name__ == "__main__":
    # 빠른 시연 모드
    print("\n" + "=" * 80)
    print("PRACTICAL EXAMPLE: ResNet on CIFAR-10")
    print("=" * 80)
    
    print("\nThis script demonstrates:")
    print("1. Loading and preprocessing CIFAR-10 dataset")
    print("2. Training ResNet with proper hyperparameters")
    print("3. Using learning rate scheduling")
    print("4. Evaluating model performance")
    print("5. Saving best model checkpoints")
    
    print("\n" + "=" * 80)
    print("Running quick demo (2 epochs)...")
    print("=" * 80)
    
    # 빠른 시연 실행
    model, history = quick_demo(epochs=2)
    
    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    
    print("\nTo train a full model (100 epochs, ~93% accuracy):")
    print("  python 06_practical_example.py --full-training")
    print("\nExpected results after full training:")
    print("  - ResNet-18: ~93-94% test accuracy")
    print("  - ResNet-34: ~94-95% test accuracy")
    print("  - ResNet-50: ~94-95% test accuracy")
    
    print("\n" + "=" * 80 + "\n")```

## 논의

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 딥러닝 구조 설계에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
학습 루프에서 `optimizer.zero_grad()` 호출을 없애면 어떤 일이 일어나는지 설명하라. 고친 코드를 실행하고 학습 손실의 수렴에 미치는 영향을 서술하라.

??? success "연습문제 1 풀이"
    `optimizer.zero_grad()`가 없으면 PyTorch가 새 경사를 기존 `.grad` 텐서에 덮어쓰지 않고 더하기 때문에 반복에 걸쳐 경사가 누적된다. 이는 사실상 학습률에 누적된 단계 수를 곱하는 셈이어서 최적화가 점점 크고 불규칙한 걸음을 내딛게 된다. 학습 손실은 매끄럽게 수렴하는 대신 심하게 진동하거나 발산한다. 해결책은 간단하다. `loss.backward()`를 호출하기 전에 언제나 경사를 0으로 만들어라.

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
조기 종료를 구현하라. 매 에폭 후 검증 손실을 추적하고, 10 에폭 연속으로 개선이 없으면 학습을 멈춘다. 가장 좋은 모델 가중치를 저장하고 복원하라.

??? success "연습문제 4 풀이"
    인내 횟수 카운터와 최저 손실 추적기를 추가한다.
    ```python
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    for epoch in range(num_epochs):
        # ... 학습 단계 ...
        val_loss = evaluate(model, val_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        if patience_counter >= 10:
            print(f'Early stopping at epoch {epoch}')
            model.load_state_dict(best_state)
            break
    ```
    이렇게 하면 따로 떼어 둔 데이터에서 모델이 더 나아지지 않을 때 멈추므로 과적합을 막을 수 있다.
