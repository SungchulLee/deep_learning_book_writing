# 학습 비교

학습 비교: ResNet과 평범한 신경망. 학습 실험으로 잔차 연결의 이점을 보인다.

합성곱 구조는 요즘 컴퓨터 비전 시스템의 뼈대를 이룬다. 이 구현은 PyTorch로 잔차 신경망 설계의 핵심 개념을 보이며, 이미지 데이터에서 공간적인 특징의 위계가 어떻게 학습되는지 드러낸다.

## 코드

```python
"""
학습 비교: ResNet과 평범한 신경망
============================================
학습 실험으로 잔차 연결의 이점을 보인다.
수렴 속도와 기울기의 흐름, 마지막 정확도를 견준다.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

# ========================================================================
# 메인
# ========================================================================


class PlainNet(nn.Module):
    """
    잔차 연결이 없는 깊은 신경망
    """
    def __init__(self, num_classes=10):
        super(PlainNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 합성곱 층 쌓기 (건너뛰기 연결 없음)
        self.layers = nn.ModuleList()
        channels = 64
        for i in range(8):  # 깊이 8층
            self.layers.append(nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResidualNet(nn.Module):
    """
    잔차 연결이 있는 깊은 신경망
    """
    def __init__(self, num_classes=10):
        super(ResidualNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 잔차 블록 더미
        self.layers = nn.ModuleList()
        channels = 64
        for i in range(8):  # 잔차 블록 8개
            self.layers.append(self._make_residual_block(channels))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
    
    def _make_residual_block(self, channels):
        """잔차 블록 하나를 만든다"""
        return nn.ModuleDict({
            'conv1': nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            'bn1': nn.BatchNorm2d(channels),
            'conv2': nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            'bn2': nn.BatchNorm2d(channels)
        })
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        
        for layer in self.layers:
            identity = x
            out = torch.relu(layer['bn1'](layer['conv1'](x)))
            out = layer['bn2'](layer['conv2'](out))
            x = torch.relu(out + identity)  # 잔차 연결
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def create_synthetic_dataset(num_samples=1000, image_size=32):
    """
    빠른 학습 비교를 위해 인공 데이터셋을 만든다
    """
    # 무작위 이미지
    X = torch.randn(num_samples, 3, image_size, image_size)
    # 무작위 이름표
    y = torch.randint(0, 10, (num_samples,))
    
    return TensorDataset(X, y)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    한 세대 학습시킨다
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """
    검증 집합에서 모델을 평가한다
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def compute_gradient_stats(model):
    """
    기울기의 흐름을 보이려고 기울기 통계를 계산한다
    """
    total_norm = 0
    max_grad = 0
    min_grad = float('inf')
    
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            max_grad = max(max_grad, p.grad.data.abs().max().item())
            min_grad = min(min_grad, p.grad.data.abs().min().item())
    
    total_norm = total_norm ** 0.5
    
    return total_norm, max_grad, min_grad


def compare_training(num_epochs=20, batch_size=32):
    """
    평범한 신경망과 잔차 신경망의 학습을 견준다
    """
    print("=" * 80)
    print("Training Comparison: Plain Network vs Residual Network")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # 데이터셋들을 만든다
    print("\nCreating synthetic dataset...")
    train_dataset = create_synthetic_dataset(num_samples=1000)
    val_dataset = create_synthetic_dataset(num_samples=200)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 모델 만들기
    print("Initializing models...")
    plain_model = PlainNet(num_classes=10).to(device)
    residual_model = ResidualNet(num_classes=10).to(device)
    
    # 매개변수 수 견주기
    plain_params = sum(p.numel() for p in plain_model.parameters())
    residual_params = sum(p.numel() for p in residual_model.parameters())
    
    print(f"Plain Network parameters: {plain_params:,}")
    print(f"Residual Network parameters: {residual_params:,}")
    
    # 학습 준비
    criterion = nn.CrossEntropyLoss()
    plain_optimizer = optim.Adam(plain_model.parameters(), lr=0.001)
    residual_optimizer = optim.Adam(residual_model.parameters(), lr=0.001)
    
    # 학습 기록
    history = {
        'plain': {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'grad_norm': []},
        'residual': {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'grad_norm': []}
    }
    
    print(f"\nTraining for {num_epochs} epochs...")
    print("-" * 80)
    
    for epoch in range(num_epochs):
        # 평범한 신경망 학습
        plain_train_loss, plain_train_acc = train_epoch(
            plain_model, train_loader, criterion, plain_optimizer, device)
        plain_val_loss, plain_val_acc = evaluate(
            plain_model, val_loader, criterion, device)
        
        # 역전파를 한 번 더 한 뒤 기울기 통계 얻기
        inputs, targets = next(iter(train_loader))
        inputs, targets = inputs.to(device), targets.to(device)
        plain_optimizer.zero_grad()
        outputs = plain_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        plain_grad_norm, _, _ = compute_gradient_stats(plain_model)
        
        # 잔차 신경망 학습
        residual_train_loss, residual_train_acc = train_epoch(
            residual_model, train_loader, criterion, residual_optimizer, device)
        residual_val_loss, residual_val_acc = evaluate(
            residual_model, val_loader, criterion, device)
        
        # 기울기 통계 얻기
        residual_optimizer.zero_grad()
        outputs = residual_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        residual_grad_norm, _, _ = compute_gradient_stats(residual_model)
        
        # 이력 기록
        history['plain']['train_loss'].append(plain_train_loss)
        history['plain']['train_acc'].append(plain_train_acc)
        history['plain']['val_loss'].append(plain_val_loss)
        history['plain']['val_acc'].append(plain_val_acc)
        history['plain']['grad_norm'].append(plain_grad_norm)
        
        history['residual']['train_loss'].append(residual_train_loss)
        history['residual']['train_acc'].append(residual_train_acc)
        history['residual']['val_loss'].append(residual_val_loss)
        history['residual']['val_acc'].append(residual_val_acc)
        history['residual']['grad_norm'].append(residual_grad_norm)
        
        if (epoch + 1) % 5 == 0:
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Plain    - Loss: {plain_train_loss:.4f}, Acc: {plain_train_acc:.2f}%, "
                  f"Val Acc: {plain_val_acc:.2f}%, Grad: {plain_grad_norm:.4f}")
            print(f"  Residual - Loss: {residual_train_loss:.4f}, Acc: {residual_train_acc:.2f}%, "
                  f"Val Acc: {residual_val_acc:.2f}%, Grad: {residual_grad_norm:.4f}")
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    
    # 마지막 비교
    print("\nFinal Results:")
    print(f"  Plain Network    - Val Acc: {history['plain']['val_acc'][-1]:.2f}%")
    print(f"  Residual Network - Val Acc: {history['residual']['val_acc'][-1]:.2f}%")
    print(f"  Improvement: {history['residual']['val_acc'][-1] - history['plain']['val_acc'][-1]:.2f}%")
    
    return history


def plot_comparison(history, save_path='training_comparison.png'):
    """
    학습 비교 결과를 그린다
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Plain Network vs Residual Network', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['plain']['train_loss']) + 1)
    
    # 학습 손실
    axes[0, 0].plot(epochs, history['plain']['train_loss'], 'b-', label='Plain', linewidth=2)
    axes[0, 0].plot(epochs, history['residual']['train_loss'], 'r-', label='Residual', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].set_title('Training Loss Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 학습 정확도
    axes[0, 1].plot(epochs, history['plain']['train_acc'], 'b-', label='Plain', linewidth=2)
    axes[0, 1].plot(epochs, history['residual']['train_acc'], 'r-', label='Residual', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Training Accuracy (%)')
    axes[0, 1].set_title('Training Accuracy Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 검증 정확도
    axes[1, 0].plot(epochs, history['plain']['val_acc'], 'b-', label='Plain', linewidth=2)
    axes[1, 0].plot(epochs, history['residual']['val_acc'], 'r-', label='Residual', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Validation Accuracy (%)')
    axes[1, 0].set_title('Validation Accuracy Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 기울기의 노름
    axes[1, 1].plot(epochs, history['plain']['grad_norm'], 'b-', label='Plain', linewidth=2)
    axes[1, 1].plot(epochs, history['residual']['grad_norm'], 'r-', label='Residual', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Gradient Norm')
    axes[1, 1].set_title('Gradient Flow Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    
    return fig


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RESIDUAL CONNECTIONS - TRAINING COMPARISON")
    print("=" * 80)
    
    print("\nThis experiment demonstrates:")
    print("1. Faster convergence with residual connections")
    print("2. Better gradient flow (higher gradient norms)")
    print("3. Higher final accuracy")
    print("4. More stable training")
    
    # 비교 실행
    history = compare_training(num_epochs=20, batch_size=32)
    
    # 결과 그리기
    plot_comparison(history, save_path='/home/claude/residual_connections/training_comparison.png')
    
    print("\n" + "=" * 80)
    print("Key Observations:")
    print("=" * 80)
    print("1. Residual networks maintain higher gradient norms throughout training")
    print("2. This enables better optimization and faster convergence")
    print("3. The skip connections act as 'gradient highways' to deeper layers")
    print("=" * 80 + "\n")```

## 논의

이 구현은 클래스 두 개(`PlainNet`, `ResidualNet`)를 정의하며, 이들이 어우러져 완전한 잔차 신경망 구조를 이룬다. 클래스마다 별개의 부품을 감싸므로 코드가 모듈식이고 넓히기 쉽다. `forward` 메서드가 PyTorch의 자동 미분이 쓰는 계산 그래프를 정의한다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 딥러닝 구조 설계에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
`PlainNet`의 순전파를 따라가며 텐서의 모양을 추적하라. 기본 매개변수로 입력 표본 4개의 배치에 대해 주요 연산(합성곱, 풀링, 선형층)마다의 모양을 적어라.

??? success "연습문제 1 풀이"
    입력 모양에서 출발하여 각 층을 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 (덧대기가 없으면) $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌거나 (`padding=k//2`이면) 그대로 유지된다. 커널이 2인 풀링은 공간 차원을 절반으로 만든다. 선형 층은 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임에 유의하며 추적한다. 중간 모양을 합성곱 층에서는 $(B, C, H, W)$로, 평탄화 후에는 $(B, F)$로 적는다.

---

**연습문제 2.**
$64 \times 64$ 크기의 RGB 이미지(입력 모양 $3 \times 64 \times 64$)를 받도록 구조를 수정하라. 모든 층의 차원을 그에 맞게 고치고 모델이 오류 없이 실행되는지 확인하라.

??? success "연습문제 2 풀이"
    첫 합성곱 층의 `in_channels`을 지금 값에서 3으로 바꾸어라. 식 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$으로 합성곱과 풀링 층마다의 공간 차원을 다시 계산하라. 마지막 합성곱/풀링 층의 펼친 출력에 맞게 첫 선형층의 `in_features`을 고쳐라. `model = PlainNet(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`으로 확인하라.

---

**연습문제 3.**
같은 입출력 차원에서 표준 합성곱과 깊이별 분리 합성곱의 매개변수 개수와 FLOPs를 비교하라. 계산 절감이 가장 큰 것은 언제인가?

??? success "연습문제 3 풀이"
    표준 `Conv2d(C_in, C_out, k)`은 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개의 매개변수를 갖는다. 깊이별 분리 합성곱은 이를 둘로 나눈다. (1) 깊이별: $C_{{\text{{in}}}} \times k^2$개(입력 채널마다 필터 하나), (2) 점별: $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 합성곱)이다. 매개변수의 비는 대략 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 매개변수가 약 $8{-}9\times$ 적어진다. 절감은 $C_{{\text{{out}}}}$과 $k$가 모두 클 때 가장 크다.

---

**연습문제 4.**
층이나 블록의 수를 설정할 수 있도록 `PlainNet`을 확장하라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`으로 훑는다. (보통의 파이썬 리스트가 아니라) `nn.ModuleList`을 써야 PyTorch가 최적화를 위해 모든 매개변수를 등록한다. `for n in [2, 4, 8]: model = PlainNet(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.
