# 학습

PixelCNN 그림 만들어 내기를 위한 익히기 대본. 이 대본은 다음을 보인다.

자기 되돌이 모델은 앞선 모든 낱개를 조건으로 삼아 낱개마다 미리 헤아려 자료를 만든다. 이 단원은 자기 되돌이 모델 부품의 짜기를 보이며 차례대로 만들어 내는 과정과 그 얼개의 요구를 그려 보인다.

## 코드

```python
"""
PixelCNN 그림 만들어 내기를 위한 익히기 대본

이 대본은 다음을 보인다.
1. MNIST 자료 묶음으로 PixelCNN 익히기
2. 자기 되돌이 그림 만들어 내기
3. 만든 표본 그려 보기
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ========================================================================
# 메인
# ========================================================================

from pixelcnn import PixelCNN


def binarize(x: torch.Tensor) -> torch.Tensor:
    """
    그림을 검정과 흰색 두값으로 만든다.
    
    이는 문제를 단순하게 만든다. 곧 화소마다 있을 수 있는 값 256가지를
    헤아리는 대신 두값(0이나 1)만 헤아린다.
    
    인수:
        x: 값이 [0, 1]인 그림 텐서
        
    반환값:
        두값으로 만든 그림(0 또는 1)
    """
    return (x > 0.5).float()


def train_epoch(model: nn.Module,
                dataloader: torch.utils.data.DataLoader,
                optimizer: optim.Optimizer,
                device: str) -> float:
    """
    한 세대를 학습한다.
    
    인수:
        model: PixelCNN 모델
        dataloader: 학습 데이터로더
        optimizer: 최적화기
        device: 학습에 쓸 장치
        
    반환값:
        바퀴의 평균 손실
    """
    model.train()
    total_loss = 0
    
    # 두값 어긋 엔트로피 손실
    # 화소마다 1(흰색)일 확률을 헤아린다
    criterion = nn.BCEWithLogitsLoss()
    
    for images, _ in dataloader:  # 만들어 내기에는 이름표가 필요 없다
        # 기기로 옮기고 두값으로 만든다
        images = binarize(images.to(device))
        
        # 순전파
        logits = model(images)
        
        # 손실을 계산한다
        # 실제 그림의 화소 값을 헤아리려 한다
        loss = criterion(logits, images)
        
        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model: nn.Module,
            dataloader: torch.utils.data.DataLoader,
            device: str) -> float:
    """
    시험 묶음에서 모델을 따진다.
    
    인수:
        model: PixelCNN 모델
        dataloader: 시험 자료 불러개
        device: 평가할 장치
        
    반환값:
        시험 묶음의 평균 손실
    """
    model.eval()
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = binarize(images.to(device))
            logits = model(images)
            loss = criterion(logits, images)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def visualize_samples(model: nn.Module,
                     real_images: torch.Tensor,
                     device: str,
                     n_samples: int = 16):
    """
    표본을 만들어 그려 본다.
    
    인수:
        model: 익힌 PixelCNN
        real_images: 견주기 위한 실제 그림
        device: 만들어 낼 기기
        n_samples: 만들 표본의 개수
    """
    model.eval()
    
    # 새 그림을 만든다
    print("Generating samples (this may take a minute)...")
    with torch.no_grad():
        generated = model.generate(
            shape=(n_samples, 28, 28),
            device=device
        )
    
    # 그림을 만든다
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    
    # 실제 그림을 위 두 줄에 그린다
    for i in range(2):
        for j in range(8):
            idx = i * 8 + j
            axes[i, j].imshow(real_images[idx, 0].cpu(), cmap='gray')
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_ylabel('Real', rotation=0, labelpad=30, fontsize=12)
    
    # 만든 그림을 아래 두 줄에 그린다
    for i in range(2, 4):
        for j in range(8):
            idx = (i - 2) * 8 + j
            axes[i, j].imshow(generated[idx, 0].cpu(), cmap='gray')
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_ylabel('Generated', rotation=0, labelpad=30, fontsize=12)
    
    plt.suptitle('Real Images vs Generated Images', fontsize=16, y=0.98)
    plt.tight_layout()
    
    return fig


def main():
    """
    으뜸 익히기 물길
    """
    print("=" * 70)
    print("PixelCNN: Autoregressive Image Generation")
    print("=" * 70)
    
    # ==================== 채비 ====================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    if device == 'cpu':
        print("\nWARNING: Training on CPU will be slow!")
        print("PixelCNN generates pixel-by-pixel, which is computationally intensive.")
        print("Consider using a smaller model or fewer epochs for CPU training.\n")
    
    # 초매개변수
    BATCH_SIZE = 64
    N_EPOCHS = 20  # CPU에서 익히면 줄인다
    LEARNING_RATE = 0.001
    N_CHANNELS = 64
    N_RESIDUAL_BLOCKS = 5
    
    print(f"Hyperparameters:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {N_EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Feature Channels: {N_CHANNELS}")
    print(f"  Residual Blocks: {N_RESIDUAL_BLOCKS}")
    
    # ==================== 자료 불러오기 ====================
    print(f"\n{'='*70}")
    print("Step 1: Loading MNIST dataset...")
    print(f"{'='*70}")
    
    # 바꾸기: 텐서로 바꾸고 [0, 1]으로 고르게 맞춘다
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # MNIST 불러오기
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # 데이터로더들을 만든다
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    
    print(f"\n✓ Loaded MNIST dataset")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Image size: 28x28")
    
    # ==================== 모델 첫자리매김 ====================
    print(f"\n{'='*70}")
    print("Step 2: Initializing PixelCNN...")
    print(f"{'='*70}")
    
    model = PixelCNN(
        n_channels=N_CHANNELS,
        n_residual_blocks=N_RESIDUAL_BLOCKS
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel initialized")
    print(f"  Parameters: {n_params:,}")
    
    # 최적화기
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # ==================== 익히기 ====================
    print(f"\n{'='*70}")
    print("Step 3: Training PixelCNN...")
    print(f"{'='*70}")
    print("\nNote: PixelCNN training is slow because each pixel depends on")
    print("all previous pixels. This is the price of autoregressive modeling!")
    print()
    
    train_losses = []
    test_losses = []
    
    for epoch in tqdm(range(N_EPOCHS), desc="Training"):
        # 학습
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # 평가한다
        test_loss = evaluate(model, test_loader, device)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        # 진행 상황 출력
        if (epoch + 1) % 5 == 0:
            print(f"\nEpoch {epoch+1}/{N_EPOCHS}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Test Loss: {test_loss:.4f}")
    
    print(f"\n✓ Training complete!")
    print(f"  Final train loss: {train_losses[-1]:.4f}")
    print(f"  Final test loss: {test_losses[-1]:.4f}")
    
    # ==================== 그려 보기 ====================
    print(f"\n{'='*70}")
    print("Step 4: Creating visualizations...")
    print(f"{'='*70}")
    
    # 익히기 곡선을 그린다
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    plt.plot(test_losses, label='Test Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross-Entropy Loss')
    plt.title('PixelCNN: Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pixelcnn_training.png', dpi=150)
    print("✓ Saved pixelcnn_training.png")
    
    # 견주려고 실제 그림 몇 장을 얻는다
    real_images, _ = next(iter(test_loader))
    real_images = binarize(real_images[:16])
    
    # 표본을 만들어 그려 본다
    fig = visualize_samples(model, real_images, device, n_samples=16)
    plt.savefig('pixelcnn_samples.png', dpi=150)
    print("✓ Saved pixelcnn_samples.png")
    
    # ==================== 만들어 내기 보여 주기 ====================
    print(f"\n{'='*70}")
    print("Step 5: Demonstrating autoregressive generation...")
    print(f"{'='*70}")
    
    print("\nGenerating a single image step-by-step...")
    print("Watch how the image is filled pixel by pixel!")
    
    # 그림 하나를 만들며 중간 걸음을 보인다
    model.eval()
    height, width = 28, 28
    
    # 만들어 내는 과정을 보이는 그림 낱장을 만든다
    frames = []
    sample = torch.zeros(1, 1, height, width).to(device)
    
    # 화소 50개마다 그림 낱장을 만들어 갈무리한다
    pixel_count = 0
    with torch.no_grad():
        for i in range(height):
            for j in range(width):
                logits = model(sample)
                probs = torch.sigmoid(logits[:, :, i, j])
                sample[:, :, i, j] = torch.bernoulli(probs)
                
                pixel_count += 1
                if pixel_count % 50 == 0 or pixel_count == height * width:
                    frames.append(sample.clone())
    
    # 만들어 내는 과정을 그려 본다
    n_frames = len(frames)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    
    for idx, frame in enumerate(frames):
        axes[idx].imshow(frame[0, 0].cpu(), cmap='gray')
        axes[idx].set_title(f'Pixel {(idx+1)*50}' if idx < n_frames-1 else 'Complete')
        axes[idx].axis('off')
    
    plt.suptitle('Autoregressive Generation Process', fontsize=14)
    plt.tight_layout()
    plt.savefig('generation_process.png', dpi=150)
    print("✓ Saved generation_process.png")
    
    # ==================== 간추리기 ====================
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print("\nKey Observations:")
    print("1. PixelCNN generates images pixel-by-pixel")
    print("2. Each pixel depends on all previous pixels (autoregressive)")
    print("3. Generation is slow but produces diverse samples")
    print("4. The model learned MNIST digit structure!")
    print("\nCheck the generated PNG files for visualizations.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()```

## 논의

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 결은 더 복잡한 경우로 자연스럽게 넓혀진다. 웃매개변수와 얼개 변형, 여러 자료 묶음을 실험하면 그림 만들어 내기 일에 대한 이해가 깊어지고 실제 직관이 쌓인다.

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
