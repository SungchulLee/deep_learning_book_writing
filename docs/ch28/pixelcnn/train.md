# 이진 MNIST의 PixelCNN 학습

이 절은 [이진 PixelCNN](pixelcnn.md)에 MNIST를 학습하는 절차다. 회색조를 임계로 이진화하고 픽셀별 베르누이 음의 로그확률을 최소화한다. 학습은 모든 픽셀의 logits를 함께 계산하지만 새 이미지 생성은 래스터 순서로 진행한다.

## 1. 학습 코드

```python
"""
PixelCNN 이미지 생성을 위한 학습 스크립트

이 스크립트는 다음을 보인다.
1. MNIST 데이터셋으로 PixelCNN 학습
2. 자기회귀 이미지 생성
3. 생성한 표본 시각화
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

import torch.nn.functional as F

# --- pixelcnn ---
"""
자기회귀 이미지 생성을 위한 단순한 PixelCNN

PixelCNN은 래스터 스캔 순서(왼쪽에서 오른쪽, 위에서 아래)로 픽셀 하나씩 이미지를 생성한다.
픽셀마다 앞서 생성한 모든 픽셀을 조건으로 예측한다.

핵심 개념에 집중한 교육용 단순화 구현이다.
"""


# ========================================================================
# 메인
# ========================================================================


class MaskedConv2d(nn.Conv2d):
    """
    자기회귀 이미지 생성을 위한 마스크 컨볼루션.
    
    PixelCNN의 핵심 아이디어: 픽셀마다 앞선 픽셀(위쪽과 왼쪽)에만
    의존하도록 컨볼루션을 마스킹한다.
    
    마스크 종류:
    - A 마스크: 첫 레이어용이며 현재 픽셀을 뺀다
    - B 마스크: 뒤 레이어용이며 현재 픽셀을 넣는다
    """
    
    def __init__(self, mask_type: str, *args, **kwargs):
        """
        마스크 컨볼루션을 초기화한다.
        
        인수:
            mask_type: 'A'나 'B'
            *args, **kwargs: nn.Conv2d에 줄 인자
        """
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        
        assert mask_type in ['A', 'B'], "mask_type must be 'A' or 'B'"
        self.mask_type = mask_type
        
        # 마스크를 버퍼로 등록한다(학습 중에 갱신되지 않는다)
        self.register_buffer('mask', torch.zeros_like(self.weight))
        self.create_mask()
    
    def create_mask(self):
        """
        자기회귀 마스크를 만든다.
        
        마스크는 다음을 보장한다.
        - 위쪽 픽셀은 볼 수 있다
        - 왼쪽 픽셀은 볼 수 있다
        - 현재 픽셀: B 마스크에서만 보인다
        - 아래쪽과 오른쪽 픽셀은 볼 수 없다
        """
        # 차원을 얻는다
        # 웨이트 형태: [출력 채널, 입력 채널, 커널 높이, 커널 너비]
        k_h, k_w = self.weight.shape[2:]
        
        # 마스크를 모두 1로 초기화한다
        self.mask.fill_(1)
        
        # 아래 반을 0으로 만든다
        self.mask[:, :, k_h // 2 + 1:, :] = 0
        
        # 가운데 줄의 오른쪽을 0으로 만든다
        # A 마스크: 가운데 픽셀을 뺀다
        # B 마스크: 가운데 픽셀을 넣는다
        if self.mask_type == 'A':
            self.mask[:, :, k_h // 2, k_w // 2:] = 0
        else:  # mask_type == 'B'
            self.mask[:, :, k_h // 2, k_w // 2 + 1:] = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        마스크된 웨이트로 하는 순전파.
        
        인수:
            x: 입력 텐서
            
        반환값:
            마스크 컨볼루션의 출력
        """
        # 컨볼루션 앞에서 웨이트에 마스크를 곱한다
        # 그러면 허용된 픽셀만 쓰인다
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class ResidualBlock(nn.Module):
    """
    마스크 컨볼루션을 갖춘 잔차 블록.
    
    구조:
        입력 -> 마스크 컨볼루션 -> ReLU -> 마스크 컨볼루션 -> 입력과 더하기
    """
    
    def __init__(self, channels: int):
        """
        잔차 블록을 초기화한다.
        
        인수:
            channels: 채널의 수
        """
        super(ResidualBlock, self).__init__()
        
        # 첫 컨볼루션 뒤로는 모두 B 마스크다
        self.conv1 = MaskedConv2d('B', channels, channels // 2, 
                                  kernel_size=1, padding=0)
        self.conv2 = MaskedConv2d('B', channels // 2, channels // 2,
                                  kernel_size=3, padding=1)
        self.conv3 = MaskedConv2d('B', channels // 2, channels,
                                  kernel_size=1, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """잔차 블록을 지나는 순전파."""
        residual = x
        
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        
        # 잔차 연결
        return out + residual


class PixelCNN(nn.Module):
    """
    이진(검정/흰색) 이미지 생성을 위한 단순한 PixelCNN.
    
    이 자기회귀 모델은 픽셀 하나씩 이미지를 생성한다.
    P(이미지) = P(x₁) × P(x₂|x₁) × P(x₃|x₁,x₂) × ... × P(xₙ|x₁,...,xₙ₋₁)
    
    여기서 xᵢ는 픽셀 값이다.
    """
    
    def __init__(self, 
                 n_channels: int = 64,
                 n_residual_blocks: int = 5):
        """
        PixelCNN을 초기화한다.
        
        인수:
            n_channels: 피처 채널의 수
            n_residual_blocks: 잔차 블록의 수
        """
        super(PixelCNN, self).__init__()
        
        # 첫 레이어는 A 마스크를 쓴다(현재 픽셀을 뺀다)
        self.input_conv = MaskedConv2d('A', 1, n_channels,
                                       kernel_size=7, padding=3)
        
        # 잔차 블록
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(n_channels) for _ in range(n_residual_blocks)
        ])
        
        # 출력 레이어
        self.output_conv1 = MaskedConv2d('B', n_channels, n_channels,
                                         kernel_size=1)
        self.output_conv2 = MaskedConv2d('B', n_channels, n_channels,
                                         kernel_size=1)
        
        # 마지막 레이어: 픽셀마다 확률을 예측한다
        # 이진 이미지에서는 채널 1개를 출력한다(흰색일 확률)
        self.final_conv = MaskedConv2d('B', n_channels, 1,
                                       kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        PixelCNN을 지나는 순전파.
        
        인수:
            x: 입력 이미지 [배치 크기, 1, 높이, 너비]
               [0, 1] 안의 값
               
        반환값:
            픽셀마다의 로짓 [배치 크기, 1, 높이, 너비]
            확률을 얻으려 시그모이드를 쓴다
        """
        # 첫 마스크 컨볼루션
        out = F.relu(self.input_conv(x))
        
        # 잔차 블록
        for block in self.residual_blocks:
            out = block(out)
        
        # 출력 컨볼루션
        out = F.relu(self.output_conv1(out))
        out = F.relu(self.output_conv2(out))
        
        # 마지막 예측
        out = self.final_conv(out)
        
        return out
    
    @torch.no_grad()
    def generate(self, 
                 shape: tuple,
                 device: str = 'cpu') -> torch.Tensor:
        """
        이미지를 자기회귀로 생성한다.
        
        이것이 자기회귀 생성의 핵심이다.
        1. 빈 이미지(모두 0)에서 시작한다
        2. 픽셀 자리마다(위에서 아래로, 왼쪽에서 오른쪽으로):
           a. 픽셀이 1일 확률을 예측한다
           b. 베르누이 분포에서 뽑는다
           c. 픽셀을 채운다
        3. 완성된 이미지를 돌려준다
        
        인수:
            shape: (배치 크기, 높이, 너비)
            device: 생성에 쓸 장치
            
        반환값:
            생성한 이미지 [배치 크기, 1, 높이, 너비]
        """
        self.eval()
        
        batch_size, height, width = shape
        
        # 빈 이미지(모두 0)에서 시작한다
        samples = torch.zeros(batch_size, 1, height, width).to(device)
        
        # 픽셀 하나씩 생성한다
        # 래스터 스캔 순서: 위에서 아래로, 왼쪽에서 오른쪽으로
        for i in range(height):
            for j in range(width):
                # 현재 픽셀의 예측을 얻는다
                # 참고: 앞서 생성한 픽셀을 모두 쓴다
                logits = self.forward(samples)
                
                # 현재 픽셀 자리의 확률을 얻는다
                probs = torch.sigmoid(logits[:, :, i, j])
                
                # 베르누이 분포에서 뽑는다
                # 그래서 생성이 확률적이 된다
                samples[:, :, i, j] = torch.bernoulli(probs)
        
        return samples


if __name__ == "__main__":
    """
    데모: 더미 데이터로 PixelCNN을 시험한다
    """
    
    print("=" * 70)
    print("Testing Simplified PixelCNN")
    print("=" * 70)
    
    # 모델 생성
    model = PixelCNN(n_channels=32, n_residual_blocks=3)
    
    # 파라메터 개수 세기
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel has {n_params:,} parameters")
    
    # 순전파 시험
    batch_size = 4
    height, width = 28, 28  # MNIST 크기
    
    # 임시 입력 만들기
    x = torch.rand(batch_size, 1, height, width)
    
    # 순전파
    output = model(x)
    
    print(f"\nForward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    
    # 생성을 시험한다
    print(f"\nGenerating images...")
    print(f"  This will take a while (generating pixel by pixel)...")
    
    # 데모용으로 작은 이미지를 생성한다
    small_shape = (2, 8, 8)  # 8x8 크기 이미지 2장
    generated = model.generate(small_shape, device='cpu')
    
    print(f"  Generated shape: {generated.shape}")
    print(f"  Sample pixel values: {generated[0, 0, :3, :3]}")
    
    print("\n✓ PixelCNN working correctly!")
    print("\nNote: For real training, use the train.py script")
    print("which trains on actual image data (like MNIST)")


def binarize(x: torch.Tensor) -> torch.Tensor:
    """
    이미지를 검정과 흰색으로 이진화한다.
    
    이는 문제를 단순하게 만든다. 곧 픽셀마다 가능한 값 256가지를
    예측하는 대신 두 값(0이나 1)만 예측한다.
    
    인수:
        x: 값이 [0, 1]인 이미지 텐서
        
    반환값:
        이진화한 이미지(0 또는 1)
    """
    return (x > 0.5).float()


def train_epoch(model: nn.Module,
                dataloader: torch.utils.data.DataLoader,
                optimizer: optim.Optimizer,
                device: str) -> float:
    """
    한 에포크를 학습한다.
    
    인수:
        model: PixelCNN 모델
        dataloader: 학습 데이터로더
        optimizer: 옵티마이저
        device: 학습에 쓸 장치
        
    반환값:
        에포크의 평균 로스
    """
    model.train()
    total_loss = 0
    
    # 이진 교차 엔트로피 로스
    # 픽셀마다 1(흰색)일 확률을 예측한다
    criterion = nn.BCEWithLogitsLoss()
    
    for images, _ in dataloader:  # 생성에는 라벨이 필요 없다
        # 장치로 옮기고 이진화한다
        images = binarize(images.to(device))
        
        # 순전파
        logits = model(images)
        
        # 로스를 계산한다
        # 실제 이미지의 픽셀 값을 예측하려 한다
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
    시험 세트에서 모델을 평가한다.
    
    인수:
        model: PixelCNN 모델
        dataloader: 시험 데이터로더
        device: 평가할 장치
        
    반환값:
        시험 세트의 평균 로스
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
    표본을 생성해 시각화한다.
    
    인수:
        model: 학습된 PixelCNN
        real_images: 견주기 위한 실제 이미지
        device: 생성에 쓸 장치
        n_samples: 생성할 표본의 개수
    """
    model.eval()
    
    # 새 이미지를 생성한다
    print("Generating samples (this may take a minute)...")
    with torch.no_grad():
        generated = model.generate(
            shape=(n_samples, 28, 28),
            device=device
        )
    
    # 그림을 만든다
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    
    # 실제 이미지를 위 두 줄에 그린다
    for i in range(2):
        for j in range(8):
            idx = i * 8 + j
            axes[i, j].imshow(real_images[idx, 0].cpu(), cmap='gray')
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_ylabel('Real', rotation=0, labelpad=30, fontsize=12)
    
    # 생성한 이미지를 아래 두 줄에 그린다
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
    메인 학습 파이프라인
    """
    print("=" * 70)
    print("PixelCNN: Autoregressive Image Generation")
    print("=" * 70)
    
    # ==================== 설정 ====================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    if device == 'cpu':
        print("\nWARNING: Training on CPU will be slow!")
        print("PixelCNN generates pixel-by-pixel, which is computationally intensive.")
        print("Consider using a smaller model or fewer epochs for CPU training.\n")
    
    # 하이퍼파라메터
    BATCH_SIZE = 64
    N_EPOCHS = 20  # CPU에서 학습하면 줄인다
    LEARNING_RATE = 0.001
    N_CHANNELS = 64
    N_RESIDUAL_BLOCKS = 5
    
    print(f"Hyperparameters:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {N_EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Feature Channels: {N_CHANNELS}")
    print(f"  Residual Blocks: {N_RESIDUAL_BLOCKS}")
    
    # ==================== 데이터 불러오기 ====================
    print(f"\n{'='*70}")
    print("Step 1: Loading MNIST dataset...")
    print(f"{'='*70}")
    
    # 변환: 텐서로 바꾸고 [0, 1]로 정규화한다
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
    
    # ==================== 모델 초기화 ====================
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
    
    # 옵티마이저
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # ==================== 학습 ====================
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
    
    # ==================== 시각화 ====================
    print(f"\n{'='*70}")
    print("Step 4: Creating visualizations...")
    print(f"{'='*70}")
    
    # 학습 곡선을 그린다
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
    
    # 견주려고 실제 이미지 몇 장을 얻는다
    real_images, _ = next(iter(test_loader))
    real_images = binarize(real_images[:16])
    
    # 표본을 생성해 시각화한다
    fig = visualize_samples(model, real_images, device, n_samples=16)
    plt.savefig('pixelcnn_samples.png', dpi=150)
    print("✓ Saved pixelcnn_samples.png")
    
    # ==================== 생성 데모 ====================
    print(f"\n{'='*70}")
    print("Step 5: Demonstrating autoregressive generation...")
    print(f"{'='*70}")
    
    print("\nGenerating a single image step-by-step...")
    print("Watch how the image is filled pixel by pixel!")
    
    # 이미지 하나를 생성하며 중간 스텝을 보인다
    model.eval()
    height, width = 28, 28
    
    # 생성 과정을 보이는 프레임을 만든다
    frames = []
    sample = torch.zeros(1, 1, height, width).to(device)
    
    # 픽셀 50개마다 프레임을 만들어 저장한다
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
    
    # 생성 과정을 시각화한다
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
    
    # ==================== 요약 ====================
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
    main()
```

기본 설정은 64채널·잔차 블록 5개, 배치 64, Adam 0.001과 20에포크다. 입력에 ToTensor를 적용한 뒤 x>0.5인 픽셀만 1로 바꾼다. 0.5 자체는 0이며 확률적으로 이진화하는 방법과 다르다. 학습·평가·실제 이미지 표시에 같은 변환을 적용한다.

import하는 로컬 pixelcnn.py와 같은 모듈이 검색되어야 하며 모델·입력은 선택한 CUDA 또는 CPU에 둔다. 기본 파라메터 수는 78,785개다. 마스크의 인과성이 지켜지므로 현재 이미지를 입력과 타깃으로 함께 주어도 현재 픽셀 값을 직접 복사하는 일반 오토인코더 로스와 다르다.

## 2. 로스의 단위와 계산비용

BCEWithLogitsLoss는 Sigmoid를 포함한 안정적인 BCE 계산으로 이진 타깃을 평가한다. 기본 mean은 배치·채널·공간 원소 전체 평균이라 반환은 **픽셀당 NLL의 nat 단위**다. 28×28 표본당 NLL 합은 이 값에 784를 곱하고, 이진 bits/pixel은 ln 2로 나눈다. 8비트 연속화 보정 8을 여기에 더하는 것은 맞지 않는다.

에포크 함수는 배치 평균들을 단순 평균하므로 마지막 배치 크기가 다르면 전체 표본·픽셀 평균과 다르다. 정확한 데이터셋 지표에는 처리 표본 수 가중이 필요하다. train 곡선은 학습 도중 바뀌는 모델, test 곡선은 종료 모델의 평가값이다.

코드 출력 안내문은 “각 픽셀 의존성 때문에 학습이 순차적”이라는 인상을 주지만, 마스크 CNN의 학습 forward는 알려진 이미지에서 모든 위치를 병렬 계산한다. 생성은 784번의 전체 이미지 forward를 반복하므로 느릴 수 있다. 학습 속도와 생성 병목을 구분해야 한다.

## 3. 평가·표본 격자

평가는 eval·no_grad이며 라벨은 사용하지 않는 무조건부 이미지 모형이다. 시험 로스를 매 에포크 기록하므로 이를 하이퍼파라메터·종료 선택에 쓰면 별도 검증셋과 최종 시험셋을 구분해야 한다. 현재 main에는 체크포인트 저장·최적 모델 복원·조기 종료가 없다.

visualize_samples는 고정 4×8 격자에 실제 16장과 생성 16장을 표시한다. 인수 n_samples를 임의로 바꿔도 격자 반복 수가 바뀌지 않아 16보다 작으면 인덱스 오류, 크면 일부만 표시할 수 있다. real_images에도 적어도 16장이 필요하다. 표본은 픽셀별 베르누이 추출이지만 격자의 다양성·품질은 별도 평가가 필요하다.

## 4. 생성 과정 그림의 프레임 수 문제

마지막 단계는 50픽셀마다 중간 상태를 저장하고 마지막 완성 상태도 더한다. 784픽셀에서는 50,…,750의 15개와 마지막 1개로 총 16프레임이다. 그런데 표시 축은 2×5로 10개뿐이다. 따라서 현재 루프는 11번째 프레임에서 축 인덱스를 벗어나 generation_process.png 저장까지 완료하지 못한다.

따라서 코드 끝의 완료 안내문을 이 경로의 성공 증거로 읽으면 안 된다. 자동으로 “다양한 숫자 구조를 배웠다”는 결론을 내리기보다 독립 시험 NLL과 품질·다양성을 함께 살펴야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
그래디언트 초기화와 마스크 적용은 왜 별개의 연산인가?

</div>

??? success "풀이"

    zero_grad는 이전 step의 grad를 비우고 마스크는 순전파에서 금지된 입력 의존을 차단한다. 현재 weight.data 마스킹은 금지 웨이트의 grad를 자동으로 0으로 만드는 방식이 아니다. 초기화를 생략하면 과거 배치 그래디언트까지 누적되어 다른 업데이트가 된다. 두 기능을 혼동하지 않고 실제 순전파의 인과성과 최적화의 그래디언트 처리를 각각 확인해야 한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
본문의 옵티마이저 비교와 공통 평가 단위를 설명하라.

</div>

??? success "풀이"

    이미 Adam 0.001을 쓰므로 같은 설정으로 다시 지정하는 것은 새 비교가 아니다. 다른 옵티마이저·러닝레이트에는 동일한 데이터 이진화·분할·구조와 같은 튜닝 예산을 적용한다. 검증 픽셀당 NLL을 표본 가중 평균하고 생성의 품질·다양성·비용도 따로 비교한다. 학습 로스나 단일 시드의 이미지로 일반적인 우열을 정하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
낮은 로스와 실행 완료 메시지만으로 무엇을 확인할 수 없는가?

</div>

??? success "풀이"

    타깃 누출이 없는지, 다양한 숫자를 생성하는지, 독립 데이터에 잘 맞는지, 모든 시각화 경로가 끝나는지는 별도다. 특히 본문 생성 과정은 16프레임을 10축에 넣어 실패한다. 마스크 경로·타깃·프레임 수를 정적으로 확인하고 검증 NLL·대표 표본·다양성을 함께 평가한다. 현재·미래 픽셀을 볼 수 있는 임의 BatchNorm이나 비인과 경로를 추가해 로스를 낮추면 확률 계약을 깨뜨릴 수 있다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
조기 종료 예시를 적용하려면 무엇을 보완해야 하는가?

</div>

??? success "풀이"

    현재 evaluate는 model·dataloader·device를 받으므로 아래 두 인수 호출은 맞지 않는다. 학습 데이터에서 검증셋을 만들고 동일 이진화·로스 단위로 평가한다. state_dict().copy는 얕은 복사여서 독립 웨이트 보존에 부족하며 파일 저장이나 깊은 복제가 필요하다. 정상 종료에도 최적 모델을 복원하고 구조·전처리·선택 규칙을 기록해야 한다. 최종 시험셋은 선택에 반복 사용하지 않는다.

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


## 정리하며

이진 PixelCNN은 모든 위치의 베르누이 logits를 병렬로 학습하고 픽셀별로 표본을 생성한다. 로스의 픽셀당 단위·이진화·검증 분할을 맞추고, 고정 격자·프레임 수 같은 시각화 계약도 확인해야 한다. 생성 품질은 완료 문구나 학습 로스와 별도로 평가한다.
