# 이진 MNIST의 PixelCNN 학습

이 절은 [이진 PixelCNN](pixelcnn.md)에 MNIST를 학습하는 절차다. 회색조를 문턱으로 이진화하고 픽셀별 베르누이 음의 로그확률을 최소화한다. 학습은 모든 픽셀의 logits를 함께 계산하지만 새 이미지 생성은 래스터 순서로 진행한다.

## 1. 학습 코드

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
    시험 배치에서 모델을 따진다.
    
    인수:
        model: PixelCNN 모델
        dataloader: 시험 자료 불러개
        device: 평가할 장치
        
    반환값:
        시험 배치의 평균 손실
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
    main()
```

기본 설정은 64채널·잔차 블록 5개, 배치 64, Adam 0.001과 20에포크다. 입력에 ToTensor를 적용한 뒤 x>0.5인 픽셀만 1로 바꾼다. 0.5 자체는 0이며 확률적으로 이진화하는 방법과 다르다. 학습·평가·실제 이미지 표시에 같은 변환을 적용한다.

import하는 로컬 pixelcnn.py와 같은 모듈이 검색되어야 하며 모델·입력은 선택한 CUDA 또는 CPU에 둔다. 기본 파라미터 수는 78,785개다. 마스크의 인과성이 지켜지므로 현재 이미지를 입력과 타깃으로 함께 주어도 현재 픽셀 값을 직접 복사하는 일반 오토인코더 손실과 다르다.

## 2. 손실의 단위와 계산비용

BCEWithLogitsLoss는 Sigmoid를 포함한 안정적인 BCE 계산으로 이진 타깃을 평가한다. 기본 mean은 배치·채널·공간 원소 전체 평균이라 반환은 **픽셀당 NLL의 nat 단위**다. 28×28 표본당 NLL 합은 이 값에 784를 곱하고, 이진 bits/pixel은 ln 2로 나눈다. 8비트 연속화 보정 8을 여기에 더하는 것은 맞지 않는다.

에포크 함수는 배치 평균들을 단순 평균하므로 마지막 배치 크기가 다르면 전체 표본·픽셀 평균과 다르다. 정확한 데이터셋 지표에는 처리 표본 수 가중이 필요하다. train 곡선은 학습 도중 바뀌는 모델, test 곡선은 종료 모델의 평가값이다.

코드 출력 안내문은 “각 픽셀 의존성 때문에 학습이 순차적”이라는 인상을 주지만, 마스크 CNN의 학습 forward는 알려진 이미지에서 모든 위치를 병렬 계산한다. 생성은 784번의 전체 이미지 forward를 반복하므로 느릴 수 있다. 학습 속도와 생성 병목을 구분해야 한다.

## 3. 평가·표본 격자

평가는 eval·no_grad이며 라벨은 사용하지 않는 무조건부 이미지 모형이다. 테스트 손실을 매 에포크 기록하므로 이를 하이퍼파라미터·종료 선택에 쓰면 별도 검증셋과 최종 테스트셋을 구분해야 한다. 현재 main에는 체크포인트 저장·최적 모델 복원·조기 종료가 없다.

visualize_samples는 고정 4×8 격자에 실제 16장과 생성 16장을 표시한다. 인수 n_samples를 임의로 바꿔도 격자 반복 수가 바뀌지 않아 16보다 작으면 인덱스 오류, 크면 일부만 표시할 수 있다. real_images에도 적어도 16장이 필요하다. 표본은 픽셀별 베르누이 추출이지만 격자의 다양성·품질은 별도 평가가 필요하다.

## 4. 생성 과정 그림의 프레임 수 문제

마지막 단계는 50픽셀마다 중간 상태를 저장하고 마지막 완성 상태도 더한다. 784픽셀에서는 50,…,750의 15개와 마지막 1개로 총 16프레임이다. 그런데 표시 축은 2×5로 10개뿐이다. 따라서 현재 루프는 11번째 프레임에서 축 인덱스를 벗어나 generation_process.png 저장까지 완료하지 못한다.

코드·기존 출력·그림은 보존했으므로 완료 안내문을 이 경로의 성공 증거로 읽으면 안 된다. 이 편집에서는 학습·샘플링·그림 생성을 실행하지 않았다. 자동으로 “다양한 숫자 구조를 배웠다”는 결론을 내리기보다 독립 테스트 NLL과 품질·다양성을 함께 살펴야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
기울기 초기화와 마스크 적용은 왜 별개의 연산인가?

</div>

??? success "연습문제 1 풀이"

    zero_grad는 이전 step의 grad를 비우고 마스크는 순전파에서 금지된 입력 의존을 차단한다. 현재 weight.data 마스킹은 금지 가중치의 grad를 자동으로 0으로 만드는 방식이 아니다. 초기화를 생략하면 과거 배치 기울기까지 누적되어 다른 업데이트가 된다. 두 기능을 혼동하지 않고 실제 순전파의 인과성과 최적화의 기울기 처리를 각각 확인해야 한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
본문의 옵티마이저 비교와 공통 평가 단위를 설명하라.

</div>

??? success "연습문제 2 풀이"

    이미 Adam 0.001을 쓰므로 같은 설정으로 다시 지정하는 것은 새 비교가 아니다. 다른 옵티마이저·학습률에는 동일한 자료 이진화·분할·구조와 같은 튜닝 예산을 적용한다. 검증 픽셀당 NLL을 표본 가중 평균하고 생성의 품질·다양성·비용도 따로 비교한다. 훈련 손실이나 단일 시드의 이미지로 일반적인 우열을 정하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
낮은 손실과 실행 완료 메시지만으로 무엇을 확인할 수 없는가?

</div>

??? success "연습문제 3 풀이"

    타깃 누출이 없는지, 다양한 숫자를 생성하는지, 독립 자료에 잘 맞는지, 모든 시각화 경로가 끝나는지는 별도다. 특히 본문 생성 과정은 16프레임을 10축에 넣어 실패한다. 마스크 경로·타깃·프레임 수를 정적으로 확인하고 검증 NLL·대표 표본·다양성을 함께 평가한다. 현재·미래 픽셀을 볼 수 있는 임의 BatchNorm이나 비인과 경로를 추가해 손실을 낮추면 확률 계약을 깨뜨릴 수 있다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
조기 종료 예시를 적용하려면 무엇을 보완해야 하는가?

</div>

??? success "연습문제 4 풀이"

    현재 evaluate는 model·dataloader·device를 받으므로 아래 두 인수 호출은 맞지 않는다. 학습 자료에서 검증셋을 만들고 동일 이진화·손실 단위로 평가한다. state_dict().copy는 얕은 복사여서 독립 가중치 보존에 부족하며 파일 저장이나 깊은 복제가 필요하다. 정상 종료에도 최적 모델을 복원하고 구조·전처리·선택 규칙을 기록해야 한다. 최종 테스트는 선택에 반복 사용하지 않는다.

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

이진 PixelCNN은 모든 위치의 베르누이 logits를 병렬로 학습하고 픽셀별로 표본을 생성한다. 손실의 픽셀당 단위·이진화·검증 분할을 맞추고, 고정 격자·프레임 수 같은 시각화 계약도 확인해야 한다. 생성 품질은 완료 문구나 훈련 손실과 별도로 평가한다.
