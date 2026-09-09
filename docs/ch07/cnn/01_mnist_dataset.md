# MNIST 데이터셋 시각화

MNIST 데이터셋은 손으로 쓴 숫자의 회색조 이미지 7만 장으로 이루어진, 기계 학습에서 가장 바탕이 되는 표준 자료 가운데 하나이다. 이 데이터셋을 불러오고 살피고 그려 보는 법을 익히는 것이 이미지 분류용 합성곱 신경망을 세우는 첫걸음이다. 28×28 화소 이미지와 잘 정의된 10개 부류라는 MNIST의 단순함은 PyTorch 데이터 파이프라인을 배우기에 알맞은 출발점이 되어 준다.

## 1. 코드

```python
"""
01_mnist_dataset.py
===================
MNIST 데이터셋 시각화

첫 CNN 실습이다! 손으로 쓴 숫자로 이루어진 이름난 MNIST 데이터셋을 살펴본다.
이 데이터셋이 배우기에 알맞은 까닭은 다음과 같다.
- 단순한 회색조 이미지 (화소 28x28)
- 뚜렷하고 잘 정의된 부류 (숫자 0-9)
- 빨리 학습시킬 만큼 작다
- 뜻있는 무늬를 배울 만큼 크다

배울 내용:
- PyTorch로 데이터셋 불러오기
- 데이터의 모양과 짜임 이해하기
- 이미지 데이터 그려 보기
- DataLoader와 배치 다루기

난이도: 쉬움
예상 시간: 30분

지은이: PyTorch CNN 실습
날짜: 2025년 11월
"""

import matplotlib.pyplot as plt
import cnn_utils as utils

# =============================================================================
# 1절: 설정과 준비
# =============================================================================

print("=" * 70)
print("MNIST Dataset Exploration")
print("=" * 70)

# 명령줄 인자 구문 분석
# 기본 설정(배치 크기, 씨앗 등)을 불러온다
cfg = utils.parse_args()

print("\nConfiguration:")
print(f"  Learning rate: {cfg.lr}")
print(f"  Batch size: {cfg.batch_size}")
print(f"  Test batch size: {cfg.test_batch_size}")
print(f"  Random seed: {cfg.seed}")
print(f"  Device: {cfg.device}")

# 재현성을 위해 난수 씨앗 고정
# 실행할 때마다 같은 결과가 나오도록 한다
utils.set_seed(seed=cfg.seed)
print(f"\nRandom seed set to {cfg.seed} for reproducibility")

# =============================================================================
# 2절: 데이터 적재
# =============================================================================

print("\n" + "=" * 70)
print("Loading MNIST Dataset")
print("=" * 70)

# DataLoader 매개변수 설정
train_kwargs = {'batch_size': cfg.batch_size, 'shuffle': True}
test_kwargs = {'batch_size': cfg.test_batch_size, 'shuffle': False}

# MNIST 데이터셋 불러오기
# - 학습 집합: 이미지 60,000장
# - 시험 집합: 이미지 10,000장
# - 이미지마다: 28x28 회색조 (채널 1개)
# - 부류: 10개 (숫자 0-9)
trainloader, testloader = utils.load_data(train_kwargs, test_kwargs)

print("\nDataset loaded successfully!")
print(f"  Training batches: {len(trainloader)}")
print(f"  Test batches: {len(testloader)}")
print(f"  Images per training batch: {cfg.batch_size}")
print(f"  Total training images: {len(trainloader) * cfg.batch_size}")

# 살펴볼 표본 배치 하나 가져오기
sample_images, sample_labels = next(iter(trainloader))
print(f"\nSample batch shape:")
print(f"  Images: {sample_images.shape}")  # (배치 크기, 채널, 높이, 너비)
print(f"  Labels: {sample_labels.shape}")  # (배치 크기,)
print(f"\nImage tensor details:")
print(f"  - Batch size: {sample_images.shape[0]}")
print(f"  - Channels: {sample_images.shape[1]} (grayscale)")
print(f"  - Height: {sample_images.shape[2]} pixels")
print(f"  - Width: {sample_images.shape[3]} pixels")

# =============================================================================
# 3절: 데이터 시각화
# =============================================================================

print("\n" + "=" * 70)
print("Visualizing MNIST Images")
print("=" * 70)

# 이미지 64장을 보이려고 8x8 격자 만들기
fig, axes = plt.subplots(8, 8, figsize=(10, 10))
fig.suptitle('MNIST Handwritten Digits Sample', fontsize=16)

# 이미지 배치 하나 가져오기
for images, labels in trainloader:
    # 격자를 훑으며 이미지 채우기
    for ax, image, label in zip(axes.reshape(-1), images, labels):
        # matplotlib을 위해 (C, H, W)에서 (H, W, C)로 바꾸기
        # 회색조에서는 채널 차원을 짜내면 된다
        img_display = image.squeeze().cpu().numpy()
        
        # 더 잘 보이도록 [-1, 1]에서 [0, 1]로 되돌리기
        img_display = img_display / 2 + 0.5
        
        # 이미지 보이기
        ax.imshow(img_display, cmap="gray")
        ax.axis("off")
        ax.set_title(f"{label.item()}", fontsize=10)
    
    # 첫 배치만 처리
    break

plt.tight_layout()
print("\nDisplaying 64 sample images...")
print("Each image shows a handwritten digit (0-9)")
print("Close the plot window to continue.")
plt.show()

# =============================================================================
# 4절: 데이터 분석
# =============================================================================

print("\n" + "=" * 70)
print("Dataset Statistics")
print("=" * 70)

# 학습 집합의 이름표 분포 분석
label_counts = [0] * 10
for _, labels in trainloader:
    for label in labels:
        label_counts[label.item()] += 1

print("\nClass distribution in training set:")
print(f"{'Digit':<10} | {'Count':<10} | {'Percentage':<10}")
print("-" * 35)
total = sum(label_counts)
for digit, count in enumerate(label_counts):
    percentage = 100.0 * count / total
    print(f"{digit:<10} | {count:<10} | {percentage:.2f}%")

print(f"\nTotal training samples: {total}")
print("Note: The dataset is well-balanced across all digits!")

# =============================================================================
# 5절: 화소 값 이해하기
# =============================================================================

print("\n" + "=" * 70)
print("Understanding Pixel Values")
print("=" * 70)

# 분석할 이미지 하나 가져오기
sample_image = sample_images[0].squeeze().cpu().numpy()

print("\nSample image statistics:")
print(f"  Shape: {sample_image.shape}")
print(f"  Data type: {sample_image.dtype}")
print(f"  Min value: {sample_image.min():.4f}")
print(f"  Max value: {sample_image.max():.4f}")
print(f"  Mean value: {sample_image.mean():.4f}")
print(f"  Std deviation: {sample_image.std():.4f}")

print("\nPixel value interpretation:")
print("  - Images are normalized to range [-1, 1]")
print("  - -1.0 represents black pixels")
print("  - +1.0 represents white pixels")
print("  - Values in between represent shades of gray")
print("  - This normalization helps neural networks train better!")


if __name__ == "__main__":
    pass
```

## 2. 논의

MNIST 데이터셋은 딥러닝의 "Hello World" 노릇을 한다. 이미지 하나하나가 단일 채널 텐서로 저장된 $28 \times 28$ 회색조 이미지여서 표본마다 모양이 $(1, 28, 28)$이다. PyTorch `DataLoader`로 불러오면 이미지가 모양 $(B, 1, 28, 28)$의 텐서로 묶이며 여기서 $B$은 배치 크기이다. 데이터셋에는 학습 이미지 6만 장과 시험 이미지 1만 장이 있고 숫자 10개 부류에 고르게 나뉘어 있다.

아주 중요한 전처리 단계가 정규화이다. 범위가 $[0, 255]$인 원래 화소 값을 `ToTensor()`이 먼저 $[0, 1]$으로 바꾸고, 평균 0.5와 표준편차 0.5를 쓰는 `Normalize` 변환이 다시 $[-1, 1]$으로 옮긴다. 이렇게 0을 중심으로 놓으면 입력 분포의 큰 양의 치우침을 신경망의 가중치가 메울 필요가 없어져 기울기 기반 최적화가 더 빨리 수렴한다.

`DataLoader`이라는 추상은 배치 묶기, 섞기, 병렬 적재를 맡는다. 학습 데이터를 섞는 일은 모델이 예제의 순서에서 비롯한 헛된 규칙을 배우지 않게 하는 데 중요하다. MNIST는 부류의 분포가 고르므로(숫자마다 표본 약 6000개) 부류별 가중치를 손보지 않아도 단순한 정확도만으로 뜻이 통한다.

### 데이터셋을 부르는 두 줄 읽기

실제로 쓰는 것은 결국 이 두 줄이다. 인자 하나하나가 무엇을 정하는지 짚어 둘 값어치가 있다.

```python
train_dataset = datasets.MNIST(
    root="./data",          # 저장 경로 지정
    train=True,             # 학습 집합 받기 (False로 두면 시험 집합)
    download=True,          # 없으면 내려받기
    transform=transform,    # 전처리 object 얹기
)

train_loader = DataLoader(
    train_dataset,
    batch_size=1024,        # 한 번에 넘길 장 수
    shuffle=False,
)
```

- `root` -- 데이터셋을 어디에 저장할지 정한다. 처음 한 번만 내려받고 다음부터는 이 경로에서 읽는다.
- `train=True` / `train=False` -- 같은 함수로 학습 집합(6만 장)과 시험 집합(1만 장)을 갈라 받는다.
- `transform` -- 이미지를 텐서로 바꾸고 정규화하는 절차를 하나의 object로 얹는다.
- `batch_size` -- 한 번에 GPU로 넘길 장 수이다. $1024 = 2^{10}$처럼 2의 거듭제곱으로 잡는 것이 관행인데, 그 까닭은 [GPU 연산 - CUDA와 장치 관리](../../ch03/tensors/23_gpu_operations.md)에서 설명한다.

### 밑금으로서의 82%

이 데이터셋에는 학습을 전혀 하지 않고도 세울 수 있는 밑금이 있다. 숫자마다 학습 이미지를 모두 평균 내어 판(template) 열 장을 만들고, 새 이미지를 가장 가까운 판의 숫자로 부르는 것이다. 역전파도, 하이퍼파라미터 조정도 없다.

$$
\hat{y} = \operatorname*{arg\,min}_{k \in \{0, \ldots, 9\}} \lVert x - \bar{x}_k \rVert^2
$$

**시험 정확도 82.03%.** 평균을 내는 일뿐이고 몇 초면 끝나는데도 열 개 중 여덟 개를 맞힌다. 이 장의 합성곱 신경망은 이 82%를 넘어야 값어치가 있는 것이며, 그래서 이 수치를 먼저 손에 쥐고 가는 편이 좋다.

전체 82%는 클래스마다 사정이 아주 다르다는 것을 감춘다. `1`은 96.2%로 가장 잘 맞고(획이 하나뿐이라 필체가 달라져도 평균에서 크게 벗어나지 않는다), `5`는 68.6%로 가장 나쁘며 오답의 대부분이 `3`으로 쏠린다(5 → 3이 118건). `4`와 `9`도 서로를 오간다(116건과 83건).

여기에 이 방법의 한계가 그대로 드러난다. 평균 이미지는 그 클래스의 모든 필체를 한 장으로 뭉갠 것이라, 같은 숫자를 쓰는 방식이 여럿일 때 어느 쪽과도 멀어진다. 또 화소를 자리 그대로 견주므로 숫자가 조금만 옆으로 밀리거나 기울어도 거리가 크게 늘어난다. **합성곱 신경망이 무엇을 더 해 주는지가 바로 이 두 지점이다.**

실행 가능한 코드와 자세한 분석(선형 분류기와의 동치 관계, 클래스별 정확도 표, 표본 수에 따른 곡선)은 [2.1 템플릿 학습](../../ch02/mnist/01_template_learning.md)에 있다. 그 절을 시작으로 2장은 템플릿 학습 → 선형 모델 → 다층 퍼셉트론 → 합성곱 신경망의 네 걸음을 같은 데이터셋 위에서 견준다.

## 연습문제

**연습문제 1.**
배치 크기를 64에서 256으로 바꾸고 학습 배치가 몇 개 나오는지 계산하라. 학습 집합에 이미지가 6만 장 있을 때 배치의 수가 정확히 $60000 / 256$이 아닐 수 있는 까닭을 설명하라.

??? success "연습문제 1 풀이"
    배치 크기가 256이고 학습 이미지가 6만 장이면 꽉 찬 배치가 $\lfloor 60000 / 256 \rfloor = 234$개 나오고 이미지 $60000 - 234 \times 256 = 96$장이 남는다. PyTorch `DataLoader`은 기본적으로 그 96장으로 마지막 작은 배치를 만들어 모두 235개가 된다. `drop_last=True`을 주면 마지막 미완성 배치를 버려 정확히 234개가 된다.

---

**연습문제 2.**
정규화 변환 `Normalize((0.5,), (0.5,))`이 화소 값을 $[0, 1]$에서 $[-1, 1]$으로 옮기는 까닭을 설명하라. 그 변환의 식을 적어라.

??? success "연습문제 2 풀이"
    `Normalize` 변환은 $\mu = 0.5$, $\sigma = 0.5$일 때 식 $x' = (x - \mu) / \sigma$을 적용한다. 입력의 최솟값 $x = 0$에서는 $x' = (0 - 0.5) / 0.5 = -1$이고, 최댓값 $x = 1$에서는 $x' = (1 - 0.5) / 0.5 = 1$이다. 따라서 출력 범위가 $[-1, 1]$이다. 이는 데이터를 0 둘레로 모으고 크기를 1 안팎으로 맞추는 선형 사상인데, 처음의 무작위 가중치도 대개 0을 중심으로 놓이므로 신경망 학습에 도움이 된다.

---

**연습문제 3.**
MNIST 학습 집합 전체에 대해 화소별 평균과 표준편차를 (0.5라는 박아 넣은 값을 쓰지 않고) 계산하는 코드를 작성하라. 더 엄밀한 정규화 방식에서 이 통계량을 어떻게 쓸지 논하라.

??? success "연습문제 3 풀이"
    ```python
    import torch
    from torchvision import datasets, transforms

    dataset = datasets.MNIST(root='./data', train=True,
                             transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(dataset, batch_size=1000)

    mean = 0.0
    sq_mean = 0.0
    count = 0
    for images, _ in loader:
        mean += images.sum()
        sq_mean += (images ** 2).sum()
        count += images.numel()

    mean /= count
    std = (sq_mean / count - mean ** 2).sqrt()
    print(f"Mean: {mean:.4f}, Std: {std:.4f}")
    ```
    MNIST의 참된 통계량은 평균이 약 0.1307, 표준편차가 약 0.3081이다. 뭉뚱그린 0.5 대신 이 값을 쓰면 평균이 0이고 분산이 1인 분포가 되는데, 이것이 더 꼼꼼한 정규화의 표준 관행이다. 0.5라는 값은 간단하지만 참으로 표준화된 분포를 주지는 못한다.

## 정리하며

**다룬 것** — MNIST 데이터셋 시각화

MNIST 데이터셋은 딥러닝의 "Hello World" 노릇을 한다.

앞의 연습문제 3개로 직접 확인할 수 있다.
