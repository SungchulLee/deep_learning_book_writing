# MNIST 데이터셋 시각화

MNIST는 28×28 회색조 손글씨 숫자 이미지의 학습셋 60000장과 테스트셋 10000장으로 구성된다. 이 페이지는 데이터 로딩, 텐서 모양과 전처리, 클래스 분포와 표본 시각화를 설명한다.

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
# 배치 수 x 배치 크기로 세면 안 된다 -- 938 x 64 = 60032 가 되어,
# 32장뿐인 마지막 배치를 64장으로 쳐서 32장을 더 세게 된다.
# 실제 장 수는 데이터셋에서 직접 읽는다.
print(f"  Total training images: {len(trainloader.dataset)}")

# 살펴볼 표본 배치 하나 가져오기
sample_images, sample_labels = next(iter(trainloader))
print("\nSample batch shape:")
print(f"  Images: {sample_images.shape}")  # (배치 크기, 채널, 높이, 너비)
print(f"  Labels: {sample_labels.shape}")  # (배치 크기,)
print("\nImage tensor details:")
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
        # matplotlib은 (H, W) 또는 (H, W, C)를 받는데 텐서는 (C, H, W)이다
        # 회색조는 C = 1 이므로 자리를 바꿀 것 없이 채널 축을 짜내면 (H, W)가 된다
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

**출력:**

```
======================================================================
MNIST Dataset Exploration
======================================================================

Configuration:
  Learning rate: 0.01
  Batch size: 64
  Test batch size: 1000
  Random seed: 1
  Device: mps

Random seed set to 1 for reproducibility

======================================================================
Loading MNIST Dataset
======================================================================

Dataset loaded successfully!
  Training batches: 938
  Test batches: 10
  Images per training batch: 64
  Total training images: 60000

Sample batch shape:
  Images: torch.Size([64, 1, 28, 28])
  Labels: torch.Size([64])

Image tensor details:
  - Batch size: 64
  - Channels: 1 (grayscale)
  - Height: 28 pixels
  - Width: 28 pixels

======================================================================
Visualizing MNIST Images
======================================================================

Displaying 64 sample images...
Each image shows a handwritten digit (0-9)
Close the plot window to continue.

======================================================================
Dataset Statistics
======================================================================

Class distribution in training set:
Digit      | Count      | Percentage
-----------------------------------
0          | 5923       | 9.87%
1          | 6742       | 11.24%
2          | 5958       | 9.93%
3          | 6131       | 10.22%
4          | 5842       | 9.74%
5          | 5421       | 9.04%
6          | 5918       | 9.86%
7          | 6265       | 10.44%
8          | 5851       | 9.75%
9          | 5949       | 9.91%

Total training samples: 60000
Note: The dataset is well-balanced across all digits!

======================================================================
Understanding Pixel Values
======================================================================

Sample image statistics:
  Shape: (28, 28)
  Data type: float32
  Min value: -1.0000
  Max value: 1.0000
  Mean value: -0.8363
  Std deviation: 0.4804

Pixel value interpretation:
  - Images are normalized to range [-1, 1]
  - -1.0 represents black pixels
  - +1.0 represents white pixels
  - Values in between represent shades of gray
  - This normalization helps neural networks train better!
```

## 2. 논의

변환 후 이미지 하나는 $(1,28,28)$이고 배치는 $(B,1,28,28)$이다. 클래스별 표본 수는 비슷하지만 정확히 같지는 않다. 기본 DataLoader는 마지막 작은 배치를 유지하므로 표본 수를 배치 수×설정 배치 크기로 계산하면 과대 추정할 수 있다. 예를 들어 60000장을 배치 64로 읽으면 938개 배치 중 마지막은 32장이다. len(loader.dataset)은 데이터셋 크기를 나타내며 사용자 정의 샘플러나 drop_last를 쓰면 실제 처리 표본 수와도 구분해야 한다.

MNIST의 원본 8비트 이미지는 ToTensor()로 $[0,1]$ 범위의 텐서가 되고 Normalize((0.5,),(0.5,))를 거치면 $x\mapsto2x-1$에 따라 $[-1,1]$로 변환된다. 이 범위 변환이 데이터의 실제 평균을 0, 분산을 1로 만드는 것은 아니다. 전처리는 모델과 함께 일관되게 사용하고, 선택의 효과는 검증 성능으로 판단한다.

DataLoader는 배치 구성과 셔플, 선택적인 병렬 로딩을 담당한다. 셔플은 배치 구성의 순서 의존성을 줄일 수 있지만 각 배치의 클래스 균형을 보장하지 않는다. 전체 정확도와 함께 클래스별 재현율이나 혼동 행렬을 확인하면 어떤 숫자를 혼동하는지 알 수 있다.

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
- `train=True` / `train=False` -- 같은 함수로 학습셋(6만 장)과 테스트셋(1만 장)을 갈라 받는다.
- `transform` -- 이미지를 텐서로 바꾸고 정규화하는 절차를 하나의 object로 얹는다.
- `batch_size` — 기본 배치 구성에서 한 번에 반환할 표본 수이다. GPU 이동은 학습 루프에서 별도로 처리한다. 2의 거듭제곱일 필요는 없으며 메모리·처리량과 검증 성능에 맞춰 선택한다. [GPU 연산과 장치 관리](../../ch06/tensors/23_gpu_operations.md)도 참고한다.

### 기준선으로서의 82%

간단한 기준선은 학습 이미지의 클래스별 평균을 템플릿으로 저장하고 가장 가까운 템플릿의 숫자를 예측하는 방법이다. 역전파는 사용하지 않지만 레이블이 붙은 학습 데이터에서 대표값을 추정하므로 넓은 의미의 학습에 해당한다.

$$
\hat{y} = \operatorname*{arg\,min}_{k \in \{0, \ldots, 9\}} \lVert x - \bar{x}_k \rVert^2
$$

[템플릿 학습](../../ch03/mnist/01_template_learning.md)에 기록된 테스트 정확도는 82.03%이다. 해당 거리와 전처리에서 얻은 참고 기준선이며 모든 구성에서 반드시 넘어야 하는 이론적 한계는 아니다. CNN의 효과를 비교하려면 같은 분할과 평가 조건을 유지한다.

전체 정확도는 클래스별 차이를 감춘다. 연결된 템플릿 예제에서는 숫자 1의 재현율이 96.2%, 5는 68.6%이며 일부 클래스 쌍의 혼동이 크다. 이런 차이는 평균 템플릿과의 거리로 관찰한 결과이며, 획 수 같은 한 가지 시각적 이유로 원인을 확정할 수는 없다.

클래스별 템플릿 하나는 다양한 필체를 충분히 표현하지 못할 수 있고 픽셀 위치별 거리는 이동·회전에 민감하다. CNN은 국소 특징과 계층적 조합을 학습하는 다른 표현을 제공하지만, 이동과 회전에 자동으로 완전한 불변성을 갖는 것은 아니다.

실행 가능한 코드와 자세한 분석(선형 분류기와의 동치 관계, 클래스별 정확도 표, 표본 수에 따른 곡선)은 [3.1 템플릿 학습](../../ch03/mnist/01_template_learning.md)에 있다. 그 절을 시작으로 3장은 템플릿 학습(3.1) → 선형 모델과 소프트맥스(3.2) → 다층 퍼셉트론(3.3) → 합성곱 신경망(3.4)의 네 걸음을 같은 데이터셋 위에서 견준다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
5절은 표본 이미지 한 장의 평균이 $-0.8363$이라고 찍는다. 정규화하기 **전**의 원래 픽셀 평균은 얼마였는가? 그 값이 0.5보다 훨씬 작은 까닭도 한 줄로 적어라.

</div>

??? success "연습문제 1 풀이"
    정규화는 $x' = (x - 0.5)/0.5 = 2x - 1$이므로 되돌리는 식은 $x = (x' + 1)/2$이다. 따라서 원래 평균은

    $$
    x = \frac{-0.8363 + 1}{2} = 0.0819
    $$

    이다. MNIST 이미지는 28×28 = 784픽셀 가운데 대부분이 배경(검정, 값 0)이고 획이 지나가는 픽셀만 1에 가깝다. 그래서 평균이 0에 훨씬 가깝게 쏠린다. 데이터셋 전체로 재면 평균이 0.1307(연습문제 4 참조)이고, 정규화 뒤에는 $2 \times 0.1307 - 1 = -0.7386$이 된다. 표본 한 장의 $-0.8363$은 이 값 둘레에서 흔들리는 계산이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
배치 크기를 64에서 256으로 바꾸고 학습 배치가 몇 개 나오는지 계산하라. 학습셋에 이미지가 6만 장 있을 때 배치의 수가 정확히 $60000 / 256$이 아닐 수 있는 까닭을 설명하라.

</div>

??? success "연습문제 2 풀이"
    배치 크기가 256이고 학습 이미지가 6만 장이면 꽉 찬 배치가 $\lfloor 60000 / 256 \rfloor = 234$개 나오고 이미지 $60000 - 234 \times 256 = 96$장이 남는다. PyTorch `DataLoader`는 기본적으로 그 96장으로 마지막 작은 배치를 만들어 모두 235개가 된다. `drop_last=True`를 주면 마지막 미완성 배치를 버려 정확히 234개가 된다.

    본문의 코드가 전체 장 수를 `len(trainloader) * cfg.batch_size`가 아니라 `len(trainloader.dataset)`으로 세는 까닭이 여기에 있다. 배치 크기 64에서 앞의 식은 $938 \times 64 = 60032$를 주는데, 마지막 배치는 32장뿐이므로 32장을 더 센 값이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
Normalize((0.5,),(0.5,))의 식과 범위를 설명하고 평균 0을 보장하는지도 답하라.

</div>

??? success "연습문제 3 풀이"
    $x\in[0,1]$에 대해 $x\prime=(x-0.5)/0.5=2x-1$이므로 범위는 $[-1,1]$이다. 실제 입력 평균이 0.5일 때만 변환 후 평균이 0이다. 평균·분산으로 표준화하는 것과 단순한 범위 변환을 구분한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
MNIST 학습셋 전체 픽셀의 평균과 표준편차를 (0.5라는 박아 넣은 값을 쓰지 않고) 계산하는 코드를 작성하라. 더 엄밀한 정규화 방식에서 이 통계량을 어떻게 쓸지 논하라.

</div>

??? success "연습문제 4 풀이"
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
    학습셋의 [0,1] 픽셀 전체에서 계산하는 평균과 표준편차는 약 0.1307과 0.3081이다. 이는 표본 집계 방식과 분산 규약에 따른 경험적 통계량이다. 학습셋에서 정한 값을 고정해 검증·테스트에도 적용하면 학습셋 기준의 표준화를 할 수 있다. 이 방식이 0.5를 이용한 범위 변환보다 모든 모델에서 더 좋은 성능을 보장하지는 않는다.

---

## 정리하며

MNIST에서 데이터셋 크기, 배치 수와 실제 처리 표본 수를 구분한다. 전처리 후 텐서는 배치×1×28×28이고 범위 변환과 통계적 표준화는 다르다. 전체 정확도와 클래스별 결과를 함께 보고, 템플릿 분류기는 학습 데이터에서 추정한 간단한 기준선으로 해석한다.
