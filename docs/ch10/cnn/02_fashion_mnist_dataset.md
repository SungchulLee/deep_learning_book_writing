# Fashion-MNIST 데이터셋 시각화

Fashion-MNIST는 고전적인 MNIST 데이터셋을 그대로 갈아 끼울 수 있는 요즘 판본으로, 손으로 쓴 숫자 대신 옷가지의 회색조 이미지를 쓴다. Zalando Research가 28×28 회색조 이미지와 10개 부류라는 편리한 형식을 지키면서도 더 까다로운 표준 자료를 주려고 만들었다. 앞 쪽 [MNIST 데이터셋](01_mnist_dataset.md)과 짜임이 똑같으므로, 두 쪽을 나란히 놓고 읽으면 데이터셋의 복잡함이 모델 성능에 어떻게 영향을 주는지 알 수 있다.

## 1. 코드

```python
"""
02_fashion_mnist_dataset.py
============================
Fashion-MNIST 데이터셋 시각화

Fashion-MNIST는 MNIST를 대신하는 더 까다로운 요즘 데이터셋이다!
손으로 쓴 숫자 대신 옷가지의 회색조 이미지가 들어 있다.

왜 Fashion-MNIST인가?
- MNIST와 형식이 같다 (28x28 회색조)
- 더 까다롭고 현실적이다
- 모델의 일반화를 시험하기에 낫다
- 요즘 연구와 교육에서 쓰인다

배울 내용:
- 형식이 같은 여러 데이터셋 다루기
- 숫자가 아닌 데이터의 부류 이름표 이해하기
- 데이터셋의 난이도 견주기
- 뜻있는 이름이 붙은 범주형 데이터 다루기

난이도: 쉬움
예상 시간: 30분

지은이: PyTorch CNN 실습
날짜: 2025년 11월
"""

import matplotlib.pyplot as plt
import cnn_utils as utils

# =============================================================================
# 1절: Fashion-MNIST 부류 이름표
# =============================================================================

print("=" * 70)
print("Fashion-MNIST Dataset Exploration")
print("=" * 70)

FASHION_MNIST_LABELS = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

print("\nFashion-MNIST Classes:")
print("-" * 40)
for idx, name in FASHION_MNIST_LABELS.items():
    print(f"  Class {idx}: {name}")

# =============================================================================
# 2절: 설정과 데이터 적재
# =============================================================================

cfg = utils.parse_args()
utils.set_seed(seed=cfg.seed)

train_kwargs = {'batch_size': cfg.batch_size, 'shuffle': True}
test_kwargs = {'batch_size': cfg.test_batch_size, 'shuffle': False}

# MNIST를 부르던 줄에 fashion_mnist=True 하나만 더 준 것이다.
# 뒤의 코드는 한 줄도 고치지 않는다.
# - 학습 집합: 이미지 60,000장 (부류마다 6,000장)
# - 시험 집합: 이미지 10,000장 (부류마다 1,000장)
# - 이미지마다: 28x28 회색조 (채널 1개)
trainloader, testloader = utils.load_data(
    train_kwargs, test_kwargs, fashion_mnist=True
)

print("\nDataset loaded successfully!")
# 배치 수 x 배치 크기로 세면 안 된다 -- 938 x 64 = 60032 가 되어,
# 32장뿐인 마지막 배치를 64장으로 쳐서 32장을 더 세게 된다.
# 실제 장 수는 데이터셋에서 직접 읽는다.
print(f"  Training images: {len(trainloader.dataset)}")
print(f"  Test images: {len(testloader.dataset)}")
print(f"  Training batches: {len(trainloader)}")

# 살펴볼 표본 배치 하나 가져오기
sample_images, sample_labels = next(iter(trainloader))
print("\nSample batch shape:")
print(f"  Images: {sample_images.shape}")  # (배치 크기, 채널, 높이, 너비)
print(f"  Labels: {sample_labels.shape}")  # (배치 크기,)

# =============================================================================
# 3절: 시각화
# =============================================================================

fig, axes = plt.subplots(8, 8, figsize=(12, 12))
fig.suptitle('Fashion-MNIST Clothing Items Sample', fontsize=16)

for images, labels in trainloader:
    for ax, image, label in zip(axes.reshape(-1), images, labels):
        # matplotlib은 (H, W) 또는 (H, W, C)를 받는데 텐서는 (C, H, W)이다
        # 회색조는 C = 1 이므로 자리를 바꿀 것 없이 채널 축을 짜내면 (H, W)가 된다
        img_display = image.squeeze().cpu().numpy()

        # 더 잘 보이도록 [-1, 1]에서 [0, 1]로 되돌리기
        img_display = img_display / 2 + 0.5

        ax.imshow(img_display, cmap="gray")
        ax.axis("off")
        class_name = FASHION_MNIST_LABELS[label.item()]
        ax.set_title(class_name, fontsize=8)

    # 첫 배치만 처리
    break

plt.tight_layout()
plt.show()

# =============================================================================
# 4절: 데이터셋 통계
# =============================================================================

class_counts = [0] * 10
for _, labels in trainloader:
    for label in labels:
        class_counts[label.item()] += 1

print("\nClass distribution:")
total = sum(class_counts)
for class_id, count in enumerate(class_counts):
    percentage = 100.0 * count / total
    name = FASHION_MNIST_LABELS[class_id]
    print(f"  {name:15s} (Class {class_id}): {count} ({percentage:.2f}%)")


if __name__ == "__main__":
    pass
```

**출력:**

```
======================================================================
Fashion-MNIST Dataset Exploration
======================================================================

Fashion-MNIST Classes:
----------------------------------------
  Class 0: T-shirt/top
  Class 1: Trouser
  Class 2: Pullover
  Class 3: Dress
  Class 4: Coat
  Class 5: Sandal
  Class 6: Shirt
  Class 7: Sneaker
  Class 8: Bag
  Class 9: Ankle boot

Dataset loaded successfully!
  Training images: 60000
  Test images: 10000
  Training batches: 938

Sample batch shape:
  Images: torch.Size([64, 1, 28, 28])
  Labels: torch.Size([64])

Class distribution:
  T-shirt/top     (Class 0): 6000 (10.00%)
  Trouser         (Class 1): 6000 (10.00%)
  Pullover        (Class 2): 6000 (10.00%)
  Dress           (Class 3): 6000 (10.00%)
  Coat            (Class 4): 6000 (10.00%)
  Sandal          (Class 5): 6000 (10.00%)
  Shirt           (Class 6): 6000 (10.00%)
  Sneaker         (Class 7): 6000 (10.00%)
  Bag             (Class 8): 6000 (10.00%)
  Ankle boot      (Class 9): 6000 (10.00%)
```

!!! note "처음 돌릴 때만 나오는 줄"
    데이터가 아직 `./data`에 없으면 위 출력의 부류 목록과 `Dataset loaded successfully!` 사이에 `Downloading ...` / `Extracting ...` 열두 줄이 더 찍힌다. 내려받기가 한 번 끝나면 다시 나오지 않으므로 여기에는 싣지 않았다.

## 2. 논의

Fashion-MNIST는 MNIST와 텐서의 짜임이 같아 이미지마다 단일 채널 $28 \times 28$ 회색조이지만, 시각적으로는 훨씬 복잡하다. 손으로 쓴 숫자는 획의 모양이 뚜렷한 반면 옷가지는 같은 부류 안의 변화가 크다. 티셔츠 하나만 해도 방향과 질감과 맵시가 여러 가지이다. 그래서 Fashion-MNIST가 실제 분류의 어려움을 더 잘 흉내 낸다.

가장 헷갈리는 부류 쌍은 분류의 중요한 개념인 부류 사이의 유사성을 잘 보여 준다. 티셔츠(0번 부류)와 셔츠(6번 부류)는 윤곽이 비슷하고, 풀오버(2번)와 코트(4번)도 그렇다. 이 책이 직접 잰 값으로 견주면 차이가 뚜렷하다. 합성곱 신경망은 숫자 MNIST에서 시험 정확도 99.21%를 내지만([3.4 MNIST 합성곱 신경망](../../ch03/mnist/04_cnn.md)), 이 절과 같은 CNN을 Fashion-MNIST에 학습시킨 [Fashion-MNIST 분류기](05_fashion_mnist_classifier.md)는 85.84%에 그친다. 두 실험은 최적화기와 에포크 수가 달라 소수점까지 맞대 놓을 것은 아니지만, 13%p에 이르는 이 간격을 만든 것은 구조가 아니라 데이터셋이다. 이룰 수 있는 성능을 데이터셋의 성질이 근본적으로 정한다는 뜻이다.

실용적인 면에서 Fashion-MNIST는 MNIST와 똑같은 PyTorch API를 쓴다. 2절에서 바뀐 것은 `utils.load_data(...)` 호출에 `fashion_mnist=True` 하나를 더 준 것뿐이고, 그 안에서 `datasets.MNIST`가 `datasets.FashionMNIST`로 갈릴 뿐 변환도 `DataLoader`도 그대로다. 이는 표준화된 데이터셋 인터페이스의 힘을 보여 준다. 연구자가 학습 코드를 고치지 않고 표준 자료를 갈아 끼울 수 있어 방법 사이의 공정한 비교가 가능해진다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
4절의 출력은 열 부류가 모두 정확히 6000장, 10.00%라고 찍는다. 앞 쪽 [MNIST 데이터셋](01_mnist_dataset.md)의 같은 표는 5421에서 6742까지 흔들린다. 두 데이터셋이 이렇게 다른 까닭은 무엇인가? 아무것도 배우지 않고 한 부류만 찍는 분류기의 정확도를 두 데이터셋에서 각각 계산해 보고, 시험 정확도 하나로 모델을 재는 일에 이 차이가 어떤 뜻을 갖는지 적어라.

</div>

??? success "연습문제 1 풀이"
    MNIST는 이미 있던 손글씨 표본을 모아 놓은 것이라 부류의 크기가 저절로 정해졌고, 그래서 대략 고를 뿐 정확히 고르지는 않다. Fashion-MNIST는 Zalando가 **일부러** 부류마다 같은 장 수를 뽑아 만든 것이라 학습 6000장, 시험 1000장이 부류마다 딱 맞는다.

    아무것도 배우지 않고 늘 같은 부류만 찍는 분류기를 시험 집합에서 재면 이렇게 된다.

    - Fashion-MNIST: 부류마다 1000장이므로 어느 부류를 찍든 $1000 / 10000 = 10.00\%$이다.
    - MNIST: 가장 흔한 `1`을 찍으면 $1135 / 10000 = 11.35\%$, 가장 드문 `5`를 찍으면 $892 / 10000 = 8.92\%$이다.

    Fashion-MNIST에서는 분포가 고르므로 정확도 하나만 보아도 곧바로 뜻이 통한다. 10%가 바닥이고, 부류마다 같은 무게로 채점되기 때문이다. 반면 분포가 기울어진 데이터셋에서는 정확도가 무엇을 잘 맞히는지 감춘다. 흔한 부류만 잘 맞혀도 수가 올라가므로 부류별 정확도나 혼동 행렬을 함께 보아야 한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
Fashion-MNIST의 10개 부류마다 평균 화소 밝기를 계산하라. 평균 밝기가 가장 높은 부류와 가장 낮은 부류는 무엇인가? 그것이 물리적으로 말이 되는 까닭을 설명하라.

</div>

??? success "연습문제 2 풀이"
    ```python
    class_pixel_sums = [0.0] * 10
    class_counts = [0] * 10
    for images, labels in trainloader:
        for img, lbl in zip(images, labels):
            class_pixel_sums[lbl.item()] += img.mean().item()
            class_counts[lbl.item()] += 1
    for i in range(10):
        avg = class_pixel_sums[i] / class_counts[i]
        print(f"{FASHION_MNIST_LABELS[i]:15s}: {avg:.4f}")
    ```

    ```
    T-shirt/top    : -0.3488
    Trouser        : -0.5542
    Pullover       : -0.2466
    Dress          : -0.4822
    Coat           : -0.2293
    Sandal         : -0.7265
    Shirt          : -0.3364
    Sneaker        : -0.6646
    Bag            : -0.2929
    Ankle boot     : -0.3976
    ```

    이 수들은 정규화를 거친 뒤의 값이라 $[-1, 1]$에 놓인다. $-1$이 검은 배경이고 $+1$이 흰 옷감이므로, 값이 $-1$에 가까울수록 배경 화소가 많다는 뜻이다. 원래 $[0, 1]$ 눈금으로 되돌리려면 $(x' + 1)/2$를 쓰면 된다.

    가장 밝은 것은 코트(4번 부류)가 $-0.2293$이고 풀오버(2번 부류)가 $-0.2466$으로 바로 뒤를 따른다. 가장 어두운 것은 샌들(5번 부류)로 $-0.7265$이고 운동화(7번 부류)가 $-0.6646$으로 그다음이다. 밝기의 차례는 결국 **그 물건이 28×28 화면을 얼마나 덮는가**를 그대로 잰 것이다. 코트와 풀오버는 소매를 펼친 채 화면을 거의 가득 채우고, 샌들은 작고 앞이 트인 물건이라 배경이 대부분이다.

    바지(1번 부류)가 $-0.5542$로 어두운 쪽에 붙는다는 점이 이 읽기를 확인해 준다. 바지는 길이는 길지만 폭이 좁은 세로 모양이라, 세로로는 화면을 다 쓰면서도 가로로는 많이 비운다. 옷감이 이미지의 넓은 부분을 덮으리라는 짐작과는 반대이며, 실제로 바지보다 어두운 부류는 신발 둘뿐이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
Fashion-MNIST 분류기의 혼동 행렬에서 티셔츠/윗옷(0)과 셔츠(6) 사이의 비대각 성분이 큰 까닭을 개념적으로 설명하라. CNN이 둘을 가르려면 어떤 시각적 특징을 배울 수 있겠는가?

</div>

??? success "연습문제 3 풀이"
    티셔츠와 셔츠는 전체 모양이 비슷하다. 둘 다 소매와 몸통이 있는 윗옷이다. 혼동이 생기는 까닭은 $28 \times 28$ 해상도에서는 깃의 모양, 단추의 자리, 소매의 길이 같은 잔 세부를 가려내기 어렵기 때문이다. 연습문제 2의 표도 이 짐작과 들어맞는다. 티셔츠가 $-0.3488$, 셔츠가 $-0.3364$로 평균 밝기가 0.012밖에 차이 나지 않아, 화면을 덮는 넓이만으로는 둘을 가를 수 없다.

    CNN은 깃의 모양(티셔츠는 대개 둥근 목선이고 셔츠는 깃이 있다), 소매의 비율, 셔츠 앞섶이 트인 무늬 같은 특징을 잡아 둘을 가를 수 있다. 필터가 많은 더 깊은 신경망이 이런 미묘한 질감과 모양의 차이를 더 잘 붙잡는다. [Fashion-MNIST 분류기](05_fashion_mnist_classifier.md)가 이 쌍을 실제 모델에서 다시 다룬다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
데이터 증강(무작위 좌우 뒤집기와 작은 회전)이 Fashion-MNIST의 정확도를 높이는지 시험할 실험을 설계하라. 고친 변환 파이프라인을 적고, 좌우 뒤집기가 안전한 부류와 그렇지 않은 부류를 수로 가려내라.

</div>

??? success "연습문제 4 풀이"
    ```python
    from torchvision import transforms

    augmented_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    ```

    기하 변환 둘은 PIL 이미지 위에서 도는 것이므로 `ToTensor()` **앞**에 두어야 한다. 그리고 평가의 일관성을 지키려면 증강 변환은 시험 집합이 아니라 학습 집합에만 적용해야 한다.

    좌우 뒤집기가 안전한지는 부류마다 재어 보면 된다. 이미지와 그것을 좌우로 뒤집은 것 사이의 평균 절대 차이 $\frac{1}{784}\sum_{i,j} |x_{ij} - x_{i, 29-j}|$를 $[0, 1]$ 눈금에서 부류마다 평균하면 이렇게 나온다.

    ```
    Ankle boot      (Class 9): 0.3269
    Sandal          (Class 5): 0.1554
    Sneaker         (Class 7): 0.1284
    Bag             (Class 8): 0.1017
    Trouser         (Class 1): 0.0903
    Shirt           (Class 6): 0.0823
    Pullover        (Class 2): 0.0787
    Coat            (Class 4): 0.0774
    Dress           (Class 3): 0.0715
    T-shirt/top     (Class 0): 0.0707
    ```

    윗옷과 원피스는 0.07~0.08에 모여 있어 거의 좌우 대칭이다. 뒤집어도 그럴듯한 같은 부류의 이미지가 되므로 증강이 안전하다. 신발 셋은 사정이 다르다. 앵클부츠가 0.3269로 티셔츠의 4.6배이고 샌들과 운동화가 그 뒤를 잇는데, Fashion-MNIST의 신발이 모두 같은 쪽을 보고 찍혀 있기 때문이다. 이 부류를 뒤집으면 데이터셋에 없는 방향의 이미지를 만들어 넣는 셈이라, 시험 집합에는 없는 변화를 배우게 된다.

    그러므로 실험은 두 갈래로 나누어 재는 것이 옳다. 뒤집기를 모든 부류에 주는 갈래와 신발 셋을 빼고 주는 갈래를 견주고, 어느 쪽이든 씨앗을 여럿 써서 퍼짐과 함께 적어야 한다. 한 번 잰 값의 차이는 차이가 아니다. 같은 부류 안의 변화가 큰 코트, 풀오버, 원피스가 회전 증강에서 가장 이득을 볼 만한데, 더해진 변화가 여러 자세에 걸친 일반화를 돕기 때문이다.

---

## 정리하며

**다룬 것** — Fashion-MNIST 데이터셋 시각화

Fashion-MNIST는 텐서의 짜임도 전처리도 MNIST와 똑같아서, 데이터를 부르는 줄에 `fashion_mnist=True` 하나만 더하면 나머지 코드가 그대로 돈다. 다른 것은 내용물뿐이다. 부류마다 학습 6000장이 정확히 맞춰져 있어 정확도 하나로 재도 뜻이 통하지만, 같은 CNN이 MNIST에서 99.21%를 내고 여기서는 85.84%에 그친다. 이 13%p가 이 절이 보이려는 것이다.

앞의 연습문제 4개로 직접 확인할 수 있다.
