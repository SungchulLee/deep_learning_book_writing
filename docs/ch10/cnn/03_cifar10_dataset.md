# CIFAR-10 데이터셋 시각화

CIFAR-10은 색 이미지와 실제 물체 인식을 들여와 MNIST 계열보다 복잡함이 한 단계 크게 올라간다. 동물과 탈것을 아우르는 10개 물체 부류에 걸쳐 $32 \times 32$ RGB 사진 6만 장이 들어 있고, 학습 집합 5만 장과 시험 집합 1만 장으로 나뉜다. 앞의 두 쪽 [MNIST 데이터셋](01_mnist_dataset.md)과 [Fashion-MNIST 데이터셋](02_fashion_mnist_dataset.md)은 회색조 $28 \times 28$이라 채널 축이 사실상 없었지만, 여기서는 채널이 셋이다. 그래서 이 쪽에서는 세 채널 입력을 처리할 줄 알아야 하고, 자연 이미지 분류가 손글씨 인식보다 왜 근본적으로 더 어려운지 알아야 한다.

## 1. 코드

```python
"""
03_cifar10_dataset.py
=====================
CIFAR-10 데이터셋 시각화

CIFAR-10은 색 이미지와 실제 물체 인식을 들여온다!
이 데이터셋은 MNIST 계열보다 훨씬 까다롭다.

CIFAR-10(캐나다 고등 연구원):
- 32x32 RGB(색) 이미지 60,000장 (학습 50,000장 / 시험 10,000장)
- 실제 세상의 물체 부류 10개
- 배경이 있는 자연 이미지
- 자세와 조명과 크기의 변화

난이도: 쉬움
예상 시간: 30분

지은이: PyTorch CNN 실습
날짜: 2025년 11월
"""

import matplotlib.pyplot as plt
import cnn_utils as utils

# =============================================================================
# 1절: CIFAR-10 부류 이름표
# =============================================================================

CIFAR10_LABELS = {
    0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
    5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
}

# =============================================================================
# 2절: 설정과 데이터 적재
# =============================================================================

cfg = utils.parse_args()
utils.set_seed(seed=cfg.seed)

train_kwargs = {'batch_size': cfg.batch_size, 'shuffle': True}
test_kwargs = {'batch_size': cfg.test_batch_size, 'shuffle': False}

# MNIST를 부르던 줄에 cifar10=True 하나만 더 준 것이다.
# - 학습 집합: 이미지 50,000장 (부류마다 5,000장)
# - 시험 집합: 이미지 10,000장 (부류마다 1,000장)
# - 이미지마다: 32x32 RGB (채널 3개)
trainloader, testloader = utils.load_data(
    train_kwargs, test_kwargs, cifar10=True
)

# 배치 수 x 배치 크기로 세면 안 된다 -- 782 x 64 = 50048 이 되어,
# 16장뿐인 마지막 배치를 64장으로 쳐서 48장을 더 세게 된다.
# 실제 장 수는 데이터셋에서 직접 읽는다.
print(f"Training images: {len(trainloader.dataset)}")
print(f"Test images: {len(testloader.dataset)}")
print(f"Training batches: {len(trainloader)}")

sample_images, sample_labels = next(iter(trainloader))
print(f"Image shape: {sample_images.shape}")
print(f"  Channels: {sample_images.shape[1]} (RGB color)")
print(f"  Height: {sample_images.shape[2]}, Width: {sample_images.shape[3]}")
print(f"Label shape: {sample_labels.shape}")

# 채널별 통계량 분석 -- 정규화를 거친 뒤라 [-1, 1] 눈금의 값이다.
sample_img = sample_images[0]
for ch, color in enumerate(['Red', 'Green', 'Blue']):
    channel_data = sample_img[ch].cpu().numpy()
    print(f"  {color}: min={channel_data.min():.3f}, "
          f"max={channel_data.max():.3f}, mean={channel_data.mean():.3f}")

# =============================================================================
# 3절: 시각화
# =============================================================================

fig, axes = plt.subplots(8, 8, figsize=(12, 12))
fig.suptitle('CIFAR-10 Natural Images Sample', fontsize=16)

for images, labels in trainloader:
    for ax, image, label in zip(axes.reshape(-1), images, labels):
        # matplotlib은 (H, W) 또는 (H, W, C)를 받는데 텐서는 (C, H, W)이다.
        # 회색조 쪽에서는 C = 1 이라 채널 축을 짜내기만 하면 되었지만,
        # 여기서는 C = 3 이라 축을 정말로 옮겨야 한다: (3, 32, 32) -> (32, 32, 3).
        img_display = image.permute(1, 2, 0).cpu().numpy()

        # 더 잘 보이도록 [-1, 1]에서 [0, 1]로 되돌리기
        img_display = img_display / 2 + 0.5

        # 부동소수점 반올림으로 아주 조금 벗어난 값을 imshow가 싫어한다
        img_display = img_display.clip(0, 1)

        ax.imshow(img_display)
        ax.axis("off")
        ax.set_title(CIFAR10_LABELS[label.item()], fontsize=8)

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
    name = CIFAR10_LABELS[class_id]
    print(f"  {name:15s}: {count} ({100.0 * count / total:.2f}%)")
print(f"  Total: {total}")


if __name__ == "__main__":
    pass
```

**출력:**

```
Files already downloaded and verified
Files already downloaded and verified
Training images: 50000
Test images: 10000
Training batches: 782
Image shape: torch.Size([64, 3, 32, 32])
  Channels: 3 (RGB color)
  Height: 32, Width: 32
Label shape: torch.Size([64])
  Red: min=-0.898, max=0.890, mean=0.376
  Green: min=-0.945, max=0.937, mean=0.381
  Blue: min=-0.992, max=0.937, mean=0.324

Class distribution:
  airplane       : 5000 (10.00%)
  automobile     : 5000 (10.00%)
  bird           : 5000 (10.00%)
  cat            : 5000 (10.00%)
  deer           : 5000 (10.00%)
  dog            : 5000 (10.00%)
  frog           : 5000 (10.00%)
  horse          : 5000 (10.00%)
  ship           : 5000 (10.00%)
  truck          : 5000 (10.00%)
  Total: 50000
```

!!! note "처음 돌릴 때만 나오는 줄"
    데이터가 아직 `./data`에 없으면 첫머리의 `Files already downloaded and verified` 두 줄 자리에 `Downloading ...` 과 `Extracting ...` 이 대신 찍힌다. 내려받기가 한 번 끝나면 다시 나오지 않으므로 여기에는 싣지 않았다. 두 줄인 까닭은 `load_data`가 학습 집합과 시험 집합을 따로 만들면서 저마다 `download=True`로 부르기 때문이다.

## 2. 논의

MNIST에서 CIFAR-10으로 넘어가면 컴퓨터 비전의 근본적인 어려움 몇 가지가 드러난다. CIFAR-10 이미지는 모양이 $(3, 32, 32)$으로, 공간 해상도 $32 \times 32$에 색 채널 세 개(빨강, 초록, 파랑)를 나타낸다. 곧 이미지마다 값이 $3 \times 32 \times 32 = 3{,}072$개로, MNIST의 화소 784개보다 $3072 / 784 = 3.92$배 많다. 첫 합성곱 층은 입력 채널을 1개가 아니라 3개 받는다. 가중치만 세면 정확히 세 배($3 \times 3 \times 1 \times 32 = 288$개에서 $3 \times 3 \times 3 \times 32 = 864$개로)이지만, 편향 32개는 입력 채널 수와 상관없이 그대로이므로 층 전체로는 $896 / 320 = 2.8$배가 된다. 연습문제 2에서 이 셈을 직접 해 본다.

자연 사진에는 숫자 인식에 없던 복잡함이 있다. 배경이 제각각이고, 가려지고, 조명과 크기가 바뀌고, 자세가 다양하다. 고양이는 아무 쪽이나 볼 수 있고, 가구에 일부가 가려질 수도 있고, 밝은 햇빛이나 어두운 실내에서 찍힐 수도 있다. 같은 부류 안의 이런 큰 변화에 $32 \times 32$라는 낮은 해상도가 겹쳐 CIFAR-10은 훨씬 어렵다. 이 책이 직접 잰 값으로 견주면 그 간격이 눈에 보인다. 합성곱 층 두 개짜리 같은 CNN이 숫자 MNIST에서는 시험 정확도 99.21%를 내지만([3.4 MNIST 합성곱 신경망](../../ch03/mnist/04_cnn.md)), Fashion-MNIST에서는 85.84%에 그친다([Fashion-MNIST 분류기](05_fashion_mnist_classifier.md)). CIFAR-10은 합성곱 층을 넷 쓰는 더 깊은 구조(`cnn_utils.py`의 `CNN_CIFAR10`)로 14세대를 학습시킨 [CIFAR-10 심화 CNN](07_cifar10_advanced.md)조차 54.82%에 머문다. 세 실험은 최적화기와 세대 수가 달라 소수점까지 맞대 놓을 것은 아니고, 마지막 것은 앞의 둘보다 오히려 깊은 구조를 썼다. 그런데도 99% → 86% → 55%라는 차례가 나온다. 이 차례를 만든 것은 구조가 아니라 데이터셋이다.

정규화 방식은 세 채널로 넓혀진다. `cnn_utils.py`의 `load_data`는 `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`를 써서 채널마다 따로 평균 0.5와 표준편차 0.5로 나누므로, 모든 화소 값이 $[-1, 1]$으로 옮겨 간다. 화면에 보이려면 역변환 $(x / 2 + 0.5)$으로 $[0, 1]$ 범위를 되찾아야 한다. 여기에 더해 축을 옮기는 일이 하나 더 붙는다. matplotlib은 $(H, W)$ 또는 $(H, W, C)$를 받는데 PyTorch 텐서는 $(C, H, W)$이다. 회색조 쪽에서는 $C = 1$이라 `.squeeze()`로 길이 1짜리 채널 축을 떨어뜨리면 그대로 $(H, W)$가 되어 축을 옮길 일이 없었다. 여기서는 $C = 3$이라 짜낼 축이 없고, `.permute(1, 2, 0)`이 실제로 축의 차례를 바꾸어 $(3, 32, 32)$를 $(32, 32, 3)$으로 만든다. 회색조 쪽의 `.squeeze()`를 "자리 바꾸기"로 잘못 읽고 오면 이 차이를 놓치게 된다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
2절의 출력은 표본 이미지 한 장의 채널별 평균을 `Red 0.376`, `Green 0.381`, `Blue 0.324`로 찍는다. 이 값들은 정규화를 거친 뒤의 $[-1, 1]$ 눈금이다. 원래 $[0, 1]$ 눈금으로 되돌려라. 되돌린 값을 CIFAR-10 학습 집합 전체의 채널별 평균(0.4914, 0.4822, 0.4465)과 견주면 이 한 장은 데이터셋 평균보다 밝은가, 어두운가?

</div>

??? success "연습문제 1 풀이"
    정규화가 $x' = (x - 0.5) / 0.5 = 2x - 1$이므로 역변환은 $x = (x' + 1) / 2 = x' / 2 + 0.5$이다. 3절의 `img_display / 2 + 0.5`이 바로 이 식이다.

    - 빨강: $0.376 / 2 + 0.5 = 0.688$
    - 초록: $0.381 / 2 + 0.5 = 0.6905$
    - 파랑: $0.324 / 2 + 0.5 = 0.662$

    세 값이 모두 데이터셋 평균 0.4914, 0.4822, 0.4465보다 0.19에서 0.22만큼 높으므로, 이 한 장은 데이터셋 평균보다 뚜렷이 밝다.

    다만 한 장은 데이터셋이 아니다. 이 이미지는 초록이 빨강보다 조금 높지만($0.6905 > 0.688$), 학습 집합 전체에서는 반대로 빨강이 가장 높고 파랑이 가장 낮다($0.4914 > 0.4822 > 0.4465$). 한 장의 채널 차례로 데이터셋의 색 치우침을 읽어서는 안 된다는 뜻이며, 연습문제 3에서 부류마다 이 차례를 재어 본다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
크기가 $3 \times 3$인 출력 필터 32개를 쓰는 CNN의 첫 합성곱 층에서, 입력 채널이 3개일 때(CIFAR-10)와 1개일 때(MNIST) 매개변수의 총수를 계산하라. 편향 항도 넣어라. 편향을 빼고 세면 비가 어떻게 달라지는가?

</div>

??? success "연습문제 2 풀이"
    편향이 있는 `Conv2d(in_channels, 32, kernel_size=3)` 층에 대해 다음과 같다.

    - MNIST (채널 1개): $(3 \times 3 \times 1 + 1) \times 32 = 10 \times 32 = 320$개의 매개변수
    - CIFAR-10 (채널 3개): $(3 \times 3 \times 3 + 1) \times 32 = 28 \times 32 = 896$개의 매개변수

    CIFAR-10 쪽 첫 층의 매개변수가 $896 / 320 = 2.8$배 많다. 필터마다 색 채널 세 개에 걸친 공간 무늬를 한꺼번에 배워야 하기 때문이다.

    편향을 빼고 가중치만 세면 $288$개와 $864$개가 되어 비는 정확히 $864 / 288 = 3$배이다. 비가 3에서 2.8로 내려앉는 까닭은 편향이 출력 필터마다 하나씩, 곧 양쪽 모두 32개로 입력 채널 수와 무관하게 붙기 때문이다. 층 전체로 보면 이 32개가 분모 쪽에 상대적으로 더 무겁게 실린다. 그러므로 "채널이 세 배니 매개변수도 세 배"라는 말은 가중치에 대해서만 정확하다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
모든 채널에 하나의 전역 평균을 쓰기보다 채널별 정규화 통계량(R, G, B마다 따로 평균과 표준편차)을 계산하는 편이 나은 까닭을 설명하라. CIFAR-10 학습 집합에서 채널별 평균과 표준편차를 실제로 재어라. 이어서 부류마다 채널별 평균을 재어, 파랑이 빨강보다 높은 부류가 어느 것인지 수로 가려내라.

</div>

??? success "연습문제 3 풀이"
    먼저 데이터셋 전체의 채널별 통계량이다. `trainloader`가 주는 것은 정규화된 값이므로 $x / 2 + 0.5$로 $[0, 1]$ 눈금을 되찾은 뒤 잰다.

    ```python
    import torch

    s = torch.zeros(3)
    sq = torch.zeros(3)
    n = 0
    for images, _ in trainloader:
        raw = images / 2 + 0.5
        s += raw.sum(dim=(0, 2, 3))
        sq += (raw ** 2).sum(dim=(0, 2, 3))
        n += raw.shape[0] * raw.shape[2] * raw.shape[3]
    mean = s / n
    std = (sq / n - mean ** 2).sqrt()
    print("Mean:", [f"{v:.4f}" for v in mean])
    print("Std: ", [f"{v:.4f}" for v in std])
    ```

    ```
    Mean: ['0.4914', '0.4822', '0.4465']
    Std:  ['0.2470', '0.2435', '0.2616']
    ```

    이 평균 셋은 CIFAR-10을 쓰는 코드에서 널리 인용되는 값과 같다. 표준편차는 갈린다. 남의 코드에서는 0.2023, 0.1994, 0.2010을 훨씬 자주 보게 되는데, 이 값은 틀린 값이 아니라 **다른 것을 잰 값**이다.

    ```python
    x = torch.tensor(trainloader.dataset.data).float() / 255   # (50000, 32, 32, 3)

    print("화소 전체        ", [f"{v:.4f}" for v in x.std(dim=(0, 1, 2))])
    print("장마다 재어 평균 ", [f"{v:.4f}" for v in x.std(dim=(1, 2)).mean(dim=0)])
    ```

    ```
    화소 전체         ['0.2470', '0.2435', '0.2616']
    장마다 재어 평균  ['0.2023', '0.1994', '0.2010']
    ```

    널리 퍼진 그 수는 **한 장 안에서 표준편차를 잰 뒤 5만 장에 걸쳐 평균낸** 값이다. 소수점 넷째 자리까지 그대로 맞는다.

    두 값이 다른 까닭은 흩어짐이 두 갈래이기 때문이다. 한 장 안에서 화소가 흩어진 정도가 있고, 장과 장 사이에 밝기가 흩어진 정도가 따로 있다. 앞의 것만 재면 뒤의 것이 빠지므로 장마다 잰 값이 더 작게 나온다. 빠진 몫은 장마다 잰 평균의 표준편차 0.1284, 0.1258, 0.1533이고, 두 갈래는 제곱으로 더해진다 — $\sqrt{0.2023^2 + 0.1284^2} = 0.2396$으로 0.2470에 가깝다(딱 맞지 않는 것은 장마다의 표준편차를 제곱평균이 아니라 산술평균으로 냈기 때문이다).

    `transforms.Normalize`는 데이터셋 전체에 상수 하나를 나누므로 여기에 맞는 것은 화소 전체에 걸친 0.2470, 0.2435, 0.2616이다. 이 책은 4장에서도 이 값을 쓴다. 차이가 크지는 않아서 — 20%쯤 — 어느 쪽을 써도 학습은 되지만, 두 수가 같은 것을 재다 어긋난 것이 아니라 애초에 다른 것을 잰 값임은 알고 쓰는 편이 낫다.

    이제 부류마다 채널별 평균을 잰다.

    ```python
    channel_sums = torch.zeros(10, 3)
    counts = torch.zeros(10)
    for images, labels in trainloader:
        raw = images / 2 + 0.5
        means = raw.mean(dim=(2, 3))
        for m, lbl in zip(means, labels):
            channel_sums[lbl] += m
            counts[lbl] += 1

    avg = channel_sums / counts.unsqueeze(1)
    for c in avg.mean(1).argsort(descending=True):
        r, g, b = avg[c]
        print(f"  {CIFAR10_LABELS[c.item()]:12s}: "
              f"R={r:.4f} G={g:.4f} B={b:.4f}  mean={avg[c].mean():.4f}")
    ```

    ```
      airplane    : R=0.5257 G=0.5603 B=0.5889  mean=0.5583
      ship        : R=0.4902 G=0.5254 B=0.5547  mean=0.5234
      truck       : R=0.4987 G=0.4853 B=0.4781  mean=0.4874
      bird        : R=0.4893 G=0.4915 B=0.4240  mean=0.4683
      horse       : R=0.5020 G=0.4799 B=0.4169  mean=0.4662
      dog         : R=0.4999 G=0.4646 B=0.4165  mean=0.4604
      automobile  : R=0.4712 G=0.4545 B=0.4472  mean=0.4576
      cat         : R=0.4955 G=0.4564 B=0.4155  mean=0.4558
      deer        : R=0.4716 G=0.4652 B=0.3782  mean=0.4383
      frog        : R=0.4701 G=0.4384 B=0.3452  mean=0.4179
    ```

    파랑이 빨강보다 높은 부류는 둘뿐이다. 비행기가 $0.5889 > 0.5257$이고 배가 $0.5547 > 0.4902$이며, 나머지 여덟 부류는 모두 빨강이 파랑보다 높다. 이 둘이 곧 하늘과 바다를 배경으로 찍히는 부류이므로, "야외 장면은 파랑 채널이 높다"는 짐작이 수로 확인되는 셈이다. 두 부류는 평균 밝기로도 1위(0.5583)와 2위(0.5234)이다. 반대쪽 끝은 개구리로, 파랑이 0.3452까지 내려가 채널 사이의 벌어짐이 $0.4701 - 0.3452 \approx 0.125$로 열 부류 가운데 가장 크다. 풀과 흙을 배경으로 찍힌 결과이다.

    전역 정규화 하나만 쓰면 이 분포들이 뒤섞인다. 세 채널의 평균이 0.4914에서 0.4465까지 벌어져 있으므로 하나의 평균으로 빼면 파랑 채널은 평균이 0이 되지 않고 양이나 음으로 치우친 채 남고, 표준편차도 0.2435에서 0.2616까지 달라 채널마다 크기가 어긋난다. 그러면 신경망이 메워야 할 비대칭이 생겨 모델의 용량이 낭비되고 수렴이 느려질 수 있다. 채널별 정규화는 신경망이 균형 잡힌 표현에서 출발하도록 해 준다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
CIFAR-10 학습 집합에는 이미지가 5만 장(부류마다 5000장) 있고 MNIST에는 6만 장(부류마다 약 6000장) 있다. CIFAR-10을 증강하여 실질적으로 20만 장의 학습 집합을 만들고 싶다면, 자연 이미지에 알맞은 증강 기법 네 가지를 서술하고 각각이 의미 이름표를 지키는 까닭을 설명하라. 그 가운데 무작위 좌우 뒤집기에 대해서는, 앞 쪽 [Fashion-MNIST 데이터셋](02_fashion_mnist_dataset.md)의 연습문제 4가 신발 세 부류에서 뒤집기가 위험하다고 재어 보인 것과 같은 재기를 CIFAR-10에서도 해 보고 결론이 같은지 확인하라.

</div>

??? success "연습문제 4 풀이"
    알맞은 증강 기법 네 가지는 다음과 같다.

    1. **무작위 좌우 뒤집기**: 이미지를 좌우로 뒤집어도 물체의 부류는 그대로인데, 자연의 물체는 뒤집어도 같아 보이기 때문이다. 왼쪽을 향한 비행기도 여전히 비행기이다.

    2. **덧대기 뒤 무작위 잘라내기**: 이미지 사방에 화소 4개를 덧댄 뒤 다시 $32 \times 32$으로 무작위로 잘라 내면 작은 평행 이동을 흉내 낸다. 화소 몇 개만큼 옮긴 트럭도 여전히 트럭이다.

    3. **색 흔들기**: 밝기, 대비, 채도를 무작위로 조절하면 여러 조명 조건을 흉내 낸다. 더 밝은 빛 아래의 개도 여전히 개이다.

    4. **무작위 회전 (작은 각도, 예를 들어 $\pm 15$도)**: 자연 이미지의 물체는 조금 기울어 보일 수 있다. 조금 기운 배도 여전히 배로 알아볼 수 있다.

    이 기법들이 모두 쓸 만한 학습 예제를 만드는 까닭은 그 변환이 물체의 근본적인 정체를 바꾸지 않고 위치, 방향, 조명 같은 부수적인 성질만 바꾸기 때문이다. 넷을 함께 쓰면 원본 5만 장에서 세대마다 서로 다른 변형이 뽑히므로 실질적인 학습 집합이 20만 장 규모로 불어난다.

    좌우 뒤집기가 안전한지는 부류마다 재어 보면 된다. 이미지와 그것을 좌우로 뒤집은 것 사이의 평균 절대 차이를 $[0, 1]$ 눈금에서 부류마다 평균한다.

    ```python
    flip_sums = torch.zeros(10)
    counts = torch.zeros(10)
    for images, labels in trainloader:
        raw = images / 2 + 0.5
        d = (raw - torch.flip(raw, dims=[3])).abs().mean(dim=(1, 2, 3))
        for v, lbl in zip(d, labels):
            flip_sums[lbl] += v
            counts[lbl] += 1

    asym = flip_sums / counts
    for c in asym.argsort(descending=True):
        print(f"  {CIFAR10_LABELS[c.item()]:12s}: {asym[c]:.4f}")
    ```

    ```
      truck       : 0.1898
      cat         : 0.1851
      dog         : 0.1771
      automobile  : 0.1748
      horse       : 0.1677
      frog        : 0.1502
      bird        : 0.1395
      deer        : 0.1304
      ship        : 0.1297
      airplane    : 0.1191
    ```

    결론은 Fashion-MNIST와 다르다. 열 부류가 0.1191에서 0.1898 사이에 촘촘히 모여 있어 가장 큰 값이 가장 작은 값의 $0.1898 / 0.1191 = 1.59$배에 지나지 않는다. Fashion-MNIST에서는 앵클부츠가 0.3269로 티셔츠 0.0707의 4.6배까지 벌어졌고, 그것은 그 데이터셋의 신발이 모두 같은 쪽을 보고 찍혔기 때문이었다. CIFAR-10에는 그렇게 튀는 부류가 없다. 자동차와 트럭과 말이 사진마다 왼쪽이나 오른쪽을 아무렇게나 보고 있어 뒤집은 이미지도 데이터셋 안에 이미 있는 종류이다. 그러므로 무작위 좌우 뒤집기는 열 부류 모두에 안전하며, 실제로 CIFAR-10 학습법에서 거의 언제나 기본으로 쓰인다.

    값 자체가 Fashion-MNIST보다 전반적으로 큰 것(0.12~0.19 대 0.07~0.33)은 뒤집기가 위험하다는 뜻이 아니다. 자연 사진에는 배경 질감이 있어 좌우 어느 쪽으로도 화소가 많이 어긋나기 때문이며, 대칭성과는 다른 이야기이다. 그래도 증강의 효과를 주장하려면 씨앗을 여럿 써서 퍼짐과 함께 적어야 한다. 한 번 잰 값의 차이는 차이가 아니다.

---

## 정리하며

**다룬 것** — CIFAR-10 데이터셋 시각화

CIFAR-10도 데이터를 부르는 줄에 `cifar10=True` 하나만 더하면 앞 두 쪽의 코드가 그대로 돌지만, 화면에 보이는 대목에서만은 그렇지 않다. 채널이 셋이라 `.squeeze()`가 할 일이 없고 `.permute(1, 2, 0)`으로 축을 정말 옮겨야 한다. 학습 5만 장이 부류마다 정확히 5000장씩 고르게 나뉘어 있어 정확도 하나로 재도 뜻이 통하며, 이미지마다 값이 3,072개로 MNIST의 3.92배다. 그리고 이 책이 잰 값으로 MNIST 99.21%, Fashion-MNIST 85.84%, CIFAR-10 54.82%라는 차례가 나온다. 이 차례가 이 절이 보이려는 것이다.

앞의 연습문제 4개로 직접 확인할 수 있다.
