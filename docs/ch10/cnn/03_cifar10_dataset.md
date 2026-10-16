# CIFAR-10 데이터셋 시각화

CIFAR-10은 32×32 RGB 이미지의 학습셋 50000장과 테스트셋 10000장으로 구성된다. MNIST 계열의 채널 1과 달리 색 채널 3을 사용하므로 첫 컨볼루션의 입력 채널과 시각화의 축 순서를 맞춰야 한다.

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
    데이터가 아직 `./data`에 없으면 소개의 `Files already downloaded and verified` 두 줄 자리에 `Downloading ...` 과 `Extracting ...` 이 대신 찍힌다. 내려받기가 한 번 끝나면 다시 나오지 않으므로 여기에는 싣지 않았다. 두 줄인 까닭은 `load_data`가 학습셋과 테스트셋을 따로 만들면서 저마다 `download=True`로 부르기 때문이다.

## 2. 논의

MNIST에서 CIFAR-10으로 넘어가면 컴퓨터 비전의 근본적인 어려움 몇 가지가 드러난다. CIFAR-10 이미지는 모양이 $(3, 32, 32)$으로, 공간 해상도 $32 \times 32$에 색 채널 세 개(빨강, 초록, 파랑)를 나타낸다. 곧 이미지마다 값이 $3 \times 32 \times 32 = 3{,}072$개로, MNIST의 픽셀 784개보다 $3072 / 784 = 3.92$배 많다. 첫 컨볼루션 레이어는 입력 채널을 1개가 아니라 3개 받는다. 웨이트만 세면 정확히 세 배($3 \times 3 \times 1 \times 32 = 288$개에서 $3 \times 3 \times 3 \times 32 = 864$개로)이지만, 바이어스 32개는 입력 채널 수와 상관없이 그대로이므로 레이어 전체로는 $896 / 320 = 2.8$배가 된다. 연습문제 2에서 이 계산을 직접 해 본다.

자연 사진에는 배경·가림·조명·자세와 크기의 변화가 있다. 이 책의 예제에 기록된 MNIST 99.21%, Fashion-MNIST 85.84%, CIFAR-10 심화 CNN 54.82%는 서로 다른 학습 설정에서 얻은 값이다. 구조, 옵티마이저와 에포크 수도 달라 데이터셋 난이도만의 인과 효과로 해석하거나 각 데이터셋의 달성 가능한 성능으로 취급할 수 없다.

정규화 방식은 세 채널로 넓혀진다. `cnn_utils.py`의 `load_data`는 `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`를 써서 채널마다 따로 평균 0.5와 표준편차 0.5로 나누므로, 모든 픽셀 값이 $[-1, 1]$으로 옮겨 간다. 화면에 보이려면 역변환 $(x / 2 + 0.5)$으로 $[0, 1]$ 범위를 되찾아야 한다. 여기에 더해 축을 옮기는 일이 하나 더 붙는다. matplotlib은 $(H, W)$ 또는 $(H, W, C)$를 받는데 PyTorch 텐서는 $(C, H, W)$이다. 회색조 쪽에서는 $C = 1$이라 `.squeeze()`로 길이 1짜리 채널 축을 떨어뜨리면 그대로 $(H, W)$가 되어 축을 옮길 일이 없었다. 여기서는 $C = 3$이라 짜낼 축이 없고, `.permute(1, 2, 0)`이 실제로 축의 차례를 바꾸어 $(3, 32, 32)$를 $(32, 32, 3)$으로 만든다. 회색조 쪽의 `.squeeze()`를 "자리 바꾸기"로 잘못 읽고 오면 이 차이를 놓치게 된다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
2절의 출력은 표본 이미지 한 장의 채널별 평균을 `Red 0.376`, `Green 0.381`, `Blue 0.324`로 찍는다. 이 값들은 정규화를 거친 뒤의 $[-1, 1]$ 눈금이다. 원래 $[0, 1]$ 눈금으로 되돌려라. 되돌린 값을 CIFAR-10 학습셋 전체의 채널별 평균(0.4914, 0.4822, 0.4465)과 견주면 이 한 장은 데이터셋 평균보다 밝은가, 어두운가?

</div>

??? success "연습문제 1 풀이"
    정규화가 $x' = (x - 0.5) / 0.5 = 2x - 1$이므로 역변환은 $x = (x' + 1) / 2 = x' / 2 + 0.5$이다. 3절의 `img_display / 2 + 0.5`이 바로 이 식이다.

    - 빨강: $0.376 / 2 + 0.5 = 0.688$
    - 초록: $0.381 / 2 + 0.5 = 0.6905$
    - 파랑: $0.324 / 2 + 0.5 = 0.662$

    세 값이 모두 데이터셋 평균 0.4914, 0.4822, 0.4465보다 0.19에서 0.22만큼 높으므로, 이 한 장은 데이터셋 평균보다 뚜렷이 밝다.

    다만 한 장은 데이터셋이 아니다. 이 이미지는 초록이 빨강보다 조금 높지만($0.6905 > 0.688$), 학습셋 전체에서는 반대로 빨강이 가장 높고 파랑이 가장 낮다($0.4914 > 0.4822 > 0.4465$). 한 장의 채널 차례로 데이터셋의 색 바이어스를 읽어서는 안 된다는 뜻이며, 연습문제 3에서 클래스마다 이 차례를 재어 본다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
크기가 $3 \times 3$인 출력 필터 32개를 쓰는 CNN의 첫 컨볼루션 레이어에서, 입력 채널이 3개일 때(CIFAR-10)와 1개일 때(MNIST) 파라메터의 총수를 계산하라. 바이어스 항도 넣어라. 바이어스를 빼고 세면 비가 어떻게 달라지는가?

</div>

??? success "연습문제 2 풀이"
    바이어스가 있는 `Conv2d(in_channels, 32, kernel_size=3)` 레이어에 대해 다음과 같다.

    - MNIST (채널 1개): $(3 \times 3 \times 1 + 1) \times 32 = 10 \times 32 = 320$개의 파라메터
    - CIFAR-10 (채널 3개): $(3 \times 3 \times 3 + 1) \times 32 = 28 \times 32 = 896$개의 파라메터

    CIFAR-10 쪽 첫 레이어의 파라메터가 $896 / 320 = 2.8$배 많다. 필터마다 색 채널 세 개에 걸친 공간 무늬를 한꺼번에 배워야 하기 때문이다.

    바이어스를 빼고 웨이트만 세면 $288$개와 $864$개가 되어 비는 정확히 $864 / 288 = 3$배이다. 비가 3에서 2.8로 내려앉는 까닭은 바이어스가 출력 필터마다 하나씩, 곧 양쪽 모두 32개로 입력 채널 수와 무관하게 붙기 때문이다. 레이어 전체로 보면 이 32개가 분모 쪽에 상대적으로 더 무겁게 실린다. 그러므로 "채널이 세 배니 파라메터도 세 배"라는 말은 웨이트에 대해서만 정확하다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
모든 채널에 하나의 전역 평균을 쓰기보다 채널별 정규화 통계량(R, G, B마다 따로 평균과 표준편차)을 계산하는 편이 나은 까닭을 설명하라. CIFAR-10 학습셋에서 채널별 평균과 표준편차를 실제로 재어라. 이어서 클래스마다 채널별 평균을 재어, 파랑이 빨강보다 높은 클래스가 어느 것인지 수로 가려내라.

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

    전체 픽셀의 분산은 이미지 내부 분산의 평균과 이미지별 평균의 분산으로 분해할 수 있다. 같은 모집단 분산 규약과 동일한 이미지 크기에서 $\operatorname{Var}(X)=E[\operatorname{Var}(X\mid I)]+\operatorname{Var}(E[X\mid I])$이다. 이미지별 표준편차를 먼저 산술평균한 값의 제곱은 이미지별 분산의 평균과 다르므로, 그 값을 제곱해 더하면 정확히 일치하지 않는다.

    전체 픽셀을 기준으로 평균 0·분산 1로 표준화하려면 같은 학습셋 집계 규약으로 계산한 채널별 통계량을 사용한다. 이미지별 표준편차 평균을 쓰는 설정은 다른 스케일링이며 반드시 잘못된 것은 아니다. 목적과 구현을 명시하고 테스트에서 새로 추정하지 않는다.

    이제 클래스마다 채널별 평균을 잰다.

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

    기록된 클래스 평균에서 비행기와 배는 파랑 채널이 빨강보다 높다. 하늘이나 바다의 배경이 기여한다는 가설을 세울 수 있지만 클래스 평균만으로 그 원인을 입증할 수는 없다. 개구리의 채널 차이도 실제 이미지의 전경·배경을 따로 살펴 해석해야 한다.

    하나의 공통 평균·표준편차를 쓰면 채널마다 평균과 분산이 다르게 남을 수 있다. 채널별 표준화는 이 차이를 조정하는 선택이며, 실제 학습의 속도나 정확도를 언제나 개선한다고 보장하지는 않는다. 사전 학습 웨이트를 사용할 때에는 그 웨이트의 입력 전처리 규약을 우선 맞춘다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
CIFAR-10에서 좌우 뒤집기, 패딩 뒤 무작위 자르기, 색상 변형과 작은 회전을 사용할 때의 가정과 한계를 설명하라. 원본 5만 장의 온라인 증강이 독립 표본 20만 장과 같은지, 아래 좌우 픽셀 차이로 증강 안전성을 판정할 수 있는지도 논하라.

</div>

??? success "연습문제 4 풀이"
    알맞은 증강 기법 네 가지는 다음과 같다.

    1. **무작위 좌우 뒤집기**: 이미지를 좌우로 뒤집어도 물체의 클래스는 그대로인데, 자연의 물체는 뒤집어도 같아 보이기 때문이다. 왼쪽을 향한 비행기도 여전히 비행기이다.

    2. **패딩 뒤 무작위 잘라내기**: 이미지 사방에 픽셀 4개를 덧댄 뒤 다시 $32 \times 32$으로 무작위로 잘라 내면 작은 평행 이동을 흉내 낸다. 픽셀 몇 개만큼 옮긴 트럭도 여전히 트럭이다.

    3. **색상 변형**: 밝기, 대비, 채도를 무작위로 조절하면 여러 조명 조건을 흉내 낸다. 더 밝은 빛 아래의 개도 여전히 개이다.

    4. **무작위 회전 (작은 각도, 예를 들어 $\pm 15$도)**: 자연 이미지의 물체는 조금 기울어 보일 수 있다. 조금 기운 배도 여전히 배로 알아볼 수 있다.

    변환이 작고 물체를 식별할 정보가 남으면 레이블이 유지될 수 있다. 그러나 자르기로 물체를 지우거나 강한 색상 변형·회전으로 비현실적인 이미지를 만들 수도 있다. 여러 증강을 조합한다고 독립적인 학습 이미지 20만 장이 생기는 것은 아니다. 온라인 증강은 같은 원본에서 서로 의존하는 변형을 반복해서 생성한다.

    아래 평균 절대 차이는 원본과 뒤집힌 이미지의 비대칭 정도를 측정한다. 배경과 질감의 차이도 포함하므로 이 값만으로 클래스 의미가 유지되는지나 테스트 성능이 개선되는지 판정할 수 없다.

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

    기록된 클래스별 값은 약 0.1191~0.1898이다. Fashion-MNIST의 값과의 차이는 두 데이터의 영상 구조가 다르다는 관찰이다. CIFAR-10의 물체 클래스는 좌우 방향에 레이블이 의존하지 않는 경우가 많지만, 모든 이미지에서 변환의 적합성이나 성능 개선을 이 통계가 보장하지는 않는다.

    증강 효과는 별도 검증셋에서 같은 구조·학습 예산으로 증강 유무를 비교하고 여러 시드의 변동을 함께 보고한다. 픽셀 차이의 크기와 분류 성능의 차이는 서로 다른 지표이다.

---

## 정리하며

CIFAR-10의 배치 입력은 B×3×32×32이고 시각화에는 CHW→HWC 축 변환과 필요한 역정규화가 필요하다. 전체·이미지별 통계량의 집계 방식, 클래스 평균의 해석과 증강의 타당성을 구분한다. 서로 다른 설정의 정확도 차이를 데이터셋 효과 하나로 돌리지 않는다.
