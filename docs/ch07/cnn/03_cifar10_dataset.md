# CIFAR-10 데이터셋 시각화

CIFAR-10은 색 이미지와 실제 물체 인식을 들여와 MNIST 계열보다 복잡함이 한 단계 크게 올라간다. 동물과 탈것을 아우르는 10개 물체 부류에 걸쳐 $32 \times 32$ RGB 사진 6만 장이 들어 있다. CIFAR-10을 다루려면 세 채널 입력을 처리할 줄 알아야 하고, 자연 이미지 분류가 손글씨 인식보다 왜 근본적으로 더 어려운지 알아야 한다.

## 1. 코드

```python
"""
03_cifar10_dataset.py
=====================
CIFAR-10 데이터셋 시각화

CIFAR-10은 색 이미지와 실제 물체 인식을 들여온다!
이 데이터셋은 MNIST 계열보다 훨씬 까다롭다.

CIFAR-10(캐나다 고등 연구원):
- 32x32 RGB(색) 이미지 60,000장
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

trainloader, testloader = utils.load_data(
    train_kwargs, test_kwargs, cifar10=True
)

sample_images, sample_labels = next(iter(trainloader))
print(f"Image shape: {sample_images.shape}")
print(f"  Channels: {sample_images.shape[1]} (RGB color)")
print(f"  Height: {sample_images.shape[2]}, Width: {sample_images.shape[3]}")

# 채널별 통계량 분석
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
        img_display = image.permute(1, 2, 0).cpu().numpy()
        img_display = img_display / 2 + 0.5
        img_display = img_display.clip(0, 1)
        ax.imshow(img_display)
        ax.axis("off")
        ax.set_title(CIFAR10_LABELS[label.item()], fontsize=8)
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

total = sum(class_counts)
for class_id, count in enumerate(class_counts):
    name = CIFAR10_LABELS[class_id]
    print(f"  {name:15s}: {count} ({100.0 * count / total:.2f}%)")


if __name__ == "__main__":
    pass
```

## 2. 논의

MNIST에서 CIFAR-10으로 넘어가면 컴퓨터 비전의 근본적인 어려움 몇 가지가 드러난다. CIFAR-10 이미지는 모양이 $(3, 32, 32)$으로, 공간 해상도 $32 \times 32$에 색 채널 세 개(빨강, 초록, 파랑)를 나타낸다. 곧 이미지마다 값이 $3 \times 32 \times 32 = 3{,}072$개로, MNIST의 화소 784개보다 대략 네 배 많다. 첫 합성곱 층은 입력 채널을 1개가 아니라 3개 받아야 하므로 그 층의 매개변수가 세 배가 된다.

자연 사진에는 숫자 인식에 없던 복잡함이 있다. 배경이 제각각이고, 가려지고, 조명과 크기가 바뀌고, 자세가 다양하다. 고양이는 아무 쪽이나 볼 수 있고, 가구에 일부가 가려질 수도 있고, 밝은 햇빛이나 어두운 실내에서 찍힐 수도 있다. 같은 부류 안의 이런 큰 변화에 $32 \times 32$라는 낮은 해상도가 겹쳐 CIFAR-10은 훨씬 어렵다. 단순한 CNN은 MNIST에서 99%를 내지만 CIFAR-10에서는 60~70%에 그친다.

정규화 방식은 세 채널로 넓혀진다. 채널마다 따로 평균 0.5와 표준편차 0.5로 정규화하여 모든 화소 값을 $[-1, 1]$으로 옮긴다. 화면에 보이려면 역변환 $(x / 2 + 0.5)$으로 $[0, 1]$ 범위를 되찾고, 텐서를 PyTorch의 $(C, H, W)$ 형식에서 matplotlib의 $(H, W, C)$ 형식으로 바꾸어 주어야 한다.

## 연습문제

**연습문제 1.**
크기가 $3 \times 3$인 출력 필터 32개를 쓰는 CNN의 첫 합성곱 층에서, 입력 채널이 3개일 때(CIFAR-10)와 1개일 때(MNIST) 매개변수의 총수를 계산하라. 편향 항도 넣어라.

??? success "연습문제 1 풀이"
    편향이 있는 `Conv2d(in_channels, 32, kernel_size=3)` 층에 대해 다음과 같다.

    - MNIST (채널 1개): $(3 \times 3 \times 1 + 1) \times 32 = 10 \times 32 = 320$개의 매개변수
    - CIFAR-10 (채널 3개): $(3 \times 3 \times 3 + 1) \times 32 = 28 \times 32 = 896$개의 매개변수

    CIFAR-10 쪽 첫 층의 매개변수가 $896 / 320 = 2.8$배 많다. 필터마다 색 채널 세 개에 걸친 공간 무늬를 한꺼번에 배워야 하기 때문이다.

---

**연습문제 2.**
모든 채널에 하나의 전역 평균을 쓰기보다 채널별 정규화 통계량(R, G, B마다 따로 평균과 표준편차)을 계산하는 편이 나은 까닭을 설명하라. 전역 정규화 하나만 쓰면 무엇이 잘못되는가?

??? success "연습문제 2 풀이"
    자연 이미지에서는 색 채널마다 밝기 분포가 다르다. 이를테면 야외 장면은 (하늘 때문에) 파랑 채널 값이 높은 편이고, 실내 장면은 더 따뜻한(빨강이 높은) 색조를 띨 수 있다. CIFAR-10의 참된 채널별 통계량(평균 0.4914, 0.4822, 0.4465; 표준편차 0.2023, 0.1994, 0.2010)을 쓰면 채널마다 따로 평균 0, 분산 1로 표준화된다.

    전역 정규화 하나만 쓰면 이 분포들이 뒤섞여 어떤 채널은 평균이 0이 아니거나 크기가 달라진다. 그러면 신경망이 메워야 할 비대칭이 생겨 모델의 용량이 낭비되고 수렴이 느려질 수 있다. 채널별 정규화는 신경망이 균형 잡힌 표현에서 출발하도록 해 준다.

---

**연습문제 3.**
CIFAR-10 학습 집합에는 이미지가 5만 장(부류마다 5000장) 있고 MNIST에는 6만 장(부류마다 약 6000장) 있다. CIFAR-10을 증강하여 실질적으로 20만 장의 학습 집합을 만들고 싶다면, 자연 이미지에 알맞은 증강 기법 네 가지를 서술하고 각각이 의미 이름표를 지키는 까닭을 설명하라.

??? success "연습문제 3 풀이"
    알맞은 증강 기법 네 가지는 다음과 같다.

    1. **무작위 좌우 뒤집기**: 이미지를 좌우로 뒤집어도 물체의 부류는 그대로인데, 자연의 물체는 뒤집어도 같아 보이기 때문이다. 왼쪽을 향한 비행기도 여전히 비행기이다.

    2. **덧대기 뒤 무작위 잘라내기**: 이미지 사방에 화소 4개를 덧댄 뒤 다시 $32 \times 32$으로 무작위로 잘라 내면 작은 평행 이동을 흉내 낸다. 화소 몇 개만큼 옮긴 트럭도 여전히 트럭이다.

    3. **색 흔들기**: 밝기, 대비, 채도를 무작위로 조절하면 여러 조명 조건을 흉내 낸다. 더 밝은 빛 아래의 개도 여전히 개이다.

    4. **무작위 회전 (작은 각도, 예를 들어 $\pm 15$도)**: 자연 이미지의 물체는 조금 기울어 보일 수 있다. 조금 기운 배도 여전히 배로 알아볼 수 있다.

    이 기법들이 모두 쓸 만한 학습 예제를 만드는 까닭은 그 변환이 물체의 근본적인 정체를 바꾸지 않고 위치, 방향, 조명 같은 부수적인 성질만 바꾸기 때문이다.

## 정리하며

**다룬 것** — CIFAR-10 데이터셋 시각화

MNIST에서 CIFAR-10으로 넘어가면 컴퓨터 비전의 근본적인 어려움 몇 가지가 드러난다.

앞의 연습문제 3개로 직접 확인할 수 있다.
