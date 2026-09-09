# Fashion-MNIST 데이터셋 시각화

Fashion-MNIST는 고전적인 MNIST 데이터셋을 그대로 갈아 끼울 수 있는 요즘 판본으로, 손으로 쓴 숫자 대신 옷가지의 회색조 이미지를 쓴다. Zalando Research가 28×28 회색조 이미지와 10개 부류라는 편리한 형식을 지키면서도 더 까다로운 표준 자료를 주려고 만들었다. MNIST와 Fashion-MNIST의 차이를 이해하면 데이터셋의 복잡함이 모델 성능에 어떻게 영향을 주는지 알 수 있다.

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

trainloader, testloader = utils.load_data(
    train_kwargs, test_kwargs, fashion_mnist=True
)

sample_images, sample_labels = next(iter(trainloader))

# =============================================================================
# 3절: 시각화
# =============================================================================

fig, axes = plt.subplots(8, 8, figsize=(12, 12))
fig.suptitle('Fashion-MNIST Clothing Items Sample', fontsize=16)

for images, labels in trainloader:
    for ax, image, label in zip(axes.reshape(-1), images, labels):
        img_display = image.squeeze().cpu().numpy()
        img_display = img_display / 2 + 0.5
        ax.imshow(img_display, cmap="gray")
        ax.axis("off")
        class_name = FASHION_MNIST_LABELS[label.item()]
        ax.set_title(class_name, fontsize=8)
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

## 2. 논의

Fashion-MNIST는 MNIST와 텐서의 짜임이 같아 이미지마다 단일 채널 $28 \times 28$ 회색조이지만, 시각적으로는 훨씬 복잡하다. 손으로 쓴 숫자는 획의 모양이 뚜렷한 반면 옷가지는 같은 부류 안의 변화가 크다. 티셔츠 하나만 해도 방향과 질감과 맵시가 여러 가지이다. 그래서 Fashion-MNIST가 실제 분류의 어려움을 더 잘 흉내 낸다.

가장 헷갈리는 부류 쌍은 분류의 중요한 개념인 부류 사이의 유사성을 잘 보여 준다. 티셔츠(0번 부류)와 셔츠(6번 부류)는 윤곽이 비슷하고, 풀오버(2번)와 코트(4번)도 그렇다. 숫자 MNIST에서 99%에 이르는 구조라도 Fashion-MNIST에서 학습하면 대개 90~92%에 머문다. 이 7~9%p의 차이는 이룰 수 있는 성능을 모델 구조만이 아니라 데이터셋의 성질이 근본적으로 정한다는 것을 보여 준다.

실용적인 면에서 Fashion-MNIST는 MNIST와 똑같은 PyTorch API를 쓰므로 데이터를 불러오는 호출에서 깃발 하나만 바꾸면 된다. 이는 표준화된 데이터셋 인터페이스의 힘을 보여 준다. 연구자가 학습 코드를 고치지 않고 표준 자료를 갈아 끼울 수 있어 방법 사이의 공정한 비교가 가능해진다.

## 연습문제

**연습문제 1.**
Fashion-MNIST의 10개 부류마다 평균 화소 밝기를 계산하라. 평균 밝기가 가장 높은 부류와 가장 낮은 부류는 무엇인가? 그것이 물리적으로 말이 되는 까닭을 설명하라.

??? success "연습문제 1 풀이"
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
    보통 바지(1번 부류)가 가장 밝은데, 옷감 화소가 이미지의 넓은 부분을 덮기 때문이다. 샌들(5번 부류)이 가장 어두운 편인데, 작고 앞이 트인 물건이라 배경(검은) 화소가 많기 때문이다. 이는 옷의 물리적인 크기와 덮는 넓이를 그대로 드러낸다.

---

**연습문제 2.**
Fashion-MNIST 분류기의 혼동 행렬에서 티셔츠/윗옷(0)과 셔츠(6) 사이의 비대각 성분이 큰 까닭을 개념적으로 설명하라. CNN이 둘을 가르려면 어떤 시각적 특징을 배울 수 있겠는가?

??? success "연습문제 2 풀이"
    티셔츠와 셔츠는 전체 모양이 비슷하다. 둘 다 소매와 몸통이 있는 윗옷이다. 혼동이 생기는 까닭은 $28 \times 28$ 해상도에서는 깃의 모양, 단추의 자리, 소매의 길이 같은 잔 세부를 가려내기 어렵기 때문이다. CNN은 깃의 모양(티셔츠는 대개 둥근 목선이고 셔츠는 깃이 있다), 소매의 비율, 셔츠 앞섶이 트인 무늬 같은 특징을 잡아 둘을 가를 수 있다. 필터가 많은 더 깊은 신경망이 이런 미묘한 질감과 모양의 차이를 더 잘 붙잡는다.

---

**연습문제 3.**
데이터 증강(무작위 좌우 뒤집기와 작은 회전)이 Fashion-MNIST의 정확도를 높이는지 시험할 실험을 설계하라. 고친 변환 파이프라인을 적고 어떤 부류가 증강에서 가장 이득을 볼지 설명하라.

??? success "연습문제 3 풀이"
    ```python
    from torchvision import transforms

    augmented_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    ```
    옷은 대체로 좌우 대칭이므로 좌우 뒤집기는 거의 모든 부류에 안전하지만, 비대칭인 특징이 있는 부류에는 도움이 되지 않는다. 코트, 풀오버, 원피스처럼 같은 부류 안의 변화가 큰 것이 증강에서 가장 이득을 보는데, 더해진 변화가 여러 방향과 자세에 걸친 일반화를 돕기 때문이다. 가방은 이미 모양이 다양해 이득이 적을 수 있다. 평가의 일관성을 지키려면 증강 변환은 시험 집합이 아니라 학습 집합에만 적용해야 한다.

## 정리하며

**다룬 것** — Fashion-MNIST 데이터셋 시각화

Fashion-MNIST는 MNIST와 텐서의 짜임이 같아 이미지마다 단일 채널 $28 \times 28$ 회색조이지만, 시각적으로는 훨씬 복잡하다.

앞의 연습문제 3개로 직접 확인할 수 있다.
