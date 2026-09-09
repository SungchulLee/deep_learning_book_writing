# 본보기 배우기

원형 망을 만나기 앞서, 그 생각을 가장 벌거벗은 꼴로 해 보자. 부류마다 학습 그림을 모두 더해 평균을 낸 뒤, 새 그림은 가장 가까운 평균 그림의 부류로 매긴다. 신경망도, 역전파도, 배울 매개변수도 없다. 그런데도 MNIST에서 82%를 맞힌다.

이것은 [원형 망](prototypical.md)에서 묻힘 함수 $f_\theta$를 항등 함수로 둔 경우와 정확히 같다. 원형 망이 배운 묻힘 공간에서 부류 평균을 잡는다면, 여기서는 날것의 화소 공간에서 잡을 뿐이다. 그래서 이 쪽이 아래끝이 되고, 묻힘을 배워서 얻는 것이 무엇인지 재는 잣대가 된다.

---

## 1. 핵심 개념

생각은 두 걸음뿐이다.

1. **본보기 만들기** — 부류 $k$에 딸린 학습 그림을 모두 평균 내어 본보기 $\overline{x}_k$를 얻는다.
2. **가장 가까운 본보기 고르기** — 새 그림 $x$는 $\overline{x}_k$ 가운데 가장 가까운 것의 부류로 매긴다.

학습이라 부를 것이 평균 내기 하나뿐이라, 자료를 한 번만 훑으면 끝난다. 되풀이도 학습률도 없다.

---

## 2. 수학적 바탕

부류 $k$의 학습 표본 집합을 $\mathcal{D}_k$라 하면 본보기는 그 평균이다.

$$
\overline{x}_k = \frac{1}{|\mathcal{D}_k|} \sum_{x \in \mathcal{D}_k} x
$$

가려내기는 제곱 거리를 가장 작게 하는 부류를 고르는 것이다.

$$
\widehat{y} = \operatorname*{arg\,min}_{k \in \{0,\ldots,9\}} \lVert x - \overline{x}_k \rVert^2
$$

### 사실은 선형 가려내개다

제곱을 펼쳐 보면 이 규칙의 정체가 드러난다.

$$
\lVert x - \overline{x}_k \rVert^2 = \lVert x \rVert^2 - 2\,x^{\top}\overline{x}_k + \lVert \overline{x}_k \rVert^2
$$

첫 항 $\lVert x \rVert^2$은 $k$와 무관하므로 어느 부류를 고르든 똑같이 더해진다. 곧 가장 작게 하는 $k$를 찾는 일에서는 지워도 된다. 남은 두 항의 부호를 뒤집으면 가장 크게 하는 문제가 된다.

$$
\widehat{y} = \operatorname*{arg\,max}_{k} \left( x^{\top}\overline{x}_k - \frac{1}{2}\lVert \overline{x}_k \rVert^2 \right)
$$

이는 가중값 $w_k = \overline{x}_k$, 치우침 $b_k = -\tfrac{1}{2}\lVert \overline{x}_k \rVert^2$인 선형 가려내개와 똑같다. 곧 본보기 배우기는 **가중값을 경사 하강으로 배우는 대신 부류 평균으로 못박아 둔 선형 층** 하나인 셈이다. $\square$

이 사실이 아래끝의 성격을 말해 준다. 82%라는 값은 "가장 단순한 선형 가려내개가 화소 공간에서 얻을 수 있는 정도"이며, 여기서 더 올라가려면 가중값을 배우거나(로지스틱 회귀) 공간 자체를 바꾸어야 한다(원형 망).

---

## 3. PyTorch 구현

```python
"""
본보기 배우기: 부류마다 평균 그림을 만들어 가장 가까운 것으로 가려낸다.
신경망 학습도 역전파도 쓰지 않는다.
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# === 1. MNIST 불러오기 ===
# ToTensor는 0~255인 uint8 화소를 0~1 사이 실수로 옮긴다. 눈금을 맞추는
# 일이 여기서는 크게 중요하지 않은데, 모든 그림에 같은 변환이 걸려
# 거리의 대소 관계가 바뀌지 않기 때문이다
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# shuffle=False인 까닭은 이 방법에 학습 차례라는 것이 없어서다. 평균은
# 더하는 차례와 무관하므로 섞을 까닭이 없다.
# 배치를 쓰는 것도 경사 때문이 아니라, 6만 장을 한꺼번에 메모리에
# 올리지 않으려는 것뿐이다
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

# === 2. 부류마다 평균 그림 만들기 ===
# 합과 개수를 따로 모아 마지막에 한 번 나눈다. 배치마다 평균을 내어
# 그 평균들을 다시 평균 내면 안 된다. 배치마다 부류별 장수가 달라
# 잘못된 가중 평균이 되기 때문이다
digit_sums = torch.zeros(10, 28, 28)
digit_counts = torch.zeros(10)

for images, labels in train_loader:
    # (배치, 1, 28, 28) -> (배치, 28, 28). MNIST는 흑백이라 채널 축이
    # 1뿐이므로 눌러 없앤다
    images = images.squeeze(1)

    for digit in range(10):
        # 불리언 가리개로 이 배치 안에서 해당 숫자인 그림만 고른다
        mask = labels == digit
        # dim=0으로 더하면 뽑힌 그림들이 화소별로 더해져 (28, 28)이 된다.
        # 뽑힌 것이 하나도 없으면 0으로 채운 (28, 28)이 나와 탈이 없다
        digit_sums[digit] += images[mask].sum(dim=0)
        digit_counts[digit] += mask.sum()

# [:, None, None]로 (10,)을 (10, 1, 1)로 늘려 방송이 되게 한다.
# 이것 없이 나누면 모양이 맞지 않아 오류가 난다
average_images = digit_sums / digit_counts[:, None, None]

print(digit_counts)
print(average_images.shape)  # torch.Size([10, 28, 28])
```

**출력:**

```
tensor([5923., 6742., 5958., 6131., 5842., 5421., 5918., 6265., 5851., 5949.])
torch.Size([10, 28, 28])
```

부류별 장수가 5421부터 6742까지 고르지 않다는 점을 눈여겨보라. 그래도 평균을 내는 방식이라 이 치우침이 본보기 자체를 흔들지는 않는다. 다만 뒤에서 볼 부류별 정확도에는 영향을 준다.

### 본보기 열 장 그려 보기

```python
# 만들어진 본보기를 눈으로 보는 것이 이 방법을 이해하는 가장 빠른 길이다.
# 흐릿한 숫자 열 개가 나오는데, 그 흐릿함이 곧 이 방법의 한계를 말해 준다.
# 같은 숫자라도 사람마다 쓰는 꼴이 달라, 평균을 내면 그 차이가 뭉개진다
fig, axes = plt.subplots(2, 5, figsize=(10, 4))

for digit, ax in enumerate(axes.flat):
    ax.imshow(average_images[digit], cmap="gray")
    ax.set_title(str(digit))
    ax.axis("off")

plt.tight_layout()
plt.show()
```

### 가장 가까운 본보기로 가려내기

```python
def predict(images, templates):
    """가장 가까운 본보기의 부류를 돌려준다.

    인수:
        images:    (배치, 1, 28, 28)
        templates: (10, 28, 28)
    반환값:
        (배치,) 모양의 예측 부류
    """
    images = images.squeeze(1)

    # 방송으로 모든 (그림, 본보기) 짝의 거리를 한꺼번에 잰다.
    #   images[:, None]    -> (배치,  1, 28, 28)
    #   templates[None, :] -> (   1, 10, 28, 28)
    # 빼면 (배치, 10, 28, 28)이 되고, 화소 축 (2, 3)을 더해 없애면
    # (배치, 10)짜리 거리 표가 남는다.
    # 반복문으로 열 번 도는 것보다 훨씬 빠르지만, 배치가 크면
    # (배치, 10, 28, 28) 중간 텐서가 메모리를 꽤 먹는다
    distances = (
        images[:, None, :, :] - templates[None, :, :, :]
    ).square().sum(dim=(2, 3))

    # 제곱근을 씌우지 않는다. 제곱근은 단조 증가 함수라 가장 작은
    # 자리를 바꾸지 않으므로, 굳이 셈을 더할 까닭이 없다
    return distances.argmin(dim=1)


# === 4. 시험 자료로 따져 보기 ===
correct = 0
total = 0

for images, labels in test_loader:
    predictions = predict(images, average_images)

    correct += (predictions == labels).sum().item()
    total += labels.size(0)

accuracy = correct / total

print(f"Test accuracy: {accuracy:.2%}")
```

**출력:**

```
Test accuracy: 82.03%
```

---

## 4. 이 값을 어떻게 볼 것인가

82.03%는 두 방향에서 읽어야 한다.

아래에서 보면 꽤 높다. 열 부류를 마구 찍으면 10%이므로, 평균 내기 하나로 그보다 여덟 배 넘게 잘한다. 배울 매개변수가 하나도 없고 자료를 한 번만 훑는데도 그렇다.

위에서 보면 한참 낮다. 같은 MNIST에서 로지스틱 회귀는 92% 언저리, 작은 합성곱 신경망은 99%를 넘긴다. 그 사이의 간격이 곧 "배우는 일"이 벌어 주는 몫이다.

간격이 생기는 까닭은 화소 공간에서 거리를 재기 때문이다.

- **한 부류 안의 여러 꼴을 평균이 뭉갠다.** 1을 곧게 쓰는 사람과 비스듬히 쓰는 사람의 그림을 평균 내면 어느 쪽과도 닮지 않은 흐릿한 자국이 남는다.
- **밀고 돌리는 것에 약하다.** 같은 숫자라도 두 화소만 옆으로 밀리면 제곱 거리가 크게 뛴다. 화소를 자리별로 견주기 때문이다.
- **획의 굵기에 휘둘린다.** 굵게 쓴 그림은 켜진 화소가 많아 어느 본보기와도 거리가 멀어진다.

원형 망이 하는 일이 바로 이 셋을 없애는 것이다. 날것의 화소 대신 배운 묻힘 공간에서 평균을 잡으면, 같은 부류의 서로 다른 꼴이 그 공간에서는 가까이 모이도록 $f_\theta$가 학습된다. 곧 본보기 배우기는 원형 망에서 "어디서 평균을 잡을 것인가"라는 물음만 빼놓은 셈이다.

---

## 연습문제

**연습문제 1.**
2절에서 제곱 거리 규칙이 가중값 $\overline{x}_k$, 치우침 $-\tfrac{1}{2}\lVert \overline{x}_k \rVert^2$인 선형 가려내개와 같음을 보였다. 이 가중값과 치우침을 `nn.Linear(784, 10)`에 직접 넣고, 그 층의 출력에 `argmax`를 취한 결과가 `predict`와 똑같은지 확인하라.

??? success "연습문제 1 풀이"
    본보기를 펴서 가중값에, 그 노름의 절반에 음수를 붙여 치우침에 넣는다.
    ```python
    import torch.nn as nn

    layer = nn.Linear(784, 10)
    with torch.no_grad():
        # nn.Linear의 weight는 (출력, 입력) 모양이라 (10, 784)로 편다
        layer.weight.copy_(average_images.reshape(10, 784))
        layer.bias.copy_(-0.5 * average_images.reshape(10, 784).pow(2).sum(dim=1))

    images, labels = next(iter(test_loader))
    with torch.no_grad():
        linear_pred = layer(images.reshape(-1, 784)).argmax(dim=1)
    template_pred = predict(images, average_images)
    print(torch.equal(linear_pred, template_pred))  # True
    ```
    둘이 정확히 같게 나온다. 곧 이 방법은 이미 선형 가려내개이며, 다른 점은 가중값을 경사 하강으로 배우지 않고 부류 평균으로 못박아 두었다는 것뿐이다. 로지스틱 회귀가 92% 언저리까지 올라가는 것은 같은 꼴의 가중값을 자료에 맞추어 움직였기 때문이다.

---

**연습문제 2.**
부류별 정확도를 재어 어느 숫자가 가장 자주 틀리는지 찾아라. 그 숫자가 무엇과 헷갈리는지 혼동 표로 확인하고, 본보기 그림을 보며 까닭을 설명하라.

??? success "연습문제 2 풀이"
    부류마다 맞은 수와 전체 수를 따로 센다.
    ```python
    correct_per = torch.zeros(10)
    total_per = torch.zeros(10)
    confusion = torch.zeros(10, 10, dtype=torch.long)

    for images, labels in test_loader:
        pred = predict(images, average_images)
        for t, p in zip(labels, pred):
            confusion[t, p] += 1
            total_per[t] += 1
            correct_per[t] += (t == p)

    for d in range(10):
        print(f"{d}: {correct_per[d] / total_per[d]:.2%}")
    ```
    부류별 정확도는 다음과 같다.

    | 숫자 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
    |---|---|---|---|---|---|---|---|---|---|---|
    | 정확도 | 89.6% | 96.2% | 75.7% | 80.6% | 82.6% | 68.6% | 86.3% | 83.3% | 73.7% | 80.7% |

    1(96.2%)과 0(89.6%)이 가장 잘 맞고, 5(68.6%), 8(73.7%), 2(75.7%)가 가장 자주 틀린다.

    혼동 표를 보면 5와 8 모두 **3으로 가장 많이 흘러가고, 그다음이 1**이다. 3으로 가는 것은 짐작대로다. 3, 5, 8의 평균 그림은 모두 가운데가 굵고 위아래가 둥근 비슷한 자국이라 화소 자리로만 견주면 서로 가깝다.

    1로 흘러가는 것은 뜻밖으로 보이지만 2절의 식이 까닭을 말해 준다. 고르는 규칙은 $x^{\top}\overline{x}_k - \tfrac{1}{2}\lVert \overline{x}_k \rVert^2$을 가장 크게 하는 $k$였다. 1의 평균 그림은 가는 세로줄뿐이라 켜진 화소가 적고, 따라서 $\lVert \overline{x}_1 \rVert^2$이 열 부류 가운데 가장 작다. 곧 치우침 항의 벌이 가장 가벼워, 어느 그림에나 1이 만만한 후보로 남는다. 부류마다 본보기의 크기가 다르면 이런 치우침이 생긴다.

---

**연습문제 3.**
평균 대신 부류별 **중앙값** 그림을 본보기로 삼아 보라. 정확도가 오르는가 내리는가? 왜 그런지 설명하라.

??? success "연습문제 3 풀이"
    화소마다 중앙값을 잡으려면 부류별 그림을 모두 쌓아 두어야 한다.
    ```python
    by_digit = [[] for _ in range(10)]
    for images, labels in train_loader:
        images = images.squeeze(1)
        for d in range(10):
            by_digit[d].append(images[labels == d])

    median_images = torch.stack([
        torch.cat(by_digit[d]).median(dim=0).values for d in range(10)
    ])
    ```
    정확도가 82.03%에서 **76.59%**로 떨어진다. 중앙값은 튀는 값에 덜 흔들린다는 장점이 있지만, MNIST 화소는 대부분 0이라 부류에 따라서는 화소 절반 이상이 0이어서 중앙값이 0으로 눌린다. 그러면 본보기가 지나치게 성기어져 획의 가장자리 정보를 잃는다. 튀는 값이 문제가 아닌 자료에서는 평균이 더 많은 정보를 담는다는 것을 보여 준다.

---

**연습문제 4.**
본보기를 만들 때 부류마다 학습 그림을 $n$장만 쓰도록 바꾸고, $n = 1, 5, 10, 100, 1000$에 대해 정확도를 그려라. 이 곡선이 소수 예시 학습과 어떻게 이어지는가?

??? success "연습문제 4 풀이"
    부류마다 앞의 $n$장만 모아 평균을 낸다.
    ```python
    def build_templates(n_per_class):
        sums = torch.zeros(10, 28, 28)
        counts = torch.zeros(10)
        for images, labels in train_loader:
            images = images.squeeze(1)
            for d in range(10):
                if counts[d] >= n_per_class:
                    continue
                take = images[labels == d][: int(n_per_class - counts[d])]
                sums[d] += take.sum(dim=0)
                counts[d] += len(take)
        return sums / counts[:, None, None]
    ```
    | 부류마다 장수 | 1 | 5 | 10 | 100 | 1000 | 전체(약 6000) |
    |---|---|---|---|---|---|---|
    | 정확도 | 50.99% | 62.26% | 66.48% | 76.92% | 81.00% | 82.03% |

    그림 **한 장**만으로도 50%를 넘는다는 점이 먼저 눈에 띈다. 마구 찍는 10%의 다섯 배다. 그 뒤로 수익이 빠르게 줄어, 장수를 100배 늘린 $n=10$에서 $n=1000$ 사이가 겨우 15%포인트를 벌 뿐이고, $n=1000$과 전체(약 여섯 배 더 많은 자료)의 차이는 1%포인트뿐이다.

    이 곡선이 소수 예시 학습의 출발점이자 그 한계다. 부류 평균은 표본이 적어도 꽤 튼튼하게 잡히므로 새 부류를 보기 몇 장만으로 다룰 수 있다. 하지만 자료를 아무리 부어도 82% 언저리에서 멈춘다. 막고 있는 것은 표본 수가 아니라 화소 공간에서 거리를 잰다는 사실이기 때문이다.

    [원형 망](prototypical.md)이 손대는 곳이 바로 거기다. 평균을 잡는 방식은 그대로 두고 평균을 잡는 **자리**만 배운 묻힘 공간으로 옮겨, 이 곡선의 천장 자체를 끌어올린다.

## 정리하며

**다룬 것** — 본보기 배우기

부류마다 평균 그림을 만들고 가장 가까운 것을 고르는 것만으로 MNIST에서 82.03%를 얻는다. 배울 매개변수도, 역전파도, 되풀이도 없다.

제곱 거리를 펼쳐 보면 이 규칙이 가중값을 부류 평균으로 못박은 선형 가려내개임이 드러난다. 그래서 이 값은 "배우지 않은 선형 가려내개"의 아래끝이 되고, 로지스틱 회귀(92% 언저리)와 합성곱 신경망(99% 넘김)까지의 간격이 곧 배우는 일이 벌어 주는 몫이다.

[원형 망](prototypical.md)은 같은 뼈대에서 평균을 잡는 자리만 배운 묻힘 공간으로 옮긴 것이다. 앞의 연습문제 4개로 직접 확인할 수 있다.
