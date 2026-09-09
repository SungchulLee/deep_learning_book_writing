# 템플릿 학습: 학습하지 않는 첫 분류기

이 장은 MNIST 하나로 네 걸음을 걷는다. 그 첫걸음은 학습을 전혀 하지 않는 모델이다. 무엇을 배우기 전에, 배우지 않고 어디까지 갈 수 있는지부터 보아 두자.

MNIST 손글씨 숫자를 분류하는 가장 단순한 방법은 이렇다. 클래스마다 학습 이미지를 모두 평균 내어 **템플릿**을 만들고, 새 이미지는 가장 가까운 템플릿의 클래스로 예측한다. 학습할 가중치도, 역전파도, 반복 루프도 없다. 데이터를 한 번 훑으면 끝난다.

이 방법이 시험 데이터에서 **82.03%**를 맞힌다.

이 페이지를 첫 예제로 두는 데에는 두 가지 이유가 있다.

첫째, 이 장에서 배울 도구가 여기 거의 다 나온다. 텐서 만들기, 인덱싱, 브로드캐스팅, 축약 연산, `DataLoader`, GPU로 옮기기까지 모두 쓰인다. 다만 아직 자동 미분과 최적화기는 쓰지 않는다.

둘째, 82.03%가 이 장의 바닥이 된다. 뒤따르는 세 걸음(2.2 선형 모델, 2.3 다층 퍼셉트론, 2.4 합성곱 신경망)은 저마다 생각을 하나씩 더하는데, 그 값어치는 모두 이 82%를 얼마나 끌어올리느냐로 매겨진다.

---

## 1. 핵심 개념

방법은 두 단계뿐이다.

1. **템플릿 만들기** — 클래스 $k$에 속하는 학습 이미지를 모두 평균 내어 템플릿 $\overline{x}_k$를 얻는다.
2. **가장 가까운 템플릿 고르기** — 새 이미지 $x$는 $\overline{x}_k$ 가운데 가장 가까운 것의 클래스로 예측한다.

---

## 2. 수학적 배경

클래스 $k$의 학습 표본 집합을 $\mathcal{D}_k$라 하면 템플릿은 그 평균이다.

$$
\overline{x}_k = \frac{1}{|\mathcal{D}_k|} \sum_{x \in \mathcal{D}_k} x
$$

분류는 제곱 거리를 최소화하는 클래스를 고르는 것이다.

$$
\widehat{y} = \operatorname*{arg\,min}_{k \in \{0,\ldots,9\}} \lVert x - \overline{x}_k \rVert^2
$$

### 사실은 선형 분류기이다

제곱을 전개하면 이 규칙의 정체가 드러난다.

$$
\lVert x - \overline{x}_k \rVert^2 = \lVert x \rVert^2 - 2\,x^{\top}\overline{x}_k + \lVert \overline{x}_k \rVert^2
$$

첫 항 $\lVert x \rVert^2$은 $k$와 무관하므로 어느 클래스를 고르든 똑같이 더해진다. 곧 최소화 문제에서는 지워도 된다. 남은 두 항의 부호를 뒤집으면 최대화 문제가 된다.

$$
\widehat{y} = \operatorname*{arg\,max}_{k} \left( x^{\top}\overline{x}_k - \frac{1}{2}\lVert \overline{x}_k \rVert^2 \right)
$$

이는 가중치 $w_k = \overline{x}_k$, 편향 $b_k = -\tfrac{1}{2}\lVert \overline{x}_k \rVert^2$인 선형 분류기와 정확히 같다. 곧 템플릿 학습은 **가중치를 경사 하강법으로 학습하는 대신 클래스 평균으로 고정해 둔 선형층** 하나인 셈이다. $\square$

이 사실이 82.03%의 성격을 말해 준다. 이 값은 "학습하지 않은 선형 분류기"가 픽셀 공간에서 얻는 성능이며, 여기서 더 올라가려면 가중치를 학습하거나(로지스틱 회귀) 공간 자체를 바꾸어야 한다(합성곱 신경망).

---

## 3. PyTorch 구현

```python
"""
템플릿 학습: 클래스마다 평균 이미지를 만들어 가장 가까운 것으로 분류한다.
신경망 학습도 역전파도 쓰지 않는다.
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# === 1. MNIST 불러오기 ===
# ToTensor는 0~255인 uint8 픽셀을 0~1 사이 실수로 바꾼다. 여기서는 이
# 스케일 조정이 결과에 영향을 주지 않는데, 모든 이미지에 같은 변환이
# 걸리므로 거리의 대소 관계가 바뀌지 않기 때문이다
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

# shuffle=False인 까닭은 이 방법에 학습 순서라는 것이 없어서다. 평균은
# 더하는 순서와 무관하므로 섞을 이유가 없다.
# 배치를 쓰는 것도 경사 때문이 아니라, 6만 장을 한꺼번에 메모리에
# 올리지 않으려는 것뿐이다
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

# === 2. 클래스마다 평균 이미지 만들기 ===
# 합과 개수를 따로 모아 마지막에 한 번 나눈다. 배치마다 평균을 내어
# 그 평균들을 다시 평균 내면 안 된다. 배치마다 클래스별 장수가 달라
# 잘못된 가중 평균이 되기 때문이다
digit_sums = torch.zeros(10, 28, 28)
digit_counts = torch.zeros(10)

for images, labels in train_loader:
    # (배치, 1, 28, 28) -> (배치, 28, 28). MNIST는 흑백이라 채널 축이
    # 1뿐이므로 squeeze로 없앤다
    images = images.squeeze(1)

    for digit in range(10):
        # 불리언 마스크로 이 배치 안에서 해당 숫자인 이미지만 고른다
        mask = labels == digit
        # dim=0으로 축약하면 선택된 이미지들이 픽셀별로 더해져 (28, 28)이
        # 된다. 선택된 것이 하나도 없어도 0으로 채운 (28, 28)이 나와
        # 오류가 나지 않는다
        digit_sums[digit] += images[mask].sum(dim=0)
        digit_counts[digit] += mask.sum()

# [:, None, None]로 (10,)을 (10, 1, 1)로 만들어 브로드캐스팅이 되게 한다.
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

클래스별 장수가 5421부터 6742까지 고르지 않다는 점에 주목하라. 평균을 내는 방식이라 이 불균형이 템플릿 자체를 왜곡하지는 않지만, 뒤에서 볼 클래스별 정확도에는 영향을 준다.

### 템플릿 열 장 그려 보기

```python
# 만들어진 템플릿을 눈으로 보는 것이 이 방법을 이해하는 가장 빠른 길이다.
# 흐릿한 숫자 열 개가 나오는데, 그 흐릿함이 곧 이 방법의 한계를 말해 준다.
# 같은 숫자라도 사람마다 쓰는 모양이 달라, 평균을 내면 그 차이가 뭉개진다
fig, axes = plt.subplots(2, 5, figsize=(10, 4))

for digit, ax in enumerate(axes.flat):
    ax.imshow(average_images[digit], cmap="gray")
    ax.set_title(str(digit))
    ax.axis("off")

plt.tight_layout()
plt.show()
```

### 가장 가까운 템플릿으로 분류하기

```python
def predict(images, templates):
    """가장 가까운 템플릿의 클래스를 반환한다.

    인수:
        images:    (배치, 1, 28, 28)
        templates: (10, 28, 28)
    반환값:
        (배치,) 모양의 예측 클래스
    """
    images = images.squeeze(1)

    # 브로드캐스팅으로 모든 (이미지, 템플릿) 쌍의 거리를 한 번에 계산한다.
    #   images[:, None]    -> (배치,  1, 28, 28)
    #   templates[None, :] -> (   1, 10, 28, 28)
    # 빼면 (배치, 10, 28, 28)이 되고, 픽셀 축 (2, 3)을 축약하면
    # (배치, 10)짜리 거리 표가 남는다.
    # 반복문으로 열 번 도는 것보다 훨씬 빠르지만, 배치가 크면
    # (배치, 10, 28, 28) 중간 텐서가 메모리를 꽤 차지한다
    distances = (
        images[:, None, :, :] - templates[None, :, :, :]
    ).square().sum(dim=(2, 3))

    # 제곱근을 씌우지 않는다. 제곱근은 단조 증가 함수라 최솟값의 위치를
    # 바꾸지 않으므로, 굳이 계산을 더할 이유가 없다
    return distances.argmin(dim=1)


# === 4. 시험 데이터로 평가하기 ===
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

아래에서 보면 꽤 높다. 열 클래스를 무작위로 찍으면 10%이므로, 평균 내기 하나로 그보다 여덟 배 넘게 잘한다. 학습할 매개변수가 하나도 없고 데이터를 한 번만 훑는데도 그렇다.

위에서 보면 한참 낮다. 같은 MNIST에서 로지스틱 회귀는 92% 언저리, 작은 합성곱 신경망은 99%를 넘긴다. 그 사이의 간격이 곧 학습이 벌어 주는 몫이다.

간격이 생기는 까닭은 픽셀 공간에서 거리를 재기 때문이다.

- **한 클래스 안의 여러 모양을 평균이 뭉갠다.** 1을 곧게 쓰는 사람과 비스듬히 쓰는 사람의 이미지를 평균 내면 어느 쪽과도 닮지 않은 흐릿한 자국이 남는다.
- **평행 이동과 회전에 약하다.** 같은 숫자라도 두 픽셀만 옆으로 밀리면 제곱 거리가 크게 뛴다. 픽셀을 위치별로 대응시켜 비교하기 때문이다.
- **획의 굵기에 휘둘린다.** 굵게 쓴 이미지는 켜진 픽셀이 많아 어느 템플릿과도 거리가 멀어진다.

뒤따르는 세 걸음이 이 세 가지를 차례로 걷어낸다. [2.2 선형 모델](02_linear_softmax.md)은 가중치를 고정하는 대신 데이터에 맞추어 움직이고, [2.3 다층 퍼셉트론](03_mlp.md)은 층을 쌓아 한 클래스의 여러 필체를 따로 다루며, [2.4 합성곱 신경망](04_cnn.md)은 화소의 이웃 관계를 되찾아 평행 이동에 강해진다. 그 학습을 실제로 굴러가게 하는 도구(자동 미분, 경사 하강법)는 2.4절 뒤에 이어진다.

같은 아이디어를 배운 표현 위에서 되살린 것이 [원형 망](../../ch11/metric_learning/prototypical.md)이다. 클래스를 그 표본들의 평균으로 나타내고 가장 가까운 것을 고른다는 뼈대는 그대로 두고, 평균을 날것의 픽셀이 아니라 학습된 임베딩 공간에서 잡는다.

---

## 연습문제

**연습문제 1.**
2절에서 제곱 거리 규칙이 가중치 $\overline{x}_k$, 편향 $-\tfrac{1}{2}\lVert \overline{x}_k \rVert^2$인 선형 분류기와 같음을 보였다. 이 가중치와 편향을 `nn.Linear(784, 10)`에 직접 넣고, 그 층의 출력에 `argmax`를 취한 결과가 `predict`와 똑같은지 확인하라.

??? success "연습문제 1 풀이"
    템플릿을 펼쳐서 가중치에, 그 노름의 절반에 음수를 붙여 편향에 넣는다.
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
    둘이 정확히 같게 나온다. 곧 이 방법은 이미 선형 분류기이며, 다른 점은 가중치를 경사 하강법으로 학습하지 않고 클래스 평균으로 고정해 두었다는 것뿐이다. 로지스틱 회귀가 92% 언저리까지 올라가는 것은 같은 형태의 가중치를 데이터에 맞추어 움직였기 때문이다.

---

**연습문제 2.**
클래스별 정확도를 재어 어느 숫자가 가장 자주 틀리는지 찾아라. 그 숫자가 무엇과 혼동되는지 혼동 행렬로 확인하고, 템플릿 이미지를 보며 까닭을 설명하라.

??? success "연습문제 2 풀이"
    클래스마다 맞은 수와 전체 수를 따로 센다.
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
    클래스별 정확도는 다음과 같다.

    | 숫자 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
    |---|---|---|---|---|---|---|---|---|---|---|
    | 정확도 | 89.6% | 96.2% | 75.7% | 80.6% | 82.6% | 68.6% | 86.3% | 83.3% | 73.7% | 80.7% |

    1(96.2%)과 0(89.6%)이 가장 잘 맞고, 5(68.6%), 8(73.7%), 2(75.7%)가 가장 자주 틀린다.

    혼동 행렬을 보면 5와 8 모두 **3으로 가장 많이 흘러가고, 그다음이 1**이다. 3으로 가는 것은 짐작대로다. 3, 5, 8의 평균 이미지는 모두 가운데가 굵고 위아래가 둥근 비슷한 자국이라 픽셀 위치로만 비교하면 서로 가깝다.

    1로 흘러가는 것은 뜻밖으로 보이지만 2절의 식이 까닭을 말해 준다. 고르는 규칙은 $x^{\top}\overline{x}_k - \tfrac{1}{2}\lVert \overline{x}_k \rVert^2$을 최대화하는 $k$였다. 1의 평균 이미지는 가는 세로줄뿐이라 켜진 픽셀이 적고, 따라서 $\lVert \overline{x}_1 \rVert^2 = 28.84$로 열 클래스 가운데 가장 작다(가장 큰 0은 68.45다). 곧 편향 항의 벌점이 가장 가벼워, 어느 이미지에나 1이 만만한 후보로 남는다. 클래스마다 템플릿의 크기가 다르면 이런 편향이 생긴다.

---

**연습문제 3.**
평균 대신 클래스별 **중앙값** 이미지를 템플릿으로 삼아 보라. 정확도가 오르는가 내리는가? 왜 그런지 설명하라.

??? success "연습문제 3 풀이"
    픽셀마다 중앙값을 구하려면 클래스별 이미지를 모두 쌓아 두어야 한다.
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
    정확도가 82.03%에서 **76.59%**로 떨어진다. 중앙값은 이상치에 덜 흔들린다는 장점이 있지만, MNIST 픽셀은 대부분 0이라 클래스에 따라서는 픽셀 절반 이상이 0이어서 중앙값이 0으로 눌린다. 그러면 템플릿이 지나치게 성겨져 획의 가장자리 정보를 잃는다. 이상치가 문제가 아닌 데이터에서는 평균이 더 많은 정보를 담는다는 것을 보여 준다.

---

**연습문제 4.**
템플릿을 만들 때 클래스마다 학습 이미지를 $n$장만 쓰도록 바꾸고, $n = 1, 5, 10, 100, 1000$에 대해 정확도를 그려라. 이 곡선에서 무엇을 읽을 수 있는가?

??? success "연습문제 4 풀이"
    클래스마다 앞의 $n$장만 모아 평균을 낸다.
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

    | 클래스마다 장수 | 1 | 5 | 10 | 100 | 1000 | 전체(약 6000) |
    |---|---|---|---|---|---|---|
    | 정확도 | 50.99% | 62.26% | 66.48% | 76.92% | 81.00% | 82.03% |

    이미지 **한 장**만으로도 50%를 넘는다는 점이 먼저 눈에 띈다. 무작위로 찍는 10%의 다섯 배다. 그 뒤로 수익이 빠르게 줄어, $n=1000$과 전체(약 여섯 배 더 많은 데이터)의 차이는 1%포인트뿐이다.

    곧 이 방법을 막고 있는 것은 **데이터의 양이 아니다.** 아무리 부어도 82% 언저리에서 멈춘다. 막고 있는 것은 픽셀 공간에서 거리를 잰다는 사실이며, 그래서 다음 걸음이 필요해진다.

## 정리하며

**다룬 것** — 템플릿 학습

클래스마다 평균 이미지를 만들고 가장 가까운 것을 고르는 것만으로 MNIST에서 82.03%를 얻는다. 학습할 매개변수도, 역전파도, 반복 루프도 없다.

제곱 거리를 전개하면 이 규칙이 가중치를 클래스 평균으로 고정한 선형 분류기임이 드러난다. 그래서 이 값은 "학습하지 않은 선형 분류기"의 바닥이 되고, 로지스틱 회귀(92% 언저리)와 합성곱 신경망(99% 넘김)까지의 간격이 곧 학습이 벌어 주는 몫이다.

이 예제에는 이 장에서 쓰는 도구가 대부분 등장한다. 텐서 생성, 인덱싱, 브로드캐스팅, 축약 연산, `DataLoader`가 그것이다. 아직 쓰지 않은 것은 자동 미분과 경사 하강법뿐이며, 그 둘이 바로 다음 걸음부터 82%의 벽을 넘는 데 쓰인다. 앞의 연습문제 4개로 직접 확인할 수 있다.
