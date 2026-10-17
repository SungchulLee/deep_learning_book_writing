# 데이터로더 변환

이미지 transform과 레이블 target_transform을 Dataset에서 적용한다. 원핫 목표는 로스 함수의 입력 규약에 맞춰 선택하며 항상 필요한 것은 아니다.

## 1. 코드

```python
"""
변환을 쓰는 DataLoader — FashionMNIST
==========================================
``Lambda``를 쓰는 ``target_transform``으로 정수 레이블을 원-핫 텐서로 바꾸고,
이미지에는 ``ToTensor``를 함께 쓰는 법을 보인다.

사용법
-----
    python dataloader_transforms.py
"""

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Lambda, ToTensor

# ========================================================================
# 메인
# ========================================================================

LABELS_MAP = {
    0: "T-Shirt", 1: "Trouser", 2: "Pullover", 3: "Dress",
    4: "Coat",    5: "Sandal",  6: "Shirt",    7: "Sneaker",
    8: "Bag",     9: "Ankle Boot",
}


def load_dataloader():
    one_hot = Lambda(
        lambda y: torch.zeros(10, dtype=torch.float).scatter_(
            0, torch.tensor(y), value=1
        )
    )

    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
        target_transform=one_hot,
    )
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
        target_transform=one_hot,
    )

    train_loader = DataLoader(train_data, batch_size=20, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=20, shuffle=True)
    return train_loader, test_loader


def draw_batch(dataloader):
    images, labels = next(iter(dataloader))

    fig, axes = plt.subplots(2, 10, figsize=(12, 3))
    for ax, img, label in zip(axes.reshape(-1), images, labels):
        ax.imshow(img.squeeze(), cmap="binary")
        # 원-핫 벡터에서 클래스 인덱스 찾기
        class_idx = label.argmax().item()
        ax.set_title(LABELS_MAP[class_idx])
        ax.axis("off")
    plt.show()


def main():
    train_loader, _ = load_dataloader()
    draw_batch(train_loader)

    # 첫 표본의 원-핫 레이블 보이기
    images, labels = next(iter(train_loader))
    print(f"Image shape:  {images[0].shape}")
    print(f"Label (one-hot): {labels[0]}")
    print(f"Class index:  {labels[0].argmax().item()}")
    print(f"Class name:   {LABELS_MAP[labels[0].argmax().item()]}")


if __name__ == "__main__":
    main()
```

**출력:**

```
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz
Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz
Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw

Image shape:  torch.Size([1, 28, 28])
Label (one-hot): tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.])
Class index:  5
Class name:   Sandal
```

## 2. 논의

이 예제는 ToTensor로 이미지를 바꾸고 Lambda의 scatter_로 정수 레이블을 10차원 원핫 벡터로 만든다. 배치 목표는 `(B, 10)` 부동소수점 텐서가 된다. 인덱스 목표를 사용하는 CrossEntropyLoss라면 원래 long 인덱스를 유지해도 된다.

정규화는 채널별 정한 평균을 빼고 표준편차로 나눈다. 데이터셋 통계량이 근사값이거나 새로운 분포라면 각 이미지나 배치의 평균 0·표준편차 1을 보장하지는 않는다.

무작위 어그멘테이션은 학습에 사용하고 평가에서는 정한 전처리를 유지한다. 이 코드의 테스트 DataLoader도 shuffle=True이지만, 전체 표본의 집계 정확도는 순서에 의존하지 않아야 한다. 표본별 예측을 저장할 때는 인덱스 대응을 유지해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
Normalize((0.2860,), (0.3530,))가 모든 이미지의 평균을 정확히 0으로 만드는가?

</div>

??? success "풀이"
    아니다. 모든 픽셀에 같은 채널 통계량을 사용한다. 그 통계량이 맞는 데이터 분포의 전체 채널 평균과 분산을 조정하며 개별 이미지의 통계량은 다를 수 있다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
정답 0.9, 오답 각각 0.1/9의 스무딩과 PyTorch label_smoothing=0.1의 차이를 설명하라.

</div>

??? success "풀이"
    전자는 정답 밖에만 질량을 분배한다. PyTorch의 균등 혼합은 정답 0.91, 다른 9개에 각각 0.01이다. 둘 다 합은 1이지만 같은 계수에서 수치는 다르다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
4~8 픽셀 크기의 직사각형을 만들 때 높이와 너비를 어떻게 정하는가?

</div>

??? success "풀이"
    높이와 너비를 각각 별도 randint 호출로 정한다. 정수 하나를 반환하는 호출을 h, w 두 변수에 바로 풀어 넣을 수 없다. 이미지 범위 안의 좌표를 정하고 원본 보존이 필요하면 복사본을 수정한다.

## 정리하며

이미지와 목표의 변환을 따로 정의하고 로스가 받는 모양·자료형에 맞춘다. 고정 통계 정규화와 표본별 정규화를 구분하며 학습과 평가 전처리를 기록한다.
