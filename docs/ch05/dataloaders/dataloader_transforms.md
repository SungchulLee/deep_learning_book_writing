# 데이터로더 변환

데이터 변환은 데이터를 불러오는 동안 입력과 레이블에 적용하는 전처리 단계이다. PyTorch는 `torchvision.transforms`로 조립할 수 있는 변환을 제공하여 이미지를 텐서로 바꾸고, 화소값을 정규화하며, `Lambda`로 레이블을 원-핫 벡터로 바꾸는 일까지 할 수 있다. 변환을 알맞게 설정해야 데이터가 학습에 효과적인 형식과 척도를 갖춘다.

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

## 2. 논의

PyTorch 데이터셋의 `target_transform` 매개변수는 레이블이 학습 루프에 들어가기 전에 임의의 변환을 적용하게 해 준다. 여기서는 `Lambda`가 `scatter_`를 써서 정수 클래스 레이블을 10차원 원-핫 벡터로 바꾼다. `scatter_` 연산은 클래스에 해당하는 인덱스에 1을 놓고 나머지는 0으로 채운다.

원-핫 부호화는 손실 함수가 클래스에 대한 확률 분포를 받을 때(예: 부드러운 교차 엔트로피)나 레이블 평활화를 적용하고 싶을 때 쓸모 있다. 다만 PyTorch의 표준 `nn.CrossEntropyLoss`는 정수 클래스 인덱스를 바로 받으므로 원-핫 부호화가 언제나 필요하지는 않다.

이미지 변환(`ToTensor`, `Normalize`)과 레이블 변환을 데이터셋 정의 한 곳에 모아 두면 전처리 논리가 한데 모여 재현하기 쉬워진다. 무작위 잘라내기, 뒤집기, 색 흔들기 같은 증강을 이어 붙일 때 이 방식이 특히 힘을 발휘한다.

## 연습문제

**연습문제 1.**
FashionMNIST 이미지의 평균이 0, 표준편차가 1이 되도록 정규화하는 변환을 이미지 파이프라인에 넣어라. mean=0.2860, std=0.3530을 쓰라.

??? success "연습문제 1 풀이"
    `transform=ToTensor()`을 `transform=transforms.Compose([ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])`으로 바꾼다. `Normalize` 변환은 채널마다 평균을 빼고 표준편차로 나눈다.

---


**연습문제 2.**
딱딱한 원-핫 벡터 대신 평활화 계수가 $\alpha = 0.1$인 부드러운 레이블을 만들도록 목표 변환을 고쳐라.

??? success "연습문제 2 풀이"
    `Lambda(lambda y: torch.full((10,), 0.1/9).scatter_(0, torch.tensor(y), 1.0 - 0.1))`을 쓴다. 참 클래스에 $1 - \alpha = 0.9$을 놓고 나머지 9개 클래스에 $\alpha/(K-1) \approx 0.011$씩 나누어 준다.

---


**연습문제 3.**
이미지의 직사각형 영역을 무작위로 지우는(컷아웃 증강) 사용자 정의 변환 클래스를 작성하라. 지운 영역은 0으로 채우고 크기는 4x4에서 8x8 화소 사이에서 무작위로 정하라.

??? success "연습문제 3 풀이"
    `__call__(self, img)`을 갖는 클래스를 만들어 왼쪽 위 좌표 `(r, c)`와 크기 `(h, w)`을 `h, w = random.randint(4, 8)`으로 무작위로 정한다. `img[:, r:r+h, c:c+w] = 0`으로 두고 `img`을 돌려준다. 이 변환을 Compose 파이프라인의 `ToTensor()` 뒤에 넣는다.

## 정리하며

**다룬 것** — 데이터로더 변환

PyTorch 데이터셋의 `target_transform` 매개변수는 레이블이 학습 루프에 들어가기 전에 임의의 변환을 적용하게 해 준다.

앞의 연습문제 3개로 직접 확인할 수 있다.
