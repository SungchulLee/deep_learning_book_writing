# 이미지 한 장 그리기

데이터셋의 표본을 하나씩 살펴보면 데이터 적재와 전처리가 제대로 되는지 확인할 수 있다. 이 스크립트는 DataLoader로 FashionMNIST에서 이미지 한 장을 불러와 레이블과 함께 보여 준다. PyTorch 데이터 적재 파이프라인의 가장 간단한 예이다.

## 1. 코드

```python
"""
이미지 한 장 그리기 — FashionMNIST
==============================
DataLoader로 FashionMNIST를 불러와 표본 이미지 한 장을 레이블과 함께
보여 준다.

사용법
-----
    python draw_one_image.py
"""

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# ========================================================================
# 메인
# ========================================================================

LABELS_MAP = {
    0: "T-Shirt", 1: "Trouser", 2: "Pullover", 3: "Dress",
    4: "Coat",    5: "Sandal",  6: "Shirt",    7: "Sneaker",
    8: "Bag",     9: "Ankle Boot",
}


def load_dataloader():
    train_data = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )
    test_data = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )
    train_loader = DataLoader(train_data, batch_size=20, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=20, shuffle=True)
    return train_loader, test_loader


def draw_one_image(dataloader):
    images, labels = next(iter(dataloader))
    img, label = images[0], labels[0]

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(img.squeeze(), cmap="binary")
    ax.set_title(LABELS_MAP[label.item()])
    ax.axis("off")
    plt.show()


def main():
    train_loader, _ = load_dataloader()
    draw_one_image(train_loader)


if __name__ == "__main__":
    main()```

## 2. 논의

이미지 한 장을 띄워 보는 것은 가장 단순한 데이터 확인 방법이다. 이 스크립트는 최소한의 DataLoader 파이프라인을 보인다. `ToTensor()` 변환으로 데이터셋을 만들고, DataLoader로 감싸고, `next(iter(...))`으로 배치 하나를 꺼내고, 그 첫 표본을 고른다.

`images[0]`(모양이 `[1, 28, 28]`인 텐서)과 `labels[0]`(스칼라 텐서)의 차이를 구별하는 것이 중요하다. `squeeze()`은 표시를 위해 채널 차원을 없애고, `label.item()`은 스칼라 텐서를 사전의 키로 쓸 수 있는 파이썬 정수로 바꾼다.

이 방식은 더 복잡한 데이터셋으로도 자연스럽게 넓혀진다. RGB 이미지라면 `squeeze()`을 빼거나 `permute(1, 2, 0)`으로 PyTorch의 채널 우선 형식을 matplotlib의 채널 나중 형식으로 바꾼다.

## 연습문제

**연습문제 1.**
코드를 따라가며 쓰인 주요 자료 구조를 찾아라. 각각에 대해 자료형, (해당한다면) 모양, 파이프라인에서의 구실을 적어라.

??? success "연습문제 1 풀이"
    코드를 꼼꼼히 읽으며 변수 대입마다 살펴본다. 텐서는 `.shape`과 `.dtype`을 확인하고, 클래스는 `__init__`의 매개변수와 `forward`/`__call__`의 서명을 확인한다. 이름, 자료형, 모양, 구실을 열로 하는 표에 정리한다.

---


**연습문제 2.**
오류 처리와 입력 검증을 넣도록 코드를 고쳐라. 이 코드를 실전에 쓸 수 있게 하려면 어떤 검사를 더하겠는가?

??? success "연습문제 2 풀이"
    입력에 자료형 검사(`isinstance`), 모양 검증(`assert tensor.dim() == expected`), 값 범위 검사(예: 확률이 [0,1] 안인지)를 넣고, 입출력 연산은 try-except로 감싼다. 빈 배치나 NaN 같은 경계 상황에는 경고를 남긴다. 매개변수와 반환값의 자료형을 적은 독스트링을 붙인다.

---


**연습문제 3.**
직접 고른 새로운 쓰임새를 지원하도록 코드를 확장하라. 무엇을 왜 바꿀지 설명하라.

??? success "연습문제 3 풀이"
    알맞은 확장을 하나 고른다(예: 다른 데이터셋, 지표 추가, 새 모델 변형). 필요한 변경을 설명한다. 새 임포트, 클래스 정의 수정, 초매개변수 갱신, 새로운 시각화나 기록 등이다. 핵심 변경을 구현하고 간단한 시험으로 올바름을 확인한다.

## 정리하며

**다룬 것** — 이미지 한 장 그리기

이미지 한 장을 띄워 보는 것은 가장 단순한 데이터 확인 방법이다.

앞의 연습문제 3개로 직접 확인할 수 있다.
