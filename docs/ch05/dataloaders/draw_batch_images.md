# 배치 이미지 그리기

학습 데이터의 배치를 눈으로 확인하는 것은 어떤 딥러닝 파이프라인에서도 매우 중요한 디버깅 단계이다. 표본 이미지를 레이블과 함께 격자로 보여 주면, 학습에 시간을 들이기 전에 데이터 적재, 변환, 레이블 대응이 옳은지 확인할 수 있다. 이 스크립트는 FashionMNIST로 배치 시각화를 보인다.

## 코드

```python
"""
배치 이미지 그리기 — FashionMNIST
=================================
FashionMNIST에서 배치 하나를 불러와 표본 이미지 20장을 2 × 10 격자에
레이블과 함께 보여 준다.

사용법
-----
    python draw_batch_images.py
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


def draw_batch(dataloader):
    images, labels = next(iter(dataloader))

    fig, axes = plt.subplots(2, 10, figsize=(12, 3))
    for ax, img, label in zip(axes.reshape(-1), images, labels):
        ax.imshow(img.squeeze(), cmap="binary")
        ax.set_title(LABELS_MAP[label.item()])
        ax.axis("off")
    plt.show()


def main():
    train_loader, _ = load_dataloader()
    draw_batch(train_loader)


if __name__ == "__main__":
    main()```

## 논의

배치 시각화는 딥러닝 과제를 시작할 때의 기본 점검 구실을 한다. 이미지와 레이블의 격자를 살펴보면 잘못된 정규화, 어긋난 레이블, 깨진 이미지 적재, 뜻밖의 데이터 분포 같은 문제를 학습을 조용히 망치기 전에 잡아낼 수 있다.

`FashionMNIST` 데이터셋은 MNIST와 같은 28x28 회색조 형식을 쓰지만 숫자 대신 옷가지를 담고 있다. 레이블 대응(0=T-Shirt, 1=Trouser 등)은 눈으로 확인해야 한다. 전처리 파이프라인의 레이블 오류는 손실값만으로는 알아채기 어렵기 때문이다.

`matplotlib` 격자 표시는 회색조 이미지에 이진 색지도를 쓰는 `imshow`를 이용한다. DataLoader에 `shuffle=True`을 두면 그릴 때마다 데이터의 무작위 부분집합이 나오므로 클래스의 다양성과 이미지의 품질을 고르게 볼 수 있다.

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

