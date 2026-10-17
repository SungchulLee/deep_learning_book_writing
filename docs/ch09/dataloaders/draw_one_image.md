# 이미지 한 장 그리기

DataLoader에서 배치 하나를 꺼내 그 안의 첫 FashionMNIST 이미지를 표시한다. 데이터셋 인덱스와 배치 내부 인덱스를 구분한다.

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
    main()
```

## 2. 논의

ToTensor가 만든 이미지와 정수 레이블을 DataLoader가 배치로 묶는다. shuffle=True는 접근 순서를 무작위로 바꾸지만 같은 표본이 반환하는 이미지와 레이블의 대응을 바꾸지는 않는다.

회색조는 `(H, W)`, RGB는 `(H, W, C)` 형식으로 matplotlib에 전달한다. GPU 텐서나 경사를 추적하는 텐서를 표시하려면 필요에 따라 detach().cpu()로 바꾼다. 입력을 정규화했다면 표시 전에 역변환하거나 색 범위를 명시한다.

시각화는 잘못된 전처리와 레이블을 찾는 데 유용하지만 일부 표본을 봤다고 전체 데이터 품질이나 균형이 검증되지는 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
images[0]과 labels[0]의 모양과 표시 방법을 설명하라.

</div>

??? success "풀이"
    이미지는 `(1, 28, 28)`이고 레이블은 스칼라 텐서이다. 이미지의 채널 축만 제거해 표시하고 레이블은 item()으로 정수로 바꾼다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
shuffle=True이면 한 배치에 모든 클래스가 고르게 들어오는가?

</div>

??? success "풀이"
    아니다. 일반 셔플은 무작위 순서를 정할 뿐 클래스별 개수를 맞추지 않는다. 여러 배치와 전체 클래스 수를 따로 확인한다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
RGB 이미지 표시로 확장할 때 어떤 축 변환이 필요한가?

</div>

??? success "풀이"
    일반적인 CHW 텐서는 permute(1, 2, 0)로 HWC로 바꾼다. squeeze만으로 채널 순서를 바꿀 수 없으며, 무조건 모든 크기 1인 축을 제거하면 배치·공간 축도 사라질 수 있다.

## 정리하며

배치 시각화는 모양·값 범위·레이블 대응을 확인하는 도구이다. 채널 축과 배치 축을 구분하고 일부 표본 관찰과 전체 데이터 검증의 차이를 유지한다.
