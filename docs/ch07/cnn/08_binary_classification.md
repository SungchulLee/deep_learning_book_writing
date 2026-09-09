# 이진 분류

이 스크립트는 MNIST 숫자에 대한 45가지 쌍별 이진 분류 문제를 모두 살피며, 숫자 쌍마다(0 대 1, 0 대 2, …, 8 대 9) 따로 CNN을 학습시킨다. 모든 쌍의 학습 곡선을 견주면 어떤 숫자들이 시각적으로 가장 비슷한지 알아내고 어떤 부류가 얽히느냐에 따라 분류의 어려움이 어떻게 달라지는지 이해할 수 있다.

## 1. 코드

```python
"""
08_binary_classification.py
============================
MNIST 이진 분류 분석

난이도: 고급
예상 시간: 2~3시간

지은이: PyTorch CNN 실습
날짜: 2025년 11월
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

print("=" * 70)
print("Binary Classification Analysis on MNIST")
print("=" * 70)


class BinaryCNN(nn.Module):
    """이진 분류를 위한 단순한 CNN."""
    def __init__(self):
        super(BinaryCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# MNIST 데이터 불러오기
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
all_images, all_labels = next(iter(train_loader))

# 숫자별로 데이터 정리
digit_images = {}
digit_labels = {}
for i in range(10):
    mask = all_labels == i
    digit_images[i] = all_images[mask]
    digit_labels[i] = all_labels[mask]


def train_binary(model, images_0, labels_0, images_1, labels_1, epochs=10):
    """두 부류로 학습시키고 손실의 자취를 돌려준다."""
    images = torch.cat([images_0, images_1])
    labels = torch.cat([torch.zeros(len(labels_0), dtype=torch.long),
                        torch.ones(len(labels_1), dtype=torch.long)])
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_trace = []
    for epoch in range(epochs):
        total_loss = 0
        for batch_images, batch_labels in loader:
            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_trace.append(total_loss / len(loader))
    return loss_trace


# 모든 쌍별 분류 실행
fig, axes = plt.subplots(10, 10, figsize=(15, 15))

for i in range(10):
    for j in range(10):
        if i >= j:
            axes[i, j].axis('off')
            continue
        print(f"Training classifier: {i} vs {j}")
        model = BinaryCNN()
        loss_trace = train_binary(
            model, digit_images[i], digit_labels[i],
            digit_images[j], digit_labels[j], epochs=10
        )
        axes[i, j].plot(loss_trace, 'r-', linewidth=2)
        axes[i, j].set_title(f'{i} vs {j}', fontsize=8)
        axes[i, j].set_ylim([0, 1])
        axes[i, j].axis('off')
        axes[j, i].plot(loss_trace, 'b-', linewidth=2)
        axes[j, i].set_title(f'{j} vs {i}', fontsize=8)
        axes[j, i].set_ylim([0, 1])
        axes[j, i].axis('off')

plt.suptitle('MNIST Binary Classification Learning Curves', fontsize=16)
plt.tight_layout()
plt.show()


if __name__ == "__main__":
    pass
```

## 2. 논의

10개 부류 문제를 45개의 이진 부분 문제로 쪼개면 분류 지형의 짜임이 드러난다. 이진 분류기마다 숫자 두 부류만으로 학습하는데, 쉬운 쌍에서는 거의 완벽한 정확도로 수렴하고 어려운 쌍에서는 애를 먹는다. 학습 곡선이 이를 곧바로 드러낸다. 빠른 수렴(손실이 재빨리 0 가까이 떨어짐)은 시각적으로 뚜렷이 다른 숫자를 뜻하고, 느리거나 불안정한 수렴은 시각적 유사성을 뜻한다.

0 대 1이나 1 대 7 같은 쉬운 쌍은 공간 구조가 근본적으로 다른 숫자, 곧 둥근 모양과 곧은 세로획을 다룬다. CNN은 학습 첫 몇 번의 반복에서 배운 단순한 특징만으로 이들을 가른다. 3 대 5나 4 대 9 같은 어려운 쌍은 굽거나 각진 획이 비슷하여, 특정 구간의 정확한 곡률 같은 더 미묘한 구별 특징을 배워야 한다.

이 분석은 다부류 분류기의 오류를 이해하는 데 실용적인 뜻이 있다. 10개 부류 모델이 "4"를 "9"로 잘못 볼 때, 그것은 이진 분석이 드러낸 바로 그 시각적 모호함을 반영한다. 쌍별 분해는 SVM 같은 몇몇 전통적인 분류기가 쓰는 일대일 전략과도 이어지는데, 거기서는 다부류 문제를 여러 이진 결정으로 줄인 뒤 투표로 합친다.

## 연습문제

**연습문제 1.**
학습 곡선에서 가장 쉬운 숫자 쌍 셋과 가장 어려운 쌍 셋을 찾아라. 어려운 쌍마다 어떤 시각적 특징 때문에 구별이 어려운지 서술하라.

??? success "연습문제 1 풀이"
    대체로 가장 쉬운 쌍은 **0 대 1**(원과 세로선), **1 대 8**(획 하나와 복잡한 고리), **0 대 7**(둥근 것과 각진 것)이다. 이들은 1~2세대 안에 손실이 거의 0으로 수렴한다.

    대체로 가장 어려운 쌍은 **3 대 5**(둘 다 위에 가로획이 있고 아래가 굽어 있으며, 가운데 곡선의 방향만 다르다), **4 대 9**(둘 다 오른쪽에 세로획이 있고 위쪽이 닫혔거나 거의 닫혀 있다), **7 대 9**(비스듬한 획의 모양이 비슷하고, 9의 닫힌 고리와 7의 열린 각이 핵심 차이이다)이다.

    어려운 쌍은 획의 방향, 차지하는 공간, 겹치는 화소 분포 같은 구조적 특징을 함께 지녀서, 특징 공간에서 결정 경계가 좁고 배우기 어렵다.

---

**연습문제 2.**
모델 45개를 따로 학습시키는 대신 하나의 모델로 쌍별 혼동 분석을 하는 방법을 설명하라. 두 방식에는 각각 어떤 장단점이 있는가?

??? success "연습문제 2 풀이"
    10개 부류 모델 하나면 한 번의 학습으로 $10 \times 10$ 혼동 행렬이 나온다. 성분 $(i, j)$은 $i$번 부류의 표본 가운데 몇 개를 $j$번 부류로 예측했는지 보여 준다. 이는 같은 쌍별 혼동 정보를 주되 공유된 특징 표현에서 얻는다.

    **모델 하나의 장점**: 학습을 한 번만 하면 되고(10~50배 빠르다), 공유된 표현이 비슷한 분류 과제 사이에 지식을 옮겨 주며, 실제 배포 상황과도 맞는다.

    **이진 모델 45개의 장점**: 모델마다 다른 부류의 간섭 없이 자기 쌍에 맞추어 최적화되고, 학습 곡선이 쌍마다의 본질적인 어려움을 따로 보여 주며, 부류에 걸친 불균형한 오류 전파에 흔들리지 않는다. 이진 방식은 모델 용량도 통제한다. 쌍마다 10개 부류에 나누어 쓰는 대신 모델의 온전한 용량을 쓴다.

---

**연습문제 3.**
45개 이진 분류기마다 마지막 시험 정확도를 계산하고, 성분 $(i, j)$이 숫자 $i$과 $j$을 가르는 시험 정확도를 나타내는 $10 \times 10$ 열지도를 만들어 분석을 넓혀라. 이 열지도에서 기대되는 무늬를 서술하라.

??? success "연습문제 3 풀이"
    ```python
    accuracy_matrix = np.zeros((10, 10))
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    test_images, test_labels = next(iter(test_loader))

    for i in range(10):
        for j in range(i + 1, 10):
            model = BinaryCNN()
            train_binary(model, digit_images[i], digit_labels[i],
                        digit_images[j], digit_labels[j], epochs=10)
            # 시험 집합에서 평가
            mask = (test_labels == i) | (test_labels == j)
            test_data = test_images[mask]
            test_labs = (test_labels[mask] == j).long()
            model.eval()
            with torch.no_grad():
                outputs = model(test_data)
                _, preds = torch.max(outputs, 1)
                acc = (preds == test_labs).float().mean().item() * 100
            accuracy_matrix[i, j] = acc
            accuracy_matrix[j, i] = acc
    ```
    열지도에서는 대부분의 성분이 99%를 넘고, 어려운 쌍(3-5, 4-9, 7-9, 3-8) 둘레에 정확도가 낮은 칸이 조금 모여 있다. 대각선은 정의되지 않는다(자기끼리는 견주지 않는다). 전체적으로는 거의 고르게 높은 정확도에 군데군데 움푹 팬 곳이 있는 무늬인데, 이는 대부분의 숫자 쌍은 손쉽게 갈리지만 몇몇은 정말로 모호한 특징을 함께 지님을 드러낸다.

## 정리하며

**다룬 것** — 이진 분류

10개 부류 문제를 45개의 이진 부분 문제로 쪼개면 분류 지형의 짜임이 드러난다.

핵심 클래스는 `BinaryCNN`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
