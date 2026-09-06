# CIFAR-10

CIFAR-10은 10개 클래스에 걸친 컬러 이미지 60,000장을 담은, 널리 쓰이는 기준 데이터셋이다. 이 스크립트는 배치 정규화와 드롭아웃을 선택적으로 쓰는 LeNet 방식 CNN으로 완전한 학습 파이프라인을 구현한다. 데이터 적재, 모델 정의, 학습, 평가, 모델 저장에 이르는 딥러닝 작업 흐름 전체를 보인다.

## 코드

```python
"""
CIFAR-10 분류
=======================
배치 정규화와 드롭아웃을 선택적으로 쓰는 LeNet 방식 CNN으로 CIFAR-10 컬러
이미지 데이터셋을 학습시키는 파이프라인.

사용법
-----
    python cifar10.py
    python cifar10.py --epochs 5 --batchnorm True --dropout True
"""

import argparse
import os

# ========================================================================
# 메인
# ========================================================================

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# 전역 설정
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="CIFAR10")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--test_batch_size", type=int, default=1000)
parser.add_argument("--batchnorm", type=bool, default=False)
parser.add_argument("--dropout", type=bool, default=False)
parser.add_argument("--dropout-prob", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--cuda", action="store_true", default=True)
parser.add_argument("--mps", action="store_true", default=True)
ARGS = parser.parse_args()

np.random.seed(ARGS.seed)
torch.manual_seed(ARGS.seed)

ARGS.use_cuda = ARGS.cuda and torch.cuda.is_available()
ARGS.use_mps = ARGS.mps and torch.backends.mps.is_available()
if ARGS.use_cuda:
    ARGS.device = torch.device("cuda")
else:
    ARGS.device = torch.device("cpu")

ARGS.train_kwargs = {"batch_size": ARGS.batch_size, "shuffle": True, "num_workers": 2}
ARGS.test_kwargs = {"batch_size": ARGS.test_batch_size, "shuffle": False, "num_workers": 2}
if ARGS.use_cuda:
    cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
    ARGS.train_kwargs.update(cuda_kwargs)
    ARGS.test_kwargs.update(cuda_kwargs)

ARGS.classes = (
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
ARGS.path = "./model/model.pth"
os.makedirs("./model", exist_ok=True)

# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------


def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    testset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    trainloader = DataLoader(trainset, **ARGS.train_kwargs)
    testloader = DataLoader(testset, **ARGS.test_kwargs)
    return trainloader, testloader


# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------


class Net(nn.Module):
    def __init__(self, dropout_prob=ARGS.dropout_prob):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.batchnorm1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        if ARGS.batchnorm:
            x = self.pool(F.relu(self.batchnorm1(self.conv1(x))))
            x = self.pool(F.relu(self.batchnorm2(self.conv2(x))))
        else:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        if ARGS.dropout:
            x = self.dropout1(F.relu(self.fc1(x)))
            x = self.dropout2(F.relu(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ---------------------------------------------------------------------------
# 유틸리티
# ---------------------------------------------------------------------------


def show_batch_or_ten_images(dataloader, model):
    images, labels = next(iter(dataloader))
    outputs = model(images)
    _, predicted_labels = torch.max(outputs, 1)

    _, axes = plt.subplots(1, min(ARGS.batch_size, 10), figsize=(20, 3))
    for i, (img, lab, pred) in enumerate(zip(images, labels, predicted_labels)):
        img = img / 2 + 0.5
        img = np.transpose(img.numpy(), (1, 2, 0))
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(
            f"label: {ARGS.classes[lab]}\npred: {ARGS.classes[pred]}"
        )
        if i + 1 == min(ARGS.batch_size, 10):
            break
    plt.show()


# ---------------------------------------------------------------------------
# 학습과 평가
# ---------------------------------------------------------------------------


def train(model, trainloader, optimizer, loss_ftn):
    model.train()
    for epoch in range(ARGS.epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            loss = loss_ftn(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0


def compute_accuracy(model, testloader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in testloader:
            _, predicted = torch.max(model(images).data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy on 10 000 test images: {100 * correct // total} %")

    correct_pred = {c: 0 for c in ARGS.classes}
    total_pred = {c: 0 for c in ARGS.classes}
    with torch.no_grad():
        for images, labels in testloader:
            _, predictions = torch.max(model(images), 1)
            for lab, pred in zip(labels, predictions):
                if lab == pred:
                    correct_pred[ARGS.classes[lab]] += 1
                total_pred[ARGS.classes[lab]] += 1
    for cls, cnt in correct_pred.items():
        print(f"  class {cls:>5s}: {100 * cnt / total_pred[cls]:.1f} %")


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------


def main():
    trainloader, testloader = load_data()
    model = Net()
    loss_ftn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=ARGS.lr, momentum=ARGS.momentum)

    show_batch_or_ten_images(testloader, model)
    train(model, trainloader, optimizer, loss_ftn)
    show_batch_or_ten_images(testloader, model)

    torch.save(model.state_dict(), ARGS.path)
    model = Net()
    model.load_state_dict(torch.load(ARGS.path))
    compute_accuracy(model, testloader)


if __name__ == "__main__":
    main()```

## 논의

이 파이프라인은 PyTorch의 표준 학습 양식을 따른다. 데이터 변환을 정의하고, 데이터셋과 데이터 로더를 만들고, 모델을 정의하고, 손실 함수와 최적화기를 준비하고, 몇 에포크 동안 학습시킨 뒤 평가한다. LeNet 방식 CNN은 합성곱 층 두 개 뒤에 완전 연결층 세 개를 두며, 배치 정규화와 드롭아웃은 명령행 인수로 켜고 끈다.

배치 정규화는 미니배치 안에서 활성화를 정규화하여 내부 공변량 변화를 줄이고 더 높은 학습률을 쓸 수 있게 한다. 드롭아웃은 학습 중에 활성화를 무작위로 0으로 만들어 신경망이 중복된 표현을 배우게 하여 일반화를 개선한다. 평가할 때에는 `model.eval()`으로 두 기법을 모두 끈다.

끝에 나오는 클래스별 정확도 보고서는 모델이 어떤 범주를 어려워하는지 드러낸다. CIFAR-10에서는 "cat"과 "dog"처럼 모양과 결이 비슷한 범주가 자주 헷갈리는 반면, "ship"과 "airplane"처럼 뚜렷이 다른 범주는 정확도가 높은 편이다.

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

