# MNIST

MNIST는 손글씨 숫자 분류의 고전적인 기준으로, 0부터 9까지의 숫자를 담은 회색조 이미지 70,000장으로 이루어진다. 이 스크립트는 드롭아웃과 배치 정규화를 선택할 수 있는 2층 CNN으로 완전한 학습 파이프라인을 제공하며 학습률 스케줄링과 클래스별 정확도 보고를 포함한 종합 평가까지 다룬다.

## 코드

```python
"""
MNIST 분류
====================
배치 정규화와 드롭아웃을 선택적으로 쓰는 간단한 2층 CNN으로 MNIST 손글씨
숫자 데이터셋을 학습시키는 완전한 파이프라인.

사용법
-----
    python mnist.py
    python mnist.py --epochs 5 --lr 0.01 --dropout True --batchnorm True
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

parser = argparse.ArgumentParser(description="MNIST")
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--epochs", type=int, default=14)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--test-batch-size", type=int, default=1000)
parser.add_argument("--batchnorm", type=bool, default=False)
parser.add_argument("--dropout", type=bool, default=True)
parser.add_argument("--dropout_prob_1", type=float, default=0.25)
parser.add_argument("--dropout_prob_2", type=float, default=0.5)
parser.add_argument("--scheduler", type=bool, default=True)
parser.add_argument("--gamma", type=float, default=0.7)
parser.add_argument("--dry_run", action="store_true", default=False)
parser.add_argument("--log_interval", type=int, default=100)
parser.add_argument("--cuda", action="store_true", default=True)
parser.add_argument("--mps", action="store_true", default=True)
parser.add_argument("--seed", type=int, default=1)
ARGS = parser.parse_args()

np.random.seed(ARGS.seed)
torch.manual_seed(ARGS.seed)

ARGS.use_cuda = ARGS.cuda and torch.cuda.is_available()
ARGS.use_mps = ARGS.mps and torch.backends.mps.is_available()
if ARGS.use_cuda:
    ARGS.device = torch.device("cuda")
elif ARGS.use_mps:
    ARGS.device = torch.device("mps")
else:
    ARGS.device = torch.device("cpu")

ARGS.train_kwargs = {"batch_size": ARGS.batch_size}
ARGS.test_kwargs = {"batch_size": ARGS.test_batch_size}
if ARGS.use_cuda:
    cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
    ARGS.train_kwargs.update(cuda_kwargs)
    ARGS.test_kwargs.update(cuda_kwargs)

ARGS.classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
ARGS.path = "./model/model.pth"
os.makedirs("./model", exist_ok=True)

# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------


def load_data():
    transform = transforms.ToTensor()
    trainset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, **ARGS.train_kwargs)
    testloader = DataLoader(testset, **ARGS.test_kwargs)
    return trainloader, testloader


# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(ARGS.dropout_prob_1)
        self.dropout2 = nn.Dropout(ARGS.dropout_prob_2)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        if ARGS.batchnorm:
            x = F.relu(self.batchnorm1(self.conv1(x)))
            x = self.pool(F.relu(self.batchnorm2(self.conv2(x))))
        else:
            x = F.relu(self.conv1(x))
            x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        if ARGS.dropout:
            x = F.relu(self.fc1(self.dropout1(x)))
            x = self.fc2(self.dropout2(x))
        else:
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# 유틸리티
# ---------------------------------------------------------------------------


def show_batch_or_ten_images(dataloader, model):
    for images, labels in dataloader:
        outputs = model(images.to(ARGS.device))
        _, predicted_labels = torch.max(outputs, 1)

        fig, axes = plt.subplots(1, min(ARGS.batch_size, 10), figsize=(12, 3))
        for i, (img, lab, pred) in enumerate(
            zip(images, labels, predicted_labels.cpu())
        ):
            img = img / 2 + 0.5
            img = np.transpose(img.numpy(), (1, 2, 0))
            axes[i].imshow(img, cmap="binary")
            axes[i].axis("off")
            axes[i].set_title(
                f"label: {ARGS.classes[lab]}\npred: {ARGS.classes[pred]}"
            )
            if i == 9:
                break
        plt.show()
        break


# ---------------------------------------------------------------------------
# 학습과 평가
# ---------------------------------------------------------------------------


def train(model, trainloader, loss_ftn, optimizer, scheduler):
    model.train()
    for epoch in range(ARGS.epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(ARGS.device), labels.to(ARGS.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_ftn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % ARGS.log_interval == ARGS.log_interval - 1:
                print(
                    f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / ARGS.log_interval:.7f}"
                )
                running_loss = 0.0
        if ARGS.scheduler:
            scheduler.step()
        if ARGS.dry_run:
            break


def compute_accuracy(model, testloader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(ARGS.device), labels.to(ARGS.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy on 10 000 test images: {100 * correct // total} %")

    correct_pred = {c: 0 for c in ARGS.classes}
    total_pred = {c: 0 for c in ARGS.classes}
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(ARGS.device), labels.to(ARGS.device)
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
    model = Net().to(ARGS.device)
    loss_ftn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=ARGS.lr, momentum=ARGS.momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=ARGS.gamma)

    show_batch_or_ten_images(testloader, model)
    train(model, trainloader, loss_ftn, optimizer, scheduler)
    show_batch_or_ten_images(testloader, model)

    torch.save(model.state_dict(), ARGS.path)
    model = Net().to(ARGS.device)
    model.load_state_dict(torch.load(ARGS.path))

    compute_accuracy(model, testloader)


if __name__ == "__main__":
    main()```

## 논의

이 MNIST 파이프라인은 실전에 쓸 만한 관행을 여럿 보인다. 초매개변수를 위한 명령행 인수 처리, 자동 장치 선택(CUDA, MPS, CPU), 설정 가능한 드롭아웃 비율, 계단식 감쇠를 쓰는 학습률 스케줄러이다.

이 CNN 구조는 합성곱 층 두 개(필터 32개와 64개) 뒤에 최댓값 풀링, 드롭아웃, 완전 연결층 두 개를 둔다. 평가할 때 `model.eval()`을 부르면 드롭아웃이 꺼져 예측이 결정적이 된다. 이를 부르지 않으면 예측이 확률적이어서 정확도 측정을 믿을 수 없다.

클래스별 정확도 보고서는 흔히 헷갈리는 숫자를 짚어 준다. MNIST에서는 4와 9, 3과 8처럼 획의 모양이 비슷한 숫자가 잘못 분류될 가능성이 크다. 이 분석은 학습 데이터를 더 모아야 할지, 구조를 바꾸어야 할지 판단하는 데 도움이 된다.

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

