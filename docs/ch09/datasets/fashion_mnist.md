# 패션 MNIST

FashionMNIST의 28×28 회색조 이미지를 CNN으로 분류하는 예제이다. 데이터 변환, 배치 처리와 학습·평가 모드의 역할을 연결한다.

## 1. 코드

```python
"""
FashionMNIST 분류
============================
배치 정규화와 드롭아웃을 선택적으로 쓰는 LeNet 방식 CNN으로 FashionMNIST
데이터셋을 학습시키는 파이프라인.

사용법
-----
    python fashion_mnist.py
    python fashion_mnist.py --epochs 5 --batchnorm True --dropout True
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

parser = argparse.ArgumentParser(description="FashionMNIST")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--test-batch-size", type=int, default=1000)
parser.add_argument("--batchnorm", type=bool, default=False)
parser.add_argument("--dropout", type=bool, default=False)
parser.add_argument("--dropout-prob", type=float, default=0.5)
parser.add_argument("--scheduler", type=bool, default=False)
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
    trainset = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    testset = datasets.FashionMNIST(
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
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.batchnorm1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(256, 120)
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
            loss = loss_ftn(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0
        if ARGS.scheduler:
            scheduler.step()


def compute_accuracy(model, testloader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(ARGS.device), labels.to(ARGS.device)
            _, predicted = torch.max(model(images).data, 1)
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
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    show_batch_or_ten_images(testloader, model)
    train(model, trainloader, loss_ftn, optimizer, scheduler)
    show_batch_or_ten_images(testloader, model)

    torch.save(model.state_dict(), ARGS.path)
    model = Net().to(ARGS.device)
    model.load_state_dict(torch.load(ARGS.path))
    compute_accuracy(model, testloader)


if __name__ == "__main__":
    main()
```

**출력:**

```
Accuracy on 10 000 test images: 69 %
  class     0: 72.7 %
  class     1: 88.3 %
  class     2: 66.8 %
  class     3: 78.5 %
  class     4: 18.0 %
  class     5: 87.6 %
  class     6: 17.4 %
  class     7: 84.2 %
  class     8: 91.1 %
  class     9: 86.4 %
```

## 2. 논의

합성곱 채널 수는 1→6→16이며 공간 크기는 28→24→12→8→4로 변한다. 마지막 특징은 16×4×4=256이고 완전 연결층 세 개로 10개 로짓을 만든다. 명령행 인수로 배치 정규화와 드롭아웃 사용 여부를 조절한다.

model.eval()은 드롭아웃을 끄고 기본 배치 정규화가 이동 통계량을 쓰게 한다. 배치 정규화 자체를 끄는 것은 아니다. 경사 기록을 끄는 no_grad와도 구분한다.

클래스별 정확도는 해당 클래스 표본 중 맞힌 비율이다. 어떤 클래스 쌍을 혼동하는지는 혼동 행렬이나 실제 오분류를 확인해야 한다. 특정 클래스가 항상 쉽거나 어렵다고 단정하지 않고, 테스트 결과로 설정을 조정할 때는 검증 분할을 별도로 마련한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
합성곱 뒤 평탄화 차원이 정해지는 과정을 설명하라.

</div>

??? success "연습문제 1 풀이"
    채널과 공간 크기를 각각 추적한다. 이 예제에서는 공간 크기가 28→24→12→8→4이고 마지막 특징은 16×4×4=256이고 완전 연결층 세 개로 10개 로짓을 만든다. 배치 차원은 유지한다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
평가 모드가 배치 정규화와 드롭아웃에 미치는 영향을 설명하라.

</div>

??? success "연습문제 2 풀이"
    드롭아웃은 항등 동작을 하고 기본 배치 정규화는 누적 이동 통계량을 사용한다. 모델을 학습 모드에 둔 채 경사만 끄면 이 동작은 바뀌지 않는다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
클래스별 정확도만으로 혼동하는 클래스 쌍을 알 수 있는가?

</div>

??? success "연습문제 3 풀이"
    아니다. 클래스별 정답 비율은 오답이 어느 클래스로 갔는지 보여 주지 않는다. 실제·예측 클래스의 교차표와 표본 수를 함께 확인한다.

## 정리하며

입력 채널과 공간 크기를 추적하면 평탄화 뒤 선형층 차원을 확인할 수 있다. 평가에서는 모드와 경사 기록을 각각 설정하고, 클래스별 지표의 의미와 모델 선택용 검증을 구분한다.
