# MNIST

MNIST의 28×28 회색조 이미지를 CNN으로 분류하는 예제이다. 데이터 변환, 배치 처리와 학습·평가 모드의 역할을 연결한다.

## 1. 코드

```python
"""
MNIST 분류
====================
배치 정규화와 드롭아웃을 선택적으로 쓰는 간단한 2레이어 CNN으로 MNIST 손글씨
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
    main()
```

??? note "전체 출력 (137줄)"

    ```
    [1,   100] loss: 1.2268413
    [1,   200] loss: 0.6762528
    [1,   300] loss: 0.5423621
    [1,   400] loss: 0.3955110
    [1,   500] loss: 0.3451793
    [1,   600] loss: 0.2722260
    [1,   700] loss: 0.2495641
    [1,   800] loss: 0.2336635
    [1,   900] loss: 0.1787234
    [2,   100] loss: 0.1616425
    [2,   200] loss: 0.1707605
    [2,   300] loss: 0.1476284
    [2,   400] loss: 0.1317965
    [2,   500] loss: 0.1430719
    [2,   600] loss: 0.1194900
    [2,   700] loss: 0.1308619
    [2,   800] loss: 0.1342847
    [2,   900] loss: 0.1086862
    [3,   100] loss: 0.1064620
    [3,   200] loss: 0.1040552
    [3,   300] loss: 0.0994132
    [3,   400] loss: 0.0862094
    [3,   500] loss: 0.0988619
    [3,   600] loss: 0.0963443
    [3,   700] loss: 0.0918163
    [3,   800] loss: 0.0987642
    [3,   900] loss: 0.0817158
    [4,   100] loss: 0.0893050
    [4,   200] loss: 0.0867413
    [4,   300] loss: 0.0741176
    [4,   400] loss: 0.0735338
    [4,   500] loss: 0.0767195
    [4,   600] loss: 0.0756848
    [4,   700] loss: 0.0754038
    [4,   800] loss: 0.0842071
    [4,   900] loss: 0.0690014
    [5,   100] loss: 0.0724872
    [5,   200] loss: 0.0716603
    [5,   300] loss: 0.0712729
    [5,   400] loss: 0.0624583
    [5,   500] loss: 0.0673477
    [5,   600] loss: 0.0660317
    [5,   700] loss: 0.0700283
    [5,   800] loss: 0.0740323
    [5,   900] loss: 0.0593979
    [6,   100] loss: 0.0657551
    [6,   200] loss: 0.0623210
    [6,   300] loss: 0.0621727
    [6,   400] loss: 0.0507785
    [6,   500] loss: 0.0615643
    [6,   600] loss: 0.0630819
    [6,   700] loss: 0.0552831
    [6,   800] loss: 0.0651452
    [6,   900] loss: 0.0553009
    [7,   100] loss: 0.0608834
    [7,   200] loss: 0.0645298
    [7,   300] loss: 0.0582906
    [7,   400] loss: 0.0519960
    [7,   500] loss: 0.0591846
    [7,   600] loss: 0.0568298
    [7,   700] loss: 0.0571153
    [7,   800] loss: 0.0612060
    [7,   900] loss: 0.0528919
    [8,   100] loss: 0.0607193
    [8,   200] loss: 0.0600888
    [8,   300] loss: 0.0493189
    [8,   400] loss: 0.0489381
    [8,   500] loss: 0.0500094
    [8,   600] loss: 0.0516449
    [8,   700] loss: 0.0567348
    [8,   800] loss: 0.0576415
    [8,   900] loss: 0.0458774
    [9,   100] loss: 0.0539871
    [9,   200] loss: 0.0557084
    [9,   300] loss: 0.0490090
    [9,   400] loss: 0.0420257
    [9,   500] loss: 0.0533018
    [9,   600] loss: 0.0534333
    [9,   700] loss: 0.0510741
    [9,   800] loss: 0.0527045
    [9,   900] loss: 0.0453930
    [10,   100] loss: 0.0519041
    [10,   200] loss: 0.0477362
    [10,   300] loss: 0.0456561
    [10,   400] loss: 0.0433271
    [10,   500] loss: 0.0490685
    [10,   600] loss: 0.0475104
    [10,   700] loss: 0.0471566
    [10,   800] loss: 0.0540270
    [10,   900] loss: 0.0426829
    [11,   100] loss: 0.0562640
    [11,   200] loss: 0.0526303
    [11,   300] loss: 0.0487617
    [11,   400] loss: 0.0494923
    [11,   500] loss: 0.0458794
    [11,   600] loss: 0.0466035
    [11,   700] loss: 0.0497668
    [11,   800] loss: 0.0531775
    [11,   900] loss: 0.0467396
    [12,   100] loss: 0.0532734
    [12,   200] loss: 0.0509588
    [12,   300] loss: 0.0438266
    [12,   400] loss: 0.0471100
    [12,   500] loss: 0.0462306
    [12,   600] loss: 0.0484390
    [12,   700] loss: 0.0465859
    [12,   800] loss: 0.0532990
    [12,   900] loss: 0.0411312
    [13,   100] loss: 0.0494849
    [13,   200] loss: 0.0491107
    [13,   300] loss: 0.0457686
    [13,   400] loss: 0.0384080
    [13,   500] loss: 0.0432677
    [13,   600] loss: 0.0500087
    [13,   700] loss: 0.0474824
    [13,   800] loss: 0.0512351
    [13,   900] loss: 0.0427363
    [14,   100] loss: 0.0520116
    [14,   200] loss: 0.0492149
    [14,   300] loss: 0.0465975
    [14,   400] loss: 0.0385379
    [14,   500] loss: 0.0438415
    [14,   600] loss: 0.0512444
    [14,   700] loss: 0.0473516
    [14,   800] loss: 0.0501071
    [14,   900] loss: 0.0458168
    Accuracy on 10 000 test images: 98 %
      class     0: 99.5 %
      class     1: 99.6 %
      class     2: 98.8 %
      class     3: 99.2 %
      class     4: 98.9 %
      class     5: 99.1 %
      class     6: 98.5 %
      class     7: 98.7 %
      class     8: 98.9 %
      class     9: 97.8 %
    ```


## 2. 논의

컨볼루션 채널 수는 1→32→64이며 공간 크기는 28→26→24→12로 변한다. 마지막 컨볼루션 피처는 64×12×12=9216이고 완전 연결 레이어 두 개로 10개 로짓을 만든다. 명령행 인수로 배치 정규화와 드롭아웃 사용 여부를 조절한다.

model.eval()은 드롭아웃을 끄고 기본 배치 정규화가 이동 통계량을 쓰게 한다. 배치 정규화 자체를 끄는 것은 아니다. 경사 기록을 끄는 no_grad와도 구분한다.

클래스별 정확도는 해당 클래스 표본 중 맞힌 비율이다. 어떤 클래스 쌍을 혼동하는지는 혼동 행렬이나 실제 오분류를 확인해야 한다. 특정 클래스가 항상 쉽거나 어렵다고 단정하지 않고, 테스트 결과로 설정을 조정할 때는 검증 분할을 별도로 마련한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
컨볼루션 뒤 평탄화 차원이 정해지는 과정을 설명하라.

</div>

??? success "풀이"
    채널과 공간 크기를 각각 추적한다. 이 예제에서는 공간 크기가 28→26→24→12이고 마지막 컨볼루션 피처는 64×12×12=9216이고 완전 연결 레이어 두 개로 10개 로짓을 만든다. 배치 차원은 유지한다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
평가 모드가 배치 정규화와 드롭아웃에 미치는 영향을 설명하라.

</div>

??? success "풀이"
    드롭아웃은 항등 동작을 하고 기본 배치 정규화는 누적 이동 통계량을 사용한다. 모델을 학습 모드에 둔 채 경사만 끄면 이 동작은 바뀌지 않는다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
클래스별 정확도만으로 혼동하는 클래스 쌍을 알 수 있는가?

</div>

??? success "풀이"
    아니다. 클래스별 정답 비율은 오답이 어느 클래스로 갔는지 보여 주지 않는다. 실제·예측 클래스의 교차표와 표본 수를 함께 확인한다.

## 정리하며

입력 채널과 공간 크기를 추적하면 평탄화 뒤 선형 레이어 차원을 확인할 수 있다. 평가에서는 모드와 경사 기록을 각각 설정하고, 클래스별 지표의 의미와 모델 선택용 검증을 구분한다.
