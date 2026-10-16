# 텐서로 만드는 데이터셋

TensorDataset으로 피처와 목표 텐서를 같은 인덱스에 묶어 선형 회귀를 학습한다. 메모리에 있는 텐서를 DataLoader로 배치 처리하는 기본 예제이다.

## 1. 코드

```python
"""
텐서로 만드는 데이터셋 — 선형 회귀
========================================
간단한 선형 회귀 과제에 TensorDataset과 DataLoader를 쓰는 법을 보인다.
참 모델은  y = 1 + 2x + ε,  ε ~ N(0, 0.1²) 이다.

사용법
-----
    python dataset_from_tensor.py
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
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# 전역 설정
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="dataset_from_tensor")
parser.add_argument("--lr", type=float, default=1e-1)
parser.add_argument("--epochs", type=int, default=1_000)
parser.add_argument("--batch_size", type=int, default=100)
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
ARGS.test_kwargs = {"batch_size": ARGS.batch_size}
if ARGS.use_cuda:
    cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
    ARGS.train_kwargs.update(cuda_kwargs)
    ARGS.test_kwargs.update(cuda_kwargs)

ARGS.path = "./model/model.pth"
os.makedirs("./model", exist_ok=True)

# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------


def load_data():
    x_train = np.random.uniform(size=(ARGS.batch_size, 1))
    x_test = np.random.uniform(size=(ARGS.batch_size, 1))

    y_train = 1 + 2 * x_train + np.random.normal(scale=0.1, size=(ARGS.batch_size, 1))
    y_test = 1 + 2 * x_test + np.random.normal(scale=0.1, size=(ARGS.batch_size, 1))

    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)

    trainloader = DataLoader(train_ds, **ARGS.train_kwargs)
    testloader = DataLoader(test_ds, **ARGS.test_kwargs)
    return trainloader, testloader


# ---------------------------------------------------------------------------
# 유틸리티
# ---------------------------------------------------------------------------


def plot_test_model(model, testloader):
    for x_batch, y_batch in testloader:
        pred = model(x_batch.to(ARGS.device))
        _, ax = plt.subplots(figsize=(12, 3))
        ax.plot(x_batch.squeeze().detach(), y_batch.squeeze().detach(), "k.", label="data")
        ax.plot(x_batch.squeeze().detach(), pred.detach().cpu(), "r-", label="pred")
        ax.legend()
        plt.show()
        break


# ---------------------------------------------------------------------------
# 학습
# ---------------------------------------------------------------------------


def train(model, loss_fn, opt, trainloader):
    model.train()
    w_trace, b_trace, loss_trace = [], [], []

    for _ in range(ARGS.epochs):
        for xb, yb in trainloader:
            xb, yb = xb.to(ARGS.device), yb.to(ARGS.device)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()

            w_trace.append(model.weight.item())
            b_trace.append(model.bias.item())
            loss_trace.append(loss.item())

    return np.array(w_trace), np.array(b_trace), np.array(loss_trace)


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------


def main():
    trainloader, testloader = load_data()

    model = nn.Linear(1, 1).to(ARGS.device)
    loss_fn = F.mse_loss
    opt = optim.SGD(model.parameters(), lr=ARGS.lr)

    w_trace, b_trace, loss_trace = train(model, loss_fn, opt, trainloader)

    _, axes = plt.subplots(1, 3, figsize=(12, 3))
    axes[0].plot(w_trace, label="estimated slope")
    axes[0].plot(2 * np.ones_like(w_trace), "--r", label="true slope")
    axes[1].plot(b_trace, label="estimated bias")
    axes[1].plot(np.ones_like(b_trace), "--r", label="true bias")
    axes[2].plot(loss_trace, label="loss")
    for ax in axes:
        ax.legend()
    plt.show()

    torch.save(model.state_dict(), ARGS.path)
    model = nn.Linear(1, 1).to(ARGS.device)
    model.load_state_dict(torch.load(ARGS.path))

    plot_test_model(model, testloader)


if __name__ == "__main__":
    main()
```

## 2. 논의

TensorDataset에 넣는 텐서들은 첫 축의 길이가 같아야 한다. 인덱스 하나에 대해 각 텐서의 같은 위치를 튜플로 반환하므로 피처와 목표의 대응을 유지할 수 있다.

학습은 MSE와 SGD로 선형 계수를 추정한다. 잡음과 표본 변동, 유한한 반복 때문에 데이터 생성의 계수 2와 절편 1에 정확히 일치할 필요는 없다. 미니배치 로스도 매 단계 단조 감소한다고 보장되지 않는다.

state_dict를 저장해 복원할 때는 같은 구조와 전처리로 새 모델을 만들어야 한다. 저장 성공과 모델의 통계적 적합성은 다른 검증이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
피처 텐서 길이가 100이고 목표 길이가 99이면 왜 사용할 수 없는가?

</div>

??? success "연습문제 1 풀이"
    각 인덱스에 대응하는 피처와 목표가 모두 있어야 한다. TensorDataset은 첫 차원 크기의 일치를 요구하므로 데이터 생성·필터링 단계에서 정렬과 길이를 맞춘다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
로스가 매 미니배치마다 줄지 않아도 학습 실패가 아닐 수 있는 이유를 설명하라.

</div>

??? success "연습문제 2 풀이"
    각 배치의 표본과 난이도가 달라 측정한 로스도 다르다. 전체 평가 모드의 로스와 추세를 함께 확인한다. 큰 진동이나 발산은 러닝레이트와 입력 스케일을 점검한다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
저장한 모델의 예측을 재현하는 데 필요한 항목을 설명하라.

</div>

??? success "연습문제 3 풀이"
    구조, 웨이트·버퍼, 입력의 자료형과 전처리, 평가 모드를 맞춘다. 학습 재개에는 옵티마이저와 반복 위치 등 추가 상태가 필요하다.

## 정리하며

TensorDataset은 첫 축이 정렬된 여러 텐서를 표본 단위로 묶는다. 회귀의 추정 오차와 학습 변동을 고려하고 웨이트 복원과 예측 재현의 조건을 구분한다.
