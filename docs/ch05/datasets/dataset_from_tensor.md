# 텐서로 만드는 데이터셋

TensorDataset은 날 텐서를 PyTorch 데이터셋으로 감싸므로 합성 데이터나 미리 계산해 둔 데이터를 DataLoader로 흘려보내기 쉽다. 이 스크립트는 참 관계가 $y = 1 + 2x + \varepsilon$인 간단한 선형 회귀 파이프라인을 SGD로 학습시키고 참값과 나란히 그려 보인다.

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
    main()```

## 2. 논의

TensorDataset은 날 텐서로 PyTorch 데이터셋을 만드는 가장 간단한 방법이다. 여러 텐서(특징과 레이블)를 감싸고 인덱싱하면 그에 맞는 조각을 돌려주므로, 표 형식 데이터, 합성 실험, 데이터가 텐서로 메모리에 들어가는 모든 상황에 알맞다.

학습 루프는 반복에 따른 가중치, 편향, 손실을 기록하여 수렴을 눈으로 보여 준다. 참 모델이 $y = 1 + 2x + \varepsilon$이므로 학습된 매개변수는 대략 $w \approx 2$(기울기)과 $b \approx 1$(절편)으로 수렴해야 한다. 손실의 자취는 볼록 목적 함수에 대한 경사 하강법 특유의 지수적 감소를 보인다.

학습이 끝나면 `torch.save(model.state_dict(), path)`으로 모델을 저장하고 다시 불러와 제대로 저장되었는지 확인한다. 시험 집합 그림은 학습된 선형 함수가 데이터에 잘 맞음을 보여 주며, 예측을 잡음이 섞인 참값 위에 겹쳐 그린다.

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

**다룬 것** — 텐서로 만드는 데이터셋

TensorDataset은 날 텐서로 PyTorch 데이터셋을 만드는 가장 간단한 방법이다.

앞의 연습문제 3개로 직접 확인할 수 있다.
