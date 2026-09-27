"""선형 모델 절의 주장 셋을 수로 확인한다.

    주장 1  찾아야 할 수가 7,850개다
    주장 2  1단계 템플릿 학습은 **이미 이 쪽의 모델**이었다 (수가 못박혀 있었을 뿐)
    주장 3  화소를 뒤섞어도 정확도가 달라지지 않는다 (= 이웃 관계를 안 쓴다)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms

SEED, BATCH, LR, EPOCHS = 42, 128, 1e-3, 10
MEAN, STD = 0.1307, 0.3081

# 자료를 한 번만 읽어 텐서로 펼쳐 둔다 (ToTensor만 — 정규화는 뒤에서 따로 건다)
raw = transforms.ToTensor()
tr_ds = torchvision.datasets.MNIST("./data", train=True, download=True, transform=raw)
te_ds = torchvision.datasets.MNIST("./data", train=False, download=True, transform=raw)


def flatten(ds):
    X = torch.stack([x for x, _ in ds]).reshape(len(ds), -1)   # (n, 784)
    y = torch.tensor([y for _, y in ds])
    return X, y


Xtr, ytr = flatten(tr_ds)
Xte, yte = flatten(te_ds)
print(f"자료  학습 {tuple(Xtr.shape)}  시험 {tuple(Xte.shape)}\n")


# ==========================================================================
# 주장 1 — 매개변수는 7,850개
# ==========================================================================
print("=" * 64)
print("주장 1  찾아야 할 수는 7,850개")
print("=" * 64)

model = nn.Linear(784, 10)
n_w = model.weight.numel()
n_b = model.bias.numel()
print(f"  가중치 A : {model.weight.shape[1]} x {model.weight.shape[0]} = {n_w:,}")
print(f"  편향   b : {n_b:,}")
print(f"  합       : {n_w + n_b:,}")
print(f"  torch가 세어 준 값: {sum(p.numel() for p in model.parameters()):,}")


# ==========================================================================
# 주장 2 — 템플릿 학습은 이미 선형 모델이었다
# ==========================================================================
print("\n" + "=" * 64)
print("주장 2  템플릿 학습 = 가중치가 못박힌 선형 모델")
print("=" * 64)

# 클래스마다 평균 이미지를 만든다. 이것이 '템플릿'이다.
means = torch.stack([Xtr[ytr == k].mean(0) for k in range(10)])   # (10, 784)

# (가) 가장 가까운 템플릿을 고르는 규칙 — 1단계가 한 일
d = torch.cdist(Xte, means) ** 2            # (10000, 10) 제곱 거리
pred_nc = d.argmin(1)
acc_nc = 100.0 * (pred_nc == yte).float().mean().item()

# (나) 같은 것을 선형 모델로 적는다.
#     ||x - m_k||^2 을 펴면 ||x||^2 - 2 x·m_k + ||m_k||^2 이고,
#     ||x||^2 은 k에 따라 변하지 않으므로 argmin 에서 빼도 된다. 남는 것은
#         argmax_k ( x·m_k - 0.5||m_k||^2 )
#     곧 A의 열이 m_k, b가 -0.5||m_k||^2 인 선형 분류기다.
A = means.T                                  # (784, 10)
b = -0.5 * (means ** 2).sum(1)               # (10,)
pred_lin = (Xte @ A + b).argmax(1)
acc_lin = 100.0 * (pred_lin == yte).float().mean().item()

print(f"  (가) 가장 가까운 템플릿 고르기      : {acc_nc:.2f}%")
print(f"  (나) A=평균, b=-0.5||평균||^2 인 선형 : {acc_lin:.2f}%")
print(f"  두 방법의 예측이 몇 개나 다른가      : {int((pred_nc != pred_lin).sum())}개")
print("\n  하나도 다르지 않다. 두 식이 같은 것이기 때문이다.")

# 정규화를 걸어도 템플릿 학습의 값이 바뀌지 않는지 확인한다.
#   (x - MEAN)/STD 는 모든 화소에 똑같이 걸리는 아핀 변환이므로
#   거리를 1/STD^2 배로 줄일 뿐, 어느 것이 가장 가까운지는 바꾸지 않는다.
Xtr_n, Xte_n = (Xtr - MEAN) / STD, (Xte - MEAN) / STD
means_n = torch.stack([Xtr_n[ytr == k].mean(0) for k in range(10)])
acc_nc_n = 100.0 * ((torch.cdist(Xte_n, means_n) ** 2).argmin(1) == yte).float().mean().item()
print(f"  정규화를 걸고 다시 재면              : {acc_nc_n:.2f}%  (같아야 한다)")


# ==========================================================================
# 주장 2 (이어서) — 같은 7,850개를 '학습하면' 얼마나 오르는가
# ==========================================================================
def train_linear(X, y, Xt, yt, seed=SEED, epochs=EPOCHS):
    """같은 nn.Linear(784, 10)을 자료에 맞추어 움직인다."""
    torch.manual_seed(seed)
    m = nn.Linear(784, 10)
    opt = optim.Adam(m.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    ld = DataLoader(TensorDataset(X, y), batch_size=BATCH, shuffle=True, generator=g)
    for _ in range(epochs):
        m.train()
        for xb, yb in ld:
            opt.zero_grad(); crit(m(xb), yb).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return 100.0 * (m(Xt).argmax(1) == yt).float().mean().item()


acc_trained = train_linear(Xtr_n, ytr, Xte_n, yte)
print(f"\n  같은 7,850개를 **학습하면**          : {acc_trained:.2f}%")
print(f"  못박아 두었을 때와의 차이            : {acc_trained - acc_nc:+.2f}%포인트")
print("\n  매개변수의 수는 그대로다. 값을 자료에 맞추어 움직인 몫이다.")


# ==========================================================================
# 주장 3 — 화소를 뒤섞어도 정확도가 그대로다
# ==========================================================================
print("\n" + "=" * 64)
print("주장 3  화소를 뒤섞어도 달라지지 않는다")
print("=" * 64)

# 모든 이미지에 **같은** 순열을 건다. 사람 눈에는 잡음이 되지만
# 784개 숫자 꾸러미로서는 이름표만 바뀐 셈이다.
g = torch.Generator().manual_seed(0)
perm = torch.randperm(784, generator=g)
Xtr_p, Xte_p = Xtr_n[:, perm], Xte_n[:, perm]

acc_shuf = train_linear(Xtr_p, ytr, Xte_p, yte)
print(f"  뒤섞지 않은 자료 : {acc_trained:.2f}%")
print(f"  뒤섞은 자료      : {acc_shuf:.2f}%")
print(f"  차이             : {acc_shuf - acc_trained:+.2f}%포인트")
print("\n  순열은 A의 행을 맞바꾸는 것과 같고, 학습은 맞바꾼 자리에서")
print("  같은 값을 찾아내면 된다. 곧 이 모델은 입력을 **순서 없는**")
print("  784개 숫자 꾸러미로 본다. 이웃 관계는 쓰지 않는다.")
