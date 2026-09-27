"""교차 엔트로피 손실 절의 주장들을 수로 확인한다.

    주장 1  가능도를 곱으로 두면 부동소수점에서 0이 된다
    주장 2  로그를 취하면 크기가 표본 수에 선형으로만 큰다
    주장 3  가능도 최대화와 교차 엔트로피 최소화는 **같은 것**이다
    주장 4  그림 한 장이 지나가는 길 — 학습 전과 후
    주장 5  확신하고 틀리면 손실이 아주 커진다
    주장 6  시작할 때의 손실은 log 10 = 2.3026 이어야 한다
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms

SEED, BATCH, LR, EPOCHS = 42, 128, 1e-3, 10


# ==========================================================================
# 주장 1 — 곱으로 두면 0이 된다
# ==========================================================================
print("=" * 64)
print("주장 1  가능도를 곱으로 두면 부동소수점에서 0이 된다")
print("=" * 64)

n = 60000
p = 0.99                                  # 표본마다 확률이 0.99인 '아주 좋은' 모델
print(f"  표본 {n:,}개, 표본마다 확률 {p}")
print(f"  0.99^{n} = 10^({n * math.log10(p):.1f})")
print(f"  float32가 담을 수 있는 가장 작은 양수 = {np.finfo(np.float32).tiny:.3e}")
print(f"  float64                              = {np.finfo(np.float64).tiny:.3e}")

# 실제로 60,000번 곱해 보면 무슨 일이 일어나는가
for dtype, name in ((np.float32, "float32"), (np.float64, "float64")):
    prod = dtype(1.0)
    snapshots = {}
    for i in range(1, n + 1):
        prod = dtype(prod * dtype(p))
        if i in (5000, 10000, 20000, 60000):
            snapshots[i] = prod
    print(f"\n  {name}로 곱해 나가면")
    for i, v in snapshots.items():
        print(f"    {i:>6,}번: {v:.6e}")

# float32 쪽은 0이 되는 것이 아니라 **한 값에 멈춘다**. 까닭이 재미있다.
tiny32 = np.nextafter(np.float32(0), np.float32(1))     # 가장 작은 비정규수
stuck = np.float32(7.006492e-44)
print(f"\n  float32가 멈춘 값 {stuck:.6e} 은 가장 작은 비정규수의 "
      f"{stuck/tiny32:.0f}배다.")
print("  비정규수 구간에서는 값 사이 간격이 **절대적**이라, 50칸짜리 수에")
print("  0.99를 곱하면 49.5칸이 되고 이는 49와 50의 한가운데다. 짝수 쪽으로")
print("  반올림하는 규칙이 50을 고르므로 **제자리로 돌아온다.**")

print("\n  0이 되면 매개변수를 어느 쪽으로 움직여도 가능도가 0이라 아무 정보가 없다.")


# ==========================================================================
# 주장 2 — 로그를 취하면 표본 수에 선형으로만 큰다
# ==========================================================================
print("\n" + "=" * 64)
print("주장 2  로그를 취하면 크기가 선형으로만 큰다")
print("=" * 64)
print(f"  {'표본 수':>9} | {'곱 (10^?)':>12} | {'로그 합':>10}")
print("  " + "-" * 36)
for m in (100, 1000, 10000, 60000):
    print(f"  {m:>9,} | {m * math.log10(p):>12.1f} | {m * math.log(p):>10.1f}")
print(f"\n  60,000개에서 곱은 10^-262 이지만 로그 합은 {n * math.log(p):.0f} 다.")
print("  담을 수 없는 수가 담을 수 있는 수가 된다.")


# ==========================================================================
# 자료와 모델 — 아래 주장들에 쓴다
# ==========================================================================
tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.1307,), (0.3081,))])
tr = torchvision.datasets.MNIST("./data", train=True, download=True, transform=tf)
te = torchvision.datasets.MNIST("./data", train=False, download=True, transform=tf)
Xtr = torch.stack([x for x, _ in tr]).reshape(len(tr), -1)
ytr = torch.tensor([y for _, y in tr])
Xte = torch.stack([x for x, _ in te]).reshape(len(te), -1)
yte = torch.tensor([y for _, y in te])

torch.manual_seed(SEED)
model = nn.Linear(784, 10)


def logits_probs(m, x):
    with torch.no_grad():
        z = m(x)
        return z, torch.softmax(z, dim=-1)


# ==========================================================================
# 주장 6 (먼저) — 시작할 때의 손실은 log 10 이어야 한다
# ==========================================================================
print("\n" + "=" * 64)
print("주장 6  시작할 때의 손실")
print("=" * 64)

crit = nn.CrossEntropyLoss()
with torch.no_grad():
    start_loss = crit(model(Xtr), ytr).item()
print(f"  학습 전 학습 자료 평균 손실 : {start_loss:.4f}")
print(f"  log 10                      : {math.log(10):.4f}")
print(f"  차이                        : {abs(start_loss - math.log(10)):.4f}")
print("\n  아무것도 배우지 않았으면 열 갈래에 1/10씩 주므로 -log(1/10) = log 10 이다.")


# ==========================================================================
# 주장 4 (앞쪽) — 학습 전, 그림 한 장이 지나가는 길
# ==========================================================================
print("\n" + "=" * 64)
print("주장 4  그림 한 장이 지나가는 길")
print("=" * 64)

img, label = Xte[0], int(yte[0])
z0, p0 = logits_probs(model, img)
print(f"  시험 자료 첫 그림, 참 레이블 {label}")
print("\n  [학습 전]")
print("    로짓 z : " + " ".join(f"{v:+6.2f}" for v in z0))
print("    확률 p : " + " ".join(f"{v:6.3f}" for v in p0))
print(f"    확률의 합 = {p0.sum():.4f}")
print(f"    손실 = -log p_{label} = -log {p0[label]:.4f} = {-math.log(p0[label]):.4f}")


# --- 학습한다 ---
opt = optim.Adam(model.parameters(), lr=LR)
g = torch.Generator().manual_seed(SEED)
ld = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH, shuffle=True, generator=g)
for _ in range(EPOCHS):
    model.train()
    for xb, yb in ld:
        opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
model.eval()

z1, p1 = logits_probs(model, img)
print(f"\n  [{EPOCHS} 에포크 학습 뒤]")
print("    로짓 z : " + " ".join(f"{v:+6.2f}" for v in z1))
print("    확률 p : " + " ".join(f"{v:6.3f}" for v in p1))
print(f"    손실 = -log p_{label} = -log {p1[label]:.4f} = {-math.log(p1[label]):.4f}")
print(f"\n  로짓이 {z1.min():+.2f} 에서 {z1.max():+.2f} 까지 벌어졌다.")
print("  학습이 한 일은 로짓을 **크게 벌리는 것**이다.")


# ==========================================================================
# 주장 3 — 가능도 최대화 = 교차 엔트로피 최소화
# ==========================================================================
print("\n" + "=" * 64)
print("주장 3  가능도 최대화와 교차 엔트로피 최소화는 같은 것")
print("=" * 64)

with torch.no_grad():
    logp = torch.log_softmax(model(Xte), dim=1)
    # (가) 로그 가능도: 참 클래스의 로그 확률을 모두 더한다
    ll = logp[torch.arange(len(yte)), yte].sum().item()
    # (나) 교차 엔트로피: 원-핫 표적과의 교차 엔트로피를 평균낸다
    onehot = torch.zeros(len(yte), 10)
    onehot[torch.arange(len(yte)), yte] = 1.0
    ce = -(onehot * logp).sum(1).mean().item()
    # (다) PyTorch의 손실 함수
    ce_torch = crit(model(Xte), yte).item()

print(f"  (가) 로그 가능도 sum(log p_t)     = {ll:.6f}")
print(f"  (나) 교차 엔트로피 -mean(q . log p) = {ce:.6f}")
print(f"  (다) nn.CrossEntropyLoss           = {ce_torch:.6f}")
print(f"\n  -(가)/n = {-ll/len(yte):.6f}   <- (나)와 같아야 한다")
print(f"  (나)와 (다)의 차이 = {abs(ce - ce_torch):.2e}")
print("\n  셋이 같은 수다. 부호를 뒤집고 표본 수로 나눈 것뿐이므로,")
print("  가능도를 가장 크게 하는 매개변수가 곧 손실을 가장 작게 한다.")


# ==========================================================================
# 주장 5 — 확신하고 틀리면 아주 세게 벌한다
# ==========================================================================
print("\n" + "=" * 64)
print("주장 5  확신하고 틀릴 때의 손실")
print("=" * 64)

with torch.no_grad():
    losses = -logp[torch.arange(len(yte)), yte]          # 표본마다의 손실
    worst = int(losses.argmax())
    pred = int(model(Xte[worst]).argmax())

print(f"  가장 크게 틀린 표본 (시험 자료 {len(yte):,}장 가운데)")
print(f"    참 레이블        : {int(yte[worst])}")
print(f"    예측             : {pred}")
print(f"    참 클래스의 확률  : {math.exp(-losses[worst].item()):.3e}")
print(f"    손실             : {losses[worst].item():.1f}")
print(f"\n  가장 작은 손실    : {abs(losses.min().item()):.4f}")
print(f"  같은 손실 함수가 {abs(losses.min().item()):.4f} 와 {losses.max().item():.1f} 를 함께 낸다.")

# 배치 평균에 미치는 몫
typical = losses.median().item()
batch = (losses[worst].item() + 127 * typical) / 128
print(f"\n  가운뎃값 손실 {typical:.4f} 짜리 127장에 이 한 장을 섞으면")
print(f"    배치 평균 = ({losses[worst].item():.1f} + 127 x {typical:.4f}) / 128 = {batch:.4f}")
print(f"    127장만이었다면 {typical:.4f} 이므로 {batch/typical:.1f}배로 뛴다.")
print("\n  손실이 갑자기 튀면 대개 이런 표본 몇 개 때문이다.")
