"""소프트맥스 절의 주장 다섯을 수로 확인한다.

    주장 1  이동 불변 — 모든 로짓에 같은 수를 더해도 확률이 같다
    주장 2  정의를 그대로 옮기면 넘친다. 최댓값을 빼면 넘치지 않는다
    주장 3  자유도가 하나 남는다 — 한 로짓을 0으로 못박아도 잃는 것이 없다
    주장 4  온도가 분포를 딱딱하게도 부드럽게도 만든다
    주장 5  소프트맥스를 두 번 걸면 **오류 없이** 학습이 나빠진다
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms


# ==========================================================================
# 주장 1 — 이동 불변성
# ==========================================================================
print("=" * 64)
print("주장 1  모든 로짓에 같은 수를 더해도 확률이 같다")
print("=" * 64)


def softmax(y):
    """정의를 그대로 옮긴 것. 일부러 최댓값을 빼지 않았다."""
    e = np.exp(y)
    return e / e.sum()


a = np.array([1.0, 2.0, 3.0])
b = a + 100.0                      # 모두 100씩 밀었다
print(f"  로짓 {a}  ->  {softmax(a).round(6)}")
print(f"  로짓 {b}  ->  {softmax(b).round(6)}")
print(f"  두 확률의 가장 큰 차이: {np.abs(softmax(a) - softmax(b)).max():.2e}")
print("\n  뜻이 있는 것은 로짓 사이의 **차이**뿐이다.")


# ==========================================================================
# 주장 2 — 정의대로 옮기면 넘친다
# ==========================================================================
print("\n" + "=" * 64)
print("주장 2  최댓값을 빼야 넘치지 않는다")
print("=" * 64)


def softmax_stable(y):
    """실제 구현. 먼저 최댓값을 뺀다 — 이동 불변이라 값은 같다."""
    e = np.exp(y - y.max())
    return e / e.sum()


# float32가 담을 수 있는 가장 큰 수와, exp가 그것을 넘는 자리
print(f"  float32의 최댓값      = {np.finfo(np.float32).max:.4e}")
print(f"  exp(88)  = {np.exp(np.float32(88)):.4e}   (아직 담긴다)")
print(f"  exp(89)  = {np.exp(np.float32(89)):.4e}   (넘쳤다)")

for y in (np.array([1.0, 2.0, 3.0], dtype=np.float32),
          np.array([100.0, 101.0, 102.0], dtype=np.float32),
          np.array([1000.0, 1.0, 1.0], dtype=np.float32)):
    naive = softmax(y)
    stable = softmax_stable(y)
    print(f"\n  로짓 {y}")
    print(f"    정의 그대로 : {naive}")
    print(f"    최댓값 빼기 : {stable}")

print("\n  로짓이 100만 되어도 정의대로 옮긴 쪽은 nan이 된다.")
print("  inf/inf 이기 때문이며, 오류가 아니라 조용히 nan이 흐른다.")


# ==========================================================================
# 주장 3 — 자유도가 하나 남는다
# ==========================================================================
print("\n" + "=" * 64)
print("주장 3  한 로짓을 0으로 못박아도 잃는 것이 없다")
print("=" * 64)

rng = np.random.RandomState(0)
y = rng.randn(10) * 3                      # 아무 로짓 열 개
y_fixed = y - y[0]                         # 첫 로짓을 0으로 밀어 둔다

print(f"  원래 로짓의 첫 값   : {y[0]:.4f}")
print(f"  민 뒤의 첫 값       : {y_fixed[0]:.4f}")
print(f"  두 확률의 가장 큰 차이: {np.abs(softmax_stable(y) - softmax_stable(y_fixed)).max():.2e}")
print("\n  로짓 10개로 적지만 확률을 정하는 자유도는 9다.")


# ==========================================================================
# 주장 4 — 온도
# ==========================================================================
print("\n" + "=" * 64)
print("주장 4  온도가 분포를 딱딱하게도 부드럽게도 만든다")
print("=" * 64)


def entropy(p):
    """섀넌 엔트로피. 고를수록 크고, 한 곳에 몰릴수록 0에 가깝다."""
    q = p[p > 0]
    return float(-(q * np.log(q)).sum())


logits = np.array([3.0, 1.0, 0.2, 0.0, -1.0, -1.5, -2.0, -2.2, -3.0, -3.5])
print(f"  고른 분포의 엔트로피(위 끝) = {np.log(10):.4f}\n")
print(f"  {'T':>6} | {'가장 큰 확률':>11} | {'엔트로피':>9}")
print("  " + "-" * 32)
for T in (0.1, 0.5, 1.0, 2.0, 10.0, 100.0):
    p = softmax_stable(logits / T)
    print(f"  {T:>6} | {p.max():>11.4f} | {entropy(p):>9.4f}")

print("\n  T가 작으면 argmax에 가까워지고(딱딱), 크면 1/10로 고르게 된다(부드럽).")


# ==========================================================================
# 주장 5 — 소프트맥스를 두 번 걸면 오류 없이 나빠진다
# ==========================================================================
print("\n" + "=" * 64)
print("주장 5  소프트맥스를 두 번 — 오류는 없고 학습만 나빠진다")
print("=" * 64)

# 먼저 무엇이 일어나는지 수로 본다
p_once = softmax_stable(logits)
p_twice = softmax_stable(p_once)          # 확률을 다시 로짓인 양 넣는다
print(f"  한 번 건 뒤 : 가장 큰 확률 {p_once.max():.4f}, 엔트로피 {entropy(p_once):.4f}")
print(f"  두 번 건 뒤 : 가장 큰 확률 {p_twice.max():.4f}, 엔트로피 {entropy(p_twice):.4f}")
print(f"  (고른 분포의 엔트로피가 {np.log(10):.4f}이니 두 번 건 쪽은 거의 고르다)")

# 이제 실제로 학습시켜 본다. **오류가 나지 않는다**는 것이 요점이다.
SEED, BATCH, EPOCHS = 42, 128, 10
tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.1307,), (0.3081,))])
tr = torchvision.datasets.MNIST("./data", train=True, download=True, transform=tf)
te = torchvision.datasets.MNIST("./data", train=False, download=True, transform=tf)
Xtr = torch.stack([x for x, _ in tr]).reshape(len(tr), -1)
ytr = torch.tensor([y for _, y in tr])
Xte = torch.stack([x for x, _ in te]).reshape(len(te), -1)
yte = torch.tensor([y for _, y in te])


def train(double_softmax, opt_name="Adam"):
    """double_softmax=True면 모델 끝에 nn.Softmax를 덧붙인다 (하지 말아야 할 일).

    에포크마다의 정확도와, 첫 배치에서의 기울기 크기를 함께 돌려준다.
    """
    torch.manual_seed(SEED)
    layers = [nn.Linear(784, 10)]
    if double_softmax:
        layers.append(nn.Softmax(dim=1))       # CrossEntropyLoss가 또 걸 것이다
    m = nn.Sequential(*layers)
    opt = (optim.Adam(m.parameters(), lr=1e-3) if opt_name == "Adam"
           else optim.SGD(m.parameters(), lr=0.1))
    crit = nn.CrossEntropyLoss()               # 안에서 log_softmax를 이미 한다
    g = torch.Generator().manual_seed(SEED)
    ld = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH,
                    shuffle=True, generator=g)

    first_grad, accs = None, []
    for _ in range(EPOCHS):
        m.train()
        for xb, yb in ld:
            opt.zero_grad()
            crit(m(xb), yb).backward()
            if first_grad is None:             # 첫 배치의 기울기 크기
                first_grad = m[0].weight.grad.norm().item()
            opt.step()
        m.eval()
        with torch.no_grad():
            accs.append(100.0 * (m(Xte).argmax(1) == yte).float().mean().item())
    return accs, first_grad


acc_ok, g_ok = train(False)
acc_bad, g_bad = train(True)

print(f"\n  첫 배치의 기울기 크기")
print(f"    로짓 그대로   : {g_ok:.4f}")
print(f"    Softmax 덧붙임: {g_bad:.4f}   ({g_bad/g_ok:.2f}배)")

print(f"\n  에포크마다의 시험 정확도")
print(f"  {'에포크':>5} | {'로짓 그대로':>10} | {'Softmax 덧붙임':>14} | {'차이':>8}")
print("  " + "-" * 50)
for i, (x, y) in enumerate(zip(acc_ok, acc_bad), 1):
    print(f"  {i:>5} | {x:>9.2f}% | {y:>13.2f}% | {y-x:>+7.2f}%p")

# 최적화기를 바꾸어도 마찬가지인지 본다
sgd_ok, _ = train(False, "SGD")
sgd_bad, _ = train(True, "SGD")
print(f"\n  최적화기를 바꾸면 (10에포크 끝값)")
print(f"    Adam : {acc_ok[-1]:.2f}% -> {acc_bad[-1]:.2f}%  ({acc_bad[-1]-acc_ok[-1]:+.2f}%p)")
print(f"    SGD  : {sgd_ok[-1]:.2f}% -> {sgd_bad[-1]:.2f}%  ({sgd_bad[-1]-sgd_ok[-1]:+.2f}%p)")
