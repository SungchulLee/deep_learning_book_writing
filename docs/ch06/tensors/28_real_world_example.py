"""튜토리얼 28: 파이프라인이 제 고장을 알아보는가.

MNIST 숫자 분류기를 처음부터 끝까지 짠다. 그러나 이 쪽의 일은 98%를
받아 보는 데서 끝나지 않는다. 끝까지 이어 붙인 보기가 값을 가지는
까닭은 **잘못 이어 붙였을 때 숫자가 그것을 알려 주기 때문**이고,
그러니 알려 주기는 하는지를 재어 보아야 한다.

그래서 바르게 돌린 다음, 네 가지로 고장을 내고 어느 고장이 숫자에
나타나는지 본다.

실행: python 28_real_world_example.py   (repo 뿌리에서, ./data 를 읽는다)
"""
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

# ========================================================================
# 장치 — 있는 것을 쓴다
# ========================================================================

def pick_device():
    """cuda → mps → cpu. 이 책은 셋 다에서 돌아야 한다."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEV = pick_device()

# ========================================================================
# 모델
# ========================================================================

class ConvNet(nn.Module):
    """28x28 한 채널 → 열 클래스. 파라미터 225,034개."""

    def __init__(self, p_drop=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)   # 28 → 26
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # 13 → 11
        self.pool = nn.MaxPool2d(2)                    # 26 → 13, 11 → 5
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.drop = nn.Dropout(p_drop)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.flatten(1)                 # view(-1, ...) 보다 뜻이 분명하다
        x = torch.relu(self.fc1(x))
        return self.fc2(self.drop(x))

# ========================================================================
# 학습과 평가 — 고장을 낼 수 있게 손잡이를 둔다
# ========================================================================

def one_epoch(model, opt, crit, loader, *, zero_grad=True, scale=1.0):
    """한 에폭. zero_grad=False 와 scale=255 가 뒤에서 낼 고장이다."""
    model.train()
    tot = correct = seen = 0.0
    for x, y in loader:
        x, y = x.to(DEV) * scale, y.to(DEV)
        if zero_grad:
            opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        tot += loss.item() * y.numel()
        correct += (out.argmax(1) == y).sum().item()
        seen += y.numel()
    return tot / seen, 100 * correct / seen


def accuracy(model, loader, *, eval_mode=True, scale=1.0):
    """eval_mode=False 가 'eval() 빼먹기' 고장이다."""
    model.eval() if eval_mode else model.train()
    correct = seen = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEV) * scale, y.to(DEV)
            correct += (model(x).argmax(1) == y).sum().item()
            seen += y.numel()
    return 100 * correct / seen


def fresh(seed, p_drop=0.5, lr=1e-3, sgd=False):
    torch.manual_seed(seed)
    m = ConvNet(p_drop).to(DEV)
    opt = (torch.optim.SGD if sgd else torch.optim.Adam)(m.parameters(), lr=lr)
    return m, opt, nn.CrossEntropyLoss()


def train(loader, *, seed, epochs, **kw):
    m, opt, crit = fresh(seed)
    for _ in range(epochs):
        one_epoch(m, opt, crit, loader, **kw)
    return m

# ========================================================================
# 메인
# ========================================================================

TFM = transforms.ToTensor()
TR = datasets.MNIST("./data", train=True, download=True, transform=TFM)
TE = datasets.MNIST("./data", train=False, download=True, transform=TFM)
trl = DataLoader(TR, batch_size=128, shuffle=True)
tel = DataLoader(TE, batch_size=512)


def main():
    print(f"장치 {DEV}   학습 {len(TR)}장   검증 {len(TE)}장")

    # === 1. 바르게 이어 붙인 파이프라인 ===
    print("\n=== 1. 바른 파이프라인, 4 에폭 ===")
    m, opt, crit = fresh(0)
    print(f"파라미터 {sum(p.numel() for p in m.parameters()):,}개")
    for ep in range(1, 5):
        ls, ac = one_epoch(m, opt, crit, trl)
        print(f"에폭 {ep}  학습 로스 {ls:.4f}  학습 {ac:.2f}%  "
              f"검증 {accuracy(m, tel):.2f}%")

    # === 2. 라벨을 섞어 본다 ===
    # 라벨을 무작위로 섞으면 배울 규칙이 없어진다. 그런데도 검증 점수가
    # 높게 나오면 파이프라인이 어딘가에서 답을 흘리고 있다는 뜻이다.
    print("\n=== 2. 라벨을 섞으면 (학습 10,000장) ===")
    torch.manual_seed(0)
    idx = torch.randperm(len(TR))[:10000].tolist()
    sub_x = torch.stack([TR[i][0] for i in idx])
    sub_y = torch.tensor([TR[i][1] for i in idx])
    fake_y = sub_y[torch.randperm(10000)]
    print(f"섞고 나서 우연히 맞은 라벨 "
          f"{100 * (fake_y == sub_y).float().mean():.2f}%")
    fake = DataLoader(TensorDataset(sub_x, fake_y), batch_size=128, shuffle=True)
    for p_drop in (0.5, 0.0):
        m2, opt2, crit2 = fresh(0, p_drop=p_drop)
        print(f"  드롭아웃 p={p_drop}")
        for ep in range(1, 61):
            ls, ac = one_epoch(m2, opt2, crit2, fake)
            if ep in (10, 20, 30, 40, 60):
                print(f"    {ep:2d} 에폭  학습 로스 {ls:.4f}  학습 {ac:6.2f}%  "
                      f"검증 {accuracy(m2, tel):5.2f}%")

    # === 3. 조용한 고장 셋 ===
    print("\n=== 3. 조용한 고장 (2 에폭, 씨앗 셋) ===")
    leak = DataLoader(Subset(TR, list(range(10000))), batch_size=512)
    keys = ("정상", "eval 빼기", "검증 누수", "zero_grad 빼기")
    rows = {k: [] for k in keys}
    for seed in (0, 1, 2):
        base = train(trl, seed=seed, epochs=2)
        rows["정상"].append(accuracy(base, tel))
        rows["eval 빼기"].append(accuracy(base, tel, eval_mode=False))
        rows["검증 누수"].append(accuracy(base, leak))
        nz = train(trl, seed=seed, epochs=2, zero_grad=False)
        rows["zero_grad 빼기"].append(accuracy(nz, tel))
    for k in keys:
        v = rows[k]
        print(f"{k:14s} {sum(v) / 3:6.2f}%   퍼짐 {min(v):.2f}~{max(v):.2f}   "
              f"({', '.join(f'{x:.2f}' for x in v)})")

    # === 4. 같은 고장이 판에 따라 보이기도, 안 보이기도 한다 ===
    print("\n=== 4. 검증 누수를 외우는 판에서 다시 ===")
    for size, epochs in ((1000, 40), (60000, 2)):
        sub = Subset(TR, list(range(size)))
        mm = train(DataLoader(sub, batch_size=128, shuffle=True),
                   seed=0, epochs=epochs)
        s = accuracy(mm, DataLoader(sub, batch_size=512))
        h = accuracy(mm, tel)
        print(f"  학습 {size:>5}장 {epochs:2d}에폭   본 그림 {s:.2f}%   "
              f"못 본 그림 {h:.2f}%   부풀림 {s - h:+.2f}점")

    # === 5. 고장이라 불리지만 고장이 아닌 것 ===
    print("\n=== 5. 입력을 0~255로 넣으면 (2 에폭) ===")
    for name, kw in (("Adam lr=0.001", dict(lr=1e-3)),
                     ("SGD  lr=0.01 ", dict(lr=1e-2, sgd=True))):
        got = []
        for scale in (1.0, 255.0):
            m3, opt3, crit3 = fresh(0, **kw)
            ls = 0.0
            for _ in range(2):
                ls, _ = one_epoch(m3, opt3, crit3, trl, scale=scale)
            got.append((accuracy(m3, tel, scale=scale), ls))
        (a1, l1), (a2, l2) = got
        print(f"  {name}   0~1 {a1:6.2f}% (로스 {l1:.4f})   "
              f"0~255 {a2:6.2f}% (로스 {l2:.4f})")


if __name__ == "__main__":
    main()
