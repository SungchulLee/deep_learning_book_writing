"""부호를 아무 데서나 뽑으면 어떻게 되는가.

5.1절에서 학습한 AE_MLP를 그대로 쓴다. 되살리기는 배웠으니, 이번에는
없던 부호를 넣어 없던 그림을 만들어 본다. mnist_judge.pt는 이 절 첫 쪽에서
학습해 둔 것을 읽어 쓴다.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms

SEED, BATCH, LR, AE_EPOCHS = 42, 100, 1e-3, 100
N_SAMPLE = 5000
MEAN, STD = 0.1307, 0.3081
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
plt.rcParams["svg.fonttype"] = "path"

tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((MEAN,), (STD,))])
tr_ds = torchvision.datasets.MNIST("./data", train=True, download=True, transform=tf)
te_ds = torchvision.datasets.MNIST("./data", train=False, download=True, transform=tf)


def materialize(ds):
    xs = [x for x, _ in DataLoader(ds, batch_size=2000, shuffle=False)]
    return torch.cat(xs).flatten(1)


Xtr, Xte = materialize(tr_ds), materialize(te_ds)


# === 심판 (5.1절 첫 쪽에서 학습해 둔 것) ====================================
class JudgeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2); self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128); self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        return self.fc2(self.dropout(torch.relu(self.fc1(x.flatten(1)))))


judge = JudgeCNN().to(device)
judge.load_state_dict(torch.load("mnist_judge.pt", weights_only=True))
judge.eval()


@torch.no_grad()
def read(flat):
    """심판에게 읽히고 (평균 확신도, 클래스 균형도, 최대 클래스 몫, 그 숫자)를 돌려준다."""
    p = torch.cat([torch.softmax(judge(flat[i:i + 1000].reshape(-1, 1, 28, 28).to(device)), 1).cpu()
                   for i in range(0, len(flat), 1000)])
    conf, lab = p.max(1)
    share = torch.bincount(lab, minlength=10).float() / len(lab)
    nz = share[share > 0]
    balance = float(-(nz * nz.log()).sum() / torch.tensor(10.0).log())
    return conf.mean().item(), balance, share.max().item(), int(share.argmax())


# === 5.1절의 AE_MLP 그대로 ==================================================
class MLPAE(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, k))
        self.dec = nn.Sequential(nn.Linear(k, 256), nn.ReLU(), nn.Linear(256, 784))

    def forward(self, x):
        return self.dec(self.enc(x))


def train_ae(k):
    torch.manual_seed(SEED)
    m = MLPAE(k).to(device)
    opt = optim.Adam(m.parameters(), lr=LR)
    g = torch.Generator().manual_seed(SEED)
    loader = DataLoader(TensorDataset(Xtr), batch_size=BATCH, shuffle=True, generator=g)
    for _ in range(AE_EPOCHS):
        m.train()
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            F.mse_loss(m(xb), xb).backward()
            opt.step()
    return m.eval()


if __name__ == "__main__":
    print(f"진짜 시험 이미지에 심판이 주는 확신도 {read(Xte)[0]:.3f}")

    rows = []
    for k in (2, 64):
        m = train_ae(k)
        with torch.no_grad():
            z = torch.cat([m.enc(Xte[i:i + 1000].to(device)).cpu()
                           for i in range(0, len(Xte), 1000)])
        with torch.no_grad():
            rec = torch.cat([m(Xte[i:i + 1000].to(device)).cpu()
                             for i in range(0, len(Xte), 1000)])
        print(f"\n=== 잠재 차원 {k}")
        print(f"  진짜 부호: 좌표 std {z.std(0).mean():.2f}  "
              f"범위 [{z.min():.1f}, {z.max():.1f}]")
        print(f"  되살린 그림  확신도 {read(rec)[0]:.3f}")

        torch.manual_seed(0)
        for name, pts in [("N(0, I)", torch.randn(N_SAMPLE, k)),
                          ("맞춘 정규분포", torch.randn(N_SAMPLE, k) * z.std(0) + z.mean(0))]:
            with torch.no_grad():
                img = m.dec(pts.to(device)).cpu()
            conf, bal, top, digit = read(img)
            print(f"  {name:12s} 확신도 {conf:.3f}  클래스 균형도 {bal:.3f}  "
                  f"가장 많은 클래스 {100 * top:.1f}% (숫자 {digit})")
            rows.append((f"k={k}  {'N(0,I)' if 'N(' in name else 'fitted'}", img[:10]))

    # === 열 장씩 그려 둔다 ===================================================
    fig, axes = plt.subplots(len(rows), 10, figsize=(10, 4.6))
    for r, (label, imgs) in enumerate(rows):
        for c in range(10):
            axes[r, c].imshow((imgs[c].reshape(28, 28) * STD + MEAN).clamp(0, 1),
                              cmap="gray", vmin=0, vmax=1)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        axes[r, 0].text(-0.15, 0.5, label, transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=9)
    fig.subplots_adjust(left=0.17, right=0.99, top=0.99, bottom=0.01,
                        wspace=0.05, hspace=0.08)
    fig.savefig("ae_sampling.svg", transparent=True)
    print("\nwrote ae_sampling.svg")
