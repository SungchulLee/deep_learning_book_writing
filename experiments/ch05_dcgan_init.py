"""DCGAN 논문의 웨이트 초기화 N(0, 0.02) 가 이 책의 작은 DCGAN 에서도 일을 하는가.

`docs/ch05/generative/02_dcgan.md` 의 모델·규약을 글자 그대로 가져와, 초기화만
바꾸어 시드 다섯 벌씩 돌린다. 5.3절 개요 5절이 네 모델을 다섯 벌로 잰 것과
같은 자(확신도·클래스 균형도·최대 클래스 몫·클래스 안 다양함)를 쓴다.

세 갈래를 견준다.

  none    아무것도 안 한다 — 쪽이 지금 싣고 있는 것
  conv    논문·튜토리얼의 weights_init 그대로. Conv* 는 N(0, 0.02),
          BatchNorm* 은 웨이트 N(1, 0.02) 에 바이어스 0.
          **Linear 는 건드리지 않는다** — 논문의 구조에는 Linear 가 없어서
          규칙이 적혀 있지 않기 때문이다.
  all     거기에 Linear 까지 N(0, 0.02) 로. 이 쪽의 생성기는 파라메터
          552,513개 가운데 401,408개가 첫 Linear 에 있으므로, conv 갈래만으로는
          대부분을 손대지 않은 채 재게 된다.

실행: python experiments/ch05_dcgan_init.py        (저장소 뿌리에서)
      3 갈래 x 5 시드 = 15 번 학습한다. 한 번에 15분 남짓 걸린다.
결과: experiments/ch05_dcgan_init.json
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms

# === 규약 — 02_dcgan.md 와 같다 ===============================================
BATCH, ZDIM, EPOCHS = 100, 64, 30
N_SAMPLE = 5000
SEEDS = (42, 43, 44, 45, 46)
JM, JS = 0.1307, 0.3081
OUT = Path("experiments/ch05_dcgan_init.json")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.5,), (0.5,))])
tr = torchvision.datasets.MNIST("./data", train=True, download=True, transform=tf)
te = torchvision.datasets.MNIST("./data", train=False, download=True, transform=tf)
Xtr = torch.cat([x for x, _ in DataLoader(tr, batch_size=2000, shuffle=False)])
Xte = torch.cat([x for x, _ in DataLoader(te, batch_size=2000, shuffle=False)])
yte = torch.cat([y for _, y in DataLoader(te, batch_size=2000, shuffle=False)])


# === 평가 분류기 — 5.1절에서 학습해 둔 것 =====================================
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
def predict(x_pm1):
    xj = (((x_pm1 + 1) / 2) - JM) / JS
    p = torch.cat([torch.softmax(judge(xj[i:i + 1000].to(device)), 1).cpu()
                   for i in range(0, len(xj), 1000)])
    conf, pred = p.max(1)
    return conf, pred


def within_class_var(imgs01, group, min_n=20):
    vs, ws = [], []
    for d in range(10):
        g = imgs01[group == d]
        if len(g) < min_n:
            continue
        vs.append(g.flatten(1).var(0).mean().item()); ws.append(len(g))
    return sum(v * w for v, w in zip(vs, ws)) / sum(ws) if vs else 0.0


REAL_VAR = within_class_var((Xte + 1) / 2, yte)


# === 모델 — 02_dcgan.md 와 같다 ===============================================
class GConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(ZDIM, 128 * 7 * 7),
                                nn.BatchNorm1d(128 * 7 * 7), nn.ReLU())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh())

    def forward(self, z):
        return self.net(self.fc(z).reshape(-1, 128, 7, 7))


class DConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Flatten(), nn.Linear(128 * 7 * 7, 1))

    def forward(self, x):
        return self.net(x)


# === 초기화 세 갈래 ===========================================================
def weights_init(m):
    """논문·튜토리얼이 적어 둔 것 그대로. Conv* 와 BatchNorm* 만 건드린다."""
    name = m.__class__.__name__
    if name.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif name.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def weights_init_all(m):
    """위에 Linear 를 더한 것. 논문에는 없는 갈래다."""
    name = m.__class__.__name__
    if name.find("Linear") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    else:
        weights_init(m)


ARMS = {"none": None, "conv": weights_init, "all": weights_init_all}


def train_one(seed, init_fn):
    torch.manual_seed(seed)
    G, D = GConv().to(device), DConv().to(device)
    if init_fn is not None:
        G.apply(init_fn); D.apply(init_fn)
    oG = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    oD = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    gg = torch.Generator().manual_seed(seed)
    ld = DataLoader(TensorDataset(Xtr), batch_size=BATCH, shuffle=True, generator=gg)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(EPOCHS):
        G.train(); D.train()
        for (xb,) in ld:
            xb = xb.to(device); n = xb.size(0)
            ones = torch.ones(n, 1, device=device)
            zeros = torch.zeros(n, 1, device=device)
            fake = G(torch.randn(n, ZDIM, device=device))
            lossD = bce(D(xb), ones) + bce(D(fake.detach()), zeros)
            oD.zero_grad(); lossD.backward(); oD.step()
            lossG = bce(D(fake), ones)
            oG.zero_grad(); lossG.backward(); oG.step()
    G.eval()
    with torch.no_grad():
        z = torch.randn(N_SAMPLE, ZDIM, generator=torch.Generator().manual_seed(seed))
        s = torch.cat([G(z[i:i + 1000].to(device)).cpu()
                       for i in range(0, N_SAMPLE, 1000)])
    conf, pred = predict(s)
    share = torch.bincount(pred, minlength=10).float()
    share = share / share.sum()
    nz = share[share > 0]
    return {
        "conf": conf.mean().item(),
        "balance": (-(nz * nz.log()).sum() / torch.tensor(10.0).log()).item(),
        "top": share.max().item(),
        "within": within_class_var((s + 1) / 2, pred) / REAL_VAR,
    }


def rng(vals):
    return min(vals), max(vals), sum(vals) / len(vals)


if __name__ == "__main__":
    res = {}
    for arm, fn in ARMS.items():
        res[arm] = []
        for sd in SEEDS:
            t0 = time.time()
            r = train_one(sd, fn)
            r["seed"] = sd; r["sec"] = round(time.time() - t0)
            res[arm].append(r)
            print(f"  {arm:5s} seed {sd}  확신도 {r['conf']:.3f}  "
                  f"균형도 {r['balance']:.3f}  최대몫 {r['top']:.3f}  "
                  f"다양함 {r['within']:.3f}  ({r['sec']}s)", flush=True)
        OUT.write_text(json.dumps(res, indent=1))

    print("\n  다섯 벌 범위 (최소~최대, 평균)")
    for k in ("conf", "balance", "top", "within"):
        print(f"    {k}")
        for arm in ARMS:
            lo, hi, mu = rng([r[k] for r in res[arm]])
            print(f"      {arm:5s} {lo:.3f}~{hi:.3f}  평균 {mu:.3f}")
    print(f"\n  wrote {OUT}")
