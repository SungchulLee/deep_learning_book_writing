# cVAE — 원하는 숫자를 만들기

[VAE](01_vae.md)는 뽑으면 **무엇이 나올지 모르는** 모델이었다. 라벨을 함께 넣으면 고를 수 있게 된다. 인코더와 디코더 모두에 원-핫 라벨을 이어 붙인다.

```python
"""cVAE. 라벨을 함께 넣어 원하는 숫자를 만든다.

mnist_judge.pt는 5.1절에서 학습해 둔 것을 읽어 쓴다.
구조와 규약은 5.1절 AE_MLP와 같다. 달라지는 것은 손실뿐이다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms

SEED, BATCH, LR, EPOCHS = 42, 100, 1e-3, 100
N_SAMPLE = 5000
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.1307,), (0.3081,))])
tr_ds = torchvision.datasets.MNIST("./data", train=True, download=True, transform=tf)
te_ds = torchvision.datasets.MNIST("./data", train=False, download=True, transform=tf)


def materialize(ds):
    xs, ys = [], []
    for x, y in DataLoader(ds, batch_size=2000, shuffle=False):
        xs.append(x.flatten(1)); ys.append(y)
    return torch.cat(xs), torch.cat(ys)


Xtr, ytr = materialize(tr_ds)
Xte, yte = materialize(te_ds)


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
def judge_probs(flat):
    return torch.cat([torch.softmax(
        judge(flat[i:i + 1000].reshape(-1, 1, 28, 28).to(device)), 1).cpu()
        for i in range(0, len(flat), 1000)])


def sample_stats(flat):
    """표본을 심판에 넣어 확신도와 클래스 고름을 잰다.

    고름은 심판이 읽은 클래스 분포의 엔트로피를 균등분포로 나눈 값이다.
    1이면 열 가지가 고르게 나왔고, 0에 가까우면 한 가지로 쏠렸다.
    """
    p = judge_probs(flat)
    conf, pred = p.max(1)
    share = torch.bincount(pred, minlength=10).float()
    share = share / share.sum()
    nz = share[share > 0]
    bal = (-(nz * nz.log()).sum() / torch.tensor(10.0).log()).item()
    return conf.mean().item(), bal, share.max().item()


class VAE(nn.Module):
    """부호를 점이 아니라 분포로 내놓는다. n_cond>0이면 조건부(cVAE)."""

    def __init__(self, k, n_cond=0):
        super().__init__()
        self.k, self.n_cond = k, n_cond
        self.enc = nn.Sequential(nn.Linear(784 + n_cond, 256), nn.ReLU())
        self.mu = nn.Linear(256, k)
        self.logvar = nn.Linear(256, k)
        self.dec = nn.Sequential(nn.Linear(k + n_cond, 256), nn.ReLU(),
                                 nn.Linear(256, 784))

    def encode(self, x, c=None):
        h = self.enc(x if c is None else torch.cat([x, c], 1))
        return self.mu(h), self.logvar(h)

    def decode(self, z, c=None):
        return self.dec(z if c is None else torch.cat([z, c], 1))

    def forward(self, x, c=None):
        mu, logvar = self.encode(x, c)
        # 재매개변수화: 무작위성을 eps로 밀어내어 mu, logvar로 기울기가 흐르게 한다
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode(z, c), mu, logvar


def onehot(y):
    return F.one_hot(y, 10).float()


def train_vae(k, beta, cond):
    torch.manual_seed(SEED)
    m = VAE(k, 10 if cond else 0).to(device)
    opt = optim.Adam(m.parameters(), lr=LR)
    g = torch.Generator().manual_seed(SEED)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH,
                        shuffle=True, generator=g)
    for _ in range(EPOCHS):
        m.train()
        for xb, yb in loader:
            xb = xb.to(device)
            c = onehot(yb).to(device) if cond else None
            opt.zero_grad()
            xh, mu, lv = m(xb, c)
            # 화소에 대해 합, 표본에 대해 평균 — 가우스 가능도의 꼴
            rec = F.mse_loss(xh, xb, reduction="sum") / xb.size(0)
            kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / xb.size(0)
            (rec + beta * kl).backward()
            opt.step()
    return m.eval()


# === 라벨을 인코더와 디코더 양쪽에 넣는다 ===================================
for k in (2, 64):
    m = train_vae(k, beta=1.0, cond=True)

    with torch.no_grad():
        # 복원: 진짜 라벨을 함께 준다 (그래서 정체가 부풀려진다)
        rec = torch.cat([
            m.decode(m.encode(Xte[i:i + 1000].to(device),
                              onehot(yte[i:i + 1000]).to(device))[0],
                     onehot(yte[i:i + 1000]).to(device)).cpu()
            for i in range(0, len(Xte), 1000)])

        # 만들기: 열 가지를 고루 요청하고, 그대로 나왔는지 심판에게 묻는다
        g = torch.Generator().manual_seed(SEED)
        z = torch.randn(N_SAMPLE, k, generator=g)
        want = torch.arange(N_SAMPLE) % 10
        c = onehot(want)
        s = torch.cat([m.decode(z[i:i + 1000].to(device),
                                c[i:i + 1000].to(device)).cpu()
                       for i in range(0, N_SAMPLE, 1000)])

    got = judge_probs(s).argmax(1)
    cond_acc = 100.0 * (got == want).float().mean()
    tag = f"cvae{k}"
    print(f"  {tag:7s} 복원 {((rec - Xte) ** 2).mean():.5f}  "
          f"조건 정확도 {cond_acc:.2f}%")
```

**출력:**

```
  cvae2   복원 0.34787  조건 정확도 99.66%
  cvae64  복원 0.07618  조건 정확도 73.64%
```

만들 때는 원하는 숫자의 원-핫과 $z \sim N(0, I)$을 함께 넣는다. 그리고 **정말 그 숫자가 나왔는지 심판에게 물으면** 조건이 먹혔는지를 수로 알 수 있다. 만들어 내는 모델에 정확도를 매길 수 있는 드문 자리이며, 책이 [3장](../../ch03/mnist/04_cnn.md)에서 분류기를 만들어 두었기에 가능하다.

---

## 1. 부호가 클수록 조건이 덜 먹힌다

| | 복원 MSE | 조건 정확도 |
|---|---|---|
| cVAE-2 | **0.3479** | **99.66%** |
| cVAE-64 | 0.0762 | **73.64%** |

99.66%에서 73.64%로 떨어진다. 부호를 키우면 되살리기는 좋아지는데 **말을 듣지 않는다.**

---

## 2. 복원 쪽 수가 까닭을 말해 준다

같은 부호 크기의 조건 없는 [VAE](01_vae.md)와 견주어 보자.

| 부호 | VAE 복원 | cVAE 복원 | 라벨이 도운 몫 |
|---|---|---|---|
| 2 | 0.4274 | **0.3479** | **19% 좋아짐** |
| 64 | 0.0750 | 0.0762 | 없음 |

**부호가 2일 때는 라벨이 되살리기를 크게 돕는다.** 2차원으로는 *어느 숫자인지*와 *어떤 모양인지*를 함께 담을 수 없는데, 라벨이 앞엣것을 맡아 주니 부호는 모양만 담으면 된다. 기울기, 굵기, 획의 끝 모양 같은 것들이다. 그래서 복원이 19% 좋아지고, 디코더는 라벨에 **기댈 수밖에** 없으므로 조건도 잘 먹는다.

**부호가 64일 때는 라벨이 되살리기에 아무 보탬이 안 된다.** 64차원이면 숫자의 정체를 $z$ 안에 담고도 자리가 남기 때문이다. 그러면 디코더는 라벨을 **무시해도 된다.** 만들 때 $z \sim N(0, I)$이 제멋대로의 정체를 싣고 오면, 디코더는 우리가 고른 라벨이 아니라 $z$를 따른다.

**두 수가 같은 것을 가리킨다.** 라벨이 되살리기를 돕지 않는 자리에서 조건도 먹지 않는다. 서로 다른 두 측정이 하나의 설명으로 모인다는 점이 이 결과를 믿을 만하게 만든다.

그리고 [VAE 쪽](01_vae.md)에서 본 것과 같은 이야기다. 부호를 키우면 잠재 공간을 **다루기가 어려워진다.** 뽑기도 어려워지고 조건도 먹지 않는다.

---

## 3. 이 표의 다른 수치는 읽지 말 것

!!! danger "고름과 정체는 여기서 뜻이 없다"
    cVAE의 **고름**은 1.000과 0.964인데, 이것은 성취가 아니다. 열 가지를 **고르게 요청했으므로** 고르게 나오는 것이 당연하다. 고름은 모델이 **스스로 클래스를 고를 때만** 모드 붕괴를 잡아낸다.

    cVAE의 **정체**(99.96%, 98.79%)도 부풀려져 있다. 복원할 때 디코더에게 **진짜 라벨을 알려 주기** 때문이다. 무엇을 그릴지 듣고 그린 그림이다. 조건 없는 모델의 정체와 같은 자리에 놓고 견주면 안 된다.

    조건부 모델에서 뜻이 있는 자는 **조건 정확도**와 **복원 MSE**뿐이다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
조건 정확도가 부호 64에서 73.64%로 떨어졌다. 부호 크기를 바꾸지 않고 이를 고칠 방법을 두 가지 들고, 각각이 무엇을 대가로 치르는지 말하라.

</div>

??? success "연습문제 1 풀이"
    문제의 뿌리는 **디코더가 라벨을 쓰지 않아도 되는 것**이다. $z$ 안에 정체가 들어 있기 때문이다. 고치려면 $z$가 정체를 담지 못하게 하거나, 라벨을 쓰지 않으면 손해를 보게 해야 한다.

    **하나, $\beta$를 키운다.** KL을 세게 걸면 $z$가 앞분포로 눌려 정보를 덜 담게 되고, 부족한 만큼 디코더가 라벨에 기댄다. 대가는 [β-VAE 쪽](02_beta_vae.md)에서 본 그대로다. 되살리기가 나빠진다.

    **둘, 손실에 항을 더한다.** 만들어 낸 표본을 분류기에 넣어 요청한 라벨과 맞는지를 손실에 넣는다. 조건을 **직접** 요구하는 셈이다. 대가는 분류기가 따로 필요하다는 것과, 분류기가 좋아하는 전형적인 모양으로 쏠릴 위험이다([β-VAE 쪽](02_beta_vae.md)의 함정과 같다).

    셋째 길도 있다. $z$에서 정체를 **적대적으로 지우는** 것이다. $z$로부터 라벨을 맞히려는 작은 분류기를 두고 인코더는 그것을 방해하도록 학습시킨다. 대가는 학습이 불안정해지는 것이며, 그 불안정함이 [5.3절](../generative/index.md)의 주제이기도 하다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
cVAE-2의 부호 2개는 무엇을 담고 있겠는가? 확인하는 방법을 적어라.

</div>

??? success "연습문제 2 풀이"
    라벨이 정체를 맡았으므로 부호에 남는 것은 **모양**이다. 기울기, 획 굵기, 고리 크기 같은 것들이 후보다.

    확인은 **같은 $z$를 열 가지 라벨에 걸어 보는 것**으로 한다.

    ```python
    z = torch.randn(1, 2).repeat(10, 1)          # 같은 부호
    c = F.one_hot(torch.arange(10), 10).float()  # 라벨만 0~9
    imgs = model.decode(z, c)
    ```

    부호가 정말 모양만 담고 있다면, 열 장이 **서로 다른 숫자인데 같은 기울기와 굵기**를 지녀야 한다. $z$를 바꾸어 여러 줄을 그리면 각 줄이 한 가지 필체가 된다.

    반대로 열 장이 제멋대로면 부호와 라벨이 얽혀 있다는 뜻이다. cVAE-64에서는 그렇게 될 가능성이 높고, 그것이 조건 정확도 73.64%의 다른 얼굴이다.

    이 그림은 **풀림**(disentanglement)을 눈으로 보는 흔한 방법이기도 하다. 라벨이라는 하나의 축이 나머지와 깔끔히 갈라졌는지를 본다.
