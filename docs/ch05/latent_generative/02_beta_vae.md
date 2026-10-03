# β-VAE — 되살리기와 덮기를 맞바꾸는 손잡이

[VAE](01_vae.md)의 손실에 이미 $\beta$가 있었다.

$$
L = \lVert \hat{x} - x \rVert^2 \;+\; \beta \, D_{\mathrm{KL}}\!\left(N(\mu, \sigma^2) \,\|\, N(0, I)\right)
$$

앞 쪽은 $\beta = 1$로 두었다. 이 쪽은 그 값을 돌려 본다.

---

## 1. 훑어 보면 두 열이 나란히 단조롭다

부호 64에서 잰 결과다. 맨 윗줄은 KL 항이 아예 없는 것, 곧 [5.1절의 오토인코더](../dimreduction/03_ae_mlp.md)다.

```python
"""beta-VAE. KL 항의 무게를 돌려 가며 되살리기와 덮기를 맞바꾼다.

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


# === 부호 64에서 beta 만 바꾸어 가며 ========================================
for beta in (0.5, 1.0, 4.0):
    m = train_vae(64, beta=beta, cond=False)

    with torch.no_grad():
        rec = torch.cat([m.decode(m.encode(Xte[i:i + 1000].to(device))[0]).cpu()
                         for i in range(0, len(Xte), 1000)])
        g = torch.Generator().manual_seed(SEED)
        z = torch.randn(N_SAMPLE, 64, generator=g)
        s = torch.cat([m.decode(z[i:i + 1000].to(device)).cpu()
                       for i in range(0, N_SAMPLE, 1000)])

    pred = judge_probs(rec).argmax(1)
    ident = 100.0 * (pred == yte).float().mean()
    conf, bal, _ = sample_stats(s)
    print(f"  vae64_beta{beta}  복원 {((rec - Xte) ** 2).mean():.5f}  "
          f"정체 {ident:.2f}%  확신도 {conf:.3f}  고름 {bal:.3f}")
```

**출력:**

```
  vae64_beta0.5  복원 0.05773  정체 98.25%  확신도 0.774  고름 0.784
  vae64_beta1.0  복원 0.07499  정체 98.06%  확신도 0.769  고름 0.877
  vae64_beta4.0  복원 0.13512  정체 95.52%  확신도 0.794  고름 0.950
```

| 부호 64 | 복원 MSE | 정체 | 확신도 | 고름 |
|---|---|---|---|---|
| KL 없음 ($\beta = 0$) | **0.0546** | 98.51% | 0.777 | 0.541 |
| $\beta = 0.5$ | 0.0577 | 98.25% | 0.774 | 0.784 |
| $\beta = 1.0$ | 0.0750 | 98.06% | 0.769 | 0.877 |
| $\beta = 4.0$ | 0.1351 | 95.52% | 0.794 | **0.950** |

**복원은 꾸준히 나빠지고 덮기는 꾸준히 좋아진다.** $\beta$는 그 사이를 조절하는 손잡이다.

그리고 **확신도는 어느 줄에서도 크게 움직이지 않는다**(0.769~0.794). 손잡이가 건드리는 것은 "얼마나 숫자다운가"가 아니라 "몇 가지를 만드는가"다.

---

## 2. 주고받는 비율이 고르지 않다

| | 고름이 오른 몫 | 복원이 나빠진 몫 |
|---|---|---|
| $\beta{=}0$ → $\beta{=}0.5$ | +0.243 | **+5.7%** |
| $\beta{=}0.5$ → $\beta{=}1.0$ | +0.093 | +30% |
| $\beta{=}1.0$ → $\beta{=}4.0$ | +0.073 | +80% |

**첫 걸음이 가장 싸다.** 되살리기를 5.7%만 내주고 덮기의 대부분(0.541 → 0.784)을 산다. 그 뒤로는 같은 만큼을 사는 데 점점 비싸진다.

곧 $\beta$를 고를 때 기본값 1이 반드시 옳은 것은 아니다. 되살리기가 중요한 쓰임새라면 0.5 언저리가 더 나은 거래일 수 있다.

---

## 3. 이 자가 못 보는 것

$\beta = 4$ 줄을 다시 보라. 복원은 가장 나쁜데 **확신도는 가장 높다**(0.794). 이상해 보이지만 까닭이 있다. KL이 세지면 디코더가 매끄럽고 **전형적인** 숫자를 내놓게 되고, 심판은 별난 진짜 3보다 반듯한 3에 더 확신한다.

여기서 이 절이 쓰는 자의 한계가 드러난다.

!!! warning "고름은 클래스 **사이**의 다양함만 잰다"
    숫자 열 가지를 딱 한 가지 모양씩만, 늘 똑같이 그려 내는 모델을 생각해 보자. 확신도는 1에 가깝고 고름도 1이 된다. **이 절의 자로는 만점이지만 만들어 내는 모델로서는 쓸모가 없다.** 같은 3만 5,000번 그리는 것과 다양한 3을 그리는 것을 구별하지 못한다.

    $\beta = 4$가 그쪽으로 가고 있을 수 있고, 이 절의 수치는 그것을 알려 주지 않는다.

    [5.3절](../generative/index.md)이 이 빈틈을 메우려 자를 하나 더 만든다. 같은 클래스로 읽힌 표본들 사이의 화소 분산을 진짜 자료의 그것으로 나눈 값이다. 그 자로 재면 VAE($\beta{=}1$)는 0.628, 곧 **진짜의 63%밖에 다양하지 않다.**

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
본문은 고름이 클래스 안의 다양함을 못 본다고 했다. 그것을 재는 자를 설계하고, $\beta = 4$가 정말 전형적인 숫자만 그리는지 확인하는 방법을 적어라.

</div>

??? success "연습문제 1 풀이"
    **같은 클래스로 읽힌 표본들 사이의 거리를 재면 된다.**

    ```python
    pred = judge(samples).argmax(1)
    for d in range(10):
        group = samples[pred == d]
        within[d] = group.var(0).mean()          # 화소별 분산의 평균
    ```

    이 값을 **진짜 자료의 같은 클래스 안 분산**과 견주는 것이 요점이다. 그래야 기준이 생긴다. 표본의 분산이 진짜보다 훨씬 작으면 전형적인 모양으로 쏠렸다는 뜻이다. [5.3절](../generative/index.md)이 정확히 이 자를 만들어 쓴다.

    $\beta$를 키워 가며 이 값을 그리면 예상대로인지 알 수 있다. 예상은 $\beta$가 커질수록 클래스 안 분산이 줄어드는 것이다. KL이 셀수록 부호가 앞분포로 눌려 서로 비슷해지고, 디코더가 내놓는 모양의 폭도 좁아지기 때문이다.

    이렇게 하면 $\beta$의 맞바꿈이 두 가지가 아니라 **세 가지**임이 드러난다. 되살리기, 클래스 사이 덮기, 클래스 안 다양함이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
$\beta$를 아주 크게 두면 어떤 일이 일어나겠는가? 손실의 두 항을 보고 극단을 따져라.

</div>

??? success "연습문제 2 풀이"
    $\beta \to \infty$이면 손실이 사실상 KL 항만 남는다. 그것을 0으로 만드는 방법은 모든 입력에 대해 $\mu = 0$, $\sigma = 1$을 내놓는 것, 곧 **인코더가 입력을 완전히 무시하는 것**이다.

    그러면 $z$는 입력과 아무 상관 없는 순수한 잡음이 되고, 디코더는 그 잡음에서 최선을 다해야 한다. 제곱오차를 줄이는 최선은 **모든 학습 그림의 평균 하나**를 늘 내놓는 것이다.

    곧 극단에서는 **표본이 전부 같아진다.** 흐릿한 평균 그림 하나뿐이다. 이때 고름은 어떻게 될까? 그 한 장이 어느 클래스로 읽히든 5,000장이 모두 같은 클래스가 되므로 **고름이 0에 가까워진다.**

    그러므로 $\beta$를 키우는 것이 끝없이 좋을 수는 없고, 어딘가에서 고름이 도로 무너진다. 본문의 표는 4까지만 재었으므로 그 꺾이는 자리를 보지 못했다. 16, 64로 넓혀 재면 나타날 것이다.

    이 현상을 **뒤확률 붕괴**(posterior collapse)라 하며, 인코더가 입력을 무시하게 되는 실패다. 큰 VAE에서 실제로 자주 겪는 문제다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
본문은 "첫 걸음이 가장 싸다"고 적는다. 이것을 수로 만들어라 — 복원이 1% 나빠질 때마다 고름이 얼마씩 오르는지를 세 걸음에서 각각 셈하고, 첫 걸음이 나머지보다 몇 배 싼지 답하라. 그리고 전체 고름 상승분 가운데 첫 걸음이 가져오는 몫도 구하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    rows = [("β=0 → 0.5",   0.0546,  0.05773, 0.541, 0.784),
            ("β=0.5 → 1.0", 0.05773, 0.07499, 0.784, 0.877),
            ("β=1.0 → 4.0", 0.07499, 0.13512, 0.877, 0.950)]
    eff = []
    for name, r0, r1, b0, b1 in rows:
        dr = 100 * (r1 - r0) / r0          # 복원이 나빠진 몫 (%)
        db = b1 - b0                        # 고름이 오른 몫
        eff.append(db / dr)
        print(f"{name:14} 고름 {db:+.3f}  복원 {dr:5.1f}%  1%당 {db/dr:.4f}")
    print(f"첫 걸음이 둘째보다 {eff[0]/eff[1]:.0f}배, 셋째보다 {eff[0]/eff[2]:.0f}배 싸다")
    ```

    ```
    β=0 → 0.5      고름 +0.243  복원   5.7%  1%당 0.0424
    β=0.5 → 1.0    고름 +0.093  복원  29.9%  1%당 0.0031
    β=1.0 → 4.0    고름 +0.073  복원  80.2%  1%당 0.0009
    첫 걸음이 둘째보다 14배, 셋째보다 47배 싸다
    ```

    **첫 걸음이 둘째보다 14배, 셋째보다 47배 싸다.** 그리고 전체 고름 상승분
    $0.950 - 0.541 = 0.409$ 가운데 첫 걸음이 $0.243$, 곧 **59%**를 가져온다.

    곧 $\beta$를 0에서 0.5로 올리는 것만으로 살 것의 절반 넘게를 사고, 값은 복원 5.7%뿐이다. 그 뒤로는 같은 만큼을 사는 데 값이 열 배, 쉰 배로 뛴다.

    **그래서 기본값 1이 당연한 선택은 아니다.** 0.5에서 1로 가는 걸음은 고름 0.093을 복원 30%에 사는 거래인데, 되살리기를 쓰는 일이라면 받아들이기 어려운 값이다. 기본값은 "대개 무난한 값"일 뿐 내 쓰임새에 맞는 값이 아니다.

    다만 이 셈은 **네 점으로 그린 곡선**이다. $\beta$를 0.1이나 0.25에서 재면 첫 걸음이 더 싼 자리가 더 있을 수 있고, 0.5와 1 사이에서 꺾이는 자리를 이 네 점으로는 집어낼 수 없다.

