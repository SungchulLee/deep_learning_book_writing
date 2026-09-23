# DDIM — 걸음을 건너뛴다

[앞 쪽](01_ddpm.md)이 남긴 값은 컸다. 표본 하나에 그물을 **1,000번** 지나야 했고, 5,000장에 42분이 들었다. [DCGAN](../generative/02_dcgan.md)은 한 번이면 끝난다.

이 쪽은 **모델을 손대지 않는다.** 같은 `mnist_ddpm.pt`를 그대로 읽어 뽑는 방법만 바꾼다.

---

## 1. 무엇을 바꾸는가

DDPM의 되돌리기는 걸음마다 잡음을 새로 더한다. 그래서 1,000걸음을 하나도 건너뛸 수 없다. 앞 걸음에서 더한 잡음을 다음 걸음이 알아야 하기 때문이다.

DDIM은 **그 잡음을 빼 버린다.** 걸음마다 이렇게 한다.

1. 지금 자리 $x_t$에서 $\varepsilon_\theta(x_t, t)$을 맞혀 **원본이 무엇일지 추측**한다

$$
\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\, \varepsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}
$$

2. 그 추측을 **다음 걸음 자리로 다시 보낸다**

$$
x_{t'} = \sqrt{\bar{\alpha}_{t'}}\, \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t'}}\, \varepsilon_\theta(x_t, t)
$$

무작위 항이 없으므로 $t'$은 $t-1$일 필요가 없다. **1,000에서 900으로 건너뛰어도 된다.**

!!! note "덤으로 따라오는 것"
    무작위성이 사라졌으므로 **처음 뽑은 $z$가 그림을 온전히 정한다.** 같은 $z$는 언제나 같은 그림이 된다. [VAE](../latent_generative/01_vae.md)나 [GAN](../generative/01_gan.md)이 그랬던 것처럼 잠재 공간이 생기는 셈이고, DDPM에는 없던 성질이다.

---

## 2. 코드

```python
"""DDIM. 걸음을 건너뛰어 같은 모델에서 더 싸게 뽑는다.

앞 쪽에서 학습해 둔 mnist_ddpm.pt를 읽어 쓴다. 모델은 손대지 않는다.
자료·심판·U-Net 정의는 앞 쪽과 같다.
"""
import json, time
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision, torchvision.transforms as transforms

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEED, BATCH, LR, EPOCHS, T = 42, 100, 1e-3, 30, 1000
N_SAMPLE = 5000
JM, JS = 0.1307, 0.3081

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
tr = torchvision.datasets.MNIST("./data", train=True, transform=tf)
te = torchvision.datasets.MNIST("./data", train=False, transform=tf)
Xtr = torch.cat([x for x,_ in DataLoader(tr, batch_size=2000)])
Xte = torch.cat([x for x,_ in DataLoader(te, batch_size=2000)])
yte = torch.cat([y for _,y in DataLoader(te, batch_size=2000)])

class JudgeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,32,3,padding=1); self.conv2=nn.Conv2d(32,64,3,padding=1)
        self.pool=nn.MaxPool2d(2,2); self.dropout=nn.Dropout(0.25)
        self.fc1=nn.Linear(64*7*7,128); self.fc2=nn.Linear(128,10)
    def forward(self,x):
        x=self.pool(torch.relu(self.conv1(x))); x=self.pool(torch.relu(self.conv2(x)))
        return self.fc2(self.dropout(torch.relu(self.fc1(x.flatten(1)))))

judge = JudgeCNN().to(device)
judge.load_state_dict(torch.load("mnist_judge.pt", weights_only=True)); judge.eval()

@torch.no_grad()
def predict(x):
    xj = (((x+1)/2)-JM)/JS
    p = torch.cat([torch.softmax(judge(xj[i:i+1000].to(device)),1).cpu() for i in range(0,len(xj),1000)])
    return p.max(1)

def within_var(im01, grp, min_n=20):
    vs,ws=[],[]
    for d in range(10):
        g=im01[grp==d]
        if len(g)<min_n: continue
        vs.append(g.flatten(1).var(0).mean().item()); ws.append(len(g))
    return sum(v*w for v,w in zip(vs,ws))/sum(ws) if vs else 0.0

REAL = within_var((Xte+1)/2, yte)

def evaluate(s, tag):
    conf,pred = predict(s)
    sh = torch.bincount(pred,minlength=10).float(); sh/=sh.sum(); nz=sh[sh>0]
    bal = (-(nz*nz.log()).sum()/torch.tensor(10.0).log()).item()
    wv = within_var((s+1)/2, pred)
    r = dict(confidence=round(conf.mean().item(),4), balance=round(bal,4),
             biggest=round(sh.max().item(),4), within=round(wv/REAL,4))
    print(f"  {tag:18s} 확신도 {r['confidence']:.3f}  고름 {r['balance']:.3f}  "
          f"최대몫 {r['biggest']:.3f}  클래스안 다양함 {r['within']:.3f}", flush=True)
    return r

# === 앞으로 가는 흐름: 미리 셈해 두는 상수들 ===
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
abar = torch.cumprod(alphas, 0)

class Block(nn.Module):
    def __init__(self, cin, cout, temb):
        super().__init__()
        self.c1=nn.Conv2d(cin,cout,3,padding=1); self.c2=nn.Conv2d(cout,cout,3,padding=1)
        self.t=nn.Linear(temb,cout); self.n1=nn.GroupNorm(8,cout); self.n2=nn.GroupNorm(8,cout)
        self.skip=nn.Conv2d(cin,cout,1) if cin!=cout else nn.Identity()
    def forward(self,x,t):
        h=F.silu(self.n1(self.c1(x)))
        h=h+self.t(t)[:,:,None,None]
        h=F.silu(self.n2(self.c2(h)))
        return h+self.skip(x)

class UNet(nn.Module):
    """28 -> 14 -> 7 -> 14 -> 28. 건너뛰기 이음을 갖춘 작은 U-Net."""
    def __init__(self, ch=64, temb=128):
        super().__init__()
        self.temb=temb
        self.tmlp=nn.Sequential(nn.Linear(temb,temb), nn.SiLU(), nn.Linear(temb,temb))
        self.d1=Block(1,ch,temb); self.d2=Block(ch,ch*2,temb)
        self.mid=Block(ch*2,ch*2,temb)
        self.u2=Block(ch*4,ch,temb); self.u1=Block(ch*2,ch,temb)
        self.out=nn.Conv2d(ch,1,3,padding=1)
        self.pool=nn.AvgPool2d(2)
    def temb_of(self,t):
        half=self.temb//2
        f=torch.exp(-torch.arange(half,device=t.device)*torch.log(torch.tensor(10000.0))/(half-1))
        a=t[:,None].float()*f[None]
        return torch.cat([a.sin(),a.cos()],1)
    def forward(self,x,t):
        e=self.tmlp(self.temb_of(t))
        h1=self.d1(x,e)                      # 28
        h2=self.d2(self.pool(h1),e)          # 14
        m=self.mid(self.pool(h2),e)          # 7
        u=F.interpolate(m,scale_factor=2)    # 14
        u=self.u2(torch.cat([u,h2],1),e)
        u=F.interpolate(u,scale_factor=2)    # 28
        u=self.u1(torch.cat([u,h1],1),e)
        return self.out(u)

net = UNet().to(device)
net.load_state_dict(torch.load("mnist_ddpm.pt", weights_only=True))
net.eval()


@torch.no_grad()
def sample_ddim(net, steps, n=N_SAMPLE, bs=500):
    """걸음을 건너뛴다. 잡음을 더하지 않으므로 z가 그림을 온전히 정한다."""
    ts = torch.linspace(T - 1, 0, steps).long().tolist()      # 1000걸음에서 골라낸다
    out = []
    ab_ = abar.to(device)
    g = torch.Generator().manual_seed(SEED)
    for i in range(0, n, bs):
        x = torch.randn(bs, 1, 28, 28, generator=g).to(device)
        for j, t in enumerate(ts):
            tt = torch.full((bs,), t, device=device, dtype=torch.long)
            eps = net(x, tt)
            a = ab_[t]
            x0 = (x - (1 - a).sqrt() * eps) / a.sqrt()        # 지금 자리에서 본 원본
            if j + 1 < len(ts):
                a_next = ab_[ts[j + 1]]
                # 그 원본을 다음 걸음 자리로 다시 보낸다. 무작위 항이 없다
                x = a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps
            else:
                x = x0
        out.append(x.cpu())
    return torch.cat(out).clamp(-1, 1)


for steps in (100, 50, 20, 10):
    t0 = time.time()
    s = sample_ddim(net, steps)
    evaluate(s, f"DDIM {steps}걸음")
    print(f"      ({time.time()-t0:.0f}s)")
```

**출력:**

```
  DDIM 100걸음       확신도 0.924  고름 0.979  최대몫 0.185  클래스안 다양함 1.110
      (290s)
  DDIM 50걸음        확신도 0.923  고름 0.979  최대몫 0.187  클래스안 다양함 1.097
      (158s)
  DDIM 20걸음        확신도 0.916  고름 0.977  최대몫 0.191  클래스안 다양함 1.060
      (69s)
  DDIM 10걸음        확신도 0.904  고름 0.971  최대몫 0.204  클래스안 다양함 0.979
      (26s)
```

---

## 3. 96배 빠르고 4% 나쁘다

| | 걸음 | 확신도 | 고름 | 최대 클래스 | 클래스 안 다양함 | 5,000장 뽑는 데 |
|---|---|---|---|---|---|---|
| [DDPM](01_ddpm.md) | 1,000 | **0.940** | **0.986** | **0.175** | 1.047 | 2,495초 |
| DDIM | 100 | 0.924 | 0.979 | 0.185 | 1.110 | 290초 |
| DDIM | 50 | 0.923 | 0.979 | 0.187 | 1.097 | 158초 |
| DDIM | 20 | 0.916 | 0.977 | 0.191 | 1.060 | 69초 |
| DDIM | **10** | 0.904 | 0.971 | 0.204 | **0.979** | **26초** |

**걸음을 100분의 1로 줄이면 뽑기가 96배 빨라지고 확신도는 0.940에서 0.904로 떨어진다.** 3.8% 손해다.

눈여겨볼 자리가 둘 있다.

**100걸음에서 50걸음은 거의 공짜다.** 확신도 0.924와 0.923, 고름은 소수 셋째 자리까지 같다. 그런데 시간은 절반이다. 여기서는 **걸음을 줄이지 않을 까닭이 없다.**

**10걸음짜리가 DCGAN과 맞먹는다.** 확신도 0.904 대 0.909, 고름 0.971 대 0.970이다. 26초는 GAN의 한 번짜리 앞먹임보다는 여전히 느리지만, 5.3절이 얻은 자리를 **거의 같은 값으로** 얻는다는 뜻이다.

---

## 4. 이 자가 또 이상하게 움직인다

클래스 안 다양함 열을 걸음 수 차례로 읽어 보라.

$$
1.047 \;\to\; 1.110 \;\to\; 1.097 \;\to\; 1.060 \;\to\; 0.979
$$

**단조롭지 않다.** 1,000걸음에서 100걸음으로 갈 때 **올라갔다가**, 거기서부터 내려온다. 그리고 10걸음에서 1을 밑으로 지난다.

품질을 재는 자라면 걸음을 줄일수록 나빠져야 하는데 그렇지 않다. 확신도와 고름은 얌전히 단조롭게 내려가는데 이 자만 다르게 움직인다.

[5.3절](../generative/index.md)이 DCGAN의 1.085를 두고 적어 둔 경고를 다시 읽을 자리다.

> 1을 넘었다는 것은 **이 자가 더 이상 다양함만 재고 있지 않다**는 신호로 보는 편이 안전하다.

여기서 그 신호가 더 또렷해진다. **10걸음짜리가 1에 가장 가까운 0.979를 받는데, 확신도는 가장 낮다.** 가장 숫자답지 않은 표본이 "진짜만큼 다양하다"는 점수를 받은 것이다.

까닭을 짐작하면 이렇다. 이 자는 심판이 **같은 클래스로 읽은 표본들** 사이의 화소 분산이다. 걸음이 적어 흐릿하거나 어정쩡한 표본이 섞이면 무리 안이 지저분해져 분산이 커진다. 1,000걸음에서 100걸음으로 갈 때 오른 것은 그 몫으로 보인다. 10걸음에서 도로 내려간 것은 또 다른 이야기가 필요한데, **이 실험만으로는 가를 수 없다.**

확실한 것은 하나다. **이 열의 값이 1에 가깝다고 좋은 모델이라 읽으면 안 된다.**

---

## 5. 5장이 걸어온 길

| | 뽑을 수 있는가 | 인코더 | 확신도 | 고름 | 앞먹임 횟수 |
|---|---|---|---|---|---|
| [AE_MLP-64](../dimreduction/03_ae_mlp.md) | 억지로 | 있다 | 0.737 | 0.580 | 1 |
| [VAE-64](../latent_generative/01_vae.md) | 앞분포에서 | 있다 | 0.758 | 0.887 | 1 |
| [GAN (MLP)](../generative/01_gan.md) | 앞분포에서 | **없다** | 0.908 | 0.945 | 1 |
| [DCGAN](../generative/02_dcgan.md) | 앞분포에서 | **없다** | 0.909 | 0.970 | 1 |
| [DDPM](01_ddpm.md) | 앞분포에서 | 없다 | **0.940** | **0.986** | **1,000** |
| DDIM 10걸음 | 앞분포에서 | 없다 | 0.904 | 0.971 | **10** |

**확산은 품질을 시간으로 산다.** 5.2절의 VAE는 되살리기를 희생해 덮기를 샀고, 5.3절의 GAN은 인코더와 학습 안정성을 희생해 선명함을 샀다. 확산이 내놓은 값은 **뽑는 시간**이며, DDIM은 그 값을 얼마든지 깎을 수 있게 하되 깎은 만큼 품질을 돌려준다.

어느 자리를 고를지는 무엇이 모자란지가 정한다. 한 장을 만드는 데 26초가 아깝지 않다면 1,000걸음이 낫고, 실시간으로 만들어야 한다면 GAN이 아직 이긴다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff normal" title="중간"></span>
DDIM은 무작위 항이 없어 같은 $z$가 언제나 같은 그림이 된다. 이 성질로 무엇을 할 수 있는가?

</div>

??? success "연습문제 1 풀이"
    **잠재 공간을 훑을 수 있다.** $z_1$과 $z_2$ 사이를 이어 그 위의 점들을 모두 복호하면, [5.1절이 2차원 잠재 공간을 격자로 훑은 것](../dimreduction/03_ae_mlp.md)과 같은 그림을 얻는다. DDPM에서는 걸음마다 새 잡음이 들어가므로 같은 $z$로도 다른 그림이 나와 이 일을 할 수 없다.

    **그림을 고칠 수도 있다.** 앞으로 가는 흐름을 DDIM 식으로 거꾸로 돌리면 주어진 그림에 해당하는 $z$를 찾을 수 있고($DDIM\ inversion$), 그 $z$를 조금 움직여 되돌리면 원본과 닮은 변형을 얻는다. [5.3절 연습문제](../generative/index.md)가 "GAN에는 부호기가 없어 그림을 부호로 되돌릴 수 없다"고 적었는데, DDIM은 **학습 없이** 그 일을 근사한다.

    다만 온전한 인코더는 아니다. 되돌리기가 근사이므로 원본이 정확히 복원되지 않고, 걸음을 줄일수록 어긋남이 커진다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
본문은 걸음을 $[T-1, 0]$에서 고르게 골랐다(`torch.linspace`). 고르지 않게 고르면 나아질 수 있겠는가? 어느 쪽에 걸음을 몰아야 하는가?

</div>

??? success "연습문제 2 풀이"
    **나아질 수 있고, 실제로 그렇게 한다.**

    걸음마다 하는 일이 다르다. $t$가 클 때는 거의 잡음뿐인 것에서 큰 얼개를 정하고, $t$가 작을 때는 잔 결을 다듬는다. 그런데 $\bar{\alpha}_t$이 $t$에 따라 고르게 변하지 않는다. 본문의 선형 $\beta$ 일정에서는 **작은 $t$ 쪽에서 $\bar{\alpha}_t$이 빠르게 움직인다.**

    빠르게 움직이는 구간을 성글게 건너뛰면 어림이 크게 어긋난다. 그러므로 **작은 $t$ 쪽에 걸음을 몰아 주는 것**이 이치에 맞는다.

    실제로 쓰이는 방법들이 이 자리를 다룬다. 이차 간격(quadratic spacing)이 흔하고, [30장](../../ch30/index.md)이 다루는 DPM-Solver 같은 것들은 한 걸음 안에서 더 높은 차수의 어림을 써서 같은 걸음 수로 더 멀리 간다.

    확인하려면 본문 코드의 `torch.linspace(T - 1, 0, steps)`만 바꾸어 같은 걸음 수로 재어 보면 된다. 모델을 다시 학습할 필요가 없다는 점이 이 쪽 전체의 요점이기도 하다.
