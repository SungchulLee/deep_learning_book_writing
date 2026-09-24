# DDPM — 잡음을 맞히는 법만 배운다

[GAN](../generative/01_gan.md)은 판별기를 속이는 법을 배웠다. 확산 모델은 훨씬 수수한 것을 배운다. **그림에 잡음을 얼마나 넣었는지 맞히는 것**이다.

!!! note "31장과의 사이"
    [31장 확산 모델](../../ch30/index.md)이 이 주제를 본격적으로 다룬다. 점수 맞추기, 확률 미분 방정식 틀, 조건 만들어 내기가 거기 있다. 이 쪽이 하는 일은 다르다. **5장이 줄 세워 온 다섯 모델 옆에 한 칸을 더 놓고 같은 자로 재는 것**이다.

---

## 1. 두 흐름

**앞으로 가는 흐름**은 학습하지 않는다. 그림에 잡음을 조금씩 섞어 $T$걸음 뒤에는 순수한 잡음으로 만드는, 미리 정해 둔 절차다.

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \varepsilon,
\qquad \varepsilon \sim N(0, I)
$$

$\bar{\alpha}_t$을 미리 셈해 두면 **어느 걸음이든 한 번에 갈 수 있다.** 1,000걸음을 실제로 밟지 않아도 500걸음째 그림을 곧장 만들 수 있고, 그래서 학습이 싸다.

**되돌리는 흐름**이 배우는 대상이다. $x_t$와 $t$를 받아 **거기 섞인 $\varepsilon$을 맞힌다.**

$$
L = \left\lVert \varepsilon - \varepsilon_\theta(x_t, t) \right\rVert^2
$$

손실이 제곱오차 하나뿐이다. [5.1절의 오토인코더](../dimreduction/03_ae_mlp.md)와 같은 꼴이며, [VAE](../latent_generative/01_vae.md)의 KL 항도 [GAN](../generative/01_gan.md)의 적대적 항도 없다.

!!! note "왜 잡음을 맞히면 그림이 나오는가"
    $x_t$에서 $\varepsilon$을 알면 위 식을 뒤집어 $x_0$을 얻는다. 곧 **잡음을 맞히는 일과 원본을 되살리는 일은 같은 일**이다. 한 걸음씩 조금만 되돌리며 이를 되풀이하면 순수한 잡음에서 그림에 이른다.

---

## 2. 코드

```python
"""5.4: DDPM. 잡음에서 조금씩 되돌려 그림을 만든다.

5.3과 같은 자를 쓴다 — 심판 CNN, 확신도, 고름, 클래스 안 다양함.
GAN과 달리 규약을 깨지 않는다: Adam 1e-3, 배치 100.
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

def train():
    torch.manual_seed(SEED)
    net=UNet().to(device)
    print(f"  U-Net 매개변수 {sum(p.numel() for p in net.parameters()):,}", flush=True)
    opt=optim.Adam(net.parameters(), lr=LR)
    g=torch.Generator().manual_seed(SEED)
    ld=DataLoader(TensorDataset(Xtr), batch_size=BATCH, shuffle=True, generator=g)
    ab=abar.to(device); t0=time.time()
    for ep in range(EPOCHS):
        net.train(); tot=n=0.0
        for (xb,) in ld:
            xb=xb.to(device); b=xb.size(0)
            t=torch.randint(0,T,(b,),device=device)
            eps=torch.randn_like(xb)
            a=ab[t][:,None,None,None]
            xt=a.sqrt()*xb+(1-a).sqrt()*eps      # 한 번에 t단계로 간다
            loss=F.mse_loss(net(xt,t),eps)       # 넣은 잡음을 맞힌다
            opt.zero_grad(); loss.backward(); opt.step()
            tot+=loss.item(); n+=1
        if ep==0 or (ep+1)%10==0:
            print(f"    에포크 {ep+1:2d}/{EPOCHS}  손실 {tot/n:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    return net.eval()

@torch.no_grad()
def sample_ddpm(net, n=N_SAMPLE, bs=500):
    """조상 뽑기. T걸음을 모두 밟는다."""
    out=[]; b_=betas.to(device); a_=alphas.to(device); ab_=abar.to(device)
    g=torch.Generator().manual_seed(SEED)
    for i in range(0,n,bs):
        x=torch.randn(bs,1,28,28,generator=g).to(device)
        for t in reversed(range(T)):
            tt=torch.full((bs,),t,device=device,dtype=torch.long)
            eps=net(x,tt)
            mean=(x-b_[t]/(1-ab_[t]).sqrt()*eps)/a_[t].sqrt()
            x=mean if t==0 else mean+b_[t].sqrt()*torch.randn_like(x)
        out.append(x.cpu())
    return torch.cat(out).clamp(-1,1)


if __name__ == "__main__":
    print(f"진짜 자료의 클래스 안 화소 분산 {REAL:.5f}  (기준선 = 1.00)")
    print("=== DDPM 익히기 ===")
    net = train()
    torch.save(net.state_dict(), "mnist_ddpm.pt")

    print("\n=== 뽑기 ===")
    t0 = time.time()
    s = sample_ddpm(net)
    evaluate(s, "DDPM 1000걸음")
    print(f"      ({time.time()-t0:.0f}s)")
```

**출력:**

```
진짜 자료의 클래스 안 화소 분산 0.05272  (기준선 = 1.00)
=== DDPM 익히기 ===
  U-Net 매개변수 975,681
    에포크  1/30  손실 0.0707  (113s)
    에포크 10/30  손실 0.0246  (1212s)
    에포크 20/30  손실 0.0228  (2289s)
    에포크 30/30  손실 0.0224  (3475s)

=== 뽑기 ===
  DDPM 1000걸음      확신도 0.940  고름 0.986  최대몫 0.175  클래스안 다양함 1.047
      (2495s)
```

---

## 3. 규약을 깨지 않는다

[5.3절](../generative/index.md)은 절을 하나 따로 두어 **GAN이 이 책의 규약으로는 학습되지 않는다**고 적어야 했다. 학습률을 $10^{-3}$에서 $2 \times 10^{-4}$로 내리고 Adam의 $\beta_1$을 0.9에서 0.5로 바꿔야 했다.

**확산 모델은 그럴 필요가 없었다.** Adam $10^{-3}$, 배치 100, 씨 42 그대로다.

| | 학습률 | Adam $\beta_1$ | 손실 움직임 |
|---|---|---|---|
| [GAN](../generative/01_gan.md) | $2 \times 10^{-4}$ | 0.5 | 오르내린다 (D 0.589 → 1.037) |
| **DDPM** | $10^{-3}$ (규약) | 0.9 (기본값) | **단조롭게 내려간다** |

손실이 0.0707에서 0.0246으로 내려간 뒤 거의 평평하다(20에포크 0.0228, 30에포크 0.0224). GAN에서는 손실이 내려가는 것이 좋은 신호인지조차 분명하지 않았다. 상대가 계속 바뀌므로 판별기의 손실이 낮다는 것은 생성기가 지고 있다는 뜻일 수도 있었다.

**여기서는 손실이 그냥 손실이다.** 맞히려는 대상($\varepsilon$)이 고정되어 있으므로 낮을수록 좋다. 겨루는 상대가 없다는 것이 이만큼을 바꾼다.

---

## 4. 결과

| | 확신도 | 고름 | 최대 클래스 | 클래스 안 다양함 |
|---|---|---|---|---|
| 진짜 자료 | — | — | — | **1.00** |
| [VAE-64](../latent_generative/01_vae.md) | 0.758 | 0.887 | 0.343 | 0.628 |
| [GAN (MLP)](../generative/01_gan.md) | 0.908 | 0.945 | 0.214 | 0.953 |
| [DCGAN](../generative/02_dcgan.md) | 0.909 | 0.970 | **0.162** | 1.085 |
| **DDPM** | **0.940** | **0.986** | 0.175 | **1.047** |

확신도가 DCGAN의 0.909에서 **0.940**으로 오른다. 5.3절이 GAN에서 얻은 도약(0.758 → 0.908)만큼 크지는 않지만, 그 위에서 다시 한 걸음이다.

**클래스 안 다양함이 1.047로 진짜에 가장 가깝다.** DCGAN은 1.085로 더 많이 넘어섰다. 5.3절이 "1을 넘으면 이 자가 다양함만 재고 있지 않다는 신호"라고 적었으므로, 1에 가깝다는 것은 **그 경고에 덜 걸린다**는 뜻이다.

### 두 고름 자가 엇갈린다

눈여겨볼 칸이 있다. DDPM은 엔트로피 고름이 더 높은데(0.986 대 0.970) **최대 클래스 몫은 더 크다**(0.175 대 0.162).

같은 "고르다"를 재는 자 둘이 서로 다른 답을 냈다. 까닭은 보는 곳이 다르기 때문이다. 최대 몫은 **가장 높은 봉우리 하나**만 보고, 고름은 **열 칸 전체**를 본다. DCGAN은 봉우리가 낮은 대신 바닥에 아주 드문 클래스가 있어 고름이 깎였고, DDPM은 봉우리가 조금 높아도 나머지가 고르다.

[5.3절](../generative/index.md)이 "자를 하나 더 만들면 또 빈틈이 생긴다"고 적은 것의 구체적인 보기다. **어느 쪽이 더 고른지는 무엇을 고르다고 부를지를 정해야 답할 수 있다.**

---

## 5. 값

여기까지만 보면 확산이 모든 칸에서 이긴 것처럼 보인다. 그렇지 않다.

| | 5,000장 뽑는 데 | 그물을 지나는 횟수 |
|---|---|---|
| [DCGAN](../generative/02_dcgan.md) | 몇 초 | **1** |
| DDPM | **2,495초 (42분)** | **1,000** |

**표본 하나에 그물을 1,000번 지난다.** GAN은 한 번이다. 학습은 42분보다 조금 더 걸렸을 뿐인데(3,475초) 뽑기가 그에 맞먹는다. 이 값을 어떻게 줄이는지는 [다음 쪽](02_ddim.md)이 다룬다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
학습할 때 걸음 $t$를 `torch.randint`로 아무거나 고른다. 왜 1부터 $T$까지 차례로 밟지 않는가?

</div>

??? success "연습문제 1 풀이"
    **그럴 필요가 없고, 그러면 훨씬 비싸기 때문이다.**

    손실은 걸음마다 따로 정의된다. $t$번째 걸음의 손실은 $x_t$와 $\varepsilon$만 있으면 셈할 수 있고, $x_t$는 $\bar{\alpha}_t$ 덕에 $x_0$에서 **한 번에** 만들어진다. 앞선 걸음을 거칠 까닭이 없다.

    차례로 밟으면 배치 하나마다 1,000번의 앞먹임이 필요하다. 아무거나 고르면 1번이다. 에포크를 여러 번 돌리면 모든 걸음이 고루 뽑히므로, **기댓값에서 같은 것을 최적화하면서 1,000분의 1만 쓴다.**

    이것이 확산 모델이 학습은 싸고 뽑기는 비싼 까닭이기도 하다. 학습에서는 걸음을 건너뛸 수 있지만, 뽑기에서는 앞 걸음의 결과가 있어야 다음 걸음을 밟는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff normal" title="중간"></span>
U-Net에 때 $t$를 넣지 않고 $x_t$만 넣으면 무엇이 잘못되는가?

</div>

??? success "연습문제 2 풀이"
    **얼마나 되돌려야 하는지 알 수 없다.**

    같은 그물이 1,000가지 일을 겸한다. $t = 900$에서는 거의 잡음뿐인 것에서 큰 덩어리를 잡아내야 하고, $t = 10$에서는 거의 완성된 그림에서 잔 잡음만 걷어내야 한다. 정반대의 일이다.

    $t$를 주지 않으면 그물은 들임만 보고 짐작해야 한다. 잡음이 얼마나 섞였는지는 그림에서 어느 정도 읽히므로 아주 못 하지는 않겠지만, 알려 줄 수 있는 것을 굳이 알아맞히게 하는 셈이다.

    본문의 `temb_of`는 $t$를 사인·코사인으로 적어 넣는다. 트랜스포머가 자리를 적는 방법과 같은 수법이며, 까닭도 같다. 수 하나를 그대로 넣는 것보다 **여러 잣수로 펼쳐** 넣어야 그물이 쓰기 쉽다.
