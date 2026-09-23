# 깊이 — 3×3을 쌓는다

[4.1절](01_two_ladders.md)의 4걸음 CNN은 CIFAR-10에서 71.78%였다. **남은 오차가 28.22%포인트다.**

[3.4절](../ch03/mnist/04_cnn.md)은 MNIST에서 같은 모델의 손잡이를 여덟 가지로 돌려 보았고, 아무것도 움직이지 않았다. 남은 오차가 0.93%포인트뿐이라 잴 여지가 없었기 때문이다.

여기서는 그 여지가 서른 배다. **같은 손잡이를 돌리면 이번에는 무엇이 일어나는가?**

---

## 1. 무엇을 바꾸고 무엇을 고정하는가

분류기 머리(`fc1`, `fc2`)를 모든 갈래에서 똑같이 둔다. 풀링을 두 번 지나 $8 \times 8 \times 64$가 되는 것도 같다. **바뀌는 것은 합성곱 더미뿐**이므로, 정확도 차이를 깊이와 필터 크기의 몫으로만 읽을 수 있다.

| 갈래 | 합성곱 더미 |
|---|---|
| 2겹 3×3 | `[conv(3→32), pool]`, `[conv(32→64), pool]` — 4.1절 그대로 |
| 2겹 5×5 | 같은 깊이, 필터만 5×5 |
| 4겹 3×3 | 블록마다 두 번 — `[conv, conv, pool]` ×2 |
| 6겹 3×3 | 블록마다 세 번 |

규약은 [4.1절](01_two_ladders.md) 그대로다. 배치 100, Adam $10^{-3}$, 5 에포크이며 씨 다섯 개로 재어 퍼짐을 함께 적는다.

### 코드

```python
"""CIFAR-10에서 같은 손잡이를 돌린다. 4.1절 규약 그대로: 배치 100, Adam 1e-3, 5 에포크.

분류기 머리(fc1, fc2)는 모든 갈래에서 같다. 바뀌는 것은 합성곱 더미뿐이라
정확도 차이를 깊이와 필터 크기의 몫으로 읽을 수 있다.
"""
import json, time
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS, BATCH, LR, SEEDS = 5, 100, 1e-3, [0,1,2,3,4]

tf = transforms.Compose([transforms.ToTensor(),
      transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
tr = datasets.CIFAR10("./data", train=True, transform=tf)
te = datasets.CIFAR10("./data", train=False, transform=tf)
Xtr = torch.cat([x for x,_ in DataLoader(tr, batch_size=2000)])
ytr = torch.cat([y for _,y in DataLoader(tr, batch_size=2000)])
Xte = torch.cat([x for x,_ in DataLoader(te, batch_size=2000)])
yte = torch.cat([y for _,y in DataLoader(te, batch_size=2000)])


class Net(nn.Module):
    """블록마다 3x3을 n_per번 쌓고 풀링한다. n_per=1이면 4.1절의 4걸음이다."""
    def __init__(self, n_per=1, k=3, dropout=0.25):
        super().__init__()
        layers, cin = [], 3
        for cout in (32, 64):
            for _ in range(n_per):
                layers += [nn.Conv2d(cin, cout, k, padding=k//2), nn.ReLU()]
                cin = cout
            layers += [nn.MaxPool2d(2, 2)]
        self.features = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()
        self.fc1 = nn.Linear(64*8*8, 128)      # 모든 갈래에서 같다
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.features(x)
        return self.fc2(self.drop(torch.relu(self.fc1(x.flatten(1)))))


def run(seed, **kw):
    torch.manual_seed(seed)
    m = Net(**kw).to(device)
    n_par = sum(p.numel() for p in m.parameters())
    n_conv = sum(1 for mod in m.features if isinstance(mod, nn.Conv2d))
    opt = optim.Adam(m.parameters(), lr=LR); crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    ld = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH, shuffle=True, generator=g)
    for _ in range(EPOCHS):
        m.train()
        for xb, yb in ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(m(xb), yb).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        pred = torch.cat([m(Xte[i:i+1000].to(device)).argmax(1).cpu() for i in range(0,len(Xte),1000)])
    return 100.0*(pred==yte).float().mean().item(), n_par, n_conv


VARIANTS = [
    ("합성곱 2겹 3x3 (4.1절)",  dict(n_per=1, k=3)),
    ("합성곱 2겹 5x5",          dict(n_per=1, k=5)),
    ("합성곱 4겹 3x3",          dict(n_per=2, k=3)),
    ("합성곱 6겹 3x3",          dict(n_per=3, k=3)),
    ("합성곱 4겹 3x3, 드롭아웃 없음", dict(n_per=2, k=3, dropout=0.0)),
]
out={}
for tag, kw in VARIANTS:
    t0=time.time(); accs=[]
    for s in SEEDS:
        a,n_par,n_conv = run(s, **kw); accs.append(a)
    mean=sum(accs)/len(accs)
    out[tag]=dict(mean=round(mean,3), lo=round(min(accs),2), hi=round(max(accs),2),
                  spread=round(max(accs)-min(accs),3), params=n_par, n_conv=n_conv,
                  runs=[round(a,2) for a in accs])
    print(f"  {tag:26s} {mean:6.2f}%  퍼짐 {max(accs)-min(accs):.2f} "
          f"({min(accs):.2f}~{max(accs):.2f})  합성곱 {n_conv}층  매개변수 {n_par:,}  ({time.time()-t0:.0f}s)", flush=True)
```

**출력:**

```
  합성곱 2겹 3x3 (4.1절)          71.33%  퍼짐 1.11 (70.68~71.79)  합성곱 2층  매개변수 545,098  (147s)
  합성곱 2겹 5x5                 71.53%  퍼짐 2.29 (70.21~72.50)  합성곱 2층  매개변수 579,402  (289s)
  합성곱 4겹 3x3                 74.69%  퍼짐 1.37 (74.05~75.42)  합성곱 4층  매개변수 591,274  (421s)
  합성곱 6겹 3x3                 72.84%  퍼짐 3.25 (70.68~73.93)  합성곱 6층  매개변수 637,450  (392s)
  합성곱 4겹 3x3, 드롭아웃 없음        75.24%  퍼짐 1.23 (74.62~75.85)  합성곱 4층  매개변수 591,274  (242s)
```

---

## 2. 필터를 키우는 것은 여기서도 안 된다

| 갈래 | 평균 | 퍼짐 | 범위 | 매개변수 |
|---|---|---|---|---|
| 2겹 3×3 ([4.1절](01_two_ladders.md)) | 71.33% | 1.11 | 70.68~71.79 | 545,098 |
| 2겹 5×5 | 71.53% | **2.29** | 70.21~72.50 | 579,402 |

**+0.20%포인트, 퍼짐 안이다.** 그리고 퍼짐이 두 배로 벌어진다.

[3.4절](../ch03/mnist/04_cnn.md)에서 5×5가 아무것도 못 했을 때는 "잴 여지가 없어서"라고 읽을 수 있었다. 여기서는 그 변명이 통하지 않는다. **여지가 28%포인트나 있는데도 필터를 키우는 것으로는 그것을 먹지 못한다.**

---

## 3. 깊이는 먹는다

| 합성곱 | 평균 | 퍼짐 | 범위 | 매개변수 |
|---|---|---|---|---|
| 2겹 3×3 | 71.33% | 1.11 | 70.68~71.79 | 545,098 |
| **4겹 3×3** | **74.69%** | 1.37 | **74.05~75.42** | 591,274 |
| 6겹 3×3 | 72.84% | **3.25** | 70.68~73.93 | 637,450 |

**4겹에서 범위가 갈린다.** 가장 나쁜 실행(74.05)이 기준선의 가장 좋은 실행(71.79)보다 2.26%포인트 높다. [3.4절](../ch03/mnist/04_cnn.md)과 이 절을 통틀어 **처음으로 퍼짐을 넘는 차이**다.

### 얻은 것은 넓게 보기가 아니다

여기가 이 절의 요점이다. **5×5 한 겹과 3×3 두 겹은 받는 자리가 같다.**

3×3을 두 번 지나면 한 자리가 보는 범위가 5×5가 된다. 그런데 결과는 71.53%와 74.69%로 3.16%포인트 갈린다. 같은 넓이를 보는데 한쪽만 번다.

**다른 것은 그 사이에 ReLU가 하나 더 있다는 것뿐이다.** 곧 깊이가 버는 것은 넓게 보는 힘이 아니라 **한 번 더 굽힐 기회**다.

이것이 VGG가 큰 필터를 버리고 3×3만 쌓은 논거이며, 오늘날 표준이 3×3인 까닭이다. [다음 절](03_vgg16_transfer.md)의 VGG16에 합성곱이 13층 있는 것도 같은 생각을 끝까지 밀어붙인 결과다.

!!! note "매개변수는 근거가 되지 못한다"
    "3×3 두 겹이 5×5보다 매개변수가 적다"는 말을 흔히 듣는다. 채널 수가 그대로일 때는 맞지만($2 \times 9C^2 < 25C^2$), 이 블록은 채널을 늘리므로 성립하지 않는다. 4겹이 591,274개로 2겹 5×5의 579,402개보다 **많다.**

    게다가 `fc1` 하나가 524,416개라 세 갈래 모두 매개변수의 대부분이 거기 있다. **이 실험에서 갈린 것은 크기가 아니라 짜임이다.**

---

## 4. 6겹은 5 에포크로는 모자랐다

6겹은 4겹보다 **못하다**(72.84% 대 74.69%). 그리고 퍼짐이 3.25로 터진다. 가장 나쁜 실행(70.68)은 기준선의 가장 나쁜 실행과 똑같다 — 어떤 씨에서는 층을 네 개 더 얹고 아무것도 얻지 못했다.

깊을수록 낫다는 이야기와 어긋난다. **예산을 의심할 자리다.** [4.4절](04_augmentation.md)이 증강에서 5 에포크와 30 에포크의 결론이 뒤집히는 것을 보이므로, 같은 확인을 해 본다.

| 합성곱 | 5 에포크 | 30 에포크 | 예산으로 번 것 | 30ep 퍼짐 |
|---|---|---|---|---|
| 2겹 | 71.33% | 71.72% | **+0.39** | 0.84 |
| 4겹 | 74.69% | **76.32%** | +1.63 | 1.18 |
| 6겹 | 72.84% | 75.70% | **+2.86** | 1.86 |

*(30 에포크는 씨 세 개로 쟀다. 위 코드에서 상수 한 줄만 바꾸면 된다 —*
`EPOCHS, BATCH, LR, SEEDS = 30, 100, 1e-3, [0, 1, 2]`*.)*

**붕괴는 예산 탓이었다.** 6겹이 75.70%로 올라오고 퍼짐도 3.25에서 1.86으로 반이 된다.

그런데 그러고도 4겹을 넘지 못한다(76.32% 대 75.70%, 범위 겹침). **여섯째 층은 예산을 여섯 배 주어도 버는 것이 없다.**

### 예산을 쓸 줄 아는 것과 모르는 것

같은 표를 다르게 읽으면 더 큰 것이 보인다.

**얕은 그물은 예산을 쓰지 못한다.** 2겹은 6배 더 돌려도 +0.39%포인트로 퍼짐 안이다. [4.4절](04_augmentation.md)이 같은 설정에서 학습 정확도 99.05%를 보고하므로, 이미 외우고 있어 더 갈 데가 없다.

**깊은 그물은 쓴다.** 4겹은 +1.63, 6겹은 +2.86을 번다. 그래서 **간격이 예산과 함께 벌어진다** — 2겹과 4겹의 차이가 5 에포크에서 3.36%포인트, 30 에포크에서 4.60%포인트다.

곧 깊이는 두 가지를 한꺼번에 준다. **더 높은 천장과, 그 천장에 닿기 위한 더 큰 예산 요구**다.

---

## 5. 그래서 다음 절이 필요하다

여기까지가 맨바닥에서 얻을 수 있는 것이다.

| | 정확도 |
|---|---|
| [4.1절](01_two_ladders.md) 4걸음 | 71.78% |
| 4겹 3×3, 30 에포크 | **76.32%** |

4.5%포인트를 벌었고, 값은 층 두 개와 여섯 배의 시간이었다.

그런데 더 깊이 가려니 벽이 있다. 6겹은 예산을 여섯 배 주고서야 겨우 4겹 자리에 왔다. 층을 더 얹으려면 예산이 더 들고, 그러고도 벌 수 있을지 알 수 없다.

**이 벽을 누가 이미 넘어 두었다면?** [다음 절](03_vgg16_transfer.md)의 VGG16은 합성곱이 13층이며, ImageNet 120만 장으로 학습되어 있다. 우리가 치르지 못한 예산을 남이 치른 것이다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff normal" title="중간"></span>
본문은 5×5 한 겹과 3×3 두 겹이 "받는 자리가 같다"고 했다. 이를 확인하라. 그리고 3×3 세 겹은 몇 ×몇에 해당하는가?

</div>

??? success "연습문제 1 풀이"
    3×3 합성곱을 한 번 지나면 한 자리가 보는 범위가 한 쪽으로 1칸씩 늘어난다. 곧 받는 자리의 한 변이 $3 \to 5 \to 7$로 2씩 커진다.

    | 3×3을 겹친 수 | 받는 자리 | 매개변수(채널 $C$ 고정) |
    |---|---|---|
    | 1 | 3×3 | $9C^2$ |
    | 2 | **5×5** | $18C^2$ |
    | 3 | **7×7** | $27C^2$ |

    일반형은 $k$겹일 때 $2k+1$이다.

    채널이 고정이라면 3겹($27C^2$)이 7×7 한 겹($49C^2$)보다 싸다. VGG 논문이 든 근거가 이것이며, 겹칠수록 이득이 커진다. 다만 본문이 짚었듯 **채널을 늘리는 블록에서는 성립하지 않는다.**

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
6겹이 5 에포크에서 무너진 것을 본문은 "예산이 모자랐다"로 설명했다. 예산 말고 다른 설명은 없는가? 어떻게 가려낼 수 있는가?

</div>

??? success "연습문제 2 풀이"
    적어도 셋이 더 있다.

    **하나, 기울기가 깊이를 통과하지 못한다.** 층이 여섯이면 되짚기가 지나야 할 길이 길어진다. 30 에포크에서 회복한 것은 이 설명과도 어긋나지 않는다. 느리게라도 흐르면 오래 돌릴수록 나아지기 때문이다. 가려내려면 층마다 기울기의 크기를 재어 뒤로 갈수록 줄어드는지 보면 된다.

    **둘, 이 구조에서 6겹이 너무 깊다.** CIFAR-10은 32×32이고 풀링을 두 번 하면 8×8이 된다. 블록마다 세 겹을 쌓아도 볼 것이 더 없을 수 있다.

    **셋, 학습률이 맞지 않는다.** 깊은 그물은 흔히 더 작은 학습률을 원한다. 이 실험은 규약을 지키려고 $10^{-3}$으로 고정했으므로, 6겹에만 불리했을 수 있다.

    **가려내는 법은 하나씩 푸는 것이다.** [4.7절](07_distillation.md)이 대조군으로 이득을 둘로 나눈 것과 같은 방식이며, 예컨대 배치 정규화를 넣으면 첫째와 셋째가 함께 풀린다. 그래서 그 실험 하나로는 어느 쪽이 원인이었는지 여전히 가릴 수 없다.

    본문이 "예산 탓이었다"고 적은 것은 **예산을 늘리자 회복했다**는 사실까지만 말한 것이며, 다른 설명을 배제한 것이 아니다.
