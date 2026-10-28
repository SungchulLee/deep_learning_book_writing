# MNIST 뉴럴 넷트웍 — 전체 파이프라인

앞 쪽들은 조각을 하나씩 다루었다. 경사, 러닝레이트, 배치 크기, 옵티마이저. 이 쪽은 처음부터 끝까지 한 번 돌린다. 자료를 나누고, 모델을 세우고, 학습하고, 재고, 보고한다.

조각을 아는 것과 파이프라인을 세우는 것은 다른 일이다. **파이프라인에서만 생기는 실수** 들이 있고, 그 대부분은 예외를 던지지 않는다. 조용히 틀린 수를 만들 뿐이다. 이 쪽은 그런 자리를 하나씩 재어 본다.

---

## 1. 자료를 셋으로 나눈다

```
  MNIST 학습셋 60000 장을 48000 / 12000 으로 나눴다
  학습 48000,  검증 12000,  테스트 10000
  장치 mps
  학습 라벨 분포(%)  9.7 10.8  9.8  9.5  9.8  9.2 10.4 10.7  9.7 10.4   최소 9.2 최대 10.8
  검증 라벨 분포(%)  9.7 11.5 10.7  9.3 10.5  8.5 10.7  9.0 10.7  9.5   최소 8.5 최대 11.5
  테스트 라벨 분포(%)  7.6 11.4  9.0 10.8 10.4  9.4  9.2 11.8  9.2 11.2   최소 7.6 최대 11.8
```

**셋이 각각 다른 일을 한다.**

- **학습셋** — 파라메터를 갱신한다
- **검증셋** — 설정을 고른다(러닝레이트, 에폭 수, 모델 크기). 여기서 고르는 행위가 곧 검증셋에 맞추는 일이므로, 여기서 잰 수는 최종 성능이 아니다
- **테스트셋** — 다 끝난 뒤 **한 번만** 본다

무작위로 나누었는데 라벨 분포가 고르다. 9%에서 11% 사이다. 표본이 많기 때문이고, 자료가 적거나 라벨이 치우쳐 있으면 층화 추출을 해야 한다.

!!! note "테스트셋이 더 울퉁불퉁하다"
    7.6%에서 11.8%로 폭이 더 넓다. MNIST 테스트셋이 원래 그렇게 생겼다(여기서는 20장마다 하나씩 세어 본 것이라 그 자체로도 표본 오차가 있다). 테스트셋은 주어진 것이므로 고를 수 없고, 그래서 클래스별 정확도를 따로 보는 편이 안전하다.

---

## 2. 모델과 학습

모델은 784 → 128 → 64 → 10이고 은닉 레이어에 ReLU와 Dropout을 둔다. **마지막에 Softmax를 붙이지 않는다** — `CrossEntropyLoss`가 로짓을 받기 때문이다(연습문제 1에서 붙이면 어떻게 되는지 잰다).

```
  에폭   학습 배치평균   검증 로스   검증 정확도   그때의 lr
     0   0.4144        0.1832       94.78%     0.1000
     1   0.2259        0.1678       95.17%     0.1000
     2   0.1952        0.1534       95.64%     0.1000
     3   0.1816        0.1475       95.90%     0.1000
     4   0.1661        0.1407       96.24%     0.1000
     5   0.1540        0.1259       96.69%     0.1000
     6   0.1440        0.1215       96.83%     0.1000
     7   0.1389        0.1237       96.92%     0.1000
```

에폭 7에서 **검증 로스가 올라갔는데(0.1215 → 0.1237) 정확도는 올라갔다**(96.83% → 96.92%). 둘이 늘 같은 방향은 아니다. 로스는 확신의 정도까지 보고 정확도는 맞았는지만 보기 때문이다. 어느 쪽으로 모델을 고를지는 미리 정해 두어야 한다.

---

## 3. `model.train()`과 `model.eval()`

```
  eval()  로스 0.1237   정확도 96.92%
  train() 로스 0.1906   정확도 95.31%
  차이          +0.0669            -1.61%p
```

**같은 모델, 같은 자료인데 1.61%p 다르다.** `eval()`을 빼먹으면 이만큼 낮게 보고하게 된다.

더 고약한 것은 재현되지 않는다는 점이다.

```
  train() 모드는 돌릴 때마다 답이 달라진다
    1번째   로스 0.1899   정확도 95.26%
    2번째   로스 0.1805   정확도 95.38%
    3번째   로스 0.1900   정확도 95.34%
```

Dropout이 매번 다른 유닛을 끄기 때문이다. 평가를 두 번 하면 다른 수가 나오는데, 그 폭(95.26~95.38)이 실제 모델 간 차이와 비슷한 크기라 눈치채기 어렵다.

```
  Dropout 을 끈 모델이라면 두 모드가 같은가
    p_drop=0.0:  eval 94.77%   train 94.77%   차이 0.0000%p
```

`p_drop=0`이면 두 모드가 **완전히** 같다. 이 모델에는 BatchNorm이 없어 Dropout만이 모드에 반응하기 때문이다. BatchNorm이 있으면 `p_drop=0`이어도 갈린다 — 학습 모드에서는 배치 통계를, 평가 모드에서는 누적 통계를 쓴다.

!!! warning "`no_grad()`는 모드가 아니다"
    `torch.no_grad()`는 경사 기록을 끌 뿐 **모델의 행동을 바꾸지 않는다.** Dropout은 여전히 켜져 있다. 평가에는 둘 다 필요하다 — `model.eval()`로 행동을 바꾸고 `no_grad()`로 메모리를 아낀다.

---

## 4. "배치 평균의 평균"은 참 평균이 아니다

학습 로스를 적는 방법이 둘이다.

- **(가)** 배치마다의 평균 로스를 모아 다시 평균 낸다 — 거의 모든 학습 루프가 이렇게 한다
- **(나)** 로스의 합을 모아 전체 표본 수로 나눈다 — 참 평균

```
  배치 128   배치수  375  마지막  128장  (가) 0.16292799  (나) 0.16292799  차이 0.00e+00
  배치 127   배치수  378  마지막  121장  (가) 0.16291818  (나) 0.16292799  차이 9.81e-06
  배치 1000  배치수   48  마지막 1000장  (가) 0.16292799  (나) 0.16292799  차이 6.38e-10
```

나누어떨어지면 둘이 같다. 배치 127에서만 $9.8\times10^{-6}$ 벌어진다. **마지막 121장짜리 배치가 128장짜리와 같은 무게로 평균에 들어가기** 때문이다.

크기는 작다. $10^{-5}$ 자리이고 네 자리까지 보고한다면 보이지도 않는다. 그래도 알아 둘 값어치가 있다 — 자료가 적고 마지막 배치가 짧으면 커지고, 그때는 어느 쪽으로 쟀는지 밝혀야 한다.

**정확도에는 이 문제가 없다.** 맞힌 수를 전체 수로 나누면 그만이라 배치 경계와 무관하다.

---

## 5. 한 번 돌린 결과는 측정이 아니다

```
  씨앗   검증 정확도   테스트 정확도   둘의 차
     0       96.92%         96.85%    -0.07%p
     1       96.54%         96.62%    +0.08%p
     2       96.70%         96.78%    +0.08%p
     3       97.54%         97.44%    -0.10%p
     4       97.33%         97.53%    +0.20%p
  검증   평균 97.01%  퍼짐 96.54~97.54
  테스트 평균 97.04%  퍼짐 96.62~97.53
```

**같은 코드를 씨앗만 바꿔 돌렸는데 1%p 흩어진다.** 96.54%에서 97.54%다.

이것이 뜻하는 바는 분명하다. 어떤 변경을 하고 정확도가 0.5%p 올랐다면, **그것은 아무 일도 하지 않았을 때의 변동 범위 안이다.** 씨앗을 여럿 돌려 퍼짐을 함께 보고하지 않으면 그 0.5%p가 무엇인지 알 수 없다.

### 검증셋으로 고르고 테스트셋으로 보고한다

```
  검증으로 고르면 씨앗 3 를 고른다 (검증 97.54%)
  그 씨앗의 테스트 정확도는 97.44% 이고,
  다섯의 테스트 평균은 97.04% 다
  차이 +0.40%p
```

검증 정확도가 가장 높은 것을 고르면 테스트에서 97.44%를 얻는다. 평균보다 **0.40%p 높다.**

이것이 공짜 이득이 아니다. 다섯 중 검증에서 가장 운이 좋았던 것을 고른 것이고, 그 운의 일부가 테스트에도 묻어 왔다. 만약 검증 점수를 최종 성능으로 보고했다면 97.54%라고 적었을 텐데 실제는 97.44%다.

고를 것이 다섯이라 차이가 0.1%p지만, **수백 가지 설정을 검증셋으로 뒤지면 이 차이가 커진다.** 그래서 테스트셋을 따로 두고 한 번만 본다.

---

## 6. 스케줄러가 한 일

```
  스케줄러 켬  검증 정확도 평균 96.72%  퍼짐 96.54~96.92
              마지막 씨앗의 에폭별 lr: 0.1000 0.1000 0.1000 0.1000 0.1000 0.1000 0.1000 0.1000
  스케줄러 끔  검증 정확도 평균 96.72%  퍼짐 96.54~96.92
              마지막 씨앗의 에폭별 lr: 0.1000 0.1000 0.1000 0.1000 0.1000 0.1000 0.1000 0.1000
```

두 줄이 평균도 퍼짐도 **완전히 같다.** `ReduceLROnPlateau`가 **한 번도 켜지지 않았기** 때문이다. 러닝레이트가 여덟 에폭 내내 0.1이다.

2절의 표를 보면 까닭이 보인다. 검증 로스가 0.1832에서 0.1215까지 계속 내려갔다. `patience=1`이므로 두 에폭 연속 나빠져야 줄이는데 그런 일이 없었다.

**스케줄러를 코드에 넣었다고 일하는 것이 아니다.** 러닝레이트를 함께 기록해 실제로 켜졌는지 확인해야 한다. 그러지 않으면 "스케줄러를 써서 좋아졌다"고 적게 되고, 실제로는 아무 일도 없었다.

---

## 7. 전체 코드와 출력

MNIST를 읽으므로 저장소 뿌리에서 돌린다. 이 기계(mps)에서 약 5분이 걸린다.

```python
"""MNIST 전체 파이프라인 — 이 장에서 배운 것을 한자리에.

앞 쪽들은 조각을 하나씩 다루었다. 이 쪽은 처음부터 끝까지 한 번
돌린다. 자료를 나누고, 모델을 세우고, 학습하고, 재고, 보고한다.

조각을 아는 것과 파이프라인을 세우는 것은 다른 일이고, 그 차이가
드러나는 자리를 재어 본다.

- 검증셋과 테스트셋은 왜 따로 있는가 (수로 보면)
- model.train() 과 model.eval() 이 실제로 바꾸는 것
- '배치 평균의 평균' 과 참 평균이 얼마나 다른가
- 한 번 돌린 결과는 측정이 아니다 (씨앗을 바꾸면)
- 스케줄러가 실제로 한 일

저장소 뿌리에서 돌린다 (./data).

실행: python 01_neural_network_mnist.py
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH = 128


def header(title):
    print(f"\n=== {title} ===")


def load():
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    train_full = datasets.MNIST("./data", train=True, download=False,
                                transform=tf)
    test = datasets.MNIST("./data", train=False, download=False, transform=tf)
    g = torch.Generator().manual_seed(0)
    train, val = random_split(train_full, [48000, 12000], generator=g)
    return train, val, test


TRAIN, VAL, TEST = load()


class Net(nn.Module):
    def __init__(self, p_drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(64, 10),                 # 로짓. Softmax 를 붙이지 않는다
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader, train_mode=False):
    """로스를 **표본 수로 가중해** 더한다. 배치 평균의 평균이 아니다."""
    model.train() if train_mode else model.eval()
    tot_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEV), y.to(DEV)
            out = model(x)
            tot_loss += nn.functional.cross_entropy(
                out, y, reduction="sum").item()
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return tot_loss / n, correct / n * 100


# ========================================================================
# 1. 자료를 셋으로 나눈다
# ========================================================================

def splits():
    header("1. 학습 / 검증 / 테스트")
    print(f"  MNIST 학습셋 60000 장을 48000 / 12000 으로 나눴다")
    print(f"  학습 {len(TRAIN)},  검증 {len(VAL)},  테스트 {len(TEST)}")
    print(f"  장치 {DEV}")

    # 라벨 분포가 고르게 갈렸는지 확인한다
    def counts(ds):
        ys = torch.tensor([ds[i][1] for i in range(0, len(ds), 20)])
        return torch.bincount(ys, minlength=10).float() / len(ys) * 100

    for name, ds in (("학습", TRAIN), ("검증", VAL), ("테스트", TEST)):
        c = counts(ds)
        print(f"  {name} 라벨 분포(%) " +
              " ".join(f"{v:4.1f}" for v in c) +
              f"   최소 {c.min():.1f} 최대 {c.max():.1f}")
    print("  무작위로 나누었는데도 분포가 고르다. 표본이 많기 때문이다")
    print("  자료가 적거나 라벨이 치우쳐 있으면 층화 추출을 해야 한다")


# ========================================================================
# 2. 학습
# ========================================================================

def train_once(seed, epochs=8, lr=0.1, use_sched=True, p_drop=0.2,
               verbose=False):
    torch.manual_seed(seed)
    model = Net(p_drop).to(DEV)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=1)
    g = torch.Generator().manual_seed(seed)
    tl = DataLoader(TRAIN, batch_size=BATCH, shuffle=True, generator=g)
    vl = DataLoader(VAL, batch_size=512)

    hist = []
    for ep in range(epochs):
        model.train()
        batch_means = []
        for x, y in tl:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            batch_means.append(loss.item())
        vloss, vacc = evaluate(model, vl)
        if use_sched:
            sched.step(vloss)
        hist.append({"ep": ep, "batch_mean": float(np.mean(batch_means)),
                     "vloss": vloss, "vacc": vacc,
                     "lr": opt.param_groups[0]["lr"]})
        if verbose:
            print(f"  {ep:>4}   {hist[-1]['batch_mean']:.4f}        "
                  f"{vloss:.4f}      {vacc:6.2f}%     {hist[-1]['lr']:.4f}")
    return model, hist


def training():
    header("2. 학습 (8 에폭, SGD+모멘텀 0.9, lr 0.1, ReduceLROnPlateau)")
    print("  에폭   학습 배치평균   검증 로스   검증 정확도   그때의 lr")
    model, hist = train_once(0, verbose=True)
    return model, hist


# ========================================================================
# 3. train() 과 eval() 이 바꾸는 것
# ========================================================================

def modes(model):
    header("3. model.train() vs model.eval()")
    vl = DataLoader(VAL, batch_size=512)
    el, ea = evaluate(model, vl, train_mode=False)
    tl_, ta = evaluate(model, vl, train_mode=True)
    print(f"  eval()  로스 {el:.4f}   정확도 {ea:.2f}%")
    print(f"  train() 로스 {tl_:.4f}   정확도 {ta:.2f}%")
    print(f"  차이          {tl_-el:+.4f}            {ta-ea:+.2f}%p")
    print("  같은 모델, 같은 자료인데 다르다. Dropout 이 켜져 있기 때문이다")

    print("\n  train() 모드는 돌릴 때마다 답이 달라진다")
    for i in range(3):
        torch.manual_seed(100 + i)
        l, a = evaluate(model, vl, train_mode=True)
        print(f"    {i+1}번째   로스 {l:.4f}   정확도 {a:.2f}%")
    print("  eval() 을 빼먹으면 재현되지 않는 평가를 하게 된다")

    print("\n  Dropout 을 끈 모델이라면 두 모드가 같은가")
    m2, _ = train_once(0, epochs=1, p_drop=0.0)
    a1 = evaluate(m2, vl, train_mode=False)
    a2 = evaluate(m2, vl, train_mode=True)
    print(f"    p_drop=0.0:  eval {a1[1]:.2f}%   train {a2[1]:.2f}%   "
          f"차이 {abs(a1[1]-a2[1]):.4f}%p")
    print("  같다. 이 모델에는 BatchNorm 이 없어 Dropout 만이 모드에 반응한다")
    print("  no_grad() 는 모드와 다른 일을 한다 — 기록을 끌 뿐 행동을 안 바꾼다")


# ========================================================================
# 4. '배치 평균의 평균' 은 참 평균이 아니다
# ========================================================================

def mean_of_means(hist):
    header("4. 배치 평균의 평균 vs 표본 가중 평균")
    print("  학습 로스를 적는 두 방법이 있다")
    print("   (가) 배치마다의 평균 로스를 모아 다시 평균 낸다 (흔한 방법)")
    print("   (나) 로스의 합을 모아 전체 표본 수로 나눈다 (참 평균)")

    n_full = len(TRAIN)
    n_batches = -(-n_full // BATCH)
    print(f"\n  학습셋 {n_full}장, 배치 {BATCH} -> 배치 {n_batches}개")
    print(f"  마지막 배치 크기 {n_full - (n_batches-1)*BATCH}")
    print("  48000 = 128 * 375 이라 딱 떨어진다. 그러면 (가) 와 (나) 가 같다")

    print("\n  일부러 나누어떨어지지 않게 해 본다 (배치 127)")
    g = torch.Generator().manual_seed(0)
    model, _ = train_once(0, epochs=1)
    for bs in (128, 127, 1000):
        dl = DataLoader(TRAIN, batch_size=bs)
        model.eval()
        means, tot, n = [], 0.0, 0
        with torch.no_grad():
            for x, y in dl:
                x, y = x.to(DEV), y.to(DEV)
                out = model(x)
                means.append(nn.functional.cross_entropy(out, y).item())
                tot += nn.functional.cross_entropy(
                    out, y, reduction="sum").item()
                n += len(y)
        last = n_full - (len(means) - 1) * bs
        print(f"  배치 {bs:<5} 배치수 {len(means):>4}  마지막 {last:>4}장  "
              f"(가) {np.mean(means):.8f}  (나) {tot/n:.8f}  "
              f"차이 {abs(np.mean(means)-tot/n):.2e}")
    print("  나누어떨어지면 차이가 0 이거나(128) 부동소수점 자리다(1000).")
    print("  127 에서만 9.8e-06 으로 뜬다 — 마지막 121장짜리 배치가 128장짜리와")
    print("  같은 무게로 평균에 들어가기 때문이다")
    print("  크기는 작다. 그래도 적을 때는 어느 쪽인지 밝히는 편이 낫다")
    print("  정확도는 이 문제가 없다 — 맞힌 수를 전체 수로 나누면 그만이다")


# ========================================================================
# 5. 한 번 돌린 결과는 측정이 아니다
# ========================================================================

def spread():
    header("5. 씨앗을 바꾸면 (5번)")
    vl = DataLoader(VAL, batch_size=512)
    tl = DataLoader(TEST, batch_size=512)
    vaccs, taccs = [], []
    print("  씨앗   검증 정확도   테스트 정확도   둘의 차")
    for s in range(5):
        m, h = train_once(s)
        _, va = evaluate(m, vl)
        _, ta = evaluate(m, tl)
        vaccs.append(va)
        taccs.append(ta)
        print(f"  {s:>4}   {va:9.2f}%   {ta:11.2f}%   {ta-va:+6.2f}%p")
    print(f"  검증   평균 {np.mean(vaccs):.2f}%  "
          f"퍼짐 {min(vaccs):.2f}~{max(vaccs):.2f}")
    print(f"  테스트 평균 {np.mean(taccs):.2f}%  "
          f"퍼짐 {min(taccs):.2f}~{max(taccs):.2f}")
    print("  퍼짐이 이만큼 있다. 두 설정의 차이가 이보다 작으면 차이가 아니다")

    best = int(np.argmax(vaccs))
    print(f"\n  검증으로 고르면 씨앗 {best} 를 고른다 "
          f"(검증 {vaccs[best]:.2f}%)")
    print(f"  그 씨앗의 테스트 정확도는 {taccs[best]:.2f}% 이고,")
    print(f"  다섯의 테스트 평균은 {np.mean(taccs):.2f}% 다")
    print(f"  차이 {taccs[best]-np.mean(taccs):+.2f}%p — 검증으로 고르는 행위")
    print("  자체가 검증셋에 맞추는 일이다. 그래서 테스트셋이 따로 있다")
    print("  검증셋으로 고르고 테스트셋으로 한 번만 보고한다")


# ========================================================================
# 6. 스케줄러가 한 일
# ========================================================================

def scheduler_effect():
    header("6. ReduceLROnPlateau 가 한 일")
    vl = DataLoader(VAL, batch_size=512)
    for use in (True, False):
        accs = []
        for s in range(3):
            m, h = train_once(s, use_sched=use)
            accs.append(evaluate(m, vl)[1])
        lrs = [r["lr"] for r in h]
        lab = "스케줄러 켬" if use else "스케줄러 끔"
        print(f"  {lab}  검증 정확도 평균 {np.mean(accs):.2f}%  "
              f"퍼짐 {min(accs):.2f}~{max(accs):.2f}")
        print(f"              마지막 씨앗의 에폭별 lr: " +
              " ".join(f"{v:.4f}" for v in lrs))
    print("  두 줄이 평균도 퍼짐도 완전히 같다. 스케줄러가 **한 번도 켜지지**")
    print("  않았기 때문이다 — lr 이 여덟 에폭 내내 0.1 이다")
    print("  검증 로스가 8 에폭 동안 계속 내려가서 plateau 가 없었다")
    print("  스케줄러를 넣었다고 일하는 것이 아니다. lr 기록을 함께 찍어")
    print("  실제로 켜졌는지 확인해야 한다")


def main():
    splits()
    model, hist = training()
    modes(model)
    mean_of_means(hist)
    spread()
    scheduler_effect()


if __name__ == "__main__":
    main()
```

??? note "전체 출력 (85줄)"

    ```
    === 1. 학습 / 검증 / 테스트 ===
      MNIST 학습셋 60000 장을 48000 / 12000 으로 나눴다
      학습 48000,  검증 12000,  테스트 10000
      장치 mps
      학습 라벨 분포(%)  9.7 10.8  9.8  9.5  9.8  9.2 10.4 10.7  9.7 10.4   최소 9.2 최대 10.8
      검증 라벨 분포(%)  9.7 11.5 10.7  9.3 10.5  8.5 10.7  9.0 10.7  9.5   최소 8.5 최대 11.5
      테스트 라벨 분포(%)  7.6 11.4  9.0 10.8 10.4  9.4  9.2 11.8  9.2 11.2   최소 7.6 최대 11.8
      무작위로 나누었는데도 분포가 고르다. 표본이 많기 때문이다
      자료가 적거나 라벨이 치우쳐 있으면 층화 추출을 해야 한다

    === 2. 학습 (8 에폭, SGD+모멘텀 0.9, lr 0.1, ReduceLROnPlateau) ===
      에폭   학습 배치평균   검증 로스   검증 정확도   그때의 lr
         0   0.4144        0.1832       94.78%     0.1000
         1   0.2259        0.1678       95.17%     0.1000
         2   0.1952        0.1534       95.64%     0.1000
         3   0.1816        0.1475       95.90%     0.1000
         4   0.1661        0.1407       96.24%     0.1000
         5   0.1540        0.1259       96.69%     0.1000
         6   0.1440        0.1215       96.83%     0.1000
         7   0.1389        0.1237       96.92%     0.1000

    === 3. model.train() vs model.eval() ===
      eval()  로스 0.1237   정확도 96.92%
      train() 로스 0.1906   정확도 95.31%
      차이          +0.0669            -1.61%p
      같은 모델, 같은 자료인데 다르다. Dropout 이 켜져 있기 때문이다

      train() 모드는 돌릴 때마다 답이 달라진다
        1번째   로스 0.1899   정확도 95.26%
        2번째   로스 0.1805   정확도 95.38%
        3번째   로스 0.1900   정확도 95.34%
      eval() 을 빼먹으면 재현되지 않는 평가를 하게 된다

      Dropout 을 끈 모델이라면 두 모드가 같은가
        p_drop=0.0:  eval 94.77%   train 94.77%   차이 0.0000%p
      같다. 이 모델에는 BatchNorm 이 없어 Dropout 만이 모드에 반응한다
      no_grad() 는 모드와 다른 일을 한다 — 기록을 끌 뿐 행동을 안 바꾼다

    === 4. 배치 평균의 평균 vs 표본 가중 평균 ===
      학습 로스를 적는 두 방법이 있다
       (가) 배치마다의 평균 로스를 모아 다시 평균 낸다 (흔한 방법)
       (나) 로스의 합을 모아 전체 표본 수로 나눈다 (참 평균)

      학습셋 48000장, 배치 128 -> 배치 375개
      마지막 배치 크기 128
      48000 = 128 * 375 이라 딱 떨어진다. 그러면 (가) 와 (나) 가 같다

      일부러 나누어떨어지지 않게 해 본다 (배치 127)
      배치 128   배치수  375  마지막  128장  (가) 0.16292799  (나) 0.16292799  차이 0.00e+00
      배치 127   배치수  378  마지막  121장  (가) 0.16291818  (나) 0.16292799  차이 9.81e-06
      배치 1000  배치수   48  마지막 1000장  (가) 0.16292799  (나) 0.16292799  차이 6.38e-10
      나누어떨어지면 차이가 0 이거나(128) 부동소수점 자리다(1000).
      127 에서만 9.8e-06 으로 뜬다 — 마지막 121장짜리 배치가 128장짜리와
      같은 무게로 평균에 들어가기 때문이다
      크기는 작다. 그래도 적을 때는 어느 쪽인지 밝히는 편이 낫다
      정확도는 이 문제가 없다 — 맞힌 수를 전체 수로 나누면 그만이다

    === 5. 씨앗을 바꾸면 (5번) ===
      씨앗   검증 정확도   테스트 정확도   둘의 차
         0       96.92%         96.85%    -0.07%p
         1       96.54%         96.62%    +0.08%p
         2       96.70%         96.78%    +0.08%p
         3       97.54%         97.44%    -0.10%p
         4       97.33%         97.53%    +0.20%p
      검증   평균 97.01%  퍼짐 96.54~97.54
      테스트 평균 97.04%  퍼짐 96.62~97.53
      퍼짐이 이만큼 있다. 두 설정의 차이가 이보다 작으면 차이가 아니다

      검증으로 고르면 씨앗 3 를 고른다 (검증 97.54%)
      그 씨앗의 테스트 정확도는 97.44% 이고,
      다섯의 테스트 평균은 97.04% 다
      차이 +0.40%p — 검증으로 고르는 행위
      자체가 검증셋에 맞추는 일이다. 그래서 테스트셋이 따로 있다
      검증셋으로 고르고 테스트셋으로 한 번만 보고한다

    === 6. ReduceLROnPlateau 가 한 일 ===
      스케줄러 켬  검증 정확도 평균 96.72%  퍼짐 96.54~96.92
                  마지막 씨앗의 에폭별 lr: 0.1000 0.1000 0.1000 0.1000 0.1000 0.1000 0.1000 0.1000
      스케줄러 끔  검증 정확도 평균 96.72%  퍼짐 96.54~96.92
                  마지막 씨앗의 에폭별 lr: 0.1000 0.1000 0.1000 0.1000 0.1000 0.1000 0.1000 0.1000
      두 줄이 평균도 퍼짐도 완전히 같다. 스케줄러가 **한 번도 켜지지**
      않았기 때문이다 — lr 이 여덟 에폭 내내 0.1 이다
      검증 로스가 8 에폭 동안 계속 내려가서 plateau 가 없었다
      스케줄러를 넣었다고 일하는 것이 아니다. lr 기록을 함께 찍어
      실제로 켜졌는지 확인해야 한다
    ```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
모델 끝에 `nn.Softmax(dim=1)`을 붙이고 `CrossEntropyLoss`를 그대로 쓰면 어떻게 되는가? 로스와 경사를 재어 보라.

</div>

??? success "풀이"
    ```python
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    logits = torch.randn(4, 10) * 3
    y = torch.tensor([1, 3, 5, 7])

    print(f"  로짓 그대로 CrossEntropyLoss  "
          f"{nn.functional.cross_entropy(logits, y).item():.6f}")
    sm = torch.softmax(logits, 1)
    print(f"  Softmax 붙이고 CrossEntropyLoss "
          f"{nn.functional.cross_entropy(sm, y).item():.6f}")

    l1 = logits.clone().requires_grad_(True)
    nn.functional.cross_entropy(l1, y).backward()
    l2 = logits.clone().requires_grad_(True)
    nn.functional.cross_entropy(torch.softmax(l2, 1), y).backward()
    print(f"  경사 노름  올바름 {l1.grad.norm():.6f}   "
          f"Softmax 덧붙임 {l2.grad.norm():.6f}  "
          f"({l2.grad.norm() / l1.grad.norm():.4f} 배)")
    ```

    ```
      로짓 그대로 CrossEntropyLoss  5.418520
      Softmax 붙이고 CrossEntropyLoss 2.318047
      올바른 짝: log_softmax + nll   5.418520
      경사 노름  올바름 0.573800   Softmax 덧붙임 0.070368  (0.1226 배)
    ```

    **로스가 5.42에서 2.32로 좋아 보인다.** 이것이 이 버그의 고약한 점이다. 수가 나빠지면 바로 알아챌 텐데 오히려 좋아진다.

    `CrossEntropyLoss`는 내부에서 `log_softmax`를 적용한다. Softmax를 이미 적용한 값에 다시 적용하면 **Softmax를 두 번** 한 셈이다. 두 번째 Softmax의 입력이 $[0,1]$ 범위라 출력이 거의 균등해지고, 그래서 로스가 $\ln 10 = 2.303$ 근처로 눌린다.

    **진짜 피해는 경사에 있다.** 노름이 0.1226배, 곧 **8분의 1** 이다. 학습이 여덟 배 느려진다. 그리고 로스가 2.3 근처에 눌려 있으니 "수렴이 느린데 로스는 그럭저럭"으로 보여서 원인을 찾기 어렵다.

    고치는 길은 둘이다 — 모델 끝에서 Softmax를 **빼거나**, 그대로 두고 로스를 `NLLLoss`로 바꾸되 `Softmax` 대신 `LogSoftmax`를 쓰거나. 위 코드의 셋째 줄이 그 짝이 로짓 + `CrossEntropyLoss`와 같은 값을 준다는 것을 보여 준다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
Dropout 비율을 0, 0.2, 0.5로 바꾸어 재어라. 씨앗을 여럿 쓰고 퍼짐을 함께 보고하라.

</div>

??? success "풀이"
    ```
      p_drop   검증 정확도 평균   퍼짐
      0.0              96.41%   95.88~96.70
      0.2              96.23%   95.90~96.63
      0.5              92.93%   92.62~93.19
    ```

    (4에폭, 씨앗 3개)

    **0.0과 0.2의 퍼짐이 거의 완전히 겹친다**(95.88~96.70과 95.90~96.63). 차이를 말할 수 없다. 평균만 보면 Dropout을 **끈** 쪽이 0.18%p 높지만 그것도 잡음 안이다.

    0.5는 92.93%로 명백히 나쁘다. 퍼짐이 겹치지 않는다.

    **본문 모델이 `p_drop=0.2`를 쓰는 것이 이 측정으로 정당화되지 않는다.** 4에폭에서는 과적합이 일어나지 않아 정규화할 것이 없기 때문이다. 5절의 8에폭 학습에서도 검증 로스가 끝까지 내려갔다 — 아직 외우는 단계에 들어서지 않았다.

    Dropout이 값어치를 하려면 **과적합이 실제로 일어나야** 한다. 에폭을 50~100으로 늘리거나 학습셋을 1,000장으로 줄여 보면 그림이 달라질 것이다. 지금 설정에서 Dropout은 학습을 느리게 만드는 역할만 하고 있다.

    이것이 "좋다고 알려진 기법"을 재어 보지 않고 넣을 때 생기는 일이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
`transforms.Normalize((0.1307,), (0.3081,))`를 빼면 어떻게 되는가? 두 경우의 픽셀 값 범위도 함께 보라.

</div>

??? success "풀이"
    ```
      정규화    검증 정확도 평균   퍼짐
      True             96.23%   95.90~96.63
      False            96.91%   96.84~97.05
      정규화 함: 값 범위 -0.424~2.809  평균 -0.1005
      정규화 안 함: 값 범위 0.000~0.996  평균 +0.0997
    ```

    **정규화를 뺀 쪽이 더 좋다.** 96.91% 대 96.23%이고 퍼짐이 겹치지 않는다(96.84~97.05와 95.90~96.63).

    예상과 반대일 것이다. 까닭은 **MNIST 픽셀이 이미 $[0,1]$에 들어 있다** 는 데 있다. `ToTensor()`가 이미 255로 나누어 주므로 척도가 정상이다. 거기에 `Normalize`를 걸면 범위가 $[-0.42, 2.81]$로 **세 배 넓어진다.** 입력이 커지면 첫 레이어의 경사도 커지고, $\eta = 0.1$에서는 그것이 과했다.

    [러닝레이트 쪽](learning_rate.md) 연습문제 2에서 본 그대로다 — 입력의 척도를 바꾸면 러닝레이트도 다시 골라야 한다. 정규화와 러닝레이트를 **따로** 고르면 이런 일이 난다.

    **그러니 "정규화는 빼도 된다"가 결론이 아니다.** 두 결론은 이렇다.

    - MNIST처럼 이미 척도가 고른 자료에서는 정규화의 이득이 작고, 러닝레이트를 다시 맞추지 않으면 손해일 수도 있다
    - 특징마다 단위가 제각각인 자료(키와 몸무게와 소득)에서는 이야기가 전혀 다르다. 거기서는 정규화가 선택이 아니다

    `(0.1307, 0.3081)`이라는 수 자체도 확인해 둘 값어치가 있다. MNIST 학습셋 전체의 평균과 표준편차이고, **테스트셋의 값이 아니다.** 테스트셋으로 구한 통계를 쓰면 그 자체가 누수다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
검증셋까지 학습에 쓰고 그 검증셋으로 성능을 보고하면 얼마나 부풀려지는가? 테스트셋과 함께 재어 그 차이를 설명하라.

</div>

??? success "풀이"
    ```python
    from torch.utils.data import ConcatDataset

    BOTH = ConcatDataset([TRAIN, VAL])          # 48000 + 12000

    for lab, tr in (("학습셋만 (48000)", TRAIN), ("학습+검증 (60000)", BOTH)):
        va = [ev(train(tr, s), val_loader) for s in range(3)]
        ta = [ev(train(tr, s), test_loader) for s in range(3)]
        print(f"  {lab}  {np.mean(va):.2f}%   {np.mean(ta):.2f}%   "
              f"{np.mean(va) - np.mean(ta):+.2f}%p")
    ```

    ```
      학습에 쓴 것         검증 정확도   테스트 정확도   차
      학습셋만 (48000)         96.23%         96.01%    +0.22%p
      학습+검증 (60000)        97.23%         96.38%    +0.85%p
    ```

    **두 줄을 세로로 읽으면 안 되고 가로로 읽어야 한다.**

    검증 정확도만 보면 96.23%에서 97.23%로 **1.00%p 올랐다.** "자료를 더 썼더니 좋아졌다"로 읽힌다.

    그런데 테스트 정확도는 96.01%에서 96.38%로 **0.37%p** 올랐을 뿐이다. 진짜 이득은 그만큼이고, 자료를 25% 더 쓴 대가로는 그럴듯하다.

    나머지 0.63%p는 어디서 왔는가. **검증셋을 학습에 썼으므로 그 자료를 모델이 이미 본 것이다.** 마지막 열이 그것을 드러낸다 — 검증과 테스트의 차이가 0.22%p에서 0.85%p로 네 배가 되었다.

    **이 실수가 위험한 까닭.** 어디서도 예외가 나지 않고, 수가 **좋아지고**, 그 수가 보고서에 적힌다. 실제 배포에서 0.63%p 낮게 나와야 비로소 드러난다.

    실무에서 실제로 일어나는 꼴들은 이렇다.

    - 전처리 통계(평균·표준편차·스케일러)를 **나누기 전에** 전체 자료로 구한다
    - 시계열을 무작위로 나누어 미래의 표본이 학습셋에 들어간다
    - 중복된 표본이 양쪽에 나뉘어 들어간다
    - 설정을 수백 가지 시도하면서 매번 검증셋으로 고른다(느린 누수다)

    마지막 것은 본문 5절에서 이미 보았다. 다섯 중 하나만 골라도 0.40%p가 묻어 왔다.

    **쓸 수 있는 규칙.** 검증셋을 학습에 쓰는 것 자체가 나쁜 것은 아니다 — 설정을 다 고른 뒤 전체 자료로 다시 학습하는 것은 흔하고 합당한 절차다. 다만 그렇게 했다면 **검증 점수를 보고해서는 안 된다.** 보고할 수는 테스트셋의 수뿐이고, 그것도 한 번만 본다.

---

## 정리하며

1. **셋은 각각 다른 일을 한다**: 학습셋은 갱신, 검증셋은 고르기, 테스트셋은 마지막 한 번
2. **`eval()`을 빼먹으면 1.61%p 낮게, 그리고 재현되지 않게 보고한다**: Dropout이 매번 다른 유닛을 끈다. `no_grad()`는 이것을 고쳐 주지 않는다
3. **배치 평균의 평균은 나누어떨어지지 않을 때만 틀린다**: 여기서는 $10^{-5}$ 자리다. 정확도에는 이 문제가 없다
4. **씨앗만 바꿔도 1%p 흩어진다**: 96.54%에서 97.54%. 이보다 작은 개선은 개선이 아니다
5. **검증으로 고르는 것만으로 0.40%p가 묻어 온다**: 다섯 중 하나를 골랐을 뿐인데도 그렇다
6. **스케줄러는 넣었다고 켜지지 않는다**: 여덟 에폭 내내 lr이 0.1이었고 켠 쪽과 끈 쪽의 결과가 완전히 같았다. lr을 함께 기록해 확인한다
7. **Softmax를 덧붙이면 로스는 좋아 보이고 경사는 8분의 1이 된다**: 나빠지는 버그보다 좋아지는 버그가 오래 간다
8. **Dropout 0.2는 이 설정에서 값어치를 못 했다**: 0.0과 퍼짐이 겹친다. 과적합이 없으면 정규화할 것도 없다
9. **정규화도 재어 보고 넣는다**: MNIST에서는 빼는 쪽이 0.68%p 높았다. 척도를 바꾸면 러닝레이트도 다시 골라야 하기 때문이다

**참고 문헌**

- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11).
- Srivastava, N., et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *JMLR*, 15.
- Kaufman, S., et al. (2012). Leakage in Data Mining. *ACM TKDD*, 6(4).
