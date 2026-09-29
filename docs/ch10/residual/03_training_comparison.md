# 학습 비교

평범하게 쌓은 층더미와 잔차 연결을 넣은 층더미를 같은 자료로 학습시켜, 잔차 연결이 실제로 무엇을 바꾸는지 잰다. [기본 잔차 블록](01_basic_residual_block.md)이 블록 **하나**에서 기울기를 쟀다면, 이 쪽은 블록 여덟 개를 CIFAR-10으로 실제로 학습시킨다.

이런 비교는 쉽게 망가진다. 두 층더미의 층 수나 매개변수 수가 다르면, 마지막에 드러난 차이를 잔차 연결의 몫으로 돌릴 수 없다. 난수 가중치가 다른 것도 마찬가지다. 그래서 이 쪽은 층더미를 클래스 **하나**(`ConvStack`)로 만들고 `residual` 이라는 참/거짓 하나만 바꾼다. 층의 종류와 만드는 차례가 두 경우에 완전히 같으므로 같은 씨앗을 심으면 가중치가 글자 그대로 같아지고, 코드가 그것을 먼저 확인해서 찍는다.

## 1. 두 층더미의 유일한 차이

블록 하나가 하는 일을 $G_l$ 이라 쓰자 — 합성곱, 배치 정규화, ReLU, 합성곱, 배치 정규화까지다. $\sigma$ 를 ReLU라 하면

$$G_l(x) = \mathrm{BN}_2\!\left(W_2\,\sigma\!\left(\mathrm{BN}_1(W_1 x)\right)\right)$$

이다. 두 층더미는 이 $G_l$ 을 똑같이 쓰고 마지막 한 걸음만 다르다.

$$\begin{aligned}
\text{평범한 블록:}\quad & x_{l+1} = \sigma\!\left(G_l(x_l)\right) \\
\text{잔차 블록:}\quad & x_{l+1} = \sigma\!\left(G_l(x_l) + x_l\right)
\end{aligned}$$

야코비를 적으면 차이가 어디서 오는지 보인다. ReLU가 살린 자리에 1, 죽인 자리에 0을 놓은 대각행렬을 $D_l$ 이라 하면

$$\begin{aligned}
\text{평범한 블록:}\quad & \frac{\partial x_{l+1}}{\partial x_l} = D_l\, G_l'(x_l) \\
\text{잔차 블록:}\quad & \frac{\partial x_{l+1}}{\partial x_l} = D_l\left(G_l'(x_l) + I\right)
\end{aligned}$$

블록 $L$ 개를 지나면 평범한 쪽은 $\prod_l D_l G_l'$ 이라는 **곱** 하나다. 잔차 쪽은 $\prod_l D_l (G_l' + I)$ 를 펼친 $2^L$ 개 항의 **합**이 되고, 그 합 안에 $G'$ 을 한 번도 거치지 않는 항이 들어 있다. 잔차 연결이 하는 일은 이것이 전부다.

!!! warning "덧셈 분해는 하한이 아니다"
    괄호 안에 $I$ 가 들어 있다고 해서 기울기가 1 이상이 되지는 않는다. $D_l$ 이 ReLU가 죽인 자리를 0으로 만들고, $G_l'(x_l) = -I$ 이면 괄호 안이 통째로 0이다. [항등 사상](identity_mapping.md)이 같은 함정을 반례와 함께 다룬다.

## 2. 코드

```python
"""
학습 비교: 잔차 층더미와 평범한 층더미
======================================
같은 씨앗에서 가중치가 글자 그대로 같은 두 층더미를 CIFAR-10 조각으로
학습시킨다. 두 층더미는 블록 안에서 `out + x` 한 항만 다르므로, 드러나는
차이는 모두 그 항의 몫이다. 씨앗 셋을 돌려 씨앗 사이의 폭도 함께 찍는다.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

DEPTH = 8          # 블록 수 (블록마다 합성곱 2개)
WIDTH = 16         # 채널 수
N_TRAIN = 1280     # CIFAR-10 학습 조각
N_VAL = 512        # CIFAR-10 검증 조각
BATCH = 64
EPOCHS = 5
SEEDS = (0, 1, 2)

# ========================================================================
# 층더미 — `residual` 한 글자만 다르다
# ========================================================================


class ConvStack(nn.Module):
    """블록 DEPTH개를 쌓은 층더미.

    블록 하나는 conv-BN-ReLU-conv-BN 이고, `residual` 이 참이면 그 뒤에
    입력을 더한 다음 ReLU를 건다. 거짓이면 더하지 않고 바로 ReLU를 건다.
    층의 종류와 만드는 차례가 두 경우에 완전히 같으므로, 같은 씨앗을 심으면
    두 층더미의 가중치는 글자 그대로 같아진다.

    합성곱의 `bias=False`는 바로 뒤의 배치 정규화가 치우침을 지우기 때문이다.
    """

    def __init__(self, residual, depth=DEPTH, width=WIDTH, num_classes=10):
        super().__init__()
        self.residual = residual
        # stride 2 합성곱과 최대 풀링으로 32x32 를 8x8 로 줄인다. ResNet 의
        # 머리처럼 블록에 닿기 전에 두 번 반으로 줄이는 꼴이고, 블록들이
        # 8x8 에서 돌아 씨앗 셋을 돌릴 만큼 싸진다.
        self.stem = nn.Conv2d(3, width, kernel_size=3, stride=2, padding=1,
                              bias=False)
        self.stem_bn = nn.BatchNorm2d(width)
        self.stem_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleDict({
                'conv1': nn.Conv2d(width, width, 3, padding=1, bias=False),
                'bn1': nn.BatchNorm2d(width),
                'conv2': nn.Conv2d(width, width, 3, padding=1, bias=False),
                'bn2': nn.BatchNorm2d(width),
            }))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(width, num_classes)

    def forward(self, x):
        x = self.stem_pool(torch.relu(self.stem_bn(self.stem(x))))
        for blk in self.blocks:
            out = torch.relu(blk['bn1'](blk['conv1'](x)))
            out = blk['bn2'](blk['conv2'](out))
            if self.residual:
                out = out + x          # 이 한 항이 두 층더미의 유일한 차이다
            x = torch.relu(out)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def build(residual, seed):
    """씨앗을 먼저 심고 층더미를 만든다 — 같은 씨앗이면 가중치가 같다."""
    torch.manual_seed(seed)
    return ConvStack(residual)


# ========================================================================
# 자료 — CIFAR-10의 고정된 조각
# ========================================================================


def load_subsets():
    """학습 1280장, 검증 512장. 뽑는 자리는 씨앗 0으로 못 박아 모든 씨앗이
    똑같은 자료를 본다. 달라지는 것은 가중치의 초기값과 섞는 차례뿐이다."""
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_full = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=tf)
    val_full = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=tf)
    g = torch.Generator().manual_seed(0)
    tr = torch.randperm(len(train_full), generator=g)[:N_TRAIN].tolist()
    va = torch.randperm(len(val_full), generator=g)[:N_VAL].tolist()
    return Subset(train_full, tr), Subset(val_full, va)


# ========================================================================
# 기울기 재기
# ========================================================================


def grad_profile(model, x, y):
    """고정된 배치 하나로 블록별·전체 매개변수 기울기의 노름을 잰다.

    배치 정규화를 학습 모드로 두어야 층더미가 실제로 쓰는 정규화가 걸린다.
    대신 달리는 통계가 이 한 배치로 오염되지 않도록 되돌려 놓는다.
    """
    was_training = model.training
    model.train()
    saved = [(m, m.running_mean.clone(), m.running_var.clone(),
              m.num_batches_tracked.clone())
             for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    model.zero_grad(set_to_none=True)
    nn.CrossEntropyLoss()(model(x), y).backward()
    per_block = [blk['conv1'].weight.grad.norm().item() for blk in model.blocks]
    total = torch.sqrt(
        sum((p.grad ** 2).sum() for p in model.parameters())).item()
    model.zero_grad(set_to_none=True)
    for m, rm, rv, nb in saved:
        m.running_mean.copy_(rm)
        m.running_var.copy_(rv)
        m.num_batches_tracked.copy_(nb)
    model.train(was_training)
    return per_block, total


# ========================================================================
# 학습
# ========================================================================


def run_epoch(model, loader, opt=None):
    """opt가 있으면 한 세대 학습시키고, 없으면 평가한다.

    손실은 표본 수로 나눈다. 배치 수로 나누면 마지막 배치가 짧을 때
    (1280 = 64 x 20 이라 여기서는 나누어떨어지지만, 크기를 바꾸면
    곧바로 어긋난다) 짧은 배치가 지나치게 무거워진다.
    """
    train = opt is not None
    model.train(train)
    crit = nn.CrossEntropyLoss()
    loss_sum, correct, n = 0.0, 0, 0
    with torch.enable_grad() if train else torch.no_grad():
        for xb, yb in loader:
            out = model(xb)
            loss = crit(out, yb)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            loss_sum += loss.item() * yb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            n += yb.size(0)
    return loss_sum / n, 100.0 * correct / n


if __name__ == "__main__":
    print("=" * 78)
    print(f"Plain vs residual stack | {DEPTH} blocks = {2 * DEPTH + 1} convs, "
          f"width {WIDTH}, CIFAR-10 {N_TRAIN}/{N_VAL}, {EPOCHS} epochs")
    print("=" * 78)

    train_set, val_set = load_subsets()

    # --- 두 층더미가 정말 같은 가중치에서 출발하는지 ---
    plain0, res0 = build(False, 0), build(True, 0)
    same = all(torch.equal(a, b) for a, b in zip(plain0.state_dict().values(),
                                                 res0.state_dict().values()))
    print(f"\nSame weights under the same seed: {same}")
    print(f"Parameters: plain {sum(p.numel() for p in plain0.parameters()):,}, "
          f"residual {sum(p.numel() for p in res0.parameters()):,}")

    # --- 초기 기울기의 깊이별 옆모습 (가중치가 같으므로 차이는 `+ x` 뿐이다) ---
    probe_x, probe_y = next(iter(DataLoader(train_set, batch_size=BATCH)))
    prof = {res: [grad_profile(build(res, s), probe_x, probe_y) for s in SEEDS]
            for res in (False, True)}

    print(f"\nGradient norm of conv1.weight at initialization, "
          f"seeds {SEEDS} min-max")
    print(f"{'block':>5} | {'plain':>19} | {'residual':>19}")
    print("-" * 49)
    for b in range(DEPTH):
        pv = [r[0][b] for r in prof[False]]
        rv = [r[0][b] for r in prof[True]]
        print(f"{b + 1:>5} | {min(pv):>8.4f} - {max(pv):<8.4f} | "
              f"{min(rv):>8.4f} - {max(rv):<8.4f}")
    pt = [r[1] for r in prof[False]]
    rt = [r[1] for r in prof[True]]
    print(f"{'all':>5} | {min(pt):>8.4f} - {max(pt):<8.4f} | "
          f"{min(rt):>8.4f} - {max(rt):<8.4f}")

    p_ratio = [r[0][0] / r[0][-1] for r in prof[False]]
    r_ratio = [r[0][0] / r[0][-1] for r in prof[True]]
    print(f"block 1 / block {DEPTH}:  plain "
          f"{min(p_ratio):.3f} - {max(p_ratio):.3f},  residual "
          f"{min(r_ratio):.3f} - {max(r_ratio):.3f}")

    # --- 학습 ---
    hist = {False: {}, True: {}}
    for s in SEEDS:
        g = torch.Generator().manual_seed(s)
        train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,
                                  generator=g)
        val_loader = DataLoader(val_set, batch_size=BATCH)
        for res in (False, True):
            model = build(res, s)
            opt = optim.Adam(model.parameters(), lr=1e-3)
            rows = []
            for _ in range(EPOCHS):
                tr_loss, tr_acc = run_epoch(model, train_loader, opt)
                va_loss, va_acc = run_epoch(model, val_loader)
                _, gnorm = grad_profile(model, probe_x, probe_y)
                rows.append((tr_loss, tr_acc, va_loss, va_acc, gnorm))
            hist[res][s] = rows
        print(f"\n--- seed {s} ---")
        print(f"{'ep':>2} | {'plain: loss  acc   val  grad':>30} | "
              f"{'residual: loss  acc   val  grad':>30}")
        for e in range(EPOCHS):
            p = hist[False][s][e]
            r = hist[True][s][e]
            print(f"{e + 1:>2} | {p[0]:>11.4f} {p[1]:5.2f} {p[3]:5.2f} "
                  f"{p[4]:7.3f} | {r[0]:>13.4f} {r[1]:5.2f} {r[3]:5.2f} "
                  f"{r[4]:7.3f}")

    # --- 씨앗을 가로질러 본 마지막 값 ---
    print("\n" + "=" * 78)
    print(f"After {EPOCHS} epochs, across seeds {SEEDS}")
    print("=" * 78)
    summary = {}
    for name, res in (("plain   ", False), ("residual", True)):
        tr = [hist[res][s][-1][1] for s in SEEDS]
        va = [hist[res][s][-1][3] for s in SEEDS]
        gn = [hist[res][s][-1][4] for s in SEEDS]
        summary[res] = (tr, va, gn)
        print(f"{name} train acc {min(tr):5.2f}-{max(tr):5.2f}  "
              f"val acc {min(va):5.2f}-{max(va):5.2f} "
              f"(mean {sum(va) / len(va):5.2f})  grad {min(gn):.3f}-{max(gn):.3f}")

    pv, rv = summary[False][1], summary[True][1]
    gap = sum(rv) / len(rv) - sum(pv) / len(pv)
    spread = max(max(pv) - min(pv), max(rv) - min(rv))
    print(f"\nval acc: gap of means {gap:+.2f} points, widest seed spread "
          f"{spread:.2f} points")
    print("verdict: " + ("the gap is larger than the spread"
                         if abs(gap) > spread else
                         "the gap is inside the spread - not a difference"))
    print("=" * 78)
```

??? note "전체 출력 (56줄)"

    ```
    ==============================================================================
    Plain vs residual stack | 8 blocks = 17 convs, width 16, CIFAR-10 1280/512, 5 epochs
    ==============================================================================
    Files already downloaded and verified
    Files already downloaded and verified

    Same weights under the same seed: True
    Parameters: plain 38,010, residual 38,010

    Gradient norm of conv1.weight at initialization, seeds (0, 1, 2) min-max
    block |               plain |            residual
    -------------------------------------------------
        1 |   1.3619 - 1.9331   |   0.8807 - 1.0686  
        2 |   0.9176 - 1.2537   |   0.6462 - 0.7791  
        3 |   0.6744 - 0.7905   |   0.4801 - 0.6951  
        4 |   0.4846 - 0.5605   |   0.3710 - 0.5701  
        5 |   0.3502 - 0.4330   |   0.4368 - 0.5181  
        6 |   0.2423 - 0.2868   |   0.3098 - 0.4480  
        7 |   0.1982 - 0.2095   |   0.4327 - 0.4551  
        8 |   0.1571 - 0.2291   |   0.3642 - 0.4355  
      all |   2.7300 - 3.5850   |   3.1136 - 4.1751  
    block 1 / block 8:  plain 5.945 - 12.302,  residual 2.022 - 2.934

    --- seed 0 ---
    ep |   plain: loss  acc   val  grad | residual: loss  acc   val  grad
     1 |      2.2821 13.28  8.79   1.474 |        2.2374 15.78 20.51   2.473
     2 |      2.1885 18.59 15.62   1.222 |        1.9153 32.27 28.71   2.341
     3 |      2.0992 21.88 17.38   2.396 |        1.7395 35.00 36.91   2.879
     4 |      2.0171 23.28 19.92   2.439 |        1.6212 39.77 36.13   3.865
     5 |      1.9461 24.77 16.60   2.928 |        1.5291 44.45 39.26   3.103

    --- seed 1 ---
    ep |   plain: loss  acc   val  grad | residual: loss  acc   val  grad
     1 |      2.2926 12.81 10.94   1.917 |        2.3868 12.73 17.19   2.125
     2 |      2.2164 18.05 19.53   2.022 |        1.9682 24.69 25.59   2.299
     3 |      2.1188 23.75 21.29   2.448 |        1.7570 36.48 29.49   3.892
     4 |      2.0044 28.44 22.85   3.447 |        1.6044 41.41 32.23   3.759
     5 |      1.9091 30.62 22.85   3.594 |        1.4688 46.80 35.74   5.481

    --- seed 2 ---
    ep |   plain: loss  acc   val  grad | residual: loss  acc   val  grad
     1 |      2.3060 12.34 10.16   2.306 |        2.3482 16.02 15.23   2.266
     2 |      2.2249 16.64 15.43   1.827 |        1.9591 29.84 29.10   2.421
     3 |      2.1413 19.06 16.41   2.231 |        1.7492 39.30 35.35   3.621
     4 |      2.0520 22.97 18.36   3.283 |        1.5912 44.92 31.84   4.287
     5 |      1.9640 25.94 18.16   2.900 |        1.4626 47.89 32.03   5.613

    ==============================================================================
    After 5 epochs, across seeds (0, 1, 2)
    ==============================================================================
    plain    train acc 24.77-30.62  val acc 16.60-22.85 (mean 19.21)  grad 2.900-3.594
    residual train acc 44.45-47.89  val acc 32.03-39.26 (mean 35.68)  grad 3.103-5.613

    val acc: gap of means +16.47 points, widest seed spread 7.23 points
    verdict: the gap is larger than the spread
    ==============================================================================
    ```

## 3. 출력 읽기

### 3.1 같은 가중치에서 출발했는가

```
Same weights under the same seed: True
Parameters: plain 38,010, residual 38,010
```

이 두 줄이 이 쪽의 나머지를 떠받친다. 첫 줄은 두 층더미의 `state_dict` 를 항목마다 `torch.equal` 로 견준 결과다 — 가중치뿐 아니라 배치 정규화의 $\gamma$, $\beta$, 달리는 평균과 분산까지 모두 같다. 둘째 줄의 38,010은 `out + x` 가 배울 것이 없는 덧셈이라 매개변수를 하나도 쓰지 않기 때문이다.

이 두 줄이 없으면 뒤의 모든 수가 무엇의 결과인지 알 수 없다. 씨앗을 한 번만 심고 두 층더미를 잇달아 만들면 이 줄은 `False` 로 바뀐다(연습문제 3). 앞 쪽 [항등 사상](identity_mapping.md)이 바로 그 함정에 걸려 있었다.

### 3.2 초기 기울기의 깊이별 옆모습

가중치가 같으므로, 학습을 시작하기 전에 잰 기울기의 차이는 통째로 `+ x` 한 항의 몫이다.

| | 평범 | 잔차 |
|---|---|---|
| 블록 1 | 1.3619 ~ 1.9331 | 0.8807 ~ 1.0686 |
| 블록 8 | 0.1571 ~ 0.2291 | 0.3642 ~ 0.4355 |
| 블록 1 ÷ 블록 8 | 5.945 ~ 12.302 | 2.022 ~ 2.934 |
| 층더미 전체 | 2.7300 ~ 3.5850 | 3.1136 ~ 4.1751 |

읽을 것은 셋째 줄이다. 평범한 층더미는 첫 블록에서 마지막 블록으로 가며 기울기가 6\~12배 줄지만, 잔차 층더미는 2\~3배밖에 줄지 않는다. 씨앗 셋의 폭이 두 열 사이에서 겹치지 않으므로 이 차이는 씨앗 잡음보다 크다. 잔차 연결은 기울기의 **총량**이 아니라 **깊이에 따른 기울어짐**을 바꾼다.

넷째 줄이 그 점을 거꾸로 확인해 준다. 층더미 전체의 기울기 노름은 두 범위가 겹친다(평범의 3.5850이 잔차의 3.1136보다 크다). 총량만 보아서는 두 층더미를 가를 수 없다.

### 3.3 학습 곡선과 씨앗 사이의 폭

```
plain    train acc 24.77-30.62  val acc 16.60-22.85 (mean 19.21)  grad 2.900-3.594
residual train acc 44.45-47.89  val acc 32.03-39.26 (mean 35.68)  grad 3.103-5.613

val acc: gap of means +16.47 points, widest seed spread 7.23 points
verdict: the gap is larger than the spread
```

한 번 돌린 것은 재어 본 것이 아니다. 그래서 코드는 씨앗 셋을 돌리고, 평균의 차이를 씨앗 사이의 가장 넓은 폭과 나란히 찍는다. 검증 정확도의 차이는 16.47점이고 가장 넓은 씨앗 폭은 7.23점이다 — 차이가 폭보다 두 배 넘게 크다. 더 강하게는 두 범위가 아예 겹치지 않는다. 평범한 층더미의 가장 좋은 씨앗(22.85%)이 잔차 층더미의 가장 나쁜 씨앗(32.03%)에 못 미친다.

학습 정확도에서도 같다(24.77\~30.62 대 44.45\~47.89). 다섯 세대는 두 층더미 모두에게 짧아 절대 정확도 자체는 낮지만 — 열 갈래이므로 찍기가 10%다 — 견주는 데에는 지장이 없다. 두 층더미가 같은 자료를 같은 차례로 보고 같은 가중치에서 출발했기 때문이다.

한편 **기울기의 총 노름은 이 차이를 설명하지 못한다.** 씨앗 2의 첫 세대에서는 평범한 쪽이 2.306, 잔차 쪽이 2.266으로 평범한 쪽이 오히려 크다. 다섯 세대째에는 세 씨앗 모두 잔차 쪽이 크지만, "잔차 신경망은 학습 내내 더 큰 기울기 노름을 유지한다"는 식의 말은 이 표가 받쳐 주지 않는다. 받쳐 주는 것은 3.2절의 깊이별 옆모습이다.

### 3.4 이 쪽이 뒷받침하지 못하는 것

**기울기 소실이 일어난 것이 아니다.** 평범한 층더미의 블록 8에서도 기울기 노름은 0.1571\~0.2291이다. 0에 가까운 수가 아니라 블록 1의 6\~12분의 1일 뿐이다. 배치 정규화가 층마다 활성값의 크기를 다시 맞추므로 신호가 깊이에 따라 기하급수로 죽을 여지가 없다. [기본 잔차 블록](01_basic_residual_block.md)이 잰 표는 같은 이야기를 더 세게 한다 — 깊이 8과 16에서 평범한 층더미의 입력 기울기는 잔차 쪽보다 오히려 **크고**(16층에서 26,672\~28,533 대 1,680\~1,764), 문제는 소실이 아니라 폭주다. [항등 사상](identity_mapping.md)도 블록 50개에서 사후 활성화의 기울기 비를 1.052로 재어, 그 깊이에서도 사라지지 않음을 보였다.

**깊이 8은 원 논문이 말하는 깊이가 아니다.** He 등이 잔차 연결을 내놓은 자리는 110층과 1,001층이다. 이 쪽은 합성곱 17개짜리 층더미를 CIFAR-10 1,280장으로 다섯 세대 돌린 것이고, 씨앗 셋을 감당할 만큼 작게 잡은 것이다. 여기서 나온 16.47점을 깊이가 다른 구조로 옮겨 적을 수는 없다.

**절대 정확도는 이 쪽의 주장이 아니다.** 1,280장에 다섯 세대, 채널 16개로 얻은 35.68%는 CIFAR-10에서 좋은 값이 아니다. 이 쪽이 주장하는 것은 두 수의 **차이**뿐이고, 그 차이가 뜻을 가지는 까닭은 오직 두 층더미가 한 항만 빼고 모든 것을 나누어 가졌기 때문이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
출력이 찍은 매개변수 38,010개를 손으로 검산하라. 머리(`stem`, `stem_bn`), 블록 하나, 마지막 선형층으로 나누어 세고, 두 층더미의 개수가 같은 까닭도 한 줄로 적어라.

</div>

??? success "연습문제 1 풀이"
    머리의 합성곱은 `bias=False` 이므로 $3 \times 16 \times 3 \times 3 = 432$개이고, 배치 정규화는 $\gamma$ 와 $\beta$ 를 채널마다 하나씩 가지므로 $2 \times 16 = 32$개다. `MaxPool2d` 는 매개변수가 없다.

    블록 하나는 `bias=False` 인 $16 \to 16$ 합성곱 둘과 배치 정규화 둘이다.

    $$2 \times (16 \times 16 \times 3 \times 3) + 2 \times (2 \times 16) = 4608 + 64 = 4672$$

    블록이 여덟이므로 $8 \times 4672 = 37{,}376$개다. 마지막 선형층은 $16 \times 10 + 10 = 170$개다. 모두 더하면

    $$432 + 32 + 37{,}376 + 170 = 38{,}010$$

    으로 출력과 맞는다. 두 층더미가 같은 개수를 갖는 까닭은 간단하다 — `out + x` 는 배울 것이 없는 덧셈이라 매개변수를 하나도 쓰지 않는다. 잔차 연결은 **공짜**다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
$32 \times 32$ 입력이 블록에 닿을 때까지의 공간 크기를 따라가라. 그리고 블록 안의 `out` 과 `x` 가 왜 언제나 모양이 맞는지 설명하라.

</div>

??? success "연습문제 2 풀이"
    합성곱의 출력 크기는 $\left\lfloor (H + 2p - k)/s \right\rfloor + 1$ 이다.

    - 머리의 합성곱($k = 3$, $p = 1$, $s = 2$): $\lfloor (32 + 2 - 3)/2 \rfloor + 1 = 15 + 1 = 16$
    - `MaxPool2d(2, 2)`($k = 2$, $p = 0$, $s = 2$): $\lfloor (16 - 2)/2 \rfloor + 1 = 7 + 1 = 8$

    그래서 블록들은 $8 \times 8$ 에서 돌아간다. 블록 안의 합성곱은 $k = 3$, $p = 1$, $s = 1$ 이므로 $\lfloor (8 + 2 - 3)/1 \rfloor + 1 = 8$ 로 크기를 그대로 둔다. 채널도 $16 \to 16$ 으로 그대로다. 따라서 `out` 과 `x` 는 둘 다 $(B, 16, 8, 8)$ 이고 덧셈이 언제나 맞는다.

    크기나 채널이 바뀌는 자리에서는 이 말이 깨진다. 그때 지름길에 1×1 사영을 넣는 까닭을 [기본 잔차 블록](01_basic_residual_block.md)이 다룬다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
`build` 를 쓰지 않고 씨앗을 **한 번만** 심은 뒤 두 층더미를 잇달아 만들어 보라.

```python
torch.manual_seed(0)
plain = ConvStack(False)
residual = ConvStack(True)
```

`Same weights under the same seed` 가 무엇으로 바뀌는가? 왜 그런가? 그리고 그렇게 견주었을 때 마지막 정확도의 차이는 무엇을 재는 것인가?

</div>

??? success "연습문제 3 풀이"
    `False` 가 된다. `torch.manual_seed(0)` 은 전역 난수 생성기를 한 번 되돌릴 뿐이고, `ConvStack(False)` 를 만드는 동안 그 생성기가 38,010개의 값을 뽑아 앞으로 나아간다. 이어서 만드는 `ConvStack(True)` 는 **그 다음** 값들을 받는다. 두 층더미의 첫 블록 `conv1.weight` 를 빼 보면 최대 차이가 0.1631로, 가중치의 크기와 같은 자릿수다.

    그렇게 견주면 마지막 정확도의 차이는 "잔차 연결의 효과"가 아니라 "잔차 연결의 효과 + 서로 다른 초기값의 효과"다. 뒤엣것이 얼마나 큰지는 이 쪽의 출력이 직접 말해 준다 — 같은 구조라도 씨앗만 바꾸면 검증 정확도가 몇 점씩 움직인다. 초기값 차이를 섞어 넣으면 그만큼의 잡음을 결론으로 읽게 된다.

    고치는 방법이 `build(residual, seed)` 다. 씨앗을 **층을 만들기 직전에** 심고, 두 경우가 같은 종류의 층을 같은 차례로 만들게 두면 가중치가 글자 그대로 같아진다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
초기 기울기 표에서 블록 1과 블록 8의 값을 읽어라. 이 표가 "평범한 층더미에서는 기울기가 사라진다"는 말을 뒷받침하는가? 뒷받침하지 않는다면, 표가 실제로 보이는 것은 무엇인가?

</div>

??? success "연습문제 4 풀이"
    뒷받침하지 않는다. 평범한 층더미의 블록 8에서도 기울기의 노름은 0.1571\~0.2291로, 0이 아니라 블록 1의 6\~12분의 1일 뿐이다. 기울기 소실이라는 말은 $10^{-15}$ 같은 자릿수를 가리키는 말이고, 여기서 그런 일은 일어나지 않는다. 까닭은 배치 정규화다 — 층마다 활성값의 크기를 다시 맞추므로 신호가 깊이에 따라 기하급수로 줄어들 여지가 없다.

    표가 실제로 보이는 것은 **깊이에 따른 기울기의 기울어짐**이다. 평범한 쪽은 블록 1에서 블록 8로 가며 5.9\~12.3배 줄지만, 잔차 쪽은 2.0\~2.9배밖에 줄지 않는다. 씨앗 셋의 폭이 두 쪽 사이에서 겹치지 않으므로 이 차이는 씨앗 잡음보다 크다. 즉 잔차 연결이 하는 일은 기울기를 **살려 내는** 것이 아니라 **층마다 고르게 나누는** 것이다.

    [기본 잔차 블록](01_basic_residual_block.md)의 표는 같은 이야기를 반대쪽에서 보인다. 깊이 8, 16에서 평범한 층더미의 입력 기울기는 사라지기는커녕 잔차 쪽보다 훨씬 **크다**(16층에서 26,672\~28,533 대 1,680\~1,764). 배치 정규화가 있는 층더미의 문제는 소실이 아니라 척도가 깊이에 따라 제멋대로 움직이는 것이다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span>
잔차 블록의 야코비가

$$\frac{\partial x_{l+1}}{\partial x_l} = D_l\left(G_l'(x_l) + I\right)$$

임을 연쇄 법칙으로 유도하라. 그리고 이 식이 "기울기의 크기가 1 이상"을 뜻하지 **않는다**는 것을 반례 둘로 보여라.

</div>

??? success "연습문제 5 풀이"
    $u_l = G_l(x_l) + x_l$ 이라 두면 $x_{l+1} = \sigma(u_l)$ 이다. 연쇄 법칙으로

    $$\frac{\partial x_{l+1}}{\partial x_l} = \frac{\partial \sigma(u_l)}{\partial u_l} \cdot \frac{\partial u_l}{\partial x_l}$$

    이다. 앞의 인수는 ReLU의 미분이라 $u_l$ 의 성분이 양수인 자리에 1, 아닌 자리에 0을 놓은 대각행렬 $D_l$ 이다. 뒤의 인수는 $u_l = G_l(x_l) + x_l$ 을 그대로 미분한 $G_l'(x_l) + I$ 다. 곱하면 구하는 식이 된다. 평범한 블록에서는 $u_l = G_l(x_l)$ 이라 $I$ 가 없고 $D_l G_l'(x_l)$ 만 남는다.

    반례 하나. $u_l$ 의 어떤 성분이 음수이면 $D_l$ 의 그 대각 성분이 0이므로 그 줄이 통째로 0이 된다. $I$ 가 괄호 안에 있든 없든 ReLU 바깥의 문은 닫힌다.

    반례 둘. $G_l'(x_l) = -I$ 이면 $G_l'(x_l) + I = 0$ 이다. 잔차 가지가 항등 가지를 정확히 지워 버리는 경우로, $D_l$ 이 모두 1이어도 야코비가 0이다. 더 약하게 $G_l'(x_l) = -0.9 I$ 이면 야코비가 $0.1 D_l$ 이 되어 블록마다 10분의 1로 줄어든다.

    그러니 올바른 읽기는 "기울기가 1 이상"이 아니라 "기울기가 **곱이 아니라 합**으로 분해된다"이다. 합 안에 $G'$ 을 거치지 않는 항이 하나 들어 있다는 것이 이득의 전부이고, 그 항조차 $D_l$ 을 지나야 한다. [항등 사상](identity_mapping.md)이 이 구분을 더 자세히 다룬다.

---

## 정리하며

**다룬 것** — 잔차 연결 하나만 다른 두 층더미의 학습 비교

이 쪽은 견주기를 망가뜨리는 것부터 막았다. 층더미를 클래스 하나로 만들고 `residual` 참/거짓만 바꾸어, 두 층더미가 같은 씨앗에서 **글자 그대로 같은 가중치**(`True`)와 같은 매개변수 수(38,010개)를 갖게 했다. 그래서 뒤에 나온 모든 차이는 블록 안 `out + x` 한 항의 몫이다.

그렇게 재었을 때 남는 것은 둘이다. 첫째, 학습을 시작하기도 전에 기울기의 **깊이별 옆모습**이 갈린다 — 블록 1 대 블록 8의 비가 평범한 쪽 5.945\~12.302, 잔차 쪽 2.022\~2.934로 씨앗 폭을 넘어 겹치지 않는다. 둘째, 다섯 세대 뒤 검증 정확도가 19.21% 대 35.68%로, 평균의 차이 16.47점이 가장 넓은 씨앗 폭 7.23점보다 크고 두 범위가 겹치지 않는다.

같은 정도로 중요한 것은 이 쪽이 **보이지 못한** 것들이다. 층더미 전체의 기울기 노름은 두 범위가 겹쳐 아무것도 가르지 못하고, 평범한 층더미에서 기울기가 사라지지도 않았다(블록 8에서 0.1571\~0.2291). 배치 정규화가 있는 층더미에서 잔차 연결이 고치는 것은 소실이 아니라 기울어짐이다.

이 쪽이 다루는 것은 [기본 잔차 블록](01_basic_residual_block.md)이 블록 하나에서 보인 것을 층더미로 늘린 것이고, 지름길을 손대지 않은 항등으로 두는 문제는 [항등 사상](identity_mapping.md)이, 단계를 나누어 진짜 ResNet을 쌓는 일은 [ResNet 구현](02_resnet_implementation.md)이 이어받는다.
