# 학습 비교

일반 블록과 잔차 블록을 같은 ConvStack 클래스로 구성해 CIFAR-10 일부 데이터에서 비교한다. 두 모델은 같은 초기 상태와 파라메터 수를 갖도록 만들고, 초기 파라메터 그래디언트와 짧은 학습 후 평가값을 기록한다.

이 설계는 구조 차이를 줄이지만 모든 조건이 완전히 통제된 것은 아니다. 코드가 실제로 측정한 그래디언트의 대상, 배치 순서와 시드 범위를 확인해 결과를 해석한다.

## 1. 두 네트워크의 유일한 차이

블록 하나가 하는 일을 $G_l$ 이라 쓰자 — 컨볼루션, 배치 정규화, ReLU, 컨볼루션, 배치 정규화까지다. $\sigma$ 를 ReLU라 하면

$$G_l(x) = \mathrm{BN}_2\!\left(W_2\,\sigma\!\left(\mathrm{BN}_1(W_1 x)\right)\right)$$

이다. 두 네트워크는 이 $G_l$ 을 똑같이 쓰고 마지막 한 스텝만 다르다.

$$\begin{aligned}
\text{평범한 블록:}\quad & x_{l+1} = \sigma\!\left(G_l(x_l)\right) \\
\text{잔차 블록:}\quad & x_{l+1} = \sigma\!\left(G_l(x_l) + x_l\right)
\end{aligned}$$

야코비안를 적으면 차이가 어디서 오는지 보인다. ReLU가 살린 자리에 1, 죽인 자리에 0을 놓은 대각행렬을 $D_l$ 이라 하면

$$\begin{aligned}
\text{평범한 블록:}\quad & \frac{\partial x_{l+1}}{\partial x_l} = D_l\, G_l'(x_l) \\
\text{잔차 블록:}\quad & \frac{\partial x_{l+1}}{\partial x_l} = D_l\left(G_l'(x_l) + I\right)
\end{aligned}$$

여러 블록의 야코비안은 출력 쪽부터 순서대로 행렬 곱을 이룬다. 잔차 모델의 곱을 전개하면 $G_l\prime$을 거치지 않는 경로에 대응하는 항도 생긴다. 그러나 두 모델의 중간 입력이 달라져 $D_l$과 $G_l\prime$도 달라질 수 있다. 항등항 하나의 존재만으로 전체 그래디언트나 성능 차이를 예측할 수는 없다.

!!! warning "덧셈 분해는 하한이 아니다"
    괄호 안에 $I$ 가 들어 있다고 해서 기울기가 1 이상이 되지는 않는다. $D_l$ 이 ReLU가 죽인 자리를 0으로 만들고, $G_l'(x_l) = -I$ 이면 괄호 안이 통째로 0이다. [항등 사상](identity_mapping.md)이 같은 함정을 반례와 함께 다룬다.

## 2. 코드

```python
"""
학습 비교: 잔차 레이어더미와 평범한 레이어더미
======================================
같은 씨앗에서 웨이트가 글자 그대로 같은 두 레이어더미를 CIFAR-10 조각으로
학습시킨다. 두 레이어더미는 블록 안에서 `out + x` 한 항만 다르므로, 드러나는
차이는 모두 그 항의 몫이다. 씨앗 셋을 돌려 씨앗 사이의 폭도 함께 찍는다.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

DEPTH = 8          # 블록 수 (블록마다 컨볼루션 2개)
WIDTH = 16         # 채널 수
N_TRAIN = 1280     # CIFAR-10 학습 조각
N_VAL = 512        # CIFAR-10 검증 조각
BATCH = 64
EPOCHS = 5
SEEDS = (0, 1, 2)

# ========================================================================
# 레이어더미 — `residual` 한 글자만 다르다
# ========================================================================


class ConvStack(nn.Module):
    """블록 DEPTH개를 쌓은 레이어더미.

    블록 하나는 conv-BN-ReLU-conv-BN 이고, `residual` 이 참이면 그 뒤에
    입력을 더한 다음 ReLU를 건다. 거짓이면 더하지 않고 바로 ReLU를 건다.
    레이어의 종류와 만드는 차례가 두 경우에 완전히 같으므로, 같은 씨앗을 심으면
    두 레이어더미의 웨이트는 글자 그대로 같아진다.

    컨볼루션의 `bias=False`는 바로 뒤의 배치 정규화가 치우침을 지우기 때문이다.
    """

    def __init__(self, residual, depth=DEPTH, width=WIDTH, num_classes=10):
        super().__init__()
        self.residual = residual
        # stride 2 컨볼루션과 최대 풀링으로 32x32 를 8x8 로 줄인다. ResNet 의
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
                out = out + x          # 이 한 항이 두 레이어더미의 유일한 차이다
            x = torch.relu(out)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def build(residual, seed):
    """씨앗을 먼저 심고 레이어더미를 만든다 — 같은 씨앗이면 웨이트가 같다."""
    torch.manual_seed(seed)
    return ConvStack(residual)


# ========================================================================
# 자료 — CIFAR-10의 고정된 조각
# ========================================================================


def load_subsets():
    """학습 1280장, 검증 512장. 뽑는 자리는 씨앗 0으로 못 박아 모든 씨앗이
    똑같은 자료를 본다. 달라지는 것은 웨이트의 초기값과 섞는 차례뿐이다."""
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
# 그래디언트 재기
# ========================================================================


def grad_profile(model, x, y):
    """고정된 배치 하나로 블록별·전체 파라메터 그래디언트의 노름을 잰다.

    배치 정규화를 학습 모드로 두어야 레이어더미가 실제로 쓰는 정규화가 걸린다.
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

    로스는 표본 수로 나눈다. 배치 수로 나누면 마지막 배치가 짧을 때
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

    # --- 두 레이어더미가 정말 같은 웨이트에서 출발하는지 ---
    plain0, res0 = build(False, 0), build(True, 0)
    same = all(torch.equal(a, b) for a, b in zip(plain0.state_dict().values(),
                                                 res0.state_dict().values()))
    print(f"\nSame weights under the same seed: {same}")
    print(f"Parameters: plain {sum(p.numel() for p in plain0.parameters()):,}, "
          f"residual {sum(p.numel() for p in res0.parameters()):,}")

    # --- 초기 그래디언트의 깊이별 옆모습 (웨이트가 같으므로 차이는 `+ x` 뿐이다) ---
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

### 3.1 같은 웨이트에서 출발했는가

```
Same weights under the same seed: True
Parameters: plain 38,010, residual 38,010
```

이 두 줄이 이 예제의 나머지를 떠받친다. 첫 줄은 두 네트워크의 `state_dict` 를 항목마다 `torch.equal` 로 견준 결과다 — 웨이트뿐 아니라 배치 정규화의 $\gamma$, $\beta$, 이동 평균과 분산까지 모두 같다. 둘째 줄의 38,010은 `out + x` 가 배울 것이 없는 덧셈이라 파라메터를 하나도 쓰지 않기 때문이다.

웨이트와 버퍼의 일치를 확인하면 초기 상태 차이라는 요인을 줄일 수 있다. 그러나 학습 루프에서 시드별 DataLoader와 generator를 한 번 만들고 두 모델을 순서대로 학습하므로, 두 번째 모델은 첫 번째 학습이 진행시킨 generator 상태를 이어 쓴다. 같은 배치 순서를 보장하려면 모델별로 같은 시드의 generator를 새로 만들거나 상태를 복원해야 한다.

### 3.2 초기 그래디언트의 깊이별 옆모습

grad_profile은 각 블록 conv1.weight의 그래디언트 노름과 전체 파라메터 그래디언트 노름을 측정한다. 블록 입력·출력 엑티베이션에 대한 그래디언트 노름이 아니므로 레이어간 역전파 감쇠율로 직접 해석할 수 없다. 같은 초기 웨이트에서도 잔차 더하기가 중간 피처와 로스를 바꾸므로 그에 따른 전체 계산의 차이가 관찰된다.

| | 평범 | 잔차 |
|---|---|---|
| 블록 1 | 1.3619 ~ 1.9331 | 0.8807 ~ 1.0686 |
| 블록 8 | 0.1571 ~ 0.2291 | 0.3642 ~ 0.4355 |
| 블록 1 ÷ 블록 8 | 5.945 ~ 12.302 | 2.022 ~ 2.934 |
| 네트워크 전체 | 2.7300 ~ 3.5850 | 3.1136 ~ 4.1751 |

셋째 줄은 첫 블록과 마지막 블록의 선택한 웨이트 그래디언트 크기 비이다. 이 세 초기화에서는 잔차 모델의 비가 더 작다. 이를 그래디언트 전파가 6~12배 또는 2~3배 감쇠했다는 의미로 사용하지 않는다. 서로 다른 파라메터의 스케일과 해당 배치에서의 로스 민감도도 반영된다.

넷째 줄의 전체 파라메터 그래디언트 노름은 이 관측에서 범위가 겹친다. 이것은 세 시드의 관찰 범위이며 두 분포가 같다는 증명은 아니다. 선택한 그래디언트 통계 하나가 학습 성능 차이의 원인을 충분히 설명한다고 가정하지 않는다.

### 3.3 학습 곡선과 시드 사이의 폭

```
plain    train acc 24.77-30.62  val acc 16.60-22.85 (mean 19.21)  grad 2.900-3.594
residual train acc 44.45-47.89  val acc 32.03-39.26 (mean 35.68)  grad 3.103-5.613

val acc: gap of means +16.47 points, widest seed spread 7.23 points
verdict: the gap is larger than the spread
```

각 실행은 하나의 관찰이고 반복 실행은 그 변동을 파악하는 데 도움이 된다. 코드의 세 시드에서 평가 정확도 평균 차이는 16.47%포인트이다. 최대–최소 범위와의 비교는 기술적인 요약이며 차이가 범위보다 크거나 작다는 것만으로 통계적 유의성을 판정하는 일반 규칙은 아니다. 출력의 verdict 문구도 이 단순 규칙에 따른 것이다.

학습 정확도도 잔차 설정이 높게 기록되지만 두 모델의 학습 배치 순서는 일치하지 않는다. 고정된 데이터 부분집합·초기 상태를 사용한 짧은 학습의 결과로 읽고, 잔차 연결만의 인과 효과를 더 엄밀히 분리하려면 데이터 순서까지 맞춘 비교가 필요하다. val_set은 공식 테스트셋에서 선택한 512장이므로 이를 반복 선택에 쓰면 별도 최종 테스트가 필요하다.

한편 **그래디언트의 총 노름은 이 차이를 설명하지 못한다.** 시드 2의 첫 에포크에서는 평범한 쪽이 2.306, 잔차 쪽이 2.266으로 평범한 쪽이 오히려 크다. 다섯 에포크째에는 세 시드 모두 잔차 쪽이 크지만, "잔차 뉴럴 넷트웍은 학습 내내 더 큰 그래디언트 노름을 유지한다"는 식의 말은 이 표가 받쳐 주지 않는다. 받쳐 주는 것은 3.2절의 깊이별 옆모습이다.

### 3.4 이 예제가 뒷받침하지 못하는 것

**그래디언트 전파의 직접 측정은 아니다.** 블록별 숫자는 웨이트 그래디언트의 노름이다. 이 값만으로 엑티베이션 그래디언트의 소실·폭발 여부를 판정할 수 없다. 그래디언트 소실에 특정한 절대 임계값이 있는 것도 아니며 BatchNorm이 이를 원천적으로 막는 것도 아니다. 다른 페이지의 입력 그래디언트 실험은 측정 대상과 조건이 달라 별도로 해석해야 한다.

**깊이와 실험 범위**: 이 예제의 depth=8은 잔차 블록 8개를 뜻하며 스템까지 컨볼루션 17개이다. 원래 ResNet과 사전 활성화 논문은 다양한 깊이와 다른 데이터·학습 예산을 사용했다. 여기의 작은 부분집합·5에포크 결과를 깊은 모델 전반의 성능 차이로 옮길 수 없다.

**평가 범위**: 평균 35.68%는 이 부분집합·학습 예산의 관찰이다. 모델의 일반적인 CIFAR-10 성능 상한이나 잔차 구조 전체의 대표값이 아니다. 설정 선택과 최종 성능 보고를 위한 분할을 구분한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
출력이 찍은 파라메터 38,010개를 손으로 검산하라. 머리(`stem`, `stem_bn`), 블록 하나, 마지막 선형 레이어로 나누어 세고, 두 네트워크의 개수가 같은 까닭도 한 줄로 적어라.

</div>

??? success "풀이"
    스템의 컨볼루션은 `bias=False` 이므로 $3 \times 16 \times 3 \times 3 = 432$개이고, 배치 정규화는 $\gamma$ 와 $\beta$ 를 채널마다 하나씩 가지므로 $2 \times 16 = 32$개다. `MaxPool2d` 는 파라메터가 없다.

    블록 하나는 `bias=False` 인 $16 \to 16$ 컨볼루션 둘과 배치 정규화 둘이다.

    $$2 \times (16 \times 16 \times 3 \times 3) + 2 \times (2 \times 16) = 4608 + 64 = 4672$$

    블록이 여덟이므로 $8 \times 4672 = 37{,}376$개다. 마지막 선형 레이어는 $16 \times 10 + 10 = 170$개다. 모두 더하면

    $$432 + 32 + 37{,}376 + 170 = 38{,}010$$

    합계는 38010개이다. 동일 차원에서 입력을 더하는 연산은 학습 파라메터를 추가하지 않지만 덧셈과 중간 텐서 유지에는 계산·메모리 비용이 있을 수 있다. 투영 지름길은 추가 파라메터도 사용한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
$32 \times 32$ 입력이 블록에 닿을 때까지의 공간 크기를 따라가라. 그리고 블록 안의 `out` 과 `x` 가 왜 언제나 모양이 맞는지 설명하라.

</div>

??? success "풀이"
    컨볼루션의 출력 크기는 $\left\lfloor (H + 2p - k)/s \right\rfloor + 1$ 이다.

    - 스템의 컨볼루션($k = 3$, $p = 1$, $s = 2$): $\lfloor (32 + 2 - 3)/2 \rfloor + 1 = 15 + 1 = 16$
    - `MaxPool2d(2, 2)`($k = 2$, $p = 0$, $s = 2$): $\lfloor (16 - 2)/2 \rfloor + 1 = 7 + 1 = 8$

    그래서 블록들은 $8 \times 8$ 에서 돌아간다. 블록 안의 컨볼루션은 $k = 3$, $p = 1$, $s = 1$ 이므로 $\lfloor (8 + 2 - 3)/1 \rfloor + 1 = 8$ 로 크기를 그대로 둔다. 채널도 $16 \to 16$ 으로 그대로다. 따라서 `out` 과 `x` 는 둘 다 $(B, 16, 8, 8)$ 이고 덧셈이 언제나 맞는다.

    크기나 채널이 바뀌는 자리에서는 이 말이 깨진다. 그때 지름길에 1×1 투영을 넣는 까닭을 [기본 잔차 블록](01_basic_residual_block.md)이 다룬다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
`build` 를 쓰지 않고 시드를 **한 번만** 심은 뒤 두 네트워크를 잇달아 만들어 보라.

```python
torch.manual_seed(0)
plain = ConvStack(False)
residual = ConvStack(True)
```

`Same weights under the same seed` 가 무엇으로 바뀌는가? 왜 그런가? 그리고 그렇게 견주었을 때 마지막 정확도의 차이는 무엇을 재는 것인가?

</div>

??? success "풀이"
    두 모델을 시드 고정 후 차례로 생성하면 난수 상태가 진행되어 초기 웨이트가 달라진다. 다만 38010개 파라메터 전부가 난수로 초기화되는 것은 아니다. BatchNorm의 스케일·바이어스와 같은 상수 초기화도 있으며 난수 소비량을 파라메터 수와 동일시하지 않는다.

    그렇게 견주면 마지막 정확도의 차이는 "잔차 연결의 효과"가 아니라 "잔차 연결의 효과 + 서로 다른 초기값의 효과"다. 뒤엣것이 얼마나 큰지는 이 예제의 출력이 직접 말해 준다 — 같은 구조라도 시드만 바꾸면 검증 정확도가 몇 점씩 움직인다. 초기값 차이를 섞어 넣으면 그만큼의 잡음을 결론으로 읽게 된다.

    고치는 방법이 `build(residual, seed)` 다. 시드를 **레이어를 만들기 직전에** 심고, 두 경우가 같은 종류의 레이어를 같은 차례로 만들게 두면 웨이트가 글자 그대로 같아진다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
블록별 그래디언트 표의 측정 대상과 이 표만으로 확인할 수 없는 것을 설명하라.

</div>

??? success "풀이"
    표는 각 블록 conv1.weight의 그래디언트 노름과 전체 파라메터 그래디언트 노름이다. 선택된 레이어들의 로스 민감도 비교에는 사용할 수 있지만 엑티베이션 그래디언트가 블록 사이에서 얼마나 감쇠했는지를 직접 나타내지 않는다. 그래디언트 소실·폭발을 판단하려면 측정 대상과 스케일을 정하고 깊이·상류 그래디언트·초기화와 입력 조건을 함께 비교해야 한다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span>
잔차 블록의 야코비안가

$$\frac{\partial x_{l+1}}{\partial x_l} = D_l\left(G_l'(x_l) + I\right)$$

임을 연쇄 법칙으로 유도하라. 그리고 이 식이 "그래디언트의 크기가 1 이상"을 뜻하지 **않는다**는 것을 반례 둘로 보여라.

</div>

??? success "풀이"
    $u_l = G_l(x_l) + x_l$ 이라 두면 $x_{l+1} = \sigma(u_l)$ 이다. 연쇄 법칙으로

    $$\frac{\partial x_{l+1}}{\partial x_l} = \frac{\partial \sigma(u_l)}{\partial u_l} \cdot \frac{\partial u_l}{\partial x_l}$$

    이다. 앞의 인수는 ReLU의 미분이라 $u_l$ 의 성분이 양수인 자리에 1, 아닌 자리에 0을 놓은 대각행렬 $D_l$ 이다. 뒤의 인수는 $u_l = G_l(x_l) + x_l$ 을 그대로 미분한 $G_l'(x_l) + I$ 다. 곱하면 구하는 식이 된다. 평범한 블록에서는 $u_l = G_l(x_l)$ 이라 $I$ 가 없고 $D_l G_l'(x_l)$ 만 남는다.

    반례 하나. $u_l$ 의 어떤 성분이 음수이면 $D_l$ 의 그 대각 성분이 0이므로 그 줄이 통째로 0이 된다. $I$ 가 괄호 안에 있든 없든 ReLU 바깥의 문은 닫힌다.

    반례 둘. $G_l'(x_l) = -I$ 이면 $G_l'(x_l) + I = 0$ 이다. 잔차 가지가 항등 가지를 정확히 지워 버리는 경우로, $D_l$ 이 모두 1이어도 야코비안가 0이다. 더 약하게 $G_l'(x_l) = -0.9 I$ 이면 야코비안가 $0.1 D_l$ 이 되어 블록마다 10분의 1로 줄어든다.

    전체 야코비안은 블록별 행렬 곱이다. 잔차의 항등항으로 그 곱을 경로별 합으로 전개할 수 있지만, 그 자체가 노름의 하한이나 성능 개선 보장은 아니다. 직접 경로도 사후 활성화의 ReLU 마스크를 통과한다. 벡터의 특정 방향 소실과 전체 행렬이 0인 경우도 구분한다.

---

## 정리하며

두 모델의 초기 웨이트와 파라메터 수는 맞지만 배치 순서는 같은 generator 상태로 재설정하지 않았다. 블록별 그래디언트 표는 선택된 웨이트의 그래디언트이며 엑티베이션 간 전파율과 다르다. 세 시드에서의 정확도와 그래디언트 범위를 기술 통계로 보고하고, 구조만의 효과와 최종 일반화 성능을 확인하려면 추가 통제와 독립 평가가 필요하다.
