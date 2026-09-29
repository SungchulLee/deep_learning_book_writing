# 항등 사상

He 등이 "Identity Mappings in Deep Residual Networks"(2016)에서 소개한 심층 잔차 신경망의 항등 사상은 원래 ResNet 설계를 크게 다듬은 것이다. 잔차 블록 안의 부품 순서를 바꾸어 배치 정규화와 ReLU를 합성곱 뒤가 아니라 *앞*에 두면 건너뛰기 연결이 참된 항등 사상이 되어, 기울기가 더 깨끗하게 흐르고 1000층이 넘는 신경망도 학습할 수 있게 된다.

이 절은 왜 순수한 항등 지름길이 최적인지, 사전 활성화 블록 설계가 그것을 어떻게 이루는지, 그리고 기울기 전파에 어떤 수학적 결과가 따르는지를 살펴본다. 주장을 말로만 두지 않고, 3절에서 50블록짜리 신경망의 기울기 크기를 직접 재어 어디까지가 측정되는 사실이고 어디부터가 원리상의 이야기인지를 갈라 둔다.

---

## 1. 왜 항등 사상이 중요한가

원래 ResNet 논문은 다음 식으로 잔차 함수를 배우자고 제안했다.

$$y_l = h(x_l) + F(x_l, W_l)$$

$$x_{l+1} = f(y_l)$$

여기서 $h(x_l)$은 건너뛰기 연결, $F$은 잔차 함수, $f$은 더하기 뒤의 활성화 함수(ReLU)이다.

### 건너뛰기 연결의 변형 실험

He 등은 $h(x_l)$의 여러 형태를 CIFAR-10에서 110층 ResNet으로 체계적으로 시험했다. 아래 수치는 논문 표 1의 시험 오차이다.

| 지름길의 종류 | $h(x_l)$ | 시험 오차 (ResNet-110) |
|---------------|----------|----------------|
| 항등 (원래) | $x_l$ | **6.61%** |
| 상수 배율 ($\lambda = 0.5$) | $0.5 \cdot x_l$ | 12.35% |
| 배타적 문 달기 | $(1 - g(x_l)) \cdot x_l$, 잔차 가지는 $g(x_l)$배 | 8.70% |
| 지름길에만 문 달기 | $(1 - g(x_l)) \cdot x_l$, 잔차 가지는 그대로 | 6.91% |
| 1×1 합성곱 | $W \cdot x_l$ | 12.22% |
| 드롭아웃 (0.5) | $\text{dropout}(x_l)$ | 수렴 실패 |

**핵심 발견**: 순수한 항등 사상이 가장 잘 통한다. 학습된 문 달기처럼 이로워 보이는 것을 포함하여, 건너뛰기 연결에 무엇을 손대든 성능이 나빠진다. 손댄 정도와 나빠진 정도가 함께 간다는 것도 눈여겨볼 만하다 — 문을 거의 열어 둔 채 학습할 수 있는 "지름길에만 문 달기"는 6.91%로 항등에 가깝지만, 지름길을 고정된 0.5배로 줄이면 12.35%로 두 배 가까이 나빠진다.

### 수학적 설명

건너뛰기 연결이 항등이고 **더하기 뒤의 $f$ 또한 항등이면** 순전파가 깔끔하게 펼쳐진다. 두 조건이 함께 필요하다는 점이 중요하다. $h$만 항등으로 두고 $f = \text{ReLU}$를 그대로 두면 아래의 펼침은 성립하지 않는다.

$$x_L = x_l + \sum_{i=l}^{L-1} F_i(x_i)$$

어떤 깊은 층도 얕은 층에 누적된 잔차를 더한 것으로 곧바로 나타난다. 항등이 아닌 지름길 $h(x_l) = \lambda x_l$을 쓰면(여전히 $f$는 항등으로 두고) 펼침이 다음과 같이 된다.

$$x_L = \lambda^{L-l} x_l + \sum_{i=l}^{L-1} \lambda^{L-1-i} F_i(x_i)$$

지수 인수 $\lambda^{L-l}$이 신호를 키우거나($\lambda > 1$) 줄여($\lambda < 1$), 건너뛰기 연결이 풀려던 바로 그 기울기 흐름 문제를 되살린다. $\lambda = 0.5$이고 $L - l = 50$이면 이 인수는 $0.5^{50} \approx 8.9 \times 10^{-16}$이다. 3절에서 실제로 재어 보면 이 자릿수가 그대로 나온다.

---

## 2. 사전 활성화라는 착상

1절이 요구하는 두 조건 가운데 $h = \text{항등}$은 원래 ResNet도 이미 만족한다. 문제는 $f$이다. 사전 활성화는 $f$를 없애는 — 더 정확히는 $f$가 하던 일을 잔차 가지 안으로 옮기는 — 재배치다.

### 원래 ResNet 블록 (사후 활성화)

```text
입력 ──────┬─────────────────────────────────────┐
           │                                      │
           ▼                                      │
      [Conv 3×3]                                  │
           ▼                                      │
      [BatchNorm]                                 │
           ▼                                      │
        [ReLU]                                    │ (항등 또는 사영)
           ▼                                      │
      [Conv 3×3]                                  │
           ▼                                      │
      [BatchNorm]                                 │
           ▼                                      │
         (+)  ◄───────────────────────────────────┘
           ▼
        [ReLU]  ◄─── 이 ReLU가 다음 블록의 항등 경로까지 건드린다
           ▼
         출력
```

**문제**: 더하기 뒤에 ReLU가 있으므로 건너뛰기 연결을 지나 다음 블록으로 가는 신호*까지* ReLU를 거치게 되어 순수한 항등 사상이 깨진다. 출력 $x_{l+1} = \text{ReLU}(x_l + F(x_l))$은 언제나 음이 아니므로 항등 경로가 $\mathbb{R}_{\geq 0}$으로 제약된다.

### 사전 활성화 ResNet 블록

```text
입력 ──────┬─────────────────────────────────────┐
           │                                      │
           ▼                                      │
      [BatchNorm]                                 │
           ▼                                      │
        [ReLU]                                    │
           ▼                                      │
      [Conv 3×3]                                  │
           ▼                                      │
      [BatchNorm]                                 │
           ▼                                      │
        [ReLU]                                    │ (순수한 항등)
           ▼                                      │
      [Conv 3×3]                                  │
           ▼                                      │
         (+)  ◄───────────────────────────────────┘
           ▼
     출력 (다음 블록의 입력으로 그대로 이어진다)
```

**해결**: 배치 정규화와 ReLU를 합성곱 앞으로 옮기면 $f$가 항등이 되고, 따라서 건너뛰기 연결도 참된 항등 사상이 된다. 신호가 잇따른 블록을 손대지 않은 채 흐르고 출력은 다음과 같이 간단해진다.

$$x_{l+1} = x_l + F(\hat{f}(x_l), W_l)$$

여기서 $\hat{f}$은 잔차 가지 안에서만 적용되는 사전 활성화(배치 정규화 뒤 ReLU)를 뜻한다. 지름길에 더해지는 것은 $\hat{f}(x_l)$이 아니라 손대지 않은 $x_l$이다 — 이 구별이 4절 구현에서 한 줄 차이로 갈리며, 잘못 쓰면 이 절의 모든 식이 무너진다.

---

## 3. 수학적 분석

### 정보의 전파

사전 활성화를 쓰면 순전파가 다음과 같이 된다.

$$x_{l+1} = x_l + F(\hat{f}(x_l), W_l)$$

이 점화식을 펼치면 다음과 같다.

$$x_L = x_l + \sum_{i=l}^{L-1} F(\hat{f}(x_i), W_i)$$

이는 어떤 깊은 층 $x_L$도 얕은 층 $x_l$과 중간의 모든 잔차 함수의 합임을 보인다. 곱해지는 인수가 전혀 없고 관계가 순전히 덧셈적이다.

### 기울기의 흐름

위 식을 $x_l$로 미분하면 기울기도 덧셈으로 갈라진다.

$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \left(1 + \frac{\partial}{\partial x_l}\sum_{i=l}^{L-1}F_i\right)$$

"1" 항이 $\mathcal{L}$에서 $x_l$까지 어떤 비선형에도 막히지 않는 곧바른 기울기 경로를 준다. 이것이 $\lambda$ 지름길과 갈리는 자리다. $\lambda$ 지름길에서는 같은 자리에 $\lambda^{L-l}$이 놓이므로 $\lambda < 1$이면 깊이에 따라 지수로 죽는다.

!!! warning "'기울기의 크기가 1 이상'은 아니다"
    이 식이 보장하는 것은 **덧셈 분해**이지 하한이 아니다. $\partial \sum F / \partial x_l$이 정확히 $-1$이 되면 괄호 안은 0이 되어 기울기는 사라진다. He 등이 말한 것은 그런 일이 미니배치의 **모든** 표본에서 동시에 일어날 가능성이 낮다는 것이지, 수학적으로 불가능하다는 것이 아니다. 이 차이는 연습문제 1에서 반례로 다시 다룬다.

### 기울기 경로의 비교

| 항목 | 원래 ResNet (사후 활성화) | 사전 활성화 ResNet |
|------|-----------------|----------------------|
| 항등 경로의 기울기 | 블록마다 더하기 뒤 ReLU의 마스크를 지난다 | 어떤 비선형도 지나지 않는다 |
| 기울기에 붙는 인수 | $\prod_{i=l}^{L-1} \mathbb{1}[y_i > 0]$ 이 끼어든다 | 덧셈항 $1$이 그대로 남는다 |
| 사라질 수 있는가? | 원리상 그렇다 (죽은 ReLU가 연쇄된다) | $1 + \partial \sum F / \partial x_l$ 이 0이 되지 않는 한 아니다 |

이 표의 마지막 줄은 **원리**를 적은 것이지 아무 깊이에서나 관측되는 현상이 아니다. 어느 깊이에서 실제로 갈리는지는 바로 아래에서 재어 본다.

### 수를 재어 확인하기

세 가지를 확인한다. 하나, 펼침 공식 $x_L = x_l + \sum F_i$이 정말 성립하는가. 둘, 기울기가 정말 $\partial \mathcal{L} / \partial x_L$과 잔차항의 **합**으로 갈라지는가. 셋, 50블록을 쌓았을 때 $\lVert \partial \mathcal{L} / \partial x_l \rVert$이 깊이에 따라 어떻게 달라지는가.

셋째 실험에서는 위쪽에서 내려오는 기울기 $\partial \mathcal{L} / \partial x_L$을 세 경우에 똑같이 고정하고 그것으로 나눈 비를 본다. 그렇게 하지 않으면 순전파의 크기 차이(0.5배 지름길은 순전파도 함께 죽는다)가 섞여 들어와 역전파의 감쇠를 따로 볼 수 없다.

```python
"""항등 지름길이 기울기에 무엇을 하는지 직접 재어 본다."""

import torch
import torch.nn as nn
import torch.nn.functional as F

CH, DEPTH = 8, 50


class ToyPreActUnit(nn.Module):
    """사전 활성화 잔차 가지 F. 지름길은 이 단위 밖에서 더한다."""

    def __init__(self, ch: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        return self.conv2(out)


class ToyPostActUnit(nn.Module):
    """원래 ResNet 단위. 더하기 뒤에 ReLU가 온다."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


# === 1. 펼침 공식 x_L = x_l + sum F_i 를 확인한다
def check_unrolling(depth: int = 8, seed: int = 0) -> None:
    torch.manual_seed(seed)
    units = nn.ModuleList([ToyPreActUnit(CH) for _ in range(depth)]).eval()
    torch.manual_seed(123)
    x0 = torch.randn(4, CH, 8, 8)

    x, residual_sum = x0, torch.zeros_like(x0)
    for unit in units:
        fx = unit(x)
        residual_sum = residual_sum + fx
        x = x + fx

    gap = (x - (x0 + residual_sum)).abs().max().item()
    print(f"[1] max |x_L - (x_0 + sum F_i)| = {gap:.2e}   (블록 {depth}개)")


# === 2. 기울기가 덧셈으로 갈라지는지 확인한다
def check_additive_gradient(depth: int = 8, seed: int = 0) -> None:
    torch.manual_seed(seed)
    units = nn.ModuleList([ToyPreActUnit(CH) for _ in range(depth)]).eval()
    torch.manual_seed(123)
    x0 = torch.randn(4, CH, 8, 8, requires_grad=True)

    x = x0
    for unit in units:
        x = x + unit(x)
    x_L = x

    loss = 0.5 * (x_L ** 2).sum()
    g_L = torch.autograd.grad(loss, x_L, retain_graph=True)[0]
    g_0 = torch.autograd.grad(x_L, x0, grad_outputs=g_L, retain_graph=True)[0]
    # x_L - x_0 이 곧 sum F_i 이므로, 그것만 따로 미분하면 덧셈항이 나온다.
    g_F = torch.autograd.grad(x_L - x0, x0, grad_outputs=g_L, retain_graph=True)[0]

    print(f"[2] ||dL/dx_L||                 = {g_L.norm().item():.4f}")
    print(f"    ||dL/dx_L * d(sum F)/dx_0|| = {g_F.norm().item():.4f}")
    print(f"    ||dL/dx_0||                 = {g_0.norm().item():.4f}")
    print(f"    max |dL/dx_0 - (dL/dx_L + dL/dx_L * d(sum F)/dx_0)| "
          f"= {(g_0 - (g_L + g_F)).abs().max().item():.2e}")


# === 3. 깊이에 따른 기울기 크기: 항등 지름길 · 배율 지름길 · 사후 활성화
def grad_profile(kind: str, lam: float = 1.0, seed: int = 0) -> list:
    torch.manual_seed(seed)
    unit_cls = ToyPostActUnit if kind == "post" else ToyPreActUnit
    units = nn.ModuleList([unit_cls(CH) for _ in range(DEPTH)]).eval()

    torch.manual_seed(123)
    x = torch.randn(4, CH, 8, 8, requires_grad=True)
    xs, h = [x], x
    for unit in units:
        h = unit(h) if kind == "post" else lam * h + unit(h)
        h.retain_grad()
        xs.append(h)

    # 위쪽에서 내려오는 기울기를 세 경우에 똑같이 고정한다.
    # 그래야 견주는 것이 순전파의 크기가 아니라 역전파의 감쇠가 된다.
    torch.manual_seed(7)
    (xs[-1] * torch.randn_like(xs[-1])).sum().backward()

    base = xs[-1].grad.norm().item()
    return [t.grad.norm().item() / base for t in xs]


if __name__ == "__main__":
    check_unrolling()
    check_additive_gradient()

    cases = (("pre-act, identity ", "pre", 1.0),
             ("pre-act, 0.5x     ", "pre", 0.5),
             ("post-act (original)", "post", 1.0))

    print(f"\n[3] ||dL/dx_l|| / ||dL/dx_L||   (블록 {DEPTH}개)")
    print(f"{'l =':<20}" + "".join(f"{l:>11}" for l in (50, 40, 30, 20, 10, 0)))
    for label, kind, lam in cases:
        p = grad_profile(kind, lam)
        print(label + "".join(f"{p[l]:>11.3e}" for l in (50, 40, 30, 20, 10, 0)))

    print("\n[4] 씨앗 다섯 개에서 본 ||dL/dx_0|| / ||dL/dx_L||")
    for label, kind, lam in cases:
        vals = [grad_profile(kind, lam, seed=s)[0] for s in range(5)]
        print(f"{label} 가운뎃값 {sorted(vals)[2]:.3e}   "
              f"범위 {min(vals):.3e} ~ {max(vals):.3e}")
```

**출력:**

```
[1] max |x_L - (x_0 + sum F_i)| = 4.77e-07   (블록 8개)
[2] ||dL/dx_L||                 = 51.1830
    ||dL/dx_L * d(sum F)/dx_0|| = 33.0093
    ||dL/dx_0||                 = 71.4416
    max |dL/dx_0 - (dL/dx_L + dL/dx_L * d(sum F)/dx_0)| = 4.77e-07

[3] ||dL/dx_l|| / ||dL/dx_L||   (블록 50개)
l =                          50         40         30         20         10          0
pre-act, identity   1.000e+00  1.094e+00  1.228e+00  1.376e+00  1.625e+00  1.831e+00
pre-act, 0.5x       1.000e+00  1.237e-03  1.878e-06  3.038e-09  5.109e-12  7.685e-15
post-act (original)  1.000e+00  8.999e-01  9.384e-01  1.024e+00  1.209e+00  1.052e+00

[4] 씨앗 다섯 개에서 본 ||dL/dx_0|| / ||dL/dx_L||
pre-act, identity  가운뎃값 1.675e+00   범위 1.567e+00 ~ 1.831e+00
pre-act, 0.5x      가운뎃값 5.682e-15   범위 5.092e-15 ~ 7.685e-15
post-act (original) 가운뎃값 1.036e+00   범위 8.786e-01 ~ 1.115e+00
```

읽는 법은 이렇다.

- **[1], [2]는 식이 맞는다는 확인이다.** 펼침 공식의 어긋남 $4.77 \times 10^{-7}$과 덧셈 분해의 어긋남 $4.77 \times 10^{-7}$은 둘 다 float32의 반올림 크기다. [2]의 세 수는 삼각부등식과도 맞는다 — $71.44 \le 51.18 + 33.01 = 84.19$이며, 두 항이 같은 방향을 가리키지 않기 때문에 합보다 작다.
- **[3]의 첫 줄이 이 절의 핵심 주장이다.** 50블록을 거슬러 올라가는 동안 기울기가 줄기는커녕 1.00에서 1.83으로 **늘었다**. 덧셈항 1이 살아 있고, 잔차 가지들이 그 위에 조금씩 얹힌 결과다.
- **[3]의 둘째 줄이 대조군이다.** 지름길을 0.5배로 줄이자 같은 50블록에서 $7.7 \times 10^{-15}$로, 열네 자릿수가 사라졌다. 1절이 예고한 $0.5^{50} \approx 8.9 \times 10^{-16}$이 지배하는 크기다.

!!! warning "셋째 줄은 이 쪽의 주장을 그대로 받쳐 주지 않는다"
    사후 활성화의 비는 50블록에서도 1.05로, 사전 활성화와 구별되지 않는다(씨앗 다섯 개의 범위 0.879\~1.115가 사전 활성화의 1.567\~1.831과 겹치지도 않지만, 둘 다 1 언저리라는 점에서 같다). **50블록 깊이에서 사후 활성화의 기울기는 사라지지 않는다.** 더하기 뒤의 ReLU가 항등 경로를 막는다는 말은 원리로는 옳지만, 이 깊이에서 재면 잡히지 않는 크기다. 두 설계의 차이가 수치로 드러나는 곳은 5절의 1001층이다 — 그리고 거기서도 차이는 "수렴하느냐 마느냐"가 아니라 오차 7.61% 대 4.92%다.

---

## 4. 구현

### 사전 활성화 기본 블록

구현에서 가장 틀리기 쉬운 줄은 지름길을 고르는 한 줄이다. 사전 활성화를 계산한 `out`을 지름길로 넘겨주고 싶은 유혹이 있지만, 그러면 2절이 공들여 세운 항등 경로가 그 자리에서 깨진다. 지름길로 가는 것은 손대지 않은 `x`이고, 사영이 필요할 때만 — 채널 수나 해상도가 바뀔 때만 — 이미 정규화된 `out`에 1×1 합성곱을 걸친다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Type, List


class PreActBasicBlock(nn.Module):
    """
    사전 활성화 기본 블록.

    구조: BN → ReLU → Conv → BN → ReLU → Conv → 더하기

    배치 정규화와 ReLU가 합성곱 뒤에 오는 원래 BasicBlock과 달리
    여기서는 합성곱 앞에 와서 참된 항등 지름길을 만든다.
    """

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ):
        super(PreActBasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # 사전 활성화: 합성곱 앞에 배치 정규화와 ReLU
        self.bn1 = norm_layer(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )

        self.bn2 = norm_layer(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 사전 활성화
        out = self.bn1(x)
        out = F.relu(out)

        # 지름길: 모양이 그대로면 손대지 않은 x 를 쓴다.
        # 여기에 out(= ReLU(BN(x)))을 넣으면 지름길이 항등이 아니게 되어
        # x_L = x_l + sum F 가 그 자리에서 무너진다.
        # 모양이 바뀔 때만 사영을 쓰며, 그 입력은 이미 정규화된 out 이다.
        identity = x if self.downsample is None else self.downsample(out)

        # 첫 합성곱 (사전 활성화된 입력에)
        out = self.conv1(out)

        # 둘째 사전 활성화와 합성곱
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)

        # 건너뛰기 연결 (순수한 덧셈, 뒤에 활성화 없음)
        out = out + identity

        return out
```

### 사전 활성화 병목 블록

```python
class PreActBottleneck(nn.Module):
    """
    사전 활성화 병목 블록.

    구조: BN → ReLU → Conv1×1 → BN → ReLU → Conv3×3 → BN → ReLU → Conv1×1 → 더하기

    확장 배수 = 4 (병목 블록의 표준).
    지름길 규칙은 PreActBasicBlock 과 같다 — 손대지 않은 x, 사영이 필요할 때만 out.
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        width: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ):
        super(PreActBottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # 실제 너비 계산
        actual_width = int(width * (base_width / 64.0)) * groups

        # 1×1 축소를 위한 사전 활성화
        self.bn1 = norm_layer(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, actual_width,
            kernel_size=1, bias=False
        )

        # 3×3 처리를 위한 사전 활성화
        self.bn2 = norm_layer(actual_width)
        self.conv2 = nn.Conv2d(
            actual_width, actual_width,
            kernel_size=3, stride=stride, padding=1,
            groups=groups, bias=False
        )

        # 1×1 확장을 위한 사전 활성화
        self.bn3 = norm_layer(actual_width)
        self.conv3 = nn.Conv2d(
            actual_width, width * self.expansion,
            kernel_size=1, bias=False
        )

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 첫 사전 활성화
        out = self.bn1(x)
        out = F.relu(out)

        # 지름길은 손대지 않은 입력이다 (사영이 필요할 때만 out 을 쓴다)
        identity = x if self.downsample is None else self.downsample(out)

        # 1×1 축소
        out = self.conv1(out)

        # 3×3 처리
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)

        # 1×1 확장
        out = self.bn3(out)
        out = F.relu(out)
        out = self.conv3(out)

        # 덧셈 (뒤에 활성화 없음)
        out = out + identity

        return out
```

### 완전한 사전 활성화 ResNet

```python
class PreActResNet(nn.Module):
    """
    사전 활성화 ResNet.

    원래 ResNet과의 핵심 차이:
    1. 잔차 블록에서 배치 정규화와 ReLU를 합성곱 앞으로 옮겼다
    2. 분류기 앞에 마지막 배치 정규화를 두었다 (마지막 블록에 사후 활성화가 없으므로)
    3. 항등 경로를 지나는 기울기의 흐름이 더 깨끗하다
    """

    def __init__(
        self,
        block: Type[nn.Module],
        layers: List[int],
        num_classes: int = 1000,
        in_channels: int = 3,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ):
        super(PreActResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_planes = 64
        self.groups = groups
        self.base_width = width_per_group

        # 첫 합성곱 (배치 정규화와 ReLU 없음 — 첫 블록에 들어간다)
        self.conv1 = nn.Conv2d(
            in_channels, self.in_planes,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 잔차 단계들
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 마지막 배치 정규화와 활성화 (사전 활성화 방식을 완성한다)
        self.bn_final = norm_layer(512 * block.expansion)
        self.relu_final = nn.ReLU(inplace=True)

        # 분류기
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.in_planes != planes * block.expansion:
            # 참고: 하향 표본화에는 배치 정규화가 없다 (사전 활성화가 정규화를 맡는다)
            downsample = nn.Conv2d(
                self.in_planes, planes * block.expansion,
                kernel_size=1, stride=stride, bias=False
            )

        layers = []
        layers.append(block(
            self.in_planes, planes, stride, downsample,
            norm_layer=norm_layer
        ))

        self.in_planes = planes * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(
                self.in_planes, planes,
                norm_layer=norm_layer
            ))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 풀링 전 마지막 사전 활성화
        x = self.bn_final(x)
        x = self.relu_final(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# 생성 함수들
def preact_resnet18(num_classes: int = 1000) -> PreActResNet:
    return PreActResNet(PreActBasicBlock, [2, 2, 2, 2], num_classes)


def preact_resnet50(num_classes: int = 1000) -> PreActResNet:
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3], num_classes)


def preact_resnet152(num_classes: int = 1000) -> PreActResNet:
    return PreActResNet(PreActBottleneck, [3, 8, 36, 3], num_classes)
```

### 지름길이 정말 항등인지 확인하기

지름길이 항등인지 눈으로 읽어 확인하기는 어렵다. 대신 잔차 가지를 0으로 만들어 두고 블록을 통과시키면 된다. $F = 0$이면 블록은 항등 함수여야 하므로 출력과 입력의 차가 정확히 0이어야 한다. 앞서 말한 한 줄을 잘못 쓴 판본은 여기서 바로 걸린다.

```python
# === 지름길이 정말 항등인가 ===
def shortcut_is_identity() -> None:
    torch.manual_seed(0)
    block = PreActBasicBlock(16, 16).eval()
    nn.init.zeros_(block.conv2.weight)          # F = 0 이 되게 만든다

    x = torch.randn(2, 16, 8, 8)
    out = block(x)
    print(f"F = 0 일 때 max |block(x) - x| = {(out - x).abs().max().item():.2e}")

    wrong = F.relu(block.bn1(x))                # identity = out 으로 잘못 쓴 경우
    print(f"지름길을 ReLU(BN(x))로 잘못 두면   = {(wrong - x).abs().max().item():.2e}")


if __name__ == "__main__":
    shortcut_is_identity()

    for name, factory in (("preact_resnet18", preact_resnet18),
                          ("preact_resnet50", preact_resnet50)):
        torch.manual_seed(0)
        model = factory(num_classes=10).eval()
        n = sum(p.numel() for p in model.parameters())
        y = model(torch.zeros(2, 3, 224, 224))
        print(f"{name:<16} 매개변수 {n:>10,}   출력 {tuple(y.shape)}")
```

**출력:**

```
F = 0 일 때 max |block(x) - x| = 0.00e+00
지름길을 ReLU(BN(x))로 잘못 두면   = 3.33e+00
preact_resnet18  매개변수 11,179,850   출력 (2, 10)
preact_resnet50  매개변수 23,520,842   출력 (2, 10)
```

첫 줄의 `0.00e+00`은 반올림 수준이 아니라 **정확한** 0이다. 지름길이 입력 텐서를 그대로 더하기 때문에 계산 자체가 일어나지 않는다. 둘째 줄이 그 대비다 — 지름길을 잘못 두면 같은 자리에서 3.33만큼 어긋나며, 이것이 블록마다 쌓인다.

---

## 5. 실험 결과

### CIFAR-10에서의 성능 비교

He 등의 논문 표 3에 실린 시험 오차이다.

| 깊이 | 원래 ResNet | 사전 활성화 ResNet |
|-------|-----------------|----------------------|
| 110 | 6.61% | 6.37% |
| 164 | 5.93% | 5.46% |
| 1001 | 7.61% | **4.92%** |

**핵심 관찰**: 1001층에서 원래 ResNet은 *수렴한다* — 다만 7.61%로, 여섯 배 얕은 164층의 5.93%보다 **나쁘다**. 깊이를 여섯 배로 늘려 놓고 오차를 1.68%포인트 잃은 것이다. 사전 활성화는 같은 깊이에서 5.46%에서 4.92%로 계속 좋아진다. 그러니까 사전 활성화가 고치는 것은 "학습이 되느냐"가 아니라 "깊이가 이득으로 바뀌느냐"이다. 보통 깊이(110\~164층)에서 이득이 0.24\~0.47%포인트로 미미한 것도 같은 이유에서다.

### 활성화 순서에 대한 제거 실험

He 등은 배치 정규화와 ReLU를 놓을 수 있는 여러 자리를 시험했다(논문 표 2).

| 순서 | ResNet-110 | ResNet-164 |
|----------|------|------|
| 사후 활성화 (원래) | 6.61% | 5.93% |
| 더하기 뒤에 배치 정규화 | 8.17% | 6.50% |
| 더하기 앞에 ReLU | 7.84% | 6.14% |
| ReLU만 앞으로 | 6.71% | 5.91% |
| 완전한 사전 활성화 (배치 정규화 + ReLU) | **6.37%** | **5.46%** |

여기서 읽어야 할 것은 "둘 다 조금씩 이롭다"가 아니다. ReLU만 앞으로 옮기면 164층에서 5.93%가 5.91%가 되어 사실상 달라지지 않는다. 이득은 배치 정규화까지 함께 옮겨 5.46%가 될 때 비로소 나타난다. 앞의 두 줄은 순서를 어설프게 건드리면 오히려 나빠진다는 것을 보인다 — 더하기 뒤에 배치 정규화를 두면 8.17%로 기준선보다 1.56%포인트 나쁘다.

---

## 6. 비교: 원래 방식과 사전 활성화

| 항목 | 원래 ResNet | 사전 활성화 ResNet |
|--------|-----------------|----------------------|
| 배치 정규화와 ReLU의 위치 | 합성곱 뒤 | 합성곱 앞 |
| 건너뛰기 연결 | 항등에 사후 ReLU | 순수한 항등 |
| 마지막 층의 출력 | 활성화됨 | 마지막에 배치 정규화와 ReLU가 필요 |
| 기울기 경로 | 활성화를 $L$번 지남 | 곧바른 항등 경로 |
| 아주 깊을 때 (1000층 이상) | 수렴은 하나 얕은 판본보다 나쁨 | 깊이가 계속 이득이 됨 |
| 미리 학습된 가중치 | 널리 있음 | 드묾 |

---

## 7. 사전 활성화 ResNet을 쓸 때

### 권장하는 상황

1. **아주 깊은 신경망 (100층 이상)**: 깊이를 늘린 만큼 오차가 줄어들게 하려면 필요하다
2. **초심층 구조 연구**: 500~1000층이 넘는 신경망에서 깊이가 이득으로 남는다
3. **학습의 안정성이 중요할 때**: 학습 동역학이 더 안정적이다
4. **조밀 예측 과제**: 분할을 위한 특징 표현이 더 깨끗하다

### 원래 ResNet으로 충분할 때

1. **보통 깊이 (18~50층)**: 두 판본의 성능이 비슷하다 (110층에서도 차이가 0.24%포인트다)
2. **미리 학습된 가중치를 쓸 때**: 대부분의 사전 학습 모델이 원래 ResNet의 순서를 쓴다
3. **실전 배포**: 프레임워크의 지원이 낫고 사전 학습 검사점이 더 많다

---

## 8. 트랜스포머 구조와의 관계

사전 활성화 설계는 오늘날 주류인 "사전 정규화" 트랜스포머 구조에 곧바로 영향을 주었다. 사전 정규화 트랜스포머에서는 층 정규화를 어텐션과 순방향 부분층 *앞*에 적용하여 똑같이 순수한 항등 건너뛰기 연결을 만든다.

$$x_{l+1} = x_l + \text{Attention}(\text{LN}(x_l))$$

$$x_{l+2} = x_{l+1} + \text{FFN}(\text{LN}(x_{l+1}))$$

3절의 펼침과 글자 그대로 같은 꼴이므로, $x_L = x_l + \sum F_i$과 기울기의 덧셈 분해도 그대로 따라온다. 이 관계는 항등 사상의 원리가 구조를 가리지 않음을 보여 준다 — 잔차 연결이 있는 어떤 깊은 신경망에도 이롭다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
잔차 연결 $y = x + F(x)$에서 야코비 행렬이 $I + \partial F / \partial x$로 갈라짐을 보여라. 그리고 "잔차 연결을 지나는 기울기의 크기는 언제나 1 이상이다"라는 흔한 서술이 왜 틀린지 반례를 들어 보여라.

</div>

??? success "연습문제 1 풀이"
    **덧셈 분해.** $y = x + F(x)$을 $x$로 미분하면 미분의 선형성에서 곧바로

    $$\frac{\partial y}{\partial x} = I + \frac{\partial F}{\partial x}$$

    를 얻는다. 잔차 블록이 $L$개 이어지면 연쇄 법칙으로

    $$\frac{\partial y_L}{\partial x_0} = \prod_{l=1}^{L} \left(I + \frac{\partial F_l}{\partial x_{l-1}}\right)$$

    이고, 이 곱을 펼치면 $2^L$개의 항이 나오는데 그 가운데 하나가 $I$이다. 모든 $\partial F_l / \partial x_{l-1}$이 아무리 작아도 이 $I$ 항은 그대로 남는다. $\lambda$ 지름길이라면 같은 자리에 $\lambda^L I$이 놓여 $\lambda < 1$일 때 지수로 사라진다는 점이 대비된다.

    **반례.** "크기가 1 이상"은 성립하지 않는다. $F$가 선형이고 $F(x) = -x$이면, 즉 가중치 행렬이 $W = -I$이면

    $$\frac{\partial y}{\partial x} = I + (-I) = 0$$

    이 되어 기울기는 정확히 0이다. 한 블록만으로도 반례가 된다. 더 부드러운 반례로 $F(x) = -0.9x$을 $L$개 이으면 야코비는 $(0.1)^L I$이 되어 지수로 사라진다.

    따라서 옳은 서술은 하한이 아니라 **분해**이다. 항등 지름길은 기울기에 $1$이라는 덧셈항을 만들어 주며, 이 항이 0으로 지워지려면 잔차 가지의 야코비가 미니배치의 모든 표본에서 $-I$에 정확히 맞아떨어져야 한다. He 등의 주장은 그런 일이 일어나기 어렵다는 것이지 불가능하다는 것이 아니다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
사전 활성화 ResNet(BN-ReLU-Conv)과 사후 활성화(Conv-BN-ReLU)를 비교하라. 아주 깊은 신경망에는 어느 쪽이 나은가?

</div>

??? success "연습문제 2 풀이"
    아주 깊은 신경망(100층 초과)에는 사전 활성화(He 등, 2016)가 낫다. 사후 활성화에서는 더하기 뒤의 ReLU가 $f$ 자리에 남아 있어 건너뛰기 경로의 신호까지 그것을 지나고, 그 결과 $x_L = x_l + \sum F_i$이라는 깨끗한 펼침이 성립하지 않는다. 사전 활성화는 $f$를 항등으로 만들어 이 펼침과 기울기의 덧셈 분해를 되살린다.

    다만 "그래서 사후 활성화는 기울기가 사라진다"고까지 말하면 지나치다. 3절 [3]의 측정에서 50블록짜리 사후 활성화 신경망의 $\lVert \partial \mathcal{L} / \partial x_0 \rVert / \lVert \partial \mathcal{L} / \partial x_L \rVert$은 1.05로 사전 활성화의 1.83과 자릿수가 같다. 실제로 갈리는 것은 5절의 1001층에서이며, 거기서도 사후 활성화는 수렴하되 오차가 7.61%로 164층보다 나쁘다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
차원이 바뀔 때 쓰는 사영 지름길을 갖춘 잔차 블록을 PyTorch로 구현하라.

</div>

??? success "연습문제 3 풀이"
    아래는 원래 ResNet(사후 활성화) 방식이다. 차원이 맞을 때는 `nn.Identity`로 손대지 않고, 보폭이나 채널 수가 바뀔 때만 1×1 합성곱과 배치 정규화로 사영한다.

    ```python
    class ResBlock(nn.Module):
        def __init__(self, in_ch, out_ch, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_ch)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_ch)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch)
            ) if stride != 1 or in_ch != out_ch else nn.Identity()

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return F.relu(out + self.shortcut(x))
    ```

    합성곱마다 `bias=False`인 까닭은 바로 뒤의 배치 정규화가 평균을 빼면서 치우침을 지우기 때문이다. 사전 활성화 판본으로 바꾸려면 4절의 `PreActBasicBlock`처럼 정규화와 활성화를 합성곱 앞으로 옮기고, 지름길에는 `x`를 그대로 넘기면 된다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
잔차 신경망과 앙상블 학습의 관계를 설명하라.

</div>

??? success "연습문제 4 풀이"
    Veit 등(2016)은 ResNet이 얕은 신경망의 앙상블처럼 움직임을 보였다. 잔차 연결을 풀어 보면 블록이 $L$개인 ResNet에 길이가 서로 다른 경로가 $2^L$개 있다. 이는 연습문제 1에서 $\prod (I + \partial F_l / \partial x_{l-1})$을 펼쳐 $2^L$개 항을 얻은 것과 같은 계산이다. 기울기는 대부분 짧은 경로(블록 3~5개)로 흐르는데, 이는 ResNet이 얕은 부분 신경망의 앙상블을 암묵적으로 학습시킴을 시사한다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
3절 [3]에서 0.5배 지름길의 기울기 비는 50블록 뒤에 $7.7 \times 10^{-15}$이었고, 씨앗 다섯 개의 가운뎃값은 $5.7 \times 10^{-15}$이었다. 1절의 식이 예측하는 값은 얼마이며, 측정값이 그보다 큰 까닭은 무엇인가? 같은 계산을 항등 지름길($\lambda = 1$)에 대해서도 해 보라.

</div>

??? success "연습문제 5 풀이"
    1절의 펼침 $x_L = \lambda^{L-l} x_l + \sum_i \lambda^{L-1-i} F_i(x_i)$을 $x_l$로 미분하면 지름길이 주는 항은 $\lambda^{L-l}$이다. $\lambda = 0.5$, $L - l = 50$이므로

    $$0.5^{50} = 8.88 \times 10^{-16}$$

    이다. 측정된 가운뎃값 $5.68 \times 10^{-15}$은 이보다 약 6.4배 크다. 남는 몫은 합 쪽에서 온다 — 잔차 가지를 거치는 경로들도 기울기를 나르며, 그 가운데 $i$가 큰(즉 $L$에 가까운) 항들은 $\lambda$가 몇 번밖에 곱해지지 않아 지름길 항보다 크게 살아남는다. 그래도 자릿수를 정하는 것은 $\lambda^{50}$이고, 측정값이 이론값의 한 자릿수 안에 있다는 것이 바로 그 증거다.

    $\lambda = 1$이면 지름길 항은 $1^{50} = 1$이므로 감쇠가 전혀 없다. 측정된 가운뎃값 1.675에서 1을 넘는 0.675가 곧 잔차 가지들이 얹은 몫이며, 이때는 그것이 기울기를 **키우는** 쪽으로 작용한다. 두 경우의 차이는 잔차 가지에 있지 않고 오로지 $\lambda^{50}$에 있다 — 이것이 "지름길에 손대지 말라"는 이 절의 결론이 뜻하는 바다.

---

## 정리하며

사전 활성화 ResNet은 중요한 순서 변경을 들여온다.

| 변경 | 영향 |
|--------|--------|
| 합성곱 앞의 배치 정규화 | 합성곱마다의 입력을 정규화한다 |
| 합성곱 앞의 ReLU | 정규화된 특징에 활성화를 적용한다 |
| 더하기 뒤에 아무것도 두지 않음 | $f$가 항등이 되어 지름길로 기울기가 순수하게 흐른다 |
| 분류기 앞의 마지막 배치 정규화 | 사전 활성화 방식을 완성한다 |

핵심 원리는 **건너뛰기 연결이 손대지 않은 항등 사상이어야 한다**는 것이다. 지름길 경로에 어떤 변환을 적용하든, 그것이 학습된 것(합성곱)이든 고정된 것(배율 조정)이든 비선형(ReLU)이든 기울기의 흐름과 학습 성능을 해친다.

다만 이 쪽이 재어 본 바로는, 그 해악이 **아무 깊이에서나 보이지는 않는다**. 지름길을 0.5배로 줄이면 50블록에서 이미 열네 자릿수가 사라지지만, 더하기 뒤에 ReLU를 두는 정도의 손댐은 같은 깊이에서 재도 잡히지 않는다. 사후 활성화와 사전 활성화가 수치로 갈리는 곳은 1001층이고, 거기서도 차이는 수렴 여부가 아니라 오차 7.61% 대 4.92%다. 원리는 깊이를 가리지 않지만 그 값은 깊이에 달려 있다.

**참고 문헌**

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Identity Mappings in Deep Residual Networks. *ECCV 2016*.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.
3. Veit, A., Wilber, M., & Belongie, S. (2016). Residual Networks Behave Like Ensembles of Relatively Shallow Networks. *NeurIPS 2016*.
4. Xiong, R., Yang, Y., He, J., Zheng, K., Zheng, S., Xing, C., Zhang, H., Lan, Y., Wang, L., & Liu, T. (2020). On Layer Normalization in the Transformer Architecture. *ICML 2020*.
