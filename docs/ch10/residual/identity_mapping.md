# 항등 사상

사전 활성화(Pre-activation) ResNet은 잔차 분기의 컨볼루션 앞에 BatchNorm과 ReLU를 배치하고 더하기 뒤의 활성화를 없앤다. 차원을 유지하는 블록에서는 입력을 그대로 더하는 항등 경로를 만들 수 있다. 이 절은 그 경로의 수식, 구현과 제한된 수치 예제를 살펴본다.

항등 경로는 깊은 모델의 최적화를 돕는 설계이지만 모든 구조에서 최적이거나 그래디언트 크기를 일정하게 보장하는 방법은 아니다. 채널·해상도 변경을 위한 투영 블록은 순수한 항등 블록과 구분한다.

## 1. 왜 항등 사상이 중요한가

원래 ResNet 논문은 다음 식으로 잔차 함수를 배우자고 제안했다.

$$y_l = h(x_l) + F(x_l, W_l)$$

$$x_{l+1} = f(y_l)$$

여기서 $h(x_l)$은 스킵 연결, $F$은 잔차 함수, $f$은 더하기 뒤의 엑티베이션 함수(ReLU)이다.

### 스킵 연결의 변형 실험

He 등은 $h(x_l)$의 여러 형태를 CIFAR-10에서 110레이어 ResNet으로 체계적으로 시험했다. 아래 수치는 논문 표 1의 테스트 오차이다.

| 지름길의 종류 | $h(x_l)$ | 테스트 오차 (ResNet-110) |
|---------------|----------|----------------|
| 항등 (원래) | $x_l$ | **6.61%** |
| 상수 배율 ($\lambda = 0.5$) | $0.5 \cdot x_l$ | 12.35% |
| 배타적 게이팅 | $(1 - g(x_l)) \cdot x_l$, 잔차 가지는 $g(x_l)$배 | 8.70% |
| 지름길에만 게이팅 | $(1 - g(x_l)) \cdot x_l$, 잔차 가지는 그대로 | 6.91% |
| 1×1 컨볼루션 | $W \cdot x_l$ | 12.22% |
| 드롭아웃 (0.5) | $\text{dropout}(x_l)$ | 수렴 실패 |

**해석 범위**: 표는 논문의 특정 CIFAR-10 설정에서 항등 지름길이 비교한 변형보다 낮은 테스트 오차를 보인 결과이다. 지름길을 변경하면 모든 데이터와 모델에서 반드시 성능이 나빠진다는 보편적인 정리는 아니다. 게이팅이나 투영의 효과도 해당 구조와 최적화 조건에서 비교한다.

### 수학적 설명

스킵 연결이 항등이고 **더하기 뒤의 $f$ 또한 항등이면** 순전파가 깔끔하게 펼쳐진다. 두 조건이 함께 필요하다는 점이 중요하다. $h$만 항등으로 두고 $f = \text{ReLU}$를 그대로 두면 아래의 펼침은 성립하지 않는다.

$$x_L = x_l + \sum_{i=l}^{L-1} F_i(x_i)$$

어떤 깊은 레이어도 얕은 레이어에 누적된 잔차를 더한 것으로 곧바로 나타난다. 항등이 아닌 지름길 $h(x_l) = \lambda x_l$을 쓰면(여전히 $f$는 항등으로 두고) 펼침이 다음과 같이 된다.

$$x_L = \lambda^{L-l} x_l + \sum_{i=l}^{L-1} \lambda^{L-1-i} F_i(x_i)$$

상수 양의 배율을 사용하면 순수 지름길 항은 $\lambda^{L-l}$로 바뀐다. $0<\lambda<1$이면 감소하고 $\lambda>1$이면 증가한다. 전체 신호와 그래디언트는 잔차 경로의 기여도 포함하므로 이 항만으로 전체 노름의 증가·감소를 결정할 수는 없다. $\lambda=0.5$, 깊이 차 50의 순수 지름길 계수는 약 $8.9\times10^{-16}$이다.

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

**문제**: 더하기 뒤에 ReLU가 있으므로 스킵 연결을 지나 다음 블록으로 가는 신호*까지* ReLU를 거치게 되어 순수한 항등 사상이 깨진다. 출력 $x_{l+1} = \text{ReLU}(x_l + F(x_l))$은 언제나 음이 아니므로 항등 경로가 $\mathbb{R}_{\geq 0}$으로 제약된다.

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

**해결**: 배치 정규화와 ReLU를 컨볼루션 앞으로 옮기면 $f$가 항등이 되고, 따라서 스킵 연결도 참된 항등 사상이 된다. 신호가 잇따른 블록을 손대지 않은 채 흐르고 출력은 다음과 같이 간단해진다.

$$x_{l+1} = x_l + F(\hat{f}(x_l), W_l)$$

여기서 $\hat{f}$은 잔차 가지 안에서만 적용되는 사전 활성화(배치 정규화 뒤 ReLU)를 뜻한다. 지름길에 더해지는 것은 $\hat{f}(x_l)$이 아니라 손대지 않은 $x_l$이다 — 이 구별이 4절 구현에서 한 줄 차이로 갈리며, 잘못 쓰면 이 절의 모든 식이 무너진다.

---

## 3. 수학적 분석

### 정보의 전파

사전 활성화를 쓰면 순전파가 다음과 같이 된다.

$$x_{l+1} = x_l + F(\hat{f}(x_l), W_l)$$

이 점화식을 펼치면 다음과 같다.

$$x_L = x_l + \sum_{i=l}^{L-1} F(\hat{f}(x_i), W_i)$$

순전파는 같은 차원 구간에서 얕은 피처와 중간 잔차들의 합으로 표현된다. 잔차 $F_i$ 자체는 앞선 피처와 파라메터에 복잡하게 의존하므로, 이 덧셈 표현이 내부 컨볼루션이나 역전파의 행렬 곱을 없앤다는 뜻은 아니다.

### 그래디언트의 흐름

벡터 피처에서 야코비안을 사용하면 열벡터 그래디언트 g_l은 다음처럼 쓸 수 있다. 여기서 각 F_i의 미분은 중간 피처를 통한 의존성까지 포함한 전체 미분이다.

$$g_l=\left(I+\frac{\partial}{\partial x_l}\sum_{i=l}^{L-1}F_i\right)^\top g_L$$

항등행렬 I에 대응하는 직접 경로가 한 항으로 남는다. 전체 그래디언트는 이 항과 다른 경로의 벡터 합이므로 상쇄되거나 커질 수 있다. 배율 지름길에서도 전체 야코비안에는 잔차 가지의 미분이 포함된다.

!!! warning "항등 항은 그래디언트 노름의 하한이 아니다"
    $F(x)=-x$이면 한 블록의 야코비안은 $I-I=0$이다. 전체 야코비안이 0이 아니어도 특정 상류 그래디언트가 그 전치의 영공간에 놓이면 해당 기울기가 0이 될 수 있다. 모든 표본의 잔차 야코비안이 정확히 −I여야만 소실이 가능하다는 조건은 필요하지 않다.

### 그래디언트 경로의 비교

| 항목 | 원래 ResNet (사후 활성화) | 사전 활성화 ResNet |
|------|-----------------|----------------------|
| 항등 경로의 그래디언트 | 블록마다 더하기 뒤 ReLU의 마스크를 지난다 | 어떤 비선형도 지나지 않는다 |
| 그래디언트에 붙는 인수 | $\prod_{i=l}^{L-1} \mathbb{1}[y_i > 0]$ 이 끼어든다 | 덧셈항 $1$이 그대로 남는다 |
| 그래디언트 소실 가능성 | ReLU와 전체 야코비안에 의존 | 잔차와 항등 경로의 상쇄·특잇값에 의존 |

이 표의 마지막 줄은 **원리**를 적은 것이지 아무 깊이에서나 관측되는 현상이 아니다. 어느 깊이에서 실제로 갈리는지는 바로 아래에서 재어 본다.

### 수를 재어 확인하기

세 가지를 확인한다. 하나, 펼침 공식 $x_L = x_l + \sum F_i$이 정말 성립하는가. 둘, 그래디언트가 정말 $\partial \mathcal{L} / \partial x_L$과 잔차항의 **합**으로 갈라지는가. 셋, 50블록을 쌓았을 때 $\lVert \partial \mathcal{L} / \partial x_l \rVert$이 깊이에 따라 어떻게 달라지는가.

셋째 실험에서는 위쪽에서 내려오는 그래디언트 $\partial \mathcal{L} / \partial x_L$을 세 경우에 똑같이 고정하고 그것으로 나눈 비를 본다. 그렇게 하지 않으면 순전파의 크기 차이(0.5배 지름길은 순전파도 함께 죽는다)가 섞여 들어와 역전파의 감쇠를 따로 볼 수 없다.

```python
"""항등 지름길이 그래디언트에 무엇을 하는지 직접 재어 본다."""

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


# === 2. 그래디언트가 덧셈으로 갈라지는지 확인한다
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


# === 3. 깊이에 따른 그래디언트 크기: 항등 지름길 · 배율 지름길 · 사후 활성화
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

    # 위쪽에서 내려오는 그래디언트를 세 경우에 똑같이 고정한다.
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

- **[1], [2]**: 기록된 약 $4.77\times10^{-7}$ 차이는 해당 float32 예제에서 수식과 계산이 수치적으로 가깝다는 관찰이다. 허용 오차는 값의 스케일과 연산 수에 따라 달라진다.
- **[3]의 첫 줄**: 해당 사전 활성화 모델의 50블록 그래디언트 노름 비는 약 1.83이다. 모든 깊이·웨이트에서 감소하지 않는다는 보장은 아니다.
- **[3]의 둘째 줄**: 배율 0.5인 모델에서 비가 약 $7.7\times10^{-15}$이다. 전체 야코비안의 효과이며 순수 지름길 계수 하나의 예측값으로 해석하지 않는다.

!!! note "이 실험과 논문의 성능 비교는 다르다"
    사후 활성화 모델도 이 수치 예제에서 그래디언트가 사라지지 않는다. 반면 사전·사후 활성화의 기록된 노름 비 범위는 달라, 둘이 완전히 구별되지 않는다는 표현도 부정확하다. 이 실험은 특정 초기화에서의 그래디언트 비교이며, 논문의 학습 후 테스트 오차 차이와 직접적인 원인–결과 관계를 입증하지 않는다.

---

## 4. 구현

### 사전 활성화 기본 블록

구현에서 가장 틀리기 쉬운 줄은 지름길을 고르는 한 줄이다. 사전 활성화를 계산한 `out`을 지름길로 넘겨주고 싶은 유혹이 있지만, 그러면 2절이 공들여 세운 항등 경로가 그 자리에서 깨진다. 지름길로 가는 것은 손대지 않은 `x`이고, 투영이 필요할 때만 — 채널 수나 해상도가 바뀔 때만 — 이미 정규화된 `out`에 1×1 컨볼루션을 걸친다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Type, List


class PreActBasicBlock(nn.Module):
    """
    사전 활성화 기본 블록.

    구조: BN → ReLU → Conv → BN → ReLU → Conv → 더하기

    배치 정규화와 ReLU가 컨볼루션 뒤에 오는 원래 BasicBlock과 달리
    여기서는 컨볼루션 앞에 와서 참된 항등 지름길을 만든다.
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

        # 사전 활성화: 컨볼루션 앞에 배치 정규화와 ReLU
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

        # 첫 컨볼루션 (사전 활성화된 입력에)
        out = self.conv1(out)

        # 둘째 사전 활성화와 컨볼루션
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
    1. 잔차 블록에서 배치 정규화와 ReLU를 컨볼루션 앞으로 옮겼다
    2. 분류기 앞에 마지막 배치 정규화를 두었다 (마지막 블록에 사후 활성화가 없으므로)
    3. 항등 경로를 지나는 그래디언트의 흐름이 더 깨끗하다
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

        # 첫 컨볼루션 (배치 정규화와 ReLU 없음 — 첫 블록에 들어간다)
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
        print(f"{name:<16} 파라메터 {n:>10,}   출력 {tuple(y.shape)}")
```

**출력:**

```
F = 0 일 때 max |block(x) - x| = 0.00e+00
지름길을 ReLU(BN(x))로 잘못 두면   = 3.33e+00
preact_resnet18  파라메터 11,179,850   출력 (2, 10)
preact_resnet50  파라메터 23,520,842   출력 (2, 10)
```

출력은 소수점 형식으로 차이가 0으로 표시된다는 뜻이다. 숫자가 표시상 0이라는 사실만으로 일반적인 정확한 항등성이 증명되지는 않는다. 차원이 같은 블록에서 잔차를 정확히 0으로 두고 지름길이 x를 그대로 반환하는 구조라면 대수적으로 y=x임을 확인할 수 있다. 이때에도 덧셈 연산 자체는 수행된다.

---

## 5. 실험 결과

### CIFAR-10에서의 성능 비교

He 등의 논문 표 3에 실린 테스트 오차이다.

| 깊이 | 원래 ResNet | 사전 활성화 ResNet |
|-------|-----------------|----------------------|
| 110 | 6.61% | 6.37% |
| 164 | 5.93% | 5.46% |
| 1001 | 7.61% | **4.92%** |

**해석 범위**: 표는 해당 논문 설정에서 사전 활성화가 비교한 깊이마다 낮은 테스트 오차를 보인 결과이다. 특정 깊이에서의 테스트 오차만으로 최적화가 수학적으로 수렴했는지 판정하거나, 모든 더 깊은 모델에서 개선이 계속된다고 보장하지 않는다.

### 활성화 순서에 대한 절제 실험

He 등은 배치 정규화와 ReLU를 놓을 수 있는 여러 자리를 시험했다(논문 표 2).

| 순서 | ResNet-110 | ResNet-164 |
|----------|------|------|
| 사후 활성화 (원래) | 6.61% | 5.93% |
| 더하기 뒤에 배치 정규화 | 8.17% | 6.50% |
| 더하기 앞에 ReLU | 7.84% | 6.14% |
| ReLU만 앞으로 | 6.71% | 5.91% |
| 완전한 사전 활성화 (배치 정규화 + ReLU) | **6.37%** | **5.46%** |

ReLU만 앞으로 옮긴 조건과 BatchNorm까지 함께 옮긴 조건의 결과가 다르다. 이는 해당 실험에서 두 변경의 조합을 비교한 것이며, 오차 차이가 얼마나 안정적인지나 다른 과제로 전이되는지는 추가 검증이 필요하다.

---

## 6. 비교: 원래 방식과 사전 활성화

| 항목 | 원래 ResNet | 사전 활성화 ResNet |
|--------|-----------------|----------------------|
| 배치 정규화와 ReLU의 위치 | 컨볼루션 뒤 | 컨볼루션 앞 |
| 스킵 연결 | 항등에 사후 ReLU | 순수한 항등 |
| 마지막 레이어의 출력 | 활성화됨 | 마지막에 배치 정규화와 ReLU가 필요 |
| 그래디언트 경로 | 활성화를 $L$번 지남 | 곧바른 항등 경로 |
| 논문의 1001레이어 실험 | 7.61% 테스트 오차 | 4.92% 테스트 오차 |
| 체크포인트 사용 | 해당 구조·전처리와의 호환성 확인 | 해당 구조·전처리와의 호환성 확인 |

---

## 7. 사전 활성화 ResNet을 쓸 때

사전·사후 활성화는 구조 선택의 후보이다. 비교에서는 같은 데이터 분할, 조율·학습 예산과 평가 기준을 사용하고 깊이에 따른 정확도와 비용을 확인한다. 사전 활성화가 100레이어 이상에서 반드시 필요하거나 모든 조밀 예측에서 더 좋은 피처를 준다고 가정하지 않는다.

사전 학습 체크포인트를 쓰면 블록 순서와 투영 방식, 전처리를 그 구현과 맞춰야 한다. 사후 활성화 웨이트를 같은 모양의 사전 활성화 구조에 단순 복사해도 같은 함수를 계산한다는 보장은 없다.

---

## 8. 트랜스포머 구조와의 관계

Pre-LN 트랜스포머도 잔차 분기의 어텐션·FFN 앞에 LayerNorm을 두고 입력을 직접 더한다. 이 점에서 같은 차원의 사전 활성화 잔차 블록과 구조적 공통점이 있다. 역사적인 직접 영향이나 모든 트랜스포머의 우위는 이 식만으로 주장하지 않는다.

$$x_{l+1} = x_l + \text{Attention}(\text{LN}(x_l))$$

$$x_{l+2} = x_{l+1} + \text{FFN}(\text{LN}(x_{l+1}))$$

차원이 같고 더하기 뒤 추가 변환이 없는 구간이면 앞의 덧셈 표현을 적용할 수 있다. 그러나 어텐션과 정규화의 야코비안·초기화·스케일링이 전체 그래디언트를 결정하므로 잔차 연결이 있는 모든 모델에서 개선을 보장하지 않는다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
잔차 블록의 야코비안을 구하고 항등 경로가 그래디언트 노름의 양의 하한을 보장하지 않는 반례를 제시하라.

</div>

??? success "풀이"
    $y=x+F(x)$의 야코비안은 $I+J_F(x)$이다. 여러 블록의 야코비안은 출력 쪽부터 순서대로 $(I+J_{F_L})\cdots(I+J_{F_1})$을 곱한다. $F=-x$이면 한 블록부터 0이고 $F=-0.9x$이면 L개 블록에서 $0.1^LI$이다. 행렬 전체가 0이 아니어도 특정 그래디언트 방향이 소실될 수 있다. 항등항의 존재와 전체 노름의 하한은 별개이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
사전 활성화와 사후 활성화 잔차 블록의 차이와 성능 비교의 범위를 설명하라.

</div>

??? success "풀이"
    사후 활성화는 더하기 뒤 ReLU가 있으므로 y=x+F(x) 형태의 항등 출력이 일반적으로 성립하지 않는다. 사전 활성화는 BatchNorm·ReLU를 잔차 분기 앞쪽에 두어 차원을 유지하는 구간의 직접 경로를 항등으로 만든다. 논문의 특정 깊은 모델에서는 오차가 낮았지만 항상 더 좋거나 사후 활성화가 특정 깊이부터 학습 불가능하다는 뜻은 아니다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
차원이 바뀔 때 쓰는 투영 지름길을 갖춘 잔차 블록을 PyTorch로 구현하라.

</div>

??? success "풀이"
    아래는 원래 ResNet(사후 활성화) 방식이다. 차원이 맞을 때는 `nn.Identity`로 손대지 않고, 스트라이드나 채널 수가 바뀔 때만 1×1 컨볼루션과 배치 정규화로 투영한다.

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

    컨볼루션 뒤 BatchNorm에서는 학습 모드의 채널별 상수 바이어스가 배치 평균에서 상쇄되어 bias=False를 흔히 사용한다. 평가 모드에서는 고정된 이동 통계와 바이어스를 함께 고려해야 한다. 사전 활성화로 바꿀 때에는 활성화 순서뿐 아니라 차원 변경 시 투영의 입력과 항등 구간의 지름길을 해당 설계에 맞춰 정한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
잔차 네트워크의 경로 분해를 독립 모델의 앙상블과 동일시할 수 있는가?

</div>

??? success "풀이"
    블록 야코비안 곱을 전개하면 짧고 긴 경로에 대응하는 항들이 나오고, 경로별 해석은 모델 분석에 유용하다. 그러나 같은 파라메터와 중간 피처를 공유하므로 독립적으로 학습한 $2^L$개 모델의 평균과 일반적으로 동일하지 않다. 어떤 길이의 경로가 우세한지도 구조·초기화·학습과 측정 방식에 따라 달라진다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
배율 0.5와 깊이 차 50의 순수 지름길 계수를 계산하고 전체 그래디언트 노름 비와 어떻게 구별할지 설명하라.

</div>

??? success "풀이"
    직접 경로의 계수는 $0.5^{50}\approx8.88\times10^{-16}$이다. 전체 그래디언트에는 잔차 가지를 거치는 모든 경로의 벡터 기여가 포함되므로 측정 비가 이 값과 같아야 하는 것은 아니다. 항등 지름길에서도 전체 노름 비 1.675에서 1을 뺀 0.675가 잔차 기여의 노름이 되는 것은 아니다. 일반적으로 $\|a+b\|\ne\|a\|+\|b\|$이다. 지름길 변경은 중간 피처와 잔차 야코비안도 바꾸므로 두 모델의 차이를 직접 경로 계수만으로 분리하지 않는다.

---

## 정리하며

차원을 유지하는 사전 활성화 블록은 y=x+F(x) 형태의 직접 경로를 제공한다. 이는 유용한 덧셈 분해이지만 그래디언트 소실·폭발을 금지하거나 모든 깊이에서 성능을 보장하지 않는다. 수치 예제의 노름 비와 논문의 테스트 오차는 서로 다른 측정이며, 투영이 있는 구간은 항등 구간과 구분한다.

**참고 문헌**

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Identity Mappings in Deep Residual Networks. *ECCV 2016*.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.
3. Veit, A., Wilber, M., & Belongie, S. (2016). Residual Networks Behave Like Ensembles of Relatively Shallow Networks. *NeurIPS 2016*.
4. Xiong, R., Yang, Y., He, J., Zheng, K., Zheng, S., Xing, C., Zhang, H., Lan, Y., Wang, L., & Liu, T. (2020). On Layer Normalization in the Transformer Architecture. *ICML 2020*.
