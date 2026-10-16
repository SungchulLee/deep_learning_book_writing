# 기본 잔차 블록

잔차 블록은 잔차 분기 F(x)와 지름길 S(x)를 더한다. 이 예제의 사후 활성화 블록은 $y=\operatorname{ReLU}(F(x)+S(x))$를 계산하며, 두 3×3 컨볼루션과 BatchNorm으로 F를 구성한다.

차원이 같으면 입력을 직접 더하고 채널 수나 공간 크기가 바뀌면 투영으로 모양을 맞춘다. 이런 파라메터화는 깊은 모델의 최적화를 돕지만 학습 성공이나 그래디언트 크기를 보장하지는 않는다.

## 1. 코드

```python
"""
기본 잔차 블록 구현
====================================
뉴럴 넷트웍의 잔차 연결(건너뛰기 연결) 소개.

핵심 개념:
H(x)를 배우는 대신 F(x) = H(x) - x를 배우므로 출력은 F(x) + x이다.
덕분에 그래디언트가 건너뛰기 연결을 타고 뉴럴 넷트웍을 곧바로 흐른다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class BasicBlock(nn.Module):
    """
    ResNet을 위한 기본 잔차 블록

    구조:
    입력 ─┬─ Conv ─ BN ─ ReLU ─ Conv ─ BN ─ (+) ─ ReLU ─ 출력
          └────────── 건너뛰기 연결 ──────────┘

    더하기는 마지막 ReLU 앞에 온다 (He 등, 2015의 사후 활성화 방식).
    건너뛰기 연결은 모양이 맞으면 항등 사상이고, 모양이 어긋나면
    1x1 컨볼루션 + BN으로 맞춘 사영이다.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        # 첫 컨볼루션 레이어 (stride를 여기서 준다: 공간 크기가 줄어드는 유일한 자리)
        # 바로 뒤에 BN이 오므로 bias=False로 둔다. BN이 평균을 빼 버려
        # 컨볼루션의 치우침 항은 출력에 아무 영향도 주지 못한다.
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 둘째 컨볼루션 레이어 (stride=1, 채널 수 그대로)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 건너뛰기 연결 (지름길). 빈 Sequential은 입력을 그대로 돌려주므로
        # 이 경우의 지름길은 파라메터가 0개인 항등 사상이다.
        self.shortcut = nn.Sequential()

        # 모양이 어긋나면 (공간 크기가 줄거나 채널 수가 바뀌면) 더할 수가 없다.
        # 1x1 컨볼루션에 같은 stride를 주어 모양을 맞춘다 (사영 지름길).
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # 주 경로: Conv - BN - ReLU - Conv - BN (마지막에 ReLU가 없다)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # 건너뛰기 연결 더하기
        out += self.shortcut(x)

        # 마지막 활성화 (더하기 뒤에 온다)
        out = F.relu(out)

        return out


class PlainBlock(nn.Module):
    """
    견주기 위한 평범한 블록 (건너뛰기 연결 없음)

    BasicBlock과 레이어의 종류·순서·생성 순서가 모두 같고 더하기만 없다.
    따라서 같은 씨앗에서 만들면 두 블록의 웨이트가 글자 그대로 같아진다.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(PlainBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


# ========================================================================
# 그래디언트 흐름 재기
# ========================================================================


def input_grad_norm(block_cls, depth, seed):
    """
    block_cls를 depth개 쌓은 뉴럴 넷트웍에 입력을 넣고, 출력의 합을 역전파했을 때
    입력에 맺히는 그래디언트의 노름을 돌려준다. 씨앗을 먼저 심으므로 입력과
    웨이트가 씨앗 하나로 결정된다.
    """
    torch.manual_seed(seed)
    x = torch.randn(1, 64, 32, 32, requires_grad=True)
    net = nn.Sequential(*[block_cls(64, 64) for _ in range(depth)])
    net(x).sum().backward()
    return x.grad.norm().item()


def demonstrate_gradient_flow(depths=(1, 4, 8, 16), seeds=(0, 1, 2)):
    """
    잔차 블록과 평범한 블록을 같은 웨이트로 견주어 그래디언트의 흐름을 잰다
    """
    print("=" * 60)
    print("Gradient Flow Demonstration")
    print("=" * 60)

    # 같은 씨앗이면 두 블록의 웨이트가 정말 같은지 먼저 확인한다.
    torch.manual_seed(0)
    res_block = BasicBlock(64, 64)
    torch.manual_seed(0)
    plain_block = PlainBlock(64, 64)
    same = all(torch.equal(a, b) for a, b in
               zip(res_block.parameters(), plain_block.parameters()))
    print(f"\nSame weights under the same seed: {same}")
    print("So the only difference between the two columns is the skip.")

    print("\n depth |  residual: min - max  |     plain: min - max")
    print("-" * 58)
    for depth in depths:
        res = [input_grad_norm(BasicBlock, depth, s) for s in seeds]
        plain = [input_grad_norm(PlainBlock, depth, s) for s in seeds]
        print(f" {depth:5d} | {min(res):9.2f} - {max(res):9.2f}"
              f" | {min(plain):11.2f} - {max(plain):11.2f}")

    print("\nOne block: the skip adds the identity path, so the residual")
    print("gradient is about 2.4x the plain one.")
    print("Sixteen blocks: the plain gradient has grown ~300x from its own")
    print("depth-1 value while the residual one grew ~8x.")
    print("The skip keeps the gradient scale nearly depth-independent.")
    print("=" * 60)


# ========================================================================
# 모양과 파라메터 확인
# ========================================================================


def test_blocks():
    """
    기본 잔차 블록이 제대로 움직이는지 시험한다
    """
    print("\n" + "=" * 60)
    print("Testing Residual Blocks")
    print("=" * 60)

    torch.manual_seed(0)

    # 차원이 같은 경우 시험
    print("\n1. Same dimensions (64 -> 64)")
    block1 = BasicBlock(64, 64)
    x1 = torch.randn(2, 64, 32, 32)
    out1 = block1(x1)
    print(f"   Input shape:  {x1.shape}")
    print(f"   Output shape: {out1.shape}")
    print(f"   Shortcut modules: {len(block1.shortcut)} (identity, no parameters)")

    # 차원이 다른 경우 시험
    print("\n2. Different dimensions (64 -> 128, stride=2)")
    block2 = BasicBlock(64, 128, stride=2)
    x2 = torch.randn(2, 64, 32, 32)
    out2 = block2(x2)
    print(f"   Input shape:  {x2.shape}")
    print(f"   Output shape: {out2.shape}")
    print("   Notice: Spatial dimensions halved, channels doubled")
    print(f"   Shortcut modules: {len(block2.shortcut)} (1x1 conv + BN projection)")
    print(f"   Shortcut output shape: {block2.shortcut(x2).shape}")

    # 파라메터 개수 세기
    print("\n3. Parameter count")
    total1 = sum(p.numel() for p in block1.parameters())
    print(f"   BasicBlock(64, 64):            {total1:,}")
    print(f"     conv1 3*3*64*64 = {block1.conv1.weight.numel():,}"
          f"   bn1 2*64 = {sum(p.numel() for p in block1.bn1.parameters())}")
    print(f"     conv2 3*3*64*64 = {block1.conv2.weight.numel():,}"
          f"   bn2 2*64 = {sum(p.numel() for p in block1.bn2.parameters())}")

    total2 = sum(p.numel() for p in block2.parameters())
    short2 = sum(p.numel() for p in block2.shortcut.parameters())
    print(f"   BasicBlock(64, 128, stride=2): {total2:,}")
    print(f"     of which the 1x1 projection: {short2:,}"
          f" ({100 * short2 / total2:.1f}%)")

    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RESIDUAL CONNECTIONS - BASIC CONCEPTS")
    print("=" * 60)

    print("\nKey Benefits of Residual Connections:")
    print("1. Easier gradient flow (keeps the gradient scale steady with depth)")
    print("2. Enables training of very deep networks (100+ layers)")
    print("3. Learning identity function is easy (F(x) = 0)")
    print("4. Better optimization landscape")

    # 시험 실행
    test_blocks()

    # 그래디언트의 흐름 보이기
    demonstrate_gradient_flow()

    print("\n" + "=" * 60)
    print("Next: See 02_resnet_implementation.md for full ResNet architecture")
    print("=" * 60 + "\n")
```

**출력:**

```
============================================================
RESIDUAL CONNECTIONS - BASIC CONCEPTS
============================================================

Key Benefits of Residual Connections:
1. Easier gradient flow (keeps the gradient scale steady with depth)
2. Enables training of very deep networks (100+ layers)
3. Learning identity function is easy (F(x) = 0)
4. Better optimization landscape

============================================================
Testing Residual Blocks
============================================================

1. Same dimensions (64 -> 64)
   Input shape:  torch.Size([2, 64, 32, 32])
   Output shape: torch.Size([2, 64, 32, 32])
   Shortcut modules: 0 (identity, no parameters)

2. Different dimensions (64 -> 128, stride=2)
   Input shape:  torch.Size([2, 64, 32, 32])
   Output shape: torch.Size([2, 128, 16, 16])
   Notice: Spatial dimensions halved, channels doubled
   Shortcut modules: 2 (1x1 conv + BN projection)
   Shortcut output shape: torch.Size([2, 128, 16, 16])

3. Parameter count
   BasicBlock(64, 64):            73,984
     conv1 3*3*64*64 = 36,864   bn1 2*64 = 128
     conv2 3*3*64*64 = 36,864   bn2 2*64 = 128
   BasicBlock(64, 128, stride=2): 230,144
     of which the 1x1 projection: 8,448 (3.7%)
============================================================
============================================================
Gradient Flow Demonstration
============================================================

Same weights under the same seed: True
So the only difference between the two columns is the skip.

 depth |  residual: min - max  |     plain: min - max
----------------------------------------------------------
     1 |    220.03 -    221.34 |       91.79 -       94.37
     4 |    385.83 -    392.50 |      286.62 -      295.06
     8 |    757.48 -    781.23 |     1314.94 -     1350.58
    16 |   1679.75 -   1764.44 |    26671.74 -    28533.47

One block: the skip adds the identity path, so the residual
gradient is about 2.4x the plain one.
Sixteen blocks: the plain gradient has grown ~300x from its own
depth-1 value while the residual one grew ~8x.
The skip keeps the gradient scale nearly depth-independent.
============================================================

============================================================
Next: See 02_resnet_implementation.md for full ResNet architecture
============================================================
```

## 2. 논의

**블록 안의 순서.** 주 경로는 Conv → BN → ReLU → Conv → BN이고, 둘째 BN 뒤에는 ReLU가 없다. 마지막 ReLU는 더하기 **뒤**에 온다.

$$y = \operatorname{ReLU}\bigl(F(x) + \mathcal{S}(x)\bigr)$$

여기서 $F$는 컨볼루션 두 개짜리 주 경로, $\mathcal{S}$는 지름길이다. 코드의 `F`는 `torch.nn.functional`의 관례적인 별칭이라 수식의 $F$와 글자가 겹치지만, 둘은 서로 다른 것이다. 둘째 BN 바로 뒤에 ReLU를 걸면 $F(x) \ge 0$이 되어 잔차가 입력을 깎아내릴 수 없게 되므로, 그 자리를 비워 두는 것이 이 구조의 핵심이다. 반대로 마지막 ReLU가 더하기 뒤에 있다는 것은 건너뛰기 경로 또한 블록마다 ReLU를 한 번씩 지난다는 뜻이기도 하다. 이 점을 문제 삼아 더하기를 활성화 밖으로 빼내는 사전 활성화 설계는 [항등 사상](identity_mapping.md) 쪽에서 다룬다.

**컨볼루션의 바이어스**: 이 구현은 컨볼루션 뒤 BatchNorm이 있으므로 bias=False를 사용한다. 학습 모드에서 같은 배치 통계로 정규화하면 채널별 상수 바이어스는 평균에서 상쇄된다. 그러나 평가 모드의 이동 평균을 고정한 채 바이어스만 추가·제거하면 출력이 달라질 수 있다. 바이어스가 모든 모드에서 무조건 사라진다는 뜻은 아니다.

**지름길의 두 모습.** 출력의 1번 시험에서 `Shortcut modules: 0`, 2번 시험에서 `Shortcut modules: 2`가 찍힌다. 입출력 모양이 같으면 지름길은 파라메터가 0개인 항등 사상이고, 모양이 어긋나면 1×1 컨볼루션과 BN으로 이루어진 **투영 지름길**이 된다. 모양이 맞아떨어지는 까닭은 두 경로의 크기 공식이 같기 때문이다. 공간 크기는 다음을 따른다.

$$H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2p - k}{s} \right\rfloor + 1$$

주 경로의 첫 컨볼루션은 $k = 3$, $p = 1$이므로 $\lfloor (H_{\text{in}} - 1)/s \rfloor + 1$이고, 지름길의 1×1 컨볼루션은 $k = 1$, $p = 0$이므로 역시 $\lfloor (H_{\text{in}} - 1)/s \rfloor + 1$이다. 두 식이 글자 그대로 같으니 어떤 $H_{\text{in}}$과 어떤 $s$에서도 두 경로의 공간 크기가 저절로 맞는다. 출력이 보이는 $32 \to 16$이 그 한 경우다. `stride`를 주는 자리는 `conv1`과 지름길 둘뿐이며, `conv2`는 언제나 `stride=1`이다.

**파라메터의 값.** 3×3 컨볼루션은 편향이 없으므로 $9 C_{\text{in}} C_{\text{out}}$개, 배치 정규화는 채널마다 크기와 바이어스 하나씩이므로 $2C_{\text{out}}$개를 쓴다. 채널이 $C$로 유지되는 블록은 그래서 다음을 갖는다.

$$2(9C^2 + 2C) = 18C^2 + 4C$$

$C=64$의 차원 유지 블록은 73984개이다. BasicBlock(64,128,stride=2)는 투영을 포함해 230144개이고 투영의 파라메터는 8448개이다. 전체 네트워크에서 투영이 차지하는 비율은 채널·블록 수와 투영 설계에 따라 다시 계산해야 한다.

**그래디언트 표의 조건**: 비교 대상은 같은 종류·순서의 레이어를 만들고 출력에 기록된 웨이트 일치를 확인한다. 스킵 더하기는 이후 중간 엑티베이션과 BatchNorm 통계·야코비안까지 바꾸므로 두 모델의 그래디언트 차이를 단순히 동일한 잔차 그래디언트에 I 하나를 더한 값으로만 설명할 수 없다. 블록 수와 컨볼루션 레이어 수도 구분해 표의 depth 인수를 읽는다.

**해석 범위**: 세 시드의 이 예제에서는 잔차 모델의 입력 그래디언트 노름이 비교 모델보다 깊이에 덜 민감했다. 관측 범위가 겹치지 않는 것만으로 모든 시드 분포의 차이를 입증하거나 증가율의 일반 법칙을 정할 수는 없다. BatchNorm이 있다고 그래디언트 소실이 불가능한 것도 아니다. 결과는 초기화 직후의 특정 입력·상류 그래디언트에서 얻은 값이며 학습 후 성능과는 구분한다. [학습 비교](03_training_comparison.md)에서 별도 학습 기록을 살펴본다.

**다음 쪽과의 이름 차이.** 여기서는 지름길을 블록 스스로 만들어 `self.shortcut`에 담았다. 다음 쪽 [ResNet 구현](02_resnet_implementation.md)의 `BasicBlock`은 같은 것을 바깥에서 `downsample` 인자로 받는다. 한 단계 안에서 투영이 필요한 것은 첫 블록뿐이므로, `_make_layer`가 투영을 만들어 첫 블록에만 넘기고 나머지 블록은 `downsample=None`으로 둔다. 만드는 자리만 다를 뿐 계산하는 함수는 똑같다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
식 $18C^2 + 4C$로 `BasicBlock(128, 128)`의 파라메터 개수를 구하고, `sum(p.numel() for p in BasicBlock(128, 128).parameters())`로 확인하라.

</div>

??? success "연습문제 1 풀이"
    $18 \cdot 128^2 + 4 \cdot 128 = 18 \cdot 16384 + 512 = 294{,}912 + 512 = 295{,}424$이다.

    ```python
    b = BasicBlock(128, 128)
    print(sum(p.numel() for p in b.parameters()))   # 295424
    ```

    내역은 conv1 147456개, bn1 256개, conv2 147456개, bn2 256개이다. 채널을 두 배로 늘리면 이차항인 컨볼루션 웨이트는 네 배지만 일차항인 BatchNorm 계수는 두 배이므로 전체가 정확히 네 배는 아니다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
BasicBlock(64,64)의 두 컨볼루션에 바이어스를 추가하면 계수가 얼마나 늘고, BatchNorm에서 어떤 조건으로 상쇄되는가?

</div>

??? success "연습문제 2 풀이"
    각 출력 채널당 하나씩 총 128개가 늘어 74112개이다. 학습 모드의 배치 정규화에서는 $(Wx+b)-(\mu_{Wx}+b)=Wx-\mu_{Wx}$로 상쇄된다. 평가에서는 이동 평균이 고정되므로 바이어스와 통계를 함께 맞추지 않으면 같은 함수라는 보장이 없다. BatchNorm의 선형 변환이 있어 일반적으로 앞선 바이어스를 생략하는 설정을 사용할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
`BasicBlock`의 `__init__`에서 `if stride != 1 or in_channels != out_channels:` 가지를 지워 투영 지름길이 만들어지지 않게 한 뒤, `BasicBlock(64, 128, stride=2)`에 $(2, 64, 32, 32)$ 텐서를 넣어라. 어떤 오류가 나며 그 수는 무엇을 가리키는가?

</div>

??? success "연습문제 3 풀이"
    지름길이 빈 `nn.Sequential()`로 남아 입력을 그대로 돌려주므로, 더하기에서 주 경로의 $(2, 128, 16, 16)$과 입력의 $(2, 64, 32, 32)$를 맞붙이게 된다. 실행하면 다음이 뜬다.

    ```
    RuntimeError: The size of tensor a (16) must match the size of tensor b (32)
    at non-singleton dimension 3
    ```

    16은 `stride=2`를 지난 주 경로의 너비, 32는 손대지 않은 입력의 너비다. PyTorch가 마지막 축부터 맞춰 보므로 너비에서 먼저 걸리지만, 채널 축(128 대 64)도 똑같이 어긋나 있다. 투영 지름길은 이 두 어긋남을 1×1 컨볼루션 하나로 한꺼번에 고친다. 출력 채널 수로 채널을 맞추고, 같은 `stride`로 공간 크기를 맞춘다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
`demonstrate_gradient_flow(depths=(16, 32))`로 깊이 32를 함께 재어라. 깊이 16에서 32로 갈 때 두 레이어들의 연결의 그래디언트는 각각 몇 배가 되는가?

</div>

??? success "연습문제 4 풀이"
    시드 0, 1, 2에서 얻는 표의 두 줄은 다음과 같다.

    ```
     depth |  residual: min - max  |     plain: min - max
        16 |   1679.75 -   1764.44 |    26671.74 -    28533.47
        32 |   3840.75 -   4058.23 | 12094596.00 - 14029711.00
    ```

    기록된 값은 깊이 16→32에서 잔차 쪽이 약 2.3배, 비교 모델은 약 470배이다. 이 두 깊이의 값만으로 각각 선형·지수 성장 법칙을 입증하지는 않는다. 잔차 모델의 야코비안도 블록별 행렬의 곱이며, 항등 경로가 소실·폭발을 자동으로 막는 것은 아니다. 상류 그래디언트와 초기화·입력 등 비교 조건을 함께 기록한다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span>
$F(x) = 0$이면 블록이 항등 사상이 된다고들 말한다. `bn2`의 크기와 바이어스를 0으로 두어 $F(x) = 0$을 실제로 만든 뒤, 블록이 정말 항등인지 확인하라. 무엇이 나오며, 그 까닭은 무엇인가?

</div>

??? success "연습문제 5 풀이"
    `bn2.weight = 0`, `bn2.bias = 0`이면 `bn2`의 출력이 통째로 0이므로 $F(x) = 0$이다. 확인해 보자.

    ```python
    torch.manual_seed(0)
    blk = BasicBlock(64, 64).eval()
    with torch.no_grad():
        blk.bn2.weight.zero_()
        blk.bn2.bias.zero_()
    x = torch.randn(1, 64, 8, 8)
    with torch.no_grad():
        y = blk(x)
    print(torch.allclose(y, torch.relu(x), atol=1e-6))   # True
    print(torch.allclose(y, x, atol=1e-6))               # False
    print(round(x.min().item(), 4), y.min().item())      # -3.8394 0.0
    ```

    블록은 항등이 아니라 $\operatorname{ReLU}$가 된다. 더하기 뒤에 마지막 ReLU가 한 번 더 걸리기 때문이다.

    $$y = \operatorname{ReLU}(0 + x) = \operatorname{ReLU}(x)$$

    이 표본에서는 $x$의 48.75%가 음수라 절반 가까운 성분이 0으로 눌린다. 따라서 이 블록이 공짜로 얻는 것은 정확히는 "항등 사상"이 아니라 "음이 아닌 입력 위에서의 항등 사상"이다. 앞 블록의 출력은 그 자신의 마지막 ReLU를 거쳐 이미 음이 아니므로 ResNet 안에서는 이 구별이 대체로 무해하지만, 건너뛰기 경로가 참된 항등이 되게 하려면 마지막 ReLU를 치워야 한다. 그 설계가 [항등 사상](identity_mapping.md) 쪽의 사전 활성화 블록이다.

---

## 정리하며

BasicBlock은 두 컨볼루션 분기와 지름길을 더한 뒤 ReLU를 적용한다. 같은 채널 C에서 파라메터는 $18C^2+4C$이며, 차원이 바뀌면 투영 비용을 추가한다. F=0이고 지름길이 항등이어도 사후 ReLU 때문에 전체 블록은 일반적으로 ReLU(x)이다. 그래디언트 수치 예제의 결과를 모든 깊이에서의 안정성 보장으로 해석하지 않는다. [ResNet 구현](02_resnet_implementation.md)과 [항등 사상](identity_mapping.md)에서 전체 구조와 사전 활성화를 이어 본다.
