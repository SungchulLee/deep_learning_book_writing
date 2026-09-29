# 기본 잔차 블록

기본 잔차 블록 구현. 신경망의 잔차 연결(건너뛰기 연결)을 소개한다.

층을 더 쌓으면 적어도 나빠지지는 않아야 마땅하다. 얕은 신경망 뒤에 항등 사상 층을 덧붙이기만 해도 같은 함수를 나타낼 수 있기 때문이다. 그런데 He 등이 "Deep Residual Learning for Image Recognition"(2015)에서 관찰한 것은 그 반대였다. CIFAR-10에서 56층 평범한 합성곱 신경망은 20층짜리보다 *학습* 오차부터 더 높았다. 과적합이 아니라 최적화가 실패한 것이다.

잔차 블록은 층이 배워야 할 목표를 바꾸어 이 문제를 푼다. 블록이 나타내려는 사상을 $H(x)$라 하자. 층더미가 $H$를 곧바로 배우게 하는 대신, 입력과의 차이인 **잔차** $F(x) = H(x) - x$를 배우게 하고 입력을 되더해 준다.

$$H(x) = F(x) + x$$

항등 사상이 최적이라면, 평범한 층더미는 여러 비선형 층을 겹쳐 항등을 흉내내야 하지만 잔차 블록은 가중치를 0 쪽으로 밀어 $F(x) = 0$을 만드는 것으로 끝난다. 이 쪽은 그 $F$를 3×3 합성곱 두 개로 구현하고, 더하기가 성립하도록 모양을 맞추는 방법을 보이며, 이 구조가 기울기의 크기에 무엇을 하는지를 잰다. 여기서 정한 이름($F$, 건너뛰기 연결, 사영 지름길)을 이 절의 나머지 쪽들이 그대로 물려받는다.

## 1. 코드

```python
"""
기본 잔차 블록 구현
====================================
신경망의 잔차 연결(건너뛰기 연결) 소개.

핵심 개념:
H(x)를 배우는 대신 F(x) = H(x) - x를 배우므로 출력은 F(x) + x이다.
덕분에 기울기가 건너뛰기 연결을 타고 신경망을 곧바로 흐른다.
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
    1x1 합성곱 + BN으로 맞춘 사영이다.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        # 첫 합성곱 층 (stride를 여기서 준다: 공간 크기가 줄어드는 유일한 자리)
        # 바로 뒤에 BN이 오므로 bias=False로 둔다. BN이 평균을 빼 버려
        # 합성곱의 치우침 항은 출력에 아무 영향도 주지 못한다.
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 둘째 합성곱 층 (stride=1, 채널 수 그대로)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 건너뛰기 연결 (지름길). 빈 Sequential은 입력을 그대로 돌려주므로
        # 이 경우의 지름길은 매개변수가 0개인 항등 사상이다.
        self.shortcut = nn.Sequential()

        # 모양이 어긋나면 (공간 크기가 줄거나 채널 수가 바뀌면) 더할 수가 없다.
        # 1x1 합성곱에 같은 stride를 주어 모양을 맞춘다 (사영 지름길).
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

    BasicBlock과 층의 종류·순서·생성 순서가 모두 같고 더하기만 없다.
    따라서 같은 씨앗에서 만들면 두 블록의 가중치가 글자 그대로 같아진다.
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
# 기울기 흐름 재기
# ========================================================================


def input_grad_norm(block_cls, depth, seed):
    """
    block_cls를 depth개 쌓은 신경망에 입력을 넣고, 출력의 합을 역전파했을 때
    입력에 맺히는 기울기의 노름을 돌려준다. 씨앗을 먼저 심으므로 입력과
    가중치가 씨앗 하나로 결정된다.
    """
    torch.manual_seed(seed)
    x = torch.randn(1, 64, 32, 32, requires_grad=True)
    net = nn.Sequential(*[block_cls(64, 64) for _ in range(depth)])
    net(x).sum().backward()
    return x.grad.norm().item()


def demonstrate_gradient_flow(depths=(1, 4, 8, 16), seeds=(0, 1, 2)):
    """
    잔차 블록과 평범한 블록을 같은 가중치로 견주어 기울기의 흐름을 잰다
    """
    print("=" * 60)
    print("Gradient Flow Demonstration")
    print("=" * 60)

    # 같은 씨앗이면 두 블록의 가중치가 정말 같은지 먼저 확인한다.
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
# 모양과 매개변수 확인
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

    # 매개변수 개수 세기
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

    # 기울기의 흐름 보이기
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

여기서 $F$는 합성곱 두 개짜리 주 경로, $\mathcal{S}$는 지름길이다. 코드의 `F`는 `torch.nn.functional`의 관례적인 별칭이라 수식의 $F$와 글자가 겹치지만, 둘은 서로 다른 것이다. 둘째 BN 바로 뒤에 ReLU를 걸면 $F(x) \ge 0$이 되어 잔차가 입력을 깎아내릴 수 없게 되므로, 그 자리를 비워 두는 것이 이 구조의 핵심이다. 반대로 마지막 ReLU가 더하기 뒤에 있다는 것은 건너뛰기 경로 또한 블록마다 ReLU를 한 번씩 지난다는 뜻이기도 하다. 이 점을 문제 삼아 더하기를 활성화 밖으로 빼내는 사전 활성화 설계는 [항등 사상](identity_mapping.md) 쪽에서 다룬다.

**치우침을 두지 않는 이유.** 두 합성곱 모두 `bias=False`이다. 바로 뒤의 배치 정규화가 채널마다 평균을 빼므로 합성곱의 치우침 $b$는 출력에 전혀 나타나지 않는다. 즉 $\operatorname{BN}(Wx + b) = \operatorname{BN}(Wx)$이다. `bias=True`로 두면 `BasicBlock(64, 64)`의 매개변수가 128개(64 + 64) 늘지만 — 73,984의 0.17% — 함수는 한 치도 달라지지 않는다.

**지름길의 두 모습.** 출력의 1번 시험에서 `Shortcut modules: 0`, 2번 시험에서 `Shortcut modules: 2`가 찍힌다. 입출력 모양이 같으면 지름길은 매개변수가 0개인 항등 사상이고, 모양이 어긋나면 1×1 합성곱과 BN으로 이루어진 **사영 지름길**이 된다. 모양이 맞아떨어지는 까닭은 두 경로의 크기 공식이 같기 때문이다. 공간 크기는 다음을 따른다.

$$H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2p - k}{s} \right\rfloor + 1$$

주 경로의 첫 합성곱은 $k = 3$, $p = 1$이므로 $\lfloor (H_{\text{in}} - 1)/s \rfloor + 1$이고, 지름길의 1×1 합성곱은 $k = 1$, $p = 0$이므로 역시 $\lfloor (H_{\text{in}} - 1)/s \rfloor + 1$이다. 두 식이 글자 그대로 같으니 어떤 $H_{\text{in}}$과 어떤 $s$에서도 두 경로의 공간 크기가 저절로 맞는다. 출력이 보이는 $32 \to 16$이 그 한 경우다. `stride`를 주는 자리는 `conv1`과 지름길 둘뿐이며, `conv2`는 언제나 `stride=1`이다.

**매개변수의 값.** 3×3 합성곱은 치우침이 없으므로 $9 C_{\text{in}} C_{\text{out}}$개, 배치 정규화는 채널마다 크기와 치우침 하나씩이므로 $2C_{\text{out}}$개를 쓴다. 채널이 $C$로 유지되는 블록은 그래서 다음을 갖는다.

$$2(9C^2 + 2C) = 18C^2 + 4C$$

$C = 64$이면 $18 \cdot 4096 + 256 = 73{,}984$로, 출력에 찍힌 값과 그 내역(36,864 + 128을 두 번)이 정확히 이것이다. 채널이 바뀌는 `BasicBlock(64, 128, stride=2)`는 230,144개를 쓰고 그 가운데 1×1 사영이 8,448개(3.7%)다. 사영은 단계가 바뀌는 블록에서만 나타나므로 ResNet 전체에서 차지하는 몫은 이보다도 작다.

**기울기 표를 읽는 법.** 표 위의 `Same weights under the same seed: True`가 이 비교의 전제다. 두 클래스는 층을 같은 종류·같은 순서로 만들기 때문에 같은 씨앗을 심으면 난수 발생기에서 같은 수를 같은 순서로 받아 간다. 그래서 두 열의 차이는 오직 더하기 하나다. 블록이 하나일 때 잔차 쪽 기울기는 220 언저리, 평범한 쪽은 92 언저리로 2.4배인데, 이는 $\partial(F(x) + x)/\partial x = F'(x) + I$의 항등 항이 그대로 더해진 결과다. 정작 중요한 것은 깊이를 늘렸을 때다. 평범한 층더미는 16층에서 제 1층 값의 약 300배(26,672–28,533)로 불어나는 반면, 잔차 층더미는 8배(1,680–1,764)에 그친다.

**이 실험이 보이는 것과 보이지 않는 것.** 세 씨앗의 최소–최대 폭이 두 열 사이에서 겹치지 않으므로, 위의 차이는 씨앗 잡음보다 크다. 다만 여기서 평범한 층더미의 기울기는 사라지는 것이 아니라 커진다. 배치 정규화가 층마다 활성값의 크기를 다시 맞춰 주기 때문이며, 흔히 말하는 "기울기 소실"은 정규화가 없을 때의 이야기다. 이 표가 재는 것은 한쪽으로의 붕괴가 아니라 **척도의 안정성**이다. 건너뛰기 연결이 있으면 입력에 맺히는 기울기의 크기가 깊이에 거의 무관해진다. 또한 이것은 초기화 직후 한 번의 역전파일 뿐 학습을 돌린 것이 아니다. 같은 깊이에서 학습 곡선이 실제로 어떻게 갈리는지는 [학습 비교](03_training_comparison.md) 쪽이 잰다.

**다음 쪽과의 이름 차이.** 여기서는 지름길을 블록 스스로 만들어 `self.shortcut`에 담았다. 다음 쪽 [ResNet 구현](02_resnet_implementation.md)의 `BasicBlock`은 같은 것을 바깥에서 `downsample` 인자로 받는다. 한 단계 안에서 사영이 필요한 것은 첫 블록뿐이므로, `_make_layer`가 사영을 만들어 첫 블록에만 넘기고 나머지 블록은 `downsample=None`으로 둔다. 만드는 자리만 다를 뿐 계산하는 함수는 똑같다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
식 $18C^2 + 4C$로 `BasicBlock(128, 128)`의 매개변수 개수를 구하고, `sum(p.numel() for p in BasicBlock(128, 128).parameters())`로 확인하라.

</div>

??? success "연습문제 1 풀이"
    $18 \cdot 128^2 + 4 \cdot 128 = 18 \cdot 16384 + 512 = 294{,}912 + 512 = 295{,}424$이다.

    ```python
    b = BasicBlock(128, 128)
    print(sum(p.numel() for p in b.parameters()))   # 295424
    ```

    내역은 `conv1` $9 \cdot 128 \cdot 128 = 147{,}456$, `bn1` 256, `conv2` 147,456, `bn2` 256이다. 채널을 두 배로 늘리면 매개변수는 네 배가 된다($C=64$의 73,984와 견주어 보라). 합성곱의 매개변수가 $C^2$에 비례하기 때문이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
`conv1`과 `conv2`를 `bias=True`로 바꾸면 매개변수가 몇 개 늘어나는가? 그런데도 블록이 계산하는 함수는 왜 달라지지 않는가?

</div>

??? success "연습문제 2 풀이"
    합성곱마다 출력 채널 수만큼 치우침이 붙으므로 $64 + 64 = 128$개가 늘어 74,112개가 된다. 73,984의 0.17%다.

    함수가 달라지지 않는 까닭은 바로 뒤에 배치 정규화가 오기 때문이다. 치우침 $b$를 더하면 채널의 모든 값이 $b$만큼 평행이동하는데, 배치 정규화는 채널의 평균 $\mu$를 빼므로 그 평행이동이 평균에도 똑같이 실려 지워진다.

    $$\frac{(Wx + b) - (\mu_{Wx} + b)}{\sigma} = \frac{Wx - \mu_{Wx}}{\sigma}$$

    치우침 노릇은 배치 정규화 자신의 $\beta$ 매개변수가 대신한다. 그래서 BN이 뒤따르는 합성곱에는 언제나 `bias=False`를 준다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
`BasicBlock`의 `__init__`에서 `if stride != 1 or in_channels != out_channels:` 가지를 지워 사영 지름길이 만들어지지 않게 한 뒤, `BasicBlock(64, 128, stride=2)`에 $(2, 64, 32, 32)$ 텐서를 넣어라. 어떤 오류가 나며 그 수는 무엇을 가리키는가?

</div>

??? success "연습문제 3 풀이"
    지름길이 빈 `nn.Sequential()`로 남아 입력을 그대로 돌려주므로, 더하기에서 주 경로의 $(2, 128, 16, 16)$과 입력의 $(2, 64, 32, 32)$를 맞붙이게 된다. 실행하면 다음이 뜬다.

    ```
    RuntimeError: The size of tensor a (16) must match the size of tensor b (32)
    at non-singleton dimension 3
    ```

    16은 `stride=2`를 지난 주 경로의 너비, 32는 손대지 않은 입력의 너비다. PyTorch가 마지막 축부터 맞춰 보므로 너비에서 먼저 걸리지만, 채널 축(128 대 64)도 똑같이 어긋나 있다. 사영 지름길은 이 두 어긋남을 1×1 합성곱 하나로 한꺼번에 고친다. 출력 채널 수로 채널을 맞추고, 같은 `stride`로 공간 크기를 맞춘다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
`demonstrate_gradient_flow(depths=(16, 32))`로 깊이 32를 함께 재어라. 깊이 16에서 32로 갈 때 두 층더미의 기울기는 각각 몇 배가 되는가?

</div>

??? success "연습문제 4 풀이"
    씨앗 0, 1, 2에서 얻는 표의 두 줄은 다음과 같다.

    ```
     depth |  residual: min - max  |     plain: min - max
        16 |   1679.75 -   1764.44 |    26671.74 -    28533.47
        32 |   3840.75 -   4058.23 | 12094596.00 - 14029711.00
    ```

    잔차 쪽은 약 1,720에서 약 3,950으로 2.3배가 되었고, 평범한 쪽은 약 27,600에서 약 1,300만으로 470배가 되었다. 깊이를 두 배로 늘렸을 뿐인데 평범한 층더미의 기울기는 또 한 번 자릿수 두 개 넘게 뛴다. 층마다 거의 일정한 배수가 곱해져 깊이에 대해 지수로 자라기 때문이다. 잔차 층더미에서는 그 곱셈 사슬에 항등 항 $I$가 더해져 있어 자람이 거의 선형에 머문다. 깊이 32에서 두 값의 비는 이미 3,000배가 넘는다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span>
$F(x) = 0$이면 블록이 항등 사상이 된다고들 말한다. `bn2`의 크기와 치우침을 0으로 두어 $F(x) = 0$을 실제로 만든 뒤, 블록이 정말 항등인지 확인하라. 무엇이 나오며, 그 까닭은 무엇인가?

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

**다룬 것** — 기본 잔차 블록

잔차 블록은 $H(x)$ 대신 $F(x) = H(x) - x$를 배우고 입력을 되더한다. `BasicBlock`은 그 $F$를 3×3 합성곱 두 개(Conv → BN → ReLU → Conv → BN)로 두고 더하기를 마지막 ReLU 앞에 놓는다. 모양이 맞으면 지름길은 매개변수가 없는 항등이고, `stride`나 채널 수가 바뀌면 1×1 합성곱과 BN으로 된 사영이 대신한다.

수로 남는 것은 셋이다. 채널이 $C$인 블록의 매개변수는 $18C^2 + 4C$개($C = 64$에서 73,984개)이고, `BasicBlock(64, 128, stride=2)`에서 사영이 차지하는 몫은 3.7%이며, 같은 가중치로 견준 입력 기울기는 깊이 16에서 잔차 쪽 1,680–1,764 대 평범한 쪽 26,672–28,533이다. 건너뛰기 연결이 하는 일은 기울기를 키우는 것이 아니라 깊이에 대해 그 크기를 붙들어 두는 것이다.

이 블록을 단계별로 쌓아 ResNet-18부터 ResNet-152까지를 만드는 일은 [ResNet 구현](02_resnet_implementation.md)에서, 마지막 ReLU를 치워 건너뛰기 경로를 참된 항등으로 만드는 변형은 [항등 사상](identity_mapping.md)에서 이어 본다.
