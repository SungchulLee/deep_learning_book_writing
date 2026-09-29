# 전치 합성곱

**전치 합성곱**(**분수 보폭 합성곱**)은 보통 합성곱을 입력에 대해 미분한 연산이다. 표준 합성곱이 대체로 공간 차원을 줄이는 데 반해 전치 합성곱은 이를 **늘리므로**, 신경망에서 학습 가능한 상향 표본화의 대표적인 방법이 된다. 흔히 "역합성곱"이라 부르지만 그 이름은 틀렸다 — 이 연산은 합성곱을 **되돌리지 않는다**.

전치 합성곱은 다음에서 꼭 필요한 부품이다.

- 의미 분할을 위한 **인코더-디코더 구조** (U-Net, SegNet)
- 이미지 합성을 위한 **생성 모델** (GAN, VAE)
- 이미지를 키우는 **초해상도 신경망**
- 여러 규모의 물체 탐지를 위한 **특징 피라미드 신경망**

> **용어에 대한 참고**: "역합성곱"(deconvolution)은 신호 처리에서 합성곱을 실제로 되돌리는 별개의 연산을 가리킨다. 전치 합성곱은 그것이 아니다. 수학적으로 정확한 말은 "전치 합성곱"이며, 합성곱의 퇴플리츠 행렬을 **전치**한 것을 곱한다는 뜻이다. 전치는 역행렬이 아니므로 값은 돌아오지 않는다 — 돌아오는 것은 **크기**뿐이다. 1절 끝에서 이를 직접 재어 본다.

---

## 1. 수학적 바탕

### 행렬 곱으로 본 합성곱

전치 합성곱을 이해하려면 먼저 표준 합성곱을 행렬 곱으로 나타내야 한다. 1차원 입력 $\mathbf{x} \in \mathbb{R}^5$과 핵 $\mathbf{k} = [k_0, k_1, k_2]$에 대해 유효 합성곱 $\mathbf{y} = \mathbf{C}\mathbf{x}$은 다음을 쓴다.

$$\mathbf{C} = \begin{bmatrix}
k_0 & k_1 & k_2 & 0 & 0 \\
0 & k_0 & k_1 & k_2 & 0 \\
0 & 0 & k_0 & k_1 & k_2
\end{bmatrix}$$

이는 $\mathbb{R}^5 \to \mathbb{R}^3$으로 보낸다(하향 표본화).

### 전치한 연산

**전치 합성곱**은 $\mathbf{C}^\top$을 쓴다.

$$\mathbf{C}^\top = \begin{bmatrix}
k_0 & 0 & 0 \\
k_1 & k_0 & 0 \\
k_2 & k_1 & k_0 \\
0 & k_2 & k_1 \\
0 & 0 & k_2
\end{bmatrix}$$

이는 $\mathbb{R}^3 \to \mathbb{R}^5$으로 보낸다(상향 표본화). 핵의 가중치는 같지만 이어지는 방식이 뒤집힌다.

$\mathbf{C}$는 $3 \times 5$ 행렬이라 애초에 역행렬을 가질 수 없고, $\mathbf{C}^\top \mathbf{C}$도 계수가 3뿐인 $5 \times 5$ 행렬이라 항등행렬이 아니다. 그러니 $\mathbf{C}^\top$을 곱하는 것은 $\mathbf{C}$를 되돌리는 일이 아니다.

### 핵심 착상: 합성곱의 기울기

입력에 대한 합성곱의 역전파가 바로 전치 합성곱이다.

$$\frac{\partial L}{\partial \mathbf{x}} = \mathbf{C}^\top \frac{\partial L}{\partial \mathbf{y}}$$

그래서 역전파에서 전치 합성곱이 자연스럽게 나타난다. 아래 코드는 이것을 확인한 뒤, 이어서 같은 핵으로 되돌려 보아 **크기는 돌아와도 값은 돌아오지 않는다**는 것까지 잰다.

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)

# 순전파: conv2d
x = torch.randn(1, 3, 8, 8, requires_grad=True)
w = torch.randn(16, 3, 3, 3, requires_grad=True)
y = F.conv2d(x, w, padding=1)

# 역전파: conv_transpose2d와 같다
grad_output = torch.randn_like(y)
y.backward(grad_output)

# 손수 확인
grad_input_manual = F.conv_transpose2d(grad_output, w, padding=1)
print(f"Gradient match: {torch.allclose(x.grad, grad_input_manual, atol=1e-5)}")

# 전치는 역행렬이 아니다: 같은 핵으로 되돌려도 값은 돌아오지 않는다
torch.manual_seed(0)
a = torch.randn(1, 1, 8, 8)
k = torch.randn(1, 1, 3, 3)
b = F.conv2d(a, k, padding=1)
a_back = F.conv_transpose2d(b, k, padding=1)
print(f"Shape restored:  {tuple(a.shape) == tuple(a_back.shape)}")
print(f"Values restored: {torch.allclose(a, a_back, atol=1e-4)}")
print(f"Relative error:  {(a_back - a).norm() / a.norm():.4f}")
```

**출력:**

```
Gradient match: True
Shape restored:  True
Values restored: False
Relative error:  6.6400
```

상대 오차가 6.64다 — 되돌린 값이 원래 값보다 여섯 배 넘게 어긋났다는 뜻이다. 8×8이 8×8로 돌아온 것은 크기가 맞아떨어졌을 뿐, 값은 전혀 복원되지 않았다.

---

## 2. 전치 합성곱이 움직이는 방식

### 상향 표본화 장치

전치 합성곱은 다음 세 걸음으로 이해할 수 있다. 핵 크기를 $k$, 보폭을 $s$, 덧대기를 $p$라 하자.

1. (보폭이 1보다 크면) 입력 원소 사이에 **0을 $s-1$개씩 끼워 넣는다** — $i \times i$ 입력이 $\big(s(i-1)+1\big) \times \big(s(i-1)+1\big)$이 된다
2. 각 변에 **0을 $k-1-p$개 덧댄다**
3. 뒤집은 핵으로 **보폭 1 표준 합성곱**을 적용한다

2번의 덧대기 양에 주의하라. 덧대기가 $p$가 아니라 $k-1-p$이므로, `padding`을 **키우면 출력이 작아진다** — 보통 합성곱과 정반대다. 전치 합성곱의 `padding` 인자는 "0을 더 넣어라"가 아니라 "가장자리를 그만큼 잘라내라"에 가깝다.

3×3 핵을 쓰는 보폭 2, 덧대기 1 전치 합성곱에서는 다음과 같다. 덧대기 양은 $k-1-p = 3-1-1 = 1$이므로 각 변에 한 줄씩이다.

```
Input (2×2):       Insert zeros (3×3):      Pad (5×5):           Convolve with 3×3
┌───┬───┐          ┌───┬───┬───┐           ┌───┬───┬───┬───┬───┐
│ a │ b │    →      │ a │ 0 │ b │     →     │ 0 │ 0 │ 0 │ 0 │ 0 │  →  3×3 output
├───┼───┤          ├───┼───┼───┤           ├───┼───┼───┼───┼───┤
│ c │ d │          │ 0 │ 0 │ 0 │           │ 0 │ a │ 0 │ b │ 0 │
└───┴───┘          ├───┼───┼───┤           ├───┼───┼───┼───┼───┤
                   │ c │ 0 │ d │           │ 0 │ 0 │ 0 │ 0 │ 0 │
                   └───┴───┴───┘           ├───┼───┼───┼───┼───┤
                                           │ 0 │ c │ 0 │ d │ 0 │
                                           ├───┼───┼───┼───┼───┤
                                           │ 0 │ 0 │ 0 │ 0 │ 0 │
                                           └───┴───┴───┴───┴───┘
```

5×5를 3×3 핵으로 보폭 1 유효 합성곱하면 $5-3+1 = 3$이므로 출력은 **3×3**이다. 2×2를 정확히 두 배인 4×4로 키우려면 뒤에 나올 `output_padding=1`을 더해 오른쪽과 아래에 한 줄씩 더 붙여야 한다.

세 걸음이 정말 `conv_transpose2d`와 같은 값을 내는지 손으로 만들어 견주어 보자.

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)


def transposed_by_hand(x, w, stride, padding):
    """0 끼워 넣기 → 덧대기 → 보통 합성곱, 세 걸음으로 손수 만든 전치 합성곱."""
    i, k = x.shape[-1], w.shape[-1]

    # 1. 원소 사이에 0을 stride - 1개씩 끼워 넣는다
    n = stride * (i - 1) + 1
    z = torch.zeros(x.shape[0], x.shape[1], n, n)
    z[:, :, ::stride, ::stride] = x

    # 2. 각 변에 k - 1 - padding개의 0을 덧댄다 (padding이 클수록 적게 덧댄다)
    pad = k - 1 - padding
    z = F.pad(z, (pad, pad, pad, pad))

    # 3. 뒤집은 핵으로 보폭 1 보통 합성곱
    return F.conv2d(z, torch.flip(w, [-1, -2]).transpose(0, 1)), z.shape[-1]


x = torch.randn(1, 2, 4, 4)
w = torch.randn(2, 3, 3, 3)

for stride, padding in [(1, 0), (2, 0), (2, 1), (3, 1)]:
    ref = F.conv_transpose2d(x, w, stride=stride, padding=padding)
    got, padded = transposed_by_hand(x, w, stride, padding)
    print(f"s={stride} p={padding}: 0 끼운 뒤 덧댄 크기 {padded}×{padded}"
          f" → 출력 {got.shape[-1]}×{got.shape[-1]}"
          f" (conv_transpose2d와 같은가: {torch.allclose(ref, got, atol=1e-5)})")

# 그림의 2×2 입력: p=1이면 3×3, output_padding=1을 더해야 4×4가 된다
x22 = torch.randn(1, 1, 2, 2)
w33 = torch.randn(1, 1, 3, 3)
for op in (0, 1):
    out = F.conv_transpose2d(x22, w33, stride=2, padding=1, output_padding=op)
    print(f"2×2 입력, k=3 s=2 p=1 output_padding={op} → {out.shape[-1]}×{out.shape[-1]}")
```

**출력:**

```
s=1 p=0: 0 끼운 뒤 덧댄 크기 8×8 → 출력 6×6 (conv_transpose2d와 같은가: True)
s=2 p=0: 0 끼운 뒤 덧댄 크기 11×11 → 출력 9×9 (conv_transpose2d와 같은가: True)
s=2 p=1: 0 끼운 뒤 덧댄 크기 9×9 → 출력 7×7 (conv_transpose2d와 같은가: True)
s=3 p=1: 0 끼운 뒤 덧댄 크기 12×12 → 출력 10×10 (conv_transpose2d와 같은가: True)
2×2 입력, k=3 s=2 p=1 output_padding=0 → 3×3
2×2 입력, k=3 s=2 p=1 output_padding=1 → 4×4
```

$p$를 0에서 1로 키우자 덧댄 크기가 11에서 9로 줄고 출력도 9에서 7로 줄었다. 앞에서 말한 "덧대기를 키우면 출력이 작아진다"가 이것이다.

### 출력 크기 공식

전치 합성곱에 대해 다음과 같다.

$$H_{out} = (H_{in} - 1) \times s - 2p + d(K - 1) + p_{out} + 1$$

여기서 각 기호는 다음과 같다.

- $H_{in}$: 입력의 높이
- $s$: 보폭
- $p$: 덧대기
- $d$: 팽창률
- $K$: 핵 크기
- $p_{out}$: 출력 덧대기 (모호함을 없앤다)

3×3 핵으로 보폭 2 상향 표본화를 하는 흔한 경우($s = 2$, $p = 1$, $d = 1$, $K = 3$)에 항을 하나씩 넣으면 다음과 같다.

$$H_{out} = (H_{in} - 1) \times 2 - 2 \times 1 + 1 \times (3 - 1) + p_{out} + 1 = 2 H_{in} - 1 + p_{out}$$

즉 이 설정만으로는 **정확히 두 배가 되지 않는다**. $p_{out} = 0$이면 $2H_{in} - 1$이고, $p_{out} = 1$을 주어야 비로소 $2H_{in}$이다. 다음 절에서 16×16이 각각 31×31과 32×32로 가는 것이 이 두 경우다.

---

## 3. PyTorch 구현

### 기본 사용법

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

# 보통 합성곱 (하향 표본화)
conv = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)

# 전치 합성곱 (상향 표본화)
conv_transpose = nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2,
                                    padding=1, output_padding=1)

x = torch.randn(1, 64, 32, 32)

# 하향 표본화
y = conv(x)
print(f"Conv: {x.shape} → {y.shape}")  # [1, 64, 32, 32] → [1, 32, 16, 16]

# 상향 표본화
z = conv_transpose(y)
print(f"ConvT: {y.shape} → {z.shape}")  # [1, 32, 16, 16] → [1, 64, 32, 32]
```

**출력:**

```
Conv: torch.Size([1, 64, 32, 32]) → torch.Size([1, 32, 16, 16])
ConvT: torch.Size([1, 32, 16, 16]) → torch.Size([1, 64, 32, 32])
```

### `output_padding` 매개변수

보폭이 1보다 크면 보통 합성곱에서 여러 입력 크기가 같은 출력 크기를 낼 수 있다. 이를테면 $K=3$, $s=2$, $p=1$에서는 31×31 입력과 32×32 입력이 모두 16×16 출력을 낸다. 거꾸로 갈 때 둘 중 어느 것을 돌려줄지 정해 주는 것이 `output_padding`이다.

```python
# output_padding이 없으면 2 × 16 - 1 = 31×31이 된다
conv_t_no_op = nn.ConvTranspose2d(32, 64, 3, stride=2, padding=1)
# output_padding=1이면 2 × 16 = 32×32가 보장된다
conv_t_with_op = nn.ConvTranspose2d(32, 64, 3, stride=2, padding=1, output_padding=1)

y = torch.randn(1, 32, 16, 16)
print(f"Without output_padding: {conv_t_no_op(y).shape}")     # [1, 64, 31, 31]
print(f"With output_padding=1: {conv_t_with_op(y).shape}")    # [1, 64, 32, 32]
```

**출력:**

```
Without output_padding: torch.Size([1, 64, 31, 31])
With output_padding=1: torch.Size([1, 64, 32, 32])
```

`output_padding`은 오른쪽과 아래에 줄을 더 붙여 크기만 맞출 뿐, 그 자리에 실제로 값을 계산해 넣지는 않는다. 그래서 크기를 맞추는 데는 쓰되 정보를 늘리는 장치로 여겨서는 안 된다.

### 매개변수의 수

전치 합성곱은 입력 채널과 출력 채널을 맞바꾼 보통 합성곱과 매개변수 수가 같다.

```python
# 보통: 채널 64개 → 32개
conv = nn.Conv2d(64, 32, 3, bias=False)
print(f"Conv2d params: {sum(p.numel() for p in conv.parameters()):,}")
# 32 × 64 × 3 × 3 = 18,432

# 전치: 채널 32개 → 64개
conv_t = nn.ConvTranspose2d(32, 64, 3, bias=False)
print(f"ConvTranspose2d params: {sum(p.numel() for p in conv_t.parameters()):,}")
# 32 × 64 × 3 × 3 = 18,432 (같다!)
```

**출력:**

```
Conv2d params: 18,432
ConvTranspose2d params: 18,432
```

---

## 4. 바둑판 무늬 흠 문제

### 무엇이 문제인가

보폭이 1보다 큰 전치 합성곱은 **바둑판 무늬 흠**을 만드는 것으로 악명 높다. 까닭은 출력의 자리마다 입력에서 값을 받는 횟수가 다르기 때문이다. 이 횟수는 재기 쉽다 — 핵과 입력을 모두 1로 두면 출력의 값이 곧 받은 횟수다.

```python
import torch
import torch.nn.functional as F


def overlap_counts(kernel_size, stride, padding=0, in_size=3):
    """출력의 각 자리가 입력에서 몇 번 값을 받는지 센다.

    핵을 모두 1로, 입력도 모두 1로 두면 출력의 값이 곧 받은 횟수다.
    """
    w = torch.ones(1, 1, kernel_size, kernel_size)
    x = torch.ones(1, 1, in_size, in_size)
    return F.conv_transpose2d(x, w, stride=stride,
                              padding=padding).squeeze().int()


for k, s, p in [(3, 2, 0), (4, 2, 0), (2, 2, 0)]:
    c = overlap_counts(k, s, p)
    print(f"kernel={k}, stride={s}  (K % s = {k % s})  →  출력 {tuple(c.shape)}")
    print(c.numpy())
    print()
```

**출력:**

```
kernel=3, stride=2  (K % s = 1)  →  출력 (7, 7)
[[1 1 2 1 2 1 1]
 [1 1 2 1 2 1 1]
 [2 2 4 2 4 2 2]
 [1 1 2 1 2 1 1]
 [2 2 4 2 4 2 2]
 [1 1 2 1 2 1 1]
 [1 1 2 1 2 1 1]]

kernel=4, stride=2  (K % s = 0)  →  출력 (8, 8)
[[1 1 2 2 2 2 1 1]
 [1 1 2 2 2 2 1 1]
 [2 2 4 4 4 4 2 2]
 [2 2 4 4 4 4 2 2]
 [2 2 4 4 4 4 2 2]
 [2 2 4 4 4 4 2 2]
 [1 1 2 2 2 2 1 1]
 [1 1 2 2 2 2 1 1]]

kernel=2, stride=2  (K % s = 0)  →  출력 (6, 6)
[[1 1 1 1 1 1]
 [1 1 1 1 1 1]
 [1 1 1 1 1 1]
 [1 1 1 1 1 1]
 [1 1 1 1 1 1]
 [1 1 1 1 1 1]]
```

$K=3$, $s=2$에서는 안쪽이 2와 4를 한 칸 걸러 오간다 — 이것이 바둑판 무늬다. $K=4$, $s=2$에서는 안쪽이 모두 4로 고르고, $K=2$, $s=2$에서는 겹침이 아예 없어 모든 자리가 1이다. 앞의 두 경우는 가장자리 두 줄이 값을 덜 받는데, 이는 어느 핵 크기에서나 생기는 테두리 효과일 뿐 바둑판 무늬와는 다른 이야기다($K=2$에서는 그마저 없다). 무늬인지 테두리인지는 **안쪽**을 보면 갈린다.

바둑판 무늬가 생기는 조건은 $K$가 $s$로 나누어떨어지지 않는 것이다. 위 세 경우의 `K % s`가 그것을 그대로 보여 준다.

### 해법 1: 핵 크기를 보폭의 배수로

보폭으로 딱 나누어떨어지는 핵 크기를 쓴다. 위에서 센 횟수가 그대로 근거다.

```python
import torch.nn as nn

# 나쁨: stride=2, kernel=3 → 안쪽이 2와 4를 오간다 (3 % 2 = 1)
bad = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)

# 나음: stride=2, kernel=4 → 안쪽이 모두 4 (4 % 2 = 0)
better = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)

# 이것도 좋음: stride=2, kernel=2 → 겹침 없음 (2 % 2 = 0)
good = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
```

다만 이 해법이 없애는 것은 **기하학적인** 원인뿐이다. 겹침 횟수가 고르더라도 학습된 가중치가 자리마다 다르면 약한 무늬가 남을 수 있다.

### 해법 2: 크기 조정 뒤 합성곱 (권장)

상향 표본화와 합성곱을 떼어 놓아 전치 합성곱을 아예 쓰지 않는 더 깔끔한 방법이다. 보간은 모든 출력 자리를 같은 방식으로 채우므로 겹침이 고르지 않을 여지가 없다.

```python
import torch
import torch.nn as nn

torch.manual_seed(0)


class UpsampleConv(nn.Module):
    """
    보간 뒤 합성곱으로 상향 표본화한다.
    전치 합성곱에서 오는 바둑판 무늬 흠을 피한다.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 scale_factor=2, mode='bilinear'):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode,
                                    align_corners=False if mode != 'nearest' else None)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=kernel_size // 2)

    def forward(self, x):
        x = self.upsample(x)
        return self.conv(x)


# 비교
x = torch.randn(1, 64, 16, 16)

# 전치 합성곱
conv_t = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)

# 크기 조정 뒤 합성곱 (흠 없음)
resize_conv = UpsampleConv(64, 32, scale_factor=2)

print(f"ConvTranspose: {conv_t(x).shape}")     # [1, 32, 32, 32]
print(f"Resize+Conv:   {resize_conv(x).shape}")  # [1, 32, 32, 32]
```

**출력:**

```
ConvTranspose: torch.Size([1, 32, 32, 32])
Resize+Conv:   torch.Size([1, 32, 32, 32])
```

### 해법 3: 부화소 합성곱 (PixelShuffle)

초해상도에서 쓰는 방법으로, 채널을 공간 차원으로 다시 늘어놓는다.

```python
class SubPixelUpsample(nn.Module):
    """
    효율적인 상향 표본화를 위한 부화소 합성곱(PixelShuffle).

    보통 합성곱으로 채널을 r²개 만든 뒤 다시 늘어놓아
    공간 해상도를 r배로 키운다.
    """

    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * upscale_factor**2,
                              kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        return self.shuffle(x)


# 예
sub_pixel = SubPixelUpsample(64, 32, upscale_factor=2)
x = torch.randn(1, 64, 16, 16)
out = sub_pixel(x)
print(f"Sub-pixel: {x.shape} → {out.shape}")  # [1, 64, 16, 16] → [1, 32, 32, 32]
```

**출력:**

```
Sub-pixel: torch.Size([1, 64, 16, 16]) → torch.Size([1, 32, 32, 32])
```

---

## 5. 인코더-디코더 구조

앞 절의 상향 표본화 방법들이 실제로 쓰이는 자리가 디코더다. 인코더가 줄여 놓은 해상도를 원래대로 되돌리는 것이 디코더의 일이며, 그 되돌리기가 전치 합성곱이다.

### 단순한 오토인코더

```python
import torch
import torch.nn as nn

torch.manual_seed(0)


class ConvAutoencoder(nn.Module):
    """
    복호에 전치 합성곱을 쓰는 합성곱 오토인코더.
    """

    def __init__(self):
        super().__init__()

        # 인코더: 차츰 하향 표본화
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),    # 224 → 112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),   # 112 → 56
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 56 → 28
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # 디코더: 차츰 상향 표본화 (K=4, s=2 이므로 겹침이 고르다)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 28 → 56
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 56 → 112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),     # 112 → 224
            nn.Sigmoid(),  # 출력은 [0, 1]
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


model = ConvAutoencoder()
x = torch.randn(2, 3, 224, 224)
reconstruction = model(x)
print(f"Input: {x.shape}")
print(f"Reconstruction: {reconstruction.shape}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Input: torch.Size([2, 3, 224, 224])
Reconstruction: torch.Size([2, 3, 224, 224])
Parameters: 1,030,723
```

디코더의 세 층은 모두 $K=4$, $s=2$, $p=1$이므로 $(H-1) \times 2 - 2 + 3 + 0 + 1 = 2H$, 곧 정확히 두 배다. $K=3$을 썼다면 `output_padding=1`이 따로 필요했을 것이다.

### U-Net 방식 (건너뛰기 연결이 있는)

오토인코더는 병목을 지나면서 위치 정보를 잃는다. U-Net은 인코더의 특징을 같은 해상도의 디코더로 곧장 건네주어 이를 메운다.

```python
class UNetDecoder(nn.Module):
    """
    인코더에서 오는 건너뛰기 연결이 있는 U-Net 디코더 블록.

    특징 맵을 상향 표본화하여 짝이 되는 인코더 특징과 이어 붙인 뒤
    합성곱을 적용한다.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        # 상향 표본화 (전치 합성곱 또는 크기 조정 뒤 합성곱)
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                     kernel_size=4, stride=2, padding=1)

        # 이어 붙인 뒤: (in_channels//2 + skip_channels) → out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)

        # 공간 크기가 어긋나는 경우 처리 (차원이 홀수일 때 생길 수 있다).
        # 채널 수는 일부러 다를 수 있으므로 shape 전체가 아니라 shape[2:]만 견준다.
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:])

        x = torch.cat([x, skip], dim=1)  # 채널을 따라 이어 붙이기
        return self.conv(x)


torch.manual_seed(0)
block = UNetDecoder(in_channels=256, skip_channels=128, out_channels=128)
deep = torch.randn(1, 256, 28, 28)     # 더 깊은 층에서 올라온 특징
skip = torch.randn(1, 128, 56, 56)     # 짝이 되는 인코더 특징
out = block(deep, skip)

upsampled = block.up(deep)
print(f"up:     {tuple(deep.shape)} → {tuple(upsampled.shape)}")
print(f"concat: {upsampled.shape[1]} + {skip.shape[1]} = "
      f"{upsampled.shape[1] + skip.shape[1]} 채널")
print(f"out:    {tuple(out.shape)}")
print(f"params: {sum(p.numel() for p in block.parameters()):,}")
```

**출력:**

```
up:     (1, 256, 28, 28) → (1, 128, 56, 56)
concat: 128 + 128 = 256 채널
out:    (1, 128, 56, 56)
params: 967,552
```

28×28이 56×56으로 올라가 같은 해상도의 건너뛰기 특징과 만나고, 채널이 128 + 128 = 256으로 합쳐졌다가 합성곱을 지나 128로 줄어든다.

위 `forward`에서 크기를 견줄 때 `x.shape`이 아니라 `x.shape[2:]`를 쓰는 것이 중요하다. 올려 보낸 쪽의 채널 수(`in_channels // 2`)와 건너뛰기 쪽의 채널 수(`skip_channels`)는 일반적으로 다르므로, `shape` 전체를 견주면 공간 크기가 멀쩡한데도 늘 다르다고 판정되어 쓸데없는 보간이 매번 일어난다.

---

## 6. 상향 표본화 방법 견주기

| 방법 | 학습 가능 | 흠 | 매개변수 | 속도 |
|--------|-----------|-----------|------------|-------|
| **ConvTranspose2d** | 그렇다 | 바둑판 무늬 (K % s ≠ 0일 때) | $C_{in} \times C_{out} \times K^2$ | 빠름 |
| **쌍선형 보간 뒤 합성곱** | 일부 | 깨끗함 | $C_{in} \times C_{out} \times K^2$ | 보통 |
| **최근접 보간 뒤 합성곱** | 일부 | 네모진 무늬 | $C_{in} \times C_{out} \times K^2$ | 보통 |
| **PixelShuffle** | 그렇다 | 깨끗함 | $C_{in} \times C_{out} \times r^2 \times K^2$ | 빠름 |
| **쌍선형 보간만** | 아니다 | 매끈함 (흐릿함) | 0 | 매우 빠름 |

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

# 모든 방법: 채널 64개, 16×16 → 32×32

x = torch.randn(1, 64, 16, 16)

methods = {
    'ConvTranspose (K=4, s=2)': nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
    'ConvTranspose (K=2, s=2)': nn.ConvTranspose2d(64, 32, 2, stride=2),
    'PixelShuffle': nn.Sequential(
        nn.Conv2d(64, 32 * 4, 3, padding=1),
        nn.PixelShuffle(2)
    ),
    'Bilinear + Conv': nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.Conv2d(64, 32, 3, padding=1)
    ),
}

for name, module in methods.items():
    out = module(x)
    params = sum(p.numel() for p in module.parameters())
    print(f"{name:<30}: {x.shape} → {out.shape}, params: {params:,}")
```

**출력:**

```
ConvTranspose (K=4, s=2)      : torch.Size([1, 64, 16, 16]) → torch.Size([1, 32, 32, 32]), params: 32,800
ConvTranspose (K=2, s=2)      : torch.Size([1, 64, 16, 16]) → torch.Size([1, 32, 32, 32]), params: 8,224
PixelShuffle                  : torch.Size([1, 64, 16, 16]) → torch.Size([1, 32, 32, 32]), params: 73,856
Bilinear + Conv               : torch.Size([1, 64, 16, 16]) → torch.Size([1, 32, 32, 32]), params: 18,464
```

네 방법이 모두 16×16을 32×32로 보내지만 매개변수는 8,224에서 73,856까지 아홉 배 가까이 차이 난다. PixelShuffle이 가장 무거운 것은 $r^2 = 4$배의 채널을 먼저 만들어야 하기 때문이다.

---

## 7. 1차원 전치 합성곱

이미지만이 아니라 소리나 시계열처럼 축이 하나인 자료에도 같은 연산이 그대로 쓰인다.

```python
torch.manual_seed(0)

# 시간 방향 상향 표본화를 위한 1차원 전치 합성곱
conv_t1d = nn.ConvTranspose1d(
    in_channels=64,
    out_channels=32,
    kernel_size=4,
    stride=2,
    padding=1
)

x = torch.randn(1, 64, 50)  # 시각 50개
out = conv_t1d(x)
print(f"1D ConvTranspose: {x.shape} → {out.shape}")  # [1, 32, 100]
```

**출력:**

```
1D ConvTranspose: torch.Size([1, 64, 50]) → torch.Size([1, 32, 100])
```

공식은 그대로다: $(50 - 1) \times 2 - 2 \times 1 + 1 \times (4-1) + 0 + 1 = 100$.

---

## 8. 핵심 정리

1. **전치 합성곱은 합성곱 행렬의 역행렬이 아니라 전치**이다. 합성곱을 되돌리지 않는다 — 1절에서 8×8을 되돌려 보았을 때 크기는 8×8로 맞았지만 상대 오차는 6.64였다
2. 역전파에서 입력에 대한 보통 합성곱의 **기울기로 자연스럽게 나타난다**
3. 전치 합성곱의 `padding`은 **출력을 줄인다**. 0을 끼워 넣은 뒤 각 변에 덧대는 양이 $p$가 아니라 $k-1-p$이기 때문이다
4. **바둑판 무늬 흠**은 핵 크기가 보폭으로 나누어떨어지지 않아 겹침이 고르지 않을 때 생긴다. $K=3$, $s=2$에서 겹침 횟수가 안쪽에서 2와 4를 오가는 것을 4절에서 셌다. $K = 2s$이나 $K = s$을 쓰면 피할 수 있다
5. 흠 없는 상향 표본화에는 **크기 조정 뒤 합성곱**(쌍선형 보간 뒤 보통 합성곱)이 나을 때가 많다
6. **PixelShuffle**(부화소 합성곱)은 채널을 다시 늘어놓아 효율적이고 흠 없는 상향 표본화를 준다
7. **output_padding**은 보통 합성곱에서 여러 입력 크기가 같은 출력 크기로 갈 때 생기는 모호함을 없앤다. $K=3$, $s=2$, $p=1$에서 출력은 $2H_{in} - 1 + p_{out}$이므로, 정확히 두 배를 얻으려면 $p_{out} = 1$이 필요하다

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
전치 합성곱의 출력 크기 공식 $o = (i-1) \times s - 2p + k + p_{out}$을 유도하라. 여기서 $i$는 전치 합성곱의 입력 크기, $o$는 그 출력 크기이고 $d = 1$로 둔다.

</div>

??? success "연습문제 1 풀이"
    보통 합성곱과 전치 합성곱은 크기를 서로 반대로 옮긴다. 헷갈리지 않도록 보통 합성곱 쪽 이름을 따로 두자. 보통 합성곱이 크기 $n$을 크기 $m$으로 보낸다고 하면

    $$m = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1$$

    이다. 전치 합성곱은 이 대응을 거꾸로 밟으므로 그 입력은 $i = m$, 그 출력은 $o = n$이다. 바닥 함수를 잠시 떼고 $n$에 대해 풀면

    $$n = (m - 1)s - 2p + k$$

    이므로 $o = (i-1)s - 2p + k$을 얻는다.

    바닥 함수를 떼면서 잃은 것이 $p_{out}$이다. $n + 2p - k$가 $s$로 나누어떨어지지 않으면 나머지 $r \in \{0, 1, \dots, s-1\}$만큼 서로 다른 $n$이 같은 $m$으로 간다. 이를테면 $k=3$, $s=2$, $p=1$일 때 $n = 31$과 $n = 32$가 모두 $m = 16$이 된다. 거꾸로 갈 때는 그 가운데 어느 것을 돌려줄지 정해 주어야 하고, 그 선택이 곧

    $$o = (i-1)s - 2p + k + p_{out}, \qquad 0 \le p_{out} < s$$

    의 $p_{out}$이다. 팽창률 $d$까지 넣으면 핵의 실효 크기가 $d(k-1)+1$이 되므로 $k$를 그것으로 바꾸어 본문의 공식이 된다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
전치 합성곱이 바둑판 무늬 흠을 만들 수 있는 까닭과 그것을 피하는 방법을 설명하라.

</div>

??? success "연습문제 2 풀이"
    보폭이 1보다 크면 전치 합성곱은 입력 원소 사이에 0을 끼워 넣은 뒤 핵을 훑으므로, 출력의 자리마다 입력에서 값을 받는 횟수가 달라진다. 4절에서 센 대로 $K=3$, $s=2$에서는 안쪽 자리가 2번과 4번을 한 칸 걸러 오간다. 이 주기적인 세기 차이가 바둑판 무늬로 보인다.

    피하는 방법은 세 가지다.

    1. **핵 크기를 보폭의 배수로** 한다. $K=4$, $s=2$에서는 안쪽 겹침이 모두 4로 고르고, $K=2$, $s=2$에서는 겹침이 아예 없어 모두 1이다
    2. **크기 조정 뒤 합성곱**을 쓴다. 최근접이나 쌍선형 보간으로 먼저 키운 뒤 보폭 1 합성곱을 얹으면, 보간이 모든 자리를 같은 방식으로 채우므로 겹침이 고르지 않을 여지가 없다
    3. **PixelShuffle**을 쓴다. 채널을 공간으로 늘어놓을 뿐이라 겹침 자체가 없다

    다만 1번은 기하학적 원인만 없앤다. 겹침 횟수가 고르더라도 학습된 가중치가 자리마다 다르면 약한 무늬가 남을 수 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
(가) 전치 합성곱과 (나) 쌍선형 보간 뒤 합성곱으로 상향 표본화 모듈을 각각 구현하고, 출력 크기·매개변수 수·겹침의 고르기를 견주어라.

</div>

??? success "연습문제 3 풀이"
    ```python
    import torch
    import torch.nn as nn

    torch.manual_seed(0)

    x = torch.randn(1, 64, 16, 16)

    # 방법 (가): 전치 합성곱 — 상향 표본화 필터까지 배운다
    up_a = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)

    # 방법 (나): 쌍선형 보간 뒤 합성곱 — 상향 표본화는 고정, 합성곱만 배운다
    up_b = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.Conv2d(64, 32, kernel_size=3, padding=1)
    )

    for name, m in [("(가) ConvTranspose", up_a), ("(나) Bilinear+Conv", up_b)]:
        out = m(x)
        print(f"{name:<22}: → {tuple(out.shape)}, "
              f"params {sum(p.numel() for p in m.parameters()):,}")

    # 겹침이 고른지: 핵을 1로 두고 받은 횟수를 센다
    ones = torch.ones(1, 1, 8, 8)
    counts_a = nn.functional.conv_transpose2d(
        ones, torch.ones(1, 1, 4, 4), stride=2, padding=1).squeeze()
    print(f"(가) 받은 횟수 — 안쪽 최소 {counts_a[1:-1, 1:-1].min():.0f}, "
          f"최대 {counts_a[1:-1, 1:-1].max():.0f}")
    print("(나) 받은 횟수 — 보간은 모든 자리를 똑같이 채우므로 고르다")
    ```

    **출력:**

    ```
    (가) ConvTranspose     : → (1, 32, 32, 32), params 32,800
    (나) Bilinear+Conv     : → (1, 32, 32, 32), params 18,464
    (가) 받은 횟수 — 안쪽 최소 4, 최대 4
    (나) 받은 횟수 — 보간은 모든 자리를 똑같이 채우므로 고르다
    ```

    둘 다 16×16을 32×32로 보내지만 (가)가 매개변수를 1.8배 더 쓴다(32,800 대 18,464). 상향 표본화 필터까지 학습 대상이기 때문이다. 겹침은 $K=4$, $s=2$를 골랐으므로 (가)도 안쪽이 모두 4로 고르다 — 같은 코드를 $K=3$으로 바꾸면 안쪽이 1과 4 사이를 오가며 바둑판 무늬의 조건이 갖추어진다. 고를 때의 기준은 이렇다. 표현력이 필요하고 $K$를 $s$의 배수로 둘 수 있으면 (가), 흠에 민감한 생성 과제라면 (나)다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
전치 합성곱은 어떤 구조에서 흔히 쓰이는가?

</div>

??? success "연습문제 4 풀이"
    디코더 신경망에서 쓰인다. U-Net(분할), 오토인코더, GAN(생성기의 상향 표본화), 초해상도 신경망이 그 예이다. 고정된 상향 표본화 방법의 학습 가능한 짝으로, 신경망이 알맞은 상향 표본화 필터를 배우게 해 준다.

---

## 정리하며

| 항목 | 설명 |
|--------|-------------|
| **연산** | 합성곱 행렬의 전치로, 낮은 해상도를 높은 해상도로 보낸다 |
| **합성곱과의 관계** | 입력에 대한 conv2d의 기울기 (역행렬이 **아니다**) |
| **출력 크기** | $(H_{in}-1) \times s - 2p + d(K-1) + p_{out} + 1$ |
| **덧대기의 방향** | $p$를 키우면 출력이 **작아진다** (덧대는 양이 $K-1-p$이므로) |
| **흔한 쓰임** | 디코더 신경망, GAN, 분할, 초해상도 |
| **주된 함정** | $K \% s \neq 0$일 때의 바둑판 무늬 흠 |
| **모범 관행** | $s$으로 나누어떨어지는 $K$을 쓰거나 크기 조정 뒤 합성곱을 쓴다 |

**참고 문헌**

1. Dumoulin, V., & Visin, F. (2016). "A guide to convolution arithmetic for deep learning." *arXiv preprint arXiv:1603.07285*.

2. Long, J., Shelhamer, E., & Darrell, T. (2015). "Fully Convolutional Networks for Semantic Segmentation." *CVPR*.

3. Odena, A., Dumoulin, V., & Olah, C. (2016). "Deconvolution and Checkerboard Artifacts." *Distill*. https://distill.pub/2016/deconv-checkerboard/

4. Shi, W., et al. (2016). "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network." *CVPR*.

5. Ronneberger, O., Fischer, P., & Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." *MICCAI*.
