# 비국소 신경망

합성곱은 이웃만 본다. $3 \times 3$ 핵 하나가 닿는 곳은 아홉 칸뿐이고, 멀리 떨어진 두 위치를 잇는 길은 층을 쌓아 만드는 수밖에 없다. [수용 영역](../cnn/receptive_field.md)에서 본 대로 $3 \times 3$ 보폭 1 합성곱을 $L$층 쌓으면 수용 영역은 $2L + 1$까지밖에 자라지 않으므로, $28 \times 28$짜리 특징 맵의 모서리와 모서리를 잇는 데만 14층이 든다.

비국소 연산은 그 14층을 한 층으로 바꾼다. 위치 $i$의 반응을 **모든** 위치 $j$에 걸친 가중 평균으로 계산하고, 그 가중치를 특징끼리의 유사도에서 그때그때 정하는 것이다. 이 쪽은 그 한 층이 무엇을 계산하는지, 무엇을 주고 무엇을 받는지 — 위치 수 $N = HW$에 대해 $N \times N$짜리 행렬을 만들어야 한다는 값을 치른다 — 를 식과 실제 `nn.Module`로 함께 본다.

[압축-여기](squeeze_excitation.md)가 **채널**에 문을 달았다면 비국소 블록은 **위치**에 문을 단다. 둘은 같은 절의 짝이며, 뒤의 것이 트랜스포머의 자기 어텐션과 사실상 같은 연산이다.

---

## 1. 핵심 개념

- **먼 거리 의존**: 특징 맵 전체에 걸친 특징 사이 관계를 한 층에서 곧바로 계산
- **어텐션 같은 장치**: 모든 위치의 특징이 얼마나 이바지할지 학습으로 가중
- **깊이와의 맞바꿈**: $3 \times 3$ 합성곱을 $\lceil (\max(H, W) - 1)/2 \rceil$층 쌓아야 얻는 전역 수용 영역을 한 층에서 얻는다
- **값**: 그 대가로 $N \times N$짜리 유사도 행렬이 필요하며, 이것이 시간으로는 $O(N^2 C)$, 메모리로는 $O(N^2)$이다
- **일반성**: 공간 차원과 시간 차원 모두에 적용 가능 — 영상에서는 $N = THW$이다

---

## 2. 수학적 틀

### 비국소 연산

비국소 연산은 위치 $i$에서의 반응을 다음과 같이 계산한다.

$$\mathbf{y}_i = \frac{1}{C(\mathbf{x})} \sum_{j} f(\mathbf{x}_i, \mathbf{x}_j) \, g(\mathbf{x}_j)$$

여기서 각 기호는 다음과 같다.

- $\mathbf{x}_i \in \mathbb{R}^{C}$은 위치 $i$의 특징 벡터이고, $i$은 $N = HW$개의 공간 위치(영상이면 $N = THW$개의 시공간 위치) 가운데 하나를 가리킨다
- $f(\mathbf{x}_i, \mathbf{x}_j)$은 두 위치의 관계를 재는 스칼라다
- $g(\mathbf{x}_j)$은 위치 $j$의 특징을 사영한다 — 실제 구현에서는 $1 \times 1$ 합성곱이다
- $C(\mathbf{x})$은 정규화 인수다

합 기호 $\sum_j$이 $j$의 **모든** 값을 훑는다는 것이 이 식의 전부다. 합성곱이라면 $j$가 $i$의 이웃으로 제한되고, $f$은 $i$와 $j$의 **상대 위치**에만 달린 고정 가중치가 된다. 비국소 연산은 그 두 제약을 함께 푼다.

### 쌍 관계 함수와 정규화

$f$과 $C(\mathbf{x})$은 짝으로 고른다. Wang 등(2018)이 견준 네 가지는 다음과 같다.

| 이름 | $f(\mathbf{x}_i, \mathbf{x}_j)$ | $C(\mathbf{x})$ |
|---|---|---|
| 가우스 | $e^{\mathbf{x}_i^T \mathbf{x}_j}$ | $\sum_j f(\mathbf{x}_i, \mathbf{x}_j)$ |
| 내장 가우스 | $e^{\theta(\mathbf{x}_i)^T \phi(\mathbf{x}_j)}$ | $\sum_j f(\mathbf{x}_i, \mathbf{x}_j)$ |
| 내적 | $\theta(\mathbf{x}_i)^T \phi(\mathbf{x}_j)$ | $N$ |
| 이어 붙이기 | $\text{ReLU}\big(\mathbf{w}^T[\theta(\mathbf{x}_i) \oplus \phi(\mathbf{x}_j)]\big)$ | $N$ |

여기서 $\theta$과 $\phi$은 $1 \times 1$ 합성곱으로 만든 임베딩이고, $\oplus$은 이어 붙이기를 뜻한다.

!!! tip "소프트맥스는 내장 가우스의 다른 이름이다"
    내장 가우스에서 $C(\mathbf{x}) = \sum_j f$을 쓰면 $\dfrac{f(\mathbf{x}_i, \mathbf{x}_j)}{C(\mathbf{x})} = \dfrac{e^{\theta(\mathbf{x}_i)^T \phi(\mathbf{x}_j)}}{\sum_{j'} e^{\theta(\mathbf{x}_i)^T \phi(\mathbf{x}_{j'})}}$이 되어 글자 그대로 $j$에 대한 소프트맥스다. 그래서 아래 코드가 `torch.softmax(scores, dim=-1)` 한 줄로 끝난다. 반면 내적과 이어 붙이기는 $C(\mathbf{x}) = N$, 곧 **단순 평균**이라 소프트맥스가 아니다. 논문이 재어 본 바로는 네 가지의 정확도 차이가 크지 않으며, 중요한 것은 $f$의 선택이 아니라 비국소 연산을 쓴다는 사실 자체다.

!!! warning "배율 조정 내적이 아니다"
    표의 셋째 줄은 [배율 조정 내적](../../ch12/transformer_architecture/scaled_dot_product.md) 어텐션과 이름이 비슷하지만 같지 않다. 트랜스포머는 $\theta^T\phi$을 $\sqrt{d}$로 나눈 뒤 소프트맥스를 씌우지만, 비국소의 "내적" 형태에는 $1/\sqrt{d}$ 배율도 소프트맥스도 없다. 트랜스포머의 어텐션에 대응하는 것은 표의 **둘째** 줄인 내장 가우스다.

---

## 3. 구현 구조

### 블록의 짜임

비국소 **연산**을 기존 신경망에 끼워 넣을 수 있는 **블록**으로 만드는 것은 마지막의 잔차 연결이다.

$$\mathbf{z}_i = W_z \mathbf{y}_i + \mathbf{x}_i$$

여기서 $W_z$은 중간 차원을 $C$로 되돌리는 $1 \times 1$ 합성곱이다. 잔차 형태이므로 [항등 사상](../residual/identity_mapping.md)에서 본 성질을 그대로 물려받지만, 여기서는 그보다 실용적인 쓰임이 있다.

!!! tip "$W_z$을 0으로 시작하면 블록은 항등이다"
    $W_z$의 가중치와 치우침을 모두 0으로 두면 $\mathbf{z}_i = \mathbf{x}_i$이 된다 — 근사가 아니라 정확히 같다. 그래서 이미 학습한 신경망의 아무 자리에나 비국소 블록을 끼워 넣어도 **끼워 넣은 순간의 동작이 조금도 달라지지 않고**, 학습이 진행되면서 $W_z$이 0에서 벗어나는 만큼만 비국소 항이 들어온다. 아래 출력 `[2]`가 이것을 잰다. 논문이 ImageNet에서 미리 학습한 ResNet에 블록을 붙여 실험할 수 있었던 까닭이 이것이다.

블록 전체를 한 줄로 적으면 다음과 같다.

$$\text{NL}(\mathbf{X}) = W_z \Big( \text{softmax}\big(\theta(\mathbf{X})^T \phi(\mathbf{X})\big) \, g(\mathbf{X})^T \Big)^T + \mathbf{X}$$

여기서 $\theta, \phi, g$은 모두 $C$채널을 $C/2$채널로 줄이는 $1 \times 1$ 합성곱이고, 소프트맥스는 둘째 축(곧 $j$)에 대해 취한다.

### 계산 복잡도

중간 차원을 $d = C/2$로 두면 블록이 하는 곱셈-덧셈은 두 갈래다.

$$\text{MAC} = \underbrace{2 N^2 d}_{\theta^T\phi \text{ 와 } A g} + \underbrace{4 N C d}_{1 \times 1 \text{ 합성곱 넷}} = N^2 C + 2 N C^2$$

메모리는 다른 이야기다. 어텐션 행렬 $A \in \mathbb{R}^{N \times N}$을 표본마다 만들어야 하고, 역전파를 위해 그대로 들고 있어야 한다.

$$\text{메모리}(A) = B \times N^2 \times 4\ \text{바이트}\quad (\text{float32})$$

!!! warning "비국소 블록을 낮은 해상도에만 두는 까닭은 연산량이 아니라 메모리다"
    아래 출력 `[4]`가 이 둘을 갈라 보여 준다. ResNet-50의 layer1($56 \times 56$)에서도 비국소 블록의 연산량은 같은 자리 $3 \times 3$ 합성곱의 1.583배에 지나지 않는다 — 감당 못 할 값이 아니다. 그런데 $A$ 하나가 37.52 MiB이므로 배치 32면 1.17 GiB이고, 역전파를 위해 이것을 들고 있어야 한다. 같은 블록을 layer3($14 \times 14$)에 두면 $A$는 0.15 MiB로 줄어든다. 한 변이 4분의 1이 되었으므로 $N^2$은 정확히 $4^4 = 256$분의 1이다(9,834,496 대 38,416). 막는 것은 $N^2 C$이 아니라 $N^2$이다.

### 주장을 실제 모듈에 물어보기

지금까지 적은 것 — $W_z = 0$이면 블록이 정확히 항등이다, 어텐션 행렬의 각 행이 1로 더해진다, 매개변수는 $4C \cdot \tfrac{C}{2}$에 치우침을 더한 값이다, 유사도 행렬은 대칭이 **아니다**, $A$가 해상도의 제곱으로 불어난다, 병목이 값을 정확히 반으로 줄인다 — 은 모두 코드로 확인할 수 있는 주장이다. 아래 프로그램이 하나씩 확인한다.

```python
"""비국소 블록이 말하는 것을 하나씩 재어 본다.

잔차 형태 z = W_z y + x 가 W_z 를 0 으로 놓았을 때 정말 항등인지,
유사도 행렬이 대칭인지, N x N 행렬이 해상도에 따라 얼마나 커지는지,
병목과 하향 표본화가 실제로 얼마를 덜어 주는지를 모듈에 직접 물어본다.
"""

import torch
import torch.nn as nn

torch.manual_seed(0)


# === 비국소 블록 (내장 가우스 형태) ===

class NonLocalBlock2D(nn.Module):
    """Wang 등(2018)의 비국소 블록. 병목 C -> C/2, 잔차 출력, W_z 는 0 으로 시작."""

    def __init__(self, channels, sub_sample=False, zero_init=True):
        super().__init__()
        self.inter = channels // 2
        self.theta = nn.Conv2d(channels, self.inter, 1)
        self.phi = nn.Conv2d(channels, self.inter, 1)
        self.g = nn.Conv2d(channels, self.inter, 1)
        self.W_z = nn.Conv2d(self.inter, channels, 1)
        self.pool = nn.MaxPool2d(2) if sub_sample else nn.Identity()
        if zero_init:
            nn.init.zeros_(self.W_z.weight)
            nn.init.zeros_(self.W_z.bias)

    def affinity(self, x):
        """소프트맥스를 거친 N x N 어텐션 행렬을 돌려준다."""
        B, _, H, W = x.shape
        theta = self.theta(x).view(B, self.inter, H * W)              # B, C/2, N
        phi = self.pool(self.phi(x)).flatten(2)                       # B, C/2, N_hat
        scores = theta.transpose(1, 2) @ phi                          # B, N, N_hat
        return torch.softmax(scores, dim=-1)

    def forward(self, x):
        B, _, H, W = x.shape
        attn = self.affinity(x)                                       # B, N, N_hat
        g = self.pool(self.g(x)).flatten(2)                           # B, C/2, N_hat
        y = (g @ attn.transpose(1, 2)).view(B, self.inter, H, W)      # B, C/2, H, W
        return self.W_z(y) + x


# === [1] 모양과 매개변수 ===

def report_shapes():
    print("[1] 모양과 매개변수  (C = 512, 28x28, 배치 2)")
    x = torch.randn(2, 512, 28, 28)
    block = NonLocalBlock2D(512)
    z = block(x)
    attn = block.affinity(x)
    print(f"    입력        {tuple(x.shape)}")
    print(f"    어텐션 행렬 {tuple(attn.shape)}")
    print(f"    출력        {tuple(z.shape)}")
    n = 28 * 28
    print(f"    N = H*W = {n},  N^2 = {n * n}")
    print(f"    행 합 = 1 인가: 최대 어긋남 {(attn.sum(-1) - 1).abs().max():.2e}")
    p = sum(q.numel() for q in block.parameters())
    formula = 4 * 512 * 256 + 3 * 256 + 512
    print(f"    매개변수 {p:,}개 (식 4*C*(C/2) + 3*(C/2) + C = {formula:,})")


# === [2] 시작할 때 항등인가 ===

def report_identity():
    print("\n[2] W_z = 0 일 때 블록은 항등인가")
    x = torch.randn(2, 64, 16, 16)
    zero = NonLocalBlock2D(64, zero_init=True)
    print(f"    W_z 를 0 으로: max |block(x) - x| = {(zero(x) - x).abs().max():.2e}")
    print(f"    정확히 0 인가: {torch.equal(zero(x), x)}")
    rand = NonLocalBlock2D(64, zero_init=False)
    print(f"    기본 초기화로: max |block(x) - x| = {(rand(x) - x).abs().max():.2e}")


# === [3] 유사도 행렬은 대칭이 아니다 ===

def report_symmetry():
    print("\n[3] 유사도 행렬은 대칭인가  (C = 64, 8x8, N = 64)")
    x = torch.randn(1, 64, 8, 8)
    block = NonLocalBlock2D(64)
    scores = (block.theta(x).flatten(2).transpose(1, 2) @ block.phi(x).flatten(2))
    attn = torch.softmax(scores, dim=-1)
    print(f"    theta != phi: max |S - S^T| = {(scores - scores.transpose(1, 2)).abs().max():.3f}")
    print(f"                  max |A - A^T| = {(attn - attn.transpose(1, 2)).abs().max():.3f}")
    shared = block.theta(x).flatten(2)
    s_sym = shared.transpose(1, 2) @ shared
    a_sym = torch.softmax(s_sym, dim=-1)
    print(f"    theta == phi: max |S - S^T| = {(s_sym - s_sym.transpose(1, 2)).abs().max():.3e}")
    print(f"                  max |A - A^T| = {(a_sym - a_sym.transpose(1, 2)).abs().max():.3f}")


# === [4] 해상도마다의 값 ===

def report_cost():
    print("\n[4] ResNet-50 의 네 단계에서 드는 값  (한 표본, float32)")
    stages = [("layer1", 256, 56), ("layer2", 512, 28),
              ("layer3", 1024, 14), ("layer4", 2048, 7)]
    print("  (a) A 행렬의 크기")
    print("  stage       C     HxW       N          N^2      mem(A)")
    for name, c, s in stages:
        n = s * s
        print(f"  {name:8s} {c:5d}  {s:3d}x{s:<3d} {n:6d} {n * n:12,d} "
              f"{n * n * 4 / 2 ** 20:8.2f} MiB")
    print("  (b) 곱셈-덧셈 횟수 (유사도·집계 2*N^2*(C/2), 1x1 합성곱 4*N*C*(C/2))")
    print("  stage      유사도·집계   1x1 합성곱       합계      3x3 합성곱   비")
    for name, c, s in stages:
        n = s * s
        pair, conv1 = n * n * c, 2 * n * c * c
        conv3 = n * 9 * c * c
        print(f"  {name:8s} {pair / 1e9:9.3f} G {conv1 / 1e9:9.3f} G "
              f"{(pair + conv1) / 1e9:9.3f} G {conv3 / 1e9:9.3f} G {(pair + conv1) / conv3:7.3f}")
    n = 112 * 112
    print(f"  112x112 이면 N = {n:,}, N^2 = {n * n:,}, "
          f"A 하나가 {n * n * 4 / 2 ** 30:.2f} GiB")
    print("  영상은 N = T*H*W 라 프레임 수가 그대로 N 에 곱해진다 (28x28 에서)")
    for t in (4, 8, 32):
        n = t * 28 * 28
        print(f"    T={t:3d}: N = {n:6,d}, N^2 = {n * n:15,d}, "
              f"A 하나가 {n * n * 4 / 2 ** 20:8.1f} MiB")


# === [5] 병목과 하향 표본화가 덜어 주는 몫 ===

def report_savings():
    print("\n[5] 병목과 하향 표본화  (C = 512, 28x28)")
    c, s = 512, 28
    n = s * s
    for name, d in (("중간 차원 C/2 = 256", c // 2), ("중간 차원 C   = 512", c)):
        pair, conv = 2 * n * n * d, 4 * n * c * d
        print(f"    {name}: 유사도·집계 {pair / 1e9:.3f} G + 1x1 합성곱 "
              f"{conv / 1e9:.3f} G = 합계 {(pair + conv) / 1e9:.3f} G")
    print("    두 항이 모두 중간 차원에 정비례하므로 병목이 더는 몫은 정확히 2배다")

    x = torch.randn(2, c, s, s)
    plain, sub = NonLocalBlock2D(c), NonLocalBlock2D(c, sub_sample=True)
    a_plain, a_sub = plain.affinity(x), sub.affinity(x)
    print(f"    하향 표본화 없이: A {tuple(a_plain.shape)}, 원소 {a_plain[0].numel():,}개")
    print(f"    2x2 최대 풀링   : A {tuple(a_sub.shape)}, 원소 {a_sub[0].numel():,}개 "
          f"({a_plain[0].numel() / a_sub[0].numel():.1f}배 적다)")
    print(f"    출력 모양은 그대로: {tuple(sub(x).shape)}")


# === [6] 합성곱을 몇 층 쌓아야 그만큼 보는가 ===

def report_depth():
    print("\n[6] 3x3 stride 1 합성곱으로 같은 범위를 덮으려면")
    for s in (56, 28, 14, 7):
        layers = -(-(s - 1) // 2)                  # 2L + 1 >= s
        print(f"    {s:2d}x{s:<2d} 지도: {layers:2d}층  (수용 영역 {2 * layers + 1} >= {s})")
    print("    비국소 블록은 어느 쪽이든 1층이다")


if __name__ == "__main__":
    report_shapes()
    report_identity()
    report_symmetry()
    report_cost()
    report_savings()
    report_depth()
```

**출력:**

```
[1] 모양과 매개변수  (C = 512, 28x28, 배치 2)
    입력        (2, 512, 28, 28)
    어텐션 행렬 (2, 784, 784)
    출력        (2, 512, 28, 28)
    N = H*W = 784,  N^2 = 614656
    행 합 = 1 인가: 최대 어긋남 7.15e-07
    매개변수 525,568개 (식 4*C*(C/2) + 3*(C/2) + C = 525,568)

[2] W_z = 0 일 때 블록은 항등인가
    W_z 를 0 으로: max |block(x) - x| = 0.00e+00
    정확히 0 인가: True
    기본 초기화로: max |block(x) - x| = 1.13e+00

[3] 유사도 행렬은 대칭인가  (C = 64, 8x8, N = 64)
    theta != phi: max |S - S^T| = 10.108
                  max |A - A^T| = 0.650
    theta == phi: max |S - S^T| = 0.000e+00
                  max |A - A^T| = 0.175

[4] ResNet-50 의 네 단계에서 드는 값  (한 표본, float32)
  (a) A 행렬의 크기
  stage       C     HxW       N          N^2      mem(A)
  layer1     256   56x56    3136    9,834,496    37.52 MiB
  layer2     512   28x28     784      614,656     2.34 MiB
  layer3    1024   14x14     196       38,416     0.15 MiB
  layer4    2048    7x7       49        2,401     0.01 MiB
  (b) 곱셈-덧셈 횟수 (유사도·집계 2*N^2*(C/2), 1x1 합성곱 4*N*C*(C/2))
  stage      유사도·집계   1x1 합성곱       합계      3x3 합성곱   비
  layer1       2.518 G     0.411 G     2.929 G     1.850 G   1.583
  layer2       0.315 G     0.411 G     0.726 G     1.850 G   0.392
  layer3       0.039 G     0.411 G     0.450 G     1.850 G   0.243
  layer4       0.005 G     0.411 G     0.416 G     1.850 G   0.225
  112x112 이면 N = 12,544, N^2 = 157,351,936, A 하나가 0.59 GiB
  영상은 N = T*H*W 라 프레임 수가 그대로 N 에 곱해진다 (28x28 에서)
    T=  4: N =  3,136, N^2 =       9,834,496, A 하나가     37.5 MiB
    T=  8: N =  6,272, N^2 =      39,337,984, A 하나가    150.1 MiB
    T= 32: N = 25,088, N^2 =     629,407,744, A 하나가   2401.0 MiB

[5] 병목과 하향 표본화  (C = 512, 28x28)
    중간 차원 C/2 = 256: 유사도·집계 0.315 G + 1x1 합성곱 0.411 G = 합계 0.726 G
    중간 차원 C   = 512: 유사도·집계 0.629 G + 1x1 합성곱 0.822 G = 합계 1.451 G
    두 항이 모두 중간 차원에 정비례하므로 병목이 더는 몫은 정확히 2배다
    하향 표본화 없이: A (2, 784, 784), 원소 614,656개
    2x2 최대 풀링   : A (2, 784, 196), 원소 153,664개 (4.0배 적다)
    출력 모양은 그대로: (2, 512, 28, 28)

[6] 3x3 stride 1 합성곱으로 같은 범위를 덮으려면
    56x56 지도: 28층  (수용 영역 57 >= 56)
    28x28 지도: 14층  (수용 영역 29 >= 28)
    14x14 지도:  7층  (수용 영역 15 >= 14)
     7x7  지도:  3층  (수용 영역 7 >= 7)
    비국소 블록은 어느 쪽이든 1층이다
```

!!! note "합계가 해상도에 따라 별로 줄지 않는 까닭"
    출력 `[4](b)`의 "1x1 합성곱" 열은 네 단계에서 모두 0.411 G로 **같다**. ResNet의 단계마다 해상도가 반으로 줄면서 채널이 두 배가 되므로 $NC^2$이 그대로 보존되기 때문이다. 그래서 layer3에서는 이 항이 합계의 91.3%를 차지하고, 유사도 계산은 8.7%밖에 되지 않는다. $O(N^2C)$이라는 표기만 보면 낮은 해상도에서 비국소가 거의 공짜가 될 것 같지만, 실제로는 $2NC^2$이라는 바닥이 있어 layer4에서도 $3 \times 3$ 합성곱의 0.225배는 든다.

---

## 4. 합성곱을 쌓는 방식과의 견줌

**관계를 곧바로 모형화**: 위 출력 `[6]`이 이 이야기의 전부다. $28 \times 28$ 지도에서 모서리와 모서리를 잇는 데 $3 \times 3$ 보폭 1 합성곱은 14층이 들지만 비국소 블록은 1층이다. 층 수는 $\lceil (\max(H, W) - 1)/2 \rceil$, 곧 한 변의 길이에 비례해 자란다.

!!! warning "$O(\log HW)$이 아니다"
    "합성곱 몇 층이면 전역이 되는가"에 로그가 나오려면 층 사이에 보폭 2의 하향 표본화가 있어야 한다. 그때는 수용 영역이 층마다 두 배로 뛰므로 $O(\log \max(H, W))$층이면 된다. 그러나 **한 해상도 안에서** 보폭 1 합성곱만 쌓을 때는 수용 영역이 $2L + 1$로 **선형**으로만 자라므로 $\Theta(\max(H, W))$층이 필요하다. 비국소 블록이 대신하는 것은 뒤쪽이다. [팽창 합성곱](../cnn/dilated_convolutions.md)은 같은 문제를 해상도를 낮추지 않고 푸는 또 다른 방법이다.

**학습 가능한 관련성**: 합성곱의 가중치는 상대 위치 $i - j$에만 달린 고정 값이라, 학습이 끝나면 어떤 입력에도 같은 가중치가 쓰인다. 비국소의 가중치는 $\theta(\mathbf{x}_i)^T\phi(\mathbf{x}_j)$이므로 **입력마다 다시 계산된다**. 같은 위치 쌍이라도 그림이 달라지면 다른 가중치를 받는다.

!!! danger "비국소 연산은 대칭이 아니다"
    "$i$과 $j$의 관계"라는 말 때문에 $f(\mathbf{x}_i, \mathbf{x}_j) = f(\mathbf{x}_j, \mathbf{x}_i)$일 것 같지만 그렇지 않다. 출력 `[3]`이 두 단계로 보여 준다. 첫째, $\theta \ne \phi$이므로 점수 행렬부터 대칭이 아니다($\max|S - S^\top| = 10.108$). 둘째, **설령 $\theta = \phi$으로 묶어 점수 행렬을 대칭으로 만들어도**($\max|S - S^\top| = $ 0.000e+00) 소프트맥스를 씌우면 대칭이 깨진다($\max|A - A^\top| = 0.175$). 정규화 인수 $C(\mathbf{x}) = \sum_j f(\mathbf{x}_i, \mathbf{x}_j)$이 행마다 다르기 때문이다. $A$은 대칭 행렬이 아니라 **행마다 합이 1인 행렬**이며, 출력 `[1]`이 그 성질을 잰다(어긋남 7.15e-07, float32 반올림). 곧 "$i$이 $j$를 얼마나 보는가"와 "$j$이 $i$를 얼마나 보는가"는 서로 다른 수다.

---

## 5. 컴퓨터 비전에서의 활용

앞 절이 공간 차원만 다루었지만 식 어디에도 차원이 둘이어야 할 이유는 없다. 지표 $i$을 시공간으로 넓히면 그대로 영상에 쓸 수 있다.

### 영상 이해

영상에서는 위치가 $(t, i, j)$이고, 합이 세 지표 모두를 훑는다.

$$\mathbf{y}_{t,i,j} = \frac{1}{C(\mathbf{x})} \sum_{t', i', j'} f\big(\mathbf{x}_{t,i,j}, \mathbf{x}_{t',i',j'}\big) \, g\big(\mathbf{x}_{t',i',j'}\big)$$

3차원 합성곱이 한 층에서 시간 축으로 커널 크기만큼(대개 3프레임)만 보는 데 견주어, 비국소는 첫 프레임의 한 점과 마지막 프레임의 한 점을 한 층에서 잇는다. 대가는 그대로 $N = THW$에 실린다.

!!! warning "영상에서는 프레임 수가 곧바로 $N$에 곱해진다"
    출력 `[4]`의 마지막 세 줄이 이것이다. $28 \times 28$ 특징 맵이라도 32프레임을 함께 넣으면 $N = 25{,}088$이 되어 $A$ 하나가 2,401.0 MiB, 곧 2.34 GiB다. 표본 **하나**의 어텐션 행렬 하나가 그렇다. 그래서 영상 쪽 구현은 프레임을 4~8개로 끊거나 시간 축을 먼저 줄인 뒤에 비국소를 쓴다. [때에 걸친 어텐션](../../ch21/video/09_temporal_attention.md)과 [느림빠름 그물](../../ch21/video/10_slowfast_network.md)이 이 문제를 다루는 두 갈래다.

### 세밀한 인식

새의 부리와 꽁지처럼 서로 떨어진 부분의 관계가 갈래를 가르는 과제에서, 비국소 블록은 그 두 자리를 잇는 가중치를 직접 학습한다. 합성곱만으로는 두 자리가 같은 수용 영역에 들어올 만큼 깊어져야 비로소 그 관계가 표현된다.

### 3차원 점구름 처리

점구름에는 격자가 없으므로 합성곱을 그대로 쓸 수 없지만, 비국소 연산은 $f$과 $g$만 있으면 되고 $j$이 어떤 격자 위에 있어야 할 까닭이 없다. $\theta, \phi$을 점 특징에 대한 다층 퍼셉트론으로 바꾸면 식이 그대로 성립한다.

---

## 6. CNN 구조에 넣기

Wang 등(2018)은 ResNet의 res3와 res4 단계, 곧 $28 \times 28$과 $14 \times 14$ 자리에 블록을 넣었다. 출력 `[4]`가 그 선택을 설명한다.

- **앞쪽 층**($56 \times 56$ 이상): 지역 합성곱만 쓴다. 연산량은 $3 \times 3$ 합성곱의 1.583배로 감당할 만하지만 $A$ 하나가 37.52 MiB라 배치를 키울 수 없고, 저수준 특징에는 전역 맥락이 별로 쓸모가 없다
- **중간 층**($28 \times 28$, $14 \times 14$): 의미 있는 특징이 이미 만들어졌고 $A$이 2.34 MiB와 0.15 MiB로 가벼운 자리다. 논문이 블록을 넣는 곳이다
- **뒤쪽 층**($7 \times 7$): $N = 49$밖에 안 되어 수용 영역이 이미 사실상 전역이므로 비국소가 새로 주는 것이 적다

블록은 잔차 블록 **뒤**에 놓고 $W_z$을 0으로 두면, 3절에서 본 대로 끼워 넣는 순간 신경망의 출력이 전혀 달라지지 않는다.

---

## 7. 계산을 줄이는 전략

**병목 설계**: $\theta, \phi, g$의 출력 채널을 $C$가 아니라 $d = C/2$로 둔다. 유사도 항 $2N^2d$과 합성곱 항 $4NCd$이 **둘 다** $d$에 정비례하므로, 값은 정확히 반이 된다.

$$\text{MAC}(d) = 2N^2 d + 4NCd = 2d\,(N^2 + 2NC)$$

출력 `[5]`가 $C = 512$, $28 \times 28$에서 1.451 G와 0.726 G로 이것을 확인한다. 흔히 "$\ll$"로 적히지만 실제로 얻는 것은 **2배**이며, 그 이상을 얻으려면 $d$를 더 줄여야 한다.

**공간 하향 표본화**: $\phi$과 $g$ 뒤에만 보폭 $r$의 최대 풀링을 두어 $j$이 훑는 위치를 $N$에서 $N/r^2$으로 줄인다. $i$의 개수는 그대로이므로 출력 모양은 변하지 않는다.

$$\text{유사도·집계} = 2 \cdot N \cdot \frac{N}{r^2} \cdot d$$

$1 \times 1$ 합성곱 항 $4NCd$은 풀링이 합성곱 **뒤**에 오므로 줄지 않는다. $r = 2$이면 유사도 항이 4분의 1이 되고, 어텐션 행렬도 $N \times N$에서 $N \times N/4$로 줄어든다. 출력 `[5]`에서 $784 \times 784$가 $784 \times 196$이 되고 원소 수가 614,656개에서 153,664개로 정확히 4배 줄어든 것이 이것이다. 출력 모양은 `(2, 512, 28, 28)` 그대로다.

**갈래 나누기**: 특징 맵을 겹치지 않는 영역으로 나누어 영역 안에서만 비국소 연산을 하고 합친다. $N$개 위치를 $k$개 영역으로 나누면 $N^2$이 $k \cdot (N/k)^2 = N^2/k$이 되지만, 영역을 가로지르는 관계는 그만큼 잃는다. 스윈 트랜스포머의 창 어텐션이 이 생각을 밀고 나간 것이다.

---

## 8. 관련 주제

- [압축-여기](squeeze_excitation.md) — 같은 절의 짝. 채널에 문을 달아 $N$에 이차인 값을 치르지 않는 쪽
- [합성곱 신경망에서 트랜스포머로](cnn_to_vit_bridge.md) — 비국소 블록이 순수 CNN과 ViT 사이의 어느 단계인지
- [자기 어텐션](../../ch11/attention/self_attention.md) — 내장 가우스 형태의 비국소 연산이 곧 이것이다
- [배율 조정 내적](../../ch12/transformer_architecture/scaled_dot_product.md) — 2절의 경고가 가리키는 차이
- [ViT 훑어보기](../../ch12/transformers_vision/vit.md) — 합성곱을 다 걷어 내고 어텐션만 남기면 어디에 닿는지
- [수용 영역](../cnn/receptive_field.md) — 4절의 $2L + 1$이 나오는 곳
- [항등 사상](../residual/identity_mapping.md) — 3절의 잔차 형태가 기대는 성질
- [CBAM](../../ch21/image_classification/24_cbam.md) — 채널 어텐션과 공간 어텐션을 잇대어 쓰는 확장
- [때에 걸친 어텐션](../../ch21/video/09_temporal_attention.md) — 영상에서 $N = THW$을 감당하는 방법

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
$C = 512$, 특징 맵이 $28 \times 28$, 중간 차원 $d = C/2$일 때 비국소 블록의 매개변수 개수를 세어라. 치우침까지 넣어라.

</div>

??? success "연습문제 1 풀이"
    $1 \times 1$ 합성곱이 넷이다.

    - $\theta, \phi, g$: 각각 $C \times d = 512 \times 256 = 131{,}072$개의 가중치와 $d = 256$개의 치우침
    - $W_z$: $d \times C = 256 \times 512 = 131{,}072$개의 가중치와 $C = 512$개의 치우침

    합하면 가중치 $4 \times 131{,}072 = 524{,}288$개, 치우침 $3 \times 256 + 512 = 1{,}280$개, 모두 **525,568개**다. 출력 `[1]`이 `sum(p.numel() for p in block.parameters())`로 센 값과 같다.

    공간 크기 $28 \times 28$은 이 셈에 들어오지 않는다는 데 주의하라. 매개변수는 $4C d + 3d + C$으로 해상도와 무관하고, 해상도에 달린 것은 연산량과 메모리뿐이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
$3 \times 3$ 보폭 1 합성곱을 $L$층 쌓았을 때의 수용 영역을 $L$의 식으로 적고, $56 \times 56$ 특징 맵의 한 모서리에서 반대 모서리까지 닿으려면 몇 층이 필요한지 구하라. 비국소 블록은 몇 층인가?

</div>

??? success "연습문제 2 풀이"
    층마다 양쪽으로 1칸씩 늘어나므로 $L$층 뒤의 수용 영역은 $2L + 1$이다. $2L + 1 \ge 56$을 풀면 $L \ge 27.5$, 곧 $L = 28$이다. 출력 `[6]`이 이 값을 찍는다(수용 영역 57).

    비국소 블록은 1층이다. 두 위치의 거리가 얼마든 $\sum_j$이 그 $j$를 이미 포함하고 있기 때문이다.

    다만 "$O(\log HW)$층이면 된다"는 흔한 말은 **보폭 2의 하향 표본화가 사이사이에 있을 때**만 맞다. 그때는 수용 영역이 층마다 두 배가 되므로 $\log_2 56 \approx 5.8$, 곧 6층이면 된다. 한 해상도 안에서 보폭 1 합성곱만 쌓으면 선형으로만 자란다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
공간 차원이 $H \times W$, 채널이 $C$, 중간 차원이 $C/2$일 때 비국소 블록과 같은 자리 $3 \times 3$ 합성곱($C \to C$)의 곱셈-덧셈 횟수를 적고, ResNet-50의 layer1과 layer3에서 비를 구하라. 어느 항이 어디서 이기는가?

</div>

??? success "연습문제 3 풀이"
    $N = HW$으로 두면 비국소는 유사도·집계에 $2N^2 \cdot \tfrac{C}{2} = N^2 C$, $1 \times 1$ 합성곱 넷에 $4NC \cdot \tfrac{C}{2} = 2NC^2$이 들어 합계 $N^2C + 2NC^2$이다. $3 \times 3$ 합성곱은 $9NC^2$이다. 비는 다음과 같다.

    $$\frac{N^2 C + 2NC^2}{9NC^2} = \frac{N}{9C} + \frac{2}{9}$$

    layer1은 $C = 256$, $N = 3136$이므로 $\dfrac{3136}{2304} + \dfrac{2}{9} = 1.361 + 0.222 = 1.583$이고, layer3은 $C = 1024$, $N = 196$이므로 $\dfrac{196}{9216} + \dfrac{2}{9} = 0.021 + 0.222 = 0.243$이다. 출력 `[4](b)`의 마지막 열과 같다.

    첫째 항이 $N/9C$이므로 **$N$이 $C$보다 크면 유사도가, 작으면 $1 \times 1$ 합성곱이 이긴다.** layer1은 $3136 > 256$이라 유사도가 86.0%(2.518 G / 2.929 G), layer3은 $196 < 1024$이라 $1 \times 1$ 합성곱이 91.3%(0.411 G / 0.450 G)다. 그리고 둘째 항 $2/9 = 0.222$은 $N$을 아무리 줄여도 남는 바닥이다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
비국소 블록을 신경망의 맨 앞이 아니라 합성곱 층 여러 개 뒤에 두는 까닭은 무엇인가? 연산량과 메모리를 갈라서 답하라.

</div>

??? success "연습문제 4 풀이"
    흔한 답은 "앞쪽은 연산량을 감당할 수 없어서"이지만, 출력 `[4]`를 보면 그것만으로는 부족하다.

    **연산량**: layer1($56 \times 56$)에서도 비국소는 같은 자리 $3 \times 3$ 합성곱의 1.583배다. 비싸긴 해도 못 쓸 값은 아니다.

    **메모리**: 같은 자리에서 어텐션 행렬 $A$ 하나가 37.52 MiB이고 배치 32면 1.17 GiB다. 역전파를 위해 이것을 끝까지 들고 있어야 하므로 배치 크기가 곧바로 깎인다. layer3($14 \times 14$)에서는 0.15 MiB, 곧 $N^2$의 비로 정확히 256분의 1이다(9,834,496 대 38,416). 막는 것은 이쪽이다. 첫 합성곱 뒤의 $112 \times 112$ 자리라면 $N = 12{,}544$, $A$ 하나가 0.59 GiB로 아예 불가능하다.

    **표현**: 앞쪽 특징은 모서리와 색 같은 저수준이라 "멀리 떨어진 두 자리가 비슷한가"를 물어도 쓸모 있는 답이 나오지 않는다. 전역 맥락은 의미 있는 특징 위에서 값어치가 있다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span>
$\theta$이 만든 임베딩을 열로 늘어놓은 행렬을 $\Theta \in \mathbb{R}^{d \times N}$이라 하자. $\theta = \phi$으로 묶으면 점수 행렬 $S = \Theta^\top \Theta$은 대칭이 된다. 그런데도 어텐션 행렬 $A = \text{softmax}(S)$은 대칭이 아니다. 왜 그런지 보이고, $A$이 대칭이 될 필요충분조건을 구하라.

</div>

??? success "연습문제 5 풀이"
    $A_{ij} = \dfrac{e^{S_{ij}}}{\sum_{k} e^{S_{ik}}}$이고 $A_{ji} = \dfrac{e^{S_{ji}}}{\sum_{k} e^{S_{jk}}}$이다. $S$이 대칭이면 분자는 $e^{S_{ij}} = e^{S_{ji}}$으로 같지만, 분모는 $i$번째 행의 합과 $j$번째 행의 합이라 서로 다르다. 곧

    $$\frac{A_{ij}}{A_{ji}} = \frac{\sum_k e^{S_{jk}}}{\sum_k e^{S_{ik}}}$$

    이므로 $A$이 대칭일 필요충분조건은 $\sum_k e^{S_{ik}}$이 $i$에 무관한 것, 곧 **모든 행의 정규화 인수 $C(\mathbf{x})$이 같은 것**이다($S$이 대칭이고 모든 원소가 양수이므로 이 조건은 $i$과 $j$을 바꾸어도 그대로다). 행마다 원소의 배열은 달라도 합만 같으면 되지만, 학습으로 얻은 $\Theta$에서 그런 일은 우연이 아니면 일어나지 않는다.

    출력 `[3]`의 아래 두 줄이 이 계산을 확인한다. $\theta = \phi$으로 묶어 $\max|S - S^\top| = 0$을 만들어도 $\max|A - A^\top| = 0.175$이다. 소프트맥스는 행별 연산이므로 대칭성을 보존하지 않는다.

    이 때문에 비국소 블록은 "두 위치를 잇는 무향 간선"이 아니라 **방향 있는 간선**을 학습한다. 배경의 한 점이 사람의 손을 강하게 볼 수 있으면서 손은 그 배경 점을 거의 보지 않을 수 있다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
$W_z$의 가중치와 치우침을 0으로 두면 블록이 정확히 항등이라는 것을 식으로 보여라. 또 $W_z$에 오는 기울기가 0이 아님을 보이고, 그래서 학습이 왜 멈추지 않는지 설명하라.

</div>

??? success "연습문제 6 풀이"
    **항등**: 블록은 $\mathbf{z} = W_z \mathbf{y} + \mathbf{x}$이다. $W_z$이 $1 \times 1$ 합성곱이므로 위치마다 $\mathbf{z}_i = W \mathbf{y}_i + \mathbf{b} + \mathbf{x}_i$이고, $W = 0$, $\mathbf{b} = 0$이면 $\mathbf{z}_i = \mathbf{x}_i$이다. $\mathbf{y}_i$이 무엇이든 상관없으므로 근사가 아니라 정확히 같다. 출력 `[2]`의 `torch.equal(zero(x), x) = True`이 이것이고, 기본 초기화에서는 같은 입력에 대해 최대 1.13만큼 어긋난다.

    **기울기**: 손실 $L$에 대해 $\dfrac{\partial L}{\partial W} = \sum_i \dfrac{\partial L}{\partial \mathbf{z}_i} \mathbf{y}_i^\top$이다. 이 식에 $W$이 들어 있지 않다는 것이 핵심이다. $W = 0$이어도 $\mathbf{y}_i \ne 0$이고 $\partial L / \partial \mathbf{z}_i \ne 0$이면 기울기는 0이 아니다. 그리고 $\mathbf{y}_i$은 $\theta, \phi, g$이 만드는 값이라 $W$과 무관하게 0이 아니다.

    **왜 멈추지 않는가**: 순전파에서는 $W = 0$이 비국소 가지를 완전히 막지만, 역전파에서 $W$ 자신에게 오는 기울기는 그 가지를 지나지 않고 $\partial L/\partial \mathbf{z}$에서 곧바로 온다. 그래서 첫 갱신에서 $W$이 0을 벗어나고, 그 뒤로는 $\theta, \phi, g$에도 $\partial L/\partial \mathbf{y} = W^\top \partial L/\partial \mathbf{z}$을 거쳐 기울기가 흐른다. 곧 **첫 걸음에서만 막히고 그 뒤로는 열린다**. 잔차 가지의 마지막 층을 0으로 초기화하는 수법이 널리 쓰이는 까닭이 이것이며, [항등 사상](../residual/identity_mapping.md)의 잔차 블록도 같은 구조다.

---

## 정리하며

비국소 연산은 $\mathbf{y}_i = \frac{1}{C(\mathbf{x})}\sum_j f(\mathbf{x}_i, \mathbf{x}_j) g(\mathbf{x}_j)$ 한 줄이며, 합성곱과 다른 점은 $j$이 이웃이 아니라 전부를 훑는다는 것과 가중치가 상대 위치가 아니라 특징에서 나온다는 것 둘뿐이다. 내장 가우스 형태에서 $C(\mathbf{x}) = \sum_j f$을 쓰면 이 식은 글자 그대로 소프트맥스 어텐션이 된다.

이 쪽이 실제로 재어 본 것은 셋이다. 첫째, $W_z$을 0으로 초기화하면 블록은 **정확히** 항등이다(`torch.equal` 이 `True`). 그래서 이미 학습한 신경망 아무 데나 끼워 넣어도 그 순간의 동작이 달라지지 않으면서, $W_z$에 오는 기울기는 0이 아니므로 학습은 멈추지 않는다. 둘째, 어텐션 행렬은 대칭이 **아니다** — $\theta = \phi$으로 묶어 점수 행렬을 대칭으로 만들어도 소프트맥스의 행별 정규화가 대칭을 깬다($\max|A - A^\top| = 0.175$). 셋째, 값은 $N^2C + 2NC^2$이고 메모리는 $BN^2$인데, 비국소 블록을 낮은 해상도에만 두게 만드는 것은 앞이 아니라 뒤다. ResNet-50의 layer1에서 연산량은 $3 \times 3$ 합성곱의 1.583배로 감당할 만하지만 어텐션 행렬 하나가 37.52 MiB이고, layer3에서는 0.15 MiB로 256분의 1이 된다.

병목($d = C/2$)은 값을 정확히 2배 줄이고, $\phi$과 $g$에 보폭 2 풀링을 두는 하향 표본화는 유사도 항을 4배 줄인다. 영상에서는 $N = THW$이라 32프레임이면 $28 \times 28$에서도 어텐션 행렬 하나가 2.34 GiB이므로, 이 두 수법이 선택이 아니라 필수가 된다.
