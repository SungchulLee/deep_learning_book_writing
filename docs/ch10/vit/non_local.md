# 비국소 신경망

비국소(Non-local) 연산은 위치별 특징을 모든 위치의 특징과 비교해 가중 결합한다. 이 페이지는 embedded Gaussian 형태의 비국소 블록을 셀프 어텐션과 연결하고, 전체 위치 쌍의 계산·메모리 비용을 분석한다.

국소 합성곱도 깊이·다운샘플링·팽창률을 통해 넓은 문맥을 처리할 수 있다. 비국소 연산은 먼 위치를 한 번의 집계로 직접 연결하는 다른 방식이며, 위치별 배율만 곱하는 공간 게이트와는 구분한다.

## 1. 핵심 개념

- **먼 거리 의존**: 특징 맵 전체에 걸친 특징 사이 관계를 한 층에서 곧바로 계산
- **어텐션 같은 장치**: 모든 위치의 특징이 얼마나 이바지할지 학습으로 가중
- **직접 연결**: 모든 위치를 한 번의 집계에 포함한다. 국소 합성곱과의 거리 비교에서는 출력의 위치와 경계 규약을 함께 정해야 한다.
- **비용**: 아래의 밀집 구현은 N×N 행렬을 명시적으로 만들며 유사도·집계와 채널 투영의 비용을 모두 가진다.
- **일반성**: 공간 차원과 시간 차원 모두에 적용 가능 — 영상에서는 $N = THW$이다

---

## 2. 수학적 틀

### 비국소 연산

비국소 연산은 위치 $i$에서의 반응을 다음과 같이 계산한다.

$$\mathbf{y}_i = \frac{1}{C(\mathbf{x})} \sum_{j} f(\mathbf{x}_i, \mathbf{x}_j) \, g(\mathbf{x}_j)$$

여기서 각 기호는 다음과 같다.

- $\mathbf{x}_i \in \mathbb{R}^{C}$은 위치 $i$의 특징 벡터이고, $i$은 $N = HW$개의 공간 위치(영상이면 $N = THW$개의 시공간 위치) 가운데 하나를 가리킨다
- $f(\mathbf{x}_i, \mathbf{x}_j)$은 두 위치의 관계를 재는 스칼라다
- $g(\mathbf{x}_j)$은 위치 $j$의 특징을 투영한다 — 실제 구현에서는 $1 \times 1$ 합성곱이다
- $C(\mathbf{x})$은 정규화 인수다

j를 모든 위치에 걸쳐 합산하면 멀리 있는 특징도 한 번에 결합할 수 있다. 일반적인 다채널 합성곱의 연결은 국소 영역에 제한되고 채널별 행렬 가중치를 사용한다. 따라서 모든 합성곱을 스칼라 유사도 함수 f의 고정값으로 완전히 동일하게 표현한 것이라고 가정하지 않는다. C(x)는 일반적으로 질의 위치 i에 따라 달라지는 정규화 인수이다.

### 쌍 관계 함수와 정규화

$f$과 $C(\mathbf{x})$은 짝으로 고른다. Wang 등(2018)이 견준 네 가지는 다음과 같다.

| 이름 | $f(\mathbf{x}_i, \mathbf{x}_j)$ | $C(\mathbf{x})$ |
|---|---|---|
| 가우스 | $e^{\mathbf{x}_i^T \mathbf{x}_j}$ | $\sum_j f(\mathbf{x}_i, \mathbf{x}_j)$ |
| embedded Gaussian | $e^{\theta(\mathbf{x}_i)^T \phi(\mathbf{x}_j)}$ | $\sum_j f(\mathbf{x}_i, \mathbf{x}_j)$ |
| 내적 | $\theta(\mathbf{x}_i)^T \phi(\mathbf{x}_j)$ | $N$ |
| 이어 붙이기 | $\text{ReLU}\big(\mathbf{w}^T[\theta(\mathbf{x}_i) \oplus \phi(\mathbf{x}_j)]\big)$ | $N$ |

여기서 $\theta$과 $\phi$은 $1 \times 1$ 합성곱으로 만든 임베딩이고, $\oplus$은 이어 붙이기를 뜻한다.

!!! note "embedded Gaussian의 행별 정규화"
    지수화한 점수를 질의 위치 i의 행 합으로 나누면 j축 소프트맥스가 된다. 반면 내적·이어 붙이기 형태의 1/N은 가중 합을 N으로 스케일하는 것이며, f가 일정한 1이 아닌 한 값 특징의 단순 평균과 같지 않다. 함수 선택의 성능 효과는 데이터와 구현에 따라 비교해야 한다.

!!! warning "스케일 내적 어텐션이 아니다"
    표의 셋째 줄은 [스케일 내적 어텐션](../../ch12/transformer_architecture/scaled_dot_product.md) 어텐션과 이름이 비슷하지만 같지 않다. 트랜스포머는 $\theta^T\phi$을 $\sqrt{d}$로 나눈 뒤 소프트맥스를 씌우지만, 비국소의 "내적" 형태에는 $1/\sqrt{d}$ 배율도 소프트맥스도 없다. 트랜스포머의 어텐션에 대응하는 것은 표의 **둘째** 줄인 embedded Gaussian다.

---

## 3. 구현 구조

### 블록의 구조

비국소 **연산**을 기존 신경망에 끼워 넣을 수 있는 **블록**으로 만드는 것은 마지막의 잔차 연결이다.

$$\mathbf{z}_i = W_z \mathbf{y}_i + \mathbf{x}_i$$

여기서 $W_z$은 중간 차원을 $C$로 되돌리는 $1 \times 1$ 합성곱이다. 잔차 형태이므로 [항등 사상](../residual/identity_mapping.md)에서 본 성질을 그대로 물려받지만, 여기서는 그보다 실용적인 쓰임이 있다.

!!! note "출력 투영의 0 초기화"
    유한한 중간 값에 대해 출력 투영의 가중치·편향이 0이면 잔차 가지가 0이 되어 z=x이다. 차원이 맞는 위치에 넣으면 초기 순전파를 유지하는 데 유용하다. 그러나 NaN·Inf가 있거나 주변 상태·난수 소비가 바뀌는 상황까지 모든 프로그램 동작이 같다는 보장은 아니다. 출력 투영이 이후 실제로 갱신되는지도 기울기로 확인해야 한다.

블록 전체를 한 줄로 적으면 다음과 같다.

$$\text{NL}(\mathbf{X}) = W_z \Big( \text{softmax}\big(\theta(\mathbf{X})^T \phi(\mathbf{X})\big) \, g(\mathbf{X})^T \Big)^T + \mathbf{X}$$

여기서 $\theta, \phi, g$은 모두 $C$채널을 $C/2$채널로 줄이는 $1 \times 1$ 합성곱이고, 소프트맥스는 둘째 축(곧 $j$)에 대해 취한다.

### 계산 복잡도

중간 차원을 $d = C/2$로 두면 블록이 하는 곱셈-덧셈은 두 갈래다.

$$\text{MAC} = \underbrace{2 N^2 d}_{\theta^T\phi \text{ 와 } A g} + \underbrace{4 N C d}_{1 \times 1 \text{ 합성곱 넷}} = N^2 C + 2 N C^2$$

아래의 명시적 행렬 구현은 표본마다 N×N 어텐션을 만든다. 다음 식은 그 행렬 하나의 float32 저장량이며 모델 전체나 최대 학습 메모리가 아니다. 역전파 저장·재계산 전략과 융합 구현에 따라 실제 메모리 경로가 달라질 수 있다.

$$\text{메모리}(A) = B \times N^2 \times 4\ \text{바이트}\quad (\text{float32})$$

!!! note "해상도와 비용"
    높은 해상도에서는 N² 행렬의 메모리가 큰 부담이 될 수 있고 계산량도 함께 커진다. 어떤 자원이 실제 병목인지는 채널 수·배치·장치와 구현에 따라 다르다. 예를 들어 56×56의 float32 행렬 하나는 약 37.52MiB, 14×14에서는 약 0.15MiB이다. 이 값만으로 앞쪽 배치가 불가능하거나 연산 비용이 무시할 만하다고 판정하지 않는다.

### 주장을 실제 모듈에 물어보기

지금까지 적은 것 — $W_z = 0$이면 블록이 정확히 항등이다, 어텐션 행렬의 각 행이 1로 더해진다, 매개변수는 $4C \cdot \tfrac{C}{2}$에 편향을 더한 값이다, 유사도 행렬은 대칭이 **아니다**, $A$가 해상도의 제곱으로 불어난다, 병목이 값을 정확히 반으로 줄인다 — 은 모두 코드로 확인할 수 있는 주장이다. 아래 프로그램이 하나씩 확인한다.

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

## 4. 합성곱을 쌓는 방식과의 비교

**위치 간 거리**: 3×3·보폭 1·팽창률 1의 합성곱 L층은 이론적 수용 영역의 너비가 2L+1이다. 그러나 같은 격자의 한 모서리 출력에서 반대 모서리 입력까지 연결하려면 각 층에서 최대 한 칸 이동하므로 $L\ge\max(H-1,W-1)$이 필요하다. 28×28에서는 27층이다. 코드의 14층 계산은 수용 영역의 너비가 28 이상이 되는 기준이며 모서리 간 연결 기준과 다르다.

다운샘플링이나 지수적 팽창률을 사용하면 더 적은 층으로 넓은 경계 범위를 얻을 수 있지만 해상도와 연결 밀도·좌표 대응을 함께 고려해야 한다. 특정한 격자 크기만으로 모든 설계의 정확한 층 수를 정할 수는 없다. [팽창 합성곱](../cnn/dilated_convolutions.md)도 관련 절충을 다룬다.

**학습 가능한 관련성**: 합성곱의 가중치는 상대 위치 $i - j$에만 달린 고정 값이라, 학습이 끝나면 어떤 입력에도 같은 가중치가 쓰인다. 비국소의 가중치는 $\theta(\mathbf{x}_i)^T\phi(\mathbf{x}_j)$이므로 **입력마다 다시 계산된다**. 같은 위치 쌍이라도 그림이 달라지면 다른 가중치를 받는다.

!!! danger "비국소 연산은 대칭이 아니다"
    "$i$과 $j$의 관계"라는 말 때문에 $f(\mathbf{x}_i, \mathbf{x}_j) = f(\mathbf{x}_j, \mathbf{x}_i)$일 것 같지만 그렇지 않다. 출력 `[3]`이 두 단계로 보여 준다. 첫째, $\theta \ne \phi$이므로 점수 행렬부터 대칭이 아니다($\max|S - S^\top| = 10.108$). 둘째, **설령 $\theta = \phi$으로 묶어 점수 행렬을 대칭으로 만들어도**($\max|S - S^\top| = $ 0.000e+00) 소프트맥스를 씌우면 대칭이 깨진다($\max|A - A^\top| = 0.175$). 정규화 인수 $C(\mathbf{x}) = \sum_j f(\mathbf{x}_i, \mathbf{x}_j)$이 행마다 다르기 때문이다. $A$은 대칭 행렬이 아니라 **행마다 합이 1인 행렬**이며, 출력 `[1]`이 그 성질을 잰다(어긋남 7.15e-07, float32 반올림). 곧 "$i$이 $j$를 얼마나 보는가"와 "$j$이 $i$를 얼마나 보는가"는 서로 다른 수다.

---

## 5. 컴퓨터 비전에서의 활용

앞 절이 공간 차원만 다루었지만 식 어디에도 차원이 둘이어야 할 이유는 없다. 지표 $i$을 시공간으로 넓히면 그대로 영상에 쓸 수 있다.

### 영상 이해

영상에서는 위치가 $(t, i, j)$이고, 합이 세 지표 모두를 훑는다.

$$\mathbf{y}_{t,i,j} = \frac{1}{C(\mathbf{x})} \sum_{t', i', j'} f\big(\mathbf{x}_{t,i,j}, \mathbf{x}_{t',i',j'}\big) \, g\big(\mathbf{x}_{t',i',j'}\big)$$

3차원 합성곱이 한 층에서 시간 축으로 커널 크기만큼(대개 3프레임)만 보는 데 비교하여, 비국소는 첫 프레임의 한 점과 마지막 프레임의 한 점을 한 층에서 잇는다. 대가는 그대로 $N = THW$에 실린다.

!!! warning "영상에서는 프레임 수가 곧바로 $N$에 곱해진다"
    출력 `[4]`의 마지막 세 줄이 이것이다. $28 \times 28$ 특징 맵이라도 32프레임을 함께 넣으면 $N = 25{,}088$이 되어 $A$ 하나가 2,401.0 MiB, 곧 2.34 GiB다. 표본 **하나**의 어텐션 행렬 하나가 그렇다. 그래서 영상 쪽 구현은 프레임을 4~8개로 끊거나 시간 축을 먼저 줄인 뒤에 비국소를 쓴다. [시간 어텐션](../../ch21/video/09_temporal_attention.md)과 [SlowFast 네트워크](../../ch21/video/10_slowfast_network.md)이 이 문제를 다루는 두 갈래다.

### 세밀한 인식

먼 부위의 특징을 직접 결합할 수 있어 세밀한 분류의 후보 연산이 된다. 그러나 비국소 가중치가 원하는 의미 관계를 실제로 학습했는지는 과제 성능과 추가 분석이 필요하다. 어텐션 값만으로 인과적 판단 근거를 확인할 수는 없다.

### 3차원 점구름 처리

불규칙한 점 집합에도 위치별 특징의 쌍 관계를 사용할 수 있다. 일반 이미지 합성곱의 규칙 격자를 그대로 적용하기 어렵지만 점·그래프·희소 합성곱 같은 다른 방법도 있다. 비국소 연산에서는 좌표·위치 인코딩과 순열 처리 방식을 과제에 맞춰 정해야 한다.

---

## 6. CNN 구조에 넣기

블록 위치를 정할 때에는 입력 위치 수, 채널 수와 처리할 문맥을 함께 고려한다. 낮은 해상도에 두면 밀집 어텐션 행렬은 작아지지만 채널 투영 비용과 이미 손실된 위치 정보도 고려해야 한다. 초기 특징의 전역 관계가 항상 무의미하거나 마지막 단계에서는 추가 효과가 없다고 단정하지 않는다.

잔차 출력 투영을 0으로 초기화하면 차원이 맞고 중간 계산이 유한한 조건에서 초기 순전파를 항등으로 유지할 수 있다. 적절한 위치와 개수, 성능 효과는 별도 검증으로 정한다.

---

## 7. 계산을 줄이는 전략

**병목 설계**: $\theta, \phi, g$의 출력 채널을 $C$가 아니라 $d = C/2$로 둔다. 유사도 항 $2N^2d$과 합성곱 항 $4NCd$이 **둘 다** $d$에 정비례하므로, 값은 정확히 반이 된다.

$$\text{MAC}(d) = 2N^2 d + 4NCd = 2d\,(N^2 + 2NC)$$

출력 `[5]`가 $C = 512$, $28 \times 28$에서 1.451 G와 0.726 G로 이것을 확인한다. 흔히 "$\ll$"로 적히지만 실제로 얻는 것은 **2배**이며, 그 이상을 얻으려면 $d$를 더 줄여야 한다.

**공간 다운샘플링**: $\phi$과 $g$ 뒤에만 보폭 $r$의 최대 풀링을 두어 $j$이 훑는 위치를 $N$에서 $N/r^2$으로 줄인다. $i$의 개수는 그대로이므로 출력 모양은 변하지 않는다.

$$\text{유사도·집계} = 2 \cdot N \cdot \frac{N}{r^2} \cdot d$$

$1 \times 1$ 합성곱 항 $4NCd$은 풀링이 합성곱 **뒤**에 오므로 줄지 않는다. $r = 2$이면 유사도 항이 4분의 1이 되고, 어텐션 행렬도 $N \times N$에서 $N \times N/4$로 줄어든다. 출력 `[5]`에서 $784 \times 784$가 $784 \times 196$이 되고 원소 수가 614,656개에서 153,664개로 정확히 4배 줄어든 것이 이것이다. 출력 모양은 `(2, 512, 28, 28)` 그대로다.

**윈도 분할**: 특징 맵을 겹치지 않는 영역으로 나누어 영역 안에서만 비국소 연산을 하고 합친다. $N$개 위치를 $k$개 영역으로 나누면 $N^2$이 $k \cdot (N/k)^2 = N^2/k$이 되지만, 영역을 가로지르는 관계는 그만큼 잃는다. 스윈 트랜스포머의 창 어텐션이 이 생각을 밀고 나간 것이다.

---

## 8. 관련 주제

- [Squeeze-and-Excitation](squeeze_excitation.md) — 같은 절의 짝. 채널에 문을 달아 $N$에 이차인 값을 치르지 않는 쪽
- [합성곱 신경망에서 트랜스포머로](cnn_to_vit_bridge.md) — 비국소 블록이 순수 CNN과 ViT 사이의 어느 단계인지
- [셀프 어텐션](../../ch11/attention/self_attention.md) — embedded Gaussian 형태의 비국소 연산이 곧 이것이다
- [스케일 내적 어텐션](../../ch12/transformer_architecture/scaled_dot_product.md) — 2절의 경고가 가리키는 차이
- [ViT 개요](../../ch12/transformers_vision/vit.md) — 합성곱을 다 걷어 내고 어텐션만 남기면 어디에 닿는지
- [수용 영역](../cnn/receptive_field.md) — 4절의 $2L + 1$이 나오는 곳
- [항등 사상](../residual/identity_mapping.md) — 3절의 잔차 형태가 기대는 성질
- [CBAM](../../ch21/image_classification/24_cbam.md) — 채널 어텐션과 공간 어텐션을 잇대어 쓰는 확장
- [시간 어텐션](../../ch21/video/09_temporal_attention.md) — 영상에서 $N = THW$을 감당하는 방법

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
$C = 512$, 특징 맵이 $28 \times 28$, 중간 차원 $d = C/2$일 때 비국소 블록의 매개변수 개수를 세어라. 편향까지 넣어라.

</div>

??? success "연습문제 1 풀이"
    $1 \times 1$ 합성곱이 넷이다.

    - $\theta, \phi, g$: 각각 $C \times d = 512 \times 256 = 131{,}072$개의 가중치와 $d = 256$개의 편향
    - $W_z$: $d \times C = 256 \times 512 = 131{,}072$개의 가중치와 $C = 512$개의 편향

    합하면 가중치 $4 \times 131{,}072 = 524{,}288$개, 편향 $3 \times 256 + 512 = 1{,}280$개, 모두 **525,568개**다. 출력 `[1]`이 `sum(p.numel() for p in block.parameters())`로 센 값과 같다.

    공간 크기 $28 \times 28$은 이 계산에 들어오지 않는다는 데 주의하라. 매개변수는 $4C d + 3d + C$으로 해상도와 무관하고, 해상도에 달린 것은 연산량과 메모리뿐이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
56×56 격자에서 3×3 보폭 1 합성곱의 수용 영역 너비가 격자 크기 이상이 되는 층 수와 모서리 간 연결 층 수를 구분하라.

</div>

??? success "연습문제 2 풀이"
    팽창률 1에서 너비는 2L+1이므로 너비 56 이상에는 L=28이 필요하다. 그러나 한 모서리 출력에서 반대 모서리 입력까지의 이동 거리는 각 축 55이므로 같은 해상도에서 L=55가 필요하다. 코드의 28은 첫 기준이다. 모든 위치를 집계하는 비국소 연산은 한 층에 그 위치를 포함하지만 실제 영향이 크다는 보장은 별개이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
공간 차원이 $H \times W$, 채널이 $C$, 중간 차원이 $C/2$일 때 비국소 블록과 같은 자리 $3 \times 3$ 합성곱($C \to C$)의 곱셈-덧셈 횟수를 적고, ResNet-50의 layer1과 layer3에서 비를 구하라. 어느 항이 어디서 이기는가?

</div>

??? success "연습문제 3 풀이"
    $N = HW$으로 두면 비국소는 유사도·집계에 $2N^2 \cdot \tfrac{C}{2} = N^2 C$, $1 \times 1$ 합성곱 넷에 $4NC \cdot \tfrac{C}{2} = 2NC^2$이 들어 합계 $N^2C + 2NC^2$이다. $3 \times 3$ 합성곱은 $9NC^2$이다. 비는 다음과 같다.

    $$\frac{N^2 C + 2NC^2}{9NC^2} = \frac{N}{9C} + \frac{2}{9}$$

    layer1은 $C = 256$, $N = 3136$이므로 $\dfrac{3136}{2304} + \dfrac{2}{9} = 1.361 + 0.222 = 1.583$이고, layer3은 $C = 1024$, $N = 196$이므로 $\dfrac{196}{9216} + \dfrac{2}{9} = 0.021 + 0.222 = 0.243$이다. 출력 `[4](b)`의 마지막 열과 같다.

    유사도·집계 항은 $N^2C$, 투영 항은 $2NC^2$이므로 전자가 더 크려면 $N>2C$이다. N>C만으로는 충분하지 않다. layer1의 3136>512에서는 전자가 크고 layer3의 196<2048에서는 투영이 크다. 표의 비용은 소프트맥스·편향·메모리 이동 등을 제외한 MAC 기준이다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
비국소 블록을 넣을 위치를 정할 때 연산량과 메모리를 어떻게 비교해야 하는가?

</div>

??? success "연습문제 4 풀이"
    N×N 행렬의 저장량은 배치와 N²에 비례하고 채널 투영은 NCd, 유사도·집계는 N²d에 비례한다. 입력·출력 특징과 역전파 저장 공간도 추가된다. 특정 MiB 값만으로 불가능하다고 정하지 말고 장치 예산, 실제 구현의 재계산·융합과 처리 시간을 확인한다. 해상도·채널 수와 필요한 문맥이 다른 위치의 품질도 검증해야 한다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span>
$\theta$이 만든 임베딩을 열로 늘어놓은 행렬을 $\Theta \in \mathbb{R}^{d \times N}$이라 하자. $\theta = \phi$으로 묶으면 점수 행렬 $S = \Theta^\top \Theta$은 대칭이 된다. 그런데도 어텐션 행렬 $A = \text{softmax}(S)$은 대칭이 아니다. 왜 그런지 보이고, $A$이 대칭이 될 필요충분조건을 구하라.

</div>

??? success "연습문제 5 풀이"
    $A_{ij} = \dfrac{e^{S_{ij}}}{\sum_{k} e^{S_{ik}}}$이고 $A_{ji} = \dfrac{e^{S_{ji}}}{\sum_{k} e^{S_{jk}}}$이다. $S$이 대칭이면 분자는 $e^{S_{ij}} = e^{S_{ji}}$으로 같지만, 분모는 $i$번째 행의 합과 $j$번째 행의 합이라 서로 다르다. 곧

    $$\frac{A_{ij}}{A_{ji}} = \frac{\sum_k e^{S_{jk}}}{\sum_k e^{S_{ik}}}$$

    대칭인 유한한 점수 행렬 $S$에 대해 $e^{S_{ij}}>0$이므로, $A$가 대칭일 필요충분조건은 모든 행의 정규화 합 $\sum_k e^{S_{ik}}$가 같은 것이다. 행의 원소 배열이 달라도 합이 같을 수 있다. 일반적인 학습 결과에서 이 조건이 보장되지는 않지만 특정 입력이나 매개변수에서는 성립할 수 있다.

    출력 `[3]`의 아래 두 줄이 이 계산을 확인한다. $\theta = \phi$으로 묶어 $\max|S - S^\top| = 0$을 만들어도 $\max|A - A^\top| = 0.175$이다. 소프트맥스는 행별 연산이므로 대칭성을 보존하지 않는다.

    이 때문에 비국소 블록은 "두 위치를 잇는 무향 간선"이 아니라 **방향 있는 간선**을 학습한다. 배경의 한 점이 사람의 손을 강하게 볼 수 있으면서 손은 그 배경 점을 거의 보지 않을 수 있다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
$W_z$의 가중치와 편향을 0으로 두면 블록이 정확히 항등이라는 것을 식으로 보여라. 또 $W_z$에 오는 기울기가 0이 아님을 보이고, 그래서 학습이 왜 멈추지 않는지 설명하라.

</div>

??? success "연습문제 6 풀이"
    **항등**: 블록은 $\mathbf{z} = W_z \mathbf{y} + \mathbf{x}$이다. $W_z$이 $1 \times 1$ 합성곱이므로 위치마다 $\mathbf{z}_i = W \mathbf{y}_i + \mathbf{b} + \mathbf{x}_i$이고, $W = 0$, $\mathbf{b} = 0$이면 $\mathbf{z}_i = \mathbf{x}_i$이다. $\mathbf{y}_i$이 무엇이든 상관없으므로 근사가 아니라 정확히 같다. 출력 `[2]`의 `torch.equal(zero(x), x) = True`이 이것이고, 기본 초기화에서는 같은 입력에 대해 최대 1.13만큼 어긋난다.

    **기울기**: 출력 투영 가중치 W의 기울기는 $\sum_i(\partial L/\partial z_i)y_i^\top$이다. W=0이어도 0이 아닐 수 있으나, 두 인자가 각각 0이 아니더라도 곱의 합이 상쇄될 수 있어 항상 0이 아니라고 보장할 수는 없다. 편향의 기울기도 상류 기울기의 합이다.

    **학습 가능성**: W=0이면 내부 y에 전달되는 기울기는 0이다. 출력 투영이 비영 기울기로 갱신되면 이후 내부 투영에도 학습 신호가 전달될 수 있다. 모든 입력·손실에서 첫 갱신이 반드시 W를 0에서 벗어나게 하는 것은 아니므로 실제 기울기를 확인한다.

---

## 정리하며

embedded Gaussian 비국소 블록은 모든 위치의 점수에 행별 소프트맥스를 적용해 값 특징을 결합한다. 점수 행렬이 대칭이어도 정규화 후에는 일반적으로 대칭이 아니다. 명시적 밀집 구현의 비용에는 $2N^2d+4NCd$ MAC과 N² 행렬 저장량이 포함된다. 출력 투영의 0 초기화는 조건을 만족할 때 항등 출력을 제공하지만 모든 기울기가 비영이라는 보장은 아니다. 해상도·채널·출력 위치와 비교 기준을 명시해야 한다.
