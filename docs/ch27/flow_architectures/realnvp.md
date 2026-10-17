# RealNVP와 아핀 커플링 레이어

RealNVP는 입력 좌표를 두 그룹으로 나누고 한 그룹에서 계산한 스케일·이동으로 나머지를 변환한다. 이 커플링 구조는 역변환을 명시적으로 구할 수 있고 로그 행렬식도 쉽게 계산하게 한다. [RealNVP 논문](https://arxiv.org/abs/1605.08803)은 이를 이미지 밀도 추정의 다중 스케일 구조로 확장했다. 여기서는 먼저 벡터용 핵심 구현을 살펴본다.

## 1. 커플링 변환과 야코비안

입력 u=(u_A,u_B)를 나눠 다음과 같이 변환한다.

$$v_A=u_A,\qquad v_B=u_B\odot\exp(s(u_A))+t(u_A).$$

출력에서 u_A=v_A를 그대로 얻으므로 역변환은 다음과 같다.

$$u_B=(v_B-t(v_A))\odot\exp(-s(v_A)).$$

s·t를 예측하는 조건 네트워크 자체는 가역적일 필요가 없다. 다만 역변환에서 같은 조건 입력에 같은 값을 계산해야 한다. 재현되지 않는 드롭아웃이나 배치에 따라 달라지는 상태를 아무 조치 없이 넣으면 표본별 가역 변환·밀도 계산의 계약을 깨뜨릴 수 있다.

좌표를 A,B 순서로 정렬하면 야코비안은 블록 삼각 행렬이다.

$$J=\begin{pmatrix}I&0\\ *&\operatorname{diag}(\exp(s(u_A)))\end{pmatrix},\qquad
\log|\det J|=\sum_j s_j(u_A).$$

조건 네트워크의 미분을 포함하는 왼쪽 아래 블록은 행렬식에 직접 필요하지 않지만 s의 값 자체는 행렬식에 영향을 준다. s·t의 계산비용까지 사라지는 것은 아니다. 네트워크 계산 뒤의 로그 스케일 합산이 변환 좌표 수에 대해 선형 시간이다.

## 2. 아핀 커플링 구현

```python
import torch
import torch.nn as nn
import numpy as np

class AffineCouplingLayer(nn.Module):
    """아핀 짝지음 레이어 하나."""

    def __init__(self, dim, mask, hidden_dims=(256, 256)):
        super().__init__()
        self.dim = dim
        self.register_buffer("mask", mask.float())
        d_cond = int(mask.sum().item())
        d_trans = dim - d_cond

        layers = []
        prev = d_cond
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, d_trans * 2))
        self.net = nn.Sequential(*layers)
        # 거의 항등인 첫자리매김
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _st(self, x_cond):
        params = self.net(x_cond)
        s, t = params.chunk(2, dim=-1)
        s = torch.tanh(s) * 2          # 안정을 위해 로그 잣수를 가둔다
        return s, t

    def forward(self, x):
        x_cond = x[:, self.mask.bool()]
        s, t = self._st(x_cond)
        y = x.clone()
        y[:, ~self.mask.bool()] = x[:, ~self.mask.bool()] * torch.exp(s) + t
        return y, s.sum(dim=-1)

    def inverse(self, y):
        y_cond = y[:, self.mask.bool()]   # x_cond과 같다
        s, t = self._st(y_cond)
        x = y.clone()
        x[:, ~self.mask.bool()] = (y[:, ~self.mask.bool()] - t) * torch.exp(-s)
        return x, -s.sum(dim=-1)
```

입력은 `(B,dim)`, 마스크는 dim 길이의 0·1 벡터여야 한다. 생성자는 mask.sum으로 조건 차원을 정하고 실제 인덱싱은 bool로 하므로 임의의 실수 마스크를 주면 차원이 맞지 않을 수 있다. 일반적인 다변량 분할에서는 조건·변환 그룹이 모두 비어 있지 않게 한다. mask는 버퍼로 등록되어 모델 장치 이동과 state_dict에 포함된다.

forward는 `(변환값,표본별 로그 절댓값 행렬식)`을 반환한다. inverse는 역변환의 로그 행렬식이므로 부호가 반대다. 마지막 선형 레이어의 웨이트·바이어스를 0으로 초기화해 s=t=0, 곧 항등 변환에서 시작한다.

본문은 로그 스케일에 2 tanh를 적용해 단일 레이어의 배율을 exp(−2)~exp(2)로 제한한다. 이는 모든 깊이의 누적 스케일이나 이동값까지 제한하는 것은 아니다.

```python
# 가둔 로그 잣수
s = torch.tanh(s_raw) * scale_bound      # 예컨대 scale_bound = 2

# 다른 길: 매끄럽게 양수로 만드는 부드러운 정류 선형
scale = torch.nn.functional.softplus(s_raw)
```

위 두 줄은 서로 다른 스케일 파라메터화의 예다. softplus 값을 직접 양의 배율 a로 쓴다면 변환에는 a, 로그 행렬식에는 log a를 사용해야 한다. 이를 s 자리에 그대로 대입해 exp(s)를 쓰는 것과 같지 않다. 양수 조건만으로 작은 배율의 수치 불안정이 사라지지도 않는다.

## 3. 마스크와 좌표 혼합

항상 같은 그룹만 변환하면 나머지 좌표는 끝까지 그대로 남는다. 번갈아 마스크를 쓰거나 가역 순열을 사이에 넣어 서로 다른 좌표가 조건·변환 역할을 맡게 한다.

```python
def alternating_mask(dim, layer_idx):
    mask = torch.zeros(dim)
    if layer_idx % 2 == 0:
        mask[:dim // 2] = 1
    else:
        mask[dim // 2:] = 1
    return mask
```

이 함수는 앞·뒤 그룹을 교대로 선택하며 dim이 홀수이면 그룹 크기도 번갈아 달라진다. 충분한 좌표 혼합은 표현에 중요하지만 유한 레이어가 모든 밀도를 정확히 표현하거나 모든 쌍의 의존성을 학습한다는 보장은 아니다. dim=1의 빈 조건 그룹은 일반적인 비선형 다변량 커플링의 효과를 주지 못한다.

## 4. 벡터 모델과 이미지 구조의 구분

원 논문의 이미지 설계에서는 바둑판·채널 마스크와 다중 스케일 처리를 결합한다. 아래는 개념적 블록 순서이며 이 페이지의 벡터 코드가 모두 구현하는 것은 아니다.

```
For each scale:
    For L coupling blocks:
        Affine coupling (checkerboard mask)
        Batch normalisation
    Affine coupling (channel mask)
    Factor out half channels → base distribution
```

```python
class RealNVP(nn.Module):
    """벡터 자료를 위한 RealNVP."""

    def __init__(self, dim, n_layers=8, hidden_dims=(256, 256)):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            mask = alternating_mask(dim, i)
            self.layers.append(AffineCouplingLayer(dim, mask, hidden_dims))

    def forward(self, x):
        ld = torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:
            x, log_det = layer.forward(x)
            ld += log_det
        return x, ld

    def inverse(self, z):
        ld = torch.zeros(z.shape[0], device=z.device)
        for layer in reversed(self.layers):
            z, log_det = layer.inverse(z)
            ld += log_det
        return z, ld

    def log_prob(self, x):
        z, ld = self.forward(x)
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
        return log_pz + ld

    def sample(self, n, device="cpu"):
        z = torch.randn(n, self.dim, device=device)
        x, _ = self.inverse(z)
        return x
```

이 RealNVP 클래스의 forward 방향은 **데이터 x→기저 z**다. log_prob는 다음 식을 계산한다.

$$\log p_X(x)=\log p_Z(f(x))+\log|\det Df(x)|.$$

기저는 dim차원 표준정규이고 sample은 z를 뽑아 inverse로 데이터 공간에 보낸다. inverse에서는 레이어 순서를 뒤집고 로그 행렬식을 더한다. 여기의 “정확한 우도”는 주어진 연속 모형의 변수 변환식을 근사 사후 없이 계산한다는 뜻이며 학습된 모형이 실제 분포와 정확히 같거나 부동소수점 오차가 없다는 뜻은 아니다.

이미지 다중 스케일 구조에서는 squeeze가 공간 위치를 채널로 재배열한다. 원소 수를 보존하는 가역 순열이므로 자체 로그 절댓값 행렬식은 0이다. factor-out은 일부 좌표를 더 깊은 변환에서 제외하고 기저 변수의 일부로 취급하며 버리는 연산이 아니다.

```
Input (3, 32, 32)
  → Squeeze → (12, 16, 16) → Coupling blocks → Factor out (6, 16, 16)
  → Squeeze → (24, 8, 8)   → Coupling blocks → Factor out (12, 8, 8)
  → Squeeze → (48, 4, 4)   → Coupling blocks → Base distribution
```

위 모양 예시는 원소 수를 보존한다. 분리된 6×16×16, 12×8×8과 최종 48×4×4를 합하면 원래 3×32×32와 같다. 어떤 계층이 자동으로 세부·의미 정보를 담당하는지는 보장되지 않는다. 본문 벡터 클래스에는 squeeze·factor-out·이미지용 정규화가 없다.

## 5. 학습과 수치 조건

```python
def train_realnvp(model, data, epochs=100, batch_size=256, lr=1e-3):
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data),
        batch_size=batch_size, shuffle=True,
    )
    for epoch in range(epochs):
        for (batch,) in loader:
            loss = -model.log_prob(batch).mean()
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
```

음의 평균 로그밀도를 최소화하는 최대우도 학습이다. DataLoader는 data의 장치를 자동 변경하지 않으므로 모델·배치의 장치와 자료형을 맞춰야 한다. 그래디언트 클리핑은 일부 불안정을 완화하지만 역변환 정확성·좋은 밀도·생성 품질을 보장하지 않는다. 검증·조기 종료·체크포인트·실험 결과는 이 함수에 없다.

sample의 기본 CPU와 기본 자료형도 모델과 맞춰야 하며 eval·no_grad가 자동 적용되지 않는다. ld 초기 텐서는 입력 dtype을 명시적으로 복제하지 않아 혼합 정밀도·다른 dtype에서는 누적 정밀도 폴리시도 확인해야 한다. 정수 이미지에 연속 밀도를 적용하려면 dequantization과 전처리 야코비안 등 별도 처리가 필요하다.

덧셈 커플링은 s=0인 특별한 경우로 단일 레이어의 부피를 보존한다. 아핀 커플링은 비자명한 스케일까지 허용하지만 구조 확장이 곧 유한 학습의 성능 개선을 뜻하지는 않는다. 가역성·계산비용·표현력의 절충은 [정규화 흐름 리뷰](https://jmlr.org/papers/v22/19-1028.html)에서 체계적으로 다룬다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 페이지의 forward 방향에서 변수 변환 공식과 부호를 설명하라.

</div>

??? success "풀이"

    z=f(x)이고 x=f⁻¹(z)이면 p_X(x)=p_Z(f(x))|det Df(x)|다. 따라서 데이터→기저 forward의 로그 행렬식은 log_prob에 더한다. 생성 방향의 로그 행렬식을 대신 사용하면 부호를 바꿔야 한다. 합성 변환은 각 레이어의 로그 절댓값 행렬식을 합한다. 연속 밀도는 좌표 단위·전처리에 의존하며 정수 이미지의 확률 질량과 동일하지 않다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
커플링 레이어의 역변환과 로그 행렬식이 쉬운 이유를 설명하라.

</div>

??? success "풀이"

    한 그룹이 그대로 남아 조건 네트워크 입력을 복구할 수 있다. 다른 그룹은 양의 배율과 이동이므로 성분별 역변환이 가능하다. 야코비안은 블록 삼각이고 변환 그룹의 대각은 exp(s)이므로 로그 행렬식은 Σs다. 조건 네트워크의 전체 야코비안을 만들 필요는 없지만 그 네트워크 계산비용과 수치 안정성은 여전히 중요하다. 이 코드는 실제로 각 표본에 결정적인 MLP를 사용한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
교대 마스크가 해결하는 제약과 해결하지 못하는 제약을 설명하라.

</div>

??? success "풀이"

    같은 조건 그룹을 계속 고정하면 해당 좌표가 영원히 항등 변환으로 남는다. 교대 마스크·순열은 서로 다른 좌표가 변환되게 한다. 그러나 유한 레이어 수와 조건 네트워크 용량·학습은 여전히 표현 한계를 가지며 좌표를 번갈아 바꾼다고 임의 분포를 정확히 맞추는 것은 아니다. 가역 선형 혼합 등 다른 방법도 각 연산의 로그 행렬식을 함께 계산해야 한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
흐름·VAE·GAN의 계산 특성을 성능 보장과 구분해 비교하라.

</div>

??? success "풀이"

    일반적인 같은 차원의 흐름은 기저 밀도와 가역 변환으로 로그밀도를 직접 계산하고 잠재 좌표를 결정적으로 복구한다. VAE는 보통 근사 사후와 ELBO로 학습하며 GAN은 보통 명시적인 계산 가능 밀도 없이 생성자를 학습한다. 그러나 생성 품질·안정성·의미 있는 보간의 우열은 구조·자료·학습 조건에 달려 있다. 모든 흐름에서 양방향이 같은 비용으로 병렬 계산되는 것도 아니다. RealNVP의 커플링은 양방향 명시적 계산을 제공하지만 같은 차원 가역성 등의 제약이 있다. 정확한 좌표 역변환을 VAE의 확률적 사후 추론과 동일한 문제로 읽지 않는다.



## 정리하며

RealNVP의 커플링은 한 그룹을 조건으로 다른 그룹을 선형 변환해 가역성과 쉬운 로그 행렬식을 얻는다. 데이터→기저 방향의 부호, 조건 네트워크 비용과 샘플링 계약을 확인한다. 벡터 예제와 이미지 다중 스케일 구조는 구분하며 계산 가능한 우도를 실제 분포 적합성·생성 품질의 보장으로 해석하지 않는다.

**참고 문헌**

1. Dinh, L., Krueger, D. & Bengio, Y. (2015). NICE: Non-linear Independent Components Estimation. *ICLR Workshop*.
2. Dinh, L., Sohl-Dickstein, J. & Bengio, S. (2017). Density Estimation Using Real-NVP. *ICLR*.
3. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.
