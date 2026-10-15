# VAE의 인코더 네트워크

VAE의 인코더는 입력마다 근사 사후 q_φ(z|x)의 파라미터를 계산한다. 이 절은 여러 입력에 하나의 추론 네트워크를 공유하는 **상각 추론(amortized inference)**과 대각 가우시안의 평균·로그 분산 헤드를 설명한다. 분포를 출력한다는 것만으로 참 사후의 정확한 추론이나 보정된 불확실성이 보장되지는 않는다.

## 1. 입력에서 분포 파라미터로

고전적인 표본별 변분 최적화는 각 x의 변분 파라미터를 따로 갱신한다. 상각 추론은 공유 네트워크로 이를 예측해 새 입력에도 한 번의 순전파로 초기 분포를 얻는다. 여기서 “상각”은 표본별 추론 비용을 공유 학습으로 나누어 부담한다는 뜻이다.

대각 가우시안의 출력은 μ(x)∈ℝᵏ와 ℓ(x)=logσ²(x)∈ℝᵏ다. 이 둘로 평균과 양의 대각 분산 expℓ을 정한다. k차원 결정적 코드와 비교하면 파라미터 출력 원소 수는 2k다. 일반 공분산·혼합·흐름 사후를 사용하면 필요한 출력과 계산이 달라진다.

```
Input x ──► [Shared Hidden Layers] ──┬──► fc_mu ──► μ ∈ ℝ^d
                                      │
                                      └──► fc_logvar ──► log(σ²) ∈ ℝ^d
```

## 2. MLP 인코더와 두 출력 헤드

```python
import torch
import torch.nn as nn

class MLPEncoder(nn.Module):
    """
    벡터 들임(예컨대 펼친 MNIST)을 위한 온전히 이어진 인코더.
    
    구조: input_dim → hidden_dim → hidden_dim → (μ, logvar)
    """
    
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=32):
        super().__init__()
        
        # 공유 특징 추출
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 분포 매개변수를 위한 따로 둔 머리
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x):
        """
        인수:
            x: 들임 텐서 [배치 크기, 들임 차원]
        반환값:
            mu: 평균 [배치 크기, 숨은 차원]
            logvar: 로그 흩어짐 [배치 크기, 숨은 차원]
        """
        h = self.shared(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
```

기본 `MLPEncoder`는 B×784 → B×256 → B×256 특징을 만들고 두 선형 헤드로 B×32 배열 두 개를 반환한다. 평균과 로그 분산은 모두 실수값 파라미터이므로 마지막 선형 출력에 별도 활성화를 두지 않은 설계다. 두 헤드를 합쳐 2k출력 선형층 하나를 만든 뒤 쪼개는 것도 같은 파라미터화를 표현할 수 있다.

다만 두 헤드로 바꾸는 것만으로 생성 모델이 완성되지는 않는다. 사전·관측 분포와 재매개변수화, 학습 목표도 함께 정해야 한다. 공통 은닉 너비 h에서 한 k출력 헤드를 추가하면 바이어스를 포함해 (h+1)k개 파라미터가 늘어난다.

## 3. 합성곱 인코더의 모양 조건

```python
class ConvEncoder(nn.Module):
    """
    그림 들임을 위한 누비기 인코더.
    
    공간 차원을 차츰 내림 표집하면서
    채널 깊이를 늘린 뒤 숨은 매개변수로 쏜다.
    """
    
    def __init__(self, in_channels=1, hidden_channels=32, latent_dim=32):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # 28x28 → 14x14
            nn.Conv2d(in_channels, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels),
            
            # 14x14 → 7x7
            nn.Conv2d(hidden_channels, hidden_channels * 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels * 2),
            
            # 7x7 → 4x4
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels * 4),
        )
        
        # 펼치고 쏜다
        flat_dim = hidden_channels * 4 * 4 * 4  # 들임 크기에 달렸다
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)
    
    def forward(self, x):
        """
        인수:
            x: 들임 그림 [배치 크기, 채널, 높이, 너비]
        """
        h = self.conv_layers(x)
        h = h.view(h.size(0), -1)  # 펼친다
        return self.fc_mu(h), self.fc_logvar(h)
```

기본 입력 B×1×28×28은 stride 합성곱으로 B×32×14×14 → B×64×7×7 → B×128×4×4가 된다. 펼친 크기는 2048이며 각 헤드가 B×32를 반환한다. `flat_dim`은 마지막 공간 크기를 4×4로 하드코딩하므로 임의 해상도에 자동으로 맞는 구조는 아니다. 32×32도 이 구성에서는 4×4가 되지만 다른 크기는 직접 계산해야 한다.

BatchNorm은 학습 중 표본들을 결합하는 통계를 사용하고 평가 때 기본적으로 실행 통계를 사용한다. 입력별 독립 사후라는 단순한 설명과 배치 의존 동작을 구분한다. 활성화·정규화 순서도 설계 선택이며 같은 이름의 CNN이라도 계산이 달라질 수 있다.

## 4. 로그 분산과 안정화

로그 분산은 양수 제약 없이 선형층으로 출력할 수 있고 표준편차는 exp(ℓ/2)로 만든다. 대안으로 softplus 등 양수 변환을 사용할 수도 있다. 로그 출력도 극단값의 지수 계산에서 overflow·underflow가 생길 수 있어 무조건 안정적인 것은 아니다.

```python
def safe_reparameterize(mu, logvar, max_logvar=10.0):
    """가둔 로그 흩어짐으로 다시 매개변수화한다."""
    logvar = torch.clamp(logvar, min=-max_logvar, max=max_logvar)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps
```

`safe_reparameterize`는 샘플링에 사용할 ℓ을 [−max_logvar,max_logvar]로 자른다. 입력 인수의 유효성·dtype을 확인해야 하며, clamp 구간 밖에서는 그 경로의 기울기가 0이 된다. KL을 원래의 자르지 않은 ℓ로 계산하면 샘플 분포와 KL이 설명하는 분포가 달라진다. 실제 목적에 사용한 동일 분포를 일관되게 계산해야 한다.

## 5. 근사 간격과 상각 간격

고정된 생성 파라미터 θ와 변분 계열 Q에서 표본별 최적 ELBO를 L_Q*(x)=sup_{q∈Q}L(θ,q;x)로 두면 상각 간격은 다음과 같다.

$$L_Q^*(x)-L(\theta,q_\phi(\cdot|x);x)\ge0.$$

참 로그우도와의 전체 간격에는 Q 자체가 참 사후를 표현하지 못하는 근사 간격도 있다. 더 깊은 인코더는 파라미터 예측 함수를 유연하게 하지만 대각 가우시안이라는 분포 계열의 한계는 유지한다. 흐름 사후는 Q 자체도 바꾸므로 상각 간격만 줄이는 조치라고 부를 수 없다. [추론의 부분 최적성을 분석한 연구](https://proceedings.mlr.press/v80/cremer18a.html)는 두 요인을 나누어 설명한다.

표본별 파라미터를 몇 번 다듬어 ELBO가 올랐다면 초기 인코더 출력보다 나은 값이 있음을 알 수 있다. 유한 최적화가 최적 q를 찾았다는 보장은 없어 그 차이를 정확한 상각 간격으로 확정하지 않는다. 고정 θ, 독립 평가 표본과 같은 Monte Carlo 조건을 사용해야 한다.

## 6. 초기화·구조·평가

```python
# 첫 흩어짐이 웬만하도록 로그 흩어짐 머리의 치우침을 첫자리매김한다
nn.init.constant_(encoder.fc_logvar.bias, -1.0)
```

로그 분산 헤드의 바이어스를 −1로 두는 조각은 출력의 상수 항만 바꾼다. 가중치가 0이 아니면 모든 입력의 ℓ이 −1이 되지는 않는다. μ=0,분산 1로 시작하려면 두 헤드의 전체 출력과 초기화 조건을 맞추어야 한다.

출력 헤드 뒤의 활성화나 BatchNorm은 선택한 분포 파라미터화·배치 의존성을 바꿀 수 있어 의도를 명시한다. 모든 경우에 금지된 연산은 아니며, 제한된 μ가 있어도 가우시안 샘플의 지지집합은 분산이 양수이면 전체 실수 공간이다. 은닉·잠재 너비 비율에 모든 데이터에 통하는 2~4배 규칙도 없다.

학습에는 보통 재매개변수화 표본으로 기대 복원 항을 추정한다. 평가 시 μ를 디코딩하면 결정적 복원 진단이지만 E_q[g(z)]나 ELBO의 기대값과 같지 않다. `eval()`은 사용자 정의 샘플링을 자동으로 끄지 않는다. 분류 특징으로 μ, μ·분산 결합 또는 표본 평균을 쓸지는 태스크에서 검증한다. 사후 분산은 잠재 추론의 양이며 분류 확신도·분포 밖 이상 점수로 자동 보정되는 것은 아니다.

## 연습문제


<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
대각 가우시안 인코더의 출력 개수와 추가 헤드의 파라미터 수를 구하라.

</div>

??? success "연습문제 1 풀이"

    k차원에 평균과 로그 분산이 필요해 표본당 2k개를 출력한다. h입력 k출력 선형 헤드는 hk+k개이므로 h=256,k=16이면 4112개가 추가된다. 아래 self.head 조각은 클래스 문맥에 넣고 h,k를 정의해야 한다. 이 출력 변경만으로 생성 모형·학습이 완성되지는 않는다.

    ```python
    self.head = nn.Linear(256, 2 * k)     # mu 와 logvar
    mu, logvar = self.head(h).chunk(2, dim=1)
    ```


---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
재매개변수화의 미분과 다른 추정 방식의 존재를 설명하라.

</div>

??? success "연습문제 2 풀이"

    독립 잡음 ε에서 z=μ+σ⊙ε로 두면 경로별 미분이 μ에 대해 1, σ에 대해 ε다. 일반 sample 호출과 이 경로를 구분한다. 다른 점수 함수 추정기나 이산 변수의 완화 등도 있어 모든 샘플링의 미분이 원리적으로 불가능한 것은 아니다. 해당 분포·추정기의 편향과 분산을 확인해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
평가에서 평균 코드와 사후 샘플을 사용할 때 계산하는 양은 어떻게 다른가?

</div>

??? success "연습문제 3 풀이"

    g(μ)는 결정적 대표 복원이고 E_q[g(z)]와 일반적으로 다르다. 기대 로그우도와 ELBO를 추정하려면 그 기대값에 맞는 샘플링·평균을 사용한다. 샘플을 쓰는 평가도 고정 시드와 반복으로 재현성을 관리할 수 있다. 샘플링을 끄더라도 해석적 KL이 분산에 의존한다는 사실은 남는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
입력과 무관한 고정 σ와 학습 가능한 공유 σ의 차이를 설명하라.

</div>

??? success "연습문제 4 풀이"

    고정 σ에서는 KL의 분산 부분이 상수라 μ에 대한 L2와 상수로 정리된다. σ도 공유 파라미터로 학습하면 분산 항의 그래디언트가 남고 복원 샘플링에도 영향을 준다. 입력별 σ는 더 유연하지만 큰 값이 반드시 애매한 숫자나 낮은 분류 확신도를 뜻하지는 않는다.



---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
인코더 깊이와 분포 계열 변경이 해결하는 간격을 구분하라.

</div>

??? success "연습문제 5 풀이"

    깊이는 입력에서 같은 계열의 파라미터를 예측하는 함수를 바꾼다. 대각 가우시안의 구조 한계는 남는다. 흐름·혼합 등은 계열을 바꿔 근사 간격도 바꾼다. 고정 생성 모델에서 유한 표본별 refinement로 얻은 ELBO 개선은 정확한 전역 상각 간격 자체는 아니다.



---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
재매개변수화 그래디언트의 변동이 σ만으로 정해지는가?

</div>

??? success "연습문제 6 풀이"

    디코더·손실의 도함수, 입력과 잡음·미니배치·표본 수·파라미터화가 함께 영향을 준다. 큰 σ가 항상 큰 그래디언트 분산을 만든다고 단정할 수 없다. clamp는 분포와 기울기를 바꾸므로 KL과 샘플링의 일관성을 유지하고 안정화 효과를 검증한다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
인코더와 디코더의 상대 크기를 정하는 기준을 설명하라.

</div>

??? success "연습문제 7 풀이"

    인코더 추론 오차와 디코더 표현력, 학습 비용·샘플 품질·후속 태스크를 함께 평가한다. 차원 수가 늘어나는 방향이라는 이유만으로 디코더의 문제가 항상 더 어렵거나 더 크게 해야 하는 것은 아니다. 강한 디코더와 사후 붕괴의 관계도 진단할 수 있다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
2k출력 하나와 k출력 두 개의 동등성 및 아래 코드의 실행 한계를 설명하라.

</div>

??? success "연습문제 8 풀이"

    같은 입력에서 선형 가중치와 바이어스를 행 방향으로 붙이거나 나누면 같은 두 출력을 표현한다. 실제 파라미터 수는 같고 속도는 환경에 달려 있다. 아래 nn.Linear를 호출 시 새로 만드는 조각은 등록·장치·dtype과 가중치 보존이 없어 완성된 학습 모듈이 아니다. __init__에 계층을 등록해야 한다.

    ```python
    # 방식 1: 한 층에서 2k개를 내고 쪼갠다
    h = nn.Linear(256, 2 * k)(x)
    mu, logvar = h.chunk(2, dim=1)

    # 방식 2: 층을 둘 둔다
    mu     = nn.Linear(256, k)(x)
    logvar = nn.Linear(256, k)(x)
    ```


---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff easy" title="쉬움"></span>
μ나 logvar에 출력 활성화를 걸면 항상 VAE 정의를 위반하는가?

</div>

??? success "연습문제 9 풀이"

    아니다. 사후 계열을 제한하는 설계일 수 있다. 예를 들어 μ의 ReLU는 비음수 평균을 제한하지만 양의 분산 가우시안의 샘플 지지집합까지 비음수로 만들지 않는다. 보통은 실수 선형 헤드를 사용하되 제한 목적과 KL·샘플링의 분포 정의를 일관되게 유지한다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
VAE의 μ와 σ를 분류·이상 탐지에 활용할 때 필요한 검증을 설명하라.

</div>

??? success "연습문제 10 풀이"

    μ는 결정적 특징으로 유용할 수 있지만 항상 최선은 아니다. 분산 결합·표본 평균 등도 별도 태스크에서 비교할 수 있다. 큰 σ를 곧 낮은 분류 확신도나 이상이라고 해석하지 말고 독립 레이블·분포 변화에 대한 보정과 분리 성능을 평가한다.



## 정리하며

VAE 인코더는 입력별 사후 파라미터를 공유 네트워크로 계산한다. 출력 차원·공분산 해석과 샘플링·KL의 일관성을 맞추고 상각 간격과 분포 계열의 근사 한계를 구분한다. 평균 복원·확률 평가와 불확실성 활용은 서로 다른 조건에서 검증해야 한다.
