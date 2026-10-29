# VAE의 사전 분포 선택

잠재 사전 p(z)는 생성 시 잠재 표현을 뽑는 분포이며 학습 ELBO의 KL 기준이 된다. 표준정규, 혼합모형과 학습 가능한 사전은 서로 다른 계산·표현 조건을 제공한다. 복잡한 사전이 항상 더 좋은 생성 결과를 주거나 단순한 사전이 데이터의 여러 클래스를 표현하지 못하는 것은 아니다.

## 1. 사전과 집계 사후의 역할

학습 중 디코더는 주로 데이터와 근사 사후 q_φ(z|x)에서 온 잠재 표현을 보고, 생성에서는 p(z)를 사용한다. 집계 사후 q(z)=E_xq_φ(z|x)와 사전이 크게 다르면 생성에서 덜 학습된 영역을 방문할 수 있다. 그러나 분포 불일치가 모든 나쁜 출력의 유일한 원인은 아니며 디코더 표현·학습과 관측 모형도 영향을 준다.

기대 KL은 다음 두 항으로 나뉜다.

$$\mathbb E_xD_{KL}(q(z|x)\|p(z))=I_q(X;Z)+D_{KL}(q(z)\|p(z)).$$

따라서 표준 VAE가 집계 불일치를 전혀 벌하지 않는다는 설명은 틀리다. 입력 정보 비용도 함께 벌하므로 충분한 표현과 사전 일치를 동시에 달성하는 것은 절충 문제다. KL을 크게 가중하면 정보만 줄어드는지, 불일치도 줄어드는지는 실제로 진단해야 한다.

## 2. 표준정규 사전의 이점과 한계

N(0,I)는 별도 학습 파라미터 없이 쉽게 샘플링하고, 대각 가우시안 사후와의 KL을 해석적으로 계산할 수 있다. 등방성은 잠재 좌표에서 특정 방향을 선호하지 않는 사전의 성질이지 모든 인코더·디코더가 회전 대칭이라는 뜻은 아니다. 가우시안의 최대 엔트로피 성질도 고정 평균·공분산 등 명시한 제약 아래의 결과다.

단일 잠재 가우시안과 비선형 디코더를 결합하면 데이터 공간의 여러 모드를 표현할 수 있다. 표준정규를 고른다고 실제 데이터가 단봉·구 대칭이라고 가정하는 것은 아니다. 다만 유한한 디코더·사후 계열·학습 조건에서 더 유연한 사전이 집계 잠재 표현에 잘 맞을 여지는 있다.

## 3. 이른바 prior hole

사전은 상당한 질량을 두지만 집계 사후는 상대적으로 드문 영역을 prior hole이라고 부르기도 한다. 양의 분산 가우시안 사후를 유한 혼합하면 수학적으로 밀도가 정확히 0인 영역은 없으므로 여기서 “구멍”은 보통 상대적인 저밀도를 뜻한다.

```
Prior p(z):              Aggregated q(z):        Gap:
   ┌─────────┐             ┌─────────┐          ┌─────────┐
   │  ░░░░░  │             │         │          │  ░░░░░  │
   │ ░░░░░░░ │             │  ▓▓ ▓▓  │          │ ░░   ░░ │
   │ ░░░░░░░ │      -      │  ▓▓ ▓▓  │    =     │ ░░   ░░ │
   │ ░░░░░░░ │             │         │          │ ░░░░░░░ │
   │  ░░░░░  │             │         │          │  ░░░░░  │
   └─────────┘             └─────────┘          └─────────┘
   Smooth bell          Clustered data          Decoder unseen
```

위 그림은 개념적 도식이며 밀도의 단순 뺄셈이 정확한 실패 확률을 주는 것은 아니다. 사전 표본과 q 표본의 밀도·이웃·판별 성능을 비교할 수 있지만 μ(x)만 모으면 실제 q(z) 표본과 다른 분포를 평가하게 된다.

생성 이미지의 분류기 확신도는 간접적인 품질 자료일 뿐 구멍의 위치·질량이나 진짜 유효률을 직접 측정하지 않는다. [앞 장의 기존 기록](../../ch25/limits/latent_sampling.md)에서 0.2%와 57.4%는 최대 분류 확률 임계를 넘은 비율이며 그 반대를 생성 실패율로 읽어서는 안 된다.

## 4. 가우시안 혼합 사전

웨이트 π_k≥0, Σπ_k=1과 양의 공분산을 사용하면 다음 사전을 정의할 수 있다.

$$p(z)=\sum_{k=1}^{K}\pi_k\mathcal N(z;\mu_k,\Sigma_k).$$

성분 수는 잠재 분포의 모델링 선택이며 데이터 클래스 수와 같아야 할 이유는 없다. 혼합의 로그밀도는 성분별 로그밀도에 로그 웨이트를 더한 뒤 logsumexp로 안정적으로 계산한다. 가우시안 q와 혼합 p 사이 KL은 일반적으로 간단한 닫힌 형태가 없어 E_q[log q−log p] 등을 추정한다.

```python
import torch
import torch.nn as nn

class GaussianMixturePrior(nn.Module):
    """배울 수 있는 정규 섞기 사전 분포."""
    
    def __init__(self, latent_dim, num_components=10):
        super().__init__()
        self.num_components = num_components
        
        # 배울 수 있는 섞기 파라메터
        self.logits = nn.Parameter(torch.zeros(num_components))
        self.means = nn.Parameter(torch.randn(num_components, latent_dim) * 0.5)
        self.logvars = nn.Parameter(torch.zeros(num_components, latent_dim))
    
    def log_prob(self, z):
        """섞기 아래에서 log p(z)을 셈한다."""
        # z: [배치, 숨은 차원]
        # 퍼뜨리려 넓힌다: [배치, K, 숨은 차원]
        z_exp = z.unsqueeze(1)
        
        # 성분마다의 로그 확률
        log_var = self.logvars.unsqueeze(0)
        means = self.means.unsqueeze(0)
        
        log_p_per_component = -0.5 * (log_var + (z_exp - means).pow(2) / log_var.exp())
        log_p_per_component = log_p_per_component.sum(dim=2)  # [배치, K]
        
        # 무게로 섞는다
        log_weights = torch.log_softmax(self.logits, dim=0)
        log_p = torch.logsumexp(log_p_per_component + log_weights, dim=1)
        
        return log_p
    
    def sample(self, num_samples):
        """섞기 사전 분포에서 뽑는다."""
        # 성분을 고른다
        weights = torch.softmax(self.logits, dim=0)
        indices = torch.multinomial(weights, num_samples, replacement=True)
        
        # 고른 성분에서 뽑는다
        means = self.means[indices]
        stds = torch.exp(0.5 * self.logvars[indices])
        
        return means + stds * torch.randn_like(means)
```

이 클래스의 `log_prob`에는 가우시안 정규화 상수 **−(d/2)log(2π)**가 빠져 있다. 따라서 반환값은 정규화된 로그밀도보다 (d/2)log(2π)만큼 크다. 같은 차원에서 특정 파라미터 그래디언트는 바뀌지 않을 수 있지만 KL·ELBO의 절대값과 확률 해석은 틀어진다. 표준 가우시안 q의 정확한 log_prob에서 이것을 빼면 KL이 잘못 작아지거나 음수가 될 수 있다.

`sample`은 웨이트로 성분을 고른 뒤 해당 평균·로그 분산에서 대각 가우시안 표본을 만든다. z는 B×d라는 전제가 있고 차원·성분 수·장치·분산의 수치 범위를 검증해야 한다. 정의된 함수를 VAE 로스와 생성 루프에 연결하는 코드는 이 페이지에 없다.

## 5. VampPrior

[VampPrior](https://proceedings.mlr.press/v84/tomczak18a.html)는 학습 가능한 의사 입력 u_k에서 인코더가 만든 사후들의 혼합을 사전으로 사용한다.

$$p(z)=\frac1K\sum_{k=1}^{K}q_\phi(z|u_k).$$

이는 사전과 인코더를 공유 파라미터로 연결하는 선택이며 집계 사후와 정확히 같아진다는 보장은 없다. 의사 입력 수가 표본 수보다 작다는 것만으로 암기나 과적합이 불가능해지는 것도 아니다.

```python
class VampPrior(nn.Module):
    """VampPrior: 사후 분포를 변분으로 섞기."""
    
    def __init__(self, encoder, input_dim, num_pseudoinputs=100):
        super().__init__()
        self.encoder = encoder
        self.pseudoinputs = nn.Parameter(torch.randn(num_pseudoinputs, input_dim) * 0.05)
    
    def get_prior_params(self):
        """가짜 들임에서 섞기 성분 파라메터를 얻는다."""
        with torch.no_grad():
            mu, logvar = self.encoder(torch.sigmoid(self.pseudoinputs))
        return mu, logvar
```

이 `get_prior_params`는 **no_grad 안에서** 의사 입력을 인코더에 넣는다. 따라서 반환한 μ·logvar로 사전 로스를 계산해도 이 경로에서 `pseudoinputs`와 인코더로 그래디언트가 흐르지 않는다. 학습 가능한 VampPrior를 완성한 구현이 아니다. 혼합 log_prob와 sample도 없다. `sigmoid(pseudoinputs)`는 [0,1] 벡터 입력을 가정하므로 실제 인코더의 입력 모양·전처리와도 맞추어야 한다.

## 6. 흐름 사전과 조건부 사전

가역 미분 가능 변환 z=f(z₀), z₀∼p₀에서 다음이 성립한다.

$$\log p(z)=\log p_0(f^{-1}(z))+\log|\det J_{f^{-1}}(z)|.$$

역변환과 로그 행렬식을 계산할 수 있게 설계하면 사전 밀도와 샘플링을 다룰 수 있다. 그렇다고 가우시안 q에 대한 KL 기대값까지 자동으로 닫힌 형태인 것은 아니다. 변환 비용과 추정 변동도 포함해 평가한다.

조건부 모델은 p(z|y)와 p_θ(x|z,y), q_φ(z|x,y)를 사용할 수 있다. 조건부 사전은 조건별 잠재 표현 분포를 바꾸고, 조건부 디코더는 같은 잠재 표현을 조건에 따라 다르게 해석한다. 어느 하나가 클래스별 잠재 표현의 완전 분리나 클래스·스타일 독립을 보장하지 않는다. 조건을 실제로 활용하는지 별도 검증한다.

## 7. 사전 스케일과 평가 계획

p(z)=N(0,s²I)의 양수 s는 이상적으로 잠재 단위 변환으로 흡수할 수 있다. z′=sz에 맞춰 사후 평균과 표준편차를 s배, 로그 분산을 2log s만큼 이동하고 디코더 입력 웨이트를 1/s로 바꾸면 같은 분포를 표현할 수 있다. 하지만 웨이트 감쇠·아키텍처 제약·초기값·최적화가 있으면 실제 학습까지 같지는 않다. KL의 상대 무게 β를 바꾸는 것과도 다르다.

사전을 비교할 때는 같은 관측 분포·로스 정규화·데이터 분할과 학습 예산을 기록한다. 보류 ELBO 또는 일관된 우도 추정, 품질·다양성과 입력 정보 활용도를 함께 본다. 잠재 두 표본 판별기를 쓸 때는 독립 평가와 검정력, 균형 레이블을 확인한다. 판별 정확도 50%가 분포 일치의 증명은 아니며 가까운 이웃 거리도 차원·표본 수·척도에 의존한다.

## 연습문제


<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
표준정규 사전의 편의와 그 편의가 보장하지 않는 것을 설명하라.

</div>

??? success "풀이"

    학습할 사전 파라미터가 없고 샘플링과 가우시안 사후 KL 계산이 쉽다. 그러나 집계 사후 일치나 좋은 생성 품질·최적 사전이라는 보장은 없다. 비선형 디코더는 단순 잠재에서 다봉 데이터 분포를 표현할 수 있다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
prior hole을 정확한 0 밀도와 상대적 저밀도로 구분하라.

</div>

??? success "풀이"

    양의 분산 가우시안 혼합은 보통 전 실수 공간에 양의 밀도를 둔다. 여기 구멍은 사전에 비해 집계 사후가 상대적으로 드문 영역을 뜻한다. 그 영역에서 나쁜 생성이 생길 수 있지만 확신도 임계의 실패가 모두 이 원인이라는 증거는 아니다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
기대 사전 KL의 분해가 말하는 절충을 설명하라.

</div>

??? success "풀이"

    E_xKL(q(z|x)||p(z))=I_q(X;Z)+KL(q(z)||p(z))다. 집계 불일치도 직접 포함되므로 VAE가 이를 전혀 벌하지 않는 것은 아니다. 정보량과 불일치를 함께 줄이려 하므로 어느 항이 바뀌는지 따로 진단한다. 정보량을 낮추면 항상 잠재 표현이 쓸모없다는 극단적 결론도 피한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
혼합 사전의 log_prob를 확인할 때 빠진 상수가 중요한 이유를 설명하라.

</div>

??? success "풀이"

    가우시안 정규화 상수 −d log(2π)/2가 필요하다. 이를 빼먹으면 현재 코드의 log_prob가 그만큼 커져 정확한 KL과 ELBO 수치가 아니다. logsumexp 안정화와 웨이트 정규화만으로 이 상수 누락이 해결되지는 않는다.



---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
사전 표준편차 s를 바꾸는 단위 변환에 필요한 사후·디코더 변경을 적어라.

</div>

??? success "풀이"

    양수 s에 대해 사후 평균·표준편차를 s배 하고 로그 분산에 2log s를 더하며 디코더 첫 입력 웨이트를 1/s로 바꾸면 이상적인 리파라메리제이션을 만들 수 있다. 실제 정규화·최적화·구조 제약에서는 동등하지 않을 수 있다. β는 두 로스의 상대 무게라 같은 변화가 아니다.



---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
활성 차원 수로 사전의 적절성이나 내재 차원을 곧바로 판단할 수 있는가?

</div>

??? success "풀이"

    활성 임계·KL 무게·사후 계열·관측 분포와 학습 경로에 따라 수가 달라진다. 비활성이 필요한 용량 선택인지 붕괴인지 별도로 확인해야 한다. 더 많은 차원이 재구성을 못 개선했다는 한 결과만으로 적정 차원이나 원인을 확정할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff hard" title="어려움"></span>
사전과 집계 사후의 차이를 진단하는 평가를 설계하라.

</div>

??? success "풀이"

    실제 q(z|x) 샘플과 p(z) 샘플을 모아 독립 두 표본 판별이나 적절한 밀도·거리 검사를 할 수 있다. 평균 μ만 쓴 분포와 혼동하지 않는다. 표본 수·고차원 척도와 검정력을 맞추고 디코딩 품질·다양성은 별도로 평가한다. 판별 실패나 최근접 거리 하나가 완전 일치를 증명하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
학습 가능한 사전이 집계 사후와 같아지면 정보 패널티도 사라지는가?

</div>

??? success "풀이"

    아니다. 집계 KL이 0이면 기대 사전 KL에는 I_q(X;Z)가 남는다. 유연한 사전이 과적합할 가능성은 있지만 자동으로 재구성 0이나 학습 잠재 표현의 델타 혼합이 되는 것은 아니다. VampPrior도 작은 의사 입력 수만으로 암기를 금지하지 않으며 현재 코드는 no_grad 때문에 그 입력의 학습 경로부터 끊겨 있다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff easy" title="쉬움"></span>
조건부 사전과 조건부 디코더의 효과 및 보장되지 않는 분리를 설명하라.

</div>

??? success "풀이"

    p(z|y)는 조건별 잠재 표현 생성 분포를, p(x|z,y)는 복원·생성의 조건 해석을 바꾼다. 둘을 함께 또는 따로 쓸 수 있다. 조건별 사전이 반드시 분리되거나 디코더에 y를 넣으면 잠재 표현에서 클래스 정보가 완전히 사라진다는 보장은 없다. 독립성과 속성 제어는 별도 목적·검증이 필요하다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
나쁜 생성 결과에서 사전 변경을 검토하기 전에 무엇을 확인하는가?

</div>

??? success "풀이"

    관측 분포와 출력 표시, 복원·KL 합산, 모델 학습과 사후 붕괴·데이터 전처리를 먼저 점검한다. 흐릿함·낮은 다양성·얼룩은 여러 원인이 가능하다. 표준 사전과 유연한 사전을 같은 조건에서 비교하고 조건부 태스크는 실제 사용 가능한 조건과 목적에 맞추어 선택한다.



## 정리하며

사전은 생성 잠재 표현 분포와 ELBO의 기준을 정한다. 표준정규의 단순성, 혼합·VampPrior·흐름의 유연성과 비용을 비교하되 구현의 정규화 상수와 그래디언트 경로를 확인한다. 집계 사후 불일치·입력 정보·생성 품질은 관련되지만 같은 지표가 아니다.
