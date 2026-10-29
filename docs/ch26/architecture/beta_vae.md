# β-VAE: KL 웨이트와 표현의 절충

β-VAE는 재구성 로스와 KL의 상대적인 비중을 조절한다. 인코더·디코더 구조를 유지한 채 잠재 정보의 비용을 바꾸므로 재구성, 사전 일치와 표현 해석에 영향을 줄 수 있다. β를 키운다고 각 잠재 축이 회전·굵기 같은 의미 요인 하나씩을 반드시 학습하는 것은 아니다.

## 1. 구현

```python
"""얽힘 풀린 나타냄을 배우는 베타 변분 오토인코더."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BetaVAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=32, beta=4.0):
        super().__init__()
        self.input_dim, self.latent_dim, self.beta = input_dim, latent_dim, beta
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def loss_function(self, reconstruction, x, mu, logvar):
        BCE = F.binary_cross_entropy(reconstruction, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + self.beta * KLD, BCE, KLD

    def traverse_latent_dimension(self, dim_idx, num_steps=10, range_limit=3.0, device='cpu'):
        z = torch.zeros(num_steps, self.latent_dim).to(device)
        z[:, dim_idx] = torch.linspace(-range_limit, range_limit, num_steps).to(device)
        return self.decoder(z)

if __name__ == '__main__':
    model = BetaVAE(input_dim=784, latent_dim=10, beta=4.0)
    # BCE 로스는 목표가 [0,1]이어야 한다. randn은 음수를 내므로 rand를 쓴다
    x = torch.rand(32, 784)
    reconstruction, mu, logvar = model(x)
    loss, bce, kld = model.loss_function(reconstruction, x, mu, logvar)
    print(f"Loss: {loss.item():.4f}, Recon: {bce.item():.4f}, KL: {kld.item():.4f}")
```

기본 구조는 [VAE 모듈](vae.md)과 같은 완전 연결 네트워크다. 입력은 `(B,input_dim)`이고 출력은 재구성 확률·평균·로그분산 튜플이다. 클래스 기본 잠재 차원은 32지만 실행 예에서는 10으로 설정한다. 여기에는 컨볼루션 버전이나 학습 반복문이 없다.

**기존 출력 기록:**

```
Loss: 17434.7734, Recon: 17430.4121, KL: 1.0901
```

무작위 입력에 대한 한 번의 로스 계산이며 학습 결과가 아니다. BCE와 KL은 모두 배치 전체의 합이다. β는 학습 가능한 파라미터가 아닌 일반 속성이며 loss_function은 `(BCE+β×KL,BCE,KL)`을 반환한다.

## 2. 목적 함수의 의미

표본당 기대 재구성 로스를 R=−E_q log p(x|z), 사전 KL을 K라고 하면 최소화 목표는 R+βK다. β=1은 음의 ELBO이고 β>1은 KL을 더 강하게 벌한다. β=0이어도 표본을 뽑는 인코더 구조는 남으므로 자동으로 결정적 오토인코더와 같아지는 것은 아니다.

대응하는 최대화 목표 J_β=−R−βK는 ELBO−(β−1)K다. 따라서 β≥1이면 여전히 로그 주변우도 이하인 하한이지만 β=1의 표준 ELBO와는 다르다. 0≤β<1에서는 일반적인 하한 보장이 없다. 이 해석에는 정규화된 관측 모형과 기대값 기준이 필요하며, 회색조 BCE 관례는 [앞 절의 관측 모형 설명](vae.md)을 따른다.

## 3. 잠재 축 탐색과 분리된 표현

`traverse_latent_dimension`은 다른 좌표를 0으로 두고 한 좌표를 −range_limit부터 +range_limit까지 바꾼다. 반환값은 픽셀 확률이므로 표시하려면 이미지 모양으로 바꾼다. 이 함수는 장치·자료형을 모델에 자동으로 맞추거나 그래디언트 기록을 중지하지 않는다. dim_idx와 num_steps도 유효해야 한다.

학습된 여러 기준 잠재 표현 주변에서 탐색하면 한 축이 어떤 변화와 연결되는지 살펴볼 수 있다. 원점만 통과하는 경로는 집계 사후에서 드문 곳일 수 있으며 시각적 변화만으로 의미 요인의 독립성과 완전한 분리를 입증할 수는 없다.

[β-VAE 원 논문](https://openreview.net/references/pdf?id=rJi_RxvQg)은 제약된 변분 학습으로 분리된 표현을 얻는 접근을 제안했다. 이후 [비지도 표현 분리의 가정에 관한 연구](https://proceedings.mlr.press/v97/locatello19a.html)는 귀납 바이어스와 평가 조건의 중요성을 보였다. 사전 좌표의 독립성과 데이터의 참 의미 요인 복원은 서로 다른 요구다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
β 후보를 비교하는 실험을 실행 없이 설계하라. 어떤 결과를 미리 단정하면 안 되는가?

</div>

??? success "풀이"

    전처리·잠재 차원·구조·평가 예산을 맞추고 여러 초기화에서 후보별 재구성, 표준 ELBO, 차원별 KL, 사전 표본과 잠재 축 탐색을 비교한다. 목적이 요인 분리라면 참 요인 정보가 있는 별도 평가도 필요하다. MNIST에서 β=4~10이 항상 최적이거나 β=50에서 반드시 모든 차원이 붕괴한다고 미리 답할 수 없다. 본문은 이 실험을 실행하거나 결과를 제시하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
β=1일 때 코드가 최소화하는 값과 ELBO의 관계를 설명하라.

</div>

??? success "풀이"

    ELBO는 E_q log p(x|z)−KL(q||p)다. 이진 베르누이 관측에서는 BCE가 한 잠재 표본으로 추정한 음의 재구성 로그우도이고 KL은 해석적으로 계산한다. β=1이면 합은 음의 ELBO의 몬테카를로 추정이다. 코드가 배치 합을 반환하므로 표본 평균 ELBO와 비교하려면 부호와 표본 수를 맞춘다. MSE도 관측 분산과 정규화 상수를 지정한 가우시안 모형에서 해석해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
요인 분리 평가에 필요한 정보와 검증 절차를 설명하라.

</div>

??? success "풀이"

    참 생성 요인을 아는 데이터에서 요인별 개입이나 요인·잠재 변수 사이의 관계를 정의된 지표로 측정한다. 예를 들어 한 요인을 고정하고 나머지를 바꿀 때 정규화된 잠재 분산 패턴을 이용하는 분류 평가를 설계할 수 있다. 학습·평가 분할, 잠재 스케일 정규화, 붕괴 차원 처리와 점수 정의를 명시해야 한다. 단순히 큰 분산의 축을 고르는 임의 규칙을 표준 지표라고 부르거나 β=4의 일정한 개선율을 가정하지 않는다. 서로 다른 지표가 다른 속성을 평가할 수도 있다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
β가 커지면 최적 재구성 로스와 KL은 어떤 관계를 가지는가?

</div>

??? success "풀이"

    같은 후보 모형 집합에서 β₂>β₁≥0이고 각 목적의 전역 최적해를 구했다면 두 최적성 부등식을 더해 K₂≤K₁을 얻는다. 이어 R₁≤R₂가 따른다. 즉 이상적인 최적해에서는 KL을 줄이는 대신 재구성 비용이 커질 수 있다. 실제 비볼록 학습은 서로 다른 지역해·학습 오차·표본 변동 때문에 이 단조성을 보장하지 않는다. 구체적인 감소 비율은 실행 결과 없이 제시할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
β=1이 사전 표본의 품질을 항상 최대로 만드는가?

</div>

??? success "풀이"

    아니다. β=1의 근거는 표준 ELBO이며 특정 이미지 품질 지표의 최적값이라는 보장은 없다. 후보 선택에는 다양성·품질·목적별 유용성과 검증 자료를 함께 사용한다. 분류기 최대 확률이 0.9를 넘는 비율은 보조 지표일 뿐 진짜 유효률이나 모드 포괄도를 직접 재지 않는다. β가 작은 모델의 품질 저하를 모두 prior hole, 큰 모델의 저하를 모두 정보 소실로 설명할 수도 없다.



---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
β≠1로 학습한 모델을 우도 기준으로 비교할 수 있는가?

</div>

??? success "풀이"

    학습 목표가 달라도 같은 정규화된 관측 모형·전처리에서 표준 ELBO나 중요도 가중 평가량을 별도로 계산해 비교할 수 있다. β≠1이면 어떤 하한도 없다는 설명은 틀리다. β≥1의 J_β는 표준 ELBO보다 작거나 같은 하한이다. 다만 β가 다른 훈련 목적의 숫자를 같은 우도 지표로 나열하면 안 된다. 평가에서 β=1로 계산한다는 것은 학습을 다시 수행한다는 뜻이 아니라 표준 재구성·KL 조합을 보고한다는 뜻이다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff hard" title="어려움"></span>
독립 사전과 작은 KL이 의미 요인별 잠재 축을 보장하지 않는 이유는 무엇인가?

</div>

??? success "풀이"

    기대 KL은 I_q(X;Z)+KL(q(z)||p(z))다. 독립 표준정규 사전에 집계 분포가 가까워지는 것과 각 좌표가 참 의미 요인에 대응하는 것은 다르다. 등방 가우시안은 회전해도 같은 분포이며 정보를 거의 전달하지 않는 잠재 표현도 사전과 잘 맞을 수 있다. 낮은 KL이나 사용 차원 감소만으로 요인 분리 개선을 주장할 수 없다. 데이터 생성의 가정·모델 바이어스·필요한 지도 정보를 밝히고 별도 지표로 검증해야 한다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
고정 β와 KL warm-up은 어떻게 다른가?

</div>

??? success "풀이"

    KL warm-up은 학습 초기에 β를 작게 두고 정한 일정에 따라 목표값으로 올리는 방식이다. 초기 재구성 신호가 잠재 경로를 형성하도록 돕는 전략이지만 붕괴 방지나 품질 개선을 보장하지 않는다. 한 번 KL이 작아진 차원도 조건에 따라 다시 활성화될 수 있다. 최종 β가 같아도 학습 경로와 도달한 해는 달라질 수 있으므로 일정과 종료 시점을 기록한다. 본문 구현에는 자동 일정이 없으며 model.beta를 바꾸는 것은 호출부의 역할이다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
잠재 차원과 β를 어떤 기준으로 조절할 수 있는가?

</div>

??? success "풀이"

    두 값은 정보 용량·정규화·비용에 함께 영향을 주므로 예산 안에서 공동 비교하는 것이 원칙이다. 중간 크기 차원을 고정해 β를 탐색한 뒤 차원 후보를 좁히는 순차 전략도 가능하지만 보편적인 최적 순서는 없다. 큰 잠재 차원은 헤드·디코더 파라미터와 계산비용을 늘리고 모델이 불필요한 차원을 항상 정확히 끄지는 않는다. 작은 KL 차원의 개수는 임계에 따른 진단 지표이며 실제 모델을 그 수의 결정적 차원으로 축소한 것과 동일하지 않다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
재구성 항의 스케일 변화와 β 변화는 어떤 조건에서 대응하는가?

</div>

??? success "풀이"

    β>0일 때 R+βK와 R/β+K는 양의 상수 배율만 달라 같은 전역 최소해를 가진다. 재구성만 픽셀 평균으로 바꾸고 KL은 그대로 두면 상대적인 KL 비중이 픽셀 수만큼 커진다. 반대로 두 항을 같은 배치 수로 나누면 상대 비중은 유지된다. 고정 관측 분산 σ²인 가우시안의 재구성 항은 제곱오차 합/(2σ²)에 상수를 더한 값이므로 분산은 이 비중에도 관여한다. 학습 가능한 분산이면 로그 정규화 항까지 필요해 단순한 β 교체로 끝나지 않는다. 목적의 상수 배율이 같다고 유한 학습의 궤적이 동일한 것도 아니며 러닝레이트·옵티마이저·정규화·클리핑의 영향을 받는다.



## 정리하며

β-VAE는 KL의 상대 비용을 조절하는 모델이다. β=1의 표준 ELBO, β≥1의 보수적인 하한과 실제 생성·표현 품질을 구분한다. 로스의 합산 기준을 맞추고 잠재 축 탐색과 정량 평가로 목적에 맞는 설정을 비교한다.
