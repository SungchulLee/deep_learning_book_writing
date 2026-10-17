# 변분 오토인코더의 기본 구조

변분 오토인코더(VAE)는 잠재 사전 p(z), 디코더 관측 분포 p_θ(x|z)와 근사 사후 q_φ(z|x)를 사용하는 확률 모형이다. 이 예제는 대각 가우시안 인코더와 Sigmoid 디코더의 구조를 정의한다. 본 코드의 실행부는 파라미터 수만 출력하며 학습된 생성 품질을 보여 주지는 않는다.

## 1. 예제 코드와 출력

```python
"""VAE - 변분 오토인코더."""
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=20):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, input_dim), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

if __name__ == "__main__":
    model = VAE()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

```
Parameters: 1,082,680
```

## 2. 인코더와 잠재 분포

기본 입력 차원은 784, 코드 차원은 20이다. 공통 인코더 784→512→256 뒤에 `fc_mu`와 `fc_logvar` 두 선형 계층이 각각 B×20 배열을 만든다. `logvar`는 로그 표준편차가 아니라 로그 **분산**이다.

$$q_\phi(z|x)=\mathcal N(\mu(x),\operatorname{diag}(\exp(\ell(x)))),\qquad\ell=\log\sigma^2.$$

기본 파라미터 수는 1,082,680개다. 생성 모수 θ의 점추정을 학습하는 이 코드가 모든 네트워크 웨이트에 대한 완전한 베이즈 사후를 추론하는 것은 아니다. VAE의 변분 추론은 여기서 입력별 잠재 사후를 근사하는 데 사용한다.

`forward`는 생성자 `input_dim`과 무관하게 `x.view(-1,784)`를 사용한다. 다른 입력 차원으로 초기화하면 인코더 차원이 맞지 않을 수 있으며, 원소 수가 우연히 나누어떨어져도 배치 의미가 바뀔 수 있다. 모양·연속성과 유효한 잠재 차원을 확인해야 한다.

## 3. 리파라메리제이션의 역할

코드는 ε∼N(0,I)를 파라미터와 독립적으로 뽑고 z=μ+exp(ℓ/2)⊙ε를 계산한다. 잡음 실현값을 고정한 경로에서 μ와 ℓ로 미분할 수 있어 기대 복원 항의 경로별 그래디언트를 추정한다. 샘플링 자체가 수학적으로 어떤 방법으로도 미분 불가능하다는 뜻은 아니다. 리파라메리제이션 외에 다른 추정기도 있으며 PyTorch 분포의 `sample`과 리파라메리제이션된 `rsample`도 구분한다.

여기 `forward`는 항상 새 ε를 뽑으므로 `eval()`을 호출해도 확률적이다. 평균 코드만 사용할 평가와 사후 표본을 사용할 평가를 구분해 별도로 정의해야 한다.

## 4. ELBO와 KL 항

VAE는 보통 다음 [ELBO](https://arxiv.org/abs/1312.6114)를 최대화한다.

$$\mathcal L(x)=\mathbb E_{q_\phi(z|x)}[\log p_\theta(x|z)]
-D_{KL}(q_\phi(z|x)\|p(z))\le\log p_\theta(x).$$

표준정규 사전과 대각 가우시안 사후의 KL은 다음과 같다.

$$D_{KL}=\frac12\sum_j(\mu_j^2+\exp(\ell_j)-1-\ell_j).$$

KL을 0으로 만드는 것은 μ=0, ℓ=0이다. 입력별 분포가 모두 사전과 같으면 코드가 입력 정보를 전달하지 않는 사후 붕괴가 된다. KL은 무너짐을 항상 막는 항이 아니며 오히려 이 입력 독립 해를 선호할 수 있다. 큰 KL도 입력 정보뿐 아니라 집계 사후와 사전의 불일치에서 나올 수 있다.

## 5. 관측 분포와 샘플링

Sigmoid 출력은 조건부 독립 베르누이 관측을 사용할 경우 픽셀의 확률이다. 정규 잠재 z를 디코딩한 확률 이미지와, 그 확률에서 다시 베르누이 픽셀을 뽑은 관측 표본은 다르다. 연속 회색 픽셀의 BCE를 쓴다면 실제 관측 우도와 소프트 표적 목적의 차이를 설명해야 한다.

MSE는 고정 분산 가우시안 관측으로 해석할 수 있지만 분산 계수와 상수·합산 단위를 정해야 ELBO와 대응한다. 조건부 독립 관측도 잠재 변수를 적분하면 주변 분포에서 픽셀 의존성을 표현할 수 있다. 관측 가정 하나가 모든 흐릿함의 유일한 원인은 아니다.

VAE가 확률 분포를 정의하고 샘플링 절차를 제공한다는 사실과 좋은 표본을 생성한다는 것은 다르다. KL이 잠재공간의 모든 빈 구간을 메우거나 재구성·다양성을 보장하지 않는다. 일반 AE에 별도 잠재 분포를 결합하는 생성 접근도 가능하다.

## 6. 학습·품질 평가의 범위

본 코드에는 로스 함수·학습 루프·데이터셋이 없다. 생성 예제와 로스 조각은 아래 연습문제에서 읽되 필요한 import·모양·장치를 맞추어야 한다. 다른 페이지의 분류기 확신도 임계를 “숫자로 보이는 비율”이나 생성 실패율로 바꾸어 해석하지 않는다.

잠재 차원, KL 무게와 스케줄은 독립 검증에서 비교한다. 차원별 KL이나 μ의 분산은 활용도를 보는 진단이지만 참 내재 차원을 직접 알려 주지 않는다. 사후 붕괴에서 회복이 원리적으로 불가능한 것도 아니며 KL 워밍업·Free Bits 역시 항상 해결을 보장하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
대각 가우시안 q=N(μ,diag(σ²))와 표준정규 p 사이의 KL을 유도하라. 등방 분산은 어떻게 다른가?

</div>

??? success "풀이"

    가우시안 로그밀도 차이의 q 기댓값에서 E[z_j²]=μ_j²+σ_j²를 쓰면 KL=1/2 Σ_j(μ_j²+σ_j²−1−logσ_j²)다. ℓ_j=logσ_j²이면 코드의 식과 같다. 등방 분산 σ²I는 모든 σ_j가 같은 특수 경우이고, 코드의 일반 대각 사후는 차원마다 다르다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
아래 생성 조각이 모델의 관측 표본을 완전히 뽑는지와 실행 전제를 설명하라.

</div>

??? success "풀이"

    기본 20차원 CPU 코드에서 decode한 값은 Sigmoid 확률 이미지다. 베르누이 관측 표본이면 그 확률에서 픽셀을 추가로 뽑는다. 학습 모델과 장치·잠재 차원을 맞추고 plt import가 필요하다. GPU 출력은 .numpy() 전에 CPU로 옮겨야 한다. 본문은 학습하지 않아 이 조각만으로 좋은 숫자가 나온다고 가정할 수 없다.

    ```python
    model.eval()
    with torch.no_grad():
        z = torch.randn(100, 20)
        samples = model.decode(z).view(-1, 28, 28)
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for i in range(100):
        axes[i//10, i%10].imshow(samples[i].numpy(), cmap='gray')
        axes[i//10, i%10].axis('off')
    plt.show()
    ```


---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
리파라메리제이션으로 어떤 그래디언트 경로가 생기는가?

</div>

??? success "풀이"

    파라미터 독립 ε를 고정한 경로 z=μ+σ⊙ε로 복원 로스가 μ·σ와 인코더까지 미분된다. sample처럼 경로를 기록하지 않는 호출에서는 복원 항의 경로별 그래디언트가 없지만, 별도의 해석적 KL 그래디언트까지 사라지는 것은 아니다. rsample이나 다른 추정 방식도 구분한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
일반 AE와 VAE의 출력·학습 목표 차이를 설명하라.

</div>

??? success "풀이"

    일반 AE는 보통 결정적 코드를, 이 VAE는 μ와 로그 분산을 반환해 코드를 샘플링한다. 확률적 디코더 해석과 사전, ELBO 목적을 함께 정한다. KL이나 샘플링 하나를 제거했다고 반드시 동일한 일반 AE가 되는 것은 아니며 남은 분산 경로와 목표를 확인해야 한다.



---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
아래 로스 조각의 합산 축과 필요한 입력 조건을 설명하라.

</div>

??? success "풀이"

    복원 BCE와 KL을 각각 모든 표본·픽셀 또는 코드에 합한 뒤 배치 크기로 나눈다. B×784로 맞춘 표적과 출력, B×k의 mu·logvar 및 F import가 필요하다. 복원만 원소 평균으로 바꾸면 KL의 상대 무게가 784배 달라진다. 둘을 동일 상수로 나누는 것과 한 항만 바꾸는 것은 다르다.

    ```python
    def loss_function(out, x, mu, logvar):
        # 다시 세우기: 화소에 대해 더하고 표본에 대해 평균
        rec = F.binary_cross_entropy(out, x, reduction='sum') / x.size(0)
        # KL: 닫힌 꼴. 차원에 대해 더하고 표본에 대해 평균
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum() / x.size(0)
        return rec + kl
    ```


---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
VAE가 생성 분포를 정의하는 근거와 확률 이미지를 표시하는 절차의 차이를 설명하라.

</div>

??? success "풀이"

    p_θ(x)=∫p_θ(x|z)p(z)dz가 주변 분포를 정의한다. 잠재를 뽑고 관측 분포에서 뽑으면 그 모델의 관측 표본이다. Bernoulli 평균 이미지만 표시하는 것은 조건부 평균 시각화다. 분포 정의나 샘플링 가능성이 실제 품질을 증명하지는 않는다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff hard" title="어려움"></span>
BCE와 MSE 복원 항의 우도 해석 및 흐릿함의 원인을 설명하라.

</div>

??? success "풀이"

    이진 관측 BCE는 조건부 독립 Bernoulli 음의 로그우도다. 연속 회색 표적에는 소프트 목적과 실제 연속 우도를 구분한다. MSE는 고정 분산 Gaussian 우도의 계수·상수를 맞춘 형태다. 잠재 혼합은 주변 의존성을 만들 수 있어 조건부 독립 하나만으로 흐릿함을 전부 설명할 수는 없다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
잠재 차원과 활성 차원의 수로 내재 차원을 정할 수 있는가?

</div>

??? success "풀이"

    활성 기준과 학습 목적, KL 무게·관측 분포·최적화에 따라 수가 달라지므로 직접 추정이 아니다. 여유 차원을 늘리면 파라미터·계산 비용도 커진다. 복원·ELBO·샘플 품질과 차원별 통계를 검증해 선택하고 고정 표에서 항상 16이 최선이라고 일반화하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
KL이 빨리 0으로 가는 이유와 이를 진단할 지표를 설명하라.

</div>

??? success "풀이"

    μ=0,σ=1은 입력에 무관하게 KL=0을 만드는 해다. 복원보다 이 경로가 먼저 최적화될 수 있지만 항상 같은 동학은 아니다. KL·복원·μ 분산·차원별 KL과 디코더의 코드 의존성을 함께 본다. 워밍업·Free Bits를 검토할 수 있으나 항상 회복하거나 비활성 차원이 절대 회복하지 못한다는 보장은 없다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
재구성·표현·생성·이상 탐지에서 각각 필요한 평가를 설명하라.

</div>

??? success "풀이"

    재구성은 사후 샘플 또는 평균 코드 조건을 명시해 측정한다. 표현은 μ 등 코드 정의와 보류 태스크를, 생성은 사전·관측 샘플 구분과 품질·다양성을 평가한다. ELBO를 이상 점수로 사용할 수 있지만 높은 우도나 하한이 항상 정상임을 보장하지 않는다. 일반 AE에서도 별도 이상 점수 평가가 가능하다.



## 정리하며

VAE는 잠재 사전·근사 사후·관측 분포를 정하고 ELBO를 학습한다. 리파라메리제이션은 복원 항의 그래디언트 경로를 제공하지만 KL과 샘플링이 품질·붕괴 방지를 보장하지는 않는다. 구현의 입력 차원, 확률 이미지와 관측 표본, 합산 축과 실제 학습 범위를 구분한다.
