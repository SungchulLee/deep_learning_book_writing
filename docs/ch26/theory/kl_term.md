# VAE의 KL 발산 항

VAE에서는 근사 사후와 사전 사이의 KL 항, 그리고 ELBO와 로그우도 사이 간격인 사후 KL을 구분해야 한다. 같은 KL 정의를 사용하지만 비교하는 분포와 역할이 다르다. 이 절은 대각 가우시안의 계산과 정보량 해석, 진단 시 주의점을 정리한다.

## 1. 정의와 비음수성

같은 기준 측도에 대한 밀도 q,p에서 다음과 같이 정의한다.

$$D_{KL}(q\|p)=\mathbb E_q[\log q-\log p].$$

q가 양의 질량을 두는 곳에 p가 0이면 KL은 무한대일 수 있다. KL≥0이고 0이면 두 확률 법칙이 거의 모든 곳에서 같다. 대칭성과 삼각부등식을 만족하지 않아 일반적인 거리는 아니다. 자연로그를 쓰면 단위는 nat이며 bit로 바꾸려면 log 2로 나눈다.

Jensen 부등식으로 −E_q log(p/q)≥−log E_q(p/q)를 얻는다. 오른쪽 기대값은 q의 지지집합 위 p의 질량이므로 1 이하이고, 따라서 하한은 0 이상이다. 이를 무조건 ∫p=1이라고 두려면 지지집합 조건을 확인해야 한다. 등호 조건은 q=p에 대응한다.

엔트로피 H(q)=−E_qlog q와 교차 엔트로피 H(q,p)=−E_qlog p가 유한한 조건에서는 H(q,p)=H(q)+KL(q||p)다. 이산 확률의 부호화 길이 해석과 연속 밀도의 미분 엔트로피는 다르다. 미분 엔트로피는 음수일 수 있으며 임의의 연속값을 정확히 표현하는 비트 수로 바로 읽지 않는다.

## 2. 두 KL 방향과 근사 계열

고정 목표 P를 제한된 Q 계열로 근사할 때 KL(P||Q)는 P가 중요한 곳의 Q 부족을 벌하고, KL(Q||P)는 Q가 질량을 두는 곳의 낮은 P를 벌한다. 이를 각각 질량 포괄·모드 선택 성향으로 설명하기도 하지만 모든 분포·근사 계열에서 적용되는 정리는 아니다. Q가 P를 정확히 포함하면 양 방향 모두 P가 최적이다.

VAE의 ELBO 정규화는 KL(q_φ(z|x)||p(z))이며 간격은 KL(q_φ(z|x)||p_θ(z|x))다. 첫 p는 사전, 두 번째는 참 사후다. 표준정규 사전은 모든 실수 z에서 밀도가 양수이므로 “지지집합 안에만 둔다”는 표현만으로 정규화 효과를 설명할 수 없다.

## 3. ELBO와 사후 KL 간격

고정 x에서 유효한 모형·근사 사후와 기대값 조건 아래 다음이 성립한다.

$$\log p_\theta(x)=\underbrace{\mathbb E_q[\log p_\theta(x,z)-\log q(z|x)]}_{\mathcal L(x)}
+D_{KL}(q(z|x)\|p_\theta(z|x)).$$

결합 분포를 p_θ(x|z)p(z)로 분해하면 다음과 같다.

$$\mathcal L(x)=\mathbb E_q[\log p_\theta(x|z)]-D_{KL}(q(z|x)\|p(z)).$$

간격의 비음수성으로 ELBO≤log p_θ(x)를 얻는다. 방향은 이 항등식과 변분 추론에서 나오며 단순히 사전의 어떤 모드를 고르기 위해 임의 선택한 것은 아니다. q는 입력에 맞춰 기대값을 추정하기 쉽게 해 주지만 참 사후의 KL 간격 자체는 보통 직접 계산하기 어렵다. θ도 같이 학습하면 참 사후와 로그우도 자체가 바뀌므로 인코더 학습을 고정된 간격 감소만으로 설명하지 않는다. [원 논문](https://arxiv.org/abs/1312.6114)을 참고한다.

## 4. 가우시안 KL 유도

1차원 q=N(μ₁,σ₁²), p=N(μ₂,σ₂²)이고 분산이 양수이면 다음과 같다.

$$D_{KL}(q\|p)=\log\frac{\sigma_2}{\sigma_1}
+\frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2}-\frac12.$$

표준정규 p에서 E_qz²=μ²+σ²를 사용하면 KL=(μ²+σ²−1−logσ²)/2다. 대각 공분산의 좌표별 합은 다음과 같다.

$$D_{KL}(q\|N(0,I))=\frac12\sum_j(\mu_j^2+\exp(\ell_j)-1-\ell_j),\qquad\ell_j=\log\sigma_j^2.$$

μ_j에 대한 미분은 μ_j, ℓ_j에 대한 미분은 (exp(ℓ_j)−1)/2다. KL 항만 보면 μ=0, σ²=1을 선호한다. 전체 복원 목적이 함께 있으면 각 입력 사후가 반드시 그 값에 도달하는 것은 아니다.

```python
import torch

def kl_divergence_standard_normal(mu, logvar):
    """
    N(mu, diag(exp(logvar)))에서 N(0, I)으로의 KL 벌어짐.
    
    D_KL = -0.5 * sum(1 + log(σ²) - μ² - σ²)
    
    인수:
        mu: 평균 [배치 크기, 숨은 차원]
        logvar: 로그 흩어짐 [배치 크기, 숨은 차원]
    
    반환값:
        표본마다의 KL 벌어짐 [배치 크기]
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

def kl_divergence_two_gaussians(mu1, logvar1, mu2, logvar2):
    """
    대각 정규 분포 둘 사이의 KL 벌어짐.
    
    D_KL(q || p). 여기서 q = N(mu1, exp(logvar1)), p = N(mu2, exp(logvar2))
    
    인수:
        mu1, logvar1: q의 파라메터
        mu2, logvar2: p의 파라메터
    
    반환값:
        표본마다의 KL 벌어짐 [배치 크기]
    """
    var1 = logvar1.exp()
    var2 = logvar2.exp()
    
    kl = 0.5 * (logvar2 - logvar1 + var1 / var2 
                 + (mu1 - mu2).pow(2) / var2 - 1)
    return kl.sum(dim=1)

def kl_per_dimension(mu, logvar):
    """
    숨은 차원마다의 KL 벌어짐 몫.
    사후 분포 무너짐을 진단하는 데 쓸모 있다.
    
    인수:
        mu: 평균 [배치 크기, 숨은 차원]
        logvar: 로그 흩어짐 [배치 크기, 숨은 차원]
    
    반환값:
        차원마다의 평균 KL [숨은 차원]
    """
    kl_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl_dim.mean(dim=0)
```

함수들은 B×k의 서로 호환되는 텐서를 전제로 한다. 앞 두 함수는 코드 축을 더해 B개 KL을 반환하고 `kl_per_dimension`은 배치 축을 평균해 k개를 반환한다. 1차원·추가 배치 축에는 `dim=1`을 그대로 적용할 수 없다. 장치·dtype과 분산의 유효 범위를 맞추어야 한다. exp(logvar)는 여전히 overflow·underflow를 일으킬 수 있어 로그 분산 출력이 수치 문제를 전부 해결하지는 않는다.

## 5. 차원별 KL 진단

```python
def analyze_kl_dimensions(model, data_loader, device):
    """깨어 있는 숨은 차원과 잠든 숨은 차원을 가려낸다."""
    model.eval()
    all_kl = []
    
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.view(data.size(0), -1).to(device)
            mu, logvar = model.encode(data)
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            all_kl.append(kl.cpu())
    
    mean_kl = torch.cat(all_kl, dim=0).mean(dim=0)
    
    active = (mean_kl > 0.1).sum().item()
    total = mean_kl.shape[0]
    
    print(f"Active dimensions: {active}/{total}")
    print(f"Total KL: {mean_kl.sum():.2f}")
    
    return mean_kl
```

이 함수는 모든 표본의 차원별 KL을 모아 표본 평균을 계산하므로 배치 크기가 달라도 같은 웨이트다. 대신 전체 B×k 결과를 CPU에 보관해 메모리를 사용한다. 빈 로더이면 `cat`이 실패한다. `model.encode`는 (μ,logvar) 계약이어야 하며 모델의 장치는 호출하는 쪽에서 맞춘다.

평균 KL>0.1 nat를 활성 차원으로 세지만 이는 진단용 임의 임계다. 다른 곳의 0.01 임계와 같은 개수를 기대할 수 없다. KL이 큰 차원이 반드시 많은 입력 정보를 담는 것도 아니며, 작은 차원별 KL은 사전과의 근접성이지 정확한 정보량 측정이 아니다.

## 6. 집계 사후와 상호정보량

q(x,z)=p_data(x)q(z|x), q(z)=E_xq(z|x)로 두고 유한성 등 필요한 조건 아래 다음 항등식을 얻는다.

$$\mathbb E_xD_{KL}(q(z|x)\|p(z))=I_q(X;Z)+D_{KL}(q(z)\|p(z)).$$

따라서 기대 KL은 상호정보량의 상계지만 두 항의 합이다. 입력에 무관한 q(z|x)=r(z)라도 r가 p와 다르면 KL이 클 수 있다. 높은 KL을 곧 좋은 정보·좋은 재구성이라 부를 수 없는 이유다. 반대로 기대 KL이 정확히 0이면 거의 모든 입력의 q가 p와 같아 I=0이다.

이 분해는 특정 확률 결합에 대한 결과다. 일반 AE의 작은 실수 코드 차원이나 L1·야코비안 패널티만으로 같은 정보 상계가 생기는 것은 아니다. VAE의 평균 코드 μ(x) 그림도 집계 사후 q(z)의 모든 정보를 보여 주지 않는다.

## 7. 붕괴와 샘플링의 해석

q(z|x)=p(z)인 모든 입력은 같은 코드 분포를 갖는다. 그러나 강력한 디코더는 코드 없이도 복잡한 주변 분포를 표현할 수 있어 붕괴가 언제나 “평균 이미지만 출력”한다는 뜻은 아니다. 일부 차원 비활성은 필요한 용량 선택일 수 있으며 모든 차원이 0이 아니어야 좋은 모델이라는 규칙도 없다.

KL을 제거하면 표준 ELBO 목표는 달라진다. 분산을 줄여 결정적 복원에 가까워질 수 있지만 유한 학습에서 반드시 σ→0이거나 정확한 일반 AE가 된다고 보장하지 않는다. 생성 모형 p(z)p_θ(x|z)의 정의가 없어지는 것도 아니지만, 사전에서 좋은 표본을 얻도록 학습되었다고 보기 어려울 수 있다.

## 8. 합산과 계산 비용

관측 로그우도는 표본의 관측 차원에 대해 합하고 KL은 코드 차원에 대해 합한 뒤 배치에서 평균하는 것이 표준 ELBO의 한 구현이다. 두 항을 같이 같은 상수로 나누는 것과 복원 항만 d로 나누는 것은 다르다. 후자는 원래 합산 목적에 비해 KL의 상대 무게가 d배가 된다.

해석적 가우시안 KL은 잠재 Monte Carlo 추정의 추가 잡음을 제거하지만 미니배치와 복원 샘플의 잡음은 남는다. 두 항을 샘플링하면 분산이 정확히 두 배가 되는 것은 아니며 상관과 추정기에 달려 있다. 복잡한 사전·사후도 유효한 선택이고 표현 이득과 추정 비용을 비교해야 한다. 흐름 변환의 로그밀도를 계산할 수 있어도 KL의 기대값까지 항상 닫힌 형태인 것은 아니다.

## 연습문제


<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
표준정규에 대한 대각 가우시안 KL과 0이 되는 조건을 적어라.

</div>

??? success "풀이"

    KL=1/2 Σ_j(μ_j²+σ_j²−1−logσ_j²)이며 모든 μ_j=0,σ_j²=1이면 0이다. 일반 대각 가우시안은 차원별 σ_j가 다르다. 해석식은 잠재 표본을 뽑지 않고 계산하지만 입력 미니배치의 추정 변동까지 없애지는 않는다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
μ와 로그 분산 ℓ에 대한 KL 미분과 선호값을 구하라.

</div>

??? success "풀이"

    ∂KL/∂μ=μ, ∂KL/∂ℓ=(expℓ−1)/2다. KL만 최소화하면 μ=0,ℓ=0을 선호한다. 개별 코드 z를 원점으로 보내는 것과 사후 분포를 표준정규로 맞추는 것은 다르다. 복원 항과 결합하면 입력별 최적 분포가 반드시 사전과 같지는 않다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
ELBO와 로그우도의 간격을 적고 계산 가능성의 차이를 설명하라.

</div>

??? success "풀이"

    log p_θ(x)=ELBO+KL(q_φ(z|x)||p_θ(z|x))다. ELBO는 적절한 q 표본과 계산 가능한 결합 로그밀도로 추정할 수 있다. 간격은 참 사후를 필요로 해 보통 직접 계산하기 어렵다. 모든 모형의 적분이 불가능한 것은 아니지만 뉴럴 넷트웍 잠재 모형에서는 닫힌 형태가 일반적으로 없고 단순 사전 샘플 추정이 비효율적일 수 있다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
VAE의 사전 KL과 사후 KL의 대상·역할을 구분하라.

</div>

??? success "풀이"

    사전 KL은 ELBO의 정규화 항 KL(q(z|x)||p(z))다. 사후 KL은 ELBO 간격 KL(q(z|x)||p_θ(z|x))다. 사전은 샘플링 가능해도 참 사후는 보통 다루기 어렵다. 방향의 모드 선택·포괄 해석은 제한된 근사 계열의 성향이며 모든 분포에서 반드시 한 모드만 택하는 법칙은 아니다.



---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
KL 항이 없으면 모형 정의와 학습 목표가 각각 어떻게 달라지는가?

</div>

??? success "풀이"

    p(z)와 p_θ(x|z)를 남기면 생성 확률 법칙은 여전히 정의된다. 다만 재구성만 학습하는 목표는 표준 ELBO가 아니고 사전 샘플과 학습 사후 사이의 일치를 제어하지 않는다. 분산이 작아질 수 있지만 반드시 0으로 수렴하거나 일반 AE와 동일해지는 것은 아니다.



---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
기대 KL이 입력 정보량만을 측정하지 않는 이유를 식으로 설명하라.

</div>

??? success "풀이"

    E_x KL(q(z|x)||p(z))=I_q(X;Z)+KL(q(z)||p(z))다. 입력과 무관한 r(z)를 모든 q로 써도 r≠p이면 KL이 양수다. 따라서 큰 KL은 집계 사후 불일치일 수도 있고 정보·샘플 품질을 단독으로 증명하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
닫힌 형태 KL의 이점과 복잡한 분포를 사용할 때의 절충을 설명하라.

</div>

??? success "풀이"

    추가 잠재 샘플 없이 정확한 함수값·미분을 구해 그 추정 잡음을 피한다. 그러나 복원·미니배치 변동은 남는다. 복잡한 분포는 표현 이득과 Monte Carlo 비용을 함께 평가해야 하며 항상 손해라고 할 수 없다. 밀도 계산 가능성과 KL 기대값의 닫힌 형태도 구분한다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff easy" title="쉬움"></span>
로그 분산 출력의 장점과 수치적 한계를 설명하라.

</div>

??? success "풀이"

    실수 선형 출력에서 양의 분산을 expℓ로 만들고 KL에 로그 분산을 직접 쓸 수 있다. 하지만 큰 ℓ에서 exp가 overflow, 작은 ℓ에서 underflow할 수 있다. dtype과 범위·로스 유한성을 확인해야 하며 무조건 안정하다고 할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
KL이 0인 차원이 있으면 반드시 모델 실패인가?

</div>

??? success "풀이"

    일부 차원의 사후가 사전과 같으면 그 차원은 입력별 정보를 전달하지 않는다. 필요한 용량보다 넓다면 자연스러울 수 있다. 전체 입력의 기대 KL이 0이면 모든 코드의 정보량이 0이지만 디코더가 항상 평균만 출력한다는 뜻은 아니다. 목적별 품질과 디코더 의존성을 함께 진단한다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
복원 원소 평균과 코드 합 평균을 함께 쓰면 상대 무게가 어떻게 바뀌는가?

</div>

??? success "풀이"

    표본별 관측 합을 쓰던 기준에서 복원만 관측 차원 d로 나누면 원래 목적의 d배 KL 무게와 같은 상대 균형이 된다. 두 항을 같은 상수로 나누면 상대 균형은 유지된다. 합산 축·관측 분포 계수와 β를 함께 명시해야 공정하게 비교할 수 있다.



## 정리하며

KL 계산에서는 분포의 방향과 지지집합, 로그 단위와 합산 축을 확인한다. VAE의 사전 정규화와 사후 추론 간격은 다른 KL이며 기대 사전 KL은 입력 정보와 집계 사후 불일치를 함께 포함한다. 해석식·차원별 진단과 생성 품질을 구분해 평가한다.
