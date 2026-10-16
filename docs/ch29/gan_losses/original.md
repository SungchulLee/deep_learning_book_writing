# 원래 GAN 로스와 비포화 생성기 로스

GAN의 판별기와 생성기는 서로 다른 부호·타깃으로 학습한다. 원래 minimax 생성기 로스와 비포화(non-saturating) 로스는 같은 이상적 분포 일치점을 겨냥하지만 그래디언트와 게임의 구조가 다르다. 확률 D,그 이전의 logit a,생성기 파라미터에 대한 미분을 구분해야 포화를 정확히 설명할 수 있다.

## 1. 판별기와 원래 생성기 목적

판별기 최소화 로스는 다음과 같다.

$$L_D=-\mathbb E_{p_{data}}\log D(x)-\mathbb E_{p_Z}\log(1-D(G(z))).$$

원래 minimax 생성기 최소화 로스는 다음과 같다.

$$L_G^{MM}=\mathbb E_{p_Z}\log(1-D(G(z))).$$

확률 BCE(D,0)=−log(1−D)이므로 **음의 BCE(D,0)**를 최소화해야 원래 생성기 로스가 된다. 양의 BCE(D,0)를 줄이면 G가 D의 가짜 판정을 돕게 되어 목적이 반대다.

```python
import torch
import torch.nn as nn

class OriginalGANLoss:
    """본디 GAN 잃음(Goodfellow 외, 2014)."""
    
    def __init__(self):
        self.criterion = nn.BCELoss()
    
    def discriminator_loss(self, d_real, d_fake):
        """
        판별기 잃음: -E[log D(x)] - E[log(1 - D(G(z)))]
        """
        batch_size_real = d_real.size(0)
        batch_size_fake = d_fake.size(0)
        
        real_labels = torch.ones(batch_size_real, 1, device=d_real.device)
        fake_labels = torch.zeros(batch_size_fake, 1, device=d_fake.device)
        
        real_loss = self.criterion(d_real, real_labels)
        fake_loss = self.criterion(d_fake, fake_labels)
        
        return real_loss + fake_loss, {
            'real_loss': real_loss.item(),
            'fake_loss': fake_loss.item()
        }
    
    def generator_loss(self, d_fake):
        """
        본디 생성기 잃음: E[log(1 - D(G(z)))]
        """
        batch_size = d_fake.size(0)
        fake_labels = torch.zeros(batch_size, 1, device=d_fake.device)
        return -self.criterion(d_fake, fake_labels)
```

OriginalGANLoss는 이 부호를 사용한다. D 입력은 `(B,1)`의 확률이어야 하며 실제·생성 배치의 평균 BCE를 각각 구해 합한다. 레이블은 입력과 같은 장치지만 기본 부동소수점으로 생성되므로 다른 자료형 폴리시도 맞춰야 한다. 반환 로스 텐서와 .item() 로그용 수치를 구분한다.

## 2. 확률·logit·파라미터 그래디언트

p=D(G(z))=σ(a)로 두면 다음과 같다.

| 미분 대상 | 원래 log(1−p) | 비포화 −log p |
|---|---|---|
| 확률 p | −1/(1−p) | −1/p |
| logit a | −p | p−1 |

p≈0에서 원래 로스의 **확률 미분**은 약 −1이라 0이 아니다. 하지만 Sigmoid의 p(1−p)가 곱해진 logit 미분은 −p로 작아진다. 비포화 logit 미분은 약 −1이다. 생성기 파라미터 그래디언트는 여기에 ∇_x a와 ∇_θG도 연결되므로 어느 로스도 항상 유용하고 큰 최종 그래디언트를 보장하지는 않는다.

```python
import numpy as np
import matplotlib.pyplot as plt

def compare_gradients():
    """본디 로스와 포화하지 않는 로스의 그래디언트 크기를 견준다."""
    
    d_gz = np.linspace(0.001, 0.999, 1000)
    
    # 본디: 기울기가 -1/(1-D)이다
    grad_original = -1 / (1 - d_gz)
    
    # 포화하지 않음: 기울기가 -1/D이다
    grad_ns = -1 / d_gz
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 로스
    loss_original = np.log(1 - d_gz)
    loss_ns = -np.log(d_gz)
    
    axes[0].plot(d_gz, loss_original, label='Original: log(1-D)')
    axes[0].plot(d_gz, loss_ns, label='Non-saturating: -log(D)')
    axes[0].set_xlabel('D(G(z))')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Generator Loss Functions')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-5, 5)
    
    # 그래디언트 크기
    axes[1].plot(d_gz, np.abs(grad_original), label='|∂(log(1-D))/∂D|')
    axes[1].plot(d_gz, np.abs(grad_ns), label='|∂(-log D)/∂D|')
    axes[1].set_xlabel('D(G(z))')
    axes[1].set_ylabel('Gradient Magnitude')
    axes[1].set_title('Gradient Magnitudes')
    axes[1].legend()
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    # 그래디언트 비율
    ratio = np.abs(grad_ns) / np.abs(grad_original)
    axes[2].plot(d_gz, ratio)
    axes[2].set_xlabel('D(G(z))')
    axes[2].set_ylabel('|NS gradient| / |Original gradient|')
    axes[2].set_title('Gradient Ratio (NS / Original)')
    axes[2].axhline(y=1, color='red', linestyle='--', alpha=0.5)
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
```

비교 함수는 확률 p에 대한 로스·미분을 그린다. 제목의 Gradient Magnitude를 생성기 전체 파라미터 그래디언트로 읽지 않는다. logit 그래디언트를 보려면 위 사슬 규칙을 적용해야 한다. 함수는 그림 객체를 반환하며 호출·새 실험 결과가 이 페이지에 포함된 것은 아니다.

## 3. 비포화 로스 구현

비포화 생성기는 실제 레이블 1의 BCE를 줄인다.

$$L_G^{NS}=-\mathbb E_z\log D(G(z)).$$

```python
import torch
import torch.nn as nn

class NonSaturatingGANLoss:
    """
    포화하지 않는 맞겨루기 생성기 로스.
    
    생성기는 log(1 - D(G(z)))을 가장 작게 하는 대신 log(D(G(z)))을 가장 크게 한다.
    G이 서툴 때 더 센 그래디언트를 준다.
    """
    
    def __init__(self):
        self.criterion = nn.BCELoss()
    
    def discriminator_loss(self, d_real, d_fake):
        """
        여느 판별기 잃음(본디 GAN과 같다).
        
        L_D = -E[log D(x)] - E[log(1 - D(G(z)))]
        """
        batch_real = d_real.size(0)
        batch_fake = d_fake.size(0)
        
        real_labels = torch.ones(batch_real, 1, device=d_real.device)
        fake_labels = torch.zeros(batch_fake, 1, device=d_fake.device)
        
        real_loss = self.criterion(d_real, real_labels)
        fake_loss = self.criterion(d_fake, fake_labels)
        
        total_loss = real_loss + fake_loss
        
        return total_loss, {
            'real_loss': real_loss.item(),
            'fake_loss': fake_loss.item(),
            'd_real': d_real.mean().item(),
            'd_fake': d_fake.mean().item()
        }
    
    def generator_loss(self, d_fake):
        """
        포화하지 않는 생성기 로스.
        
        L_G = -E[log D(G(z))]
        
        log(1-D)을 가장 작게 하는 대신 -log(D)을 가장 작게 한다.
        이는 실제 이름표를 쓴 두값 어긋 엔트로피로 짠다.
        """
        batch_size = d_fake.size(0)
        
        # 생성기 익히기에서는 가짜 표본을 실제로 다룬다
        real_labels = torch.ones(batch_size, 1, device=d_fake.device)
        
        # BCE(D(G(z)), 1) = -log(D(G(z)))
        loss = self.criterion(d_fake, real_labels)
        
        return loss
```

D의 목적은 그대로 두고 G의 목적만 바꾸므로 두 업데이트가 하나의 V를 반대 부호로 최적화하는 엄밀한 영합 게임은 아니다. p_G=p_data,D*=1/2의 이상적 고정점을 공유할 수 있어도 학습 경로나 수렴 보장이 같다는 뜻은 아니다.

```python
class NonSaturatingGANLossManual:
    """로그를 드러내어 셈하는 포화하지 않는 로스."""
    
    def discriminator_loss(self, d_real, d_fake):
        """판별기 잃음: -E[log D(x)] - E[log(1 - D(G(z)))]"""
        eps = 1e-8  # 수치의 안정을 위해
        
        real_loss = -torch.log(d_real + eps).mean()
        fake_loss = -torch.log(1 - d_fake + eps).mean()
        
        return real_loss + fake_loss, {
            'd_real': d_real.mean().item(),
            'd_fake': d_fake.mean().item()
        }
    
    def generator_loss(self, d_fake):
        """G loss: -E[log D(G(z))]"""
        eps = 1e-8
        return -torch.log(d_fake + eps).mean()
```

수동 log(p+ε) 구현은 0 근처를 완화하지만 원래 목적·미분과 정확히 같지 않고 Sigmoid가 이미 정밀도를 잃은 경우를 복구하지 못한다. 확률 입력은 유한한 0~1 범위여야 한다. 판별기가 logits를 내도록 설계하면 [BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)가 Sigmoid와 로그를 결합해 더 안정적으로 계산한다. 이때 판별기 끝 Sigmoid를 중복 적용하면 안 된다.

## 4. 비포화 교대 학습

```python
def train_step_nonsaturating(G, D, real_data, latent_dim, 
                             g_optimizer, d_optimizer, device):
    """포화하지 않는 로스를 쓴 익히기 스텝."""
    
    loss_fn = NonSaturatingGANLoss()
    batch_size = real_data.size(0)
    
    # ==================
    # 판별기를 익힌다
    # ==================
    d_optimizer.zero_grad()
    
    # 실제 자료
    d_real = D(real_data)
    
    # 가짜 자료
    z = torch.randn(batch_size, latent_dim, device=device)
    fake_data = G(z)
    d_fake = D(fake_data.detach())
    
    d_loss, d_info = loss_fn.discriminator_loss(d_real, d_fake)
    d_loss.backward()
    d_optimizer.step()
    
    # ===============
    # 생성기를 익힌다(포화하지 않음)
    # ===============
    g_optimizer.zero_grad()
    
    z = torch.randn(batch_size, latent_dim, device=device)
    fake_data = G(z)
    d_fake = D(fake_data)
    
    g_loss = loss_fn.generator_loss(d_fake)
    g_loss.backward()
    g_optimizer.step()
    
    return g_loss.item(), d_loss.item(), d_info
```

D 단계에서 fake_data.detach를 사용해 G 그래프를 분리하고,D를 갱신한 뒤 새 D forward를 거쳐 G 로스를 계산한다. 모델·실제 자료·잡음의 장치·자료형,생성 범위와 판별기 `(B,1)` 확률 출력은 호출부가 맞춰야 한다. 함수는 real_data 이동이나 모델 모드 설정을 하지 않는다.

G 단계에서도 D의 grad는 계산되지만 D optimizer를 step하지 않는다. D를 거치는 입력 그래디언트는 필요하므로 D 출력 자체를 detach하면 G가 학습되지 않는다. 파라미터 고정과 BatchNorm·dropout 등 상태 동작은 별도로 관리해야 한다. 새 잡음은 이 구현의 선택이며 반드시 같은 z를 재사용할 수 없는 것은 아니다.

다음 기존 주석의 첫 줄은 원래 로스를 양의 BCE(D,0) 최소화라고 적어 **부호가 잘못되어 있다**. 위 OriginalGANLoss의 음의 BCE가 원래 minimax 목적에 맞는 구현이다.

```python
# 본디: BCE(d_fake, 0) = -log(1 - D(G(z)))을 가장 작게 한다
# 포화하지 않음: BCE(d_fake, 1) = -log(D(G(z)))을 가장 작게 한다
```

```python
# 여느 맞겨루기 생성기 익히기는 기본으로 포화하지 않는 로스를 쓴다
def recommended_generator_loss(discriminator, fake_data):
    """맞겨루기 생성기를 익히는 권하는 방식."""
    d_fake = discriminator(fake_data)
    
    # 포화하지 않는 로스
    real_labels = torch.ones_like(d_fake)
    loss = nn.BCELoss()(d_fake, real_labels)
    
    return loss
```

recommended_generator_loss도 확률을 받는 비포화 BCE 예시다. 함수 이름이 특정 로스의 보편적 우월성·안정성을 보장하지는 않는다.

## 5. 최적 판별기와 발산 해석

[원 GAN 이론](https://arxiv.org/abs/1406.2661)은 균형 잡힌 분류의 제한 없는 최적 D를 원래 전체 밸류 함수에 대입해 다음을 얻는다.

$$\max_DV(D,G)=-\log4+2\operatorname{JS}(p_{data}\|p_G).$$

이것은 현재 유한 D를 고정한 한 번의 생성기 업데이트가 항상 정확한 JS 경사라는 뜻이 아니다. 비포화 목적에 D*를 대입하면 −E_(p_G)log[p_data/(p_data+p_G)]이며 **순수한 KL(p_G||p_data)와 같지 않다**. 대입한 함수 전체를 미분하는 것과 실제 업데이트처럼 현재 D를 고정해 미분하는 것도 구분해야 한다.

[비포화 GAN의 발산 분석](https://arxiv.org/abs/2010.08029)은 적절한 조건에서의 f-divergence 해석을 다룬다. 이를 근거로 모든 비포화 GAN이 단순한 역방향 KL을 최소화한다거나 모드 붕괴가 반드시 로스 선택 하나에서 생긴다고 결론 내리지 않는다. VAE도 단순한 “순방향 KL 모드 덮기” 하나로 설명할 수 없다.

## 6. 라벨 스무딩·정규화와 평가

실제 라벨을 α<1,생성 라벨을 0으로 두면 같은 비중의 이상적 판별기는 D*=α p_data/(p_data+p_G)로 바뀐다. p_G=p_data에서 값은 α/2이며 α=0.9이면 0.45다. 이것이 유한 D 출력의 강제 상한이나 가짜 점수 0 방지 보장은 아니다.

로스 변형·러닝레이트·정규화·업데이트 비율은 서로 상호작용한다. WGAN은 실수 critic과 Lipschitz 제약을 포함한 다른 목적이며 단순히 BCE 레이블이나 이름만 바꾼 것이 아니다. 그래디언트 벌점은 특정 표본에서의 소프트 제약이므로 전역 1-Lipschitz를 자동 강제하지 않는다. 어떤 로스도 보편적으로 가장 안정되거나 모든 위치에서 0 아닌 유용한 그래디언트를 주지는 않는다.

로스·D 출력·그래디언트는 진단 자료이며 품질·다양성 평가는 별도다. FID 하나도 전체 품질·분포 일치의 충분한 지표가 아니다. 같은 평가기·전처리·표본 수와 검증 예산에서 비교하며 이 페이지에는 라벨 스무딩 등의 새 성능 실험이 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
원래 전체 밸류와 JS 발산의 관계를 유도하라.

</div>

??? success "연습문제 1 풀이"

    m=(p_data+p_G)/2라 두면 D*를 대입한 V는 E_data log(p_data/m)+E_G log(p_G/m)−2log2다. 앞 두 항은 KL(p_data||m)+KL(p_G||m)=2JS이므로 −log4+2JS다. 공통 지배 측도와 적절한 0 항 관례로 해석하며 D를 제한 없이 최적화한 분포 목적의 결과다. 현재 뉴럴 넷트웍 D의 미니배치 로스가 이 값과 항상 같지는 않다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
원래·비포화·Wasserstein 생성기 목적의 그래디언트를 어떻게 비교해야 하는가?

</div>

??? success "연습문제 2 풀이"

    원래·비포화 Sigmoid 목적의 logit 미분은 각각 −p와 p−1이다. WGAN의 −critic 출력 미분은 출력에 대해 −1이지만 입력·생성기 파라미터 미분이 언제나 0이 아닌 것은 아니다. critic 함수 계열·Lipschitz 처리·최적화 상태가 다르므로 로스 이름만으로 안정성 순위를 매기지 않는다. 각 목적·정규화·평가 계약을 맞춰 비교한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
이상적 분포 일치점에서 D와 두 생성기 로스의 값은 무엇인가?

</div>

??? success "연습문제 3 풀이"

    p_G=p_data이고 최적 D를 쓰면 혼합 지지집합 거의 모든 곳에서 D=1/2다. 원래 전체 V는 −2log2,원래 G 항은 −log2,비포화 G는 log2다. 본문 D 로스는 두 평균 BCE의 합이므로 2log2다. 데이터·생성 질량이 모두 없는 곳의 D는 미정이며 유한한 파라미터 게임의 모든 균형이 이 이상적 경우와 같지는 않다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
웨이트 clipping과 WGAN-GP가 전역 1-Lipschitz를 보장하는가?

</div>

??? success "연습문제 4 풀이"

    웨이트를 유한 구간에 두면 제한된 구조의 Lipschitz 상한을 통제할 수 있지만 그 상한이 자동으로 정확히 1인 것은 아니다. WGAN-GP는 선택한 보간점의 입력 그래디언트 노름을 1로 유도하는 벌점으로 전역 제약의 엄밀한 강제가 아니다. 용량·수치·최적화 문제와 검증 성능을 함께 살펴야 하며 GP가 모든 구조에서 항상 더 낫다는 정리는 아니다. 실수 critic과 해당 목적을 함께 정의해야 한다.



---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span>
p=0.001에서 원래·비포화 logit 그래디언트의 크기를 비교하라.

</div>

??? success "연습문제 5 풀이"

    원래는 p=0.001,비포화는 1−p=0.999이므로 국소 크기 비는 999다. 확률 p에 대한 미분과 혼동하지 않는다. 같은 현재 판별기·표본의 공통 사슬 미분에서는 이 배율이 연결되지만 배치 평균·다른 방향·정밀도·클리핑·optimizer까지 포함한 업데이트가 항상 999배라는 뜻은 아니다. 원래 목적이 수학적으로 항상 학습 불가능한 것도 아니다. 아래는 비포화 확률 BCE 골격이며 입력·레이블 장치를 맞춰야 한다.

    ```python
    lg = bce(d(g(z)), torch.ones(n, 1))     # 포화되지 않는 꼴
    ```


---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
두 로스가 같은 “최적점”을 갖는다는 말을 어떤 조건에서 사용할 수 있는가?

</div>

??? success "연습문제 6 풀이"

    이상적 표현과 최적 판별기를 가정한 분포 일치 p_G=p_data,D*=1/2가 공통 목표라는 뜻으로 사용할 수 있다. 실제 게임의 모든 국소 정지점·안정성·경로가 같다는 뜻은 아니다. 최적 D를 비포화 식에 대입한 값도 순수한 역방향 KL과 같지 않다. 모드 찾기·덮기 비유는 제약된 근사·목적·지원집합에 따라 달라 로스 이름만으로 붕괴를 단정하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
확률 BCE 코드의 생성기 부호와 detach 위치를 설명하라.

</div>

??? success "연습문제 7 풀이"

    아래 첫 골격의 G는 실제 레이블 1의 BCE를 최소화해 비포화 목적을 사용한다. D 학습의 fake만 분리하고 G 학습은 D를 통해 G까지 미분해야 한다. 뒤 골격은 음의 BCE(D,0)로 원래 목적을 표현한다. BCEWithLogitsLoss로 바꾸려면 D가 logits를 반환해야 하며 단순히 로스 클래스만 바꾸고 기존 Sigmoid를 남기지 않는다. 모든 타깃·출력의 모양·장치·자료형도 맞춘다.

    ```python
    # 판별기: 참은 1, 거짓은 0
    ld = bce(d(real), ones) + bce(d(fake.detach()), zeros)

    # 생성기: 거짓에 1을 준다  <- 이것이 포화되지 않는 꼴
    lg = bce(d(g(z)), ones)
    ```
    ```python
    lg = -bce(d(g(z)), zeros)          # 원래 꼴. 쓰지 않는다
    ```


---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
한쪽 라벨 스무딩의 최적 판별기와 구현 계약을 설명하라.

</div>

??? success "연습문제 8 풀이"

    실제 타깃 α,생성 타깃 0이면 D*=αp_data/(p_data+p_G)다. 유한 모델에 강제 상한을 두는 것이 아니고 D(fake)가 0에 가까워지지 않는다는 보장도 없다. 아래 real_target·zeros는 출력과 같은 장치·자료형·모양이어야 하며 D 업데이트의 fake는 G 그래프에서 분리해야 한다. G 타깃도 함께 바꿀지는 별도 목적 선택이다. α=0.9는 후보일 뿐 모든 작업의 최적값이나 일정한 FID 개선을 보장하지 않는다.

    ```python
    real_target = torch.full((n, 1), 0.9)      # 1.0 대신
    ld = bce(d(real), real_target) + bce(d(fake), zeros)
    ```


---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
D·G 로스와 판별기 평균을 함께 기록할 때 어떤 결론을 피해야 하는가?

</div>

??? success "연습문제 9 풀이"

    진동이 항상 정상이고 D 로스 감소가 항상 실패라는 식의 단정은 피한다. D와 G는 상대가 바뀌는 목적을 보므로 값의 해석이 상호 의존적이다. 평균 D(real)·D(fake)가 0.5여도 약한 D나 평균 상쇄일 수 있어 균형 증명이 아니다. 표본·품질·다양성·검증 분류·수치 상태를 함께 기록하며 FID만이 유일한 개선 지표인 것도 아니다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
로스 교체와 정규화를 비교 실험에서 어떻게 분리하는가?

</div>

??? success "연습문제 10 풀이"

    먼저 출력이 확률인지 logits인지,부호·레이블·분리 그래프가 맞는지 확인한다. 같은 구조·자료·평가와 튜닝 예산에서 로스와 정규화의 조합을 비교하면 개별 효과와 상호작용을 볼 수 있다. 로스마다 필요한 함수 제약도 포함해야 하므로 WGAN의 critic 제약을 빼고 이름만 비교하지 않는다. 비포화→정규화→Wasserstein이라는 단일 순서가 모든 작업에 최선이라는 결론은 근거 없이 제시할 수 없다.



## 정리하며

원래 minimax 생성기 로스는 음의 가짜 타깃 BCE,비포화 로스는 실제 타깃 BCE다. 확률·logit·파라미터 그래디언트와 최적 판별기 대입 분석을 구분한다. 수치 안정성·라벨 스무딩·정규화는 실제 출력·목적 계약에 맞춰 적용하고 품질·다양성을 별도로 검증한다.
