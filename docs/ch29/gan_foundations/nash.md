# GAN의 내시 균형과 학습 동역학

내시 균형(Nash equilibrium)은 상대의 전략을 고정했을 때 어느 쪽도 혼자 바꾸어 목적을 개선할 수 없는 상태다. 균형이 존재한다는 주장,기울기가 0이라는 주장,실제 학습이 그곳으로 수렴한다는 주장은 다르다. GAN에서는 이 구분이 특히 중요하다.

## 1. 영합 게임의 균형

원래 GAN에서 G는 V(D,G)를 최소화하고 D는 최대화한다. 전역 내시 균형은 다음 부등식을 만족한다.

$$V(D,G^*)\le V(D^*,G^*)\le V(D^*,G),\qquad\forall D,G.$$

이는 모든 허용 전략에 대한 단독 변경을 비교한다. 파라미터 근방의 작은 변경만 비교하면 국소 내시 균형이다. 단순 정지점은 한쪽의 국소 최적 반응조차 아닐 수 있다. [앞 절](adversarial.md)의 비포화 생성기 로스처럼 두 목적이 바뀌면 같은 V의 상승·하강인 영합 동역학으로 곧바로 분석할 수 없다.

## 2. 이상적 GAN의 분포 일치

충분한 분포·판별기 표현을 허용한 원래 게임에서, 고정 G의 최적 판별기는 혼합 지지집합에서 다음과 같다.

$$D_G^*(x)=\frac{p_{data}(x)}{p_{data}(x)+p_G(x)}.$$

이를 대입한 생성 분포 목적은 다음과 같다.

$$C(G)=\max_DV(D,G)=-\log4+2\operatorname{JS}(p_{data}\|p_G).$$

따라서 가능한 분포 계열이 참 분포를 포함하면 이상적 전역 최솟값은 p_G=p_data에서 −log4다. D=1/2를 전체 공간에서 택하면 이 분포와 함께 하나의 균형을 구성한다. 단,최적 판별기의 값은 혼합이 질량을 두는 영역에서만 정해지고 밀도 분모가 0인 영역은 미정이다. 분포의 동일성과 생성기 파라미터의 유일성도 다르다.

[원 GAN 논문](https://arxiv.org/abs/1406.2661)의 이상적 분포 분석을 유한 뉴럴 넷트웍·미니배치·고정 러닝레이트에 대한 일반적인 수렴 보장으로 확대하지 않는다. 최적 D를 매번 얻는 것과 제한된 파라미터 공간에서 몇 번 업데이트하는 것은 다른 절차다.

```python
import torch

def ideal_gan_training(G, D, data_loader, n_iterations):
    """
    담이가 끝없고 걸음마다 가장 좋게 고친다고 보는
    이상적인 맞겨루기 생성기 익히기.
    """
    for i in range(n_iterations):
        # 걸음 1: D을 가장 좋은 자리까지 고친다(실제로는 어림한다)
        for _ in range(k_d):  # 판별기 걸음 k_d번
            d_loss = compute_d_loss(D, G, data_loader)
            update_discriminator(D, d_loss)
        
        # 걸음 2: G을 고친다
        g_loss = compute_g_loss(G, D)
        update_generator(G, g_loss)
    
    # 모이면: p_g ≈ p_data, D(x) ≈ 0.5
    return G, D
```

위는 개념적 교대 학습 골격이다. k_d,로스 함수와 update 함수가 정의되지 않았으며 단순히 이 반복문을 적었다고 판별기가 매번 최적화되거나 균형에 도달하는 것은 아니다. 생성 분포 목적 C는 **최소화**해야 하며 V를 높이는 생성기 업데이트로 설명하면 부호가 반대다.

## 3. 로스 표면과 순환

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_loss_landscape():
    """맞겨루기 생성기의 로스 지형은 복잡하며 볼록-오목이 아니다."""
    # 파라메터 공간에서 지형에는 안장점이 많다
    # 그리고 볼록하지 않은 자리
    
    theta_g = np.linspace(-2, 2, 100)
    theta_d = np.linspace(-2, 2, 100)
    TG, TD = np.meshgrid(theta_g, theta_d)
    
    # 단순하게 만든 로스 지형(그림 삼아)
    # 실제 맞겨루기 생성기의 지형은 차원이 높고 더 복잡하다
    V = np.sin(TG) * np.cos(TD) + 0.1 * (TG**2 - TD**2)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(TG, TD, V, levels=20, cmap='RdYlBu')
    plt.colorbar(label='V(D, G)')
    plt.xlabel('θ_G')
    plt.ylabel('θ_D')
    plt.title('GAN Loss Landscape (Simplified)')
```

이 함수는 임의의 두 변수 함수를 그리는 도구다. 실제 훈련된 GAN의 목적 표면을 측정한 것이 아니다. 비볼록·비오목성 외에도 상충하는 업데이트의 회전 성분 때문에 단순 게임에서도 수렴이 어려울 수 있다.

```
되풀이 1: G이 봉우리 A에 집중하고 D은 A을 물리치는 법을 배운다
되풀이 2: G이 봉우리 B으로 옮기고 D은 B을 물리치는 법을 배운다
Iteration 3: G switches back to A (D "forgot"), ...
```

기존 순환 도식은 직관 자료이며 동역학을 입증하는 실험 기록은 아니다. 이를 명확히 보이는 예는 min_x max_y xy다. 연속 시간 동시 경사장은 ẋ=−y,ẏ=x로 x²+y²가 보존되어 일반 초기값에서 원점을 계속 돈다. 균형 (0,0)이 있어도 그 동역학이 수렴하지 않는 것이다.

## 4. 국소 선형화와 안정성

원래 매끄러운 영합 목적의 연속 시간 경사장을 선형화하면 다음 블록 행렬을 얻는다.

$$J=\begin{pmatrix}-V_{GG}&-V_{GD}\\V_{DG}&V_{DD}\end{pmatrix}.$$

정지점에서 모든 고유값의 실수부가 음수이면 적절한 매끄러움 아래 국소 점근 안정성을 얻는다. 양의 실수부가 있으면 불안정하다. 실수부 0인 고유값이 있으면 선형화만으로 비선형 안정성을 확정할 수 없다. 교차 미분이 있다고 항상 순허수 고유값만 생기는 것은 아니며 자기 곡률과 함께 작용한다.

```python
def analyze_local_dynamics(V_gg, V_gd, V_dd):
    """
    균형 가까이의 액션을 살핀다.
    
    인수:
        V_gg: ∂²V/∂θ_G²
        V_gd: ∂²V/∂θ_G∂θ_D
        V_dd: ∂²V/∂θ_D²
    """
    import numpy as np
    
    # 그래디언트 액션의 야코비 행렬
    J = np.array([
        [-V_gg, -V_gd],
        [V_gd.T, V_dd]  # 참고: V_dg = V_gd.T
    ])
    
    eigenvalues = np.linalg.eigvals(J)
    
    print("Eigenvalues:", eigenvalues)
    print("Real parts:", eigenvalues.real)
    print("Imaginary parts:", eigenvalues.imag)
    
    # 안정을 살핀다
    if all(eigenvalues.real < 0):
        print("Stable equilibrium")
    elif any(eigenvalues.real > 0):
        print("Unstable equilibrium")
    else:
        print("Marginal stability (oscillation possible)")
```

예시 analyze_local_dynamics의 np.array는 일반적인 다차원 Hessian 블록을 2×2 블록 행렬로 조립하지 못한다. 벡터 파라미터라면 np.block 등으로 전체 행렬을 구성해야 한다. 입력 타입에 따라 스칼라의 .T도 정의되지 않을 수 있다. 코드의 “Marginal stability” 출력은 영 실수부에서 결론이 미정이라는 수학적 한계까지 해결한 것은 아니다.

연속 시간 J의 안정성과 이산 업데이트도 구분한다. 동시 Euler 갱신의 국소 행렬은 I+ηJ이므로 그 고유값의 절댓값이 1보다 작아야 한다. 교대·Adam·서로 다른 러닝레이트에는 해당 업데이트의 별도 선형화가 필요하다.

## 5. 업데이트 횟수와 두 시간 척도

```python
def train_with_multiple_d_steps(G, D, data, n_d_steps=5):
    """생성기 한 걸음마다 판별기를 여러 걸음 익힌다."""
    
    for _ in range(n_d_steps):
        # D을 고친다
        d_loss = discriminator_loss(D, G, data)
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
    
    # G을 한 번 고친다
    g_loss = generator_loss(G, D)
    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()
```

D를 더 자주 업데이트하는 것은 최적 반응에 가까워지려는 전략이지만 횟수를 늘린다고 항상 G 학습이 좋아지는 것은 아니다. 로스 함수·optimizer 등은 이 골격 외부에 있으며 분리 그래프와 상태 변경도 맞춰야 한다.

```python
# 판별기가 더 빨리 배운다
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0004)

# 생성기가 더 느리게 배운다
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0001)
```

서로 다른 러닝레이트를 쓰는 예다. [TTUR 논문](https://arxiv.org/abs/1706.08500)의 수렴 정리는 시간 척도·러닝레이트 일정·안정성과 확률적 근사 등에 대한 가정을 둔다. 상수 Adam 러닝레이트 0.0004와 0.0001을 지정한 것만으로 정리의 조건이나 국소 내시 수렴이 보장되지는 않는다. 속도 비율은 데이터·구조·목적별로 검증해야 한다.

## 6. 정규화의 역할과 구현 계약

[Spectral normalization](https://arxiv.org/abs/1802.05957)은 웨이트의 스펙트럼 노름을 조절해 판별기 연산의 크기를 제어하는 접근이다.

```python
from torch.nn.utils import spectral_norm

class StableDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            spectral_norm(nn.Linear(784, 256)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(256, 256)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(256, 1)),
            nn.Sigmoid()
        )
```

예시는 선형 레이어에 정규화를 걸지만 StableDiscriminator에는 forward가 없어 그대로 호출할 수 없다. nn import도 필요하다. 전체 함수의 Lipschitz 성질은 각 레이어·활성화·잔차 등 전체 합성에 달려 있고 수치 노름 추정도 고려해야 한다. 정규화만으로 모든 GAN의 수렴을 보장하지 않는다.

[WGAN-GP](https://arxiv.org/abs/1704.00028)는 critic의 입력 그래디언트 노름을 표본 보간점에서 벌하는 방법을 제안했다.

```python
def gradient_penalty(D, real_data, fake_data, lambda_gp=10):
    """1-립시츠 묶음을 위한 WGAN-GP 그래디언트 벌점."""
    batch_size = real_data.size(0)
    
    # 아무 사이 메우기
    alpha = torch.rand(batch_size, 1, device=real_data.device)
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated.requires_grad_(True)
    
    # 판별기 내놓기
    d_interpolated = D(interpolated)
    
    # 경사를 계산한다
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True
    )[0]
    
    # 벌점
    gradient_norm = gradients.norm(2, dim=1)
    penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
    
    return penalty
```

이 gradient_penalty의 alpha `(B,1)`와 norm(dim=1)은 벡터 자료 `(B,d)`를 전제로 한다. 이미지 `(B,C,H,W)`에는 배치별 scalar 보간을 위한 차원 확장과 비배치 차원 전체의 노름 계산이 필요하다. 또한 배치 표본을 섞는 판별기에서는 출력 합의 입력 그래디언트가 표본별 그래디언트와 같지 않을 수 있다.

이 코드는 보간점의 노름을 1로 유도하는 **소프트 벌점**이며 전역 1-Lipschitz를 엄밀히 강제하지 않는다. WGAN critic은 통상 확률 Sigmoid 판별기와 다른 실수 점수·목적을 사용하므로 앞 StableDiscriminator에 함수를 붙였다고 WGAN-GP가 완성되는 것은 아니다. 생성기 그래프를 어디서 분리할지도 별도 학습 계약이다.

## 7. 균형 진단의 한계

```python
def check_equilibrium_indicators(D, G, real_data, fake_data):
    """익히기가 균형에 가까워지는지 살핀다."""
    
    with torch.no_grad():
        d_real = D(real_data).mean().item()
        d_fake = D(fake_data).mean().item()
    
    indicators = {
        'D(real)': d_real,
        'D(fake)': d_fake,
        'D_accuracy': (d_real + (1 - d_fake)) / 2,
    }
    
    # 균형에서: 모든 x에 대해 D(x) ≈ 0.5
    equilibrium_score = 1 - abs(d_real - 0.5) - abs(d_fake - 0.5)
    indicators['equilibrium_score'] = equilibrium_score
    
    print(f"D(real): {d_real:.4f}")
    print(f"D(fake): {d_fake:.4f}")
    print(f"Equilibrium score: {equilibrium_score:.4f}")
    
    if d_real > 0.9 and d_fake < 0.1:
        print("Warning: D is too strong, G may have vanishing gradients")
    elif abs(d_real - 0.5) < 0.1 and abs(d_fake - 0.5) < 0.1:
        print("Near equilibrium!")
    
    return indicators
```

D_accuracy는 임계값 분류의 실제 정답률이 아니라 [평균 D(real)+평균(1−D(fake))]/2라는 소프트 점수다. equilibrium_score도 두 **평균**이 0.5에 가까운지 볼 뿐 분포 일치·최적 반응·내시 균형을 검사하지 않는다. 약한 D나 값이 상쇄된 평균도 높은 점수를 만들 수 있다. G 인수는 함수에서 사용되지 않고 no_grad는 eval 설정이나 버퍼 고정을 수행하지 않는다.

```python
def plot_convergence(d_real_history, d_fake_history):
    """균형으로 모여 가는 모습을 그려 본다."""
    plt.figure(figsize=(10, 5))
    
    iterations = range(len(d_real_history))
    
    plt.plot(iterations, d_real_history, label='D(real)', alpha=0.7)
    plt.plot(iterations, d_fake_history, label='D(fake)', alpha=0.7)
    plt.axhline(y=0.5, color='black', linestyle='--', label='Equilibrium')
    
    plt.xlabel('Iteration')
    plt.ylabel('Discriminator Output')
    plt.title('Convergence Toward Nash Equilibrium')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
```

그래프가 0.5 근처에서 평평해져도 수렴의 증거로 충분하지 않다. 검증된 판별 진단,품질·다양성 지표와 표본·학습 경로를 함께 살펴야 한다. FID 정체나 고정 잠재 격자의 정체 역시 특정 평가가 변하지 않는다는 뜻이며 내시 조건이나 최적 생성기의 증명은 아니다.

혼합 전략은 생성기·판별기 **파라미터 전략에 대한 확률분포**이며,고정 G에 다양한 z를 주는 것과 구분한다. 유한 게임의 혼합 균형 존재 정리를 비콤팩트 뉴럴 넷트웍 파라미터 공간에 조건 없이 적용할 수 없다. 국소 균형은 작은 단독 전략 변경에 대한 목적 조건이지 모든 학습 알고리즘의 동적 안정성을 뜻하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
원래 GAN의 전역 내시 부등식과 실용적 비포화 목적을 구분하라.

</div>

??? success "연습문제 1 풀이"

    원래 영합 게임에서는 V(D,G*)≤V(D*,G*)≤V(D*,G)가 모든 허용 D,G에 대해 성립해야 한다. G와 D의 한쪽 변경만으로 각 목적을 개선할 수 없다는 뜻이다. 비포화 G는 −log D를 최소화하므로 실제 두 업데이트가 같은 V의 상승·하강은 아니다. 분포 일치의 이상적 고정점이 공통일 수 있어도 목적과 동역학 전체가 같다고 해석하면 안 된다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
최적 판별기의 식에서 지지집합과 제한된 모델 계열의 역할을 설명하라.

</div>

??? success "연습문제 2 풀이"

    실제·생성 혼합에 대한 양의 밀도 합이 있는 곳에서는 점별 분류 목적을 최대화해 D*=p_data/(p_data+p_G)를 얻는다. 한쪽 밀도가 0이면 경계값 0·1,둘 다 0이면 D는 목적에서 미정이다. 유한 뉴럴 넷트웍은 이 모든 점별 값을 동시에 표현·학습하지 못할 수 있다. 식을 이론적으로 쓸 수 있다는 것과 현재 D가 그 값이라는 것은 다르다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
모드 붕괴가 게임 중 나타날 수 있는 이유와 진단을 설명하라.

</div>

??? success "연습문제 3 풀이"

    고정 D에 대한 기대 생성기 목적은 좋은 점수 영역에 질량을 모으도록 할 수 있다. 제한 없는 분포·달성 가능한 최적점이라는 조건에서는 한 최적점에 집중한 최적 반응도 가능하지만 모든 G가 이를 구현하거나 붕괴가 반드시 일어나는 것은 아니다. 다양성 신호·정규화·업데이트를 바꿔 검증하되 품질·포괄도·중복을 따로 평가한다. 상대 속도 하나로 모든 붕괴·순환 원인을 설명하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
D(G(z))≈0에서 원래·비포화 로스의 logit 그래디언트를 계산하라.

</div>

??? success "연습문제 4 풀이"

    D=σ(a)일 때 ∂_a log(1−D)=−D이고 ∂_a[−log D]=D−1이다. 앞 값은 작아지고 뒤 값은 약 −1이므로 비포화 목적은 Sigmoid 로스 경로의 작은 그래디언트를 완화한다. 하지만 a의 입력·생성기 파라미터 미분이 작거나 좋지 않은 방향이면 유용한 업데이트가 보장되지 않는다. 모든 강한 판별기에서 모든 목적의 그래디언트가 사라진다는 명제는 아니다.



---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
전역·국소 내시 균형과 동적 안정성을 구분하라.

</div>

??? success "연습문제 5 풀이"

    전역 내시는 모든 허용 단독 변경을,국소 내시는 파라미터 근방의 단독 변경을 비교한다. 정지점은 1차 그래디언트만 0일 수 있어 각 방향의 최적성도 확인해야 한다. 동적 안정성은 정한 업데이트가 주변에서 그 점으로 가는지의 문제다. 같은 균형도 알고리즘·러닝레이트에 따라 회전·발산·수렴이 달라질 수 있다. 이상적 p_G=p_data 사례는 유한 파라미터 학습의 보편적 결과가 아니다.



---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
min_x max_y xy의 연속 경사장이 수렴하지 않음을 보여라.

</div>

??? success "연습문제 6 풀이"

    ẋ=−y,ẏ=x이면 d(x²+y²)/dt=2x(−y)+2yx=0이다. 원점 이외의 시작점은 반지름을 유지해 주기적으로 돈다. 이는 연속 시간 동시 경사장의 결과이며 “교대와 동시가 모두 정확히 이 ODE를 따른다”는 뜻은 아니다. 단일 목적의 경사 하강도 임의 러닝레이트에서 로스 감소·수렴이 보장되는 것은 아니므로 대조 조건을 명확히 한다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
고정 판별기에 대한 생성기의 최적 반응과 GAN의 분포 목표는 어떻게 다른가?

</div>

??? success "연습문제 7 풀이"

    현재 D의 점수가 높은 영역에 질량을 모으면 고정 D의 기대 생성 로스를 줄일 수 있다. 그러나 D도 바뀌어 그 영역을 구별하게 될 수 있고 실제 G의 표현 제약도 남는다. 최적 판별기를 다시 구한 C(G)의 분포 일치 목적과 현재 D에 대한 한 번의 최적 반응은 다르다. 모드 붕괴의 가능성을 설명하는 관점이지 모든 붕괴가 같은 과정이나 모든 제안 기법이 이를 막는다는 증명은 아니다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
이상적 분포 수렴 분석과 실제 학습의 가정 차이를 설명하라.

</div>

??? success "연습문제 8 풀이"

    이상적 분석은 충분한 표현·정확한 분포 기대·최적 판별기·적절한 분포 업데이트 등을 가정한다. 실제는 유한 자료·제한된 파라미터 계열·확률적 그래디언트·불완전 상대 최적화를 사용하고 로스도 바꿀 수 있다. 각 차이는 일반화·안정성에 영향을 주며 어느 하나를 항상 가장 큰 원인으로 정할 수 없다. 이론을 근거로 구체적인 정규화 조합이나 특정 FID 개선율을 보장하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
xy 게임의 동시·교대 이산 업데이트를 비교하고 아래 골격의 한계를 설명하라.

</div>

??? success "연습문제 9 풀이"

    동시 업데이트 행렬은 [[1,−η],[η,1]]로 반지름 제곱을 1+η²배 키운다. x를 먼저 갱신한 교대 행렬은 [[1,−η],[η,1−η²]]로 행렬식 1이다. 0<η<2에서 일반적으로 유계 회전하지만 원점으로의 수렴은 보장되지 않는다. 따라서 교대가 항상 더 낫거나 수렴한다는 결론은 아니다. 아래 코드는 ld·lg의 계산 시점이 생략되어 실제로 새 D에서 G 그래디언트를 다시 구하는지 알 수 없다. optimizer.step 순서만 바꾼 것과 올바른 교대 재계산을 구분한다.

    ```python
    od.zero_grad(); ld.backward(); od.step()      # 먼저 판별기
    og.zero_grad(); lg.backward(); og.step()      # 새 판별기로 생성기
    ```


---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
실용적 체크포인트 선택과 내시 수렴 판정을 구분하라.

</div>

??? success "연습문제 10 풀이"

    검증용 품질·다양성·사용 목적 지표로 좋은 체크포인트를 고를 수 있지만 이것이 내시 균형을 입증하지는 않는다. D 평균 0.5·FID 정체·클래스 엔트로피 안정도 각각 제한된 진단이다. 평가기·표본 수·전처리를 고정하고 변동을 보고하며 독립 테스트 자료는 선택에서 분리한다. 필요하면 상대를 고정한 추가 최적 반응 실험으로 개선 여지를 진단할 수 있어도 전역 최적성의 증명은 아니다.



## 정리하며

내시 균형은 단독 전략 변경에 대한 최적 반응 조건이며 실제 학습 수렴과 다르다. 원래 게임의 분포 이론,연속 시간 선형 안정성,이산·교대·Adam 업데이트를 구분해야 한다. 정규화와 평가지표는 진단·설계 도구이며 그 자체로 균형이나 분포 일치를 보증하지 않는다.
