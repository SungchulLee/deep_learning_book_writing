# 내시 균형
맞겨루기 만들개 익히기는 놀이 이론의 눈으로 이해할 수 있다. **내시 균형** 개념은 맞겨루기 만들개가 언제 어떻게 모이는지 이해하는 이론의 바탕을 준다.

---

## 1. 놀이 이론 틀

### 두 사람 놀이로 적기

맞겨루기 만들개는 다음과 같은 두 사람 놀이를 뜻매김한다.

- **첫째 사람(만들개)**: 방책 $G$(매개변수 $\theta_G$)을 고른다
- **둘째 사람(가름개)**: 방책 $D$(매개변수 $\theta_D$)을 고른다
- **셈속 함수**: $V(D, G)$

이는 **영합 놀이**이다. 곧 D은 $V$을 가장 크게 하고 G은 가장 작게 한다.

### 내시 균형 뜻매김

**내시 균형**은 어느 쪽도 혼자 방책을 바꾸어 셈속을 높일 수 없는 방책 짝 $(G^*, D^*)$이다.

$$V(D, G^*) \leq V(D^*, G^*) \leq V(D^*, G) \quad \forall D, G$$

곧 다음을 뜻한다:

- (G*이 주어졌을 때) D은 방책을 바꾸어 나아질 수 없다
- (D*이 주어졌을 때) G은 방책을 바꾸어 나아질 수 없다

---

## 2. 내시 균형이 있음

### 정리: 맞겨루기 만들개의 균형

맞겨루기 만들개의 값 함수에 대해:

$$V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

다음과 같은 내시 균형이 있다.

$$p_g^* = p_{\text{data}}$$

$$D^*(x) = \frac{1}{2} \quad \forall x$$

### 밝힘 밑그림

**걸음 1**: 어떤 만들개 G에 대해서든 가장 좋은 가름개는 다음과 같다.

$$D^*_G(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$$

**걸음 2**: $D^*_G$을 $V$에 넣으면:

$$C(G) = V(D^*_G, G) = -\log 4 + 2 \cdot \text{JSD}(p_{\text{data}} \| p_g)$$

**걸음 3**: 젠슨-섀넌 벌어짐은 $p_g = p_{\text{data}}$일 때 가장 작다(0이 된다).

**걸음 4**: 균형에서:

- $p_g^* = p_{\text{data}}$이 $C(G)$을 가장 작게 한다
- $D^*(x) = \frac{p_{\text{data}}(x)}{2 \cdot p_{\text{data}}(x)} = \frac{1}{2}$

---

## 3. 맞겨루기 만들개 균형의 성질

### 균형에서

| 양 | 값 |
|----------|-------|
| 만들개 분포 | $p_g = p_{\text{data}}$ |
| 가름개 내놓기 | 모든 $x$에 대해 $D(x) = 0.5$ |
| 값 함수 | $V(D^*, G^*) = -\log 4 \approx -1.386$ |
| 젠슨-섀넌 벌어짐 | $\text{JSD}(p_{\text{data}} \| p_g) = 0$ |

### 해석

균형에서:

- 만들개가 흠 없는 표본을 낸다
- 가름개가 실제와 가짜를 가려내지 못한다
- 가름개의 가장 좋은 방책은 아무렇게나 찍기(50/50)이다

---

## 4. 균형으로 모이기

### 가장 좋은 경우의 모임

가장 좋은 조건에서는 번갈아 하는 기울기 내려가기가 내시 균형으로 모인다.

```python
import torch

def ideal_gan_training(G, D, data_loader, n_iterations):
    """
    담이가 끝없고 걸음마다 가장 좋게 고친다고 보는
    이상적인 맞겨루기 만들개 익히기.
    """
    for i in range(n_iterations):
        # 걸음 1: D을 가장 좋은 자리까지 고친다(실제로는 어림한다)
        for _ in range(k_d):  # 가름개 걸음 k_d번
            d_loss = compute_d_loss(D, G, data_loader)
            update_discriminator(D, d_loss)
        
        # 걸음 2: G을 고친다
        g_loss = compute_g_loss(G, D)
        update_generator(G, g_loss)
    
    # 모이면: p_g ≈ p_data, D(x) ≈ 0.5
    return G, D
```

### 이론의 보장

**명제**(Goodfellow 외, 2014): G과 D의 담이가 넉넉하고 걸음마다 가름개가 주어진 G에서 가장 좋은 자리에 이를 수 있으며 p_g이 다음 잣대를 높이도록 고쳐진다면,

$$\mathbb{E}_{x \sim p_{\text{data}}}[\log D^*_G(x)] + \mathbb{E}_{x \sim p_g}[\log(1 - D^*_G(x))]$$

$p_g$은 $p_{\text{data}}$으로 모인다.

---

## 5. 모이기가 어려운 까닭

### 문제 1: 볼록하지 않은 가장 좋게 하기

맞겨루기 만들개의 목표는 볼록-오목이 아니다.

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_loss_landscape():
    """맞겨루기 만들개의 손실 지형은 복잡하며 볼록-오목이 아니다."""
    # 매개변수 공간에서 지형에는 안장점이 많다
    # 그리고 볼록하지 않은 자리
    
    theta_g = np.linspace(-2, 2, 100)
    theta_d = np.linspace(-2, 2, 100)
    TG, TD = np.meshgrid(theta_g, theta_d)
    
    # 단순하게 만든 손실 지형(그림 삼아)
    # 실제 맞겨루기 만들개의 지형은 차원이 높고 더 복잡하다
    V = np.sin(TG) * np.cos(TD) + 0.1 * (TG**2 - TD**2)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(TG, TD, V, levels=20, cmap='RdYlBu')
    plt.colorbar(label='V(D, G)')
    plt.xlabel('θ_G')
    plt.ylabel('θ_D')
    plt.title('GAN Loss Landscape (Simplified)')
```

### 문제 2: 흔들림

모이는 대신 G과 D이 흔들릴 수 있다.

```
되풀이 1: G이 봉우리 A에 집중하고 D은 A을 물리치는 법을 배운다
되풀이 2: G이 봉우리 B으로 옮기고 D은 B을 물리치는 법을 배운다
Iteration 3: G switches back to A (D "forgot"), ...
```

### 문제 3: 유한한 담이

실제 신경망은 담이가 유한하다.

- D이 참으로 가장 좋은 가름개를 나타낼 수 없다
- G이 있을 수 있는 모든 분포를 나타낼 수 없다

---

## 6. 안정성 분석

### 가까이에서의 안정

균형 가까이에서는 기울기 움직임의 야코비 행렬로 안정을 살필 수 있다.

$$\frac{d\theta_G}{dt} = -\nabla_{\theta_G} V$$

$$\frac{d\theta_D}{dt} = +\nabla_{\theta_D} V$$

야코비 행렬:

$$J = \begin{pmatrix} -\nabla^2_{\theta_G} V & -\nabla_{\theta_G \theta_D} V \\ \nabla_{\theta_D \theta_G} V & \nabla^2_{\theta_D} V \end{pmatrix}$$

### 고윳값 살피기

안정되려면 J의 고윳값이 음의 실수부를 가져야 한다. 그러나 맞겨루기 만들개에서는:

- 섞인 편미분이 순허수 고윳값을 만든다
- 이는 모임이 아니라 **돌기**로 이어진다
- 흔들리는 움직임을 설명해 준다

```python
def analyze_local_dynamics(V_gg, V_gd, V_dd):
    """
    균형 가까이의 움직임을 살핀다.
    
    인수:
        V_gg: ∂²V/∂θ_G²
        V_gd: ∂²V/∂θ_G∂θ_D
        V_dd: ∂²V/∂θ_D²
    """
    import numpy as np
    
    # 기울기 움직임의 야코비 행렬
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

---

## 7. 실제로 모이게 하기

### 방책 1: D을 여러 번 걸음

D을 가장 좋은 자리에 가깝게 두려 G보다 더 많이 익힌다.

```python
def train_with_multiple_d_steps(G, D, data, n_d_steps=5):
    """만들개 한 걸음마다 가름개를 여러 걸음 익힌다."""
    
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

### 방책 2: 두 시간 잣수 고침 규칙(TTUR)

서로 다른 배움 빠르기를 쓴다.

```python
# 가름개가 더 빨리 배운다
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0004)

# 만들개가 더 느리게 배운다
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0001)
```

### 방책 3: 스펙트럼 고르게 맞추기

가름개의 립시츠 상수를 묶는다.

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

### 방책 4: 기울기 벌점

가름개 기울기 크기에 벌점을 더한다.

```python
def gradient_penalty(D, real_data, fake_data, lambda_gp=10):
    """1-립시츠 묶음을 위한 WGAN-GP 기울기 벌점."""
    batch_size = real_data.size(0)
    
    # 아무 사이 메우기
    alpha = torch.rand(batch_size, 1, device=real_data.device)
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated.requires_grad_(True)
    
    # 가름개 내놓기
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

---

## 8. 균형 지켜보기

### 모임의 표시

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

### 눈으로 지켜보기

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

---

## 9. 여느 내시 균형 너머

### 섞인 내시 균형

어떤 경우에는 풀이가 사람들이 아무렇게나 고르는 **섞인 균형**이다.

- G이 만들개에 대한 분포에서 뽑는다
- D이 가름개에 대한 분포에서 뽑는다

### 가까운 내시 균형

볼록하지 않으므로 맞겨루기 만들개는 흔히 **가까운** 내시 균형을 찾는다.

- 작은 흔들림에는 안정되다
- 온마당에서 가장 좋지는 않을 수 있다
- 가까운 균형이 여럿 있을 수 있다

---

## 연습문제

**연습문제 1.**
맞겨루기 만들개 익히기 목표를 최소최대 놀이로 설명하라. 각 사람은 무엇을 가장 좋게 하는가?

??? success "연습문제 1 풀이"
    맞겨루기 만들개의 목표는 다음과 같다.

    $$\min_G \max_D \; \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

    **가름개** $D$은 이를 가장 크게 한다. 곧 실제 자료에는 $D(x) \to 1$, 가짜 자료에는 $D(G(z)) \to 0$을 바란다. **만들개** $G$은 가장 작게 한다. 곧 $D(G(z)) \to 1$(가름개 속이기)을 바란다. 내시 균형에서 $G$은 참 자료 분포에서 표본을 만들고 어디서나 $D(x) = 1/2$이다.

---

**연습문제 2.**
만들개가 붙박였을 때 가장 좋은 가름개가 $D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$임을 밝혀라.

??? success "연습문제 2 풀이"
    만든 분포가 $p_g$인 붙박인 $G$에서 가름개는 다음을 가장 크게 한다.

    $$V(D) = \int \left[ p_{\text{data}}(x) \log D(x) + p_g(x) \log(1 - D(x)) \right] dx$$

    $D(x)$에 대해 미분해 0으로 두면:

    $$\frac{p_{\text{data}}(x)}{D(x)} - \frac{p_g(x)}{1 - D(x)} = 0 \implies D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$$

    이차 미분이 음수이므로 이것이 최댓값임이 굳어진다. $\square$

---

**연습문제 3.**
맞겨루기 만들개 익히기에서 봉우리 무너짐이란 무엇인가? 이를 누그러뜨리는 방식 둘을 설명하라.

??? success "연습문제 3 풀이"
    봉우리 무너짐은 만들개가 자료 분포의 봉우리 가운데 일부에서만 표본을 내고 나머지를 무시할 때 일어난다. 예컨대 숫자로 익힌 맞겨루기 만들개가 "1"과 "7"만 만들 수 있다. **누그러뜨리는 방식**: (1) **작은 묶음 가름**: 가름개가 묶음 전체의 통계를 받아 다양함이 모자라면 벌을 준다. (2) **바서슈타인 맞겨루기 만들개(WGAN)**: 젠슨-섀넌 벌어짐 대신 바서슈타인 거리를 써서 분포가 겹치지 않아도 0이 아닌 기울기를 주어 익히기를 안정시키고 봉우리 무너짐을 줄인다. 그 밖에 펼친 맞겨루기 만들개, 스펙트럼 고르게 맞추기, 차츰 키우기가 있다.

---

**연습문제 4.**
가름개가 너무 셀 때 본디 맞겨루기 만들개 손실의 기울기가 사라질 수 있는 까닭을 설명하라.

??? success "연습문제 4 풀이"
    가름개가 만들개보다 훨씬 나으면 만든 표본에서 $D(G(z)) \approx 0$이다. $\log(1 - D(G(z)))$에서 오는 만들개의 기울기는 $\frac{-D'(G(z))}{1 - D(G(z))}$이며 $D(G(z)) \approx 0$일 때 $-D'(G(z))$(작다)에 가까워진다. 더 결정적으로 손실 $\log(1 - D(G(z)))$이 $\log(1) = 0$ 가까이에서 포화해 기울기가 거의 0이 된다. 실제의 손질은 **포화하지 않는 손실**이다. 곧 $\log(1 - D(G(z)))$을 가장 작게 하는 대신 만들개가 $\log D(G(z))$을 가장 크게 하며, 이는 $D(G(z))$이 작을 때 센 기울기를 지닌다.

## 정리하며

| 개념 | 설명 |
|---------|-------------|
| **내시 균형** | 어느 쪽도 혼자서는 나아질 수 없다 |
| **맞겨루기 만들개의 균형** | $p_g = p_{\text{data}}$, $D^* = 0.5$ |
| **모임** | 가장 좋은 조건에서는 이론으로 보장된다 |
| **실제의 문제** | 볼록하지 않음, 흔들림, 유한한 담이 |
| **안정시키기** | TTUR, 스펙트럼 잣대, 기울기 벌점 |
| **지켜보기** | D(실제)와 D(가짜)가 0.5으로 가는지 좇는다 |

내시 균형을 아는 것은 맞겨루기 만들개 익히기의 이론 바탕이 되고 실제의 개선을 이끌어 낸다. 완벽히 모이기는 어렵지만 이 통찰이 더 안정된 익히기 알고리즘을 만드는 길잡이가 된다.
