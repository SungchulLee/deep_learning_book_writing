# 본디 맞겨루기 만들개 손실
Goodfellow 외가 2014년에 내놓은 본디 맞겨루기 만들개 손실은 만들어 내는 모델을 두값 어긋 엔트로피 가르기를 쓴 최소최대 놀이로 적는다.

---

## 1. 수학적 정식화

### 값 함수

본디 맞겨루기 만들개의 목표는 다음과 같다.

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

### 가름개 손실

가름개는 $V(D, G)$을 가장 크게 하며 이는 다음을 가장 작게 하는 것과 같다.

$$\mathcal{L}_D = -\mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] - \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

이는 다음을 갖춘 **두값 어긋 엔트로피**이다.

- 실제 표본은 1로 이름표를 붙인다
- 가짜 표본은 0으로 이름표를 붙인다

### 만들개 손실

만들개는 $V(D, G)$을 가장 작게 한다.

$$\mathcal{L}_G = \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

만들개는 $D(G(z)) \to 1$을 바라므로 $\log(1 - D(G(z))) \to -\infty$이다.

---

## 2. 구현

```python
import torch
import torch.nn as nn

class OriginalGANLoss:
    """본디 GAN 잃음(Goodfellow 외, 2014)."""
    
    def __init__(self):
        self.criterion = nn.BCELoss()
    
    def discriminator_loss(self, d_real, d_fake):
        """
        가름개 잃음: -E[log D(x)] - E[log(1 - D(G(z)))]
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
        본디 만들개 잃음: E[log(1 - D(G(z)))]
        """
        batch_size = d_fake.size(0)
        fake_labels = torch.zeros(batch_size, 1, device=d_fake.device)
        return -self.criterion(d_fake, fake_labels)
```

---

## 3. 경사 분석

### 포화 문제

D이 가짜 자료에 자신 있으면($D(G(z)) \approx 0$):

$$\nabla_{\theta_G} \mathcal{L}_G = -\mathbb{E}_z\left[\frac{\nabla_{\theta_G} D(G(z))}{1 - D(G(z))}\right]$$

분모 $1 - D(G(z)) \approx 1$이어서 G이 센 배움 신호를 필요로 할 때 기울기가 작아진다.

---

## 4. 이론적 성질

### 가장 좋은 가름개

$$D^*_G(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$$

### 젠슨-섀넌 벌어짐

가장 좋은 D에서 만들개는 다음을 가장 작게 한다.

$$C(G) = -\log 4 + 2 \cdot \text{JSD}(p_{\text{data}} \| p_g)$$

---

## 5. 한계

1. **기울기 포화** - 익히기 앞머리에 G의 기울기가 여리다
2. **봉우리 무너짐** - 분포가 겹치지 않으면 젠슨-섀넌 벌어짐이 평평해질 수 있다
3. **익히기의 불안정** - 웃매개변수를 꼼꼼히 맞추어야 한다

---

## 6. 왜 필요한가

### 포화 문제

본디 맞겨루기 만들개에서 만들개는 다음을 가장 작게 한다.

$$\mathcal{L}_G^{\text{original}} = \mathbb{E}_z[\log(1 - D(G(z)))]$$

익히기 앞머리에 $D(G(z)) \approx 0$일 때:

- $\log(1 - D(G(z))) \approx \log(1) = 0$
- 기울기 $\frac{-1}{1-D(G(z))} \approx -1$(가둬져 있다)

바로 G이 서툴러 센 배움 신호가 필요할 때 기울기가 여리다.

### 풀이

$\log(1 - D(G(z)))$을 가장 작게 하는 대신 $\log D(G(z))$을 가장 크게 한다.

$$\mathcal{L}_G^{\text{NS}} = -\mathbb{E}_z[\log D(G(z))]$$

$D(G(z)) \approx 0$일 때:

- 기울기 $\frac{-1}{D(G(z))} \to -\infty$

G이 나아져야 할 때 센 기울기 신호가 온다!

---

## 7. 수학으로 견주기

### 본디 손실

$$\mathcal{L}_G^{\text{original}} = \mathbb{E}_z[\log(1 - D(G(z)))]$$

$D(G(z))$에 대한 기울기:

$$\frac{\partial \mathcal{L}_G^{\text{original}}}{\partial D(G(z))} = \frac{-1}{1 - D(G(z))}$$

### 포화하지 않는 손실

$$\mathcal{L}_G^{\text{NS}} = -\mathbb{E}_z[\log D(G(z))]$$

$D(G(z))$에 대한 기울기:

$$\frac{\partial \mathcal{L}_G^{\text{NS}}}{\partial D(G(z))} = \frac{-1}{D(G(z))}$$

---

## 8. 기울기 견주기

```python
import numpy as np
import matplotlib.pyplot as plt

def compare_gradients():
    """본디 손실과 포화하지 않는 손실의 기울기 크기를 견준다."""
    
    d_gz = np.linspace(0.001, 0.999, 1000)
    
    # 본디: 기울기가 -1/(1-D)이다
    grad_original = -1 / (1 - d_gz)
    
    # 포화하지 않음: 기울기가 -1/D이다
    grad_ns = -1 / d_gz
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 손실
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
    
    # 기울기 크기
    axes[1].plot(d_gz, np.abs(grad_original), label='|∂(log(1-D))/∂D|')
    axes[1].plot(d_gz, np.abs(grad_ns), label='|∂(-log D)/∂D|')
    axes[1].set_xlabel('D(G(z))')
    axes[1].set_ylabel('Gradient Magnitude')
    axes[1].set_title('Gradient Magnitudes')
    axes[1].legend()
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    # 기울기 비율
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

---

## 9. 구현

### 손실 갈래

```python
import torch
import torch.nn as nn

class NonSaturatingGANLoss:
    """
    포화하지 않는 맞겨루기 만들개 손실.
    
    만들개는 log(1 - D(G(z)))을 가장 작게 하는 대신 log(D(G(z)))을 가장 크게 한다.
    G이 서툴 때 더 센 기울기를 준다.
    """
    
    def __init__(self):
        self.criterion = nn.BCELoss()
    
    def discriminator_loss(self, d_real, d_fake):
        """
        여느 가름개 잃음(본디 GAN과 같다).
        
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
        포화하지 않는 만들개 손실.
        
        L_G = -E[log D(G(z))]
        
        log(1-D)을 가장 작게 하는 대신 -log(D)을 가장 작게 한다.
        이는 실제 이름표를 쓴 두값 어긋 엔트로피로 짠다.
        """
        batch_size = d_fake.size(0)
        
        # 만들개 익히기에서는 가짜 표본을 실제로 다룬다
        real_labels = torch.ones(batch_size, 1, device=d_fake.device)
        
        # BCE(D(G(z)), 1) = -log(D(G(z)))
        loss = self.criterion(d_fake, real_labels)
        
        return loss
```

### 다른 짜기(두값 어긋 엔트로피 없이)

```python
class NonSaturatingGANLossManual:
    """로그를 드러내어 셈하는 포화하지 않는 손실."""
    
    def discriminator_loss(self, d_real, d_fake):
        """가름개 잃음: -E[log D(x)] - E[log(1 - D(G(z)))]"""
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

---

## 10. 포화하지 않는 손실로 익히기

```python
def train_step_nonsaturating(G, D, real_data, latent_dim, 
                             g_optimizer, d_optimizer, device):
    """포화하지 않는 손실을 쓴 익히기 걸음."""
    
    loss_fn = NonSaturatingGANLoss()
    batch_size = real_data.size(0)
    
    # ==================
    # 가름개를 익힌다
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
    # 만들개를 익힌다(포화하지 않음)
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

---

## 11. 이론적 분석

### 같은 붙박이점

두 손실의 가장 좋은 만들개는 같다.

**본디**: $\min_G \mathbb{E}_z[\log(1 - D^*(G(z)))]$

**포화하지 않음**: $\min_G -\mathbb{E}_z[\log D^*(G(z))]$

가장 좋은 자리에서는 둘 다 $p_g = p_{\text{data}}$이고 $D^*(x) = 0.5$이다.

### 서로 다른 가장 좋게 하기 움직임

기울기의 움직임이 다르다.

| D(G(z)) | 본디 기울기 | 포화하지 않는 기울기 |
|---------|-------------------|-------------|
| 0.01 | -1.01 | -100 |
| 0.1 | -1.11 | -10 |
| 0.5 | -2.0 | -2.0 |
| 0.9 | -10 | -1.11 |
| 0.99 | -100 | -1.01 |

엇갈리는 점 $D(G(z)) = 0.5$에서는 둘의 기울기 크기가 같다.

### 은근한 벌어짐

포화하지 않는 손실은 다른 벌어짐을 가장 작게 한다.

$$\mathcal{L}_G^{\text{NS}} = -\mathbb{E}_{x \sim p_g}[\log D^*(x)]$$

가장 좋은 가름개에서:

$$= -\mathbb{E}_{x \sim p_g}\left[\log \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}\right]$$

이는 젠슨-섀넌 벌어짐이 아니라 **거꾸로 된 쿨백-라이블러 벌어짐**과 이어진다.

---

## 12. 이점

### 1. 앞머리 기울기가 더 세다

G이 서툴러 $D(G(z)) \approx 0$일 때:

- 본디: 기울기 $\approx -1$
- 포화하지 않음: 기울기 $\to -\infty$

### 2. 더 안정된 익히기

겪어 본 바로 포화하지 않는 손실은 다음으로 이어진다.

- 처음 배움이 더 빠르다
- 익히기 움직임이 더 한결같다
- 마지막 표본 품질이 더 낫다

### 3. 단순한 짜기

두값 어긋 엔트로피에서 이름표만 바꾸면 된다.

```python
# 본디: BCE(d_fake, 0) = -log(1 - D(G(z)))을 가장 작게 한다
# 포화하지 않음: BCE(d_fake, 1) = -log(D(G(z)))을 가장 작게 한다
```

---

## 13. 한계

### 1. 여전히 무너질 수 있다

포화하지 않는 손실만으로는 봉우리 무너짐을 막지 못한다.

### 2. 다른 벌어짐

거꾸로 된 쿨백-라이블러를 가장 작게 하며 이는 봉우리를 찾는 움직임이 다르다.

- KL(p_data || p_g): 봉우리 덮기(변분 자기 부호기 같다)
- KL(p_g || p_data): 봉우리 찾기(맞겨루기 만들개 같다)

### 3. 영합 놀이가 아니다

포화하지 않는 손실에서는 G과 D이 영합 놀이를 하지 않는다.

- D은 여전히 본디 목표를 가장 크게 한다
- G은 다른 목표를 가장 크게 한다

이는 모임 살피기에 영향을 줄 수 있다.

---

## 14. 실전 권고

```python
# 여느 맞겨루기 만들개 익히기는 기본으로 포화하지 않는 손실을 쓴다
def recommended_generator_loss(discriminator, fake_data):
    """맞겨루기 만들개를 익히는 권하는 방식."""
    d_fake = discriminator(fake_data)
    
    # 포화하지 않는 손실
    real_labels = torch.ones_like(d_fake)
    loss = nn.BCELoss()(d_fake, real_labels)
    
    return loss
```

---

## 연습문제

**연습문제 1.**
본디 맞겨루기 만들개 손실과 젠슨-섀넌 벌어짐의 관계를 이끌어 내라.

??? success "연습문제 1 풀이"
    가장 좋은 가름개 $D^*(x) = \frac{p_d(x)}{p_d(x) + p_g(x)}$을 값 함수에 넣으면:

    $$V(G, D^*) = \mathbb{E}_{p_d}\left[\log \frac{p_d}{p_d + p_g}\right] + \mathbb{E}_{p_g}\left[\log \frac{p_g}{p_d + p_g}\right]$$

    $$= \mathbb{E}_{p_d}\left[\log \frac{p_d}{(p_d + p_g)/2}\right] + \mathbb{E}_{p_g}\left[\log \frac{p_g}{(p_d + p_g)/2}\right] - 2\log 2$$

    $$= D_{\text{KL}}(p_d \| m) + D_{\text{KL}}(p_g \| m) - 2\log 2 = 2 \, \text{JSD}(p_d \| p_g) - 2\log 2$$

    여기서 $m = (p_d + p_g)/2$이다. $\square$

---

**연습문제 2.**
본디 맞겨루기 만들개 손실, 포화하지 않는 손실, 바서슈타인 손실을 기울기의 움직임과 익히기의 안정 면에서 견주어라.

??? success "연습문제 2 풀이"
    | 손실 | 만들개 목표 | 기울기 문제 | 안정 |
    |------|-------------------|----------------|-----------|
    | **본디** | $\min \log(1 - D(G(z)))$ | $D$이 셀 때 사라진다 | 나쁘다 |
    | **포화하지 않음** | $\max \log D(G(z))$ | 앞머리 기울기가 세다 | 더 낫다 |
    | **바서슈타인** | $\min -D(G(z))$(평가개) | 선형이며 사라지지 않는다 | 가장 좋다 |

    바서슈타인 거리는 포화하는 젠슨-섀넌 벌어짐과 달리 이어져 있고 미분할 수 있으므로 바서슈타인 손실은 (분포가 겹치지 않아도) 어디서나 기울기를 준다.

---

**연습문제 3.**
맞겨루기 만들개 놀이의 내시 균형에서 $p_g = p_{\text{data}}$이고 모든 $x$에 대해 $D(x) = 1/2$임을 보여라.

??? success "연습문제 3 풀이"
    균형에서 $G$은 $G$에 대해 $V(G, D^*)$을 가장 작게 했다. 젠슨-섀넌 벌어짐 나누기에서 $V(G, D^*) = 2\text{JSD}(p_d \| p_g) - 2\log 2 \geq -2\log 2$이며, 등호는 $\text{JSD} = 0$일 때만, 곧 $p_g = p_d$일 때만 성립한다. $p_g = p_d$을 $D^*$에 넣으면 $D^*(x) = \frac{p_d(x)}{p_d(x) + p_d(x)} = \frac{1}{2}$이다. $\square$

---

**연습문제 4.**
WGAN-GP의 립시츠 묶음을 설명하고 무게 자르기보다 기울기 벌점을 더 낫게 여기는 까닭을 밝혀라.

??? success "연습문제 4 풀이"
    WGAN은 평가개(가름개)가 1-립시츠이기를 요구한다. 곧 모든 $x, y$에 대해 $|D(x) - D(y)| \leq \|x - y\|$이다. **무게 자르기**는 무게를 $[-c, c]$으로 묶어 이를 지키게 하지만 다음을 낳는다. (1) 담이를 덜 쓴다(무게 공간의 대부분을 쓰지 않는다). (2) $c$에 따라 기울기가 터지거나 사라진다. (3) 평가개가 단순한 함수만 배운다. **기울기 벌점**(WGAN-GP)은 $\hat{x}$이 실제 표본과 가짜 표본 사이를 메울 때 $\lambda \mathbb{E}_{\hat{x}}[(\|\nabla_x D(\hat{x})\| - 1)^2]$을 더한다. 이는 알맞은 점에서 립시츠 묶음을 부드럽게 지키게 하여 평가개가 담이를 온전히 쓰게 하고 익히기를 더 안정시킨다.

## 정리하며

| 갈래 | 본디 맞겨루기 만들개 손실 |
|--------|-------------------|
| D 손실 | 두값 어긋 엔트로피(실제=1, 가짜=0) |
| G 손실 | min log(1 - D(G(z))) |
| 벌어짐 | 젠슨-섀넌 |
| 문제 | 기울기 포화 |

---

# 포화하지 않는 손실

포화하지 않는 손실은 본디 맞겨루기 만들개의 만들개 손실을 실제에 맞게 고친 것으로, 만들개가 가장 필요로 하는 익히기 앞머리에 더 센 기울기를 준다.

| 갈래 | 본디 손실 | 포화하지 않는 손실 |
|--------|---------------|---------------------|
| **G 목표** | min E[log(1-D)] | min -E[log D] |
| **앞머리 기울기** | 여리다 | 세다 |
| **벌어짐** | 젠슨-섀넌 | 거꾸로 된 쿨백-라이블러와 이어짐 |
| **영합 놀이** | 그렇다 | 아니다 |
| **쓰임** | 이론용 | 실제용(기본) |

포화하지 않는 손실은 맞겨루기 만들개 익히기의 **여느 고르기**이다. 본디 손실과 같은 붙박이점을 지키면서 실제에 더 알맞은 가장 좋게 하기 움직임을 준다.
