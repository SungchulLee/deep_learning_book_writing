# 사전 분포 고르기
숨은 공간의 사전 분포 $p(z)$을 고르고 설계하기.

---

## 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

- 변분 자기 부호기의 만들어 내기와 벌주기에서 사전 분포의 노릇 설명하기
- 표준 정규 사전 분포가 왜 붙박이 고름인지 이해하기
- 다른 사전 분포와 그 맞바꿈 적기
- 표준이 아닌 사전 분포를 쓴 변분 자기 부호기 짜기

---

## 사전 분포의 노릇

사전 분포 $p(z)$은 변분 자기 부호기에서 결정적인 노릇을 둘 한다. **익히는 동안** KL 벌어짐 $D_{KL}(q_\phi(z|x) \| p(z))$이 부호기가 내놓는 것을 사전 분포에 맞도록 옭아매어 숨은 공간의 기하를 빚는다. **만들어 낼 때**는 $z \sim p(z)$을 뽑아 푸므로 사전 분포가 숨은 부호를 뽑는 분포를 정한다.

사전 분포와 모은 사후 분포 $q_\phi(z) = \mathbb{E}_{p_{\text{data}}}[q_\phi(z|x)]$이 어긋나면 만들어 내기가 나빠진다. 곧 풀개가 만들어 낼 때 익히는 동안 본 적 없는 숨은 부호를 만난다.

---

## 표준 정규 사전 분포

### 왜 N(0, I)인가?

방향에 무관한 정규 사전 분포 $p(z) = \mathcal{N}(0, I)$이 표준 고름인 까닭은 여럿이다:

| 성질 | 좋은 점 |
|----------|---------|
| **닫힌 꼴 KL** | 정규 $q_\phi(z\|x)$과의 KL 벌어짐을 손으로 셈할 수 있다 |
| **뽑기 쉬움** | 뽑기가 뻔하다 |
| **방향에 무관함** | 숨은 공간에 두드러진 방향이 없다 |
| **최대 엔트로피** | 평균과 흩어짐이 주어진 분포 가운데 정규 분포의 엔트로피가 가장 크다 |

### 한계

표준 정규 사전 분포는 지나치게 옭아맬 수 있다. 모은 사후 분포가 봉우리 하나이고 공처럼 대칭이라고 여기는데 이는 자료의 참 짜임과 맞지 않을 수 있다. 뚜렷한 무리나 다양체 짜임이 있는 복잡한 자료에서는 정규 사전 분포가 덜 좋은 숨은 배치를 강요한다.

---

## 사전 분포의 구멍 문제

### 무엇인가?

**사전 분포의 구멍**은 모은 사후 분포 $q_\phi(z)$이 사전 분포 $p(z)$과 맞지 않을 때 생긴다. $p(z)$에는 질량이 있으나 $q_\phi(z)$에는 없는 자리가 "구멍"이다. 풀개가 그런 숨은 부호에 뜻 있는 것을 내는 법을 배우지 못해 만들어 내기 품질이 나빠진다.

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

### 결과

사전 분포에서 뽑으면 이런 구멍에 떨어져 품질이 낮거나 실제 같지 않은 것이 나올 수 있다. 자료 짜임이 복잡할 때의 근본 한계이다.

---

## 다른 사전 분포

### 정규 분포를 섞은 사전 분포

섞은 사전 분포는 봉우리 여럿인 짜임을 잡을 수 있다:

$$p(z) = \sum_{k=1}^{K} \pi_k \mathcal{N}(z; \mu_k, \Sigma_k)$$

```python
import torch
import torch.nn as nn

class GaussianMixturePrior(nn.Module):
    """배울 수 있는 정규 섞기 사전 분포."""
    
    def __init__(self, latent_dim, num_components=10):
        super().__init__()
        self.num_components = num_components
        
        # 배울 수 있는 섞기 매개변수
        self.logits = nn.Parameter(torch.zeros(num_components))
        self.means = nn.Parameter(torch.randn(num_components, latent_dim) * 0.5)
        self.logvars = nn.Parameter(torch.zeros(num_components, latent_dim))
    
    def log_prob(self, z):
        """섞기 아래에서 log p(z)을 셈한다."""
        # z: [묶음, 숨은 차원]
        # 퍼뜨리려 넓힌다: [묶음, K, 숨은 차원]
        z_exp = z.unsqueeze(1)
        
        # 성분마다의 로그 확률
        log_var = self.logvars.unsqueeze(0)
        means = self.means.unsqueeze(0)
        
        log_p_per_component = -0.5 * (log_var + (z_exp - means).pow(2) / log_var.exp())
        log_p_per_component = log_p_per_component.sum(dim=2)  # [묶음, K]
        
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

### VampPrior(사후 분포를 변분으로 섞기)

VampPrior(Tomczak & Welling, 2018)은 배운 가짜 들임에서 따진 부호기 내놓기를 섞어 사전 분포를 정한다:

$$p(z) = \frac{1}{K}\sum_{k=1}^{K} q_\phi(z | u_k)$$

여기서 $u_1, \ldots, u_K$은 자료 공간의 배울 수 있는 가짜 들임이다. 그러면 사전 분포가 모은 사후 분포의 짜임에 저절로 맞는다.

```python
class VampPrior(nn.Module):
    """VampPrior: 사후 분포를 변분으로 섞기."""
    
    def __init__(self, encoder, input_dim, num_pseudoinputs=100):
        super().__init__()
        self.encoder = encoder
        self.pseudoinputs = nn.Parameter(torch.randn(num_pseudoinputs, input_dim) * 0.05)
    
    def get_prior_params(self):
        """가짜 들임에서 섞기 성분 매개변수를 얻는다."""
        with torch.no_grad():
            mu, logvar = self.encoder(torch.sigmoid(self.pseudoinputs))
        return mu, logvar
```

### 고르게 하는 흐름으로 배우는 사전 분포

고르게 하는 흐름은 표준 정규 분포를 표현력이 더 좋은 사전 분포로 바꿀 수 있다:

$$z_0 \sim \mathcal{N}(0, I), \quad z = f(z_0), \quad p(z) = p(z_0)|det \frac{\partial f^{-1}}{\partial z}|$$

그러면 밀도를 다룰 수 있게 지키면서도 사전 분포가 복잡하고 봉우리 여럿인 짜임을 잡는다.

---

## 조건부 사전 분포

조건부 변분 자기 부호기에서는 사전 분포가 조건 앎에 기댈 수 있다:

$$p(z|y) = \mathcal{N}(z; \mu_{\text{prior}}(y), \sigma^2_{\text{prior}}(y))$$

배운 조건부 사전 분포는 모든 조건이 같은 $\mathcal{N}(0, I)$ 사전 분포를 나눠 쓰도록 강요하는 대신 갈래나 조건마다 숨은 공간의 다른 자리를 자연스럽게 차지하게 한다.

---

## 실전 권고

대개의 쓰임새에서는 표준 정규 분포 $\mathcal{N}(0, I)$으로 시작하라. 단순하고 잘 알려졌으며 웬만큼 복잡한 자료에 잘 통한다. 다시 세우기는 좋은데 만들어 내기 품질이 나쁠 때, 자료에 뚜렷한 무리 짜임이 있을 때(섞은 사전 분포), 사전 분포가 자료에 맞춰지길 바랄 때(VampPrior), 복잡하고 봉우리 여럿인 분포를 다룰 때(흐름 바탕 사전 분포) 다른 사전 분포를 헤아려라.

---

## 요약

| 사전 분포 | KL 셈하기 | 표현력 | 복잡함 |
|-------|---------------|----------------|------------|
| **표준 정규** | 닫힌 꼴 | 낮음 | 가장 적음 |
| **정규 분포 섞기** | 뽑기가 필요 | 가운데 | 웬만함 |
| **VampPrior** | 뽑기가 필요 | 높음 | 웬만함 |
| **흐름 바탕** | 정확(변수 바꿈) | 높음 | 높음 |
| **조건부** | 꼴에 따라 다름 | 일마다 다름 | 웬만함 |

---

## 다음은

다음 절은 부호기가 들임 자료를 무시하는 흔한 익히기 어긋남인 사후 분포 무너짐을 다룬다.

## 연습문제

### 연습 1: 사전 분포 그려 보기

MNIST에서 표준 정규 사전 분포와 성분 10개짜리 섞은 사전 분포로 변분 자기 부호기를 익혀라. 모은 사후 분포와 사전 분포를 2차원으로 그려 보아라.

### 연습 2: 만들어 내기 품질

표준 사전 분포와 섞은 사전 분포에서 만든 표본의 FID 점수(또는 눈에 보이는 품질)를 견주어라.

### 연습 3: VampPrior

가짜 들임 50개로 VampPrior을 짜라. 배운 가짜 들임이 익히기 자료를 닮았는가?

---
