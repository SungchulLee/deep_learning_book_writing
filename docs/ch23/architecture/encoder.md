# 부호기 그물
고르게 나눈 변분 추론: 자료를 어림 사후 분포에 옮기는 법 배우기.

---

## 1. 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

- 분포 매개변수를 내놓는 부호기 얼개 설계하기
- $\mu$과 $\log\sigma^2$을 위한 두 머리 얼개 짜기
- 고르게 나눈 추론과 그 셈의 이점 이해하기
- 자료 갈래에 따라 여러 층 인식개 부호기와 누비기 부호기 세우기

---

## 2. 부호기의 노릇

### 추론에서 고르게 나누기로

고전 변분 추론에서는 자료 점 $x_i$마다 따로 변분 매개변수를 가장 좋게 한다. 변분 자기 부호기의 부호기는 **고르게 나눈 추론**을 한다. 곧 신경망 하나 $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma^2_\phi(x))$이 어떤 들임이든 그 어림 사후 분포에 옮긴다. 표본마다의 가장 좋음을 셈의 효율과 맞바꾸는 것이며, 추론에 앞먹임 한 번만 든다.

### 부호기가 내놓는 것

부호기 그물은 들임 $x$을 받아 벡터 둘을 낸다:

- **평균** $\mu_\phi(x) \in \mathbb{R}^d$: 어림 사후 분포의 가운데
- **로그 흩어짐** $\log\sigma^2_\phi(x) \in \mathbb{R}^d$: 퍼진 정도(수치 안정을 위해 로그 공간에서)

```
Input x ──► [Shared Hidden Layers] ──┬──► fc_mu ──► μ ∈ ℝ^d
                                      │
                                      └──► fc_logvar ──► log(σ²) ∈ ℝ^d
```

---

## 3. 여러 층 인식개 부호기

### 구조

```python
import torch
import torch.nn as nn

class MLPEncoder(nn.Module):
    """
    벡터 들임(예컨대 펼친 MNIST)을 위한 온전히 이어진 부호기.
    
    얼개: input_dim → hidden_dim → hidden_dim → (μ, logvar)
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
            x: 들임 텐서 [묶음 크기, 들임 차원]
        반환값:
            mu: 평균 [묶음 크기, 숨은 차원]
            logvar: 로그 흩어짐 [묶음 크기, 숨은 차원]
        """
        h = self.shared(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
```

### 왜 머리를 나누는가?

평균과 로그 흩어짐은 내놓기의 성질이 다르다. 평균 $\mu$은 갇혀 있지 않고 숨은 특징의 온전한 표현력에서 덕을 본다. 로그 흩어짐 $\log\sigma^2$은 불확실함을 다스리며 흔히 다른 잣수로 모인다. 마지막 쏘기만 빼고 층을 모두 나눠 쓰면 그물이 공통 특징 나타냄을 배우면서 마지막 대응만 특화할 수 있다.

---

## 4. 누비기 부호기

### 그림 자료에

```python
class ConvEncoder(nn.Module):
    """
    그림 들임을 위한 누비기 부호기.
    
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
            x: 들임 그림 [묶음 크기, 채널, 높이, 너비]
        """
        h = self.conv_layers(x)
        h = h.view(h.size(0), -1)  # 펼친다
        return self.fc_mu(h), self.fc_logvar(h)
```

---

## 5. 로그 흩어짐 매개변수화

### 왜 시그마를 곧바로 내놓지 않는가?

표준 편차 $\sigma$은 양수여야 한다. $\sigma$을 곧바로 내놓으면 양수임을 강제할 깨어남 함수(부드러운 정류 선형 같은)가 필요하다. 대신 $\log\sigma^2$을 내놓고 $\sigma = \exp(0.5 \cdot \log\sigma^2)$을 셈하면 깨어남 함수 없이 저절로 양수가 된다.

| 매개변수화 | 양수임 | 기울기의 몸가짐 | 표준 고름 |
|-----------------|------------|-------------------|-----------------|
| $\sigma$ 곧바로 | 부드러운 정류 선형이나 지수가 필요 | 기울기가 사라질 수 있다 | 아니오 |
| $\sigma^2$ | 부드러운 정류 선형이 필요 | 제곱 잣수 문제 | 아니오 |
| $\log\sigma^2$ | 지수로 저절로 | 얌전하다 | **예** |

### 수치적 안정성

로그 흩어짐이 아주 크거나 작으면 값을 가두어 수치 넘침을 막는다:

```python
def safe_reparameterize(mu, logvar, max_logvar=10.0):
    """가둔 로그 흩어짐으로 다시 매개변수화한다."""
    logvar = torch.clamp(logvar, min=-max_logvar, max=max_logvar)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps
```

---

## 6. 고르게 나누기의 벌어짐

### 무엇인가?

**고르게 나누기의 벌어짐**은 고르게 나눈 부호기가 이룬 증거 하한과 표본마다 가장 좋게 해서 이룰 수 있는 증거 하한의 차이다:

$$\text{고르게 나누기의 벌어짐} = \mathcal{L}^*(\theta; x) - \mathcal{L}(\theta, \phi; x)$$

여기서 $\mathcal{L}^*$은 $x$마다 따로 가장 좋은 $q^*(z|x)$을 쓴다.

### 왜 중요한가

부호기는 모든 자료 점에 두루 통해야 하므로 들임마다 사후 분포를 완벽히 어림할 수 없다. 이것이 고르게 나누기의 값이다. 곧 표본마다의 가장 좋음을 내주고 빠른 추론을 얻는다.

### 벌어짐 줄이기

고르게 나누기의 벌어짐을 줄이는 전략에는 표현력이 더 좋은 부호기 얼개 쓰기, 부호기가 내놓는 것에 고르게 하는 흐름 쓰기($q_\phi(z|x)$이 흐름 바탕 분포가 된다), 시험 때 기울기로 몇 걸음 다듬기(반쯤 고르게 나눈 추론)가 있다.

---

## 7. 설계 지침

### 숨은 차원 고르기

부호기는 자료에서 사후 매개변수로 가는 대응을 배울 담이를 넉넉히 갖춰야 한다. 어림잡아 숨은 차원은 적어도 숨은 차원의 2~4배여야 하고 숨은 층의 수는 자료의 복잡함에 따라 늘려야 한다.

### 초기화

여느 첫자리매김(자비에/허)이 대개 잘 통한다. 어떤 이는 단위 흩어짐이 아니라 웬만한 흩어짐으로 시작하려 로그 흩어짐 머리의 치우침을 작은 음수(예컨대 -1)로 첫자리매김한다:

```python
# 첫 흩어짐이 웬만하도록 로그 흩어짐 머리의 치우침을 첫자리매김한다
nn.init.constant_(encoder.fc_logvar.bias, -1.0)
```

### 묶음 고르게 맞추기

부호기의 묶음 고르게 맞추기는 익히기 안정에 도움이 되지만 $\mu$과 $\log\sigma^2$ 머리 뒤에는 쓰면 **안 된다**. 분포 매개변수화를 흔들기 때문이다.

---

## 8. 다음은

다음 절은 숨은 부호를 자료 공간으로 되돌리는 풀개 그물을 다룬다.

---

## 연습문제

### 연습 1: 얼개 덜어내기

MNIST에서 숨은 층이 1, 2, 4개인 부호기 얼개를 견주어라. 다시 세우기 품질과 익히기 빠르기를 재어라.

### 연습 2: 숨은 차원 훑기

숨은 차원 $d \in \{2, 8, 16, 32, 64, 128\}$으로 부호기를 익혀라. $d$에 대한 다시 세우기 어긋남과 KL 벌어짐을 그려라.

### 연습 3: 고르게 나누기의 벌어짐

익힌 변분 자기 부호기에서 부호기가 낸 증거 하한과, 표본마다 $(μ, \log σ²)$을 기울기로 100걸음 가장 좋게 한 뒤의 증거 하한을 견주어라. 벌어짐을 알려라.

---

## 정리하며

| 조각 | 몫 | 짜기 |
|-----------|---------|----------------|
| **나눠 쓰는 층** | 들임에서 특징을 뽑는다 | 정류 선형을 곁들인 온전히 이어진 층이나 누비기 층 |
| **평균 머리** | $\mu_\phi(x)$을 내놓는다 | 선형 쏘기, 갇히지 않음 |
| **로그 흩어짐 머리** | $\log\sigma^2_\phi(x)$을 내놓는다 | 선형 쏘기, 옭아매지 않음 |
| **다시 매개변수화** | $z = \mu + \sigma \cdot \epsilon$을 뽑는다 | 뒤먹임 퍼뜨리기를 되게 한다 |

---
