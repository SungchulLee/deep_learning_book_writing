# KL 벌어짐 항
변분 자기 부호기에서 KL 벌주기의 수학 성질과 셈하기 세부.

---

## 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

- KL 벌어짐과 그 핵심 성질을 정의하기
- 정규 분포의 닫힌 꼴 KL 벌어짐을 이끌어 내기
- 앞 KL과 뒤 KL을 설명하고 변분 자기 부호기가 왜 뒤 KL을 쓰는지 밝히기
- 변분 자기 부호기 익히기에서 KL 벌어짐의 앎 이론상 노릇 이해하기

---

## KL 벌어짐: 정의와 성질

### 정의

분포 $q$에서 분포 $p$으로의 **쿨백-라이블러 벌어짐**은 다음과 같다:

$$D_{KL}(q \| p) = \mathbb{E}_{q(x)}\left[\log \frac{q(x)}{p(x)}\right] = \int q(x) \log \frac{q(x)}{p(x)} dx$$

### 근본 성질

| 성질 | 말 | 뜻하는 바 |
|----------|-----------|-------------|
| **음이 아님** | $D_{KL}(q \| p) \geq 0$ | 증거 하한이 하한임을 보장한다 |
| **같을 때만 0** | $D_{KL}(q \| p) = 0 \Leftrightarrow q = p$ | 어림이 완벽하면 KL이 0이다 |
| **대칭이 아님** | 두루 $D_{KL}(q \| p) \neq D_{KL}(p \| q)$ | 방향이 중요하다 |
| **잣대가 아님** | 삼각 부등식이 성립하지 않는다 | 거리로 쓸 수 없다 |

### 음이 아님의 증명(깁스 부등식)

볼록 함수 $-\log$에 옌센 부등식을 쓰면:

$$D_{KL}(q \| p) = -\mathbb{E}_q\left[\log \frac{p(x)}{q(x)}\right] \geq -\log \mathbb{E}_q\left[\frac{p(x)}{q(x)}\right] = -\log \int p(x) dx = 0$$

---

## 엔트로피와 엇갈린 엔트로피의 관계

### 근본 관계

$$\underbrace{H(q, p)}_{\text{Cross-entropy}} = \underbrace{H(q)}_{\text{Entropy}} + \underbrace{D_{KL}(q \| p)}_{\text{KL divergence}}$$

여기서 각 기호는 다음과 같다.

- **엔트로피:** $H(q) = -\mathbb{E}_q[\log q(x)]$ — 줄일 수 없는 불확실함
- **엇갈린 엔트로피:** $H(q, p) = -\mathbb{E}_q[\log p(x)]$ — $p$의 부호를 쓸 때 드는 비트
- **KL 벌어짐:** $q$ 대신 $p$을 써서 더 드는 비트

$D_{KL} \geq 0$이므로 엇갈린 엔트로피는 늘 엔트로피 이상이다. $q$이 고정이면 엇갈린 엔트로피를 가장 작게 하는 것은 KL 벌어짐을 가장 작게 하는 것과 같다.

---

## 순방향 KL과 역방향 KL

### 앞 KL: D_KL(p || q)|KL(p || q) — 평균을 좇음

$$D_{KL}(p \| q) = \mathbb{E}_p\left[\log \frac{p(x)}{q(x)}\right]$$

이는 $p$의 확률이 높은 곳에서 $q$의 확률이 낮으면 벌을 준다. 그 결과 $q$이 $p$의 확률이 낮은 자리에도 확률을 주더라도 $p$의 **모든 봉우리를 덮는다**.

### 뒤 KL: D_KL(q || p)|KL(q || p) — 봉우리를 좇음

$$D_{KL}(q \| p) = \mathbb{E}_q\left[\log \frac{q(x)}{p(x)}\right]$$

이는 $p$의 확률이 낮은 곳에서 $q$의 확률이 높으면 벌을 준다. 그 결과 $q$이 $p$의 **확률이 높은 자리에 몰리며** 어떤 봉우리는 놓칠 수 있다.

### 변분 자기 부호기는 뒤 KL을 쓴다

증거 하한에서는 어림 사후 분포에서 사전 분포로의 뒤 KL인 $D_{KL}(q_\phi(z|x) \| p(z))$을 가장 작게 한다. 그러면 $q_\phi(z|x)$이 $p(z)$의 확률이 낮은 자리에 질량을 두지 않게 되어 숨은 부호가 사전 분포의 "받침" 안에 머문다.

더 근본으로는 증거 하한의 벌어짐이 어림 사후 분포에서 참 사후 분포로의 뒤 KL인 $D_{KL}(q_\phi(z|x) \| p_\theta(z|x))$이며, 이는 부호기가 봉우리를 모두 덮으려 퍼지기보다 참 사후 분포의 확률 높은 자리에 몰리기 쉽다는 뜻이다.

---

## 정규분포의 KL 발산

### 한 변수 경우

$q = \mathcal{N}(\mu_1, \sigma_1^2)$이고 $p = \mathcal{N}(\mu_2, \sigma_2^2)$이면:

$$D_{KL}(q \| p) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

### 변분 자기 부호기의 특별한 경우: q_phi(z|x)과 표준 정규 분포

$q = \mathcal{N}(\mu, \sigma^2)$이고 $p = \mathcal{N}(0, 1)$이면:

$$D_{KL}(q \| p) = -\frac{1}{2}\left(1 + \log\sigma^2 - \mu^2 - \sigma^2\right)$$

**이끌어 내기:**

$$D_{KL} = \mathbb{E}_q\left[\log \frac{q(z)}{p(z)}\right] = \mathbb{E}_q[\log q(z)] - \mathbb{E}_q[\log p(z)]$$

$$= -\frac{1}{2}(1 + \log 2\pi\sigma^2) - \left(-\frac{1}{2}\mathbb{E}_q[z^2] - \frac{1}{2}\log 2\pi\right)$$

$$= -\frac{1}{2}\log\sigma^2 - \frac{1}{2} + \frac{1}{2}\mathbb{E}_q[z^2]$$

$\mathbb{E}_q[z^2] = \mu^2 + \sigma^2$(정규 분포의 2차 적률)이므로:

$$= -\frac{1}{2}\log\sigma^2 - \frac{1}{2} + \frac{1}{2}(\mu^2 + \sigma^2) = -\frac{1}{2}(1 + \log\sigma^2 - \mu^2 - \sigma^2)$$

### 여러 변수 경우

$q = \mathcal{N}(\mu, \text{diag}(\sigma_1^2, \ldots, \sigma_d^2))$이고 $p = \mathcal{N}(0, I)$이면:

$$D_{KL}(q \| p) = -\frac{1}{2}\sum_{j=1}^{d}(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2)$$

이는 **차원에 걸친 합**으로 쪼개지므로 셈하기 편하고 차원마다 살필 수 있다.

---

## PyTorch 구현

```python
import torch

def kl_divergence_standard_normal(mu, logvar):
    """
    N(mu, diag(exp(logvar)))에서 N(0, I)으로의 KL 벌어짐.
    
    D_KL = -0.5 * sum(1 + log(σ²) - μ² - σ²)
    
    인수:
        mu: 평균 [묶음 크기, 숨은 차원]
        logvar: 로그 흩어짐 [묶음 크기, 숨은 차원]
    
    반환값:
        표본마다의 KL 벌어짐 [묶음 크기]
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def kl_divergence_two_gaussians(mu1, logvar1, mu2, logvar2):
    """
    대각 정규 분포 둘 사이의 KL 벌어짐.
    
    D_KL(q || p). 여기서 q = N(mu1, exp(logvar1)), p = N(mu2, exp(logvar2))
    
    인수:
        mu1, logvar1: q의 매개변수
        mu2, logvar2: p의 매개변수
    
    반환값:
        표본마다의 KL 벌어짐 [묶음 크기]
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
        mu: 평균 [묶음 크기, 숨은 차원]
        logvar: 로그 흩어짐 [묶음 크기, 숨은 차원]
    
    반환값:
        차원마다의 평균 KL [숨은 차원]
    """
    kl_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl_dim.mean(dim=0)
```

---

## KL 항 살피기

### KL이 0일 때

모든 $x$에 대해 $q_\phi(z|x) = \mathcal{N}(0, I)$이면 $D_{KL}(q_\phi(z|x) \| p(z)) = 0$이며 부호기가 들임을 아예 무시한다는 뜻이다. 이것이 **사후 분포 무너짐**이다. 곧 부호기가 들임과 무관하게 같은 분포를 낸다.

### KL이 클 때

KL이 크면 부호기가 숨은 공간을 많이 써서 들임 $x$마다 상당한 앎을 담는다는 뜻이다. 다시 세우기에는 좋지만 KL이 지나치면 숨은 공간이 사전 분포에서 크게 벗어나 만들어 내기 품질을 해칠 수 있다.

### 차원마다 살피기

숨은 차원마다 $D_{KL}$을 살피면 어느 차원이 "깨어 있고"(자료의 앎을 담고) 어느 차원이 "잠들었는지"(사전 분포로 무너졌는지) 드러난다:

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

---

## 서로 앎과의 이음

자료 분포에 걸친 KL의 기댓값은 서로 앎과 맞닿는다:

$$\mathbb{E}_{p_{\text{data}}(x)}[D_{KL}(q_\phi(z|x) \| p(z))] = I_q(X; Z) + D_{KL}(q_\phi(z) \| p(z))$$

따라서 KL 항은 자료와 숨은 부호 사이의 서로 앎과, 모은 사후 분포와 사전 분포의 어긋남에 모두 벌을 준다. 이 쪼개기가 베타 변분 자기 부호기와 얽힘 풀기에 쓰이는 전체 상관 쪼개기를 이해하는 열쇠이다.

---

## 요약

| 개념 | 식 | 변분 자기 부호기에서의 노릇 |
|---------|---------|-------------|
| **표준 정규 분포로의 KL** | $-\frac{1}{2}\sum(1 + \log\sigma^2 - \mu^2 - \sigma^2)$ | 벌주기 항 |
| **앞 KL** | $D_{KL}(p \| q)$ | 평균을 좇음(변분 자기 부호기에서는 쓰지 않는다) |
| **뒤 KL** | $D_{KL}(q \| p)$ | 최빈값 찾기(VAE 익히기) |
| **KL = 0** | $q(z\|x) = p(z)$ | 사후 분포 무너짐 |
| **쪼개기** | $\text{서로 앎} + \text{가장자리 KL}$ | 앎 이론의 관점 |

---

## 다음은

다음 절은 증거 하한의 가능도 조각과 여러 풀개 분포 고름을 다루는 다시 세우기 항을 살핀다.

## 연습문제

### 익힘 1: 닫힌 꼴 이끌어 내기

정의에서 KL 벌어짐 식 $D_{KL}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, 1))$을 걸음마다 보이며 이끌어 내어라.

### 익힘 2: 수치로 따져 보기

표본 10만 개를 쓴 몬테카를로 어림과 견주어 닫힌 꼴 KL을 확인하라.

### 익힘 3: 앞 KL과 뒤 KL 견주기

봉우리 둘인 목표 $p(x) = 0.5\mathcal{N}(-3, 1) + 0.5\mathcal{N}(3, 1)$과 정규 어림 $q(x) = \mathcal{N}(\mu, \sigma^2)$에 대해 앞 KL과 뒤 KL 아래에서 가장 좋은 $q$을 수치로 찾아라. 차이를 그려 보아라.

---
