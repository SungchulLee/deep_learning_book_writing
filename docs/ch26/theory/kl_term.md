# KL 벌어짐 항
변분 자기 부호기에서 KL 벌주기의 수학 성질과 셈하기 세부.

---

## 1. 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

- KL 벌어짐과 그 핵심 성질을 정의하기
- 정규 분포의 닫힌 꼴 KL 벌어짐을 이끌어 내기
- 앞 KL과 뒤 KL을 설명하고 변분 자기 부호기가 왜 뒤 KL을 쓰는지 밝히기
- 변분 자기 부호기 익히기에서 KL 벌어짐의 앎 이론상 노릇 이해하기

---

## 2. KL 벌어짐: 정의와 성질

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

## 3. 엔트로피와 엇갈린 엔트로피의 관계

### 근본 관계

$$\underbrace{H(q, p)}_{\text{Cross-entropy}} = \underbrace{H(q)}_{\text{Entropy}} + \underbrace{D_{KL}(q \| p)}_{\text{KL divergence}}$$

여기서 각 기호는 다음과 같다.

- **엔트로피:** $H(q) = -\mathbb{E}_q[\log q(x)]$ — 줄일 수 없는 불확실함
- **엇갈린 엔트로피:** $H(q, p) = -\mathbb{E}_q[\log p(x)]$ — $p$의 부호를 쓸 때 드는 비트
- **KL 벌어짐:** $q$ 대신 $p$을 써서 더 드는 비트

$D_{KL} \geq 0$이므로 엇갈린 엔트로피는 늘 엔트로피 이상이다. $q$이 고정이면 엇갈린 엔트로피를 가장 작게 하는 것은 KL 벌어짐을 가장 작게 하는 것과 같다.

---

## 4. 순방향 KL과 역방향 KL

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

## 5. 정규분포의 KL 발산

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

## 6. PyTorch 구현

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
        mu1, logvar1: q의 매개변수
        mu2, logvar2: p의 매개변수
    
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

---

## 7. KL 항 살피기

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

## 8. 서로 앎과의 이음

자료 분포에 걸친 KL의 기댓값은 서로 앎과 맞닿는다:

$$\mathbb{E}_{p_{\text{data}}(x)}[D_{KL}(q_\phi(z|x) \| p(z))] = I_q(X; Z) + D_{KL}(q_\phi(z) \| p(z))$$

따라서 KL 항은 자료와 숨은 부호 사이의 서로 앎과, 모은 사후 분포와 사전 분포의 어긋남에 모두 벌을 준다. 이 쪼개기가 베타 변분 자기 부호기와 얽힘 풀기에 쓰이는 전체 상관 쪼개기를 이해하는 열쇠이다.

---

## 9. 다음은

다음 절은 증거 하한의 가능도 조각과 여러 풀개 분포 고름을 다루는 다시 세우기 항을 살핀다.

---

## 연습문제


<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
두 가우시안 $q = \mathcal{N}(\mu, \sigma^2)$와 $p = \mathcal{N}(0,1)$ 사이의 KL 벌어짐을 닫힌 꼴로 적어라. $\mu = 0$, $\sigma = 1$이면 얼마인가?

</div>

??? success "연습문제 1 풀이"
    $$D_{\mathrm{KL}}\big(\mathcal{N}(\mu,\sigma^2) \,\|\, \mathcal{N}(0,1)\big)
      = \tfrac{1}{2}\left( \mu^2 + \sigma^2 - 1 - \log \sigma^2 \right)$$

    $\mu = 0$, $\sigma = 1$이면 $\tfrac{1}{2}(0 + 1 - 1 - 0) = 0$이다. 두 분포가 같으니
    당연하다.

    차원이 $k$개이고 서로 독립이면 차원마다 더하면 된다.

    $$D_{\mathrm{KL}} = \tfrac{1}{2} \sum_{j=1}^{k}
      \left( \mu_j^2 + \sigma_j^2 - 1 - \log \sigma_j^2 \right)$$

    코드에서 보는 `-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()`이 정확히
    이 식이다. 부호를 바꾸어 정리했을 뿐이다.

    이 항이 닫힌 꼴로 떨어지는 것이 변분 자기 부호기를 다루기 쉽게 만드는 큰 이유다.
    표집 없이 정확히 셈할 수 있으므로 기울기에 잡음이 섞이지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
KL 항이 $\mu$와 $\sigma$를 각각 어느 쪽으로 미는가? 식의 세 조각을 나누어 설명하라.

</div>

??? success "연습문제 2 풀이"
    $\tfrac{1}{2}(\mu^2 + \sigma^2 - 1 - \log\sigma^2)$을 조각내어 보자.

    | 조각 | 미는 방향 |
    |---|---|
    | $\mu^2$ | $\mu \to 0$. 코드를 원점으로 당긴다 |
    | $\sigma^2$ | $\sigma \to 0$. 퍼짐을 줄인다 |
    | $-\log\sigma^2$ | $\sigma \to \infty$. 퍼짐을 **늘린다** |

    뒤의 두 조각이 맞선다. $\sigma^2 - \log\sigma^2$을 미분해 0으로 두면
    $1 - 1/\sigma^2 = 0$에서 $\sigma = 1$이 나오므로, 둘의 균형점이 정확히
    $\sigma = 1$이다.

    곧 KL 항은 **각 코드를 원점으로 당기면서 퍼짐은 1에 맞추라**고 말한다. 앞의 힘만
    있으면 모든 코드가 0으로 뭉쳐 구별이 사라지고, 뒤의 힘만 있으면 퍼짐이 멋대로
    커진다. 두 힘이 함께 있어야 코드가 $\mathcal{N}(0,I)$ 모양으로 자리 잡는다.

    $-\log\sigma^2$ 항이 특히 중요하다. 이것이 없으면 $\sigma \to 0$이 되어 부호기가
    결정적 함수로 되돌아가고, 그러면 [자기 부호기](../../ch25/index.md)와 같아진다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
증거 하한(ELBO)을 유도하라. 왜 $\log p(x)$를 직접 최대화하지 않는가?

</div>

??? success "연습문제 3 풀이"
    직접 최대화하지 못하는 까닭부터 보자.

    $$p(x) = \int p(x \mid z)\, p(z)\, dz$$

    이 적분은 숨은 변수 공간 전체에 걸친 것이라 닫힌 꼴로 풀리지 않고, 몬테카를로로
    어림하자니 $p(z)$에서 뽑은 대부분의 $z$가 이 $x$와 무관해 $p(x \mid z)$가 0에
    가깝다. 표본이 엄청나게 많이 필요하다.

    그래서 **제안 분포** $q(z \mid x)$를 끌어들인다. 임의의 $q$에 대해

    $$\log p(x) = \log \int p(x \mid z) p(z) \, dz
      = \log \int q(z \mid x) \frac{p(x \mid z) p(z)}{q(z \mid x)} dz$$

    이고, 옌센 부등식($\log$이 오목하므로 $\log \mathbb{E} \ge \mathbb{E} \log$)에 따라

    $$\log p(x) \;\ge\; \mathbb{E}_{q(z \mid x)}
      \left[ \log \frac{p(x \mid z) p(z)}{q(z \mid x)} \right]
      = \underbrace{\mathbb{E}_{q}[\log p(x \mid z)]}_{\text{다시 세우기}}
      - \underbrace{D_{\mathrm{KL}}(q(z \mid x) \,\|\, p(z))}_{\text{KL}}$$

    오른쪽이 **증거 하한**이다. $\square$

    여기서 두 가지가 드러난다.

    **첫째, 우리가 따로 재던 두 가지가 한 식의 두 항이다.** 다시 세우기와 KL은 임의로
    섞은 것이 아니라 $\log p(x)$의 하한을 유도했더니 나온 것이다.

    **둘째, 틈이 정확히 얼마인지 알 수 있다.** 다르게 전개하면

    $$\log p(x) = \text{ELBO} + D_{\mathrm{KL}}\big(q(z \mid x) \,\|\, p(z \mid x)\big)$$

    이므로, 하한이 느슨한 정도가 곧 제안 분포와 **참된 사후 분포** 사이의 KL이다.
    $q$가 참된 사후 분포와 같아지면 등호가 성립한다. 부호기를 익히는 일이 곧 그
    틈을 줄이는 일이다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
KL 벌어짐은 대칭이 아니다. 변분 자기 부호기가 $D_{\mathrm{KL}}(q \| p)$를 쓰고 $D_{\mathrm{KL}}(p \| q)$를 쓰지 않는 까닭은 무엇인가?

</div>

??? success "연습문제 4 풀이"
    먼저 두 방향의 성격이 다르다.

    | | 벌하는 것 | 결과 |
    |---|---|---|
    | $D_{\mathrm{KL}}(q \| p)$ | $p$가 작은 곳에서 $q$가 큰 것 | $q$가 $p$ 안에 **움츠린다** |
    | $D_{\mathrm{KL}}(p \| q)$ | $p$가 큰 곳에서 $q$가 작은 것 | $q$가 $p$를 **덮으려 한다** |

    변분 자기 부호기가 앞쪽을 쓰는 까닭은 **셈할 수 있기 때문**이다. 앞의 유도에서
    보았듯 기댓값이 $q$에 대한 것이라 우리가 뽑을 수 있는 분포에서 표본을 얻는다.
    뒤쪽은 기댓값이 $p(z \mid x)$에 대한 것인데, 그 분포는 우리가 모르는 것이라
    애초에 뽑을 수가 없다.

    결과적으로 생기는 성질도 알아 둘 만하다. $q$가 움츠리는 쪽이므로 참된 사후 분포가
    여러 봉우리를 가질 때 변분 자기 부호기는 그중 **하나만** 잡는 경향이 있다. 이것이
    생성 표본이 흐릿해지는 한 원인으로 지목되기도 한다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
KL 항이 없으면 무슨 일이 벌어지는가? 그 모델은 무엇이 되는가?

</div>

??? success "연습문제 5 풀이"
    $\beta = 0$으로 두는 셈이므로 손실이 다시 세우기만 남는다. 그러면 KL이 $\sigma$를
    1 쪽으로 당기던 힘이 사라져 부호기가 $\sigma \to 0$으로 간다. 잡음이 없는 편이
    되돌리기에 유리하기 때문이다.

    $\sigma = 0$이면 $z = \mu(x)$로 결정적이 되므로, 이 모델은 정확히
    [자기 부호기](../../ch25/index.md)다.

    그리고 자기 부호기가 못 하는 일을 똑같이 못 하게 된다.
    [23.5절](../../ch25/limits/latent_sampling.md)에서 잰 대로 $\mathcal{N}(0,I)$에서
    뽑은 표본 가운데 숫자로 보이는 것이 0.2%뿐이다.

    곧 KL 항은 덧붙인 정칙화가 아니라 **이 모델을 만들어 내는 모델로 만드는 바로 그
    항**이다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
KL 항을 정보 이론으로 읽으면 무엇인가?

</div>

??? success "연습문제 6 풀이"
    KL 항은 **코드가 입력에 대해 담을 수 있는 정보량의 상한**이다.

    자료 전체에 걸쳐 평균 내면

    $$\mathbb{E}_{p(x)}\big[ D_{\mathrm{KL}}(q(z \mid x) \,\|\, p(z)) \big]
      \;\ge\; I(x; z)$$

    로 코드와 자료 사이의 **서로 앎**(mutual information)을 위에서 묶는다. 곧 KL을
    작게 하라는 것은 "코드에 담는 정보를 아껴 쓰라"는 말이다.

    이 관점에서 보면 [25장의 여러 벌점](../../ch25/ae/loss_functions.md)과 한 줄에
    선다. 좁은 병목, 성김, 잡음, 오그림이 모두 정보량에 예산을 매기는 방법이었고,
    KL도 그중 하나다.

    다른 점은 그 예산이 **뽑을 수 있는 분포에 대해** 매겨진다는 것이다. $q$를 $p$에
    가깝게 하라는 요구이므로, 예산을 지키면 자동으로 $p$에서 뽑아도 말이 되는 코드
    공간이 남는다. 다른 벌점들은 정보량을 줄이되 어떤 모양으로 줄지는 말해 주지 않는다.

    이 읽기가 $\beta$의 뜻도 설명한다. $\beta$를 키우는 것은 정보 예산을 줄이는 것이고,
    실제로 재어 보면 살아 있는 차원 수가 $\beta$에 따라 16개에서 5개까지 줄어든다
    ([베타 VAE](../architecture/beta_vae.md)).

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
KL 항을 표집으로 어림하지 않고 닫힌 꼴로 셈하는 것이 왜 중요한가?

</div>

??? success "연습문제 7 풀이"
    기울기의 **잡음**이 줄기 때문이다.

    두 항 모두 표집으로 어림하면 기울기에 두 배의 잡음이 실린다. KL이 닫힌 꼴이면
    잡음이 다시 세우기 항에서만 오므로 학습이 훨씬 안정된다.

    닫힌 꼴이 가능한 것은 $q$와 $p$를 모두 가우시안으로 고른 덕이다. 이것이 실용적
    선택인 까닭 가운데 하나다. 사전 분포를 복잡한 것으로 바꾸면 표현력은 늘지만
    KL을 표집해야 해서 잡음이 늘고, 대개 손해가 더 크다.

    사전 분포를 바꾸고 싶을 때 흐름 모델을 끼워 넣는 방법이 있는데, 그때도 야코비
    행렬식을 닫힌 꼴로 셈할 수 있는 흐름을 고르는 것이 핵심이다
    ([27장](../../ch27/index.md)).

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff easy" title="쉬움"></span>
코드에서 왜 $\sigma$가 아니라 $\log \sigma^2$을 내놓게 하는가?

</div>

??? success "연습문제 8 풀이"
    세 가지 이유가 있다.

    **부호 제약이 없다.** $\sigma$는 양수여야 하므로 그물이 음수를 내놓으면 곤란하다.
    $\log \sigma^2$은 실수 전체를 돌아다녀도 되므로 선형층 출력을 그대로 쓸 수 있다.

    **수치적으로 안정하다.** $\sigma$가 0에 가까워지면 $\log \sigma$가 $-\infty$로
    가는데, 로그 공간에서 다루면 그 자리를 그냥 큰 음수로 표현한다.

    **KL 식이 간단해진다.** 앞의 닫힌 꼴에 $\log \sigma^2$이 그대로 들어가므로
    변환 없이 쓸 수 있다.

    다시 뽑을 때만 $\sigma = \exp(\tfrac{1}{2}\log\sigma^2)$으로 되돌린다. 코드의
    `(0.5 * logvar).exp()`가 그것이다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
KL 값이 0이 되는 것은 좋은 일인가?

</div>

??? success "연습문제 9 풀이"
    좋지 않다. **사후 분포 무너짐**(posterior collapse)이라 부르는 실패다.

    $D_{\mathrm{KL}}(q(z \mid x) \| p(z)) = 0$이라는 것은 $q(z \mid x) = p(z)$라는 뜻이고,
    이는 코드가 **입력에 전혀 기대지 않는다**는 말이다. 어떤 $x$를 넣어도 같은
    $\mathcal{N}(0,I)$이 나오므로 코드가 정보를 하나도 나르지 않는다.

    그러면 풀개는 코드를 무시하고 자료의 평균 같은 것만 내놓게 된다. 손실에서 KL은
    0으로 완벽하지만 다시 세우기가 엉망이 된다.

    실제로 차원마다 KL을 재면 이 일이 **일부 차원에서만** 벌어지는 것을 볼 수 있다.
    $\beta = 1$에서 16개 가운데 4개가 KL 0.01 미만으로 죽어 있고, $\beta = 4$에서는
    9개가 죽는다([베타 VAE](../architecture/beta_vae.md)).

    그러므로 KL은 작을수록 좋은 것도 클수록 좋은 것도 아니다. **적당히 0이 아니어야**
    한다. 차원별로 들여다보는 것이 총합만 보는 것보다 훨씬 많은 것을 알려 주는 까닭이다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
다시 세우기 항과 KL 항의 크기 비가 손실의 `reduction` 방식에 따라 달라진다. 이것이 왜 문제이며 어떻게 다루어야 하는가?

</div>

??? success "연습문제 10 풀이"
    다시 세우기 항은 화소 784개에 걸친 합이고 KL 항은 숨은 차원 16개에 걸친 합이다.
    화소 축을 평균 내면 다시 세우기가 784배 작아지는데 KL은 그대로이므로, 두 항의
    비가 784배 달라진다.

    곧 `reduction`을 바꾸는 것만으로 **암묵적인 $\beta$를 784배 바꾸는** 셈이다.

    | 방식 | 실효 $\beta$ |
    |---|---|
    | 화소 합 + 표본 평균 | 1 (관례) |
    | 화소 평균 + 표본 평균 | 784 |

    그래서 논문이나 코드를 읽을 때 $\beta$ 값만 보아서는 안 되고 두 항을 어떻게
    줄였는지 함께 보아야 한다. 같은 $\beta = 1$이 전혀 다른 균형을 뜻할 수 있다.

    실제로 재어 보면 이 차이가 결정적이다. $\beta$를 0.25에서 8로 바꾸는 것만으로
    다시 세우기가 70.60에서 134.92로, 살아 있는 차원이 16개에서 5개로 달라진다.
    784배 어긋나면 사실상 다른 모델이 된다.

    다루는 법은 간단하다. **화소는 더하고 표본만 평균 낸다**는 관례를 지키고, 그
    관례를 코드에 주석으로 밝혀 둔다.

## 정리하며

| 개념 | 식 | 변분 자기 부호기에서의 노릇 |
|---------|---------|-------------|
| **표준 정규 분포로의 KL** | $-\frac{1}{2}\sum(1 + \log\sigma^2 - \mu^2 - \sigma^2)$ | 벌주기 항 |
| **앞 KL** | $D_{KL}(p \| q)$ | 평균을 좇음(변분 자기 부호기에서는 쓰지 않는다) |
| **뒤 KL** | $D_{KL}(q \| p)$ | 최빈값 찾기(VAE 익히기) |
| **KL = 0** | $q(z\|x) = p(z)$ | 사후 분포 무너짐 |
| **쪼개기** | $\text{서로 앎} + \text{가장자리 KL}$ | 앎 이론의 관점 |

---
