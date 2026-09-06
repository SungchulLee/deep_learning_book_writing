# 다층 모형
다층(섞인 효과) 모형은 무리 수준의 변동을 담는 짜임새 있는 확률 효과를 들여와 층층 베이즈 모형을 넓힌다. 학교 안의 학생, 포트폴리오 안의 거래, 피험자마다 되풀이한 측정처럼 관찰이 무리 수준의 짜임을 나누어 갖는 겹친 데이터를 뜯어보는 데 꼭 필요하다.

---

## 모형 명세

### 절편이 달라지는 모형

가장 단순한 다층 모형은 기울기는 함께 쓰면서 절편만 무리마다 다르게 한다.

$$
y_{ij} = \alpha_{j} + \beta x_{ij} + \epsilon_{ij}, \quad \epsilon_{ij} \sim \mathcal{N}(0, \sigma^2)
$$

여기서 $i$은 무리 $j$ 안의 관찰을 가리키고 다음이 성립한다.

$$
\alpha_j \sim \mathcal{N}(\mu_\alpha, \sigma_\alpha^2)
$$

무리의 절편 $\alpha_j$은 공통 모집단 분포에서 나오며, 이로써 **반쯤 모으기**가 가능해져 무리마다의 어림값이 다른 무리에서 힘을 빌린다.

### 절편과 기울기가 함께 달라지기

더 풍부한 모형은 절편과 기울기를 모두 무리마다 다르게 한다.

$$
y_{ij} = \alpha_j + \beta_j x_{ij} + \epsilon_{ij}
$$

$$
\begin{pmatrix} \alpha_j \\ \beta_j \end{pmatrix} \sim \mathcal{N}\left( \begin{pmatrix} \mu_\alpha \\ \mu_\beta \end{pmatrix}, \boldsymbol{\Sigma} \right)
$$

공분산 행렬 $\boldsymbol{\Sigma}$은 무리 수준 절편과 기울기 사이의 상관을 담는다. 이를테면 바탕 수익률이 높은 무리가 어떤 요인에도 더 민감할 수 있다.

---

## 세 가지 모으기 전략

### 온전히 모으기(모형 하나)

무리의 짜임을 아예 아랑곳하지 않는다: $y_{ij} = \alpha + \beta x_{ij} + \epsilon_{ij}$

**문제**: 무리 수준의 변동을 낮잡고, 모집단과 다른 무리의 어림값을 치우치게 한다.

### 모으지 않기(따로따로 모형)

무리마다 따로 모형을 맞춘다: $y_{ij} = \alpha_j + \beta_j x_{ij} + \epsilon_{ij}$

**문제**: 관찰이 적은 무리에서 어림값이 잡음투성이가 되고 정보를 나누어 쓰지 못한다.

### 반쯤 모으기(다층)

무리의 어림값이 모집단의 평균 쪽으로 오그라들며, 얼마나 오그라들지는 무리 수준 정보와 모집단 수준 정보의 정밀도 차이가 정한다.

$$
\hat{\alpha}_j^{\text{partial}} = \lambda_j \hat{\alpha}_j^{\text{no pool}} + (1 - \lambda_j) \hat{\alpha}^{\text{complete pool}}
$$

여기서 $\lambda_j$은 무리 $j$의 관찰 수와 무리 사이의 흩어짐에 달렸다.

**핵심 성질**: 관찰이 적은 무리일수록 모집단의 평균 쪽으로 더 오그라든다.

---

## PyTorch 구현

```python
import torch
import torch.distributions as dist

class MultilevelModel:
    """
    절편이 달라지는 베이즈 여러 층 모형.
    
    다음에 대한 깁스 표집을 구현한다:
        y_{ij} = alpha_j + beta * x_{ij} + epsilon_{ij}
        alpha_j ~ N(mu_alpha, sigma_alpha^2)
    """
    
    def __init__(self, n_groups: int):
        self.n_groups = n_groups
    
    def fit_gibbs(self, x, y, group_ids, n_samples=2000, warmup=500):
        """
        절편이 달라지는 모형의 깁스 표집기.
        
        매개변수
        ----------
        x : 꼴 (N,)의 텐서
        y : 꼴 (N,)의 텐서
        group_ids : 꼴 (N,)의 텐서, 정수 무리 이름표
        """
        N = len(y)
        
        # 매개변수를 초기화한다
        alpha = torch.zeros(self.n_groups)
        beta = torch.tensor(0.0)
        mu_alpha = torch.tensor(0.0)
        sigma2 = torch.tensor(1.0)
        sigma2_alpha = torch.tensor(1.0)
        
        samples = {
            'alpha': [], 'beta': [], 'mu_alpha': [],
            'sigma2': [], 'sigma2_alpha': []
        }
        
        for t in range(n_samples + warmup):
            # --- alpha_j(무리 절편) 표집 ---
            for j in range(self.n_groups):
                mask = (group_ids == j)
                n_j = mask.sum().float()
                if n_j == 0:
                    alpha[j] = dist.Normal(mu_alpha, sigma2_alpha.sqrt()).sample()
                    continue
                
                resid_j = y[mask] - beta * x[mask]
                
                # 뒤확률 정밀도와 평균
                precision_j = n_j / sigma2 + 1.0 / sigma2_alpha
                mean_j = (resid_j.sum() / sigma2 + mu_alpha / sigma2_alpha) / precision_j
                alpha[j] = dist.Normal(mean_j, (1.0 / precision_j).sqrt()).sample()
            
            # --- beta(공통 기울기) 표집 ---
            resid = y - alpha[group_ids]
            precision_beta = (x ** 2).sum() / sigma2 + 0.01  # 약한 앞확률
            mean_beta = (x * resid).sum() / sigma2 / precision_beta
            beta = dist.Normal(mean_beta, (1.0 / precision_beta).sqrt()).sample()
            
            # --- mu_alpha(모집단 평균) 표집 ---
            precision_mu = self.n_groups / sigma2_alpha + 0.001
            mean_mu = alpha.sum() / sigma2_alpha / precision_mu
            mu_alpha = dist.Normal(mean_mu, (1.0 / precision_mu).sqrt()).sample()
            
            # --- sigma2(관측 잡음) 표집 ---
            resid_all = y - alpha[group_ids] - beta * x
            ss = (resid_all ** 2).sum()
            sigma2 = dist.InverseGamma(
                torch.tensor(N / 2.0 + 1.0),
                ss / 2.0 + 0.1
            ).sample()
            
            # --- sigma2_alpha(무리 사이 흩어짐) 표집 ---
            ss_alpha = ((alpha - mu_alpha) ** 2).sum()
            sigma2_alpha = dist.InverseGamma(
                torch.tensor(self.n_groups / 2.0 + 1.0),
                ss_alpha / 2.0 + 0.1
            ).sample()
            
            if t >= warmup:
                samples['alpha'].append(alpha.clone())
                samples['beta'].append(beta.item())
                samples['mu_alpha'].append(mu_alpha.item())
                samples['sigma2'].append(sigma2.item())
                samples['sigma2_alpha'].append(sigma2_alpha.item())
        
        return {k: torch.tensor(v) if not isinstance(v[0], torch.Tensor) 
                else torch.stack(v) for k, v in samples.items()}
```

---

## 계량 금융에서의 쓰임

### 횡단면 자산 가격 결정

다층 모형은 자산 가격 결정의 패널 데이터를 자연스럽게 다룬다.

$$
r_{it} = \alpha_i + \beta_i f_t + \epsilon_{it}
$$

여기서 자산 수준 매개변수 $(\alpha_i, \beta_i)$은 모집단 분포에서 나온다. 이는 다음을 준다.

- 잡음 많은 알파 어림값을 0 쪽으로 오그라뜨리기(여러 번 검정하는 문제를 다룬다)
- 지난 기록이 짧은 자산의 베타 어림 개선
- 자산마다 원칙 있는 불확실성 재기

### 포트폴리오 위험 쪼개기

효과가 달라지는 모형은 포트폴리오의 위험을 다음으로 쪼갠다.

- **자산 안의 흩어짐**($\sigma^2$): 개별 위험
- **자산 사이의 흩어짐**($\sigma_\alpha^2$): 기대 수익률의 체계적인 퍼짐
- **모집단 매개변수**($\mu_\alpha, \mu_\beta$): 시장 전체의 위험 요인

---

## 요약

| 개념 | 핵심 |
|---------|-----------|
| **반쯤 모으기** | 무리의 짜임을 아랑곳하지 않는 것과 지나치게 맞추는 것 사이의 가장 좋은 절충 |
| **오그라들기** | 데이터가 적은 무리일수록 모집단의 평균 쪽으로 더 끌린다 |
| **달라지는 효과** | 절편, 기울기, 또는 둘 다 무리마다 달라질 수 있다 |
| **공분산 짜임** | 다변량 확률 효과가 무리 매개변수 사이의 상관을 담는다 |

---

## 참고 문헌

- Gelman, A., & Hill, J. (2006). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press.
- McElreath, R. (2020). *Statistical Rethinking* (2nd ed.). CRC Press. 13~14장.
- Raudenbush, S. W., & Bryk, A. S. (2002). *Hierarchical Linear Models* (2nd ed.). Sage.

## 연습문제

**연습문제 1.**
이 쪽이 다루는 핵심 개념과 그것이 베이즈 통계에서 하는 몫을 설명하라.

??? success "연습문제 1 풀이"
    이 쪽은 베이즈 추론의 근본 부품인 다층 모형을(를) 다룬다. 이는 데이터로 믿음을 고치고, 불확실성을 수로 나타내며, 불확실함 속에서 결정을 내리는 더 넓은 틀과 이어진다. 베이즈의 눈은 앞선 앎을 아우르고 불확실성을 분석 전체로 퍼뜨리는 원칙 있는 길을 준다.

---

**연습문제 2.**
주된 수학적 결과를 끌어내거나 밝히고 그 뜻을 설명하라.

??? success "연습문제 2 풀이"
    핵심 결과는 앞선 정보가 베이즈 정리를 거쳐 관찰한 데이터와 어우러져 고쳐진 추론을 낳는 모습을 보여 준다. 이 결과가 뜻깊은 까닭은, 매개변수의 불확실성을 아랑곳하지 않는 점 어림 방법과 달리 불확실성을 셈에 넣으면서 데이터에서 배우는 앞뒤 맞는 틀을 주기 때문이다.

---

**연습문제 3.**
이 주제에서 베이즈 방법과 빈도주의 대안을 견주어라.

??? success "연습문제 3 풀이"
    베이즈 방법은 온전한 뒤확률 분포, 자연스러운 불확실성 재기, 앞선 앎을 아우르는 원칙 있는 길을 준다. 빈도주의 대안은 표집 분포에 기대고, 큰 표본 어림이 필요할 수 있으며, 매개변수를 붙박인 미지수로 다룬다. 표본이 작을 때는 앞확률의 벌주기 효과 덕분에 베이즈 방법이 더 나을 때가 많다.

---

**연습문제 4.**
이 개념의 간단한 보기를 파이토치나 넘파이로 파이썬에 구현하라.

??? success "연습문제 4 풀이"
    ```python
    import numpy as np
    # 구현은 주제에 따라 달라진다.
    # 켤레 모형: 닫힌 꼴 뒤확률 새로 고치기.
    # 켤레가 아닌 모형: MCMC 또는 변분 추론.
    # 핵심 걸음: 앞확률 정하기, 가능도 셈하기, 뒤확률 이끌어 내기/어림하기.
    ```
