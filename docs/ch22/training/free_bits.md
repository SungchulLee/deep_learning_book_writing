# Free Bits
숨은 차원마다 최소 앎을 보장해 사후 분포 무너짐 막기.

---

## 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

- 공짜 비트 재주와 그 까닭 설명하기
- 변분 자기 부호기 익히기에 공짜 비트 짜기
- 공짜 비트와 KL 달구고 식히기 견주기
- 알맞은 공짜 비트 문턱 고르기

---

## The Free Bits Technique

### 동기

KL 달구고 식히기는 (익히기 초반에) 때로 사후 분포 무너짐을 다룬다. **공짜 비트**(Kingma et al., 2016)는 숨은 차원마다 벌 없이 적어도 $\lambda$ 냇의 앎을 담을 수 있도록 KL 항을 고쳐 짜임에서 다룬다.

### 정식화

여느 KL 항을 다음으로 갈음한다:

$$D_{KL}^{\text{free}} = \sum_{j=1}^{d} \max\left(\lambda, \, D_{KL,j}(q_\phi(z_j|x) \| p(z_j))\right)$$

여기서 $D_{KL,j}$은 차원 $j$의 KL 몫이고 $\lambda$은 공짜 비트 문턱이다. $D_{KL,j} < \lambda$이면 KL을 $\lambda$으로 붙박아 두므로 그 차원을 사전 분포 쪽으로 더 미는 기울기가 없다.

### 구현

```python
import torch
import torch.nn.functional as F

def vae_loss_free_bits(recon_x, x, mu, logvar, free_bits=0.5):
    """
    공짜 비트 제약을 곁들인 변분 자기 부호기 손실.
    
    인수:
        recon_x: 다시 세운 내놓기
        x: 본디 들임
        mu: 부호기의 평균 [묶음 크기, 숨은 차원]
        logvar: 부호기의 로그 흩어짐 [묶음 크기, 숨은 차원]
        free_bits: 차원마다의 최소 KL(냇 단위)
    
    반환값:
        total_loss, recon_loss, kl_loss
    """
    # 되살림 손실
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # 차원마다의 KL: [묶음 크기, 숨은 차원]
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    
    # 묶음에 걸쳐 먼저 평균 낸 뒤 공짜 비트를 적용한다
    kl_mean_per_dim = kl_per_dim.mean(dim=0)  # [숨은 차원]
    kl_free = torch.clamp(kl_mean_per_dim, min=free_bits)
    kl_loss = kl_free.sum() * x.size(0)  # 묶음 잣수로 되돌린다
    
    total_loss = recon_loss + kl_loss
    return total_loss, recon_loss, kl_loss
```

---

## 람다 고르기

| $\lambda$ Value | Effect |
|-----------------|--------|
| $\lambda = 0$ | 여느 변분 자기 부호기(공짜 비트 없음) |
| $\lambda = 0.1$ | 무너짐을 약하게 막음 |
| $\lambda = 0.5$ | 웬만함 — 좋은 붙박이 |
| $\lambda = 2.0$ | 셈 — 차원마다 상당한 앎을 담아야 한다 |

가장 좋은 $\lambda$은 자료의 복잡함과 숨은 차원에 달렸다. 숨은 차원 32개인 MNIST에서는 흔히 $\lambda \in [0.1, 1.0]$이 잘 통한다. 자료가 더 복잡하거나 숨은 차원이 더 높으면 더 작은 값으로도 넉넉할 수 있다.

---

## Free Bits vs KL Annealing

| Aspect | KL Annealing | Free Bits |
|--------|-------------|-----------|
| **장치** | 때에 따름(차례표 바탕) | 짜임에 따름(차원마다의 문턱) |
| **웃매개변수** | 몸풀기 기간 | 문턱 $\lambda$ |
| **사후 분포 무너짐** | 몸풀기 동안 막는다 | 늘 막는다 |
| **아울러 쓸 수 있는가** | 예 — 흔히 함께 쓴다 | 예 |

실전에서는 두 재주를 아우르면 가장 튼튼하게 익힌다.

---

## 요약

| 개념 | 핵심 |
|---------|-----------|
| **공짜 비트** | 차원마다의 최소 KL: $\max(\lambda, D_{KL,j})$ |
| **효과** | 차원 하나하나가 무너지는 것을 막는다 |
| **붙박이 $\lambda$** | 일에 따라 0.1~1.0 냇 |
| **아우르기** | 가장 좋은 결과를 얻으려 KL 달구고 식히기와 함께 쓴다 |

---

## 다음은

다음 절은 묶음 크기가 변분 자기 부호기 익히기의 움직임에 미치는 영향을 살핀다.

## 연습문제

### Exercise 1: Free Bits Sweep

$\lambda \in \{0, 0.1, 0.5, 1.0, 2.0\}$으로 변분 자기 부호기를 익혀라. 깨어 있는 차원과 다시 세우기 품질을 그려라.

### Exercise 2: Combined Strategy

(가) 달구고 식히기만, (나) 공짜 비트만, (다) 둘 다를 견주어라. 어느 것이 가장 좋은 증거 하한을 이루는가?

---
