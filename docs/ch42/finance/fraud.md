# 사기 탐지 모델의 강건성

사기 탐지 시스템은 본래 적대적인 환경이다. 사기꾼은 사기의 목적은 지킨 채 탐지를 회피하는 거래를 끊임없이 만들어 낸다. 그래서 적대적 강건성은 이론상의 걱정이 아니라 운영에 꼭 필요한 요건이 된다. 적대적 샘플이 주로 연구 주제로 다뤄지는 이미지 분류와 달리, 사기 탐지는 전략을 끊임없이 바꾸는 **실제 적대자**와 마주한다.

---

## 1. 사기 탐지의 위협 모델

### 적대자의 모습

사기꾼은 다음 조건 아래에서 움직인다.

- **앎**: 흔히 블랙박스나 그레이박스다. 승인/거절 결과는 보지만 모델 내부를 들여다보는 일은 드물다
- **목표**: 표적 회피. 사기 거래를 정상 거래처럼 보이게 한다
- **제약**: 사기의 금전적 목적을 지켜야 한다(돈이 실제로 옮겨져야 하고, 훔친 물건을 받아야 한다)
- **질의 예산**: 사기를 한 번 시도할 때의 비용과 적발 위험 때문에 제한된다

### 형식화

$f(\mathbf{x}) = 1$이 사기를 뜻하는 사기 탐지기 $f_\theta: \mathbb{R}^d \to \{0, 1\}$을 두자. 적대자는 다음을 찾는다.

$$
\mathbf{x}_{\text{evasion}} = \arg\min_{\mathbf{x}' \in \mathcal{C}} f_\theta(\mathbf{x}')
$$

이때 제약 집합 $\mathcal{C}$는 사기의 의미를 보존한다(그 거래가 여전히 적대자의 금전적 목적을 이뤄야 한다).

### 피처 공간의 섭동

$\ell_p$ 노름을 쓰는 이미지 공격과 달리, 사기 공격은 도메인에 맞는 제약이 걸린 **피처 공간**에서 움직인다.

| 피처 종류 | 바꿀 수 있나? | 제약 |
|-------------|-------------|------------|
| 거래 금액 | 일부 | 금전적 목적을 이뤄야 한다 |
| 가맹점 업종 | 그렇다 | 유효한 업종 가운데서 고른다 |
| 거래 시각 | 그렇다 | 영업 시간 안이어야 한다 |
| 기기 지문 | 그렇다 | 위조하거나 새 기기를 쓴다 |
| IP 주소 | 그렇다 | VPN/프록시를 쓴다 |
| 거래 빈도 | 일부 | 거래를 실제로 마쳐야 한다 |
| 카드 실물 제시 여부 | 고정 | 물리적으로 제약된다 |

---

## 2. 사기 탐지의 적대적 학습

### 사기 탐지에 맞춘 적대적 학습 프레임워크

표준 적대적 학습은 표 형식의 금융 데이터에 맞게 고쳐야 한다. 아래 코드는 피처마다 섭동 예산을 따로 두고, 범주형 피처는 고정하며, 사기를 놓치는 오류(거짓 음성)에 더 큰 가중치를 준다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

class FraudRobustTrainer:
    """
    속임수 탐지에 맞춘 맞서며 익히기.
    
    그림 맞서며 익히기와 다른 핵심:
    - 결마다 다른 섭동 예산
    - 옭아맴을 아는 섭동(클래스 결, 옳은 자리)
    - 어긋난 로스(놓침이 헛 정확도보다 값이 크다)
    """
    
    def __init__(
        self,
        model: nn.Module,
        feature_budgets: torch.Tensor,
        categorical_mask: torch.Tensor,
        num_iter: int = 10,
        alpha_scale: float = 2.0,
        fn_weight: float = 10.0,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.feature_budgets = feature_budgets  # 결마다의 엡실론
        self.categorical_mask = categorical_mask  # 클래스 결이면 1
        self.num_iter = num_iter
        self.alpha_scale = alpha_scale
        self.fn_weight = fn_weight  # 놓침에 주는 짐
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
    
    def _constrained_pgd(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        결마다 옭아맴을 지닌 PGD.
        
        이어지는 결: 결마다의 엡실론 안에서 흔든다
        클래스 결: 붙박아 둔다(또는 옳은 값으로만 흔든다)
        """
        eps = self.feature_budgets.to(self.device)
        cat_mask = self.categorical_mask.to(self.device)
        alpha = self.alpha_scale * eps / self.num_iter
        
        # 첫자리를 잡는다
        delta = torch.zeros_like(x)
        cont_mask = 1 - cat_mask
        
        # 이어지는 결만 아무 첫자리
        delta = delta + cont_mask * torch.empty_like(x).uniform_(-1, 1) * eps
        
        for _ in range(self.num_iter):
            x_adv = (x + delta).requires_grad_(True)
            logits = self.model(x_adv)
            
            # 짐 준 로스: 비껴감을 벌한다(속임을 옳은 것으로 가름)
            loss = F.cross_entropy(logits, y, reduction='none')
            # 속임 보기에 짐을 더 준다(적대자가 회피하려 한다)
            weights = torch.where(y == 1, self.fn_weight, 1.0)
            loss = (weights * loss).mean()
            
            self.model.zero_grad()
            loss.backward()
            grad = x_adv.grad.data
            
            with torch.no_grad():
                # 이어지는 결만 고친다
                delta = delta + cont_mask * alpha * grad.sign()
                delta = torch.clamp(delta, -eps, eps) * cont_mask
        
        return torch.clamp(x + delta, 0, 1).detach()
    
    def train_epoch(self, train_loader, optimizer):
        """속임을 아는 맞서며 익히기로 한 판 익힌다."""
        self.model.train()
        total_loss = 0
        total = 0
        
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            
            # 맞서는 보기를 만든다
            x_adv = self._constrained_pgd(x, y)
            
            # 맞서는 보기로 익힌다
            optimizer.zero_grad()
            logits = self.model(x_adv)
            
            # 어긋난 로스
            weights = torch.where(y == 1, self.fn_weight, 1.0)
            loss = (weights * F.cross_entropy(
                logits, y, reduction='none'
            )).mean()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(y)
            total += len(y)
        
        return {'loss': total_loss / total}
```

---

## 3. 강건한 사기 탐지를 평가하는 지표

사기 탐지에 표준 정확도만으로는 모자란다. 사기 거래는 아주 드물어서, 모두 정상이라고 답해도 정확도가 99%를 넘기 때문이다. 적대적 환경에서 중요한 지표는 이렇다.

| 지표 | 정의 | 목표 |
|--------|-----------|--------|
| 강건한 참 양성 비율 | 적대적 회피 아래에서의 참 양성 비율(재현율) | 최대화 |
| 임계값에서의 거짓 양성 비율 | 운영 임계값에서의 거짓 양성 비율 | 최소화 |
| 강건한 AUPRC | 공격 아래 정밀도-재현율 곡선 아래 넓이 | 최대화 |
| 회피율 | 탐지를 회피한 사기의 비율 | 최소화 |

---

## 4. 실무 권장 사항

1. **피처마다 예산을 달리한다**: 피처마다 바꿀 수 있는 정도가 다르다
2. **범주형 제약을 지킨다**: 적대자가 이산적인 피처를 마음대로 바꿀 수는 없다
3. **비대칭으로 학습한다**: 놓친 사기에 거짓 양성보다 훨씬 큰 가중치를 준다
4. **회피에 쓰이는 피처를 모니터링한다**: 적대자가 어느 피처를 가장 많이 바꾸는지 추적한다
5. **방어를 앙상블로**: 규칙 기반 탐지와 머신러닝 기반 탐지를 함께 써서 강건성을 높인다

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
선형 분류기 $f(x) = w^T x + b$에서 예측한 클래스를 바꾸는 데 드는 가장 작은 $\ell_\infty$ 섭동을 계산하여라. 이것이 뉴럴 네트워크의 강건성과 어떻게 이어지는지 설명하여라.

</div>

??? success "풀이"
    선형 분류기에서 $\ell_\infty$ 노름으로 잰 결정 경계까지의 거리는 $\frac{|w^T x + b|}{\|w\|_1}$이다. 가장 작은 섭동은 $\delta^* = \frac{|w^T x + b|}{\|w\|_1} \cdot \text{sign}(w)$이다. 뉴럴 네트워크에서는 국소 선형 근사 $f(x + \delta) \approx f(x) + \nabla_x f \cdot \delta$이 FGSM(그래디언트의 부호를 쓴다)이 왜 잘 통하는지를 설명해 준다. 차원이 높은 모델이 취약한 까닭은 $\|w\|_1$은 차원과 함께 커지는데 $|w^T x + b|$은 꼭 그렇지 않아 강건성의 여유가 줄어들기 때문이다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 절에서 다룬 공격이나 방어를 CIFAR-10의 ResNet-18 모델에 구현하여라. $\epsilon = 8/255$의 PGD-20 공격 아래에서 클린 정확도와 강건 정확도를 보고하여라.

</div>

??? success "풀이"
    일반 ResNet-18은 클린 정확도가 $\sim$93%이지만 PGD-20($\epsilon = 8/255$, 스텝 크기 $2/255$) 아래의 강건 정확도는 $\sim$0%이다. 이 절의 방법을 적용한 결과는 기법에 따라 다르다. 적대적 학습은 클린 정확도 $\sim$83%에 강건 정확도 $\sim$50%이고, 인증 가능한 방어는 더 낮지만 증명할 수 있는 바운드를 준다. 정확도와 강건성의 트레이드오프는 근본적인 것이라, 강건성을 높이면 클린 정확도가 흔히 5~15% 떨어진다. 무작위 시드 3개의 평균과 표준편차로 보고하여라. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
섭동 공 안에서 클래스별 데이터의 서포트가 서로 겹친다고 볼 때, 모델이 용량을 키우지 않고서는 어떤 방어도 클린 데이터의 높은 정확도와 $\ell_\infty$ 섭동에 대한 높은 강건성을 함께 이룰 수 없음을 증명하여라.

</div>

??? success "풀이"
    두 클래스의 서포트가 거리 $\epsilon$ 안에서 겹치면(곧 $\|x_1 - x_2\|_\infty \leq 2\epsilon$인 $x_1 \in \text{클래스 1}, x_2 \in \text{클래스 2}$이 있으면), $x_1$과 $x_2$ 둘 다에서 강건한 분류기는 적어도 하나를 틀리게 가를 수밖에 없다(섭동 공이 겹치기 때문이다). 이것이 정확도와 강건성의 근본적인 트레이드오프다. 겹치는 서포트의 비중이 피할 수 없는 정확도 손실을 정한다. 자연 이미지 분포에서는 $\epsilon = 8/255$에서 겹침이 꽤 있어, 관찰되는 10~15%의 정확도 하락을 설명해 준다. 모델 용량을 키우면(더 넓은 네트워크) 복잡한 강건 결정 경계를 더 잘 그려 트레이드오프를 어느 정도 완화할 수 있다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
금융 머신러닝 시스템(사기 탐지나 거래 신호 생성 등)에서 적대적 강건성이 어떻게 드러나는지 논하여라. 위협 모델이 비전과 어떻게 다른가?

</div>

??? success "풀이"
    금융에서 적대자는 탐지 시스템에 맞추어 전략을 바꾸는 교묘한 집단(사기꾼, 시장 조작자)이다. 비전과 다른 핵심은 이렇다. (1) 섭동할 수 있는 피처가 경제적으로 그럴듯한 범위로 제약된다(사기꾼이 자기 거래 이력을 통째로 바꿀 수는 없다). (2) 공격이 순차적이고 적응적이다(적대자가 시스템의 응답을 보고 고친다). (3) 거짓 양성과 거짓 음성의 비용이 비대칭이다(정상 거래를 막는 것과 사기를 놓치는 것). (4) $\ell_p$ 노름은 의미가 없고 도메인에 맞는 섭동 모델이 있어야 한다. 방어는 적응형 적대자에게도 강건해야 하므로, 탐지 기준이 알려지면 회피할 수 있는 탐지 기반 방법은 쓸모가 크게 줄어든다. $\square$

## 정리하며

이 절에서는 사기 탐지의 위협 모델, 피처별 제약을 넣은 적대적 학습, 강건한 사기 탐지를 평가하는 지표, 실무 권장 사항을 살펴보았다. 사기 탐지에서 섭동은 $\ell_p$ 공이 아니라 사기꾼이 실제로 바꿀 수 있는 피처의 범위로 정해진다.

**살펴볼 거리**

1. Cartella, F., et al. (2021). "Adversarial Attacks on Fraud Detection Systems." Future Generation Computer Systems.
2. Chen, H., et al. (2020). "Robustness of Machine Learning Based Fraud Detection." ACM SIGKDD Workshop.
