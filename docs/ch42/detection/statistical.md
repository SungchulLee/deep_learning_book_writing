# 통계적 적대적 샘플 탐지

모델을 적대적 섭동에 강건하게 만드는 대신, 적대적 입력이 분류기에 닿기 전에 **탐지하는** 방법도 있다. 통계적 탐지 방법은 입력과 모델 내부의 피처를 분석해 클린 샘플과 적대적 샘플을 가른다.

---

## 1. 탐지 방법

### 피처 분포 분석

적대적 샘플은 네트워크의 중간 레이어에서 **비정상적인 활성화 패턴**을 내는 일이 잦다. 탐지 방법은 테스트 입력의 활성화 통계를 클린 데이터의 분포와 비교한다.

**마할라노비스 거리 탐지기**(이 등, 2018):

클래스 $c$와 레이어 $\ell$마다 클린 활성화에 가우스 분포를 적합한다(공분산은 클래스끼리 공유한다).

$$
(\boldsymbol{\mu}_c^\ell, \boldsymbol{\Sigma}^\ell) = \text{fit}(\{h^\ell(\mathbf{x}) : y = c\})
$$

탐지 점수는 레이어마다 가장 가까운 클래스까지의 마할라노비스 거리(에 음수를 붙인 값)를 모두 더한 것이다. 마할라노비스 거리는 공분산으로 축마다 스케일을 맞춘 거리다.

$$
M(\mathbf{x}) = \sum_\ell \max_c \left[ -(h^\ell(\mathbf{x}) - \boldsymbol{\mu}_c^\ell)^\top (\boldsymbol{\Sigma}^\ell)^{-1} (h^\ell(\mathbf{x}) - \boldsymbol{\mu}_c^\ell) \right]
$$

적대적 샘플은 마할라노비스 거리가 큰(점수가 낮은) 경향이 있다.

### 예측의 일관성

참 클래스를 바꾸지 않아야 할 입력 변환 아래에서 모델의 예측이 **일관되는지** 살핀다. 아래 코드는 변환으로 작은 가우스 잡음을 쓴다.

```python
import torch
import torch.nn as nn
from typing import Dict

class ConsistencyDetector:
    """
    아무 바꿈 아래 미루어 봄이 한결같은지로
    맞서는 보기를 짚어낸다.
    
    맑은 들임은 작은 바꿈 아래에서도 미루어 봄이 한결같고,
    맞서는 보기는 그렇지 않다.
    """
    
    def __init__(
        self, model: nn.Module, num_transforms: int = 20,
        noise_std: float = 0.05, threshold: float = 0.7
    ):
        self.model = model
        self.num_transforms = num_transforms
        self.noise_std = noise_std
        self.threshold = threshold
        self.model.eval()
    
    def detect(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        맞서는 보기를 짚어낸다.
        
        한결같음 점수와 참거짓 판단을 돌려준다.
        """
        device = next(self.model.parameters()).device
        x = x.to(device)
        
        with torch.no_grad():
            # 본디 미루어 봄
            base_pred = self.model(x).argmax(dim=1)
            
            # 아무 잡음 아래의 미루어 봄
            consistent = torch.zeros(len(x), device=device)
            for _ in range(self.num_transforms):
                noise = torch.randn_like(x) * self.noise_std
                noisy_pred = self.model(x + noise).argmax(dim=1)
                consistent += (noisy_pred == base_pred).float()
            
            consistency_score = consistent / self.num_transforms
        
        return {
            'consistency_score': consistency_score,
            'is_adversarial': consistency_score < self.threshold,
            'base_prediction': base_pred
        }
```

### 로짓 분석

적대적 샘플은 클린 입력과 다른 로짓 분포를 내는 일이 잦다.

- **더 큰 엔트로피**: 확신도가 낮은 예측(일부 공격에서)
- **비정상적인 로짓 마진**: 상위 클래스끼리의 로짓 차가 이상하다
- **다른 소프트맥스 분포**: 통계적 검정으로 탐지할 수 있다

---

## 2. 한계

통계적 탐지기에는 근본적인 어려움이 있다. 탐지기 자신도 공격받을 수 있다는 것이다. 탐지 방식을 아는 **적응형 적대자**는 탐지기까지 속이는 적대적 샘플을 만들 수 있다. 칼리니와 와그너(2017)는 탐지 방법 열 가지를 이렇게 모두 우회했다. 그래서 화이트박스 상황에서는 공격자에게 유리한 군비 경쟁이 벌어진다.

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

통계적 탐지는 방어를 보완하는 한 겹이며, 적응형이 아닌 공격에 특히 잘 듣는다. 다만 이것 하나에만 기대서는 안 된다. 교묘한 적대자 앞에서는 더욱 그렇다.

**살펴볼 거리**

1. Lee, K., et al. (2018). "A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks." NeurIPS.
2. Ma, X., et al. (2018). "Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality." ICLR.
3. Carlini, N., & Wagner, D. (2017). "Adversarial Examples Are Not Easily Detected: Bypassing Ten Detection Methods." ACM Workshop on AI Security.
