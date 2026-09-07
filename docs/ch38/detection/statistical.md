# 자로 맞서는 보기 알아내기
## 들머리

모형을 맞서는 흔듦에 든든하게 만드는 대신, 맞서는 들임이 가름개에 닿기 앞서 **짚어내는** 길도 있다. 자로 알아내는 방법은 들임과 모형 속의 결을 살펴 맑은 보기와 맞서는 보기를 가른다.

## 알아내는 길

### 결 분포 살피기

맞서는 보기는 그물의 가운데 켜에서 **여느 것과 다른 살림 결**을 내는 일이 잦다. 알아내는 방법은 시험 들임의 살림 자를 맑은 자료의 분포와 견준다.

**마할라노비스 거리 알아내개**(이 등, 2018):

갈래 $c$과 켜 $\ell$마다 맑은 살림에 가우스를 맞춘다.

$$
(\boldsymbol{\mu}_c^\ell, \boldsymbol{\Sigma}^\ell) = \text{fit}(\{h^\ell(\mathbf{x}) : y = c\})
$$

알아내기 점수는 켜에 걸친 마할라노비스 거리를 아우른다.

$$
M(\mathbf{x}) = \sum_\ell \max_c \left[ -(h^\ell(\mathbf{x}) - \boldsymbol{\mu}_c^\ell)^\top (\boldsymbol{\Sigma}^\ell)^{-1} (h^\ell(\mathbf{x}) - \boldsymbol{\mu}_c^\ell) \right]
$$

맞서는 보기는 마할라노비스 거리가 크고(점수가 낮고) 하는 쪽이다.

### 미루어 봄의 한결같음

참 갈래를 바꾸지 않아야 할 들임 바꿈 아래에서 모형의 미루어 봄이 **한결같은지** 살핀다.

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

### 로짓 살피기

맞서는 보기는 맑은 들임과 다른 로짓 분포를 내는 일이 잦다.

- **더 큰 엔트로피**: 덜 자신하는 미루어 봄(어떤 치기에서)
- **여느 것과 다른 로짓 틈**: 으뜸 갈래끼리의 틈이 이상하다
- **다른 소프트맥스 분포**: 통계 시험으로 짚어낼 수 있다

## 한계

자로 알아내는 개에는 밑바탕 어려움이 있다. 그 자신이 치일 수 있다는 것이다. 알아내는 얼개를 아는 **맞추어 오는 겨루는 이**는 알아내개까지 속이는 맞서는 보기를 지을 수 있다. 이는 흰 상자 자리에서 치는 쪽에 이로운 무기 겨루기를 낳는다.

## 간추림

자로 알아내기는 막이를 채워 주는 한 켜이며, 맞추어 오지 않는 치기에 특히 잘 듣는다. 다만 이것 하나에만 기대서는 안 된다. 꾀 많은 겨루는 이 앞에서는 더욱 그렇다.

## 살펴볼 거리

1. Lee, K., et al. (2018). "A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks." NeurIPS.
2. Ma, X., et al. (2018). "Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality." ICLR.
3. Carlini, N., & Wagner, D. (2017). "Adversarial Examples Are Not Easily Detected: Bypassing Ten Detection Methods." ACM Workshop on AI Security.

## 익힘 문제

**익힘 1.**
선형 가름개 $f(x) = w^T x + b$에서 미루어 본 갈래를 바꾸는 데 드는 가장 작은 $\ell_\infty$ 흔듦을 셈하여라. 이것이 신경 그물의 든든함과 어떻게 이어지는지 밝혀라.

??? success "익힘 1 풀이"
    선형 가름개에서 $\ell_\infty$ 노름으로 잰 판단의 금까지의 거리는 $\frac{|w^T x + b|}{\|w\|_1}$이다. 가장 작은 흔듦은 $\delta^* = \frac{|w^T x + b|}{\|w\|_1} \cdot \text{sign}(w)$이다. 신경 그물에서는 그 자리의 선형 어림 $f(x + \delta) \approx f(x) + \nabla_x f \cdot \delta$이 FGSM(기울기의 부호를 쓴다)이 왜 잘 듣는지를 밝혀 준다. 차수가 높은 모형이 무른 까닭은 $\|w\|_1$은 차수와 함께 커지는데 $|w^T x + b|$은 꼭 그렇지 않아 든든함의 여유가 줄어들기 때문이다. $\square$

---

**익힘 2.**
이 마디에서 다룬 치기나 막이를 CIFAR-10의 ResNet-18 모형에 짜 넣어라. $\epsilon = 8/255$의 PGD-20 치기 아래에서 맑은 맞음과 든든한 맞음을 알려라.

??? success "익힘 2 풀이"
    여느 ResNet-18은 맑은 맞음이 $\sim$93%이지만 PGD-20($\epsilon = 8/255$, 걸음 크기 $2/255$) 아래의 든든한 맞음은 $\sim$0%이다. 이 마디의 방법을 걸면 결과는 재주에 따라 다르다. 맞서며 익히기는 맑은 맞음 $\sim$83%에 든든한 맞음 $\sim$50%이고, 밝혀 낸 막이는 더 낮지만 증명할 수 있는 테두리를 준다. 맞음과 든든함의 맞바꿈은 밑바탕부터 있는 것이라, 든든함을 높이면 맑은 맞음이 흔히 5~15% 든다. 아무렇게나 하는 씨앗 3개의 평균과 잣대 어긋남으로 알려라. $\square$

---

**익힘 3.**
흔듦 공 안에서 갈래별 자료의 밑자리가 서로 겹친다고 볼 때, 모형이 담는 힘을 키우지 않고서는 어떤 막이도 맑은 자료의 높은 맞음과 $\ell_\infty$ 흔듦에 대한 높은 든든함을 함께 이룰 수 없음을 증명하여라.

??? success "익힘 3 풀이"
    두 갈래의 밑자리가 거리 $\epsilon$ 안에서 겹치면(곧 $\|x_1 - x_2\|_\infty \leq 2\epsilon$인 $x_1 \in \text{갈래 1}, x_2 \in \text{갈래 2}$이 있으면), $x_1$과 $x_2$ 둘 다에서 든든한 가름개는 적어도 하나를 틀리게 가를 수밖에 없다(흔듦 공이 겹치기 때문이다). 이것이 맞음과 든든함의 밑바탕 맞바꿈이다. 겹치는 밑자리의 몫이 피할 수 없는 맞음 잃음을 정한다. 여느 그림 분포에서는 $\epsilon = 8/255$에서 겹침이 꽤 있어, 살펴본 10~15%의 맞음 떨어짐을 밝혀 준다. 모형이 담는 힘을 키우면(더 너른 그물) 얽힌 든든한 판단의 금을 더 잘 그려 맞바꿈을 얼마쯤 눅일 수 있다. $\square$

---

**익힘 4.**
금융 기계 배움 얼개(속임수 알아내기나 거래 신호 만들기 따위)에서 맞섬의 든든함이 어떻게 드러나는지 다루어라. 으름 얼개가 보기 다룸과 어떻게 다른가?

??? success "익힘 4 풀이"
    금융에서 겨루는 이는 알아내는 얼개에 맞추어 스스로 움직이는 꾀 많은 무리(속임수꾼, 저자 흔드는 이)다. 보기 다룸과 다른 고갱이는 이렇다. (1) 흔들 수 있는 밭이 돈으로 될 만한 것에 옭매인다(속임수꾼이 제 거래 자취를 통째로 바꿀 수는 없다). (2) 치기가 잇따르며 맞추어 간다(겨루는 이가 얼개의 되받음을 보고 손본다). (3) 헛 맞음과 놓침의 값이 서로 어긋난다(옳은 거래를 막는 것과 속임수를 놓치는 것). (4) $\ell_p$ 노름은 뜻이 없고 밭에 맞는 흔듦 모형이 있어야 한다. 막이는 맞추어 오는 겨루는 이에게도 든든해야 하므로, 알아내는 잣대가 알려지면 비껴갈 수 있는 알아내기 바탕의 길은 많이 걸러진다. $\square$
