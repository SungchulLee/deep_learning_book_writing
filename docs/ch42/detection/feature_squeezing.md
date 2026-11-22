# Feature Squeezing

**피처 스퀴징**(feature squeezing, 쉬 등, 2018)은 원래 입력과 복잡도를 줄인("쥐어짠") 입력에 대한 예측을 비교해 적대적 샘플을 탐지한다. 핵심 통찰은 적대적 섭동이 특정한, 흔히 미세한 픽셀 패턴에 기대므로 입력을 단순하게 만들면 효과를 잃는 반면 클린 이미지의 예측은 거의 그대로라는 것이다.

---

## 1. 핵심 통찰

입력 $\mathbf{x}$와 스퀴징 함수 집합 $\{s_1, s_2, \ldots\}$이 있을 때 예측을 비교한다.

$$
\text{Detection Score} = \max_j \|f(\mathbf{x}) - f(s_j(\mathbf{x}))\|_1
$$

여기서 $f$는 소프트맥스 확률 벡터다. 예측의 가장 큰 차가 임계값을 넘으면 그 입력을 적대적 샘플로 판정한다.

---

## 2. 스퀴징 연산

### 비트 깊이 줄이기

색 채널마다의 비트 수를 줄여 픽셀 값을 몇 단계로 양자화한다.

$$
s_{\text{bit}}(\mathbf{x}; b) = \text{round}(\mathbf{x} \cdot 2^b) / 2^b
$$

이를테면 8비트(256단계)에서 4비트(16단계)로 줄이면 이미지의 구조는 지키면서 미세한 섭동을 없앤다.

### 공간 평활화

고주파 섭동을 흐리는 공간 필터를 건다.

- **미디언 필터**: 비선형이며 에지를 보존하고 소금후추 잡음에 잘 듣는다
- **가우스 블러**: 선형 평활화로 고주파 잡음을 줄인다
- **비지역 평균(non-local means)**: 비슷한 패치끼리 가중 평균을 내어 잡음을 지운다

### JPEG 압축

JPEG는 DCT 영역에서 고주파 성분을 거칠게 양자화하므로 적대적 섭동의 상당 부분을 저절로 없앤다. 위의 비지역 평균과 JPEG 압축은 아래 구현에는 넣지 않았다.

---

## 3. PyTorch로 구현하기

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional

class FeatureSqueezing:
    """
    맞서는 보기를 탐지하는 결 쥐어짜기.
    
    본디 들임과 쥐어짠 들임의 미루어 봄을 견준다.
    미루어 봄의 차가 크면 맞서서 흔들었다는 뜻이다.
    
    Parameters
    ----------
    model : nn.Module
        타깃 분류기
    squeezers : list[callable]
        쥐어짜는 함수의 목록
    threshold : float
        미루어 봄의 L1 차에 대한 탐지 임계
    """
    
    def __init__(
        self,
        model: nn.Module,
        threshold: float = 0.1,
        bit_depth: int = 4,
        median_kernel: int = 3,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.threshold = threshold
        self.bit_depth = bit_depth
        self.median_kernel = median_kernel
        self.device = device or next(model.parameters()).device
        self.model.eval()
    
    def _squeeze_bit_depth(
        self, x: torch.Tensor, bits: int
    ) -> torch.Tensor:
        """들임의 비트 깊이를 줄인다."""
        levels = 2 ** bits
        return torch.round(x * levels) / levels
    
    def _squeeze_median(
        self, x: torch.Tensor, kernel_size: int = 3
    ) -> torch.Tensor:
        """가운뎃값 필터를 건다."""
        # 클래스마다 가운뎃값 필터
        pad = kernel_size // 2
        x_pad = F.pad(x, [pad] * 4, mode='reflect')
        
        # 펼쳐서 조각을 얻는다
        patches = x_pad.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
        return patches.contiguous().view(*patches.shape[:4], -1).median(dim=-1)[0]
    
    def _squeeze_gaussian(
        self, x: torch.Tensor, sigma: float = 1.0
    ) -> torch.Tensor:
        """가우스 흐리기를 건다."""
        k = int(4 * sigma + 1)
        if k % 2 == 0:
            k += 1
        
        coords = torch.arange(k, dtype=torch.float32, device=x.device) - k // 2
        kernel_1d = torch.exp(-coords**2 / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d.outer(kernel_1d)
        
        C = x.shape[1]
        kernel = kernel_2d.expand(C, 1, k, k)
        
        return F.conv2d(x, kernel, padding=k//2, groups=C)
    
    def detect(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        결 쥐어짜기로 맞서는 보기를 짚어낸다.
        
        탐지 점수와 참거짓 판단을 돌려준다.
        """
        x = x.to(self.device)
        
        with torch.no_grad():
            # 본디 미루어 봄
            pred_orig = F.softmax(self.model(x), dim=1)
            
            # 쥐어짠 들임의 미루어 봄
            max_diff = torch.zeros(len(x), device=self.device)
            
            # 쥐어짜개 1: 비트 깊이 줄이기
            x_squeezed = self._squeeze_bit_depth(x, self.bit_depth)
            pred_sq = F.softmax(self.model(x_squeezed), dim=1)
            diff = (pred_orig - pred_sq).abs().sum(dim=1)
            max_diff = torch.max(max_diff, diff)
            
            # 쥐어짜개 2: 가운뎃값 필터
            x_squeezed = self._squeeze_median(x, self.median_kernel)
            pred_sq = F.softmax(self.model(x_squeezed), dim=1)
            diff = (pred_orig - pred_sq).abs().sum(dim=1)
            max_diff = torch.max(max_diff, diff)
            
            # 쥐어짜개 3: 가우스 흐리기
            x_squeezed = self._squeeze_gaussian(x, sigma=1.0)
            pred_sq = F.softmax(self.model(x_squeezed), dim=1)
            diff = (pred_orig - pred_sq).abs().sum(dim=1)
            max_diff = torch.max(max_diff, diff)
        
        return {
            'detection_score': max_diff,
            'is_adversarial': max_diff > self.threshold
        }
```

---

## 4. 한계

- **적응형 공격**: 스퀴징 연산을 아는 적대자는 스퀴징을 거쳐도 살아남는 섭동을 만들 수 있다
- **정확도에 미치는 영향**: 클린 입력도 스퀴징하면 예측이 바뀌어 거짓 양성이 생길 수 있다
- **임계값 선택**: 탐지 임계값은 클린 데이터로 신중하게 보정해야 한다

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
    일반 ResNet-18은 클린 정확도가 $\sim$93%이지만 PGD-20($\epsilon = 8/255$, 스텝 크기 $2/255$) 아래의 강건 정확도는 $\sim$0%이다. 이 절의 방법을 적용한 결과는 기법에 따라 다르다. 적대적 트레인은 클린 정확도 $\sim$83%에 강건 정확도 $\sim$50%이고, 인증 가능한 방어는 더 낮지만 증명할 수 있는 바운드를 준다. 정확도와 강건성의 트레이드오프는 근본적인 것이라, 강건성을 높이면 클린 정확도가 흔히 5~15% 떨어진다. 무작위 시드 3개의 평균과 표준편차로 보고하여라. $\square$

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

피처 스퀴징은 모델을 가리지 않는 단순한 탐지 수단이다. 단독 방어보다는 여러 겹으로 쌓은 방어의 한 부분으로 쓸 때 가장 잘 듣고, 적응형이 아닌 적대자를 상대할 때 특히 그렇다.

**살펴볼 거리**

1. Xu, W., Evans, D., & Qi, Y. (2018). "Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks." NDSS.
