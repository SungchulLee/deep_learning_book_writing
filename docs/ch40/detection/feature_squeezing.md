# 결 쥐어짜기

**결 쥐어짜기**(쉬 등, 2018)은 본디 들임과 얽힘을 줄인("쥐어짠") 들임에 대한 미루어 봄을 견주어 맞서는 보기를 짚어낸다. 고갱이 깨침은 맞서는 흔듦이 꼭 집은, 흔히 은근한 낱그림점 결에 기대므로 들임을 눌러 담으면 무너지는 반면 맑은 그림은 거의 그대로라는 것이다.

---

## 1. 고갱이 깨침

들임 $\mathbf{x}$과 쥐어짜는 함수 모임 $\{s_1, s_2, \ldots\}$이 있을 때 미루어 봄을 견준다.

$$
\text{Detection Score} = \max_j \|f(\mathbf{x}) - f(s_j(\mathbf{x}))\|_1
$$

미루어 봄의 가장 큰 차가 문턱을 넘으면 그 들임에 맞선다는 표시를 붙인다.

---

## 2. 쥐어짜는 셈

### 비트 깊이 줄이기

빛깔 갈래마다의 비트 수를 줄여 낱그림점 값을 띄엄하게 만든다.

$$
s_{\text{bit}}(\mathbf{x}; b) = \text{round}(\mathbf{x} \cdot 2^b) / 2^b
$$

이를테면 8비트(256단계)에서 4비트(16단계)로 줄이면 그림의 얼개는 지키면서 은근한 흔듦을 없앤다.

### 자리 매끄럽게 하기

높은 잦기의 흔듦을 흐리는 자리 거르개를 건다.

- **가운뎃값 거르개**: 곧지 않고 모서리를 지키며 소금후추 흔듦에 잘 듣는다
- **가우스 흐리기**: 선형으로 매끄럽게 하여 높은 잦기 잡음을 줄인다
- **먼 데까지 보는 평균**: 조각끼리의 닮음에 기대어 맞추어 가며 잡음을 지운다

### JPEG 눌러 담기

JPEG은 DCT 밭에서 높은 잦기 몫을 버리므로 맞서는 흔듦을 절로 많이 없앤다.

---

## 3. PyTorch로 짜기

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional

class FeatureSqueezing:
    """
    맞서는 보기를 알아내는 결 쥐어짜기.
    
    본디 들임과 쥐어짠 들임의 미루어 봄을 견준다.
    미루어 봄의 차가 크면 맞서서 흔들었다는 뜻이다.
    
    Parameters
    ----------
    model : nn.Module
        과녁 가름개
    squeezers : list[callable]
        쥐어짜는 함수의 목록
    threshold : float
        미루어 봄의 L1 차에 대한 알아내기 문턱
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
        """가운뎃값 거르개를 건다."""
        # 갈래마다 가운뎃값 거르개
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
        
        알아내기 점수와 참거짓 판단을 돌려준다.
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
            
            # 쥐어짜개 2: 가운뎃값 거르개
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

- **맞추어 오는 치기**: 쥐어짜는 셈을 아는 겨루는 이는 쥐어짜도 살아남는 흔듦을 지을 수 있다
- **맞음에 미침**: 맑은 들임을 쥐어짜도 미루어 봄이 바뀌어 헛 맞음이 생길 수 있다
- **문턱 고르기**: 알아내기 문턱은 맑은 자료로 조심스레 눈금을 맞춰야 한다

---

## 연습문제

**연습문제 1.**
선형 가름개 $f(x) = w^T x + b$에서 미루어 본 갈래를 바꾸는 데 드는 가장 작은 $\ell_\infty$ 흔듦을 셈하여라. 이것이 신경 그물의 든든함과 어떻게 이어지는지 밝혀라.

??? success "연습문제 1 풀이"
    선형 가름개에서 $\ell_\infty$ 노름으로 잰 판단의 금까지의 거리는 $\frac{|w^T x + b|}{\|w\|_1}$이다. 가장 작은 흔듦은 $\delta^* = \frac{|w^T x + b|}{\|w\|_1} \cdot \text{sign}(w)$이다. 신경 그물에서는 그 자리의 선형 어림 $f(x + \delta) \approx f(x) + \nabla_x f \cdot \delta$이 FGSM(기울기의 부호를 쓴다)이 왜 잘 듣는지를 밝혀 준다. 차수가 높은 모형이 무른 까닭은 $\|w\|_1$은 차수와 함께 커지는데 $|w^T x + b|$은 꼭 그렇지 않아 든든함의 여유가 줄어들기 때문이다. $\square$

---

**연습문제 2.**
이 마디에서 다룬 치기나 막이를 CIFAR-10의 ResNet-18 모형에 짜 넣어라. $\epsilon = 8/255$의 PGD-20 치기 아래에서 맑은 맞음과 든든한 맞음을 알려라.

??? success "연습문제 2 풀이"
    여느 ResNet-18은 맑은 맞음이 $\sim$93%이지만 PGD-20($\epsilon = 8/255$, 걸음 크기 $2/255$) 아래의 든든한 맞음은 $\sim$0%이다. 이 마디의 방법을 걸면 결과는 재주에 따라 다르다. 맞서며 익히기는 맑은 맞음 $\sim$83%에 든든한 맞음 $\sim$50%이고, 밝혀 낸 막이는 더 낮지만 증명할 수 있는 테두리를 준다. 맞음과 든든함의 맞바꿈은 밑바탕부터 있는 것이라, 든든함을 높이면 맑은 맞음이 흔히 5~15% 든다. 아무렇게나 하는 씨앗 3개의 평균과 잣대 어긋남으로 알려라. $\square$

---

**연습문제 3.**
흔듦 공 안에서 갈래별 자료의 밑자리가 서로 겹친다고 볼 때, 모형이 담는 힘을 키우지 않고서는 어떤 막이도 맑은 자료의 높은 맞음과 $\ell_\infty$ 흔듦에 대한 높은 든든함을 함께 이룰 수 없음을 증명하여라.

??? success "연습문제 3 풀이"
    두 갈래의 밑자리가 거리 $\epsilon$ 안에서 겹치면(곧 $\|x_1 - x_2\|_\infty \leq 2\epsilon$인 $x_1 \in \text{갈래 1}, x_2 \in \text{갈래 2}$이 있으면), $x_1$과 $x_2$ 둘 다에서 든든한 가름개는 적어도 하나를 틀리게 가를 수밖에 없다(흔듦 공이 겹치기 때문이다). 이것이 맞음과 든든함의 밑바탕 맞바꿈이다. 겹치는 밑자리의 몫이 피할 수 없는 맞음 잃음을 정한다. 여느 그림 분포에서는 $\epsilon = 8/255$에서 겹침이 꽤 있어, 살펴본 10~15%의 맞음 떨어짐을 밝혀 준다. 모형이 담는 힘을 키우면(더 너른 그물) 얽힌 든든한 판단의 금을 더 잘 그려 맞바꿈을 얼마쯤 눅일 수 있다. $\square$

---

**연습문제 4.**
금융 기계 배움 얼개(속임수 알아내기나 거래 신호 만들기 따위)에서 맞섬의 든든함이 어떻게 드러나는지 다루어라. 으름 얼개가 보기 다룸과 어떻게 다른가?

??? success "연습문제 4 풀이"
    금융에서 겨루는 이는 알아내는 얼개에 맞추어 스스로 움직이는 꾀 많은 무리(속임수꾼, 저자 흔드는 이)다. 보기 다룸과 다른 고갱이는 이렇다. (1) 흔들 수 있는 밭이 돈으로 될 만한 것에 옭매인다(속임수꾼이 제 거래 자취를 통째로 바꿀 수는 없다). (2) 치기가 잇따르며 맞추어 간다(겨루는 이가 얼개의 되받음을 보고 손본다). (3) 헛 맞음과 놓침의 값이 서로 어긋난다(옳은 거래를 막는 것과 속임수를 놓치는 것). (4) $\ell_p$ 노름은 뜻이 없고 밭에 맞는 흔듦 모형이 있어야 한다. 막이는 맞추어 오는 겨루는 이에게도 든든해야 하므로, 알아내는 잣대가 알려지면 비껴갈 수 있는 알아내기 바탕의 길은 많이 걸러진다. $\square$

## 정리하며

결 쥐어짜기는 모형을 가리지 않는 단순한 알아내기 켜를 준다. 홀로 쓰는 막이보다는 여러 켜 막이의 한 몫으로 가장 잘 듣고, 맞추어 오지 않는 겨루는 이에게 특히 그렇다.

**살펴볼 거리**

1. Xu, W., Evans, D., & Qi, Y. (2018). "Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks." NDSS.
