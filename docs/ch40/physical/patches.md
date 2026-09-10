# 맞서는 헝겊

**맞서는 헝겊**은 참 세상에 찍어 붙여 가름개를 속일 수 있는, 그 자리에 몰린 맞서는 흔듦이다. 낱그림점 모두를 알아챌 수 없게 고치는 $\ell_p$ 마디 흔듦과 달리, 헝겊은 들임의 작은 자리를 크기 옭아맴 없이 고친다. 그래서 참 세상 얼개를 치는 손에 잡히는 길이 된다.

---

## 1. 수학 꼴

맞서는 헝겊 $P$은 헝겊 붙이는 셈 $A$으로 들임 $\mathbf{x}$에 붙는다.

$$
\mathbf{x}_{\text{patched}} = A(\mathbf{x}, P, l, t) = (1 - M_l) \odot \mathbf{x} + M_l \odot t(P)
$$

여기서

- $M_l$은 헝겊을 붙일 자리 $l$을 정하는 두 값 가리개
- $t(\cdot)$은 자리 바꿈을 건다(돌림, 크기, 비스듬함)
- $\odot$은 낱낱의 곱을 뜻한다

### 두루 쓰는 헝겊 다듬기

**두루 쓰는 맞서는 헝겊**은 밑그림이 무엇이든 모형을 속인다.

$$
P^* = \arg\max_P \mathbb{E}_{\mathbf{x} \sim \mathcal{D}} \mathbb{E}_{l \sim \mathcal{U}} \mathbb{E}_{t \sim \mathcal{T}} \left[ \mathcal{L}(f_\theta(A(\mathbf{x}, P, l, t)), y_{\text{target}}) \right]
$$

바람은 다음에 걸쳐 잡는다.

- 자료 분포에서 아무렇게나 뽑은 그림
- 헝겊을 붙일 아무 자리
- 참 세상의 아무 바꿈

### 바꿈에 걸친 바람(EOT)

참 세상의 흔들림에 든든하도록 헝겊은 **바꿈에 걸친 바람**(아탈리에 등, 2018)으로 다듬는다.

$$
\nabla_P \mathbb{E}_{t \sim \mathcal{T}} [\mathcal{L}(f(t(A(\mathbf{x}, P))))] \approx \frac{1}{K} \sum_{k=1}^K \nabla_P \mathcal{L}(f(t_k(A(\mathbf{x}, P))))
$$

바꿈에는 돌림, 크기 바꿈, 밝기 바꿈, 비스듬히 틀기가 든다.

---

## 2. PyTorch로 짜기

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class AdversarialPatch:
    """
    두루 쓰는 맞서는 헝겊 만들개.
    
    어떤 그림에 붙여도 모형이 과녁 갈래를 미루어 보게 하는
    헝겊을 만든다.
    
    Parameters
    ----------
    model : nn.Module
        과녁 가름개
    patch_size : tuple
        헝겊의 크기 (H, W)
    target_class : int
        헝겊이 불러낼 갈래
    """
    
    def __init__(
        self,
        model: nn.Module,
        patch_size: Tuple[int, int] = (8, 8),
        target_class: int = 0,
        learning_rate: float = 0.01,
        num_transforms: int = 10,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.patch_size = patch_size
        self.target_class = target_class
        self.lr = learning_rate
        self.num_transforms = num_transforms
        self.device = device or next(model.parameters()).device
        
        self.model.eval()
        
        # 아무 헝겊으로 첫자리를 잡는다
        self.patch = torch.rand(
            1, 3, *patch_size, device=self.device, requires_grad=True
        )
    
    def _apply_patch(
        self, x: torch.Tensor, patch: torch.Tensor,
        location: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """아무 자리나 정해 준 자리에 헝겊을 붙인다."""
        B, C, H, W = x.shape
        ph, pw = self.patch_size
        
        if location is None:
            r = torch.randint(0, H - ph + 1, (1,)).item()
            c = torch.randint(0, W - pw + 1, (1,)).item()
        else:
            r, c = location
        
        x_patched = x.clone()
        patch_resized = F.interpolate(patch, size=(ph, pw), mode='bilinear')
        patch_resized = torch.clamp(patch_resized, 0, 1)
        x_patched[:, :, r:r+ph, c:c+pw] = patch_resized.expand(B, -1, -1, -1)
        
        return x_patched
    
    def _random_transform(self, x: torch.Tensor) -> torch.Tensor:
        """EOT을 위해 아무 밝기/맞섬을 건다."""
        brightness = 0.9 + 0.2 * torch.rand(1, device=self.device)
        return torch.clamp(x * brightness, 0, 1)
    
    def train_patch(
        self, train_loader, epochs: int = 10
    ) -> torch.Tensor:
        """
        두루 쓰는 맞서는 헝겊을 익힌다.
        
        익힌 헝겊 텐서를 돌려준다.
        """
        optimizer = torch.optim.Adam([self.patch], lr=self.lr)
        target = torch.tensor(
            [self.target_class], device=self.device
        )
        
        for epoch in range(epochs):
            total_loss = 0
            success = 0
            total = 0
            
            for x, y in train_loader:
                x = x.to(self.device)
                batch_target = target.expand(x.shape[0])
                
                optimizer.zero_grad()
                
                # EOT: 아무 바꿈에 걸친 평균 잃음
                loss = 0
                for _ in range(self.num_transforms):
                    x_patched = self._apply_patch(x, self.patch)
                    x_patched = self._random_transform(x_patched)
                    logits = self.model(x_patched)
                    loss += F.cross_entropy(logits, batch_target)
                
                loss /= self.num_transforms
                loss.backward()
                optimizer.step()
                
                # 헝겊을 옳은 자리로 자른다
                with torch.no_grad():
                    self.patch.clamp_(0, 1)
                
                total_loss += loss.item() * len(x)
                with torch.no_grad():
                    pred = self.model(self._apply_patch(x, self.patch)).argmax(1)
                    success += (pred == batch_target).sum().item()
                    total += len(x)
            
            print(f"{epoch+1}판: 잃음={total_loss/total:.4f}, "
                  f"먹힘={success/total:.2%}")
        
        return self.patch.detach()
```

---

## 3. 참 세상에서 헤아릴 것

### 찍어 내기와 내놓기

맞서는 헝겊은 참 세상의 흐름을 견뎌야 한다.

1. **셈틀에서 찍어 내기로**: 빛깔 눈금 맞추기, 찍개의 결 마디
2. **둘레의 것들**: 빛, 그림자, 보는 자리
3. **찍개가 담기**: 렌즈 뒤틀림, 절로 빛 맞추기, 잡음

EOT 익힘은 참 세상 바꿈의 분포에서 뽑아 이를 헤아린다.

### 헝겊의 결

| 결 | 셈틀 헝겊 | 참 세상 헝겊 |
|----------|----------------|-----------------|
| 흔듦 갈래 | 낱그림점 켜 | 찍어 낸 감 |
| 옭아맴 | 자리 + 크기 | 자리 + 크기 + 든든함 |
| 바꿔도 그대로임 | 골라 씀 | 꼭 있어야 함 |
| 먹힌 비율 | 90% 넘음 | 60~85% |

---

## 4. 금융에 쓰기

주로 셈틀 보기에서 살폈지만 헝겊의 깨침은 금융 자리로도 넓어진다.

- **결 끼워 넣기**: 거래 적바림에 맞서는 결 몇 개를 끼워 넣는 것과 같다
- **재개 흔들기**: 자료를 모으는 재개를 몸소 건드리기(물품 거래에 쓰는 별그림 따위)
- **글월 위조**: 저절로 다루는 흐름을 속이려 훑어 담은 글월을 그 자리에서 조금 고치기

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

이 마당은 수학 꼴、PyTorch로 짜기、참 세상에서 헤아릴 것、금융에 쓰기을 차례로 짚었다.

**살펴볼 거리**

1. Brown, T. B., et al. (2017). "Adversarial Patch." arXiv preprint arXiv:1712.09665.
2. Athalye, A., et al. (2018). "Synthesizing Robust Adversarial Examples." ICML.
3. Eykholt, K., et al. (2018). "Robust Physical-World Attacks on Deep Learning Visual Classification." CVPR.
