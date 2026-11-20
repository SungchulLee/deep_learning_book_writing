# 적대적 패치

**적대적 패치**는 현실 세계에 찍어 붙여 분류기를 속일 수 있는, 그 자리에 몰린 적대적 섭동이다. 픽셀 모두를 알아챌 수 없게 고치는 $\ell_p$ 마디 섭동과 달리, 패치는 입력의 작은 자리를 크기 제약 없이 고친다. 그래서 현실 세계 구조를 치는 실용적인 길이 된다.

---

## 1. 수학 형태

적대적 패치 $P$은 패치 붙이는 계산 $A$으로 입력 $\mathbf{x}$에 붙는다.

$$
\mathbf{x}_{\text{patched}} = A(\mathbf{x}, P, l, t) = (1 - M_l) \odot \mathbf{x} + M_l \odot t(P)
$$

여기서

- $M_l$은 패치를 붙일 자리 $l$을 정하는 두 값 마스크
- $t(\cdot)$은 자리 변환을 건다(돌림, 크기, 비스듬함)
- $\odot$은 낱낱의 곱을 뜻한다

### 두루 쓰는 패치 다듬기

**두루 쓰는 적대적 패치**는 밑그림이 무엇이든 모델을 속인다.

$$
P^* = \arg\max_P \mathbb{E}_{\mathbf{x} \sim \mathcal{D}} \mathbb{E}_{l \sim \mathcal{U}} \mathbb{E}_{t \sim \mathcal{T}} \left[ \mathcal{L}(f_\theta(A(\mathbf{x}, P, l, t)), y_{\text{target}}) \right]
$$

바람은 다음에 걸쳐 잡는다.

- 데이터 분포에서 무작위로 뽑은 그림
- 패치를 붙일 아무 자리
- 현실 세계의 아무 바꿈

### 변환에 대한 기댓값(EOT)

현실 세계의 흔들림에 강건하도록 패치는 **변환에 대한 기댓값**(아탈리에 등, 2018)으로 다듬는다.

$$
\nabla_P \mathbb{E}_{t \sim \mathcal{T}} [\mathcal{L}(f(t(A(\mathbf{x}, P))))] \approx \frac{1}{K} \sum_{k=1}^K \nabla_P \mathcal{L}(f(t_k(A(\mathbf{x}, P))))
$$

바꿈에는 돌림, 크기 변환, 밝기 변환, 비스듬히 틀기가 든다.

---

## 2. PyTorch로 구현하기

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class AdversarialPatch:
    """
    두루 쓰는 맞서는 헝겊 생성기.
    
    어떤 그림에 붙여도 모형이 타깃 클래스를 미루어 보게 하는
    헝겊을 만든다.
    
    Parameters
    ----------
    model : nn.Module
        타깃 분류기
    patch_size : tuple
        헝겊의 크기 (H, W)
    target_class : int
        헝겊이 불러낼 클래스
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
        """EOT을 위해 아무 밝기/적대적 공격을 건다."""
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
                
                # EOT: 아무 바꿈에 걸친 평균 로스
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
            
            print(f"{epoch+1}판: 로스={total_loss/total:.4f}, "
                  f"먹힘={success/total:.2%}")
        
        return self.patch.detach()
```

---

## 3. 현실 세계에서 헤아릴 것

### 찍어 내기와 출력

적대적 패치는 현실 세계의 흐름을 견뎌야 한다.

1. **디지털에서 찍어 내기로**: 색 보정, 프린터의 피처 마디
2. **둘레의 것들**: 빛, 그림자, 보는 자리
3. **프린터가 담기**: 렌즈 뒤틀림, 절로 빛 맞추기, 잡음

EOT 학습은 현실 세계 바꿈의 분포에서 뽑아 이를 헤아린다.

### 패치의 피처

| 피처 | 디지털 패치 | 물리 패치 |
|----------|----------------|-----------------|
| 섭동 클래스 | 픽셀 레이어 | 찍어 낸 감 |
| 제약 | 자리 + 크기 | 자리 + 크기 + 강건성 |
| 바꿔도 그대로임 | 골라 씀 | 꼭 있어야 함 |
| 먹힌 비율 | 90% 넘음 | 60~85% |

---

## 4. 금융 응용

주로 디지털 보기에서 살폈지만 패치의 통찰은 금융 자리로도 넓어진다.

- **피처 끼워 넣기**: 거래 적바림에 적대적 패턴 몇 개를 끼워 넣는 것과 같다
- **재개 흔들기**: 데이터를 모으는 재개를 몸소 건드리기(물품 거래에 쓰는 별그림 따위)
- **텍스트 위조**: 저절로 다루는 흐름을 속이려 훑어 담은 텍스트를 그 자리에서 조금 고치기

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
    두 클래스의 서포트가 거리 $\epsilon$ 안에서 겹치면(곧 $\|x_1 - x_2\|_\infty \leq 2\epsilon$인 $x_1 \in \text{클래스 1}, x_2 \in \text{클래스 2}$이 있으면), $x_1$과 $x_2$ 둘 다에서 강건한 분류기는 적어도 하나를 틀리게 가를 수밖에 없다(섭동 공이 겹치기 때문이다). 이것이 정확도와 강건성의 기초 트레이드오프다. 겹치는 서포트의 비중이 피할 수 없는 정확도 손실을 정한다. 자연 이미지 분포에서는 $\epsilon = 8/255$에서 겹침이 꽤 있어, 관찰되는 10~15%의 정확도 하락을 설명해 준다. 모델 용량을 키우면(더 넓은 네트워크) 복잡한 강건 결정 경계를 더 잘 그려 트레이드오프를 어느 정도 완화할 수 있다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
금융 머신러닝 시스템(사기 탐지나 거래 신호 생성 등)에서 적대적 강건성이 어떻게 드러나는지 논하여라. 위협 모델이 비전과 어떻게 다른가?

</div>

??? success "풀이"
    금융에서 적대자는 탐지 시스템에 맞추어 전략을 바꾸는 교묘한 집단(사기꾼, 시장 조작자)이다. 비전과 다른 핵심은 이렇다. (1) 섭동할 수 있는 피처가 경제적으로 그럴듯한 범위로 제약된다(사기꾼이 자기 거래 이력을 통째로 바꿀 수는 없다). (2) 공격이 순차적이고 적응적이다(적대자가 시스템의 응답을 보고 고친다). (3) 거짓 양성과 거짓 음성의 비용이 비대칭이다(정상 거래를 막는 것과 사기를 놓치는 것). (4) $\ell_p$ 노름은 의미가 없고 도메인에 맞는 섭동 모델이 있어야 한다. 방어는 적응형 적대자에게도 강건해야 하므로, 탐지 기준이 알려지면 회피할 수 있는 탐지 기반 방법은 쓸모가 크게 줄어든다. $\square$

## 정리하며

이 절에서는 수학 형태, PyTorch로 구현하기, 현실 세계에서 헤아릴 것, 금융 응용을 살펴보았다.

**살펴볼 거리**

1. Brown, T. B., et al. (2017). "Adversarial Patch." arXiv preprint arXiv:1712.09665.
2. Athalye, A., et al. (2018). "Synthesizing Robust Adversarial Examples." ICML.
3. Eykholt, K., et al. (2018). "Robust Physical-World Attacks on Deep Learning Visual Classification." CVPR.
