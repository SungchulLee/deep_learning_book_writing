# 초점 손실

---

## 1. 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

- 촘촘한 물체 알아내기의 갈래 치우침 문제를 이해한다
- 초점 매개변수를 넣어 보통의 엇갈린 엔트로피에서 초점 손실을 이끌어 낸다
- 두 갈래와 여러 갈래 자리매김 모두에 대해 초점 손실을 짠다
- 장면마다 알맞은 웃매개변수(alpha, gamma)를 고른다

---

## 2. 왜 하는가: 쉬운 보기 문제

촘촘한 알아내기에서 후보 자리의 거의 다는 쉬운 음성(뒷바탕)이다. 보통의 엇갈린 엔트로피는 모든 보기를 똑같이 다루므로, 쉬운 음성 수천 개에서 쌓인 손실이 몇 안 되는 어렵고 알찬 보기의 신호를 뒤덮는다.

---

## 3. 초점 손실의 정의

초점 잃음은 엇결 엔트로피에 조절 값 $(1 - p_t)^\gamma$을 더한다.

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

여기서 $p_t$은 맞는 갈래에 대해 모델이 어림한 확률이다.

### 초점 매개변수 gamma의 효과

| $p_t$(자신도) | 엇결 엔트로피 잃음 | 초점 잃음($\gamma=2$) | 줄어든 비율 |
|---------------------|---------|-----------------|-----------|
| 0.9(쉬움) | 0.105 | 0.001 | **100×** |
| 0.5(가운데) | 0.693 | 0.173 | 4× |
| 0.1(어려움) | 2.303 | 1.867 | 1.2× |

$\gamma = 2$이면 잘 가려낸 보기의 짐이 100배 넘게 줄지만 어려운 보기는 거의 그대로다. 그래서 어려운 아님 보기를 따로 캐내지 않아도 알려 주는 바가 큰 보기에 익힘이 절로 몰린다.

### 알파 균형 인자

$\alpha_t$은 초점 짐과 따로 갈래 수준의 짐을 준다. 흔한 값은 $\alpha = 0.25$이다(더 잦은 바탕 갈래의 짐을 줄인다).

---

## 4. PyTorch 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    초점 손실: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    인수:
        alpha: 양성 갈래의 균형 인자(붙박이: 0.25)
        gamma: 초점 매개변수(붙박이: 2.0)
               gamma=0이면 보통의 엇갈린 엔트로피로 줄어든다
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        return (alpha_weight * focal_weight * bce).mean()

class MultiClassFocalLoss(nn.Module):
    """나누기와 알아내기를 위한 여러 갈래 초점 손실."""
    
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None,
                 ignore_index: int = -1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction='none',
                             ignore_index=self.ignore_index)
        
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1).clamp(0, logits.size(1) - 1)).squeeze(1)
        
        focal_weight = (1 - pt) ** self.gamma
        
        valid_mask = (targets != self.ignore_index).float()
        focal_loss = focal_weight * ce * valid_mask
        
        return focal_loss.sum() / (valid_mask.sum() + 1e-6)
```

---

## 5. 초매개변수 길잡이

| 설정 | $\gamma$ | $\alpha$ | 적바림 |
|---------|----------|----------|-------|
| 약한 치우침 | 1.0 | 0.5 | 약한 초점 |
| 보통의 알아내기 | 2.0 | 0.25 | 레티나넷 붙박이 |
| 극단적 치우침 | 3.0–5.0 | 0.1 | 의료, 작은 물체 |
| 고른 자료 뭉치 | 0.0 | 0.5 | 엇갈린 엔트로피로 줄어듦 |

---

## 연습문제

**연습문제 1.**
한 단계 알아내개와 두 단계 알아내개의 차이를 설명하여라. 빠르기와 정확도 사이의 근본 맞바꿈은 무엇인가?

??? success "연습문제 1 풀이"
    **두 단계 알아내개**(보기로 더 빠른 R-CNN)는 먼저 자리 제안을 만들고 제안마다 갈래를 매기고 다듬는다. 정확하지만 제안마다 다루기 때문에 느리다. **한 단계 알아내개**(보기로 YOLO, SSD)는 특징 지도에서 두름 상자와 갈래 확률을 한 번에 곧바로 어림하여 정확도를 조금 내주고 훨씬 빠른 미룸을 얻는다. 맞바꿈은 이렇다. 두 단계 알아내개는 작고 겹치는 물체를 잘 알아내지만 초당 5~15틀로 돌고, 한 단계 알아내개는 mAP가 조금 낮은 대신 초당 30~155틀 넘게 낸다.

---

**연습문제 2.**
겹침 비(교집합 나누기 합집합) 식을 이끌어 내고 두름 상자를 값매김할 때 왜 단순한 L2 거리보다 낫게 여기는지 설명하여라.

??? success "연습문제 2 풀이"
    두 두름 상자 $A$과 $B$에 대해:

    $$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}$$

    겹침 비를 낫게 여기는 까닭은 이렇다. (1) 잣수에 안 바뀐다(화소 10개의 어긋남은 큰 물체보다 작은 물체에 더 크게 다가온다). (2) 자연스레 $[0, 1]$에 놓여 좋음 점수로 읽을 수 있다. (3) 상자 자리표 사이의 L2 거리는 겹침을 담아내지 못해 두 상자의 L2 거리가 작아도 겹침이 0일 수 있다(보기로 하나가 다른 하나 안에 있는 경우와 나란히 놓인 경우).

---

**연습문제 3.**
최대가 아닌 것 누르기(NMS)를 짜고 알아내기 물길에서 그것이 하는 몫을 설명하여라.

??? success "연습문제 3 풀이"
    ```python
    import numpy as np

    def nms(boxes, scores, iou_threshold=0.5):
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            remaining = order[1:]
            ious = compute_iou(boxes[i], boxes[remaining])
            mask = ious <= iou_threshold
            order = remaining[mask]
        return keep
    ```
    NMS는 같은 물체를 거듭 알아낸 것을 없앤다. 후보 상자에 점수를 매긴 뒤 점수가 가장 높은 상자를 고르고 겹침 비가 문턱값을 넘는 상자(겹친 것일 가능성이 높다)를 모두 없애기를 되풀이한다.

---

**연습문제 4.**
물체 알아내기의 갈래 치우침 문제와 초점 손실이 그것을 어떻게 다루는지 설명하여라.

??? success "연습문제 4 풀이"
    한 단계 알아내개에서는 닻 상자 대부분이 바탕(쉬운 아님 보기)이고 물체를 담은 것은 몇 안 된다. 여느 엇결 엔트로피 잃음은 쉬운 아님 보기가 워낙 많아 그쪽에 휘둘리므로 어려운 맞음 보기의 기울기 신호가 묻힌다. **초점 잃음**은 조절 값을 더한다. $\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$이다. $\gamma > 0$이면 쉬운 보기($p_t$이 높은 것)의 짐이 지수로 줄어 어려운 보기에 익힘이 몰린다. $\gamma = 2$과 $\alpha = 0.25$을 쓰면 RetinaNet은 한 단계의 빠르기를 지키면서 두 단계 알아내개에 맞먹는 맞음을 이룬다.

## 정리하며

초점 잃음은 쉬운 보기의 짐을 줄여 촘촘한 알아내기의 밑바탕 갈래 치우침을 다룬다. $\gamma = 2$과 $\alpha = 0.25$을 쓰면 RetinaNet이 어떤 표집 요령도 없이 두 단계 알아내개의 맞음에 맞먹게 되니, 맞음의 틈이 얼개의 한계가 아니라 갈래 치우침에서 왔음을 보여 준다.

**참고 문헌**

1. Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. ICCV.
