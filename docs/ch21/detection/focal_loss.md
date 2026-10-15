# Focal loss와 쉬운 예제의 가중치

촘촘한 객체 탐지에서는 객체가 없는 위치가 많다. 개별 손실이 작아도 쉬운 배경 예제가 매우 많으면 전체 손실·기울기에 큰 영향을 줄 수 있다. Focal loss는 잘 맞힌 예제의 기여를 줄이는 방식이다. 교차 엔트로피가 모든 예제에서 같은 크기의 손실이나 기울기를 만든다는 뜻은 아니다.

## 1. 이진 focal loss의 정의

이진 정답 y∈{0,1}, 양성 확률 p에 대해 p_t=p이면 y=1, p_t=1−p이면 y=0이다. α_t는 양성에 α, 음성에 1−α를 적용한다.

$$\operatorname{FL}(p_t)=-\alpha_t(1-p_t)^\gamma\log p_t.$$

γ≥0이며 γ=0이면 **α 가중** 이진 교차 엔트로피다. α=0.5이면 일반 BCE의 절반이므로 숫자까지 같은 손실은 아니다. γ는 난이도에 따른 가중, α는 정답 클래스에 따른 가중을 제어한다. 확률과 가중치도 모델 출력에 의존하므로 기울기는 단순히 기존 BCE 기울기에 고정 가중치를 곱한 것과 다르다.

## 2. γ와 α를 읽는 방법

다음은 α를 제외한 손실 값의 비교다. 비율은 손실 값에 대한 것이며 기울기 감소율은 아니다.

| p_t | 교차 엔트로피 | γ=2 focal 항 | focal/CE |
|---|---|---|---|
| 0.9 | 약 0.10536 | 약 0.001054 | 0.01 |
| 0.5 | 약 0.69315 | 약 0.17329 | 0.25 |
| 0.1 | 약 2.30259 | 약 1.86509 | 0.81 |

p_t=0.9에서는 정확히 100분의 1이고 더 쉬운 예제에서는 더 작아진다. γ를 크게 하면 어려운 예제·오주석·이상치에 상대적으로 더 집중할 수 있으므로 클수록 좋은 것은 아니다. α=0.25는 양성에 0.25, 음성에 0.75를 주므로 ‘α가 작으면 배경 가중치를 낮춘다’는 설명은 이 정의와 반대다. 전경·배경 불균형의 처리는 α와 focal 조절, 목표 배정·집계를 함께 보아야 한다.

## 3. 이진·다중 클래스 구현

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

`FocalLoss`는 같은 모양의 로짓·0/1 목표를 받고 모든 원소의 평균을 낸다. N×K 독립 sigmoid 출력에도 적용할 수 있지만 상호 배타적인 클래스의 softmax 목적과는 다르다. 목표가 0~1의 soft label이면 p_t 정의도 혼합값이 되어 원래 hard-label focal loss와 같은 해석을 그대로 적용하지 않는다. 이 구현에는 ignore mask가 없다.

`MultiClassFocalLoss`는 클래스 차원 1의 softmax 로짓과 정수 클래스 목표를 받아 유효 위치의 평균을 계산한다. `ignore_index`인 위치는 제외하며 모두 무시되면 0을 반환한다. 그러나 인수 `alpha`는 저장만 하고 실제 손실에는 사용하지 않는다. 따라서 클래스 가중 focal loss가 구현되었다고 볼 수 없다. 무시되지 않은 정답은 유효한 클래스 범위 안이어야 하며 gather의 clamp가 cross_entropy의 잘못된 정답 ID까지 허용하는 것은 아니다.

## 4. 손실 집계와 비교 설계

위 이진 구현은 전체 원소 평균이고 다중 클래스 구현은 유효 위치 평균이다. 검출기에서 양성 앵커 수로 정규화하는 손실과 배율이 다르다. 클래스·공간 차원 수, 무시 위치, 배치별 양성 수와 회귀 손실 계수를 함께 확인해야 한다.

[RetinaNet 논문](https://arxiv.org/abs/1708.02002)의 α=0.25, γ=2는 해당 실험의 설정이다. 불균형 수준만 보고 γ=3~5나 α=0.1을 보편적 처방으로 정할 수 없다. 같은 데이터 분할·평가에서 CE 또는 가중 CE와 비교하고 클래스별 지표와 오주석 영향을 검토한다. focal loss가 모든 샘플링 기법을 불필요하게 만들거나 구조의 한계를 없애는 것은 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
γ=0, α=0.5일 때 첫 구현과 일반 BCE 평균의 관계를 설명하라.

</div>

??? success "연습문제 1 풀이"

    모든 예제의 alpha_weight가 0.5이고 focal_weight는 1이므로 반환값은 BCE 평균의 절반이다. 분류 목적의 최소점이 같을 수 있어도 다른 손실과 합칠 때 상대 배율은 달라진다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
p_t=0.9와 0.1에서 γ=2의 손실 가중치를 비교하라. α=0.25는 음성에 어떤 가중치를 주는가?

</div>

??? success "연습문제 2 풀이"

    focal 가중치는 각각 0.01과 0.81이다. α=0.25이면 음성의 alpha_weight는 0.75다. 이 수치는 손실의 가중치이고 모델 출력에 따른 가중치 변화까지 포함한 경사의 비율은 아니다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
focal loss와 아래 NMS의 역할이 어떻게 다른가? 코드의 실행에 필요한 별도 정의도 확인하라.

</div>

??? success "연습문제 3 풀이"

    focal loss는 학습 중 예제별 기여를 조절하고 NMS는 추론 후보의 중복을 정리한다. 하나가 다른 하나를 대체하지 않는다. 아래 NumPy 함수는 compute_iou 정의가 별도로 필요하고 이미 디코딩된 상자·점수를 가정한다. 임계값과 같은 IoU는 유지한다.

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


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
MultiClassFocalLoss에서 alpha를 바꾸면 결과가 달라지는가? 이진 구현과 평균 기준은 같은가?

</div>

??? success "연습문제 4 풀이"

    alpha는 사용하지 않으므로 다른 조건이 같으면 결과에 영향이 없다. 다중 클래스는 유효 위치 수로 나누고 이진 구현은 모든 원소 평균을 내므로 집계 단위도 다르다. 클래스 가중치를 실제 적용할 때는 정답별 가중과 정규화 기준을 명시해야 한다.



## 정리하며

Focal loss는 잘 맞힌 예제의 손실을 상대적으로 줄인다. γ의 난이도 가중과 α의 클래스 가중, sigmoid·softmax 목표와 평균 기준을 구분한다. 여기서는 다중 클래스 구현의 alpha가 사용되지 않으므로 설명과 실제 계산을 일치시켜 읽어야 한다.

**참고 문헌**

1. Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. ICCV.
