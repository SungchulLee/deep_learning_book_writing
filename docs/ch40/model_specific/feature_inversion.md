# 피처 역변환(Feature Inversion)

**피처 역변환**은 중간 표현에서 입력을 재구성해, 네트워크가 레이어마다 어떤 정보를 남기는지 드러낸다. 기여도 분석을 보완하는 이 방법은 *어떤* 피처가 중요한지뿐 아니라 *모델이 실제로 무엇을 보는지*를 처리 단계마다 보여 준다.

---

## 1. 수학 기초

레이어 $l$의 피처 표현 $\Phi_l(\mathbf{x})$가 주어지면 피처 역변환은 다음을 찾는다.

$$
\mathbf{x}^* = \arg\min_{\mathbf{x}} \|\Phi_l(\mathbf{x}) - \Phi_l(\mathbf{x}_0)\|^2 + \lambda R(\mathbf{x})
$$

여기서 $R(\mathbf{x})$는 자연스러워 보이는 이미지가 나오도록 이끄는 정칙화 항(총변동(total variation), $L^2$ 크기)이다. 총변동은 이웃한 픽셀 사이 차이의 절댓값을 모두 더한 값으로(아래 구현은 평균을 쓴다), 이를 줄이면 이미지가 매끄러워진다.

---

## 2. 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureInversion:
    """가운데 결에서 들임을 되세운다."""

    def __init__(self, model, target_layer, device):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.target_features = None

        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'current_features', o)
        )

    def total_variation(self, x):
        """자리를 매끄럽게 하는 온 흔들림 정칙화 항."""
        diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        return diff_h.mean() + diff_w.mean()

    def invert(
        self, target_input, n_steps=500, lr=0.05,
        tv_weight=1e-3, l2_weight=1e-5
    ):
        """레이어의 결에서 들임을 되세운다."""
        self.model.eval()

        with torch.no_grad():
            self.model(target_input.to(self.device))
            target_features = self.current_features.clone()

        # 잡음에서 시작한다
        x = torch.randn_like(target_input, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([x], lr=lr)

        for step in range(n_steps):
            optimizer.zero_grad()
            self.model(x)

            # 결을 맞추는 로스
            feat_loss = F.mse_loss(self.current_features, target_features)

            # 정칙화
            tv_loss = tv_weight * self.total_variation(x)
            l2_loss = l2_weight * x.pow(2).mean()

            loss = feat_loss + tv_loss + l2_loss
            loss.backward()
            optimizer.step()

        return x.detach()
```

---

## 3. 읽는 법

피처 역변환은 근본적인 사실 하나를 드러낸다. **앞쪽 레이어는 공간적인 세부 패턴을 남기되 의미 정보가 약하고, 뒤쪽 레이어는 의미 정보를 남기되 공간적인 세부를 잃는다.** 이렇게 레이어를 따라 정보가 점점 추상화되기 때문에 (뒤쪽 레이어를 대상으로 하는) Grad-CAM이 해상도는 낮아도 클래스를 잘 가려내는 히트맵을 내는 것이다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
XOR를 계산하는 ReLU 네트워크의 은닉 레이어가 $\Phi(\mathbf{x}) = (\text{ReLU}(x_1 - x_2), \text{ReLU}(x_2 - x_1))$이라 하자. $\mathbf{x}_0 = [1, 1]$의 표현을 역변환하라. 정칙화가 $R(\mathbf{x}) = \|\mathbf{x}\|^2$이고 $\lambda$가 아주 작을 때 $\mathbf{x}^*$는 무엇이며, 이 결과는 이 레이어에 대해 무엇을 말해 주는가?

</div>

??? success "풀이"
    $\Phi(\mathbf{x}_0) = (0, 0)$이다. 이 표현을 정확히 재현하는 입력은 $x_1 = x_2$인 직선 전체이므로, 피처 맞춤 항만으로는 해가 하나로 정해지지 않는다. 그 직선 위에서 $\|\mathbf{x}\|^2$을 가장 작게 하는 점은 $\mathbf{x}^* = [0, 0]$이다. 곧 $[1, 1]$을 역변환하면 $[0, 0]$이 나온다. 이 레이어는 두 입력의 차이만 남기고 「두 값이 얼마나 큰가」는 버렸다는 뜻이다. 역변환 결과에서 원래 입력과 같은 부분은 레이어가 남긴 정보이고 달라진 부분은 버린 정보이며, 달라진 부분을 무엇으로 채우는지는 정칙화가 정한다. 그래서 역변환 이미지를 읽을 때는 정칙화가 그려 넣은 것과 레이어가 남긴 것을 갈라 보아야 한다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
이 절의 설명 방법이 완전성 공리를 만족하는지, 곧 어떤 베이스라인 $x_0$에 대해 모든 피처 기여도의 합이 $f(x) - f(x_0)$과 같은지 증명하거나 반례를 들어라.

</div>

??? success "풀이"
    완전성 공리(섀플리 값 이론에서는 효율성이라고도 한다)는 기여도의 합이 입력에서의 모델 출력과 베이스라인에서의 출력의 차이와 같다는 것이다. 이 방법이 완전성을 만족하는지는 그 구성에 달렸다. 그래디언트 방법은 완전성을 만족하지 못한다(그래디언트는 한 점의 국소적인 양이고 경로를 따라 적분한 것이 아니다). Integrated Gradients는 구성 자체로 완전성을 만족한다(경로를 따라 미적분의 기본정리를 쓴다). SHAP 값은 섀플리 공리로 효율성을 만족한다. 완전성을 어기는 방법은 기여도를 너무 크거나 작게 매길 수 있어, 기여도 전체를 전역 설명으로 믿기 어렵게 만든다. 피처 역변환은 피처마다 기여도를 매기지 않고 입력 공간의 이미지 하나를 내놓으므로, 완전성 공리가 적용되는 방법이 아니다. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 방법이 내놓는 설명이 얼마나 믿을 만한지 평가하는 실험을 설계하라. 방법이 중요하다고 짚은 피처가 실제로 모델에 중요한지를 삽입 곡선과 삭제 곡선으로 재어라.

</div>

??? success "풀이"
    절차는 이렇다. (1) 시험 이미지마다 피처 기여도를 계산한다. (2) 삭제: 기여도가 큰 차례로 피처를 하나씩 가리며 모델의 확신도가 떨어지는 모습을 기록한다. 믿을 만한 설명이면 확신도가 빠르게 떨어진다. (3) 삽입: 빈 베이스라인에서 시작해 기여도가 큰 차례로 피처를 하나씩 드러내며 확신도가 오르는 모습을 기록한다. 믿을 만한 설명이면 확신도가 빠르게 오른다. (4) 두 곡선 아래 넓이를 계산한다. (5) 무작위 순서(베이스라인)와 다른 방법에 견준다. 믿을 만한 방법이면 삭제 곡선의 넓이가 작고 삽입 곡선의 넓이가 커야 한다. 통계적으로 믿을 만하려면 시험 표본 1000개 넘게 되풀이한다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
이 설명 방법을 신용 부도를 예측하는 금융 모델에 어떻게 적용할 수 있는지 논하라. 설명이 채워야 할 규제 요건은 무엇인가?

</div>

??? success "풀이"
    신용 모델에는 규제(ECOA, GDPR 22조)가 불리한 결정마다 그 사람에게 맞춘 설명을 요구한다. 방법은 다음을 내놓아야 한다. (1) 거절에 가장 크게 기여한 요인(불리한 처분 사유). (2) 일관된 설명(비슷한 신청자는 비슷한 설명을 받는다). (3) 실행 가능한 설명(신청자가 무엇을 바꾸어야 하는지 안다). 이 절의 설명 방법으로 피처 중요도를 짚을 수 있으나, 강건성(입력이 조금 바뀌었다고 설명이 확 달라지면 안 된다)과 충실성(중요한 피처를 없애면 예측이 바뀌어야 한다)을 평가해야 한다. 보호 속성은 대리 차별이 드러나지 않도록 조심히 다루어야 한다. $\square$

## 정리하며

피처 역변환은 네트워크의 레이어마다 어떤 정보가 남고 어떤 정보가 버려지는지를 보여 기여도 분석 방법을 보완하며, 모델의 내부 표현을 전체적으로 이해하게 해 준다.

**살펴볼 거리**

1. Mahendran, A., & Vedaldi, A. (2015). "Understanding Deep Image Representations by Inverting Them." *CVPR*.

2. Dosovitskiy, A., & Brox, T. (2016). "Inverting Visual Representations with Convolutional Networks." *CVPR*.
