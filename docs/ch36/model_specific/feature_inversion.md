# 결 되돌리기
## 들머리

**결 되돌리기**는 가운데 나타냄에서 들임을 되세워, 그물이 켜마다 어떤 소식을 남기는지 드러낸다. 몫 매기기를 채워 주는 이 길은 *어떤* 결이 중요한지뿐 아니라 *모형이 참으로 무엇을 보는지*를 다루는 마디마다 보여 준다.

## 수학 밑바탕

켜 $l$의 결 나타냄 $\Phi_l(\mathbf{x})$이 주어지면 결 되돌리기는 다음을 찾는다.

$$
\mathbf{x}^* = \arg\min_{\mathbf{x}} \|\Phi_l(\mathbf{x}) - \Phi_l(\mathbf{x}_0)\|^2 + \lambda R(\mathbf{x})
$$

여기서 $R(\mathbf{x})$은 자연스러워 보이는 그림이 나오도록 이끄는 정칙화 항(온 흔들림, $L^2$ 크기)이다.

## 짜보기

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
        """켜의 결에서 들임을 되세운다."""
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

            # 결을 맞추는 잃음
            feat_loss = F.mse_loss(self.current_features, target_features)

            # 정칙화
            tv_loss = tv_weight * self.total_variation(x)
            l2_loss = l2_weight * x.pow(2).mean()

            loss = feat_loss + tv_loss + l2_loss
            loss.backward()
            optimizer.step()

        return x.detach()
```

## 읽는 법

결 되돌리기는 밑바탕이 되는 깨침을 드러낸다. **앞선 켜는 자리의 잔 무늬를 남기되 뜻 속살을 잃고, 뒤쪽 켜는 뜻 속살을 남기되 자리의 잔 무늬를 잃는다.** 이렇게 켜를 따라 점점 추려지기 때문에 (뒤쪽 켜를 겨누는) Grad-CAM이 성기면서도 갈래를 가려내는 열 그림을 내는 것이다.

## 간추림

결 되돌리기는 그물의 켜마다 어떤 소식이 남고 어떤 소식이 버려지는지를 보여 몫 매기기 방법을 채워 주며, 모형의 안쪽 나타냄을 통째로 알아보게 해 준다.

## 살펴볼 거리

1. Mahendran, A., & Vedaldi, A. (2015). "Understanding Deep Image Representations by Inverting Them." *CVPR*.

2. Dosovitskiy, A., & Brox, T. (2016). "Inverting Visual Representations with Convolutional Networks." *CVPR*.

## 익힘 문제

**익힘 1.**
이 마디에서 밝힌 풀이 방법을, XOR 들임을 가르는 ReLU 살림의 두 켜 신경 그물에 걸어라. 들임 $x = [1, 1]$에 대한 풀이를 셈하여라.

??? success "익힘 1 풀이"
    짐이 $W_1, b_1, W_2, b_2$인 익힌 XOR 그물에서 내놓기는 $f(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$이다. 풀이 방법은 들임 결마다 몫을 내놓는다. $x = [1, 1]$(갈래 0)이면 두 결 모두 음수 가름에 이바지한다. 몫 값은 방법마다 다르다. 기울기 바탕 방법은 $\partial f / \partial x_i$을 셈하고, 흔들어 보는 방법은 결을 가렸을 때 내놓기가 얼마나 바뀌는지 잰다. XOR 문제는 판단 금이 선형이 아니므로 선형 풀이 방법이 그르칠 수 있음을 보여 준다. $\square$

---

**익힘 2.**
이 마디의 풀이 방법이 온전함 공리를 채우는지, 곧 어떤 밑금 $x_0$에 대해 모든 결 몫의 합이 $f(x) - f(x_0)$과 같은지 증명하거나 뒤집어라.

??? success "익힘 2 풀이"
    온전함 공리(섀플리 값 이론에서는 효율이라고도 한다)는 몫의 합이 들임에서의 모형 내놓기와 밑금에서의 내놓기의 차이와 같다는 것이다. 이 방법이 온전함을 채우는지는 그 세움새에 달렸다. 기울기 방법은 온전함을 채우지 못한다(기울기는 그 자리의 것이고 길을 따라 쌓은 것이 아니다). 쌓은 기울기는 세움새 자체로 온전함을 채운다(길을 따라 미적분의 밑정리를 쓴다). SHAP 값은 섀플리 공리로 효율을 채운다. 온전함을 어기는 방법은 몫을 너무 많거나 적게 매길 수 있어, 온 몫을 온 세상 풀이로 믿기 어렵게 만든다. $\square$

---

**익힘 3.**
이 방법이 내놓는 풀이가 얼마나 미더운지 따지는 시험을 꾸며라. 짚어 준 결이 참으로 모형에 중요한지를 넣기와 빼기 곡선으로 재어라.

??? success "익힘 3 풀이"
    절차는 이렇다. (1) 시험 그림마다 결 몫을 셈한다. (2) 빼기: 몫이 큰 차례로 결을 하나씩 가리며 모형의 자신함이 떨어지는 모습을 적는다. 미더운 풀이면 자신함이 빠르게 떨어진다. (3) 넣기: 빈 밑금에서 시작해 몫이 큰 차례로 결을 하나씩 드러내며 자신함이 오르는 모습을 적는다. 미더운 풀이면 자신함이 빠르게 오른다. (4) 두 곡선의 아래 넓이를 셈한다. (5) 아무렇게나 매긴 차례(밑금)와 다른 방법에 견준다. 미더운 방법이면 빼기 넓이가 작고 넣기 넓이가 커야 한다. 통계로 미더우려면 시험 표본 1000개 넘게 되풀이한다. $\square$

---

**익힘 4.**
이 풀이 방법을 신용 부도를 미루어 보는 금융 모형에 어떻게 걸 수 있는지 다루어라. 풀이가 채워야 할 규정 요건은 무엇인가?

??? success "익힘 4 풀이"
    신용 모형에는 규정(ECOA, GDPR 22조)이 불리한 판단마다 그 사람에게 맞춘 풀이를 바란다. 방법은 다음을 내놓아야 한다. (1) 물리침에 가장 크게 이바지한 인자(불리한 처분 까닭). (2) 한결같은 풀이(비슷한 신청자는 비슷한 풀이를 받는다). (3) 손에 잡히는 풀이(신청자가 무엇을 바꾸어야 하는지 안다). 이 마디의 풀이 방법으로 결의 중요함을 짚을 수 있으나, 든든함(들임이 조금 바뀌었다고 풀이가 확 달라지면 안 된다)과 옳음(중요한 결을 없애면 미루어 봄이 바뀌어야 한다)을 따져 보아야 한다. 지켜야 할 됨됨이는 대리 차별이 드러나지 않도록 조심히 다루어야 한다. $\square$
