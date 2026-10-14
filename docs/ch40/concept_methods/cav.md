# 개념 활성화 벡터(CAV)

**개념 활성화 벡터(CAV)**는 신경망이 사람이 알아들을 수 있는 어떤 개념을 배웠는지 따지는 길을 준다. 추론을 날 입력 특징(픽셀, 단어 쏘아 넣기)로 풀이하는 대신, CAV는 "줄무늬 있음", "털 있음", "많이 출렁임", "평균으로 되돌아감" 같은 높은 층의 개념으로 풀이한다.

김 외(2018)가 들여온 CAV는 신경망이 셈하는 것(높은 차원 밭의 활성화)과 사람이 알아듣는 것(개념) 사이의 틈을 잇는다.

---

## 1. 수학 기초

### 활성화 밭에서의 개념 쪽

학습된 신경망에서 입력 앙상블에 대한 층 $l$의 활성화를 생각하자. 층 $l$에서 개념 $k$의 CAV는 활성화 밭에서 그 개념 쪽을 가리키는 벡터 $v_l^k$이다.

이 쪽을 찾으려면

1. 개념 $k$을 드러내는 **양의 보기** $P_k$을 모은다
2. **음의 보기** $N_k$(아무거나 또는 그 개념이 없는 것)을 모은다
3. 층 $l$의 활성화 $h_l(x)$에 선형 분류기를 학습된이다

$$
v_l^k = \arg\min_v \sum_{x \in P_k} \ell(\sigma(v^\top h_l(x)), 1) + \sum_{x \in N_k} \ell(\sigma(v^\top h_l(x)), 0)
$$

판단 금에 대한 법선 벡터가 곧 CAV $v_l^k$이다.

### 개념에 대한 예민함

층 $l$에서 갈래 $c$이 개념 $k$에 대해 지니는 **개념 예민함**은 이렇다.

$$
S_{c,k,l}(x) = \nabla_{h_l(x)} f_c(x) \cdot v_l^k
$$

활성화를 개념 쪽으로 움직일 때 클래스 점수가 얼마나 바뀌는지를 잰다.

---

## 2. PyTorch 짜보기

```python
import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression

class ConceptActivationVector:
    """개념 살아남 벡터를 셈하고 쓴다."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def get_activations(self, inputs: torch.Tensor) -> np.ndarray:
        """겨눈 켜의 살아남을 뽑아낸다."""
        self.model.eval()
        with torch.no_grad():
            self.model(inputs)

        act = self.activations
        if act.dim() > 2:
            act = act.mean(dim=tuple(range(2, act.dim())))
        return act.cpu().numpy()

    def train_cav(
        self,
        concept_examples: torch.Tensor,
        random_examples: torch.Tensor
    ) -> np.ndarray:
        """
        선형 분류기를 맞추어 CAV를 익힌다.

        Returns:
            cav_vector: 판단 금의 법선
        """
        pos_act = self.get_activations(concept_examples)
        neg_act = self.get_activations(random_examples)

        X = np.vstack([pos_act, neg_act])
        y = np.array([1] * len(pos_act) + [0] * len(neg_act))

        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X, y)

        cav = clf.coef_[0]
        cav = cav / np.linalg.norm(cav)

        accuracy = clf.score(X, y)
        print(f"CAV 분류기 맞음률: {accuracy:.3f}")

        return cav

    def conceptual_sensitivity(
        self,
        input_tensor: torch.Tensor,
        cav: np.ndarray,
        target_class: int
    ) -> float:
        """겨눈 갈래가 개념 쪽에 얼마나 예민한지 셈한다."""
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(False)

        self.model(input_tensor)
        act = self.activations
        act.requires_grad_(True)

        output = self.model.fc(act.mean(dim=(2, 3)) if act.dim() > 2 else act)
        output[0, target_class].backward()

        grad = act.grad
        if grad.dim() > 2:
            grad = grad.mean(dim=tuple(range(2, grad.dim())))

        grad_np = grad[0].cpu().numpy()
        cav_tensor = cav

        sensitivity = np.dot(grad_np, cav_tensor)
        return sensitivity
```

---

## 3. 계량 금융 응용

CAV로 금융 모델이 뜻있는 경제 개념을 배웠는지 따질 수 있다.

| 개념 | 양의 보기 | 쓰일 자리 |
|---------|-------------------|----------|
| "많이 출렁이는 판" | VIX > 25인 때 | 위험 모델 따지기 |
| "평균으로 되돌아감" | 옮김 평균으로 돌아오는 자산 | 꾀 풀이하기 |
| "밀림" | 요즘 크게 오른 자산 | 인자 모델 살피기 |
| "신용 죄임" | 벌어진 값 차이가 나타난 때 | 신용 위험 모델 |

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 마디에서 밝힌 풀이 방법을, XOR 입력을 가르는 ReLU 활성화의 두 층 신경망에 걸어라. 입력 $x = [1, 1]$에 대한 풀이를 계산하여라.

</div>

??? success "연습문제 1 풀이"
    가중치가 $W_1, b_1, W_2, b_2$인 학습된 XOR 네트워크에서 내놓기는 $f(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$이다. 풀이 방법은 입력 특징마다 몫을 내놓는다. $x = [1, 1]$(갈래 0)이면 두 특징 모두 음수 분류에 이바지한다. 몫 값은 방법마다 다르다. 기울기 기반 방법은 $\partial f / \partial x_i$을 계산하고, 흔들어 보는 방법은 특징을 가렸을 때 내놓기가 얼마나 바뀌는지 잰다. XOR 문제는 판단 금이 선형이 아니므로 선형 풀이 방법이 그르칠 수 있음을 보여 준다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
이 마디의 풀이 방법이 온전함 공리를 채우는지, 곧 어떤 밑금 $x_0$에 대해 모든 특징 몫의 합이 $f(x) - f(x_0)$과 같은지 증명하거나 뒤집어라.

</div>

??? success "연습문제 2 풀이"
    온전함 공리(섀플리 값 이론에서는 효율이라고도 한다)는 몫의 합이 입력에서의 모델 배포와 밑금에서의 내놓기의 차이와 같다는 것이다. 이 방법이 온전함을 채우는지는 그 세움새에 달렸다. 기울기 방법은 온전함을 채우지 못한다(기울기는 그 자리의 것이고 길을 따라 쌓은 것이 아니다). Integrated Gradients는 세움새 자체로 온전함을 채운다(길을 따라 미적분의 밑정리를 쓴다). SHAP 값은 섀플리 공리로 효율을 채운다. 온전함을 어기는 방법은 몫을 너무 많거나 적게 매길 수 있어, 온 몫을 온 세상 풀이로 믿기 어렵게 만든다. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 방법이 내놓는 풀이가 얼마나 미더운지 따지는 시험을 꾸며라. 짚어 준 특징이 참으로 모델에 중요한지를 넣기와 빼기 곡선으로 재어라.

</div>

??? success "연습문제 3 풀이"
    절차는 이렇다. (1) 시험 그림마다 특징 몫을 계산한다. (2) 빼기: 몫이 큰 차례로 특징을 하나씩 가리며 모델의 자신함이 떨어지는 모습을 적는다. 미더운 풀이면 자신함이 빠르게 떨어진다. (3) 넣기: 빈 밑금에서 시작해 몫이 큰 차례로 특징을 하나씩 드러내며 자신함이 오르는 모습을 적는다. 미더운 풀이면 자신함이 빠르게 오른다. (4) 두 곡선의 아래 넓이를 계산한다. (5) 아무렇게나 매긴 차례(밑금)와 다른 방법에 견준다. 미더운 방법이면 빼기 넓이가 작고 넣기 넓이가 커야 한다. 통계로 미더우려면 시험 표본 1000개 넘게 되풀이한다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
이 풀이 방법을 신용 부도를 예측하는 금융 모델에 어떻게 걸 수 있는지 다루어라. 풀이가 채워야 할 규정 요건은 무엇인가?

</div>

??? success "연습문제 4 풀이"
    신용 모델에는 규정(ECOA, GDPR 22조)이 불리한 판단마다 그 사람에게 맞춘 풀이를 바란다. 방법은 다음을 내놓아야 한다. (1) 물리침에 가장 크게 이바지한 인자(불리한 처분 까닭). (2) 한결같은 풀이(비슷한 신청자는 비슷한 풀이를 받는다). (3) 손에 잡히는 풀이(신청자가 무엇을 바꾸어야 하는지 안다). 이 마디의 풀이 방법으로 특징의 중요함을 짚을 수 있으나, 든든함(입력이 조금 바뀌었다고 풀이가 확 달라지면 안 된다)과 옳음(중요한 특징을 없애면 추론이 바뀌어야 한다)을 따져 보아야 한다. 지켜야 할 성능은 대리 차별이 드러나지 않도록 조심히 다루어야 한다. $\square$

## 정리하며

CAV는 사람이 알아들을 수 있는 개념에 맞물리는 쪽을 활성화 밭에서 찾아 개념 층의 풀이를 준다. 신경망의 속살과 밭 밝은 이의 앎 사이를 잇는다.

**살펴볼 거리**

1. Kim, B., et al. (2018). "Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors." *ICML*.

2. Ghorbani, A., et al. (2019). "Towards Automatic Concept-based Explanations." *NeurIPS*.
