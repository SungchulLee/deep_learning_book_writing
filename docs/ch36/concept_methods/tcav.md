# TCAV: 개념 살아남 벡터로 따지기
## 들머리

**CAV로 따지기(TCAV)**는 개념 살아남 벡터를 넓혀, 어떤 개념이 모형의 미루어 봄에 얼마나 중요한지를 수와 통계로 따진다. CAV가 살아남 밭에서 개념의 쪽을 세우는 데 견주어, TCAV은 **어떤 갈래의 들임 가운데 몇 몫이 그 개념에 양수로 흔들리는지**를 잰다.

## 수학 밑바탕

### TCAV 점수

갈래 $c$, 개념 $k$, 켜 $l$에 대한 TCAV 점수는 이렇다.

$$
\text{TCAV}_{c,k,l} = \frac{|\{x \in X_c : S_{c,k,l}(x) > 0\}|}{|X_c|}
$$

여기서 $S_{c,k,l}(x) = \nabla_{h_l(x)} f_c(x) \cdot v_l^k$은 개념 예민함이고 $X_c$은 갈래 $c$에 드는 들임의 모임이다.

TCAV 점수가 0.8이면 갈래 $c$ 들임의 80%이 개념 $k$에 양수로 흔들린다는 뜻이다.

### 통계로 따지기

뜻있음을 따지려고 TCAV은 아무렇게나 만든 CAV를 영 가설로 쓴다.

1. 아무 개념(아무렇게나 고른 그림 모둠)으로 CAV를 여럿 익힌다
2. 아무 CAV마다 TCAV 점수를 셈한다
3. 참 개념의 TCAV 점수가 아무 것과 뜻있게 다른지 따진다

TCAV 점수가 0.5(우연)와 통계로 뜻있게 다르면 그 개념을 뜻있다고 여긴다.

## PyTorch 짜보기

```python
import torch
import numpy as np
from scipy.stats import ttest_1samp

class TCAV:
    """개념 살아남 벡터로 따지기."""

    def __init__(self, model, target_layer, device):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.cav_module = ConceptActivationVector(model, target_layer)

    def compute_tcav_score(
        self,
        class_inputs: torch.Tensor,
        cav: np.ndarray,
        target_class: int
    ) -> float:
        """개념과 갈래에 대한 TCAV 점수를 셈한다."""
        positive_count = 0
        total = len(class_inputs)

        for i in range(total):
            sensitivity = self.cav_module.conceptual_sensitivity(
                class_inputs[i:i+1].to(self.device),
                cav, target_class
            )
            if sensitivity > 0:
                positive_count += 1

        return positive_count / total

    def tcav_with_significance(
        self,
        class_inputs: torch.Tensor,
        concept_examples: torch.Tensor,
        random_example_sets: list,
        target_class: int,
        alpha: float = 0.05
    ) -> dict:
        """
        통계로 뜻있음을 따지며 TCAV 점수를 셈한다.

        Args:
            class_inputs: 겨눈 갈래의 들임
            concept_examples: 그 개념의 보기
            random_example_sets: 영 가설에 쓸 아무 보기 모둠의 목록
            target_class: 따질 갈래
            alpha: 뜻있음 문턱
        """
        # 개념 CAV를 익힌다
        random_neg = random_example_sets[0]
        concept_cav = self.cav_module.train_cav(concept_examples, random_neg)
        concept_score = self.compute_tcav_score(
            class_inputs, concept_cav, target_class
        )

        # 영 가설에 쓸 아무 CAV를 익힌다
        random_scores = []
        for i in range(0, len(random_example_sets) - 1, 2):
            random_cav = self.cav_module.train_cav(
                random_example_sets[i], random_example_sets[i + 1]
            )
            score = self.compute_tcav_score(
                class_inputs, random_cav, target_class
            )
            random_scores.append(score)

        # 0.5에 대한 양쪽 t 검정
        t_stat, p_value = ttest_1samp(
            [concept_score] + random_scores, 0.5
        )

        return {
            'tcav_score': concept_score,
            'random_scores': random_scores,
            'p_value': p_value,
            'significant': p_value < alpha,
            'concept_meaningful': concept_score > 0.5 and p_value < alpha
        }
```

## 계량 금융에 쓰기

TCAV으로 금융 모형이 경제로 뜻있는 개념을 배웠는지 따질 수 있다.

```python
def test_financial_concepts(model, layer, device):
    """돌아옴을 미루어 보는 모형이 금융 개념을 쓰는지 따진다."""

    tcav = TCAV(model, layer, device)

    concepts_to_test = {
        'momentum': momentum_examples,
        'mean_reversion': reversion_examples,
        'volatility_regime': vol_regime_examples,
        'credit_stress': credit_stress_examples
    }

    for concept_name, examples in concepts_to_test.items():
        result = tcav.tcav_with_significance(
            class_inputs=positive_return_samples,
            concept_examples=examples,
            random_example_sets=random_sets,
            target_class=1  # 양수 돌아옴 갈래
        )

        sig = "***" if result['significant'] else "뜻없음"
        print(f"{concept_name:20s}: TCAV={result['tcav_score']:.3f} "
              f"p={result['p_value']:.4f} {sig}")
```

## 간추림

TCAV은 신경 그물에서 개념의 중요함을 따지는 엄밀하고 수로 재는 틀을 준다. 통계로 따지므로 참으로 뜻있는 개념만 가려내며, 풀이의 됨됨이를 따져야 하는 규정된 자리에 알맞다.

## 살펴볼 거리

1. Kim, B., et al. (2018). "Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors." *ICML*.

2. Ghorbani, A., Wexler, J., Zou, J., & Kim, B. (2019). "Towards Automatic Concept-based Explanations." *NeurIPS*.

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
