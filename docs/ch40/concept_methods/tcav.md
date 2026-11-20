# TCAV: 개념 활성화 벡터를 이용한 평가

**CAV를 이용한 검정(TCAV, Testing with CAV)**은 개념 활성화 벡터를 넓혀, 어떤 개념이 모델의 예측에 얼마나 중요한지를 정량적·통계적으로 평가한다. CAV가 활성화 공간에서 개념의 방향을 찾는 데 비해, TCAV는 **어떤 클래스의 입력 가운데 몇 퍼센트가 그 개념에 양의 민감도를 보이는지**를 잰다.

---

## 1. 수학 기초

### TCAV 점수

클래스 $c$, 개념 $k$, 레이어 $l$에 대한 TCAV 점수는 이렇다.

$$
\text{TCAV}_{c,k,l} = \frac{|\{x \in X_c : S_{c,k,l}(x) > 0\}|}{|X_c|}
$$

여기서 $S_{c,k,l}(x) = \nabla_{h_l(x)} f_c(x) \cdot v_l^k$는 개념 민감도이고 $X_c$는 클래스 $c$에 드는 입력의 집합이다.

TCAV 점수가 0.8이면 클래스 $c$ 입력의 80%가 개념 $k$에 양의 민감도를 보인다는 뜻이다.

### 통계적 검정

유의성을 평가하려고 TCAV는 무작위로 만든 CAV를 귀무가설로 쓴다.

1. 무작위 개념(무작위로 고른 이미지 집합)으로 CAV를 여럿 학습한다
2. 무작위 CAV마다 TCAV 점수를 계산한다
3. 실제 개념의 TCAV 점수가 무작위 CAV의 점수와 유의하게 다른지 검정한다

Kim 등(2018)은 개념 CAV도 음성 집합을 바꿔 가며 여러 번 학습해 점수의 분포를 얻고, 그 분포가 무작위 CAV 점수의 분포와 유의하게 다를 때 그 개념을 유의하다고 본다. 무작위 CAV의 점수는 0.5(우연) 근처에 모일 것으로 기대된다.

아래 구현은 이를 크게 간추렸다. 개념 점수 하나와 무작위 점수들을 한데 묶어 0.5와 견주는 단일 표본 t 검정을 쓰므로, 무작위 점수가 많을수록 이 검정은 사실상 「무작위 CAV의 점수가 0.5에서 벗어났는가」를 묻게 된다. 논문의 두 표본 검정과 같은 것이 아니다.

---

## 2. PyTorch 구현

```python
import torch
import numpy as np
from scipy.stats import ttest_1samp

class TCAV:
    """개념 살아남 벡터로 평가."""

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
        """개념과 클래스에 대한 TCAV 점수를 셈한다."""
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
        통계로 뜻있음을 평가하며 TCAV 점수를 셈한다.

        Args:
            class_inputs: 겨눈 클래스의 들임
            concept_examples: 그 개념의 보기
            random_example_sets: 영 가설에 쓸 아무 보기 모둠의 목록
            target_class: 따질 클래스
            alpha: 뜻있음 임계
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

---

## 3. 계량 금융 응용

TCAV로 금융 모델이 경제적으로 의미 있는 개념을 배웠는지 검정할 수 있다.

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
            target_class=1  # 양수 돌아옴 클래스
        )

        sig = "***" if result['significant'] else "뜻없음"
        print(f"{concept_name:20s}: TCAV={result['tcav_score']:.3f} "
              f"p={result['p_value']:.4f} {sig}")
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
레이어 $l$의 활성화가 2차원 $h = (h_1, h_2)$이고 CAV가 $v_l^k = (1, 1)/\sqrt{2}$, 클래스 점수가 $f_c(h) = h_1^2 - h_2$라 하자. 클래스 $c$의 입력 다섯 개의 활성화에서 $h_1$이 각각 $0.2, 0.4, 0.6, 0.8, 1.0$일 때 TCAV 점수를 계산하여라. 이 점수가 말해 주지 않는 것은 무엇인가?

</div>

??? success "풀이"
    $\nabla_h f_c = (2h_1, -1)$이므로 $S_{c,k,l} = (2h_1 - 1)/\sqrt{2}$이고, $h_1 > 0.5$일 때만 양수다. 다섯 입력 가운데 $0.6, 0.8, 1.0$ 세 개가 양수이므로 $\text{TCAV}_{c,k,l} = 3/5 = 0.6$이다. 이 점수는 민감도의 부호만 세고 크기는 버린다. $h_1 = 0.6$의 민감도 $0.14$와 $h_1 = 1.0$의 민감도 $0.71$이 똑같이 한 표로 세어지고, 음수 쪽 입력이 아무리 크게 음수여도 점수에는 한 표로만 들어간다. 그래서 TCAV 점수가 높다는 것은 「많은 입력에서 개념이 점수를 올리는 쪽」이라는 뜻이지 「개념이 점수를 크게 좌우한다」는 뜻이 아니다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
이 절의 설명 방법이 완전성 공리를 만족하는지, 곧 어떤 베이스라인 $x_0$에 대해 모든 피처 기여도의 합이 $f(x) - f(x_0)$과 같은지 증명하거나 반례를 들어라.

</div>

??? success "풀이"
    완전성 공리(섀플리 값 이론에서는 효율성이라고도 한다)는 기여도의 합이 입력에서의 모델 출력과 베이스라인에서의 출력의 차이와 같다는 것이다. 이 방법이 완전성을 만족하는지는 그 구성에 달렸다. 그래디언트 방법은 완전성을 만족하지 못한다(그래디언트는 한 점의 국소적인 양이고 경로를 따라 적분한 것이 아니다). Integrated Gradients는 구성 자체로 완전성을 만족한다(경로를 따라 미적분의 기본정리를 쓴다). SHAP 값은 섀플리 공리로 효율성을 만족한다. 완전성을 어기는 방법은 기여도를 너무 크거나 작게 매길 수 있어, 기여도 전체를 전역 설명으로 믿기 어렵게 만든다. TCAV 점수는 입력 피처에 기여도를 나누는 방법이 아니라 개념 하나에 대한 비율이므로 완전성 공리의 대상이 아니다. 여러 개념의 TCAV 점수를 더해도 아무 양과 같아지지 않는다. $\square$

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

TCAV는 뉴럴 네트워크에서 개념의 중요도를 엄밀하고 정량적으로 평가하는 틀을 준다. 통계적 검정을 거치므로 실제로 의미 있는 개념만 가려내며, 설명의 품질을 검증해야 하는 규제 환경에 알맞다.

**살펴볼 거리**

1. Kim, B., et al. (2018). "Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors." *ICML*.

2. Ghorbani, A., Wexler, J., Zou, J., & Kim, B. (2019). "Towards Automatic Concept-based Explanations." *NeurIPS*.
