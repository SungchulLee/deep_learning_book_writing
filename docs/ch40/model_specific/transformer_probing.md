# 트랜스포머 프로빙

**프로빙 분류기**는 동결된(frozen) 활성화 위에 단순한(흔히 선형) 분류기를 트레인해, 트랜스포머의 잠재 표현에 어떤 정보가 담겼는지 알아낸다. **BertViz** 같은 시각화 도구와 함께 쓰면 트랜스포머가 레이어마다 무엇을 배우는지 폭넓게 그려 볼 수 있다.

---

## 1. 프로빙 방법

### 선형 프로브

레이어 $l$의 동결된 표현 $h_l(x)$가 있을 때, 언어적 속성이나 과제별 속성 $y$를 예측하는 선형 분류기를 트레인한다.

$$
\hat{y} = \sigma(W h_l(x) + b)
$$

정확도가 높으면 레이어 $l$이 속성 $y$에 대한 정보를 담고 있다는 뜻이다. 프로브가 단순하므로, 그 정보를 프로브가 스스로 계산해 낸 것이 아니라 표현 안에 이미 있다고 볼 근거가 된다(완전한 보장은 아니다).

### 프로브가 드러내는 것

레이어 번호는 12레이어짜리 BERT-base를 기준으로 한 대략의 경향이다(Tenney 등, 2019).

| 레이어 | 흔히 담기는 것 |
|-------|-------------------|
| 0(임베딩) | 단어의 속성, 위치 |
| 1~3 | 품사 레이블, 단어 형태 |
| 4~8 | 문장 구조, 의존 관계 |
| 9~12 | 의미, 상호참조(coreference), 감성 |

### 구현

```python
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class TransformerProbe:
    """트랜스포머 표현을 프로빙하는 분류기."""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def extract_representations(self, texts, layer):
        """지정한 레이어의 은닉 상태를 뽑아낸다."""
        self.model.eval()
        representations = []

        for text in texts:
            inputs = self.tokenizer(
                text, return_tensors='pt',
                padding=True, truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)

            hidden = outputs.hidden_states[layer]
            cls_rep = hidden[0, 0].cpu().numpy()  # [CLS] 토큰
            representations.append(cls_rep)

        return np.array(representations)

    def probe_layer(self, texts, labels, layer, test_size=0.2):
        """지정한 레이어에서 프로브를 트레인하고 평가한다."""
        from sklearn.model_selection import train_test_split

        reps = self.extract_representations(texts, layer)

        X_train, X_test, y_train, y_test = train_test_split(
            reps, labels, test_size=test_size, random_state=42
        )

        probe = LogisticRegression(max_iter=1000)
        probe.fit(X_train, y_train)

        accuracy = probe.score(X_test, y_test)
        return accuracy

    def probe_all_layers(self, texts, labels):
        """정보가 어디에 담겼는지 보려고 모든 레이어를 프로빙한다."""
        n_layers = self.model.config.num_hidden_layers + 1

        results = {}
        for layer in range(n_layers):
            acc = self.probe_layer(texts, labels, layer)
            results[layer] = acc
            print(f"레이어 {layer:2d}: 정확도 = {acc:.3f}")

        return results
```

---

## 2. BertViz와 함께 쓰기

BertViz는 어텐션 패턴을 인터랙티브하게 시각화해 준다.

```python
from bertviz import head_view, model_view

def visualize_with_bertviz(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    # 인터랙티브 시각화
    head_view(outputs.attentions, tokens)
    model_view(outputs.attentions, tokens)
```

---

## 3. 금융 개념 프로빙

```python
def probe_financial_model(model, tokenizer, device):
    """금융 언어 모델이 도메인 지식을 지녔는지 프로빙한다."""

    probe = TransformerProbe(model, tokenizer, device)

    # 평가: 모델이 감성을 담고 있는가?
    texts = financial_texts  # 금융 뉴스 목록
    sentiment_labels = [0, 1, 1, 0, ...]  # 약세/강세

    print("레이어마다의 감성 정보:")
    results = probe.probe_all_layers(texts, sentiment_labels)

    best_layer = max(results, key=results.get)
    print(f"감성 프로빙에 가장 좋은 레이어: {best_layer} (정확도={results[best_layer]:.3f})")
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
XOR를 계산하는 ReLU 네트워크의 은닉 레이어가 $h(x) = (\text{ReLU}(x_1 - x_2), \text{ReLU}(x_2 - x_1))$이라 하자. 네 입력 $\{0,1\}^2$에 대해 속성 「$x_1 \ne x_2$」를 맞히는 선형 프로브를 입력 레이어($h_0 = x$)와 은닉 레이어 $h$ 위에 각각 트레인하면 정확도는 최대 얼마인가? $x = [1, 1]$은 각 레이어에서 어디에 놓이는가?

</div>

??? success "풀이"
    입력 레이어에서는 $\{(0,0), (1,1)\}$과 $\{(0,1), (1,0)\}$이 직선 하나로 갈리지 않으므로, 선형 프로브의 정확도는 네 점 가운데 셋을 맞히는 75%가 최대다. 은닉 레이어에서는 $(0,0)$과 $(1,1)$이 둘 다 $h = (0, 0)$으로, $(1,0)$과 $(0,1)$이 각각 $(1,0)$과 $(0,1)$로 간다. 속성이 $h_1 + h_2 > 0.5$로 선형 분리되므로 정확도는 100%다. $x = [1, 1]$은 입력 레이어에서는 $[0, 0]$과 떨어져 있지만 은닉 레이어에서는 그것과 같은 점이 된다. 정보는 입력 레이어에도 다 있었는데 선형 프로브가 꺼내지 못했을 뿐이다. 그래서 프로브 정확도가 오른 것은 「레이어가 그 정보를 새로 만들었다」가 아니라 「선형으로 읽을 수 있게 바꾸었다」는 뜻이다. 프로브를 MLP로 키우면 입력 레이어에서도 100%가 나오므로, 프로브의 단순함이 결론을 좌우한다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
이 절의 설명 방법이 완전성 공리를 만족하는지, 곧 어떤 베이스라인 $x_0$에 대해 모든 피처 기여도의 합이 $f(x) - f(x_0)$과 같은지 증명하거나 반례를 들어라.

</div>

??? success "풀이"
    완전성 공리(섀플리 값 이론에서는 효율성이라고도 한다)는 기여도의 합이 입력에서의 모델 출력과 베이스라인에서의 출력의 차이와 같다는 것이다. 이 방법이 완전성을 만족하는지는 그 구성에 달렸다. 그래디언트 방법은 완전성을 만족하지 못한다(그래디언트는 한 점의 국소적인 양이고 경로를 따라 적분한 것이 아니다). Integrated Gradients는 구성 자체로 완전성을 만족한다(경로를 따라 미적분의 기본정리를 쓴다). SHAP 값은 섀플리 공리로 효율성을 만족한다. 완전성을 어기는 방법은 기여도를 너무 크거나 작게 매길 수 있어, 기여도 전체를 전역 설명으로 믿기 어렵게 만든다. 프로빙은 피처마다 기여도를 매기지 않고 레이어마다 정확도 하나를 내놓으므로 완전성 공리의 대상이 아니다. 어텐션 가중치도 행마다 합이 1일 뿐 $f(x) - f(x_0)$과는 아무 관계가 없다. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 방법이 내놓는 설명이 얼마나 믿을 만한지 평가하는 실험을 설계하라. 방법이 중요하다고 짚은 피처가 실제로 모델에 중요한지를 삽입 곡선과 삭제 곡선으로 재어라.

</div>

??? success "풀이"
    절차는 이렇다. (1) 시험 이미지마다 피처 기여도를 계산한다. (2) 삭제: 기여도가 큰 차례로 피처를 하나씩 가리며 모델의 확신도가 떨어지는 모습을 기록한다. 믿을 만한 설명이면 확신도가 빠르게 떨어진다. (3) 삽입: 빈 베이스라인에서 시작해 기여도가 큰 차례로 피처를 하나씩 드러내며 확신도가 오르는 모습을 기록한다. 믿을 만한 설명이면 확신도가 빠르게 오른다. (4) 두 곡선 아래 넓이를 계산한다. (5) 무작위 순서(베이스라인)와 다른 방법과 비교한다. 믿을 만한 방법이면 삭제 곡선의 넓이가 작고 삽입 곡선의 넓이가 커야 한다. 통계적으로 믿을 만하려면 시험 표본 1000개 넘게 되풀이한다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
이 설명 방법을 신용 부도를 예측하는 금융 모델에 어떻게 적용할 수 있는지 논하라. 설명이 채워야 할 규제 요건은 무엇인가?

</div>

??? success "풀이"
    신용 모델에는 규제(ECOA, GDPR 22조)가 불리한 결정마다 그 사람에게 맞춘 설명을 요구한다. 방법은 다음을 내놓아야 한다. (1) 거절에 가장 크게 기여한 요인(불리한 처분 사유). (2) 일관된 설명(비슷한 신청자는 비슷한 설명을 받는다). (3) 실행 가능한 설명(신청자가 무엇을 바꾸어야 하는지 안다). 이 절의 설명 방법으로 피처 중요도를 짚을 수 있으나, 강건성(입력이 조금 바뀌었다고 설명이 확 달라지면 안 된다)과 충실성(중요한 피처를 없애면 예측이 바뀌어야 한다)을 평가해야 한다. 보호 속성은 대리 차별이 드러나지 않도록 조심히 다루어야 한다. $\square$

## 정리하며

프로빙 분류기는 트랜스포머가 레이어마다 어떤 정보를 담는지 드러내어, 어텐션 시각화를 보완하는 표현 수준의 이해를 준다. BertViz의 인터랙티브 시각화와 함께 쓰면 트랜스포머를 해석하는 탄탄한 도구가 된다.

**살펴볼 거리**

1. Belinkov, Y. (2022). "Probing Classifiers: Promises, Shortcomings, and Advances." *Computational Linguistics*.

2. Vig, J. (2019). "A Multiscale Visualization of Attention in the Transformer Model." *ACL Demo*.

3. Tenney, I., et al. (2019). "BERT Rediscovers the Classical NLP Pipeline." *ACL*.

4. Hewitt, J., & Manning, C. D. (2019). "A Structural Probe for Finding Syntax in Word Representations." *NAACL*.
