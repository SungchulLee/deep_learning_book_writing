# 변환기 더듬기

**더듬는 가름개**는 얼려 둔 살아남 위에 단순한(흔히 선형) 가름개를 익혀, 변환기의 숨은 나타냄에 어떤 소식이 담겼는지 캐낸다. **BertViz** 같은 그림 그리기 연장과 아우르면 변환기가 켜마다 무엇을 배우는지 두루 그릴 수 있다.

---

## 1. 더듬는 길

### 선형 더듬개

켜 $l$의 얼려 둔 나타냄 $h_l(x)$이 있을 때, 말결이나 일감에 매인 됨됨이 $y$을 두고 선형 가름개를 익힌다.

$$
\hat{y} = \sigma(W h_l(x) + b)
$$

맞음률이 높으면 켜 $l$이 됨됨이 $y$에 대한 소식을 담고 있다는 뜻이다. 더듬개가 단순하므로 그 소식이 더듬개가 지어낸 것이 아니라 나타냄 안에 있다는 것이 보장된다.

### 더듬개가 드러내는 것

| 켜 | 흔히 담기는 것 |
|-------|-------------------|
| 0(쏘아 넣기) | 낱말의 됨됨이, 자리 |
| 1~3 | 품사 이름표, 낱말 꼴 |
| 4~8 | 월 얼개, 매인 얽힘 |
| 9~12 | 뜻, 같은 것 가리키기, 느낌 |

### 짜보기

```python
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class TransformerProbe:
    """변환기 나타냄을 더듬는 가름개."""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def extract_representations(self, texts, layer):
        """정한 켜의 숨은 상태를 뽑아낸다."""
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
            cls_rep = hidden[0, 0].cpu().numpy()  # [CLS] 낱말
            representations.append(cls_rep)

        return np.array(representations)

    def probe_layer(self, texts, labels, layer, test_size=0.2):
        """정한 켜에서 더듬개를 익히고 따진다."""
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
        """소식이 어디에 담겼는지 보려고 모든 켜를 더듬는다."""
        n_layers = self.model.config.num_hidden_layers + 1

        results = {}
        for layer in range(n_layers):
            acc = self.probe_layer(texts, labels, layer)
            results[layer] = acc
            print(f"켜 {layer:2d}: 맞음률 = {acc:.3f}")

        return results
```

---

## 2. BertViz과 아우르기

BertViz은 눈길 결을 주고받으며 그려 준다.

```python
from bertviz import head_view, model_view

def visualize_with_bertviz(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    # 주고받는 그림
    head_view(outputs.attentions, tokens)
    model_view(outputs.attentions, tokens)
```

---

## 3. 금융 개념 더듬기

```python
def probe_financial_model(model, tokenizer, device):
    """금융 글 모형이 그 밭의 앎을 지녔는지 더듬는다."""

    probe = TransformerProbe(model, tokenizer, device)

    # 따짐: 모형이 느낌을 담고 있는가?
    texts = financial_texts  # 금융 소식 목록
    sentiment_labels = [0, 1, 1, 0, ...]  # 내림세/오름세

    print("켜마다의 느낌 담김:")
    results = probe.probe_all_layers(texts, sentiment_labels)

    best_layer = max(results, key=results.get)
    print(f"느낌에 가장 나은 켜: {best_layer} (맞음률={results[best_layer]:.3f})")
```

---

## 연습문제

**연습문제 1.**
이 마디에서 밝힌 풀이 방법을, XOR 들임을 가르는 ReLU 살림의 두 켜 신경 그물에 걸어라. 들임 $x = [1, 1]$에 대한 풀이를 셈하여라.

??? success "연습문제 1 풀이"
    짐이 $W_1, b_1, W_2, b_2$인 익힌 XOR 그물에서 내놓기는 $f(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$이다. 풀이 방법은 들임 결마다 몫을 내놓는다. $x = [1, 1]$(갈래 0)이면 두 결 모두 음수 가름에 이바지한다. 몫 값은 방법마다 다르다. 기울기 바탕 방법은 $\partial f / \partial x_i$을 셈하고, 흔들어 보는 방법은 결을 가렸을 때 내놓기가 얼마나 바뀌는지 잰다. XOR 문제는 판단 금이 선형이 아니므로 선형 풀이 방법이 그르칠 수 있음을 보여 준다. $\square$

---

**연습문제 2.**
이 마디의 풀이 방법이 온전함 공리를 채우는지, 곧 어떤 밑금 $x_0$에 대해 모든 결 몫의 합이 $f(x) - f(x_0)$과 같은지 증명하거나 뒤집어라.

??? success "연습문제 2 풀이"
    온전함 공리(섀플리 값 이론에서는 효율이라고도 한다)는 몫의 합이 들임에서의 모형 내놓기와 밑금에서의 내놓기의 차이와 같다는 것이다. 이 방법이 온전함을 채우는지는 그 세움새에 달렸다. 기울기 방법은 온전함을 채우지 못한다(기울기는 그 자리의 것이고 길을 따라 쌓은 것이 아니다). 쌓은 기울기는 세움새 자체로 온전함을 채운다(길을 따라 미적분의 밑정리를 쓴다). SHAP 값은 섀플리 공리로 효율을 채운다. 온전함을 어기는 방법은 몫을 너무 많거나 적게 매길 수 있어, 온 몫을 온 세상 풀이로 믿기 어렵게 만든다. $\square$

---

**연습문제 3.**
이 방법이 내놓는 풀이가 얼마나 미더운지 따지는 시험을 꾸며라. 짚어 준 결이 참으로 모형에 중요한지를 넣기와 빼기 곡선으로 재어라.

??? success "연습문제 3 풀이"
    절차는 이렇다. (1) 시험 그림마다 결 몫을 셈한다. (2) 빼기: 몫이 큰 차례로 결을 하나씩 가리며 모형의 자신함이 떨어지는 모습을 적는다. 미더운 풀이면 자신함이 빠르게 떨어진다. (3) 넣기: 빈 밑금에서 시작해 몫이 큰 차례로 결을 하나씩 드러내며 자신함이 오르는 모습을 적는다. 미더운 풀이면 자신함이 빠르게 오른다. (4) 두 곡선의 아래 넓이를 셈한다. (5) 아무렇게나 매긴 차례(밑금)와 다른 방법에 견준다. 미더운 방법이면 빼기 넓이가 작고 넣기 넓이가 커야 한다. 통계로 미더우려면 시험 표본 1000개 넘게 되풀이한다. $\square$

---

**연습문제 4.**
이 풀이 방법을 신용 부도를 미루어 보는 금융 모형에 어떻게 걸 수 있는지 다루어라. 풀이가 채워야 할 규정 요건은 무엇인가?

??? success "연습문제 4 풀이"
    신용 모형에는 규정(ECOA, GDPR 22조)이 불리한 판단마다 그 사람에게 맞춘 풀이를 바란다. 방법은 다음을 내놓아야 한다. (1) 물리침에 가장 크게 이바지한 인자(불리한 처분 까닭). (2) 한결같은 풀이(비슷한 신청자는 비슷한 풀이를 받는다). (3) 손에 잡히는 풀이(신청자가 무엇을 바꾸어야 하는지 안다). 이 마디의 풀이 방법으로 결의 중요함을 짚을 수 있으나, 든든함(들임이 조금 바뀌었다고 풀이가 확 달라지면 안 된다)과 옳음(중요한 결을 없애면 미루어 봄이 바뀌어야 한다)을 따져 보아야 한다. 지켜야 할 됨됨이는 대리 차별이 드러나지 않도록 조심히 다루어야 한다. $\square$

## 정리하며

더듬는 가름개는 변환기가 켜마다 어떤 소식을 담는지 드러내어, 눈길 그림 그리기를 채워 주는 얼개 낱의 이해를 준다. BertViz으로 주고받으며 살피는 것과 아우르면 변환기 풀이하기의 든든한 연장이 된다.

**살펴볼 거리**

1. Belinkov, Y. (2022). "Probing Classifiers: Promises, Shortcomings, and Advances." *Computational Linguistics*.

2. Vig, J. (2019). "A Multiscale Visualization of Attention in the Transformer Model." *ACL Demo*.

3. Tenney, I., et al. (2019). "BERT Rediscovers the Classical NLP Pipeline." *ACL*.

4. Hewitt, J., & Manning, C. D. (2019). "A Structural Probe for Finding Syntax in Word Representations." *NAACL*.
