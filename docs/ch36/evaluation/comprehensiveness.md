# 두루 갖춤 따지기
## 들머리

**두루 갖춤**은 풀이가 중요한 결을 몇몇이 아니라 모두 담았는지를 잰다. 그 짝인 **넉넉함**은 짚어 준 결만으로 그 미루어 봄을 되살릴 수 있는지 따진다.

## 자

### 두루 갖춤 점수

앞선 $k$개의 결을 없애고 미루어 봄이 얼마나 바뀌는지 잰다.

$$
\text{두루 갖춤}(E, x) = f(x) - f(x_{\setminus E})
$$

클수록 그 풀이가 중요한 결을 담았다는 뜻이다.

### 넉넉함 점수

앞선 $k$개의 결만 남기고 미루어 봄이 얼마나 지켜지는지 잰다.

$$
\text{넉넉함}(E, x) = f(x) - f(x_E)
$$

작을수록 짚어 준 결만으로 넉넉하다는 뜻이다.

### 짜보기

```python
import torch
import numpy as np

def comprehensiveness_sufficiency(
    model, input_tensor, attribution, target_class,
    k_values=[0.1, 0.2, 0.3, 0.5]
):
    """문턱을 달리하며 두루 갖춤과 넉넉함을 셈한다."""
    model.eval()

    with torch.no_grad():
        base_score = torch.softmax(model(input_tensor), dim=1)[0, target_class].item()

    attr_flat = attribution.flatten()
    sorted_idx = np.argsort(np.abs(attr_flat))[::-1]
    n_features = len(attr_flat)

    results = {}
    for k in k_values:
        n_top = int(k * n_features)
        top_indices = sorted_idx[:n_top]

        # 두루 갖춤: 앞선 k개를 없앤다
        removed = input_tensor.clone().flatten()
        removed[top_indices] = 0
        with torch.no_grad():
            removed_score = torch.softmax(model(removed.reshape(input_tensor.shape)), dim=1)[0, target_class].item()

        # 넉넉함: 앞선 k개만 남긴다
        kept = torch.zeros_like(input_tensor).flatten()
        kept[top_indices] = input_tensor.flatten()[top_indices]
        with torch.no_grad():
            kept_score = torch.softmax(model(kept.reshape(input_tensor.shape)), dim=1)[0, target_class].item()

        results[k] = {
            'comprehensiveness': base_score - removed_score,
            'sufficiency': base_score - kept_score
        }

    return results
```

## 읽는 법

좋은 풀이라면 **두루 갖추면서**(두루 갖춤 점수가 크고) **넉넉해야**(넉넉함 점수가 작아야) 한다. 이 두 자는 서로 채워 준다. 두루 갖춤만 보면 늘 모든 결을 고르는 꼼수에 넘어갈 수 있다.

## 간추림

두루 갖춤과 넉넉함은 풀이가 걸린 결을 모두 담았는지, 그리고 그 결만으로 미루어 봄을 되살릴 수 있는지를 수로 잰다.

## 살펴볼 거리

1. DeYoung, J., et al. (2020). "ERASER: A Benchmark to Evaluate Rationalized NLP Models." *ACL*.
2. Carton, S., Rathore, A., & Tan, C. (2020). "Evaluating and Characterizing Human Rationales." *EMNLP*.

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
