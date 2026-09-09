# 밝혀 낸 맞음

**밝혀 낸 맞음**은 주어진 예산 안의 어떤 흔듦에도 모형의 미루어 봄이 옳음을 증명할 수 있는 시험 보기의 몫을 수로 나타낸다. 치기의 세기에 매인 겪은 든든한 맞음과 달리, 밝혀 낸 맞음은 참 든든함의 다짐된 아래끝을 준다.

---

## 1. 꼴로 뜻매김하기

### 반지름 r에서 밝혀 낸 맞음

$$
\text{Certified Acc}(r) = \frac{1}{N} \sum_{i=1}^N \mathbf{1}\left[f(\mathbf{x}_i) = y_i \text{ and } R(\mathbf{x}_i) \geq r\right]
$$

여기서 $R(\mathbf{x}_i)$은 보기 $i$에서 밝혀 낸 반지름이다.

### 다른 자와의 사이

$$
\text{Certified Acc}(r) \leq \text{True Robust Acc}(r) \leq \text{Empirical Robust Acc}(r)
$$

- **밝혀 낸 맞음**은 아래끝이다. 참으로 든든한 미루어 봄도 밝히지 못할 수 있다
- **겪은 든든한 맞음**은 위끝이다. 치기가 가장 좋은 맞서는 보기를 못 찾을 수 있다
- 그 틈이 "밝히기의 틈"을 잰다

---

## 2. 밝혀 낸 맞음 셈하기

### 아무렇게나 매끄럽게 하기에서

```python
import torch
from typing import Dict, List

def compute_certified_accuracy(
    smoother,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    radii: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
    n: int = 10000,
    alpha: float = 0.001
) -> Dict[str, float]:
    """
    여러 반지름에서 밝혀 낸 맞음을 셈한다.
    
    Parameters
    ----------
    smoother : RandomizedSmoothing
        밝힐 수 있는 매끄럽게 한 가름개
    test_images, test_labels : torch.Tensor
        시험 자료 꾸러미
    radii : list[float]
        밝혀 낸 맞음을 셈할 반지름
    n : int
        밝히는 데 쓸 몬테카를로 표본 수
    alpha : float
        믿음 켜
    
    Returns
    -------
    results : 반지름을 밝혀 낸 맞음에 맞춘 사전
    """
    num_examples = len(test_images)
    predictions = []
    certified_radii = []
    
    for i in range(num_examples):
        pred, cert_radius = smoother.certify(
            test_images[i], n=n, alpha=alpha
        )
        predictions.append(pred)
        certified_radii.append(cert_radius)
    
    predictions = torch.tensor(predictions)
    certified_radii = torch.tensor(certified_radii)
    correct = (predictions == test_labels)
    
    results = {'clean_accuracy': correct.float().mean().item()}
    
    for r in radii:
        certified_at_r = correct & (certified_radii >= r)
        results[f'certified_r={r}'] = certified_at_r.float().mean().item()
    
    if correct.any():
        results['avg_radius'] = certified_radii[correct].mean().item()
    else:
        results['avg_radius'] = 0.0
    
    return results
```

### IBP/CROWN에서

```python
def certified_accuracy_ibp(model, test_loader, epsilon, device='cuda'):
    """IBP 테두리로 밝혀 낸 맞음을 셈한다."""
    certified = 0
    correct = 0
    total = 0
    
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        lb, ub = compute_ibp_bounds(model, x, epsilon)
        
        with torch.no_grad():
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
        
        true_lb = lb.gather(1, y.view(-1, 1)).squeeze()
        ub_copy = ub.clone()
        ub_copy.scatter_(1, y.view(-1, 1), float('-inf'))
        max_other_ub = ub_copy.max(dim=1)[0]
        
        is_certified = (true_lb > max_other_ub) & (pred == y)
        certified += is_certified.sum().item()
        total += len(y)
    
    return {
        'clean_accuracy': correct / total,
        'certified_accuracy': certified / total
    }
```

---

## 3. 잣대 재기

### CIFAR-10(L2, 아무렇게나 매끄럽게 하기)

| 방법 | $\sigma$ | $r{=}0.25$에서 밝힘 | $r{=}0.5$에서 밝힘 | $r{=}1.0$에서 밝힘 |
|--------|----------|---------------------|--------------------|--------------------|
| 코언 등 | 0.25 | 60% | 43% | — |
| 살만 등 | 0.25 | 68% | 49% | — |
| 코언 등 | 0.50 | 54% | 41% | 26% |
| 살만 등 | 0.50 | 59% | 44% | 32% |

### CIFAR-10(L-무한, IBP/CROWN)

| 방법 | $\varepsilon$ | 밝혀 낸 맞음 |
|--------|--------------|-------------------|
| IBP | 2/255 | 33% |
| CROWN-IBP | 2/255 | 38% |
| IBP | 8/255 | 7% |
| CROWN-IBP | 8/255 | 12% |

---

## 연습문제

**연습문제 1.**
선형 가름개 $f(x) = w^T x + b$에서 미루어 본 갈래를 바꾸는 데 드는 가장 작은 $\ell_\infty$ 흔듦을 셈하여라. 이것이 신경 그물의 든든함과 어떻게 이어지는지 밝혀라.

??? success "연습문제 1 풀이"
    선형 가름개에서 $\ell_\infty$ 노름으로 잰 판단의 금까지의 거리는 $\frac{|w^T x + b|}{\|w\|_1}$이다. 가장 작은 흔듦은 $\delta^* = \frac{|w^T x + b|}{\|w\|_1} \cdot \text{sign}(w)$이다. 신경 그물에서는 그 자리의 선형 어림 $f(x + \delta) \approx f(x) + \nabla_x f \cdot \delta$이 FGSM(기울기의 부호를 쓴다)이 왜 잘 듣는지를 밝혀 준다. 차수가 높은 모형이 무른 까닭은 $\|w\|_1$은 차수와 함께 커지는데 $|w^T x + b|$은 꼭 그렇지 않아 든든함의 여유가 줄어들기 때문이다. $\square$

---

**연습문제 2.**
이 마디에서 다룬 치기나 막이를 CIFAR-10의 ResNet-18 모형에 짜 넣어라. $\epsilon = 8/255$의 PGD-20 치기 아래에서 맑은 맞음과 든든한 맞음을 알려라.

??? success "연습문제 2 풀이"
    여느 ResNet-18은 맑은 맞음이 $\sim$93%이지만 PGD-20($\epsilon = 8/255$, 걸음 크기 $2/255$) 아래의 든든한 맞음은 $\sim$0%이다. 이 마디의 방법을 걸면 결과는 재주에 따라 다르다. 맞서며 익히기는 맑은 맞음 $\sim$83%에 든든한 맞음 $\sim$50%이고, 밝혀 낸 막이는 더 낮지만 증명할 수 있는 테두리를 준다. 맞음과 든든함의 맞바꿈은 밑바탕부터 있는 것이라, 든든함을 높이면 맑은 맞음이 흔히 5~15% 든다. 아무렇게나 하는 씨앗 3개의 평균과 잣대 어긋남으로 알려라. $\square$

---

**연습문제 3.**
흔듦 공 안에서 갈래별 자료의 밑자리가 서로 겹친다고 볼 때, 모형이 담는 힘을 키우지 않고서는 어떤 막이도 맑은 자료의 높은 맞음과 $\ell_\infty$ 흔듦에 대한 높은 든든함을 함께 이룰 수 없음을 증명하여라.

??? success "연습문제 3 풀이"
    두 갈래의 밑자리가 거리 $\epsilon$ 안에서 겹치면(곧 $\|x_1 - x_2\|_\infty \leq 2\epsilon$인 $x_1 \in \text{갈래 1}, x_2 \in \text{갈래 2}$이 있으면), $x_1$과 $x_2$ 둘 다에서 든든한 가름개는 적어도 하나를 틀리게 가를 수밖에 없다(흔듦 공이 겹치기 때문이다). 이것이 맞음과 든든함의 밑바탕 맞바꿈이다. 겹치는 밑자리의 몫이 피할 수 없는 맞음 잃음을 정한다. 여느 그림 분포에서는 $\epsilon = 8/255$에서 겹침이 꽤 있어, 살펴본 10~15%의 맞음 떨어짐을 밝혀 준다. 모형이 담는 힘을 키우면(더 너른 그물) 얽힌 든든한 판단의 금을 더 잘 그려 맞바꿈을 얼마쯤 눅일 수 있다. $\square$

---

**연습문제 4.**
금융 기계 배움 얼개(속임수 알아내기나 거래 신호 만들기 따위)에서 맞섬의 든든함이 어떻게 드러나는지 다루어라. 으름 얼개가 보기 다룸과 어떻게 다른가?

??? success "연습문제 4 풀이"
    금융에서 겨루는 이는 알아내는 얼개에 맞추어 스스로 움직이는 꾀 많은 무리(속임수꾼, 저자 흔드는 이)다. 보기 다룸과 다른 고갱이는 이렇다. (1) 흔들 수 있는 밭이 돈으로 될 만한 것에 옭매인다(속임수꾼이 제 거래 자취를 통째로 바꿀 수는 없다). (2) 치기가 잇따르며 맞추어 간다(겨루는 이가 얼개의 되받음을 보고 손본다). (3) 헛 맞음과 놓침의 값이 서로 어긋난다(옳은 거래를 막는 것과 속임수를 놓치는 것). (4) $\ell_p$ 노름은 뜻이 없고 밭에 맞는 흔듦 모형이 있어야 한다. 막이는 맞추어 오는 겨루는 이에게도 든든해야 하므로, 알아내는 잣대가 알려지면 비껴갈 수 있는 알아내기 바탕의 길은 많이 걸러진다. $\square$

## 정리하며

| 자 | 다짐 | 값 | 촘촘함 |
|--------|-----------|------|-----------|
| 겪은 든든한 맞음 | 없음 | 낮음~가운데 | 위끝 |
| 밝혀 낸 맞음(아무렇게나 매끄럽게) | 낌새의 것 | 높음 | 가운데 |
| 밝혀 낸 맞음(IBP) | 붙박인 것 | 낮음 | 헐거움 |
| 밝혀 낸 맞음(CROWN) | 붙박인 것 | 가운데 | 더 촘촘함 |

밝혀 낸 맞음은 가장 엄밀한 든든함 자로, 흔듦 예산 안의 어떤 치기에도 버티는 다짐을 준다.

**살펴볼 거리**

1. Cohen, J., Rosenfeld, E., & Kolter, Z. (2019). "Certified Adversarial Robustness via Randomized Smoothing." ICML.
2. Gowal, S., et al. (2019). "Scalable Verified Training for Provably Robust Image Classification." ICCV.
3. Salman, H., et al. (2019). "Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers." NeurIPS.
