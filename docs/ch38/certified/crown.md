# CROWN: 볼록 눅임에 기댄 밝히기
## 들머리

**CROWN**(장 등, 2018)과 그 갈래는 곧지 않은 살림 함수를 **선형으로 눅여** IBP보다 촘촘한 밝힘 테두리를 준다. IBP은 아래끝과 위끝을 따로 퍼뜨리지만, CROWN은 들임에 매인 테두리를 셈해 알맞은 셈 값으로 훨씬 촘촘한 밝힘을 낸다.

## 수학 밑바탕

### ReLU의 선형 눅임

테두리를 퍼뜨릴 때 어려운 대목은 곧지 않은 살림을 다루는 일이다. 테두리가 $x \in [\underline{x}, \overline{x}]$인 ReLU $z = \max(0, x)$에 CROWN은 **선형 눅임**을 쓴다.

**자리 1:** $\underline{x} \geq 0$(늘 살아 있음): 꼭 $z = x$

**자리 2:** $\overline{x} \leq 0$(늘 죽어 있음): 꼭 $z = 0$

**자리 3:** $\underline{x} < 0 < \overline{x}$(흔들리는 신경 낱자리):

$$
\alpha x \leq z \leq \frac{\overline{x}}{\overline{x} - \underline{x}}(x - \underline{x})
$$

여기서 $\alpha \in [0, 1]$은 아래끝의 기울기 매개변수로, 배우거나 어림으로 정한다.

### 테두리 퍼뜨리기

CROWN은 마지막 켜의 테두리를 들임의 **선형 함수**으로 적는다.

$$
\underline{z}_L = \mathbf{A}^L \mathbf{x} + \mathbf{b}^L_\text{lower}, \quad \overline{z}_L = \mathbf{A}^U \mathbf{x} + \mathbf{b}^U_\text{upper}
$$

켜를 거슬러 대입하면 마지막 테두리가 들임에 선형으로 매이므로, 들임 흔듦 모임에 걸쳐 잘 드는 다듬기를 할 수 있다.

### CROWN-IBP

**CROWN-IBP**은 CROWN의 촘촘함과 IBP의 잘 듦을 아우른다.

$$
\mathcal{L} = \beta \cdot \mathcal{L}_{\text{CROWN}} + (1 - \beta) \cdot \mathcal{L}_{\text{IBP}}
$$

익히는 동안 $\beta$을 1(CROWN, 촘촘하나 느림)에서 0(IBP, 빠름)으로 차츰 낮추어 두 길의 나은 점을 함께 얻는다.

## Auto-LiRPA

**Auto-LiRPA**(쉬 등, 2020)은 어떤 셈 그림에도 선형 눅임에 기댄 흔듦 살피기를 저절로 해 주는 두루 쓰는 틀로, CROWN을 단순한 앞먹임 그물 너머로 넓힌다.

```python
# auto_LiRPA 묶음을 쓴다
# pip install auto_LiRPA
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

# 모형을 감싼다
bounded_model = BoundedModule(model, torch.zeros(1, 3, 32, 32))

# 흔듦을 매긴다
ptb = PerturbationLpNorm(norm=float('inf'), eps=8/255)
bounded_input = BoundedTensor(x, ptb)

# 테두리를 셈한다
lb, ub = bounded_model.compute_bounds(
    x=(bounded_input,), method='CROWN'
)

# 밝히기: k ≠ y에 대해 lb[y] > max(ub[k])인지 살핀다
```

## 밝히는 방법 견주기

| 방법 | 테두리의 촘촘함 | 셈 값 | 크게 늘리기 |
|--------|----------------|-------------------|-------------|
| IBP | 헐거움 | 낮음 | 좋음 |
| CROWN | 촘촘함 | 높음 | 가운데 |
| CROWN-IBP | 가운데 | 가운데 | 좋음 |
| α-CROWN | 가장 촘촘함 | 가장 높음 | 마디 있음 |
| SDP 눅임 | 아주 촘촘함 | 아주 높음 | 작은 그물만 |

## 간추림

CROWN과 그 갈래는 붙박인 밝혀 낸 든든함에서 가장 앞선 것으로, 셈 값을 더 들여 IBP보다 촘촘한 테두리를 준다. CROWN-IBP을 아우르면 밝힐 수 있게 든든한 그물을 익히는 데 참으로 가장 좋은 맞바꿈이 되고, Auto-LiRPA은 이 깨침을 두루 쓰이는 얼개로 넓힌다.

## 살펴볼 거리

1. Zhang, H., et al. (2018). "Efficient Neural Network Robustness Certification with General Activation Functions." NeurIPS.
2. Xu, K., et al. (2020). "Automatic Perturbation Analysis for Scalable Certified Robustness and Beyond." NeurIPS.
3. Zhang, H., et al. (2020). "General Cutting Planes for Bound-Propagation-Based Neural Network Verification." NeurIPS.

## 익힘 문제

**익힘 1.**
선형 가름개 $f(x) = w^T x + b$에서 미루어 본 갈래를 바꾸는 데 드는 가장 작은 $\ell_\infty$ 흔듦을 셈하여라. 이것이 신경 그물의 든든함과 어떻게 이어지는지 밝혀라.

??? success "익힘 1 풀이"
    선형 가름개에서 $\ell_\infty$ 노름으로 잰 판단의 금까지의 거리는 $\frac{|w^T x + b|}{\|w\|_1}$이다. 가장 작은 흔듦은 $\delta^* = \frac{|w^T x + b|}{\|w\|_1} \cdot \text{sign}(w)$이다. 신경 그물에서는 그 자리의 선형 어림 $f(x + \delta) \approx f(x) + \nabla_x f \cdot \delta$이 FGSM(기울기의 부호를 쓴다)이 왜 잘 듣는지를 밝혀 준다. 차수가 높은 모형이 무른 까닭은 $\|w\|_1$은 차수와 함께 커지는데 $|w^T x + b|$은 꼭 그렇지 않아 든든함의 여유가 줄어들기 때문이다. $\square$

---

**익힘 2.**
이 마디에서 다룬 치기나 막이를 CIFAR-10의 ResNet-18 모형에 짜 넣어라. $\epsilon = 8/255$의 PGD-20 치기 아래에서 맑은 맞음과 든든한 맞음을 알려라.

??? success "익힘 2 풀이"
    여느 ResNet-18은 맑은 맞음이 $\sim$93%이지만 PGD-20($\epsilon = 8/255$, 걸음 크기 $2/255$) 아래의 든든한 맞음은 $\sim$0%이다. 이 마디의 방법을 걸면 결과는 재주에 따라 다르다. 맞서며 익히기는 맑은 맞음 $\sim$83%에 든든한 맞음 $\sim$50%이고, 밝혀 낸 막이는 더 낮지만 증명할 수 있는 테두리를 준다. 맞음과 든든함의 맞바꿈은 밑바탕부터 있는 것이라, 든든함을 높이면 맑은 맞음이 흔히 5~15% 든다. 아무렇게나 하는 씨앗 3개의 평균과 잣대 어긋남으로 알려라. $\square$

---

**익힘 3.**
흔듦 공 안에서 갈래별 자료의 밑자리가 서로 겹친다고 볼 때, 모형이 담는 힘을 키우지 않고서는 어떤 막이도 맑은 자료의 높은 맞음과 $\ell_\infty$ 흔듦에 대한 높은 든든함을 함께 이룰 수 없음을 증명하여라.

??? success "익힘 3 풀이"
    두 갈래의 밑자리가 거리 $\epsilon$ 안에서 겹치면(곧 $\|x_1 - x_2\|_\infty \leq 2\epsilon$인 $x_1 \in \text{갈래 1}, x_2 \in \text{갈래 2}$이 있으면), $x_1$과 $x_2$ 둘 다에서 든든한 가름개는 적어도 하나를 틀리게 가를 수밖에 없다(흔듦 공이 겹치기 때문이다). 이것이 맞음과 든든함의 밑바탕 맞바꿈이다. 겹치는 밑자리의 몫이 피할 수 없는 맞음 잃음을 정한다. 여느 그림 분포에서는 $\epsilon = 8/255$에서 겹침이 꽤 있어, 살펴본 10~15%의 맞음 떨어짐을 밝혀 준다. 모형이 담는 힘을 키우면(더 너른 그물) 얽힌 든든한 판단의 금을 더 잘 그려 맞바꿈을 얼마쯤 눅일 수 있다. $\square$

---

**익힘 4.**
금융 기계 배움 얼개(속임수 알아내기나 거래 신호 만들기 따위)에서 맞섬의 든든함이 어떻게 드러나는지 다루어라. 으름 얼개가 보기 다룸과 어떻게 다른가?

??? success "익힘 4 풀이"
    금융에서 겨루는 이는 알아내는 얼개에 맞추어 스스로 움직이는 꾀 많은 무리(속임수꾼, 저자 흔드는 이)다. 보기 다룸과 다른 고갱이는 이렇다. (1) 흔들 수 있는 밭이 돈으로 될 만한 것에 옭매인다(속임수꾼이 제 거래 자취를 통째로 바꿀 수는 없다). (2) 치기가 잇따르며 맞추어 간다(겨루는 이가 얼개의 되받음을 보고 손본다). (3) 헛 맞음과 놓침의 값이 서로 어긋난다(옳은 거래를 막는 것과 속임수를 놓치는 것). (4) $\ell_p$ 노름은 뜻이 없고 밭에 맞는 흔듦 모형이 있어야 한다. 막이는 맞추어 오는 겨루는 이에게도 든든해야 하므로, 알아내는 잣대가 알려지면 비껴갈 수 있는 알아내기 바탕의 길은 많이 걸러진다. $\square$
