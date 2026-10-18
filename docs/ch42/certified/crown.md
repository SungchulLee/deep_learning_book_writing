# CROWN: 볼록 완화에 기댄 밝히기

**CROWN**(장 등, 2018)과 그 클래스는 곧지 않은 엑티베이션 함수를 **선형으로 눅여** IBP보다 촘촘한 밝힘 테두리를 준다. IBP은 아래끝과 위끝을 따로 퍼뜨리지만, CROWN은 입력에 매인 테두리를 셈해 알맞은 계산 비용으로 훨씬 촘촘한 밝힘을 낸다.

---

## 1. 수학 기초

### ReLU의 선형 눅임

테두리를 퍼뜨릴 때 어려운 대목은 곧지 않은 활성화를 다루는 일이다. 테두리가 $x \in [\underline{x}, \overline{x}]$인 ReLU $z = \max(0, x)$에 CROWN은 **선형 눅임**을 쓴다.

**자리 1:** $\underline{x} \geq 0$(늘 살아 있음): 꼭 $z = x$

**자리 2:** $\overline{x} \leq 0$(늘 죽어 있음): 꼭 $z = 0$

**자리 3:** $\underline{x} < 0 < \overline{x}$(흔들리는 뉴럴 유닛):

$$
\alpha x \leq z \leq \frac{\overline{x}}{\overline{x} - \underline{x}}(x - \underline{x})
$$

여기서 $\alpha \in [0, 1]$은 아래끝의 그래디언트 파라메터로, 배우거나 어림으로 정한다.

### 테두리 퍼뜨리기

CROWN은 마지막 켜의 테두리를 입력의 **선형 함수**으로 적는다.

$$
\underline{z}_L = \mathbf{A}^L \mathbf{x} + \mathbf{b}^L_\text{lower}, \quad \overline{z}_L = \mathbf{A}^U \mathbf{x} + \mathbf{b}^U_\text{upper}
$$

켜를 거슬러 대입하면 마지막 테두리가 입력에 선형으로 매이므로, 입력 섭동 모임에 걸쳐 잘 드는 다듬기를 할 수 있다.

### CROWN-IBP

**CROWN-IBP**은 CROWN의 촘촘함과 IBP의 잘 듦을 아우른다.

$$
\mathcal{L} = \beta \cdot \mathcal{L}_{\text{CROWN}} + (1 - \beta) \cdot \mathcal{L}_{\text{IBP}}
$$

익히는 동안 $\beta$을 1(CROWN, 촘촘하나 느림)에서 0(IBP, 빠름)으로 차츰 낮추어 두 길의 나은 점을 함께 얻는다.

---

## 2. Auto-LiRPA

**Auto-LiRPA**(쉬 등, 2020)은 어떤 계산 그림에도 선형 눅임에 기댄 섭동 살피기를 저절로 해 주는 두루 쓰는 틀로, CROWN을 단순한 순방향 네트워크 너머로 넓힌다.

```python
# auto_LiRPA 배치를 쓴다
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

---

## 3. 밝히는 방법 비교

| 방법 | 테두리의 촘촘함 | 계산 비용 | 크게 늘리기 |
|--------|----------------|-------------------|-------------|
| IBP | 헐거움 | 낮음 | 좋음 |
| CROWN | 촘촘함 | 높음 | 가운데 |
| CROWN-IBP | 가운데 | 가운데 | 좋음 |
| α-CROWN | 가장 촘촘함 | 가장 높음 | 마디 있음 |
| SDP 눅임 | 아주 촘촘함 | 아주 높음 | 작은 네트워크만 |

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
선형 분류기 $f(x) = w^T x + b$에서 예측한 클래스를 바꾸는 데 드는 가장 작은 $\ell_\infty$ 섭동을 계산하여라. 이것이 뉴럴 넷트웍의 강건성과 어떻게 이어지는지 밝혀라.

</div>

??? success "풀이"
    선형 분류기에서 $\ell_\infty$ 노름으로 잰 결정 경계까지의 거리는 $\frac{|w^T x + b|}{\|w\|_1}$이다. 가장 작은 섭동은 $\delta^* = \frac{|w^T x + b|}{\|w\|_1} \cdot \text{sign}(w)$이다. 뉴럴 넷트웍에서는 그 자리의 선형 어림 $f(x + \delta) \approx f(x) + \nabla_x f \cdot \delta$이 FGSM(그래디언트의 부호를 쓴다)이 왜 잘 듣는지를 밝혀 준다. 차원이 높은 모델이 무른 까닭은 $\|w\|_1$은 차원과 함께 커지는데 $|w^T x + b|$은 꼭 그렇지 않아 강건성의 여유가 줄어들기 때문이다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 절에서 다룬 공격나 방어를 CIFAR-10의 ResNet-18 모델에 짜 넣어라. $\epsilon = 8/255$의 PGD-20 공격 아래에서 클린 정확도와 강건 정확도를 알려라.

</div>

??? success "풀이"
    여느 ResNet-18은 클린 정확도가 $\sim$93%이지만 PGD-20($\epsilon = 8/255$, 스텝 크기 $2/255$) 아래의 강건 정확도는 $\sim$0%이다. 이 절의 방법을 걸면 피처와는 재주에 따라 다르다. 적대적 학습은 클린 정확도 $\sim$83%에 강건 정확도 $\sim$50%이고, 인증 가능한 방어는 더 낮지만 증명할 수 있는 테두리를 준다. 정확도와 강건성의 트레이드오프는 기초부터 있는 것이라, 강건성을 높이면 클린 정확도가 흔히 5~15% 든다. 아무렇게나 하는 시드 3개의 평균과 표준편차로 알려라. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
섭동 공 안에서 클래스별 데이터의 밑자리가 서로 겹친다고 볼 때, 모델이 담는 힘을 키우지 않고서는 어떤 방어도 맑은 데이터의 높은 정확도와 $\ell_\infty$ 섭동에 대한 높은 강건성을 함께 이룰 수 없음을 증명하여라.

</div>

??? success "풀이"
    두 클래스의 밑자리가 거리 $\epsilon$ 안에서 겹치면(곧 $\|x_1 - x_2\|_\infty \leq 2\epsilon$인 $x_1 \in \text{클래스 1}, x_2 \in \text{클래스 2}$이 있으면), $x_1$과 $x_2$ 둘 다에서 강건한 분류기는 적어도 하나를 틀리게 가를 수밖에 없다(섭동 공이 겹치기 때문이다). 이것이 정확도와 강건성의 기초 트레이드오프다. 겹치는 밑자리의 몫이 피할 수 없는 정확도 로스를 정한다. 여느 그림 분포에서는 $\epsilon = 8/255$에서 겹침이 꽤 있어, 살펴본 10~15%의 정확도 떨어짐을 밝혀 준다. 모델이 담는 힘을 키우면(더 너른 네트워크) 얽힌 강건한 결정 경계를 더 잘 그려 트레이드오프를 얼마쯤 눅일 수 있다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
금융 머신러닝 구조(속임수 알아내기나 거래 신호 만들기 따위)에서 맞섬의 강건성이 어떻게 드러나는지 다루어라. 위협 모델이 보기 다룸과 어떻게 다른가?

</div>

??? success "풀이"
    금융에서 겨루는 이는 알아내는 구조에 맞추어 스스로 움직이는 꾀 많은 무리(속임수꾼, 저자 흔드는 이)다. 보기 다룸과 다른 핵심은 이렇다. (1) 흔들 수 있는 밭이 돈으로 될 만한 것에 옭매인다(속임수꾼이 제 거래 자취를 통째로 바꿀 수는 없다). (2) 공격이 잇따르며 맞추어 간다(겨루는 이가 구조의 되받음을 보고 손본다). (3) 헛 정확도와 놓침의 값이 서로 어긋난다(옳은 거래를 막는 것과 속임수를 놓치는 것). (4) $\ell_p$ 노름은 뜻이 없고 밭에 맞는 섭동 모델이 있어야 한다. 방어는 맞추어 오는 겨루는 이에게도 든든해야 하므로, 알아내는 잣대가 알려지면 비껴갈 수 있는 알아내기 기초의 길은 많이 걸러진다. $\square$

## 정리하며

CROWN과 그 클래스는 붙박인 강건성 인증에서 가장 앞선 것으로, 계산 비용을 더 들여 IBP보다 촘촘한 테두리를 준다. CROWN-IBP을 아우르면 밝힐 수 있게 강건한 네트워크를 익히는 데 참으로 가장 좋은 트레이드오프가 되고, Auto-LiRPA은 이 깨침을 두루 쓰이는 구조로 넓힌다.

**살펴볼 거리**

1. Zhang, H., et al. (2018). "Efficient Neural Network Robustness Certification with General Activation Functions." NeurIPS.
2. Xu, K., et al. (2020). "Automatic Perturbation Analysis for Scalable Certified Robustness and Beyond." NeurIPS.
3. Zhang, H., et al. (2020). "General Cutting Planes for Bound-Propagation-Based Neural Network Verification." NeurIPS.
