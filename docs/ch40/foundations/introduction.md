# 맞섬에 든든하기 들머리

---

## 1. 맞서는 보기를 찾아내다

신경 그물이 공들여 지은 들임 흔듦에 무르다는 것을 찾아낸 일은 요즘 깊은 배움에서 가장 크게 걸린 일 가운데 하나다. 세게디 등(2014)은 알아챌 수 없는 흔듦만으로 가장 앞선 그림 가름개가 크게 어그러질 수 있음을 처음으로 보였고, 맞겨루기 기계 배움이라는 밭이 통째로 열렸다.

**맞서는 보기**은 작고 흔히 알아챌 수 없는 흔듦으로 고친 들임으로, 익힌 모형이 크게 자신하며 틀린 날임을 내게 만든다. 꼴로 적으면, 가름개 $f: \mathcal{X} \to \mathcal{Y}$과 참 이름표가 $y$인 들임 $\mathbf{x}$에 대해 맞서는 보기 $\mathbf{x}'$은 다음을 채운다.

$$
f(\mathbf{x}') \neq y \quad \text{and} \quad d(\mathbf{x}', \mathbf{x}) \leq \varepsilon
$$

여기서 $d(\cdot, \cdot)$은 거리 자이고 $\varepsilon$은 작은 흔듦 예산이다.

---

## 2. 맞서는 보기는 왜 있는가

서로 채워 주는 몇 가지 짐작이 이 일을 밝혀 주며, 저마다 신경 그물의 꼴과 배움의 움직임에서 다른 결을 비춘다.

### 선형 짐작

굿펠로 등(2015)은 맞섬에 무름이 차수 높은 밭에서 신경 그물이 **그 자리에서 선형으로** 움직이는 데서 온다고 내놓았다. 선형 모형 $f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$을 보자. 흔듦 $\boldsymbol{\delta}$이 날임을 바꾸는 정도는

$$
f(\mathbf{x} + \boldsymbol{\delta}) - f(\mathbf{x}) = \mathbf{w}^\top \boldsymbol{\delta}
$$

$\ell_\infty$ 옭아맴 $\|\boldsymbol{\delta}\|_\infty \leq \varepsilon$ 아래에서 이 바뀜을 가장 크게 하려면 자리마다 흔듦을 이렇게 둔다.

$$
\delta_i^* = \varepsilon \cdot \text{sign}(w_i)
$$

그러면 가장 큰 바뀜은

$$
\mathbf{w}^\top \boldsymbol{\delta}^* = \varepsilon \|\mathbf{w}\|_1
$$

차수가 높으면(이미지넷에서는 $d = 3 \times 224 \times 224 \approx 150{,}000$) 아주 작은 $\varepsilon$으로도 $\varepsilon \|\mathbf{w}\|_1$이 커진다. 가까운 이웃에서 거의 선형으로 움직이는 신경 그물은 이 무름을 물려받는다. 일차 테일러 펼침이 이를 또렷이 보여 준다.

$$
\mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y) \approx \mathcal{L}(f_\theta(\mathbf{x}), y) + \boldsymbol{\delta}^\top \nabla_\mathbf{x} \mathcal{L}(f_\theta(\mathbf{x}), y)
$$

### 차수 높은 밭의 꼴

차수 높은 밭의 꼴은 언뜻 어긋나 보이는 결을 여럿 낳으며, 이것이 맞서는 보기를 쉽게 만든다.

- **껍질에 몰림**: 차수 높은 공은 부피의 거의가 껍질 가까이 몰려, 아무렇게나 잡은 점은 거의 언제나 감싸는 자리의 금 가까이 있다
- **판단의 금이 가까움**: 차수가 높으면 판단의 금은 감싸는 부피에 견주어 겉넓이가 엄청나므로, 옳게 가른 점도 거의가 금 가까이 있다
- **거리가 몰림**: 아무렇게나 잡은 점끼리의 거리가 평균 언저리로 몰려 "가깝다"와 "멀다"의 가름이 흐려진다

이 꼴의 결들은, 차수가 낮으면 하잘것없어 보이는 작은 흔듦이 결 밭에서는 꽤 먼 거리를 지나 판단의 금을 넘을 수 있음을 뜻한다.

### 든든하지 않은 결

일리아스 등(2019)은 맞섬에 무름이 모형이 들임의 결을 쓰는 방식에서 오는 것임을 보이며 또 다른 눈을 주었다. 자료에는 두 갈래의 결이 있다.

- **든든한 결**($\rho$-든든): 이름표와 얽히고 마디 있는 흔듦 아래에서도 미루어 보는 힘을 잃지 않는 결. 꼴로 적으면, $\mathbb{E}[y \cdot \phi(\mathbf{x})] \geq \gamma$이고 $\mathbb{E}[\inf_{\|\boldsymbol{\delta}\| \leq \rho} y \cdot \phi(\mathbf{x} + \boldsymbol{\delta})] \geq \gamma$이면 결 $\phi$은 $(\gamma, \rho)$-든든하다.

- **든든하지 않은 결**: 여느 분포에서는 이름표와 얽히지만 맞서는 흔듦에 아주 예민한 결.

여느 익힘은 든든한 결과 든든하지 않은 결을 **함께** 배워 맞음을 가장 크게 한다. 든든하지 않은 결도 참으로 미루어 보는 힘이 있으므로(잡음이 아니다) 없애면 여느 맞음이 떨어진다. 이것이 든든함과 맞음의 맞바꿈을 밝혀 준다.

---

## 3. 든든함과 맞음의 맞바꿈

여느 됨됨이와 든든한 됨됨이 사이에는 밑바탕부터 있는 팽팽함이 있다. 두 무릅씀 자를 매기자.

$$
\begin{aligned}
\text{Standard Risk: } R_{\text{std}}(f) &= \mathbb{E}_{(\mathbf{x},y) \sim \mathcal{D}}[\mathbf{1}[f(\mathbf{x}) \neq y]] \\
\text{Robust Risk: } R_{\text{rob}}(f) &= \mathbb{E}_{(\mathbf{x},y) \sim \mathcal{D}}\left[\max_{\|\boldsymbol{\delta}\| \leq \varepsilon} \mathbf{1}[f(\mathbf{x} + \boldsymbol{\delta}) \neq y]\right]
\end{aligned}
$$

> **정리(치프라스 등, 2019):** 어떤 자료 분포에서는 든든한 맞음이 가장 좋은 가름개라면 여느 맞음은 반드시 가장 좋은 것보다 못하다.

쉽게 말하면, 든든한 가름개는 참으로 미루어 보는 신호를 지닌 든든하지 않은 결을 못 본 척해야 한다. 겪어 보면 CIFAR-10 같은 여느 잣대에서 맞섬에 든든하도록 익힐 때 맑은 맞음이 5~15% 떨어지는 것으로 드러난다.

---

## 4. 가장 좋게 하기로 적은 치기

맞서는 치기는 거의가 옭아맨 가장 좋게 하기 문제로 적힌다.

**과녁 없는 치기**(어떻게든 틀리게 가르게 하기):

$$
\boldsymbol{\delta}^* = \arg\max_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y)
$$

**과녁 있는 치기**(고른 미루어 봄 $y_{\text{target}}$을 억지로 내게 하기):

$$
\boldsymbol{\delta}^* = \arg\min_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y_{\text{target}})
$$

막이 문제는 그 짝이다. 가장 나쁜 자리의 잃음을 가장 작게 하는 매개변수 $\theta$을 찾는다.

$$
\min_\theta \mathbb{E}_{(\mathbf{x},y) \sim \mathcal{D}} \left[ \max_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y) \right]
$$

이 가장 작게-가장 크게 꼴이 맞서며 익히기를 떠받치고, 이어지는 마디에서 다룰 치기와 막이의 갈래를 통틀어 이끈다.

---

## 5. 계량 금융과의 걸림

맞섬에 든든하기는 금융 쓰임에서 학문에만 머무는 걱정이 아니다.

- **모형의 온전함**: 기계 배움 신호를 쓰는 거래 얼개는 들임 자료를 맞서서 흔드는 일(속인 저자 자료 줄기 따위)에 든든해야 한다
- **규정 맞추기**: 서비스에 내놓은 금융 모형은 보여 줄 수 있는 든든함 다짐이 있어야 하며, 미쁨 점수 매기기와 무릅씀 따지기에서 더욱 그렇다
- **속임수 알아내기**: 겨루는 이는 속이려는 뜻은 지킨 채 알아내는 얼개를 비껴가는 들임을 부지런히 지어낸다
- **저자 흔들기**: 맞서는 치기를 알면 흔들기에 버티는 살핌 얼개를 꾸미는 데 도움이 된다

이 장에서는 이론의 깨침과 짜기를 저마다 손에 잡히는 금융 쓰임에 이어 붙인다.

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

이 마당은 맞서는 보기를 찾아내다、맞서는 보기는 왜 있는가、든든함과 맞음의 맞바꿈、가장 좋게 하기로 적은 치기을 차례로 짚었다.

**살펴볼 거리**

1. Szegedy, C., et al. (2014). "Intriguing Properties of Neural Networks." ICLR.
2. Goodfellow, I., Shlens, J., & Szegedy, C. (2015). "Explaining and Harnessing Adversarial Examples." ICLR.
3. Ilyas, A., et al. (2019). "Adversarial Examples Are Not Bugs, They Are Features." NeurIPS.
4. Tsipras, D., et al. (2019). "Robustness May Be at Odds with Accuracy." ICLR.
5. Gilmer, J., et al. (2018). "Adversarial Spheres." ICLR Workshop.
