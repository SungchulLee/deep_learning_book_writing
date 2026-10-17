# 적대적 강건성 입문

---

## 1. 적대적 샘플을 찾아내다

뉴럴 넷트웍이 공들여 지은 입력 섭동에 무르다는 것을 찾아낸 일은 요즘 딥러닝에서 가장 크게 걸린 일 가운데 하나다. 세게디 등(2014)은 알아챌 수 없는 섭동만으로 가장 앞선 이미지 분류기가 크게 어그러질 수 있음을 처음으로 보였고, 적대적 머신러닝이라는 밭이 통째로 열렸다.

**적대적 샘플**은 작고 흔히 알아챌 수 없는 섭동으로 고친 입력으로, 학습된 모델이 크게 자신하며 틀린 출력을 내게 만든다. 꼴로 적으면, 분류기 $f: \mathcal{X} \to \mathcal{Y}$과 참 레이블이 $y$인 입력 $\mathbf{x}$에 대해 적대적 샘플 $\mathbf{x}'$은 다음을 채운다.

$$
f(\mathbf{x}') \neq y \quad \text{and} \quad d(\mathbf{x}', \mathbf{x}) \leq \varepsilon
$$

여기서 $d(\cdot, \cdot)$은 거리 자이고 $\varepsilon$은 작은 섭동 예산이다.

---

## 2. 적대적 샘플은 왜 있는가

서로 채워 주는 몇 가지 짐작이 이 일을 밝혀 주며, 저마다 뉴럴 넷트웍의 꼴과 학습의 액션에서 다른 피처를 비춘다.

### 선형 짐작

굿펠로 등(2015)은 적대적 취약성이 고차원 공간에서 뉴럴 넷트웍이 **그 자리에서 선형으로** 움직이는 데서 온다고 내놓았다. 선형 모델 $f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$을 보자. 섭동 $\boldsymbol{\delta}$이 출력을 바꾸는 정도는

$$
f(\mathbf{x} + \boldsymbol{\delta}) - f(\mathbf{x}) = \mathbf{w}^\top \boldsymbol{\delta}
$$

$\ell_\infty$ 제약 $\|\boldsymbol{\delta}\|_\infty \leq \varepsilon$ 아래에서 이 바뀜을 가장 크게 하려면 자리마다 섭동을 이렇게 둔다.

$$
\delta_i^* = \varepsilon \cdot \text{sign}(w_i)
$$

그러면 가장 큰 바뀜은

$$
\mathbf{w}^\top \boldsymbol{\delta}^* = \varepsilon \|\mathbf{w}\|_1
$$

차원이 높으면(이미지넷에서는 $d = 3 \times 224 \times 224 \approx 150{,}000$) 아주 작은 $\varepsilon$으로도 $\varepsilon \|\mathbf{w}\|_1$이 커진다. 가까운 이웃에서 거의 선형으로 움직이는 뉴럴 넷트웍은 이 취약성을 물려받는다. 일차 테일러 전개가 이를 또렷이 보여 준다.

$$
\mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y) \approx \mathcal{L}(f_\theta(\mathbf{x}), y) + \boldsymbol{\delta}^\top \nabla_\mathbf{x} \mathcal{L}(f_\theta(\mathbf{x}), y)
$$

### 고차원 공간의 꼴

고차원 공간의 꼴은 언뜻 어긋나 보이는 피처를 여럿 낳으며, 이것이 적대적 샘플을 쉽게 만든다.

- **껍질에 몰림**: 차원 높은 공은 부피의 거의가 껍질 가까이 몰려, 아무렇게나 잡은 점은 거의 언제나 감싸는 자리의 금 가까이 있다
- **결정 경계가 가까움**: 차원이 높으면 결정 경계는 감싸는 부피에 비교하여 겉넓이가 엄청나므로, 옳게 가른 점도 거의가 금 가까이 있다
- **거리가 몰림**: 아무렇게나 잡은 점끼리의 거리가 평균 언저리로 몰려 "가깝다"와 "멀다"의 분류가 흐려진다

이 꼴의 피처들은, 차원이 낮으면 하잘것없어 보이는 작은 섭동이 피처 공간에서는 꽤 먼 거리를 지나 결정 경계를 넘을 수 있음을 뜻한다.

### 강건하지 않은 피처

일리아스 등(2019)은 적대적 취약성이 모델이 입력의 피처를 쓰는 방식에서 오는 것임을 보이며 또 다른 눈을 주었다. 데이터에는 두 클래스의 피처가 있다.

- **강건한 피처**($\rho$-든든): 레이블과 얽히고 마디 있는 섭동 아래에서도 예측력을 잃지 않는 피처. 꼴로 적으면, $\mathbb{E}[y \cdot \phi(\mathbf{x})] \geq \gamma$이고 $\mathbb{E}[\inf_{\|\boldsymbol{\delta}\| \leq \rho} y \cdot \phi(\mathbf{x} + \boldsymbol{\delta})] \geq \gamma$이면 피처 $\phi$은 $(\gamma, \rho)$-든든하다.

- **강건하지 않은 피처**: 여느 분포에서는 레이블과 얽히지만 적대적 섭동에 아주 예민한 피처.

여느 학습은 강건한 피처와 강건하지 않은 피처를 **함께** 배워 정확도를 가장 크게 한다. 강건하지 않은 피처도 참으로 예측력이 있으므로(잡음이 아니다) 없애면 표준 정확도가 떨어진다. 이것이 강건성과 정확도의 트레이드오프를 밝혀 준다.

---

## 3. 강건성과 정확도의 트레이드오프

여느 성능과 강건한 성능 사이에는 기초부터 있는 팽팽함이 있다. 두 위험 자를 매기자.

$$
\begin{aligned}
\text{Standard Risk: } R_{\text{std}}(f) &= \mathbb{E}_{(\mathbf{x},y) \sim \mathcal{D}}[\mathbf{1}[f(\mathbf{x}) \neq y]] \\
\text{Robust Risk: } R_{\text{rob}}(f) &= \mathbb{E}_{(\mathbf{x},y) \sim \mathcal{D}}\left[\max_{\|\boldsymbol{\delta}\| \leq \varepsilon} \mathbf{1}[f(\mathbf{x} + \boldsymbol{\delta}) \neq y]\right]
\end{aligned}
$$

> **정리(치프라스 등, 2019):** 어떤 데이터 분포에서는 강건 정확도가 가장 좋은 분류기라면 표준 정확도는 반드시 가장 좋은 것보다 못하다.

쉽게 말하면, 강건한 분류기는 참으로 예측하는 신호를 지닌 강건하지 않은 피처를 못 본 척해야 한다. 겪어 보면 CIFAR-10 같은 여느 잣대에서 맞섬에 든든하도록 익힐 때 클린 정확도가 5~15% 떨어지는 것으로 드러난다.

---

## 4. 최적화로 적은 공격

적대적 공격은 거의가 제약된 최적화 문제로 적힌다.

**비표적 공격**(어떻게든 틀리게 가르게 하기):

$$
\boldsymbol{\delta}^* = \arg\max_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y)
$$

**표적 공격**(고른 추론 $y_{\text{target}}$을 억지로 내게 하기):

$$
\boldsymbol{\delta}^* = \arg\min_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y_{\text{target}})
$$

방어 문제는 그 짝이다. 가장 나쁜 자리의 로스를 가장 작게 하는 파라메터 $\theta$을 찾는다.

$$
\min_\theta \mathbb{E}_{(\mathbf{x},y) \sim \mathcal{D}} \left[ \max_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y) \right]
$$

이 가장 작게-가장 크게 꼴이 적대적 학습을 떠받치고, 이어지는 마디에서 다룰 공격과 방어의 클래스를 통틀어 이끈다.

---

## 5. 계량 금융과의 걸림

적대적 강건성은 금융 쓰임에서 학문에만 머무는 걱정이 아니다.

- **모델의 온전함**: 머신러닝 신호를 쓰는 거래 구조는 입력 데이터를 맞서서 흔드는 일(속인 시장 데이터 줄기 따위)에 든든해야 한다
- **규정 맞추기**: 서비스에 내놓은 금융 모델은 보여 줄 수 있는 강건성 다짐이 있어야 하며, 미쁨 점수 매기기와 위험 따지기에서 더욱 그렇다
- **속임수 알아내기**: 겨루는 이는 속이려는 뜻은 지킨 채 알아내는 구조를 비껴가는 입력을 부지런히 지어낸다
- **시장 조작**: 적대적 공격을 알면 흔들기에 버티는 살핌 구조를 꾸미는 데 도움이 된다

이 장에서는 이론의 깨침과 짜기를 저마다 손에 잡히는 금융 쓰임에 이어 붙인다.

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
이 마디에서 다룬 공격나 방어를 CIFAR-10의 ResNet-18 모델에 짜 넣어라. $\epsilon = 8/255$의 PGD-20 공격 아래에서 클린 정확도와 강건 정확도를 알려라.

</div>

??? success "풀이"
    여느 ResNet-18은 클린 정확도가 $\sim$93%이지만 PGD-20($\epsilon = 8/255$, 스텝 크기 $2/255$) 아래의 강건 정확도는 $\sim$0%이다. 이 마디의 방법을 걸면 피처와는 재주에 따라 다르다. 적대적 학습은 클린 정확도 $\sim$83%에 강건 정확도 $\sim$50%이고, 인증 가능한 방어는 더 낮지만 증명할 수 있는 테두리를 준다. 정확도와 강건성의 트레이드오프는 기초부터 있는 것이라, 강건성을 높이면 클린 정확도가 흔히 5~15% 든다. 아무렇게나 하는 시드 3개의 평균과 표준편차로 알려라. $\square$

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

이 절에서는 적대적 샘플을 찾아내다, 적대적 샘플은 왜 있는가, 강건성과 정확도의 트레이드오프, 최적화로 적은 공격을 살펴보았다.

**살펴볼 거리**

1. Szegedy, C., et al. (2014). "Intriguing Properties of Neural Networks." ICLR.
2. Goodfellow, I., Shlens, J., & Szegedy, C. (2015). "Explaining and Harnessing Adversarial Examples." ICLR.
3. Ilyas, A., et al. (2019). "Adversarial Examples Are Not Bugs, They Are Features." NeurIPS.
4. Tsipras, D., et al. (2019). "Robustness May Be at Odds with Accuracy." ICLR.
5. Gilmer, J., et al. (2018). "Adversarial Spheres." ICLR Workshop.
