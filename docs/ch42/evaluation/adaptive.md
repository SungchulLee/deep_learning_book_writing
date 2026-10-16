# 적응적 공격과 그래디언트 마스킹

적대적 강건성의 자취에는 처음엔 든든해 보였다가 나중에 **맞추어 오는 공격**, 곧 그 방어 구조를 콕 집어 넘어서려 꾸민 공격에 무너진 방어가 널려 있다. 그래디언트 가리기를 알고 맞추어 오는 공격을 꾸밀 줄 아는 일은 솔직한 강건성 따짐에 중요하다.

---

## 1. 그래디언트 가리기

### 그래디언트 가리기란

**그래디언트 가리기**은 방어가 참 강건성은 주지 못한 채 그래디언트 신호를 흐리거나 망가뜨릴 때 생긴다. 모델이 든든해 보이는 것은 참으로 든든해서가 아니라 그래디언트가 알려 주는 바가 없기 때문이다.

### 그래디언트 가리기의 클래스

1. **부서진 그래디언트**: 미분할 수 없는 계산(JPEG 눌러 담기, 수 줄이기, 입력 임계 자르기)이 계산 그림을 끊는다
2. **흔들리는 그래디언트**: 아무렇게나 하는 방어(시험 때 드롭아웃, 아무 크기 바꿈)가 잡음 섞인 못 미더운 그래디언트를 만든다
3. **사라지거나 터지는 그래디언트**: 깊은 방어 켜나 남다른 구조가 계산에서 그래디언트 탈을 낳는다
4. **흐려 놓은 그래디언트**: 그래디언트 계산을 헷갈리게 하려고 일부러 꾸민 방어

### 짚어내는 어림 규칙

다음 몇 가지가 그래디언트 가리기를 알리는 붉은 깃발이다.

1. **한 스텝 공격이 회귀 공격보다 잘 먹힌다**: FGSM이 PGD보다 잘 먹히면 그래디언트가 못 미덥다
2. **블랙박스 공격이 화이트박스보다 잘 먹힌다**: 전이되는 공격이 곧바로 그래디언트를 쓰는 공격보다 잘 들면 방어가 그래디언트를 가리고 있다
3. **마디 없는 공격이 안 먹힌다**: $\varepsilon$을 키워도 공격이 더 먹히지 않으면 다듬기가 막혀 있다
4. **아무 잡음이 맞먹는다**: 아무 섭동이 그래디언트 기반의 섭동과 엇비슷하게 먹히면 그래디언트가 쓸모 있는 신호를 못 준다

```python
import torch

def check_gradient_masking(model, x, y, epsilon=8/255):
    """
    그래디언트 가리기를 살피는 진단.
    
    진단 표시를 담은 사전을 돌려준다.
    """
    results = {}
    
    # 시험 1: FGSM과 PGD
    from attacks import FGSM, PGD
    fgsm = FGSM(model, epsilon=epsilon)
    pgd = PGD(model, epsilon=epsilon, num_iter=40)
    
    x_fgsm = fgsm.generate(x, y)
    x_pgd = pgd.generate(x, y)
    
    fgsm_success = evaluate_success(model, x_fgsm, y)
    pgd_success = evaluate_success(model, x_pgd, y)
    
    results['fgsm_success'] = fgsm_success
    results['pgd_success'] = pgd_success
    results['fgsm_stronger'] = fgsm_success > pgd_success + 0.05
    
    # 시험 2: 엡실론에 따라 한결같이 느는지
    success_curve = []
    for eps in [0.01, 0.02, 0.04, 0.08, 0.16]:
        attack = PGD(model, epsilon=eps, num_iter=20)
        x_adv = attack.generate(x, y)
        success_curve.append(evaluate_success(model, x_adv, y))
    
    results['monotonic'] = all(
        s1 <= s2 + 0.02 for s1, s2 in zip(success_curve[:-1], success_curve[1:])
    )
    
    # 시험 3: 아무 잡음과 견주기
    noise = torch.empty_like(x).uniform_(-epsilon, epsilon)
    x_random = torch.clamp(x + noise, 0, 1)
    random_success = evaluate_success(model, x_random, y)
    
    results['random_success'] = random_success
    results['random_competitive'] = random_success > 0.5 * pgd_success
    
    # 통틀어 본 판단
    results['gradient_masking_suspected'] = (
        results['fgsm_stronger'] or
        not results['monotonic'] or
        results['random_competitive']
    )
    
    return results
```

---

## 2. 맞추어 오는 공격 꾸미기

### 되돌아 스텝을 미분할 수 있게 어림하기(BPDA)

미분할 수 없는 몫 $g$을 지닌 방어에서는 되돌아 스텝 동안 $g$을 미분할 수 있는 어림 $\hat{g}$으로 갈음한다.

$$
\text{Forward: } f(g(\mathbf{x})), \quad \text{Backward: } \nabla f(\hat{g}(\mathbf{x}))
$$

$\hat{g}$으로 흔히 고르는 것: 제 자리 함수($g$이 거의 그대로 두는 것이면), 또는 학습된 뉴럴 넷트웍 어림.

### 바꿈에 걸친 바람(EOT)

아무렇게나 하는 방어에서는 그 아무렇게나임에 걸친 바람으로 그래디언트를 계산한다.

$$
\nabla_\mathbf{x} \mathbb{E}_{t \sim \mathcal{T}} [\mathcal{L}(f(t(\mathbf{x})), y)] \approx \frac{1}{K} \sum_{k=1}^K \nabla_\mathbf{x} \mathcal{L}(f(t_k(\mathbf{x})), y)
$$

### 방어 클래스마다 즐겨 쓸 공격

| 방어 클래스 | 맞추어 오는 공격 꾀 |
|-------------|------------------------|
| 미분할 수 없는 미리 다루기 | BPDA(제 자리 함수나 배운 어림) |
| 아무렇게나 하는 방어 | EOT(아무렇게나임에 걸쳐 그래디언트를 고르게 함) |
| 앙상블/골라 샘플링 | 클래스를 하나씩 치고 많은 쪽을 고른다 |
| 알아내기 + 물리기 | 분류기와 알아내개를 함께 친다 |
| 입력 바꿈 | 바꿈을 뚫고 다듬는다 |

---

## 3. 따짐 살핌표

새 방어라면 다음을 살펴 강건성을 따진다.

- [ ] 아무렇게나 여러 번 다시 비롯하는 PGD-100 이상
- [ ] 오토어택(파라메터 없는 앙상블)
- [ ] 그래디언트 가리기 진단(위의 시험 4가지)
- [ ] 방어에 미분할 수 없는 몫이 있으면 BPDA
- [ ] 방어가 아무렇게나 한다면 EOT
- [ ] 방어 없는 대신 모델에서 전이되는 공격
- [ ] 회귀를 넉넉히 준 C&W 공격
- [ ] 여러 노름의 위협($\ell_\infty$, $\ell_2$)

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
선형 분류기 $f(x) = w^T x + b$에서 예측한 클래스를 바꾸는 데 드는 가장 작은 $\ell_\infty$ 섭동을 계산하여라. 이것이 뉴럴 넷트웍의 강건성과 어떻게 이어지는지 밝혀라.

</div>

??? success "연습문제 1 풀이"
    선형 분류기에서 $\ell_\infty$ 노름으로 잰 결정 경계까지의 거리는 $\frac{|w^T x + b|}{\|w\|_1}$이다. 가장 작은 섭동은 $\delta^* = \frac{|w^T x + b|}{\|w\|_1} \cdot \text{sign}(w)$이다. 뉴럴 넷트웍에서는 그 자리의 선형 어림 $f(x + \delta) \approx f(x) + \nabla_x f \cdot \delta$이 FGSM(그래디언트의 부호를 쓴다)이 왜 잘 듣는지를 밝혀 준다. 차원이 높은 모델이 무른 까닭은 $\|w\|_1$은 차원과 함께 커지는데 $|w^T x + b|$은 꼭 그렇지 않아 강건성의 여유가 줄어들기 때문이다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 마디에서 다룬 공격나 방어를 CIFAR-10의 ResNet-18 모델에 짜 넣어라. $\epsilon = 8/255$의 PGD-20 공격 아래에서 클린 정확도와 강건 정확도를 알려라.

</div>

??? success "연습문제 2 풀이"
    여느 ResNet-18은 클린 정확도가 $\sim$93%이지만 PGD-20($\epsilon = 8/255$, 스텝 크기 $2/255$) 아래의 강건 정확도는 $\sim$0%이다. 이 마디의 방법을 걸면 피처와는 재주에 따라 다르다. 적대적 학습은 클린 정확도 $\sim$83%에 강건 정확도 $\sim$50%이고, 인증 가능한 방어는 더 낮지만 증명할 수 있는 테두리를 준다. 정확도와 강건성의 트레이드오프는 기초부터 있는 것이라, 강건성을 높이면 클린 정확도가 흔히 5~15% 든다. 아무렇게나 하는 시드 3개의 평균과 표준편차로 알려라. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
섭동 공 안에서 클래스별 데이터의 밑자리가 서로 겹친다고 볼 때, 모델이 담는 힘을 키우지 않고서는 어떤 방어도 맑은 데이터의 높은 정확도와 $\ell_\infty$ 섭동에 대한 높은 강건성을 함께 이룰 수 없음을 증명하여라.

</div>

??? success "연습문제 3 풀이"
    두 클래스의 밑자리가 거리 $\epsilon$ 안에서 겹치면(곧 $\|x_1 - x_2\|_\infty \leq 2\epsilon$인 $x_1 \in \text{클래스 1}, x_2 \in \text{클래스 2}$이 있으면), $x_1$과 $x_2$ 둘 다에서 강건한 분류기는 적어도 하나를 틀리게 가를 수밖에 없다(섭동 공이 겹치기 때문이다). 이것이 정확도와 강건성의 기초 트레이드오프다. 겹치는 밑자리의 몫이 피할 수 없는 정확도 로스를 정한다. 여느 그림 분포에서는 $\epsilon = 8/255$에서 겹침이 꽤 있어, 살펴본 10~15%의 정확도 떨어짐을 밝혀 준다. 모델이 담는 힘을 키우면(더 너른 네트워크) 얽힌 강건한 결정 경계를 더 잘 그려 트레이드오프를 얼마쯤 눅일 수 있다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
금융 머신러닝 구조(속임수 알아내기나 거래 신호 만들기 따위)에서 맞섬의 강건성이 어떻게 드러나는지 다루어라. 위협 모델이 보기 다룸과 어떻게 다른가?

</div>

??? success "연습문제 4 풀이"
    금융에서 겨루는 이는 알아내는 구조에 맞추어 스스로 움직이는 꾀 많은 무리(속임수꾼, 저자 흔드는 이)다. 보기 다룸과 다른 핵심은 이렇다. (1) 흔들 수 있는 밭이 돈으로 될 만한 것에 옭매인다(속임수꾼이 제 거래 자취를 통째로 바꿀 수는 없다). (2) 공격이 잇따르며 맞추어 간다(겨루는 이가 구조의 되받음을 보고 손본다). (3) 헛 정확도와 놓침의 값이 서로 어긋난다(옳은 거래를 막는 것과 속임수를 놓치는 것). (4) $\ell_p$ 노름은 뜻이 없고 밭에 맞는 섭동 모델이 있어야 한다. 방어는 맞추어 오는 겨루는 이에게도 든든해야 하므로, 알아내는 잣대가 알려지면 비껴갈 수 있는 알아내기 기초의 길은 많이 걸러진다. $\square$

## 정리하며

그래디언트 가리기는 맞섬 방어가 어그러지는 가장 흔한 피처가다. 솔직한 따짐에는 그 방어 구조를 콕 집어 꾸민 맞추어 오는 공격이 있어야 한다. 오토어택이 도움이 되지만 새로운 방어 구조에는 모자랄 수 있으니, 늘 그 방어에 맞춘 공격로 채워야 한다.

**살펴볼 거리**

1. Athalye, A., Carlini, N., & Wagner, D. (2018). "Obfuscated Gradients Give a False Sense of Security." ICML.
2. Tramer, F., et al. (2020). "On Adaptive Attacks to Adversarial Example Defenses." NeurIPS.
3. Carlini, N., et al. (2019). "On Evaluating Adversarial Robustness." arXiv preprint arXiv:1902.06705.
