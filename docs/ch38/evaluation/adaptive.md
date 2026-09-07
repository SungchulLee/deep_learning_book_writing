# 맞추어 오는 치기와 기울기 가리기
## 들머리

맞섬에 든든하기의 자취에는 처음엔 든든해 보였다가 나중에 **맞추어 오는 치기**, 곧 그 막이 얼개를 콕 집어 넘어서려 꾸민 치기에 무너진 막이가 널려 있다. 기울기 가리기를 알고 맞추어 오는 치기를 꾸밀 줄 아는 일은 솔직한 든든함 따짐에 종요롭다.

## 기울기 가리기

### 기울기 가리기란

**기울기 가리기**은 막이가 참 든든함은 주지 못한 채 기울기 신호를 흐리거나 망가뜨릴 때 생긴다. 모형이 든든해 보이는 것은 참으로 든든해서가 아니라 기울기가 알려 주는 바가 없기 때문이다.

### 기울기 가리기의 갈래

1. **부서진 기울기**: 미분할 수 없는 셈(JPEG 눌러 담기, 수 줄이기, 들임 문턱 자르기)이 셈 그림을 끊는다
2. **흔들리는 기울기**: 아무렇게나 하는 막이(시험 때 드롭아웃, 아무 크기 바꿈)가 잡음 섞인 못 미더운 기울기를 만든다
3. **사라지거나 터지는 기울기**: 깊은 막이 켜나 남다른 얼개가 셈에서 기울기 탈을 낳는다
4. **흐려 놓은 기울기**: 기울기 셈을 헷갈리게 하려고 일부러 꾸민 막이

### 짚어내는 어림 규칙

다음 몇 가지가 기울기 가리기를 알리는 붉은 깃발이다.

1. **한 걸음 치기가 되돌이 치기보다 잘 먹힌다**: FGSM이 PGD보다 잘 먹히면 기울기가 못 미덥다
2. **검은 상자 치기가 흰 상자보다 잘 먹힌다**: 옮아가는 치기가 곧바로 기울기를 쓰는 치기보다 잘 들면 막이가 기울기를 가리고 있다
3. **마디 없는 치기가 안 먹힌다**: $\varepsilon$을 키워도 치기가 더 먹히지 않으면 다듬기가 막혀 있다
4. **아무 잡음이 맞먹는다**: 아무 흔듦이 기울기 바탕의 흔듦과 엇비슷하게 먹히면 기울기가 쓸모 있는 신호를 못 준다

```python
import torch

def check_gradient_masking(model, x, y, epsilon=8/255):
    """
    기울기 가리기를 살피는 진단.
    
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

## 맞추어 오는 치기 꾸미기

### 되돌아 걸음을 미분할 수 있게 어림하기(BPDA)

미분할 수 없는 몫 $g$을 지닌 막이에서는 되돌아 걸음 동안 $g$을 미분할 수 있는 어림 $\hat{g}$으로 갈음한다.

$$
\text{Forward: } f(g(\mathbf{x})), \quad \text{Backward: } \nabla f(\hat{g}(\mathbf{x}))
$$

$\hat{g}$으로 흔히 고르는 것: 제 자리 함수($g$이 거의 그대로 두는 것이면), 또는 익힌 신경 그물 어림.

### 바꿈에 걸친 바람(EOT)

아무렇게나 하는 막이에서는 그 아무렇게나임에 걸친 바람으로 기울기를 셈한다.

$$
\nabla_\mathbf{x} \mathbb{E}_{t \sim \mathcal{T}} [\mathcal{L}(f(t(\mathbf{x})), y)] \approx \frac{1}{K} \sum_{k=1}^K \nabla_\mathbf{x} \mathcal{L}(f(t_k(\mathbf{x})), y)
$$

### 막이 갈래마다 즐겨 쓸 치기

| 막이 갈래 | 맞추어 오는 치기 꾀 |
|-------------|------------------------|
| 미분할 수 없는 미리 다루기 | BPDA(제 자리 함수나 배운 어림) |
| 아무렇게나 하는 막이 | EOT(아무렇게나임에 걸쳐 기울기를 고르게 함) |
| 모둠/골라 뽑기 | 갈래를 하나씩 치고 많은 쪽을 고른다 |
| 알아내기 + 물리기 | 가름개와 알아내개를 함께 친다 |
| 들임 바꿈 | 바꿈을 뚫고 다듬는다 |

## 따짐 살핌표

새 막이라면 다음을 살펴 든든함을 따진다.

- [ ] 아무렇게나 여러 번 다시 비롯하는 PGD-100 이상
- [ ] 오토어택(매개변수 없는 모둠)
- [ ] 기울기 가리기 진단(위의 시험 4가지)
- [ ] 막이에 미분할 수 없는 몫이 있으면 BPDA
- [ ] 막이가 아무렇게나 한다면 EOT
- [ ] 막이 없는 대신 모형에서 옮아가는 치기
- [ ] 되돌이를 넉넉히 준 C&W 치기
- [ ] 여러 노름의 으름($\ell_\infty$, $\ell_2$)

## 간추림

기울기 가리기는 맞섬 막이가 어그러지는 가장 흔한 결이다. 솔직한 따짐에는 그 막이 얼개를 콕 집어 꾸민 맞추어 오는 치기가 있어야 한다. 오토어택이 도움이 되지만 새로운 막이 얼개에는 모자랄 수 있으니, 늘 그 막이에 맞춘 치기로 채워야 한다.

## 살펴볼 거리

1. Athalye, A., Carlini, N., & Wagner, D. (2018). "Obfuscated Gradients Give a False Sense of Security." ICML.
2. Tramer, F., et al. (2020). "On Adaptive Attacks to Adversarial Example Defenses." NeurIPS.
3. Carlini, N., et al. (2019). "On Evaluating Adversarial Robustness." arXiv preprint arXiv:1902.06705.

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
