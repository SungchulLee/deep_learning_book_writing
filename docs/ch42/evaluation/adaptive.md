# 적응적 공격과 그래디언트 마스킹

적대적 강건성의 역사에는 처음엔 강건해 보였다가 나중에 **적응형 공격**, 곧 그 방어 구조를 겨냥해 우회하도록 설계한 공격에 무너진 방어가 수두룩하다. 그래디언트 마스킹을 이해하고 적응형 공격을 설계할 줄 아는 일은 정직한 강건성 평가에 꼭 필요하다.

---

## 1. 그래디언트 마스킹

### 그래디언트 마스킹이란

**그래디언트 마스킹**은 방어가 참 강건성은 주지 못한 채 그래디언트 신호를 흐리거나 망가뜨릴 때 생긴다. 모델이 강건해 보이는 것은 실제로 강건해서가 아니라 그래디언트가 공격자에게 쓸모 있는 정보를 주지 않기 때문이다.

### 그래디언트 마스킹의 종류

1. **부서진 그래디언트**(shattered): 미분할 수 없는 연산(JPEG 압축, 양자화, 입력 임계값 처리)이 계산 그래프를 끊는다
2. **확률적 그래디언트**(stochastic): 무작위 방어(테스트 때의 드롭아웃, 무작위 리사이즈)가 잡음 섞인, 믿을 수 없는 그래디언트를 만든다
3. **사라지거나 폭발하는 그래디언트**: 깊은 방어 레이어나 특이한 구조가 그래디언트 계산을 수치적으로 망가뜨린다
4. **난독화된 그래디언트**(obfuscated): 그래디언트 계산을 헷갈리게 하려고 일부러 설계한 방어

### 탐지를 위한 경험 규칙

다음 몇 가지가 그래디언트 마스킹을 알리는 위험 신호다.

1. **한 스텝 공격이 반복 공격보다 잘 통한다**: FGSM이 PGD보다 잘 통하면 그래디언트를 믿을 수 없다
2. **블랙박스 공격이 화이트박스보다 잘 통한다**: 전이 공격이 그래디언트를 직접 쓰는 공격보다 잘 들면 방어가 그래디언트를 가리고 있다
3. **제한 없는 공격이 실패한다**: $\varepsilon$을 크게 키워도 공격 성공률이 100%에 다가가지 않으면 최적화가 막혀 있다
4. **무작위 잡음이 맞먹는다**: 무작위 섭동이 그래디언트 기반 섭동과 엇비슷하게 통하면 그래디언트가 쓸모 있는 신호를 주지 못한다

아래 진단 코드는 이 가운데 1, 3, 4번을 시험한다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


# === 공격 ===

class FGSM:
    """Fast Gradient Sign Method — 그래디언트의 부호로 한 스텝.

        x_adv = clip(x + epsilon * sign(grad_x loss))

    한 번만 계산하므로 싸다. 다만 로스가 그 자리에서 선형이라고 보는 것이라,
    굽은 자리에서는 약하다. 그것이 PGD가 여러 스텝을 밟는 까닭이다.
    """

    def __init__(self, model: nn.Module, epsilon: float = 0.03,
                 clip_min: float = 0.0, clip_max: float = 1.0):
        self.model = model
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max

    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()                     # 드롭아웃 등이 그래디언트를 흔들지 않게

        x_adv = x.clone().detach().requires_grad_(True)
        loss = F.cross_entropy(self.model(x_adv), y)
        grad, = torch.autograd.grad(loss, x_adv)

        out = x + self.epsilon * grad.sign()
        out = out.clamp(self.clip_min, self.clip_max)

        if was_training:
            self.model.train()
        return out.detach()


class PGD:
    """Projected Gradient Descent — 작은 스텝을 여러 번 밟고 매번 볼 안으로 투영한다.

        x_0   = x + U(-epsilon, epsilon)          (무작위로 시작한다)
        x_t+1 = proj_ball( x_t + alpha * sign(grad) )

    무작위로 시작하는 것이 중요하다. 늘 x에서 출발하면 그래디언트가 0인 자리에
    걸려 공격이 실패하는 것을 "강건성"으로 잘못 읽게 된다.
    """

    def __init__(self, model: nn.Module, epsilon: float = 0.03,
                 alpha: float = None, num_iter: int = 40,
                 random_start: bool = True,
                 clip_min: float = 0.0, clip_max: float = 1.0):
        self.model = model
        self.epsilon = epsilon
        # 스텝 크기를 따로 주지 않으면 볼을 두어 번 건널 만큼으로 잡는다
        self.alpha = alpha if alpha is not None else max(epsilon / 4, 2.5 * epsilon / num_iter)
        self.num_iter = num_iter
        self.random_start = random_start
        self.clip_min = clip_min
        self.clip_max = clip_max

    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()

        if self.random_start:
            delta = torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
            x_adv = (x + delta).clamp(self.clip_min, self.clip_max)
        else:
            x_adv = x.clone()
        x_adv = x_adv.detach()

        for _ in range(self.num_iter):
            x_adv.requires_grad_(True)
            loss = F.cross_entropy(self.model(x_adv), y)
            grad, = torch.autograd.grad(loss, x_adv)
            x_adv = x_adv.detach() + self.alpha * grad.sign()
            # epsilon 볼 안으로 투영한 뒤, 이미지가 가질 수 있는 값으로 자른다
            x_adv = torch.min(torch.max(x_adv, x - self.epsilon), x + self.epsilon)
            x_adv = x_adv.clamp(self.clip_min, self.clip_max).detach()

        if was_training:
            self.model.train()
        return x_adv


# === 진단 ===

@torch.no_grad()
def evaluate_success(model, x_adv, y):
    """공격 성공률 — 적대적 입력에서 예측이 정답과 달라진 비율(0~1).

    처음부터 틀린 표본도 「성공」으로 센다. 그래서 깨끗한 입력의 오류율이
    이 값의 바닥이 된다.
    """
    was_training = model.training
    model.eval()
    pred = model(x_adv).argmax(dim=1)
    if was_training:
        model.train()
    return (pred != y).float().mean().item()


def check_gradient_masking(model, x, y, epsilon=8/255):
    """
    그래디언트 가리기를 살피는 진단.
    
    진단 표시를 담은 사전을 돌려준다.
    """
    results = {}
    
    # 시험 1: FGSM과 PGD
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

## 2. 적응형 공격 설계

### 역전파 미분 근사(BPDA)

미분할 수 없는 구성 요소 $g$를 지닌 방어에서는 역전파 단계에서만 $g$를 미분할 수 있는 근사 $\hat{g}$로 바꾼다(Backward Pass Differentiable Approximation). 순전파는 원래의 $g$를 그대로 쓴다.

$$
\text{Forward: } f(g(\mathbf{x})), \quad \text{Backward: } \nabla f(\hat{g}(\mathbf{x}))
$$

$\hat{g}$로 흔히 고르는 것은 항등 함수($g(\mathbf{x}) \approx \mathbf{x}$일 때)나 학습된 뉴럴 네트워크 근사다.

### 변환에 대한 기댓값(EOT)

무작위 방어에서는 그 무작위성에 대한 기댓값으로 그래디언트를 계산한다.

$$
\nabla_\mathbf{x} \mathbb{E}_{t \sim \mathcal{T}} [\mathcal{L}(f(t(\mathbf{x})), y)] \approx \frac{1}{K} \sum_{k=1}^K \nabla_\mathbf{x} \mathcal{L}(f(t_k(\mathbf{x})), y)
$$

### 방어 종류별 권장 공격

| 방어 종류 | 적응형 공격 전략 |
|-------------|------------------------|
| 미분할 수 없는 전처리 | BPDA(항등 함수나 학습된 근사) |
| 무작위 방어 | EOT(무작위성에 걸쳐 그래디언트를 평균) |
| 앙상블/무작위 선택 | 구성 모델을 모두 함께 공격하거나 다수결을 겨냥한다 |
| 탐지 + 거부 | 분류기와 탐지기를 함께 속이도록 공격한다 |
| 입력 변환 | 변환을 통과하도록 최적화한다(BPDA·EOT) |

---

## 3. 평가 체크리스트

새 방어라면 다음을 확인해 강건성을 평가한다.

- [ ] 무작위 재시작을 여러 번 하는 PGD-100 이상
- [ ] AutoAttack(하이퍼파라미터 조정이 필요 없는 앙상블)
- [ ] 그래디언트 마스킹 진단(위의 위험 신호 4가지)
- [ ] 방어에 미분할 수 없는 구성 요소가 있으면 BPDA
- [ ] 방어가 무작위성을 쓰면 EOT
- [ ] 방어가 없는 대리 모델에서 전이되는 공격
- [ ] 반복을 넉넉히 준 C&W 공격
- [ ] 여러 노름의 위협($\ell_\infty$, $\ell_2$)

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
선형 분류기 $f(x) = w^T x + b$에서 예측한 클래스를 바꾸는 데 드는 가장 작은 $\ell_\infty$ 섭동을 계산하여라. 이것이 뉴럴 네트워크의 강건성과 어떻게 이어지는지 설명하여라.

</div>

??? success "풀이"
    선형 분류기에서 $\ell_\infty$ 노름으로 잰 결정 경계까지의 거리는 $\frac{|w^T x + b|}{\|w\|_1}$이다. 가장 작은 섭동은 $\delta^* = \frac{|w^T x + b|}{\|w\|_1} \cdot \text{sign}(w)$이다. 뉴럴 네트워크에서는 국소 선형 근사 $f(x + \delta) \approx f(x) + \nabla_x f \cdot \delta$이 FGSM(그래디언트의 부호를 쓴다)이 왜 잘 통하는지를 설명해 준다. 차원이 높은 모델이 취약한 까닭은 $\|w\|_1$은 차원과 함께 커지는데 $|w^T x + b|$은 꼭 그렇지 않아 강건성의 여유가 줄어들기 때문이다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 절에서 다룬 공격이나 방어를 CIFAR-10의 ResNet-18 모델에 구현하여라. $\epsilon = 8/255$의 PGD-20 공격 아래에서 클린 정확도와 강건 정확도를 보고하여라.

</div>

??? success "풀이"
    일반 ResNet-18은 클린 정확도가 $\sim$93%이지만 PGD-20($\epsilon = 8/255$, 스텝 크기 $2/255$) 아래의 강건 정확도는 $\sim$0%이다. 이 절의 방법을 적용한 결과는 기법에 따라 다르다. 적대적 학습은 클린 정확도 $\sim$83%에 강건 정확도 $\sim$50%이고, 인증 가능한 방어는 더 낮지만 증명할 수 있는 바운드를 준다. 정확도와 강건성의 트레이드오프는 근본적인 것이라, 강건성을 높이면 클린 정확도가 흔히 5~15% 떨어진다. 무작위 시드 3개의 평균과 표준편차로 보고하여라. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
섭동 공 안에서 클래스별 데이터의 서포트가 서로 겹친다고 볼 때, 모델이 용량을 키우지 않고서는 어떤 방어도 클린 데이터의 높은 정확도와 $\ell_\infty$ 섭동에 대한 높은 강건성을 함께 이룰 수 없음을 증명하여라.

</div>

??? success "풀이"
    두 클래스의 서포트가 거리 $\epsilon$ 안에서 겹치면(곧 $\|x_1 - x_2\|_\infty \leq 2\epsilon$인 $x_1 \in \text{클래스 1}, x_2 \in \text{클래스 2}$이 있으면), $x_1$과 $x_2$ 둘 다에서 강건한 분류기는 적어도 하나를 틀리게 가를 수밖에 없다(섭동 공이 겹치기 때문이다). 이것이 정확도와 강건성의 근본적인 트레이드오프다. 겹치는 서포트의 비중이 피할 수 없는 정확도 손실을 정한다. 자연 이미지 분포에서는 $\epsilon = 8/255$에서 겹침이 꽤 있어, 관찰되는 10~15%의 정확도 하락을 설명해 준다. 모델 용량을 키우면(더 넓은 네트워크) 복잡한 강건 결정 경계를 더 잘 그려 트레이드오프를 어느 정도 완화할 수 있다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
금융 머신러닝 시스템(사기 탐지나 거래 신호 생성 등)에서 적대적 강건성이 어떻게 드러나는지 논하여라. 위협 모델이 비전과 어떻게 다른가?

</div>

??? success "풀이"
    금융에서 적대자는 탐지 시스템에 맞추어 전략을 바꾸는 교묘한 집단(사기꾼, 시장 조작자)이다. 비전과 다른 핵심은 이렇다. (1) 섭동할 수 있는 피처가 경제적으로 그럴듯한 범위로 제약된다(사기꾼이 자기 거래 이력을 통째로 바꿀 수는 없다). (2) 공격이 순차적이고 적응적이다(적대자가 시스템의 응답을 보고 고친다). (3) 거짓 양성과 거짓 음성의 비용이 비대칭이다(정상 거래를 막는 것과 사기를 놓치는 것). (4) $\ell_p$ 노름은 의미가 없고 도메인에 맞는 섭동 모델이 있어야 한다. 방어는 적응형 적대자에게도 강건해야 하므로, 탐지 기준이 알려지면 회피할 수 있는 탐지 기반 방법은 쓸모가 크게 줄어든다. $\square$

## 정리하며

그래디언트 마스킹은 적대적 방어가 실패하는 가장 흔한 원인이다. 정직한 평가에는 그 방어 구조를 겨냥해 설계한 적응형 공격이 있어야 한다. AutoAttack이 도움이 되지만 새로운 방어 구조에는 모자랄 수 있으니, 늘 그 방어에 맞춘 공격으로 보완해야 한다.

**살펴볼 거리**

1. Athalye, A., Carlini, N., & Wagner, D. (2018). "Obfuscated Gradients Give a False Sense of Security." ICML.
2. Tramer, F., et al. (2020). "On Adaptive Attacks to Adversarial Example Defenses." NeurIPS.
3. Carlini, N., et al. (2019). "On Evaluating Adversarial Robustness." arXiv preprint arXiv:1902.06705.
