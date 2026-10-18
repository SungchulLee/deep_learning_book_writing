# AutoAttack

**오토어택**(크로체 & 하인, 2020)은 맞섬에 강건성을 미덥게 따지려고 꾸민, 파라메터 없는 앙상블 기초의 공격이다. 서로 채워 주는 공격 꾀 여럿을 아울러, 그래디언트 가리기나 잘못 고른 하이퍼파라미터 탓에 든든하다고 잘못 말할 위험을 가장 작게 한다. 오토어택은 적대적 강건성 글에서 강건 정확도를 알리는 사실상의 잣대가 되었다.

---

## 1. 왜 하는가

처음에는 든든해 보이던 방어가 더 세거나 더 곱게 맞춘 공격로 따져 보니 무너진 일이 많다. 흔히 어그러지는 피처는 이렇다.

- **그래디언트 가리기**: 참 강건성 대신 그래디언트를 감추는 방어
- **잘못 고른 하이퍼파라미터**: 스텝이 너무 적거나 스텝 크기가 틀린 PGD
- **로스 함수 하나**: 엇갈린 엔트로피가 모든 방어에 가장 잘 듣는 공격 목표는 아니다

오토어택은 **하이퍼파라미터를 맞출 것 없이** 서로 다른 공격의 앙상블을 써서 이를 푼다.

---

## 2. 이루는 몫

오토어택은 서로 채워 주는 공격 넷을 차례로 돌린다.

### 1. APGD-CE(엇갈린 엔트로피를 쓰는 오토-PGD)

**맞추어 가는 스텝 크기**를 쓰도록 나아진 PGD이다.

$$
\alpha^{(t+1)} = \begin{cases}
\alpha^{(t)} / 2 & \text{if loss hasn't improved in } w \text{ steps} \\
\alpha^{(t)} & \text{otherwise}
\end{cases}
$$

여느 PGD보다 나아진 핵심은 이렇다.

- **밀어 나감**: $\mathbf{z}^{(t+1)} = \rho \cdot \mathbf{z}^{(t)} + \nabla_\mathbf{x} \mathcal{L}$
- **찰칵 자리로 되돌리기**: 스텝 크기가 줄면 찾아 둔 가장 좋은 점으로 되돌린다
- **스텝 크기를 맞출 것 없음**: 로스의 자취를 보고 절로 맞춘다

### 2. APGD-DLR(로짓 차 비교를 쓰는 오토-PGD)

엇갈린 엔트로피 대신 **로짓 차 비교** 로스를 쓴다.

$$
\mathcal{L}_{\text{DLR}} = -\frac{z_y - \max_{i \neq y} z_i}{z_{\pi_1} - z_{\pi_3}}
$$

여기서 $z_{\pi_1} \geq z_{\pi_2} \geq \ldots$은 줄 세운 로짓이다. DLR 로스는

- 잣대가 바뀌어도 그대로다(아래가 로짓의 자리로 잣대를 맞춘다)
- 자신하는 추론 언저리에서 엇갈린 엔트로피가 배부르는 탈을 비껴간다
- 자신함을 높게 지키는 방어에 더 잘 듣는다

### 3. FAB(빠르게 맞추어 가는 금)

결정 경계로 거듭 되비추어 작은 섭동을 찾는 **가장 작은 노름 공격**이다.

- 이제의 점을 곧게 편 결정 경계로 되비춘다
- 타깃 클래스 여럿을 두루 뒤진다
- 예산이 붙박인 PGD의 섭동보다 작은 섭동을 찾는 일이 잦다

방어의 로스 터가 매끄럽지 않을 때 FAB이 더욱 쓸모 있다.

### 4. 네모 공격(블랙박스)

아무렇게나 잡은 네모 꼴 섭동을 쓰는 **스코어 기반 블랙박스** 공격이다.

- 그래디언트가 있어야 하지 않다(그래디언트를 가리는 방어를 잡아낸다)
- 그 자리에 몰린 아무 뒤지기 꾀를 쓴다
- 로스를 가장 크게 하도록 네모 헝겊을 거듭 고친다

블랙박스 공격을 넣는 일이 중요하다. 방어가 그래디언트를 가리면 APGD은 듣지 않아도 네모 공격은 먹힐 수 있다.

---

## 3. 따지는 절차

오토어택은 공격을 차례로 돌린다. 공격이 끝날 때마다 살아남은 보기(아직 틀리게 갈리지 않은 보기)만 다음 공격로 넘긴다.

```
온 시험 꾸러미 → APGD-CE → 살아남음 → APGD-DLR → 살아남음 → FAB → 살아남음 → 네모 → 마지막 든든한 맞음
```

마지막 강건 정확도는 공격 넷을 모두 이겨 낸 몫이다.

### 흔한 출력

```
처음 맞음:  87.20%
apgd-ce:           든든한 맞음: 53.40% (- 33.80%)
apgd-t (DLR):     든든한 맞음: 51.20% (- 2.20%)
fab-t:             든든한 맞음: 50.80% (- 0.40%)
square:            든든한 맞음: 50.60% (- 0.20%)
```

---

## 4. PyTorch로 쓰기

### 여느 따짐

```python
# 깔기: pip install autoattack
from autoattack import AutoAttack

# 여느 따짐(치기 4개)
adversary = AutoAttack(
    model, 
    norm='Linf', 
    eps=8/255, 
    version='standard'
)
x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=100)

# 빠른 따짐(APGD-CE + APGD-DLR만)
adversary_fast = AutoAttack(
    model, 
    norm='Linf', 
    eps=8/255, 
    version='plus'
)

# 골라 쓰는 치기 배치
adversary_custom = AutoAttack(model, norm='Linf', eps=8/255)
adversary_custom.attacks_to_run = ['apgd-ce', 'apgd-t']
```

### 따짐 흐름에 붙이기

```python
import torch
from typing import Dict

def autoattack_evaluation(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 8/255,
    norm: str = 'Linf',
    version: str = 'standard',
    batch_size: int = 100
) -> Dict[str, float]:
    """
    오토어택 따짐을 돌린다.
    
    Parameters
    ----------
    model : nn.Module
        따질 모형
    x, y : torch.Tensor
        시험 들임과 이름표
    epsilon : float
        흔듦 예산
    norm : str
        노름 클래스('Linf' 또는 'L2')
    version : str
        'standard'(치기 4개) 또는 'plus'(치기 2개)
    
    Returns
    -------
    results : clean_accuracy과 robust_accuracy을 담은 사전
    """
    from autoattack import AutoAttack
    
    model.eval()
    
    # 맑은 맞음
    with torch.no_grad():
        device = next(model.parameters()).device
        clean_pred = model(x.to(device)).argmax(dim=1)
        clean_acc = (clean_pred == y.to(device)).float().mean().item()
    
    # 오토어택
    adversary = AutoAttack(model, norm=norm, eps=epsilon, version=version)
    x_adv = adversary.run_standard_evaluation(x, y, bs=batch_size)
    
    with torch.no_grad():
        adv_pred = model(x_adv.to(device)).argmax(dim=1)
        robust_acc = (adv_pred == y.to(device)).float().mean().item()
    
    return {
        'clean_accuracy': clean_acc,
        'robust_accuracy': robust_acc,
        'attack_success_rate': 1 - robust_acc,
        'norm': norm,
        'epsilon': epsilon
    }
```

---

## 5. 오토어택을 쓸 때

### 늘 쓸 자리

- 글이나 보고서에 **강건 정확도를 알릴** 때
- 잣대 잡힌 데이터에서 **방어를 견줄** 때
- 내놓은 모델의 **마지막 따짐**

### 다른 길을 헤아릴 자리

- **빠르게 만들어 볼 때**: 거듭 고쳐 가는 데는 PGD-20이 훨씬 빠르다
- **학습 안쪽 돌기**: 적대적 학습에는 너무 느리다
- **남다른 위협 모델**: 오토어택은 여느 $\ell_p$ 공을 가정한다

---

## 6. 낱낱의 공격과 비교

| 피처 | PGD-100 | C&W | 오토어택 |
|--------|---------|-----|------------|
| 파라메터 없음 | 아니다 | 아니다 | **그렇다** |
| 그래디언트 가리기를 잡아냄 | 아니다 | 얼마쯤 | **그렇다** |
| 로스 함수 여럿 | 아니다 | 아니다 | **그렇다** |
| 블랙박스 몫 | 아니다 | 아니다 | **그렇다** |
| 빠르기 | 빠름 | 느림 | 가운데 |
| 미더움 | 좋음 | 좋음 | **가장 좋음** |

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

| 피처 | 자세한 것 |
|---------|--------|
| **뜻** | 미덥고 잣대 잡힌 강건성 따짐 |
| **이루는 몫** | APGD-CE, APGD-DLR, FAB, 네모 공격 |
| **핵심 피처** | 파라메터 없는 앙상블 |
| **쓸 자리** | 마지막 따짐과 알리기 |
| **나은 점** | 그래디언트 가리기를 잡아내고 맞출 것이 없다 |

오토어택이 맞섬에 강건성 따짐의 으뜸 잣대가 된 까닭은 바로, 손으로 맞춘 공격 파라메터의 불확실성을 없애고 여러 공격 꾀를 두루 덮어 주기 때문이다.

**살펴볼 거리**

1. Croce, F., & Hein, M. (2020). "Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-Free Attacks." ICML.
2. Croce, F., & Hein, M. (2020). "Minimally Distorted Adversarial Examples with a Fast Adaptive Boundary Attack." ICML.
3. Andriushchenko, M., et al. (2020). "Square Attack: A Query-Efficient Black-Box Adversarial Attack via Random Search." ECCV.
