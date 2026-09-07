# 오토어택
## 들머리

**오토어택**(크로체 & 하인, 2020)은 맞섬에 든든함을 미덥게 따지려고 꾸민, 매개변수 없는 모둠 바탕의 치기다. 서로 채워 주는 치기 꾀 여럿을 아울러, 기울기 가리기나 잘못 고른 하이퍼파라미터 탓에 든든하다고 잘못 말할 무릅씀을 가장 작게 한다. 오토어택은 맞섬에 든든하기 글에서 든든한 맞음을 알리는 사실상의 잣대가 되었다.

## 왜 하는가

처음에는 든든해 보이던 막이가 더 세거나 더 곱게 맞춘 치기로 따져 보니 무너진 일이 많다. 흔히 어그러지는 결은 이렇다.

- **기울기 가리기**: 참 든든함 대신 기울기를 감추는 막이
- **잘못 고른 하이퍼파라미터**: 걸음이 너무 적거나 걸음 크기가 틀린 PGD
- **잃음 함수 하나**: 엇갈린 엔트로피가 모든 막이에 가장 잘 듣는 치기 목표는 아니다

오토어택은 **하이퍼파라미터를 맞출 것 없이** 서로 다른 치기의 모둠을 써서 이를 푼다.

## 이루는 몫

오토어택은 서로 채워 주는 치기 넷을 차례로 돌린다.

### 1. APGD-CE(엇갈린 엔트로피를 쓰는 오토-PGD)

**맞추어 가는 걸음 크기**를 쓰도록 나아진 PGD이다.

$$
\alpha^{(t+1)} = \begin{cases}
\alpha^{(t)} / 2 & \text{if loss hasn't improved in } w \text{ steps} \\
\alpha^{(t)} & \text{otherwise}
\end{cases}
$$

여느 PGD보다 나아진 고갱이는 이렇다.

- **밀어 나감**: $\mathbf{z}^{(t+1)} = \rho \cdot \mathbf{z}^{(t)} + \nabla_\mathbf{x} \mathcal{L}$
- **찰칵 자리로 되돌리기**: 걸음 크기가 줄면 찾아 둔 가장 좋은 점으로 되돌린다
- **걸음 크기를 맞출 것 없음**: 잃음의 자취를 보고 절로 맞춘다

### 2. APGD-DLR(로짓 차 견줌을 쓰는 오토-PGD)

엇갈린 엔트로피 대신 **로짓 차 견줌** 잃음을 쓴다.

$$
\mathcal{L}_{\text{DLR}} = -\frac{z_y - \max_{i \neq y} z_i}{z_{\pi_1} - z_{\pi_3}}
$$

여기서 $z_{\pi_1} \geq z_{\pi_2} \geq \ldots$은 줄 세운 로짓이다. DLR 잃음은

- 잣대가 바뀌어도 그대로다(아래가 로짓의 자리로 잣대를 맞춘다)
- 자신하는 미루어 봄 언저리에서 엇갈린 엔트로피가 배부르는 탈을 비껴간다
- 자신함을 높게 지키는 막이에 더 잘 듣는다

### 3. FAB(빠르게 맞추어 가는 금)

판단의 금으로 거듭 되비추어 작은 흔듦을 찾는 **가장 작은 노름 치기**다.

- 이제의 점을 곧게 편 판단의 금으로 되비춘다
- 과녁 갈래 여럿을 두루 뒤진다
- 예산이 붙박인 PGD의 흔듦보다 작은 흔듦을 찾는 일이 잦다

막이의 잃음 터가 매끄럽지 않을 때 FAB이 더욱 쓸모 있다.

### 4. 네모 치기(검은 상자)

아무렇게나 잡은 네모 꼴 흔듦을 쓰는 **점수 바탕 검은 상자** 치기다.

- 기울기가 있어야 하지 않다(기울기를 가리는 막이를 잡아낸다)
- 그 자리에 몰린 아무 뒤지기 꾀를 쓴다
- 잃음을 가장 크게 하도록 네모 헝겊을 거듭 고친다

검은 상자 치기를 넣는 일이 종요롭다. 막이가 기울기를 가리면 APGD은 듣지 않아도 네모 치기는 먹힐 수 있다.

## 따지는 절차

오토어택은 치기를 차례로 돌린다. 치기가 끝날 때마다 살아남은 보기(아직 틀리게 갈리지 않은 보기)만 다음 치기로 넘긴다.

```
온 시험 꾸러미 → APGD-CE → 살아남음 → APGD-DLR → 살아남음 → FAB → 살아남음 → 네모 → 마지막 든든한 맞음
```

마지막 든든한 맞음은 치기 넷을 모두 이겨 낸 몫이다.

### 흔한 날임

```
처음 맞음:  87.20%
apgd-ce:           든든한 맞음: 53.40% (- 33.80%)
apgd-t (DLR):     든든한 맞음: 51.20% (- 2.20%)
fab-t:             든든한 맞음: 50.80% (- 0.40%)
square:            든든한 맞음: 50.60% (- 0.20%)
```

## PyTorch로 쓰기

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

# 골라 쓰는 치기 묶음
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
        노름 갈래('Linf' 또는 'L2')
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

## 오토어택을 쓸 때

### 늘 쓸 자리

- 글이나 보고서에 **든든한 맞음을 알릴** 때
- 잣대 잡힌 자료에서 **막이를 견줄** 때
- 내놓은 모형의 **마지막 따짐**

### 다른 길을 헤아릴 자리

- **빠르게 만들어 볼 때**: 거듭 고쳐 가는 데는 PGD-20이 훨씬 빠르다
- **익힘 안쪽 돌기**: 맞서며 익히기에는 너무 느리다
- **남다른 으름 얼개**: 오토어택은 여느 $\ell_p$ 공을 가정한다

## 낱낱의 치기와 견주기

| 결 | PGD-100 | C&W | 오토어택 |
|--------|---------|-----|------------|
| 매개변수 없음 | 아니다 | 아니다 | **그렇다** |
| 기울기 가리기를 잡아냄 | 아니다 | 얼마쯤 | **그렇다** |
| 잃음 함수 여럿 | 아니다 | 아니다 | **그렇다** |
| 검은 상자 몫 | 아니다 | 아니다 | **그렇다** |
| 빠르기 | 빠름 | 느림 | 가운데 |
| 미더움 | 좋음 | 좋음 | **가장 좋음** |

## 간추림

| 결 | 자세한 것 |
|---------|--------|
| **뜻** | 미덥고 잣대 잡힌 든든함 따짐 |
| **이루는 몫** | APGD-CE, APGD-DLR, FAB, 네모 치기 |
| **고갱이 결** | 매개변수 없는 모둠 |
| **쓸 자리** | 마지막 따짐과 알리기 |
| **나은 점** | 기울기 가리기를 잡아내고 맞출 것이 없다 |

오토어택이 맞섬에 든든함 따짐의 으뜸 잣대가 된 까닭은 바로, 손으로 맞춘 치기 매개변수의 아리송함을 없애고 여러 치기 꾀를 두루 덮어 주기 때문이다.

## 살펴볼 거리

1. Croce, F., & Hein, M. (2020). "Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-Free Attacks." ICML.
2. Croce, F., & Hein, M. (2020). "Minimally Distorted Adversarial Examples with a Fast Adaptive Boundary Attack." ICML.
3. Andriushchenko, M., et al. (2020). "Square Attack: A Query-Efficient Black-Box Adversarial Attack via Random Search." ECCV.

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
