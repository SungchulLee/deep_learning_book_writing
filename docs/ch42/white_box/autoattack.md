# AutoAttack

**AutoAttack**(Croce & Hein, 2020)은 적대적 강건성을 믿을 만하게 평가하려고 설계한, 파라메터 없는 앙상블 공격이다. 서로 보완하는 공격 전략 여럿을 묶어, 그래디언트 마스킹(그래디언트를 쓸모없게 가려 공격을 헛돌게 하는 현상)이나 잘못 고른 하이퍼파라미터 탓에 강건하다고 잘못 보고할 위험을 가장 작게 한다. AutoAttack은 적대적 강건성 문헌에서 강건 정확도를 보고하는 사실상의 표준이 되었다.

---

## 1. 왜 하는가

처음에는 강건해 보이던 방어가 더 세거나 더 정교하게 맞춘 공격으로 평가하니 무너진 일이 많다. 흔한 원인은 이렇다.

- **그래디언트 마스킹**: 실제로 강건해진 것이 아니라 그래디언트만 감춘 방어
- **잘못 고른 하이퍼파라미터**: 스텝이 너무 적거나 스텝 크기가 틀린 PGD
- **로스 함수 하나**: 크로스 엔트로피가 모든 방어에 가장 잘 통하는 공격 목표는 아니다

AutoAttack은 **하이퍼파라미터를 맞출 필요 없이** 서로 다른 공격의 앙상블을 써서 이를 푼다.

---

## 2. 구성 요소

AutoAttack은 서로 보완하는 공격 넷을 차례로 돌린다.

### 1. APGD-CE(크로스 엔트로피를 쓰는 Auto-PGD)

**적응형 스텝 크기**를 쓰도록 개선한 PGD다.

$$
\alpha^{(t+1)} = \begin{cases}
\alpha^{(t)} / 2 & \text{if loss hasn't improved in } w \text{ steps} \\
\alpha^{(t)} & \text{otherwise}
\end{cases}
$$

일반 PGD와 달라진 핵심은 이렇다.

- **모멘텀**: $\mathbf{z}^{(t+1)} = \rho \cdot \mathbf{z}^{(t)} + \nabla_\mathbf{x} \mathcal{L}$
- **체크포인트로 되돌리기**: 스텝 크기를 줄일 때 그때까지 찾은 가장 좋은 점으로 돌아간다
- **스텝 크기를 맞출 필요 없음**: 로스의 이력을 보고 자동으로 맞춘다

### 2. APGD-DLR(로짓 차이 비율을 쓰는 Auto-PGD)

크로스 엔트로피 대신 **로짓 차이 비율**(Difference of Logits Ratio, DLR) 로스를 쓴다.

$$
\mathcal{L}_{\text{DLR}} = -\frac{z_y - \max_{i \neq y} z_i}{z_{\pi_1} - z_{\pi_3}}
$$

여기서 $z_{\pi_1} \geq z_{\pi_2} \geq \ldots$는 크기순으로 정렬한 로짓이다. DLR 로스는

- 로짓을 평행 이동하거나 양수배로 스케일을 바꿔도 값이 그대로다(분모가 로짓 사이의 간격으로 정규화한다)
- 확신도가 높은 예측 근처에서 크로스 엔트로피가 포화해 그래디언트가 사라지는 문제를 피한다
- 확신도를 높게 유지하는 방어에 더 잘 통한다

### 3. FAB(Fast Adaptive Boundary)

결정 경계로 반복해서 투영하여 작은 섭동을 찾는 **최소 노름 공격**이다.

- 현재 점을 선형화한 결정 경계로 투영한다
- 여러 타깃 클래스를 탐색한다
- 예산이 고정된 PGD의 섭동보다 작은 섭동을 찾는 일이 잦다

방어의 로스 지형이 매끄럽지 않을 때 FAB가 특히 쓸모 있다.

### 4. Square Attack(블랙박스)

무작위로 고른 정사각형 영역의 섭동을 쓰는 **스코어 기반 블랙박스** 공격이다.

- 그래디언트가 필요 없다(그래디언트를 가리는 방어를 잡아낸다)
- 국소화된 무작위 탐색 전략을 쓴다
- 로스를 가장 크게 하도록 정사각형 패치를 반복해서 고친다

블랙박스 공격을 넣는 일이 중요하다. 방어가 그래디언트를 가리면 APGD는 통하지 않아도 Square Attack은 통할 수 있다.

---

## 3. 평가하는 절차

AutoAttack은 공격을 차례로 돌린다. 공격이 끝날 때마다 살아남은 샘플(아직 오분류되지 않은 샘플)만 다음 공격으로 넘긴다.

```
온 시험 꾸러미 → APGD-CE → 살아남음 → APGD-DLR → 살아남음 → FAB → 살아남음 → 네모 → 마지막 강건한 정확도
```

마지막 강건 정확도는 공격 넷을 모두 이겨 낸 샘플의 비율이다.

### 흔한 출력

아래는 `autoattack` 패키지가 찍는 로그의 형식을 보이는 예시이며, 이 쪽에서 잰 값이 아니다.

```
처음 정확도:  87.20%
apgd-ce:           강건한 정확도: 53.40% (- 33.80%)
apgd-t (DLR):     강건한 정확도: 51.20% (- 2.20%)
fab-t:             강건한 정확도: 50.80% (- 0.40%)
square:            강건한 정확도: 50.60% (- 0.20%)
```

---

## 4. PyTorch로 쓰기

### 표준 평가

```python
# 깔기: pip install autoattack
from autoattack import AutoAttack

# 여느 평가(치기 4개)
adversary = AutoAttack(
    model, 
    norm='Linf', 
    eps=8/255, 
    version='standard'
)
x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=100)

# 빠른 평가(APGD-CE + APGD-DLR만)
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

### 평가 흐름에 붙이기

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
    AutoAttack 평가를 돌린다.
    
    Parameters
    ----------
    model : nn.Module
        따질 모형
    x, y : torch.Tensor
        시험 들임과 이름표
    epsilon : float
        섭동 예산
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
    
    # 맑은 정확도
    with torch.no_grad():
        device = next(model.parameters()).device
        clean_pred = model(x.to(device)).argmax(dim=1)
        clean_acc = (clean_pred == y.to(device)).float().mean().item()
    
    # AutoAttack
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

## 5. AutoAttack을 쓸 때

### 늘 쓸 자리

- 논문이나 보고서에 **강건 정확도를 보고할** 때
- 표준 벤치마크에서 **방어를 비교할** 때
- 배포할 모델의 **최종 평가**

### 다른 길을 헤아릴 자리

- **프로토타이핑**: 빠르게 고쳐 가며 볼 때는 PGD-20이 훨씬 빠르다
- **학습의 안쪽 루프**: 적대적 학습에 쓰기에는 너무 느리다
- **표준이 아닌 위협 모델**: AutoAttack은 표준 $\ell_p$ 공을 가정한다

---

## 6. 개별 공격과 비교

| 항목 | PGD-100 | C&W | AutoAttack |
|--------|---------|-----|------------|
| 파라메터 없음 | 아니다 | 아니다 | **그렇다** |
| 그래디언트 마스킹을 잡아냄 | 아니다 | 일부 | **그렇다** |
| 로스 함수 여럿 | 아니다 | 아니다 | **그렇다** |
| 블랙박스 공격 포함 | 아니다 | 아니다 | **그렇다** |
| 속도 | 빠름 | 느림 | 중간 |
| 신뢰성 | 좋음 | 좋음 | **가장 좋음** |

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
선형 분류기 $f(x) = w^T x + b$에서 예측한 클래스를 바꾸는 데 드는 가장 작은 $\ell_\infty$ 섭동을 계산하여라. 이것이 뉴럴 네트워크의 강건성과 어떻게 이어지는지 설명하여라.

</div>

??? success "풀이"
    선형 분류기에서 $\ell_\infty$ 노름으로 잰 결정 경계까지의 거리는 $\frac{|w^T x + b|}{\|w\|_1}$이다. 가장 작은 섭동은 $\delta^* = \frac{|w^T x + b|}{\|w\|_1} \cdot \text{sign}(w)$이다. 뉴럴 네트워크에서는 국소 선형 근사 $f(x + \delta) \approx f(x) + \nabla_x f \cdot \delta$가 FGSM(그래디언트의 부호를 쓴다)이 왜 잘 통하는지를 설명해 준다. 차원이 높은 모델이 취약한 까닭은 $\|w\|_1$은 차원과 함께 커지는데 $|w^T x + b|$은 꼭 그렇지 않아 강건성의 여유가 줄어들기 때문이다. $\square$

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
    두 클래스의 서포트가 거리 $\epsilon$ 안에서 겹치면(곧 $\|x_1 - x_2\|_\infty \leq 2\epsilon$인 $x_1 \in \text{클래스 1}, x_2 \in \text{클래스 2}$가 있으면), $x_1$과 $x_2$ 둘 다에서 강건한 분류기는 적어도 하나를 틀리게 가를 수밖에 없다(섭동 공이 겹치기 때문이다). 이것이 정확도와 강건성의 근본적인 트레이드오프다. 겹치는 서포트의 비중이 피할 수 없는 정확도 손실을 정한다. 자연 이미지 분포에서는 $\epsilon = 8/255$에서 겹침이 꽤 있어, 관찰되는 10~15%의 정확도 하락을 설명해 준다. 모델 용량을 키우면(더 넓은 네트워크) 복잡한 강건 결정 경계를 더 잘 그려 트레이드오프를 어느 정도 완화할 수 있다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
금융 머신러닝 시스템(사기 탐지나 거래 신호 생성 등)에서 적대적 강건성이 어떻게 드러나는지 논하여라. 위협 모델이 비전과 어떻게 다른가?

</div>

??? success "풀이"
    금융에서 적대자는 탐지 시스템에 맞추어 전략을 바꾸는 교묘한 집단(사기꾼, 시장 조작자)이다. 비전과 다른 핵심은 이렇다. (1) 섭동할 수 있는 피처가 경제적으로 그럴듯한 범위로 제약된다(사기꾼이 자기 거래 이력을 통째로 바꿀 수는 없다). (2) 공격이 순차적이고 적응적이다(적대자가 시스템의 응답을 보고 고친다). (3) 거짓 양성과 거짓 음성의 비용이 비대칭이다(정상 거래를 막는 것과 사기를 놓치는 것). (4) $\ell_p$ 노름은 의미가 없고 도메인에 맞는 섭동 모델이 있어야 한다. 방어는 적응형 적대자에게도 강건해야 하므로, 탐지 기준이 알려지면 회피할 수 있는 탐지 기반 방법은 쓸모가 크게 줄어든다. $\square$

## 정리하며

| 항목 | 내용 |
|---------|--------|
| **목적** | 믿을 만하고 표준화된 강건성 평가 |
| **구성 요소** | APGD-CE, APGD-DLR, FAB, Square Attack |
| **핵심 특징** | 파라메터 없는 앙상블 |
| **용도** | 최종 평가와 보고 |
| **나은 점** | 그래디언트 마스킹을 잡아내고 맞출 하이퍼파라미터가 없다 |

AutoAttack이 적대적 강건성 평가의 표준이 된 까닭은, 손으로 맞춘 공격 파라메터의 불확실성을 없애고 여러 공격 전략을 두루 덮어 주기 때문이다.

**살펴볼 거리**

1. Croce, F., & Hein, M. (2020). "Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-Free Attacks." ICML.
2. Croce, F., & Hein, M. (2020). "Minimally Distorted Adversarial Examples with a Fast Adaptive Boundary Attack." ICML.
3. Andriushchenko, M., et al. (2020). "Square Attack: A Query-Efficient Black-Box Adversarial Attack via Random Search." ECCV.
