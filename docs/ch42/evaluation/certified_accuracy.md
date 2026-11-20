# 인증 정확도

**인증 정확도**는 주어진 예산 안의 어떤 섭동에도 모델의 예측이 올바르게 유지됨을 증명할 수 있는 테스트 샘플의 비율이다. 공격의 세기에 좌우되는 경험적 강건 정확도와 달리, 인증 정확도는 참 강건성의 보장된 아래끝을 준다.

---

## 1. 형식적 정의

### 반지름 r에서 인증 정확도

$$
\text{Certified Acc}(r) = \frac{1}{N} \sum_{i=1}^N \mathbf{1}\left[f(\mathbf{x}_i) = y_i \text{ and } R(\mathbf{x}_i) \geq r\right]
$$

여기서 $R(\mathbf{x}_i)$는 샘플 $i$의 인증 반지름이다.

### 다른 지표와의 관계

$$
\text{Certified Acc}(r) \leq \text{True Robust Acc}(r) \leq \text{Empirical Robust Acc}(r)
$$

- **인증 정확도**는 아래끝이다. 실제로 강건한 예측도 인증하지 못할 수 있다
- **경험적 강건 정확도**는 위끝이다. 공격이 가장 나쁜 적대적 샘플을 못 찾을 수 있다
- 둘의 차이가 인증 방법이 얼마나 느슨한지("인증 격차")를 잰다

---

## 2. 인증 정확도 계산

### 무작위 평활화에서

```python
import torch
from typing import Dict, List

def compute_certified_accuracy(
    smoother,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    radii: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
    n: int = 10000,
    alpha: float = 0.001
) -> Dict[str, float]:
    """
    여러 반지름에서 밝혀 낸 정확도를 셈한다.
    
    Parameters
    ----------
    smoother : RandomizedSmoothing
        밝힐 수 있는 매끄럽게 한 분류기
    test_images, test_labels : torch.Tensor
        시험 자료 꾸러미
    radii : list[float]
        밝혀 낸 정확도를 셈할 반지름
    n : int
        밝히는 데 쓸 몬테카를로 표본 수
    alpha : float
        신뢰 수준
    
    Returns
    -------
    results : 반지름을 밝혀 낸 정확도에 맞춘 사전
    """
    num_examples = len(test_images)
    predictions = []
    certified_radii = []
    
    for i in range(num_examples):
        pred, cert_radius = smoother.certify(
            test_images[i], n=n, alpha=alpha
        )
        predictions.append(pred)
        certified_radii.append(cert_radius)
    
    predictions = torch.tensor(predictions)
    certified_radii = torch.tensor(certified_radii)
    correct = (predictions == test_labels)
    
    results = {'clean_accuracy': correct.float().mean().item()}
    
    for r in radii:
        certified_at_r = correct & (certified_radii >= r)
        results[f'certified_r={r}'] = certified_at_r.float().mean().item()
    
    if correct.any():
        results['avg_radius'] = certified_radii[correct].mean().item()
    else:
        results['avg_radius'] = 0.0
    
    return results
```

### IBP/CROWN에서

```python
def certified_accuracy_ibp(model, test_loader, epsilon, device='cuda'):
    """IBP 테두리로 밝혀 낸 정확도를 셈한다."""
    certified = 0
    correct = 0
    total = 0
    
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        lb, ub = compute_ibp_bounds(model, x, epsilon)
        
        with torch.no_grad():
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
        
        true_lb = lb.gather(1, y.view(-1, 1)).squeeze()
        ub_copy = ub.clone()
        ub_copy.scatter_(1, y.view(-1, 1), float('-inf'))
        max_other_ub = ub_copy.max(dim=1)[0]
        
        is_certified = (true_lb > max_other_ub) & (pred == y)
        certified += is_certified.sum().item()
        total += len(y)
    
    return {
        'clean_accuracy': correct / total,
        'certified_accuracy': certified / total
    }
```

---

## 3. 벤치마크 결과

### CIFAR-10(L2, 무작위 평활화)

| 방법 | $\sigma$ | $r{=}0.25$에서 인증 정확도 | $r{=}0.5$에서 인증 정확도 | $r{=}1.0$에서 인증 정확도 |
|--------|----------|---------------------|--------------------|--------------------|
| 코언 등 | 0.25 | 60% | 43% | — |
| 살만 등 | 0.25 | 68% | 49% | — |
| 코언 등 | 0.50 | 54% | 41% | 26% |
| 살만 등 | 0.50 | 59% | 44% | 32% |

### CIFAR-10(L-무한, IBP/CROWN)

| 방법 | $\varepsilon$ | 인증 정확도 |
|--------|--------------|-------------------|
| IBP | 2/255 | 33% |
| CROWN-IBP | 2/255 | 38% |
| IBP | 8/255 | 7% |
| CROWN-IBP | 8/255 | 12% |

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

| 지표 | 보장 | 계산 비용 | 타이트함 |
|--------|-----------|------|-----------|
| 경험적 강건 정확도 | 없음 | 낮음~중간 | 위끝 |
| 인증 정확도(무작위 평활화) | 확률적 | 높음 | 중간 |
| 인증 정확도(IBP) | 결정론적 | 낮음 | 헐거움 |
| 인증 정확도(CROWN) | 결정론적 | 중간 | 더 타이트함 |

인증 정확도는 가장 엄밀한 강건성 지표로, 섭동 예산 안의 어떤 공격에도 버티는 보장을 준다.

**살펴볼 거리**

1. Cohen, J., Rosenfeld, E., & Kolter, Z. (2019). "Certified Adversarial Robustness via Randomized Smoothing." ICML.
2. Gowal, S., et al. (2019). "Scalable Verified Training for Provably Robust Image Classification." ICCV.
3. Salman, H., et al. (2019). "Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers." NeurIPS.
