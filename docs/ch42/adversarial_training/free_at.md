# Free Adversarial Training

**Free AT**(Free Adversarial Training, 샤파히 등, 2019)는 적대적 학습의 가장 큰 병목, 곧 계산 비용을 다룬다. 표준 PGD 적대적 학습은 안쪽 최대화에 배치마다 순전파-역전파가 $K$번 들어 표준 학습보다 7~10배 느리다. Free AT는 그래디언트를 재사용해 **표준 학습과 엇비슷한 비용**으로 그에 버금가는 강건성을 얻는다.

---

## 1. 왜 하는가

표준 적대적 학습에서 계산의 흐름은 이렇다.

```
배치마다:
    1. PGD 안쪽 돌기(앞으로-되돌아 스텝 K번) → x_adv을 만든다
    2. x_adv으로 앞으로 스텝 → 로스를 셈한다
    3. 되돌아 스텝 → θ을 고친다
```

비용의 대부분은 1단계에 든다. Free AT는 1단계에서 계산한 그래디언트가 섭동 $\boldsymbol{\delta}$뿐 아니라 모델 파라메터 $\theta$를 갱신하는 데도 쓸모 있다는 점에 착안한다.

---

## 2. 수학 기초

### 핵심 통찰: 그래디언트 재사용

PGD에서는 섭동을 갱신하려고 $\nabla_\mathbf{x} \mathcal{L}$을 계산한다. 그런데 연쇄 법칙에 따라

$$
\nabla_\mathbf{x} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y)
$$

을 구하는 역전파는 $f_\theta$를 거쳐 모델 파라메터까지 지나간다. 그래서 **같은 역전파 한 번**으로 모델을 갱신할 $\nabla_\theta \mathcal{L}$도 함께 얻을 수 있다.

### Free AT 알고리즘

**알고리즘: Free AT**

에폭마다 각 미니배치를 $m$번 되풀이한다.

```
잔 배치 (x, y)마다 m번 되풀이:
    1. 로짓을 셈한다: z = f_θ(x + δ)
    2. 로스를 셈한다: L = CrossEntropy(z, y)
    3. 되돌아 스텝 한 번 → ∇_x L과 ∇_θ L을 함께 얻는다
    4. 섭동을 고친다: δ ← δ + ε · sign(∇_x L), 그다음 되비춘다
    5. 파라메터를 고친다: θ ← θ - η · ∇_θ L
```

섭동 $\boldsymbol{\delta}$는 $m$번의 반복에 걸쳐 이어지므로, 사실상 PGD $m$스텝을 밟으면서 모델도 $m$번 갱신한다.

### 에폭 수의 트레이드오프

표준 학습이 배치 크기 $B$로 $E$에폭을 돈다면, Free AT는 배치마다 $m$번 반복하며 $E/m$에폭을 돈다. 전체 그래디언트 계산 횟수는 표준 학습과 같지만 그래디언트 하나가 두 가지 역할을 한다.

---

## 3. PyTorch로 구현하기

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, List
from tqdm import tqdm

class FreeAdversarialTrainer:
    """
    값싼 맞서며 익히기.
    
    그래디언트를 섭동 고침과 파라메터 고침에 함께 되써서
    여느 익힘과 엇비슷한 값으로 적대적 강건성을
    이룬다.
    
    Parameters
    ----------
    model : nn.Module
        익힐 모형
    epsilon : float
        섭동 예산
    m : int
        잔 배치를 되풀이하는 횟수(배치마다의 PGD 스텝)
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 8/255,
        m: int = 8,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.epsilon = epsilon
        self.m = m
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer
    ) -> Dict[str, float]:
        """
        값싼 맞서며 익히기로 한 판 익힌다.
        
        배치마다 m번 되풀이한다. 섭동은 되풀이에 걸쳐
        이어지며 쌓인다.
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Free AT')
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            
            # 섭동의 첫자리를 잡는다(m번 되풀이에 걸쳐 이어진다)
            delta = torch.zeros_like(x, requires_grad=False)
            
            for _ in range(self.m):
                # 이제의 섭동을 건다
                x_adv = torch.clamp(x + delta, 0, 1)
                x_adv.requires_grad_(True)
                
                # 앞으로 스텝
                logits = self.model(x_adv)
                loss = F.cross_entropy(logits, y)
                
                # 되돌아 스텝 한 번: ∇_x과 ∇_θ을 함께 셈한다
                optimizer.zero_grad()
                loss.backward()
                
                # 모형 파라메터를 고친다(∇_θ으로)
                optimizer.step()
                
                # 섭동을 고친다(∇_x으로)
                with torch.no_grad():
                    grad = x_adv.grad.data
                    delta = delta + self.epsilon * grad.sign()
                    delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                
                total_loss += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                total += len(y)
            
            pbar.set_postfix({
                'loss': f'{total_loss/total:.4f}',
                'acc': f'{correct/total:.2%}'
            })
        
        return {
            'loss': total_loss / total,
            'accuracy': correct / total
        }
    
    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        optimizer: Optional[optim.Optimizer] = None
    ) -> List[Dict]:
        """
        온전한 익힘 돌기.
        
        붙임말: 배치마다 m번 다루므로
        판 수는 epochs/m으로 돈다.
        """
        if optimizer is None:
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=0.1, momentum=0.9, weight_decay=5e-4
            )
        
        # 되풀이 값에 맞춰 판 수를 손본다
        adjusted_epochs = max(epochs // self.m, 1)
        
        history = []
        for epoch in range(1, adjusted_epochs + 1):
            metrics = self.train_epoch(train_loader, optimizer)
            history.append(metrics)
            print(f"{epoch}/{adjusted_epochs}판: "
                  f"로스={metrics['loss']:.4f}, "
                  f"정확도={metrics['accuracy']:.2%}")
        
        return history
```

---

## 4. 계산 비용 비교

| 방법 | 배치마다 순전파 | 배치마다 역전파 | 상대 비용 |
|--------|---------------------|----------------------|---------------|
| 표준 학습 | 1 | 1 | 1배 |
| PGD 적대적 학습($K=10$) | 11 | 11 | 약 10배 |
| Free AT($m=8$) | 8 | 8 | 약 1.2배(에폭 수를 $1/m$로 줄여 나눔) |
| Fast AT | 2 | 2 | 약 2배 |

Free AT는 반복 비용을 에폭 수를 줄여 상쇄하므로 PGD 적대적 학습보다 약 8배 빠르다.

---

## 5. 강건성 결과

CIFAR-10, $\varepsilon = 8/255$일 때:

| 방법 | 학습 시간 | 클린 정확도 | 강건 정확도(PGD-20) |
|--------|-------------|-----------|---------------------|
| 표준 학습 | 1배 | 95% | 0% |
| PGD 적대적 학습 | 10배 | 85% | 48% |
| Free AT($m=8$) | 약 1.2배 | 83% | 43% |

Free AT는 강건성을 조금 희생하고 계산을 크게 아낀다.

---

## 6. 한계

- **강건성이 조금 약함**: PGD 적대적 학습보다 강건 정확도가 약 3~5% 낮다
- **급격한 과적합(catastrophic overfitting)의 위험**: 강건성이 갑자기 무너질 수 있다
- **하이퍼파라미터에 예민함**: 반복 횟수 $m$을 신중하게 골라야 한다

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

Free AT는 역전파 한 번으로 얻은 그래디언트를 섭동 갱신과 파라메터 갱신에 함께 써서, 표준 학습과 엇비슷한 비용으로 PGD 적대적 학습에 가까운 강건성을 얻는다. 그 대가는 조금 낮은 강건 정확도와 반복 횟수 $m$에 대한 민감성이다.

**살펴볼 거리**

1. Shafahi, A., et al. (2019). "Adversarial Training for Free!" NeurIPS.
