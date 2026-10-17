# Free Adversarial Training

**값싼 적대적 학습**(샤파히 등, 2019)은 적대적 학습의 가장 큰 목, 곧 계산 비용을 다룬다. 여느 PGD 적대적 학습은 안쪽 최대화에 배치마다 앞으로-되돌아 스텝이 $K$번 들어 여느 학습보다 7~10배 느리다. 값싼 적대적 학습은 그래디언트를 되써서 **여느 학습과 엇비슷한 값**으로 맞먹는 강건성을 이룬다.

---

## 1. 왜 하는가

여느 적대적 학습에서 계산의 흐름은 이렇다.

```
배치마다:
    1. PGD 안쪽 돌기(앞으로-되돌아 스텝 K번) → x_adv을 만든다
    2. x_adv으로 앞으로 스텝 → 잃음을 셈한다
    3. 되돌아 스텝 → θ을 고친다
```

값의 거의는 스텝1에 든다. 값싼 적대적 학습은 스텝1에서 셈한 그래디언트가 섭동 $\boldsymbol{\delta}$뿐 아니라 모델 파라메터 $\theta$을 고치는 데도 쓸모 있음을 알아챈다.

---

## 2. 수학 기초

### 핵심 깨침: 그래디언트 되쓰기

PGD에서는 섭동을 고치려 $\nabla_\mathbf{x} \mathcal{L}$을 계산한다. 그런데 사슬 규칙에 따라

$$
\nabla_\mathbf{x} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y)
$$

은 $f_\theta$을 거쳐 모델 파라메터에도 매인다. **같은 되돌아 스텝**으로 모델을 고칠 $\nabla_\theta \mathcal{L}$을 한꺼번에 셈할 수 있다.

### 값싼 적대적 학습 알고리즘

**알고리즘: 값싼 적대적 학습**

판마다 잔 배치를 $m$번 되풀이한다.

```
잔 배치 (x, y)마다 m번 되풀이:
    1. 로짓을 셈한다: z = f_θ(x + δ)
    2. 잃음을 셈한다: L = CrossEntropy(z, y)
    3. 되돌아 스텝 한 번 → ∇_x L과 ∇_θ L을 함께 얻는다
    4. 흔듦을 고친다: δ ← δ + ε · sign(∇_x L), 그다음 되비춘다
    5. 파라메터를 고친다: θ ← θ - η · ∇_θ L
```

섭동 $\boldsymbol{\delta}$은 $m$번의 되풀이에 걸쳐 이어지므로, 사실상 PGD $m$스텝을 밟으면서 모델도 $m$번 고친다.

### 판 수의 트레이드오프

여느 학습이 배치 크기 $B$으로 $E$판을 돈다면, 값싼 적대적 학습은 배치마다 $m$번 되풀이하며 $E/m$판을 돈다. 온 그래디언트 계산의 수는 여느 학습과 같지만 그래디언트마다 두 몫을 한다.

---

## 3. PyTorch로 짜기

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
    
    그래디언트를 흔듦 고침과 파라메터 고침에 함께 되써서
    여느 익힘과 엇비슷한 값으로 맞섬의 든든함을
    이룬다.
    
    Parameters
    ----------
    model : nn.Module
        익힐 모형
    epsilon : float
        흔듦 예산
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
        
        배치마다 m번 되풀이한다. 흔듦은 되풀이에 걸쳐
        이어지며 쌓인다.
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Free AT')
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            
            # 흔듦의 첫자리를 잡는다(m번 되풀이에 걸쳐 이어진다)
            delta = torch.zeros_like(x, requires_grad=False)
            
            for _ in range(self.m):
                # 이제의 흔듦을 건다
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
                
                # 흔듦을 고친다(∇_x으로)
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
                  f"잃음={metrics['loss']:.4f}, "
                  f"맞음={metrics['accuracy']:.2%}")
        
        return history
```

---

## 4. 계산 비용 비교

| 방법 | 배치마다 앞으로 스텝 | 배치마다 되돌아 스텝 | 견준 값 |
|--------|---------------------|----------------------|---------------|
| 여느 학습 | 1 | 1 | 1배 |
| PGD 적대적 학습($K=10$) | 11 | 11 | 약 10배 |
| 값싼 적대적 학습($m=8$) | 8 | 8 | 약 1.2배(고르게 나눔) |
| 빠른 적대적 학습 | 2 | 2 | 약 2배 |

값싼 적대적 학습은 값을 되풀이에 고르게 나누어 PGD 적대적 학습보다 약 8배 빠르다.

---

## 5. 강건성 피처와

CIFAR-10, $\varepsilon = 8/255$일 때:

| 방법 | 학습 때 | 클린 정확도 | 강건 정확도(PGD-20) |
|--------|-------------|-----------|---------------------|
| 여느 것 | 1배 | 95% | 0% |
| PGD 적대적 학습 | 10배 | 85% | 48% |
| 값싼 적대적 학습($m=8$) | 약 1.2배 | 83% | 43% |

값싼 적대적 학습은 강건성을 조금 내주고 계산을 크게 아낀다.

---

## 6. 한계

- **강건성이 조금 여림**: PGD 적대적 학습보다 강건 정확도가 약 3~5% 낮다
- **무너지듯 지나친 맞춤의 위험**: 강건성이 갑자기 무너질 수 있다
- **하이퍼파라미터에 예민함**: 되풀이 값 $m$을 조심스레 골라야 한다

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

이 절에서는 왜 하는가, 수학 기초, PyTorch로 짜기, 계산 비용 비교를 살펴보았다.

**살펴볼 거리**

1. Shafahi, A., et al. (2019). "Adversarial Training for Free!" NeurIPS.
