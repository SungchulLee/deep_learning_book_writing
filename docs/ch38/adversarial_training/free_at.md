# 값싼 맞서며 익히기
## 들머리

**값싼 맞서며 익히기**(샤파히 등, 2019)은 맞서며 익히기의 가장 큰 목, 곧 셈 값을 다룬다. 여느 PGD 맞서며 익히기는 안쪽 가장 크게 하기에 묶음마다 앞으로-되돌아 걸음이 $K$번 들어 여느 익힘보다 7~10배 느리다. 값싼 맞서며 익히기는 기울기를 되써서 **여느 익힘과 엇비슷한 값**으로 맞먹는 든든함을 이룬다.

## 왜 하는가

여느 맞서며 익히기에서 셈의 흐름은 이렇다.

```
묶음마다:
    1. PGD 안쪽 돌기(앞으로-되돌아 걸음 K번) → x_adv을 만든다
    2. x_adv으로 앞으로 걸음 → 잃음을 셈한다
    3. 되돌아 걸음 → θ을 고친다
```

값의 거의는 1걸음에 든다. 값싼 맞서며 익히기는 1걸음에서 셈한 기울기가 흔듦 $\boldsymbol{\delta}$뿐 아니라 모형 매개변수 $\theta$을 고치는 데도 쓸모 있음을 알아챈다.

## 수학 밑바탕

### 고갱이 깨침: 기울기 되쓰기

PGD에서는 흔듦을 고치려 $\nabla_\mathbf{x} \mathcal{L}$을 셈한다. 그런데 사슬 규칙에 따라

$$
\nabla_\mathbf{x} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y)
$$

은 $f_\theta$을 거쳐 모형 매개변수에도 매인다. **같은 되돌아 걸음**으로 모형을 고칠 $\nabla_\theta \mathcal{L}$을 한꺼번에 셈할 수 있다.

### 값싼 맞서며 익히기 알고리즘

**알고리즘: 값싼 맞서며 익히기**

판마다 잔 묶음을 $m$번 되풀이한다.

```
잔 묶음 (x, y)마다 m번 되풀이:
    1. 로짓을 셈한다: z = f_θ(x + δ)
    2. 잃음을 셈한다: L = CrossEntropy(z, y)
    3. 되돌아 걸음 한 번 → ∇_x L과 ∇_θ L을 함께 얻는다
    4. 흔듦을 고친다: δ ← δ + ε · sign(∇_x L), 그다음 되비춘다
    5. 매개변수를 고친다: θ ← θ - η · ∇_θ L
```

흔듦 $\boldsymbol{\delta}$은 $m$번의 되풀이에 걸쳐 이어지므로, 사실상 PGD $m$걸음을 밟으면서 모형도 $m$번 고친다.

### 판 수의 맞바꿈

여느 익힘이 묶음 크기 $B$으로 $E$판을 돈다면, 값싼 맞서며 익히기는 묶음마다 $m$번 되풀이하며 $E/m$판을 돈다. 온 기울기 셈의 수는 여느 익힘과 같지만 기울기마다 두 몫을 한다.

## PyTorch으로 짜기

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
    
    기울기를 흔듦 고침과 매개변수 고침에 함께 되써서
    여느 익힘과 엇비슷한 값으로 맞섬의 든든함을
    이룬다.
    
    Parameters
    ----------
    model : nn.Module
        익힐 모형
    epsilon : float
        흔듦 예산
    m : int
        잔 묶음을 되풀이하는 횟수(묶음마다의 PGD 걸음)
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
        
        묶음마다 m번 되풀이한다. 흔듦은 되풀이에 걸쳐
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
                
                # 앞으로 걸음
                logits = self.model(x_adv)
                loss = F.cross_entropy(logits, y)
                
                # 되돌아 걸음 한 번: ∇_x과 ∇_θ을 함께 셈한다
                optimizer.zero_grad()
                loss.backward()
                
                # 모형 매개변수를 고친다(∇_θ으로)
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
        
        붙임말: 묶음마다 m번 다루므로
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

## 셈 값 견주기

| 방법 | 묶음마다 앞으로 걸음 | 묶음마다 되돌아 걸음 | 견준 값 |
|--------|---------------------|----------------------|---------------|
| 여느 익힘 | 1 | 1 | 1배 |
| PGD 맞서며 익히기($K=10$) | 11 | 11 | 약 10배 |
| 값싼 맞서며 익히기($m=8$) | 8 | 8 | 약 1.2배(고르게 나눔) |
| 빠른 맞서며 익히기 | 2 | 2 | 약 2배 |

값싼 맞서며 익히기는 값을 되풀이에 고르게 나누어 PGD 맞서며 익히기보다 약 8배 빠르다.

## 든든함 결과

CIFAR-10, $\varepsilon = 8/255$일 때:

| 방법 | 익힘 때 | 맑은 맞음 | 든든한 맞음(PGD-20) |
|--------|-------------|-----------|---------------------|
| 여느 것 | 1배 | 95% | 0% |
| PGD 맞서며 익히기 | 10배 | 85% | 48% |
| 값싼 맞서며 익히기($m=8$) | 약 1.2배 | 83% | 43% |

값싼 맞서며 익히기는 든든함을 조금 내주고 셈을 크게 아낀다.

## 한계

- **든든함이 조금 여림**: PGD 맞서며 익히기보다 든든한 맞음이 약 3~5% 낮다
- **무너지듯 지나친 맞춤의 무릅씀**: 든든함이 갑자기 무너질 수 있다
- **하이퍼파라미터에 예민함**: 되풀이 값 $m$을 조심스레 골라야 한다

## 살펴볼 거리

1. Shafahi, A., et al. (2019). "Adversarial Training for Free!" NeurIPS.

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
