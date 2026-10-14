# 적대적 학습

**적대적 학습**은 적대적 공격을 막는 가장 잘 듣는 길이다. 학습에 적대적 샘플을 섞어, 학습 점마다 $\varepsilon$ 공 안에서 든든하도록 모델을 가르친다. 이 마디는 여느 적대적 학습(PGD-AT), TRADES, MART과 참으로 헤아릴 것을 다룬다.

---

## 1. 수학 기초

### 여느 학습과 강건한 학습

**여느 겪은 위험 최소화(ERM):**

$$
\min_\theta \mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{D}} \left[ \mathcal{L}(f_\theta(\mathbf{x}), y) \right]
$$

이는 여느 자리의 성능을 다듬지만 적대적 섭동을 못 본 척한다.

**든든하게 다듬기(적대적 학습):**

$$
\min_\theta \mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{D}} \left[ \max_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y) \right]
$$

이는 **가장 작게-가장 크게** 문제다.

- **안쪽 최대화**: 가장 나쁜 섭동을 찾는다(공격)
- **바깥 최소화**: 그에 든든하도록 학습된이다

### 풀이

학습 보기 $(\mathbf{x}, y)$마다

1. 손실을 가장 크게 하는 적대적 섭동 $\boldsymbol{\delta}^*$을 찾는다
2. $\mathbf{x} + \boldsymbol{\delta}^*$의 손실이 가장 작아지도록 매개변수를 고친다

모델은 $\mathbf{x}$뿐 아니라 $\mathbf{x}$ 둘레 $\varepsilon$ 공 전체를 옳게 가르도록 배운다.

### PGD에 기댄 적대적 학습

안쪽 최대화를 다룰 수 없으므로 PGD으로 어림한다.

$$
\boldsymbol{\delta}^* \approx \text{PGD}(\mathbf{x}, y, \varepsilon, \alpha, K)
$$

**알고리즘: PGD 적대적 학습**

```
판마다:
    잔 배치 (x, y)마다:
        1. 맞서는 보기를 만든다: x_adv = PGD(x, y, ε, α, K)
        2. 잃음을 셈한다: L = CrossEntropy(f_θ(x_adv), y)
        3. 매개변수를 고친다: θ ← θ - η∇_θ L
```

---

## 2. TRADES: 이론에 닿는 트레이드오프

### 왜 하는가

여느 적대적 학습은 클린 정확도를 너무 많이 내준다. **TRADES**(장 등, 2019)은 클린 정확도와 강건 정확도의 트레이드오프를 드러내 놓고 저울질한다.

### 꼴

TRADES은 강건한 손실을 쪼갠다.

$$
\mathcal{L}_{\text{TRADES}} = \mathcal{L}_{\text{CE}}(f_\theta(\mathbf{x}), y) + \beta \cdot \text{KL}(f_\theta(\mathbf{x}) \| f_\theta(\mathbf{x}_{\text{adv}}))
$$

여기서

- 첫째 항: 여느 엇갈린 엔트로피(클린 정확도)
- 둘째 항: 맑은 추론과 맞서는 추론 사이의 KL 갈림(그 자리의 매끄러움)
- $\beta$: 트레이드오프 매개변수(흔히 1~6)

### 느낌으로 알기

- **맑은 손실**: 본디 데이터에서 추론이 맞도록 한다
- **KL 항**: 맑은 입력과 흔든 입력에서 추론이 **한결같도록** 이끈다
- 함께: 든든하면서도 맞는 추론

### 여느 적대적 학습과 다른 핵심

| 특징 | 여느 적대적 학습 | TRADES |
|--------|-------------|--------|
| 손실의 타깃 | 적대적 샘플만 | 맑음 + 한결같음 |
| 트레이드오프 다루기 | 넌지시($\varepsilon$으로) | 드러내 놓고($\beta$으로) |
| 클린 정확도 | 낮음 | 높음 |
| 강건 정확도 | 높음 | 조금 낮음 |

---

## 3. MART: 잘못 분류를 아는 강건한 학습

### 왜 하는가

보기가 다 똑같이 종요롭지는 않다. **MART**(왕 등, 2020)은 잘못 가른 보기에 더 힘을 모은다.

### 꼴

$$
\mathcal{L}_{\text{MART}} = \text{BCE}(f_\theta(\mathbf{x}_{\text{adv}}), y) + \lambda \cdot (1 - p_y(\mathbf{x})) \cdot \text{KL}(f_\theta(\mathbf{x}) \| f_\theta(\mathbf{x}_{\text{adv}}))
$$

여기서

- $p_y(\mathbf{x})$은 맑은 입력에서 참 갈래의 낌새
- $(1 - p_y(\mathbf{x}))$은 이미 어려운 보기에 짐을 더 준다

### 느낌으로 알기

- 잘못 가른 보기($p_y$이 낮음)가 다독임에서 더 큰 짐을 받는다
- 옳게 가른 보기($p_y$이 높음)는 더 작은 짐을 받는다
- 방어의 힘을 가장 있어야 할 곳에 모은다

---

## 4. PyTorch로 짜기

### 여느 적대적 학습

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, List
from tqdm import tqdm

class AdversarialTrainer:
    """
    PGD에 기댄 여느 맞서며 익히기.
    
    푸는 문제: min_θ E[ max_{||δ||≤ε} L(f_θ(x+δ), y) ]
    
    Parameters
    ----------
    model : nn.Module
        익힐 모형
    epsilon : float
        흔듦 예산(기본값: CIFAR-10에서 8/255)
    alpha : float
        PGD 걸음 크기
    num_iter : int
        익히는 동안의 PGD 되돌이
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 8/255,
        alpha: float = 2/255,
        num_iter: int = 10,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
    
    def _pgd_attack(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """PGD 맞서는 보기를 만든다."""
        x_adv = x + torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)
        
        for _ in range(self.num_iter):
            x_adv.requires_grad_(True)
            loss = F.cross_entropy(self.model(x_adv), y)
            
            self.model.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                x_adv = x_adv + self.alpha * x_adv.grad.sign()
                delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
                x_adv = torch.clamp(x + delta, 0, 1)
        
        return x_adv.detach()
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer
    ) -> Dict[str, float]:
        """한 판 익힌다."""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            
            # 맞서는 보기를 만든다
            x_adv = self._pgd_attack(x, y)
            
            # 맞서는 보기로 앞으로 걸음
            optimizer.zero_grad()
            logits = self.model(x_adv)
            loss = F.cross_entropy(logits, y)
            
            # 되돌아 걸음
            loss.backward()
            optimizer.step()
            
            # 자를 좇는다
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
    
    def evaluate(
        self,
        test_loader: DataLoader,
        attack_iter: int = 20
    ) -> Dict[str, float]:
        """맑은 맞음과 든든한 맞음을 따진다."""
        self.model.eval()
        
        clean_correct = 0
        robust_correct = 0
        total = 0
        
        # 따질 때는 더 센 치기를 쓴다
        original_iter = self.num_iter
        self.num_iter = attack_iter
        
        for x, y in tqdm(test_loader, desc='Evaluating'):
            x, y = x.to(self.device), y.to(self.device)
            
            # 맑은 맞음
            with torch.no_grad():
                clean_pred = self.model(x).argmax(1)
                clean_correct += (clean_pred == y).sum().item()
            
            # 든든한 맞음
            x_adv = self._pgd_attack(x, y)
            with torch.no_grad():
                robust_pred = self.model(x_adv).argmax(1)
                robust_correct += (robust_pred == y).sum().item()
            
            total += len(y)
        
        self.num_iter = original_iter
        
        return {
            'clean_accuracy': clean_correct / total,
            'robust_accuracy': robust_correct / total
        }
    
    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """온전한 익힘 돌기."""
        if optimizer is None:
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=0.1, momentum=0.9, weight_decay=5e-4
            )
        
        if scheduler is None:
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[int(0.5*epochs), int(0.75*epochs)],
                gamma=0.1
            )
        
        history = {
            'train_loss': [], 'train_acc': [],
            'clean_acc': [], 'robust_acc': []
        }
        
        best_robust = 0
        
        for epoch in range(1, epochs + 1):
            print(f"\n{epoch}/{epochs}판")
            
            # 익힌다
            train_metrics = self.train_epoch(train_loader, optimizer)
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            
            # 따진다
            eval_metrics = self.evaluate(test_loader)
            history['clean_acc'].append(eval_metrics['clean_accuracy'])
            history['robust_acc'].append(eval_metrics['robust_accuracy'])
            
            print(f"  맑음: {eval_metrics['clean_accuracy']:.2%}, "
                  f"든든함: {eval_metrics['robust_accuracy']:.2%}")
            
            # 가장 좋은 것을 담는다
            if save_path and eval_metrics['robust_accuracy'] > best_robust:
                best_robust = eval_metrics['robust_accuracy']
                torch.save(self.model.state_dict(), save_path)
                print(f"  가장 좋은 모형을 담았다(든든함: {best_robust:.2%})")
            
            scheduler.step()
        
        return history

class TRADESTrainer(AdversarialTrainer):
    """
    TRADES: 이론에 닿는 맞바꿈.
    
    잃음: L_CE(f(x), y) + β · KL(f(x) || f(x_adv))
    """
    
    def __init__(self, *args, beta: float = 6.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer
    ) -> Dict[str, float]:
        """TRADES 익힘 한 판."""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='TRADES Training')
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            
            # 맑은 앞으로 걸음
            logits_clean = self.model(x)
            loss_natural = F.cross_entropy(logits_clean, y)
            
            # 맞서는 보기를 만든다(KL을 가장 크게)
            x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)
            
            for _ in range(self.num_iter):
                x_adv.requires_grad_(True)
                with torch.no_grad():
                    p_clean = F.softmax(logits_clean, dim=1)
                
                loss_kl = F.kl_div(
                    F.log_softmax(self.model(x_adv), dim=1),
                    p_clean,
                    reduction='batchmean'
                )
                
                self.model.zero_grad()
                loss_kl.backward()
                
                with torch.no_grad():
                    x_adv = x_adv + self.alpha * x_adv.grad.sign()
                    delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
                    x_adv = torch.clamp(x + delta, 0, 1)
            
            # TRADES 잃음
            logits_adv = self.model(x_adv)
            loss_robust = F.kl_div(
                F.log_softmax(logits_adv, dim=1),
                F.softmax(logits_clean.detach(), dim=1),
                reduction='batchmean'
            )
            
            loss = loss_natural + self.beta * loss_robust
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(y)
            correct += (logits_clean.argmax(1) == y).sum().item()
            total += len(y)
            
            pbar.set_postfix({'loss': f'{total_loss/total:.4f}'})
        
        return {'loss': total_loss / total, 'accuracy': correct / total}
```

### 쓰는 보기

```python
import torchvision
import torchvision.transforms as transforms

# CIFAR-10을 얹는다
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

# 모형을 만든다
model = torchvision.models.resnet18(num_classes=10)

# 여느 맞서며 익히기
trainer = AdversarialTrainer(model, epsilon=8/255)
history = trainer.train(train_loader, test_loader, epochs=100, save_path='robust_model.pth')

# 또는 TRADES을 쓴다
trainer_trades = TRADESTrainer(model, epsilon=8/255, beta=6.0)
history = trainer_trades.train(train_loader, test_loader, epochs=100)
```

---

## 5. 참으로 헤아릴 것

### 하이퍼파라미터

| 매개변수 | 여느 적대적 학습 | TRADES | 붙임말 |
|-----------|-------------|--------|-------|
| $\varepsilon$ | 8/255 | 8/255 | 섭동 예산 |
| $\alpha$ | 2/255 | 2/255 | 걸음 크기 |
| $K$(학습) | 7~10 | 10 | PGD 되돌이 |
| $K$(따짐) | 20~100 | 20~100 | 따질 때는 더 세게 |
| $\beta$ | 해당 없음 | 1~6 | TRADES 트레이드오프 |
| 학습률 | 0.1 | 0.1 | 줄여 가며 |
| 판 수 | 100~200 | 100~200 | 여느 학습보다 많다 |

### 계산 비용

적대적 학습은 여느 학습보다 **7~10배 느리다**.

- 배치마다 PGD에 앞으로-되돌아 걸음이 $K$번 든다
- 흔히 PGD 10걸음이면 기울기 계산이 10배 는다

### 흔한 탈

1. **무너지듯 지나친 맞춤**: 강건 정확도가 갑자기 떨어진다
   - 풀이: 강건 정확도를 지켜보고 일찍 멈춘다

2. **맑음-강건성 트레이드오프**: 강건한 모델은 클린 정확도가 낮다
   - 그럴 만하다: 클린 정확도가 약 5~15% 떨어진다
   - 트레이드오프를 다루려면 TRADES을 쓴다

3. **학습 공격에 지나치게 맞춤**: PGD에는 든든하나 다른 공격에는 아니다
   - 풀이: 공격 여럿으로 따진다(오토어택)

### 그럴 법한 특징과(CIFAR-10)

| 방법 | 클린 정확도 | 강건 정확도(PGD-20) |
|--------|-----------|---------------------|
| 여느 학습 | 95% | 0% |
| 여느 적대적 학습 | 85% | 48% |
| TRADES (β=6) | 87% | 46% |
| 가장 앞선 것 | 90% | 60% |

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
선형 분류기 $f(x) = w^T x + b$에서 예측한 갈래를 바꾸는 데 드는 가장 작은 $\ell_\infty$ 섭동을 계산하여라. 이것이 신경망의 강건성과 어떻게 이어지는지 밝혀라.

</div>

??? success "연습문제 1 풀이"
    선형 분류기에서 $\ell_\infty$ 노름으로 잰 결정 경계까지의 거리는 $\frac{|w^T x + b|}{\|w\|_1}$이다. 가장 작은 섭동은 $\delta^* = \frac{|w^T x + b|}{\|w\|_1} \cdot \text{sign}(w)$이다. 신경망에서는 그 자리의 선형 어림 $f(x + \delta) \approx f(x) + \nabla_x f \cdot \delta$이 FGSM(기울기의 부호를 쓴다)이 왜 잘 듣는지를 밝혀 준다. 차원이 높은 모델이 무른 까닭은 $\|w\|_1$은 차원과 함께 커지는데 $|w^T x + b|$은 꼭 그렇지 않아 강건성의 여유가 줄어들기 때문이다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 마디에서 다룬 공격나 방어를 CIFAR-10의 ResNet-18 모델에 짜 넣어라. $\epsilon = 8/255$의 PGD-20 공격 아래에서 클린 정확도와 강건 정확도를 알려라.

</div>

??? success "연습문제 2 풀이"
    여느 ResNet-18은 클린 정확도가 $\sim$93%이지만 PGD-20($\epsilon = 8/255$, 걸음 크기 $2/255$) 아래의 강건 정확도는 $\sim$0%이다. 이 마디의 방법을 걸면 특징과는 재주에 따라 다르다. 적대적 학습은 클린 정확도 $\sim$83%에 강건 정확도 $\sim$50%이고, 인증 가능한 방어는 더 낮지만 증명할 수 있는 테두리를 준다. 정확도와 강건성의 트레이드오프는 기초부터 있는 것이라, 강건성을 높이면 클린 정확도가 흔히 5~15% 든다. 아무렇게나 하는 시드 3개의 평균과 표준편차로 알려라. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
섭동 공 안에서 갈래별 데이터의 밑자리가 서로 겹친다고 볼 때, 모델이 담는 힘을 키우지 않고서는 어떤 방어도 맑은 데이터의 높은 정확도와 $\ell_\infty$ 섭동에 대한 높은 강건성을 함께 이룰 수 없음을 증명하여라.

</div>

??? success "연습문제 3 풀이"
    두 갈래의 밑자리가 거리 $\epsilon$ 안에서 겹치면(곧 $\|x_1 - x_2\|_\infty \leq 2\epsilon$인 $x_1 \in \text{갈래 1}, x_2 \in \text{갈래 2}$이 있으면), $x_1$과 $x_2$ 둘 다에서 강건한 분류기는 적어도 하나를 틀리게 가를 수밖에 없다(섭동 공이 겹치기 때문이다). 이것이 정확도와 강건성의 기초 트레이드오프다. 겹치는 밑자리의 몫이 피할 수 없는 정확도 손실을 정한다. 여느 그림 분포에서는 $\epsilon = 8/255$에서 겹침이 꽤 있어, 살펴본 10~15%의 정확도 떨어짐을 밝혀 준다. 모델이 담는 힘을 키우면(더 너른 네트워크) 얽힌 강건한 결정 경계를 더 잘 그려 트레이드오프를 얼마쯤 눅일 수 있다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
금융 머신러닝 구조(속임수 알아내기나 거래 신호 만들기 따위)에서 맞섬의 강건성이 어떻게 드러나는지 다루어라. 위협 모델이 보기 다룸과 어떻게 다른가?

</div>

??? success "연습문제 4 풀이"
    금융에서 겨루는 이는 알아내는 구조에 맞추어 스스로 움직이는 꾀 많은 무리(속임수꾼, 저자 흔드는 이)다. 보기 다룸과 다른 핵심은 이렇다. (1) 흔들 수 있는 밭이 돈으로 될 만한 것에 옭매인다(속임수꾼이 제 거래 자취를 통째로 바꿀 수는 없다). (2) 공격이 잇따르며 맞추어 간다(겨루는 이가 구조의 되받음을 보고 손본다). (3) 헛 정확도와 놓침의 값이 서로 어긋난다(옳은 거래를 막는 것과 속임수를 놓치는 것). (4) $\ell_p$ 노름은 뜻이 없고 밭에 맞는 섭동 모델이 있어야 한다. 방어는 맞추어 오는 겨루는 이에게도 든든해야 하므로, 알아내는 잣대가 알려지면 비껴갈 수 있는 알아내기 기초의 길은 많이 걸러진다. $\square$

## 정리하며

| 방법 | 식 | 핵심 특징 |
|--------|---------|-------------|
| **여느 적대적 학습** | $\min_\theta \mathbb{E}[\max_\delta \mathcal{L}(f(\mathbf{x}+\boldsymbol{\delta}), y)]$ | 적대적 샘플의 가장 큰 손실 |
| **TRADES** | $\mathcal{L}_{\text{CE}} + \beta \cdot \text{KL}$ | 정확도 트레이드오프를 드러내 놓음 |
| **MART** | $(1-p_y)$으로 짐을 줌 | 어려운 보기에 힘을 모음 |

계산이 더 들고 정확도를 내주어야 하는데도, 강건한 모델을 얻는 데는 적대적 학습이 여전히 으뜸 잣대다.

**살펴볼 거리**

1. Madry, A., et al. (2018). "Towards Deep Learning Models Resistant to Adversarial Attacks." ICLR.
2. Zhang, H., et al. (2019). "Theoretically Principled Trade-off between Robustness and Accuracy." ICML.
3. Wang, Y., et al. (2020). "Improving Adversarial Robustness Requires Revisiting Misclassified Examples." ICLR.
