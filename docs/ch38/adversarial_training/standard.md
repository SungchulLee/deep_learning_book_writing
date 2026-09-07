# 맞서며 익히기
## 들머리

**맞서며 익히기**은 맞서는 치기를 막는 가장 잘 듣는 길이다. 익힘에 맞서는 보기를 섞어, 익힘 점마다 $\varepsilon$ 공 안에서 든든하도록 모형을 가르친다. 이 마디는 여느 맞서며 익히기(PGD-AT), TRADES, MART과 참으로 헤아릴 것을 다룬다.

## 수학 밑바탕

### 여느 익힘과 든든한 익힘

**여느 겪은 무릅씀 가장 작게 하기(ERM):**

$$
\min_\theta \mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{D}} \left[ \mathcal{L}(f_\theta(\mathbf{x}), y) \right]
$$

이는 여느 자리의 됨됨이를 다듬지만 맞서는 흔듦을 못 본 척한다.

**든든하게 다듬기(맞서며 익히기):**

$$
\min_\theta \mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{D}} \left[ \max_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y) \right]
$$

이는 **가장 작게-가장 크게** 문제다.

- **안쪽 가장 크게 하기**: 가장 나쁜 흔듦을 찾는다(치기)
- **바깥 가장 작게 하기**: 그에 든든하도록 익힌다

### 풀이

익힘 보기 $(\mathbf{x}, y)$마다

1. 잃음을 가장 크게 하는 맞서는 흔듦 $\boldsymbol{\delta}^*$을 찾는다
2. $\mathbf{x} + \boldsymbol{\delta}^*$의 잃음이 가장 작아지도록 매개변수를 고친다

모형은 $\mathbf{x}$뿐 아니라 $\mathbf{x}$ 둘레 $\varepsilon$ 공 전체를 옳게 가르도록 배운다.

### PGD에 기댄 맞서며 익히기

안쪽 가장 크게 하기를 다룰 수 없으므로 PGD으로 어림한다.

$$
\boldsymbol{\delta}^* \approx \text{PGD}(\mathbf{x}, y, \varepsilon, \alpha, K)
$$

**알고리즘: PGD 맞서며 익히기**

```
판마다:
    잔 묶음 (x, y)마다:
        1. 맞서는 보기를 만든다: x_adv = PGD(x, y, ε, α, K)
        2. 잃음을 셈한다: L = CrossEntropy(f_θ(x_adv), y)
        3. 매개변수를 고친다: θ ← θ - η∇_θ L
```

## TRADES: 이론에 닿는 맞바꿈

### 왜 하는가

여느 맞서며 익히기는 맑은 맞음을 너무 많이 내준다. **TRADES**(장 등, 2019)은 맑은 맞음과 든든한 맞음의 맞바꿈을 드러내 놓고 저울질한다.

### 꼴

TRADES은 든든한 잃음을 쪼갠다.

$$
\mathcal{L}_{\text{TRADES}} = \mathcal{L}_{\text{CE}}(f_\theta(\mathbf{x}), y) + \beta \cdot \text{KL}(f_\theta(\mathbf{x}) \| f_\theta(\mathbf{x}_{\text{adv}}))
$$

여기서

- 첫째 항: 여느 엇갈린 엔트로피(맑은 맞음)
- 둘째 항: 맑은 미루어 봄과 맞서는 미루어 봄 사이의 KL 갈림(그 자리의 매끄러움)
- $\beta$: 맞바꿈 매개변수(흔히 1~6)

### 느낌으로 알기

- **맑은 잃음**: 본디 자료에서 미루어 봄이 맞도록 한다
- **KL 항**: 맑은 들임과 흔든 들임에서 미루어 봄이 **한결같도록** 이끈다
- 함께: 든든하면서도 맞는 미루어 봄

### 여느 맞서며 익히기와 다른 고갱이

| 결 | 여느 맞서며 익히기 | TRADES |
|--------|-------------|--------|
| 잃음의 과녁 | 맞서는 보기만 | 맑음 + 한결같음 |
| 맞바꿈 다루기 | 넌지시($\varepsilon$으로) | 드러내 놓고($\beta$으로) |
| 맑은 맞음 | 낮음 | 높음 |
| 든든한 맞음 | 높음 | 조금 낮음 |

## MART: 잘못 가름을 아는 든든한 익히기

### 왜 하는가

보기가 다 똑같이 종요롭지는 않다. **MART**(왕 등, 2020)은 잘못 가른 보기에 더 힘을 모은다.

### 꼴

$$
\mathcal{L}_{\text{MART}} = \text{BCE}(f_\theta(\mathbf{x}_{\text{adv}}), y) + \lambda \cdot (1 - p_y(\mathbf{x})) \cdot \text{KL}(f_\theta(\mathbf{x}) \| f_\theta(\mathbf{x}_{\text{adv}}))
$$

여기서

- $p_y(\mathbf{x})$은 맑은 들임에서 참 갈래의 낌새
- $(1 - p_y(\mathbf{x}))$은 이미 어려운 보기에 짐을 더 준다

### 느낌으로 알기

- 잘못 가른 보기($p_y$이 낮음)가 다독임에서 더 큰 짐을 받는다
- 옳게 가른 보기($p_y$이 높음)는 더 작은 짐을 받는다
- 막이의 힘을 가장 있어야 할 곳에 모은다

## PyTorch로 짜기

### 여느 맞서며 익히기

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

## 참으로 헤아릴 것

### 하이퍼파라미터

| 매개변수 | 여느 맞서며 익히기 | TRADES | 붙임말 |
|-----------|-------------|--------|-------|
| $\varepsilon$ | 8/255 | 8/255 | 흔듦 예산 |
| $\alpha$ | 2/255 | 2/255 | 걸음 크기 |
| $K$(익힘) | 7~10 | 10 | PGD 되돌이 |
| $K$(따짐) | 20~100 | 20~100 | 따질 때는 더 세게 |
| $\beta$ | 해당 없음 | 1~6 | TRADES 맞바꿈 |
| 배움 비율 | 0.1 | 0.1 | 줄여 가며 |
| 판 수 | 100~200 | 100~200 | 여느 익힘보다 많다 |

### 셈 값

맞서며 익히기는 여느 익힘보다 **7~10배 느리다**.

- 묶음마다 PGD에 앞으로-되돌아 걸음이 $K$번 든다
- 흔히 PGD 10걸음이면 기울기 셈이 10배 는다

### 흔한 탈

1. **무너지듯 지나친 맞춤**: 든든한 맞음이 갑자기 떨어진다
   - 풀이: 든든한 맞음을 지켜보고 일찍 멈춘다

2. **맑음-든든함 맞바꿈**: 든든한 모형은 맑은 맞음이 낮다
   - 그럴 만하다: 맑은 맞음이 약 5~15% 떨어진다
   - 맞바꿈을 다루려면 TRADES을 쓴다

3. **익힘 치기에 지나치게 맞춤**: PGD에는 든든하나 다른 치기에는 아니다
   - 풀이: 치기 여럿으로 따진다(오토어택)

### 그럴 법한 결과(CIFAR-10)

| 방법 | 맑은 맞음 | 든든한 맞음(PGD-20) |
|--------|-----------|---------------------|
| 여느 익힘 | 95% | 0% |
| 여느 맞서며 익히기 | 85% | 48% |
| TRADES (β=6) | 87% | 46% |
| 가장 앞선 것 | 90% | 60% |

## 간추림

| 방법 | 식 | 고갱이 결 |
|--------|---------|-------------|
| **여느 맞서며 익히기** | $\min_\theta \mathbb{E}[\max_\delta \mathcal{L}(f(\mathbf{x}+\boldsymbol{\delta}), y)]$ | 맞서는 보기의 가장 큰 잃음 |
| **TRADES** | $\mathcal{L}_{\text{CE}} + \beta \cdot \text{KL}$ | 맞음 맞바꿈을 드러내 놓음 |
| **MART** | $(1-p_y)$으로 짐을 줌 | 어려운 보기에 힘을 모음 |

셈이 더 들고 맞음을 내주어야 하는데도, 든든한 모형을 얻는 데는 맞서며 익히기가 여전히 으뜸 잣대다.

## 살펴볼 거리

1. Madry, A., et al. (2018). "Towards Deep Learning Models Resistant to Adversarial Attacks." ICLR.
2. Zhang, H., et al. (2019). "Theoretically Principled Trade-off between Robustness and Accuracy." ICML.
3. Wang, Y., et al. (2020). "Improving Adversarial Robustness Requires Revisiting Misclassified Examples." ICLR.

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
