# 적대적 학습

**적대적 학습**은 적대적 공격을 막는 가장 효과적인 방법이다. 학습에 적대적 샘플을 섞어, 학습 데이터 점마다 그 둘레의 $\varepsilon$ 공 안에서 강건하도록 모델을 가르친다. 이 절은 표준 적대적 학습(PGD-AT), TRADES, MART와 실전에서 고려할 점을 다룬다.

---

## 1. 수학 기초

### 표준 학습과 강건한 학습

**표준 경험적 위험 최소화(ERM):**

$$
\min_\theta \mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{D}} \left[ \mathcal{L}(f_\theta(\mathbf{x}), y) \right]
$$

이는 평균적인 성능을 최적화하지만 적대적 섭동은 고려하지 않는다.

**강건 최적화(적대적 학습):**

$$
\min_\theta \mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{D}} \left[ \max_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y) \right]
$$

이는 **min-max**(최소-최대) 문제다.

- **안쪽 최대화**: 가장 나쁜 섭동을 찾는다(공격)
- **바깥 최소화**: 그 섭동에 강건하도록 학습한다

### 해석

학습 샘플 $(\mathbf{x}, y)$마다

1. 로스를 가장 크게 하는 적대적 섭동 $\boldsymbol{\delta}^*$을 찾는다
2. $\mathbf{x} + \boldsymbol{\delta}^*$의 로스가 가장 작아지도록 파라메터를 고친다

모델은 $\mathbf{x}$뿐 아니라 $\mathbf{x}$ 둘레 $\varepsilon$ 공 전체를 올바르게 분류하도록 배운다.

### PGD 기반 적대적 학습

안쪽 최대화는 정확히 풀 수 없으므로 PGD로 근사한다.

$$
\boldsymbol{\delta}^* \approx \text{PGD}(\mathbf{x}, y, \varepsilon, \alpha, K)
$$

**알고리즘: PGD 적대적 학습**

```
판마다:
    잔 배치 (x, y)마다:
        1. 맞서는 보기를 만든다: x_adv = PGD(x, y, ε, α, K)
        2. 로스를 셈한다: L = CrossEntropy(f_θ(x_adv), y)
        3. 파라메터를 고친다: θ ← θ - η∇_θ L
```

---

## 2. TRADES: 이론에 근거한 트레이드오프

### 왜 하는가

표준 적대적 학습은 클린 정확도를 너무 많이 희생한다. **TRADES**(장 등, 2019)는 클린 정확도와 강건 정확도의 트레이드오프를 명시적으로 조절한다.

### 형태

TRADES는 강건 로스를 두 항으로 나눈다.

$$
\mathcal{L}_{\text{TRADES}} = \mathcal{L}_{\text{CE}}(f_\theta(\mathbf{x}), y) + \beta \cdot \text{KL}(f_\theta(\mathbf{x}) \| f_\theta(\mathbf{x}_{\text{adv}}))
$$

여기서

- 첫째 항: 표준 크로스 엔트로피(클린 정확도)
- 둘째 항: 클린 입력의 예측과 적대적 입력의 예측 사이의 KL 발산(국소적인 매끄러움)
- $\beta$: 트레이드오프 파라메터(흔히 1~6)

### 직관

- **클린 로스**: 원본 데이터에서 예측이 맞도록 한다
- **KL 항**: 클린 입력과 섭동된 입력에서 예측이 **일관되도록** 이끈다
- 둘을 합치면: 강건하면서도 정확한 예측

### 표준 적대적 학습과의 차이

| 항목 | 표준 적대적 학습 | TRADES |
|--------|-------------|--------|
| 로스의 타깃 | 적대적 샘플만 | 원본 + 일관성 |
| 트레이드오프 조절 | 암묵적($\varepsilon$으로) | 명시적($\beta$로) |
| 클린 정확도 | 낮음 | 높음 |
| 강건 정확도 | 높음 | 조금 낮음 |

---

## 3. MART: 오분류를 고려한 강건 학습

### 왜 하는가

모든 샘플이 똑같이 중요하지는 않다. **MART**(왕 등, 2020)는 오분류된 샘플에 더 집중한다.

### 형태

$$
\mathcal{L}_{\text{MART}} = \text{BCE}(f_\theta(\mathbf{x}_{\text{adv}}), y) + \lambda \cdot (1 - p_y(\mathbf{x})) \cdot \text{KL}(f_\theta(\mathbf{x}) \| f_\theta(\mathbf{x}_{\text{adv}}))
$$

여기서

- $p_y(\mathbf{x})$는 클린 입력에서 참 클래스의 확률
- $(1 - p_y(\mathbf{x}))$는 이미 어려운 샘플에 가중치를 더 준다

### 직관

- 오분류된 샘플($p_y$가 낮음)이 정칙화 항에서 더 큰 가중치를 받는다
- 올바르게 분류된 샘플($p_y$가 높음)은 더 작은 가중치를 받는다
- 방어 역량을 가장 필요한 곳에 모은다

---

## 4. PyTorch로 구현하기

### 표준 적대적 학습

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
        섭동 예산(기본값: CIFAR-10에서 8/255)
    alpha : float
        PGD 스텝 크기
    num_iter : int
        익히는 동안의 PGD 회귀
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
            
            # 맞서는 보기로 앞으로 스텝
            optimizer.zero_grad()
            logits = self.model(x_adv)
            loss = F.cross_entropy(logits, y)
            
            # 되돌아 스텝
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
        """맑은 정확도와 강건한 정확도를 따진다."""
        self.model.eval()
        
        clean_correct = 0
        robust_correct = 0
        total = 0
        
        # 따질 때는 더 센 치기를 쓴다
        original_iter = self.num_iter
        self.num_iter = attack_iter
        
        for x, y in tqdm(test_loader, desc='Evaluating'):
            x, y = x.to(self.device), y.to(self.device)
            
            # 맑은 정확도
            with torch.no_grad():
                clean_pred = self.model(x).argmax(1)
                clean_correct += (clean_pred == y).sum().item()
            
            # 강건한 정확도
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
            
            print(f"  원본: {eval_metrics['clean_accuracy']:.2%}, "
                  f"강건성: {eval_metrics['robust_accuracy']:.2%}")
            
            # 가장 좋은 것을 담는다
            if save_path and eval_metrics['robust_accuracy'] > best_robust:
                best_robust = eval_metrics['robust_accuracy']
                torch.save(self.model.state_dict(), save_path)
                print(f"  가장 좋은 모형을 담았다(강건성: {best_robust:.2%})")
            
            scheduler.step()
        
        return history

class TRADESTrainer(AdversarialTrainer):
    """
    TRADES: 이론에 닿는 맞바꿈.
    
    로스: L_CE(f(x), y) + β · KL(f(x) || f(x_adv))
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
            
            # 맑은 앞으로 스텝
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
            
            # TRADES 로스
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

### 사용 예

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

## 5. 실전에서 고려할 점

### 하이퍼파라미터

| 파라메터 | 표준 적대적 학습 | TRADES | 비고 |
|-----------|-------------|--------|-------|
| $\varepsilon$ | 8/255 | 8/255 | 섭동 예산 |
| $\alpha$ | 2/255 | 2/255 | 스텝 크기 |
| $K$(학습) | 7~10 | 10 | PGD 반복 횟수 |
| $K$(평가) | 20~100 | 20~100 | 평가에는 더 센 공격 |
| $\beta$ | 해당 없음 | 1~6 | TRADES 트레이드오프 |
| 러닝레이트 | 0.1 | 0.1 | 점차 줄인다 |
| 에폭 수 | 100~200 | 100~200 | 표준 학습보다 많다 |

### 계산 비용

적대적 학습은 표준 학습보다 **7~10배 느리다**.

- 배치마다 PGD에 순전파-역전파가 $K$번 든다
- PGD를 10스텝 쓰면 그래디언트 계산이 약 10배 는다

### 흔한 문제

1. **강건 과적합(robust overfitting)**: 학습이 길어지면(흔히 러닝레이트를 처음 줄인 뒤) 테스트 강건 정확도가 떨어진다
   - 해결: 강건 정확도를 지켜보며 조기 종료한다
   - FGSM 한 스텝으로 학습할 때 강건 정확도가 갑자기 0 가까이 무너지는 현상은 따로 **급격한 과적합**(catastrophic overfitting)이라 부른다(Fast AT 절에서 다룬다)

2. **클린-강건 트레이드오프**: 강건한 모델은 클린 정확도가 낮다
   - 예상되는 폭: 클린 정확도가 약 5~15% 떨어진다
   - 트레이드오프를 조절하려면 TRADES를 쓴다

3. **학습에 쓴 공격에 대한 과적합**: PGD에는 강건하나 다른 공격에는 그렇지 않다
   - 해결: 여러 공격으로 평가한다(AutoAttack)

### 대표적인 결과(CIFAR-10)

| 방법 | 클린 정확도 | 강건 정확도(PGD-20) |
|--------|-----------|---------------------|
| 표준 학습 | 95% | 0% |
| 표준 적대적 학습 | 85% | 48% |
| TRADES (β=6) | 87% | 46% |
| 최신 기법 | 90% | 60% |

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

| 방법 | 식 | 핵심 특징 |
|--------|---------|-------------|
| **표준 적대적 학습** | $\min_\theta \mathbb{E}[\max_\delta \mathcal{L}(f(\mathbf{x}+\boldsymbol{\delta}), y)]$ | 적대적 샘플에서의 최대 로스 |
| **TRADES** | $\mathcal{L}_{\text{CE}} + \beta \cdot \text{KL}$ | 정확도 트레이드오프를 명시적으로 조절 |
| **MART** | $(1-p_y)$로 가중치를 줌 | 어려운 샘플에 집중 |

계산 비용이 크고 클린 정확도를 희생해야 하는데도, 강건한 모델을 얻는 데는 적대적 학습이 여전히 표준 방법이다.

**살펴볼 거리**

1. Madry, A., et al. (2018). "Towards Deep Learning Models Resistant to Adversarial Attacks." ICLR.
2. Zhang, H., et al. (2019). "Theoretically Principled Trade-off between Robustness and Accuracy." ICML.
3. Wang, Y., et al. (2020). "Improving Adversarial Robustness Requires Revisiting Misclassified Examples." ICLR.
