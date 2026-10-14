# Carlini-Wagner(C&W) 공격

**칼리니-와그너(C&W) 공격**은 적대적 샘플 만들기를 제약 없는 최적화 문제로 다시 적은, 최적화 기초의 적대적 공격이다. 칼리니와 와그너(2017)가 내놓았으며, 기울기 기반의 방법보다 훨씬 세고, 여전히 틀리게 가르게 하면서도 더 작은 섭동을 찾아내는 일이 잦다.

---

## 1. 수학 기초

### 문제 다시 적기

섭동의 크기를 곧바로 제약하는 FGSM/PGD과 달리 C&W은 공격을 이렇게 적는다.

$$
\min_{\boldsymbol{\delta}} \|\boldsymbol{\delta}\|_p + c \cdot f(\mathbf{x} + \boldsymbol{\delta})
$$

여기서

- $\|\boldsymbol{\delta}\|_p$은 섭동의 노름(흔히 $p=2$)
- $c > 0$은 트레이드오프 붙박이
- $f(\cdot)$은 틀리게 가르게 이끄는 목표 함수

### 로짓에 기댄 목표

C&W은 로짓 $Z(\mathbf{x})$(소프트맥스 앞의 출력)에 기대어 공들여 꾸민 목표를 쓴다.

**비표적 공격:**

$$
f(\mathbf{x}') = \max\left(\max_{i \neq y} Z(\mathbf{x}')_i - Z(\mathbf{x}')_y, -\kappa\right)
$$

**표적 공격(타깃 갈래 $t$):**

$$
f(\mathbf{x}') = \max\left(Z(\mathbf{x}')_y - Z(\mathbf{x}')_t, -\kappa\right)
$$

여기서 $\kappa \geq 0$은 **자신함 매개변수**다.

- $\kappa = 0$: 틀리게 가르기만 하면 된다
- $\kappa > 0$: 자신함의 여유를 두고 틀리게 가른다

### 변수 바꾸기

상자 제약 $\mathbf{x}' \in [0, 1]^d$을 다루려고 C&W은 다음을 쓴다.

$$
\mathbf{x}' = \frac{1}{2}(\tanh(\mathbf{w}) + 1)
$$

여기서 $\mathbf{w}$은 제약 없는 최적화 변수다. 이러면

- $\tanh(\mathbf{w}) \in (-1, 1)$
- $\mathbf{x}' \in (0, 1)$이 절로 지켜진다

섭동은 이렇게 된다.

$$
\boldsymbol{\delta} = \frac{1}{2}(\tanh(\mathbf{w}) + 1) - \mathbf{x}
$$

### 마지막 최적화

$\mathbf{w}$에 대해 가장 좋게 한다.

$$
\min_{\mathbf{w}} \left\|\frac{1}{2}(\tanh(\mathbf{w}) + 1) - \mathbf{x}\right\|_2^2 + c \cdot f\left(\frac{1}{2}(\tanh(\mathbf{w}) + 1)\right)
$$

### c을 두 쪽 갈라 찾기

가장 좋은 $c$은 미리 알 수 없다. C&W은 두 쪽 갈라 찾기를 쓴다.

1. 첫자리: $c_{\text{low}} = 0$, $c_{\text{high}} = 10^{10}$
2. 걸음마다:
   - $c = (c_{\text{low}} + c_{\text{high}}) / 2$으로 둔다
   - 최적화를 돌린다
   - 공격이 먹히면: $c_{\text{high}} = c$(더 작게 해 본다)
   - 공격이 안 먹히면: $c_{\text{low}} = c$(더 크게 해 본다)
3. 먹힌 것 가운데 가장 작은 $c$을 돌려준다

---

## 2. 알고리즘

**알고리즘: C&W L2 공격**

**입력:** 맑은 보기 $\mathbf{x}$, 레이블 $y$, 모델 $f_\theta$

**출력:** 적대적 샘플 $\mathbf{x}_{\text{adv}}$

1. $\frac{1}{2}(\tanh(\mathbf{w}) + 1) = \mathbf{x}$이 되도록 $\mathbf{w}$의 첫자리를 잡는다
2. $c$을 두 쪽 갈라 찾는다:
   - $c$마다:
     - $\mathbf{w}$에 Adam을 $T$번 돌린다
     - 가장 좋은 적대적 샘플을 좇는다(먹히는 것 가운데 $\|\boldsymbol{\delta}\|_2$이 가장 작은 것)
3. 가장 좋은 적대적 샘플을 돌려준다

---

## 3. PyTorch로 짜기

```python
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict

class CarliniWagnerL2:
    """
    칼리니-와그너 L2 치기.
    
    틀리게 가르게 하면서 L2 흔듦을 가장 작게 하는
    가장 좋게 하기 바탕의 치기다.
    
    Parameters
    ----------
    model : nn.Module
        칠 신경 그물
    c : float
        처음 맞바꿈 붙박이
    kappa : float
        자신함 매개변수(0 = 틀리게 가르기만)
    learning_rate : float
        Adam의 배움 비율
    max_iter : int
        가장 많은 가장 좋게 하기 되돌이
    binary_search_steps : int
        c을 두 쪽 갈라 찾는 걸음 수
    """
    
    def __init__(
        self,
        model: nn.Module,
        c: float = 1.0,
        kappa: float = 0.0,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        binary_search_steps: int = 9,
        initial_const: float = 1e-3,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.c = c
        self.kappa = kappa
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.binary_search_steps = binary_search_steps
        self.initial_const = initial_const
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.device = device or next(model.parameters()).device
        
        self.model.eval()
        self.model.to(self.device)
    
    def _to_tanh_space(self, x: torch.Tensor) -> torch.Tensor:
        """[0,1]에서 tanh 밭으로 옮긴다."""
        # x = 0.5 * (tanh(w) + 1) => w = arctanh(2x - 1)
        x_scaled = x * 2 - 1  # [-1, 1]으로 잣대를 잡는다
        x_scaled = torch.clamp(x_scaled, -0.999999, 0.999999)  # 무한을 비껴간다
        return torch.atanh(x_scaled)
    
    def _from_tanh_space(self, w: torch.Tensor) -> torch.Tensor:
        """tanh 밭에서 [0,1]으로 옮긴다."""
        return 0.5 * (torch.tanh(w) + 1)
    
    def _f_objective(
        self,
        x_adv: torch.Tensor,
        y: torch.Tensor,
        targeted: bool,
        target_labels: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        C&W 목표 함수 f(x')을 셈한다.
        
        돌려주는 값은
        - 치기가 아직 안 먹히면 양수
        - 치기가 먹히면 음수(또는 0)
        """
        logits = self.model(x_adv)
        
        # 참 갈래의 로짓을 얻는다
        true_logit = logits.gather(1, y.view(-1, 1)).squeeze(1)
        
        if targeted:
            # 과녁 있음: 과녁 로짓 > 참 로짓이 되게 한다
            target_logit = logits.gather(1, target_labels.view(-1, 1)).squeeze(1)
            f = torch.clamp(true_logit - target_logit, min=-self.kappa)
        else:
            # 과녁 없음: 가장 큰 다른 로짓 > 참 로짓이 되게 한다
            # 참 갈래의 가리개를 만든다
            mask = torch.ones_like(logits).scatter_(1, y.view(-1, 1), 0)
            other_logits = logits * mask - (1 - mask) * 1e9
            max_other_logit = other_logits.max(dim=1)[0]
            f = torch.clamp(true_logit - max_other_logit, min=-self.kappa)
        
        return f
    
    def _optimize(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        c: float,
        targeted: bool,
        target_labels: Optional[torch.Tensor]
    ) -> tuple:
        """c 값을 붙박아 두고 가장 좋게 한다."""
        batch_size = x.shape[0]
        
        # tanh 밭에서 w의 첫자리를 잡는다
        w = self._to_tanh_space(x).clone().detach().requires_grad_(True)
        
        optimizer = optim.Adam([w], lr=self.learning_rate)
        
        best_adv = x.clone()
        best_l2 = torch.full((batch_size,), float('inf'), device=self.device)
        
        for _ in range(self.max_iter):
            optimizer.zero_grad()
            
            # 그림 밭으로 옮긴다
            x_adv = self._from_tanh_space(w)
            
            # 흔듦의 L2 노름을 셈한다
            delta = x_adv - x
            l2_dist = delta.view(batch_size, -1).norm(p=2, dim=1)
            
            # C&W 목표를 셈한다
            f = self._f_objective(x_adv, y, targeted, target_labels)
            
            # 온 잃음: L2 + c * f
            loss = l2_dist.sum() + c * f.sum()
            
            loss.backward()
            optimizer.step()
            
            # 가장 좋은 맞서는 보기를 좇는다
            with torch.no_grad():
                # 어느 보기에서 먹혔는지 살핀다
                logits = self.model(x_adv)
                pred = logits.argmax(dim=1)
                
                if targeted:
                    success = (pred == target_labels)
                else:
                    success = (pred != y)
                
                # 먹혔고 L2이 더 작으면 가장 좋은 것을 고친다
                improved = success & (l2_dist < best_l2)
                best_adv[improved] = x_adv[improved]
                best_l2[improved] = l2_dist[improved]
        
        return best_adv, best_l2
    
    def generate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        targeted: bool = False,
        target_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        C&W 치기로 맞서는 보기를 만든다.
        
        Parameters
        ----------
        x : torch.Tensor
            맑은 그림
        y : torch.Tensor
            참 이름표
        targeted : bool
            과녁 있는 치기를 한다
        target_labels : torch.Tensor, 골라 씀
            과녁 있는 치기의 과녁 이름표
            
        Returns
        -------
        x_adv : torch.Tensor
            맞서는 보기
        """
        x = x.to(self.device)
        y = y.to(self.device)
        if targeted and target_labels is not None:
            target_labels = target_labels.to(self.device)
        
        batch_size = x.shape[0]
        
        # 두 쪽 갈라 찾기의 테두리를 잡는다
        c_low = torch.zeros(batch_size, device=self.device)
        c_high = torch.full((batch_size,), 1e10, device=self.device)
        
        # 통틀어 가장 좋은 것을 좇는다
        best_adv = x.clone()
        best_l2 = torch.full((batch_size,), float('inf'), device=self.device)
        
        # c을 두 쪽 갈라 찾는다
        for step in range(self.binary_search_steps):
            if step == 0:
                c = torch.full((batch_size,), self.initial_const, device=self.device)
            else:
                c = (c_low + c_high) / 2
            
            c_mean = c.mean().item()
            
            # 가장 좋게 하기를 돌린다
            adv, l2 = self._optimize(x, y, c_mean, targeted, target_labels)
            
            # 가장 좋은 것을 고친다
            improved = l2 < best_l2
            best_adv[improved] = adv[improved]
            best_l2[improved] = l2[improved]
            
            # 먹혔는지 살피고 테두리를 고친다
            with torch.no_grad():
                pred = self.model(adv).argmax(dim=1)
                if targeted:
                    success = (pred == target_labels)
                else:
                    success = (pred != y)
            
            # 두 쪽 갈라 찾기의 테두리를 고친다
            c_high[success] = c[success]
            c_low[~success] = c[~success]
        
        return best_adv
    
    def evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_adv: torch.Tensor,
        verbose: bool = True
    ) -> Dict[str, float]:
        """C&W 치기가 얼마나 잘 먹히는지 따진다."""
        with torch.no_grad():
            x, y, x_adv = x.to(self.device), y.to(self.device), x_adv.to(self.device)
            
            clean_pred = self.model(x).argmax(dim=1)
            adv_pred = self.model(x_adv).argmax(dim=1)
            
            clean_acc = (clean_pred == y).float().mean().item()
            robust_acc = (adv_pred == y).float().mean().item()
            
            delta = (x_adv - x).view(len(x), -1)
            l2_norms = delta.norm(p=2, dim=1)
        
        metrics = {
            'clean_accuracy': clean_acc,
            'robust_accuracy': robust_acc,
            'attack_success_rate': 1 - robust_acc,
            'avg_l2': l2_norms.mean().item(),
            'median_l2': l2_norms.median().item(),
            'max_linf': delta.abs().max().item()
        }
        
        if verbose:
            print("=" * 50)
            print("C&W L2 치기 결과")
            print("=" * 50)
            print(f"맑은 맞음:      {metrics['clean_accuracy']:.2%}")
            print(f"든든한 맞음:     {metrics['robust_accuracy']:.2%}")
            print(f"치기가 먹힌 비율: {metrics['attack_success_rate']:.2%}")
            print(f"평균 L2 흔듦: {metrics['avg_l2']:.4f}")
            print(f"가운뎃값 L2:           {metrics['median_l2']:.4f}")
            print("=" * 50)
        
        return metrics
```

---

## 4. 비교: PGD과 C&W

| 특징 | PGD | C&W |
|--------|-----|-----|
| **제약** | 굳음($\|\boldsymbol{\delta}\| \leq \varepsilon$) | 부드러움(다독임) |
| **최적화** | 기울기 오름 | 제약 없는 자리에 Adam |
| **목표** | 손실을 가장 크게 | 섭동을 가장 작게 + 틀리게 가르기 |
| **가장 작은 섭동을 찾음** | 아니다 | 그렇다 |
| **빠르기** | 빠름 | 느림 |
| **세기** | 계산 | 아주 계산 |

---

## 5. 핵심 특징

### 센 데

1. **가장 작은 섭동을 찾는다**: $\varepsilon$이 붙박인 공격과 다르다
2. **기울기 가리기를 비껴간다**: 최적화가 더 든든하다
3. **아주 잘 먹힌다**: 여러 방어를 무너뜨린다
4. **너그럽다**: 여러 노름($\ell_0$, $\ell_2$, $\ell_\infty$)

### 한계

1. **계산이 비싸다**: 두 쪽 갈라 찾기 × 많은 되돌이
2. **하이퍼파라미터에 예민하다**: $c$, 학습률, 되돌이
3. **따짐에는 지나치다**: PGD으로도 넉넉한 일이 잦다

---

## 6. C&W을 쓸 때

- **방어 따지기**: PGD이 기울기 가리기에 속을 수 있을 때
- **가장 작은 섭동**: 알아챌 수 없음의 테두리 찾기
- **표적 공격**: 꼭 집어 다뤄야 할 때
- **연구**: 결정 경계의 꼴 알아보기

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

이 마당은 수학 기초, 알고리즘, PyTorch로 짜기, 비교: PGD과 C&W을 차례로 짚었다.

**살펴볼 거리**

1. Carlini, N., & Wagner, D. (2017). "Towards Evaluating the Robustness of Neural Networks." IEEE S&P.
2. Chen, P. Y., et al. (2018). "EAD: Elastic-Net Attacks." AAAI.
