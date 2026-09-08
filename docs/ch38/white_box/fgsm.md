# 빠른 기울기 부호 방법(FGSM)

**빠른 기울기 부호 방법(FGSM)**은 굿펠로 등(2015)이 내놓은, 기울기에 기댄 맞서는 치기의 밑바탕이다. 기울기를 한 번만 셈하면 되는 단순함 덕에 셈이 잘 들고, 더 정교한 치기의 벽돌이 된다.

---

## 1. 수학 밑바탕

### 선형 짐작

FGSM은 신경 그물이 차수 높은 밭에서 거의 선형으로 움직인다는 살핌에서 비롯한다. 선형 모형을 보자.

$$
f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}
$$

흔듦 $\boldsymbol{\delta}$에 대해 날임의 바뀜은

$$
f(\mathbf{x} + \boldsymbol{\delta}) - f(\mathbf{x}) = \mathbf{w}^\top \boldsymbol{\delta}
$$

$\ell_\infty$ 옭아맴 $\|\boldsymbol{\delta}\|_\infty \leq \varepsilon$ 아래에서 이를 가장 크게 하는 흔듦은

$$
\delta_i^* = \varepsilon \cdot \text{sign}(w_i)
$$

그러면 가장 큰 바뀜은

$$
\mathbf{w}^\top \boldsymbol{\delta}^* = \varepsilon \|\mathbf{w}\|_1
$$

차수가 높으면 작은 $\varepsilon$으로도 $\varepsilon \|\mathbf{w}\|_1$이 꽤 커진다.

### 신경 그물로 넓히기

잃음 함수 $\mathcal{L}$을 지닌 신경 그물 $f_\theta$에서 일차 테일러 펼침으로 곧게 편다.

$$
\mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y) \approx \mathcal{L}(f_\theta(\mathbf{x}), y) + \boldsymbol{\delta}^\top \nabla_\mathbf{x} \mathcal{L}(f_\theta(\mathbf{x}), y)
$$

기울기 $\nabla_\mathbf{x} \mathcal{L}$이 선형 어림에서 "짐" 노릇을 한다. 같은 이치를 쓰면

$$
\boxed{\mathbf{x}_{\text{adv}} = \mathbf{x} + \varepsilon \cdot \text{sign}(\nabla_\mathbf{x} \mathcal{L}(f_\theta(\mathbf{x}), y))}
$$

이것이 **FGSM 치기**이다.

### 느낌으로 알기

기울기 $\nabla_\mathbf{x} \mathcal{L}$은 들임 밭에서 잃음을 가장 크게 올리는 방향을 알려 준다. 부호 함수는 자리마다 방향($+1$ 또는 $-1$)만 뽑아내고, 우리는 그 방향으로 받아 주는 가장 큰 걸음($\varepsilon$)을 밟는다.

**그림으로 보기:**

```
맑은 들임 x ──────────────> 잃음 L(f(x), y)
                기울기
                  ∇L
                   │
                   ▼
흔든 x + ε·sign(∇L) ───> 잃음 L(f(x_adv), y) ≫ L(f(x), y)
```

---

## 2. 알고리즘

### 과녁 없는 FGSM

**들임:** 맑은 보기 $\mathbf{x}$, 참 이름표 $y$, 모형 $f_\theta$, 엡실론 $\varepsilon$

**날임:** 맞서는 보기 $\mathbf{x}_{\text{adv}}$

1. 잃음을 셈한다: $\mathcal{L} = \text{CrossEntropy}(f_\theta(\mathbf{x}), y)$
2. 기울기를 셈한다: $\mathbf{g} = \nabla_\mathbf{x} \mathcal{L}$
3. 흔듦을 셈한다: $\boldsymbol{\delta} = \varepsilon \cdot \text{sign}(\mathbf{g})$
4. 맞서는 보기를 만든다: $\mathbf{x}_{\text{adv}} = \text{clip}(\mathbf{x} + \boldsymbol{\delta}, 0, 1)$

**번거로움:** $O(1)$ — 앞으로-되돌아 걸음 한 번

### 과녁 있는 FGSM

갈래 $y_{\text{target}}$을 노린 과녁 있는 치기에서는

$$
\mathbf{x}_{\text{adv}} = \mathbf{x} - \varepsilon \cdot \text{sign}(\nabla_\mathbf{x} \mathcal{L}(f_\theta(\mathbf{x}), y_{\text{target}}))
$$

**빼기 부호**에 눈여겨보라. 잃음을 내려가며 미루어 봄을 과녁 쪽으로 끌어당긴다.

### L2 FGSM 갈래

$\ell_\infty$ 대신 $\ell_2$으로 옭아매면

$$
\mathbf{x}_{\text{adv}} = \mathbf{x} + \varepsilon \cdot \frac{\nabla_\mathbf{x} \mathcal{L}}{\|\nabla_\mathbf{x} \mathcal{L}\|_2}
$$

기울기를 길이 1로 맞춘 뒤 $\varepsilon$으로 잣대를 잡는다.

---

## 3. PyTorch로 짜기

### 온전한 FGSM 갈래

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

class FGSM:
    """
    빠른 기울기 부호 방법(FGSM) 치기.
    
    잃음을 가장 크게 하는 방향으로 기울기 한 걸음을 밟아
    맞서는 보기를 만든다.
    
    수학 꼴
    ------------------------
    x_adv = x + ε · sign(∇_x L(f(x), y))
    
    여기서:
    - ∇_x L은 들임에 대한 잃음의 기울기
    - ε은 흔듦 예산
    - sign(·)은 낱낱의 부호를 돌려준다
    
    Parameters
    ----------
    model : nn.Module
        칠 신경 그물
    epsilon : float
        가장 큰 L∞ 흔듦(기본값: 그림에서 8/255)
    loss_fn : nn.Module, 골라 씀
        잃음 함수(기본값: CrossEntropyLoss)
    clip_min : float
        옳은 들임의 가장 작은 값(기본값: 0.0)
    clip_max : float
        옳은 들임의 가장 큰 값(기본값: 1.0)
    device : torch.device, 골라 씀
        셈할 장치
    
    보기
    -------
    >>> model = load_pretrained_model()
    >>> attack = FGSM(model, epsilon=8/255)
    >>> x_adv = attack.generate(images, labels)
    >>> metrics = attack.evaluate(images, labels, x_adv)
    >>> print(f"치기가 먹힌 비율: {metrics['attack_success_rate']:.2%}")
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 8/255,
        loss_fn: Optional[nn.Module] = None,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = loss_fn if loss_fn else nn.CrossEntropyLoss()
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.device = device or next(model.parameters()).device
        
        # 모형을 따짐 모드로 둔다
        self.model.eval()
        self.model.to(self.device)
    
    def generate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        targeted: bool = False,
        target_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        FGSM으로 맞서는 보기를 만든다.
        
        Parameters
        ----------
        x : torch.Tensor
            맑은 그림, 꼴 (N, C, H, W)
        y : torch.Tensor
            참 이름표, 꼴 (N,)
        targeted : bool
            True이면 과녁 있는 치기를 한다
        target_labels : torch.Tensor, 골라 씀
            과녁 있는 치기의 과녁 이름표
            
        Returns
        -------
        x_adv : torch.Tensor
            맞서는 보기, 꼴 (N, C, H, W)
        """
        # 들임을 장치로 옮긴다
        x = x.to(self.device)
        y = y.to(self.device)
        
        # 들임에 기울기 셈을 켠다
        x_adv = x.clone().detach().requires_grad_(True)
        
        # 앞으로 걸음
        logits = self.model(x_adv)
        
        # 잃음을 셈한다
        if targeted:
            if target_labels is None:
                raise ValueError("과녁 있는 치기에는 target_labels이 있어야 한다")
            target_labels = target_labels.to(self.device)
            loss = self.loss_fn(logits, target_labels)
        else:
            loss = self.loss_fn(logits, y)
        
        # 기울기를 셈하려 되돌아 걸음
        self.model.zero_grad()
        loss.backward()
        
        # 들임에 대한 기울기를 얻는다
        grad = x_adv.grad.data
        
        # 흔듦을 셈한다
        if targeted:
            # 과녁 있는 치기에서는 내려간다
            perturbation = -self.epsilon * torch.sign(grad)
        else:
            # 과녁 없는 치기에서는 올라간다
            perturbation = self.epsilon * torch.sign(grad)
        
        # 흔듦을 걸고 잘라 낸다
        x_adv = x + perturbation
        x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)
        
        return x_adv.detach()
    
    def generate_l2(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        targeted: bool = False,
        target_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        L2-FGSM으로 맞서는 보기를 만든다.
        
        sign(기울기) 대신 기울기를 L2 노름 1로 맞춘다.
        
        Returns
        -------
        x_adv : torch.Tensor
            맞서는 보기
        """
        x = x.to(self.device)
        y = y.to(self.device)
        
        x_adv = x.clone().detach().requires_grad_(True)
        logits = self.model(x_adv)
        
        if targeted:
            if target_labels is None:
                raise ValueError("과녁 있는 치기에는 target_labels이 있어야 한다")
            loss = self.loss_fn(logits, target_labels.to(self.device))
        else:
            loss = self.loss_fn(logits, y)
        
        self.model.zero_grad()
        loss.backward()
        
        grad = x_adv.grad.data
        
        # 보기마다 기울기의 잣대를 맞춘다
        grad_flat = grad.view(grad.shape[0], -1)
        grad_norm = torch.norm(grad_flat, p=2, dim=1, keepdim=True)
        grad_normalized = grad_flat / (grad_norm + 1e-8)
        grad_normalized = grad_normalized.view(grad.shape)
        
        if targeted:
            perturbation = -self.epsilon * grad_normalized
        else:
            perturbation = self.epsilon * grad_normalized
        
        x_adv = x + perturbation
        x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)
        
        return x_adv.detach()
    
    def evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_adv: torch.Tensor,
        verbose: bool = True
    ) -> dict:
        """
        치기가 얼마나 잘 먹히는지 따진다.
        
        Parameters
        ----------
        x : torch.Tensor
            맑은 그림
        y : torch.Tensor
            참 이름표
        x_adv : torch.Tensor
            맞서는 그림
        verbose : bool
            결과를 찍는다
            
        Returns
        -------
        metrics : dict
            따지는 자
        """
        with torch.no_grad():
            # 맑은 미루어 봄
            clean_logits = self.model(x.to(self.device))
            clean_pred = clean_logits.argmax(dim=1)
            clean_correct = (clean_pred == y.to(self.device)).sum().item()
            
            # 맞서는 미루어 봄
            adv_logits = self.model(x_adv.to(self.device))
            adv_pred = adv_logits.argmax(dim=1)
            adv_correct = (adv_pred == y.to(self.device)).sum().item()
            
            # 흔듦의 자
            delta = (x_adv - x).view(len(x), -1)
            linf_norm = delta.abs().max(dim=1)[0].mean().item()
            l2_norm = torch.norm(delta, p=2, dim=1).mean().item()
        
        n = len(y)
        metrics = {
            'clean_accuracy': clean_correct / n,
            'robust_accuracy': adv_correct / n,
            'attack_success_rate': 1 - adv_correct / n,
            'avg_linf_perturbation': linf_norm,
            'avg_l2_perturbation': l2_norm
        }
        
        if verbose:
            print("=" * 50)
            print("FGSM 치기 결과")
            print("=" * 50)
            print(f"엡실론: {self.epsilon:.4f}")
            print(f"맑은 맞음: {metrics['clean_accuracy']:.2%}")
            print(f"든든한 맞음: {metrics['robust_accuracy']:.2%}")
            print(f"치기가 먹힌 비율: {metrics['attack_success_rate']:.2%}")
            print(f"평균 L∞ 흔듦: {linf_norm:.6f}")
            print(f"평균 L2 흔듦: {l2_norm:.4f}")
            print("=" * 50)
        
        return metrics
    
    def visualize(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_adv: torch.Tensor,
        class_names: Optional[list] = None,
        num_examples: int = 5,
        figsize: Tuple[int, int] = (15, 9)
    ) -> plt.Figure:
        """
        맑은 그림, 맞서는 그림, 흔듦을 그린다.
        
        Parameters
        ----------
        x : torch.Tensor
            맑은 그림
        y : torch.Tensor
            참 이름표
        x_adv : torch.Tensor
            맞서는 그림
        class_names : list, 골라 씀
            갈래 이름의 목록
        num_examples : int
            보일 보기의 수
        figsize : tuple
            그림 크기
            
        Returns
        -------
        fig : matplotlib.Figure
            그린 그림
        """
        # 미루어 봄을 얻는다
        with torch.no_grad():
            clean_pred = self.model(x.to(self.device)).argmax(dim=1)
            adv_pred = self.model(x_adv.to(self.device)).argmax(dim=1)
        
        # 넘파이로 옮긴다
        x_np = x[:num_examples].cpu().numpy()
        x_adv_np = x_adv[:num_examples].cpu().numpy()
        y_np = y[:num_examples].cpu().numpy()
        clean_pred_np = clean_pred[:num_examples].cpu().numpy()
        adv_pred_np = adv_pred[:num_examples].cpu().numpy()
        
        perturbations = x_adv_np - x_np
        
        fig, axes = plt.subplots(3, num_examples, figsize=figsize)
        
        for i in range(num_examples):
            # 잿빛인지 RGB인지 가린다
            if x_np.shape[1] == 1:
                clean_img = x_np[i, 0]
                adv_img = x_adv_np[i, 0]
                pert_img = perturbations[i, 0]
                cmap = 'gray'
            else:
                clean_img = np.transpose(x_np[i], (1, 2, 0))
                adv_img = np.transpose(x_adv_np[i], (1, 2, 0))
                pert_img = np.transpose(perturbations[i], (1, 2, 0))
                cmap = None
            
            # 1줄: 맑은 그림
            axes[0, i].imshow(np.clip(clean_img, 0, 1), cmap=cmap)
            true_label = class_names[y_np[i]] if class_names else y_np[i]
            pred_label = class_names[clean_pred_np[i]] if class_names else clean_pred_np[i]
            axes[0, i].set_title(f'맑음\n참: {true_label}', fontsize=9)
            axes[0, i].axis('off')
            
            # 2줄: 맞서는 그림
            axes[1, i].imshow(np.clip(adv_img, 0, 1), cmap=cmap)
            pred_label = class_names[adv_pred_np[i]] if class_names else adv_pred_np[i]
            color = 'red' if adv_pred_np[i] != y_np[i] else 'green'
            axes[1, i].set_title(f'맞섬\n미루어 봄: {pred_label}', 
                                fontsize=9, color=color)
            axes[1, i].axis('off')
            
            # 3줄: 흔듦(키워서)
            pert_magnified = pert_img * 10 + 0.5
            axes[2, i].imshow(np.clip(pert_magnified, 0, 1), cmap='RdBu_r')
            axes[2, i].set_title('흔듦\n(10배 키움)', fontsize=9)
            axes[2, i].axis('off')
        
        plt.suptitle(f'FGSM 치기 (ε = {self.epsilon:.4f})', fontsize=12)
        plt.tight_layout()
        
        return fig
```

### 쓰는 보기

```python
import torch
import torchvision
import torchvision.transforms as transforms

# CIFAR-10을 얹는다
transform = transforms.Compose([
    transforms.ToTensor(),
])

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# 미리 익힌 모형을 얹는다
model = torchvision.models.resnet18(pretrained=False, num_classes=10)
model.load_state_dict(torch.load('cifar10_resnet18.pth'))

# FGSM 치기를 만든다
attack = FGSM(model, epsilon=8/255)

# 묶음 하나를 얻는다
images, labels = next(iter(testloader))

# 맞서는 보기를 만든다
adv_images = attack.generate(images, labels)

# 따진다
metrics = attack.evaluate(images, labels, adv_images)

# 그린다
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
fig = attack.visualize(images, labels, adv_images, class_names=class_names)
plt.savefig('fgsm_visualization.png', dpi=150, bbox_inches='tight')
```

---

## 4. 살피기와 결

### 엡실론에 예민함

$\varepsilon$을 어떻게 고르느냐가 치기가 먹히는 데 크게 걸린다.

| 엡실론(8비트) | 엡실론(뜨는 소수) | 미침 |
|-----------------|-----------------|--------|
| 1/255 | 0.004 | 알아챌 수 없고 잘 안 먹힘 |
| 4/255 | 0.016 | 은근하고 웬만큼 먹힘 |
| 8/255 | 0.031 | **여느 값**, 잘 먹힘 |
| 16/255 | 0.063 | 눈에 띄고 아주 잘 먹힘 |

**해 봄: 엡실론과 치기가 먹힌 비율**

```python
def epsilon_sensitivity_study(
    attack_class,
    model,
    x,
    y,
    epsilons=[1/255, 2/255, 4/255, 8/255, 16/255, 32/255]
):
    """엡실론에 따라 치기가 먹힌 비율이 어떻게 바뀌는지 살핀다."""
    results = []
    
    for eps in epsilons:
        attack = attack_class(model, epsilon=eps)
        x_adv = attack.generate(x, y)
        metrics = attack.evaluate(x, y, x_adv, verbose=False)
        
        results.append({
            'epsilon': eps,
            'epsilon_255': int(eps * 255),
            **metrics
        })
    
    return results
```

### 센 데와 한계

**센 데:**

- 아주 빠르다(기울기 셈 한 번)
- 든든함을 따지는 쓸 만한 밑금이다
- 짜기도 알기도 쉽다
- 막이 없는 모형에는 거의 다 먹힌다

**한계:**

- 한 걸음 치기라 가장 좋지는 않다
- 맞서며 익히기로 쉽게 막힌다
- 기울기를 가린 모형에는 듣지 않을 수 있다
- 가장 작은 흔듦을 찾는 데는 맞지 않다

### 아무 잡음과 견주기

같은 크기의 아무 잡음보다 FGSM이 훨씬 잘 먹힌다.

```python
def compare_fgsm_to_random(model, x, y, epsilon):
    """FGSM과 아무 흔듦을 견준다."""
    device = next(model.parameters()).device
    
    # FGSM
    attack = FGSM(model, epsilon=epsilon)
    x_adv_fgsm = attack.generate(x, y)
    
    # 고르게 뽑은 아무 잡음
    noise = torch.empty_like(x).uniform_(-epsilon, epsilon)
    x_adv_random = torch.clamp(x + noise, 0, 1)
    
    # 둘 다 따진다
    with torch.no_grad():
        y_dev = y.to(device)
        
        fgsm_pred = model(x_adv_fgsm.to(device)).argmax(dim=1)
        fgsm_success = (fgsm_pred != y_dev).float().mean().item()
        
        random_pred = model(x_adv_random.to(device)).argmax(dim=1)
        random_success = (random_pred != y_dev).float().mean().item()
    
    print(f"FGSM이 먹힌 비율: {fgsm_success:.2%}")
    print(f"아무 잡음이 먹힌 비율: {random_success:.2%}")
    
    return fgsm_success, random_success
```

CIFAR-10에서 $\varepsilon = 8/255$일 때의 흔한 결과는

- FGSM: 먹힌 비율 약 60~80%
- 아무 잡음: 먹힌 비율 약 5~10%

---

## 5. 갈래와 넓힘

### 빠른 기울기 방법(FGM)

부호 대신 기울기를 그대로 쓴다.

$$
\mathbf{x}_{\text{adv}} = \mathbf{x} + \varepsilon \cdot \nabla_\mathbf{x} \mathcal{L}
$$

잣대를 맞추지 않으므로 $\ell_\infty$ 옭아맴을 어길 수 있다.

### 아무렇게나 비롯하는 FGSM

FGSM에 앞서 첫자리를 아무렇게나 잡는다.

$$
\begin{aligned}
\mathbf{x}' &= \mathbf{x} + \alpha \cdot \mathbf{u}, \quad \mathbf{u} \sim \text{Uniform}[-1, 1]^d \\
\mathbf{x}_{\text{adv}} &= \mathbf{x}' + (\varepsilon - \alpha) \cdot \text{sign}(\nabla_{\mathbf{x}'} \mathcal{L})
\end{aligned}
$$

이는 그 자리의 나쁜 봉우리를 벗어나게 도우며 맞서며 익히기에 쓰인다.

### 밀어 나감을 곁들인 FGSM

기울기의 자취를 쌓는다(PGD 마디에서 다룰 MI-FGSM으로 이어진다).

$$
\mathbf{g}_t = \mu \cdot \mathbf{g}_{t-1} + \frac{\nabla_\mathbf{x} \mathcal{L}}{\|\nabla_\mathbf{x} \mathcal{L}\|_1}
$$

---

## 6. 맞서며 익히기와의 이어짐

FGSM은 맞서며 익히기의 고갱이다. 가장 작게-가장 크게 하기

$$
\min_\theta \mathbb{E}_{(\mathbf{x},y)}[\max_{\|\boldsymbol{\delta}\| \leq \varepsilon} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y)]
$$

에서 안쪽의 가장 크게 하기를 FGSM으로 어림한다.

$$
\boldsymbol{\delta}^{\text{FGSM}} = \varepsilon \cdot \text{sign}(\nabla_\mathbf{x} \mathcal{L}(f_\theta(\mathbf{x}), y))
$$

FGSM 맞서는 보기로 익히면 밑바탕 든든함을 얻는다. 다만 PGD에 기댄 익힘이 더 세다.

---

## 연습문제

**연습문제 1.**
선형 가름개 $f(x) = w^T x + b$에서 미루어 본 갈래를 바꾸는 데 드는 가장 작은 $\ell_\infty$ 흔듦을 셈하여라. 이것이 신경 그물의 든든함과 어떻게 이어지는지 밝혀라.

??? success "연습문제 1 풀이"
    선형 가름개에서 $\ell_\infty$ 노름으로 잰 판단의 금까지의 거리는 $\frac{|w^T x + b|}{\|w\|_1}$이다. 가장 작은 흔듦은 $\delta^* = \frac{|w^T x + b|}{\|w\|_1} \cdot \text{sign}(w)$이다. 신경 그물에서는 그 자리의 선형 어림 $f(x + \delta) \approx f(x) + \nabla_x f \cdot \delta$이 FGSM(기울기의 부호를 쓴다)이 왜 잘 듣는지를 밝혀 준다. 차수가 높은 모형이 무른 까닭은 $\|w\|_1$은 차수와 함께 커지는데 $|w^T x + b|$은 꼭 그렇지 않아 든든함의 여유가 줄어들기 때문이다. $\square$

---

**연습문제 2.**
이 마디에서 다룬 치기나 막이를 CIFAR-10의 ResNet-18 모형에 짜 넣어라. $\epsilon = 8/255$의 PGD-20 치기 아래에서 맑은 맞음과 든든한 맞음을 알려라.

??? success "연습문제 2 풀이"
    여느 ResNet-18은 맑은 맞음이 $\sim$93%이지만 PGD-20($\epsilon = 8/255$, 걸음 크기 $2/255$) 아래의 든든한 맞음은 $\sim$0%이다. 이 마디의 방법을 걸면 결과는 재주에 따라 다르다. 맞서며 익히기는 맑은 맞음 $\sim$83%에 든든한 맞음 $\sim$50%이고, 밝혀 낸 막이는 더 낮지만 증명할 수 있는 테두리를 준다. 맞음과 든든함의 맞바꿈은 밑바탕부터 있는 것이라, 든든함을 높이면 맑은 맞음이 흔히 5~15% 든다. 아무렇게나 하는 씨앗 3개의 평균과 잣대 어긋남으로 알려라. $\square$

---

**연습문제 3.**
흔듦 공 안에서 갈래별 자료의 밑자리가 서로 겹친다고 볼 때, 모형이 담는 힘을 키우지 않고서는 어떤 막이도 맑은 자료의 높은 맞음과 $\ell_\infty$ 흔듦에 대한 높은 든든함을 함께 이룰 수 없음을 증명하여라.

??? success "연습문제 3 풀이"
    두 갈래의 밑자리가 거리 $\epsilon$ 안에서 겹치면(곧 $\|x_1 - x_2\|_\infty \leq 2\epsilon$인 $x_1 \in \text{갈래 1}, x_2 \in \text{갈래 2}$이 있으면), $x_1$과 $x_2$ 둘 다에서 든든한 가름개는 적어도 하나를 틀리게 가를 수밖에 없다(흔듦 공이 겹치기 때문이다). 이것이 맞음과 든든함의 밑바탕 맞바꿈이다. 겹치는 밑자리의 몫이 피할 수 없는 맞음 잃음을 정한다. 여느 그림 분포에서는 $\epsilon = 8/255$에서 겹침이 꽤 있어, 살펴본 10~15%의 맞음 떨어짐을 밝혀 준다. 모형이 담는 힘을 키우면(더 너른 그물) 얽힌 든든한 판단의 금을 더 잘 그려 맞바꿈을 얼마쯤 눅일 수 있다. $\square$

---

**연습문제 4.**
금융 기계 배움 얼개(속임수 알아내기나 거래 신호 만들기 따위)에서 맞섬의 든든함이 어떻게 드러나는지 다루어라. 으름 얼개가 보기 다룸과 어떻게 다른가?

??? success "연습문제 4 풀이"
    금융에서 겨루는 이는 알아내는 얼개에 맞추어 스스로 움직이는 꾀 많은 무리(속임수꾼, 저자 흔드는 이)다. 보기 다룸과 다른 고갱이는 이렇다. (1) 흔들 수 있는 밭이 돈으로 될 만한 것에 옭매인다(속임수꾼이 제 거래 자취를 통째로 바꿀 수는 없다). (2) 치기가 잇따르며 맞추어 간다(겨루는 이가 얼개의 되받음을 보고 손본다). (3) 헛 맞음과 놓침의 값이 서로 어긋난다(옳은 거래를 막는 것과 속임수를 놓치는 것). (4) $\ell_p$ 노름은 뜻이 없고 밭에 맞는 흔듦 모형이 있어야 한다. 막이는 맞추어 오는 겨루는 이에게도 든든해야 하므로, 알아내는 잣대가 알려지면 비껴갈 수 있는 알아내기 바탕의 길은 많이 걸러진다. $\square$

## 정리하며

| 결 | FGSM |
|--------|------|
| **식** | $\mathbf{x}_{\text{adv}} = \mathbf{x} + \varepsilon \cdot \text{sign}(\nabla_\mathbf{x} \mathcal{L})$ |
| **번거로움** | $O(1)$ — 앞으로-되돌아 걸음 한 번 |
| **세기** | 가운데(밑금) |
| **여느 $\varepsilon$** | CIFAR-10에서 $8/255$ |
| **고갱이 한계** | 한 걸음이라 쉽게 막힌다 |

FGSM은 맞서는 치기를 알아보는 깨침의 밑바탕을 준다. PGD과 C&W 같은 더 센 치기가 있지만, FGSM은 잘 드는 따짐과 맞서며 익히기에 여전히 값지다.

**살펴볼 거리**

1. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). "Explaining and Harnessing Adversarial Examples." ICLR.
2. Kurakin, A., Goodfellow, I., & Bengio, S. (2017). "Adversarial Examples in the Physical World." ICLR Workshop.
3. Tramèr, F., et al. (2018). "Ensemble Adversarial Training: Attacks and Defenses." ICLR.
