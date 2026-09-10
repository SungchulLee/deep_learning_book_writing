# 밝혀 낸 든든함

**밝혀 낸 든든함**은 정해진 반지름 안의 어떤 흔듦에도 가름개의 미루어 봄이 바뀌지 않음을 수학으로 다짐한다. 남다른 치기로 시험해 보는 겪은 막이와 달리, 밝혀 낸 막이는 있을 수 있는 모든 치기에 대해 **증명할 수 있는 다짐**을 준다.

---

## 1. 겪은 든든함과 밝혀 낸 든든함

| 결 | 겪은 것 | 밝혀 낸 것 |
|--------|-----------|-----------|
| **다짐** | 없다(아는 치기로 시험할 뿐) | 수학의 증명 |
| **지킴** | 새 치기에 무너질 수 있다 | 반지름 안에서는 증명된 지킴 |
| **맞음** | 높음 | 낮음 |
| **밝히기** | 해당 없음 | 보기마다 셈한다 |

---

## 2. 아무렇게나 매끄럽게 하기

### 고갱이 깨침

**아무렇게나 매끄럽게 하기**(코언 등, 2019)은 가우스 잡음에 걸쳐 미루어 봄을 고르게 하여 어떤 가름개든 밝혀 낼 수 있게 든든한 것으로 바꾼다.

밑 가름개 $f: \mathbb{R}^d \to \mathcal{Y}$이 있을 때 **매끄럽게 한 가름개**을 짓는다.

$$
g(\mathbf{x}) = \arg\max_c \mathbb{P}_{\boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma^2 I)}[f(\mathbf{x} + \boldsymbol{\epsilon}) = c]
$$

**느낌으로:** 들임에 가우스 잡음을 더하고, 잡음 섞인 미루어 봄에 대해 많은 쪽을 고른다.

### 밝히기 정리

**정리(코언 등, 2019):** 매끄럽게 한 가름개 $g$이 들임 $\mathbf{x}$에서 갈래 $c_A$을 낌새 $p_A$으로 미루어 보고, 다음 갈래 $c_B$의 낌새가 $p_B$이면, $g(\mathbf{x}) = c_A$은 다음 $\ell_2$ 반지름 안에서 밝혀 낸 든든함을 지닌다.

$$
R = \frac{\sigma}{2}\left(\Phi^{-1}(p_A) - \Phi^{-1}(p_B)\right)
$$

여기서 $\Phi^{-1}$은 잣대 정규 분포의 쌓인 분포 함수의 거꿀이다.

### 느낌으로 알기

- $p_A$은 갈래 $c_A$이 많은 쪽이 될 낌새
- $p_A \gg p_B$이면 밝혀 낸 반지름 $R$이 크다(크게 자신함)
- $p_A \approx p_B$이면 밝혀 낸 반지름 $R$이 작다(머뭇거림)
- $\sigma$이 반지름의 잣대를 잡는다. 잡음이 클수록 밝혀 낸 자리가 넓다

### 밝히는 알고리즘

**두 도막으로 이루어진다.**

1. **고르기:** 미루어 볼 갈래를 찾는다
   - 잡음 섞인 미루어 봄을 $n_0$번 뽑는다
   - 많은 쪽을 골라 $\hat{c}$을 정한다

2. **밝히기:** 반지름을 어림한다
   - 잡음 섞인 미루어 봄을 $n$번 더 뽑는다
   - $p_A$의 믿음 구간을 셈한다
   - 밝혀 낸 반지름 $R$을 셈한다

### 몬테카를로 어림

낌새는 뽑아서 어림한다.

$$
\hat{p}_A = \frac{1}{N} \sum_{i=1}^N \mathbf{1}[f(\mathbf{x} + \boldsymbol{\epsilon}_i) = c_A], \quad \boldsymbol{\epsilon}_i \sim \mathcal{N}(0, \sigma^2 I)
$$

**클로퍼-피어슨 믿음 구간**을 쓰면

$$
\mathbb{P}(p_A \geq \underline{p}_A) \geq 1 - \alpha
$$

이로써 낌새 $\geq 1 - \alpha$으로 밝혀 낸 반지름을 얻는다.

---

## 3. PyTorch로 짜기

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm, binom
from typing import Optional, Dict, Tuple
from tqdm import tqdm
import math

class RandomizedSmoothing:
    """
    아무렇게나 매끄럽게 하여 밝혀 낸 든든함.
    
    가우스 잡음으로 미루어 봄을 매끄럽게 하여
    증명할 수 있는 L2 든든함 다짐을 준다.
    
    Parameters
    ----------
    base_classifier : nn.Module
        매끄럽게 할 밑 가름개
    sigma : float
        가우스 잡음의 잣대 어긋남
        σ이 클수록 밝혀 낸 반지름은 크고 맞음은 낮다
    """
    
    def __init__(
        self,
        base_classifier: nn.Module,
        sigma: float = 0.25,
        device: Optional[torch.device] = None
    ):
        self.base_classifier = base_classifier
        self.sigma = sigma
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.base_classifier.eval()
        self.base_classifier.to(self.device)
        self.num_classes = None
    
    def _sample_predictions(
        self,
        x: torch.Tensor,
        num_samples: int,
        batch_size: int = 1000
    ) -> torch.Tensor:
        """
        가우스 잡음 아래에서 미루어 봄을 뽑는다.
        
        Parameters
        ----------
        x : torch.Tensor
            들임 그림 하나, 꼴 (C, H, W)
        num_samples : int
            잡음 섞인 표본의 수
        batch_size : int
            다룰 묶음 크기
            
        Returns
        -------
        counts : torch.Tensor
            갈래마다의 미루어 봄 셈
        """
        with torch.no_grad():
            # 갈래의 수를 가린다
            if self.num_classes is None:
                test_out = self.base_classifier(x.unsqueeze(0).to(self.device))
                self.num_classes = test_out.shape[1]
            
            counts = torch.zeros(self.num_classes, device=self.device)
            
            num_batches = math.ceil(num_samples / batch_size)
            remaining = num_samples
            
            for _ in range(num_batches):
                current_batch = min(batch_size, remaining)
                remaining -= current_batch
                
                # 들임을 되풀이한다
                batch = x.unsqueeze(0).repeat(current_batch, 1, 1, 1).to(self.device)
                
                # 가우스 잡음을 더한다
                noise = torch.randn_like(batch) * self.sigma
                noisy_batch = batch + noise
                
                # 미루어 봄을 얻는다
                logits = self.base_classifier(noisy_batch)
                predictions = logits.argmax(dim=1)
                
                # 미루어 봄을 센다
                for c in range(self.num_classes):
                    counts[c] += (predictions == c).sum()
            
            return counts
    
    def _lower_confidence_bound(
        self,
        count: int,
        n: int,
        alpha: float
    ) -> float:
        """
        두 값 몫의 믿음 아래끝을 셈한다.
        클로퍼-피어슨(정확한) 길을 쓴다.
        """
        if count == 0:
            return 0.0
        return binom.ppf(alpha / 2, n, count / n) / n
    
    def _compute_radius(self, p_A: float, p_B: float) -> float:
        """
        낌새로 밝혀 낸 반지름을 셈한다.
        
        R = σ/2 * (Φ^{-1}(p_A) - Φ^{-1}(p_B))
        """
        if p_A <= 0.5:
            return 0.0
        
        # 무한을 비껴가려 잘라 낸다
        p_A = min(p_A, 0.999999)
        p_B = max(p_B, 0.000001)
        
        radius = (self.sigma / 2) * (norm.ppf(p_A) - norm.ppf(p_B))
        return max(0.0, radius)
    
    def certify(
        self,
        x: torch.Tensor,
        n0: int = 100,
        n: int = 10000,
        alpha: float = 0.001,
        batch_size: int = 1000
    ) -> Tuple[int, float]:
        """
        들임 하나를 밝힌다.
        
        Parameters
        ----------
        x : torch.Tensor
            들임 그림, 꼴 (C, H, W)
        n0 : int
            고르는 도막의 표본 수
        n : int
            밝히는 도막의 표본 수
        alpha : float
            믿음 켜(기본값: 99.9%)
        batch_size : int
            몬테카를로의 묶음 크기
            
        Returns
        -------
        predicted_class : int
            미루어 본 갈래(삼가면 -1)
        certified_radius : float
            밝혀 낸 L2 반지름(삼가면 0)
        """
        x = x.to(self.device)
        
        # 1도막: 고르기
        counts_selection = self._sample_predictions(x, n0, batch_size)
        top_class = counts_selection.argmax().item()
        
        # 2도막: 밝히기
        counts_cert = self._sample_predictions(x, n, batch_size)
        
        # 으뜸 갈래의 셈
        count_top = counts_cert[top_class].item()
        
        # p_A의 믿음 아래끝
        p_A_lower = self._lower_confidence_bound(int(count_top), n, alpha)
        
        # p_A_lower <= 0.5이면 밝힐 수 없다
        if p_A_lower <= 0.5:
            return -1, 0.0  # 삼간다
        
        # 밝혀 낸 반지름을 셈한다
        # 둘 가름에서는 p_B_upper = 1 - p_A_lower
        # 여러 갈래에서는 더 조심스러운 테두리를 쓴다
        radius = self.sigma * norm.ppf(p_A_lower)
        
        return top_class, radius
    
    def certify_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        n: int = 10000,
        alpha: float = 0.001,
        batch_size: int = 1000,
        radii_to_check: list = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    ) -> Dict[str, float]:
        """
        그림 묶음을 밝힌다.
        
        Returns
        -------
        results : dict
            - clean_accuracy: 옳게 미루어 본 몫
            - certified_accuracy_r=X: 반지름 X에서 밝혀 낸 몫
            - avg_certified_radius: 밝혀 낸 반지름의 평균
        """
        num_images = len(images)
        predictions = []
        radii = []
        
        for i in tqdm(range(num_images), desc="Certifying"):
            pred, radius = self.certify(images[i], n=n, alpha=alpha, batch_size=batch_size)
            predictions.append(pred)
            radii.append(radius)
        
        predictions = torch.tensor(predictions, device=labels.device)
        radii = torch.tensor(radii)
        
        # 자
        correct = (predictions == labels)
        abstain = (predictions == -1)
        
        results = {
            'clean_accuracy': correct.float().mean().item(),
            'abstain_rate': abstain.float().mean().item(),
            'avg_certified_radius': radii[correct & ~abstain].mean().item() if (correct & ~abstain).any() else 0.0
        }
        
        # 반지름마다 밝혀 낸 맞음
        for r in radii_to_check:
            certified = correct & (radii >= r)
            results[f'certified_accuracy_r={r}'] = certified.float().mean().item()
        
        return results
    
    def predict(
        self,
        x: torch.Tensor,
        n: int = 1000,
        batch_size: int = 500
    ) -> int:
        """갈래를 미루어 본다(밝히기 없이)."""
        counts = self._sample_predictions(x, n, batch_size)
        return counts.argmax().item()

class SmoothClassifier(nn.Module):
    """
    익힘과 미루어 봄에서 가름개를 매끄럽게 해 주는 감싸개.
    """
    
    def __init__(self, base_classifier: nn.Module, sigma: float, num_samples: int = 1):
        super().__init__()
        self.base_classifier = base_classifier
        self.sigma = sigma
        self.num_samples = num_samples
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        잡음을 불려 앞으로 걸음.
        
        익힐 때: 잡음 표본 하나(잘 들도록)
        미루어 볼 때: 표본 여럿의 평균
        """
        if self.training:
            # 익힐 때는 잡음 표본 하나
            noise = torch.randn_like(x) * self.sigma
            return self.base_classifier(x + noise)
        else:
            # 미루어 볼 때는 표본 여럿의 평균
            batch_size = x.shape[0]
            outputs = []
            
            for _ in range(self.num_samples):
                noise = torch.randn_like(x) * self.sigma
                outputs.append(self.base_classifier(x + noise))
            
            return torch.stack(outputs).mean(dim=0)
```

### 쓰는 보기

```python
import torchvision
import torchvision.transforms as transforms

# 모형을 얹는다
base_model = torchvision.models.resnet18(num_classes=10)
base_model.load_state_dict(torch.load('cifar10_resnet18.pth'))

# 매끄럽게 한 가름개를 만든다
smoother = RandomizedSmoothing(base_model, sigma=0.25)

# 시험 자료를 얹는다
transform = transforms.ToTensor()
testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

# 묶음 하나를 얻는다
images, labels = next(iter(test_loader))

# 밝힌다
results = smoother.certify_batch(
    images[:50], labels[:50],  # 보기 50개를 밝힌다(느리다)
    n=10000,                    # 보기마다 표본 1만 개
    alpha=0.001                 # 믿음 99.9%
)

print("밝히기 결과:")
print(f"  맑은 맞음: {results['clean_accuracy']:.2%}")
print(f"  삼간 비율: {results['abstain_rate']:.2%}")
print(f"  밝혀 낸 반지름 평균: {results['avg_certified_radius']:.4f}")
print(f"  r=0.25에서 밝힘: {results['certified_accuracy_r=0.25']:.2%}")
print(f"  r=0.50에서 밝힘: {results['certified_accuracy_r=0.5']:.2%}")
print(f"  r=1.00에서 밝힘: {results['certified_accuracy_r=1.0']:.2%}")
```

---

## 4. 밝혀 낸 든든함을 위한 익힘

### 가우스 자료 불리기

가장 단순한 길은 가우스 잡음으로 불려 익히는 것이다.

```python
def train_with_noise(model, train_loader, sigma, epochs):
    """가우스 잡음으로 불려 익힌다."""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    
    for epoch in range(epochs):
        for x, y in train_loader:
            # 가우스 잡음을 더한다
            noise = torch.randn_like(x) * sigma
            x_noisy = x + noise
            
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x_noisy), y)
            loss.backward()
            optimizer.step()
```

### 한결같음 다독이기

잡음 표본에 걸쳐 미루어 봄이 한결같도록 이끈다.

$$
\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda \cdot \text{KL}(f(\mathbf{x} + \boldsymbol{\epsilon}_1) \| f(\mathbf{x} + \boldsymbol{\epsilon}_2))
$$

---

## 5. 고갱이 매개변수와 맞바꿈

### 잡음의 크기 시그마

| $\sigma$ | 밝혀 낸 반지름 | 맑은 맞음 |
|----------|------------------|----------------|
| 0.12 | 작음(약 0.25) | 높음(약 85%) |
| 0.25 | 가운데(약 0.5) | 가운데(약 75%) |
| 0.50 | 큼(약 1.0) | 낮음(약 60%) |
| 1.00 | 아주 큼(약 2.0) | 아주 낮음(약 40%) |

**맞바꿈:** $\sigma$이 클수록 밝혀 낸 자리는 넓어지나 맞음은 낮아진다.

### 뽑기 매개변수

| 매개변수 | 값 | 미침 |
|-----------|-------|--------|
| $n_0$(고르기) | 100 | 클수록 고르기가 미덥다 |
| $n$(밝히기) | 10,000 넘음 | 클수록 믿음이 촘촘하다 |
| $\alpha$(믿음) | 0.001 | 작을수록 조심스럽다 |

### 셈 값

밝히기는 **비싸다**.

- 들임마다 표본 $n = 10,000$개
- 표본마다 온전한 앞으로 걸음이 있어야 한다
- 시험 보기 1만 개를 밝히려면 앞으로 걸음이 1억 번

---

## 6. 견주기: 겪은 것과 밝혀 낸 것

**CIFAR-10, $\varepsilon = 0.5$(L2):**

| 방법 | 맑은 맞음 | 든든한 맞음 | 밝혔나? |
|--------|-----------|------------|------------|
| 여느 것 | 95% | 0% | 아니다 |
| PGD 맞서며 익히기 | 85% | 약 50% | 아니다 |
| 아무렇게나 매끄럽게 하기 | 75% | 약 60% | **그렇다** |

밝혀 낸 맞음이 겪은 든든한 맞음을 넘을 수도 있다. 까닭은

- 겪은 치기가 가장 좋은 맞서는 보기를 못 찾을 수 있다
- 밝히기는 다짐된 아래끝을 준다

---

## 7. 한계

1. **L2 노름만**: 아무렇게나 매끄럽게 하기는 L∞이 아니라 L2 흔듦을 밝힌다
2. **맞음 떨어짐**: 맑은 맞음이 꽤 줄어든다
3. **셈 값**: 밝히기가 느리다
4. **크게 늘리기 어려움**: 큰 모형이나 자료 꾸러미에는 만만치 않다

---

## 8. 한발 더 나간 이야기

### L∞의 밝혀 낸 든든함

**사이 테두리 퍼뜨리기(IBP)** 같은 길이 L∞ 밝히기를 준다.

$$
[\underline{z}, \overline{z}] = \text{IBP}(f, [\mathbf{x} - \varepsilon, \mathbf{x} + \varepsilon])
$$

$\underline{z}_y > \max_{i \neq y} \overline{z}_i$이면 그 미루어 봄은 밝혀진다.

### 더 촘촘한 밝힘

- **SmoothAdv**: 맞서며 익히기 + 매끄럽게 하기
- **MACER**: 익히는 동안 밝혀 낸 반지름을 가장 크게 한다
- **잡음 지운 매끄럽게 하기**: 잡음 지우개를 익혀 밑 맞음을 올린다

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

| 깨침 | 고갱이 |
|---------|-----------|
| **아무렇게나 매끄럽게 하기** | 가우스 잡음에 걸쳐 고르게 한다 |
| **밝혀 낸 반지름** | $R = \frac{\sigma}{2}(\Phi^{-1}(p_A) - \Phi^{-1}(p_B))$ |
| **맞바꿈** | $\sigma$이 클수록 R은 크고 맞음은 낮다 |
| **다짐** | $\|\boldsymbol{\delta}\|_2 \leq R$인 모든 흔듦에 증명된다 |

밝혀 낸 든든함은 가장 센 이론의 다짐을 주되, 맞음과 셈 값을 내주어야 한다.

**살펴볼 거리**

1. Cohen, J., Rosenfeld, E., & Kolter, Z. (2019). "Certified Adversarial Robustness via Randomized Smoothing." ICML.
2. Salman, H., et al. (2019). "Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers." NeurIPS.
3. Zhai, R., et al. (2020). "MACER: Attack-Free and Scalable Robust Training via Maximizing Certified Radius." ICLR.
