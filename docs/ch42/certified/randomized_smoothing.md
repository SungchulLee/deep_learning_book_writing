# 랜덤 평활화와 강건성 인증

**강건성 인증**은 정해진 반지름 안의 어떤 섭동에도 분류기의 예측이 바뀌지 않음을 수학적으로 보장한다. 특정 공격으로 시험해 보는 경험적 방어와 달리, 인증 가능한 방어는 있을 수 있는 모든 공격에 대해 **증명할 수 있는 보장**을 준다.

---

## 1. 경험적 강건성과 강건성 인증

| 항목 | 경험적 방어 | 인증된 방어 |
|--------|-----------|-----------|
| **보장** | 없다(알려진 공격으로 시험할 뿐) | 수학적 증명 |
| **보안** | 새 공격에 무너질 수 있다 | 반지름 안에서는 증명된 보안 |
| **정확도** | 높음 | 낮음 |
| **인증** | 해당 없음 | 샘플마다 계산한다 |

---

## 2. 무작위 평활화

### 핵심 통찰

**무작위 평활화**(randomized smoothing, 코언 등, 2019)는 가우스 잡음에 대해 예측을 평균 내어, 어떤 분류기든 인증 가능하게 강건한 분류기로 바꾼다.

기반 분류기 $f: \mathbb{R}^d \to \mathcal{Y}$가 있을 때 **평활화된 분류기**를 만든다.

$$
g(\mathbf{x}) = \arg\max_c \mathbb{P}_{\boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma^2 I)}[f(\mathbf{x} + \boldsymbol{\epsilon}) = c]
$$

**직관:** 입력에 가우스 잡음을 더해 여러 번 예측하고, 그 예측들의 다수결을 따른다.

### 인증 정리

**정리(코언 등, 2019):** 잡음을 더한 입력 $\mathbf{x} + \boldsymbol{\epsilon}$에서 기반 분류기 $f$가 클래스 $c_A$를 내놓을 확률이 $p_A$이고, 그다음으로 확률이 큰 클래스 $c_B$의 확률이 $p_B$이면, 평활화된 분류기의 예측 $g(\mathbf{x}) = c_A$는 다음 $\ell_2$ 반지름 안에서 강건성이 인증된다.

$$
R = \frac{\sigma}{2}\left(\Phi^{-1}(p_A) - \Phi^{-1}(p_B)\right)
$$

여기서 $\Phi^{-1}$은 표준 정규 분포의 누적 분포 함수의 역함수다.

### 직관

- $p_A$는 잡음 섞인 입력 하나에서 기반 분류기가 $c_A$를 고를 확률, 곧 다수결에서 $c_A$가 얻는 표의 비율
- $p_A \gg p_B$이면 인증 반지름 $R$이 크다(확신이 큼)
- $p_A \approx p_B$이면 인증 반지름 $R$이 작다(망설임)
- $\sigma$가 반지름의 스케일을 정한다. 잡음이 클수록 인증 영역이 넓다

### 인증 알고리즘

**두 단계로 이루어진다.**

1. **선택:** 예측할 클래스를 찾는다
   - 잡음 섞인 예측을 $n_0$번 뽑는다
   - 다수결로 $\hat{c}$를 정한다

2. **인증:** 반지름을 추정한다
   - 잡음 섞인 예측을 $n$번 더 뽑는다
   - $p_A$의 신뢰 구간을 계산한다
   - 인증 반지름 $R$을 계산한다

### 몬테카를로 근사

확률은 샘플링으로 근사한다.

$$
\hat{p}_A = \frac{1}{N} \sum_{i=1}^N \mathbf{1}[f(\mathbf{x} + \boldsymbol{\epsilon}_i) = c_A], \quad \boldsymbol{\epsilon}_i \sim \mathcal{N}(0, \sigma^2 I)
$$

**클로퍼-피어슨 신뢰 구간**을 쓰면

$$
\mathbb{P}(p_A \geq \underline{p}_A) \geq 1 - \alpha
$$

이로써 $1 - \alpha$ 이상의 확률로 성립하는 인증 반지름을 얻는다.

---

## 3. PyTorch로 구현하기

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
    아무렇게나 매끄럽게 하여 밝혀 낸 강건성.
    
    가우스 잡음으로 미루어 봄을 매끄럽게 하여
    증명할 수 있는 L2 강건성 보장을 준다.
    
    Parameters
    ----------
    base_classifier : nn.Module
        매끄럽게 할 밑 분류기
    sigma : float
        가우스 잡음의 표준편차
        σ이 클수록 밝혀 낸 반지름은 크고 정확도는 낮다
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
            들임 그림 하나, 형태 (C, H, W)
        num_samples : int
            잡음 섞인 표본의 수
        batch_size : int
            다룰 배치 크기
            
        Returns
        -------
        counts : torch.Tensor
            클래스마다의 미루어 봄 셈
        """
        with torch.no_grad():
            # 클래스의 수를 가린다
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
        두 값 기여도의 믿음 아래끝을 셈한다.
        클로퍼-피어슨(정확한) 길을 쓴다.
        """
        if count == 0:
            return 0.0
        return binom.ppf(alpha / 2, n, count / n) / n
    
    def _compute_radius(self, p_A: float, p_B: float) -> float:
        """
        확률으로 밝혀 낸 반지름을 셈한다.
        
        R = σ/2 * (Φ^{-1}(p_A) - Φ^{-1}(p_B))
        """
        if p_A <= 0.5:
            return 0.0
        
        # 무한을 회피하려 잘라 낸다
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
            들임 그림, 형태 (C, H, W)
        n0 : int
            고르는 도막의 표본 수
        n : int
            밝히는 도막의 표본 수
        alpha : float
            신뢰 수준(기본값: 99.9%)
        batch_size : int
            몬테카를로의 배치 크기
            
        Returns
        -------
        predicted_class : int
            미루어 본 클래스(삼가면 -1)
        certified_radius : float
            밝혀 낸 L2 반지름(삼가면 0)
        """
        x = x.to(self.device)
        
        # 1도막: 고르기
        counts_selection = self._sample_predictions(x, n0, batch_size)
        top_class = counts_selection.argmax().item()
        
        # 2도막: 밝히기
        counts_cert = self._sample_predictions(x, n, batch_size)
        
        # 으뜸 클래스의 셈
        count_top = counts_cert[top_class].item()
        
        # p_A의 믿음 아래끝
        p_A_lower = self._lower_confidence_bound(int(count_top), n, alpha)
        
        # p_A_lower <= 0.5이면 밝힐 수 없다
        if p_A_lower <= 0.5:
            return -1, 0.0  # 삼간다
        
        # 밝혀 낸 반지름을 셈한다
        # 둘 가름에서는 p_B_upper = 1 - p_A_lower
        # 여러 클래스에서는 더 조심스러운 테두리를 쓴다
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
        그림 배치를 밝힌다.
        
        Returns
        -------
        results : dict
            - clean_accuracy: 옳게 미루어 본 기여도
            - certified_accuracy_r=X: 반지름 X에서 밝혀 낸 기여도
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
        
        # 반지름마다 밝혀 낸 정확도
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
        """클래스를 미루어 본다(밝히기 없이)."""
        counts = self._sample_predictions(x, n, batch_size)
        return counts.argmax().item()

class SmoothClassifier(nn.Module):
    """
    익힘과 미루어 봄에서 분류기를 매끄럽게 해 주는 감싸개.
    """
    
    def __init__(self, base_classifier: nn.Module, sigma: float, num_samples: int = 1):
        super().__init__()
        self.base_classifier = base_classifier
        self.sigma = sigma
        self.num_samples = num_samples
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        잡음을 불려 앞으로 스텝.
        
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

### 사용 예

```python
import torchvision
import torchvision.transforms as transforms

# 모형을 얹는다
base_model = torchvision.models.resnet18(num_classes=10)
base_model.load_state_dict(torch.load('cifar10_resnet18.pth'))

# 매끄럽게 한 분류기를 만든다
smoother = RandomizedSmoothing(base_model, sigma=0.25)

# 시험 자료를 얹는다
transform = transforms.ToTensor()
testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

# 배치 하나를 얻는다
images, labels = next(iter(test_loader))

# 밝힌다
results = smoother.certify_batch(
    images[:50], labels[:50],  # 보기 50개를 밝힌다(느리다)
    n=10000,                    # 보기마다 표본 1만 개
    alpha=0.001                 # 믿음 99.9%
)

print("밝히기 결과:")
print(f"  맑은 정확도: {results['clean_accuracy']:.2%}")
print(f"  삼간 비율: {results['abstain_rate']:.2%}")
print(f"  밝혀 낸 반지름 평균: {results['avg_certified_radius']:.4f}")
print(f"  r=0.25에서 설명: {results['certified_accuracy_r=0.25']:.2%}")
print(f"  r=0.50에서 설명: {results['certified_accuracy_r=0.5']:.2%}")
print(f"  r=1.00에서 설명: {results['certified_accuracy_r=1.0']:.2%}")
```

---

## 4. 강건성 인증을 위한 학습

### 가우스 데이터 어그멘테이션

가장 단순한 방법은 가우스 잡음으로 데이터를 증강해 학습하는 것이다.

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

### 일관성 정칙화

서로 다른 잡음 샘플에서도 예측이 일관되도록 이끈다.

$$
\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda \cdot \text{KL}(f(\mathbf{x} + \boldsymbol{\epsilon}_1) \| f(\mathbf{x} + \boldsymbol{\epsilon}_2))
$$

---

## 5. 핵심 파라메터와 트레이드오프

### 잡음의 크기 시그마

| $\sigma$ | 인증 반지름 | 클린 정확도 |
|----------|------------------|----------------|
| 0.12 | 작음(약 0.25) | 높음(약 85%) |
| 0.25 | 중간(약 0.5) | 중간(약 75%) |
| 0.50 | 큼(약 1.0) | 낮음(약 60%) |
| 1.00 | 아주 큼(약 2.0) | 아주 낮음(약 40%) |

**트레이드오프:** $\sigma$가 클수록 인증 영역은 넓어지나 정확도는 낮아진다.

### 샘플링 파라메터

| 파라메터 | 값 | 영향 |
|-----------|-------|--------|
| $n_0$(선택) | 100 | 클수록 클래스 선택이 믿을 만하다 |
| $n$(인증) | 10,000 이상 | 클수록 신뢰 구간이 좁다 |
| $\alpha$(유의 수준) | 0.001 | 작을수록 보수적이다 |

### 계산 비용

인증은 **비싸다**.

- 입력마다 샘플 $n = 10,000$개
- 샘플마다 순전파를 한 번씩 해야 한다
- 테스트 샘플 1만 개를 인증하려면 순전파가 1억 번 든다

---

## 6. 비교: 경험적 강건성과 인증된 강건성

**CIFAR-10, $\varepsilon = 0.5$(L2):**

| 방법 | 클린 정확도 | 강건 정확도 | 인증되나? |
|--------|-----------|------------|------------|
| 표준 학습 | 95% | 0% | 아니다 |
| PGD 적대적 학습 | 85% | 약 50% | 아니다 |
| 무작위 평활화 | 75% | 약 60% | **그렇다** |

이 표에서는 무작위 평활화의 인증 정확도가 PGD 적대적 학습의 경험적 강건 정확도보다 높다. 서로 다른 모델을 견준 것이기 때문이다. 같은 모델이라면 순서가 거꾸로다.

- 인증 정확도는 참 강건 정확도의 아래끝이다(보장된 값)
- 경험적 강건 정확도는 위끝이다. 공격이 가장 나쁜 적대적 샘플을 못 찾으면 실제보다 높게 나온다

---

## 7. 한계

1. **L2 노름만**: 무작위 평활화는 L∞가 아니라 L2 섭동을 인증한다
2. **정확도 하락**: 클린 정확도가 꽤 줄어든다
3. **계산 비용**: 인증이 느리다
4. **확장의 어려움**: 큰 모델이나 데이터셋에는 만만치 않다

---

## 8. 심화 주제

### L∞의 강건성 인증

**구간 바운드 전파(IBP)** 같은 방법이 L∞ 인증을 준다.

$$
[\underline{z}, \overline{z}] = \text{IBP}(f, [\mathbf{x} - \varepsilon, \mathbf{x} + \varepsilon])
$$

$\underline{z}_y > \max_{i \neq y} \overline{z}_i$이면 그 예측은 인증된다.

### 더 타이트한 인증

- **SmoothAdv**: 적대적 학습 + 평활화
- **MACER**: 학습하는 동안 인증 반지름을 최대화한다
- **디노이징 평활화(denoised smoothing)**: 디노이저를 학습해 기반 분류기의 정확도를 올린다

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

| 통찰 | 핵심 |
|---------|-----------|
| **무작위 평활화** | 가우스 잡음에 대해 예측을 평균 낸다 |
| **인증 반지름** | $R = \frac{\sigma}{2}(\Phi^{-1}(p_A) - \Phi^{-1}(p_B))$ |
| **트레이드오프** | $\sigma$가 클수록 $R$은 크고 정확도는 낮다 |
| **보장** | $\|\boldsymbol{\delta}\|_2 \leq R$인 모든 섭동에 대해 증명된다 |

강건성 인증은 가장 강한 이론적 보장을 주지만, 그 대가로 정확도와 계산 비용을 치러야 한다.

**살펴볼 거리**

1. Cohen, J., Rosenfeld, E., & Kolter, Z. (2019). "Certified Adversarial Robustness via Randomized Smoothing." ICML.
2. Salman, H., et al. (2019). "Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers." NeurIPS.
3. Zhai, R., et al. (2020). "MACER: Attack-Free and Scalable Robust Training via Maximizing Certified Radius." ICLR.
