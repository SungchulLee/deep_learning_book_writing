# 인셉션 점수(IS)
## 훑어보기

인셉션 점수(IS)는 만들어 내는 모델, 그 가운데서도 맞겨루기 만들개 그물(GAN)을 따지는 데 가장 널리 쓰이는 자의 하나다. Salimans 외(2016)가 내놓았으며, 만들어 낸 그림의 품질과 다양함을 한꺼번에 담아내는 스칼라 값 하나를 준다.

!!! info "배움 목표"
    이 마당을 마치면 다음을 할 수 있다.
    
    - 인셉션 점수의 수학 바탕을 이해한다
    - PyTorch로 인셉션 점수 셈을 맨바닥부터 짠다
    - 인셉션 점숫값을 올바로 읽고 그 한계를 안다
    - 실제 따지기 흐름에 인셉션 점수를 쓴다

## 수학 바탕

### 고갱이 식

인셉션 점수는 다음과 같이 뜻매김한다.

$$
\text{IS} = \exp\left(\mathbb{E}_{x \sim p_g}\left[D_{KL}(p(y|x) \| p(y))\right]\right)
$$

여기서

- $x$는 만들개 분포 $p_g$에서 뽑은, 만들어 낸 그림이다
- $p(y|x)$는 그림 $x$가 주어졌을 때의 조건 갈래 분포다(InceptionV3이 준다)
- $p(y) = \mathbb{E}_{x}[p(y|x)]$는 가장자리 갈래 분포다
- $D_{KL}$은 쿨백-라이블러 갈림이다

### 각 조각의 뜻

**조건 분포 $p(y|x)$:**

이는 인셉션 가름개가 그림의 갈래를 얼마나 자신 있게 보는지를 나타낸다. 뾰족하게 솟은 분포는 가름개가 자신 있다는 뜻이며, 그림에 또렷이 알아볼 수 있는 물체가 있음을 넌지시 알려 준다.

$$
p(y|x) = \text{softmax}(f_{\text{Inception}}(x))
$$

여기서 $f_{\text{Inception}}(x)$는 ImageNet 갈래 1000개에 대한 로짓을 돌려준다.

**가장자리 분포 $p(y)$:**

이는 만들어 낸 모든 그림에 걸친 갈래 분포의 평균이다.

$$
p(y) = \frac{1}{N}\sum_{i=1}^{N} p(y|x_i)
$$

가장자리 분포가 고르면 만들개가 여러 갈래를 두루 덮는 다양한 그림을 내놓는다는 뜻이다.

**KL 갈림:**

KL 갈림은 조건 분포가 가장자리 분포와 얼마나 다른지를 잰다.

$$
D_{KL}(p(y|x) \| p(y)) = \sum_{c=1}^{C} p(y=c|x) \log\frac{p(y=c|x)}{p(y=c)}
$$

### 인셉션 점수가 참으로 재는 것

| 조각 | 값이 크면 | 값이 작으면 |
|-----------|---------------------|---------------------|
| $p(y\|x)$의 엔트로피 | 헤아림이 흐릿하다 | 헤아림이 자신 있다(품질) |
| $p(y)$의 엔트로피 | 갈래가 다양하다(다양함) | 최빈값 무너짐 |
| KL 갈림 | 품질과 다양함을 모두 갖춤 | 품질이 나쁘거나 다양함이 적음 |

인셉션 점수는 두 갈래를 한꺼번에 담아낸다.

- **품질**: 그림마다 자신 있는 가름이 나와야 한다($p(y|x)$의 엔트로피가 낮다)
- **다양함**: 만들어 낸 그림이 여러 갈래를 두루 덮어야 한다($p(y)$의 엔트로피가 높다)

## 수학으로 이끌어 내기

### KL 갈림 펼치기

뜻매김에서 비롯한다.

$$
\begin{aligned}
D_{KL}(p(y|x) \| p(y)) &= \sum_{y} p(y|x) \log\frac{p(y|x)}{p(y)} \\
&= \sum_{y} p(y|x) \log p(y|x) - \sum_{y} p(y|x) \log p(y) \\
&= -H(y|x) + H_{\text{cross}}(p(y|x), p(y))
\end{aligned}
$$

여기서 $H(y|x)$는 조건 엔트로피다.

### 기댓값

만들어 낸 표본에 대해 기댓값을 취하면 다음과 같다.

$$
\mathbb{E}_x[D_{KL}(p(y|x) \| p(y))] = -\mathbb{E}_x[H(y|x)] + H(y)
$$

첫째 마디는 **평균 조건 엔트로피**(품질을 보려면 낮을수록 좋다)를 나타내고, 둘째 마디는 **가장자리 엔트로피**(다양함을 보려면 높을수록 좋다)다.

### 마지막 점수

$$
\text{IS} = \exp\left(H(y) - \mathbb{E}_x[H(y|x)]\right)
$$

이는 만들개가 자신 있는 헤아림으로 내놓을 수 있는 **실질 갈래 수**로 읽을 수 있다.

## PyTorch 짜기

### 맨바닥부터 온전히 짜기

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from scipy import stats


class InceptionScoreCalculator:
    """
    자세한 풀이를 곁들인 두루 갖춘 인셉션 점수 셈틀.
    
    인셉션 점수는 미리 익힌 인셉션 그물의 갈래 헤아림을 살펴
    만들어 낸 그림의 품질과 다양함을 함께 잰다.
    """
    
    def __init__(self, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        인셉션 점수 셈틀의 첫자리를 잡는다.
        
        인자:
            device: 셈할 장치('cuda' 또는 'cpu')
        """
        self.device = device
        self.inception_model = None
        
    def _load_inception(self):
        """ImageNet으로 미리 익힌 InceptionV3 모델을 불러온다."""
        from torchvision.models import inception_v3, Inception_V3_Weights
        
        # 미리 익힌 InceptionV3을 불러온다
        self.inception_model = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1,
            transform_input=False  # 미리 다듬기는 우리가 손수 한다
        )
        self.inception_model.eval()
        self.inception_model.to(self.device)
        
        # 곁들이 내놓음을 끈다
        self.inception_model.aux_logits = False
        
    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        InceptionV3에 맞게 그림을 미리 다듬는다.
        
        InceptionV3은 다음을 바란다.
        - 크기가 299×299인 그림
        - ImageNet 평균과 표준편차로 잣대를 맞춘 그림
        
        인자:
            images: [0, 1] 범위의 들임 그림 [B, C, H, W]
            
        돌려주는 값:
            인셉션에 바로 넣을 수 있게 다듬은 그림
        """
        # 필요하면 299×299로 크기를 바꾼다
        if images.shape[2] != 299 or images.shape[3] != 299:
            images = F.interpolate(
                images, 
                size=(299, 299), 
                mode='bilinear', 
                align_corners=False
            )
        
        # 필요하면 잿빛을 RGB로 바꾼다
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        
        # ImageNet 통계로 잣대를 맞춘다
        # 눈여겨볼 것: 인셉션은 속으로 [-1, 1] 범위를 바란다
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
        
        images = (images - mean) / std
        
        return images
    
    def get_predictions(self, 
                       images: torch.Tensor, 
                       batch_size: int = 32) -> np.ndarray:
        """
        그림 묶음에 대한 인셉션 헤아림을 얻는다.
        
        인자:
            images: [0, 1] 범위의 만들어 낸 그림 [N, C, H, W]
            batch_size: 다룰 때 쓰는 묶음 크기
            
        돌려주는 값:
            소프트맥스 확률 [N, 1000]
        """
        if self.inception_model is None:
            self._load_inception()
            
        all_probs = []
        n_images = len(images)
        
        with torch.no_grad():
            for i in range(0, n_images, batch_size):
                batch = images[i:i+batch_size].to(self.device)
                batch = self._preprocess_images(batch)
                
                # 인셉션을 지나 앞으로 걸음
                logits = self.inception_model(batch)
                
                # 소프트맥스를 걸어 확률을 얻는다
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
        
        return np.concatenate(all_probs, axis=0)
    
    def calculate_inception_score(self,
                                  images: torch.Tensor,
                                  splits: int = 10,
                                  batch_size: int = 32) -> Tuple[float, float]:
        """
        믿음 구간과 함께 인셉션 점수를 셈한다.
        
        차례:
        1. 그림마다 인셉션에서 p(y|x)을 얻는다
        2. 흩어짐을 셈하려고 자료를 `splits`개 무리로 나눈다
        3. 조각마다:
           a. 가장자리 분포 p(y) = mean(p(y|x))을 셈한다
           b. 표본마다 KL(p(y|x) || p(y))을 셈한다
           c. KL을 평균 내고 지수를 취한다
        4. 조각들의 평균과 표준편차를 돌려준다
        
        인자:
            images: [0, 1] 범위의 만들어 낸 그림 [N, C, H, W]
            splits: 표준편차를 셈할 때 나눌 조각의 수
            batch_size: 인셉션 미룸에 쓰는 묶음 크기
            
        돌려주는 값:
            (인셉션 점수 평균, 표준편차) 튜플
        """
        # 헤아림을 얻는다
        probs = self.get_predictions(images, batch_size)
        
        # 조각을 나누어 인셉션 점수를 셈한다
        scores = []
        n = len(probs)
        split_size = n // splits
        
        for k in range(splits):
            # 조각을 가져온다
            start = k * split_size
            end = start + split_size if k < splits - 1 else n
            part = probs[start:end]
            
            # 가장자리 분포를 셈한다: p(y) = (1/N) Σ p(y|x_i)
            p_y = np.mean(part, axis=0, keepdims=True)
            
            # 표본마다 KL 갈림을 셈한다
            # KL(p(y|x) || p(y)) = Σ p(y|x) * log(p(y|x) / p(y))
            eps = 1e-16
            part = np.clip(part, eps, 1.0)
            p_y = np.clip(p_y, eps, 1.0)
            
            # 로그 비
            log_ratio = np.log(part) - np.log(p_y)
            
            # 표본마다의 KL 갈림
            kl_per_sample = np.sum(part * log_ratio, axis=1)
            
            # KL을 평균 내고 지수를 취한다
            mean_kl = np.mean(kl_per_sample)
            is_score = np.exp(mean_kl)
            
            scores.append(is_score)
        
        return float(np.mean(scores)), float(np.std(scores))


def compute_inception_score_step_by_step(probs: np.ndarray) -> dict:
    """
    배움을 돕고자 중간 결과까지 자세히 내며 인셉션 점수를 셈한다.
    
    이 함수는 인셉션 점수 셈을 읽기 쉬운 걸음으로 나누어
    각 조각이 무엇을 재는지 알기 쉽게 한다.
    
    인자:
        probs: 인셉션이 낸 갈래 확률 [N, C]
        
    돌려주는 값:
        중간 값과 마지막 인셉션 점수를 담은 사전
    """
    eps = 1e-16
    probs = np.clip(probs, eps, 1.0)
    
    # 걸음 1: 가장자리 분포 p(y)을 셈한다
    # 이는 모든 표본에 걸친 갈래 분포를 나타낸다
    p_y = np.mean(probs, axis=0)
    
    # 걸음 2: 가장자리 분포의 엔트로피 H(y)을 셈한다
    # 엔트로피가 클수록 표본이 다양하다(갈래를 더 두루 덮는다)
    h_marginal = -np.sum(p_y * np.log(p_y))
    
    # 걸음 3: 표본마다 조건 엔트로피 H(y|x)을 셈한다
    # 조건 엔트로피가 작을수록 헤아림이 자신 있다(품질이 높다)
    h_conditional_per_sample = -np.sum(probs * np.log(probs), axis=1)
    h_conditional = np.mean(h_conditional_per_sample)
    
    # 걸음 4: KL 갈림을 셈한다
    # 기댓값으로 보면 KL(p(y|x) || p(y)) = H(y) - H(y|x)
    # 다만 정확함을 위해 곧바로 셈한다
    kl_per_sample = np.sum(probs * (np.log(probs) - np.log(p_y)), axis=1)
    mean_kl = np.mean(kl_per_sample)
    
    # 걸음 5: 마지막 인셉션 점수 = exp(mean_kl)
    inception_score = np.exp(mean_kl)
    
    # 덧붙이는 눈썰미
    effective_classes = np.exp(h_marginal)  # 실제로 쓰인 갈래의 실질 수
    avg_confidence = np.exp(-h_conditional)  # 헤아림의 평균 자신도
    
    return {
        'inception_score': inception_score,
        'mean_kl_divergence': mean_kl,
        'marginal_entropy': h_marginal,
        'conditional_entropy': h_conditional,
        'effective_classes': effective_classes,
        'avg_confidence': avg_confidence,
        'marginal_distribution': p_y
    }
```

### 실제로 쓰는 보기

```python
import torch
import matplotlib.pyplot as plt


def demonstrate_inception_score():
    """
    품질이 다른 여러 상황에서 인셉션 점수 셈을 보인다.
    """
    n_samples = 1000
    n_classes = 10  # 보이기 위해 단출하게 줄였다
    
    print("=" * 70)
    print("Inception Score Demonstration")
    print("=" * 70)
    
    # 상황 1: 높은 품질 + 높은 다양함(가장 바람직)
    print("\n📊 Scenario 1: High Quality + High Diversity")
    print("-" * 50)
    
    probs_ideal = np.zeros((n_samples, n_classes))
    for i in range(n_samples):
        class_idx = i % n_classes  # 고르게 덮는다
        probs_ideal[i, class_idx] = 0.9
        probs_ideal[i, :] += 0.01  # 작고 고른 잡음
    probs_ideal = probs_ideal / probs_ideal.sum(axis=1, keepdims=True)
    
    results_ideal = compute_inception_score_step_by_step(probs_ideal)
    print(f"  IS: {results_ideal['inception_score']:.2f}")
    print(f"  Effective classes: {results_ideal['effective_classes']:.2f}")
    print(f"  Average confidence: {results_ideal['avg_confidence']:.4f}")
    
    # 상황 2: 품질이 낮다(헤아림이 흐릿하다)
    print("\n📊 Scenario 2: Low Quality (Uncertain Predictions)")
    print("-" * 50)
    
    probs_uncertain = np.ones((n_samples, n_classes)) / n_classes
    
    results_uncertain = compute_inception_score_step_by_step(probs_uncertain)
    print(f"  IS: {results_uncertain['inception_score']:.2f}")
    print(f"  Effective classes: {results_uncertain['effective_classes']:.2f}")
    print(f"  Average confidence: {results_uncertain['avg_confidence']:.4f}")
    print("  Note: Minimum IS = 1.0 when all predictions are uniform")
    
    # 상황 3: 봉우리 무너짐(갈래가 하나뿐)
    print("\n📊 Scenario 3: Mode Collapse (Single Class)")
    print("-" * 50)
    
    probs_collapse = np.zeros((n_samples, n_classes))
    probs_collapse[:, 0] = 0.95
    probs_collapse[:, 1:] = 0.05 / (n_classes - 1)
    
    results_collapse = compute_inception_score_step_by_step(probs_collapse)
    print(f"  IS: {results_collapse['inception_score']:.2f}")
    print(f"  Effective classes: {results_collapse['effective_classes']:.2f}")
    print(f"  Note: Confident but not diverse!")
    
    return {
        'ideal': results_ideal,
        'uncertain': results_uncertain,
        'collapse': results_collapse
    }


# 보임을 돌린다
results = demonstrate_inception_score()
```

**출력:**

```
======================================================================
Inception Score Demonstration
======================================================================

📊 Scenario 1: High Quality + High Diversity
--------------------------------------------------
  IS: 6.06
  Effective classes: 10.00
  Average confidence: 0.6064

📊 Scenario 2: Low Quality (Uncertain Predictions)
--------------------------------------------------
  IS: 1.00
  Effective classes: 10.00
  Average confidence: 0.1000
  Note: Minimum IS = 1.0 when all predictions are uniform

📊 Scenario 3: Mode Collapse (Single Class)
--------------------------------------------------
  IS: 1.00
  Effective classes: 1.36
  Note: Confident but not diverse!
```

## 인셉션 점숫값 읽기

### 흔한 범위

| 인셉션 점숫값 | 품질 수준 | 풀이 |
|----------|---------------|----------------|
| 2.0 미만 | 매우 나쁨 | 그림을 알아볼 수 없거나 헤아림이 매우 흐릿하다 |
| 2.0~5.0 | 나쁨에서 보통 | 짜임새는 있으나 품질이나 다양함이 모자라다 |
| 5.0~8.0 | 좋음 | 또렷한 그림에 웬만한 다양함을 갖췄다 |
| 8.0 초과 | 아주 좋음 | 품질 높고 다양한 그림을 만들어 낸다 |
| 11.2쯤 | 참 ImageNet | 참 ImageNet 그림으로 잰 잣대 |

### 이론상의 한계

**가장 작은 인셉션 점수 = 1.0**: 모든 $x$에서 $p(y|x) = p(y)$일 때다(헤아림이 고르다).

**가장 큰 인셉션 점수**: 이론상 갈래 수(ImageNet이면 1000)로 막히며, 그림마다 서로 다른 갈래로 나무랄 데 없이 갈릴 때 이른다.

## 한계와 함정

### 1. 외워 버림을 알아채지 못한다

인셉션 점수는 새 그림을 만들어 내는 모델과 익힘 자료를 그저 외운 모델을 가려내지 못한다.

```python
def demonstrate_memorization_blindness():
    """
    모델이 익힘 자료를 외워도 인셉션 점수는 알아채지 못함을 보인다.
    """
    # 같은 그림 10장을 나무랄 데 없이 만들어 내는 모델도
    # 그 그림들이 자신 있게 갈리면 인셉션 점수가 높게 나온다
    n_unique = 10
    n_total = 1000
    
    probs_memorized = np.zeros((n_total, 10))
    for i in range(n_total):
        class_idx = i % n_unique  # 서로 다른 "그림"은 10장뿐이다
        probs_memorized[i, class_idx] = 0.95
        probs_memorized[i, :] += 0.005
    
    probs_memorized = probs_memorized / probs_memorized.sum(axis=1, keepdims=True)
    results = compute_inception_score_step_by_step(probs_memorized)
    
    print(f"IS with memorization: {results['inception_score']:.2f}")
    print("This is HIGH despite only 10 unique images!")
```

### 2. 갈래 안의 다양함을 놓친다

인셉션 점수는 갈래 사이의 다양함만 잴 뿐 갈래 안의 눈에 보이는 다양함은 재지 않는다.

- 똑같은 고양이 그림 1000장 → 높은 인셉션 점수("고양이"로 자신 있게 갈린다)
- 그러나 눈에 보이는 다양함은 하나도 없다!

### 3. 자료 묶음에 매여 있다

인셉션 점수는 ImageNet 같은 자연 그림에서만 뜻이 있다. 다음에서는 어그러질 수 있다.

- 의료 그림
- 인공위성 그림
- 추상 미술
- 특정 마당의 그림

### 4. 주무를 수 있다

맞겨루기 꾀로 인셉션 점수를 억지로 부풀릴 수 있다.

```python
def demonstrate_gaming_is():
    """
    맞겨루기 꾀로 인셉션 점수를 어떻게 '주무를' 수 있는지 보인다.
    """
    # 꾀: 갈래마다 그림을 꼭 하나씩만 만든다
    n_classes = 1000
    probs_gamed = np.eye(n_classes)  # 갈래마다 나무랄 데 없이 갈린다
    
    results = compute_inception_score_step_by_step(probs_gamed)
    print(f"Gamed IS: {results['inception_score']:.2f}")
    print("Maximum possible IS with only 1000 unique images!")
```

## 가장 좋은 버릇

### 1. 표본 수

```python
def analyze_sample_size_effect(generator, sample_sizes=[100, 500, 1000, 5000, 10000]):
    """
    표본 수가 인셉션 점수의 든든함에 어떤 영향을 주는지 살핀다.
    """
    calculator = InceptionScoreCalculator()
    
    results = []
    for n in sample_sizes:
        images = generator.generate(n)
        is_mean, is_std = calculator.calculate_inception_score(images)
        results.append({
            'n_samples': n,
            'is_mean': is_mean,
            'is_std': is_std,
            'relative_std': is_std / is_mean
        })
    
    return results
```

**권함:**

- 가장 적어도: 표본 5,000개
- 권함: 표본 10,000개 이상
- 늘 믿음 구간을 함께 알린다

### 2. 흩어짐을 어림하기 위한 조각 나누기

```python
# 흔한 방식: 조각 10개
is_mean, is_std = calculator.calculate_inception_score(images, splits=10)

# 이렇게 알린다: 인셉션 점수 = 평균 ± 표준편차
print(f"IS = {is_mean:.2f} ± {is_std:.2f}")
```

### 3. 다른 자와 함께 쓰기

인셉션 점수만 홀로 써서는 안 된다. 늘 다음과 함께 쓰라.

- **FID**: 최빈값 무너짐을 더 잘 알아낸다
- **정밀도·재현율**: 품질과 덮음의 절충을 잰다
- **눈으로 살피기**: 사람의 판단은 여전히 꼭 있어야 한다

## 정보 이론과의 이음

인셉션 점수에는 아름다운 정보 이론의 풀이가 있다.

$$
\text{IS} = \exp\left(I(X; Y)\right)
$$

여기서 $I(X; Y)$는 만들어 낸 그림 $X$와 그 헤아린 갈래 $Y$ 사이의 서로 정보다.

**서로 정보는 다음과 같이 갈린다.**

$$
I(X; Y) = H(Y) - H(Y|X)
$$

- **$H(Y)$**: 갈래 헤아림의 엔트로피(다양함)
- **$H(Y|X)$**: 헤아림의 평균 흐릿함(품질)

서로 정보가 클수록 다음을 뜻한다.

- 만들어 낸 그림이 갈래 이름표에 대한 앎을 더 많이 담는다
- 품질과 다양함이 모두 좋게 이바지한다

## 간추림

!!! success "고갱이 얻음"
    
    1. **IS Formula**: $\text{IS} = \exp(\mathbb{E}[D_{KL}(p(y|x) \| p(y))])$
    
    2. **Measures Both**: Quality (confident predictions) and diversity (class coverage)
    
    3. **Range**: 1.0 (minimum) to ~1000 (theoretical max), real ImageNet ≈ 11.2
    
    4. **한계**: 외워 버림을 알아채지 못하고, 갈래 안의 다양함을 놓치며, ImageNet에 매여 있다
    
    5. **가장 좋은 버릇**: 표본 10,000개 이상, 조각 10개를 쓰고 FID와 눈으로 살피기를 함께 쓴다

## 참고 문헌

1. Salimans, T., et al. (2016). "Improved Techniques for Training GANs." *NeurIPS*.

2. Barratt, S., & Sharma, R. (2018). "A Note on the Inception Score." *ICML Workshop*.

3. Borji, A. (2019). "Pros and Cons of GAN Evaluation Measures." *Computer Vision and Image Understanding*.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
프레셰 인셉션 거리(FID)를 뜻매김하고, GAN을 따질 때 인셉션 점수보다 이를 더 치는 까닭을 밝혀라.

</div>

??? success "연습문제 1 풀이"
    FID는 참 그림과 만들어 낸 그림의 Inception-v3 특징 분포를 다변량 가우스 $\mathcal{N}(\mu_r, \Sigma_r)$와 $\mathcal{N}(\mu_g, \Sigma_g)$로 보고 다음을 셈한다.

    $$\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$

    FID를 더 치는 까닭은 이렇다. (1) 만들어 낸 그림을 참 그림과 견준다(인셉션 점수는 만들어 낸 그림만 따진다). (2) 최빈값 무너짐을 알아낸다(평균과 공분산이 달라진다). (3) 사람의 판단과 더 잘 들어맞는다. (4) FID가 낮을수록 느낌의 좋음과 더 잘 맞아떨어진다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
인셉션 점수의 한계는 무엇인가? 나쁜 표본을 내놓으면서도 높은 인셉션 점수를 얻을 수 있는가?

</div>

??? success "연습문제 2 풀이"
    인셉션 점수는 $\exp(\mathbb{E}_x [D_{\text{KL}}(p(y|x) \| p(y))])$이며 $p(y|x)$는 만들어 낸 그림 $x$에 대한 인셉션 가름개의 헤아림이다. 한계는 이렇다. (1) 품질(뾰족하고 가를 수 있음)과 다양함(갈래에 두루 퍼짐)만 잴 뿐 익힘 자료에 얼마나 충실한지는 재지 않는다. (2) 갈래마다 나무랄 데 없는 그림을 하나씩만 내놓아도 점수가 높지만 갈래 안의 다양함은 놓친다. (3) 인셉션 모델의 치우침에 흔들린다. (4) 갈래 안의 결과 결무늬 품질을 담아내지 못한다. 그렇다. ImageNet 갈래마다 대표 그림을 하나씩 외우기만 해도 높은 인셉션 점수를 얻을 수 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
만들어 내는 모델의 정밀도·재현율 자가 가름에서 쓰는 것과 어떻게 다른지 밝혀라.

</div>

??? success "연습문제 3 풀이"
    만들어 내기 자리에서는(Kynkaanniemi 외, 2019) 이렇다. **정밀도**는 만들어 낸 표본 가운데 참 자료 분포의 받침 안에 드는 몫을 잰다(품질, 곧 충실함). **재현율**은 참 자료 가운데 만들어 낸 분포의 받침 안에 드는 몫을 잰다(다양함, 곧 덮음). 정밀도가 높고 재현율이 낮으면 최빈값 무너짐이다(봉우리는 적지만 그럴듯하다). 정밀도가 낮고 재현율이 높으면 품질은 나쁘나 다양하다. 띄엄띄엄한 맞음을 세는 가름의 정밀도·재현율과 달리, 만들어 내기의 정밀도·재현율은 특징 자리에서 $k$번째 가장 가까운 이웃까지의 거리로 분포의 받침을 어림한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
만들어 내는 모델을 살필 때 왜 여러 따지기 자를 함께 써야 하는가?

</div>

??? success "연습문제 4 풀이"
    어느 자 하나도 만들어 내기 품질의 모든 면을 담아내지 못한다. **FID**는 분포가 두루 얼마나 닮았는지 재지만 품질과 다양함을 뒤섞는다. **인셉션 점수**는 품질과 다양함을 담아내지만 익힘 자료에 얼마나 충실한지는 놓친다. **정밀도·재현율**은 품질과 다양함을 갈라 보여 주지만 어떤 특징 뽑개와 $k$를 고르는지에 달렸다. **느낌의 자**(LPIPS)는 그림 하나하나의 품질은 재지만 다양함은 재지 않는다. 여러 자를 함께 쓰면 온 그림이 보인다. FID가 낮고 정밀도가 높으며 재현율이 낮은 모델은 최빈값이 무너진 것이고, 재현율은 높으나 정밀도가 낮은 모델은 다양하지만 품질이 낮은 표본을 내놓는다. 마지막 판단에서는 사람이 따지는 것이 여전히 으뜸 잣대다.


---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
인셉션 점수는 무엇을 재는가? 식의 두 조각이 각각 무엇을 벌하는가?

</div>

??? success "연습문제 5 풀이"
    $$\mathrm{IS} = \exp\Big( \mathbb{E}_x \big[ D_{\mathrm{KL}}(p(y \mid x) \,\|\, p(y)) \big] \Big)$$

    KL 안의 두 분포가 각각 다른 것을 본다.

    | 조각 | 무엇을 바라는가 | 무엇을 재는가 |
    |---|---|---|
    | $p(y \mid x)$ | 뾰족하기를 | 표본 하나가 또렷한가 (품질) |
    | $p(y)$ | 고르기를 | 여러 부류를 두루 내는가 (다양성) |

    KL이 크려면 낱낱은 뾰족하고 전체 평균은 고루 퍼져야 한다. 곧 "하나하나는 분명한
    무엇이면서 서로 다른 것들"을 바라는 셈이다.

    지수를 취하는 것은 읽기 좋게 만드는 것뿐이다. 부류가 $C$개일 때 가장 큰 값이 $C$이고
    가장 작은 값이 1이므로, MNIST에서는 1에서 10 사이가 된다.

    실제로 재어 보면 참 시험 자료가 **9.467**이다. 10에 가깝지만 닿지는 않으며, 분류기가
    완벽하지 않기 때문이다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
인셉션 점수가 놓치는 것은 무엇인가? 수치로 보이라.

</div>

??? success "연습문제 6 풀이"
    **부류 안의 다양성**을 전혀 보지 못한다. 이렇게 확인할 수 있다.

    부류마다 그림 한 장씩 열 장만 골라 100번 되풀이한 1,000개 묶음을 만든다. 서로 다른
    그림이 **열 장뿐**이다. 그런데도 점수가 이렇다.

    | 묶음 | 서로 다른 그림 | IS | FID |
    |---|---|---|---|
    | 참 자료 1,000개 | 1,000 | 9.290 | 18.7 |
    | 그림 10장을 100번 되풀이 | **10** | **9.164** | **356.5** |

    IS가 9.164로 참 자료의 9.290과 **거의 같다.** 되풀이한 묶음이 조건을 완벽히 만족하기
    때문이다. 낱낱은 또렷한 숫자이고($p(y\mid x)$가 뾰족하다) 열 부류가 정확히 고르다
    ($p(y)$가 균일하다).

    FID는 356.5로 곧바로 잡아낸다. 특징 공간의 **분포**를 견주므로 다양성이 없으면
    공분산이 어긋나는 것을 본다.

    이 한 줄이 왜 FID가 IS를 대체했는지를 설명한다. IS로는 **그림 열 장을 외운 모델이
    완벽한 점수를 받는다.**

    IS가 아무것도 못 보는 것은 아니다. 부류 수준의 쏠림은 본다. 참 자료인데 부류를 셋으로
    줄이면 IS가 3.170으로 떨어진다. 곧 **부류 사이의 다양성은 보고 부류 안의 다양성은
    못 본다.**

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff hard" title="어려움"></span>
인셉션 점수를 여러 조각으로 나누어 재는 관례가 있다. 조심할 점은 무엇인가?

</div>

??? success "연습문제 7 풀이"
    표본을 열 조각으로 나누어 각각 점수를 재고 평균과 표준편차를 적는 것이 관례다.
    편차를 보고할 수 있어 좋다.

    그런데 **표본의 순서에 딸린다.** 조각마다 $p(y)$를 따로 셈하기 때문이다.

    부류 순서대로 늘어놓은 참 자료 1,000개로 재어 보면 이렇다.

    | | 순서대로 | 섞어서 |
    |---|---|---|
    | 참 자료 1,000개 (부류 고르게) | **2.146** | **9.290** |
    | 그림 10장을 100번 되풀이 | 9.418 | 9.164 |

    첫 줄이 네 배 넘게 차이 난다. 부류 순서대로 늘어놓으면 조각마다 한두 부류만 들어가므로
    그 조각 안의 $p(y)$가 뾰족해지고 KL이 작아진다. **같은 자료에 같은 식인데 값이
    2.146과 9.290으로 갈린다.**

    그러므로 **반드시 섞은 뒤 나누어야** 한다.

    ```python
    rng = np.random.default_rng(seed)
    rng.shuffle(probs)                      # 이 한 줄이 빠지면 값이 틀린다
    ```

    나는 이 장의 수치를 재면서 실제로 이것을 빠뜨렸다. 처음 얻은 표에서 참 자료가
    2.146으로 나와 되풀이한 묶음(9.418)보다 낮았는데, 그 이상한 결과가 버그를 알려 준
    셈이다. **값이 뜻과 어긋나면 대개 짜기를 의심해야 한다.**

    조각 수도 정해 두어야 한다. 열 조각이 관례이고, 표본이 적으면 조각마다 표본이 너무
    적어져 불안정해진다.

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
MNIST에서 인셉션 그물을 그대로 쓸 수 있는가?

</div>

??? success "연습문제 8 풀이"
    쓸 수는 있지만 뜻이 약하다.

    인셉션 그물은 ImageNet의 1,000부류로 익힌 것이라 손글씨 숫자를 본 적이 없다. MNIST
    그림을 넣으면 $p(y\mid x)$가 엉뚱한 부류에 퍼지므로, 뾰족함과 고름을 재는 일이
    자료와 무관해진다.

    그래서 MNIST에서는 **MNIST로 익힌 분류기**를 쓰는 것이 관례다. 이 장의 수치도 그렇게
    쟀고(시험 정확도 98.50%), 128차원 은닉층을 특징으로 쓴다.

    대가가 있다. **논문의 값과 직접 견줄 수 없다.** 인셉션 기준 FID 20과 이 장의 FID 20은
    다른 잣대의 값이다.

    그러므로 이런 수치를 적을 때는 **무엇으로 재었는지 반드시 밝혀야** 한다. 같은 글 안에서
    모델끼리 견주는 데는 문제가 없고, 다른 글의 수와 견줄 때만 조심하면 된다.

    입력 크기와 채널도 손봐야 한다. 인셉션은 $299\times299$ 세 채널을 받으므로 MNIST를
    키우고 채널을 늘려야 하는데, 그 보간이 또 하나의 임의적 선택을 더한다. 이 역시 값을
    견주기 어렵게 만드는 요인이다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
인셉션 점수를 올리는 쉬운 방법이 있는가?

</div>

??? success "연습문제 9 풀이"
    있고, 그것이 이 잣대의 약점이다.

    **부류마다 가장 잘 나온 그림 하나씩만 내놓으면** 점수가 거의 최대가 된다. 앞에서 본
    대로 열 장만으로 9.164가 나온다.

    다른 길도 있다.

    - 분류기가 확신하는 쪽으로 그림을 다듬는다(대립 표본을 만드는 것과 같은 일이다)
    - 애매한 표본을 골라 버린다. 낱낱의 뾰족함이 올라간다
    - 부류 비율을 억지로 고르게 맞춘다

    셋째가 특히 흔하다. 조건부 모델에서 부류를 고루 지정해 뽑으면 $p(y)$가 완벽히 균일해져
    점수가 올라가는데, 모델이 좋아진 것은 아니다.

    이런 일이 가능한 까닭은 IS가 **모델의 분포를 자료의 분포와 견주지 않기** 때문이다.
    만들어 낸 표본만 보고 점수를 매기므로 참 자료가 식에 들어오지 않는다. 참 자료와
    견주는 잣대라면 열 장을 되풀이하는 것이 곧바로 들킨다.

    [24장에서 겪은 온도 함정](../../ch24/training/generate_samples.md)과 같은 이야기다.
    **잣대를 올리는 방법과 모델을 좋게 하는 방법이 갈릴 수 있다.** 그런 잣대는 고를 때
    쓰면 안 된다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
인셉션 점수와 조건부 엔트로피의 관계를 적어라.

</div>

??? success "연습문제 10 풀이"
    KL을 펼치면 엔트로피 둘의 차이가 된다.

    $$\mathbb{E}_x\big[D_{\mathrm{KL}}(p(y\mid x) \| p(y))\big]
      = \underbrace{H(y)}_{\text{전체 엔트로피}} - \underbrace{H(y \mid x)}_{\text{조건부 엔트로피}}
      = I(x; y)$$

    곧 **IS의 로그가 표본과 부류 사이의 서로 앎**이다.

    $$\mathrm{IS} = \exp\big( I(x;y) \big)$$

    이렇게 보면 두 조각이 더 또렷해진다. $H(y)$가 크기를 바라는 것이 다양성이고
    $H(y\mid x)$가 작기를 바라는 것이 품질이다.

    그리고 한계도 또렷해진다. **서로 앎은 $y$에 대한 것**이다. $y$가 부류 표지뿐이므로
    부류를 넘어서는 정보는 애초에 식에 들어올 자리가 없다. 그림 열 장을 되풀이해도
    $I(x;y)$가 줄지 않는 까닭이다. 부류를 맞히는 데 필요한 정보는 그대로 있다.

    상한도 바로 나온다. $I(x;y) \le H(y) \le \log C$이므로 $\mathrm{IS} \le C$다. MNIST에서
    10이다.

    [KL 항의 정보 이론적 읽기](../../ch24/theory/kl_term.md)와 같은 도구가 여기서도
    쓰인다는 점을 눈여겨볼 만하다.

---

<div class="drillbox" markdown>

**연습문제 11.** <span class="diff easy" title="쉬움"></span>
인셉션 점수를 보고할 때 무엇을 함께 적어야 하는가?

</div>

??? success "연습문제 11 풀이"
    적어도 이 다섯이다.

    | 적을 것 | 왜 |
    |---|---|
    | 어떤 특징 그물인가 | 인셉션인지 다른 분류기인지. 값의 뜻이 달라진다 |
    | 표본 수 | 적으면 불안정하다 |
    | 조각 수 | 평균과 편차가 달라진다 |
    | 섞었는가 | 안 섞으면 값이 크게 틀린다 |
    | 씨앗 | 재현을 위해 |

    넷째가 거의 언제나 빠진다. 그런데 참 자료에서 2.146과 9.290을 가르는 요인이었다
    (연습문제 3).

    그리고 편차를 함께 적는 것이 좋다. 이 장의 참 자료가 $9.467 \pm 0.159$이므로, 두
    모델의 점수 차이가 0.1 정도라면 뜻 있는 차이라고 말하기 어렵다.

    IS만 적는 것은 이제 권하지 않는다. **FID를 함께** 적어야 한다. IS가 못 보는 것을
    FID가 보기 때문이며, 둘이 어긋날 때 그 어긋남 자체가 정보다.

---

<div class="drillbox" markdown>

**연습문제 12.** <span class="diff hard" title="어려움"></span>
인셉션 점수가 역사적으로 중요했던 까닭은 무엇인가?

</div>

??? success "연습문제 12 풀이"
    2016년에 나왔을 때 **사람이 눈으로 세던 일을 처음으로 자동화**했기 때문이다.

    그전에는 표본을 격자로 그려 놓고 눈으로 판단하거나 사람을 모아 평가했다. 값이 비싸고
    재현되지 않으며, 논문끼리 견줄 수 없었다.

    IS는 한 줄의 수를 주었고, 그 덕에 모델을 견주고 발전을 재는 일이 가능해졌다. 잣대가
    생기자 분야가 빠르게 움직였다.

    그런데 오래 쓰이자 약점이 드러났다. 정리하면 이렇다.

    | 문제 | 무엇이 | 어느 정도인가 |
    |---|---|---|
    | 부류 안의 다양성 | 못 본다 | 그림 열 장이 9.164를 받는다 |
    | 참 자료와 견주기 | 안 한다 | 자료가 식에 없다 |
    | 특징 그물 | ImageNet에 딸린다 | 다른 자료에서는 뜻이 약하다 |
    | 겨냥하기 | 쉽다 | 부류 비율만 맞춰도 오른다 |

    FID가 앞의 둘을 고치면서 표준이 되었다. 참 자료의 특징 분포와 견주므로 다양성이
    없으면 들키고, 자료가 식에 들어온다.

    그래도 IS를 아는 것이 값진 까닭이 둘이다. 옛 논문의 수를 읽어야 하고, 무엇보다
    **잣대가 어떻게 겨냥되는지**를 보여 주는 가장 깔끔한 예다. 셋째 칸과 넷째 칸의 교훈은
    FID에도 그대로 남아 있다.

---

<div class="drillbox" markdown>

**연습문제 13.** <span class="diff med" title="중간"></span>
표본 수가 적으면 인셉션 점수가 어떻게 되는가?

</div>

??? success "연습문제 13 풀이"
    불안정해지고, 조각으로 나누면 더욱 그렇다.

    까닭이 둘이다. $p(y)$를 표본에서 추정하는데 표본이 적으면 그 추정이 튄다. 그리고
    조각마다 나누면 조각 안의 표본 수가 더 줄어든다. 열 조각이면 1,000개 표본이 조각마다
    100개가 된다.

    FID처럼 방향이 정해진 쏠림이 있는지는 덜 분명하다. FID는 표본이 적으면 **반드시 커지는**
    쏠림이 있는데([FID](fid.md)), IS는 그런 단조로운 성질이 알려져 있지 않다. 대신
    편차가 커진다.

    실무의 관례는 **50,000개**를 쓰는 것이다. 원 논문이 그렇게 했고, 그보다 적게 쓰면
    값을 논문과 견줄 수 없다.

    이 장은 10,000개로 쟀다. 모델끼리 견주는 데는 충분하지만, 다른 글의 수와 나란히
    놓을 수는 없다. **표본 수를 밝혀야 하는 까닭**이다.

---

<div class="drillbox" markdown>

**연습문제 14.** <span class="diff med" title="중간"></span>
인셉션 점수 대신 쓸 수 있는 것들을 견주어라.

</div>

??? success "연습문제 14 풀이"
    | 잣대 | 참 자료를 쓰는가 | 부류 안 다양성 | 품질과 다양성을 가르는가 |
    |---|---|---|---|
    | 인셉션 점수 | 아니다 | 못 본다 | 아니다 |
    | FID | 쓴다 | 본다 | 아니다 |
    | 정밀도와 재현율 | 쓴다 | 본다 | **가른다** |
    | 사람이 평가 | — | 본다 | 가른다 |

    FID가 기본이 된 까닭은 앞의 두 칸이다. 다만 셋째 칸이 아쉽다. FID가 하나의 수이므로
    **품질이 나빠서 큰 것인지 다양성이 부족해서 큰 것인지 알려 주지 않는다.**

    그것을 가르려는 것이 정밀도와 재현율이다
    ([정밀도와 재현율](precision_recall.md)). 정밀도가 낮으면 품질 문제, 재현율이 낮으면
    다양성 문제로 읽는다. 진단에 훨씬 쓸모 있다.

    사람이 평가하는 것은 여전히 최종 심판이다. 다만 값이 비싸고 재현이 어려우며, 사람도
    다양성은 잘 못 본다. 격자 하나를 보고 "다양한가"를 판단하기는 어렵다.

    그래서 요즘 관례는 **FID를 기본으로 적고 정밀도와 재현율로 갈라 보이며, 표본 격자를
    함께 싣는 것**이다. 어느 하나로 갈음하지 않는다.

