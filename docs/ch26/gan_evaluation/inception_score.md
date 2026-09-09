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

## 익힘 문제

**익힘 1.**
프레셰 인셉션 거리(FID)를 뜻매김하고, GAN을 따질 때 인셉션 점수보다 이를 더 치는 까닭을 밝혀라.

??? success "익힘 1 풀이"
    FID는 참 그림과 만들어 낸 그림의 Inception-v3 특징 분포를 다변량 가우스 $\mathcal{N}(\mu_r, \Sigma_r)$와 $\mathcal{N}(\mu_g, \Sigma_g)$로 보고 다음을 셈한다.

    $$\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$

    FID를 더 치는 까닭은 이렇다. (1) 만들어 낸 그림을 참 그림과 견준다(인셉션 점수는 만들어 낸 그림만 따진다). (2) 최빈값 무너짐을 알아낸다(평균과 공분산이 달라진다). (3) 사람의 판단과 더 잘 들어맞는다. (4) FID가 낮을수록 느낌의 좋음과 더 잘 맞아떨어진다.

---

**익힘 2.**
인셉션 점수의 한계는 무엇인가? 나쁜 표본을 내놓으면서도 높은 인셉션 점수를 얻을 수 있는가?

??? success "익힘 2 풀이"
    인셉션 점수는 $\exp(\mathbb{E}_x [D_{\text{KL}}(p(y|x) \| p(y))])$이며 $p(y|x)$는 만들어 낸 그림 $x$에 대한 인셉션 가름개의 헤아림이다. 한계는 이렇다. (1) 품질(뾰족하고 가를 수 있음)과 다양함(갈래에 두루 퍼짐)만 잴 뿐 익힘 자료에 얼마나 충실한지는 재지 않는다. (2) 갈래마다 나무랄 데 없는 그림을 하나씩만 내놓아도 점수가 높지만 갈래 안의 다양함은 놓친다. (3) 인셉션 모델의 치우침에 흔들린다. (4) 갈래 안의 결과 결무늬 품질을 담아내지 못한다. 그렇다. ImageNet 갈래마다 대표 그림을 하나씩 외우기만 해도 높은 인셉션 점수를 얻을 수 있다.

---

**익힘 3.**
만들어 내는 모델의 정밀도·재현율 자가 가름에서 쓰는 것과 어떻게 다른지 밝혀라.

??? success "익힘 3 풀이"
    만들어 내기 자리에서는(Kynkaanniemi 외, 2019) 이렇다. **정밀도**는 만들어 낸 표본 가운데 참 자료 분포의 받침 안에 드는 몫을 잰다(품질, 곧 충실함). **재현율**은 참 자료 가운데 만들어 낸 분포의 받침 안에 드는 몫을 잰다(다양함, 곧 덮음). 정밀도가 높고 재현율이 낮으면 최빈값 무너짐이다(봉우리는 적지만 그럴듯하다). 정밀도가 낮고 재현율이 높으면 품질은 나쁘나 다양하다. 띄엄띄엄한 맞음을 세는 가름의 정밀도·재현율과 달리, 만들어 내기의 정밀도·재현율은 특징 자리에서 $k$번째 가장 가까운 이웃까지의 거리로 분포의 받침을 어림한다.

---

**익힘 4.**
만들어 내는 모델을 살필 때 왜 여러 따지기 자를 함께 써야 하는가?

??? success "익힘 4 풀이"
    어느 자 하나도 만들어 내기 품질의 모든 면을 담아내지 못한다. **FID**는 분포가 두루 얼마나 닮았는지 재지만 품질과 다양함을 뒤섞는다. **인셉션 점수**는 품질과 다양함을 담아내지만 익힘 자료에 얼마나 충실한지는 놓친다. **정밀도·재현율**은 품질과 다양함을 갈라 보여 주지만 어떤 특징 뽑개와 $k$를 고르는지에 달렸다. **느낌의 자**(LPIPS)는 그림 하나하나의 품질은 재지만 다양함은 재지 않는다. 여러 자를 함께 쓰면 온 그림이 보인다. FID가 낮고 정밀도가 높으며 재현율이 낮은 모델은 최빈값이 무너진 것이고, 재현율은 높으나 정밀도가 낮은 모델은 다양하지만 품질이 낮은 표본을 내놓는다. 마지막 판단에서는 사람이 따지는 것이 여전히 으뜸 잣대다.
