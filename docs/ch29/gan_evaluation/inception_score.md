# 인셉션 점수(IS)

인셉션 점수(Inception Score)는 생성 이미지에 대한 고정 분류기의 예측 확률로 계산하는 지표다. [Salimans 등의 논문](https://arxiv.org/abs/1606.03498)에서 사용되었으며, 이미지별 예측의 확신과 전체 예측의 클래스 분산을 함께 요약한다. 실제 데이터 분포와 직접 비교하는 식은 아니므로 높은 점수가 곧 생성 분포의 정확성을 뜻하지는 않는다.

## 1. 정의와 엔트로피 표현

생성 이미지 X에 대한 분류기의 조건부 예측을 $q(y\mid x)$, 그 평균을 $\bar q(y)=\mathbb E_Xq(y\mid X)$라 두자.

$$\mathrm{IS}=\exp\left[\mathbb E_XD_{\mathrm{KL}}\big(q(y\mid X)\|\bar q(y)\big)\right].$$

분류 확률에 따라 클래스 변수 Y를 뽑는 결합분포 $p_G(x)q(y\mid x)$를 생각하면 다음 관계를 얻는다. 여기서 Y는 실제 정답이나 argmax 레이블이 아니라 분류 확률에 따른 확률변수다.

$$\log\mathrm{IS}=H(Y)-H(Y\mid X)=I(X;Y).$$

주변 엔트로피 H(Y)가 크면 예측이 여러 클래스에 분산되어 있고, 조건부 엔트로피 H(Y|X)가 작으면 각 이미지의 예측이 확신에 차 있다. 다만 분류기의 확신이 지각적 품질이나 정답성을 보장하지 않으며 클래스 비율이 균등한 것이 실제 데이터와 일치한다는 보장도 없다.

C개 클래스에서 0≤I(X;Y)≤log C이므로 1≤IS≤C다. 모든 이미지의 예측 확률이 같은 벡터이면 IS=1이다. 그 벡터는 균등할 필요가 없다. 상한 C는 예측이 원핫이고 주변 클래스 확률은 균등할 때 달성한다. 유한 N개 표본에서 전체 점수를 한 번 계산하면 상한은 min(N,C)로 더 제한된다.

## 2. Inception-v3를 사용하는 계산기

아래 계산기는 torchvision의 ImageNet 사전학습 Inception-v3를 불러와 1,000개 클래스 확률을 얻는다. 모델을 처음 불러올 때는 가중치 파일이 필요하다. 여기서는 코드 구조만 설명하며 실행이나 다운로드를 전제로 결과를 확인하지 않는다.

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
        그림 배치에 대한 인셉션 헤아림을 얻는다.
        
        인자:
            images: [0, 1] 범위의 만들어 낸 그림 [N, C, H, W]
            batch_size: 다룰 때 쓰는 배치 크기
            
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
            batch_size: 인셉션 미룸에 쓰는 배치 크기
            
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

입력은 N×C×H×W의 부동소수점 이미지이며 0~1 범위를 가정한다. 흑백 채널은 세 번 복제하고, 크기는 bilinear 보간으로 299×299로 맞춘 뒤 지정한 ImageNet 평균·표준편차로 정규화한다. tanh 출력을 받는다면 그 전에 범위 변환이 필요하다. 이는 이 코드의 전처리이며, 다른 IS 구현과 같은 가중치·변환·resize 관례를 쓰는지 확인하지 않고 값을 비교해서는 안 된다.

`get_predictions`는 eval 모드와 no_grad로 추론하고 CPU NumPy 확률 배열을 모은다. 빈 입력, 지원하지 않는 채널 수, 유효하지 않은 배치 크기는 처리하지 않는다. `calculate_inception_score`는 입력을 섞지 않고 연속 구간으로 나누어 구간별 점수의 평균과 표준편차를 반환한다. 1≤splits≤N이 필요하며 나머지는 마지막 구간에 포함한다.

`compute_inception_score_step_by_step`은 이미 계산한 확률 행렬에서 전체 점수를 한 번 구한다. `effective_classes=exp(H(Y))`는 주변 분포의 유효 클래스 수이고 `avg_confidence=exp(-H(Y|X))`는 엔트로피에 따른 요약값이다. 후자는 최대 예측 확률의 산술평균이 아니다. 함수는 clipping 후 재정규화하지 않으므로 유효한 입력과 작은 수치 보정이라는 조건에서 해석한다.

## 3. 확률 행렬을 이용한 예시

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

아래 출력은 직접 만든 열 클래스 확률 행렬을 계산한 예시다. 이 부분은 Inception-v3를 호출하거나 실제 이미지를 평가하지 않는다. 선택 클래스의 확률이 0.91인 첫 행렬은 완전한 원핫 예측이 아니므로 상한 10보다 작은 점수를 갖는다. 모든 행이 같은 두 번째와 세 번째 행렬은 서로 다른 확신 수준에도 IS=1이다.

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

## 4. 클래스 내부 다양성과 암기

클래스마다 대표 이미지 하나씩만 반복해도 각 이미지의 예측이 확신에 차 있고 클래스 비율이 균등하면 높은 IS를 얻을 수 있다. 반대로 동일 이미지 하나를 반복해 같은 예측만 얻으면 IS=1이다. 클래스 내부의 시각적 변이가 바뀌어도 예측 확률 분포가 같으면 IS는 같을 수 있다.

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

위 함수는 반복된 **확률 행렬**을 구성한다. 실제로 서로 다른 이미지가 열 장뿐임을 검사하거나 분류기에 이미지를 넣는 코드는 아니다. 분포 내부의 다양성을 보지 못하는 계산 원리를 보여 주는 예로 읽는다.

## 5. 분류기를 겨냥한 점수 상승

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

원핫 확률을 클래스별로 하나씩 만든 위 행렬의 전체 점수는 수치 오차를 제외하면 1,000이다. 그러나 이는 이미지 생성 실험이 아니다. 실제 모델이 평가 분류기의 확신을 유도하거나 애매한 표본을 선별해 버리면 지각적 품질이나 분포 포괄성이 개선되지 않아도 점수가 올라갈 수 있다. [IS의 한계 분석](https://arxiv.org/abs/1801.01973)과 함께 평가 목적을 검토한다.

## 6. 표본 수와 분할의 영향

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

위 함수는 외부 객체의 `generate(n)` 인터페이스를 가정한다. 이 페이지에서 정의한 학습된 생성기는 없다. 표본 수를 바꾸면 주변 확률 추정의 변동과 편향, 분할당 표본 수가 모두 달라질 수 있다. 특정 최소 개수만 넘으면 정확하다고 보장할 수 없으므로 동일 조건과 필요한 정밀도에 맞춰 정한다.

```python
# 흔한 방식: 조각 10개
is_mean, is_std = calculator.calculate_inception_score(images, splits=10)

# 이렇게 알린다: 인셉션 점수 = 평균 ± 표준편차
print(f"IS = {is_mean:.2f} ± {is_std:.2f}")
```

`±` 뒤 값은 분할별 점수의 표준편차이며 평균의 표준오차나 95% 신뢰구간이 아니다. 분할 수, 전체 표본 수와 구간 크기를 기록하고 독립 생성 반복의 변동과 구분한다. 클래스순 배열에서는 분할에 따라 클래스 구성이 크게 달라지므로 사전에 정한 섞기·시드 절차가 필요하다.

## 7. 보고와 보완 평가

분류기와 체크포인트, 데이터 전처리, 표본 수, 분할 수, 순서와 시드, 생성 조건 및 선별 규칙을 보고한다. 실제 데이터의 IS는 경험적 비교 기준이며 이론적 천장이 아니다. 데이터셋 이름만으로 특정 실제 점수나 '좋음' 구간을 고정할 수 없다. MNIST 분류기로 같은 식을 계산하면 ImageNet Inception-v3 기준 점수와 다른 지표임을 명시한다.

FID는 실제·생성 특징의 첫 두 모멘트를 비교하고, 생성 정밀도·재현율은 선택한 특징 공간의 근접성을 이용해 품질과 포괄성을 나누어 살펴본다. 어느 것도 모든 중복이나 암기를 검출하지 못한다. 표본, 최근접 이웃, 목적에 맞는 작업 성능과 함께 평가한다.

## 참고 문헌

- [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)
- [A Note on the Inception Score](https://arxiv.org/abs/1801.01973)

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
FID는 IS와 무엇을 다르게 비교하며 어떤 한계가 있는가?

</div>

??? success "연습문제 1 풀이"

    FID는 실제와 생성 특징의 평균·공분산으로 맞춘 가우시안 사이의 제곱 Wasserstein 거리를 계산한다. IS에는 없는 실제 기준 집합을 사용한다. 다만 첫 두 모멘트가 같아도 전체 분포는 다를 수 있으므로 모든 모드 붕괴를 검출한다고 보장하지 않는다. 특징 추출기, 전처리, 표본 수와 수치 계산 방식에도 의존한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
높은 IS를 얻으면서 클래스 내부 다양성이 부족한 예를 설명하라.

</div>

??? success "연습문제 2 풀이"

    클래스마다 분류기가 확신하는 대표 이미지 하나씩을 같은 비율로 반복할 수 있다. 클래스별 원핫 예측이면 상한 C에 이를 수 있어도 고유 이미지는 C개뿐이다. 동일 이미지 하나만 반복하는 경우는 평균 예측과 개별 예측이 같아 IS=1이므로 이 사례와 구분한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
생성 모델의 정밀도·재현율은 분류의 정밀도·재현율과 어떻게 다른가?

</div>

??? success "연습문제 3 풀이"

    분류에서는 정답 레이블을 기준으로 참·거짓 양성을 센다. 생성 평가의 한 방식은 실제·생성 특징에서 k번째 최근접 이웃 반지름의 구들을 만들고 서로 다른 집합의 포함 비율을 계산한다. 이는 특징 공간의 유한 표본 근사이며 참 분포의 수학적 지지집합을 정확히 아는 것이 아니다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
IS, FID와 LPIPS를 함께 해석할 때 각각의 평가 대상을 구분하라.

</div>

??? success "연습문제 4 풀이"

    IS는 분류 예측, FID는 두 특징 분포의 첫 두 모멘트, LPIPS는 두 이미지의 지각적 거리를 요약한다. LPIPS는 대응 관계 없이 이미지 하나의 절대적 품질을 매기는 지표가 아니다. 여러 지표를 쓴다고 전체 품질을 자동으로 검증하는 것은 아니므로 과제와 표본 선택에 맞는 평가를 설계한다.



---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
IS의 범위와 최솟값·최댓값의 달성 조건을 설명하라.

</div>

??? success "연습문제 5 풀이"

    0≤I(X;Y)≤H(Y)≤log C이므로 1≤IS≤C다. 모든 입력의 예측 분포가 같으면 IS=1이다. 각 입력의 예측이 원핫이고 주변 분포가 균등하면 IS=C다. 균등한 개별 예측도 최소를 만들지만 최소의 유일한 조건은 아니다. 유한 N개 전체 평가의 최대는 min(N,C)를 넘지 않는다.



---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
열 클래스에 대해 원핫 예측을 내는 이미지 하나씩을 균등 반복하면 IS는 얼마인가? FID는 반드시 나빠지는가?

</div>

??? success "연습문제 6 풀이"

    전체 점수를 한 번 구하면 조건부 엔트로피는 0, 주변 엔트로피는 log 10이므로 IS=10이다. 분할이 불균형하면 분할별 평균은 달라질 수 있다. FID는 반복으로 바뀐 특징 평균·공분산에 반응할 수 있으나 반드시 차이를 검출하는 것은 아니다. 이 페이지에는 실제 이미지로 계산한 FID 비교 실험이 없다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff hard" title="어려움"></span>
같은 확률 행렬을 다른 순서로 분할하면 점수가 달라지는 이유는 무엇인가?

</div>

??? success "연습문제 7 풀이"

    구간마다 주변 확률을 새로 추정하기 때문이다. 클래스순 입력은 구간 안에서 적은 클래스만 포함할 수 있다. 아래 섞기 조각은 probs를 제자리에서 변경하므로 원본 보존이 필요하면 복사본을 사용한다. seed는 외부 정의가 필요하다. 전체 행렬에서 splits=1로 계산한 점수는 행 순열에 불변이다.

    ```python
    rng = np.random.default_rng(seed)
    rng.shuffle(probs)                      # 이 한 줄이 빠지면 값이 틀린다
    ```


---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
MNIST를 ImageNet Inception-v3로 평가하거나 MNIST 전용 분류기로 평가할 때 무엇을 밝혀야 하는가?

</div>

??? success "연습문제 8 풀이"

    ImageNet 분류기는 숫자 구분을 위한 평가기로 학습된 것이 아니므로 예측 확신의 의미를 검토해야 한다. 채널 복제·resize 등 전처리도 기록한다. MNIST 분류기로 바꾸면 구조와 체크포인트, 별도 검증 성능을 명시하며 표준 Inception-v3 IS와 직접 비교하지 않는다. 전용 분류기를 썼다는 사실만으로 평가가 완전해지는 것은 아니다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
클래스 비율을 균등하게 지정하면 모델이 개선되지 않아도 IS가 오를 수 있는가?

</div>

??? success "연습문제 9 풀이"

    그럴 수 있다. 예측이 클래스 조건을 충실히 따르고 개별 확신이 유지된다면 주변 엔트로피가 커질 수 있다. 하지만 클래스 조건을 균등하게 입력해도 예측 분포가 반드시 균등해지는 것은 아니다. 샘플링 조건을 바꾼 결과인지 모델 자체가 개선된 것인지 구분하고, 실제 데이터의 클래스 비율과 평가 목적을 함께 기록한다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
IS의 로그가 상호정보량이라는 표현에서 Y는 무엇인가?

</div>

??? success "연습문제 10 풀이"

    생성 이미지 X와 평가 분류기의 조건부 확률 q(y|X)로 정의한 결합분포에서의 클래스 변수다. Y를 argmax 레이블로 바꾸면 조건부 엔트로피와 점수가 달라진다. $I(X;Y)=H(Y)-H(Y|X)$이며 이것이 표준 IS의 로그다. 실제 이미지의 모든 의미 정보를 측정하는 상호정보량은 아니다.



---

<div class="drillbox" markdown>

**연습문제 11.** <span class="diff easy" title="쉬움"></span>
IS의 평균과 표준편차를 보고할 때 필요한 설정을 열거하라.

</div>

??? success "연습문제 11 풀이"

    분류기·체크포인트·전처리, 클래스 수, 전체 표본 수와 분할 수, 섞기·시드, 생성 조건과 선별 절차를 기록한다. 표준편차가 분할 사이 변동인지 독립 생성 반복 사이 변동인지 명시한다. 분할 표준편차 두 개를 보고 모델 차이의 통계적 유의성을 바로 판단하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 12.** <span class="diff hard" title="어려움"></span>
IS가 도입된 의미와 이후 드러난 한계를 설명하라.

</div>

??? success "연습문제 12 풀이"

    분류기 예측을 이용한 정량적 비교 기준을 제시하여 샘플 격자 외의 평가를 제공했다. 생성 평가를 처음으로 자동화한 유일한 방법이었다고 단정할 수는 없다. 실제 기준 분포를 직접 쓰지 않고 클래스 내부 다양성과 암기를 놓치는 등의 한계 때문에 다른 평가를 함께 검토해야 한다. FID도 모든 문제를 해결한 대체물은 아니다.



---

<div class="drillbox" markdown>

**연습문제 13.** <span class="diff med" title="중간"></span>
표본 수를 늘리면 각 실험의 IS나 FID가 반드시 한 방향으로 변하는가?

</div>

??? success "연습문제 13 풀이"

    그렇지 않다. 유한 표본 편향과 표본 변동을 구분해야 하며 개별 실험값이 표본 수에 따라 단조롭게 변한다는 보장은 없다. 두 지표 모두 추정 절차와 모델에 따른 편향을 가질 수 있다. 표본 수와 분할 수를 맞추고 변동을 평가하며 특정 개수면 모든 비교에 충분하다고 주장하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 14.** <span class="diff med" title="중간"></span>
IS를 보완할 평가 방법과 각 방법의 남는 한계를 설명하라.

</div>

??? success "연습문제 14 풀이"

    FID는 실제 기준 특징의 모멘트를 비교하지만 품질과 포괄성을 한 수치에 섞는다. 정밀도·재현율 계열은 두 측면을 나누지만 특징과 이웃 설정에 의존한다. 표본 격자와 사람 평가는 세부 결함을 볼 수 있으나 표본 수·선택·평가자에 영향을 받는다. 중복·최근접 이웃 검사와 다운스트림 평가도 목적에 맞게 추가한다. 어느 방법도 단독으로 모든 생성 품질을 보증하지 않는다.



## 정리하며

IS는 고정 분류기의 예측에 대한 상호정보량을 지수화한 값이다. 평가 설정을 맞추고 클래스 내부 다양성, 실제 분포와의 일치, 암기는 별도로 살펴본다.
