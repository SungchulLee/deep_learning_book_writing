# 프레셰 인셉션 거리(FID)

프레셰 인셉션 거리(FID)는 만들어 내는 모델을, 특히 그림 만들기를 따지는 데 가장 널리 쓰이는 잣대이다. Heusel 외(2017)가 내놓은 FID는 미리 익힌 인셉션 신경망의 특징 공간에서 만든 그림의 분포와 실제 그림의 분포 사이 거리를 잰다.

!!! info "배움 목표"
    이 절을 마치면 다음을 할 수 있게 된다.
    
    - FID의 수학 바탕을 이끌어 내고 이해한다
    - 수치를 제대로 다루며 FID 셈하기를 바닥부터 짠다
    - FID가 화소 공간 대신 인셉션 특징을 쓰는 까닭을 이해한다
    - FID가 인셉션 점수보다 나은 점과 남은 한계를 안다
    - 연구와 실제 자리에서 FID를 제대로 쓴다

---

## 1. 수학적 바탕

### 프레셰 거리

프레셰 거리(정규 분포에서는 바서슈타인-2 거리라고도 한다)는 두 확률 분포 사이의 거리를 잰다. 여러 변수 정규 분포에서는 닫힌 꼴 풀이가 있다.

정규 분포 둘이 주어질 때:

- 실제 자료 특징: $\mathcal{N}(\mu_r, \Sigma_r)$
- 만든 자료 특징: $\mathcal{N}(\mu_g, \Sigma_g)$

프레셰 거리는 다음과 같다.

$$
\text{FID} = \|\mu_r - \mu_g\|_2^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)
$$

### 몫 나누어 보기

**평균 차이 항: $\|\mu_r - \mu_g\|_2^2$**

이는 특징 공간에서 "평균" 실제 그림과 만든 그림이 얼마나 다른지 잰다. 평균 차이가 크면 만들어 내기에 짜임새 있는 치우침이 있다는 뜻이다.

**함께 흩어짐 항:**

- $\text{Tr}(\Sigma_r)$: 실제 자료 특징의 온 흩어짐
- $\text{Tr}(\Sigma_g)$: 만든 자료 특징의 온 흩어짐
- $-2\text{Tr}((\Sigma_r \Sigma_g)^{1/2})$: 함께 흩어짐 겹침 벌점

함께 흩어짐 항은 생성기가 실제 자료와 같은 범위와 서로 이어짐 짜임을 내는지 담는다.

### 왜 정규 분포로 보는가?

특징 분포를 정규 분포로 보는 것은 다음 까닭으로 받아들일 만하다.

1. **중심 극한 정리**: 깊은 신경망의 깨움은 평균 내는 효과 때문에 정규 분포에 가까워진다
2. **셈으로 다룰 수 있음**: 닫힌 꼴 풀이가 있다
3. **겪어 본 확인**: 그림 특징에서 실제로 잘 듣는다
4. **수학의 바탕**: 정규 분포의 가장 좋은 나르기 거리이다

### 가장 좋은 나르기와의 이음

FID는 정규 분포의 2-바서슈타인 거리의 제곱이다.

$$
W_2^2(\mathcal{N}(\mu_1, \Sigma_1), \mathcal{N}(\mu_2, \Sigma_2)) = \text{FID}
$$

곧 FID는 한 분포를 다른 분포로 바꾸는 가장 작은 "비용"을 재며, 비용은 유클리드 거리의 제곱이다.

---

## 2. 수학으로 이끌어 내기

### 바서슈타인 거리에서 시작하기

분포 $P$과 $Q$ 사이의 2-바서슈타인 거리는 다음과 같다.

$$
W_2(P, Q) = \left(\inf_{\gamma \in \Gamma(P,Q)} \mathbb{E}_{(x,y)\sim\gamma}[\|x - y\|_2^2]\right)^{1/2}
$$

여기서 $\Gamma(P,Q)$은 가장자리 분포가 $P$과 $Q$인 모든 결합 분포의 모임이다.

### 정규 분포의 닫힌 꼴

정규 분포에서는 가장 좋은 나르기 짜임이 알려져 있어 다음을 준다.

$$
W_2^2(\mathcal{N}(\mu_1, \Sigma_1), \mathcal{N}(\mu_2, \Sigma_2)) = \|\mu_1 - \mu_2\|_2^2 + \text{Bures}(\Sigma_1, \Sigma_2)
$$

여기서 양의 정부호 행렬의 뷰레 잣대는 다음과 같다.

$$
\text{Bures}(\Sigma_1, \Sigma_2) = \text{Tr}(\Sigma_1) + \text{Tr}(\Sigma_2) - 2\text{Tr}\left((\Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2})^{1/2}\right)
$$

### 단순하게 하기

대각합의 돌림 성질과 양의 정부호성을 쓰면:

$$
\text{Tr}\left((\Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2})^{1/2}\right) = \text{Tr}\left((\Sigma_1\Sigma_2)^{1/2}\right)
$$

이것이 여느 FID 공식을 준다.

---

## 3. 왜 인셉션 특징인가?

### 화소 공간의 문제

화소 공간에서 그림을 곧바로 견주는 데는 문제가 있다.

1. **느낌과 무관함**: 화소가 조금만 옮겨도 거리는 크게 벌어지지만 눈으로는 알아채지 못한다
2. **잣수에 민감함**: 화소 거리가 해상도에 크게 매인다
3. **뜻을 모름**: 고양이와 개의 화소 통계가 비슷할 수 있다

### 특징 뽑개로서의 인셉션 신경망

(ImageNet으로 익힌) InceptionV3은 다음을 준다.

| 특징 | 좋은 점 |
|---------|---------|
| 뜻 이해 | 물체의 정체와 장면 내용을 담는다 |
| 느낌과 맞음 | 특징이 사람의 느낌과 이어진다 |
| 층층 나타냄 | 낮은 수준과 높은 수준 특징을 모두 담는다 |
| 표준화 | 같은 신경망이라 공정한 견줌이 된다 |

### Pool3 층(2048차원)

FID는 마지막 가르기 앞의 온마당 평균 모으기 층의 내놓기를 쓴다.

```
InceptionV3 얼개:
┌─────────────┐
│   들임      │ 299×299×3
├─────────────┤
│  겹말기/모으기 │ 여러 층
├─────────────┤
│  Mixed_7c   │ 8×8×2048
├─────────────┤
│  평균 모으기 │ 2048 ← FID가 이것을 쓴다!
├─────────────┤
│    FC       │ 1000 (classes)
└─────────────┘
```

---

## 4. PyTorch 구현

### 온전한 FID 셈개

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg
from typing import Tuple, Optional, Union
from torchvision.models import inception_v3, Inception_V3_Weights

class FIDCalculator:
    """
    두루 갖춘 프레셰 인셉션 거리 셈개.
    
    FID는 미리 익힌 인셉션 신경망의 특징 공간에서
    실제 그림 분포와 만든 그림 분포 사이의 거리를 잰다.
    
    FID가 낮을수록 품질이 좋고 분포가 더 비슷하다.
    
    속성:
        device: 셈할 장치
        inception: 미리 익힌 InceptionV3 모델
        feature_dim: 뽑아낸 특징의 차원(2048)
    """
    
    def __init__(self, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        FID 셈개를 첫자리매김한다.
        
        인수:
            device: 셈할 장치('cuda' 또는 'cpu')
        """
        self.device = device
        self.inception = None
        self.feature_dim = 2048
        
    def _load_inception(self):
        """
        특징 뽑기를 위해 InceptionV3을 불러와 고친다.
        
        pool3 층에서 특징을 뽑는다(2048차원)
        이는 높은 수준의 뜻 앎을 담는다.
        """
        # 미리 익힌 InceptionV3을 불러온다
        self.inception = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1,
            transform_input=False
        )
        
        # 마지막 분류 층을 없앤다
        # 모으기 층의 특징을 얻으려 한다
        self.inception.fc = nn.Identity()
        
        # 값매김 방식으로 둔다
        self.inception.eval()
        self.inception.to(self.device)
        
        # 효율을 위해 기울기를 끈다
        for param in self.inception.parameters():
            param.requires_grad = False
    
    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """
        InceptionV3용으로 그림을 미리 다듬는다.
        
        필요한 것:
        - 크기: 299×299
        - 범위: ImageNet 통계로 고르게 맞춤
        - Channels: 3 (RGB)
        
        인수:
            images: [0, 1] 범위의 들임 그림 [B, C, H, W]
            
        반환값:
            미리 다듬은 그림
        """
        # 필요하면 크기를 바꾼다
        if images.shape[2] != 299 or images.shape[3] != 299:
            images = F.interpolate(
                images,
                size=(299, 299),
                mode='bilinear',
                align_corners=False
            )
        
        # 회색을 다룬다
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        
        # ImageNet 통계로 고르게 맞춘다
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        mean = mean.to(images.device)
        std = std.to(images.device)
        
        return (images - mean) / std
    
    def extract_features(self,
                        images: torch.Tensor,
                        batch_size: int = 64) -> np.ndarray:
        """
        그림에서 InceptionV3 특징을 뽑는다.
        
        인수:
            images: [0, 1] 범위의 그림 [N, C, H, W]
            batch_size: 다룰 배치 크기
            
        반환값:
            Features [N, 2048]
        """
        if self.inception is None:
            self._load_inception()
        
        all_features = []
        n_images = len(images)
        
        with torch.no_grad():
            for i in range(0, n_images, batch_size):
                batch = images[i:i+batch_size].to(self.device)
                batch = self._preprocess(batch)
                
                # 특징을 뽑는다
                features = self.inception(batch)
                all_features.append(features.cpu().numpy())
        
        return np.concatenate(all_features, axis=0)
    
    @staticmethod
    def compute_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        특징의 평균과 함께 흩어짐을 셈한다.
        
        인수:
            features: 특징 벡터 [N, D]
            
        반환값:
            (평균 [D], 공분산 [D, D]) 튜플
            
        수학의 참고:
            μ = (1/N) Σ x_i
            Σ = (1/(N-1)) Σ (x_i - μ)(x_i - μ)ᵀ
        """
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        
        return mu, sigma
    
    @staticmethod
    def calculate_frechet_distance(mu1: np.ndarray,
                                   sigma1: np.ndarray,
                                   mu2: np.ndarray,
                                   sigma2: np.ndarray,
                                   eps: float = 1e-6) -> float:
        """
        정규 분포 둘 사이의 프레셰 거리를 셈한다.
        
        FID = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2(Σ₁Σ₂)^{1/2})
        
        인수:
            mu1: 첫째 분포의 평균 [D]
            sigma1: 첫째 분포의 공분산 [D, D]
            mu2: 둘째 분포의 평균 [D]
            sigma2: 둘째 분포의 공분산 [D, D]
            eps: 수치 안정성을 위한 작은 상수
            
        반환값:
            FID 값(스칼라이며 낮을수록 좋다)
        """
        # 넘파이 배열이 되게 한다
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        assert mu1.shape == mu2.shape, f"Mean shapes differ: {mu1.shape} vs {mu2.shape}"
        assert sigma1.shape == sigma2.shape, f"Cov shapes differ: {sigma1.shape} vs {sigma2.shape}"
        
        # 1. 평균 차이 항: ||μ₁ - μ₂||²
        diff = mu1 - mu2
        mean_term = np.dot(diff, diff)
        
        # 2. 행렬 제곱근: (Σ₁Σ₂)^{1/2}
        # 이것이 셈이 비싼 걸음이다
        
        # 함께 흩어짐의 곱
        product = sigma1 @ sigma2
        
        # scipy으로 행렬 제곱근
        covmean, _ = linalg.sqrtm(product, disp=False)
        
        # 수치 문제를 다룬다
        if not np.isfinite(covmean).all():
            print(f"Warning: Non-finite values in matrix sqrt. Adding {eps} to diagonal.")
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
        
        # 허수 몫을 다룬다(수치의 흠)
        if np.iscomplexobj(covmean):
            if np.allclose(covmean.imag, 0, atol=1e-3):
                covmean = covmean.real
            else:
                raise ValueError(f"Significant imaginary component: {np.max(np.abs(covmean.imag))}")
        
        # 3. 대각합 항
        trace_term = np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        
        # 4. 마지막 FID
        fid = mean_term + trace_term
        
        return float(fid)
    
    def calculate_fid(self,
                     real_images: torch.Tensor,
                     generated_images: torch.Tensor,
                     batch_size: int = 64) -> float:
        """
        실제 그림과 만든 그림 사이의 FID를 셈한다.
        
        온전한 물길:
        1. 실제 그림에서 인셉션 특징을 뽑는다
        2. 만든 그림에서 인셉션 특징을 뽑는다
        3. 둘 다의 통계(μ, Σ)를 셈한다
        4. 프레셰 거리를 셈한다
        
        인수:
            real_images: 참 그림 [N_r, C, H, W]
            generated_images: 만들어 낸 그림 [N_g, C, H, W]
            batch_size: 특징 뽑기의 배치 크기
            
        반환값:
            FID 점수(낮을수록 좋다)
        """
        print(f"Extracting features from {len(real_images)} real images...")
        real_features = self.extract_features(real_images, batch_size)
        
        print(f"Extracting features from {len(generated_images)} generated images...")
        gen_features = self.extract_features(generated_images, batch_size)
        
        print("Computing statistics...")
        mu_real, sigma_real = self.compute_statistics(real_features)
        mu_gen, sigma_gen = self.compute_statistics(gen_features)
        
        print("Calculating Fréchet distance...")
        fid = self.calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
        
        print(f"FID = {fid:.4f}")
        return fid
    
    def calculate_fid_from_statistics(self,
                                      mu_real: np.ndarray,
                                      sigma_real: np.ndarray,
                                      generated_images: torch.Tensor,
                                      batch_size: int = 64) -> float:
        """
        미리 셈한 실제 자료 통계로 FID를 셈한다.
        
        같은 실제 자료 묶음에 여러 생성기를 견줄 때
        이 편이 더 효율이 좋다.
        
        인수:
            mu_real: 미리 셈한 참 특징의 평균 [D]
            sigma_real: 미리 셈한 참 특징의 공분산 [D, D]
            generated_images: 만들어 낸 그림 [N, C, H, W]
            batch_size: 특징 뽑기의 배치 크기
            
        반환값:
            FID 점수
        """
        gen_features = self.extract_features(generated_images, batch_size)
        mu_gen, sigma_gen = self.compute_statistics(gen_features)
        
        return self.calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)

def save_reference_statistics(real_images: torch.Tensor,
                             save_path: str,
                             batch_size: int = 64):
    """
    견줄 자료 묶음의 통계를 미리 셈해 갈무리한다.
    
    그러면 익히는 동안 실제 자료 통계를 다시 셈하지 않고
    효율 좋게 FID를 셈할 수 있다.
    
    인수:
        real_images: 참 그림 [N, C, H, W]
        save_path: 통계를 갈무리할 길(.npz 파일)
        batch_size: 특징 뽑기의 배치 크기
    """
    calculator = FIDCalculator()
    features = calculator.extract_features(real_images, batch_size)
    mu, sigma = FIDCalculator.compute_statistics(features)
    
    np.savez(save_path, mu=mu, sigma=sigma)
    print(f"Saved statistics to {save_path}")
    print(f"  Shape: μ={mu.shape}, Σ={sigma.shape}")

def load_reference_statistics(load_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    미리 셈한 통계를 불러온다.
    
    인수:
        load_path: .npz 파일의 길
        
    반환값:
        (평균, 공분산) 튜플
    """
    data = np.load(load_path)
    return data['mu'], data['sigma']
```

### 인공 자료로 보여 주기

```python
def demonstrate_fid_computation():
    """
    다스린 상황으로 FID 셈하기를 보인다.
    """
    print("=" * 70)
    print("Fréchet Inception Distance Demonstration")
    print("=" * 70)
    
    # 보여 주기에는 인공 특징을 쓴다
    # (실제로는 인셉션에서 온다)
    n_samples = 5000
    feature_dim = 2048
    
    # 상황 1: 똑같은 분포
    print("\n📊 Scenario 1: Identical Distributions")
    print("-" * 50)
    
    real_features = np.random.randn(n_samples, feature_dim)
    gen_features = np.random.randn(n_samples, feature_dim)
    
    mu_r, sigma_r = FIDCalculator.compute_statistics(real_features)
    mu_g, sigma_g = FIDCalculator.compute_statistics(gen_features)
    
    fid = FIDCalculator.calculate_frechet_distance(mu_r, sigma_r, mu_g, sigma_g)
    print(f"FID: {fid:.4f}")
    print("Note: Small non-zero FID due to finite sample estimation")
    
    # 상황 2: 평균이 옮겨짐
    print("\n📊 Scenario 2: Shifted Mean")
    print("-" * 50)
    
    gen_features_shifted = np.random.randn(n_samples, feature_dim) + 0.5
    
    mu_g2, sigma_g2 = FIDCalculator.compute_statistics(gen_features_shifted)
    fid_shifted = FIDCalculator.calculate_frechet_distance(mu_r, sigma_r, mu_g2, sigma_g2)
    
    print(f"FID: {fid_shifted:.4f}")
    print("Note: Mean shift increases FID significantly")
    
    # 상황 3: 흩어짐이 줄었다(봉우리 무너짐 표시)
    print("\n📊 Scenario 3: Reduced Variance (Mode Collapse)")
    print("-" * 50)
    
    gen_features_collapsed = np.random.randn(n_samples, feature_dim) * 0.5
    
    mu_g3, sigma_g3 = FIDCalculator.compute_statistics(gen_features_collapsed)
    fid_collapsed = FIDCalculator.calculate_frechet_distance(mu_r, sigma_r, mu_g3, sigma_g3)
    
    print(f"FID: {fid_collapsed:.4f}")
    print("Note: Reduced variance indicates mode collapse")
    
    # 요약
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Scenario':<30} {'FID':>10}")
    print("-" * 40)
    print(f"{'Identical (baseline)':<30} {fid:>10.2f}")
    print(f"{'Shifted mean':<30} {fid_shifted:>10.2f}")
    print(f"{'Mode collapse':<30} {fid_collapsed:>10.2f}")
    print("\nLower FID = Better (more similar to real distribution)")

demonstrate_fid_computation()
```

**출력:**

```
======================================================================
Fréchet Inception Distance Demonstration
======================================================================

📊 Scenario 1: Identical Distributions
--------------------------------------------------
FID: 420.0447
Note: Small non-zero FID due to finite sample estimation

📊 Scenario 2: Shifted Mean
--------------------------------------------------
FID: 933.4830
Note: Mean shift increases FID significantly

📊 Scenario 3: Reduced Variance (Mode Collapse)
--------------------------------------------------
FID: 722.7101
Note: Reduced variance indicates mode collapse

======================================================================
Summary
======================================================================
Scenario                              FID
----------------------------------------
Identical (baseline)               420.04
Shifted mean                       933.48
Mode collapse                      722.71

Lower FID = Better (more similar to real distribution)
```

---

## 5. FID 값 풀이하기

### 자연 그림의 흔한 범위

| FID 값 | 품질 수준 | 풀이 |
|-----------|---------------|----------------|
| < 5 | 뛰어남 | 실제와 거의 가려낼 수 없다 |
| 5 - 20 | 아주 좋음 | 품질 높은 만들어 내기 |
| 20 - 50 | 좋음 | 작은 흠이나 봉우리 빠짐 |
| 50 - 100 | 보통 | 눈에 띄는 품질 문제 |
| > 100 | 나쁨 | 분포가 크게 어긋남 |

### 자료 묶음별 기준값

자료 묶음마다 FID 범위가 다르다.

| 자료 묶음 | 최고 수준 FID |
|---------|---------------------|
| CIFAR-10 | ~2 |
| CelebA-HQ 256 | ~5 |
| FFHQ 256 | ~3 |
| ImageNet 256 | ~2-5 |
| LSUN Bedroom | ~2-5 |

### FID에 영향을 주는 것

1. **표본 크기**: 표본이 많을수록 → FID가 더 안정된다
2. **그림 해상도**: 해상도가 다르면 미리 다듬기도 달라야 할 수 있다
3. **빛깔 공간**: RGB이냐 회색이냐가 특징에 영향을 준다
4. **자르기**: 맞겨루기 생성기의 자르기는 다양함을 품질과 맞바꾼다

---

## 6. 표본 크기 살피기

### 가장 낮은 조건

FID는 함께 흩어짐을 믿을 만하게 어림하려면 넉넉한 표본이 필요하다.

```python
def analyze_fid_sample_size():
    """
    표본 크기가 FID의 안정에 어떻게 영향을 주는지 살핀다.
    """
    feature_dim = 2048
    true_mu = np.zeros(feature_dim)
    true_sigma = np.eye(feature_dim)
    
    sample_sizes = [100, 500, 1000, 2048, 5000, 10000, 50000]
    n_trials = 10
    
    results = []
    
    for n in sample_sizes:
        fids = []
        for _ in range(n_trials):
            # 같은 분포에서 뽑는다
            features1 = np.random.randn(n, feature_dim)
            features2 = np.random.randn(n, feature_dim)
            
            mu1, sigma1 = FIDCalculator.compute_statistics(features1)
            mu2, sigma2 = FIDCalculator.compute_statistics(features2)
            
            fid = FIDCalculator.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
            fids.append(fid)
        
        results.append({
            'n': n,
            'mean_fid': np.mean(fids),
            'std_fid': np.std(fids)
        })
        print(f"N={n:>6}: FID = {np.mean(fids):.2f} ± {np.std(fids):.2f}")
    
    return results
```

**권하는 바:**

- **가장 적게**: 표본 2,048개(= 특징 차원)
- **권함**: 표본 10,000개 이상
- **가장 좋음**: 안정된 견줌을 위해 표본 50,000개 이상

### 부트스트랩 믿음 구간

```python
def bootstrap_fid(real_features: np.ndarray,
                  gen_features: np.ndarray,
                  n_bootstrap: int = 1000,
                  sample_size: Optional[int] = None) -> Tuple[float, float, float]:
    """
    부트스트랩 믿음 구간과 함께 FID를 셈한다.
    
    인수:
        real_features: 참 자료의 특징 [N, D]
        gen_features: 만들어 낸 자료의 특징 [N, D]
        n_bootstrap: 부트스트랩 표본 수
        sample_size: 부트스트랩 표본의 크기(기본값: min(N_real, N_gen))
        
    반환값:
        95% 믿음 구간의 (FID, 아래 끝, 위 끝) 튜플
    """
    n_real = len(real_features)
    n_gen = len(gen_features)
    
    if sample_size is None:
        sample_size = min(n_real, n_gen)
    
    bootstrap_fids = []
    
    for _ in range(n_bootstrap):
        # 부트스트랩 표본
        idx_real = np.random.choice(n_real, sample_size, replace=True)
        idx_gen = np.random.choice(n_gen, sample_size, replace=True)
        
        real_sample = real_features[idx_real]
        gen_sample = gen_features[idx_gen]
        
        mu_r, sigma_r = FIDCalculator.compute_statistics(real_sample)
        mu_g, sigma_g = FIDCalculator.compute_statistics(gen_sample)
        
        fid = FIDCalculator.calculate_frechet_distance(mu_r, sigma_r, mu_g, sigma_g)
        bootstrap_fids.append(fid)
    
    # 믿음 구간을 셈한다
    fid_mean = np.mean(bootstrap_fids)
    lower = np.percentile(bootstrap_fids, 2.5)
    upper = np.percentile(bootstrap_fids, 97.5)
    
    return fid_mean, lower, upper
```

---

## 7. FID와 인셉션 점수 견주기

| 갈래 | FID | 인셉션 점수 |
|--------|-----|-----|
| **분포 견주기** | 두 분포를 견준다 | 만든 것만 따진다 |
| **봉우리 무너짐 알아내기** | 민감하다(함께 흩어짐으로) | 덜 민감하다 |
| **필요한 표본 크기** | 크다(1만 이상) | 작다(5천 이상) |
| **견줄 자료 묶음 필요** | 그렇다 | 아니다 |
| **셈** | 더 비싸다(함께 흩어짐) | 더 빠르다 |
| **이론 바탕** | 가장 좋은 나르기 | 앎 이론 |

### 어느 쪽을 언제 쓰나

- **FID**: 그림 만들어 내기 품질의 으뜸 잣대
- **인셉션 점수**: 빠른 확인용이며 견줄 자료 묶음이 없을 때 쓸모 있다
- **둘 다**: 두루 따지려면 둘 다 알려라

---

## 8. 한계와 함정

### 1. 정규 분포 가정

FID는 특징이 정규 분포를 따른다고 본다. 이는 다음에서 어긋날 수 있다.

- 봉우리가 아주 많은 특징 공간
- 작은 표본 크기
- 마당 밖의 그림

### 2. 인셉션의 치우침

FID는 InceptionV3이 배운 나타냄에 매여 있다.

```python
def demonstrate_inception_bias():
    """
    FID가 특징 뽑개를 무엇으로 고르느냐에 달렸음을 보인다.
    """
    # InceptionV3, VGG, CLIP으로 FID를 재면 값이 달라진다
    # "옳은" FID는 뜻의 닮음을 무엇으로 보느냐에 달렸다
    print("Different feature extractors give different FIDs:")
    print("- InceptionV3: Standard choice, trained on ImageNet")
    print("- CLIP: Better for text-to-image evaluation")
    print("- SwAV: Self-supervised features, less class-biased")
```

### 3. 미리 다듬기에 민감함

미리 다듬기가 한결같지 않으면 FID가 틀린다.

```python
# 나쁨: 한결같지 않은 미리 다듬기
real_images = resize(real_images, 299)  # 쌍선형
gen_images = resize(gen_images, 299)    # 다른 방법이다!

# 좋음: 한결같은 미리 다듬기
def consistent_preprocess(images):
    return F.interpolate(images, size=(299, 299), 
                        mode='bilinear', align_corners=False)
```

### 4. FID가 모든 것을 알아내지는 못한다

FID가 놓칠 수 있는 것:

- 미묘한 흠(흐림, 잡음 결)
- 외우기(익히기 자료 베끼기)
- 통계에 영향을 주지 않는 느낌의 문제

---

## 9. 모범 사례

### 1. 자리 잡은 꾸러미를 쓰라

```python
# 권함: torch-fidelity
from torch_fidelity import calculate_metrics

metrics = calculate_metrics(
    input1='path/to/real',
    input2='path/to/generated',
    cuda=True,
    fid=True,
    verbose=True
)
print(f"FID: {metrics['frechet_inception_distance']}")
```

### 2. 견줄 통계를 미리 셈하라

```python
# 실제 자료 통계를 한 번만 갈무리한다
save_reference_statistics(real_images, 'cifar10_stats.npz')

# 익히는 동안 다시 쓴다
mu_real, sigma_real = load_reference_statistics('cifar10_stats.npz')

for epoch in range(epochs):
    # 표본 만들기
    fake_images = generator.sample(10000)
    
    # FID를 효율 좋게 셈한다
    fid = calculator.calculate_fid_from_statistics(
        mu_real, sigma_real, fake_images
    )
    print(f"Epoch {epoch}: FID = {fid:.2f}")
```

### 3. 맥락과 함께 알려라

```python
def report_fid(fid: float, n_real: int, n_gen: int):
    """맥락과 함께 FID를 제대로 알린다."""
    print(f"FID: {fid:.2f}")
    print(f"  Real samples: {n_real:,}")
    print(f"  Generated samples: {n_gen:,}")
    print(f"  Feature extractor: InceptionV3 (ImageNet)")
    print(f"  Preprocessing: 299×299, bilinear, ImageNet normalization")
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
프레셰 인셉션 거리(FID)를 뜻매김하고 맞겨루기 생성기를 따질 때 인셉션 점수보다 이를 더 낫게 여기는 까닭을 설명하라.

</div>

??? success "연습문제 1 풀이"
    FID는 실제 그림과 만든 그림의 Inception-v3 특징 분포를 여러 변수 정규 분포 $\mathcal{N}(\mu_r, \Sigma_r)$과 $\mathcal{N}(\mu_g, \Sigma_g)$으로 나타낸 뒤 다음을 셈한다.

    $$\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$

    FID를 더 낫게 여기는 까닭은 이렇다. (1) 만든 그림을 실제 그림과 견준다(인셉션 점수는 만든 그림만 따진다). (2) 봉우리 무너짐을 알아낸다(평균과 함께 흩어짐이 달라진다). (3) 사람의 판단과 더 잘 맞는다. (4) FID가 낮을수록 느낌의 품질과 더 잘 이어진다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
인셉션 점수의 한계는 무엇인가? 표본이 나쁜데도 인셉션 점수가 높을 수 있는가?

</div>

??? success "연습문제 2 풀이"
    인셉션 점수 = $\exp(\mathbb{E}_x [D_{\text{KL}}(p(y|x) \| p(y))])$이며 $p(y|x)$은 만든 그림 $x$에 대한 인셉션 판별기의 헤아림이다. 한계는 이렇다. (1) 품질(또렷하고 가를 수 있음)과 다양함(갈래에 고루 퍼짐)만 재고 익히기 자료에 얼마나 충실한지는 재지 않는다. (2) 갈래마다 완벽한 그림 하나씩만 만드는 모델도 점수가 높지만 갈래 안의 다양함은 무시한다. (3) 인셉션 모델의 치우침에 민감하다. (4) 갈래 안의 결이나 결결이 품질을 담지 못한다. 그렇다. ImageNet 갈래마다 대표 그림 하나씩을 외우면 인셉션 점수가 높아질 수 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
만들어 내는 모델의 정밀도와 재현율 잣대가 가르기에서의 그것과 어떻게 다른지 설명하라.

</div>

??? success "연습문제 3 풀이"
    만들어 내는 자리에서는(Kynkaanniemi 외, 2019) **정밀도**가 만든 표본 가운데 실제 자료 분포의 받침 안에 드는 몫을 잰다(품질/충실함). **재현율**은 실제 자료 가운데 만든 분포의 받침 안에 드는 몫을 잰다(다양함/덮기). 정밀도가 높고 재현율이 낮으면 봉우리 무너짐이다(봉우리는 적지만 그럴듯하다). 정밀도가 낮고 재현율이 높으면 품질은 나쁘지만 다양하다. 띄엄띄엄한 맞음을 세는 가르기의 정밀도/재현율과 달리 만들어 내기의 정밀도/재현율은 특징 공간에서 $k$번째 가장 가까운 이웃 거리로 분포의 받침을 어림한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
만들어 내는 모델을 따질 때 여러 잣대를 함께 써야 하는 까닭은 무엇인가?

</div>

??? success "연습문제 4 풀이"
    어느 잣대 하나도 만들어 내기 품질의 모든 면을 담지 못한다. **FID**은 전체 분포의 닮음을 재지만 품질과 다양함을 뒤섞는다. **인셉션 점수**는 품질과 다양함을 담지만 익히기 자료에 대한 충실함은 무시한다. **정밀도/재현율**은 품질과 다양함을 갈라내지만 특징 뽑개와 $k$을 어떻게 고르느냐에 매인다. **느낌 잣대**(LPIPS)는 그림 수준 품질을 재지만 다양함은 재지 않는다. 잣대를 함께 쓰면 온전한 그림이 보인다. 곧 FID가 낮고 정밀도가 높으며 재현율이 낮은 모델은 봉우리가 무너진 것이고, 재현율이 높고 정밀도가 낮은 모델은 다양하지만 품질 낮은 표본을 낸다. 마지막 판단에는 사람이 따지는 것이 여전히 으뜸 기준이다.


---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
FID의 식을 적고 두 항이 각각 무엇을 재는지 밝혀라.

</div>

??? success "연습문제 5 풀이"
    두 특징 분포를 가우시안으로 보고 그 사이의 프레셰 거리를 잰다.

    $$\mathrm{FID} = \underbrace{\|\mu_r - \mu_g\|^2}_{\text{평균의 어긋남}}
      + \underbrace{\operatorname{tr}\!\big(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\big)}_{\text{퍼짐의 어긋남}}$$

    앞 항은 특징의 **가운데**가 얼마나 다른지, 뒤 항은 **퍼진 모양**이 얼마나 다른지 본다.

    뒤 항이 다양성을 잡아내는 자리다. 모델이 몇 가지만 내놓으면 $\Sigma_g$가 작아져
    $\Sigma_r$과 어긋나므로 값이 커진다. 인셉션 점수에는 이런 항이 없어 다양성 부족을
    놓쳤다([인셉션 점수 연습문제 2](inception_score.md)).

    두 성질을 짚어 둘 만하다.

    - **작을수록 좋다.** 거리이므로 0이 완벽이다
    - **참 자료가 식에 들어온다.** $\mu_r$, $\Sigma_r$이 그것이다

    둘째가 인셉션 점수와의 근본적인 차이다. 만들어 낸 것만 보는 잣대와 자료와 견주는
    잣대는 겨냥하기의 난이도가 다르다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
FID는 표본 수에 따라 얼마나 달라지는가? 참 자료끼리 재어 확인하라.

</div>

??? success "연습문제 6 풀이"
    같은 분포에서 뽑은 두 배치가므로 참값이 0이어야 한다. 그런데 이렇다. MNIST 시험
    자료를 무작위로 절반씩 갈라 쟀다.

    | 표본 수 | 100 | 250 | 500 | 1,000 | 2,500 | 5,000 |
    |---|---|---|---|---|---|---|
    | 참 대 참 FID | 81.4 | 24.0 | 15.1 | 6.4 | 3.1 | **1.8** |

    표본이 적으면 크게 부풀려진다. 100개에서 81.4인데 이것은 **완전히 실패한 모델에서나
    볼 값**이다. 실제로는 참 자료를 참 자료와 견준 것이다.

    까닭은 공분산 추정에 있다. 128차원 공분산에 8,256개의 값을 추정해야 하는데 표본이
    100개면 턱없이 모자라다. 추정 오차가 그대로 거리로 들어간다.

    그리고 쏠림의 **방향이 정해져 있다.** 늘 크게 나온다. 잡음이 거리를 늘리기만 하기
    때문이다. 그러므로 평균을 내도 사라지지 않는다.

    실무의 결론이 둘이다.

    - **표본 수를 고정하고 밝혀라.** 표본 수가 다른 FID는 견줄 수 없다
    - **충분히 많이 써라.** 관례는 10,000개에서 50,000개다

    그리고 **참 대 참 FID를 재어 바닥을 알아 두는 것**이 좋다. 이 설정에서 1.8이 바닥이라면
    모델의 2.0은 사실상 완벽에 가깝다는 뜻이다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff hard" title="어려움"></span>
참 자료를 순서대로 절반씩 갈라 FID를 재면 표본을 늘려도 값이 줄지 않았다. 왜인가?

</div>

??? success "연습문제 7 풀이"
    **MNIST 시험 자료의 앞 절반과 뒤 절반이 실제로 다른 자료**이기 때문이다. 표본 수
    문제가 아니었다.

    | 어떻게 갈랐는가 | FID (5,000개씩) |
    |---|---|
    | 순서대로 (앞 5,000 대 뒤 5,000) | **38.1** |
    | 무작위로 | 1.8 |

    스무 배 차이다. 그리고 분류기의 정확도도 다르다.

    | | 앞 5,000개 | 뒤 5,000개 |
    |---|---|---|
    | 분류 정확도 | 97.76% | **99.24%** |

    앞쪽이 뚜렷하게 어렵다. MNIST의 시험 자료가 두 다른 출처에서 모인 것이고 섞이지 않은
    채로 놓여 있다는 것이 알려져 있는데, 그 성질이 이 수치에 그대로 나타난 셈이다.

    이 일을 겪은 과정이 교훈이다. 나는 처음에 순서대로 갈라 FID의 표본 쏠림을 재려 했고,
    값이 1,000개를 넘어서도 38 근처에서 평평해지는 것을 보았다. 0으로 가야 할 값이
    멈춰 있으니 **무언가 틀렸다**는 신호였다. 무작위로 갈라 다시 재어 보고서야 두 절반이
    다르다는 것을 알았다.

    실무의 교훈이 둘이다.

    **자료를 가를 때는 섞어라.** 순서에 뜻이 있는 자료가 생각보다 많다.

    **참값을 아는 검사를 먼저 하라.** 참 대 참 FID가 0으로 가야 한다는 것을 알고 있었기에
    이상을 알아챌 수 있었다. 그 검사가 없었다면 38이라는 바닥을 잣대의 성질로 잘못
    알았을 것이다.

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
행렬 제곱근을 셈할 때 무엇을 조심해야 하는가?

</div>

??? success "연습문제 8 풀이"
    `scipy.linalg.sqrtm`이 **복소수를 돌려줄 수 있다.**

    $\Sigma_r \Sigma_g$가 대칭이 아니므로(두 대칭 행렬의 곱은 대칭이 아니다) 수치적으로
    작은 허수부가 생긴다. 수학적으로는 실수여야 하므로 버리면 된다.

    ```python
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    ```

    `disp=False`로 두는 까닭은 경고를 직접 다루기 위해서다. 기본값이면 수렴하지 않았을 때
    화면에 찍는다.

    허수부가 **큰지 확인해 보는 것**이 좋은 버릇이다. 작으면 수치 오차이지만 크면 무언가
    잘못된 것이다.

    ```python
    assert np.abs(covmean.imag).max() < 1e-3
    ```

    공분산이 특이에 가까울 때 더 불안정해진다. 표본 수가 특징 차원보다 적으면 공분산이
    반드시 특이하므로 이 문제가 심해진다. 앞에서 본 작은 표본의 쏠림과 같은 뿌리다.

    대각선에 작은 값을 더해 안정화하는 방법이 널리 쓰인다.

    ```python
    sigma = sigma + np.eye(d) * 1e-6
    ```

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
FID가 가우시안을 가정하는 것이 문제가 되는가?

</div>

??? success "연습문제 9 풀이"
    실제 특징 분포가 가우시안이 아니므로 **가정이 틀린 것은 맞다.** 그래도 널리 쓰이는
    까닭이 있다.

    FID가 쓰는 것은 평균과 공분산, 곧 **2차 통계량까지**다. 그보다 높은 차수의 차이는
    보지 못한다. 그러므로 2차 통계량이 같으면서 실제로는 아주 다른 두 분포를 구별하지
    못한다.

    그런 예를 일부러 만들 수 있다. 특징 공간에서 평균과 공분산을 맞춰 놓은 엉뚱한 분포는
    FID가 0에 가깝게 나온다. 곧 **겨냥할 수 있다.**

    그래도 실무에서 잘 듣는 까닭은 두 가지로 이야기된다. 특징 공간이 이미 깊은 그물을
    지난 뒤라 꽤 가우시안에 가깝고, 사람이 보는 품질과의 상관이 경험적으로 높다.

    가정을 피하는 대안도 있다.

    | 잣대 | 가정 |
    |---|---|
    | FID | 가우시안 (2차 통계량만) |
    | 커널 인셉션 거리(KID) | 없다. 커널로 잰다 |
    | 정밀도와 재현율 | 없다. 이웃으로 잰다 |

    KID는 쏠림이 없다는 이점도 있어 표본이 적을 때 FID보다 낫다고 권해진다. 그런데도
    FID가 표준으로 남은 것은 관례의 힘이며, 논문끼리 견주려면 같은 잣대를 써야 하기
    때문이다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
FID 하나만 보고 무엇을 알 수 없는가?

</div>

??? success "연습문제 10 풀이"
    **왜 나쁜지**를 알 수 없다. 값이 크다는 것이 둘 가운데 어느 쪽인지 가려 주지 않는다.

    - 표본의 품질이 나쁘다 (얼룩이 섞여 있다)
    - 다양성이 부족하다 (몇 가지만 잘 만든다)

    두 경우가 같은 FID를 낼 수 있다. 하나의 수로 뭉쳐 놓았기 때문이다.

    이것이 정밀도와 재현율이 나온 까닭이다([정밀도와 재현율](precision_recall.md)).

    | | 낮으면 |
    |---|---|
    | 정밀도 | 품질 문제. 자료답지 않은 것을 낸다 |
    | 재현율 | 다양성 문제. 자료의 일부만 덮는다 |

    진단이 되므로 손볼 방향이 나온다. 재현율이 낮으면 무너짐을 다루고, 정밀도가 낮으면
    익히기를 더 하거나 얼개를 손본다.

    실제로 이 장의 두 모델이 FID로는 순서가 나오지만 그 까닭은 다르다. 그러니 FID를
    적을 때 **정밀도와 재현율을 함께** 적는 것이 요즘 관례다.

---

<div class="drillbox" markdown>

**연습문제 11.** <span class="diff med" title="중간"></span>
FID를 익히기 도중에 재어 모델을 고르는 것이 괜찮은가?

</div>

??? success "연습문제 11 풀이"
    흔히 하는 일이고 조심할 점이 몇 가지다.

    **표본 수를 고정해야 한다.** 표본 수가 다르면 다른 잣대다(연습문제 2). 에포크마다
    다른 수로 재면 곡선을 읽을 수 없다.

    **어느 자료와 견주는지 고정해야 한다.** 학습 자료와 견줄 것인가 검증 자료와 견줄
    것인가를 정해 두어야 한다. 모델을 고르는 데 쓰면서 학습 자료와 견주면, FID가 학습
    자료에 맞추는 것을 상으로 준다.

    **잣대를 겨냥하게 된다.** FID로 고르면 FID가 좋은 모델을 얻지만 그것이 사람이 보기에
    좋은 모델이라는 보장은 줄어든다. 고르는 데 쓴 잣대는 더는 공정한 심판이 아니다.

    셋째가 원칙적인 문제다. [26장의 온도](../../ch26/training/generate_samples.md)에서
    본 것처럼 잣대를 올리는 길과 모델을 좋게 하는 길이 갈릴 수 있다.

    그래서 권할 만한 방식이 이렇다. 고르는 데는 검증 자료 기준 FID를 쓰고, 마지막
    보고에는 건드리지 않은 자료로 다시 재며, **표본 격자를 눈으로 함께 본다.**

---

<div class="drillbox" markdown>

**연습문제 12.** <span class="diff easy" title="쉬움"></span>
FID를 보고할 때 무엇을 함께 적어야 하는가?

</div>

??? success "연습문제 12 풀이"
    | 적을 것 | 왜 |
    |---|---|
    | 특징 그물 | 인셉션인지 다른 것인지 |
    | 어느 층의 특징인가 | 층이 다르면 다른 값이다 |
    | 표본 수 (양쪽 모두) | 쏠림이 있다 |
    | 참 자료가 무엇인가 | 학습인지 시험인지 |
    | 참 대 참 FID | 바닥을 알려 준다 |

    마지막 줄이 흔히 빠지는데 값이 크다. 이 설정의 바닥이 1.8이라는 것을 알면 모델의
    FID 40을 읽는 눈이 달라진다.

    둘째 줄도 자주 어긋난다. 인셉션에서도 어느 층(pool3, 2048차원이 관례)을 쓰는지가
    값을 정한다.

    이 장의 수치라면 이렇게 적어야 한다. **MNIST 분류기(시험 정확도 98.50%)의 128차원
    은닉층, 표본 10,000개, 학습 자료 앞 10,000개를 기준으로, 참 대 참 바닥 1.8.**

---

<div class="drillbox" markdown>

**연습문제 13.** <span class="diff hard" title="어려움"></span>
FID를 겨냥해 속이는 모델을 만들 수 있는가?

</div>

??? success "연습문제 13 풀이"
    만들 수 있고, 두 갈래가 있다.

    **가정을 겨냥한다.** FID가 2차 통계량만 보므로, 특징 공간의 평균과 공분산을 맞추면서
    사람 눈에는 엉뚱한 표본을 내놓는 모델을 만들 수 있다. 곧 $\mu$와 $\Sigma$를 맞추는
    것을 손실로 두고 직접 최적화하면 된다.

    **학습 자료를 외운다.** 가장 단순하고 가장 고약하다. 학습 자료를 그대로 내놓으면
    FID가 바닥에 닿는다. 다양성도 완벽하고 품질도 완벽하다.

    둘째가 원칙적인 문제다. **FID는 외우기를 벌하지 않는다.** 자료의 분포와 같아지는 것을
    상으로 주는 잣대이니, 자료를 되읽는 것이 만점이다.

    그래서 FID만으로는 만들어 내는 모델을 값매김할 수 없다. 함께 보아야 하는 것들이 있다.

    - **새로움**: 만든 표본과 가장 가까운 학습 표본의 거리
    - **일반화**: 학습 자료 기준 FID와 시험 자료 기준 FID의 차이
    - **가능도**: 모델이 못 본 자료에 주는 확률

    둘째가 실용적으로 쓸 만하다. 학습 쪽 FID만 좋고 시험 쪽이 나쁘면 외우고 있는 것이다.

    이것이 이 사다리 전체의 되풀이되는 교훈이기도 하다. **잣대는 목표의 대리일 뿐이며,
    대리를 최적화하면 목표와 갈라지는 자리가 반드시 있다.**

---

<div class="drillbox" markdown>

**연습문제 14.** <span class="diff med" title="중간"></span>
특징 그물을 무엇으로 쓰는지가 FID 값에 얼마나 영향을 주는가?

</div>

??? success "연습문제 14 풀이"
    값 자체를 완전히 바꾼다. 같은 모델에 대해 다른 수가 나오므로 **서로 견줄 수 없다.**

    이 장이 MNIST 분류기를 쓴 까닭은 인셉션이 손글씨 숫자를 본 적이 없기 때문이다
    ([인셉션 점수 연습문제 4](inception_score.md)). 그 선택이 값의 눈금을 정한다.

    영향을 주는 것들이 이렇다.

    | 무엇 | 어떻게 |
    |---|---|
    | 어떤 자료로 익혔는가 | 무엇을 특징으로 보는지가 달라진다 |
    | 어느 층 | 얕으면 결, 깊으면 뜻 |
    | 차원 | 높으면 공분산 추정이 어렵다. 쏠림이 커진다 |
    | 얼마나 잘 익었는가 | 약한 그물은 차이를 못 본다 |

    셋째 줄이 실용적으로 중요하다. 인셉션의 관례인 2,048차원은 공분산에 210만 개의 값을
    추정하는 것이므로 표본이 아주 많이 필요하다. 이 장의 128차원은 그 점에서 훨씬 편하고,
    5,000개로 바닥이 1.8까지 내려간 것도 그 덕이다.

    그래서 **자기 자료에 맞는 특징 그물을 쓰는 것이 옳고, 그 사실을 밝히는 것이 함께
    옳다.** 남의 수와 견주고 싶다면 남과 같은 그물을 써야 한다.

## 정리하며

!!! success "핵심 간추리기"
    
    1. **FID 공식**: $\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2})$
    
    2. **풀이**: FID가 낮을수록 실제 자료 분포와 더 비슷하다
    
    3. **흔한 값**: 뛰어남(<5), 좋음(5-20), 보통(20-50), 나쁨(>100)
    
    4. **필요한 표본**: 적어도 2048개, 권하기로는 10,000개 이상
    
    5. **가장 좋은 방식**: 자리 잡은 꾸러미를 쓰고, 통계를 미리 셈하며, 미리 다듬기를 한결같이 하라

**참고 문헌**

1. Heusel, M., et al. (2017). "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium." *NeurIPS*.

2. Parmar, G., et al. (2021). "On Aliased Resizing and Surprising Subtleties in GAN Evaluation." *CVPR*.

3. Bińkowski, M., et al. (2018). "Demystifying MMD GANs." *ICLR*.

4. Chong, M. J., & Forsyth, D. (2020). "Effectively Unbiased FID and Inception Score and Where to Find Them." *CVPR*.
