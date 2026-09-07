# 단원 52: 프레셰 인셉션 거리(FID)

이 단원은 깊은 만들어 내는 모델의 중요한 부품인 단원 52: 프레셰 인셉션 거리(FID)을 짠다. 이 짜기를 이해하면 요즘 만들어 내는 모델에 쓰이는 얼개의 결과 익히기 절차를 꿰뚫어 볼 수 있다. 이 코드는 연구와 실제 얼개에서 널리 쓰이는 쓸모 있는 재주를 보인다.

## 코드

```python
"""
Module 52: Fréchet Inception Distance (FID)
==========================================

가장 널리 쓰이는 잣대 가운데 하나인 FID을 두루 갖추어 짠다
만들어 내는 모델을 따지는 데 쓴다.

학습 목표:
-------------------
1. FID의 수학 바탕을 이해한다
2. FID을 바닥부터 짠다
3. 특징 뽑기에 미리 익힌 InceptionV3을 쓴다
4. FID 점수를 제대로 풀이한다

핵심 공식:
-----------
FID = ||μ_real - μ_gen||² + Tr(Σ_real + Σ_gen - 2(Σ_real × Σ_gen)^{1/2})

지은이: 가르치기 인공 지능 모둠
날짜: 2025
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
from typing import Tuple
import warnings

# ========================================================================
# 메인
# ========================================================================

torch.manual_seed(42)
np.random.seed(42)


class FIDCalculator:
    """
    프레셰 인셉션 거리 셈개.
    
    수학적 바탕:
    -----------------------
    FID은 여러 변수 정규 분포 둘 사이의 거리를 잰다.
    
    Real distribution: X_real ~ N(μ_r, Σ_r)  
    Generated distribution: X_gen ~ N(μ_g, Σ_g)
    
    FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^{1/2})
    
    왜 프레셰 거리인가?
    --------------------
    1. The Fréchet distance (also called Wasserstein-2 distance) is the
       정규 분포의 가장 좋은 나르기 거리
    2. 정규 분포에는 닫힌 꼴 풀이가 있다
    3. 평균과 함께 흩어짐의 차이에 모두 민감하다
    4. Lower FID = distributions are more similar
    
    왜 인셉션 특징인가?
    ----------------------
    1. ImageNet으로 익힌 InceptionV3은 그림의 뜻 특징을 담는다
    2. Pool3 features (2048-dim) represent high-level image content
    3. 화소 수준 견줌보다 튼튼하다
    4. 사람의 판단과 잘 이어진다
    
    흔한 FID 값:
    ------------------
    - FID < 10: Excellent (near-perfect generation)
    - FID 10-50: 좋은 품질
    - FID 50-100: 보통 품질
    - FID > 100: 나쁜 품질
    """
    
    @staticmethod
    def calculate_frechet_distance(mu1: np.ndarray,
                                   sigma1: np.ndarray,
                                   mu2: np.ndarray,
                                   sigma2: np.ndarray,
                                   eps: float = 1e-6) -> float:
        """
        정규 분포 둘 사이의 프레셰 거리를 셈한다.
        
        수학의 이끌어 내기:
        -----------------------
        For X ~ N(μ₁, Σ₁) and Y ~ N(μ₂, Σ₂), the Fréchet distance is:
        
        d²_F(X,Y) = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2(Σ₁Σ₂)^{1/2})
        
        공식을 나누어 보면:
        1. ||μ₁ - μ₂||²: Difference in means (first moment)
        2. Tr(Σ₁ + Σ₂): Sum of variances
        3. -2Tr((Σ₁Σ₂)^{1/2}): Covariance overlap term
        
        인수:
            mu1: Mean of first distribution [d]
            sigma1: Covariance matrix of first distribution [d, d]
            mu2: Mean of second distribution [d]
            sigma2: Covariance matrix of second distribution [d, d]
            eps: 수치 안정성을 위한 작은 상수
        
        반환값:
            Fréchet distance (scalar)
        """
        # 들임이 넘파이 배열이 되게 한다
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        assert mu1.shape == mu2.shape, "Means must have same shape"
        assert sigma1.shape == sigma2.shape, "Covariances must have same shape"
        
        # 1. 평균 차이를 셈한다: ||μ₁ - μ₂||²
        diff = mu1 - mu2
        mean_diff_squared = np.dot(diff, diff)
        
        # 2. 행렬 제곱근을 셈한다: (Σ₁Σ₂)^{1/2}
        # 이것이 셈이 가장 비싼 걸음이다
        
        # 행렬 곱: Σ₁ @ Σ₂
        # 꼴: [d, d] @ [d, d] = [d, d]
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        # 행렬 제곱근의 수치 어긋남을 다룬다
        # 수치 문제로 sqrtm이 복소수를 돌려줄 때가 있다
        if not np.isfinite(covmean).all():
            print(f"WARNING: FID calculation produced non-finite values.")
            print(f"Adding {eps} to diagonal of covariance matrices.")
            # 안정을 위해 대각선에 작은 값을 더한다
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # 허수 몫이 작으면 실수부를 취한다
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                max_imag = np.max(np.abs(covmean.imag))
                raise ValueError(f"Imaginary component too large: {max_imag}")
            covmean = covmean.real
        
        # 3. 대각합 항을 셈한다: Tr(Σ₁ + Σ₂ - 2(Σ₁Σ₂)^{1/2})
        trace_term = np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        
        # 4. 마지막 FID = ||μ₁ - μ₂||² + Tr(...)
        fid = mean_diff_squared + trace_term
        
        return float(fid)
    
    @staticmethod
    def compute_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        특징의 평균과 함께 흩어짐을 셈한다.
        
        인수:
            features: Feature vectors [n_samples, feature_dim]
        
        반환값:
            Tuple of (mean [feature_dim], covariance [feature_dim, feature_dim])
        
        수학의 참고:
        ------------------
        Mean: μ = (1/N) Σ x_i
        Covariance: Σ = (1/N) Σ (x_i - μ)(x_i - μ)ᵀ
        
        We use rowvar=False to treat each column as a variable
        """
        # 평균을 셈한다: 표본에 걸쳐 평균 낸다
        # 꼴: [특징 차원]
        mu = np.mean(features, axis=0)
        
        # 함께 흩어짐 행렬을 셈한다
        # 꼴: [특징 차원, 특징 차원]
        # rowvar=False: 세로줄마다 변수, 가로줄마다 관측이다
        sigma = np.cov(features, rowvar=False)
        
        return mu, sigma
    
    @staticmethod
    def calculate_fid(real_features: np.ndarray,
                     generated_features: np.ndarray) -> float:
        """
        실제 특징과 만든 특징으로 FID을 셈한다.
        
        온전한 물길:
        -----------------
        1. Extract features from InceptionV3 (done before this function)
        2. Compute statistics (μ, Σ) for real data
        3. Compute statistics (μ, Σ) for generated data
        4. 프레셰 거리를 셈한다
        
        인수:
            real_features: Features from real images [n_real, 2048]
            generated_features: Features from generated images [n_gen, 2048]
        
        반환값:
            FID score (lower is better)
        
        가장 작은 표본 크기:
        --------------------
        - Absolute minimum: 2048 samples (= feature dimension)
        - 권함: 안정된 어림을 위해 표본 10,000개 이상
        - More samples = more reliable FID
        """
        print(f"Computing FID with {len(real_features)} real and "
              f"{len(generated_features)} generated samples...")
        
        # 실제 자료의 통계를 셈한다
        mu_real, sigma_real = FIDCalculator.compute_statistics(real_features)
        
        # 만든 자료의 통계를 셈한다
        mu_gen, sigma_gen = FIDCalculator.compute_statistics(generated_features)
        
        # 프레셰 거리를 셈한다
        fid = FIDCalculator.calculate_frechet_distance(
            mu_real, sigma_real, mu_gen, sigma_gen
        )
        
        print(f"✓ FID computed: {fid:.4f}")
        
        return fid


class SimpleInceptionV3Wrapper:
    """
    가르치기 위한 단순한 InceptionV3 감싸개.
    
    실제로는 다음을 쓴다.
    - torchvision.models.inception_v3(pretrained=True)
    - torch-fidelity 꾸러미
    - pytorch-fid 꾸러미
    
    이 갈래는 실제 InceptionV3 무게 없이
    핵심 개념을 보인다.
    """
    
    def __init__(self, feature_dim: int = 2048):
        """
        흉내 특징 뽑개로 첫자리매김한다.
        
        인수:
            feature_dim: Dimension of feature vectors (2048 for InceptionV3)
        """
        self.feature_dim = feature_dim
        print(f"Initialized mock InceptionV3 with {feature_dim}-dim features")
        print("NOTE: For real FID, use actual InceptionV3 pretrained on ImageNet")
    
    def extract_features(self, images: torch.Tensor) -> np.ndarray:
        """
        Extract features from images (mock implementation).
        
        실제 짜기에서는:
        1. Preprocess images to InceptionV3 format (299×299, normalized)
        2. InceptionV3을 지나는 앞먹임
        3. Extract pool3 features (2048-dim)
        4. No gradients needed (eval mode)
        
        인수:
            images: Batch of images [batch_size, C, H, W]
        
        반환값:
            Features [batch_size, feature_dim]
        """
        batch_size = images.shape[0]
        
        # 흉내 특징(실제로는 InceptionV3에서 온다)
        # 그럴듯한 통계 성질을 지닌 특징을 만든다
        features = torch.randn(batch_size, self.feature_dim)
        
        # 그림에 따라 달라지는 몫을 더한다(실제 특징 뽑기를 흉내 낸다)
        image_stats = images.mean(dim=[1,2,3], keepdim=True)
        features = features + image_stats * 0.1
        
        return features.numpy()


def demonstrate_fid_computation():
    """
    인공 자료로 FID 셈하기를 보인다.
    """
    print("=" * 70)
    print("Fréchet Inception Distance (FID) Demonstration")
    print("=" * 70)
    
    # 상황 1: 똑같은 분포(FID이 ~0이어야 한다)
    print("\nScenario 1: Identical Distributions")
    print("-" * 70)
    
    mu1 = np.array([0.0, 0.0])
    sigma1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    mu2 = np.array([0.0, 0.0])
    sigma2 = np.array([[1.0, 0.0], [0.0, 1.0]])
    
    fid_identical = FIDCalculator.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    print(f"Mean 1: {mu1}")
    print(f"Mean 2: {mu2}")
    print(f"FID: {fid_identical:.6f}")
    print("Interpretation: FID ≈ 0 indicates identical distributions")
    
    # 상황 2: 평균만 다르다
    print("\nScenario 2: Different Means (Same Covariance)")
    print("-" * 70)
    
    mu1 = np.array([0.0, 0.0])
    sigma1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    mu2 = np.array([3.0, 3.0])  # 옮겨진 평균
    sigma2 = np.array([[1.0, 0.0], [0.0, 1.0]])
    
    fid_mean = FIDCalculator.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    print(f"Mean 1: {mu1}")
    print(f"Mean 2: {mu2}")
    print(f"Mean difference norm: {np.linalg.norm(mu1 - mu2):.4f}")
    print(f"FID: {fid_mean:.4f}")
    print("Interpretation: FID increases with mean difference")
    
    # 상황 3: 함께 흩어짐이 다르다
    print("\nScenario 3: Different Covariances (Same Mean)")
    print("-" * 70)
    
    mu1 = np.array([0.0, 0.0])
    sigma1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    mu2 = np.array([0.0, 0.0])
    sigma2 = np.array([[4.0, 0.0], [0.0, 4.0]])  # 더 큰 흩어짐
    
    fid_cov = FIDCalculator.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    print(f"Covariance 1:\n{sigma1}")
    print(f"Covariance 2:\n{sigma2}")
    print(f"FID: {fid_cov:.4f}")
    print("Interpretation: FID sensitive to variance differences")
    
    # 상황 4: 둘 다 다르다
    print("\nScenario 4: Both Mean and Covariance Different")
    print("-" * 70)
    
    mu1 = np.array([0.0, 0.0])
    sigma1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    mu2 = np.array([2.0, 2.0])
    sigma2 = np.array([[3.0, 0.5], [0.5, 3.0]])  # 다른 흩어짐 + 서로 이어짐
    
    fid_both = FIDCalculator.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    print(f"Mean difference: {np.linalg.norm(mu1 - mu2):.4f}")
    print(f"FID: {fid_both:.4f}")
    print("Interpretation: FID captures both mean and covariance differences")
    
    # 요약
    print("\n" + "=" * 70)
    print("FID Comparison Summary:")
    print("=" * 70)
    print(f"Identical distributions:     FID = {fid_identical:.4f}")
    print(f"Different means:             FID = {fid_mean:.4f}")
    print(f"Different covariances:       FID = {fid_cov:.4f}")
    print(f"Both different:              FID = {fid_both:.4f}")
    print("\nKey Insight: FID increases as distributions become more different")


def demonstrate_fid_with_features():
    """
    특징 벡터로 FID 셈하기를 보인다.
    """
    print("\n" + "=" * 70)
    print("FID with Feature Vectors")
    print("=" * 70)
    
    # 인공 특징 벡터를 만든다
    n_samples = 5000
    feature_dim = 2048
    
    print(f"\nGenerating {n_samples} samples with {feature_dim} features...")
    
    # 실제 분포: N(0, I)
    real_features = np.random.randn(n_samples, feature_dim)
    
    # 만든 분포 1: 똑같다(FID이 낮아야 한다)
    gen1_features = np.random.randn(n_samples, feature_dim)
    
    # 만든 분포 2: 평균이 옮겨졌다
    gen2_features = np.random.randn(n_samples, feature_dim) + 0.5
    
    # 만든 분포 3: 흩어짐이 줄었다(봉우리 무너짐 표시)
    gen3_features = np.random.randn(n_samples, feature_dim) * 0.5
    
    # FID을 셈한다
    print("\n" + "-" * 70)
    print("FID Comparison:")
    print("-" * 70)
    
    fid1 = FIDCalculator.calculate_fid(real_features, gen1_features)
    fid2 = FIDCalculator.calculate_fid(real_features, gen2_features)
    fid3 = FIDCalculator.calculate_fid(real_features, gen3_features)
    
    print(f"\nGenerator 1 (similar):        FID = {fid1:.2f}")
    print(f"Generator 2 (shifted):        FID = {fid2:.2f}")
    print(f"Generator 3 (mode collapse):  FID = {fid3:.2f}")
    
    print("\n" + "-" * 70)
    print("Interpretation:")
    print("-" * 70)
    print("• Lower FID = Better match to real distribution")
    print("• FID sensitive to both mean shifts and variance changes")
    print("• Reduced variance (mode collapse) increases FID")


def main():
    """
    주된 보여 주기 함수.
    """
    print("\n" + "=" * 70)
    print("MODULE 52: FRÉCHET INCEPTION DISTANCE (FID)")
    print("=" * 70)
    
    # 기본 FID 셈하기를 보여 준다
    demonstrate_fid_computation()
    
    # 특징 벡터로 보여 준다
    demonstrate_fid_with_features()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. FID 뜻매김:
       - 실제 분포와 만든 분포 사이의 거리를 잰다
       - 특징 공간에서 정규 분포를 가정한다
       - Lower FID = Better generation quality
    
    2. 수학의 몫:
       - 평균 차이: ||μ_r - μ_g||²
       - Covariance term: Tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^{1/2})
       - 정규 분포의 닫힌 꼴 풀이
    
    3. 왜 InceptionV3인가?
       - 그림의 뜻 특징을 담는다
       - ImageNet으로 미리 익혔다
       - 2048차원 pool3 특징
       - 화소 공간 견줌보다 낫다
    
    4. 표본 크기가 중요하다:
       - Minimum: 2048 samples (= feature dimension)
       - 권함: 표본 10,000개 이상
       - More samples = more stable FID estimates
    
    5. 한계:
       - Assumes Gaussian distributions (may not hold)
       - 특징 뽑개를 무엇으로 고르느냐에 치우친다
       - 모든 잘못됨을 알아내지는 못한다
       - 다른 잣대와 아울러야 한다
    
    6. 흔한 값:
       - FID < 10: 뛰어난 품질
       - FID 10-50: 좋은 품질
       - FID 50-100: 보통 품질
       - FID > 100: 나쁜 품질
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()```

## 논의

이 짜기는 단원 52: 프레셰 인셉션 거리(FID)에 대해 자리 잡은 가장 좋은 방식을 따른다. 코드는 모델 뜻매김, 익히기 논리, 도구 함수를 또렷이 갈라 짜였다. 핵심 설계 결정에는 깨움 함수 고르기, 고르게 맞추기 방책, 가장 좋게 하기 웃매개변수가 들며 모두 익히기의 안정과 내놓기 품질에 크게 영향을 준다.

이 얼개는 깊은 만들어 내는 모델에 흔한 중요한 결 여럿을 보인다. 곧 여러 신경망 층을 지나며 특징을 차츰 다루기, 모델이 곁 앎을 받아들이게 하는 조건 주기 얼개, 익히는 동안 기울기가 안정되게 흐르도록 하는 꼼꼼한 첫자리매김이다.

새 자료 묶음이나 문제 마당에서는 웃매개변수 고르기와 익히기 절차를 꼼꼼히 맞추어야 할 때가 많으므로 다루는 이들은 이에 마음을 써야 한다. 코드가 조각으로 나뉘어 있어 다른 얼개, 손실 함수, 익히기 방책을 실험하기 쉽다.

## 연습문제

**연습문제 1.**
구체적인 들임 텐서로 이 단원의 으뜸 모델의 앞먹임을 좇아라. 층마다 꼴이 어떻게 바뀌는지 적고 내놓기 차원이 바라던 것과 맞는지 확인하라.

??? success "연습문제 1 풀이"
    들임 텐서에서 시작해 층마다 바뀜을 따라가라. 겹말기 층에서는 공간 차원에 공식 $H_{out} = \lfloor(H_{in} + 2p - k) / s\rfloor + 1$을 쓴다. 선형 층에서는 특징 차원의 바뀜을 좇는다. 중간 꼴을 하나씩 적고 마지막 내놓기가 그 일(그림 만들어 내기, 가르기 등)에 바라던 목표 차원과 맞는지 확인하라.

---

**연습문제 2.**
이 짜기의 핵심 웃매개변수(배움 빠르기, 묶음 크기, 얼개 고르기)를 가려내라. 다른 것을 붙박아 두고 하나씩 바꾸어 웃매개변수마다 익히기가 얼마나 민감한지 재는 실험을 짜라.

??? success "연습문제 2 풀이"
    핵심 웃매개변수에는 배움 빠르기(흔히 $10^{-4}$에서 $10^{-3}$), 묶음 크기(64-256), 층과 채널의 수, 깨움 함수가 든다. 웃매개변수마다 값을 3~5가지로 바꾸어 모델을 익히고 알맞은 잣대(손실, 표본 품질, 모이는 빠르기)를 좇아라. 결과를 그려 어느 웃매개변수가 가장 큰 영향을 주는지 가려내라. 흔히 배움 빠르기와 얼개 깊이가 가장 세게 영향을 주고, 묶음 크기는 알맞은 범위 안에서는 웬만큼 영향을 준다.

---

**연습문제 3.**
이 짜기에 새 기능을 더해 넓혀라. 곧 기울기 자르기, 배움 빠르기 차례표, 다른 손실 함수를 더하라. 고치기 앞뒤의 익히기 움직임을 견주어라.

??? success "연습문제 3 풀이"
    기울기 자르기는 `optimizer.step()` 앞에 `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`을 더한다. 배움 빠르기 차례표는 `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)`을 쓰고 바퀴마다 `scheduler.step()`을 부른다. 익히기 손실 곡선, 모이는 빠르기, 마지막 모델 품질을 견주어라. 기울기 자르기는 흔히 익히기가 치솟는 것을 막고, 코사인 식히기는 뒤 바퀴에서 더 곱게 가장 좋게 하여 마지막 솜씨를 높일 수 있다.
