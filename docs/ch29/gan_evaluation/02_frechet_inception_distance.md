# 프레셰 인셉션 거리 계산

프레셰 인셉션 거리(Fréchet Inception Distance, FID)는 실제 이미지와 생성 이미지의 피처 평균·공분산을 비교한다. 이 페이지는 미리 주어진 통계량과 임의의 가우시안 피처로 계산 원리를 보여 준다. 이름과 달리 아래 실행 예제는 실제 Inception-v3나 학습된 GAN을 사용하지 않는다.

## 1. 통계량을 이용한 계산

```python
"""
52단원: 프레셰 인셉션 거리(FID)
==========================================

가장 널리 쓰이는 잣대 가운데 하나인 FID를 두루 갖추어 짠다
만들어 내는 모델을 따지는 데 쓴다.

학습 목표:
-------------------
1. FID의 수학 바탕을 이해한다
2. FID를 바닥부터 짠다
3. 피처 뽑기에 미리 익힌 InceptionV3을 쓴다
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
    FID는 여러 변수 정규 분포 둘 사이의 거리를 잰다.
    
    참 분포: X_real ~ N(μ_r, Σ_r)
    만들어 낸 분포: X_gen ~ N(μ_g, Σ_g)
    
    FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^{1/2})
    
    왜 프레셰 거리인가?
    --------------------
    1. 프레셰 거리(바서슈타인-2 거리라고도 한다)는
       정규 분포의 가장 좋은 나르기 거리
    2. 정규 분포에는 닫힌 꼴 풀이가 있다
    3. 평균과 함께 흩어짐의 차이에 모두 민감하다
    4. FID가 낮을수록 두 분포가 더 닮았다
    
    왜 인셉션 피처인가?
    ----------------------
    1. ImageNet으로 익힌 InceptionV3은 그림의 뜻 피처를 담는다
    2. pool3 피처(2048차원)은 그림의 높은 레이어 내용을 나타낸다
    3. 화소 수준 견줌보다 튼튼하다
    4. 사람의 판단과 잘 이어진다
    
    흔한 FID 값:
    ------------------
    - FID < 10: 아주 좋음(거의 나무랄 데 없이 만들어 냄)
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
        X ~ N(μ₁, Σ₁)와 Y ~ N(μ₂, Σ₂)에 대해 프레셰 거리는 다음과 같다.
        
        d²_F(X,Y) = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2(Σ₁Σ₂)^{1/2})
        
        공식을 나누어 보면:
        1. ||μ₁ - μ₂||²: 평균의 차이(첫째 적률)
        2. Tr(Σ₁ + Σ₂): Sum of variances
        3. -2Tr((Σ₁Σ₂)^{1/2}): 공분산이 겹치는 마디
        
        인수:
            mu1: 첫째 분포의 평균 [d]
            sigma1: 첫째 분포의 공분산 행렬 [d, d]
            mu2: 둘째 분포의 평균 [d]
            sigma2: 둘째 분포의 공분산 행렬 [d, d]
            eps: 수치 안정성을 위한 작은 상수
        
        반환값:
            프레셰 거리(스칼라)
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
        # 이것이 셈이 가장 비싼 스텝이다
        
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
        피처의 평균과 함께 흩어짐을 셈한다.
        
        인수:
            features: 피처 벡터 [n_samples, feature_dim]
        
        반환값:
            (평균 [feature_dim], 공분산 [feature_dim, feature_dim]) 튜플
        
        수학의 참고:
        ------------------
        Mean: μ = (1/N) Σ x_i
        Covariance: Σ = (1/N) Σ (x_i - μ)(x_i - μ)ᵀ
        
        열마다 하나의 변수로 보려고 rowvar=False을 쓴다
        """
        # 평균을 셈한다: 표본에 걸쳐 평균 낸다
        # 꼴: [피처 차원]
        mu = np.mean(features, axis=0)
        
        # 함께 흩어짐 행렬을 셈한다
        # 꼴: [피처 차원, 피처 차원]
        # rowvar=False: 세로줄마다 변수, 가로줄마다 관측이다
        sigma = np.cov(features, rowvar=False)
        
        return mu, sigma
    
    @staticmethod
    def calculate_fid(real_features: np.ndarray,
                     generated_features: np.ndarray) -> float:
        """
        실제 피처와 만든 피처로 FID를 셈한다.
        
        온전한 물길:
        -----------------
        1. InceptionV3에서 피처를 뽑는다(이 함수에 앞서 한다)
        2. 참 자료의 통계(μ, Σ)를 셈한다
        3. 만들어 낸 자료의 통계(μ, Σ)를 셈한다
        4. 프레셰 거리를 셈한다
        
        인수:
            real_features: 참 그림에서 뽑은 피처 [n_real, 2048]
            generated_features: 만들어 낸 그림에서 뽑은 피처 [n_gen, 2048]
        
        반환값:
            FID 점수(낮을수록 좋다)
        
        가장 작은 표본 크기:
        --------------------
        - 가장 적어도: 표본 2048개(= 피처 차원)
        - 권함: 안정된 어림을 위해 표본 10,000개 이상
        - 표본이 많을수록 FID가 미덥다
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
    
    이 클래스는 실제 InceptionV3 무게 없이
    핵심 개념을 보인다.
    """
    
    def __init__(self, feature_dim: int = 2048):
        """
        흉내 피처 뽑개로 첫자리매김한다.
        
        인수:
            feature_dim: 피처 벡터의 차원(InceptionV3이면 2048)
        """
        self.feature_dim = feature_dim
        print(f"Initialized mock InceptionV3 with {feature_dim}-dim features")
        print("NOTE: For real FID, use actual InceptionV3 pretrained on ImageNet")
    
    def extract_features(self, images: torch.Tensor) -> np.ndarray:
        """
        그림에서 피처를 뽑는다(흉내만 낸 짜기).
        
        실제 짜기에서는:
        1. 그림을 InceptionV3 꼴로 미리 다듬는다(299×299, 잣대 맞춤)
        2. InceptionV3을 지나는 피드포워드
        3. pool3 피처를 뽑는다(2048차원)
        4. 그래디언트가 필요 없다(따짐 결)
        
        인수:
            images: 그림 배치 [batch_size, C, H, W]
        
        반환값:
            피처 [batch_size, feature_dim]
        """
        batch_size = images.shape[0]
        
        # 흉내 피처(실제로는 InceptionV3에서 온다)
        # 그럴듯한 통계 성질을 지닌 피처를 만든다
        features = torch.randn(batch_size, self.feature_dim)
        
        # 그림에 따라 달라지는 몫을 더한다(실제 피처 뽑기를 흉내 낸다)
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
    
    # 상황 1: 똑같은 분포(FID가 ~0이어야 한다)
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
    피처 벡터로 FID 셈하기를 보인다.
    """
    print("\n" + "=" * 70)
    print("FID with Feature Vectors")
    print("=" * 70)
    
    # 인공 피처 벡터를 만든다
    n_samples = 5000
    feature_dim = 2048
    
    print(f"\nGenerating {n_samples} samples with {feature_dim} features...")
    
    # 실제 분포: N(0, I)
    real_features = np.random.randn(n_samples, feature_dim)
    
    # 만든 분포 1: 똑같다(FID가 낮아야 한다)
    gen1_features = np.random.randn(n_samples, feature_dim)
    
    # 만든 분포 2: 평균이 옮겨졌다
    gen2_features = np.random.randn(n_samples, feature_dim) + 0.5
    
    # 만든 분포 3: 흩어짐이 줄었다(봉우리 무너짐 표시)
    gen3_features = np.random.randn(n_samples, feature_dim) * 0.5
    
    # FID를 셈한다
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
    
    # 피처 벡터로 보여 준다
    demonstrate_fid_with_features()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. FID 뜻매김:
       - 실제 분포와 만든 분포 사이의 거리를 잰다
       - 피처 공간에서 정규 분포를 가정한다
       - FID가 낮을수록 만들어 내는 품질이 좋다
    
    2. 수학의 몫:
       - 평균 차이: ||μ_r - μ_g||²
       - Covariance term: Tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^{1/2})
       - 정규 분포의 닫힌 꼴 풀이
    
    3. 왜 InceptionV3인가?
       - 그림의 뜻 피처를 담는다
       - ImageNet으로 미리 익혔다
       - 2048차원 pool3 피처
       - 화소 공간 견줌보다 낫다
    
    4. 표본 크기가 중요하다:
       - 가장 적어도: 표본 2048개(= 피처 차원)
       - 권함: 표본 10,000개 이상
       - 표본이 많을수록 FID 어림이 든든하다
    
    5. 한계:
       - 가우스 분포라고 가정한다(맞지 않을 수 있다)
       - 피처 뽑개를 무엇으로 고르느냐에 치우친다
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
    main()
```

??? note "전체 출력 (115줄)"

    ```

    ======================================================================
    MODULE 52: FRÉCHET INCEPTION DISTANCE (FID)
    ======================================================================
    ======================================================================
    Fréchet Inception Distance (FID) Demonstration
    ======================================================================

    Scenario 1: Identical Distributions
    ----------------------------------------------------------------------
    Mean 1: [0. 0.]
    Mean 2: [0. 0.]
    FID: 0.000000
    Interpretation: FID ≈ 0 indicates identical distributions

    Scenario 2: Different Means (Same Covariance)
    ----------------------------------------------------------------------
    Mean 1: [0. 0.]
    Mean 2: [3. 3.]
    Mean difference norm: 4.2426
    FID: 18.0000
    Interpretation: FID increases with mean difference

    Scenario 3: Different Covariances (Same Mean)
    ----------------------------------------------------------------------
    Covariance 1:
    [[1. 0.]
     [0. 1.]]
    Covariance 2:
    [[4. 0.]
     [0. 4.]]
    FID: 2.0000
    Interpretation: FID sensitive to variance differences

    Scenario 4: Both Mean and Covariance Different
    ----------------------------------------------------------------------
    Mean difference: 2.8284
    FID: 9.0961
    Interpretation: FID captures both mean and covariance differences

    ======================================================================
    FID Comparison Summary:
    ======================================================================
    Identical distributions:     FID = 0.0000
    Different means:             FID = 18.0000
    Different covariances:       FID = 2.0000
    Both different:              FID = 9.0961

    Key Insight: FID increases as distributions become more different

    ======================================================================
    FID with Feature Vectors
    ======================================================================

    Generating 5000 samples with 2048 features...

    ----------------------------------------------------------------------
    FID Comparison:
    ----------------------------------------------------------------------
    Computing FID with 5000 real and 5000 generated samples...
    ✓ FID computed: 420.0195
    Computing FID with 5000 real and 5000 generated samples...
    ✓ FID computed: 932.6412
    Computing FID with 5000 real and 5000 generated samples...
    ✓ FID computed: 722.7666

    Generator 1 (similar):        FID = 420.02
    Generator 2 (shifted):        FID = 932.64
    Generator 3 (mode collapse):  FID = 722.77

    ----------------------------------------------------------------------
    Interpretation:
    ----------------------------------------------------------------------
    • Lower FID = Better match to real distribution
    • FID sensitive to both mean shifts and variance changes
    • Reduced variance (mode collapse) increases FID

    ======================================================================
    KEY TAKEAWAYS
    ======================================================================

        1. FID 뜻매김:
           - 실제 분포와 만든 분포 사이의 거리를 잰다
           - 피처 공간에서 정규 분포를 가정한다
           - FID가 낮을수록 만들어 내는 품질이 좋다
        
        2. 수학의 몫:
           - 평균 차이: ||μ_r - μ_g||²
           - Covariance term: Tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^{1/2})
           - 정규 분포의 닫힌 꼴 풀이
        
        3. 왜 InceptionV3인가?
           - 그림의 뜻 피처를 담는다
           - ImageNet으로 미리 익혔다
           - 2048차원 pool3 피처
           - 화소 공간 견줌보다 낫다
        
        4. 표본 크기가 중요하다:
           - 가장 적어도: 표본 2048개(= 피처 차원)
           - 권함: 표본 10,000개 이상
           - 표본이 많을수록 FID 어림이 든든하다
        
        5. 한계:
           - 가우스 분포라고 가정한다(맞지 않을 수 있다)
           - 피처 뽑개를 무엇으로 고르느냐에 치우친다
           - 모든 잘못됨을 알아내지는 못한다
           - 다른 잣대와 아울러야 한다
        
        6. 흔한 값:
           - FID < 10: 뛰어난 품질
           - FID 10-50: 좋은 품질
           - FID 50-100: 보통 품질
           - FID > 100: 나쁜 품질
        
    ======================================================================
    ```

## 2. 평균과 공분산의 거리

실제·생성 피처의 평균과 공분산을 각각 $(\mu_r,\Sigma_r)$, $(\mu_g,\Sigma_g)$라 두면 대칭 양의 준정부호 표현은 다음과 같다.

$$\mathrm{FID}=\|\mu_r-\mu_g\|^2+
\operatorname{Tr}\!\left[\Sigma_r+\Sigma_g-
2\left(\Sigma_r^{1/2}\Sigma_g\Sigma_r^{1/2}\right)^{1/2}\right].$$

이는 두 가우시안 사이의 **제곱 2-Wasserstein 거리**다. 코드에서는 같은 trace를 주는 공분산 곱의 행렬 제곱근을 사용한다. 원소별 제곱근이나 일반적인 Cholesky 인자로 대체할 수 없다. 서로 다른 비가우시안 피처 분포도 첫 두 모멘트가 같으면 모집단 FID가 0이므로, 출력의 '분포가 동일하다'는 설명은 가우시안 근사 범위에서 읽어야 한다.

2차원 예제에서 공분산이 I이고 평균이 (0,0)에서 (3,3)으로 이동하면 평균 항은 18이다. 평균이 같고 공분산이 I에서 4I로 바뀌면 표준편차 차이의 제곱합은 2다. 이 값들은 식을 확인하는 해석적 사례이며 실제 이미지 품질 결과가 아니다.

## 3. 유한 표본 피처의 통계

`compute_statistics`는 N×d 행렬의 열별 평균과 표본 공분산을 구한다. 공분산에는 기본적으로 N−1 분모를 사용하므로 적어도 두 표본이 필요하다. 유한 표본의 공분산 계수는 최대 N−1이며 N≤d이면 특이하다. 특이 공분산도 수학적으로 FID를 정의할 수 있지만 추정과 수치 계산에는 주의해야 한다. N>d라는 조건만으로 충분한 정확도가 보장되는 것도 아니다.

5,000×2,048 피처 예제는 가우시안 난수다. 첫 번째 생성 피처는 실제 피처와 **같은 모집단에서 독립 생성한 다른 표본**이므로 표본 FID가 0일 필요가 없다. 평균을 0.5 이동한 모집단의 FID는 2048×0.5²=512이며, 표준편차를 0.5로 줄인 모집단의 FID도 2048×(1−0.5)²=512다. 코드의 표본값은 모멘트 추정 오차 때문에 이 값들과 다를 수 있다. 분산을 줄인 단봉 가우시안을 '모드 붕괴'라고 부르는 것은 여기서는 비유이며 여러 실제 모드의 누락 실험은 아니다.

유한 표본 FID에는 편향과 변동이 있고 모델에 따라 바이어스도 다를 수 있다. 표본 수가 커질수록 개별 결과가 반드시 단조롭게 감소하는 것은 아니다. [유한 표본 바이어스 연구](https://openaccess.thecvf.com/content_CVPR_2020/html/Chong_Effectively_Unbiased_FID_and_Inception_Score_and_Where_to_Find_CVPR_2020_paper.html)를 참고하고, 비교 시 실제·생성 표본 수를 각각 고정해 보고한다.

## 4. 수치 안정성과 mock 피처 추출기

코드는 공분산 곱의 제곱근이 유한하지 않을 때 대각선 ε 보정을 시도한다. 보정 후에도 유한한지 재확인하지 않으며, 복소수인 경우 허수부의 **대각선만** 검사한 뒤 실수부를 사용한다. 모든 허수 성분이나 공분산의 유효성을 검증하는 구현은 아니다. 반환값도 0 이상으로 자르지 않으므로 반올림 오차로 작은 음수가 나올 수 있다. 큰 음수·허수·비유한 값은 원인 확인이 필요하다. 사용 라이브러리 버전에 따라 `sqrtm`의 호출 방식도 확인해야 한다.

`SimpleInceptionV3Wrapper`는 학습된 피처 추출기가 아니라 난수를 만드는 mock 객체이며 main에서는 호출하지 않는다. 더욱이 N×d 난수에 N×1×1×1 이미지 평균을 더하면 브로드캐스팅 결과가 **N×1×N×d**가 되어 의도한 N×d 피처가 아니다. CUDA 이미지에서는 CPU 난수와의 장치 불일치도 생긴다. 따라서 이 메서드를 그대로 실제 이미지 FID 추출기로 설명할 수 없다.

## 5. 실제 평가와 보고

표준적인 Inception 기반 평가에서는 고정된 사전학습 피처와 정해진 전처리·구현을 사용한다. 다른 분류기의 은닉 레이어를 사용하면 유용한 도메인 지표가 될 수 있지만 동일한 표준 FID 수치로 비교하지 않는다. 피처 차원을 줄이거나 임의로 스케일을 바꾸어도 값이 달라진다.

평가 조건에는 기준 집합, 피처 모델·체크포인트, 전처리, 표본 수, 정밀도, 공분산 및 행렬 제곱근 처리와 샘플 선택을 포함한다. 실제 데이터를 무작위로 나누어 얻는 값은 유한 표본 비교의 참고값이며 절대적인 FID 바닥이 아니다. 이 페이지에는 MNIST 1.8 등의 기준값을 측정하는 실험이 없다. 'FID<10이면 우수' 같은 구간도 모든 데이터셋에 적용할 수 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
피처 입력 N×d에서 평균과 공분산의 모양은 무엇인가?

</div>

??? success "풀이"

    평균은 d, 공분산은 d×d다. 행은 독립 평가 표본, 열은 피처 좌표다. 두 집합의 표본 수는 달라도 피처 정의와 차원은 같아야 한다. 이미지 텐서를 바로 공분산 함수에 전달하는 것과 피처 추출은 다르다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
2차원에서 평균이 (0,0)과 (3,3), 두 공분산이 I일 때 FID를 구하라.

</div>

??? success "풀이"

    공분산 항은 0이고 평균 차이의 제곱합이 3²+3²=18이다. 좌표마다 c만큼 평행 이동한 d차원 피처에서는 dc²다. 스케일을 바꾸면 거리도 달라지므로 다른 피처 추출기의 절대값을 직접 비교하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
mock 피처 추출기의 브로드캐스팅 문제를 설명하라.

</div>

??? success "풀이"

    난수 피처는 N×d이지만 keepdim=True로 계산한 이미지 평균은 N×1×1×1이다. 차원을 오른쪽 정렬해 더하면 N×1×N×d로 확장되어 표본끼리 섞인 축이 생긴다. 의도한 N×d가 아니며 CPU 난수와 CUDA 이미지 평균도 함께 더할 수 없다. 이 클래스는 실제 Inception 피처를 제공하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
FID에 사용할 피처 레이어와 분류 확률의 차이는 무엇인가?

</div>

??? success "풀이"

    일반적인 Inception 기반 FID는 분류 확률 대신 선택한 은닉 피처를 사용한다. 아래 feat는 별도 CNN을 가정한 조각이며 이 페이지의 main에 연결된 128차원 학습 분류기가 아니다. 은닉 피처가 클래스 내부 차이를 보존할 수 있어도 모든 의미나 다양성을 보존한다는 보장은 없다. 레이어와 차원을 명시한다.

    ```python
    def feat(self, x):
        x = self.p(torch.relu(self.c1(x)))
        x = self.p(torch.relu(self.c2(x)))
        return torch.relu(self.f1(x.flatten(1)))     # 128차원. 여기를 쓴다
    ```


---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
실제 기준 통계를 캐시할 때 무엇을 함께 저장해야 하는가?

</div>

??? success "풀이"

    평균·공분산뿐 아니라 기준 데이터, 피처 모델과 웨이트, 전처리, 표본 수, dtype과 계산 관례를 기록한다. 아래 함수는 DEV와 net.feat 등의 외부 정의와 평가 모드를 전제로 한다. 2048차원 공분산의 float64 저장량은 약 32 MiB이며 전체 피처를 저장하는 것보다 작을 수 있다. 캐시 조건이 바뀌면 다시 계산해야 한다.

    ```python
    @torch.no_grad()
    def stats_of(images, net, bs=1000):
        fs = []
        for i in range(0, len(images), bs):
            fs.append(net.feat(images[i:i+bs].to(DEV)).cpu())
        f = torch.cat(fs).numpy().astype(np.float64)
        return f.mean(0), np.cov(f, rowvar=False)

    mu_r, sigma_r = stats_of(real_images, net)     # 한 번만
    np.savez('fid_ref.npz', mu=mu_r, sigma=sigma_r)
    ```


---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
같은 피처 배치와 평행 이동한 배치로 FID 식을 어떻게 확인하는가?

</div>

??? success "풀이"

    같은 배치를 두 번 쓰면 이론적으로 0이며 수치 오차 허용 범위에서 확인한다. 모든 좌표에 0.5를 더하면 공분산은 같아 이론값은 0.25d다. 아래 fid와 features_of는 외부 정의가 필요하다. 서로 다른 실제 표본 절반 사이의 값은 절대적인 바닥이 아니며 잡음 이미지의 값도 반드시 크다고 보장되지 않는다.

    ```python
    f = features_of(real)
    assert abs(fid(f, f)) < 1e-6

    shifted = f + 0.5
    assert abs(fid(f, shifted) - 0.25 * f.shape[1]) < 1e-3
    ```


---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
실제와 생성 피처의 표본 수가 같아야만 FID를 정의할 수 있는가?

</div>

??? success "풀이"

    아니다. 각각의 평균·공분산을 구할 수 있으면 다른 표본 수로도 계산한다. 다만 편향과 변동이 양쪽 표본 수에 영향을 받으므로 모델 비교에는 각 집합의 크기를 고정해 보고하는 것이 좋다. 표본이 적은 개별 결과가 언제나 더 크다는 단조성은 없다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
N≤d일 때 특이 공분산이 되는 이유와 대각선 ε 보정의 한계를 설명하라.

</div>

??? success "풀이"

    중심화한 N개 표본의 계수는 최대 N−1이어서 d차원 공분산의 계수도 그 이하이다. εI를 더하면 수치 계산에 도움이 될 수 있으나 추정 정확도를 회복시키거나 표본 바이어스를 제거하지는 않는다. 아래 보정 조각은 사용 위치에 따라 목적량을 바꾸며, 보정 후 유한성·허수부·최종값 검사가 필요하다.

    ```python
    eps = 1e-6
    sigma_r = sigma_r + np.eye(d) * eps       # 대각선을 조금 올린다
    ```
    ```python
    cm, _ = linalg.sqrtm(sigma_r @ sigma_g, disp=False)
    if not np.isfinite(cm).all():             # 수렴 실패를 잡아낸다
        offset = np.eye(d) * eps
        cm = linalg.sqrtm((sigma_r+offset) @ (sigma_g+offset))
    ```


---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
학습 중 FID 곡선의 변동을 해석할 때 무엇을 확인해야 하는가?

</div>

??? success "풀이"

    표본 수, 실제 기준 통계, 피처 모델과 전처리, 생성 조건을 고정하고 표본 변동을 구분한다. 아래 시드는 잠재 코드뿐 아니라 관련 난수 상태 전체에 작용할 수 있으며 이후 연산 순서도 중요하다. 독립 반복을 통해 최종 결과의 변동을 볼 수 있다. GAN 로스도 발산이나 포화의 진단에 도움이 되며 FID가 유일한 학습 진단은 아니다.

    ```python
    FID_N = 2000                      # 고정
    torch.manual_seed(0)              # 뽑기 씨앗도 고정
    ```


---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
FID 계산의 주요 비용과 차원 축소의 대가를 설명하라.

</div>

??? success "풀이"

    피처 추출, 공분산 계산, 행렬 제곱근이 주요 비용이다. 일반적인 밀집 행렬 제곱근은 대략 O(d³) 연산이 든다. 실제 통계를 캐시하거나 배치 추론을 효율화할 수 있다. 피처 차원을 낮추면 비용은 줄어도 지표 자체가 바뀌므로 원래 FID와 같은 값으로 보고하지 않는다. 고정 표본 수만으로 정확도가 보장되는 것은 아니다.



## 정리하며

FID는 고정 피처의 평균·공분산을 비교하는 지표다. 이 예제의 가우시안 난수와 mock 객체는 실제 이미지 평가 결과가 아니다. 유한 표본 오차, 피처와 전처리, 행렬 제곱근의 수치 문제를 명시하고 다른 품질 평가와 함께 해석한다.
