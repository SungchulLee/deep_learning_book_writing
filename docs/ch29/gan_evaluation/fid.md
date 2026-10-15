# 프레셰 인셉션 거리(FID)

프레셰 인셉션 거리(Fréchet Inception Distance)는 실제·생성 이미지의 고정 특징을 각각 평균과 공분산으로 요약해 비교한다. [Heusel 등의 논문](https://arxiv.org/abs/1706.08500)에서 제안되었다. 실제 기준 집합을 사용한다는 점이 인셉션 점수와 다르지만, 이미지 분포의 모든 차이를 측정하는 것은 아니다.

## 1. 정의와 최적 수송 해석

유한한 2차 모멘트를 가진 분포 P, Q의 2-Wasserstein 거리는 다음과 같다.

$$W_2(P,Q)^2=\inf_{\gamma\in\Gamma(P,Q)}\mathbb E_{(u,v)\sim\gamma}\|u-v\|^2.$$

$\Gamma(P,Q)$는 주변 분포가 P와 Q인 결합분포의 집합이다. 두 분포가 가우시안이면 이 최적 수송 비용을 평균·공분산으로 계산할 수 있다. FID는 실제·생성 **특징에 가우시안을 맞춘 뒤의 제곱 거리**다.

$$\mathrm{FID}=\|\mu_r-\mu_g\|^2+
\operatorname{Tr}\!\left(\Sigma_r+\Sigma_g-
2(\Sigma_r^{1/2}\Sigma_g\Sigma_r^{1/2})^{1/2}\right).$$

공분산 항은 Bures 거리의 **제곱**에 해당한다. 자주 쓰는 곱 형태의 trace는 적절한 행렬 제곱근 아래에서 같은 값을 준다.

$$\operatorname{Tr}(\Sigma_r^{1/2}\Sigma_g\Sigma_r^{1/2})^{1/2}
=\operatorname{Tr}(\Sigma_r\Sigma_g)^{1/2}.$$

행렬곱은 일반적으로 대칭이 아니므로 원소별 제곱근이나 Cholesky 인자로 대신할 수 없다. 공분산이 특이한 양의 준정부호 경우에도 대칭 표현으로 거리를 정의할 수 있다. FID 자체는 제곱 거리이므로 거리의 삼각부등식을 그대로 적용하지 않는다.

평균 항은 특징 중심의 차이를, 공분산 항은 퍼짐과 좌표 사이 상관 구조의 차이를 본다. 실제 특징이 가우시안이라는 보장은 없다. 중심극한정리만으로 깊은 신경망의 특징 분포 전체가 가우시안이라고 결론 내릴 수도 없다. 비가우시안 분포의 첫 두 모멘트만 비교하는 근사라는 한계를 유지한다.

## 2. 특징 추출기와 전처리

픽셀 거리는 위치 이동이나 밝기 차이에 민감하다. 사전학습 특징은 다른 수준의 구조를 반영할 수 있지만, 학습 데이터와 표현에 따른 편향을 갖는다. Inception-v3 특징이 모든 도메인에서 사람의 판단을 대변하는 것은 아니다.

```
InceptionV3 구조:
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

위 구조도는 분류기 직전의 2,048차원 전역 평균 풀링 특징을 가리킨다. 아래 구현은 torchvision Inception-v3의 fc를 Identity로 바꾸어 이 특징을 얻는다. 가중치, 전처리와 resize 구현이 다른 FID 패키지의 값과 자동으로 일치하지는 않는다.

## 3. 이미지 특징을 이용한 구현

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

입력은 0~1 범위의 N×C×H×W 부동소수점 텐서를 가정한다. 흑백은 세 채널로 복제하고 bilinear 보간으로 299×299로 맞춘 뒤 코드에 지정된 ImageNet 평균·표준편차로 정규화한다. tanh 출력은 사전에 범위를 변환해야 한다. 빈 배치나 지원하지 않는 채널 수는 검증하지 않는다.

eval과 no_grad는 각각 추론 모드와 기울기 기록을 제어한다. 추출한 특징은 CPU NumPy 배열로 모으며, 표본 공분산에는 N−1 분모를 사용한다. 최소 두 표본과 유한한 특징값이 필요하다. N≤d이면 공분산 계수는 최대 N−1이라 특이하지만, N>d라는 조건만으로 충분한 추정 정확도를 얻지는 못한다.

`calculate_frechet_distance`는 제곱근 결과가 유한하지 않을 때 대각선 ε 보정을 시도하고, 허수부가 작으면 실수부를 사용한다. 이 구현은 보정 후의 유한성을 다시 검사하거나 모든 공분산의 대칭·준정부호성을 확인하지는 않는다. 반환값도 0 이상으로 제한하지 않는다. 작은 음수는 수치 오차일 수 있지만 큰 음수·허수·비유한 값은 원인을 확인해야 한다. 라이브러리 버전과 보정 관례도 기록한다.

## 4. 가우시안 난수 예제의 해석

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

이 부분은 실제 이미지를 평가하지 않고 5,000×2,048 난수 특징을 사용한다. 같은 표준 정규분포에서 독립 생성한 두 배치의 모집단 FID는 0이지만 표본 평균·공분산은 다르다. 보존된 출력에서도 기준값이 약 420으로 나타나므로 'small non-zero'를 일반적인 작은 오차라는 뜻으로 읽으면 안 된다.

각 좌표의 평균을 0.5 옮긴 모집단의 FID는 512이며, 표준편차를 1에서 0.5로 줄인 모집단의 FID도 512다. 표본값은 모멘트 추정 오차를 포함한다. 단봉 가우시안의 분산 축소를 '모드 붕괴'라고 표시했지만 실제 이미지의 여러 모드 누락을 확인한 실험은 아니다.

## 5. 표본 수와 불확실성

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

같은 분포의 독립 배치에서도 유한 표본 FID는 일반적으로 양수다. 표본 수에 따른 편향과 변동이 있으며, 개별 실험값이 표본 수에 따라 반드시 단조롭게 변하는 것은 아니다. [유한 표본 편향 연구](https://openaccess.thecvf.com/content_CVPR_2020/html/Chong_Effectively_Unbiased_FID_and_Inception_Score_and_Where_to_Find_CVPR_2020_paper.html)처럼 모델마다 편향이 다를 수 있다는 점도 고려한다. 표본을 특징 차원의 일정 배수만큼 쓰면 모든 평가가 정확하다는 법칙은 없다.

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

위 부트스트랩은 주어진 특징 집합에서 양쪽을 복원추출하고, 반복 점수의 평균과 2.5·97.5 백분위수를 반환한다. 반환 평균은 원래 전체 표본에서 계산한 FID와 다르다. 기본 sample_size는 두 집합 크기의 최솟값이므로 원래 집합 크기가 다르면 재표집 크기도 바뀐다. 이 구간이 모집단 FID에 대해 정확한 95% 포함확률을 갖는다는 보장은 없으며, 고차원·특이 공분산과 편향의 영향을 검토해야 한다. 특징 추출기나 데이터 선택의 불확실성까지 포함하지도 않는다.

## 6. 지표의 범위와 특징 편향

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

위 함수는 다른 특징 추출기의 예를 출력할 뿐 실제 비교를 하지 않는다. CLIP이나 자기지도 특징으로 바꾸면 별도 지표가 되며, 어떤 표현이 항상 더 우수하다고 단정할 수 없다. 도메인과 평가 목적에 맞는 검증이 필요하다.

```python
# 나쁨: 한결같지 않은 미리 다듬기
real_images = resize(real_images, 299)  # 쌍선형
gen_images = resize(gen_images, 299)    # 다른 방법이다!

# 좋음: 한결같은 미리 다듬기
def consistent_preprocess(images):
    return F.interpolate(images, size=(299, 299), 
                        mode='bilinear', align_corners=False)
```

전처리의 일관성은 필수지만 위 resize 조각만으로 전체 전처리가 정의되지는 않는다. 채널, 범위, 정규화, 보간·안티앨리어싱과 압축 방식도 영향을 준다. [resize와 압축의 영향을 분석한 연구](https://arxiv.org/abs/2104.11222)에서 보듯 같은 특징 모델 이름만으로 평가가 일치하지 않는다.

## 7. 재현 가능한 평가와 통계 캐시

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

패키지 예시는 설치, 실제 이미지 경로와 장치 지원이 필요하다. 실행한 결과를 제시한 것이 아니며 CUDA 사용 가능 여부와 패키지의 평가 설정을 확인해야 한다.

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

기준 집합과 특징·전처리가 같으면 실제 평균·공분산을 재사용할 수 있다. 위 저장 함수는 수치만 저장하므로 기준 데이터, 가중치 식별자, 전처리와 표본 수 등의 메타데이터는 별도로 보관해야 한다. 예시의 generator.sample과 반복 변수는 외부 정의를 전제로 한다.

```python
def report_fid(fid: float, n_real: int, n_gen: int):
    """맥락과 함께 FID를 제대로 알린다."""
    print(f"FID: {fid:.2f}")
    print(f"  Real samples: {n_real:,}")
    print(f"  Generated samples: {n_gen:,}")
    print(f"  Feature extractor: InceptionV3 (ImageNet)")
    print(f"  Preprocessing: 299×299, bilinear, ImageNet normalization")
```

보고에는 양쪽 표본 수, 기준 집합·분할, 특징 모델·층·가중치, 전처리, dtype·구현 버전, 정규화 보정과 생성 조건을 포함한다. 임의의 'FID<5면 뛰어남' 같은 구간이나 조건 없는 최신 성능 표는 사용하지 않는다. 서로 다른 특징 눈금에서 절대값이 같아도 같은 품질을 뜻하지 않는다.

## 8. FID와 다른 평가의 역할

IS는 실제 기준 집합 없이 분류 예측을 요약하고, FID는 기준 특징의 첫 두 모멘트와 비교한다. FID가 0이어도 원래 특징 분포나 이미지 분포가 동일하다는 뜻은 아니다. 품질 저하와 다양성 부족이 같은 FID를 만들 수 있으며 암기를 직접 검출하지도 않는다.

생성 정밀도·재현율, 표본 검사와 최근접 이웃 분석, 목적에 맞는 다운스트림 성능을 함께 고려한다. LPIPS 같은 이미지 쌍의 지각적 거리를 대응 관계 없는 단일 이미지 품질 점수로 사용하지 않는다. 모델 선택에는 검증 기준을 사용하고 최종 테스트 결과와 구분한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
FID가 실제 데이터와 비교한다는 뜻과 비교하지 못하는 것을 설명하라.

</div>

??? success "연습문제 1 풀이"

    고정 특징 공간에서 실제·생성 집합의 평균·공분산을 비교한다. IS와 달리 실제 기준 집합이 필요하다. 그러나 두 모멘트만 같아도 FID가 0이 될 수 있어 전체 분포의 일치나 모든 모드의 포괄성을 보장하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
IS가 클래스 내부 반복을 놓치는 예와 FID의 보완 범위를 설명하라.

</div>

??? success "연습문제 2 풀이"

    클래스별 대표 이미지 하나씩을 균등 반복하면 확신에 찬 분류 예측으로 높은 IS를 얻을 수 있다. FID는 그런 반복이 특징의 모멘트를 바꾸면 반응할 수 있다. 모멘트가 유지되면 놓칠 수 있으므로 FID가 모든 반복을 검출한다고 할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
생성 정밀도·재현율이 사용하는 이웃 구조는 참 지지집합인가?

</div>

??? success "연습문제 3 풀이"

    아니다. 한 방식에서는 특징의 k번째 최근접 이웃 거리를 반지름으로 하는 구들의 합집합으로 유한 표본의 영역을 근사한다. 참 분포의 지지집합을 정확히 아는 것은 아니다. 특징, 표본 수와 k에 따라 값이 달라진다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
여러 평가 지표를 함께 볼 때 어떤 점을 주의해야 하는가?

</div>

??? success "연습문제 4 풀이"

    각 지표가 보는 대상과 실패 사례를 구분한다. FID의 모멘트 거리, IS의 분류 확률, 정밀도·재현율의 이웃 근사, LPIPS의 이미지 쌍 거리는 서로 다르다. 여러 수치를 나열해도 데이터 누출이나 선별 편향이 해결되는 것은 아니며 보고 목적에 맞는 절차가 필요하다.



---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
FID의 평균 항과 공분산 항을 설명하라.

</div>

??? success "연습문제 5 풀이"

    평균 항은 $\|\mu_r-\mu_g\|^2$이며 공분산 항은 $\operatorname{Tr}(\Sigma_r+\Sigma_g-2(\Sigma_r^{1/2}\Sigma_g\Sigma_r^{1/2})^{1/2})$다. 두 가우시안의 중심과 퍼짐 차이를 반영한다. 생성 다양성이 줄면 공분산이 달라질 수 있지만 모든 다양성 손실이 이 항으로 검출되는 것은 아니다.



---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
같은 분포의 독립 표본끼리 잰 FID가 양수일 수 있는 이유를 설명하라.

</div>

??? success "연습문제 6 풀이"

    표본 평균·공분산이 서로 달라지기 때문이다. 동일 모집단에서의 이론값 0과 유한 표본의 점수를 구분한다. 독립 반복으로 편향과 변동을 조사할 수 있으나 실제 데이터끼리 얻은 한 값은 절대적인 바닥이 아니다. 이 페이지의 코드는 MNIST 표본 수별 81.4~1.8 표를 재현하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff hard" title="어려움"></span>
실제 데이터를 순서대로 나눈 두 집합과 무작위로 나눈 두 집합의 FID가 다를 수 있는 이유는 무엇인가?

</div>

??? success "연습문제 7 풀이"

    순서가 클래스, 출처나 시점과 연관되어 있으면 앞뒤 집합의 특징 분포가 달라질 수 있다. 동질한 표본 비교가 목적이면 무작위 분할이 적절할 수 있지만 시간 일반화나 그룹 일반화에는 그 구조를 보존하는 분할이 필요하다. 특정 MNIST 수치나 출처 차이를 이 페이지의 가우시안 코드로 입증할 수는 없다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
아래 행렬 제곱근 조각에서 허수부를 처리하는 순서와 보정의 한계를 설명하라.

</div>

??? success "연습문제 8 풀이"

    첫 조각은 복소수 결과의 실수부를 곧바로 취한다. 그 뒤 두 번째 조각을 실행하면 이미 허수부가 사라져 원래 오차를 검사할 수 없다. 허수 크기와 유한성은 실수부를 취하기 전에 확인해야 한다. 대각선 ε 보정은 수치 안정화를 돕지만 모멘트 추정 오차를 없애지 않는다. 사용 라이브러리의 sqrtm 인터페이스도 확인한다.

    ```python
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    ```
    ```python
    assert np.abs(covmean.imag).max() < 1e-3
    ```
    ```python
    sigma = sigma + np.eye(d) * 1e-6
    ```


---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
FID의 가우시안 근사가 놓치는 것과 KID의 의미를 설명하라.

</div>

??? success "연습문제 9 풀이"

    첫 두 모멘트보다 높은 차이와 비가우시안 구조를 놓칠 수 있다. KID는 선택한 특징과 커널에 대한 MMD²의 비편향 U-통계량 추정을 사용하지만, 특징·커널 선택과 표본 변동의 영향을 받는다. 일반적인 다항 커널이 모든 분포 차이를 구별한다고 보장할 수 없고, 비편향 추정량은 유한 표본에서 음수가 될 수도 있다. 가우시안 가정이 없다는 것이 가정이나 한계가 전혀 없다는 뜻은 아니다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
높은 FID 하나로 품질 저하와 다양성 부족을 구별할 수 있는가?

</div>

??? success "연습문제 10 풀이"

    없다. 여러 다른 평균·공분산 차이가 같은 합을 만들 수 있다. 특징 통계의 변화, 생성 정밀도·재현율, 샘플과 최근접 이웃을 함께 조사한다. 재현율이 낮다는 관찰만으로 특정 학습 기법을 적용하면 해결된다고 보장할 수는 없다.



---

<div class="drillbox" markdown>

**연습문제 11.** <span class="diff med" title="중간"></span>
FID로 체크포인트를 선택할 때 검증 집합과 최종 테스트를 어떻게 구분하는가?

</div>

??? success "연습문제 11 풀이"

    고정한 검증 기준, 표본 수와 평가 절차로 체크포인트를 선택한다. 선택 과정에 사용하지 않은 테스트 데이터로 최종 평가하고 생성 조건도 기록한다. 반복 선택으로 지표나 검증 데이터에 맞출 수 있으므로 독립 지표와 표본 검사도 고려한다.



---

<div class="drillbox" markdown>

**연습문제 12.** <span class="diff easy" title="쉬움"></span>
FID 보고에 필요한 정보를 적어라.

</div>

??? success "연습문제 12 풀이"

    기준 데이터와 분할, 양쪽 표본 수, 특징 모델·가중치·층·차원, 전처리·압축, 구현 버전·정밀도와 수치 보정, 생성 조건·선별 및 반복 변동을 기록한다. 실제 데이터끼리 잰 값은 참고 자료로 제시할 수 있지만 바닥이나 완벽함의 기준으로 단정하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 13.** <span class="diff hard" title="어려움"></span>
학습 데이터를 반복 생성하는 모델을 FID만으로 탐지할 수 있는가?

</div>

??? success "연습문제 13 풀이"

    보장할 수 없다. 기준 집합을 그대로 재생하면 그 집합과 모멘트가 같아 0에 가까울 수 있다. 독립 테스트 기준에서도 차이가 작게 나올 수 있으므로 학습·테스트 FID 차이만으로 암기를 확정하거나 배제하지 않는다. 최근접 이웃, 중복 검사와 목적에 맞는 일반화 평가를 함께 사용한다.



---

<div class="drillbox" markdown>

**연습문제 14.** <span class="diff med" title="중간"></span>
특징 추출기나 특징 스케일이 바뀌면 FID가 어떻게 달라지는가?

</div>

??? success "연습문제 14 풀이"

    좌표를 공통으로 a배 하면 FID는 a²배가 된다. 특징 모델·층을 바꾸면 보존하는 정보와 눈금이 달라져 절대값을 같은 지표로 비교할 수 없다. 도메인 특징을 사용할 수 있으나 별도 지표임을 명시한다. 높은 차원은 추정 부담을 높일 수 있지만 그 자체로 편향의 크기나 품질 민감도를 완전히 정하지 않는다.



## 정리하며

FID는 고정 특징의 가우시안 근사 사이의 제곱 2-Wasserstein 거리다. 낮은 값은 첫 두 모멘트의 가까움을 뜻하며 전체 분포의 일치나 비암기를 보장하지 않는다. 표본 수·전처리·특징·수치 계산을 명시하고 편향과 변동을 함께 평가한다.

**참고 문헌**

1. Heusel, M., et al. (2017). "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium." *NeurIPS*.

2. Parmar, G., et al. (2021). "On Aliased Resizing and Surprising Subtleties in GAN Evaluation." *CVPR*.

3. Bińkowski, M., et al. (2018). "Demystifying MMD GANs." *ICLR*.

4. Chong, M. J., & Forsyth, D. (2020). "Effectively Unbiased FID and Inception Score and Where to Find Them." *CVPR*.
