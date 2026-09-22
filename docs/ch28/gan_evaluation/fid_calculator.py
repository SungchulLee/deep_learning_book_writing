"""fid_calculator — ch27/gan_evaluation/fid.md 의 코드를
모듈로 쓸 수 있게 옮겨 놓은 것이다.
"""

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
            batch_size: 다룰 묶음 크기
            
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
            batch_size: 특징 뽑기의 묶음 크기
            
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
        
        같은 실제 자료 묶음에 여러 만들개를 견줄 때
        이 편이 더 효율이 좋다.
        
        인수:
            mu_real: 미리 셈한 참 특징의 평균 [D]
            sigma_real: 미리 셈한 참 특징의 공분산 [D, D]
            generated_images: 만들어 낸 그림 [N, C, H, W]
            batch_size: 특징 뽑기의 묶음 크기
            
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
        batch_size: 특징 뽑기의 묶음 크기
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

def report_fid(fid: float, n_real: int, n_gen: int):
    """맥락과 함께 FID를 제대로 알린다."""
    print(f"FID: {fid:.2f}")
    print(f"  Real samples: {n_real:,}")
    print(f"  Generated samples: {n_gen:,}")
    print(f"  Feature extractor: InceptionV3 (ImageNet)")
    print(f"  Preprocessing: 299×299, bilinear, ImageNet normalization")
