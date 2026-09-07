# 단원 52.03: 표본 그려 보기와 품질 따지기

이 단원은 깊은 만들어 내는 모델의 중요한 부품인 단원 52.03: 표본 그려 보기와 품질 따지기을 짠다. 이 짜기를 이해하면 요즘 만들어 내는 모델에 쓰이는 얼개의 결과 익히기 절차를 꿰뚫어 볼 수 있다. 이 코드는 연구와 실제 얼개에서 널리 쓰이는 쓸모 있는 재주를 보인다.

## 코드

```python
"""
단원 52.03: 표본 그려 보기와 품질 따지기
=========================================================

이 단원은 만든 표본을 따지는 그려 보기 재주와
눈으로 살펴보는 기본 품질 잣대를 다룬다.

학습 목표:
-------------------
1. 만든 표본을 잘 보여 주는 그림을 만든다
2. 숨은 공간 사이 메우기를 한다
3. 되짓기 품질 잣대를 셈한다
4. 표본의 다양함을 눈으로 따진다

핵심 개념:
------------
- 표본의 격자 그림
- 숨은 공간 사이 메우기
- 되살림 어긋남 자(MSE, SSIM)
- 눈으로 품질 따지기

지은이: 가르치기 인공 지능 모둠
날짜: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import math

# ========================================================================
# 메인
# ========================================================================

# 난수 씨앗을 설정한다
torch.manual_seed(42)
np.random.seed(42)


class SampleGridVisualizer:
    """
    만든 표본의 격자 그림을 만든다.
    
    목적:
    -------
    만들어 내는 모델에서 눈으로 살펴보기가 결정적인 까닭은 이렇다.
    1. 잣대가 느낌의 품질 문제를 놓칠 수 있다
    2. 사람은 흠을 알아내는 데 뛰어나다
    3. 봉우리 무너짐과 다양함을 가려내는 데 도움이 된다
    4. 짜임새 있는 만들어 내기 실패를 드러낸다
    """
    
    @staticmethod
    def create_image_grid(images: torch.Tensor,
                         nrow: int = 8,
                         padding: int = 2,
                         normalize: bool = True) -> np.ndarray:
        """
        그려 보기용 그림 격자를 만든다.
        
        인수:
            images: 그림 묶음 [batch_size, channels, height, width]
            nrow: 가로줄마다 그림 수
            padding: 그림 사이 화소
            normalize: [0, 1]로 잣대를 맞출지 여부
        
        반환값:
            넘파이 배열로 된 격자 그림 [height, width, channels]
        """
        batch_size = images.shape[0]
        ncol = (batch_size + nrow - 1) // nrow  # 올림 나눗셈
        
        # 요청하면 고르게 맞춘다
        if normalize:
            images = (images - images.min()) / (images.max() - images.min() + 1e-8)
        
        # 회색과 RGB을 다룬다
        if images.shape[1] == 1:
            # 회색: [B, 1, H, W] -> [B, H, W]
            images = images.squeeze(1)
            is_grayscale = True
        else:
            # RGB: [B, 3, H, W] -> [B, H, W, 3]
            images = images.permute(0, 2, 3, 1)
            is_grayscale = False
        
        H, W = images.shape[1], images.shape[2]
        
        # 격자 바탕을 만든다
        grid_h = ncol * H + (ncol + 1) * padding
        grid_w = nrow * W + (nrow + 1) * padding
        
        if is_grayscale:
            grid = np.ones((grid_h, grid_w)) * 0.5  # 회색 바탕
        else:
            grid = np.ones((grid_h, grid_w, 3)) * 0.5
        
        # 그림을 격자에 놓는다
        for idx in range(batch_size):
            row = idx // nrow
            col = idx % nrow
            
            y = row * (H + padding) + padding
            x = col * (W + padding) + padding
            
            grid[y:y+H, x:x+W] = images[idx].numpy()
        
        return grid
    
    @staticmethod
    def plot_sample_grid(images: torch.Tensor,
                        title: str = "Generated Samples",
                        save_path: Optional[str] = None):
        """
        표본 격자를 그리고 필요하면 갈무리한다.
        
        인수:
            images: 그림 묶음 [batch_size, C, H, W]
            title: 그림의 제목
            save_path: 그림을 갈무리할 길(있으면)
        """
        grid = SampleGridVisualizer.create_image_grid(images, nrow=8)
        
        plt.figure(figsize=(12, 12))
        if len(grid.shape) == 2:  # 회색
            plt.imshow(grid, cmap='gray')
        else:  # RGB
            plt.imshow(grid)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        plt.tight_layout()


class LatentSpaceInterpolation:
    """
    숨은 공간에서 선형 사이 메우기를 한다.
    
    수학적 바탕:
    -----------------------
    숨은 벡터 z1, z2에 대해 다음과 같이 메운다.
        z(t) = (1-t) * z1 + t * z2,  where t ∈ [0, 1]
    
    목적:
    -------
    1. 숨은 공간의 매끄러움을 따진다
    2. 끊김이나 뜀을 알아낸다
    3. 배운 숨은 짜임을 그려 본다
    4. 뜻있는 사이 메우기인지 살핀다
    
    좋은 만들어 내는 모델은 숨은 부호 사이를 메울 때
    매끄러운 옮아감을 내야 한다.
    """
    
    @staticmethod
    def linear_interpolate(z1: torch.Tensor,
                          z2: torch.Tensor,
                          num_steps: int = 10) -> torch.Tensor:
        """
        숨은 벡터 둘 사이에 선형 사이 메우기를 한다.
        
        인수:
            z1: 처음 숨은 벡터 [latent_dim]
            z2: 끝 숨은 벡터 [latent_dim]
            num_steps: 사이 끼움 걸음 수
        
        반환값:
            사이를 메운 숨은 벡터 [num_steps, latent_dim]
        
        수학 공식:
        --------------------
        z(t) = (1-t) * z1 + t * z2
        where t = [0, 1/(n-1), 2/(n-1), ..., 1]
        """
        # 사이 메우기 무게를 만든다
        # 꼴: [걸음 수]
        t = torch.linspace(0, 1, num_steps)
        
        # 퍼뜨리기를 위해 차원을 늘린다
        # z1, z2: [숨은 차원] -> [1, 숨은 차원]
        # t: [걸음 수] -> [걸음 수, 1]
        t = t.unsqueeze(1)
        z1 = z1.unsqueeze(0)
        z2 = z2.unsqueeze(0)
        
        # 메운다: z(t) = (1-t) * z1 + t * z2
        # 꼴: [걸음 수, 숨은 차원]
        z_interp = (1 - t) * z1 + t * z2
        
        return z_interp
    
    @staticmethod
    def spherical_interpolate(z1: torch.Tensor,
                             z2: torch.Tensor,
                             num_steps: int = 10) -> torch.Tensor:
        """
        공 모양 선형 사이 메우기(slerp)를 한다.
        
        왜 slerp인가?
        ---------
        가우스 같은 분포(VAE의 숨은 자리)에서는
        slerp은 원점에서 거리를 한결같이 지켜
        더 자연스러운 사이 메우기를 낸다.
        
        수학 공식:
        --------------------
        slerp(z1, z2; t) = [sin((1-t)θ)/sin(θ)] * z1 + [sin(tθ)/sin(θ)] * z2
        
        where θ = arccos(z1·z2 / (||z1|| ||z2||))
        
        인수:
            z1: 처음 숨은 벡터 [latent_dim]
            z2: 끝 숨은 벡터 [latent_dim]
            num_steps: 사이 끼움 걸음 수
        
        반환값:
            사이를 메운 숨은 벡터 [num_steps, latent_dim]
        """
        # 벡터를 고르게 맞춘다
        z1_norm = F.normalize(z1, dim=0)
        z2_norm = F.normalize(z2, dim=0)
        
        # 벡터 사이의 각을 셈한다
        # θ = arccos(z1·z2)
        dot = torch.dot(z1_norm, z2_norm)
        # 수치 문제를 피하려 가둔다
        dot = torch.clamp(dot, -1.0, 1.0)
        theta = torch.acos(dot)
        
        # 벡터가 거의 나란한 경우를 다룬다
        if theta < 1e-6:
            return LatentSpaceInterpolation.linear_interpolate(z1, z2, num_steps)
        
        # 사이 메우기 무게를 만든다
        t = torch.linspace(0, 1, num_steps).unsqueeze(1)
        
        # slerp 무게를 셈한다
        sin_theta = torch.sin(theta)
        w1 = torch.sin((1 - t) * theta) / sin_theta
        w2 = torch.sin(t * theta) / sin_theta
        
        # 보간
        z_interp = w1 * z1 + w2 * z2
        
        return z_interp
    
    @staticmethod
    def visualize_interpolation(decoder,
                               z1: torch.Tensor,
                               z2: torch.Tensor,
                               num_steps: int = 10,
                               use_slerp: bool = False,
                               save_path: Optional[str] = None):
        """
        숨은 부호 둘 사이의 사이 메우기를 그려 본다.
        
        인수:
            decoder: 풀개 신경망
            z1: 시작 숨은 부호
            z2: 끝 숨은 부호
            num_steps: 사이 끼움 걸음 수
            use_slerp: 선형 대신 공 모양 사이 메우기를 쓴다
            save_path: 갈무리할 길(있으면)
        """
        # 사이 메우기를 한다
        if use_slerp:
            z_interp = LatentSpaceInterpolation.spherical_interpolate(
                z1, z2, num_steps
            )
        else:
            z_interp = LatentSpaceInterpolation.linear_interpolate(
                z1, z2, num_steps
            )
        
        # 메운 숨은 값을 푼다
        with torch.no_grad():
            images = decoder(z_interp)
        
        # 시각화 만들기
        grid = SampleGridVisualizer.create_image_grid(
            images, nrow=num_steps, padding=2
        )
        
        method = "Spherical" if use_slerp else "Linear"
        plt.figure(figsize=(15, 3))
        if len(grid.shape) == 2:
            plt.imshow(grid, cmap='gray')
        else:
            plt.imshow(grid)
        plt.title(f'{method} Interpolation in Latent Space',
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")


class ReconstructionQuality:
    """
    되짓기 품질을 따지는 잣대.
    
    흔한 잣대:
    --------------
    1. MSE(평균 제곱 어긋남): 화소마다의 차이
    2. PSNR(최고 신호 대 잡음 비): 신호 품질
    3. SSIM(짜임새 닮음): 느낌으로 본 닮음
    
    쓰임새:
    ---------
    - 변분 자기 부호기의 되짓기 품질
    - 그림에서 그림으로 옮기기
    - 누르기 따지기
    """
    
    @staticmethod
    def compute_mse(original: torch.Tensor,
                    reconstructed: torch.Tensor) -> float:
        """
        평균 제곱 어긋남을 셈한다.
        
        수학 공식:
        --------------------
        MSE = 1/(N*C*H*W) * Σ(original - reconstructed)²
        
        인수:
            original: 본디 그림 [B, C, H, W]
            reconstructed: 되살린 그림 [B, C, H, W]
        
        반환값:
            MSE 값(낮을수록 좋다)
        """
        # 제곱 차이 계산
        squared_diff = (original - reconstructed) ** 2
        
        # 모든 차원에 걸쳐 평균 낸다
        mse = torch.mean(squared_diff)
        
        return mse.item()
    
    @staticmethod
    def compute_psnr(original: torch.Tensor,
                     reconstructed: torch.Tensor,
                     max_pixel_value: float = 1.0) -> float:
        """
        봉우리 신호 대 잡음비를 셈한다.
        
        수학 공식:
        --------------------
        PSNR = 10 * log10(MAX² / MSE)
        
        여기서 MAX은 있을 수 있는 최대 화소 값이다.
        
        해석:
        --------------
        - PSNR이 높을수록 품질이 좋다
        - 봉우리 신호 대 잡음비 > 30 dB: 좋은 품질
        - 봉우리 신호 대 잡음비 > 40 dB: 뛰어난 품질
        
        인수:
            original: 본디 그림 [B, C, H, W]
            reconstructed: 되살린 그림 [B, C, H, W]
            max_pixel_value: 가장 큰 화솟값(잣대를 맞춘 그림이면 1.0)
        
        반환값:
            데시벨 단위의 PSNR(높을수록 좋다)
        """
        # 평균 제곱 어긋남을 셈한다
        mse = ReconstructionQuality.compute_mse(original, reconstructed)
        
        # 0으로 나누기를 피한다
        if mse < 1e-10:
            return 100.0  # 흠 없는 되짓기
        
        # 봉우리 신호 대 잡음비를 셈한다
        psnr = 10 * np.log10(max_pixel_value ** 2 / mse)
        
        return psnr
    
    @staticmethod
    def compute_per_sample_mse(original: torch.Tensor,
                              reconstructed: torch.Tensor) -> torch.Tensor:
        """
        표본마다 MSE를 셈한다(살피는 데 쓸모 있다).
        
        인수:
            original: 본디 그림 [B, C, H, W]
            reconstructed: 되살린 그림 [B, C, H, W]
        
        반환값:
            Per-sample MSE [B]
        """
        # 제곱 차이 계산
        squared_diff = (original - reconstructed) ** 2
        
        # C, H, W 차원에 걸쳐 평균 낸다
        per_sample_mse = torch.mean(squared_diff, dim=[1, 2, 3])
        
        return per_sample_mse


def demonstrate_sample_visualization():
    """
    표본 그려 보기 재주를 보인다.
    """
    print("=" * 70)
    print("Sample Visualization Demonstration")
    print("=" * 70)
    
    # 인공 그림을 만든다(MNIST 같은 자료를 흉내 낸다)
    batch_size = 64
    images = torch.randn(batch_size, 1, 28, 28)
    # 짜임을 조금 더한다(숫자처럼 보이게)
    images = torch.sigmoid(images * 2)
    
    print(f"\nGenerated {batch_size} synthetic images")
    print(f"Image shape: {images.shape}")
    
    # 격자 그림을 만든다
    print("\nCreating grid visualization...")
    SampleGridVisualizer.plot_sample_grid(
        images,
        title="Generated Samples (8×8 Grid)",
        save_path="/home/claude/sample_grid.png"
    )
    
    # 다양함을 살핀다
    print("\n" + "-" * 70)
    print("Diversity Analysis:")
    print("-" * 70)
    
    # 단순한 다양함 잣대로 짝마다 차이를 셈한다
    # 그림을 펼친다
    flat_images = images.reshape(batch_size, -1)
    
    # 짝마다 L2 거리를 셈한다
    # ||x_i - x_j||
    dists = torch.cdist(flat_images, flat_images, p=2)
    
    # 위 삼각을 얻는다(대각선은 뺀다)
    upper_tri = dists[torch.triu(torch.ones_like(dists), diagonal=1) == 1]
    
    print(f"Average pairwise L2 distance: {upper_tri.mean():.4f}")
    print(f"Min pairwise distance: {upper_tri.min():.4f}")
    print(f"Max pairwise distance: {upper_tri.max():.4f}")
    print(f"\nInterpretation:")
    print("  - Low average distance → Low diversity (mode collapse)")
    print("  - High average distance → High diversity")


def demonstrate_latent_interpolation():
    """
    숨은 공간 사이 메우기를 보인다.
    """
    print("\n" + "=" * 70)
    print("Latent Space Interpolation Demonstration")
    print("=" * 70)
    
    # 단순한 풀개 신경망(보여 주기용)
    class SimpleDecoder(nn.Module):
        def __init__(self, latent_dim=10):
            super().__init__()
            self.fc1 = nn.Linear(latent_dim, 128)
            self.fc2 = nn.Linear(128, 28*28)
        
        def forward(self, z):
            h = F.relu(self.fc1(z))
            x = torch.sigmoid(self.fc2(h))
            return x.view(-1, 1, 28, 28)
    
    decoder = SimpleDecoder(latent_dim=10)
    decoder.eval()
    
    # 아무 숨은 부호 둘을 뽑는다
    z1 = torch.randn(10)
    z2 = torch.randn(10)
    
    print(f"\nInterpolating between two random latent codes")
    print(f"Latent dimension: {len(z1)}")
    print(f"z1 norm: {torch.norm(z1):.4f}")
    print(f"z2 norm: {torch.norm(z2):.4f}")
    
    # 선형 사이 메우기
    print("\n" + "-" * 70)
    print("Linear Interpolation:")
    print("-" * 70)
    
    z_linear = LatentSpaceInterpolation.linear_interpolate(z1, z2, num_steps=10)
    print(f"Generated {len(z_linear)} interpolated codes")
    print(f"Norms along path: {torch.norm(z_linear, dim=1)}")
    
    # 공 모양 사이 메우기
    print("\n" + "-" * 70)
    print("Spherical Interpolation:")
    print("-" * 70)
    
    z_slerp = LatentSpaceInterpolation.spherical_interpolate(z1, z2, num_steps=10)
    print(f"Generated {len(z_slerp)} interpolated codes")
    print(f"Norms along path: {torch.norm(z_slerp, dim=1)}")
    print("\nNote: Spherical interpolation maintains constant norm")


def demonstrate_reconstruction_quality():
    """
    되짓기 품질 잣대를 보인다.
    """
    print("\n" + "=" * 70)
    print("Reconstruction Quality Metrics")
    print("=" * 70)
    
    # 본디 그림을 만든다
    batch_size = 10
    original = torch.randn(batch_size, 1, 28, 28)
    original = torch.sigmoid(original * 2)  # [0, 1]로 고르게 맞추기
    
    # 품질 수준이 다른 되짓기를 만든다
    # 흠 없는 되짓기
    perfect_recon = original.clone()
    
    # 좋은 되짓기(작은 잡음)
    good_recon = original + torch.randn_like(original) * 0.05
    good_recon = torch.clamp(good_recon, 0, 1)
    
    # 나쁜 되짓기(큰 잡음)
    poor_recon = original + torch.randn_like(original) * 0.2
    poor_recon = torch.clamp(poor_recon, 0, 1)
    
    # 지표를 계산한다
    print("\n" + "-" * 70)
    print("Reconstruction Quality Comparison:")
    print("-" * 70)
    
    reconstructions = {
        "Perfect": perfect_recon,
        "Good": good_recon,
        "Poor": poor_recon
    }
    
    print(f"\n{'Reconstruction':<15} {'MSE':<12} {'PSNR (dB)'}")
    print("-" * 70)
    
    for name, recon in reconstructions.items():
        mse = ReconstructionQuality.compute_mse(original, recon)
        psnr = ReconstructionQuality.compute_psnr(original, recon)
        print(f"{name:<15} {mse:<12.6f} {psnr:<10.2f}")
    
    print("\n" + "-" * 70)
    print("Interpretation:")
    print("-" * 70)
    print("MSE: Lower is better (0 = perfect)")
    print("PSNR: Higher is better")
    print("  - >40 dB: Excellent quality")
    print("  - 30-40 dB: Good quality")
    print("  - 20-30 dB: Fair quality")
    print("  - <20 dB: Poor quality")


def main():
    """
    표본 그려 보기를 보이는 으뜸 함수.
    """
    print("\n" + "=" * 70)
    print("MODULE 52.03: SAMPLE VISUALIZATION AND QUALITY ASSESSMENT")
    print("=" * 70)
    
    # 표본 그려 보기를 보여 준다
    demonstrate_sample_visualization()
    
    # 숨은 공간 사이 메우기를 보여 준다
    demonstrate_latent_interpolation()
    
    # 되짓기 품질을 보여 준다
    demonstrate_reconstruction_quality()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. 눈으로 살펴보기는 꼭 필요하다:
       - 잣대만으로는 느낌의 문제를 놓칠 수 있다
       - 격자 그림이 봉우리 무너짐을 드러낸다
       - 사람은 흠을 알아내는 데 뛰어나다
    
    2. 숨은 공간 사이 메우기:
       - 선형: 단순하고 빠르다
       - 공 모양: 정규 분포에 더 낫다
       - 매끄러운 옮아감은 숨은 짜임이 좋다는 표시이다
    
    3. 되짓기 잣대:
       - 평균 제곱 어긋남: 단순한 화소마다 차이
       - 봉우리 신호 대 잡음비: 데시벨로 나타낸 신호 품질
       - 봉우리 신호 대 잡음비가 클수록 흔히 품질이 좋다
    
    4. 다양함 따지기:
       - 짝마다 표본 거리
       - 격자를 눈으로 살펴보기
       - 다양함이 낮으면 봉우리 무너짐을 뜻한다
    
    5. 모범 관행:
       - 늘 표본을 그려 본다
       - 사이 메우기가 매끄러운지 살핀다
       - 품질 잣대를 여럿 쓴다
       - 수로 따지기와 결로 따지기를 아우른다
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()```

## 논의

이 짜기는 단원 52.03: 표본 그려 보기와 품질 따지기에 대해 자리 잡은 가장 좋은 방식을 따른다. 코드는 모델 뜻매김, 익히기 논리, 도구 함수를 또렷이 갈라 짜였다. 핵심 설계 결정에는 깨움 함수 고르기, 고르게 맞추기 방책, 가장 좋게 하기 웃매개변수가 들며 모두 익히기의 안정과 내놓기 품질에 크게 영향을 준다.

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
