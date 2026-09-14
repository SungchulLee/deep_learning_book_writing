# 표본 그려 보기

이 단원은 깊은 만들어 내는 모델의 중요한 부품인 표본 그려 보기와 품질 따지기을 짠다. 이 짜기를 이해하면 요즘 만들어 내는 모델에 쓰이는 얼개의 결과 익히기 절차를 꿰뚫어 볼 수 있다. 이 코드는 연구와 실제 얼개에서 널리 쓰이는 쓸모 있는 재주를 보인다.

## 1. 코드

```python
"""
표본 그려 보기와 품질 따지기
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
        save_path="sample_grid.png"
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
    main()
```

**출력:**

```
======================================================================
MODULE 52.03: SAMPLE VISUALIZATION AND QUALITY ASSESSMENT
======================================================================
======================================================================
Sample Visualization Demonstration
======================================================================

Generated 64 synthetic images
Image shape: torch.Size([64, 1, 28, 28])

Creating grid visualization...
✓ Saved: sample_grid.png

----------------------------------------------------------------------
Diversity Analysis:
----------------------------------------------------------------------
Average pairwise L2 distance: 12.4770
Min pairwise distance: 11.5609
Max pairwise distance: 13.2881

Interpretation:
  - Low average distance → Low diversity (mode collapse)
  - High average distance → High diversity

======================================================================
Latent Space Interpolation Demonstration
======================================================================

Interpolating between two random latent codes
Latent dimension: 10
z1 norm: 2.9090
z2 norm: 3.5240


... (61 lines omitted)

       - 격자를 눈으로 살펴보기
       - 다양함이 낮으면 봉우리 무너짐을 뜻한다
    
    5. 모범 관행:
       - 늘 표본을 그려 본다
       - 사이 메우기가 매끄러운지 살핀다
       - 품질 잣대를 여럿 쓴다
       - 수로 따지기와 결로 따지기를 아우른다
    
======================================================================
```

## 2. 논의

이 짜기는 표본 그려 보기와 품질 따지기에 대해 자리 잡은 가장 좋은 방식을 따른다. 코드는 모델 뜻매김, 익히기 논리, 도구 함수를 또렷이 갈라 짜였다. 핵심 설계 결정에는 깨움 함수 고르기, 고르게 맞추기 방책, 가장 좋게 하기 웃매개변수가 들며 모두 익히기의 안정과 내놓기 품질에 크게 영향을 준다.

이 얼개는 깊은 만들어 내는 모델에 흔한 중요한 결 여럿을 보인다. 곧 여러 신경망 층을 지나며 특징을 차츰 다루기, 모델이 곁 앎을 받아들이게 하는 조건 주기 얼개, 익히는 동안 기울기가 안정되게 흐르도록 하는 꼼꼼한 첫자리매김이다.

새 자료 묶음이나 문제 마당에서는 웃매개변수 고르기와 익히기 절차를 꼼꼼히 맞추어야 할 때가 많으므로 다루는 이들은 이에 마음을 써야 한다. 코드가 조각으로 나뉘어 있어 다른 얼개, 손실 함수, 익히기 방책을 실험하기 쉽다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
구체적인 들임 텐서로 이 단원의 으뜸 모델의 앞먹임을 좇아라. 층마다 꼴이 어떻게 바뀌는지 적고 내놓기 차원이 바라던 것과 맞는지 확인하라.

</div>

??? success "연습문제 1 풀이"
    들임 텐서에서 시작해 층마다 바뀜을 따라가라. 겹말기 층에서는 공간 차원에 공식 $H_{out} = \lfloor(H_{in} + 2p - k) / s\rfloor + 1$을 쓴다. 선형 층에서는 특징 차원의 바뀜을 좇는다. 중간 꼴을 하나씩 적고 마지막 내놓기가 그 일(그림 만들어 내기, 가르기 등)에 바라던 목표 차원과 맞는지 확인하라.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
이 짜기의 핵심 웃매개변수(배움 빠르기, 묶음 크기, 얼개 고르기)를 가려내라. 다른 것을 붙박아 두고 하나씩 바꾸어 웃매개변수마다 익히기가 얼마나 민감한지 재는 실험을 짜라.

</div>

??? success "연습문제 2 풀이"
    핵심 웃매개변수에는 배움 빠르기(흔히 $10^{-4}$에서 $10^{-3}$), 묶음 크기(64-256), 층과 채널의 수, 깨움 함수가 든다. 웃매개변수마다 값을 3~5가지로 바꾸어 모델을 익히고 알맞은 잣대(손실, 표본 품질, 모이는 빠르기)를 좇아라. 결과를 그려 어느 웃매개변수가 가장 큰 영향을 주는지 가려내라. 흔히 배움 빠르기와 얼개 깊이가 가장 세게 영향을 주고, 묶음 크기는 알맞은 범위 안에서는 웬만큼 영향을 준다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
이 짜기에 새 기능을 더해 넓혀라. 곧 기울기 자르기, 배움 빠르기 차례표, 다른 손실 함수를 더하라. 고치기 앞뒤의 익히기 움직임을 견주어라.

</div>

??? success "연습문제 3 풀이"
    기울기 자르기는 `optimizer.step()` 앞에 `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`을 더한다. 배움 빠르기 차례표는 `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)`을 쓰고 바퀴마다 `scheduler.step()`을 부른다. 익히기 손실 곡선, 모이는 빠르기, 마지막 모델 품질을 견주어라. 기울기 자르기는 흔히 익히기가 치솟는 것을 막고, 코사인 식히기는 뒤 바퀴에서 더 곱게 가장 좋게 하여 마지막 솜씨를 높일 수 있다.


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
표본 격자를 그릴 때 표본을 고르면 안 되는 까닭은 무엇인가?

</div>

??? success "연습문제 4 풀이"
    고른 격자는 모델이 할 수 있는 일을 부풀려 보인다. 읽는 사람이 그것을 전형적인 표본으로
    받아들이기 때문이다.

    정직한 방식은 씨앗을 고정하고 처음 $n$개를 그대로 싣는 것이다.

    ```python
    torch.manual_seed(0)
    z = torch.randn(64, latent_dim)
    imgs = model.decode(z)          # 이 64개를 그대로
    ```

    고른 격자를 싣고 싶을 때도 있다. 무엇이 가능한지 보이려는 경우다. 그때는 **골랐다는
    사실과 기준을 적으면** 된다. "1,000개 가운데 사람이 고른 64개"라고 적혀 있으면
    읽는 사람이 알맞게 읽는다.

    같은 이유로 밝혀야 하는 것이 하나 더 있다. **온도나 잘라 내기를 썼는지**다. 썼다면
    그것은 모델의 표본이 아니라 손본 표본이다
    ([24장](../../ch24/training/generate_samples.md)).

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
표본 격자에서 다양성 문제를 어떻게 알아채는가?

</div>

??? success "연습문제 5 풀이"
    격자 안에 비슷한 것이 여럿 있는지 본다. 그런데 이것이 생각보다 어렵다.

    까닭은 하나하나가 그럴듯하면 넘어가기 쉽기 때문이다. 서로 다른 그림 열 장을 되풀이한
    묶음을 격자로 그리면 64칸에 같은 그림이 여섯 번쯤 나오는데, 무심히 보면 알아채지
    못한다.

    도움이 되는 방법들이 있다.

    **격자를 크게.** 8×8보다 16×16이 되풀이를 알아보기 쉽다.

    **부류로 묶어 그린다.** 판정된 부류마다 한 줄로 놓으면 부류 안의 되풀이가 보인다.

    **가장 가까운 학습 표본을 옆에 붙인다.** 만든 표본과 그 이웃을 나란히 놓으면 외우고
    있는지도 함께 보인다.

    **가장 가까운 표본끼리 묶어 보인다.** 만든 표본들 사이의 거리를 재어 가장 닮은 짝을
    찾아 나란히 놓는다. 되풀이가 있으면 곧 드러난다.

    그래도 눈으로는 한계가 있으니 수를 함께 보아야 한다. 판정된 부류의 엔트로피가 값싸고
    쓸 만하다. 이 장의 모델들은 2.2 근처이고 고른 열 부류가 2.303이다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
숨은 공간을 따라가며 그리는 그림은 무엇을 보여 주는가?

</div>

??? success "연습문제 6 풀이"
    두 코드 사이를 이어 단계마다 풀어 한 줄로 그리는 것이다. 보여 주는 것이 몇 가지다.

    **매끄러운가.** 단계가 갑자기 튀지 않고 이어지면 숨은 공간이 매끄럽다는 신호다.

    **사이가 그럴듯한가.** 가운데 단계가 얼룩이면 그 구간이 빈 곳이다.

    **외우고 있지 않은가.** 사이 끼움이 학습 표본 사이를 매끄럽게 지나면 모델이 자료를
    그저 외운 것이 아니라는 약한 증거가 된다.

    다만 이 그림으로 모델을 견주는 데는 한계가 있다.
    [24장에서 재어 본](../../ch24/training/generate_samples.md) 대로 자기 부호기와 변분
    자기 부호기의 사이 끼움이 거의 같았다(가운데에서 확신도가 0.132 대 0.140 떨어진다).
    두 모델의 뽑기 능력은 0.2% 대 57.4%로 크게 다른데 이 그림은 그 차이를 보이지 못한다.

    까닭은 양 끝이 **자료에서 온 코드**라 그 사이도 대개 아는 영역을 지나기 때문이다.
    모델의 약점은 코드가 놓인 자리에서 먼 곳이고, 사이 끼움은 거기까지 가지 않는다.

    그러므로 이 그림은 **숨은 공간의 성질을 보이는 데** 쓰고, 모델을 견주는 데는 쓰지
    않는 것이 맞다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
적대적 생성망의 표본을 익히기 도중에 그려 볼 때 무엇을 고정해야 하는가?

</div>

??? success "연습문제 7 풀이"
    **$z$를 고정**해야 한다. 에포크마다 새로 뽑으면 그림이 달라진 것이 모델 때문인지
    $z$ 때문인지 알 수 없다.

    ```python
    fixed_z = torch.randn(64, latent_dim)      # 한 번만 뽑는다
    for ep in range(epochs):
        ...
        save_grid(g(fixed_z), f'epoch_{ep}.png')
    ```

    이렇게 하면 같은 $z$가 에포크마다 어떻게 변해 가는지 보이므로, 모델이 배우는 과정이
    한 줄의 이야기로 읽힌다. 흐릿한 덩어리에서 획이 잡히고 모양이 또렷해지는 흐름이
    보인다.

    무너짐도 이 그림에서 가장 빨리 보인다. 64칸이 서로 닮아 가기 시작하면 무너지는
    중이다. 손실로는 알 수 없다.

    그리고 `model.eval()`을 함께 챙겨야 한다. 생성기에 배치 정규화가 있으면 모드에 따라
    출력이 달라지므로, 익히기 모드로 그린 그림과 평가 모드로 그린 그림이 다르다. 하나로
    정해 두어야 에포크끼리 견줄 수 있다.

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
만든 표본이 학습 자료를 외운 것인지 어떻게 확인하는가?

</div>

??? success "연습문제 8 풀이"
    만든 표본마다 **가장 가까운 학습 표본**을 찾아 견주는 것이 기본이다.

    ```python
    d = torch.cdist(fake_feats, train_feats)
    nn_dist, nn_idx = d.min(dim=1)
    ```

    그리고 두 가지를 본다.

    **거리의 분포.** 만든 표본의 최근접 거리가 학습 표본끼리의 최근접 거리보다 눈에 띄게
    작으면 외우고 있는 것이다.

    **나란히 그린다.** 만든 표본과 그 이웃을 붙여 놓으면 눈으로 확인된다.

    화소 공간에서 재면 안 된다는 점이 중요하다. 한 화소만 움직여도 거리가 달라지므로
    외운 표본이 멀어 보일 수 있다. **특징 공간에서** 재는 것이 낫다.

    수치로 보는 다른 방법이 있다. 학습 자료 기준 FID와 시험 자료 기준 FID를 각각 재어
    견주는 것이다. 앞쪽만 좋으면 외우고 있다는 신호다
    ([FID 연습문제 19](fid.md)).

    이 확인이 필요한 까닭은 FID와 인셉션 점수가 **외우기를 벌하지 않기** 때문이다. 오히려
    FID는 자료 분포와 같아지는 것을 상으로 주므로 외우기가 만점이다. 잣대가 못 보는
    것이니 따로 보아야 한다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
표본 격자를 만들 때 값의 범위와 정렬을 어떻게 다루는가?

</div>

??? success "연습문제 9 풀이"
    **범위를 먼저 맞춘다.** `tanh` 출력이면 $[-1,1]$이므로 $[0,1]$로 옮겨야 한다.

    ```python
    imgs = (imgs + 1) / 2          # tanh 를 썼다면
    imgs = imgs.clamp(0, 1)        # 넘치는 값을 자른다
    ```

    자르는 것을 빼면 `imshow`가 알아서 눈금을 맞추는데, 그러면 **그림마다 다른 눈금**이
    쓰여 견줄 수 없다. 특히 한 칸에 아주 밝은 화소가 하나 있으면 그 칸 전체가 어두워
    보인다.

    ```python
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)     # 눈금을 못 박는다
    ```

    정렬도 정해 두어야 한다. 무작위 순서가 정직하지만 읽기는 어렵다. 판정된 부류로
    정렬하면 읽기 좋아지는 대신 **다양성이 있는 것처럼 보이는** 착시를 만들 수 있다.
    부류마다 한 줄씩 놓으면 열 부류가 다 있어 보이기 때문이다.

    그래서 둘을 함께 싣는 것이 좋다. 무작위 격자로 정직하게 보이고, 부류로 묶은 격자로
    자세히 보인다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
그림을 저장할 때 어떤 형식을 쓰겠는가?

</div>

??? success "연습문제 10 풀이"
    표본 격자는 **PNG**가 맞다. 화소 자료를 있는 그대로 보여야 하므로 벡터 형식이 줄 이득이
    없고, SVG로 저장하면 화소마다 사각형이 되어 파일이 커진다.

    이 책의 다른 그림들은 SVG를 쓴다. 곡선과 글자로 된 그림이라 확대해도 또렷하고 diff가
    되기 때문이다. 격자는 그 이유가 해당되지 않는다.

    PNG로 저장할 때 챙길 것이 둘이다.

    ```python
    plt.imsave('grid.png', grid, cmap='gray', vmin=0, vmax=1)
    ```

    **보간을 끄는 것**도 중요하다. 기본 보간이 켜져 있으면 28×28을 키울 때 뭉개져서
    실제보다 매끄러워 보인다.

    ```python
    plt.imshow(img, cmap='gray', interpolation='nearest')
    ```

    이 한 줄이 표본의 품질을 실제보다 좋게 보이는 것을 막아 준다. 격자를 크게 싣는 글에서
    자주 어긋나는 자리다.

## 정리하며

**다룬 것** — 표본 그려 보기와 품질 따지기

이 짜기는 표본 그려 보기와 품질 따지기에 대해 자리 잡은 가장 좋은 방식을 따른다.

고갱이 갈래는 `SampleGridVisualizer`, `LatentSpaceInterpolation`, `ReconstructionQuality`, `SimpleDecoder`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
