# 샘플 시각화

생성 샘플의 격자는 반복되는 패턴, 형태의 결함, 색상이나 밝기의 이상을 직접 살펴보는 도구다. 잠재 공간 보간은 선택한 경로에서 출력이 어떻게 변하는지 보여 주며, 재구성 오차는 원본과 대응되는 출력의 차이를 측정한다. 세 도구는 서로 다른 질문에 답하므로 한 가지를 전체 생성 품질의 증거로 사용하지 않는다.

## 1. 시각화와 재구성 계산 예제

아래 예제의 이미지는 학습된 GAN의 출력이 아니라 난수에 시그모이드를 적용한 패턴이다. 보간 예제의 `SimpleDecoder`도 학습하지 않은 뉴럴 넷트웍이다. 따라서 출력의 거리와 재구성 수치는 계산 방법을 보여 줄 뿐, 생성 모델의 학습 성능을 나타내지 않는다.

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
            images: 그림 배치 [batch_size, channels, height, width]
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
            images: 그림 배치 [batch_size, C, H, W]
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
    
    좋은 만들어 내는 모델은 잠재 표현 사이를 메울 때
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
            num_steps: 사이 끼움 스텝 수
        
        반환값:
            사이를 메운 숨은 벡터 [num_steps, latent_dim]
        
        수학 공식:
        --------------------
        z(t) = (1-t) * z1 + t * z2
        where t = [0, 1/(n-1), 2/(n-1), ..., 1]
        """
        # 사이 메우기 무게를 만든다
        # 꼴: [스텝 수]
        t = torch.linspace(0, 1, num_steps)
        
        # 퍼뜨리기를 위해 차원을 늘린다
        # z1, z2: [숨은 차원] -> [1, 숨은 차원]
        # t: [스텝 수] -> [스텝 수, 1]
        t = t.unsqueeze(1)
        z1 = z1.unsqueeze(0)
        z2 = z2.unsqueeze(0)
        
        # 메운다: z(t) = (1-t) * z1 + t * z2
        # 꼴: [스텝 수, 숨은 차원]
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
            num_steps: 사이 끼움 스텝 수
        
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
        잠재 표현 둘 사이의 사이 메우기를 그려 본다.
        
        인수:
            decoder: 디코더 뉴럴 넷트웍
            z1: 시작 잠재 표현
            z2: 끝 잠재 표현
            num_steps: 사이 끼움 스텝 수
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
    - 변분 오토인코더의 되짓기 품질
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
    
    # 단순한 디코더 뉴럴 넷트웍(보여 주기용)
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
    
    # 아무 잠재 표현 둘을 뽑는다
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

??? note "전체 출력 (105줄)"

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

    ----------------------------------------------------------------------
    Linear Interpolation:
    ----------------------------------------------------------------------
    Generated 10 interpolated codes
    Norms along path: tensor([2.9090, 2.7271, 2.6035, 2.5466, 2.5610, 2.6454, 2.7936, 2.9960, 3.2426,
            3.5240])

    ----------------------------------------------------------------------
    Spherical Interpolation:
    ----------------------------------------------------------------------
    Generated 10 interpolated codes
    Norms along path: tensor([2.9090, 2.9488, 3.0114, 3.0906, 3.1791, 3.2695, 3.3547, 3.4286, 3.4863,
            3.5240])

    Note: Spherical interpolation maintains constant norm

    ======================================================================
    Reconstruction Quality Metrics
    ======================================================================

    ----------------------------------------------------------------------
    Reconstruction Quality Comparison:
    ----------------------------------------------------------------------

    Reconstruction  MSE          PSNR (dB)
    ----------------------------------------------------------------------
    Perfect         0.000000     100.00    
    Good            0.002223     26.53     
    Poor            0.031011     15.08     

    ----------------------------------------------------------------------
    Interpretation:
    ----------------------------------------------------------------------
    MSE: Lower is better (0 = perfect)
    PSNR: Higher is better
      - >40 dB: Excellent quality
      - 30-40 dB: Good quality
      - 20-30 dB: Fair quality
      - <20 dB: Poor quality

    ======================================================================
    KEY TAKEAWAYS
    ======================================================================

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
        
    ======================================================================
    ```

## 2. 샘플 격자의 입력과 색상 범위

`create_image_grid`는 N×C×H×W 텐서를 받는다. C=1이면 흑백 격자를 만들고, 그 외에는 RGB용 세 채널 배열을 할당하므로 임의의 채널 수를 지원하는 함수는 아니다. nrow는 한 행의 최대 이미지 수이며 양수여야 한다. `.numpy()`를 직접 호출하므로 입력은 CPU에 있고 그래디언트 추적에서 분리되어 있어야 한다.

기본 `normalize=True`는 **배치 전체의 최솟값과 최댓값**을 0과 1로 옮긴다. 이미지별 정규화는 아니지만 서로 다른 배치·에포크에 다른 눈금이 적용될 수 있다. 상수 배치는 분자가 0이 되어 검게 표시된다. 비교용 그림에서는 원래 값의 범위를 명시하고 공통 변환과 표시 범위를 사용한다. 예를 들어 tanh 출력의 (x+1)/2 변환은 알려진 범위를 옮기는 것이며, 배치별 최솟값·최댓값 재조정과 다르다.

흑백 `imshow`는 vmin·vmax를 생략하면 전달된 배열의 범위에 따라 눈금을 잡는다. RGB 배열은 같은 방식으로 색상 맵을 자동 조정하지 않는다. 표시 범위 밖 값을 자르면 정보가 사라지므로 clipping 여부도 기록한다. 함수의 중간 회색 테두리는 데이터가 아니라 격자 배경이다.

## 3. 다양성은 거리 하나로 판단하지 않는다

예제는 평탄화한 이미지 사이의 모든 유클리드 거리를 계산하고 대각선 위쪽만 집계한다. N개 이미지에 대해 N(N−1)/2개 쌍이며, 전체 거리 행렬의 메모리는 O(N²)이다. 적어도 두 이미지가 있어야 최솟값·최댓값을 의미 있게 계산할 수 있다.

픽셀 거리에는 이동, 밝기, 노이즈도 영향을 준다. 잡음 이미지는 거리가 커도 유용한 다양성이 없을 수 있고, 같은 클래스의 좋은 이미지들은 거리가 작을 수도 있다. 따라서 출력의 '거리가 작으면 모드 붕괴, 크면 높은 다양성'은 충분한 판정 기준이 아니다. 표본 격자, 중복 검사, 과제에 맞는 피처 공간, 클래스 내·클래스 간 분포를 함께 살펴본다.

## 4. 잠재 공간 보간의 의미와 한계

선형 보간은 $z(t)=(1-t)z_1+tz_2$다. 고차원 등방 정규분포에서 독립적으로 선택한 끝점의 중간은 일반적인 표본보다 원점에 가까워질 수 있다. 그렇다고 선형 보간이 항상 나쁜 이미지를 만들거나 구면 보간이 항상 우수한 것은 아니다.

구면 보간은 두 방향 사이 각도 θ를 이용한다.

$$z(t)=\frac{\sin((1-t)\theta)}{\sin\theta}z_1+
\frac{\sin(t\theta)}{\sin\theta}z_2.$$

끝점의 노름이 같을 때 이 식은 같은 반지름의 구 위 경로를 만든다. 코드의 끝점은 독립 정규 난수이므로 노름이 일반적으로 다르며, 출력의 '일정한 노름을 유지한다'는 설명은 이 예제에 그대로 성립하지 않는다. 영벡터나 거의 반대 방향인 끝점도 주의해야 한다. 구현은 작은 θ만 선형 보간으로 대체하므로 θ≈π에서의 불안정성은 별도로 다루지 않는다.

보간 함수는 길이 D인 끝점을 가정하고 CPU에 보간 계수를 만든다. CUDA 입력과 혼용하면 장치 문제가 생길 수 있다. `visualize_interpolation`의 `no_grad`는 그래디언트 기록을 끄지만 디코더를 평가 모드로 전환하지는 않는다. 배치 정규화나 드롭아웃이 있으면 호출 전에 모드를 명확히 정해야 한다.

## 5. MSE와 PSNR은 대응된 재구성의 지표다

MSE는 원본과 재구성의 모든 원소에 대한 제곱오차 평균이다. PSNR은 정해진 최대 값 L에 대해 다음과 같다.

$$\mathrm{PSNR}=10\log_{10}\frac{L^2}{\mathrm{MSE}}.$$

`max_pixel_value=1`은 0~1 범위를 가정한다. MSE=0이면 수학적인 PSNR은 +∞지만 이 코드는 MSE<10⁻¹⁰이면 편의상 100을 반환한다. 출력의 PSNR 품질 구간은 모든 영상 과제에 통용되는 기준이 아니다.

`compute_psnr`은 배치 전체 MSE를 먼저 구하므로 이미지별 PSNR의 평균과 다를 수 있다. `compute_per_sample_mse`는 N×C×H×W의 마지막 세 축을 평균 낸다. 대응되는 원본이 없는 독립 생성 샘플에 임의의 실제 이미지를 짝지어 재구성 품질을 평가해서는 안 된다. 실제 GAN의 분포 평가에는 별도의 샘플 기반 지표가 필요하다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
N=10, C=1, H=W=28, nrow=8, padding=2일 때 격자 배열의 크기를 계산하라.

</div>

??? success "풀이"

    두 행이 필요하다. 높이는 2×28+3×2=62, 너비는 8×28+9×2=242이므로 흑백 배열은 62×242다. 마지막 행의 빈 여섯 칸은 배경으로 남는다. 이 계산에서 ncol이라는 변수는 실제로 행 수를 나타낸다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
서로 다른 에포크의 샘플을 비교할 때 배치별 최솟값·최댓값 정규화가 어떤 문제를 만들 수 있는가?

</div>

??? success "풀이"

    밝기와 대비의 실제 차이가 서로 다른 눈금에 가려질 수 있다. 생성기의 출력 범위를 기준으로 공통 변환을 정하고 흑백 표시의 vmin·vmax도 고정한다. 같은 잠재 표현, 모드, 샘플 선택 규칙을 유지하며 별도의 무작위 샘플도 확인한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
평균 픽셀 거리가 큰 배치를 곧 다양성이 좋은 배치라고 판단할 수 없는 이유를 설명하라.

</div>

??? success "풀이"

    독립적인 잡음이나 밝기 변화만으로도 거리가 커진다. 의미 있는 클래스와 클래스 내부 변이를 표현하는지는 별도 문제다. 피처 공간의 거리, 표본 검사, 클래스 분포와 중복 분석을 함께 사용하며 선택한 피처 추출기의 한계도 기록한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
보고서의 샘플 격자에서 표본 선택 규칙을 밝혀야 하는 이유는 무엇인가?

</div>

??? success "풀이"

    좋은 샘플만 선택하면 일반적인 생성 결과보다 품질이 높아 보인다. 아래 코드는 시드를 고정하는 예시지만 변수 정의, 장치, 평가 모드와 no_grad 처리가 생략되어 있다. 실제 보고에는 미리 정한 시드와 순서, 생성 수와 공개 수, 선별 여부를 기록한다. 온도나 절단 샘플링을 쓰면 그 조건에서의 생성 분포임을 밝힌다.

    ```python
    torch.manual_seed(0)
    z = torch.randn(64, latent_dim)
    imgs = model.decode(z)          # 이 64개를 그대로
    ```


---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
샘플 격자와 클래스 엔트로피로 다양성을 볼 때 각각 무엇을 놓칠 수 있는가?

</div>

??? success "풀이"

    격자는 제한된 표본만 보여 주므로 드문 모드의 누락이나 큰 데이터셋의 중복을 모두 찾지 못한다. 클래스 엔트로피가 높아도 클래스마다 같은 이미지 하나씩만 반복할 수 있고, 분류기가 잡음을 여러 클래스에 배정할 수도 있다. 균등한 열 클래스의 최대 엔트로피는 자연로그 기준 log 10≈2.303이지만 이는 이미지 다양성 전체의 최대 점수를 뜻하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
매끄러운 잠재 보간이 일반화나 비암기를 증명하는가? 구면 보간은 언제 일정한 노름을 유지하는가?

</div>

??? success "풀이"

    선택한 두 끝점 사이에서 출력이 연속적으로 변한다는 관찰만으로 일반화나 비암기를 증명할 수 없다. 매끄러운 함수도 학습 데이터를 암기할 수 있으며 다른 잠재 영역에서 실패할 수 있다. 표시된 구면 보간 공식은 같은 노름의 비영 끝점과 수치적으로 유효한 각도에서 일정한 반지름을 유지한다. 서로 다른 노름의 끝점에는 그 결론을 적용할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
학습 중 고정된 잠재 표현으로 샘플을 추적할 때 어떤 설정을 함께 고정해야 하는가?

</div>

??? success "풀이"

    잠재 표현뿐 아니라 평가 모드, 조건 레이블, 후처리와 표시 범위를 맞춘다. 아래 코드는 학습 루프 중간의 개념적 예시로, 별도의 eval/no_grad와 이전 모드 복원이 필요하다. 같은 잠재 표현의 출력을 추적하면 변화를 해석하기 쉽지만 그 소수 샘플에서 변화가 좋다는 사실이 전체 분포 개선을 보장하지 않는다.

    ```python
    fixed_z = torch.randn(64, latent_dim)      # 한 번만 뽑는다
    for ep in range(epochs):
        ...
        save_grid(g(fixed_z), f'epoch_{ep}.png')
    ```


---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
생성 샘플의 최근접 학습 이웃을 찾으면 암기 여부를 확정할 수 있는가?

</div>

??? success "풀이"

    가까운 이웃은 중복이나 암기를 조사할 단서지만 확정적 판정은 아니다. 학습 집합 크기, 데이터의 반복성, 전처리와 거리 정의에 따라 최근접 거리가 달라진다. 학습·독립 검증 집합에 대한 거리를 크기와 조건을 맞춰 비교하고 이웃을 직접 살펴본다. 아래 전체 cdist는 두 집합 크기의 곱만큼 메모리를 사용하므로 큰 집합에는 나누어 계산하는 방법이 필요하다. 피처 거리와 픽셀 거리는 서로 다른 결함을 보며 둘 다 암기를 완전히 판별하지는 못한다. FID나 인셉션 점수만으로도 암기를 배제할 수 없다.

    ```python
    d = torch.cdist(fake_feats, train_feats)
    nn_dist, nn_idx = d.min(dim=1)
    ```


---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
tanh 출력을 그릴 때 범위 변환과 표시 범위 설정은 각각 어떤 역할을 하는가?

</div>

??? success "풀이"

    아래 첫 코드는 −1~1을 0~1로 옮기고 범위 밖 값을 자른다. 두 번째 코드는 흑백 이미지의 공통 표시 범위를 지정한다. 변환과 clipping은 데이터 값에 작용하고 vmin·vmax는 표시 눈금을 정한다. RGB는 흑백 색상 맵과 동작이 다르다. 정렬·클래스별 선별도 전체 샘플의 빈도를 바꿔 보이게 할 수 있으므로 규칙을 밝히고 원래 순서의 격자를 함께 확인한다.

    ```python
    imgs = (imgs + 1) / 2          # tanh 를 썼다면
    imgs = imgs.clamp(0, 1)        # 넘치는 값을 자른다
    ```
    ```python
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)     # 눈금을 못 박는다
    ```


---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
샘플 격자를 저장할 때 PNG와 SVG의 차이, 보간 설정의 의미를 설명하라.

</div>

??? success "풀이"

    PNG는 로스 없이 래스터 격자를 저장하기에 적합하다. SVG에도 래스터 이미지를 포함할 수 있으므로 SVG 저장이 언제나 픽셀별 사각형을 만든다는 설명은 옳지 않다. 래스터 자체의 세부 정보가 벡터 형식으로 바뀌어 늘어나는 것도 아니다. 아래처럼 값 범위와 nearest 보간을 명시하면 픽셀 경계를 확인하기 쉽다. 저장 해상도와 화면 확대 방식도 명시하며 보간 유무가 실제 생성 데이터의 품질을 바꾸는 것은 아니다.

    ```python
    plt.imsave('grid.png', grid, cmap='gray', vmin=0, vmax=1)
    ```
    ```python
    plt.imshow(img, cmap='gray', interpolation='nearest')
    ```


## 정리하며

샘플 격자는 선택 규칙과 공통 표시 범위를 명시해야 비교할 수 있다. 보간은 선택한 잠재 경로의 동작을, MSE·PSNR은 대응된 재구성의 오차를 보여 준다. 픽셀 거리나 매끄러운 보간만으로 다양성, 일반화, 비암기를 확정하지 않는다.
