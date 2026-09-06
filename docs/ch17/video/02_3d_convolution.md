# 단원 34: 영상 이해

단원 34: 영상 이해 — 첫걸음 수준. 파일 02: 3차원 누비기 — 자리와 때에 걸친 특징 뽑기

이 단원은 셈틀 보기라는 더 넓은 맥락 안에서 영상 이해를 살펴본다. 여기 짠 것은 요즘 체계에서 쓰는 얼개의 새로움과 익히기 전략을 보여 주는 실전 PyTorch 코드이다.

## 코드

```python
"""
단원 34: 영상 이해 — 첫걸음 수준
파일 02: 3차원 누비기 — 자리와 때에 걸친 특징 뽑기

이 파일은 영상을 위한 3차원 누비기 그물을 다룬다:
- 3차원 누비기와 2차원 누비기 견주어 이해하기
- 자리와 때의 특징 뽑기
- PyTorch로 3차원 누비기 신경망 짜기
- C3D(3차원 누비기) 얼개
- 2차원 누비기 신경망 대안과의 견줌

수학적 바탕:
3차원 누비기 연산:
    들임 V ∈ ℝ^(T×C×H×W)과 알맹이 K ∈ ℝ^(t×c×h×w)에 대해:
    
    Output(τ, i, j) = Σ Σ Σ Σ V(τ+t', c', i+h', j+w') · K(t', c', h', w')
                      t' c' h' w'
    
2차원과의 핵심 차이:
    - 2차원 누비기: 자리 차원(H, W)만 다룬다
    - 3차원 누비기: 자리와 때의 덩어리(T, H, W)를 함께 다룬다
    
이러면 그물이 움직임 무늬를 곧바로 배울 수 있다!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import warnings

# ========================================================================
# 메인
# ========================================================================
warnings.filterwarnings('ignore')


#=============================================================================
# 1부: 2차원과 3차원 누비기 견줌
#=============================================================================

class Conv2DExample(nn.Module):
    """
    틀마다 적용하는 2차원 누비기.
    
    과정:
        틀마다 따로 다룬다 → 때의 앎이 없다!
        
    수학 연산:
        틀마다 I_t ∈ ℝ^(C×H×W)에 대해:
        Output_t = Conv2D(I_t)
        
    한계: 움직임이나 때에 걸친 흐름을 담아내지 못한다
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 64):
        super().__init__()
        
        # 보통의 2차원 누비기(자리만)
        # 알맹이 크기: (h, w) = (3, 3)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        틀마다 2차원 누비기를 쓰는 앞먹임.
        
        인수:
            x: 들임 영상 (B, T, C, H, W) 또는 (T, C, H, W)
            
        반환값:
            내놓는 특징 (B, T, out_C, H, W) 또는 (T, out_C, H, W)
            
        유의: T 차원은 지켜지지만 틀마다 따로 다룬다
        """
        # 모든 틀을 함께 다루려 꼴 바꾸기
        # (B, T, C, H, W) → (B*T, C, H, W)
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            batch_mode = True
        else:
            T, C, H, W = x.shape
            x = x.view(T, C, H, W)
            batch_mode = False
        
        # 2차원 누비기 쓰기
        out = self.conv(x)  # (B*T, out_C, H, W)
        
        # 다시 꼴 되돌리기
        if batch_mode:
            out = out.view(B, T, -1, H, W)
        else:
            out = out.view(T, -1, H, W)
        
        return out


class Conv3DExample(nn.Module):
    """
    자리와 때를 다루는 3차원 누비기.
    
    과정:
        때 덩어리를 함께 다룬다 → 움직임 무늬를 배운다!
        
    수학 연산:
        영상 V ∈ ℝ^(T×C×H×W)에 대해:
        내놓음 = Conv3D(V)
        
        3차원 알맹이는 자리와 때 모두에서 미끄러진다:
        K ∈ ℝ^(t_k × C × h_k × w_k)
        
    이점: 자리와 때의 특징(보기로 걷기, 뛰기)을 배울 수 있다
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 64):
        super().__init__()
        
        # 3차원 누비기(자리와 때)
        # 알맹이 크기: (t, h, w) = (3, 3, 3)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 3, 3),  # 때 크기, 높이, 너비
            padding=(1, 1, 1)       # 차원 지키기
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        3차원 누비기를 쓴 앞먹임.
        
        인수:
            x: 들임 영상 (B, C, T, H, W) — 유의: PyTorch 3차원 누비기 꼴!
            
        반환값:
            내놓는 특징 (B, out_C, T, H, W)
            
        중요: PyTorch Conv3d는 (B, C, T, H, W) 꼴을 바란다
        """
        # 들임 꼴 살피기
        if x.dim() == 4:
            # 필요하면 묶음 차원을 더한다
            x = x.unsqueeze(0)  # (C, T, H, W) → (1, C, T, H, W)
        
        # 3차원 누비기 쓰기
        out = self.conv(x)
        
        return out


def demonstrate_2d_vs_3d_convolution():
    """
    2차원 누비기와 3차원 누비기의 차이를 보여 준다.
    """
    print("\n" + "="*80)
    print("2D vs 3D CONVOLUTION COMPARISON")
    print("="*80)
    
    # 보기 영상 만들기
    B, T, C, H, W = 2, 16, 3, 64, 64
    video = torch.randn(B, T, C, H, W)
    
    print(f"\nInput video shape: {video.shape}")
    print(f"  B={B} (batch), T={T} (time), C={C} (channels)")
    print(f"  H={H} (height), W={W} (width)")
    
    # 2차원 누비기
    print("\n1. Applying 2D Convolution (frame-by-frame)...")
    conv2d = Conv2DExample(in_channels=C, out_channels=64)
    
    # 매개변수 개수 세기
    params_2d = sum(p.numel() for p in conv2d.parameters())
    print(f"   Parameters: {params_2d}")
    print(f"   Kernel size: (3, 3) - spatial only")
    
    output_2d = conv2d(video)
    print(f"   Output shape: {output_2d.shape}")
    print(f"   ✗ Frames processed independently - no temporal modeling")
    
    # 3차원 누비기
    print("\n2. Applying 3D Convolution (spatiotemporal)...")
    conv3d = Conv3DExample(in_channels=C, out_channels=64)
    
    # Conv3d에 맞게 다시 늘어놓기: (B, T, C, H, W) → (B, C, T, H, W)
    video_3d = video.permute(0, 2, 1, 3, 4)
    
    params_3d = sum(p.numel() for p in conv3d.parameters())
    print(f"   Parameters: {params_3d}")
    print(f"   Kernel size: (3, 3, 3) - spatiotemporal")
    
    output_3d = conv3d(video_3d)
    print(f"   Output shape: {output_3d.shape}")
    print(f"   ✓ Temporal dimension processed - learns motion!")
    
    # 매개변수 견줌
    print(f"\n3. Parameter Comparison:")
    print(f"   2D Conv: {params_2d:,} parameters")
    print(f"   3D Conv: {params_3d:,} parameters")
    print(f"   Ratio: 3D has {params_3d / params_2d:.1f}x more parameters")
    print(f"   Reason: 3D kernel has additional temporal dimension")


#=============================================================================
# 2부: 3차원 누비기 덩이
#=============================================================================

class Conv3DBlock(nn.Module):
    """
    BatchNorm과 ReLU를 갖춘 기본 3차원 누비기 덩이.
    
    구조:
        Conv3D → BatchNorm3D → ReLU → MaxPool3D
        
    이것이 대부분의 3차원 누비기 신경망 얼개의 벽돌이다.
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int, int] = (3, 3, 3),
                 stride: Tuple[int, int, int] = (1, 1, 1),
                 padding: Tuple[int, int, int] = (1, 1, 1),
                 use_pooling: bool = True):
        """
        3차원 누비기 덩이를 첫자리매김한다.
        
        인수:
            in_channels: 들임 채널 차원
            out_channels: 내놓는 채널 차원
            kernel_size: 때, 높이, 너비를 뜻하는 (t, h, w)
            stride: 차원마다의 성큼
            padding: 차원마다의 덧대기
            use_pooling: 최대 모으기를 쓸지 여부
        """
        super().__init__()
        
        # 3차원 누비기
        # 무게 꼴: (out_channels, in_channels, t, h, w)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False  # BatchNorm이 치우침을 다룬다
        )
        
        # 3차원 묶음 고르게 맞추기
        # 자리와 때 차원에 걸쳐 고르게 맞춘다
        # 채널마다 따로 통계량을 지닌다
        self.bn = nn.BatchNorm3d(out_channels)
        
        # ReLU 활성화
        self.relu = nn.ReLU(inplace=True)
        
        # 있어도 되는 3차원 최대 모으기
        # 자리와 때 차원을 줄인다
        self.use_pooling = use_pooling
        if use_pooling:
            self.pool = nn.MaxPool3d(
                kernel_size=(2, 2, 2),  # 때와 자리에 걸쳐 모으기
                stride=(2, 2, 2)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        누비기 덩이를 지나는 앞먹임.
        
        인수:
            x: 들임 텐서 (B, C, T, H, W)
            
        반환값:
            내놓는 텐서 (B, out_C, T', H', W')
            
        차원 바뀜:
            - 누비기: 성큼에 따라 크기를 지키거나 줄인다
            - 모으기: 차원마다 2분의 1로 줄인다
        """
        # 누비기: 자리와 때의 특징을 배운다
        out = self.conv(x)
        
        # 묶음 고르게 맞추기: 익히기를 든든하게 한다
        out = self.bn(out)
        
        # 깨어남: 비선형을 들여온다
        out = self.relu(out)
        
        # 모으기: 차원을 줄이고 받는 자리를 넓힌다
        if self.use_pooling:
            out = self.pool(out)
        
        return out


#=============================================================================
# 3부: C3D 얼개(고전 3차원 누비기 신경망)
#=============================================================================

class C3D(nn.Module):
    """
    C3D: 3차원 누비기 그물로 자리와 때의 특징 배우기.
    
    논문: Tran et al. "Learning Spatiotemporal Features with 3D Convolutional
           Networks" (ICCV 2015)
    
    구조:
        3x3x3 알맹이를 쓴 누비기 층 8개
        최대 모으기 층 5개
        온전히 이은 층 2개
        
    핵심 눈썰미: 그물 전체에 3x3x3 알맹이를 쓰는 것이
                 자리와 때의 특징을 담아내는 데 가장 좋다
    
    들임: 112x112 RGB 영상 틀 16개
    내놓음: 갈래 확률
    """
    
    def __init__(self, num_classes: int = 400, dropout: float = 0.5):
        """
        C3D 그물을 첫자리매김한다.
        
        인수:
            num_classes: 몸짓 갈래의 개수
            dropout: 온전히 이은 층의 떨구기 확률
        """
        super().__init__()
        
        # 1층: Conv3d (3→64)
        # 들임: (B, 3, 16, 112, 112)
        self.conv1 = Conv3DBlock(3, 64, use_pooling=True)
        # 내놓음: 모으기 뒤 (B, 64, 8, 56, 56)
        
        # 2층: Conv3d (64→128)
        self.conv2 = Conv3DBlock(64, 128, use_pooling=True)
        # 내놓음: (B, 128, 4, 28, 28)
        
        # 3a, 3b층: Conv3d (128→256)
        self.conv3a = Conv3DBlock(128, 256, use_pooling=False)
        self.conv3b = Conv3DBlock(256, 256, use_pooling=True)
        # 내놓음: (B, 256, 2, 14, 14)
        
        # 4a, 4b층: Conv3d (256→512)
        self.conv4a = Conv3DBlock(256, 512, use_pooling=False)
        self.conv4b = Conv3DBlock(512, 512, use_pooling=True)
        # 내놓음: (B, 512, 1, 7, 7)
        
        # 5a, 5b층: Conv3d (512→512)
        self.conv5a = Conv3DBlock(512, 512, use_pooling=False)
        self.conv5b = Conv3DBlock(512, 512, use_pooling=True)
        # 내놓음: (B, 512, 1, 4, 4) — 때 차원이 1로 줄었음에 유의
        
        # 완전 연결층
        # 펴기: 512 * 1 * 4 * 4 = 8192
        self.fc1 = nn.Linear(512 * 1 * 4 * 4, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        C3D를 지나는 앞먹임.
        
        인수:
            x: 들임 영상 (B, C, T, H, W)
               기대: (B, 3, 16, 112, 112)
               
        반환값:
            갈래 로짓 (B, num_classes)
        """
        # 누비기 층 — 자리와 때의 특징 뽑기
        x = self.conv1(x)    # (B, 64, 8, 56, 56)
        x = self.conv2(x)    # (B, 128, 4, 28, 28)
        
        x = self.conv3a(x)   # (B, 256, 4, 28, 28)
        x = self.conv3b(x)   # (B, 256, 2, 14, 14)
        
        x = self.conv4a(x)   # (B, 512, 2, 14, 14)
        x = self.conv4b(x)   # (B, 512, 1, 7, 7)
        
        x = self.conv5a(x)   # (B, 512, 1, 7, 7)
        x = self.conv5b(x)   # (B, 512, 1, 4, 4)
        
        # 온전히 이은 층을 위해 펴기
        x = x.flatten(start_dim=1)  # (B, 512*1*4*4)
        
        # 온전히 이은 층 — 갈래 매기기
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.fc3(x)  # 로짓
        
        return x


#=============================================================================
# 4부: 잔차 3차원 누비기 신경망(R3D)
#=============================================================================

class Residual3DBlock(nn.Module):
    """
    3차원 누비기를 위한 잔차 덩이.
    
    구조:
        x → Conv3D → BN → ReLU → Conv3D → BN → (+) → ReLU
        └──────────────────────────────────────┘
        
    잔차 이음: F(x) + x
    
    이점:
        1. 기울기가 더 잘 흐른다(깊은 그물 익히기를 돕는다)
        2. 특징을 더 잘 배운다
        3. 나빠지는 문제를 줄인다
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # 첫 합성곱 블록
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        # 둘째 합성곱 블록
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # ReLU
        self.relu = nn.ReLU(inplace=True)
        
        # 지름길 이음(차원이 안 맞을 때)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        잔차 이음을 쓴 앞먹임.
        
        인수:
            x: 들임 텐서 (B, C, T, H, W)
            
        반환값:
            내놓는 텐서 (B, out_C, T, H, W)
            
        수학 연산:
            out = ReLU(F(x) + x)
            여기서 F(x)은 잔차 함수
        """
        # 잔차 이음에 쓸 들임 갈무리
        identity = x
        
        # 첫 합성곱 블록
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # 둘째 합성곱 블록
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 잔차 이음 더하기
        identity = self.shortcut(identity)
        out += identity
        
        # 마지막 활성화
        out = self.relu(out)
        
        return out


#=============================================================================
# 5부: 그려 보기와 살피기
#=============================================================================

def visualize_3d_kernels(model: nn.Module):
    """
    배운 3차원 누비기 알맹이를 그려 본다.
    
    인수:
        model: 3차원 누비기 신경망 모델
    """
    # 첫 누비기 층 얻기
    first_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv3d):
            first_conv = module
            break
    
    if first_conv is None:
        print("No Conv3d layer found")
        return
    
    # 무게 얻기: (out_C, in_C, T, H, W)
    weights = first_conv.weight.data.cpu()
    
    print(f"\nFirst Conv3d layer:")
    print(f"  Weight shape: {weights.shape}")
    print(f"  Kernel size: {first_conv.kernel_size}")
    
    # 알맹이 일부 그려 보기
    num_filters = min(8, weights.shape[0])
    num_temporal = weights.shape[2]
    
    fig, axes = plt.subplots(num_filters, num_temporal, figsize=(12, 10))
    
    for i in range(num_filters):
        for t in range(num_temporal):
            ax = axes[i, t] if num_filters > 1 else axes[t]
            
            # 때 t의 알맹이 얻기(들임 채널로 고루냄)
            kernel = weights[i, :, t, :, :].mean(dim=0)  # (H, W)
            
            # 그려 보려고 고르게 하기
            kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min() + 1e-8)
            
            ax.imshow(kernel.numpy(), cmap='viridis')
            ax.axis('off')
            
            if t == 0:
                ax.set_ylabel(f'Filter {i}', rotation=0, labelpad=30)
            if i == 0:
                ax.set_title(f't={t}')
    
    plt.tight_layout()
    plt.savefig('/home/claude/34_video_understanding/02_3d_kernels.png',
                dpi=150, bbox_inches='tight')
    print(f"Kernel visualization saved to 02_3d_kernels.png")
    plt.close()


def analyze_feature_maps(model: nn.Module, video: torch.Tensor):
    """
    3차원 누비기 신경망의 가운데 특징 지도를 살핀다.
    
    인수:
        model: 3차원 누비기 신경망 모델
        video: 들임 영상 텐서
    """
    print("\nAnalyzing feature maps...")
    
    # 가운데 내놓음을 붙잡으려 갈고리 걸기
    activations = {}
    
    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    # 갈고리 걸기
    layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv3d):
            module.register_forward_hook(get_activation(name))
            layer_names.append(name)
    
    # 순전파
    with torch.no_grad():
        _ = model(video)
    
    # 깨어남 꼴 찍기
    print(f"\nFeature map shapes:")
    for name in layer_names[:5]:  # 처음 5개 층 보이기
        if name in activations:
            shape = activations[name].shape
            print(f"  {name}: {shape}")


def demonstrate_c3d():
    """
    C3D 얼개와 그 성질을 보여 준다.
    """
    print("\n" + "="*80)
    print("C3D ARCHITECTURE DEMONSTRATION")
    print("="*80)
    
    # C3D 모델 만들기
    print("\n1. Creating C3D model...")
    model = C3D(num_classes=101)  # UCF-101에는 갈래가 101개 있다
    
    # 매개변수 개수 세기
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1e6:.1f} MB (fp32)")
    
    # 보기 들임 만들기
    print("\n2. Testing with sample video...")
    batch_size = 2
    video = torch.randn(batch_size, 3, 16, 112, 112)  # (B, C, T, H, W)
    print(f"   Input shape: {video.shape}")
    
    # 순전파
    model.eval()
    with torch.no_grad():
        output = model(video)
    
    print(f"   Output shape: {output.shape}")
    print(f"   Output: class logits for {output.shape[1]} classes")
    
    # 확률을 얻으려 소프트맥스 쓰기
    probs = torch.softmax(output, dim=1)
    top5_probs, top5_indices = torch.topk(probs[0], 5)
    
    print(f"\n3. Top 5 predictions for first video:")
    for i, (prob, idx) in enumerate(zip(top5_probs, top5_indices)):
        print(f"   {i+1}. Class {idx.item()}: {prob.item():.4f}")
    
    # 알맹이 그려 보기
    print("\n4. Visualizing learned kernels...")
    visualize_3d_kernels(model)
    
    # 특징 지도 살피기
    analyze_feature_maps(model, video)


#=============================================================================
# 6부: 쓰는 보기와 보임
#=============================================================================

def main():
    """
    3차원 누비기를 보여 주는 주된 실행 함수.
    """
    print(__doc__)
    
    # 난수 씨앗 고정
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 보임 1: 2차원과 3차원 견줌
    demonstrate_2d_vs_3d_convolution()
    
    # 보임 2: C3D 얼개
    demonstrate_c3d()
    
    # 요약
    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print("""
    1. 3차원 누비기:
       - 2차원 누비기를 때 차원으로 넓힌다
       - 알맹이: (t, h, w)가 자리·때 덩어리를 다룬다
       - 날 화소에서 움직임 무늬를 곧바로 배운다
    
    2. 2차원과 3차원의 맞바꿈:
       - 2차원: 더 빠르고 기억 공간이 적으며 때를 나타내지 않는다
       - 3차원: 더 느리고 매개변수가 많으며 움직임을 담아낸다
       - 때 알맹이 차원 때문에 3차원은 매개변수가 약 3배 많다
    
    3. C3D 얼개:
       - 3x3x3 알맹이를 쓴 누비기 층 8개(실제로 가장 좋다)
       - 자리와 때 차원을 차츰 줄인다
       - 들임: 112x112 RGB 틀 16개
       - 갈래 101개에 매개변수 약 7800만
    
    4. 잔차 3차원 덩이:
       - 아주 깊은 3차원 누비기 신경망을 익힐 수 있게 한다
       - 잔차 이음으로 기울기가 더 잘 흐른다
       - 요즘 얼개(R3D, I3D)에 쓰인다
    
    5. 실전에서 헤아릴 점:
       - 3차원 누비기 신경망은 셈 값이 비싸다
       - 익히려면 힘센 GPU가 필요하다
       - 기억 공간 때문에 묶음 크기가 제한된다(자리·때 덩어리가 크다)
       - 이점: 날 영상에서 끝에서 끝까지 배운다
    
    다음: 3차원 누비기 신경망으로 단순 영상 갈래 매개를 세운다!
    """)


if __name__ == "__main__":
    main()```

## 논의

여기 짠 것은 함께 어울려 온전한 영상 이해 얼개를 이루는 클래스 5개(`Conv2DExample`, `Conv3DExample`, `Conv3DBlock`, `C3D`, 그 밖 1개)를 정한다. 클래스마다 뚜렷한 조각 하나를 감싸므로 코드가 단원별로 나뉘고 넓히기 쉽다. `forward` 메서드가 PyTorch의 자동 미분에 쓰이는 셈 그래프를 정한다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 컴퓨터 비전 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
`Conv2DExample`의 앞먹임을 따라가며 텐서 꼴을 좇아라. 붙박이 매개변수로 들임 표본 4개짜리 묶음에 대해 주요 연산(누비기, 모으기, 선형 층)마다 그 뒤의 꼴을 적어라.

??? success "연습문제 1 풀이"
    입력 모양에서 출발하여 각 층을 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 (덧대기가 없으면) $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌거나 (`padding=k//2`이면) 그대로 유지된다. 커널이 2인 풀링은 공간 차원을 절반으로 만든다. 선형 층은 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임에 유의하며 추적한다. 중간 모양을 합성곱 층에서는 $(B, C, H, W)$로, 평탄화 후에는 $(B, F)$로 적는다.

---

**연습문제 2.**
$64 \times 64$ 크기의 RGB 이미지(입력 모양 $3 \times 64 \times 64$)를 받도록 구조를 수정하라. 모든 층의 차원을 그에 맞게 고치고 모델이 오류 없이 실행되는지 확인하라.

??? success "연습문제 2 풀이"
    Change the first convolutional layer's `in_channels` from the current value to 3. Recalculate the spatial dimensions after each convolution and pooling layer using the formulas $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$. Update the first linear layer's `in_features` to match the flattened output of the last convolutional/pooling layer. Verify with: `model = Conv2DExample(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`.

---

**연습문제 3.**
같은 입출력 차원에서 표준 합성곱과 깊이별 분리 합성곱의 매개변수 개수와 FLOPs를 비교하라. 계산 절감이 가장 큰 것은 언제인가?

??? success "연습문제 3 풀이"
    표준 `Conv2d(C_in, C_out, k)`은 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개의 매개변수를 갖는다. 깊이별 분리 합성곱은 이를 둘로 나눈다. (1) 깊이별: $C_{{\text{{in}}}} \times k^2$개(입력 채널마다 필터 하나), (2) 점별: $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 합성곱)이다. 매개변수의 비는 대략 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 매개변수가 약 $8{-}9\times$ 적어진다. 절감은 $C_{{\text{{out}}}}$과 $k$가 모두 클 때 가장 크다.

---

**연습문제 4.**
층이나 덩이의 개수를 정할 수 있도록 `Conv2DExample`을 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`로 깊이가 바뀌는 얼개를 만들어라. 층 2, 4, 8개로 시험하여라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 되풀이한다. (그냥 파이썬 목록이 아니라) `nn.ModuleList`를 써야 PyTorch가 가장 좋게 하기에 쓸 매개변수를 모두 등록한다. 시험: `for n in [2, 4, 8]: model = Conv2DExample(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
