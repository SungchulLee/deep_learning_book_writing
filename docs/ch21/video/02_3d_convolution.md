# 3차원 합성곱과 시공간 특징

프레임별 2차원 합성곱은 각 이미지의 공간 이웃을 처리하고, 3차원 합성곱은 시간 이웃까지 함께 처리한다. 여기의 세 번째 축은 장면의 물리적 깊이가 아니라 프레임 순서다. 이 절은 단일 합성곱·3차원 블록·C3D 형태의 분류기를 비교하며 학습이나 사전 학습 가중치 로드는 수행하지 않는다.

## 1. 코드와 기존 출력

??? note "코드"

    ```python
    """
    영상 이해 — 첫걸음 수준
    파일 02: 3차원 누비기 — 자리와 때에 걸친 특징 뽑기

    이 파일은 영상을 위한 3차원 누비기 그물을 다룬다:
    - 3차원 누비기와 2차원 누비기 견주어 이해하기
    - 자리와 때의 특징 뽑기
    - PyTorch로 3차원 누비기 신경망 짜기
    - C3D(3차원 누비기) 구조
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
                # 필요하면 배치 차원을 더한다
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

        이것이 대부분의 3차원 누비기 신경망 구조의 벽돌이다.
        """

        def __init__(self,
                     in_channels: int,
                     out_channels: int,
                     kernel_size: Tuple[int, int, int] = (3, 3, 3),
                     stride: Tuple[int, int, int] = (1, 1, 1),
                     padding: Tuple[int, int, int] = (1, 1, 1),
                     use_pooling: bool = True,
                     pool_temporal: bool = True):
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

            # 3차원 배치 고르게 맞추기
            # 자리와 때 차원에 걸쳐 고르게 맞춘다
            # 채널마다 따로 통계량을 지닌다
            self.bn = nn.BatchNorm3d(out_channels)

            # ReLU 활성화
            self.relu = nn.ReLU(inplace=True)

            # 있어도 되는 3차원 최대 모으기
            # 자리와 때 차원을 줄인다
            self.use_pooling = use_pooling
            if use_pooling:
                # 마지막 켜에서는 때 차원이 이미 1이라 또 반으로 줄이면 0이 되어
                # 터진다. 그때는 자리만 모은다(참 C3D도 이렇게 한다).
                pool_k = (2, 2, 2) if pool_temporal else (1, 2, 2)
                self.pool = nn.MaxPool3d(
                    kernel_size=pool_k,
                    stride=pool_k,
                    ceil_mode=True
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

            # 배치 고르게 맞추기: 익히기를 든든하게 한다
            out = self.bn(out)

            # 활성화: 비선형을 들여온다
            out = self.relu(out)

            # 모으기: 차원을 줄이고 받는 자리를 넓힌다
            if self.use_pooling:
                out = self.pool(out)

            return out


    #=============================================================================
    # 3부: C3D 구조(고전 3차원 누비기 신경망)
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
            # 때 차원이 이미 1이므로 자리만 모은다
            self.conv5b = Conv3DBlock(512, 512, use_pooling=True, pool_temporal=False)
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
        plt.savefig('02_3d_kernels.png',
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

        # 활성화 꼴 찍기
        print(f"\nFeature map shapes:")
        for name in layer_names[:5]:  # 처음 5개 층 보이기
            if name in activations:
                shape = activations[name].shape
                print(f"  {name}: {shape}")


    def demonstrate_c3d():
        """
        C3D 구조와 그 성질을 보여 준다.
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

        # 보임 2: C3D 구조
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

        3. C3D 구조:
           - 3x3x3 알맹이를 쓴 누비기 층 8개(실제로 가장 좋다)
           - 자리와 때 차원을 차츰 줄인다
           - 들임: 112x112 RGB 틀 16개
           - 갈래 101개에 매개변수 약 7800만

        4. 잔차 3차원 덩이:
           - 아주 깊은 3차원 누비기 신경망을 익힐 수 있게 한다
           - 잔차 이음으로 기울기가 더 잘 흐른다
           - 요즘 구조(R3D, I3D)에 쓰인다

        5. 실전에서 헤아릴 점:
           - 3차원 누비기 신경망은 셈 값이 비싸다
           - 익히려면 힘센 GPU가 필요하다
           - 기억 공간 때문에 배치 크기가 제한된다(자리·때 덩어리가 크다)
           - 이점: 날 영상에서 끝에서 끝까지 배운다

        다음: 3차원 누비기 신경망으로 단순 영상 갈래 매개를 세운다!
        """)


    if __name__ == "__main__":
        main()
    ```

??? note "기존 전체 출력"

    ```

    영상 이해 — 첫걸음 수준
    파일 02: 3차원 누비기 — 자리와 때에 걸친 특징 뽑기

    이 파일은 영상을 위한 3차원 누비기 그물을 다룬다:
    - 3차원 누비기와 2차원 누비기 견주어 이해하기
    - 자리와 때의 특징 뽑기
    - PyTorch로 3차원 누비기 신경망 짜기
    - C3D(3차원 누비기) 구조
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


    ================================================================================
    2D vs 3D CONVOLUTION COMPARISON
    ================================================================================

    Input video shape: torch.Size([2, 16, 3, 64, 64])
      B=2 (batch), T=16 (time), C=3 (channels)
      H=64 (height), W=64 (width)

    1. Applying 2D Convolution (frame-by-frame)...
       Parameters: 1792
       Kernel size: (3, 3) - spatial only
       Output shape: torch.Size([2, 16, 64, 64, 64])
       ✗ Frames processed independently - no temporal modeling

    2. Applying 3D Convolution (spatiotemporal)...
       Parameters: 5248
       Kernel size: (3, 3, 3) - spatiotemporal
       Output shape: torch.Size([2, 64, 16, 64, 64])
       ✓ Temporal dimension processed - learns motion!

    3. Parameter Comparison:
       2D Conv: 1,792 parameters
       3D Conv: 5,248 parameters
       Ratio: 3D has 2.9x more parameters
       Reason: 3D kernel has additional temporal dimension

    ================================================================================
    C3D ARCHITECTURE DEMONSTRATION
    ================================================================================

    1. Creating C3D model...
       Total parameters: 78,412,325
       Trainable parameters: 78,412,325
       Model size: ~313.6 MB (fp32)

    2. Testing with sample video...
       Input shape: torch.Size([2, 3, 16, 112, 112])
       Output shape: torch.Size([2, 101])
       Output: class logits for 101 classes

    3. Top 5 predictions for first video:
       1. Class 27: 0.0101
       2. Class 24: 0.0101
       3. Class 53: 0.0101
       4. Class 67: 0.0101
       5. Class 23: 0.0101

    4. Visualizing learned kernels...

    First Conv3d layer:
      Weight shape: torch.Size([64, 3, 3, 3, 3])
      Kernel size: (3, 3, 3)
    Kernel visualization saved to 02_3d_kernels.png

    Analyzing feature maps...

    Feature map shapes:
      conv1.conv: torch.Size([2, 64, 16, 112, 112])
      conv2.conv: torch.Size([2, 128, 8, 56, 56])
      conv3a.conv: torch.Size([2, 256, 4, 28, 28])
      conv3b.conv: torch.Size([2, 256, 4, 28, 28])
      conv4a.conv: torch.Size([2, 512, 2, 14, 14])

    ================================================================================
    KEY TAKEAWAYS
    ================================================================================

        1. 3차원 누비기:
           - 2차원 누비기를 때 차원으로 넓힌다
           - 알맹이: (t, h, w)가 자리·때 덩어리를 다룬다
           - 날 화소에서 움직임 무늬를 곧바로 배운다
        
        2. 2차원과 3차원의 맞바꿈:
           - 2차원: 더 빠르고 기억 공간이 적으며 때를 나타내지 않는다
           - 3차원: 더 느리고 매개변수가 많으며 움직임을 담아낸다
           - 때 알맹이 차원 때문에 3차원은 매개변수가 약 3배 많다
        
        3. C3D 구조:
           - 3x3x3 알맹이를 쓴 누비기 층 8개(실제로 가장 좋다)
           - 자리와 때 차원을 차츰 줄인다
           - 들임: 112x112 RGB 틀 16개
           - 갈래 101개에 매개변수 약 7800만
        
        4. 잔차 3차원 덩이:
           - 아주 깊은 3차원 누비기 신경망을 익힐 수 있게 한다
           - 잔차 이음으로 기울기가 더 잘 흐른다
           - 요즘 구조(R3D, I3D)에 쓰인다
        
        5. 실전에서 헤아릴 점:
           - 3차원 누비기 신경망은 셈 값이 비싸다
           - 익히려면 힘센 GPU가 필요하다
           - 기억 공간 때문에 배치 크기가 제한된다(자리·때 덩어리가 크다)
           - 이점: 날 영상에서 끝에서 끝까지 배운다
        
        다음: 3차원 누비기 신경망으로 단순 영상 갈래 매개를 세운다!
        
    ```

## 2. 축 순서와 시간 결합

`Conv2DExample`은 배치 입력 B×T×C×H×W를 BT개의 이미지로 펼쳐 같은 2D 필터를 적용하고 B×T×C_out×H×W로 되돌린다. 프레임 간 값을 섞지 않으므로 이 층 자체에는 시간 상호작용이 없다. 이후 LSTM·시간 attention 등을 추가할 수 있어 2D 특징을 쓰는 모든 모델이 시간 정보를 이용하지 못한다는 뜻은 아니다.

`Conv3DExample`은 B×C×T×H×W를 받는다. 4차원 입력에는 배치를 추가하므로 이 경우에도 C×T×H×W여야 한다. T×C×H×W로 입력하면 채널·시간을 혼동한다. padding1의 3×3×3 합성곱은 시공간 크기를 유지하고 인접 세 프레임에서 특징을 결합한다. 아직 무작위 초기 가중치이므로 순전파 한 번만으로 움직임을 학습했다고 말할 수 없다.

같은 입출력 채널에서 bias 없는 2D와 3D의 가중치 수는 C_in C_out k_h k_w와 C_in C_out k_t k_h k_w다. k_t=3이면 가중치는 세 배지만 bias는 양쪽 모두 C_out개여서 전체 매개변수 비율은 정확히 3이 아니다. 기본 3→64에서 2D는 1792개, 3D는 5248개다. MAC 수에는 출력의 시간·공간 위치 수도 반영된다.

## 3. 블록과 C3D 형태의 예제

`Conv3DBlock`은 bias 없는 3D 합성곱·BatchNorm3d·ReLU와 선택적 풀링이다. 시간 풀링을 사용하면 T·H·W를 함께 줄이고, 사용하지 않으면 공간만 줄인다. ceil_mode=True이므로 홀수 크기는 바닥 나눗셈과 다르게 줄어든다. 배치 정규화의 학습 가능한 γ·β도 매개변수에 포함한다.

C3D 예제에 B×3×16×112×112를 넣으면 풀링 뒤 크기는 채널·시간·공간 순서로 64×8×56×56, 128×4×28×28, 256×2×14×14, 512×1×7×7, 512×1×4×4다. 펼친 8192차원이 fc1과 맞는다. 기본 생성자의 클래스 수는 400이고 시연에서는 101로 바꾸어 출력 B×101을 얻는다. 임의 T·H·W를 받아도 fc1과 항상 맞는 구조는 아니다.

이 구현은 [원 C3D](https://arxiv.org/abs/1412.0767)의 모든 풀링·정규화 세부를 재현하지 않는다. 특히 여기서는 첫 블록에서 시간을 줄이고 마지막 블록에서 공간만 줄인다. ‘C3D’ 이름과 기존 요약 문구를 그대로 원 논문의 재현이나 학습된 특징으로 받아들이지 않는다.

## 4. 잔차 경로와 시각화

`Residual3DBlock`은 공간·시간 크기를 유지하고 채널이 다를 때 1×1×1 projection으로 shortcut을 맞춘다. 더한 뒤 ReLU를 적용하므로 모든 입력에 대해 블록 전체가 항등변환인 것은 아니다. 이 블록은 별도로 정의되어 있으며 위 C3D가 실제로 사용하는 것은 아니다.

커널 그림은 초기 가중치를 채널 평균·필터별 min-max로 보여 준다. 학습된 움직임 검출기의 시각화가 아니며 필터마다 따로 스케일을 조정해 절대 크기를 비교하기도 어렵다. 무작위 입력의 top5 확률도 인식 성능이 아니다. feature hook은 제거하지 않아 같은 모델에 분석을 반복하면 hook이 누적될 수 있다.

출력의 매개변수 수×4바이트는 FP32 가중치 저장량의 근사다. 활성값·경사·옵티마이저 상태를 포함한 학습 메모리가 아니고 모든 환경의 파일 크기도 아니다. 3D 연산의 실제 비용은 시간 커널·stride·특징 크기·구현에 따라 비교한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
B=4,T=16,C=3,H=W=64에서 두 단일 합성곱 예제의 입출력을 비교하라.

</div>

??? success "연습문제 1 풀이"

    2D 예제는 4×16×3×64×64를 받아 4×16×64×64×64를 반환한다. 3D 예제는 입력을 4×3×16×64×64로 바꾸어 4×64×16×64×64를 얻는다. 같아 보이는 크기 64도 축의 의미를 구분해야 한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
C3D 예제에 16프레임 64×64 입력을 넣으면 fc1이 맞는가?

</div>

??? success "연습문제 2 풀이"

    풀링 뒤 공간은 64→32→16→8→4→2이고 시간은 16→8→4→2→1이다. 최종 512×1×2×2=2048이므로 고정 fc1의 8192와 다르다. 합성곱이 RGB를 받는다는 것만으로 전체 모델 입력이 호환되지는 않는다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
3→64, 3×3과 3×3×3 합성곱의 매개변수 수를 bias 포함해 계산하라.

</div>

??? success "연습문제 3 풀이"

    2D는 3×64×9+64=1792, 3D는 3×64×27+64=5248이다. 가중치 수는 정확히 세 배지만 공통 bias 때문에 전체 비율은 약 2.929다. 출력 위치 수와 MAC 계산도 별도로 고려한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 ModuleList 조각으로 블록 수를 바꿀 때 C3D의 선형 입력 조건을 어떻게 다루어야 하는가?

</div>

??? success "연습문제 4 풀이"

    YourBlock 정의·채널과 시간·공간 stride를 정하고 마지막 특징의 펼친 크기를 fc1에 맞춰야 한다. 필요하면 목표 크기의 적응형 풀링을 설계할 수 있지만 원 구조와 달라진다. 등록된 모듈을 반복하는 것만으로 고정8192 입력 문제가 해결되지는 않는다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

3D 합성곱은 시간 이웃을 결합하지만 학습되지 않은 순전파가 움직임 인식을 입증하지는 않는다. BCTHW·BTCHW 계약, 풀링의 시간·공간 변화와 고정 선형 입력을 추적하고 매개변수·MAC·활성 메모리를 구분한다.
