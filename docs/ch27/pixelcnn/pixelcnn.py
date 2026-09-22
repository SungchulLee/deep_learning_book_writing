"""pixelcnn — pixelcnn 페이지의 코드를 모듈로 쓸 수 있게 옮겨 놓은 것이다.

이 파일은 ch26/pixelcnn/pixelcnn.md 에서 자동으로 뽑아낸 것이므로, 고칠 일이 있으면
그 페이지를 고치고 다시 뽑아내는 편이 낫다.
"""

"""
자기 되돌이 그림 만들어 내기를 위한 단순한 PixelCNN

PixelCNN은 가로 훑기 차례(왼쪽에서 오른쪽, 위에서 아래)로 화소 하나씩 그림을 만든다.
화소마다 앞서 만든 모든 화소를 바탕으로 헤아린다.

이는 핵심 개념에 집중한 가르치기 위한 단순한 판이다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ========================================================================
# 메인
# ========================================================================


class MaskedConv2d(nn.Conv2d):
    """
    자기 되돌이 그림 만들어 내기를 위한 가린 겹말기.
    
    PixelCNN의 핵심 새로움: 화소마다 앞선 화소(위쪽과 왼쪽)에만
    매이도록 겹말기를 가린다.
    
    가림막 갈래:
    - A 갈래: 첫 층용이며 지금 화소를 뺀다
    - B 갈래: 뒤 층용이며 지금 화소를 넣는다
    """
    
    def __init__(self, mask_type: str, *args, **kwargs):
        """
        가린 겹말기를 첫자리매김한다.
        
        인수:
            mask_type: 'A'나 'B'
            *args, **kwargs: nn.Conv2d에 줄 인자
        """
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        
        assert mask_type in ['A', 'B'], "mask_type must be 'A' or 'B'"
        self.mask_type = mask_type
        
        # 가림막을 버퍼로 등록한다(익히는 동안 고쳐지지 않는다)
        self.register_buffer('mask', torch.zeros_like(self.weight))
        self.create_mask()
    
    def create_mask(self):
        """
        자기 되돌이 가림막을 만든다.
        
        가림막은 다음을 보장한다.
        - 위쪽 화소는 볼 수 있다
        - 왼쪽 화소는 볼 수 있다
        - 지금 화소: B 가림막에서만 보인다
        - 아래쪽과 오른쪽 화소는 볼 수 없다
        """
        # 차원을 얻는다
        # 무게 꼴: [내놓기 채널, 들임 채널, 알맹이 높이, 알맹이 너비]
        k_h, k_w = self.weight.shape[2:]
        
        # 가림막을 모두 1로 첫자리매김한다
        self.mask.fill_(1)
        
        # 아래 반을 0으로 만든다
        self.mask[:, :, k_h // 2 + 1:, :] = 0
        
        # 가운데 줄의 오른쪽을 0으로 만든다
        # A 가림막: 가운데 화소를 뺀다
        # B 가림막: 가운데 화소를 넣는다
        if self.mask_type == 'A':
            self.mask[:, :, k_h // 2, k_w // 2:] = 0
        else:  # mask_type == 'B'
            self.mask[:, :, k_h // 2, k_w // 2 + 1:] = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        가린 무게로 하는 앞먹임.
        
        인수:
            x: 입력 텐서
            
        반환값:
            가린 겹말기의 내놓기
        """
        # 겹말기 앞에 무게에 가림막을 곱한다
        # 그러면 허락된 화소만 쓰인다
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class ResidualBlock(nn.Module):
    """
    가린 겹말기를 갖춘 남은 덩이.
    
    짜임:
        들임 -> 가린 겹말기 -> ReLU -> 가린 겹말기 -> 들임과 더하기
    """
    
    def __init__(self, channels: int):
        """
        남은 덩이를 첫자리매김한다.
        
        인수:
            channels: 채널의 수
        """
        super(ResidualBlock, self).__init__()
        
        # 첫 겹말기 뒤로는 모두 B 갈래이다
        self.conv1 = MaskedConv2d('B', channels, channels // 2, 
                                  kernel_size=1, padding=0)
        self.conv2 = MaskedConv2d('B', channels // 2, channels // 2,
                                  kernel_size=3, padding=1)
        self.conv3 = MaskedConv2d('B', channels // 2, channels,
                                  kernel_size=1, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """남은 덩이를 지나는 앞먹임."""
        residual = x
        
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        
        # 잔차 연결
        return out + residual


class PixelCNN(nn.Module):
    """
    두값(검정/흰색) 그림 만들어 내기를 위한 단순한 PixelCNN.
    
    이 자기 되돌이 모델은 화소 하나씩 그림을 만든다.
    P(그림) = P(x₁) × P(x₂|x₁) × P(x₃|x₁,x₂) × ... × P(xₙ|x₁,...,xₙ₋₁)
    
    여기서 xᵢ은 화소 값이다.
    """
    
    def __init__(self, 
                 n_channels: int = 64,
                 n_residual_blocks: int = 5):
        """
        PixelCNN을 첫자리매김한다.
        
        인수:
            n_channels: 특징 채널의 수
            n_residual_blocks: 남은 덩이의 수
        """
        super(PixelCNN, self).__init__()
        
        # 첫 층은 A 갈래 가림막을 쓴다(지금 화소를 뺀다)
        self.input_conv = MaskedConv2d('A', 1, n_channels,
                                       kernel_size=7, padding=3)
        
        # 잔차 블록
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(n_channels) for _ in range(n_residual_blocks)
        ])
        
        # 내놓기 층
        self.output_conv1 = MaskedConv2d('B', n_channels, n_channels,
                                         kernel_size=1)
        self.output_conv2 = MaskedConv2d('B', n_channels, n_channels,
                                         kernel_size=1)
        
        # 마지막 층: 화소마다 확률을 헤아린다
        # 두값 그림에서는 채널 1개를 내놓는다(흰색일 확률)
        self.final_conv = MaskedConv2d('B', n_channels, 1,
                                       kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        PixelCNN을 지나는 앞먹임.
        
        인수:
            x: 들임 그림 [묶음 크기, 1, 높이, 너비]
               [0, 1] 안의 값
               
        반환값:
            화소마다의 로짓 [묶음 크기, 1, 높이, 너비]
            확률을 얻으려 시그모이드를 쓴다
        """
        # 첫 가린 겹말기
        out = F.relu(self.input_conv(x))
        
        # 잔차 블록
        for block in self.residual_blocks:
            out = block(out)
        
        # 내놓기 겹말기
        out = F.relu(self.output_conv1(out))
        out = F.relu(self.output_conv2(out))
        
        # 마지막 헤아림
        out = self.final_conv(out)
        
        return out
    
    @torch.no_grad()
    def generate(self, 
                 shape: tuple,
                 device: str = 'cpu') -> torch.Tensor:
        """
        그림을 자기 되돌이로 만든다.
        
        이것이 자기 되돌이 만들어 내기의 알맹이이다.
        1. 빈 그림(모두 0)에서 시작한다
        2. 화소 자리마다(위에서 아래로, 왼쪽에서 오른쪽으로):
           a. 화소가 1일 확률을 헤아린다
           b. 베르누이 분포에서 뽑는다
           c. 화소를 채운다
        3. 다 된 그림을 돌려준다
        
        인수:
            꼴: (묶음 크기, 높이, 너비)
            device: 만들어 낼 기기
            
        반환값:
            만든 그림 [묶음 크기, 1, 높이, 너비]
        """
        self.eval()
        
        batch_size, height, width = shape
        
        # 빈 바탕(모두 0)에서 시작한다
        samples = torch.zeros(batch_size, 1, height, width).to(device)
        
        # 화소 하나씩 만든다
        # 가로 훑기 차례: 위에서 아래로, 왼쪽에서 오른쪽으로
        for i in range(height):
            for j in range(width):
                # 지금 화소의 헤아림을 얻는다
                # 참고: 앞서 만든 화소를 모두 쓴다
                logits = self.forward(samples)
                
                # 지금 화소 자리의 확률을 얻는다
                probs = torch.sigmoid(logits[:, :, i, j])
                
                # 베르누이 분포에서 뽑는다
                # 그래서 만들어 내기가 확률에 따르게 된다
                samples[:, :, i, j] = torch.bernoulli(probs)
        
        return samples


if __name__ == "__main__":
    """
    보여 주기: 흉내 자료로 PixelCNN을 시험한다
    """
    
    print("=" * 70)
    print("Testing Simplified PixelCNN")
    print("=" * 70)
    
    # 모델 생성
    model = PixelCNN(n_channels=32, n_residual_blocks=3)
    
    # 매개변수 개수 세기
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel has {n_params:,} parameters")
    
    # 순전파 시험
    batch_size = 4
    height, width = 28, 28  # MNIST 크기
    
    # 임시 입력 만들기
    x = torch.rand(batch_size, 1, height, width)
    
    # 순전파
    output = model(x)
    
    print(f"\nForward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    
    # 만들어 내기를 시험한다
    print(f"\nGenerating images...")
    print(f"  This will take a while (generating pixel by pixel)...")
    
    # 보여 주기 위해 작은 그림을 만든다
    small_shape = (2, 8, 8)  # 8x8 크기 그림 2장
    generated = model.generate(small_shape, device='cpu')
    
    print(f"  Generated shape: {generated.shape}")
    print(f"  Sample pixel values: {generated[0, 0, :3, :3]}")
    
    print("\n✓ PixelCNN working correctly!")
    print("\nNote: For real training, use the train.py script")
    print("which trains on actual image data (like MNIST)")
