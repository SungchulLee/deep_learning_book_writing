# PixelCNN

자기 되돌이 그림 만들어 내기를 위한 단순한 PixelCNN. PixelCNN은 가로 훑기 차례(왼쪽에서 오른쪽, 위에서 아래)로 화소 하나씩 그림을 만든다.

자기 되돌이 모델은 앞선 모든 낱개를 조건으로 삼아 낱개마다 미리 헤아려 자료를 만든다. 이 단원은 자기 되돌이 모델 부품의 짜기를 보이며 차례대로 만들어 내는 과정과 그 얼개의 요구를 그려 보인다.

## 1. 코드

```python
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
    print("which trains on actual image data (like MNIST)")```

## 2. 논의

이 짜기는 갈래 3개(`MaskedConv2d`, `ResidualBlock`, `PixelCNN`)를 뜻매김하며 이들이 함께 온전한 자기 되돌이 모델 얼개를 이룬다. 갈래마다 뚜렷이 구분되는 부품을 감싸므로 코드가 조각으로 나뉘고 넓히기 쉽다. `forward` 방법은 PyTorch가 자동 미분에 쓰는 셈 그래프를 뜻매김한다.

여기서 보인 결은 더 복잡한 경우로 자연스럽게 넓혀진다. 웃매개변수와 얼개 변형, 여러 자료 묶음을 실험하면 그림 만들어 내기 일에 대한 이해가 깊어지고 실제 직관이 쌓인다.

## 연습문제

**연습문제 1.**
`MaskedConv2d` 앞먹임을 지나는 텐서 꼴을 좇아라. 기본 매개변수로 들임 표본 4개 묶음에 대해 큰 셈(겹말기, 모으기, 선형 층)마다 뒤의 꼴을 적어라.

??? success "연습문제 1 풀이"
    입력 모양에서 출발하여 각 층을 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 (덧대기가 없으면) $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌거나 (`padding=k//2`이면) 그대로 유지된다. 커널이 2인 풀링은 공간 차원을 절반으로 만든다. 선형 층은 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임에 유의하며 추적한다. 중간 모양을 합성곱 층에서는 $(B, C, H, W)$로, 평탄화 후에는 $(B, F)$로 적는다.

---

**연습문제 2.**
$64 \times 64$ 크기의 RGB 이미지(입력 모양 $3 \times 64 \times 64$)를 받도록 구조를 수정하라. 모든 층의 차원을 그에 맞게 고치고 모델이 오류 없이 실행되는지 확인하라.

??? success "연습문제 2 풀이"
    첫 겹말기 층의 `in_channels`을 지금 값에서 3으로 바꾸어라. 공식 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$으로 겹말기와 모으기 층마다 뒤의 공간 차원을 다시 셈하라. 첫 선형 층의 `in_features`을 마지막 겹말기/모으기 층의 펼친 내놓기에 맞게 고쳐라. `model = MaskedConv2d(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`으로 확인하라.

---

**연습문제 3.**
같은 입출력 차원에서 표준 합성곱과 깊이별 분리 합성곱의 매개변수 개수와 FLOPs를 비교하라. 계산 절감이 가장 큰 것은 언제인가?

??? success "연습문제 3 풀이"
    표준 `Conv2d(C_in, C_out, k)`은 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개의 매개변수를 갖는다. 깊이별 분리 합성곱은 이를 둘로 나눈다. (1) 깊이별: $C_{{\text{{in}}}} \times k^2$개(입력 채널마다 필터 하나), (2) 점별: $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 합성곱)이다. 매개변수의 비는 대략 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 매개변수가 약 $8{-}9\times$ 적어진다. 절감은 $C_{{\text{{out}}}}$과 $k$가 모두 클 때 가장 크다.

---

**연습문제 4.**
`MaskedConv2d`을 층이나 덩이의 수를 맞출 수 있게 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 얼개를 만들어라. 2, 4, 8층으로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되풀이하라. (여느 파이썬 목록이 아니라) `nn.ModuleList`을 쓰면 PyTorch가 모든 매개변수를 가장 좋게 하기에 등록한다. `for n in [2, 4, 8]: model = MaskedConv2d(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험하라.

## 정리하며

**다룬 것** — PixelCNN

이 짜기는 갈래 3개(`MaskedConv2d`, `ResidualBlock`, `PixelCNN`)를 뜻매김하며 이들이 함께 온전한 자기 되돌이 모델 얼개를 이룬다.

고갱이 갈래는 `MaskedConv2d`, `ResidualBlock`, `PixelCNN`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
