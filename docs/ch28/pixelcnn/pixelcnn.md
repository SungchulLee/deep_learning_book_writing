# 이진 이미지용 PixelCNN

PixelCNN은 래스터 순서, 즉 위에서 아래·각 행의 왼쪽에서 오른쪽으로 픽셀 조건부 분포를 정의한다. [PixelRNN·PixelCNN 연구](https://proceedings.mlr.press/v48/oord16.html)는 이미지의 결합 분포를 이런 조건부로 구성한다. 이 절은 단일 채널 이진 픽셀의 베르누이 확률을 예측하는 단순화된 구조다.

## 1. 마스크와 모듈 구현

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
            x: 들임 그림 [배치 크기, 1, 높이, 너비]
               [0, 1] 안의 값
               
        반환값:
            화소마다의 로짓 [배치 크기, 1, 높이, 너비]
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
            꼴: (배치 크기, 높이, 너비)
            device: 만들어 낼 기기
            
        반환값:
            만든 그림 [배치 크기, 1, 높이, 너비]
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
```

입력은 `(B,1,H,W)`이고 출력은 같은 크기의 logits다. Sigmoid를 적용한 값이 현재 픽셀을 1로 생성할 조건부 확률이다. 모델 forward 자체에는 Sigmoid가 없으므로 학습에서는 BCEWithLogitsLoss 등 logits용 손실을 사용한다. 연속 회색조의 밀도나 256단계 범주형 모형은 이 구현에 없다.

마스크 A는 커널 중심의 현재 입력 픽셀, 그 오른쪽과 아래 행을 차단한다. 첫 층에 A를 두어 타깃 픽셀을 직접 읽지 않는다. 후속 마스크 B는 중심의 **은닉 특징**을 허용하지만 그 특징이 이미 과거 픽셀에서만 계산되므로 현재 타깃의 누출을 만들지 않는다. 잔차 연결도 이 안전한 은닉 특징을 더한다.

**기존 출력 기록:**

```
======================================================================
Testing Simplified PixelCNN
======================================================================

Model has 13,921 parameters

Forward pass:
  Input shape: torch.Size([4, 1, 28, 28])
  Output shape: torch.Size([4, 1, 28, 28])

Generating images...
  This will take a while (generating pixel by pixel)...
  Generated shape: torch.Size([2, 1, 8, 8])
  Sample pixel values: tensor([[0., 0., 1.],
        [1., 1., 0.],
        [0., 1., 1.]])

✓ PixelCNN working correctly!

Note: For real training, use the train.py script
which trains on actual image data (like MNIST)
```

실행 예는 32채널·잔차 블록 3개의 무작위 모델로 순전파와 8×8 표본 생성을 수행하는 구조 확인이다. 학습 반복문은 없고 입력도 무작위 회색조이므로 이 출력이 MNIST 분포 학습이나 의미 있는 이미지 생성 품질을 입증하지는 않는다.

## 2. 모양과 수용 범위

기본 모델은 64채널과 잔차 블록 5개다. 입력 7×7·패딩 3, 잔차의 3×3·패딩 1과 1×1 층 모두 공간 크기를 유지한다. 각 잔차 블록은 채널을 C→floor(C/2)→floor(C/2)→C로 바꾸고 입력을 더한다. 유효한 중간 채널 수가 필요하므로 C≥2여야 한다.

조건은 이전 모든 픽셀을 형식상 허용하지만 실제 네트워크는 유한한 마스크 수용 영역의 일부 과거만 사용할 수 있다. 모든 이전 픽셀에 실제로 의존한다는 설명은 과하다. 기본 파라미터 집계는 78,785개이고 실행의 32채널·3블록은 13,921개다. 0으로 가리는 가중치도 저장된 파라미터 수에는 포함된다.

## 3. 가중치 마스킹의 구현상 주의

forward는 weight.data에 mask를 곱해 매 호출 전 가중치를 0으로 만든다. 현재 순전파에서 금지 위치를 차단하지만, 이 곱셈이 자동미분 그래프에 들어가지 않아 그 가중치의 grad가 반드시 0이 되는 것은 아니다. optimizer가 금지 위치를 갱신해도 다음 forward에서 다시 0으로 만든다. 상태 저장 시점·옵티마이저 상태와 기울기 클리핑 등을 해석할 때 이 차이를 알아야 한다.

일반적으로 마스크를 계산 그래프의 가중치 연산으로 적용하는 설계가 더 명확하다. 이 절에서는 코드를 보존하고 현재 방식의 효과와 한계를 설명한다. 홀수 커널·보폭 1·해당 패딩을 사용하는 현재 구조에 맞는 마스크이며 임의 stride·dilation·padding 모드로 바꿀 때는 인과성·모양을 다시 검토해야 한다.

## 4. 래스터 순서 생성

generate는 0 이미지에서 시작해 각 위치의 Sigmoid 확률로 베르누이 표본을 뽑아 채운다. 현재·미래 픽셀을 마스크가 보지 않으므로 아직 채우지 않은 0은 해당 조건부 예측에 영향을 주지 않아야 한다. 픽셀별 결정은 순차적이지만 배치의 여러 이미지는 병렬로 생성한다.

H×W번 전체 이미지 forward를 호출하므로 이 단순 구현은 비효율적이다. 학습은 이미지를 주고 모든 위치 logits를 한 번에 계산할 수 있어 생성과 계산 구조가 다르다. 입력 해상도에 고정된 선형 층이 없어 다른 H,W를 형식상 처리할 수 있지만 다른 해상도의 생성 품질은 별도다.

sample 생성 텐서의 기본 device와 부동소수점 자료형을 모델과 맞춰야 한다. 현재 generate는 eval·no_grad를 사용하고 원래 학습 모드를 복원하지 않는다. shape는 양의 배치·높이·너비로 정하고, 반환은 단일 채널의 0·1 표본이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
기본 모델과 배치 4,28×28 입력의 모양·파라미터를 계산하라.

</div>

??? success "연습문제 1 풀이"

    입력 `(4,1,28,28)`은 첫 층 뒤 `(4,64,28,28)`이다. 각 잔차 블록의 중간은 32채널, 최종·잔차 합은 64채널이다. 출력 1×1 두 층 뒤에도 64채널이고 마지막은 `(4,1,28,28)`이다. 파라미터는 첫 층 3,200개, 블록당 13,440개씩 5개, 출력 두 층 8,320개, 최종 65개로 총 78,785개다. 마스크 버퍼는 학습 파라미터가 아니다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
64×64 RGB로 확장할 때 단순 채널 변경만으로 충분한가?

</div>

??? success "연습문제 2 풀이"

    해상도 64×64 자체는 같은 크기 유지 합성곱으로 처리할 수 있다. RGB는 픽셀 내 채널 순서 R→G→B를 정의할지, 과거 공간 픽셀만 보고 세 채널을 조건부 독립으로 낼지 먼저 정해야 한다. 채널 간 자기회귀를 원하면 마스크와 출력 분포·샘플러도 채널 순서를 반영해야 한다. 현재 입력 1채널·출력 1채널·베르누이 생성에 채널 수만 바꾸어 8비트 RGB 모형이 완성되는 것은 아니다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
마스크와 depthwise 분리를 결합할 때 계산비용·인과성을 어떻게 확인하는가?

</div>

??? success "연습문제 3 풀이"

    편향 없는 표준 합성곱은 C_in C_out k², depthwise+pointwise는 C_in k²+C_in C_out개다. 하지만 마스크로 0인 가중치도 dense 커널로 계산하면 실제 연산이 자동으로 줄지 않는다. 분리 대안을 설계할 때 첫 층의 현재 픽셀 차단과 후속 특징의 인과성, 필요하다면 색 채널 순서를 유지해야 한다. 비용 비율만으로 같은 함수 계열·우도·속도를 보장할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
깊이를 설정하는 올바른 위치와 인과성 조건을 설명하라.

</div>

??? success "연습문제 4 풀이"

    PixelCNN에는 이미 n_residual_blocks가 있고 ModuleList로 등록된다. MaskedConv2d는 한 층이므로 기존 클래스에 없는 num_layers를 넘기는 예시는 맞지 않는다. 아래 YourBlock은 미정 골격이다. 깊이를 늘릴 때 첫 A층과 이후 안전한 B특징·잔차 경로를 유지하고 전체 logits가 현재·미래 입력 변화에 반응하지 않는 계약을 확인한다. 깊이 증가가 학습 품질 향상을 보장하지는 않는다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

이 PixelCNN은 마스크 A·B로 현재 타깃 누출을 막고 이진 픽셀의 베르누이 logits를 예측한다. 학습의 위치 병렬 계산과 생성의 순차 결정을 구분하고, data 기반 마스킹·유한 수용 영역·단일 채널 관측 모형의 한계를 이해해야 한다.
