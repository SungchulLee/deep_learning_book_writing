# 단원 34: 영상 이해

단원 34: 영상 이해 — 가운데 수준. 파일 05: 두 갈래 그물 — 자리 갈래와 때 갈래 녹여 붙이기

이 단원은 셈틀 보기라는 더 넓은 맥락 안에서 영상 이해를 살펴본다. 여기 짠 것은 요즘 체계에서 쓰는 얼개의 새로움과 익히기 전략을 보여 주는 실전 PyTorch 코드이다.

## 1. 코드

```python
"""
단원 34: 영상 이해 — 가운데 수준
파일 05: 두 갈래 그물 — 자리 갈래와 때 갈래 녹여 붙이기

이 파일은 몸짓 알아보기를 위한 두 갈래 얼개를 다룬다:
- 겉모습 앎을 위한 자리 갈래(RGB 틀)
- 움직임 앎을 위한 때 갈래(빛 흐름)
- 늦은 녹여 붙이기 전략
- 처음의 두 갈래 누비기 신경망 짜기

수학적 바탕:
두 갈래 얼개:
    틀이 {I_1, ..., I_T}이고 빛 흐름이 {F_1, ..., F_{T-1}}인 영상 V에 대해:
    
    자리 갈래: f_s(I_t) → p_자리
    때 갈래: f_t(F_t) → p_때
    
    마지막 어림: p = α·p_자리 + (1-α)·p_때
    
    여기서 α는 녹여 붙이기 무게(보통 0.5)

핵심 눈썰미: 겉모습과 움직임은 서로 채워 주는 실마리이다!
    - 자리: 어떤 물체/장면이 있는가
    - 때: 어떻게 움직이는가
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
import warnings

# ========================================================================
# 메인
# ========================================================================
warnings.filterwarnings('ignore')


#=============================================================================
# 1부: 자리 갈래(겉모습)
#=============================================================================

class SpatialStream(nn.Module):
    """
    겉모습 앎을 위한 자리 갈래 누비기 신경망.
    
    RGB 틀 하나를 다뤄 다음을 담아낸다:
    - 물체가 무엇인지
    - 장면의 앎
    - 자세와 겉모습
    
    보통의 2차원 누비기 신경망(보기로 ResNet, VGG)을 쓴다
    """
    
    def __init__(self, 
                 num_classes: int = 400,
                 pretrained: bool = True,
                 dropout: float = 0.5):
        """
        자리 갈래를 첫자리매김한다.
        
        인수:
            num_classes: 몸짓 갈래의 개수
            pretrained: ImageNet에서 미리 익힌 무게를 쓸지 여부
            dropout: 드롭아웃 확률
        """
        super().__init__()
        
        # 등뼈로 ResNet-50 쓰기(표준 고름)
        # 들임: RGB 틀 하나 (B, 3, H, W)
        resnet = models.resnet50(pretrained=pretrained)
        
        # 마지막 온전히 이은 층 없애기
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # 몸짓을 위한 맞춤 갈래 매개
        # ResNet-50은 2048차원 특징을 내놓는다
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2048, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        자리 갈래를 지나는 앞먹임.
        
        인수:
            x: RGB 틀 (B, 3, H, W) — 틀 하나 또는 (B, T, 3, H, W) 묶음
            
        반환값:
            갈래 점수 (B, num_classes)
            
        유의: 영상은 틀을 여럿 뽑아 어림을 고루낼 수 있다
        """
        # 틀 하나와 영상 들임을 모두 다루기
        if x.dim() == 5:  # (B, T, 3, H, W)
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)  # 모든 틀 다루기
            batch_mode = True
        else:
            batch_mode = False
            B = x.shape[0]
        
        # 특징을 뽑는다
        features = self.features(x)  # (B*T, 2048, 1, 1) 또는 (B, 2048, 1, 1)
        features = features.flatten(start_dim=1)  # (B*T, 2048)
        
        # 분류
        logits = self.classifier(features)  # (B*T, num_classes)
        
        # 영상 들임이면 때에 걸쳐 고루내기
        if batch_mode:
            logits = logits.view(B, T, -1).mean(dim=1)  # (B, num_classes)
        
        return logits


#=============================================================================
# 2부: 때 갈래(움직임)
#=============================================================================

class TemporalStream(nn.Module):
    """
    움직임 앎을 위한 때 갈래 누비기 신경망.
    
    빛 흐름을 다뤄 다음을 담아낸다:
    - 움직임 무늬
    - 빠르기와 방향
    - 때에 걸친 움직임
    
    들임: 잇단 빛 흐름 마당 L개의 쌓기
    """
    
    def __init__(self,
                 num_classes: int = 400,
                 flow_length: int = 10,
                 dropout: float = 0.5):
        """
        때 갈래를 첫자리매김한다.
        
        인수:
            num_classes: 몸짓 갈래의 개수
            flow_length: 쌓을 흐름 마당의 개수(L)
            dropout: 드롭아웃 확률
        """
        super().__init__()
        
        self.flow_length = flow_length
        
        # 들임 채널: 2*L(흐름 L개의 u와 v 성분)
        # 잇단 빛 흐름 L개를 쌓아 들임으로 삼기
        input_channels = 2 * flow_length
        
        # 들임 채널을 달리해 고친 ResNet
        resnet = models.resnet50(pretrained=False)
        
        # 2*L 채널을 받도록 첫 누비기 층 갈음
        self.conv1 = nn.Conv2d(
            input_channels,  # 3 대신 2*L채널
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # 남은 층 베끼기
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        # 분류기
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2048, num_classes)
        )
    
    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        """
        때 갈래를 지나는 앞먹임.
        
        인수:
            flow: 빛 흐름 쌓기 (B, 2*L, H, W)
                  흐름마다 채널이 2개다(u, v)
                  잇단 흐름 L개를 쌓는다
                  
        반환값:
            갈래 점수 (B, num_classes)
        """
        # 그물에 통과시키기
        x = self.conv1(flow)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        
        # 분류
        logits = self.classifier(x)
        
        return logits


#=============================================================================
# 3부: 두 갈래 녹여 붙이기
#=============================================================================

class TwoStreamNetwork(nn.Module):
    """
    녹여 붙이기를 갖춘 온전한 두 갈래 그물.
    
    몸짓 알아보기를 위해 자리 갈래와 때 갈래를 아우른다.
    
    녹여 붙이기 전략:
        1. 늦은 녹여 붙이기(고루내기): p = (p_자리 + p_때) / 2
        2. 무게를 준 녹여 붙이기: p = α·p_자리 + (1-α)·p_때
        3. 배운 녹여 붙이기: p = MLP([p_자리, p_때])
    """
    
    def __init__(self,
                 num_classes: int = 400,
                 flow_length: int = 10,
                 fusion_type: str = 'average'):
        """
        두 갈래 그물을 첫자리매김한다.
        
        인수:
            num_classes: 몸짓 갈래의 개수
            flow_length: 빛 흐름 마당의 개수
            fusion_type: 'average', 'weighted', 또는 'learned'
        """
        super().__init__()
        
        self.fusion_type = fusion_type
        
        # 자리 갈래와 때 갈래
        self.spatial_stream = SpatialStream(num_classes=num_classes)
        self.temporal_stream = TemporalStream(
            num_classes=num_classes,
            flow_length=flow_length
        )
        
        # 무게를 준 녹여 붙이기 매개변수
        if fusion_type == 'weighted':
            self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        
        # 배운 녹여 붙이기 그물
        elif fusion_type == 'learned':
            self.fusion_net = nn.Sequential(
                nn.Linear(num_classes * 2, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
    
    def forward(self,
                rgb: torch.Tensor,
                flow: torch.Tensor,
                return_separate: bool = False) -> torch.Tensor:
        """
        두 갈래 그물을 지나는 앞먹임.
        
        인수:
            rgb: RGB 틀 (B, 3, H, W) 또는 (B, T, 3, H, W)
            flow: 빛 흐름 쌓기 (B, 2*L, H, W)
            return_separate: 갈래별 내놓음을 돌려줄지 여부
            
        반환값:
            녹여 붙인 갈래 점수 (B, num_classes)
            return_separate=True이면 (녹여 붙임, 자리, 때) 튜플
        """
        # 두 갈래에서 어림 얻기
        spatial_logits = self.spatial_stream(rgb)
        temporal_logits = self.temporal_stream(flow)
        
        # 녹여 붙이기 전략
        if self.fusion_type == 'average':
            # 단순 고루내기 녹여 붙이기
            fused_logits = (spatial_logits + temporal_logits) / 2
            
        elif self.fusion_type == 'weighted':
            # 배울 수 있는 무게로 녹여 붙이기
            alpha = torch.sigmoid(self.fusion_weight)
            fused_logits = alpha * spatial_logits + (1 - alpha) * temporal_logits
            
        elif self.fusion_type == 'learned':
            # 다층 퍼셉트론으로 배운 녹여 붙이기
            combined = torch.cat([spatial_logits, temporal_logits], dim=1)
            fused_logits = self.fusion_net(combined)
        
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
        
        if return_separate:
            return fused_logits, spatial_logits, temporal_logits
        return fused_logits


#=============================================================================
# 4부: 빛 흐름 만들기(간추림)
#=============================================================================

def compute_dense_optical_flow(prev_frame: np.ndarray,
                               next_frame: np.ndarray,
                               method: str = 'farneback') -> np.ndarray:
    """
    두 틀 사이의 촘촘한 빛 흐름을 셈한다.
    
    인수:
        prev_frame: 앞 틀 (H, W, 3) 또는 (H, W)
        next_frame: 다음 틀 (H, W, 3) 또는 (H, W)
        method: 흐름 셈하기 방법
        
    반환값:
        flow: (u, v) 성분을 갖는 빛 흐름 (H, W, 2)
        
    수학적 바탕:
        빛 흐름 식(밝기가 한결같음):
        I(x, y, t) = I(x+u, y+v, t+1)
        
        선형으로 만들기:
        I_x·u + I_y·v + I_t = 0
        
        여기서 (u,v)은 흐름 벡터
    """
    # 필요하면 잿빛으로 바꾸기
    if prev_frame.ndim == 3:
        prev_gray = cv2.cvtColor((prev_frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        next_gray = cv2.cvtColor((next_frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        prev_gray = (prev_frame * 255).astype(np.uint8)
        next_gray = (next_frame * 255).astype(np.uint8)
    
    if method == 'farneback':
        # 파르네베크 방법: 다항식 펼침
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            next_gray,
            None,
            pyr_scale=0.5,  # 피라미드 잣수
            levels=3,        # 피라미드 층 수
            winsize=15,      # 창 크기
            iterations=3,    # 피라미드 켜마다의 되풀이 횟수
            poly_n=5,        # 다항식 펼침 차수
            poly_sigma=1.2,  # 다항식 펼침의 가우스 표준편차
            flags=0
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return flow


def extract_flow_stack(video: torch.Tensor,
                       flow_length: int = 10) -> torch.Tensor:
    """
    영상에서 쌓아 올린 빛 흐름을 뽑는다.
    
    인수:
        video: 영상 텐서 (T, C, H, W) 또는 (B, T, C, H, W)
        flow_length: 쌓을 흐름의 개수(L)
        
    반환값:
        flow_stack: 쌓아 올린 흐름 (B, 2*L, H, W)
    """
    if video.dim() == 4:
        video = video.unsqueeze(0)  # 묶음 차원 더하기
    
    B, T, C, H, W = video.shape
    
    # 틀이 적어도 flow_length+1개 필요하다
    if T < flow_length + 1:
        raise ValueError(f"Need at least {flow_length+1} frames, got {T}")
    
    flow_stacks = []
    
    for b in range(B):
        flows = []
        
        # 잇단 틀 사이의 빛 흐름 셈하기
        for t in range(flow_length):
            prev_frame = video[b, t].permute(1, 2, 0).cpu().numpy()  # (H, W, C)
            next_frame = video[b, t+1].permute(1, 2, 0).cpu().numpy()
            
            # 흐름 셈하기
            flow = compute_dense_optical_flow(prev_frame, next_frame)
            
            # 텐서로 바꾸고 고르게 맞추기
            flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).float()  # (2, H, W)
            flows.append(flow_tensor)
        
        # 흐름 쌓기: (L, 2, H, W) → (2*L, H, W)
        flow_stack = torch.cat(flows, dim=0)
        flow_stacks.append(flow_stack)
    
    flow_stacks = torch.stack(flow_stacks, dim=0)  # (B, 2*L, H, W)
    
    return flow_stacks


#=============================================================================
# 5부: 보임
#=============================================================================

def demonstrate_two_stream():
    """
    두 갈래 그물 얼개와 녹여 붙이기를 보여 준다.
    """
    print("\n" + "="*80)
    print("TWO-STREAM NETWORK DEMONSTRATION")
    print("="*80)
    
    # 설정
    num_classes = 400  # Kinetics-400
    batch_size = 4
    num_frames = 16
    flow_length = 10
    height, width = 224, 224
    
    print(f"\nConfiguration:")
    print(f"  Number of classes: {num_classes}")
    print(f"  Batch size: {batch_size}")
    print(f"  Video frames: {num_frames}")
    print(f"  Flow stack length: {flow_length}")
    print(f"  Frame size: {height}x{width}")
    
    # 모델 만들기
    print("\n1. Creating two-stream network...")
    
    # 여러 녹여 붙이기 전략 시험
    fusion_types = ['average', 'weighted', 'learned']
    
    for fusion_type in fusion_types:
        print(f"\n   Testing {fusion_type} fusion:")
        model = TwoStreamNetwork(
            num_classes=num_classes,
            flow_length=flow_length,
            fusion_type=fusion_type
        )
        
        # 매개변수 개수 세기
        params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {params:,}")
        
        # 보기 들임 만들기
        rgb = torch.randn(batch_size, 3, height, width)
        flow = torch.randn(batch_size, 2 * flow_length, height, width)
        
        # 순전파
        model.eval()
        with torch.no_grad():
            output, spatial_out, temporal_out = model(
                rgb, flow, return_separate=True
            )
        
        print(f"   RGB input: {rgb.shape}")
        print(f"   Flow input: {flow.shape}")
        print(f"   Spatial output: {spatial_out.shape}")
        print(f"   Temporal output: {temporal_out.shape}")
        print(f"   Fused output: {output.shape}")
        
        # 녹여 붙이기 살피기
        spatial_probs = F.softmax(spatial_out[0], dim=0)
        temporal_probs = F.softmax(temporal_out[0], dim=0)
        fused_probs = F.softmax(output[0], dim=0)
        
        top5_spatial = torch.topk(spatial_probs, 3)
        top5_temporal = torch.topk(temporal_probs, 3)
        top5_fused = torch.topk(fused_probs, 3)
        
        print(f"\n   Top-3 predictions (first sample):")
        print(f"   Spatial:  {top5_spatial.indices.tolist()} "
              f"({top5_spatial.values[0].item():.3f})")
        print(f"   Temporal: {top5_temporal.indices.tolist()} "
              f"({top5_temporal.values[0].item():.3f})")
        print(f"   Fused:    {top5_fused.indices.tolist()} "
              f"({top5_fused.values[0].item():.3f})")
    
    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print("""
    1. 두 갈래 얼개:
       - 자리 갈래: 겉모습을 위해 RGB를 다룬다
       - 때 갈래: 움직임을 위해 빛 흐름을 다룬다
       - 서로 채워 주는 앎 → 더 나은 성능
    
    2. 빛 흐름:
       - 틀 사이의 움직임 앎을 담아낸다
       - (u, v) 성분이 가로와 세로 움직임을 담는다
       - 파르네베크: 다항식 펼침을 쓴 촘촘한 빛 흐름
    
    3. 녹여 붙이기 전략:
       - 고루내기: 단순하지만 잘 된다(무게가 같음)
       - 무게: 배울 수 있는 녹여 붙이기 무게 α
       - 배움: 다층 퍼셉트론이 가장 좋은 아우름을 배운다
    
    4. 성능에서 얻은 눈썰미:
       - 두 갈래 >> RGB 한 갈래(지난 결과들)
       - 몸짓 알아보기에서 정확도 약 10~15% 나아짐
       - 때 갈래는 누비기 신경망이 틀 하나에서 얻지 못하는 것을 담아낸다
    
    5. 실전에서 헤아릴 점:
       - 빛 흐름은 셈하기에 값이 비싸다
       - 흐름을 미리 셈해 두어야 한다
       - 모델 둘 → 매개변수와 셈이 2배
       - 요즘의 대안: 3차원 누비기 신경망, 영상 변환기
    """)


def main():
    """
    두 갈래 그물의 주된 보임.
    """
    print(__doc__)
    
    # 난수 씨앗 고정
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 보여 주기를 돌린다
    demonstrate_two_stream()


if __name__ == "__main__":
    import cv2  # 설치 안 됐을 때 탈이 없도록 여기서 들여온다
    main()
```

## 2. 논의

여기 짠 것은 함께 어울려 온전한 영상 이해 얼개를 이루는 클래스 3개(`SpatialStream`, `TemporalStream`, `TwoStreamNetwork`)를 정한다. 클래스마다 뚜렷한 조각 하나를 감싸므로 코드가 단원별로 나뉘고 넓히기 쉽다. `forward` 메서드가 PyTorch의 자동 미분에 쓰이는 셈 그래프를 정한다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 컴퓨터 비전 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
`SpatialStream`의 앞먹임을 따라가며 텐서 꼴을 좇아라. 붙박이 매개변수로 들임 표본 4개짜리 묶음에 대해 주요 연산(누비기, 모으기, 선형 층)마다 그 뒤의 꼴을 적어라.

??? success "연습문제 1 풀이"
    입력 모양에서 출발하여 각 층을 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 (덧대기가 없으면) $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌거나 (`padding=k//2`이면) 그대로 유지된다. 커널이 2인 풀링은 공간 차원을 절반으로 만든다. 선형 층은 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임에 유의하며 추적한다. 중간 모양을 합성곱 층에서는 $(B, C, H, W)$로, 평탄화 후에는 $(B, F)$로 적는다.

---

**연습문제 2.**
$64 \times 64$ 크기의 RGB 이미지(입력 모양 $3 \times 64 \times 64$)를 받도록 구조를 수정하라. 모든 층의 차원을 그에 맞게 고치고 모델이 오류 없이 실행되는지 확인하라.

??? success "연습문제 2 풀이"
    첫 엮음 켜의 `in_channels`을 이제 값에서 3으로 바꾼다. 엮음과 모으기 켜마다 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$으로 자리 차수를 다시 셈한다. 첫 선형 켜의 `in_features`을 마지막 엮음/모으기 켜의 편 날임에 맞춘다. `model = SpatialStream(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`으로 따진다.

---

**연습문제 3.**
같은 입출력 차원에서 표준 합성곱과 깊이별 분리 합성곱의 매개변수 개수와 FLOPs를 비교하라. 계산 절감이 가장 큰 것은 언제인가?

??? success "연습문제 3 풀이"
    표준 `Conv2d(C_in, C_out, k)`은 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개의 매개변수를 갖는다. 깊이별 분리 합성곱은 이를 둘로 나눈다. (1) 깊이별: $C_{{\text{{in}}}} \times k^2$개(입력 채널마다 필터 하나), (2) 점별: $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 합성곱)이다. 매개변수의 비는 대략 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 매개변수가 약 $8{-}9\times$ 적어진다. 절감은 $C_{{\text{{out}}}}$과 $k$가 모두 클 때 가장 크다.

---

**연습문제 4.**
층이나 덩이의 개수를 정할 수 있도록 `SpatialStream`을 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`로 깊이가 바뀌는 얼개를 만들어라. 층 2, 4, 8개로 시험하여라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 되풀이한다. (그냥 파이썬 목록이 아니라) `nn.ModuleList`를 써야 PyTorch가 가장 좋게 하기에 쓸 매개변수를 모두 등록한다. 시험: `for n in [2, 4, 8]: model = SpatialStream(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.

## 정리하며

**다룬 것** — 단원 34: 영상 이해

여기 짠 것은 함께 어울려 온전한 영상 이해 얼개를 이루는 클래스 3개(`SpatialStream`, `TemporalStream`, `TwoStreamNetwork`)를 정한다.

고갱이 갈래는 `SpatialStream`, `TemporalStream`, `TwoStreamNetwork`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
