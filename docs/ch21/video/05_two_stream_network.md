# Two-Stream 네트워크: RGB와 옵티컬 플로우 결합

Two-Stream은 RGB의 외형 정보와 프레임 사이 optical flow의 이동 정보를 서로 다른 가지에서 처리해 결합하는 방식이다. [원 논문](https://arxiv.org/abs/1406.2199)의 아이디어를 바탕으로 이 예제는 ResNet-50 두 가지와 세 로짓 결합 방법을 보여 준다. 원 논문의 구조·학습·흐름 추정을 그대로 재현한 구현은 아니다.

## 1. 코드

??? note "코드"

    ```python
    """
    영상 이해 — 중급 수준
    파일 05: Two-Stream 네트워크 — 공간 스트림과 시간 스트림 융합

    이 파일은 액션 인식을 위한 Two-Stream 구조를 다룬다:
    - 외형 정보를 위한 공간 스트림(RGB 프레임)
    - 움직임 정보를 위한 시간 스트림(옵티컬 플로우)
    - 후기 융합 전략
    - 원래의 Two-Stream 컨볼루션 뉴럴 네트워크 구현

    수학적 바탕:
    Two-Stream 구조:
        프레임이 {I_1, ..., I_T}이고 옵티컬 플로우가 {F_1, ..., F_{T-1}}인 영상 V에 대해:

        공간 스트림: f_s(I_t) → p_공간
        시간 스트림: f_t(F_t) → p_시간

        최종 예측: p = α·p_공간 + (1-α)·p_시간

        여기서 α는 융합 웨이트(보통 0.5)

    핵심 통찰: 외형과 움직임은 상호 보완적인 단서다!
        - 공간: 어떤 물체/장면이 있는가
        - 시간: 어떻게 움직이는가
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
    # 1부: 공간 스트림(외형)
    #=============================================================================

    class SpatialStream(nn.Module):
        """
        외형 정보를 위한 공간 스트림 컨볼루션 뉴럴 네트워크.

        RGB 프레임 하나를 다뤄 다음을 담아낸다:
        - 물체가 무엇인지
        - 장면 정보
        - 자세와 외형

        보통의 2차원 컨볼루션 뉴럴 네트워크(예: ResNet, VGG)를 쓴다
        """

        def __init__(self, 
                     num_classes: int = 400,
                     pretrained: bool = True,
                     dropout: float = 0.5):
            """
            공간 스트림을 초기화한다.

            인수:
                num_classes: 액션 클래스의 개수
                pretrained: ImageNet에서 사전 학습된 웨이트를 쓸지 여부
                dropout: 드롭아웃 확률
            """
            super().__init__()

            # 백본으로 ResNet-50 쓰기(표준적인 선택)
            # 입력: RGB 프레임 하나 (B, 3, H, W)
            resnet = models.resnet50(pretrained=pretrained)

            # 마지막 완전 연결 레이어 제거
            self.features = nn.Sequential(*list(resnet.children())[:-1])

            # 액션 인식용 맞춤 분류기
            # ResNet-50은 2048차원 피처를 내놓는다
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(2048, num_classes)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            공간 스트림을 지나는 순전파.

            인수:
                x: RGB 프레임 (B, 3, H, W) — 프레임 하나 또는 (B, T, 3, H, W) 배치

            반환값:
                클래스 점수 (B, num_classes)

            유의: 영상은 프레임을 여럿 뽑아 예측을 평균할 수 있다
            """
            # 프레임 하나와 영상 입력을 모두 다루기
            if x.dim() == 5:  # (B, T, 3, H, W)
                B, T, C, H, W = x.shape
                x = x.view(B * T, C, H, W)  # 모든 프레임 다루기
                batch_mode = True
            else:
                batch_mode = False
                B = x.shape[0]

            # 피처를 뽑는다
            features = self.features(x)  # (B*T, 2048, 1, 1) 또는 (B, 2048, 1, 1)
            features = features.flatten(start_dim=1)  # (B*T, 2048)

            # 분류
            logits = self.classifier(features)  # (B*T, num_classes)

            # 영상 입력이면 시간에 걸쳐 평균하기
            if batch_mode:
                logits = logits.view(B, T, -1).mean(dim=1)  # (B, num_classes)

            return logits


    #=============================================================================
    # 2부: 시간 스트림(움직임)
    #=============================================================================

    class TemporalStream(nn.Module):
        """
        움직임 정보를 위한 시간 스트림 컨볼루션 뉴럴 네트워크.

        옵티컬 플로우를 다뤄 다음을 담아낸다:
        - 움직임 패턴
        - 속도와 방향
        - 시간에 걸친 움직임

        입력: 연속한 옵티컬 플로우 장 L개의 스택
        """

        def __init__(self,
                     num_classes: int = 400,
                     flow_length: int = 10,
                     dropout: float = 0.5):
            """
            시간 스트림을 초기화한다.

            인수:
                num_classes: 액션 클래스의 개수
                flow_length: 쌓을 흐름장의 개수(L)
                dropout: 드롭아웃 확률
            """
            super().__init__()

            self.flow_length = flow_length

            # 입력 채널: 2*L(흐름 L개의 u와 v 성분)
            # 연속한 옵티컬 플로우 L개를 쌓아 입력으로 삼기
            input_channels = 2 * flow_length

            # 입력 채널 수를 바꿔 수정한 ResNet
            resnet = models.resnet50(pretrained=False)

            # 2*L 채널을 받도록 첫 컨볼루션 레이어 교체
            self.conv1 = nn.Conv2d(
                input_channels,  # 3 대신 2*L채널
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )

            # 남은 레이어 베끼기
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
            시간 스트림을 지나는 순전파.

            인수:
                flow: 옵티컬 플로우 스택 (B, 2*L, H, W)
                      흐름마다 채널이 2개다(u, v)
                      연속한 흐름 L개를 쌓는다

            반환값:
                클래스 점수 (B, num_classes)
            """
            # 네트워크에 통과시키기
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
    # 3부: Two-Stream 융합
    #=============================================================================

    class TwoStreamNetwork(nn.Module):
        """
        융합을 갖춘 온전한 Two-Stream 네트워크.

        액션 인식을 위해 공간 스트림과 시간 스트림을 결합한다.

        융합 전략:
            1. 후기 융합(평균): p = (p_공간 + p_시간) / 2
            2. 가중 융합: p = α·p_공간 + (1-α)·p_시간
            3. 학습된 융합: p = MLP([p_공간, p_시간])
        """

        def __init__(self,
                     num_classes: int = 400,
                     flow_length: int = 10,
                     fusion_type: str = 'average'):
            """
            Two-Stream 네트워크를 초기화한다.

            인수:
                num_classes: 액션 클래스의 개수
                flow_length: 옵티컬 플로우 장의 개수
                fusion_type: 'average', 'weighted', 또는 'learned'
            """
            super().__init__()

            self.fusion_type = fusion_type

            # 공간 스트림과 시간 스트림
            self.spatial_stream = SpatialStream(num_classes=num_classes)
            self.temporal_stream = TemporalStream(
                num_classes=num_classes,
                flow_length=flow_length
            )

            # 가중 융합 파라메터
            if fusion_type == 'weighted':
                self.fusion_weight = nn.Parameter(torch.tensor(0.5))

            # 학습된 융합 네트워크
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
            Two-Stream 네트워크를 지나는 순전파.

            인수:
                rgb: RGB 프레임 (B, 3, H, W) 또는 (B, T, 3, H, W)
                flow: 옵티컬 플로우 스택 (B, 2*L, H, W)
                return_separate: 스트림별 출력을 반환할지 여부

            반환값:
                융합한 클래스 점수 (B, num_classes)
                return_separate=True이면 (융합, 공간, 시간) 튜플
            """
            # 두 스트림에서 예측 얻기
            spatial_logits = self.spatial_stream(rgb)
            temporal_logits = self.temporal_stream(flow)

            # 융합 전략
            if self.fusion_type == 'average':
                # 단순 평균 융합
                fused_logits = (spatial_logits + temporal_logits) / 2

            elif self.fusion_type == 'weighted':
                # 학습 가능한 웨이트로 융합
                alpha = torch.sigmoid(self.fusion_weight)
                fused_logits = alpha * spatial_logits + (1 - alpha) * temporal_logits

            elif self.fusion_type == 'learned':
                # 다층 퍼셉트론으로 학습된 융합
                combined = torch.cat([spatial_logits, temporal_logits], dim=1)
                fused_logits = self.fusion_net(combined)

            else:
                raise ValueError(f"Unknown fusion type: {self.fusion_type}")

            if return_separate:
                return fused_logits, spatial_logits, temporal_logits
            return fused_logits


    #=============================================================================
    # 4부: 옵티컬 플로우 계산(간략판)
    #=============================================================================

    def compute_dense_optical_flow(prev_frame: np.ndarray,
                                   next_frame: np.ndarray,
                                   method: str = 'farneback') -> np.ndarray:
        """
        두 프레임 사이의 밀집 옵티컬 플로우를 계산한다.

        인수:
            prev_frame: 앞 프레임 (H, W, 3) 또는 (H, W)
            next_frame: 다음 프레임 (H, W, 3) 또는 (H, W)
            method: 흐름 계산 방법

        반환값:
            flow: (u, v) 성분을 갖는 옵티컬 플로우 (H, W, 2)

        수학적 바탕:
            옵티컬 플로우 방정식(밝기 불변):
            I(x, y, t) = I(x+u, y+v, t+1)

            선형화:
            I_x·u + I_y·v + I_t = 0

            여기서 (u,v)는 흐름 벡터
        """
        # 필요하면 그레이스케일로 바꾸기
        if prev_frame.ndim == 3:
            prev_gray = cv2.cvtColor((prev_frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            next_gray = cv2.cvtColor((next_frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            prev_gray = (prev_frame * 255).astype(np.uint8)
            next_gray = (next_frame * 255).astype(np.uint8)

        if method == 'farneback':
            # 파르네베크 방법: 다항식 전개
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                next_gray,
                None,
                pyr_scale=0.5,  # 피라미드 스케일
                levels=3,        # 피라미드 레벨 수
                winsize=15,      # 창 크기
                iterations=3,    # 피라미드 레벨마다의 반복 횟수
                poly_n=5,        # 다항식 전개 차수
                poly_sigma=1.2,  # 다항식 전개의 가우스 표준편차
                flags=0
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        return flow


    def extract_flow_stack(video: torch.Tensor,
                           flow_length: int = 10) -> torch.Tensor:
        """
        영상에서 스택으로 쌓은 옵티컬 플로우를 뽑는다.

        인수:
            video: 영상 텐서 (T, C, H, W) 또는 (B, T, C, H, W)
            flow_length: 쌓을 흐름의 개수(L)

        반환값:
            flow_stack: 스택으로 쌓은 흐름 (B, 2*L, H, W)
        """
        if video.dim() == 4:
            video = video.unsqueeze(0)  # 배치 차원 더하기

        B, T, C, H, W = video.shape

        # 프레임이 적어도 flow_length+1개 필요하다
        if T < flow_length + 1:
            raise ValueError(f"Need at least {flow_length+1} frames, got {T}")

        flow_stacks = []

        for b in range(B):
            flows = []

            # 연속한 프레임 사이의 옵티컬 플로우 계산하기
            for t in range(flow_length):
                prev_frame = video[b, t].permute(1, 2, 0).cpu().numpy()  # (H, W, C)
                next_frame = video[b, t+1].permute(1, 2, 0).cpu().numpy()

                # 흐름 계산하기
                flow = compute_dense_optical_flow(prev_frame, next_frame)

                # 텐서로 바꾸고 정규화하기
                flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).float()  # (2, H, W)
                flows.append(flow_tensor)

            # 흐름 스택: (L, 2, H, W) → (2*L, H, W)
            flow_stack = torch.cat(flows, dim=0)
            flow_stacks.append(flow_stack)

        flow_stacks = torch.stack(flow_stacks, dim=0)  # (B, 2*L, H, W)

        return flow_stacks


    #=============================================================================
    # 5부: 시연
    #=============================================================================

    def demonstrate_two_stream():
        """
        Two-Stream 네트워크 구조와 융합을 보여 준다.
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

        # 여러 융합 전략 테스트
        fusion_types = ['average', 'weighted', 'learned']

        for fusion_type in fusion_types:
            print(f"\n   Testing {fusion_type} fusion:")
            model = TwoStreamNetwork(
                num_classes=num_classes,
                flow_length=flow_length,
                fusion_type=fusion_type
            )

            # 파라메터 개수 세기
            params = sum(p.numel() for p in model.parameters())
            print(f"   Total parameters: {params:,}")

            # 예시 입력 만들기
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

            # 융합 결과 살펴보기
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
        1. Two-Stream 구조:
           - 공간 스트림: 외형을 위해 RGB를 다룬다
           - 시간 스트림: 움직임을 위해 옵티컬 플로우를 다룬다
           - 상호 보완적인 정보 → 더 나은 성능

        2. 옵티컬 플로우:
           - 프레임 사이의 움직임 정보를 담아낸다
           - (u, v) 성분이 가로와 세로 움직임을 담는다
           - 파르네베크: 다항식 전개를 쓴 밀집 옵티컬 플로우

        3. 융합 전략:
           - 평균: 단순하지만 잘 된다(웨이트가 같음)
           - 가중: 학습 가능한 융합 웨이트 α
           - 학습: 다층 퍼셉트론이 가장 좋은 결합을 학습한다

        4. 성능에서 얻은 통찰:
           - Two-Stream >> RGB 단일 스트림(과거 결과들)
           - 액션 인식에서 정확도 약 10~15% 나아짐
           - 시간 스트림은 컨볼루션 뉴럴 네트워크가 프레임 하나에서 얻지 못하는 것을 담아낸다

        5. 실전에서 고려할 점:
           - 옵티컬 플로우는 계산 비용이 크다
           - 흐름을 미리 계산해 두어야 한다
           - 모델 둘 → 파라메터와 계산량이 2배
           - 요즘의 대안: 3차원 컨볼루션 뉴럴 네트워크, 영상 트랜스포머
        """)


    def main():
        """
        Two-Stream 네트워크의 메인 시연.
        """
        print(__doc__)

        # 난수 시드 고정
        torch.manual_seed(42)
        np.random.seed(42)

        # 시연을 실행한다
        demonstrate_two_stream()


    if __name__ == "__main__":
        import cv2  # 설치되지 않았을 때 문제가 없도록 여기서 임포트한다
        main()
    ```

## 2. RGB 가지와 프레임별 예측

`SpatialStream`은 ImageNet 사전 학습 ResNet-50의 분류 레이어를 제거하고 새 K클래스 head를 붙인다. backbone을 사전 학습했다는 사실과 새 액션 분류 head가 학습되었다는 것은 다르다. 입력의 RGB 정규화와 분포는 사전 학습 웨이트가 기대하는 것에 맞춰야 한다. `pretrained` 인수는 과거 API 형식이므로 설치한 torchvision 버전과 호환되는지 확인한다.

4차원 입력은 B×3×H×W이고, 5차원은 **B×T×3×H×W**다. 프레임별 2048차원 피처에서 로짓을 만든 뒤 시간 평균을 낸다. Conv3d의 B×C×T×H×W와 축 순서가 다르다. 시간 평균은 순서를 보지 않으므로 프레임을 재배열해도 eval 모드의 결과는 같다. 학습 중에는 BN·dropout 같은 확률적·배치 효과를 따로 고려한다.

## 3. 흐름 스택과 시간 가지

L개의 연속 프레임 쌍의 수평·수직 flow를 연결하면 B×2L×H×W다. 기본 L=10이면 20채널을 받아야 하며 원본은 적어도 11프레임이 필요하다. `TemporalStream`은 첫 컨볼루션을 이 채널 수에 맞추고, ResNet 본체도 pretrained=False로 시작한다. RGB와 달리 시간 가지는 사전 학습 웨이트를 사용하지 않는다.

`extract_flow_stack`은 BTCHW의 처음 L개 쌍만 사용하고 뒤 프레임은 사용하지 않는다. 각 프레임을 CPU NumPy로 옮겨 Farnebäck를 계산한 뒤 flow를 CPU float32 텐서로 반환한다. 이 과정은 원래 RGB에 대한 미분 가능한 경로가 아니며 GPU 모델에 넣으려면 장치도 맞춰야 한다. 입력이 requires_grad이면 detach 없이 .numpy()를 호출하는 문제도 남는다.

flow 계산 함수는 0~1 RGB 또는 grayscale을 가정해 255를 곱한다. 이미 uint8 0~255이거나 ImageNet 정규화된 값을 넣으면 올바른 영상 변환이 아니다. `cv2` import가 main 실행 조건 안에 있어 모듈을 import한 뒤 함수만 호출하면 별도로 import되지 않은 cv2 때문에 실패할 수 있다. 주 시연에서는 이 흐름 추출 함수 자체를 호출하지 않는다.

## 4. 세 로짓 결합 방식

각 가지는 B×K 로짓을 반환한다. average는 두 로짓의 산술 평균이고, weighted는 α=sigmoid(a)로 αz_RGB+(1−α)z_flow를 만든다. 초기 a=0.5이므로 α는 약 0.622이며 정확히 반반이 아니다. learned는 연결한 2K차원을 MLP에 넣는다. 학습 가능한 결합이라고 자동으로 최적의 조합을 찾는 것은 아니다.

로짓 평균 뒤 softmax와 확률을 먼저 구해 평균하는 것은 일반적으로 다르다. 동일 클래스 순서를 공유하고 두 가지의 스케일·학습 상태를 고려해야 한다. 기본 시연의 B=4, K=400에서는 분리·결합 출력이 모두 4×400이며 return_separate=True는 이 세 텐서의 튜플이다.

## 5. 시연 결과와 실제 비용

시연은 무작위 RGB·flow 입력에 새 액션 head의 출력을 계산할 뿐 액션 인식 학습이나 검증을 하지 않는다. 실제 프레임의 flow도 추출하지 않는다. 따라서 top3 값이나 요약 문구의 10~15% 개선을 이 코드의 실험 결과로 인용할 수 없다. 세 fusion 모델도 각각 새로 초기화되어 결합 방식만 통제한 성능 비교가 아니다.

두 가지 backbone과 추가 head·flow 계산 때문에 비용이 늘지만 모든 설정에서 정확히 두 배는 아니다. 시간 가지의 입력 채널과 결합 방식도 파라메터 수를 바꾼다. flow를 미리 저장하면 학습 시 추정 비용을 줄일 수 있으나 저장·전처리·배포 비용은 남는다. RGB·flow 시간 구간과 crop 좌표의 정렬도 필수다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
4×3×64×64 RGB가 SpatialStream을 통과할 때 주요 모양을 계산하라.

</div>

??? success "풀이"

    ResNet-50 stem 컨볼루션 뒤 4×64×32×32, 최대 풀링 뒤 4×64×16×16, 단계별로 4×256×16×16 · 4×512×8×8 · 4×1024×4×4 · 4×2048×2×2다. 전역 풀링 후 4×2048을 새 선형 head에 넣어 기본 4×400 로짓을 얻는다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
B×16×3×H×W에서 flow_length=10의 흐름 스택은 어떤 프레임을 쓰고 어디에 놓이는가?

</div>

??? success "풀이"

    프레임 0→1부터 9→10까지 10쌍을 계산하므로 뒤 5프레임은 사용하지 않는다. 출력은 B×20×H×W의 CPU float32다. RGB 구간과 정렬하고 시간 가지 모델의 장치·입력 범위와 맞춰야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
weighted fusion이 처음부터 두 가지를 반씩 반영하는가? 확률 평균과도 같은가?

</div>

??? success "풀이"

    초기 파라메터가 0.5이므로 sigmoid(0.5)≈0.622가 RGB 로짓의 웨이트다. 로짓 가중 평균 뒤 softmax는 확률 가중 평균과 다르다. 반반 로짓 결합에는 α=0.5에 대응하는 파라메터 0이 필요하다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 ModuleList 확장만으로 Two-Stream의 시간 표현과 학습이 완성되는가?

</div>

??? success "풀이"

    YourBlock 정의와 각 backbone 채널·stride를 맞추는 것 외에 흐름 스택의 순서·RGB 구간 정렬·입력 정규화·분류 학습이 필요하다. RGB 시간 평균만으로 순서 모델링이 추가되지는 않는다. 무작위 입력의 모양 확인은 액션 인식 평가가 아니다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

Two-Stream은 외형과 흐름을 별도 가지에서 처리한다. BTCHW·2L채널의 계약과 입력 범위·시간 정렬을 맞추고 로짓 결합과 확률 결합을 구분한다. 현재 시연의 무작위 입력과 새 head는 성능 개선을 입증하지 않는다.
