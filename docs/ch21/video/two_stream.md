# 행동 인식을 위한 Two-Stream 네트워크

Two-Stream은 RGB 외형과 optical flow 이동 정보를 별도의 네트워크에서 처리해 결합하는 방식이다. [원 논문](https://arxiv.org/abs/1406.2199)은 두 신호의 상호 보완성을 행동 인식에서 실험했다. 이 계산 구조를 사람의 시각 경로와 정확히 일치하는 모형으로 해석하거나, flow가 모든 과제에서 RGB보다 낫다는 보장으로 확대하지 않는다.

## 1. RGB 가지와 프레임 집계

RGB 한 프레임은 물체·장면·자세를 제공하지만 시간 방향이 중요한 행동은 여러 프레임이 필요할 수 있다. 그렇다고 RGB 기반 단일 경로가 시간 정보를 배울 수 없는 것은 아니다. 3D CNN·LSTM·attention 등으로도 시간 결합이 가능하다. Two-Stream의 특징은 여기서 흐름을 별도 입력으로 제공한다는 데 있다.

```python
import torch
import torch.nn as nn
import torchvision.models as models

class SpatialStream(nn.Module):
    """
    겉모습에 바탕한 알아보기를 위한 자리 갈래.
    
    옮겨 배우기에 ImageNet에서 미리 익힌 모델을 쓴다.
    핵심 눈썰미: 몸짓은 흔히 물체 및 장면과 얽혀 있다.
    """
    
    def __init__(self, 
                 num_classes: int = 101,
                 backbone: str = 'resnet50',
                 pretrained: bool = True,
                 dropout: float = 0.5):
        super().__init__()
        
        # 미리 익힌 등뼈 읽어 들이기
        if backbone == 'resnet50':
            base = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif backbone == 'resnet101':
            base = models.resnet101(pretrained=pretrained)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # 마지막 온전히 이은 층 없애기
        self.features = nn.Sequential(*list(base.children())[:-1])
        
        # 몸짓 갈래 매개
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        인수:
            x: RGB 틀 (B, 3, H, W) 또는 영상 (B, T, 3, H, W)
        반환값:
            갈래 로짓 (B, num_classes)
        """
        # 영상 들임 다루기
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            is_video = True
        else:
            is_video = False
            B = x.shape[0]
        
        # 특징을 뽑는다
        features = self.features(x)
        features = features.flatten(1)  # (B*T, feature_dim)
        
        # 분류
        logits = self.classifier(features)
        
        # 영상은 때에 걸쳐 어림 고루내기
        if is_video:
            logits = logits.view(B, T, -1).mean(dim=1)
        
        return logits
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """갈래를 매기지 않고 특징만 뽑는다."""
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            features = self.features(x).flatten(1)
            features = features.view(B, T, -1)
        else:
            features = self.features(x).flatten(1)
        return features
```

SpatialStream은 ResNet-50 또는101의 이미지 특징을 사용하고 새 행동 분류 head를 붙인다. 배치 이미지는 B×3×H×W, 비디오는 B×T×3×H×W를 받는다. 비디오의 프레임 로짓을 평균하므로 eval에서 프레임 순서에 불변이다. extract_features는 비디오일 때 B×T×2048을 반환하며 아직 시간 요약을 하지 않는다. 사전 학습된 RGB 통계·입력 범위와 새 분류 head의 학습을 구분한다.

```python
def sample_frames_spatial(video: torch.Tensor, 
                          num_samples: int = 25) -> torch.Tensor:
    """
    자리 갈래 시험을 위해 틀을 뽑는다.
    
    전략: 영상 전체에서 틀 25개를 고루 뽑는다.
    어림: 틀마다의 소프트맥스 점수를 고루낸 값.
    """
    T = video.shape[0]
    indices = torch.linspace(0, T - 1, num_samples).long()
    return video[indices]
```

균일 샘플러는 양 끝을 포함한 linspace 인덱스를 쓰고 프레임이 부족하면 인덱스가 반복될 수 있다. 빈 비디오·요청 수의 유효성 검사는 별도다. 전체 비디오 샘플링과 flow의 짧은 연속 구간이 어느 시간 범위를 대응하는지 명시해야 한다.

## 2. 흐름 스택과 초기화

```python
class TemporalStream(nn.Module):
    """
    움직임에 바탕한 알아보기를 위한 때 갈래.
    
    들임: 잇단 빛 흐름 마당 L개의 쌓기.
    흐름마다 채널이 2개다(가로 u, 세로 v).
    전체 들임 채널: 2L
    
    흔한 자리매김: L = 10(흐름 마당 10개 = 채널 20개)
    """
    
    def __init__(self,
                 num_classes: int = 101,
                 flow_length: int = 10,
                 dropout: float = 0.5):
        super().__init__()
        
        self.flow_length = flow_length
        input_channels = 2 * flow_length  # 흐름마다의 u와 v
        
        # 흐름 들임에 맞게 고친 ResNet
        # 미리 익힌 무게를 쓸 수 없다(들임 채널이 다르다)
        resnet = models.resnet50(pretrained=False)
        
        # 첫 누비기 층 갈음
        self.conv1 = nn.Conv2d(
            input_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # 미리 익힌 RGB 무게의 평균으로 첫자리매김
        # 이러면 자리 짜임에 대한 앎이 얼마간 옮겨진다
        if True:  # 있어도 되는 무게 첫자리매김
            pretrained = models.resnet50(pretrained=True)
            pretrained_weight = pretrained.conv1.weight.data
            # RGB 채널로 고루내고 흐름 채널만큼 되풀이
            mean_weight = pretrained_weight.mean(dim=1, keepdim=True)
            self.conv1.weight.data = mean_weight.repeat(1, input_channels, 1, 1)
            self.conv1.weight.data /= input_channels  # 알맞게 잣수 맞추기
        
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
        인수:
            flow: 쌓아 올린 빛 흐름 (B, 2*L, H, W)
        반환값:
            갈래 로짓 (B, num_classes)
        """
        x = self.conv1(flow)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.flatten(1)
        
        return self.classifier(x)
```

TemporalStream은 L개의 프레임 쌍을 수평·수직 두 채널씩 연결한 B×2L×H×W를 받는다. 기본 L10에서20채널이다. 첫 합성곱만 RGB 사전 학습 필터를 평균·복제해 초기화하고 나머지 ResNet은 pretrained=False의 무작위 초기값이다. 전체 시간 가지가 ImageNet 사전 학습 가중치를 이어받은 것으로 설명하지 않는다.

코드는 RGB 필터의 채널 평균을2L번 복제한 뒤2L로 나눈다. 동일한 입력을 모든 채널에 넣었을 때 총 반응은 원 RGB 필터의 **합**이 아니라 **평균**에 해당한다. 원래 RGB 응답 크기를 자동 보존하는 초기화는 아니다. flow의 단위·범위는 RGB와 다르므로 분포까지 같아지는 것도 아니다. `if True` 때문에 이 초기화와 사전 학습 가중치 로드가 항상 수행된다.

## 3. 이동량과 flow 추출

첫 프레임의 위치(x,y)가 다음 프레임의(x+Δx,y+Δy)에 대응한다고 할 때 flow는 그 프레임 쌍의 이동량이다. 물리적 속도와 같지 않으며 시간 간격·투영·카메라 운동이 영향을 준다. 밝기 보존을 작은 변화로 근사하면 I_xΔx+I_yΔy+ΔI≈0이며, 속도 u=Δx/Δt,v=Δy/Δt와 시간 미분 I_t를 쓰면 I_xu+I_yv+I_t≈0이다. 이동량과 속도의 표기를 섞지 않는다.

```python
import cv2
import numpy as np

def compute_optical_flow(frame1: np.ndarray, 
                         frame2: np.ndarray,
                         method: str = 'farneback') -> np.ndarray:
    """
    두 틀 사이의 촘촘한 빛 흐름을 셈한다.
    
    인수:
        frame1: 앞 틀 (H, W, 3) RGB, 값 [0, 1]
        frame2: 지금 틀 (H, W, 3) RGB, 값 [0, 1]
        method: 흐름 알고리즘('farneback' 또는 'tvl1')
    
    반환값:
        flow: (u, v) 성분을 갖는 빛 흐름 (H, W, 2)
        
    흐름 벡터는 화소의 옮김을 가리킨다:
        - u: 가로 움직임(양수 = 오른쪽)
        - v: 세로 움직임(양수 = 아래쪽)
    """
    # 잿빛 uint8로 바꾸기
    gray1 = cv2.cvtColor((frame1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor((frame2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    if method == 'farneback':
        # 군나르 파르네베크 알고리즘
        # 이웃을 어림하려 다항식 펼침을 쓴다
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5,    # 피라미드 잣수(0.5 = 고전 피라미드)
            levels=3,          # 피라미드 층 수
            winsize=15,        # 고루내기 창 크기
            iterations=3,      # 피라미드 켜마다의 되풀이 횟수
            poly_n=5,          # 화소 이웃의 크기
            poly_sigma=1.2,    # 다항식 펼침의 가우스 표준편차
            flags=0
        )
    elif method == 'tvl1':
        # TV-L1 알고리즘(더 정확하고 느리다)
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
        flow = tvl1.calc(gray1, gray2, None)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return flow  # (H, W, 2)

def extract_flow_stack(video: torch.Tensor, 
                       flow_length: int = 10,
                       normalize: bool = True) -> torch.Tensor:
    """
    때 갈래를 위해 영상에서 쌓아 올린 빛 흐름을 뽑는다.
    
    인수:
        video: 값이 [0, 1]인 영상 텐서 (T, C, H, W)
        flow_length: 쌓을 흐름 마당의 개수(L)
        normalize: 흐름 값을 고르게 맞출지 여부
    
    반환값:
        flow_stack: 쌓아 올린 흐름 (2*L, H, W)
    """
    T, C, H, W = video.shape
    
    if T < flow_length + 1:
        raise ValueError(f"Need at least {flow_length + 1} frames")
    
    flows = []
    for t in range(flow_length):
        # OpenCV에 쓰려고 numpy로 바꾸기
        frame1 = video[t].permute(1, 2, 0).numpy()  # (H, W, C)
        frame2 = video[t + 1].permute(1, 2, 0).numpy()
        
        # 흐름 셈하기
        flow = compute_optical_flow(frame1, frame2)  # (H, W, 2)
        
        # 흐름 값 고르게 맞추기
        if normalize:
            # 흔한 흐름 값은 화소 [-20, 20] 안이다
            # [-1, 1]로 고르게 맞추기
            flow = np.clip(flow / 20.0, -1, 1)
        
        # 텐서로 바꾸기: (2, H, W)
        flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).float()
        flows.append(flow_tensor)
    
    # 쌓기: (L, 2, H, W) → (2*L, H, W)
    flow_stack = torch.cat(flows, dim=0)
    
    return flow_stack
```

이 함수는0~1 RGB 프레임을255배해 grayscale uint8로 바꾼다. uint8 입력이나 이미 정규화된 모델 입력을 그대로 넣으면 맞지 않는다. Farnebäck와 TV-L1은 다른 추정기이며 후자는 `cv2.optflow`를 제공하는 설치 구성이 필요하다. 밝기 변화·가림·큰 이동·반복 무늬에서 오차가 생길 수 있다.

extract_flow_stack은 CPU·경사 없는 T×3×H×W에서 처음 L쌍을 계산한다. .numpy()전에 장치 이동·detach가 없으므로 GPU나 requires_grad 입력에는 그대로 쓸 수 없다. normalize=True는 성분을20으로 나누어−1~1로 자르므로20픽셀보다 큰 이동량의 구분이 사라진다. 이 값은 확률도 표준편차 정규화도 아니다. L쌍에는 적어도 L+1프레임이 필요하고 이 과정은 미분 가능한 흐름 추정 경로가 아니다.

```python
def visualize_flow(flow: np.ndarray) -> np.ndarray:
    """
    빛 흐름을 RGB 그림으로 바꾼다.
    
    부호화:
        - 색상: 흐름 방향(각)
        - 채도: 최대(상수)
        - 밝기: 흐름의 크기
    
    인수:
        flow: 빛 흐름 (H, W, 2)
    반환값:
        RGB 그림 (H, W, 3)
    """
    h, w = flow.shape[:2]
    
    # 크기와 각 셈하기
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # HSV 그림 만들기
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2  # 색상: 방향
    hsv[..., 1] = 255  # 채도: 최대
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # 밝기: 크기
    
    # RGB로 바꾸기
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return rgb
```

방향·크기의 HSV 그림은 영상별 min-max 밝기를 사용한다. 다른 그림의 밝기로 절대 속도를 비교할 수 없고 일정한 크기장도 어둡게 표시될 수 있다. flow를 resize·반사하면 벡터 크기·부호도 해당 변환에 맞춰 바꿔야 한다.

## 4. 점수 결합과 특징 결합

```python
class TwoStreamNetwork(nn.Module):
    """
    녹여 붙이기를 갖춘 온전한 두 갈래 그물.
    
    녹여 붙이기 고름:
    1. 고루내기: 소프트맥스 점수의 단순 평균
    2. 무게: 배울 수 있거나 붙박이인 무게
    3. 배움: 다층 퍼셉트론이 특징 벡터를 아우른다
    """
    
    def __init__(self,
                 num_classes: int = 101,
                 flow_length: int = 10,
                 fusion: str = 'average',
                 spatial_weight: float = 0.4):
        super().__init__()
        
        self.spatial = SpatialStream(num_classes)
        self.temporal = TemporalStream(num_classes, flow_length)
        self.fusion = fusion
        
        if fusion == 'weighted':
            # 붙박이 무게 또는 배울 수 있는 무게
            self.alpha = nn.Parameter(torch.tensor(spatial_weight))
        elif fusion == 'learned':
            # 특징을 이어 붙이고 아우름을 배우기
            self.fusion_net = nn.Sequential(
                nn.Linear(num_classes * 2, num_classes * 2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(num_classes * 2, num_classes)
            )
    
    def forward(self, 
                rgb: torch.Tensor, 
                flow: torch.Tensor) -> torch.Tensor:
        """
        인수:
            rgb: RGB 틀 (B, 3, H, W) 또는 (B, T, 3, H, W)
            flow: 빛 흐름 쌓기 (B, 2*L, H, W)
        반환값:
            녹여 붙인 갈래 로짓 (B, num_classes)
        """
        # 갈래별 어림 얻기
        spatial_logits = self.spatial(rgb)
        temporal_logits = self.temporal(flow)
        
        # 녹여 붙이기
        if self.fusion == 'average':
            # 단순 고루내기(실제로는 때 갈래가 조금 낫다)
            fused = (spatial_logits + temporal_logits) / 2
            
        elif self.fusion == 'weighted':
            # 무게를 준 아우름
            alpha = torch.sigmoid(self.alpha)  # [0, 1]에 두기
            fused = alpha * spatial_logits + (1 - alpha) * temporal_logits
            
        elif self.fusion == 'learned':
            # 이어 붙이고 배우기
            combined = torch.cat([spatial_logits, temporal_logits], dim=1)
            fused = self.fusion_net(combined)
        
        return fused
```

위 모델은 확률이 아닌 **로짓**을 결합한다. average는 로짓 평균이고 weighted는 sigmoid(alpha)로 제한한 가중 평균이다. 기본 spatial_weight0.4는 실제 가중치0.4가 아니라 sigmoid0.4≈0.599로 시작한다. learned는 연결한 로짓을 MLP로 변환한다. 클래스 순서가 같아야 하고 로짓 평균 뒤 softmax와 확률 평균은 일반적으로 다르다. 잘못된 fusion 문자열에는 반환 fused가 정의되지 않는 점도 현재 코드의 한계다.

```python
class FeatureFusionTwoStream(nn.Module):
    """
    이른 녹여 붙이기: 두 갈래의 특징을 아우른다.
    
    점수 녹여 붙이기보다 잘 나타내지만
    특징 차원을 맞춰야 한다.
    """
    
    def __init__(self, num_classes: int = 101, flow_length: int = 10):
        super().__init__()
        
        self.spatial = SpatialStream(num_classes)
        self.temporal = TemporalStream(num_classes, flow_length)
        
        # 특징 차원: ResNet-50에서 2048
        self.fusion_layer = nn.Sequential(
            nn.Linear(2048 * 2, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes)
        )
    
    def forward(self, rgb, flow):
        # 특징 뽑아내기(로짓 아님)
        spatial_feat = self.spatial.extract_features(rgb)  # (B, 2048)
        temporal_feat = self.temporal.features(flow).flatten(1)  # (B, 2048)
        
        # 이어 붙이고 녹여 붙이기
        combined = torch.cat([spatial_feat, temporal_feat], dim=1)
        return self.fusion_layer(combined)
```

FeatureFusionTwoStream은 선언된 TemporalStream에 없는 `.features`를 호출하므로 그대로 실행할 수 없다. 또 RGB 비디오의 extract_features는 B×T×2048인데 예상 시간 특징은 B×2048이므로 시간 요약 규칙 없이 바로 연결할 수 없다. 각 가지에서 같은 시간 범위의 B×2048 특징을 얻는 명시적 경로와 집계가 필요하다. 특징 연결을 선언한 것만으로 학습·정렬이 완성되는 것은 아니다.

## 5. 분리 학습과 공동 학습의 범위

```python
def train_two_stream_separate():
    """
    두 갈래 그물의 익히기 절차.
    
    1. RGB 틀로 자리 갈래 익히기(ImageNet에서 미리 익힌 것을 쓸 수 있다)
    2. 빛 흐름 쌓기로 때 갈래 익히기
    3. 시험 때 어림 녹여 붙이기
    """
    
    # 자리 갈래 익히기
    spatial_stream = SpatialStream(num_classes=101)
    spatial_optimizer = torch.optim.SGD(
        spatial_stream.parameters(),
        lr=0.01, momentum=0.9, weight_decay=5e-4
    )
    
    for epoch in range(epochs):
        for frames, labels in spatial_loader:
            # 영상마다 틀 하나 뽑기
            idx = torch.randint(0, frames.shape[1], (1,)).item()
            single_frame = frames[:, idx]
            
            output = spatial_stream(single_frame)
            loss = F.cross_entropy(output, labels)
            
            spatial_optimizer.zero_grad()
            loss.backward()
            spatial_optimizer.step()
    
    # 때 갈래 익히기(마찬가지로)
    temporal_stream = TemporalStream(num_classes=101)
    temporal_optimizer = torch.optim.SGD(
        temporal_stream.parameters(),
        lr=0.01, momentum=0.9, weight_decay=5e-4
    )
    
    for epoch in range(epochs):
        for flow_stacks, labels in temporal_loader:
            output = temporal_stream(flow_stacks)
            loss = F.cross_entropy(output, labels)
            
            temporal_optimizer.zero_grad()
            loss.backward()
            temporal_optimizer.step()
```

분리 학습 조각은 epochs·loader와 F import가 별도로 필요하며 학습한 두 가지를 반환하거나 저장·결합하는 과정도 없다. 한 RGB 프레임을 선택할 때 배치 전체에 같은 시간 인덱스를 쓰는 설계다. 목표 클래스·전처리·분할과 검증·체크포인트를 별도로 구성해야 한다.

```python
def train_two_stream_e2e(model, train_loader, optimizer, epochs):
    """녹여 붙인 두 갈래 그물을 끝에서 끝까지 익히기."""
    
    for epoch in range(epochs):
        for rgb, flow, labels in train_loader:
            rgb, flow, labels = rgb.cuda(), flow.cuda(), labels.cuda()
            
            output = model(rgb, flow)
            loss = F.cross_entropy(output, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

공동 학습 루프는 입력·레이블만 CUDA로 옮기므로 모델의 장치를 맞춰야 하고 `.train()` 호출도 없다. 호출 전에 eval로 두었다면 dropout·BN이 학습 모드로 자동 전환되지 않는다. flow 텐서를 입력으로 받는 네트워크의 공동 최적화이지 OpenCV flow 추정까지 역전파하는 end-to-end 학습은 아니다.

## 6. 성능을 비교하는 조건

이 페이지에는 실제 학습·평가 결과를 만드는 데이터나 실험이 없다. 특정 UCF-101 정확도를 원 논문의 backbone·split·flow·사전 학습·평가 조건 없이 ‘흔한 정확도’로 제시하지 않는다. RGB 단독·flow 단독·결합을 같은 조건에서 비교하고 전체 시스템 비용에 흐름 추정·저장·읽기도 포함한다.

두 네트워크의 비용은 채널 수·프레임 수·fusion과 backbone에 따라 달라 정확히 두 배로 고정되지 않는다. 흐름이 외형 정보를 전혀 담지 않는 것도 아니고 짧은 스택이 모든 장기 시간 관계를 해결하지도 않는다. 이후의 시간 모델·3D CNN·SlowFast와 비교할 때 관찰 구간과 사전 학습을 맞춘다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
RGB와 flow를 결합하는 이유와 RGB 시간 모델이 불가능하다는 주장과의 차이를 설명하라.

</div>

??? success "연습문제 1 풀이"

    RGB의 외형과 명시적으로 추정한 이동량이 상호 보완적인 입력이 될 수 있다. 그러나 RGB도 여러 프레임의3D CNN·순환·attention으로 시간 관계를 배울 수 있다. 두 가지의 성능 이점은 같은 조건의 실험으로 확인해야 한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
TemporalStream이 전체 사전 학습 가중치를 쓰는가? 첫 필터의 복제·나눗셈은 어떤 응답을 만드는가?

</div>

??? success "연습문제 2 풀이"

    나머지 ResNet은 무작위 초기값이고 첫 필터만 RGB 사전 필터에서 가져온다. 채널 평균을 복제하고2L로 나누어 동일 채널 입력의 합성 응답은 원 RGB 필터의 평균과 같다. 원 RGB 합 응답을 보존한다고 설명하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
FeatureFusionTwoStream의 두 연결 문제를 설명하라.

</div>

??? success "연습문제 3 풀이"

    TemporalStream에 features속성이 없어 호출이 실패한다. RGB 비디오 특징도B×T×2048이므로B×2048흐름 특징과 곧바로 연결할 수 없다. 명시적인 시간 특징 추출과 RGB 시간 요약·정렬이 필요하다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
공동 학습 코드가 OpenCV 흐름 추정까지 학습하는가? weighted 초기값0.4의 실제 RGB 가중치는 얼마인가?

</div>

??? success "연습문제 4 풀이"

    flow를 이미 계산한 입력으로 받으므로 OpenCV 과정에는 역전파하지 않는다. 실제 RGB 로짓 가중치는sigmoid0.4≈0.599다. 모델의train모드·장치·Fimport와 목표 텐서의 정의도 별도로 맞춰야 한다.



## 정리하며

Two-Stream을 구현하려면 RGB·flow의 시간 범위·단위·초기화·클래스 순서를 맞추고 로짓·확률·특징 결합을 구분해야 한다. 현재 특징 결합·학습 조각에는 빠진 연결이 있어 완성된 실험으로 볼 수 없다. 원 논문의 아이디어와 실제 제공 코드의 범위를 분리해 읽는다.
