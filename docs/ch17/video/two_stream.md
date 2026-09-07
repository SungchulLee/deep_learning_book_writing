# 몸짓 알아보기를 위한 두 갈래 그물
## 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

- 두 갈래 얼개를 쓰는 까닭을 이해한다
- 자리 갈래 그물과 때 갈래 그물을 짠다
- 때 갈래에 쓸 빛 흐름을 셈하고 앞손질한다
- 갈래별 어림을 아우르는 좋은 녹여 붙이기 전략을 꾸민다
- 겉모습과 움직임이 서로 채워 주는 성질을 값매김한다

## 왜 하는가: 겉모습과 움직임

### 두 갈래 가설

사람의 보기 다루기에는 뚜렷이 다른 길이 둘 있다:

1. **배쪽 갈래**(무엇): 물체 알아보기, 꼴, 빛깔
2. **등쪽 갈래**(어디/어떻게): 움직임, 자리 관계

두 갈래 그물은 다음을 따로 다뤄 이를 흉내낸다:

- **자리 갈래**: 겉모습(물체, 장면, 자세)을 위한 RGB 틀
- **때 갈래**: 움직임(빠르기, 방향, 흐름)을 위한 빛 흐름

### 왜 갈래를 나누는가?

틀 하나짜리 모델은 결정적인 때의 앎을 놓친다:

| 몸짓 | 겉모습 | 움직임이 필요한가? |
|--------|-----------|------------------|
| "서 있기"와 "걷기" | 같은 자세 | 그렇다 — 다리 움직임 |
| "마시기"와 "따르기" | 같은 물체 | 그렇다 — 손이 그리는 자취 |
| "문 열기"와 "문 닫기" | 같은 장면 | 그렇다 — 문의 방향 |

### 수학 얼거리

틀이 $\{I_1, \ldots, I_T\}$이고 빛 흐름이 $\{F_1, \ldots, F_{T-1}\}$인 영상 $V$이 주어졌을 때

**자리 갈래:**

$$p_{spatial} = f_s(I_t) \in \mathbb{R}^K$$

**때 갈래:**

$$p_{temporal} = f_t(\{F_{t}, F_{t+1}, \ldots, F_{t+L-1}\}) \in \mathbb{R}^K$$

**녹여 붙이기:**

$$p_{final} = \alpha \cdot p_{spatial} + (1-\alpha) \cdot p_{temporal}$$

여기서 $K$은 갈래 수이고 $\alpha$은 녹여 아우르는 짐이다.

## 자리 갈래

### 구조

자리 갈래는 보통의 2차원 누비기 신경망으로 RGB 틀 하나를 다룬다:

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

### 여러 틀 표집

영상 수준의 어림을 하려면 틀을 여럿 뽑는다:

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

## 때 갈래

### 빛 흐름 들임

때 갈래는 쌓아 올린 빛 흐름을 다룬다:

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

## 빛 흐름 셈하기

### 수학 바탕

빛 흐름은 밝기가 한결같다는 가정에 바탕해 틀 사이의 움직임을 어림한다:

$$I(x, y, t) = I(x + u, y + v, t + \Delta t)$$

여기서 $(u, v)$은 흐름 벡터(단위 시간당 옮김)이다.

테일러 펼침과 간추림으로 다음을 얻는다:

$$I_x u + I_y v + I_t = 0$$

여기서 $I_x$, $I_y$, $I_t$은 그림 기울기이다.

### OpenCV로 촘촘한 빛 흐름 얻기

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

### 빛 흐름 그려 보기

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

## 녹여 붙이기 전략

### 늦은 녹여 붙이기(점수 고루내기)

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

### 특징 수준의 녹여 붙이기

갈래를 매기기 앞서 특징을 아우른다:

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

## 학습 절차

### 따로 미리 익히기

처음의 두 갈래 논문은 갈래를 따로 익힌다:

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

### 끝에서 끝까지 익히기

두 갈래를 함께 익히기:

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

## 성능 살피기

### 서로 채워 주는 앎

두 갈래는 서로 다른 면을 담아낸다:

| 갈래 | 센 점 | 약한 점 |
|--------|-----------|------------|
| 자리 | 물체, 장면, 자세 | 움직임에 달린 몸짓을 가리지 못한다 |
| 때 | 움직임 무늬, 빠르기 | 물체가 무엇인지는 헤아리지 않는다 |
| 녹여 붙임 | 겉모습과 움직임 모두 | 셈이 더 든다 |

### 흔한 정확도(UCF-101)

| 방법 | 자리 | 때 | 녹여 붙임 |
|--------|---------|----------|-------|
| 처음의 두 갈래 | 73.0% | 83.7% | 88.0% |
| VGG-16을 쓸 때 | 78.4% | 85.1% | 91.4% |
| ResNet을 쓸 때 | 82.3% | 87.2% | 93.6% |

핵심 살핌: **때 갈래 혼자서도 자리 갈래보다 낫다.** 몸짓 알아보기에서 움직임이 얼마나 중요한지를 보여 준다.

## 요약

두 갈래 그물이 보여 주는 것:

1. **겉모습과 움직임**은 몸짓 알아보기에서 서로 채워 주는 실마리이다
2. **빛 흐름**은 움직임을 드러내어 나타낸다
3. **늦은 녹여 붙이기**는 단순하지만 잘 된다
4. **때의 앎**은 흔히 겉모습만보다 더 잘 가른다
5. **따로 미리 익히기**로 자리 갈래에 ImageNet 무게를 써먹을 수 있다

### 한계

- **빛 흐름 셈하기**는 값이 비싸다(흔히 미리 셈해 둔다)
- **따로 있는 그물 둘**이 매개변수와 미룸 시간을 두 배로 만든다
- **멀리 떨어진 때를 나타내지 못한다**(흐름 쌓기 길이에 갇힌다)

### 요즘의 대안

- **3차원 누비기 신경망**(C3D, I3D): RGB에서 움직임을 넌지시 배운다
- **느림빠름**: 때 해상도가 다른 두 갈래 길
- **영상 변환기**: 눈길에 바탕한 자리·때 나타내기

## 다음 걸음

- **빛 흐름 자세히**: 혼-슝크, 루카스-카나데, 배운 흐름
- **CNN-LSTM**: 되돌이로 때 나타내기
- **영상 변환기**: 영상 이해를 위한 눈길

## 연습문제

**연습문제 1.**
영상 이해를 위한 두 갈래 그물의 핵심 눈썰미를 밝히고, 날 틀 위의 한 갈래만으로는 왜 모자란지 설명하여라.

??? success "연습문제 1 풀이"
    두 갈래 그물은 **자리**(겉모습) 앎과 **때**(움직임) 앎을 서로 다른 갈래에서 다룬다. RGB 한 갈래만으로도 겉모습은 담아내지만 움직임에는 약한데, 그 까닭은 이렇다. (1) 때의 무늬는 여러 틀에 걸쳐 있어 넓은 받는 자리가 필요하다. (2) 움직임은 화소 바뀜 속에 숨어 있어 날 자료에서 배우기 어렵다. 빛 흐름 갈래는 움직임을 드러내어 부호로 담아 서로 채워 주는 신호를 준다. 두 갈래는 보통 (늦게 또는 중간에서) 녹여 붙여 겉모습과 움직임 이해를 아우르며, 한 갈래 방식보다 훨씬 낫다.

---

**연습문제 2.**
느림빠름 얼개를 설명하여라. 영상을 두 가지 틀 비율로 다루면 왜 알아보기가 나아지는가?

??? success "연습문제 2 풀이"
    SlowFast은 길 둘을 쓴다. **느린** 길은 낮은 틀 빠르기(예: 2 FPS)로 돌며 갈래를 넉넉히 두어 자리의 뜻을 잘게 담고, **빠른** 길은 높은 틀 빠르기(예: 16 FPS)로 돌며 갈래를 적게 두어 빠른 때의 움직임을 담는다. 자리의 뜻은 더디게 바뀌고(높은 틀 빠르기가 필요 없다) 움직임은 잔 때 잣대에서 일어나므로 이 꾸밈이 잘 든다. 빠른 길은 가볍고(셈의 $\sim$20%) 때의 결을 주며, 느린 길은 자리의 넉넉함을 준다. 옆으로 잇는 이음이 두 길 사이의 소식을 녹여 아우른다.

---

**연습문제 3.**
그림 가르기 얼개(보기로 ResNet)를 영상 이해로 넓힐 때의 주된 어려움은 무엇인가?

??? success "연습문제 3 풀이"
    고갱이 어려움은 이렇다. (1) **셈 값**: 때 차수를 더하면 자료가 $T\times$($T$은 틀 수)만큼 늘어 3차원 누비기가 값비싸진다. (2) **때 모형 짓기**: 2차원 누비기는 틀 하나만 보아 때의 무늬를 놓친다. 2차원 알갱이를 손쉽게 3차원으로 부풀리면(I3D 따위) 값이 비싸다. (3) **길이가 바뀌는 들임**: 영상마다 길이가 달라 때 모으기나 뽑기 꾀가 든다. (4) **멀리 걸친 매임**: 종요로운 일이 수백 틀에 걸칠 수 있어 그 자리 누비기의 받는 밭을 넘어선다. (5) **익힘 자료**: 영상 자료 묶음이 그림 자료 묶음보다 작아 지나치게 맞춰질 걱정이 있다.

---

**연습문제 4.**
영상을 나타내는 데 쓰는 3차원 누비기, (2+1)차원으로 쪼갠 누비기, 때에 걸친 스스로 눈길을 견주어라.

??? success "연습문제 4 풀이"
    | 방식 | 셈 | 때의 범위 | 익히기 |
    |----------|-------------|----------------|----------|
    | **3차원 누비기** | $O(k^3 C^2 THW)$ | 그 자리(틀 $k$개) | 값비싸고 미리 익히기가 든다 |
    | **(2+1)차원 누비기** | $O(k^2 C^2 THW + k C^2 THW)$ | 그 자리 | 다듬기 쉽고 매개변수가 적다 |
    | **때 눈길** | $O(T^2 CHW)$ | 두루 | $T$에 이차이며 너그럽다 |

    3차원 누비기는 힘세지만 값이 비싸다. (2+1)차원 쪼개기는 자리 다루기와 때 다루기를 갈라 정확도를 지키면서 매개변수를 줄인다. 때에 걸친 스스로 눈길은 멀리 떨어진 얽힘을 담아내지만 차례 길이의 제곱으로 늘어난다. 요즘 얼개(보기로 Video Swin Transformer)는 흔히 가까운 자리의 눈길과 층진 꾸밈을 아우른다.
