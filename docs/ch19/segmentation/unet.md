# U-넷: 생의학 그림 나누기를 위한 누비기 그물

---

## 1. 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

- 건너뛰는 이음을 갖춘 U-넷의 대칭 부호기-풀개 얼개를 이해한다
- PyTorch로 U-넷을 맨바닥부터 짠다
- 자리 앎을 지키는 데 건너뛰는 이음이 왜 중요한지 밝힌다
- U-넷을 두 갈래 및 여러 갈래 나누기 일에 쓴다
- 들임 크기와 복잡도 요구가 다를 때 U-넷 얼개를 맞춰 고친다
- 요즘 U-넷 변종과 나아진 점을 이해한다

---

## 2. U-넷 들어가기

U-넷은 2015년 Ronneberger, Fischer, Brox가 생의학 그림 나누기, 특히 전자 현미경 그림에서 신경 세포 짜임을 나누려고 내놓았다. 그려 보면 뚜렷한 U 꼴이 되어서 그 이름을 얻었다.

### U-넷이 판을 잡은 까닭

U-넷은 의료 영상의 결정적인 어려움을 다뤘다:

1. **익힘 자료가 적음**: 의료 자료 뭉치는 대개 작다(그림 수십에서 수백 장)
2. **정밀한 자리 잡기가 필요함**: 화소 단위로 정확한 테두리가 임상에서 중요하다
3. **갈래 치우침**: 병터나 종양은 흔히 그림의 아주 작은 몫만 차지한다
4. **빠른 미룸**: 임상 자리에서는 실시간이나 거의 실시간이 요구된다

이어 붙이기 방식의 건너뛰는 이음을 갖춘 대칭 부호기-풀개라는 이 우아한 풀이는 요즘 나누기 그물의 바탕이 되었다.

---

## 3. 얼개 깊이 파고들기

### U 꼴 풀이하기

```
                    Input (572×572×1)
                           │
    ┌──────────────────────┼──────────────────────┐
    │                      ▼                      │
    │     ┌────────────────────────────────┐     │
    │     │  Conv 3×3, 64 → Conv 3×3, 64   │     │
    │     │        (568×568×64)            │─────┼───────────────────┐
    │     └────────────┬───────────────────┘     │                   │
    │                  ▼ MaxPool 2×2             │                   │
    │     ┌────────────────────────────────┐     │                   │
    │     │  Conv 3×3, 128 → Conv 3×3, 128 │     │                   │
    │     │        (280×280×128)           │─────┼───────────────┐   │
    │     └────────────┬───────────────────┘     │               │   │
    │                  ▼ MaxPool 2×2             │               │   │
    │     ┌────────────────────────────────┐     │               │   │
    │     │  Conv 3×3, 256 → Conv 3×3, 256 │     │               │   │
    │     │        (136×136×256)           │─────┼───────────┐   │   │
    │     └────────────┬───────────────────┘     │           │   │   │
    │                  ▼ MaxPool 2×2             │           │   │   │
    │     ┌────────────────────────────────┐     │           │   │   │
    │     │  Conv 3×3, 512 → Conv 3×3, 512 │     │           │   │   │
    │     │         (64×64×512)            │─────┼───────┐   │   │   │
    │     └────────────┬───────────────────┘     │       │   │   │   │
    │                  ▼ MaxPool 2×2             │       │   │   │   │
    │     ┌────────────────────────────────┐     │       │   │   │   │
    │     │  Conv 3×3, 1024 → Conv 3×3, 1024│    │       │   │   │   │
    │     │         (28×28×1024)           │     │       │   │   │   │
    │     │         [BOTTLENECK]           │     │       │   │   │   │
    │     └────────────┬───────────────────┘     │       │   │   │   │
    │                  ▼ Up-conv 2×2             │       │   │   │   │
    │     ┌────────────────────────────────┐     │       │   │   │   │
    │     │  ← Crop & Concatenate ─────────│─────┼───────┘   │   │   │
    │     │  Conv 3×3, 512 → Conv 3×3, 512 │     │           │   │   │
    │     └────────────┬───────────────────┘     │           │   │   │
    │                  ▼ Up-conv 2×2             │           │   │   │
    │     ┌────────────────────────────────┐     │           │   │   │
    │     │  ← Crop & Concatenate ─────────│─────┼───────────┘   │   │
    │     │  Conv 3×3, 256 → Conv 3×3, 256 │     │               │   │
    │     └────────────┬───────────────────┘     │               │   │
    │                  ▼ Up-conv 2×2             │               │   │
    │     ┌────────────────────────────────┐     │               │   │
    │     │  ← Crop & Concatenate ─────────│─────┼───────────────┘   │
    │     │  Conv 3×3, 128 → Conv 3×3, 128 │     │                   │
    │     └────────────┬───────────────────┘     │                   │
    │                  ▼ Up-conv 2×2             │                   │
    │     ┌────────────────────────────────┐     │                   │
    │     │  ← Crop & Concatenate ─────────│─────┼───────────────────┘
    │     │  Conv 3×3, 64 → Conv 3×3, 64   │     │
    │     └────────────┬───────────────────┘     │
    │                  ▼ Conv 1×1                │
    │     ┌────────────────────────────────┐     │
    │     │     Output (388×388×2)         │     │
    │     └────────────────────────────────┘     │
    └────────────────────────────────────────────┘
```

### 핵심 꾸밈 원리

1. **대칭 짜임**: 부호기와 풀개의 깊이가 맞물린다
2. **이어 붙이기로 하는 건너뛰는 이음**: 자리 앎을 지킨다
3. **겹 누비기 덩이**: 켜마다 3×3 누비기가 둘이다
4. **덧대기 없음(처음 판)**: 누비기마다 내놓는 것이 조금씩 줄어든다
5. **뒤바꾼 누비기**: 배울 수 있는 키우기

---

## 4. 온전한 PyTorch 짜기

### 겹 누비기 덩이

U-넷의 근본 벽돌은 겹 누비기이다:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    겹 누비기 덩이: (Conv → BN → ReLU) × 2
    
    이것이 U-넷의 고갱이 벽돌이다. 부호기와 풀개의
    켜마다 묶음 고르게 맞추기와 ReLU 깨어남을 곁들인
    잇단 3×3 누비기 둘로 이루어진다.
    
    인수:
        in_channels: 들임 채널의 개수
        out_channels: 내놓는 채널의 개수
        mid_channels: 첫 누비기 뒤의 채널 개수(붙박이: out_channels)
    """
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        
        if mid_channels is None:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            # 첫 누비기
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            
            # 두 번째 누비기
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)
```

### 부호기(오그라드는 길)

```python
class EncoderBlock(nn.Module):
    """
    부호기 덩이: 최대 모으기 → 겹 누비기
    
    자리 차원을 2분의 1로 줄이고 겹 누비기를 쓴다.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)
```

### 풀개(부풀어 오르는 길)

```python
class DecoderBlock(nn.Module):
    """
    풀개 덩이: 키우기 → 이어 붙이기 → 겹 누비기
    
    자리 차원을 2배로 키우고 부호기 특징과 이어 붙인 뒤
    건너뛰는 이음으로 이어 붙인 뒤 겹 누비기를 쓴다.
    
    인수:
        in_channels: 들임 채널의 개수(앞 풀개 켜에서 옴)
        out_channels: 내놓는 채널의 개수
        bilinear: True이면 두 줄 사이 끼움 키우기, False이면 뒤바꾼 누비기
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        
        if bilinear:
            # 두 줄 사이 끼움 키우기 + 채널을 줄이는 1×1 누비기
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
            )
            # 이어 붙인 뒤: in_channels // 2 + in_channels // 2 = in_channels
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # 뒤바꾼 누비기(배울 수 있는 키우기)
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 
                                          kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        건너뛰는 이음을 쓴 앞먹임.
        
        인수:
            x: 앞 풀개 켜에서 온 들임
            skip: 그에 맞는 부호기 켜에서 온 건너뛰는 이음
        """
        x = self.up(x)
        
        # 모으기/되풀기로 생긴 크기 어긋남 다루기
        # 크기를 맞추는 데 필요한 덧대기 셈하기
        diff_h = skip.size()[2] - x.size()[2]
        diff_w = skip.size()[3] - x.size()[3]
        
        # skip의 크기에 맞게 x 덧대기(가운데 잘라내기 대신)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                      diff_h // 2, diff_h - diff_h // 2])
        
        # 통로 차원을 따라 이어 붙인다
        x = torch.cat([skip, x], dim=1)
        
        return self.conv(x)
```

### 온전한 U-넷 얼개

```python
class UNet(nn.Module):
    """
    뜻 나누기를 위한 U-넷 얼개.
    
    이 그물은 오그라드는 길(부호기)과 부풀어 오르는
    길(풀개)로 이루어진다. 부호기는 맥락을 담아내고 풀개는 건너뛰는 이음으로
    정밀한 자리 잡기를 가능하게 한다.
    
    인수:
        in_channels: 들임 채널의 개수(잿빛은 1, RGB는 3)
        num_classes: 출력 클래스의 수
        base_features: 첫 부호기 켜의 특징 수(붙박이: 64)
        bilinear: 두 줄 사이 끼움 키우기를 쓸지 여부(붙박이: True)
    
    보기:
        >>> model = UNet(in_channels=3, num_classes=21)
        >>> x = torch.randn(1, 3, 256, 256)
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([1, 21, 256, 256])
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 2,
                 base_features: int = 64, bilinear: bool = True):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        
        # 특징 흐름: 64 → 128 → 256 → 512 → 1024
        features = [base_features * (2 ** i) for i in range(5)]
        
        # 첫 겹 누비기(줄여 뽑기 없음)
        self.inc = DoubleConv(in_channels, features[0])
        
        # 부호기 길
        self.down1 = EncoderBlock(features[0], features[1])
        self.down2 = EncoderBlock(features[1], features[2])
        self.down3 = EncoderBlock(features[2], features[3])
        
        # 병목(두 줄 사이 끼움 계수가 마지막 부호기에 영향을 준다)
        factor = 2 if bilinear else 1
        self.down4 = EncoderBlock(features[3], features[4] // factor)
        
        # 풀개 길
        self.up1 = DecoderBlock(features[4], features[3] // factor, bilinear)
        self.up2 = DecoderBlock(features[3], features[2] // factor, bilinear)
        self.up3 = DecoderBlock(features[2], features[1] // factor, bilinear)
        self.up4 = DecoderBlock(features[1], features[0], bilinear)
        
        # 마지막 갈래 매기기 층
        self.outc = nn.Conv2d(features[0], num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        U-넷을 지나는 앞먹임.
        
        인수:
            x: 꼴이 (B, C, H, W)인 들임 텐서
        
        반환값:
            꼴이 (B, num_classes, H, W)인 내놓는 텐서
        """
        # 부호기 길(건너뛰는 이음에 쓸 특징 갈무리)
        x1 = self.inc(x)      # (B, 64, H, W)
        x2 = self.down1(x1)   # (B, 128, H/2, W/2)
        x3 = self.down2(x2)   # (B, 256, H/4, W/4)
        x4 = self.down3(x3)   # (B, 512, H/8, W/8)
        x5 = self.down4(x4)   # (B, 1024 또는 512, H/16, W/16) — 병목
        
        # 풀개 길(건너뛰는 이음 포함)
        x = self.up1(x5, x4)  # (B, 512 또는 256, H/8, W/8)
        x = self.up2(x, x3)   # (B, 256 또는 128, H/4, W/4)
        x = self.up3(x, x2)   # (B, 128 또는 64, H/2, W/2)
        x = self.up4(x, x1)   # (B, 64, H, W)
        
        # 분류
        logits = self.outc(x)  # (B, num_classes, H, W)
        
        return logits
    
    def count_parameters(self) -> int:
        """학습 가능한 전체 매개변수의 수를 센다."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

---

## 5. 건너뛰는 이음: 핵심 새로움

### 왜 더하기가 아니라 이어 붙이기인가?

U-넷은 FCN의 더하기와 달리 건너뛰는 이음에 **이어 붙이기**를 쓴다:

```python
# FCN 방식(더하기):
fused = encoder_features + decoder_features

# U-넷 방식(이어 붙이기):
fused = torch.cat([encoder_features, decoder_features], dim=1)
```

**이어 붙이기의 좋은 점:**

1. **본디 앎을 지킨다**: 부호기 특징이 그대로 남는다
2. **더 풍부한 나타냄**: 그물이 두 곳의 무게를 배울 수 있다
3. **매개변수가 더 많다**: 뒤따르는 누비기에 담는 힘이 더해진다
4. **맞물릴 필요가 없다**: 더하려고 특징이 "맞물릴" 까닭이 없다

**수학으로 본 관점:**

$\mathbf{E}$을 부호기 결, $\mathbf{D}$을 촘촘하게 한 풀개 결이라 하자.

더하기: $\mathbf{F} = \mathbf{E} + \mathbf{D}$은 그물이 소식을 아우르는 방식을 옭아맨다.

이어 붙이기: $\mathbf{F} = [\mathbf{E}; \mathbf{D}]$ 뒤에 배운 짐 $\mathbf{W}$으로 누비기를 건다.

$$\mathbf{O} = \mathbf{W}_E \cdot \mathbf{E} + \mathbf{W}_D \cdot \mathbf{D}$$

그물이 가장 좋은 아우름 무게를 배우므로 이쪽이 엄밀히 더 잘 나타낸다.

### 앎의 흐름 그려 보기

```python
def visualize_unet_activations(model, image, layer_names=None):
    """
    건너뛰는 이음을 이해하려 U-넷의 가운데 깨어남을 그려 본다.
    
    인수:
        model: 익힌 U-넷 모델
        image: 들임 그림 텐서 (1, C, H, W)
        layer_names: 그려 볼 층의 목록(없어도 됨)
    """
    import matplotlib.pyplot as plt
    
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # 갈고리 걸기
    model.inc.register_forward_hook(get_activation('enc1'))
    model.down1.register_forward_hook(get_activation('enc2'))
    model.down2.register_forward_hook(get_activation('enc3'))
    model.down3.register_forward_hook(get_activation('enc4'))
    model.down4.register_forward_hook(get_activation('bottleneck'))
    
    # 순전파
    model.eval()
    with torch.no_grad():
        output = model(image)
    
    # 시각화한다
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 부호기 깨어남 그리기
    for idx, (name, act) in enumerate(activations.items()):
        ax = axes[idx // 3, idx % 3]
        # 채널에 걸쳐 고루내기
        feature_map = act[0].mean(dim=0).cpu().numpy()
        ax.imshow(feature_map, cmap='viridis')
        ax.set_title(f'{name}: {tuple(act.shape)}')
        ax.axis('off')
    
    plt.tight_layout()
    return fig, activations
```

---

## 6. U-넷 익히기

### U-넷의 손실 함수

#### 두 갈래 엇갈린 엔트로피(두 갈래 나누기)

두 갈래 나누기(앞바탕/뒷바탕)에서는:

```python
class BCEWithLogitsLoss(nn.Module):
    """
    두 갈래 나누기를 위한 두 갈래 엇갈린 엔트로피 손실.
    
    시그모이드 깨어남과 두 갈래 엇갈린 엔트로피를 수치로 든든한 함수 하나로 아우른다.
    """
    def __init__(self, pos_weight: float = None):
        super().__init__()
        self.pos_weight = pos_weight
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        인수:
            logits: 날 어림 (B, 1, H, W)
            targets: 두 갈래 참값 (B, 1, H, W)
        """
        if self.pos_weight is not None:
            pos_weight = torch.tensor([self.pos_weight], device=logits.device)
            return F.binary_cross_entropy_with_logits(logits, targets, 
                                                       pos_weight=pos_weight)
        return F.binary_cross_entropy_with_logits(logits, targets)
```

#### 엇갈린 엔트로피(여러 갈래 나누기)

여러 갈래 나누기에서는:

```python
def segmentation_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                                ignore_index: int = 255) -> torch.Tensor:
    """
    여러 갈래 나누기를 위한 엇갈린 엔트로피 손실.
    
    인수:
        logits: 어림 (B, K, H, W). 여기서 K는 갈래의 개수
        targets: 참값 갈래 번호 (B, H, W)
        ignore_index: 무시할 번호(보기로 이름표 없는 화소)
    """
    return F.cross_entropy(logits, targets, ignore_index=ignore_index)
```

### 나누기를 위한 자료 불리기

결정적인 요점: 불리기는 그림과 마스크에 **똑같이** 적용해야 한다.

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_training_transforms(image_size: int = 256):
    """
    나누기를 위한 익히기 불리기.
    
    Albumentations는 그림과 마스크에 똑같은 바꾸기가 적용되게 한다.
    """
    return A.Compose([
        # 기하 바꾸기
        A.RandomResizedCrop(image_size, image_size, scale=(0.5, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        # 늘어나는 뒤틀기(의료 그림에 아주 좋다)
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
        
        # 세기 바꾸기(그림에만 영향, 마스크에는 없음)
        A.OneOf([
            A.RandomBrightnessContrast(p=1),
            A.GaussNoise(p=1),
            A.GaussianBlur(blur_limit=3, p=1),
        ], p=0.5),
        
        # 고르게 맞추고 텐서로 바꾸기
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_validation_transforms(image_size: int = 256):
    """검증용 바꾸기(불리기 없이 크기 바꾸기와 고르게 맞추기만)."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
```

### 완전한 학습 루프

```python
def train_unet(model, train_loader, val_loader, num_epochs=50,
               lr=1e-4, device='cuda'):
    """
    U-넷의 온전한 익히기 되풀이.
    
    배움 비율 일정 짜기, 일찍 멈추기, 모델 중간 저장을 담고 있다.
    """
    model = model.to(device)
    
    # 최적화기
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 학습률 스케줄러
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # 손실 함수(두 갈래 나누기용)
    criterion = nn.BCEWithLogitsLoss()
    
    # 익히기 잣대
    best_dice = 0.0
    patience_counter = 0
    max_patience = 10
    
    for epoch in range(num_epochs):
        # 학습 단계
        model.train()
        train_loss = 0.0
        
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # 순전파
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 검증 단계
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                # 다이스 셈하기
                probs = torch.sigmoid(outputs)
                dice = calculate_dice(probs, masks)
                
                val_loss += loss.item()
                val_dice += dice
        
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        
        # 학습률 스케줄링
        scheduler.step(val_dice)
        
        # 기록
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        # 모델 중간 저장
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), 'best_unet.pth')
            print(f"  → New best model saved! Dice: {best_dice:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 조기 종료
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return best_dice

def calculate_dice(pred: torch.Tensor, target: torch.Tensor, 
                   threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """다이스 계수를 셈한다."""
    pred_binary = (pred > threshold).float()
    
    intersection = (pred_binary * target).sum()
    dice = (2.0 * intersection + smooth) / (pred_binary.sum() + target.sum() + smooth)
    
    return dice.item()
```

---

## 7. U-넷 변종과 넓힘

### U-Net++: 겹겹이 든 건너뛰는 이음

U-Net++는 부호기와 풀개 사이에 촘촘한 건너뛰는 이음을 들여온다:

```
Standard U-Net skip:      U-Net++ nested skips:
    E1 ────────────────→ D1       E1 → X1,1 → X1,2 → X1,3 → D1
    E2 ──────────→ D2             E2 → X2,1 → X2,2 → D2
    E3 ────→ D3                   E3 → X3,1 → D3
    E4 → D4                       E4 → D4
```

### 눈길 U-넷

건너뛰는 이음에 눈길 문을 더해 풀개가 알맞은 부호기 특징에 초점을 두게 한다:

```python
class AttentionGate(nn.Module):
    """
    눈길 U-넷의 눈길 문.
    
    건너뛰는 이음에서 온 알맞은 특징을 도드라지게 하는 법을 배운다.
    """
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        
        self.W_g = nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False)
        self.W_x = nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False)
        
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        인수:
            gate: 풀개에서 온 문 신호(더 거친 잣수)
            skip: 부호기에서 온 건너뛰는 이음(더 고운 잣수)
        """
        # skip의 자리 차원에 맞게 문을 키우기
        gate_up = F.interpolate(self.W_g(gate), size=skip.shape[2:], 
                                 mode='bilinear', align_corners=True)
        
        # 아울러 눈길 셈하기
        combined = self.relu(gate_up + self.W_x(skip))
        attention = self.psi(combined)
        
        # 건너뛰는 이음에 눈길 쓰기
        return skip * attention
```

### ResU-넷

덩이마다 안에 잔차 이음을 더한다:

```python
class ResidualDoubleConv(nn.Module):
    """잔차 이음을 갖춘 겹 누비기."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        # 지름길 이음
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv(x)
        out += residual
        return self.relu(out)
```

---

## 8. 성능에서 헤아릴 점

### 기억 공간 가장 좋게 하기

U-넷은 건너뛰는 이음을 위해 부호기 특징을 쌓아 두어야 하므로 기억 공간을 많이 쓸 수 있다:

```python
# 기억 공간 요구량 어림
def estimate_memory(batch_size: int, image_size: int, base_features: int = 64):
    """U-넷의 GPU 기억 공간 요구량을 어림한다."""
    # 켜마다의 특징 지도
    levels = [
        (image_size, base_features),           # 켜 1
        (image_size // 2, base_features * 2),  # 켜 2
        (image_size // 4, base_features * 4),  # 켜 3
        (image_size // 8, base_features * 8),  # 켜 4
        (image_size // 16, base_features * 16), # 병목
    ]
    
    total_elements = 0
    for size, channels in levels:
        # 건너뛰는 이음에 쓸 부호기 갈무리
        total_elements += batch_size * channels * size * size
    
    # 앞먹임 + 뒤먹임이라 2배, float32라 4바이트
    memory_bytes = total_elements * 2 * 4
    memory_gb = memory_bytes / (1024 ** 3)
    
    return memory_gb

# 보기: batch_size=8, image_size=512
print(f"Estimated memory: {estimate_memory(8, 512):.2f} GB")
```

**출력:**

```
Estimated memory: 1.94 GB
```

### 기울기 검문점

셈을 더 하고 기억 공간을 아낀다:

```python
from torch.utils.checkpoint import checkpoint

class MemoryEfficientUNet(UNet):
    """기억 공간을 아끼려 기울기 중간 저장을 쓴 U-넷."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 부호기 덩이에 중간 저장 쓰기
        x1 = checkpoint(self.inc, x)
        x2 = checkpoint(self.down1, x1)
        x3 = checkpoint(self.down2, x2)
        x4 = checkpoint(self.down3, x3)
        x5 = checkpoint(self.down4, x4)
        
        # 풀개(효율을 위해 보통 중간 저장하지 않는다)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        return self.outc(x)
```

---

## 연습문제

**연습문제 1.**
뜻 나누기, 낱 물체 나누기, 온통 나누기의 차이를 설명하여라.

??? success "연습문제 1 풀이"
    **뜻 나누기**는 화소마다 갈래 이름표를 붙이되 같은 갈래의 서로 다른 낱 물체를 가리지 않는다. **낱 물체 나누기**는 낱낱의 물체를 알아내고 저마다 화소 수준 마스크를 주되 셀 수 있는 "것" 갈래에만 그렇게 한다. **온통 나누기**는 둘을 아우른다. 곧 화소마다 갈래 이름표를 붙이고 것 갈래에는 낱 물체 번호도 매긴다. 보기로 거리 장면에서 뜻 나누기는 모든 차를 "차"로 이름 붙이고, 낱 물체 나누기는 차를 하나하나 가려내며, 온통 나누기는 그 둘을 다 하면서 "길", "하늘" 따위도 이름 붙인다.

---

**연습문제 2.**
U-넷 얼개를 설명하고 나누기에서 건너뛰는 이음이 왜 중요한지 밝혀라.

??? success "연습문제 2 풀이"
    U-넷은 오그라드는 부호기 길(누비기와 모으기의 되풀이)과 부풀어 오르는 풀개 길(키우기와 누비기)로 이루어져 U 꼴을 이룬다. **건너뛰는 이음**은 부호기의 특징 지도를 그에 맞는 풀개 켜에 이어 붙인다. 부호기가 *무엇*(뜻 특징)을 담아내면서 *어디*(자리의 세밀함)를 잃기 때문에 이것이 결정적이다. 건너뛰는 이음은 정밀한 화소 수준 어림에 필요한 높은 해상도의 자리 앎을 주어, 풀개가 거친 뜻 앎과 고운 자리 세부를 아우르게 한다.

---

**연습문제 3.**
그림 나누기에는 어떤 손실 함수가 흔히 쓰이는가? 엇갈린 엔트로피 손실과 다이스 손실을 견주어라.

??? success "연습문제 3 풀이"
    **엇결 엔트로피 잃음**은 화소마다 따로 다룬다. $L_{CE} = -\sum_i y_i \log \hat{y}_i$이다. 눈금은 잘 맞지만 수가 많은 갈래에 휘둘릴 수 있다. **다이스 잃음**은 미루어 본 가림과 참 가림이 겹치는 정도를 잰다. $L_{Dice} = 1 - \frac{2|P \cap G|}{|P| + |G|}$이다. 다이스 잃음은 따짐 자(다이스 계수)를 곧바로 가장 좋게 하고, 화소 수와 상관없이 갈래마다 같은 짐을 주므로 갈래 치우침도 더 잘 다룬다. 참으로는 둘을 섞은 $L = \lambda L_{CE} + (1-\lambda) L_{Dice}$이 가장 잘 듣는 일이 잦다.

---

**연습문제 4.**
마스크 R-CNN이 낱 물체 나누기를 위해 더 빠른 R-CNN을 어떻게 넓히는지 밝히고 RoIAlign이 하는 몫을 설명하여라.

??? success "연습문제 4 풀이"
    마스크 R-CNN은 이미 있던 갈래 매기기 가지와 두름 상자 되돌리기 가지 곁에, 알아낸 물체마다 두 갈래 마스크를 어림하는 나란한 가지를 더한다. 핵심 새로움은 RoI 모으기를 갈음하는 **RoIAlign**이다. RoI 모으기는 양자화된 자리표(정수 화소 자리로 반올림)를 써서 특징 지도와 본디 그림 사이가 어긋난다. RoIAlign은 정확한 뜬소수점 자리에서 두 줄 사이 끼움을 써서 양자화 찌꺼기를 없앤다. 이 정밀한 맞춤이 화소 수준 마스크 어림에 결정적이며, RoI 모으기에 견주어 마스크 AP를 상대적으로 10~50% 올린다.

## 정리하며

U-넷은 이어 붙이기 방식의 건너뛰는 이음을 갖춘 우아한 대칭 부호기-풀개 꾸밈으로 뜻 나누기, 특히 의료 영상의 판을 뒤집었다. 이 얼개는 높은 해상도의 부호기 특징과 뜻이 담긴 풀개 나타냄을 아울러 자리 잡기와 맥락 사이의 맞바꿈을 잘 다룬다.

핵심 갈무리:

1. **대칭 꾸밈** 덕분에 풀개 켜마다 같은 해상도의 부호기 특징을 쓸 수 있다
2. **이어 붙이는 건너뛰는 이음**은 더하기보다 앎을 더 많이 지킨다
3. 자료 불리기를 세게 하고 건너뛰는 이음을 써서 **자료가 적어도 잘 된다**
4. **요즘 변종의 바탕**: U-Net++, 눈길 U-넷, ResU-넷

U-넷은 여전히 아주 경쟁력이 있으며, 의료 영상이나 자료가 적은 다른 나누기 일에서 흔히 가장 먼저 써 보는 얼개이다.

**참고 문헌**

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.
2. Zhou, Z., et al. (2018). UNet++: A Nested U-Net Architecture for Medical Image Segmentation. DLMIA.
3. Oktay, O., et al. (2018). Attention U-Net: Learning Where to Look for the Pancreas. MIDL.
4. Isensee, F., et al. (2021). nnU-Net: A Self-configuring Method for Deep Learning-based Biomedical Image Segmentation. Nature Methods.
