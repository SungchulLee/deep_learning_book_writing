# U-Net: 이미지 분할을 위한 인코더–디코더

U-Net은 해상도를 줄이는 인코더와 복원하는 디코더를 스킵 연결로 결합하는 분할 모델이다. Ronneberger 외의 2015년 논문은 생의학 영상에서 강한 데이터 어그멘테이션과 함께 이 구조를 평가했다. 데이터가 적어도 항상 잘 학습되거나 임상적 정확도가 자동으로 보장된다는 뜻은 아니다. 아래 코드는 원 논문의 valid convolution과 다른 padding·BN 변형이며, 기본 bilinear 경로에는 채널 불일치가 있다.

## 1. 원 구조와 패딩 변형

원 U-Net은 padding 없는 컨볼루션 때문에 피처 크기가 줄어들고 스킵을 crop해 결합한다. 다음 도식의 572→388 흐름은 이 원 구조의 크기 예다. 아래 padding=1 코드가 같은 입출력 크기 감소를 갖는다고 해석하지 않는다.

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

전치 컨볼루션은 학습 가능한 업샘플 연산이며 일반적인 컨볼루션의 역함수는 아니다. 보간과 후속 컨볼루션도 업샘플링을 구성하는 대안이다. 디코더 해상도를 키우는 것과 다운샘플링으로 잃은 정보를 완전히 되찾는 것은 구분한다.

## 2. 구성 블록

`DoubleConv`는 padding=1인 3×3 두 개에 BN·ReLU를 적용해 공간 크기를 유지한다. 원 논문의 padding 없는 블록과 다르다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    겹 컨볼루션 덩이: (Conv → BN → ReLU) × 2
    
    이것이 U-넷의 핵심 벽돌이다. 인코더와 디코더의
    켜마다 배치 고르게 맞추기와 ReLU 활성화를 곁들인
    잇단 3×3 컨볼루션 둘로 이루어진다.
    
    인수:
        in_channels: 들임 채널의 개수
        out_channels: 내놓는 채널의 개수
        mid_channels: 첫 컨볼루션 뒤의 채널 개수(붙박이: out_channels)
    """
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        
        if mid_channels is None:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            # 첫 컨볼루션
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            
            # 두 번째 컨볼루션
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)
```

`EncoderBlock`은 2×2 최대 풀링 후 채널을 바꾼다. 홀수 공간 크기는 바닥 나눗셈으로 줄어든다.

```python
class EncoderBlock(nn.Module):
    """
    인코더 덩이: 최대 모으기 → 겹 컨볼루션
    
    자리 차원을 2분의 1로 줄이고 겹 컨볼루션을 쓴다.
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

`DecoderBlock`은 두 업샘플 방식 모두 입력 채널을 절반으로 줄인 뒤 스킵과 연결한다. padding은 공간 차이를 맞추지만 채널 수의 불일치를 고치지는 않는다.

```python
class DecoderBlock(nn.Module):
    """
    디코더 덩이: 키우기 → 이어 붙이기 → 겹 컨볼루션
    
    자리 차원을 2배로 키우고 인코더 피처와 이어 붙인 뒤
    건너뛰는 이음으로 이어 붙인 뒤 겹 컨볼루션을 쓴다.
    
    인수:
        in_channels: 들임 채널의 개수(앞 디코더 켜에서 옴)
        out_channels: 내놓는 채널의 개수
        bilinear: True이면 두 줄 사이 끼움 키우기, False이면 뒤바꾼 컨볼루션
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        
        if bilinear:
            # 두 줄 사이 끼움 키우기 + 채널을 줄이는 1×1 컨볼루션
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
            )
            # 이어 붙인 뒤: in_channels // 2 + in_channels // 2 = in_channels
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # 뒤바꾼 컨볼루션(배울 수 있는 키우기)
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 
                                          kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        건너뛰는 이음을 쓴 피드포워드.
        
        인수:
            x: 앞 디코더 켜에서 온 들임
            skip: 그에 맞는 인코더 켜에서 온 건너뛰는 이음
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

다음 전체 구성에서 `bilinear=False`는 1024채널 병목을 첫 전치 컨볼루션에 전달하고 스킵을 합해 의도한 폭을 맞춘다. 그러나 **기본 bilinear=True**는 `factor=2`로 병목을 512채널로 줄이면서 첫 `DecoderBlock`을 in_channels=1024로 생성한다. 그 블록의 1×1은 1024입력을 기대하므로 512채널 병목과 맞지 않는다. 뒤 단계에도 같은 폭 설계 문제가 남는다. 코드가 옵션을 선언했다고 두 경로가 모두 유효한 구현은 아니다.

```python
class UNet(nn.Module):
    """
    뜻 나누기를 위한 U-넷 구조.
    
    이 넷트웍은 오그라드는 길(인코더)과 부풀어 오르는
    길(디코더)로 이루어진다. 인코더는 맥락을 담아내고 디코더는 건너뛰는 이음으로
    정밀한 자리 잡기를 가능하게 한다.
    
    인수:
        in_channels: 들임 채널의 개수(잿빛은 1, RGB는 3)
        num_classes: 출력 클래스의 수
        base_features: 첫 인코더 켜의 피처 수(붙박이: 64)
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
        
        # 피처 흐름: 64 → 128 → 256 → 512 → 1024
        features = [base_features * (2 ** i) for i in range(5)]
        
        # 첫 겹 컨볼루션(줄여 뽑기 없음)
        self.inc = DoubleConv(in_channels, features[0])
        
        # 인코더 길
        self.down1 = EncoderBlock(features[0], features[1])
        self.down2 = EncoderBlock(features[1], features[2])
        self.down3 = EncoderBlock(features[2], features[3])
        
        # 병목(두 줄 사이 끼움 계수가 마지막 인코더에 영향을 준다)
        factor = 2 if bilinear else 1
        self.down4 = EncoderBlock(features[3], features[4] // factor)
        
        # 디코더 길
        self.up1 = DecoderBlock(features[4], features[3] // factor, bilinear)
        self.up2 = DecoderBlock(features[3], features[2] // factor, bilinear)
        self.up3 = DecoderBlock(features[2], features[1] // factor, bilinear)
        self.up4 = DecoderBlock(features[1], features[0], bilinear)
        
        # 마지막 클래스 매기기 레이어
        self.outc = nn.Conv2d(features[0], num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        U-넷을 지나는 피드포워드.
        
        인수:
            x: 꼴이 (B, C, H, W)인 들임 텐서
        
        반환값:
            꼴이 (B, num_classes, H, W)인 내놓는 텐서
        """
        # 인코더 길(건너뛰는 이음에 쓸 피처 갈무리)
        x1 = self.inc(x)      # (B, 64, H, W)
        x2 = self.down1(x1)   # (B, 128, H/2, W/2)
        x3 = self.down2(x2)   # (B, 256, H/4, W/4)
        x4 = self.down3(x3)   # (B, 512, H/8, W/8)
        x5 = self.down4(x4)   # (B, 1024 또는 512, H/16, W/16) — 병목
        
        # 디코더 길(건너뛰는 이음 포함)
        x = self.up1(x5, x4)  # (B, 512 또는 256, H/8, W/8)
        x = self.up2(x, x3)   # (B, 256 또는 128, H/4, W/4)
        x = self.up3(x, x2)   # (B, 128 또는 64, H/2, W/2)
        x = self.up4(x, x1)   # (B, 64, H, W)
        
        # 분류
        logits = self.outc(x)  # (B, num_classes, H, W)
        
        return logits
    
    def count_parameters(self) -> int:
        """학습 가능한 전체 파라메터의 수를 센다."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

코드는 요청대로 유지했다. 실제 학습 설명은 채널·공간 조건이 맞는 모델을 전제로 읽는다. 네 번 풀링할 수 있는 입력 크기와 가장 작은 단계의 BN 표본 수도 필요하다.

## 3. 스킵의 연결과 덧셈

채널 연결은 인코더·디코더 피처를 따로 보존해 다음 컨볼루션이 각각을 조합하게 한다. 덧셈에는 두 피처의 전체 형태가 같아야 하고, 채널 연결에는 배치·공간 크기가 같으면 채널 수는 달라도 된다. 공간 정렬이나 좌표의 대응까지 불필요한 것은 아니다.

```python
# FCN 방식(더하기):
fused = encoder_features + decoder_features

# U-넷 방식(이어 붙이기):
fused = torch.cat([encoder_features, decoder_features], dim=1)
```

선형 후속레이어에서 연결하면 W_E E+W_D D처럼 두 부분의 웨이트를 다르게 둘 수 있다. 같은 폭의 덧셈 뒤 하나의 W를 적용한 W(E+D)보다 조합의 자유도가 늘 수 있지만 채널·파라메터·비용이 다르다. 모든 덧셈 기반 구조보다 엄격히 더 정확하거나 항상 정보를 보존한다는 결론은 아니다.

다음 시각화 함수는 평균 활성 지도를 그리며, 모델의 의미적 설명이나 분할 정확도를 검증하지 않는다. `layer_names`는 사용되지 않고 hook handle을 제거하지 않아 반복 호출하면 hook이 누적된다. 기본 bilinear 경로의 채널 문제도 그대로 영향을 준다. 함수는 실행하거나 그림을 재생성하지 않았다.

```python
def visualize_unet_activations(model, image, layer_names=None):
    """
    건너뛰는 이음을 이해하려 U-넷의 가운데 활성화를 그려 본다.
    
    인수:
        model: 익힌 U-넷 모델
        image: 들임 그림 텐서 (1, C, H, W)
        layer_names: 그려 볼 레이어의 목록(없어도 됨)
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
    
    # 인코더 활성화 그리기
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

## 4. 로짓·마스크와 로스

한 채널 이진 분할은 출력 N×1×H×W와 같은 형태의 부동소수점 0/1 마스크에 BCEWithLogitsLoss를 사용할 수 있다. sigmoid를 미리 취한 값을 logits 로스에 다시 넣지 않는다. pos_weight는 양성 항을 조절하며 적절한 장치·자료형과 브로드캐스트가 필요하다.

```python
class BCEWithLogitsLoss(nn.Module):
    """
    두 분류를 위한 두 클래스 엇갈린 엔트로피 로스.
    
    시그모이드 활성화와 두 클래스 엇갈린 엔트로피를 수치로 든든한 함수 하나로 아우른다.
    """
    def __init__(self, pos_weight: float = None):
        super().__init__()
        self.pos_weight = pos_weight
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        인수:
            logits: 날 어림 (B, 1, H, W)
            targets: 두 클래스 참값 (B, 1, H, W)
        """
        if self.pos_weight is not None:
            pos_weight = torch.tensor([self.pos_weight], device=logits.device)
            return F.binary_cross_entropy_with_logits(logits, targets, 
                                                       pos_weight=pos_weight)
        return F.binary_cross_entropy_with_logits(logits, targets)
```

상호 배타적인 K클래스는 N×K×H×W 로짓과 N×H×W의 정수 레이블에 교차 엔트로피를 사용한다. 클래스별로 독립적인 여러 마스크의 BCE와 다른 문제다. 무시 레이블의 처리와 모든 픽셀이 무시된 경우도 명시한다.

```python
def segmentation_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                                ignore_index: int = 255) -> torch.Tensor:
    """
    여러 분류를 위한 엇갈린 엔트로피 로스.
    
    인수:
        logits: 어림 (B, K, H, W). 여기서 K는 클래스의 개수
        targets: 참값 클래스 번호 (B, H, W)
        ignore_index: 무시할 번호(보기로 이름표 없는 화소)
    """
    return F.cross_entropy(logits, targets, ignore_index=ignore_index)
```

기본 `UNet(num_classes=2)`의 두 채널 출력에 이진 한 채널 학습 루프를 그대로 연결하면 목표의 형태와 의미가 맞지 않는다. 한 로짓의 이진 모델 또는 두 클래스 CE 중 일관된 설계를 선택해야 한다. 로스를 바꾸었다고 올바른 목표 인코딩이 자동으로 생기지는 않는다.

## 5. 이미지와 마스크의 어그멘테이션

회전·crop·resize 등 기하 변환은 이미지와 마스크의 좌표·난수를 동기화한다. 정수 클래스 마스크에는 레이블을 보존하는 보간을 사용하고 밝기·색상·노이즈 변환은 일반적으로 이미지에만 적용한다. ‘모든 어그멘테이션을 이미지와 마스크에 똑같이 적용’하는 것은 부정확하다. 입력의 의미를 바꾸지 않는 변환인지도 데이터에 맞춰 확인한다.

아래 코드는 Albumentations 호출 규약에 의존한다. 예를 들어 현재 [RandomResizedCrop 문서](https://albumentations.ai/explore/transform/RandomResizedCrop/docs/)는 size=(height,width)를 사용하므로 코드의 두 위치 인수 형식과 설치 버전의 호환성을 확인해야 한다. ImageNet 평균·표준편차는 한 전처리 선택이며 모든 영상·학습에 필수인 것은 아니다. 마스크의 채널·자료형은 선택한 BCE/CE 형식에 따로 맞춘다.

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

## 6. 학습 루프의 평가 단위

아래 루프는 BCE 기반 이진 분할을 전제로 한다. 입력 마스크를 자동으로 float·N×1×H×W로 변환하지 않고, CUDA의 가용성이나 라이브러리 버전별 scheduler 인수도 일반화하지 않는다. 올바른 모델·데이터와 실행 환경을 외부에서 구성해야 한다.

`calculate_dice`는 확률을 0.5로 임계화하고 배치 전체 픽셀을 합친 Dice를 구한다. 배치별 Dice를 단순 평균한 값은 데이터 전체의 교집합·분모를 누적한 Dice와 일반적으로 다르며 배치 구성에 따라 달라질 수 있다. 빈 양쪽 마스크는 smooth 규칙으로1이 된다. 로스 평균 역시 마지막 배치 크기가 다르면 전체 관측당 평균과 다를 수 있다.

best_dice는0에서 시작하고 엄격한 개선에서만 저장하므로 평가값이 계속0이면 체크포인트가 없을 수 있다. 반환값은 최선 Dice이며 최선 웨이트를 반환하거나 자동으로 다시 불러오지 않는다. 학습·평가 기준을 이 구현에 맞춰 해석한다.

```python
def train_unet(model, train_loader, val_loader, num_epochs=50,
               lr=1e-4, device='cuda'):
    """
    U-넷의 온전한 익히기 되풀이.
    
    러닝 비율 일정 짜기, 일찍 멈추기, 모델 중간 저장을 담고 있다.
    """
    model = model.to(device)
    
    # 최적화기
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 러닝레이트 스케줄러
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # 로스 함수(두 분류용)
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
        
        # 러닝레이트 스케줄링
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

## 7. 변형의 연결 위치

UNet++는 인코더–디코더 사이에 중첩된 연결을 구성하는 별도 모델이다. 아래 도식만으로 이 파일의 UNet이 UNet++가 되는 것은 아니다.

```
Standard U-Net skip:      U-Net++ nested skips:
    E1 ────────────────→ D1       E1 → X1,1 → X1,2 → X1,3 → D1
    E2 ──────────→ D2             E2 → X2,1 → X2,2 → D2
    E3 ────→ D3                   E3 → X3,1 → D3
    E4 → D4                       E4 → D4
```

AttentionGate는 디코더 문맥과 스킵 피처를 공간 정렬해 sigmoid 마스크를 만든다. 이 클래스를 정의해도 기본 UNet의 decoder에 자동 삽입되지는 않는다.

```python
class AttentionGate(nn.Module):
    """
    어텐션 U-넷의 어텐션 문.
    
    건너뛰는 이음에서 온 알맞은 피처를 도드라지게 하는 법을 배운다.
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
            gate: 디코더에서 온 문 신호(더 거친 잣수)
            skip: 인코더에서 온 건너뛰는 이음(더 고운 잣수)
        """
        # skip의 자리 차원에 맞게 문을 키우기
        gate_up = F.interpolate(self.W_g(gate), size=skip.shape[2:], 
                                 mode='bilinear', align_corners=True)
        
        # 아울러 어텐션 셈하기
        combined = self.relu(gate_up + self.W_x(skip))
        attention = self.psi(combined)
        
        # 건너뛰는 이음에 어텐션 쓰기
        return skip * attention
```

ResidualDoubleConv는 형태를 맞춘 지름길과 두 컨볼루션을 더한다. 이를 실제 네트워크에서 사용하려면 DoubleConv를 바꿀 위치와 채널 구성을 맞춰야 한다. 기본 bilinear 경로의 채널 오류를 이 블록 정의만으로 해결하지는 않는다.

```python
class ResidualDoubleConv(nn.Module):
    """잔차 이음을 갖춘 겹 컨볼루션."""
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

## 8. 메모리 추정과 체크포인팅

아래 메모리 함수는 각 해상도의 대표 활성 하나를 세어 ‘두 배·float32’라는 계수를 곱한 거친 추정이다. 실제 decoder 연결·BN 상태·각 컨볼루션의 저장 활성·파라메터·경사·옵티마이저·작업공간을 모두 합산한 peak 메모리가 아니다. 함수의 1024³ 단위는 GiB이며 코드 출력의 GB 표기는 이 단위로 해석한다.

```python
# 기억 공간 요구량 어림
def estimate_memory(batch_size: int, image_size: int, base_features: int = 64):
    """U-넷의 GPU 기억 공간 요구량을 어림한다."""
    # 켜마다의 피처 지도
    levels = [
        (image_size, base_features),           # 켜 1
        (image_size // 2, base_features * 2),  # 켜 2
        (image_size // 4, base_features * 4),  # 켜 3
        (image_size // 8, base_features * 8),  # 켜 4
        (image_size // 16, base_features * 16), # 병목
    ]
    
    total_elements = 0
    for size, channels in levels:
        # 건너뛰는 이음에 쓸 인코더 갈무리
        total_elements += batch_size * channels * size * size
    
    # 피드포워드 + 뒤먹임이라 2배, float32라 4바이트
    memory_bytes = total_elements * 2 * 4
    memory_gb = memory_bytes / (1024 ** 3)
    
    return memory_gb

# 보기: batch_size=8, image_size=512
print(f"Estimated memory: {estimate_memory(8, 512):.2f} GB")
```

다음은 기존 출력이며 실측 메모리가 아니다.

```
Estimated memory: 1.94 GB
```

체크포인팅은 일부 중간값을 재계산해 저장량을 줄이는 방법이다. 마지막 스킵처럼 이후 연산에 필요한 피처 자체는 여전히 보유한다. 아래 예제는 use_reentrant 등을 지정하지 않으며 재진입 방식에서 입력이 requires_grad=False인 첫 블록은 파라미터 경사 처리에 문제가 생길 수 있다. 라이브러리 방식·미분 조건을 명시해야 하고, BN 같은 상태를 가진 연산의 재계산도 이동 통계에 영향을 줄 수 있다. 채널 구성 오류는 체크포인팅으로 해결되지 않는다.

```python
from torch.utils.checkpoint import checkpoint

class MemoryEfficientUNet(UNet):
    """기억 공간을 아끼려 그래디언트 중간 저장을 쓴 U-넷."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 인코더 덩이에 중간 저장 쓰기
        x1 = checkpoint(self.inc, x)
        x2 = checkpoint(self.down1, x1)
        x3 = checkpoint(self.down2, x2)
        x4 = checkpoint(self.down3, x3)
        x5 = checkpoint(self.down4, x4)
        
        # 디코더(효율을 위해 보통 중간 저장하지 않는다)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        return self.outc(x)
```

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
원 U-Net 도식과 이 파일의 padding·BN 구현을 구분하라.

</div>

??? success "연습문제 1 풀이"
    원 도식은 padding 없는 컨볼루션과 crop 스킵으로 출력이 줄어드는 예다. 이 코드는 padding=1과 BN으로 컨볼루션 공간을 유지하고 업샘플 뒤 padding으로 정렬한다. 원 도식의572→388 크기를 현재 코드의 출력 규칙으로 적용하면 안 된다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
기본 bilinear=True의 첫 decoder에서 채널 불일치를 설명하라.

</div>

??? success "연습문제 2 풀이"
    factor=2로 병목은512채널인데 첫 DecoderBlock은 in_channels=1024로 생성된다. bilinear 업샘플 뒤 1×1이1024입력을 기대해512입력과 맞지 않는다. 공간 padding은 이 오류를 고치지 못한다. 전치 컨볼루션 경로와 같은 폭 계산을 자동으로 공유할 수는 없다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이진 한 채널 BCE와 두 클래스 CE의 형태를 쓰라.

</div>

??? success "연습문제 3 풀이"
    BCE는 N×1×H×W 로짓과 같은 형태의 float0/1 마스크를 사용한다. 두 클래스 CE는 N×2×H×W 로짓과 N×H×W의 정수0/1 레이블을 사용한다. 기본 num_classes=2와 BCE 루프는 별도 조정 없이 같은 설정이 아니다. sigmoid/softmax를 logits 로스에 중복 적용하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
학습 루프의 배치 Dice 평균과 전체 데이터 Dice가 다른 이유를 설명하라.

</div>

??? success "연습문제 4 풀이"
    각 배치의 교집합/분모 비율을 평균하면 배치가 동일 웨이트를 받는다. 전체 Dice는 교집합과 분모를 모두 누적한 비율이므로 객체·배치 크기에 따라 웨이트가 달라진다. 임계값·빈 마스크 규칙과 집계 단위를 명시해야 한다.

## 정리하며

U-Net은 스킵으로 문맥과 높은 해상도 피처를 결합한다. 이 예제의 원 논문과 다른 패딩 구조, 기본 bilinear 채널 불일치, 이진·다중 클래스 로스의 형태를 구분한다. 어그멘테이션의 좌표 동기화와 평가 집계, 메모리 추정·체크포인팅의 실제 조건도 함께 확인한다.

**참고 문헌**

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.
2. Zhou, Z., et al. (2018). UNet++: A Nested U-Net Architecture for Medical Image Segmentation. DLMIA.
3. Oktay, O., et al. (2018). Attention U-Net: Learning Where to Look for the Pancreas. MIDL.
4. Isensee, F., et al. (2021). nnU-Net: A Self-configuring Method for Deep Learning-based Biomedical Image Segmentation. Nature Methods.
