# 느림빠름 그물

---

## 1. 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

- 두 갈래 길이라는 꾸밈 철학을 이해한다
- 옆 이음을 갖춘 느림 길과 빠름 길을 짠다
- α(틀 비율)와 β(채널) 비를 맞춘다
- 몸짓 알아보기를 위해 느림빠름 그물을 익힌다
- 느림빠름을 여러 영상 이해 일에 쓴다

---

## 2. 꾸밈 철학

### 생물에서 얻은 실마리

사람의 보기 체계는 때의 잣수를 달리하며 앎을 다룬다:

- **큰세포 길**: 빠르고 자리 해상도가 낮으며 움직임에 민감하다
- **작은세포 길**: 느리고 자리 해상도가 높으며 세부에 밝다

느림빠름은 그물 길 둘로 이를 흉내낸다:

### 두 갈래 길

| 길 | 틀 비율 | 채널 | 초점 |
|---------|-----------|----------|-------|
| **느림** | 낮다(보기로 초당 4틀) | 많다(보기로 64) | 자리의 뜻 |
| **빠름** | 높다(보기로 초당 32틀) | 적다(보기로 8) | 때에 걸친 움직임 |

### 핵심 매개변수

**α(알파)**: 두 길 사이의 틀 비율 비

$$\text{Fast frames} = \alpha \times \text{Slow frames}$$

보통 α = 8이며, 이는 빠름 길이 틀을 8배 더 다룬다는 뜻이다.

**β(베타)**: 채널 비

$$\text{Fast channels} = \beta \times \text{Slow channels}$$

보통 β = 1/8이며, 이는 빠름 길의 채널이 8배 적다는 뜻이다.

---

## 3. 구조

```python
import torch
import torch.nn as nn

class SlowFast(nn.Module):
    """
    영상 알아보기를 위한 느림빠름 그물
    (Feichtenhofer et al., 2019)
    
    두 갈래 길 얼개:
    - 느림 길: 낮은 틀 비율, 많은 채널
    - 빠름 길: 높은 틀 비율, 적은 채널
    - 옆 이음이 길 사이의 앎을 녹여 붙인다
    """
    
    def __init__(self,
                 num_classes: int = 400,
                 alpha: int = 8,      # 틀 비율 비
                 beta: float = 1/8,   # 채널 비
                 slow_channels: int = 64,
                 num_frames: int = 32):
        super().__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.num_frames = num_frames
        
        fast_channels = int(slow_channels * beta)
        
        # 느림 길
        self.slow_pathway = SlowPathway(
            in_channels=3,
            base_channels=slow_channels
        )
        
        # 빠름 길
        self.fast_pathway = FastPathway(
            in_channels=3,
            base_channels=fast_channels,
            alpha=alpha
        )
        
        # 옆 이음(빠름 → 느림)
        self.lateral_connections = nn.ModuleList([
            LateralConnection(fast_channels * mult, slow_channels * mult, alpha)
            for mult in [1, 2, 4, 8]  # 단계마다
        ])
        
        # 마지막 가려내기
        slow_out = slow_channels * 8
        fast_out = fast_channels * 8
        self.head = nn.Linear(slow_out + fast_out, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        인수:
            x: 영상 텐서 (B, C, T, H, W)
               T는 alpha로 나누어떨어져야 한다
        
        반환값:
            갈래 로짓 (B, num_classes)
        """
        B, C, T, H, W = x.shape
        
        # 길마다 틀 뽑기
        # 느림: alpha번째 틀마다
        x_slow = x[:, :, ::self.alpha, :, :]  # (B, C, T/α, H, W)
        
        # 빠름: 모든 틀
        x_fast = x  # (B, C, T, H, W)
        
        # 옆 이음을 갖춘 길로 다루기
        slow_features = []
        fast_features = []
        
        x_slow, x_fast = self._forward_stem(x_slow, x_fast)
        
        for i, (slow_block, fast_block, lateral) in enumerate(
            zip(self.slow_pathway.stages, 
                self.fast_pathway.stages,
                self.lateral_connections)
        ):
            x_fast = fast_block(x_fast)
            lateral_out = lateral(x_fast)
            
            # 옆 이음을 느림 길과 녹여 붙이기
            x_slow = torch.cat([x_slow, lateral_out], dim=1)
            x_slow = slow_block(x_slow)
        
        # 전체 평균 모으기
        x_slow = x_slow.mean(dim=[2, 3, 4])  # (B, C_slow)
        x_fast = x_fast.mean(dim=[2, 3, 4])  # (B, C_fast)
        
        # 이어 붙이고 갈래 매기기
        x = torch.cat([x_slow, x_fast], dim=1)
        
        return self.head(x)

class SlowPathway(nn.Module):
    """
    느림 길: 채널을 많이 쓰고 틀은 적게 다룬다.
    
    자리의 뜻과 물체 알아보기에 초점을 둔다.
    """
    
    def __init__(self, in_channels: int, base_channels: int):
        super().__init__()
        
        # 줄기
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 
                     kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3)),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        
        # 잔차 단계(채널: 64 → 128 → 256 → 512)
        self.stages = nn.ModuleList([
            self._make_stage(base_channels, base_channels, blocks=3, temporal_stride=1),
            self._make_stage(base_channels, base_channels * 2, blocks=4, temporal_stride=1),
            self._make_stage(base_channels * 2, base_channels * 4, blocks=6, temporal_stride=2),
            self._make_stage(base_channels * 4, base_channels * 8, blocks=3, temporal_stride=2),
        ])
    
    def _make_stage(self, in_ch, out_ch, blocks, temporal_stride):
        layers = [ResBlock3D(in_ch, out_ch, temporal_stride=temporal_stride)]
        for _ in range(1, blocks):
            layers.append(ResBlock3D(out_ch, out_ch))
        return nn.Sequential(*layers)

class FastPathway(nn.Module):
    """
    빠름 길: 채널을 적게 쓰고 틀을 더 많이 다룬다.
    
    움직임과 때에 걸친 흐름에 초점을 둔다.
    """
    
    def __init__(self, in_channels: int, base_channels: int, alpha: int):
        super().__init__()
        
        self.alpha = alpha
        
        # 움직임을 담아내려 때 알맹이를 키운 줄기
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels,
                     kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3)),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        
        # 때를 줄여 뽑는 단계
        self.stages = nn.ModuleList([
            self._make_stage(base_channels, base_channels, blocks=3, temporal_stride=1),
            self._make_stage(base_channels, base_channels * 2, blocks=4, temporal_stride=2),
            self._make_stage(base_channels * 2, base_channels * 4, blocks=6, temporal_stride=2),
            self._make_stage(base_channels * 4, base_channels * 8, blocks=3, temporal_stride=2),
        ])
    
    def _make_stage(self, in_ch, out_ch, blocks, temporal_stride):
        layers = [ResBlock3D(in_ch, out_ch, temporal_stride=temporal_stride)]
        for _ in range(1, blocks):
            layers.append(ResBlock3D(out_ch, out_ch))
        return nn.Sequential(*layers)

class LateralConnection(nn.Module):
    """
    빠름 길에서 느림 길로 가는 옆 이음.
    
    느림 길의 틀 비율에 맞추려 때를 줄여 뽑고,
    그다음 녹여 붙이려 채널을 바꾼다.
    """
    
    def __init__(self, fast_channels: int, slow_channels: int, alpha: int):
        super().__init__()
        
        self.alpha = alpha
        
        # 때 줄여 뽑기: 빠름 틀 → 느림 틀
        # 성큼을 준 3차원 누비기 쓰기
        self.transform = nn.Sequential(
            nn.Conv3d(fast_channels, fast_channels * 2,
                     kernel_size=(5, 1, 1), 
                     stride=(alpha, 1, 1),
                     padding=(2, 0, 0)),
            nn.BatchNorm3d(fast_channels * 2),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x_fast: torch.Tensor) -> torch.Tensor:
        """
        인수:
            x_fast: 빠름 길의 특징 (B, C_fast, T_fast, H, W)
        
        반환값:
            느림 길에 맞춘 옆 특징 (B, C_out, T_slow, H, W)
        """
        return self.transform(x_fast)

class ResBlock3D(nn.Module):
    """3차원 잔차 덩이."""
    
    def __init__(self, in_channels, out_channels, temporal_stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, 
                               kernel_size=3, stride=(temporal_stride, 1, 1), padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Identity()
        if in_channels != out_channels or temporal_stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                         stride=(temporal_stride, 1, 1)),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(identity)
        return self.relu(out)
```

---

## 4. 자리매김 변종

```python
def slowfast_4x16_r50():
    """
    SlowFast 4×16 R50:
    - 느림: 틀 4개, ResNet-50 등뼈
    - 빠름: 틀 32개(α=8), 채널 1/8(β=1/8)
    """
    return SlowFast(
        num_classes=400,
        alpha=8,
        beta=1/8,
        slow_channels=64,
        num_frames=32
    )

def slowfast_8x8_r101():
    """
    SlowFast 8×8 R101:
    - 느림: 틀 8개
    - 빠름: 틀 32개(α=4)
    - ResNet-101 등뼈
    """
    return SlowFast(
        num_classes=400,
        alpha=4,
        beta=1/8,
        slow_channels=64,
        num_frames=32
    )
```

---

## 5. 학습

```python
def train_slowfast(model, train_loader, epochs=196):
    """
    논문에 나온 익히기 요령.
    """
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    # 반주기 코사인 일정
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    
    # 처음 34세대는 몸풀기
    warmup_epochs = 34
    
    for epoch in range(epochs):
        # 선형 워밍업
        if epoch < warmup_epochs:
            lr = 0.1 * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        model.train()
        for videos, labels in train_loader:
            # 영상은 틀 T=32인 (B, C, T, H, W)여야 한다
            videos = videos.cuda()
            labels = labels.cuda()
            
            optimizer.zero_grad()
            outputs = model(videos)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
        
        if epoch >= warmup_epochs:
            scheduler.step()
```

---

## 6. 결과

### Kinetics-400에서의 성능

| 모델 | 미리 익힘 | 상위 1 | 상위 5 |
|-------|----------|-------|-------|
| SlowFast 4×16 R50 | - | 75.6% | 92.1% |
| SlowFast 8×8 R101 | - | 77.9% | 93.2% |
| SlowFast 16×8 R101+NL | - | 79.8% | 93.9% |

### 다른 방법과의 견줌

| 방법 | 미리 익힘 | GFLOPs | 상위 1 |
|--------|----------|--------|-------|
| I3D | ImageNet | 108 | 71.1% |
| R(2+1)D | - | 152 | 72.0% |
| SlowFast 4×16 | - | 36.1 | 75.6% |
| SlowFast 8×8 | - | 65.7 | 77.0% |

핵심 발견: 느림빠름은 FLOPs를 덜 쓰고도 더 나은 정확도를 낸다.

---

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

## 정리하며

느림빠름의 핵심 새로움:

1. **두 갈래 길**이 자리의 뜻과 때에 걸친 움직임을 모두 담아낸다
2. **대칭이 아닌 꾸밈**(α, β 매개변수)이 셈을 효율적으로 나눈다
3. **옆 이음**이 길 사이로 앎이 흐르게 한다
4. 알맞은 셈 값으로 **센 성능**을 낸다

가장 알맞은 곳:

- 겉모습과 움직임이 모두 필요한 몸짓 알아보기
- 때에 걸친 움직임이 중요한 장면
- 정확도와 효율 사이의 균형
