# SlowFast 네트워크의 시간 정렬과 특징 결합

[SlowFast](https://arxiv.org/abs/1812.03982)는 같은 RGB 비디오 구간을 시간 해상도가 다른 두 경로로 처리한다. Slow는 프레임 수를 적게, Fast는 채널을 적게 사용해 외형과 시간 변화를 상호 보완적으로 표현하도록 설계한다. 이는 시각 처리에서 영감을 얻은 계산 구조이지 사람의 신경 회로를 정확히 재현하거나 두 기능이 완전히 분리됨을 입증하는 모델은 아니다.

## 1. 시간 비율과 채널 비율

같은 관찰 구간에서 Fast 프레임 수를 Slow의 α배, Fast 채널 수를 Slow의 β배로 둔다. α>1,0<β<1이 대표적인 구성이다. 예를 들어 Fast32프레임에서 α=8이면 Slow4프레임, Slow64채널에서 β=1/8이면 Fast8채널이다. 이는 특정 입력의 프레임 수·채널이지 항상 고정된4FPS·32FPS를 뜻하지 않는다. 실제 시간 간격은 원본 fps·stride로 결정한다.

입력은 B×3×T×H×W다. 아래 코드의 Slow 입력은 `::alpha`로 고르므로 길이가 ceil(T/α)가 될 수 있다. `num_frames`는 저장만 하며 입력 길이를 강제하거나 샘플링하지 않는다. α는 유효한 양의 정수이고 int(slow_channels×β)는 적어도1이어야 한다.

## 2. 구조 코드와 실행상의 연결 누락

```python
import torch
import torch.nn as nn

class SlowFast(nn.Module):
    """
    영상 알아보기를 위한 느림빠름 그물
    (Feichtenhofer et al., 2019)
    
    두 갈래 길 구조:
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

코드는 stem·stage·lateral과 분류 head의 골격을 보여 주지만 현재 상태로 완성된 SlowFast가 아니다. `forward`가 호출하는 `_forward_stem` 메서드가 정의되어 있지 않아 그대로는 실행할 수 없다. 두 stem의 호출 경로를 추가해도 아래의 채널·시간 연결 문제를 별도로 해결해야 한다.

Fast 특징을 lateral로 바꾸어 Slow와 채널 연결하면 입력 채널이 증가한다. 기본 첫 단계에서 Slow64채널과 lateral16채널을 연결하면80채널인데 첫 slow_block은64채널만 기대한다. 다음 단계도 lateral을 포함한 입력 채널을 stage 정의에 반영하지 않는다. 연결 연산을 썼다는 사실만으로 모델의 차원이 맞는 것은 아니다.

시간 정렬도 다르다. 두 번째 Fast stage는 temporal_stride=2인데 대응 Slow stage는1이다. 각 lateral이 계속 고정 α로 다운샘플링하므로 같은 단계의 시간 길이가 더 이상 맞지 않을 수 있다. stage마다 두 경로가 이미 사용한 시간 stride와 lateral의 stride·padding을 함께 추적해야 한다. LateralConnection의 slow_channels 인수는 저장·변환에 사용하지 않으며 출력 채널은 fast_channels×2로 정해진다.

## 3. 잔차 블록과 모델 이름

여기 ResBlock3D는 두3×3×3 합성곱과 필요시 projection shortcut을 사용하는 기본 블록이다. 공간 stride는1이고 시간 stride만 선택적으로 바뀐다. 표준 ResNet-50·101의 bottleneck 구조와 같은 블록이 아니며 단계 숫자만으로 원 SlowFast 모델의 재현이라고 할 수 없다.

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

두 factory는 같은 클래스·같은 stage 반복 수3·4·6·3을 사용하고 α만8과4로 바꾼다. `slowfast_8x8_r101`이라는 이름이 실제 R101 backbone을 구성하지 않는다. 또한 원 논문의 T×τ 표기에서 T는 Slow 프레임 수, τ는 원본의 샘플링 간격이다. 현재 코드에는 그 τ를 입력 비디오에서 선택하는 과정이 없어 factory 이름만으로 관찰 구간을 확정할 수 없다.

최종 head는 두 경로가 각각 slow_channels×8과 fast_channels×8의 특징을 반환한다고 가정해 평균 후 연결한다. 현재의 중간 연결 문제 때문에 이 출력까지 정상적으로 이어지는 구현으로 설명하면 안 된다.

## 4. 학습 루프의 전제

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

루프는 분류 로짓·정수 정답의 교차 엔트로피와 SGD를 사용하려는 골격이다. `F` import가 없고 loader·데이터 전처리는 별도로 필요하다. 비디오·레이블은 무조건 CUDA로 옮기지만 모델을 옮기는 줄은 없으므로 호출자가 모델 장치까지 맞춰야 한다. CPU·MPS 환경에 그대로 적용되지 않는다.

warm-up34epoch 이후 cosine scheduler를 호출하지만 T_max는 전체196epoch로 정한다. 기본 설정에서 실제 cosine step은162번뿐이라 학습 끝에 cosine 최솟값까지 도달하는 일정이 아니다. warm-up 이후 남은 기간과 스케줄러 step 기준을 맞춰 읽는다. 검증·최고 체크포인트·정확도 측정도 이 함수에는 없다.

## 5. 원 논문 결과와 축약 구현의 구분

논문의 Kinetics 결과는 backbone·관찰 구간·사전 학습·평가 clip/crop·non-local 여부 등 특정 조건의 실험이다. 이 미완성 코드의 실행 결과나 일반적인 정확도·FLOPs 순위로 제시하지 않는다. 특히 R50·R101 이름과 모델 구조가 맞지 않는 상태에서 논문 수치를 연결하면 안 된다.

Fast 비용은 일부 채널-채널 합성곱에서 대략 αβ²에 비례할 수 있지만 첫 RGB층·서로 다른 시간 커널·lateral·활성 메모리를 모두 설명하는 식은 아니다. 두 경로가 항상 더 정확하거나 Fast가 전체 비용의 고정20%라는 보장도 없다. 올바르게 연결한 모델에서 같은 평가 조건으로 비교해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
기본 첫 lateral 연결 후 Slow 입력 채널과 slow_block의 기대 채널을 비교하라.

</div>

??? success "연습문제 1 풀이"

    Slow stem64채널과 Fast8→lateral16채널을 연결하면80이다. 첫 Slow stage는64입력을 기대하므로 맞지 않는다. lateral을 포함한 입력 채널을 stage 설계에 반영해야 한다. 그 전에 미정의 _forward_stem 호출도 해결되어야 한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
Fast32프레임·α8에서 첫 시간 정렬과 다음 stage의 문제를 설명하라.

</div>

??? success "연습문제 2 풀이"

    입력 Slow길이는4다. Fast길이32를 lateral stride8로 줄이면4가 될 수 있지만 다음 Fast stage는시간을2배줄이고 Slow는유지한다. 같은 stride8lateral을 계속 쓰면2대4로 어긋날 수 있다. 누적 stride를 단계마다 추적해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
두 factory가 R50과 R101을 각각 구성한다고 말할 수 있는가?

</div>

??? success "연습문제 3 풀이"

    아니다. 같은 기본 블록과3·4·6·3반복을 사용하고 α만 다르다. bottleneck 구조·stage깊이·샘플링 간격의 구현을 확인해야 하며 이름과 논문 정확도를 직접 연결하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
기본 학습 루프의 warm-up과 cosine 기간을 계산하고 평가 범위를 설명하라.

</div>

??? success "연습문제 4 풀이"

    196epoch중34epochwarm-up 뒤162번cosine step을 하지만 T_max196이다. 따라서 cosine최솟값까지 도달하는196step일정이 아니다. 검증·체크포인트와 지표 계산도 없어 학습 완료만으로 논문 정확도를 주장할 수 없다.



## 정리하며

SlowFast의 두 시간 해상도는 stage별 정렬과 lateral 채널 계약까지 맞아야 구현된다. 현재 코드에는 stem 호출·연결 차원·모델 이름·학습 일정의 누락이 있다. 구조 아이디어와 원 논문 실험, 실제 제공된 코드의 범위를 구분한다.
