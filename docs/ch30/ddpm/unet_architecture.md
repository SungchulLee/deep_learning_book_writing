# 확산 모델을 위한 U-Net 구조

확산의 노이즈 예측기는 관측 이미지와 시간 조건을 받아 같은 공간 크기의 텐서를 출력한다. U-Net은 축소 경로에서 넓은 범위의 특징을 만들고 확대 경로에서 같은 해상도의 특징을 다시 결합하는 구조다. 이 페이지의 SimpleUNet과 일반 UNet은 서로 다른 구현이며, 후자의 순방향 순서에는 기본 설정에서도 차원이 맞지 않는 문제가 있다.

## 1. 코드

```python
"""
확산 모델을 위한 U-Net 구조

이 단원은 확산 모델에 흔히 쓰는 U-Net 구조를 짠다.
U-Net은 잡음 섞인 그림과 때 걸음 임베딩을 들임으로 받아 잡음을 헤아린다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_utils import SinusoidalPositionEmbedding

# ========================================================================
# 메인
# ========================================================================


class ResidualBlock(nn.Module):
    """
    GroupNorm과 때 임베딩을 갖춘 남은 덩이.
    """
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, 
                 num_groups: int = 8):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        
        # 때 임베딩 쏘기
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        # 잔차 연결
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        인수:
            x: 들임 텐서, 꼴 (batch, in_channels, height, width)
            time_emb: 때 묻힘, 꼴 (batch, time_emb_dim)
        
        반환값:
            내놓음 텐서, 꼴 (batch, out_channels, height, width)
        """
        residual = self.residual_conv(x)
        
        # 첫 누비기
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        
        # 때 임베딩을 더한다
        time_emb = self.time_mlp(time_emb)
        x = x + time_emb[:, :, None, None]  # 공간 차원으로 퍼뜨린다
        
        # 두 번째 누비기
        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        
        return x + residual


class AttentionBlock(nn.Module):
    """
    멀리 떨어진 매임을 담는 셀프 어텐션 덩이.
    """
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        인수:
            x: 들임 텐서, 꼴 (batch, channels, height, width)
        
        반환값:
            들임과 같은 꼴의 내놓기 텐서
        """
        batch, channels, height, width = x.shape
        residual = x
        
        x = self.norm(x)
        qkv = self.qkv(x)
        
        # 다중 머리 주의에 맞게 꼴을 바꾼다
        qkv = qkv.reshape(batch, 3, self.num_heads, channels // self.num_heads, height * width)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, 배치, 머리, hw, 차원)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 어텐션
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn / (channels // self.num_heads) ** 0.5
        attn = F.softmax(attn, dim=-1)
        
        # 값에 어텐션 적용
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(batch, channels, height, width)
        
        # 쏘고 남은 것을 더한다
        out = self.proj(out)
        return out + residual


class Downsample(nn.Module):
    """겹말기를 쓴 줄이기 층."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """자리 바꾼 겹말기를 쓴 키우기 층."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    확산 모델을 위한 U-Net 구조.
    
    그물은 다음을 지닌다.
    - 줄이기를 갖춘 인코더 길
    - 어텐션을 갖춘 병목
    - 키우기와 건너뛰기 이음을 갖춘 디코더 길
    - 때 걸음을 조건으로 삼는 때 임베딩
    """
    
    def __init__(self, 
                 in_channels: int = 1,
                 out_channels: int = 1,
                 base_channels: int = 64,
                 channel_multipliers: tuple = (1, 2, 4, 8),
                 num_res_blocks: int = 2,
                 attention_resolutions: tuple = (16,),
                 dropout: float = 0.0,
                 time_emb_dim: int = 256):
        """
        인수:
            in_channels: 들임 채널의 개수
            out_channels: 내놓는 채널의 개수
            base_channels: 바탕 채널 수
            channel_multipliers: 해상도 층마다 채널 갑절
            num_res_blocks: 해상도마다 남은 덩이의 수
            attention_resolutions: 어텐션을 쓸 해상도
            dropout: 드롭아웃 확률
            time_emb_dim: 때 임베딩의 차원
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 때 임베딩
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # 첫 합성곱
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # 인코더
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        channels = [base_channels]
        now_channels = base_channels
        
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            
            for _ in range(num_res_blocks):
                block = ResidualBlock(now_channels, out_ch, time_emb_dim)
                self.encoder_blocks.append(block)
                now_channels = out_ch
                channels.append(now_channels)
            
            # 마지막 층만 빼고 줄이기를 더한다
            if i != len(channel_multipliers) - 1:
                self.downsamples.append(Downsample(now_channels))
                channels.append(now_channels)
        
        # 병목
        self.bottleneck = nn.ModuleList([
            ResidualBlock(now_channels, now_channels, time_emb_dim),
            AttentionBlock(now_channels),
            ResidualBlock(now_channels, now_channels, time_emb_dim),
        ])
        
        # 디코더
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        for i, mult in enumerate(reversed(channel_multipliers)):
            out_ch = base_channels * mult
            
            for j in range(num_res_blocks + 1):
                # 인코더에서 오는 건너뛰기 이음
                skip_ch = channels.pop()
                block = ResidualBlock(now_channels + skip_ch, out_ch, time_emb_dim)
                self.decoder_blocks.append(block)
                now_channels = out_ch
            
            # 마지막 층만 빼고 키우기를 더한다
            if i != len(channel_multipliers) - 1:
                self.upsamples.append(Upsample(now_channels))
        
        # 내놓기
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, now_channels),
            nn.SiLU(),
            nn.Conv2d(now_channels, out_channels, kernel_size=3, padding=1),
        )
    
    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        U-Net을 지나는 앞먹임.
        
        인수:
            x: 잡음 섞인 들임, 꼴 (batch, in_channels, height, width)
            time: 때 걸음, 꼴 (batch,)
        
        반환값:
            예측한 잡음, 꼴 (batch, out_channels, height, width)
        """
        # 때 임베딩
        time_emb = self.time_embedding(time)
        
        # 첫 합성곱
        x = self.conv_in(x)
        
        # 인코더
        encoder_outputs = [x]
        
        down_idx = 0
        for block in self.encoder_blocks:
            x = block(x, time_emb)
            encoder_outputs.append(x)
        
        for downsample in self.downsamples:
            x = downsample(x)
            encoder_outputs.append(x)
        
        # 병목
        for block in self.bottleneck:
            if isinstance(block, AttentionBlock):
                x = block(x)
            else:
                x = block(x, time_emb)
        
        # 디코더
        up_idx = 0
        for block in self.decoder_blocks:
            skip = encoder_outputs.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x, time_emb)
        
        for upsample in self.upsamples:
            x = upsample(x)
        
        # 내놓기
        x = self.conv_out(x)
        
        return x


class SimpleUNet(nn.Module):
    """
    MNIST 같은 작은 자료 묶음에서 더 빨리 익히기 위한 단순한 U-Net.
    배우기에 좋은 출발점이다.
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1, 
                 base_channels: int = 32, time_emb_dim: int = 128):
        super().__init__()
        
        # 때 임베딩
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # 인코더
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.res1 = ResidualBlock(base_channels, base_channels * 2, time_emb_dim)
        self.down1 = Downsample(base_channels * 2)
        
        self.res2 = ResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.down2 = Downsample(base_channels * 4)
        
        # 병목
        self.res3 = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        
        # 디코더
        self.up1 = Upsample(base_channels * 4)
        self.res4 = ResidualBlock(base_channels * 8, base_channels * 2, time_emb_dim)
        
        self.up2 = Upsample(base_channels * 2)
        self.res5 = ResidualBlock(base_channels * 4, base_channels, time_emb_dim)
        
        # 내놓기
        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_embedding(time)
        
        # 인코더
        x1 = self.conv1(x)
        x2 = self.res1(x1, time_emb)
        x2_down = self.down1(x2)
        
        x3 = self.res2(x2_down, time_emb)
        x3_down = self.down2(x3)
        
        # 병목
        x4 = self.res3(x3_down, time_emb)
        
        # 건너뛰는 이음을 갖춘 디코더
        x5 = self.up1(x4)
        x5 = torch.cat([x5, x3], dim=1)
        x5 = self.res4(x5, time_emb)
        
        x6 = self.up2(x5)
        x6 = torch.cat([x6, x2], dim=1)
        x6 = self.res5(x6, time_emb)
        
        return self.conv_out(x6)


if __name__ == "__main__":
    # 모델을 시험한다
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # SimpleUNet을 시험한다
    model = SimpleUNet().to(device)
    x = torch.randn(4, 1, 28, 28).to(device)
    t = torch.randint(0, 1000, (4,)).to(device)
    out = model(x, t)
    print(f"SimpleUNet output shape: {out.shape}")
    
    # 온전한 UNet을 시험한다
    model = UNet(in_channels=1, base_channels=32).to(device)
    out = model(x, t)
    print(f"UNet output shape: {out.shape}")
    
    # 매개변수 개수 세기
    params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {params:,}")
```

## 2. 시간 조건 잔차 블록

ResidualBlock은 GroupNorm→SiLU→3×3 합성곱 뒤에 시간 임베딩의 선형 투영을 더한다. 다시 정규화·활성화·합성곱을 적용하고 입력의 Identity 또는 1×1 투영을 더한다. 공간 크기는 유지하며 시간 조건은 출력 채널 수만큼 만든 뒤 높이·너비에 브로드캐스팅한다.

이 시간 주입은 다른 페이지의 scale·shift 방식과 달리 **덧셈 방식**이다. GroupNorm의 입력·출력 채널은 그룹 수 8로 나누어져야 한다. 기본 채널 폭은 조건을 만족하지만 임의의 폭을 지원하는 것은 아니다.

SinusoidalPositionEmbedding은 diffusion_utils에서 가져온다. 실제 반환 폭이 2×floor(dim/2)이고 작은 폭에는 분모 문제가 있으므로 기본 base_channels=32처럼 4 이상인 짝수와 계층별 나눗셈 조건을 함께 맞춘다.

## 3. 공간 어텐션

AttentionBlock은 B×C×H×W를 헤드별 위치 시퀀스로 바꾸어 q·k·v를 계산한다. 위치 수 N=HW, 헤드 폭 Ch=C/heads이면 q·k·v는 B×heads×N×Ch다. 가중치 행렬은 B×heads×N×N이고 Ch의 제곱근으로 스케일한 뒤 softmax를 적용한다.

출력은 원래 이미지 모양으로 복원하고 1×1 투영 뒤 입력에 더한다. C는 헤드 수와 8로 나누어져야 한다. 어텐션은 위치 사이 상호작용을 제공하지만 메모리는 N²에 민감하므로 큰 해상도·배치에서 비용을 확인해야 한다.

## 4. SimpleUNet의 유효한 28×28 경로

main이 먼저 만드는 SimpleUNet은 공간 연산을 잔차 블록 사이에 배치한다. base_channels=32에서 다음과 같다.

| 특징 | 모양 |
|---|---|
| conv1 | B×32×28×28 |
| res1의 x2 | B×64×28×28 |
| down1 후 res2의 x3 | B×128×14×14 |
| down2 후 res3의 x4 | B×128×7×7 |
| up1 후 x3 연결 | B×256×14×14 |
| res4 후 up2 | B×64×28×28 |
| x2 연결 후 res5 | B×32×28×28 |
| 최종 출력 | B×1×28×28 |

up1·up2는 전치 합성곱으로 공간을 두 배 키운다. SimpleUNet에는 AttentionBlock이 없으며 생성기 손실이나 샘플링 함수도 없다. 다른 이미지 크기에서는 두 번의 올림 축소와 두 배 확대가 skip 크기와 맞는지 확인해야 한다.

## 5. 일반 UNet의 실행 순서 불일치

일반 UNet의 생성자는 수준별 잔차 블록과 downsample을 따로 저장한다. 하지만 forward는 **모든 인코더 잔차 블록을 먼저 실행한 뒤 모든 downsample을 실행**한다. 다중 해상도 경로로 교차 실행하지 않는다.

main의 base_channels=32와 배수 (1,2,4,8)에서는 잔차 블록들을 모두 지난 h가 B×256×28×28이다. 첫 downsample은 32채널용으로 만들어졌으므로 **256채널 입력을 받을 수 없어 실패한다**. 따라서 main의 두 번째 UNet 출력 모양과 파라미터 출력까지 정상 도달한다고 설명할 수 없다.

디코더도 모든 잔차 블록을 먼저 실행한 뒤 모든 upsample을 수행한다. 생성자의 skip 채널 목록은 축소가 중간에 들어가는 순서를 전제로 하지만 실제 저장·소비 순서는 다르다. 인코더 한 줄만 조정해 전체 문제가 해결된다고 볼 수 없으며 수준별 공간 연산과 skip의 대응을 함께 설계해야 한다.

입력 인수 attention_resolutions와 dropout은 현재 UNet 내부에서 사용되지 않는다. 어텐션은 bottleneck 하나에 고정되어 있고 Dropout 계층은 없다. 인수를 바꿔 해상도별 어텐션이나 Dropout 효과를 비교할 수 있는 코드라고 설명하지 않는다. down_idx와 up_idx 변수도 정의만 되어 실행 순서를 제어하지 않는다.

## 6. 구조 확인과 학습 검증의 차이

데모는 임의의 4×1×28×28 입력과 시간으로 순방향을 호출할 뿐 데이터셋·노이즈 손실·optimizer·생성 평가를 갖추지 않는다. SimpleUNet의 출력 모양이 맞아도 올바른 노이즈 예측을 학습했다는 뜻은 아니다. 일반 UNet은 위에서 설명한 채널 문제를 먼저 갖고 있다.

이 코드 블록은 보존하며 실행·모델 학습이나 그림 재생성은 수행하지 않는다. 학습 파이프라인으로 연결할 때는 사용하려는 클래스, 입력 크기, 예측 대상과 시간·장치·정규화 조건을 명시해야 한다. 구조의 이름이나 파라미터 수로 생성 품질을 평가할 수는 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
SimpleUNet에서 첫 번째 skip 연결과 두 번째 skip 연결의 모양을 설명하라.

</div>

??? success "연습문제 1 풀이"

    첫 연결은 up1의 128채널 14×14와 x3의 128채널 14×14를 붙여 256채널을 만든다. 두 번째는 up2의 64채널 28×28과 x2의 64채널 28×28을 붙여 128채널을 만든다. 이후 res5가 이를 32채널로 바꾼다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
일반 UNet의 첫 downsample이 main 설정에서 실패하는 이유는 무엇인가?

</div>

??? success "연습문제 2 풀이"

    생성자는 첫 수준의 32채널 downsample을 만들지만 forward는 모든 잔차 블록부터 실행해 256채널 특징을 전달한다. 공간 축소를 수준 사이에 배치하도록 설계된 채널 목록과 실행 순서가 다르다. 디코더의 모든 확대를 뒤로 미룬 순서도 함께 검토해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
attention_resolutions와 dropout을 바꾸면 현재 UNet의 구조가 어떻게 바뀌는가?

</div>

??? success "연습문제 3 풀이"

    둘 다 사용되지 않아 그 인수만 바꾸어서는 구조가 달라지지 않는다. 어텐션은 bottleneck에 고정되고 Dropout 계층은 없다. 해당 옵션의 효과나 노이즈 스케줄·학습 품질을 비교하는 실험도 이 페이지에는 구현되어 있지 않다.



## 정리하며

U-Net에서는 해상도 변화, 채널 수와 skip 저장·소비의 순서가 함께 맞아야 한다. SimpleUNet은 28×28의 두 단계 경로를 제공하지만 일반 UNet은 모든 축소·확대를 뒤로 미뤄 차원이 맞지 않는다. 사용되지 않는 옵션과 실제 구현을 구분해 읽는다.
