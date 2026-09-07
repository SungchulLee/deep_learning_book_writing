# 퍼짐 모델을 위한 U-Net 얼개

이 단원은 요즘 만들어 내는 모델의 핵심 부품인 퍼짐 모델을 위한 U-Net 얼개을 짠다. 여기서 보이는 개념과 재주를 알면 퍼짐 모델과 점수 바탕 만들어 내는 방법을 다루는 데 꼭 필요한 앎을 얻는다. 이 짜기는 또렷함과 실제 쓸모의 균형을 맞추어 배우기에도 실험하기에도 알맞다.

## 코드

```python
"""
퍼짐 모델을 위한 U-Net 얼개

이 단원은 퍼짐 모델에 흔히 쓰는 U-Net 얼개를 짠다.
U-Net은 잡음 섞인 그림과 때 걸음 박아 넣기를 들임으로 받아 잡음을 헤아린다.
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
    GroupNorm과 때 박아 넣기를 갖춘 남은 덩이.
    """
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, 
                 num_groups: int = 8):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        
        # 때 박아 넣기 쏘기
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
        
        # 때 박아 넣기를 더한다
        time_emb = self.time_mlp(time_emb)
        x = x + time_emb[:, :, None, None]  # 공간 차원으로 퍼뜨린다
        
        # 두 번째 누비기
        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        
        return x + residual


class AttentionBlock(nn.Module):
    """
    멀리 떨어진 매임을 담는 스스로 눈길 덩이.
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
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, 묶음, 머리, hw, 차원)
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
    퍼짐 모델을 위한 U-Net 얼개.
    
    그물은 다음을 지닌다.
    - 줄이기를 갖춘 부호기 길
    - 눈길을 갖춘 병목
    - 키우기와 건너뛰기 이음을 갖춘 풀개 길
    - 때 걸음을 조건으로 삼는 때 박아 넣기
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
            attention_resolutions: 눈길을 쓸 해상도
            dropout: 드롭아웃 확률
            time_emb_dim: 때 박아 넣기의 차원
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 때 박아 넣기
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # 첫 합성곱
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # 부호기
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
        
        # 복호기
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        for i, mult in enumerate(reversed(channel_multipliers)):
            out_ch = base_channels * mult
            
            for j in range(num_res_blocks + 1):
                # 부호기에서 오는 건너뛰기 이음
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
        # 때 박아 넣기
        time_emb = self.time_embedding(time)
        
        # 첫 합성곱
        x = self.conv_in(x)
        
        # 부호기
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
        
        # 복호기
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
        
        # 때 박아 넣기
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # 부호기
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.res1 = ResidualBlock(base_channels, base_channels * 2, time_emb_dim)
        self.down1 = Downsample(base_channels * 2)
        
        self.res2 = ResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.down2 = Downsample(base_channels * 4)
        
        # 병목
        self.res3 = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        
        # 복호기
        self.up1 = Upsample(base_channels * 4)
        self.res4 = ResidualBlock(base_channels * 8, base_channels * 2, time_emb_dim)
        
        self.up2 = Upsample(base_channels * 2)
        self.res5 = ResidualBlock(base_channels * 4, base_channels, time_emb_dim)
        
        # 내놓기
        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_embedding(time)
        
        # 부호기
        x1 = self.conv1(x)
        x2 = self.res1(x1, time_emb)
        x2_down = self.down1(x2)
        
        x3 = self.res2(x2_down, time_emb)
        x3_down = self.down2(x3)
        
        # 병목
        x4 = self.res3(x3_down, time_emb)
        
        # 건너뛰는 이음을 갖춘 풀개
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
    print(f"Number of parameters: {params:,}")```

## 논의

퍼짐 모델을 위한 U-Net 얼개의 짜기는 이 마당에 자리 잡은 방식을 따른다. 코드 짜임이 모델 뜻매김과 익히기 논리를 갈라 놓아 부품을 하나씩 고치기 쉽다. 얼개 고르기는 만들어 내는 모델 무리가 많은 실험에서 얻은 배움을 담고 있다.

이 짜기의 핵심에는 수치의 안정을 꼼꼼히 다루기, 고르게 맞추기 재주를 제대로 쓰기, 효율 좋은 셈 결이 든다. 익히기 절차에는 잡음 차례표, 기울기 다루기, 이따금의 따지기가 들며 모두 품질 높은 결과를 내는 데 결정적이다.

이 단원은 이론의 개념이 실제 짜기로 어떻게 옮겨지는지 보이며 만들어 내는 모델의 더 넓은 틀과 이어진다. 여기서 보이는 재주는 만들어 내는 모델이 이룰 수 있는 것의 가장자리를 넓히는 더 앞선 변형과 넓힘을 이해하는 바탕이 된다.

## 연습문제

**연습문제 1.**
구체적인 자료 묶음으로 이 단원의 으뜸 셈을 좇아라. 큰 걸음마다 텐서 꼴을 적고 모든 차원이 서로 맞는지 확인하라.

??? success "연습문제 1 풀이"
    모델에 알맞은 꼴의 들임 묶음에서 시작한다. 층이나 함수 부르기마다 셈을 따라가며 바뀜 뒤 텐서 꼴을 적는다. 겹말기 층에서는 내놓기 차원 공식을 쓴다. 눈길 얼개에서는 물음, 열쇠, 값의 차원이 맞는지 확인한다. 마지막 내놓기 꼴이 바라던 목표 차원과 맞는지 굳힌다. 이 익힘은 자료가 얼개를 어떻게 흐르는지에 대한 직관을 쌓아 준다.

---

**연습문제 2.**
이 단원에 쓰인 손실 함수를 가려내고 모델 매개변수에 대한 기울기를 이끌어 내라. 왜 이 손실 함수가 이 일에 알맞은지 설명하라.

??? success "연습문제 2 풀이"
    손실 함수는 모델이 헤아린 값과 목표 사이의 어긋남을 잰다. 잡음 헤아리기에서는 평균 제곱 어긋남 손실 $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$을 쓰는데, 이것이 로그 가능도의 변분 아래 한계에 맞물리기 때문이다. 매개변수 $\theta$에 대한 기울기는 $-2(\epsilon - \epsilon_\theta) \nabla_\theta \epsilon_\theta$이며 헤아림 어긋남을 줄이는 방향을 가리킨다. 이 손실을 가장 작게 하는 것이 퍼짐 모델에서 자료 로그 가능도의 아래 한계를 가장 크게 하는 것과 같으므로 알맞다.

---

**연습문제 3.**
다른 잡음 차례표를 받쳐 주도록 이 짜기를 고쳐라(예컨대 선형에서 코사인으로, 또는 그 반대로). 두 차례표의 익히기 움직임과 표본 품질을 견주어라.

??? success "연습문제 3 풀이"
    두 차례표를 모두 짜고 각각으로 모델을 익힌다. $\bar{\alpha}_t = \cos^2\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$으로 뜻매김한 코사인 차례표는 선형 차례표 $\beta_t = \beta_{\min} + t(\beta_{\max} - \beta_{\min})/T$에 견주어 잡음이 더 매끄럽게 늘어난다. 손실 곡선을 좇고 일정한 사이마다 표본을 만든다. 코사인 차례표는 신호 대 잡음비가 더 완만하게 줄어 때 걸음에 걸쳐 배움 신호가 더 고르므로 흔히 더 좋은 결과를 낸다.
