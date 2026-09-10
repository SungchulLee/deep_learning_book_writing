# 깊은 겹말기 맞겨루기 만들개

이 짜기는 Radford 외(2016)의 얼개 지침을 따른 실제 쓸 만한 품질의 DCGAN을 준다. 28x28(MNIST에 맞는) 판과 64x64(본디 논문) 판을 모두 담아 자리 바꾼 겹말기, 묶음 고르게 맞추기, 안정된 맞겨루기 만들개 익히기를 위해 권하는 깨움 함수를 제대로 쓰는 법을 보인다.

## 1. 코드

```python
"""
깊은 엮음 GAN(DCGAN)

다음 지침을 따른 DCGAN 짜기:
"Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"
(Radford et al., 2016)

핵심 얼개 원칙:
1. 모으기를 성큼 겹말기로 바꾼다
2. G과 D 모두에 묶음 고르게 맞추기를 쓴다
3. 온전 이음 숨은 층을 없앤다
4. G에서는 ReLU를 쓴다(내놓음만 Tanh)
5. D에 LeakyReLU을 쓴다
"""

import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================


class DCGANGenerator(nn.Module):
    """
    DCGAN 만들개 신경망.
    
    자리 바꾼 겹말기로 숨은 벡터 z을 그림으로 옮긴다.
    """
    
    def __init__(self, latent_dim: int = 100, image_channels: int = 1, 
                 feature_maps: int = 64):
        """
        인수:
            latent_dim: 숨은 벡터 z의 차원
            image_channels: 내놓음 통로 수(잿빛이면 1, RGB이면 3)
            feature_maps: 첫 층의 특징 지도 수(층마다 2배로 는다)
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # 28x28 그림(MNIST)에서는 7x7에서 시작해 28x28으로 키워야 한다
        # 64x64 그림에서는 4x4에서 시작해 64x64으로 키운다
        
        # 첫 쏘기와 꼴 다시 잡기
        self.project = nn.Sequential(
            nn.Linear(latent_dim, feature_maps * 8 * 7 * 7),
            nn.BatchNorm1d(feature_maps * 8 * 7 * 7),
            nn.ReLU(True)
        )
        
        # 겹말기 층(키우기)
        self.main = nn.Sequential(
            # 들임: (feature_maps*8) x 7 x 7
            
            # 층 1: 14x14으로 키운다
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 
                             kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # 내놓기: (feature_maps*4) x 14 x 14
            
            # 층 2: 28x28으로 키운다
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2,
                             kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # 내놓기: (feature_maps*2) x 28 x 28
            
            # 층 3: 그림 채널로 가는 마지막 겹말기
            nn.Conv2d(feature_maps * 2, image_channels,
                     kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
            # 내놓기: image_channels x 28 x 28
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        숨은 벡터에서 그림을 만든다.
        
        인수:
            z: 숨은 벡터, 꼴 (batch_size, latent_dim) 또는 (batch_size, latent_dim, 1, 1)
        
        반환값:
            만들어 낸 그림, 꼴 (batch_size, channels, height, width)
        """
        # 필요하면 펼치기
        if z.dim() == 4:
            z = z.view(z.size(0), -1)
        
        # 사영하고 모양 바꾸기
        x = self.project(z)
        x = x.view(x.size(0), -1, 7, 7)
        
        # 그림을 만든다
        return self.main(x)


class DCGANDiscriminator(nn.Module):
    """
    DCGAN 가름개 신경망.
    
    성큼 겹말기로 그림을 실제인지 가짜인지 가른다.
    """
    
    def __init__(self, image_channels: int = 1, feature_maps: int = 64):
        """
        인수:
            image_channels: 들임 통로 수(잿빛이면 1, RGB이면 3)
            feature_maps: 첫 층의 특징 지도 수(층마다 2배로 는다)
        """
        super().__init__()
        
        self.main = nn.Sequential(
            # 들임: image_channels x 28 x 28
            
            # 층 1: 들임 층에는 묶음 고르게 맞추기를 쓰지 않는다(DCGAN 지침)
            nn.Conv2d(image_channels, feature_maps, 
                     kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 내놓기: feature_maps x 14 x 14
            
            # 2층
            nn.Conv2d(feature_maps, feature_maps * 2,
                     kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 내놓기: (feature_maps*2) x 7 x 7
            
            # 3층
            nn.Conv2d(feature_maps * 2, feature_maps * 4,
                     kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 내놓기: (feature_maps*4) x 4 x 4
            
            # 내놓기 층: 값 하나로 겹말기 한다
            nn.Conv2d(feature_maps * 4, 1,
                     kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # 내놓기: 1 x 1 x 1
        )
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        그림을 실제인지 가짜인지 가른다.
        
        인수:
            img: 들임 그림, 꼴 (batch_size, channels, height, width)
        
        반환값:
            참일 확률, 꼴 (batch_size, 1)
        """
        output = self.main(img)
        return output.view(-1, 1)


class DCGAN64Generator(nn.Module):
    """
    64x64 그림을 위한 DCGAN 만들개(본디 논문에 더 가깝게 따른다).
    """
    
    def __init__(self, latent_dim: int = 100, image_channels: int = 3,
                 feature_maps: int = 64):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        self.main = nn.Sequential(
            # 들임: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, feature_maps * 8,
                             kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            # 상태: (feature_maps*8) x 4 x 4
            
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4,
                             kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # 상태: (feature_maps*4) x 8 x 8
            
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2,
                             kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # 상태: (feature_maps*2) x 16 x 16
            
            nn.ConvTranspose2d(feature_maps * 2, feature_maps,
                             kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # 상태: feature_maps x 32 x 32
            
            nn.ConvTranspose2d(feature_maps, image_channels,
                             kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # 내놓기: image_channels x 64 x 64
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """숨은 벡터에서 64x64 그림을 만든다."""
        if z.dim() == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)
        return self.main(z)


class DCGAN64Discriminator(nn.Module):
    """
    64x64 그림을 위한 DCGAN 가름개(본디 논문을 따른다).
    """
    
    def __init__(self, image_channels: int = 3, feature_maps: int = 64):
        super().__init__()
        
        self.main = nn.Sequential(
            # 들임: image_channels x 64 x 64
            nn.Conv2d(image_channels, feature_maps,
                     kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 상태: feature_maps x 32 x 32
            
            nn.Conv2d(feature_maps, feature_maps * 2,
                     kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 상태: (feature_maps*2) x 16 x 16
            
            nn.Conv2d(feature_maps * 2, feature_maps * 4,
                     kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 상태: (feature_maps*4) x 8 x 8
            
            nn.Conv2d(feature_maps * 4, feature_maps * 8,
                     kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 상태: (feature_maps*8) x 4 x 4
            
            nn.Conv2d(feature_maps * 8, 1,
                     kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # 내놓기: 1 x 1 x 1
        )
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """64x64 그림을 가른다."""
        output = self.main(img)
        return output.view(-1, 1)


def test_dcgan():
    """DCGAN 얼개를 시험한다."""
    print("Testing DCGAN for 28x28 images (MNIST)...")
    
    # 28x28 판을 시험한다
    gen = DCGANGenerator(latent_dim=100, image_channels=1, feature_maps=64)
    disc = DCGANDiscriminator(image_channels=1, feature_maps=64)
    
    # 순전파 시험
    z = torch.randn(16, 100)
    fake_imgs = gen(z)
    print(f"Generator output shape: {fake_imgs.shape}")
    
    d_output = disc(fake_imgs)
    print(f"Discriminator output shape: {d_output.shape}")
    
    # 매개변수 개수 세기
    g_params = sum(p.numel() for p in gen.parameters())
    d_params = sum(p.numel() for p in disc.parameters())
    print(f"Generator parameters: {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")
    
    print("\nTesting DCGAN for 64x64 images...")
    
    # 64x64 판을 시험한다
    gen64 = DCGAN64Generator(latent_dim=100, image_channels=3, feature_maps=64)
    disc64 = DCGAN64Discriminator(image_channels=3, feature_maps=64)
    
    z = torch.randn(16, 100, 1, 1)
    fake_imgs = gen64(z)
    print(f"Generator output shape: {fake_imgs.shape}")
    
    d_output = disc64(fake_imgs)
    print(f"Discriminator output shape: {d_output.shape}")
    
    g_params = sum(p.numel() for p in gen64.parameters())
    d_params = sum(p.numel() for p in disc64.parameters())
    print(f"Generator parameters: {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")
    
    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    test_dcgan()
```

**출력:**

```
Testing DCGAN for 28x28 images (MNIST)...
Generator output shape: torch.Size([16, 1, 28, 28])
Discriminator output shape: torch.Size([16, 1])
Generator parameters: 5,207,424
Discriminator parameters: 431,872

Testing DCGAN for 64x64 images...
Generator output shape: torch.Size([16, 3, 64, 64])
Discriminator output shape: torch.Size([16, 1])
Generator parameters: 3,576,704
Discriminator parameters: 2,765,568

All tests passed! ✓
```

## 2. 논의

DCGANGenerator은 두 단계 얼개를 쓴다. 곧 먼저 숨은 벡터를 묶음 고르게 맞추기를 갖춘 선형 층으로 공간 특징 지도에 쏜 뒤 자리 바꾼 겹말기로 키운다. 키우는 덩이마다 ConvTranspose2d, BatchNorm2d, ReLU의 결을 따르며 마지막 층은 Tanh으로 $[-1, 1]$ 안의 그림을 만든다. 갈래 적기와 자세한 설명글이 있어 참고 짜기로 알맞다.

DCGANDiscriminator은 성큼 겹말기로 들임 그림을 차츰 줄이면서 특징 채널 수를 늘린다. 논문 지침에 따라 첫 겹말기 층은 묶음 고르게 맞추기를 빼고 기울기 0.2인 LeakyReLU을 내내 쓴다. 마지막 겹말기는 표본마다 값 하나로 줄이고 두값 가르기를 위해 시그모이드를 지난다.

64x64 판(DCGAN64Generator과 DCGAN64Discriminator)은 본디 논문을 더 가깝게 따라 $(z, 1, 1)$ 들임에서 시작해 키우기/줄이기 네 단계의 대칭 얼개를 쓴다. 이 판은 숨은 벡터를 4차원 꼴 $(B, z, 1, 1)$으로 받으며 이는 많은 맞겨루기 만들개 틀에서 쓰는 약속이다.

## 연습문제

**연습문제 1.**
DCGANGenerator은 겹말기 층에 `bias=False`을 쓴다. 겹말기 뒤에 묶음 고르게 맞추기가 올 때 이것이 알맞은 까닭을 설명하라. 치우침을 켜면 무엇이 달라지는가?

??? success "연습문제 1 풀이"
    겹말기 뒤에 묶음 고르게 맞추기가 오면 치우침 항은 겹친다. 묶음 고르게 맞추기가 평균을 빼고(치우침 효과를 빨아들인다) 제 배울 수 있는 옮김 매개변수를 쓰기 때문이다. 치우침을 넣으면 표현력은 나아지지 않고 매개변수만 는다. `bias=False`이면 Conv2d나 ConvTranspose2d 층마다 매개변수가 줄고 묶음 고르게 맞추기의 $\beta$ 매개변수가 사실상 치우침 노릇을 한다. 치우침을 켜도 옳음이 깨지지는 않으나 기억을 낭비하고 익히기가 조금 느려진다.

---

**연습문제 2.**
익힌 DCGAN의 숨은 공간에서 선형 사이 메우기와 공 모양 사이 메우기를 견주어라. 공 모양 사이 메우기가 만든 그림 사이에 더 자연스러운 옮아감을 내는 까닭은 무엇인가?

??? success "연습문제 2 풀이"
    맞겨루기 만들개의 숨은 공간은 흔히 정규 분포를 따르며 차원이 높으면 표본이 초구면에 모인다. 선형 사이 메우기 $z(t) = (1-t)z_1 + tz_2$은 원점 가까운 밀도 낮은 자리를 지나 그럴듯하지 않은 중간 그림을 낼 수 있다. 공 모양 사이 메우기(slerp)는 원점에서 거리를 한결같이 지켜 밀도 높은 다양체 위에 머문다. 100차원 정규 분포에서 기댓값 잣대는 $\sqrt{100} = 10$이지만 선형 사이 메우기의 가운데 점은 잣대가 대략 $10/\sqrt{2} \approx 7.07$이며 이는 확률 낮은 자리에 있다. slerp은 두 점을 잇는 큰 원을 따라 메워 이를 피한다.

---

**연습문제 3.**
DCGAN 무게 첫자리매김 방식(겹말기 층은 평균 0, 표준 편차 0.02인 정규 분포, 묶음 고르게 맞추기는 평균 1, 표준 편차 0.02)을 쓰는 함수를 짜라. 만들개와 가름개 모두에서 시험하고 무게 통계를 확인하라.

??? success "연습문제 3 풀이"
    ```python
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    gen = DCGANGenerator()
    gen.apply(weights_init)
    for name, param in gen.named_parameters():
        if 'weight' in name:
            print(f"{name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")
    ```
    겹말기 무게는 평균이 0에 가깝고 표준 편차가 0.02에 가까워야 한다. 묶음 고르게 맞추기 무게는 평균이 1.0에 가깝고 표준 편차가 0.02에 가까워야 한다. 이 첫자리매김은 처음부터 깨움 크기를 알맞게 하여 익히기 앞머리의 불안정을 막는다.

## 정리하며

**다룬 것** — 깊은 겹말기 맞겨루기 만들개

DCGANGenerator은 두 단계 얼개를 쓴다.

고갱이 갈래는 `DCGANGenerator`, `DCGANDiscriminator`, `DCGAN64Generator`, `DCGAN64Discriminator`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
