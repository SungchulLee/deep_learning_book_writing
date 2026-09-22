# 깊은 겹말기 맞겨루기 만들개

이 짜기는 Radford 외(2016)의 얼개 지침을 따른 실제 쓸 만한 품질의 DCGAN을 준다. 28x28(MNIST에 맞는) 판과 64x64(본디 논문) 판을 모두 담아 자리 바꾼 겹말기, 배치 고르게 맞추기, 안정된 맞겨루기 만들개 익히기를 위해 권하는 깨움 함수를 제대로 쓰는 법을 보인다.

## 1. 코드

```python
"""
깊은 엮음 GAN(DCGAN)

다음 지침을 따른 DCGAN 짜기:
"Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"
(Radford et al., 2016)

핵심 얼개 원칙:
1. 모으기를 성큼 겹말기로 바꾼다
2. G과 D 모두에 배치 고르게 맞추기를 쓴다
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
            
            # 층 1: 들임 층에는 배치 고르게 맞추기를 쓰지 않는다(DCGAN 지침)
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

DCGANGenerator은 두 단계 얼개를 쓴다. 곧 먼저 숨은 벡터를 배치 고르게 맞추기를 갖춘 선형 층으로 공간 특징 지도에 쏜 뒤 자리 바꾼 겹말기로 키운다. 키우는 덩이마다 ConvTranspose2d, BatchNorm2d, ReLU의 결을 따르며 마지막 층은 Tanh으로 $[-1, 1]$ 안의 그림을 만든다. 갈래 적기와 자세한 설명글이 있어 참고 짜기로 알맞다.

DCGANDiscriminator은 성큼 겹말기로 들임 그림을 차츰 줄이면서 특징 채널 수를 늘린다. 논문 지침에 따라 첫 겹말기 층은 배치 고르게 맞추기를 빼고 기울기 0.2인 LeakyReLU을 내내 쓴다. 마지막 겹말기는 표본마다 값 하나로 줄이고 두값 가르기를 위해 시그모이드를 지난다.

64x64 판(DCGAN64Generator과 DCGAN64Discriminator)은 본디 논문을 더 가깝게 따라 $(z, 1, 1)$ 들임에서 시작해 키우기/줄이기 네 단계의 대칭 얼개를 쓴다. 이 판은 숨은 벡터를 4차원 꼴 $(B, z, 1, 1)$으로 받으며 이는 많은 맞겨루기 만들개 틀에서 쓰는 약속이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
DCGANGenerator은 겹말기 층에 `bias=False`을 쓴다. 겹말기 뒤에 배치 고르게 맞추기가 올 때 이것이 알맞은 까닭을 설명하라. 치우침을 켜면 무엇이 달라지는가?

</div>

??? success "연습문제 1 풀이"
    겹말기 뒤에 배치 고르게 맞추기가 오면 치우침 항은 겹친다. 배치 고르게 맞추기가 평균을 빼고(치우침 효과를 빨아들인다) 제 배울 수 있는 옮김 매개변수를 쓰기 때문이다. 치우침을 넣으면 표현력은 나아지지 않고 매개변수만 는다. `bias=False`이면 Conv2d나 ConvTranspose2d 층마다 매개변수가 줄고 배치 고르게 맞추기의 $\beta$ 매개변수가 사실상 치우침 노릇을 한다. 치우침을 켜도 옳음이 깨지지는 않으나 기억을 낭비하고 익히기가 조금 느려진다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
익힌 DCGAN의 숨은 공간에서 선형 사이 메우기와 공 모양 사이 메우기를 견주어라. 공 모양 사이 메우기가 만든 그림 사이에 더 자연스러운 옮아감을 내는 까닭은 무엇인가?

</div>

??? success "연습문제 2 풀이"
    맞겨루기 만들개의 숨은 공간은 흔히 정규 분포를 따르며 차원이 높으면 표본이 초구면에 모인다. 선형 사이 메우기 $z(t) = (1-t)z_1 + tz_2$은 원점 가까운 밀도 낮은 자리를 지나 그럴듯하지 않은 중간 그림을 낼 수 있다. 공 모양 사이 메우기(slerp)는 원점에서 거리를 한결같이 지켜 밀도 높은 다양체 위에 머문다. 100차원 정규 분포에서 기댓값 잣대는 $\sqrt{100} = 10$이지만 선형 사이 메우기의 가운데 점은 잣대가 대략 $10/\sqrt{2} \approx 7.07$이며 이는 확률 낮은 자리에 있다. slerp은 두 점을 잇는 큰 원을 따라 메워 이를 피한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
DCGAN 무게 첫자리매김 방식(겹말기 층은 평균 0, 표준 편차 0.02인 정규 분포, 배치 고르게 맞추기는 평균 1, 표준 편차 0.02)을 쓰는 함수를 짜라. 만들개와 가름개 모두에서 시험하고 무게 통계를 확인하라.

</div>

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
    겹말기 무게는 평균이 0에 가깝고 표준 편차가 0.02에 가까워야 한다. 배치 고르게 맞추기 무게는 평균이 1.0에 가깝고 표준 편차가 0.02에 가까워야 한다. 이 첫자리매김은 처음부터 깨움 크기를 알맞게 하여 익히기 앞머리의 불안정을 막는다.


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
이 코드의 생성기가 $z$를 어떻게 특징 지도로 펴는가?

</div>

??? success "연습문제 4 풀이"
    완전 연결로 $7\times7\times C$ 크기의 벡터를 만들고 되편다.

    ```python
    h = self.fc(z)                       # (B, C*7*7)
    h = h.view(-1, C, 7, 7)              # (B, C, 7, 7)
    ```

    이 층이 대개 생성기 매개변수의 큰 몫을 차지한다. $z$가 64차원이고 $C=64$면
    $64 \times 3136 = 200{,}704$개다.

    다른 방법도 있다. $z$를 $1\times1$ 특징 지도로 보고 `ConvTranspose2d`로 키우는 것이다.

    ```python
    z = z.view(-1, latent_dim, 1, 1)
    nn.ConvTranspose2d(latent_dim, 256, 7, 1, 0)     # 1x1 -> 7x7
    ```

    원 DCGAN이 이 방식을 쓴다. 매개변수 수는 비슷하고 완전 연결층이 없다는 규칙에 맞다.

    어느 쪽이든 **첫 특징 지도 크기를 $7\times7$로 두는 것**이 MNIST의 요령이다. 두 번
    키우면 28이 된다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
이 코드를 돌렸는데 표본이 모두 비슷하다. 무엇을 먼저 확인하겠는가?

</div>

??? success "연습문제 5 풀이"
    순서대로 이렇게 본다.

    1. **부류 엔트로피와 인셉션 점수를 잰다.** 무너짐인지 수로 확인한다. 인셉션 점수가
       1에 가까우면 완전히 무너진 것이다
    2. **고정된 $z$로 격자를 그려 에포크별로 본다.** 언제부터 닮아 갔는지 보인다
    3. **배치 정규화와 드롭아웃이 있는지 본다.** 이 장의 측정에서 둘을 빼자 FID가
       1792.8이 되었다
    4. **표지를 매끄럽게 해 본다.** 가장 잘 들었던 손잡이다 (FID 40.68)
    5. **$z$가 정말 쓰이는지 본다.** 같은 배치에 서로 다른 $z$를 넣어 출력이 다른지 확인한다

    5번을 값싸게 확인할 수 있다.

    ```python
    z1, z2 = torch.randn(1, k), torch.randn(1, k)
    print((g(z1) - g(z2)).abs().mean())      # 0 에 가까우면 z 를 무시한다
    ```

    손실을 보는 것은 목록에 없다는 점을 눈여겨볼 만하다. 무너진 모델의 손실이 정상으로
    보일 수 있으므로 진단에 쓸모가 없다
    ([GAN 기초 연습문제 3](45_gan.md)).

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
익히기 반복문에서 `detach()`를 어디에 왜 쓰는가?

</div>

??? success "연습문제 6 풀이"
    판별기를 갱신할 때 **거짓 그림에** 쓴다.

    ```python
    fake = g(z).detach()                  # 여기
    ld = bce(d(real), ones) + bce(d(fake), zeros)
    od.zero_grad(); ld.backward(); od.step()
    ```

    떼어 놓지 않으면 판별기의 손실에서 온 기울기가 생성기까지 흘러간다. 그러면 생성기가
    **자기를 들키게 하는 방향**으로 갱신되어 목표가 뒤집힌다.

    빠뜨리면 오류가 나지 않고 학습이 이상해질 뿐이라 찾기 어렵다. `od.step()`이 생성기
    매개변수를 건드리지는 않지만, 기울기가 생성기에 쌓여 다음 `og.step()`에 섞인다.

    생성기를 갱신할 때는 **떼지 않는다.** 기울기가 판별기를 지나 생성기까지 가야 하기
    때문이다.

    ```python
    lg = bce(d(g(z)), ones)               # detach 없다
    og.zero_grad(); lg.backward(); og.step()
    ```

    이때 판별기에도 기울기가 쌓이는데, 다음 판별기 갱신에서 `od.zero_grad()`가 지워
    주므로 문제가 없다. 순서를 지키는 것이 중요하다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
생성기 갱신에 쓰는 $z$를 판별기 갱신에 쓴 것과 같게 해도 되는가?

</div>

??? success "연습문제 7 풀이"
    같게 써도 되고 다시 뽑아도 된다. 다시 뽑는 편이 흔하고 조금 낫다고 여겨진다.

    같은 $z$를 쓰면 판별기가 방금 그 표본들을 거짓으로 보라고 배운 직후에 생성기가 같은
    표본으로 판별기를 속이려 한다. 신호가 더 날카로운 대신 그 특정 표본에 치우친다.

    다시 뽑으면 매 갱신이 새 표본을 보므로 잡음이 늘고 치우침이 줄어든다.

    ```python
    z = torch.randn(n, k, device=DEV)     # 판별기용
    fake = g(z).detach()
    ...
    z = torch.randn(n, k, device=DEV)     # 생성기용. 다시 뽑는다
    lg = bce(d(g(z)), ones)
    ```

    값은 생성기를 한 번 더 지나야 하므로 조금 더 든다. 같은 $z$를 쓰면 `g(z)`를 한 번만
    셈하고 재사용할 수 있다.

    이 장의 측정은 다시 뽑는 쪽으로 했다. 큰 차이를 보지는 못했으나, 관례를 따르는 것이
    남의 결과와 견주기에 낫다.

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
이 코드에 바서슈타인 손실을 넣으려면 무엇을 고치는가?

</div>

??? success "연습문제 8 풀이"
    네 곳이다.

    1. **판별기의 시그모이드를 뺀다.** 확률이 아니라 실수 점수를 낸다. 그래서 "비평자"라
       부른다
    2. **손실을 바꾼다.** BCE가 아니라 점수의 차이다

    ```python
    lc = -(c(real).mean() - c(fake).mean())      # 비평자
    lg = -c(g(z)).mean()                         # 생성기
    ```

    3. **리프시츠 제약을 건다.** 무게 자르기나 기울기 벌점을 쓴다

    ```python
    gp = gradient_penalty(c, real, fake)
    lc = lc + 10.0 * gp
    ```

    4. **비평자를 여러 번 갱신한다.** 보통 5:1이다. 여기서는 비평자가 강할수록 좋다

    넷째가 원래 손실과 정반대라는 점이 재미있다. 원래 손실에서는 판별기가 강하면 기울기가
    사라졌는데([GAN 기초 연습문제 2](45_gan.md)), 바서슈타인에서는 비평자가 최적에 가까울
    때 기울기가 가장 뜻 있다.

    까닭은 거리가 달라졌기 때문이다. 옌센–섀넌은 두 분포가 겹치지 않으면 상수가 되어
    기울기가 0인데, 바서슈타인 거리는 겹치지 않아도 **얼마나 멀리 옮겨야 하는지**를
    재므로 뜻 있는 기울기를 준다.

    그리고 비평자의 손실이 **바서슈타인 거리의 어림**이라 실제로 익히기를 지켜보는 데
    쓸 수 있다. 원래 손실로는 못 하던 일이다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
이 코드의 값매김을 익히기 반복문에 끼워 넣으려면 어떻게 하겠는가?

</div>

??? success "연습문제 9 풀이"
    참 자료 통계량을 미리 셈해 두고 에포크마다 FID를 잰다.

    ```python
    ref = stats_of(real_images[:10000], net)       # 한 번만
    fixed_z = torch.randn(64, k, device=DEV)       # 그림용. 고정
    for ep in range(epochs):
        train_one_epoch(...)
        g.eval()
        with torch.no_grad():
            s = torch.cat([g(torch.randn(1000, k, device=DEV)) for _ in range(2)])
            save_grid(g(fixed_z), f'ep{ep}.png')
        print(ep, fid_from_stats(features_of(s, net), *ref))
        g.train()
    ```

    챙긴 것이 넷이다.

    - 참 자료 통계량을 **한 번만** 셈한다
    - 표본 수를 **고정**한다(2,000). 에포크마다 다르면 곡선을 읽을 수 없다
    - 그림용 $z$를 **고정**한다. 변해 가는 모습이 보인다
    - `eval()`과 `train()`을 오간다. 배치 정규화가 있으므로 필요하다

    2,000개로 재면 바닥이 3 근처라는 것을 알고 있어야 한다. 그만큼의 요동은 모델이 아니라
    표본 수 탓이다([FID 짜기 연습문제 6](../gan_evaluation/02_frechet_inception_distance.md)).

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
이 코드를 CPU에서 돌리면 얼마나 걸리는가? 무엇이 병목인가?

</div>

??? success "연습문제 10 풀이"
    MNIST에 30 에포크라면 CPU에서 꽤 오래 걸리고, 애플 실리콘의 MPS나 CUDA에서는 몇
    분이면 된다.

    병목은 **한 걸음에 그물을 세 번 지나는 것**이다.

    | 무엇 | 몇 번 |
    |---|---|
    | 판별기 앞먹임 (참) | 1 |
    | 판별기 앞먹임 (거짓) | 1 |
    | 생성기 앞먹임 | 1 (또는 2) |
    | 뒤먹임 | 두 번 |

    지도 학습이 한 번 앞먹임에 한 번 뒤먹임인 것에 견주면 세 배쯤이다.

    값을 줄이는 길이 몇 가지다.

    - 배치를 크게 (GPU가 놀지 않게)
    - 생성기용 $z$를 다시 뽑지 않고 재사용 (앞먹임 한 번 아낌)
    - 값매김을 자주 하지 않기. FID가 생각보다 비싸다

    셋째가 놓치기 쉽다. 에포크마다 10,000개를 뽑아 FID를 재면 그것만으로 익히기와 비슷한
    값이 들 수 있다. 지켜보는 용도로는 2,000개면 충분하다.

## 정리하며

**다룬 것** — 깊은 겹말기 맞겨루기 만들개

DCGANGenerator은 두 단계 얼개를 쓴다.

고갱이 갈래는 `DCGANGenerator`, `DCGANDiscriminator`, `DCGAN64Generator`, `DCGAN64Discriminator`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
