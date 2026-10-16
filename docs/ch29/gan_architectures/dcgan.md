# 28×28·64×64 DCGAN 구조

이 모듈은 전치 컨볼루션 생성기와 스트라이드 컨볼루션 판별기를 두 해상도로 정의한다. [DCGAN 논문](https://arxiv.org/abs/1511.06434)의 경험적 지침을 참고하지만 28×28 버전은 큰 선형 투영을 사용하는 별도 변형이다. 앞의 [업샘플링 모듈](46_dcgan.md)과 클래스·채널 폭·파라미터 수·공간 계산이 다르므로 구분해서 읽는다.

## 1. 두 해상도의 구현

```python
"""
깊은 컨볼루션 GAN(DCGAN)

다음 지침을 따른 DCGAN 짜기:
"Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"
(Radford et al., 2016)

핵심 구조 원칙:
1. 모으기를 성큼 겹말기로 바꾼다
2. G과 D 모두에 배치 고르게 맞추기를 쓴다
3. 온전 이음 숨은 레이어를 없앤다
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
    DCGAN 생성기 뉴럴 넷트웍.
    
    자리 바꾼 겹말기로 숨은 벡터 z을 그림으로 옮긴다.
    """
    
    def __init__(self, latent_dim: int = 100, image_channels: int = 1, 
                 feature_maps: int = 64):
        """
        인수:
            latent_dim: 숨은 벡터 z의 차원
            image_channels: 내놓음 통로 수(잿빛이면 1, RGB이면 3)
            feature_maps: 첫 레이어의 피처 지도 수(레이어마다 2배로 는다)
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
        
        # 겹말기 레이어(키우기)
        self.main = nn.Sequential(
            # 들임: (feature_maps*8) x 7 x 7
            
            # 레이어 1: 14x14으로 키운다
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 
                             kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # 내놓기: (feature_maps*4) x 14 x 14
            
            # 레이어 2: 28x28으로 키운다
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2,
                             kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # 내놓기: (feature_maps*2) x 28 x 28
            
            # 레이어 3: 그림 채널로 가는 마지막 겹말기
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
    DCGAN 판별기 뉴럴 넷트웍.
    
    성큼 겹말기로 그림을 실제인지 가짜인지 가른다.
    """
    
    def __init__(self, image_channels: int = 1, feature_maps: int = 64):
        """
        인수:
            image_channels: 들임 통로 수(잿빛이면 1, RGB이면 3)
            feature_maps: 첫 레이어의 피처 지도 수(레이어마다 2배로 는다)
        """
        super().__init__()
        
        self.main = nn.Sequential(
            # 들임: image_channels x 28 x 28
            
            # 레이어 1: 들임 레이어에는 배치 고르게 맞추기를 쓰지 않는다(DCGAN 지침)
            nn.Conv2d(image_channels, feature_maps, 
                     kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 내놓기: feature_maps x 14 x 14
            
            # 2레이어
            nn.Conv2d(feature_maps, feature_maps * 2,
                     kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 내놓기: (feature_maps*2) x 7 x 7
            
            # 3레이어
            nn.Conv2d(feature_maps * 2, feature_maps * 4,
                     kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 내놓기: (feature_maps*4) x 4 x 4
            
            # 내놓기 레이어: 값 하나로 겹말기 한다
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
    64x64 그림을 위한 DCGAN 생성기(본디 논문에 더 가깝게 따른다).
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
    64x64 그림을 위한 DCGAN 판별기(본디 논문을 따른다).
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
    """DCGAN 구조를 시험한다."""
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
    
    # 파라메터 개수 세기
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

28×28 생성기는 기본 `(B,100)`을 `(B,512,7,7)`로 투영한다. 두 전치 컨볼루션으로 `(B,256,14,14)`, `(B,128,28,28)`을 만들고 일반 3×3 컨볼루션으로 `(B,1,28,28)`을 출력한다. 4차원 잡음을 받으면 먼저 펼치므로 배치당 원소 수가 latent_dim과 맞아야 한다.

28×28 판별기는 공간 28→14→7→4→1과 채널 1→64→128→256→1로 처리해 `(B,1)` 확률을 반환한다. 마지막 커널이 4라서 4×4에서 1×1이 되며,앞 절의 3×3 마지막 커널과 다르다.

64×64 생성기는 `(B,100,1,1)` 또는 이에 해당하는 2차원 벡터에서 4→8→16→32→64로 키우고 RGB를 출력한다. 판별기는 64→32→16→8→4→1로 줄인다. 이 모듈들은 목표 해상도용 구조이며 임의 해상도에서는 마지막 view(-1,1)이 공간 점수를 배치처럼 펼칠 수 있어 표본당 하나인지 확인해야 한다.

**기존 출력 기록:**

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

기본 feature_maps=64에서 28×28 G는 5,207,424개,D는 431,872개다. 64×64 RGB G는 3,576,704개,D는 2,765,568개다. 작은 이미지 G가 더 큰 것은 25,088차원 선형 투영과 그 BatchNorm1d 파라미터가 크기 때문이다. 이미지 크기만으로 전체 파라미터 수를 정할 수 없다. 실행부는 무작위 모델의 모양·집계를 확인하며 학습·생성 품질 평가를 하지 않는다.

## 2. 범위·활성화·모드

생성기의 Tanh 출력은 −1~1,판별기의 Sigmoid 출력은 0~1이다. 실제 이미지 전처리를 생성 범위와 맞추고 확률 BCE의 출력 계약을 지켜야 한다. logits 로스를 원하면 판별기 마지막 활성화도 맞춰야 한다. 컨볼루션 G의 내부 ReLU와 D의 LeakyReLU는 경험적 선택이며 모든 그래디언트·안정성을 보장하지 않는다.

28×28 G는 투영 벡터에 BatchNorm1d를 적용해 학습 모드의 배치 1 생성에서 통계를 계산할 수 없다. eval에서 running 통계를 사용하면 단일 표본 생성이 가능하지만 그 통계가 적절히 학습되었는지는 별도다. eval은 그래디언트 기록을 끄지 않으므로 생성·평가에는 no_grad 등도 따로 적용한다.

BatchNorm 뒤 컨볼루션 바이어스는 학습 배치 평균 제거·학습 이동 파라미터와 중복될 수 있어 생략하기도 한다. 그러나 모든 레이어 뒤에 BatchNorm이 있는 것은 아니고 모드·통계 계약에 따라 다르므로 bias=False가 항상 무로스인 보편 규칙은 아니다.

## 3. 별도 학습·평가의 필요

모듈에는 학습 루프·체크포인트 선택·정규화 비교 실험이 없다. [적대적 학습](../gan_foundations/adversarial.md)에서 설명한 그래프 분리·두 optimizer·상태 관리와 실제 데이터 분할을 연결해야 한다. 생성 품질·다양성·우도 계산 가능성과 파라미터 집계는 서로 다른 속성이다.

잠재 보간은 학습된 G의 경로 진단이지만 선형·구면 보간 중 하나가 항상 자연스럽다는 보장은 없다. 평가 피처·전처리·표본 수와 선택 예산을 맞추고,고정 잠재 격자 외에도 새 표본을 평가한다. 시간 비용은 모델 폭·배치·장치·평가 빈도에 따라 달라 실행 없이 특정 분 단위나 보편 배율을 제시할 수 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
BatchNorm 뒤 컨볼루션 바이어스 생략의 의미와 적용 범위를 설명하라.

</div>

??? success "연습문제 1 풀이"

    표본·공간에 공통으로 더한 채널 바이어스는 학습 BatchNorm의 평균 제거에 의해 상쇄되고 affine 이동 β로 오프셋을 표현할 수 있어 중복된다. 평가 running 통계와 학습 상태도 함께 맞춰야 이 관계를 해석할 수 있다. 마지막 Tanh·Sigmoid 앞처럼 BatchNorm이 없는 레이어에도 무조건 같은 논리가 적용되지는 않는다. 바이어스가 켜졌다고 코드가 잘못되거나 속도가 특정 비율로 느려진다고 단정하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
가우시안 잠재 공간에서 선형·구면 보간을 해석하라.

</div>

??? success "연습문제 2 풀이"

    독립 N(0,I) 끝점의 선형 중간점은 N(0,I/2)이므로 기대 제곱노름이 d/2로 줄어든다. 원점은 가우시안 점별 밀도가 가장 높지만 고차원 확률 질량의 전형적 반경은 약 √d다. “원점이 저밀도”라는 설명은 점별 밀도와 전형적 질량을 혼동한다. 일정 반경의 slerp는 정규화·끝점 반경 폴리시를 정해야 하며,경로가 그럴듯한 생성 이미지를 보장하지 않는다. 두 방식은 여러 끝점에서 목적별로 비교한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
초기화 골격이 바꾸는 레이어와 통계 검증의 한계를 설명하라.

</div>

??? success "연습문제 3 풀이"

    아래 함수는 이름에 Conv가 든 레이어의 웨이트를 N(0,0.02²),BatchNorm 웨이트를 N(1,0.02²),바이어스를 0으로 바꾼다. 생성기 project의 Linear 웨이트·바이어스는 그대로다. 예시는 gen에만 적용하므로 D에도 쓰려면 별도 적용이 필요하다. 유한 표본 평균·표준편차는 목표값과 정확히 같지 않고,이 초기화만으로 학습 불안정이 사라지지는 않는다. 실제 적용 범위와 시드를 기록한다.

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


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
기본 28×28 G의 잠재 투영 크기와 아래 대안의 차이를 설명하라.

</div>

??? success "연습문제 4 풀이"

    실제 project는 100→25,088=512×7×7이며 선형만 2,533,888개,뒤 BatchNorm affine은 50,176개다. 첫 골격은 이 투영·reshape를 일반화한 예이고 self.fc는 본문의 속성명 project와 다르다. 두 번째 ConvTranspose2d 대안은 1×1 잠재 지도에서 7×7을 만드는 다른 구조이며 채널·바이어스·정규화를 정해야 정확한 파라미터 비교가 가능하다. 7×7 시작은 28을 두 배 두 번으로 만드는 한 설계일 뿐 유일한 방법은 아니다.

    ```python
    h = self.fc(z)                       # (B, C*7*7)
    h = h.view(-1, C, 7, 7)              # (B, C, 7, 7)
    ```
    ```python
    z = z.view(-1, latent_dim, 1, 1)
    nn.ConvTranspose2d(latent_dim, 256, 7, 1, 0)     # 1x1 -> 7x7
    ```


---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
단일 잠재 입력 차이 검사와 모드 붕괴 진단의 한계를 설명하라.

</div>

??? success "연습문제 5 풀이"

    아래 z1·z2는 배치 1이라 기본 28×28 G의 학습 모드 BatchNorm1d에서 실패한다. 모델 장치·자료형을 맞추고 eval 등 적절한 조건에서 비교해야 한다. 서로 다른 z의 출력 차이만으로 전체 다양성이나 데이터 포괄도가 확인되지는 않는다. 클래스 엔트로피·인셉션 점수도 평가기 관점의 요약이며 IS≈1이 반드시 같은 이미지 생성의 증거는 아니다. 로스·출력·대표 표본과 품질·다양성 진단을 결합한다.

    ```python
    z1, z2 = torch.randn(1, k), torch.randn(1, k)
    print((g(z1) - g(z2)).abs().mean())      # 0 에 가까우면 z 를 무시한다
    ```


---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
D 단계의 detach를 빼면 G가 반드시 잘못 업데이트되는가?

</div>

??? success "연습문제 6 풀이"

    detach가 없으면 D 로스의 그래디언트가 G까지 계산되지만 D optimizer가 D만 가진다면 그 step이 G를 직접 바꾸지는 않는다. 다음 G 단계 전에 G grad를 지우면 해당 누적값이 실제 G 업데이트에 섞이지 않을 수도 있다. 따라서 “항상 반대 방향으로 학습된다”는 설명은 틀리다. 그래도 불필요한 그래디언트·그래프 수명·재사용 오류를 피하려 D 단계에서 분리하는 것이 명확하다. 아래 G 로스 경로에서는 D를 통해 G까지 미분을 유지해야 한다. 파라미터 grad와 BatchNorm·dropout 상태는 별도다.

    ```python
    fake = g(z).detach()                  # 여기
    ld = bce(d(real), ones) + bce(d(fake), zeros)
    od.zero_grad(); ld.backward(); od.step()
    ```
    ```python
    lg = bce(d(g(z)), ones)               # detach 없다
    og.zero_grad(); lg.backward(); og.step()
    ```


---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
같은 z 재사용과 새 z 사용을 어떻게 구분해 설계하는가?

</div>

??? success "연습문제 7 풀이"

    둘 다 가능하지만 D를 갱신한 뒤 G까지 연결되는 유효한 새 D forward가 필요하다. 같은 fake 그래프를 재사용하려면 D 단계에서 detach하고 G가 아직 바뀌지 않았는지 등 그래프 수명 조건을 맞춘다. 새 표본이 언제나 바이어스를 줄이거나 품질을 높이는 것은 아니다. 아래는 새 z 사용 골격이며 생략 부호 부분의 D 단계·장치·모드도 정의해야 한다. 계산비용과 확률적 변동을 같은 조건에서 비교한다.

    ```python
    z = torch.randn(n, k, device=DEV)     # 판별기용
    fake = g(z).detach()
    ...
    z = torch.randn(n, k, device=DEV)     # 생성기용. 다시 뽑는다
    lg = bce(d(g(z)), ones)
    ```


---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
Wasserstein 목적을 넣을 때 로스 외에 필요한 계약을 설명하라.

</div>

??? success "연습문제 8 풀이"

    Sigmoid 대신 실수 critic 점수,점수 차이 목적과 Lipschitz 제어를 함께 정의해야 한다. 아래 최소화 critic 로스는 E_fake c−E_real c이고 G는 −E_fake c다. GP 함수가 이미 λ를 곱하는지 확인해야 하며 앞 내시 절의 기본 gradient_penalty는 λ=10을 포함해 여기에 다시 10을 곱하면 100배가 된다. 이미지 보간·전체 입력 노름·표본별 그래디언트도 맞춘다. critic을 더 자주 갱신하는 선택이 모든 작업에 최적이거나 학습 로스가 정확한 Wasserstein 거리라는 보장은 없다.

    ```python
    lc = -(c(real).mean() - c(fake).mean())      # 비평자
    lg = -c(g(z)).mean()                         # 생성기
    ```
    ```python
    gp = gradient_penalty(c, real, fake)
    lc = lc + 10.0 * gp
    ```


---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
아래 학습 중 평가 골격에서 무엇을 고정·기록해야 하는가?

</div>

??? success "연습문제 9 풀이"

    피처 추출기·전처리·참조 자료·표본 수를 고정하고 변동을 보고한다. 골격은 매 평가 2,000개 새 생성 표본과 별도 고정 z 그림을 쓰며,stats_of·features_of·fid_from_stats는 실제 구현이 필요하다. eval·no_grad와 학습 모드 복원을 적용한다. 참조 통계를 캐시할 수 있지만 평가 자료를 반복 선택에 쓴 결과와 최종 테스트는 분리한다. 2,000개 FID의 보편적인 바닥값 3이나 충분한 정확도는 없고 피처·분포·표본 수에 따라 달라진다.

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


---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
교대 학습의 계산비용을 세는 올바른 기준을 설명하라.

</div>

??? success "연습문제 10 풀이"

    보통 D는 실제와 가짜의 D 단계 두 번에 더해 G 단계에서도 한 번 forward한다. G는 재사용 폴리시에 따라 한 번 또는 두 번 호출하며 역전파의 대상·버퍼·엑티베이션 저장도 다르다. 따라서 지도 학습보다 무조건 3배나 특정 장치에서 몇 분이라고 계산할 수 없다. 실제 모델·배치·정밀도·평가 빈도를 정한 뒤 프로파일링해야 한다. 이 페이지는 새 시간 측정을 하지 않으며 평가 표본 수의 충분성도 목적에 맞춰 정해야 한다.



## 정리하며

두 DCGAN 버전은 목표 해상도와 잠재 입력 계약이 다르다. 정확한 공간·파라미터 계산과 BatchNorm 모드를 확인하고,detach·잡음 재사용·critic 제약·평가 폴리시는 별도로 설계한다. 모양 검사나 초기화만으로 학습 품질·안정성을 보장할 수 없다.
