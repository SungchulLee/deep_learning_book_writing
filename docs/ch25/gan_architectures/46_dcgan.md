# DCGAN

깊은 겹말기 맞겨루기 만들개(DCGAN)는 온전 이음 층을 겹말기 얼개로 바꾸어 본디 맞겨루기 만들개 틀을 넓힌다. 2015년에 나온 DCGAN은 맞겨루기 만들개 익히기의 안정과 표본 품질을 크게 높인 얼개 지침을 세웠다. 핵심 원칙에는 모으기 대신 성큼 겹말기 쓰기, 묶음 고르게 맞추기, 신경망마다 알맞은 깨움 함수가 든다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
DCGAN - 깊은 겹말기 맞겨루기 만들개
논문: "Unsupervised Representation Learning with Deep Convolutional GANs" (2015)
핵심: 여러 층 신경망을 겹말기 층으로 바꾼 맞겨루기 만들개의 얼개 지침
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class DCGenerator(nn.Module):
    def __init__(self, latent_dim=100, channels=1):
        super().__init__()
        self.init_size = 7
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )
    
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class DCDiscriminator(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(channels, 16, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, 1, 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.adv_layer = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
    
    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

class DCGAN(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.generator = DCGenerator(latent_dim)
        self.discriminator = DCDiscriminator()
    
    def forward(self, z):
        return self.generator(z)

if __name__ == "__main__":
    model = DCGAN()
    print(f"Generator Parameters: {sum(p.numel() for p in model.generator.parameters()):,}")
    print(f"Discriminator Parameters: {sum(p.numel() for p in model.discriminator.parameters()):,}")```

## 2. 논의

DCGAN 얼개는 본디 논문의 중요한 설계 원칙 여럿을 따른다. 만들개는 자리 바꾼 겹말기(이따금 역겹말기라 부른다)로 옹골찬 숨은 나타냄에서 온전한 그림 해상도로 키운다. 층마다 묶음 고르게 맞추기 뒤 ReLU 깨움을 쓰되 내놓기 층만 Tanh을 쓴다. 가름개는 이를 거울처럼 뒤집어 성큼 겹말기로 줄이며 LeakyReLU 깨움과 묶음 고르게 맞추기를 쓴다(들임 층은 뺀다).

이 짜기의 얼개는 28x28 회색 그림(MNIST)을 겨눈다. 만들개는 먼저 100차원 숨은 벡터를 선형 층으로 공간 특징 지도에 쏜 뒤 키우는 겹말기 덩이를 잇달아 쓴다. 가름개는 떨구기를 곁들인 성큼 겹말기로 규칙 세우기를 하며 공간 차원을 차츰 줄이고 채널 깊이를 늘린다.

DCGAN은 그림 만들어 내기에서 겹말기 얼개가 여러 층 신경망보다 훨씬 잘 듣는다는 것을 밝혔다. 가름개가 배운 특징은 뒤따르는 일에 쓸모 있는 나타냄이 될 수 있고, 숨은 공간은 뜻있는 셈 성질을 보인다(예컨대 얼굴 속성에 대한 벡터 셈).

## 연습문제

**연습문제 1.**
차원 100인 들임 숨은 벡터에 대해 DCGenerator의 층마다 공간 차원을 셈하라. 내놓기 꼴이 $(1, 28, 28)$임을 확인하라.

??? success "연습문제 1 풀이"
    숨은 벡터 (100,)은 선형 층으로 $128 \times 7 \times 7 = 6272$에 쏘인 뒤 $(128, 7, 7)$으로 다시 꼴 잡힌다. BatchNorm2d과 Upsample(scale_factor=2) 뒤: $(128, 14, 14)$. Conv2d(128, 128, 3, 1, 1)은 공간 크기를 지킨다: $(128, 14, 14)$. 다시 Upsample(scale_factor=2) 뒤: $(128, 28, 28)$. Conv2d(128, 64, 3, 1, 1): $(64, 28, 28)$. 마지막 Conv2d(64, 1, 3, 1, 1): $(1, 28, 28)$. 내놓기가 바라던 MNIST 그림 차원과 맞는다.

---

**연습문제 2.**
DCGAN 논문이 가름개의 들임 층과 만들개의 내놓기 층에 묶음 고르게 맞추기를 쓰지 말라고 권하는 까닭을 설명하라. 이 지침을 어기면 어떤 문제가 생길 수 있는가?

??? success "연습문제 2 풀이"
    가름개 들임 층의 묶음 고르게 맞추기는 날 화소 값을 묶음에 걸쳐 고르게 맞추어 실제 그림과 가짜 그림을 가르는 중요한 분포 앎을 무너뜨릴 수 있다. 만들개 내놓기 층에서는 묶음 고르게 맞추기가 내놓기 통계를 묶어 신경망이 화소 밝기의 온 범위를 지닌 그림을 내지 못하게 한다. 이 지침을 어기면 익히기가 불안정해지거나 봉우리가 무너지거나 자연스러운 명암을 지닌 다양한 그림을 만들지 못할 수 있다.

---

**연습문제 3.**
DCGAN을 $64 \times 64$ RGB 그림을 만들도록 고쳐라. 28x28 회색 판과 64x64 RGB 판의 매개변수 수를 견주고 셈에 미치는 뜻을 논하라.

??? success "연습문제 3 풀이"
    코드에 있는 DCGAN64Generator이 이 경우를 다룬다. 곧 숨은 벡터를 자리 바꾼 겹말기로 $(512, 4, 4)$으로 다시 꼴 잡은 뒤 ConvTranspose2d 층 넷을 지나 $(3, 64, 64)$에 이르도록 키운다. DCGAN64Discriminator은 이 길을 거꾸로 간다. 매개변수 수가 크게 는다. 곧 28x28 만들개는 약 350만 개인데 64x64 판은 약 360만 개이다. 가름개는 약 25만 개에서 약 280만 개로 는다. 셈 비용은 커진 공간 차원과 늘어난 채널 모두에 따라 커져 앞먹임마다 대략 12배 많은 부동 소수점 셈이 필요하다.

## 정리하며

**다룬 것** — DCGAN

DCGAN 얼개는 본디 논문의 중요한 설계 원칙 여럿을 따른다.

고갱이 갈래는 `DCGenerator`, `DCDiscriminator`, `DCGAN`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
