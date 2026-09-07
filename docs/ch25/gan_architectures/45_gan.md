# 맞겨루기 만들개

맞겨루기 만들개(GAN)는 맞겨루기 익히기로 만들어 내는 모델에 뒤엎는 방식을 들여왔다. 신경망 둘, 곧 만들개와 가름개가 서로 겨룬다. 만들개는 그럴듯한 자료를 내려 하고 가름개는 실제 표본과 만든 표본을 가려내려 한다. 이 맞겨루는 움직임이 두 신경망을 함께 나아지게 하여 마침내 품질 높은 자료를 만들어 낸다.

## 코드

```python
#!/usr/bin/env python3
'''
GAN - 맞겨루기 만들개
논문: "Generative Adversarial Networks" (2014)
고갱이: 맞겨루며 익히는 두 그물(만들개와 가름개)
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_size=28):
        super().__init__()
        self.img_size = img_size
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, img_size * img_size * 1),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, self.img_size, self.img_size)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_size=28):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(img_size * img_size * 1, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

class GAN(nn.Module):
    def __init__(self, latent_dim=100, img_size=28):
        super().__init__()
        self.generator = Generator(latent_dim, img_size)
        self.discriminator = Discriminator(img_size)
    
    def forward(self, z):
        return self.generator(z)

if __name__ == "__main__":
    model = GAN()
    print(f"Generator Parameters: {sum(p.numel() for p in model.generator.parameters()):,}")
    print(f"Discriminator Parameters: {sum(p.numel() for p in model.discriminator.parameters()):,}")
```

## 논의

맞겨루기 만들개 얼개는 겨루는 신경망 둘로 이루어진다. 만들개는 아무 숨은 벡터 $z \sim \mathcal{N}(0, I)$을 묶음 고르게 맞추기와 LeakyReLU 깨움을 갖춘 온전 이음 층 여러 개를 지나 옮기고 마침내 Tanh 내놓기로 그림을 만든다. 가름개는 (실제이든 만든 것이든) 그림을 받아 들임이 실제인지 나타내는 확률을 내놓는다. 이 맞겨루기 채비는 최소최대 목표로 갖추어 적는다.

$$
\min_G \max_D \; \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

이 짜기의 만들개는 차츰 넓히는 얼개를 써서 그림 공간으로 쏘기 앞에 낱개를 128에서 1024까지 늘린다. 묶음 고르게 맞추기는 중간 깨움을 고르게 맞추어 익히기를 안정시키고, 기울기 0.2인 LeakyReLU은 신경 세포가 죽는 것을 막는다. 가름개는 이를 거울처럼 뒤집은 오그리는 얼개로 펼친 그림에서 시그모이드 내놓기 하나로 차원을 줄인다.

맞겨루는 움직임 탓에 맞겨루기 만들개를 익히기는 몹시 어렵기로 이름났다. 만들개가 좁은 범위의 내놓기만 내는 봉우리 무너짐과 익히기의 불안정이 흔한 문제이다. 만들개와 가름개 익히기의 균형이 결정적이다. 곧 가름개가 너무 세면 만들개의 기울기가 사라지고, 너무 여리면 쓸모 있는 신호를 주지 못한다. 이런 어려움에도 맞겨루기 만들개 틀은 놀랍도록 두루 쓸 만함이 드러났고 여러 앞선 만들어 내는 얼개의 바탕이 된다.

## 연습문제

**연습문제 1.**
`latent_dim=100`과 `img_size=28`에서 만들개와 가름개의 매개변수 총수를 셈하라. 어느 신경망이 매개변수가 더 많은가? 그것이 익히기의 안정에 왜 중요할 수 있는가?

??? success "연습문제 1 풀이"
    만들개: Linear(100, 128) = 12,928; Linear(128, 256) + BN(256) = 33,536; Linear(256, 512) + BN(512) = 132,096; Linear(512, 1024) + BN(1024) = 527,360; Linear(1024, 784) = 803,600. 만들개 매개변수 총수는 약 1,509,520이다.

    가름개: Linear(784, 512) = 401,920; Linear(512, 256) = 131,328; Linear(256, 1) = 257. 가름개 매개변수 총수는 약 533,505이다.

    만들개는 차원 낮은 숨은 공간에서 차원 높은 그림 공간으로 가는 복잡한 옮김을 배워야 하므로 매개변수가 대략 3배 많고, 가름개는 더 단순한 가르기 일을 한다. 가름개가 만들개보다 빨리 모일 수 있으므로 이 어긋남이 익히기의 안정에 영향을 줄 수 있다.

---

**연습문제 2.**
만들개는 왜 마지막 깨움으로 `nn.Tanh()`을 쓰고 가름개는 `nn.Sigmoid()`을 쓰는지 설명하라. 둘을 바꾸면 어떻게 되는가?

??? success "연습문제 2 풀이"
    만들개는 Tanh으로 $[-1, 1]$ 안의 값을 내어 고르게 맞춘 실제 그림의 범위와 맞춘다. 가름개는 시그모이드로 $[0, 1]$ 안의 확률을 내어 들임이 실제일 가능도를 나타낸다. 가름개에 Tanh을 쓰면 내놓기가 올바른 확률이 아니어서 두값 어긋 엔트로피 손실이 깨진다. 만들개에 시그모이드를 쓰면 내놓기가 $[0, 1]$으로 갇히는데, 이는 올바른 범위이긴 하나 Tanh에 견주어 가장자리 가까이에서 기울기가 여려 익히기가 느려질 수 있다.

---

**연습문제 3.**
맞겨루기 만들개를 크기 $64 \times 64$인 RGB 그림에 맞게 고쳐라. 만들개와 가름개 얼개에 어떤 바꿈이 필요한가? 고친 판을 짜고 새 매개변수 총수를 세어라.

??? success "연습문제 3 풀이"
    $64 \times 64$ RGB 그림에서는 만들개의 마지막 선형 층이 `nn.Linear(1024, 64 * 64 * 3)`으로 바뀌고 내놓기를 `(batch, 3, 64, 64)`으로 다시 꼴 잡는다. 가름개의 들임 층은 `nn.Linear(64 * 64 * 3, 512)`으로 바뀐다. 그림 공간이 커지므로 숨은 층과 담이를 더 두어야 할 수 있다. 만들개 내놓기 차원이 784에서 12,288으로 늘고 가름개 들임도 마찬가지로 는다. 그래서 매개변수가 크게 늘며(만들개 마지막 층만 해도 $1024 \times 12288 + 12288 = 12,595,200$), 더 큰 그림에는 DCGAN 같은 겹말기 얼개로 옮겨 갈 까닭이 된다.
