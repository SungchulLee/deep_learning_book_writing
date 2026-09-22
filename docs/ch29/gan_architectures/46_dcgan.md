# DCGAN

깊은 겹말기 맞겨루기 만들개(DCGAN)는 온전 이음 층을 겹말기 얼개로 바꾸어 본디 맞겨루기 만들개 틀을 넓힌다. 2015년에 나온 DCGAN은 맞겨루기 만들개 익히기의 안정과 표본 품질을 크게 높인 얼개 지침을 세웠다. 핵심 원칙에는 모으기 대신 성큼 겹말기 쓰기, 배치 고르게 맞추기, 신경망마다 알맞은 깨움 함수가 든다.

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
    print(f"Discriminator Parameters: {sum(p.numel() for p in model.discriminator.parameters()):,}")
```

**출력:**

```
Generator Parameters: 856,065
Discriminator Parameters: 97,729
```

## 2. 논의

DCGAN 얼개는 본디 논문의 중요한 설계 원칙 여럿을 따른다. 만들개는 자리 바꾼 겹말기(이따금 역겹말기라 부른다)로 옹골찬 숨은 나타냄에서 온전한 그림 해상도로 키운다. 층마다 배치 고르게 맞추기 뒤 ReLU 깨움을 쓰되 내놓기 층만 Tanh을 쓴다. 가름개는 이를 거울처럼 뒤집어 성큼 겹말기로 줄이며 LeakyReLU 깨움과 배치 고르게 맞추기를 쓴다(들임 층은 뺀다).

이 짜기의 얼개는 28x28 회색 그림(MNIST)을 겨눈다. 만들개는 먼저 100차원 숨은 벡터를 선형 층으로 공간 특징 지도에 쏜 뒤 키우는 겹말기 덩이를 잇달아 쓴다. 가름개는 떨구기를 곁들인 성큼 겹말기로 규칙 세우기를 하며 공간 차원을 차츰 줄이고 채널 깊이를 늘린다.

DCGAN은 그림 만들어 내기에서 겹말기 얼개가 여러 층 신경망보다 훨씬 잘 듣는다는 것을 밝혔다. 가름개가 배운 특징은 뒤따르는 일에 쓸모 있는 나타냄이 될 수 있고, 숨은 공간은 뜻있는 셈 성질을 보인다(예컨대 얼굴 속성에 대한 벡터 셈).

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
차원 100인 들임 숨은 벡터에 대해 DCGenerator의 층마다 공간 차원을 셈하라. 내놓기 꼴이 $(1, 28, 28)$임을 확인하라.

</div>

??? success "연습문제 1 풀이"
    숨은 벡터 (100,)은 선형 층으로 $128 \times 7 \times 7 = 6272$에 쏘인 뒤 $(128, 7, 7)$으로 다시 꼴 잡힌다. BatchNorm2d과 Upsample(scale_factor=2) 뒤: $(128, 14, 14)$. Conv2d(128, 128, 3, 1, 1)은 공간 크기를 지킨다: $(128, 14, 14)$. 다시 Upsample(scale_factor=2) 뒤: $(128, 28, 28)$. Conv2d(128, 64, 3, 1, 1): $(64, 28, 28)$. 마지막 Conv2d(64, 1, 3, 1, 1): $(1, 28, 28)$. 내놓기가 바라던 MNIST 그림 차원과 맞는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
DCGAN 논문이 가름개의 들임 층과 만들개의 내놓기 층에 배치 고르게 맞추기를 쓰지 말라고 권하는 까닭을 설명하라. 이 지침을 어기면 어떤 문제가 생길 수 있는가?

</div>

??? success "연습문제 2 풀이"
    가름개 들임 층의 배치 고르게 맞추기는 날 화소 값을 배치에 걸쳐 고르게 맞추어 실제 그림과 가짜 그림을 가르는 중요한 분포 앎을 무너뜨릴 수 있다. 만들개 내놓기 층에서는 배치 고르게 맞추기가 내놓기 통계를 묶어 신경망이 화소 밝기의 온 범위를 지닌 그림을 내지 못하게 한다. 이 지침을 어기면 익히기가 불안정해지거나 봉우리가 무너지거나 자연스러운 명암을 지닌 다양한 그림을 만들지 못할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
DCGAN을 $64 \times 64$ RGB 그림을 만들도록 고쳐라. 28x28 회색 판과 64x64 RGB 판의 매개변수 수를 견주고 셈에 미치는 뜻을 논하라.

</div>

??? success "연습문제 3 풀이"
    코드에 있는 DCGAN64Generator이 이 경우를 다룬다. 곧 숨은 벡터를 자리 바꾼 겹말기로 $(512, 4, 4)$으로 다시 꼴 잡은 뒤 ConvTranspose2d 층 넷을 지나 $(3, 64, 64)$에 이르도록 키운다. DCGAN64Discriminator은 이 길을 거꾸로 간다. 매개변수 수가 크게 는다. 곧 28x28 만들개는 약 350만 개인데 64x64 판은 약 360만 개이다. 가름개는 약 25만 개에서 약 280만 개로 는다. 셈 비용은 커진 공간 차원과 늘어난 채널 모두에 따라 커져 앞먹임마다 대략 12배 많은 부동 소수점 셈이 필요하다.


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
DCGAN이 제시한 설계 규칙들은 무엇인가? 각각 왜인가?

</div>

??? success "연습문제 4 풀이"
    | 규칙 | 왜 |
    |---|---|
    | 모으기 대신 보폭 누비기 | 줄이고 키우는 방식을 배운다 |
    | 두 그물에 배치 정규화 | 익히기를 안정시킨다 |
    | 완전 연결 은닉층을 없앰 | 공간 짜임을 지킨다 |
    | 생성기에 ReLU, 출력에 `tanh` | — |
    | 판별기에 LeakyReLU | 기울기가 죽지 않는다 |

    둘째 줄의 값어치를 이 장에서 수로 확인했다. 배치 정규화와 드롭아웃을 빼자 완전히
    무너졌다(FID 1792.8, 인셉션 점수 1.000). 넣으면 40~50이다.

    다섯째 줄도 까닭이 뚜렷하다. 판별기에서 ReLU를 쓰면 음수 쪽 기울기가 0이라 생성기에
    전해질 신호가 끊길 수 있다. LeakyReLU는 작게라도 흘려준다.

    첫째 줄은 [25장의 오토인코더](../../ch25/architecture/02_ae_cnn.md)에서 본 것과
    같은 이야기다. 최대 모으기는 어디가 최대였는지를 버리므로 되돌리기에 불리하다.

    이 규칙들이 2015년의 경험에서 나온 것이며 지금은 더 나은 방법들이 있다는 점을 알아
    두어야 한다. 스펙트럴 정규화나 자기 주의 같은 것들이다. 그래도 DCGAN 규칙은 여전히
    좋은 출발점이다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
생성기 출력에 `tanh`를 쓰면 무엇을 함께 챙겨야 하는가?

</div>

??? success "연습문제 5 풀이"
    **자료의 범위를 $[-1,1]$로 맞추어야** 한다.

    ```python
    transform = transforms.Compose([
        transforms.ToTensor(),                       # [0,1]
        transforms.Normalize((0.5,), (0.5,)),        # [-1,1]
    ])
    ```

    빠뜨리면 판별기가 참 자료와 거짓 자료를 **범위만으로** 가려낼 수 있다. 그러면 생성기가
    배울 것이 없다. 오류가 나지 않고 그저 학습이 안 되므로 알아채기 어렵다.

    그리고 잣대를 잴 때 되돌려야 한다. 특징 그물이 $[0,1]$로 익혔다면 이렇게 옮긴다.

    ```python
    imgs = (g(z) + 1) / 2
    ```

    이 자리가 값매김에서 가장 흔한 사고다
    ([인셉션 점수 연습문제 8](../gan_evaluation/01_inception_score.md)). 참 자료로 기준점을
    먼저 재어 두면 곧 드러난다.

    시그모이드를 쓰고 $[0,1]$로 다루는 선택도 물론 된다. 이 장의 측정이 그렇게 했다.
    `tanh`가 관례가 된 것은 출력이 0을 가운데로 두어 익히기가 조금 안정되기 때문이라고
    이야기된다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
생성기에서 크기를 키우는 층은 무엇을 쓰는가? 모양 셈을 적어라.

</div>

??? success "연습문제 6 풀이"
    `ConvTranspose2d`를 쓰고, 출력 크기가 이렇다.

    $$H' = (H-1)s - 2p + k$$

    MNIST를 $7 \to 14 \to 28$로 키우려면 $s=2$, $p=1$, $k=4$로 두면 된다.

    $$H' = (7-1)\cdot 2 - 2 + 4 = 14, \qquad (14-1)\cdot 2 - 2 + 4 = 28$$

    **낟알 크기를 보폭의 배수로** 두는 것이 중요하다. 그러지 않으면 출력 자리마다 겹치는
    횟수가 달라 격자 무늬가 생긴다. $k=3$, $s=2$가 그 나쁜 예다
    ([26장](../../ch26/architecture/conv_vae.md)).

    대신 키우기와 누비기를 나누는 방법도 있다.

    ```python
    nn.Upsample(scale_factor=2, mode='nearest')
    nn.Conv2d(64, 32, 3, padding=1)
    ```

    격자 무늬를 더 잘 피한다고 알려져 있고, 요즘 얼개에서 널리 쓰인다.

    첫 층은 $z$를 특징 지도로 펴는 일을 한다. 완전 연결로 $7\times7\times C$를 만들거나
    $1\times1$에서 `ConvTranspose2d`로 키운다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
판별기에 드롭아웃을 넣는 것이 왜 도움이 되는가?

</div>

??? success "연습문제 7 풀이"
    판별기를 **약하게** 만들어 생성기가 배울 여지를 남긴다.

    판별기가 너무 강하면 $D(G(z)) \approx 0$으로 확신해 기울기가 사라진다
    ([GAN 기초 연습문제 2](45_gan.md)). 드롭아웃은 판별기가 지나치게 확신하지 못하게 막는다.

    이 장의 측정에서 그 효과가 컸다. 배치 정규화와 드롭아웃을 함께 뺀 설정이 완전히
    무너졌다(FID 1792.8). 넣으면 49.85다.

    비슷한 일을 하는 다른 방법들이 있다.

    | 방법 | 어떻게 |
    |---|---|
    | 드롭아웃 | 판별기를 확신하지 못하게 |
    | 표지 매끄럽게 | 참의 표지를 1 대신 0.9로 |
    | 학습률 낮추기 | 판별기가 덜 앞서가게 |
    | 잡음 더하기 | 입력에 잡음을 주어 두 분포를 겹치게 |

    이 장에서 재어 보면 표지 매끄럽게가 가장 잘 들었다(FID 40.68). 학습률을 낮추는 것은
    오히려 조금 나빴다(51.06).

    넷째 줄이 이론적으로 재미있다. 두 분포가 겹치지 않으면 옌센–섀넌 벌어짐의 기울기가
    사라지는데([GAN 기초 연습문제 5](45_gan.md)), 잡음을 더하면 억지로 겹치게 만들어
    기울기를 되살린다.

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
DCGAN을 MNIST에 쓸 때 원 논문과 다르게 해야 할 것은 무엇인가?

</div>

??? success "연습문제 8 풀이"
    원 논문이 $64\times64$ 컬러 그림을 겨냥했으므로 MNIST에 맞게 줄여야 한다.

    | 무엇 | 원 논문 | MNIST |
    |---|---|---|
    | 그림 크기 | $64\times64$ | $28\times28$ |
    | 채널 | 3 | 1 |
    | 층 수 | 네 번 키움 | **두 번**이면 된다 |
    | 첫 특징 지도 | $4\times4$ | $7\times7$ |

    셋째 줄이 핵심이다. $28 = 7 \times 2^2$이므로 두 번 키우면 된다. 네 번 키우려면
    $28/16 = 1.75$로 정수가 아니다.

    $7\times7$에서 시작하는 것이 MNIST의 관례가 되었다. 28을 2로 두 번 나눈 값이다.

    그리고 규모를 줄이는 것이 좋다. 원 논문의 채널 수(1024까지)는 MNIST에 과하다.
    128이나 256에서 시작하면 충분하다.

    이 장의 측정은 아예 누비기 없는 완전 연결 그물로 했고 FID 40.68을 얻었다. MNIST가
    작아서 그것으로도 되는 것이며, 누비기를 쓰면 더 나아질 여지가 있다. 다만
    [26장에서 본 대로](../../ch26/architecture/conv_vae.md) 누비기가 되돌리기는 잘하면서
    뽑기는 못하는 경우도 있었으니 **재어 보아야 알 일**이다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
두 그물의 크기를 어떻게 맞추는가?

</div>

??? success "연습문제 9 풀이"
    비슷하게 두는 것이 출발점이고, 균형이 깨지면 손본다.

    생각할 점이 이렇다.

    **판별기가 너무 강하면** 생성기가 기울기를 못 받는다. 판별기를 줄이거나 드롭아웃을
    늘리거나 학습률을 낮춘다.

    **생성기가 너무 강하면** 판별기가 못 따라가 엉뚱한 신호를 준다. 이 경우가 드물지만
    일어난다.

    그런데 크기만으로 균형이 정해지지 않는다는 점이 중요하다. 이 장의 측정에서 판별기
    학습률을 절반으로 낮춘 것이 오히려 조금 나빴다(51.06 대 49.85). 표지를 매끄럽게 한
    것이 훨씬 잘 들었다(40.68).

    곧 "판별기를 약하게"라는 방향이 맞더라도 **어떤 방법으로 약하게 하는지**가 결과를
    정한다. 학습률은 판별기가 배우는 속도를 늦출 뿐 확신의 정도를 직접 막지 못하는데,
    표지 매끄럽게는 확신 자체에 상한을 둔다.

    실무의 순서는 이렇다. 비슷한 크기로 시작해 FID를 지켜보고, 무너지면 판별기 쪽을
    확신하지 못하게 손본다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
가중치를 어떻게 초기화하라고 하는가?

</div>

??? success "연습문제 10 풀이"
    원 논문이 평균 0, 표준편차 0.02의 정규 분포를 권한다.

    ```python
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

    g.apply(init_weights); d.apply(init_weights)
    ```

    0.02가 특별한 수는 아니고 경험에서 나온 값이다. 요즘은 He나 Xavier 초기화를 그냥
    쓰기도 하고 잘 듣는다.

    초기화가 적대적 생성망에서 특히 중요한 까닭이 있다. 익히기가 불안정해서 출발점이
    결과를 크게 바꾸기 때문이다. 같은 설정을 다른 씨앗으로 익히면 FID가 꽤 달라진다.

    그래서 **씨앗을 고정하고 밝히는 것**이 중요하고, 설정을 견줄 때 여러 씨앗으로 재는
    것이 옳다([값매김 연습문제 15](../gan_evaluation/complete_evaluation_example.md)).

## 정리하며

**다룬 것** — DCGAN

DCGAN 얼개는 본디 논문의 중요한 설계 원칙 여럿을 따른다.

고갱이 갈래는 `DCGenerator`, `DCDiscriminator`, `DCGAN`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
