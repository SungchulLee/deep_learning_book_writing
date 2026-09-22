# 변분 자기 부호기

2013년 Kingma와 Welling이 내놓은 변분 자기 부호기(VAE)는 자기 부호기 얼거리를 확률 숨은 공간으로 넓힌다. 들임을 붙박이 부호에 옮기는 대신 부호기가 정규 분포의 평균과 흩어짐을 내놓고 다시 매개변수화 재주로 표본을 뽑는다. 그래서 새 표본을 만들어 내는 것과 숨은 공간에 대한 원칙 있는 베이즈 추론이 모두 된다.

## 1. 코드

```python
"""VAE - 변분 자기 부호기."""
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=20):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, input_dim), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

if __name__ == "__main__":
    model = VAE()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 1,082,680
```

## 2. 논의

다시 매개변수화 재주가 변분 자기 부호기의 고갱이 새것이다. $z \sim \mathcal{N}(\mu, \sigma^2)$을 곧바로 뽑는(미분할 수 없는) 대신 $\epsilon \sim \mathcal{N}(0, I)$에 대해 $z = \mu + \sigma \odot \epsilon$으로 쓴다. 그러면 무작위가 셈 그래프 밖으로 나가 뒤먹임 퍼뜨리기 때 기울기가 $\mu$과 $\sigma$을 지나 흐를 수 있다.

변분 자기 부호기의 손실은 다시 세우기 품질(두 값 엇갈린 엔트로피나 평균 제곱 어긋남)과, 숨은 분포를 표준 정규 사전 분포 쪽으로 끄는 KL 벌어짐 항을 아우른다. KL 항 $D_\text{KL}(q(z|x) \| p(z)) = -\frac{1}{2}\sum(1 + \log\sigma^2 - \mu^2 - \sigma^2)$이 부호기가 점 어림으로 무너지는 것을 막고 숨은 공간이 매끄럽고 이어지게 한다. 이 벌주기 덕분에 변분 자기 부호기가 만들어 내는 모델이 된다. 곧 $\mathcal{N}(0, I)$에서 뽑아 풀면 그럴듯한 새 그림이 나온다.

다시 세우기와 KL 벌어짐 사이의 팽팽함이 근본 맞바꿈을 만든다. 다시 세우기 손실을 가장 작게 하면 또렷하고 세밀한 내놓기가 나오지만 숨은 공간이 조각날 수 있다. KL을 가장 작게 하면 매끄럽고 고른 숨은 공간이 되지만 다시 세운 것이 흐릿할 수 있다. 이 두 항을 저울질하는 것이 변분 자기 부호기 설계의 핵심 과제이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
$d$차원 정규 분포에서 $\mathcal{N}(\mu, \sigma^2 I)$과 $\mathcal{N}(0, I)$ 사이 KL 벌어짐의 닫힌 꼴을 이끌어 내어라. 이끌어 낸 것이 부호에 쓰인 식과 맞는지 확인하라.

</div>

??? success "연습문제 1 풀이"
    대각 정규 분포에서는 KL 벌어짐이 차원에 걸친 합으로 쪼개진다:

    $$
    D_\text{KL} = -\frac{1}{2}\sum_{j=1}^{d}\left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)
    $$

    이는 $\mu_0=0, \Sigma_0=I$을 넣은 두루 쓰는 식 $D_\text{KL}(\mathcal{N}_1 \| \mathcal{N}_0) = \frac{1}{2}\left[\text{tr}(\Sigma_0^{-1}\Sigma_1) + (\mu_0-\mu_1)^\top\Sigma_0^{-1}(\mu_0-\mu_1) - d + \ln\frac{|\Sigma_0|}{|\Sigma_1|}\right]$에서 나온다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
MNIST에 변분 자기 부호기를 익히고 $z \sim \mathcal{N}(0, I)$을 풀어 아무 표본 100개를 만들어라. 10x10 격자로 보여라. 만든 숫자를 알아볼 수 있는가?

</div>

??? success "연습문제 2 풀이"
    ```python
    model.eval()
    with torch.no_grad():
        z = torch.randn(100, 20)
        samples = model.decode(z).view(-1, 28, 28)
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for i in range(100):
        axes[i//10, i%10].imshow(samples[i].numpy(), cmap='gray')
        axes[i//10, i%10].axis('off')
    plt.show()
    ```
    만든 숫자 대부분을 알아볼 수 있으나 실제 MNIST 그림에 견주면 조금 흐릿하게 보일 수 있다. 이 흐릿함은 평균 제곱 어긋남이나 두 값 엇갈린 엔트로피 다시 세우기 손실을 쓴 변분 자기 부호기의 특징이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
다시 매개변수화 재주가 왜 필요한지 설명하라. 그 재주 없이 $\mathcal{N}(\mu, \sigma^2)$에서 $z$을 곧바로 뽑으면 어떻게 되는가?

</div>

??? success "연습문제 3 풀이"
    곧바로 뽑으면 셈 그래프에 미분할 수 없는 연산이 생긴다. 곧 분포에서 뽑는 `sample` 연산은 분포 매개변수에 대한 기울기가 없다. 기울기가 $z$을 지나 $\mu$과 $\log\sigma^2$으로 흐르지 않으면 부호기 매개변수를 뒤먹임 퍼뜨리기로 새로 고칠 수 없다. 다시 매개변수화 재주는 뽑기를 $\mu$, $\sigma$, 바깥 잡음 $\epsilon$의 정해진 함수로 다시 세워 물길 전체를 미분할 수 있게 한다.


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
변분 자기 부호기가 자기 부호기와 다른 점을 세 줄로 적어라.

</div>

??? success "연습문제 4 풀이"
    | | 자기 부호기 | 변분 자기 부호기 |
    |---|---|---|
    | 부호기가 내놓는 것 | 코드 하나 | 분포 $(\mu, \log\sigma^2)$ |
    | 코드를 얻는 법 | $z = f(x)$ | $z = \mu + \sigma\varepsilon$로 뽑는다 |
    | 손실 | 다시 세우기 | 다시 세우기 + KL |

    코드로 보면 대여섯 줄 차이다. 그런데 이 차이가 압축기와 만들어 내는 모델을 가른다.
    $\mathcal{N}(0,I)$에서 뽑은 표본이 숫자로 보이는 비율이 0.2%에서 57.4%로 달라진다
    ([23.5절](../../ch24/limits/latent_sampling.md)).

    거꾸로 셋 가운데 하나라도 빼면 자기 부호기로 되돌아간다. KL을 빼면 $\sigma \to 0$이
    되고, 뽑기를 빼면 $\sigma$가 손실에 영향을 주지 않아 역시 $\sigma$가 무의미해진다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
변분 자기 부호기의 손실을 코드로 적고 각 줄이 무엇인지 밝혀라.

</div>

??? success "연습문제 5 풀이"
    ```python
    def loss_function(out, x, mu, logvar):
        # 다시 세우기: 화소에 대해 더하고 표본에 대해 평균
        rec = F.binary_cross_entropy(out, x, reduction='sum') / x.size(0)
        # KL: 닫힌 꼴. 차원에 대해 더하고 표본에 대해 평균
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum() / x.size(0)
        return rec + kl
    ```

    두 항이 [증거 하한](../theory/kl_term.md)의 두 조각이며, 임의로 섞은 것이 아니라
    $\log p(x)$의 하한을 유도했더니 나온 것이다.

    줄이는 방식을 눈여겨볼 것이다. **화소와 차원은 더하고 표본만 평균 낸다.** 화소를
    평균 내면 다시 세우기가 784배 작아져 실효 $\beta$가 784가 되므로, 전혀 다른 모델이
    된다.

    부호가 헷갈리기 쉽다. KL의 닫힌 꼴은

    $$\tfrac{1}{2}\sum_j (\mu_j^2 + \sigma_j^2 - 1 - \log\sigma_j^2)$$

    이고, 코드의 `-0.5 * (1 + logvar - mu² - exp(logvar))`는 이것과 같다. 괄호 안의
    부호를 모두 뒤집고 $-0.5$를 곱한 꼴이다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
이 모델이 만들어 내는 모델이라고 할 수 있는 근거는 무엇인가?

</div>

??? success "연습문제 6 풀이"
    **$p(x)$를 정의하기 때문이다.**

    $$p(x) = \int p(x \mid z)\, p(z)\, dz$$

    사전 분포 $p(z) = \mathcal{N}(0,I)$와 풀개 $p(x \mid z)$가 함께 자료 공간의 확률
    분포를 정한다. 그래서 뽑을 수 있고, (하한으로) 가능도를 물을 수 있다.

    자기 부호기는 이런 식을 쓸 수 없다. 부호기와 풀개가 있을 뿐 어떤 확률 모형도 세우지
    않으므로 "이 모델이 이 그림에 주는 확률"이라는 것이 정의되지 않는다
    ([23.5절 연습문제 9](../../ch24/limits/latent_sampling.md)).

    실용적인 결과가 이것이다. 변분 자기 부호기는 뽑기가 **정의상** 가능하다.
    $z \sim \mathcal{N}(0,I)$을 뽑아 풀개에 넣으면 그것이 모델의 표본이다. 자기 부호기에서
    같은 일을 하는 것은 모델이 정의한 무엇이 아니라 그저 해 보는 것이며, 그래서 얼룩이
    나온다.

    다만 "정의된다"와 "잘된다"는 다르다. 57.4%라는 것은 42.6%가 실패한다는 뜻이다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff hard" title="어려움"></span>
다시 세우기 항을 확률로 읽으면 무엇인가? BCE를 쓰는 것은 어떤 가정인가?

</div>

??? success "연습문제 7 풀이"
    증거 하한의 첫 항은 $\mathbb{E}_q[\log p(x \mid z)]$이므로, 다시 세우기 손실은
    **음의 로그 가능도**다. 손실의 모양이 곧 $p(x \mid z)$에 대한 가정이다.

    | 손실 | 가정하는 $p(x \mid z)$ |
    |---|---|
    | BCE | 화소마다 독립인 베르누이 |
    | MSE | 화소마다 독립인 가우시안(분산 고정) |
    | L1 | 화소마다 독립인 라플라스 |

    BCE를 쓰는 것은 화소가 0 또는 1이라고 보는 것이다. MNIST 화소는 실수 $[0,1]$이므로
    엄밀히는 맞지 않고, 이를 연속 베르누이로 다듬는 논의가 따로 있다. 실용적으로는
    잘 듣는다([24장 손실 함수](../../ch24/ae/loss_functions.md)에서 BCE가 MSE보다
    나았다).

    **화소마다 독립**이라는 가정이 훨씬 심각한 문제다. 실제 그림에서 이웃 화소는 강하게
    얽혀 있는데 이 모델은 코드가 주어지면 독립이라고 본다. 그 결과가 유명한 **흐릿함**이다.
    모델이 여러 그럴듯한 그림 사이에서 고르지 못하고 평균을 내놓는다.

    이 장의 측정에서 그 흐릿함이 수로 보인다. 표본의 대비(화소 표준편차)가 0.2655인데
    실제 자료는 0.2984다([표본 만들기](../training/generate_samples.md)). 덜 날카롭다.

    이 가정을 깨는 것이 뒤 장들의 큰 줄기다. 자기 회귀 풀개는 화소 사이 의존을 직접
    모형하고, [퍼짐 모델](../../ch29/index.md)은 아예 다른 길로 간다.

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
숨은 차원을 몇으로 두어야 하는가?

</div>

??? success "연습문제 8 풀이"
    잰 값을 보면 MNIST에서는 16 근처가 좋다.

    | 숨은 차원 | 2 | 8 | 16 | 32 | 64 |
    |---|---|---|---|---|---|
    | 다시 세우기 | 136.57 | 89.19 | **81.22** | 82.06 | 83.33 |
    | 살아 있는 차원 | 2 | 8 | 12 | 17 | 11 |

    16을 넘으면 좋아지지 않고 오히려 조금 나빠진다. 그리고 늘려 준 차원을 모델이 쓰지도
    않는다. 64개를 주어도 11개만 산다.

    그래서 변분 자기 부호기에서는 이 물음이 자기 부호기에서보다 **덜 예민하다.**
    KL 항이 필요 없는 차원을 꺼 주므로 넉넉히 잡아도 크게 낭비되지 않는다. 자기 부호기는
    병목이 곧 코드 크기라서 정확히 맞추어야 했다.

    권할 만한 방법은 **넉넉히 주고 죽는 차원 수를 읽는 것**이다. 32를 주었을 때 17개가
    살았으니 자료가 쓸 수 있는 차원이 그 근처라는 뜻이다.

    다만 죽는 것이 늘 자료의 뜻은 아니다. $\beta$가 크면 쓸 만한 차원도 죽으므로
    ([자유 비트](../training/free_bits.md)) 두 원인을 가려야 한다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
익히기 초반에 KL이 먼저 줄고 다시 세우기가 늦게 줄어드는 일이 흔하다. 왜인가?

</div>

??? success "연습문제 9 풀이"
    KL을 줄이는 쉬운 답이 있기 때문이다. $\mu = 0$, $\sigma = 1$로 두면 KL이 정확히
    0이 되고, 이것은 부호기가 입력을 볼 필요조차 없는 답이다.

    반면 다시 세우기를 줄이려면 실제로 쓸모 있는 것을 배워야 하므로 오래 걸린다.

    그래서 초반에 최적화기가 값싼 쪽을 먼저 챙긴다. 위험한 것은 그 과정에서 차원이
    **죽어 버리는** 것이다. 한 번 죽은 차원은 스스로 살아나지 못한다
    ([자유 비트 연습문제 8](../training/free_bits.md)).

    두 가지 처방이 여기서 나온다.

    - **KL 달구기**: $\beta$를 0에서 1로 올려 초반에 그 압력을 없앤다
    - **자유 비트**: 차원마다 바닥을 깔아 죽는 것을 막는다

    익히기 곡선을 볼 때 두 항을 **따로** 그려야 이 일이 보인다. 합만 그리면 KL이
    떨어지면서 총손실이 내려가는 모습이 순조로워 보인다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
익힌 모델로 할 수 있는 일을 꼽아 보라.

</div>

??? success "연습문제 10 풀이"
    자기 부호기가 하던 일에 뽑기가 더해진다.

    | 하는 일 | 쓰는 부분 |
    |---|---|
    | 다시 세우기 | 부호기 + 풀개 |
    | 차원 줄이기, 특징 뽑기 | 부호기($\mu$) |
    | **새 표본 만들기** | 풀개만. $z \sim \mathcal{N}(0,I)$ |
    | 코드 사이 끼움 | 부호기 두 번 + 풀개 |
    | 이상 탐지 | 증거 하한이 낮은 것을 찾는다 |

    셋째 줄이 이 장의 새로운 것이다. **부호기를 전혀 쓰지 않는다**는 점을 눈여겨볼
    만하다. 익히고 나면 만들어 내기에는 풀개만 필요하다.

    다섯째 줄도 변분 자기 부호기라야 되는 일이다. 자기 부호기에서도 다시 세우기 오차로
    비슷한 것을 하지만, 여기서는 증거 하한이라는 확률적 근거가 있는 값을 쓸 수 있다.

    `encode`, `decode`, `forward`를 나누어 두는 설계가 이 쓰임의 다양함 때문에 값지다
    ([24장 모듈](../../ch24/architecture/autoencoder.md)).

## 정리하며

**다룬 것** — 변분 자기 부호기

다시 매개변수화 재주가 변분 자기 부호기의 고갱이 새것이다.

고갱이 갈래는 `VAE`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
