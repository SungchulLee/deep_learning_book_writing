# 베타 변분 자기 부호기

베타 변분 자기 부호기는 변분 자기 부호기 손실의 KL 벌어짐 항에 무게를 주는 웃매개변수 $\beta$을 들여와 다시 세우기 품질과 숨은 공간 얽힘 풀기의 맞바꿈을 다스린다. $\beta$이 클수록 숨은 차원마다 서로 얽히지 않고 풀이할 수 있는 흔들림 요인(예컨대 돌림, 굵기, 모양새)을 담도록 이끈다. 이 단원은 온전히 이어진 판과 누비기 판을 모두 짜고, 차원마다 무엇을 배웠는지 그려 보는 숨은 훑기 방법을 곁들인다.

## 1. 코드

```python
"""얽힘 풀린 나타냄을 배우는 베타 변분 자기 부호기."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BetaVAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=32, beta=4.0):
        super().__init__()
        self.input_dim, self.latent_dim, self.beta = input_dim, latent_dim, beta
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def loss_function(self, reconstruction, x, mu, logvar):
        BCE = F.binary_cross_entropy(reconstruction, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + self.beta * KLD, BCE, KLD

    def traverse_latent_dimension(self, dim_idx, num_steps=10, range_limit=3.0, device='cpu'):
        z = torch.zeros(num_steps, self.latent_dim).to(device)
        z[:, dim_idx] = torch.linspace(-range_limit, range_limit, num_steps).to(device)
        return self.decoder(z)

if __name__ == '__main__':
    model = BetaVAE(input_dim=784, latent_dim=10, beta=4.0)
    # BCE 손실은 목표가 [0,1]이어야 한다. randn은 음수를 내므로 rand를 쓴다
    x = torch.rand(32, 784)
    reconstruction, mu, logvar = model(x)
    loss, bce, kld = model.loss_function(reconstruction, x, mu, logvar)
    print(f"Loss: {loss.item():.4f}, Recon: {bce.item():.4f}, KL: {kld.item():.4f}")
```

**출력:**

```
Loss: 17434.7734, Recon: 17430.4121, KL: 1.0901
```

## 2. 논의

매개변수 $\beta$은 여느 자기 부호기($\beta = 0$, KL 벌주기 없음)와 지나치게 옭아맨 모델($\beta \gg 1$, 숨은 차원이 모두 사전 분포로 무너짐) 사이를 잇는다. $\beta = 1$이면 여느 변분 자기 부호기가 된다. 베타 변분 자기 부호기 논문의 핵심 눈썰미는 $\beta > 1$(흔히 4~10)이 얽힘 풀기를 이끈다는 것이다. 곧 숨은 차원마다 서로 얽히지 않은 흔들림 요인 하나를 담기 쉬워진다.

얽힘 풀기는 숨은 훑기로 가늠한다. 익힌 모델에서 한 차원만 빼고 숨은 차원을 모두 0으로 고정한 뒤 그 차원을 $-3$에서 $+3$까지 훑어 나온 숨은 벡터를 푼다. 그 차원의 얽힘이 풀렸다면 푼 그림이 다른 속성은 그대로인 채 한 속성만(예컨대 돌림만, 또는 굵기만) 바뀌어야 한다. 실전에서 완벽한 얽힘 풀기는 드물지만 베타 변분 자기 부호기는 여느 변분 자기 부호기보다 훨씬 풀이하기 쉬운 숨은 공간을 낸다.

다시 세우기와 얽힘 풀기의 맞바꿈은 근본이다. $\beta$을 키우면 얽힘 풀기는 나아지지만 KL 벌주기가 세져 부호기가 숨은 공간의 담이를 다 쓰지 못해 다시 세우기 품질이 나빠진다. 가장 좋은 $\beta$을 찾으려면 실험해야 하며 자료 묶음과 뒤따르는 쓰임새에 달렸다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
MNIST에서 $\beta \in \{0.1, 1, 4, 10, 50\}$으로 베타 변분 자기 부호기를 익혀라. 저마다 앞선 5개 차원의 숨은 훑기를 만들고 어느 $\beta$이 가장 얽힘이 풀린 나타냄을 내는지 눈으로 가늠하라.

</div>

??? success "연습문제 1 풀이"
    $\beta = 0.1$이면 다시 세운 것이 또렷하지만 숨은 훑기가 얽힌 바뀜을 보인다(속성 여럿이 한꺼번에 바뀐다). $\beta = 4$이면 차원마다 획의 굵기나 기울기 같은 뚜렷한 속성을 담기 시작한다. $\beta = 50$이면 많은 차원이 무너지고(아무 흔들림도 내지 않고) 다시 세운 것이 아주 흐릿하다. MNIST에서 알맞은 자리는 흔히 $\beta = 4$~$10$이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
$\beta = 1$일 때 베타 변분 자기 부호기의 손실이 자료의 로그 가능도에 대한 하한(증거 하한)과 같음을 밝혀라.

</div>

??? success "연습문제 2 풀이"
    증거 하한은 $\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_\text{KL}(q(z|x) \| p(z))$이다. 첫 항은 다시 세우기 손실에 음의 부호를 붙인 것(두 값 엇갈린 엔트로피나 평균 제곱 어긋남)이고 둘째는 KL 벌어짐이다. $\beta = 1$이면 베타 변분 자기 부호기의 손실이 $-\mathcal{L}$, 곧 음의 증거 하한이다. 증거 하한을 가장 크게 하는 것은 변분 자기 부호기 손실을 가장 작게 하는 것과 같다. $\beta \neq 1$이면 그 손실은 더는 옳은 증거 하한이 아니지만, 모델을 더 고른 숨은 공간 쪽으로 기울이는 쓸모 있는 익히기 목표가 된다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
얽힘 풀기를 값으로 재는 잣대를 짜라. $\beta = 1$과 $\beta = 4$으로 익힌 모델에 그것을 셈해 $\beta$이 클수록 점수가 좋은지 확인하라.

</div>

??? success "연습문제 3 풀이"
    단순한 얽힘 풀기 잣대: 숨은 차원 $j$마다 알려진 참 요인 $k$만 바꿔 자료를 만들고 부호화한 뒤 어느 숨은 차원의 흩어짐이 가장 큰지 잰다. 차원 $j$이 요인 $k$과 한결같이 일대일로 맞닿으면 얽힘이 풀린 나타냄이다. 점수는 딱 한 숨은 차원이 옳게 잡은 요인의 몫이다. dSprites처럼 참 요인이 알려진 자료 묶음에서 $\beta = 4$은 흔히 $\beta = 1$보다 20~40% 높은 점수를 낸다.


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
$\beta$를 0.25에서 8까지 올리면 다시 세우기와 KL이 각각 어떻게 되는가?

</div>

??? success "연습문제 4 풀이"
    숨은 차원 16, 20 에포크로 재면 이렇다.

    | $\beta$ | 0.25 | 0.5 | 1 | 2 | 4 | 8 |
    |---|---|---|---|---|---|---|
    | 다시 세우기 | **70.60** | 74.74 | 81.22 | 95.71 | 108.85 | 134.92 |
    | KL | 36.20 | 27.92 | 20.32 | 12.57 | 8.66 | **4.63** |

    방향이 예상대로다. $\beta$가 커지면 KL이 줄고 다시 세우기가 나빠진다. 손실에서
    KL에 더 큰 무게를 주었으니 최적화기가 그쪽을 더 줄인다.

    눈금을 눈여겨볼 만하다. $\beta$를 32배 올리는 동안 KL은 8분의 1이 되고 다시 세우기는
    두 배가 된다. 곧 **KL을 줄이는 값이 점점 비싸진다.**

    그런데 이 표만 보면 $\beta$를 어디에 두어야 할지 알 수 없다. 둘 다 손실의 조각일
    뿐이어서 "어느 쪽이 더 중한가"를 말해 주지 않기 때문이다. 다음 문제에서 제3의
    잣대를 본다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
표본의 품질이 $\beta$에 따라 어떻게 변하는가? $\beta = 1$이 가장 좋은가?

</div>

??? success "연습문제 5 풀이"
    아니다. 분류기에게 판정시키면 이렇다.

    | $\beta$ | 0.25 | 0.5 | 1 | 2 | **4** | 8 |
    |---|---|---|---|---|---|---|
    | 확신도 0.9 넘는 비율 | 40.3% | 48.4% | 57.4% | 65.2% | **66.6%** | 55.9% |

    **$\beta = 4$ 근처가 가장 좋고 $\beta = 1$은 한참 아래다.** 그리고 봉우리를 넘으면
    다시 나빠진다.

    두 방향에서 까닭을 볼 수 있다.

    $\beta$가 작을 때는 KL의 압력이 약해 코드가 사전 분포와 잘 맞지 않는다. 곧 구멍이
    많고, 뽑은 $z$가 모델이 모르는 자리에 떨어진다. 극단이
    [자기 부호기의 0.2%](../../ch24/limits/latent_sampling.md)다.

    $\beta$가 클 때는 코드가 사전 분포에 잘 맞지만 나를 정보가 없다. $\beta=8$에서 살아
    있는 차원이 5개뿐이니, 뽑기는 쉬워졌으나 풀 것이 남지 않았다.

    이것이 이 절의 핵심이다. **$\beta = 1$은 증거 하한에서 나온 값이지 표본 품질에서
    나온 값이 아니다.** 만들어 내기가 목적이면 $\beta$를 쓸어 보고 골라야 한다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
$\beta \ne 1$로 두면 증거 하한을 최대화하는 것이 아니다. 그래도 괜찮은가?

</div>

??? success "연습문제 6 풀이"
    목적이 무엇이냐에 달렸다.

    $\beta \ne 1$인 손실은 어떤 $\log p(x)$의 하한도 아니다. 그러므로 다음 두 가지를
    포기한다.

    - 다른 모델과 ELBO로 견주는 일
    - $\log p(x)$에 대한 보장

    그런데 앞 문제에서 본 대로 **$\beta = 1$이 표본 품질에서 가장 좋지도 않다.** 우리가
    정말 원하는 것이 좋은 표본이라면, 그것을 주지 못하는 이론적 정당성을 지킬 까닭이
    약하다.

    그래서 실무의 태도는 이렇다.

    | 목적 | $\beta$ |
    |---|---|
    | 밀도 추정, 가능도 보고 | 1. 다른 값을 쓰면 견줄 수 없다 |
    | 표본 만들기 | 쓸어 보고 고른다. 여기서는 4 근처 |
    | 풀 수 있는 나타냄 | 크게. 그것이 $\beta$-VAE 논문의 목적이다 |

    보고할 때 정직하면 된다. $\beta$를 밝히고, 가능도를 말할 때는 $\beta=1$로 다시
    재어 적는다. [자유 비트](../training/free_bits.md)에서도 같은 태도를 취했다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff hard" title="어려움"></span>
$\beta$를 키우면 풀 수 있는(disentangled) 나타냄이 얻어진다는 주장의 근거는 무엇인가? 이 장의 측정이 그것을 뒷받침하는가?

</div>

??? success "연습문제 7 풀이"
    근거는 [사전 분포 연습문제 3](prior.md)의 분해에 있다.

    $$\mathbb{E}_{p(x)}[\text{KL}] = I(x;z) + D_{\mathrm{KL}}(q(z) \,\|\, p(z))$$

    뒤쪽 항을 줄이면 $q(z)$가 $p(z) = \mathcal{N}(0,I)$에 가까워지고, $\mathcal{N}(0,I)$은
    **차원끼리 독립**이다. 곧 코드의 차원들이 서로 독립이 되도록 밀린다. 독립인 축이
    자료의 독립인 변화 요인과 맞아떨어지면 그것이 풀린 나타냄이다.

    이 장의 측정은 그 주장을 **직접 뒷받침하지 않는다.** 잰 것이 다시 세우기, KL,
    표본 품질뿐이고 풀림을 재지 않았기 때문이다.

    오히려 조심할 근거가 보인다. $\beta$를 키울 때 실제로 일어난 일은 **차원이 죽는
    것**이었다.

    | $\beta$ | 1 | 4 |
    |---|---|---|
    | 죽은 차원 | 4/16 | 9/16 |

    독립성이 늘어난 것이 "축마다 다른 요인을 담아서"인지 "축을 꺼서"인지 이 수치로는
    가릴 수 없다. 끈 축은 확실히 독립이지만 아무것도 나르지 않는다.

    그래서 풀림을 주장하려면 풀림을 재는 잣대가 따로 필요하다. 그리고 그 잣대들이
    대부분 **변화 요인의 참값을 아는 인공 자료**를 요구한다. MNIST에는 그 참값이 없으니
    이 장의 설정으로는 답할 수 없는 물음이다.

    측정하지 않은 것을 주장하지 않는 것이 옳다.

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
$\beta$를 익히는 동안 바꾸는 것과 고정하는 것은 어떻게 다른가?

</div>

??? success "연습문제 8 풀이"
    $\beta$를 0에서 목표값으로 서서히 올리는 것을 **KL 달구기**라 한다.

    까닭은 초기의 실패를 피하는 것이다. 부호기가 아직 쓸모 있는 코드를 못 만든 때에
    KL이 세게 누르면, 최적화기가 쉬운 답을 고른다. 차원을 꺼 버리는 것이다. 그리고
    한 번 끈 차원은 스스로 살아나지 못한다([자유 비트](../training/free_bits.md)).

    그래서 익히기를 두 시기로 나누는 셈이 된다.

    | 시기 | $\beta$ | 무엇을 배우는가 |
    |---|---|---|
    | 초기 | 작다 | 되돌리기. 코드가 쓸모를 갖춘다 |
    | 뒤 | 목표값 | 코드를 사전 분포에 맞춘다 |

    고정하는 편이 단순하고, 이 장의 측정은 모두 고정으로 했다. 그래도 4개가 죽었으니
    ($\beta=1$) 달구기가 도움이 될 여지가 있다.

    달구기의 끝값은 여전히 골라야 한다는 점을 잊지 말 것이다. 달구기는 **가는 길**을
    고르는 것이고 $\beta$ 자체는 **갈 곳**을 고르는 것이다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
$\beta$와 숨은 차원 수 가운데 어느 것을 먼저 조절해야 하는가?

</div>

??? success "연습문제 9 풀이"
    **숨은 차원을 넉넉히 잡아 두고 $\beta$를 조절하는 편**이 낫다.

    까닭은 [사전 분포 연습문제 6](prior.md)에서 본 성질이다. 변분 자기 부호기는 필요
    없는 차원을 스스로 끄므로, 차원을 넉넉히 주어도 낭비가 크지 않다. 64를 주어도
    11개만 쓴다.

    반대로 $\beta$는 스스로 정해지지 않는다. 모델이 고를 수 있는 값이 아니라 우리가
    무엇을 중히 여기는지를 나타내는 값이기 때문이다.

    다만 둘이 서로 얽혀 있다는 점은 알아 두어야 한다. $\beta$가 크면 죽는 차원이
    늘어나므로, 실효 차원은 둘이 함께 정한다. 16차원에 $\beta=8$이면 실제로는 5차원
    모델이다.

    실용적인 순서는 이렇다. 차원을 32쯤으로 넉넉히 두고 $\beta$를 쓸어 목적에 맞는
    값을 고른 뒤, 살아 있는 차원 수를 보고 필요하면 차원을 줄인다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
$\beta$를 조절하는 것과 다시 세우기 손실의 눈금을 바꾸는 것이 같은 일인가?

</div>

??? success "연습문제 10 풀이"
    같은 일이다. 그리고 이것이 실무에서 헷갈림의 큰 원인이다.

    손실이

    $$\mathcal{L} = \text{Rec} + \beta \cdot \text{KL}$$

    일 때, 양변을 $\beta$로 나누면

    $$\frac{1}{\beta}\text{Rec} + \text{KL}$$

    이 되어 최적화에는 같은 문제다. 곧 **$\beta$를 키우는 것은 다시 세우기의 무게를
    줄이는 것과 같다.**

    그러므로 다음이 모두 같은 방향의 손잡이다.

    - $\beta$를 키운다
    - 다시 세우기를 화소에 대해 평균 낸다 (784분의 1이 된다)
    - 가우시안 가능도에서 $\sigma^2$을 크게 잡는다

    세 번째가 특히 눈여겨볼 만하다. MSE를 쓰면서 $\mathcal{L} = \|x-\hat x\|^2/(2\sigma^2) + \text{KL}$로
    적으면 $\sigma^2$이 곧 $\beta$ 노릇을 한다. 흔히 $\sigma^2$을 1로 두는데, 그것은
    임의의 선택이며 암묵적으로 $\beta$를 정하는 일이다.

    결론은 [KL 항 연습문제 10](../theory/kl_term.md)과 같다. **코드를 읽을 때 $\beta$
    값만 보지 말고 두 항을 어떻게 줄였는지 함께 보아야 한다.** 논문 둘의 $\beta=1$이
    784배 다른 뜻일 수 있다.

## 정리하며

**다룬 것** — 베타 변분 자기 부호기

매개변수 $\beta$은 여느 자기 부호기($\beta = 0$, KL 벌주기 없음)와 지나치게 옭아맨 모델($\beta \gg 1$, 숨은 차원이 모두 사전 분포로 무너짐) 사이를 잇는다.

고갱이 갈래는 `BetaVAE`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
