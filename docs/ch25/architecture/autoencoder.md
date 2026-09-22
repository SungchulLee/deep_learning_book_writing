# 자기 부호기

이 단원은 차원 줄이기와 다시 세우기를 위한 말끔하고 다시 쓸 수 있는 `SimpleAutoencoder` 갈래를 정한다. 얼개는 정류 선형 깨어남과 에스자 내놓기를 갖춘 세 층짜리 온전히 이어진 부호기와 풀개 그물을 쓴다. 잡음 없애는 자기 부호기나 변분 자기 부호기 같은 더 복잡한 얼개의 바탕 벽돌 노릇을 하며 `encode`, `decode`, `loss_function` 방법을 준다.

## 1. 코드

```python
"""단순한 자기 부호기 - 정해진 부호기-풀개 기본 얼개."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))

    def loss_function(self, reconstruction, x):
        return F.mse_loss(reconstruction, x, reduction='sum')

if __name__ == '__main__':
    model = SimpleAutoencoder(input_dim=784, latent_dim=32)
    x = torch.randn(32, 784)
    reconstruction = model(x)
    loss = model.loss_function(reconstruction, x)
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Loss: {loss.item():.4f}")
```

**출력:**

```
Input shape: torch.Size([32, 784])
Reconstruction shape: torch.Size([32, 784])
Loss: 31782.0742
```

## 2. 논의

`SimpleAutoencoder`은 부호기와 풀개의 층 너비가 맞는 대칭 설계를 따른다(input_dim에서 hidden_dim, hidden_dim, latent_dim으로, 그리고 그 반대). 이 대칭이 꼭 필요하지는 않고 대칭이 아닌 얼개도 잘 되지만, 모델의 담이를 따져 보기 쉬워지고 풀개가 적어도 부호기만큼의 표현력을 갖추게 한다.

평균 제곱 어긋남 손실에서 붙박이인 `'mean'` 대신 `reduction='sum'`을 쓰는 것은 기울기의 잣수에 영향을 주는 일부러 한 고름이다. 합으로 줄이면 손실이 배치 크기와 들임 차원에 비례하므로 배움 빠르기가 사실상 그 값에 따라 잣수 맞춰진다. 변분 자기 부호기 짜기 여럿이 KL 벌어짐 항(이 또한 흔히 차원에 걸쳐 더한다)과 결을 맞추려 합으로 줄인다.

이 최소 짜기에 배치 고르게 맞추기, 떨구기, 무게 잦아듦이 없다는 것은 다스려진 바탕으로 쓸 수 있다는 뜻이다. 변분 자기 부호기나 잡음 없애는 자기 부호기로 넓힐 때는 흔히 벌주기와 고르게 맞추기 층을 더하지만, 말끔한 바탕 갈래가 있으면 고침마다의 효과를 따로 떼어 보기 쉽다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
갈래에 `get_latent_dim` 속성과 `count_parameters` 방법을 더하라. 그것으로 숨은 차원 8, 32, 128인 모델을 견주어라.

</div>

??? success "연습문제 1 풀이"
    ```python
    @property
    def get_latent_dim(self):
        return self.latent_dim

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    for dim in [8, 32, 128]:
        m = SimpleAutoencoder(latent_dim=dim)
        print(f"latent_dim={dim}: {m.count_parameters():,} parameters")
    ```
    병목에 이어진 선형 층이 비례해 커지므로 매개변수 수가 숨은 차원에 따라 선형으로 는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
평균 제곱 어긋남 손실을 두 값 엇갈린 엔트로피 손실로 갈음하라. 두 값 엇갈린 엔트로피가 옳으려면 들임 자료가 어떤 제약을 채워야 하며, 에스자 내놓기 깨어남이 왜 그것을 보장하는가?

</div>

??? success "연습문제 2 풀이"
    두 값 엇갈린 엔트로피는 어림과 목표가 모두 $[0, 1]$에 있어야 하며 이를 베르누이 확률 변수의 확률로 본다. 에스자 깨어남이 풀개의 내놓기를 $(0, 1)$에 옮겨 이 제약을 채운다. 들임 자료도 $[0, 1]$으로 골라야 한다(MNIST 그림은 `ToTensor()` 뒤 그렇게 된다). 이 범위 밖 자료에 두 값 엇갈린 엔트로피를 쓰면 음의 손실이나 NaN 기울기가 나온다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
들임 벡터 둘을 받아 부호화하고 그 숨은 부호를 고르게 벌어진 10개 점에서 선형으로 사이 끼움한 뒤 푸는 `interpolate` 방법을 짜라. MNIST 숫자 둘로 시험하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    def interpolate(self, x1, x2, n_steps=10):
        z1 = self.encode(x1.unsqueeze(0))
        z2 = self.encode(x2.unsqueeze(0))
        alphas = torch.linspace(0, 1, n_steps).unsqueeze(1)
        z_interp = (1 - alphas) * z1 + alphas * z2
        return self.decode(z_interp)
    ```
    숫자 사이 사이 끼움이 매끄러우면 숨은 공간이 뜻 있고 이어진 나타냄을 배웠다는 뜻이다. 갑작스러운 옮아감은 숨은 공간에 틈이나 끊김이 있음을 뜻한다.


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
`SimpleAutoencoder`의 매개변수를 층별로 세어 보라. 부호기와 풀개 가운데 어느 쪽이 더 무거운가?

</div>

??? success "연습문제 4 풀이"
    $784 \to 512 \to 256 \to 16$과 그 역이라면 이렇다.

    | | 가중치 | 편향 | 합 |
    |---|---|---|---|
    | 부호기 | $784{\cdot}512 + 512{\cdot}256 + 256{\cdot}16$ | $512{+}256{+}16$ | 537,360 |
    | 풀개 | $16{\cdot}256 + 256{\cdot}512 + 512{\cdot}784$ | $256{+}512{+}784$ | 538,128 |
    | 합 | | | **1,075,488** |

    대칭 설계이므로 거의 같고, 풀개가 768개 더 많다. 편향의 개수가 다르기 때문이다.
    부호기는 512, 256, 16개의 편향을 갖고 풀개는 256, 512, 784개를 갖는다.

    어느 쪽이든 **$784 \times 512$ 층이 전체의 75%**를 차지한다. 입력에 붙은 층이
    가장 무겁다는 것은 [3.4절 연습문제 2](../../ch03/mnist/04_cnn.md)에서 본 것과 같은
    이야기다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
`encode`, `decode`, `forward`를 따로 두는 설계가 왜 편리한가?

</div>

??? success "연습문제 5 풀이"
    익히는 동안에는 `forward` 하나면 된다. 셋으로 나누는 값어치는 **익히고 난 뒤**에
    드러난다.

    | 하는 일 | 쓰는 것 |
    |---|---|
    | 차원 줄이기, 특징 뽑기 | `encode`만 |
    | 코드에서 그림 만들기 | `decode`만 |
    | 다시 세우기 오차 재기 | `forward` |
    | 코드 사이 끼움 | `encode` 두 번 + `decode` 여러 번 |

    [뽑기 실험](../limits/latent_sampling.md)이 좋은 예다. 거기서는 부호기를 전혀 쓰지
    않고 $z \sim \mathcal{N}(0,I)$를 `decode`에만 넣는다. 둘이 붙어 있으면 그런 실험을
    할 수 없다.

    변분 자기 부호기로 넘어가면 이 구분이 더 중요해진다. `encode`가 값 하나가 아니라
    $(\mu, \log\sigma^2)$ 짝을 내놓게 바뀌므로, 경계가 또렷해야 무엇이 달라졌는지
    보인다([26장](../../ch26/index.md)).

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
`loss_function`을 모델 안에 두는 것과 밖에 두는 것 가운데 무엇이 나은가?

</div>

??? success "연습문제 6 풀이"
    보통 자기 부호기라면 밖에 두어도 된다. 손실이 출력과 표적만 있으면 셈해지기
    때문이다.

    안에 두는 편이 나은 경우는 **손실이 모델 내부를 알아야 할 때**다.

    - 변분 자기 부호기: KL 항이 $\mu$와 $\log\sigma^2$를 필요로 한다
    - 성긴 자기 부호기: 벌점이 은닉 깨어남을 필요로 한다
    - 오그리는 자기 부호기: 벌점이 야코비를 필요로 한다

    이들은 모두 `forward`가 돌려주는 출력만으로는 셈할 수 없다. 그래서 모델이 손실에
    필요한 것들을 함께 돌려주거나, 아예 손실 계산을 품는다.

    이 장의 모듈이 `loss_function`을 품은 것은 뒤따르는 변형들과 인터페이스를 맞추기
    위해서다. 나중에 VAE로 갈아 끼울 때 익히기 코드를 고치지 않아도 된다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff easy" title="쉬움"></span>
`SimpleAutoencoder`의 풀개 끝에 시그모이드가 붙어 있다. 왜인가? 떼면 무슨 일이 생기는가?

</div>

??? success "연습문제 7 풀이"
    MNIST 화소가 $[0,1]$이므로 출력도 그 범위에 있어야 하고, 시그모이드가 그것을
    보장한다.

    떼면 출력이 실수 전체를 돌아다닌다. MSE로 익히면 그래도 돌아가지만 화소값이 음수나
    1보다 큰 값이 나올 수 있어 그림으로 보려면 잘라 내야 한다. **BCE로 익히면 아예
    오류가 난다.** BCE는 입력이 $[0,1]$이기를 요구하기 때문이다.

    거꾸로 시그모이드가 방해가 되는 경우도 있다. 표준화한 자료처럼 값이 음수를 갖는
    자료라면 시그모이드가 표현할 수 없다. 그때는 떼고 MSE를 쓴다
    ([손실 함수 연습문제 1](../ae/loss_functions.md)).

    그리고 [주다양체 연습문제 5](04_ae_principal_manifold.md)에서 본 함정이 있다.
    "선형" 자기 부호기를 만든다면서 시그모이드를 남겨 두면 그것은 선형이 아니다.

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
이 모듈을 잡음 없애는 자기 부호기로 바꾸려면 무엇을 고쳐야 하는가?

</div>

??? success "연습문제 8 풀이"
    **모델은 한 줄도 고치지 않는다.** 익히기 반복문에서 입력만 더럽히면 된다.

    ```python
    noisy = (x + sigma * torch.randn_like(x)).clamp(0, 1)
    out = model(noisy)          # 입력은 더럽힌 것
    loss = criterion(out, x)    # 표적은 깨끗한 것
    ```

    이 비대칭이 잡음 없애는 자기 부호기의 전부다
    ([잡음 없애는 자기 부호기](03_ae_denoising.md)).

    같은 모듈로 이렇게 여러 변형을 만들 수 있다는 점이 이 설계의 값어치다. 성김은
    손실에 항을 더하면 되고, 오그림은 야코비 벌점을 더하면 된다. 바뀌는 것은 익히기
    절차이지 얼개가 아니다.

    바뀌는 폭이 얼개까지 미치는 것이 변분 자기 부호기다. 부호기가 내놓는 것의 개수가
    달라지므로 모듈 자체를 손봐야 한다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
이 모듈을 그대로 두고 변분 자기 부호기를 만들려면 무엇을 더해야 하는지 코드로 적어라.

</div>

??? success "연습문제 9 풀이"
    부호기의 마지막 층을 $2k$차원으로 늘려 $\mu$와 $\log\sigma^2$을 함께 내놓게 하고,
    다시 뽑기와 KL 항을 더한다.

    ```python
    class SimpleVAE(SimpleAutoencoder):
        def __init__(self, input_dim=784, hidden_dim=256, latent_dim=32):
            super().__init__(input_dim, hidden_dim, latent_dim * 2)   # 1. 두 배로

        def encode(self, x):
            h = super().encode(x)
            return h.chunk(2, dim=1)                                  # mu, logvar

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()      # 2. 다시 뽑기
            return self.decode(z), mu, logvar

        def loss_function(self, out, x, mu, logvar):
            rec = F.binary_cross_entropy(out, x, reduction='sum') / x.size(0)
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum() / x.size(0)
            return rec + kl                                           # 3. KL 항
        ```

    고친 곳이 셋이고 줄 수로는 대여섯 줄이다. 그런데 이 몇 줄이
    [뽑을 수 있는지](../limits/latent_sampling.md)를 0.2%에서 57.4%로 바꾼다.

    바로 그 점이 두 장을 따로 두는 까닭이기도 하다. 코드의 차이는 사소한데 모델의
    성격이 달라진다. 하나는 압축기이고 하나는 만들어 내는 모델이다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
`SimpleAutoencoder`에 입력 검증을 넣는다면 무엇을 확인해야 하는가?

</div>

??? success "연습문제 10 풀이"
    자기 부호기에서 특히 조용히 잘못되기 쉬운 것이 셋이다.

    | 확인할 것 | 어기면 |
    |---|---|
    | 모양이 `(B, input_dim)`인가 | 브로드캐스팅으로 엉뚱한 손실이 셈해질 수 있다 |
    | 값이 $[0,1]$인가 | BCE가 오류를 내거나, 시그모이드 출력과 표적의 범위가 어긋난다 |
    | 자료형이 `float32`인가 | `uint8` 그대로 넣으면 값이 0~255라 손실이 폭발한다 |

    두 번째와 세 번째가 실제로 자주 겪는 함정이다. `datasets.MNIST`의 `.data`는
    `uint8`이므로 `/255.0`을 빠뜨리면 손실이 수백에서 시작하고 그림이 새까맣게 나온다.
    오류가 나지 않고 그저 나쁜 결과가 나오므로 알아채기 어렵다.

    ```python
    assert x.dim() == 2 and x.size(1) == self.input_dim, f"모양이 어긋난다: {x.shape}"
    assert x.dtype == torch.float32, f"float32여야 한다: {x.dtype}"
    assert 0.0 <= x.min() and x.max() <= 1.0, f"[0,1]을 벗어난다: [{x.min()}, {x.max()}]"
    ```

## 정리하며

**다룬 것** — 자기 부호기

`SimpleAutoencoder`은 부호기와 풀개의 층 너비가 맞는 대칭 설계를 따른다(input_dim에서 hidden_dim, hidden_dim, latent_dim으로, 그리고 그 반대). 이 대칭이 꼭 필요하지는 않고 대칭이 아닌 얼개도 잘 되지만, 모델의 담이를 따져 보기 쉬워지고 풀개가 적어도 부호기만큼의 표현력을 갖추게 한다.

고갱이 갈래는 `SimpleAutoencoder`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
