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

## 2. 논의

다시 매개변수화 재주가 변분 자기 부호기의 고갱이 새것이다. $z \sim \mathcal{N}(\mu, \sigma^2)$을 곧바로 뽑는(미분할 수 없는) 대신 $\epsilon \sim \mathcal{N}(0, I)$에 대해 $z = \mu + \sigma \odot \epsilon$으로 쓴다. 그러면 무작위가 셈 그래프 밖으로 나가 뒤먹임 퍼뜨리기 때 기울기가 $\mu$과 $\sigma$을 지나 흐를 수 있다.

변분 자기 부호기의 손실은 다시 세우기 품질(두 값 엇갈린 엔트로피나 평균 제곱 어긋남)과, 숨은 분포를 표준 정규 사전 분포 쪽으로 끄는 KL 벌어짐 항을 아우른다. KL 항 $D_\text{KL}(q(z|x) \| p(z)) = -\frac{1}{2}\sum(1 + \log\sigma^2 - \mu^2 - \sigma^2)$이 부호기가 점 어림으로 무너지는 것을 막고 숨은 공간이 매끄럽고 이어지게 한다. 이 벌주기 덕분에 변분 자기 부호기가 만들어 내는 모델이 된다. 곧 $\mathcal{N}(0, I)$에서 뽑아 풀면 그럴듯한 새 그림이 나온다.

다시 세우기와 KL 벌어짐 사이의 팽팽함이 근본 맞바꿈을 만든다. 다시 세우기 손실을 가장 작게 하면 또렷하고 세밀한 내놓기가 나오지만 숨은 공간이 조각날 수 있다. KL을 가장 작게 하면 매끄럽고 고른 숨은 공간이 되지만 다시 세운 것이 흐릿할 수 있다. 이 두 항을 저울질하는 것이 변분 자기 부호기 설계의 핵심 과제이다.

## 연습문제

**연습문제 1.**
$d$차원 정규 분포에서 $\mathcal{N}(\mu, \sigma^2 I)$과 $\mathcal{N}(0, I)$ 사이 KL 벌어짐의 닫힌 꼴을 이끌어 내어라. 이끌어 낸 것이 부호에 쓰인 식과 맞는지 확인하라.

??? success "연습문제 1 풀이"
    대각 정규 분포에서는 KL 벌어짐이 차원에 걸친 합으로 쪼개진다:

    $$
    D_\text{KL} = -\frac{1}{2}\sum_{j=1}^{d}\left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)
    $$

    이는 $\mu_0=0, \Sigma_0=I$을 넣은 두루 쓰는 식 $D_\text{KL}(\mathcal{N}_1 \| \mathcal{N}_0) = \frac{1}{2}\left[\text{tr}(\Sigma_0^{-1}\Sigma_1) + (\mu_0-\mu_1)^\top\Sigma_0^{-1}(\mu_0-\mu_1) - d + \ln\frac{|\Sigma_0|}{|\Sigma_1|}\right]$에서 나온다.

---

**연습문제 2.**
MNIST에 변분 자기 부호기를 익히고 $z \sim \mathcal{N}(0, I)$을 풀어 아무 표본 100개를 만들어라. 10x10 격자로 보여라. 만든 숫자를 알아볼 수 있는가?

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

**연습문제 3.**
다시 매개변수화 재주가 왜 필요한지 설명하라. 그 재주 없이 $\mathcal{N}(\mu, \sigma^2)$에서 $z$을 곧바로 뽑으면 어떻게 되는가?

??? success "연습문제 3 풀이"
    곧바로 뽑으면 셈 그래프에 미분할 수 없는 연산이 생긴다. 곧 분포에서 뽑는 `sample` 연산은 분포 매개변수에 대한 기울기가 없다. 기울기가 $z$을 지나 $\mu$과 $\log\sigma^2$으로 흐르지 않으면 부호기 매개변수를 뒤먹임 퍼뜨리기로 새로 고칠 수 없다. 다시 매개변수화 재주는 뽑기를 $\mu$, $\sigma$, 바깥 잡음 $\epsilon$의 정해진 함수로 다시 세워 물길 전체를 미분할 수 있게 한다.

## 정리하며

**다룬 것** — 변분 자기 부호기

다시 매개변수화 재주가 변분 자기 부호기의 고갱이 새것이다.

고갱이 갈래는 `VAE`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
