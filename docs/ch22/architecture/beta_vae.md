# 베타 변분 자기 부호기

베타 변분 자기 부호기는 변분 자기 부호기 손실의 KL 벌어짐 항에 무게를 주는 웃매개변수 $\beta$을 들여와 다시 세우기 품질과 숨은 공간 얽힘 풀기의 맞바꿈을 다스린다. $\beta$이 클수록 숨은 차원마다 서로 얽히지 않고 풀이할 수 있는 흔들림 요인(예컨대 돌림, 굵기, 모양새)을 담도록 이끈다. 이 단원은 온전히 이어진 판과 누비기 판을 모두 짜고, 차원마다 무엇을 배웠는지 그려 보는 숨은 훑기 방법을 곁들인다.

## 코드

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
    x = torch.randn(32, 784)
    reconstruction, mu, logvar = model(x)
    loss, bce, kld = model.loss_function(reconstruction, x, mu, logvar)
    print(f"Loss: {loss.item():.4f}, Recon: {bce.item():.4f}, KL: {kld.item():.4f}")
```

## 논의

매개변수 $\beta$은 여느 자기 부호기($\beta = 0$, KL 벌주기 없음)와 지나치게 옭아맨 모델($\beta \gg 1$, 숨은 차원이 모두 사전 분포로 무너짐) 사이를 잇는다. $\beta = 1$이면 여느 변분 자기 부호기가 된다. 베타 변분 자기 부호기 논문의 핵심 눈썰미는 $\beta > 1$(흔히 4~10)이 얽힘 풀기를 이끈다는 것이다. 곧 숨은 차원마다 서로 얽히지 않은 흔들림 요인 하나를 담기 쉬워진다.

얽힘 풀기는 숨은 훑기로 가늠한다. 익힌 모델에서 한 차원만 빼고 숨은 차원을 모두 0으로 고정한 뒤 그 차원을 $-3$에서 $+3$까지 훑어 나온 숨은 벡터를 푼다. 그 차원의 얽힘이 풀렸다면 푼 그림이 다른 속성은 그대로인 채 한 속성만(예컨대 돌림만, 또는 굵기만) 바뀌어야 한다. 실전에서 완벽한 얽힘 풀기는 드물지만 베타 변분 자기 부호기는 여느 변분 자기 부호기보다 훨씬 풀이하기 쉬운 숨은 공간을 낸다.

다시 세우기와 얽힘 풀기의 맞바꿈은 근본이다. $\beta$을 키우면 얽힘 풀기는 나아지지만 KL 벌주기가 세져 부호기가 숨은 공간의 담이를 다 쓰지 못해 다시 세우기 품질이 나빠진다. 가장 좋은 $\beta$을 찾으려면 실험해야 하며 자료 묶음과 뒤따르는 쓰임새에 달렸다.

## 연습문제

**연습문제 1.**
MNIST에서 $\beta \in \{0.1, 1, 4, 10, 50\}$으로 베타 변분 자기 부호기를 익혀라. 저마다 앞선 5개 차원의 숨은 훑기를 만들고 어느 $\beta$이 가장 얽힘이 풀린 나타냄을 내는지 눈으로 가늠하라.

??? success "연습문제 1 풀이"
    $\beta = 0.1$이면 다시 세운 것이 또렷하지만 숨은 훑기가 얽힌 바뀜을 보인다(속성 여럿이 한꺼번에 바뀐다). $\beta = 4$이면 차원마다 획의 굵기나 기울기 같은 뚜렷한 속성을 담기 시작한다. $\beta = 50$이면 많은 차원이 무너지고(아무 흔들림도 내지 않고) 다시 세운 것이 아주 흐릿하다. MNIST에서 알맞은 자리는 흔히 $\beta = 4$~$10$이다.

---

**연습문제 2.**
$\beta = 1$일 때 베타 변분 자기 부호기의 손실이 자료의 로그 가능도에 대한 하한(증거 하한)과 같음을 밝혀라.

??? success "연습문제 2 풀이"
    증거 하한은 $\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_\text{KL}(q(z|x) \| p(z))$이다. 첫 항은 다시 세우기 손실에 음의 부호를 붙인 것(두 값 엇갈린 엔트로피나 평균 제곱 어긋남)이고 둘째는 KL 벌어짐이다. $\beta = 1$이면 베타 변분 자기 부호기의 손실이 $-\mathcal{L}$, 곧 음의 증거 하한이다. 증거 하한을 가장 크게 하는 것은 변분 자기 부호기 손실을 가장 작게 하는 것과 같다. $\beta \neq 1$이면 그 손실은 더는 옳은 증거 하한이 아니지만, 모델을 더 고른 숨은 공간 쪽으로 기울이는 쓸모 있는 익히기 목표가 된다.

---

**연습문제 3.**
얽힘 풀기를 값으로 재는 잣대를 짜라. $\beta = 1$과 $\beta = 4$으로 익힌 모델에 그것을 셈해 $\beta$이 클수록 점수가 좋은지 확인하라.

??? success "연습문제 3 풀이"
    단순한 얽힘 풀기 잣대: 숨은 차원 $j$마다 알려진 참 요인 $k$만 바꿔 자료를 만들고 부호화한 뒤 어느 숨은 차원의 흩어짐이 가장 큰지 잰다. 차원 $j$이 요인 $k$과 한결같이 일대일로 맞닿으면 얽힘이 풀린 나타냄이다. 점수는 딱 한 숨은 차원이 옳게 잡은 요인의 몫이다. dSprites처럼 참 요인이 알려진 자료 묶음에서 $\beta = 4$은 흔히 $\beta = 1$보다 20~40% 높은 점수를 낸다.
