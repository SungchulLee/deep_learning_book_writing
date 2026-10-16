# GAN

GAN은 2014년 글 "Generative Adversarial Networks"에서 나왔다. 네트워크 둘(생성기와 판별기)이 맞서며 학습된이다.

여기 구현은 GAN을 짧고 학습하기 좋게 보인 본이다. 코드는 핵심 구조와 앞으로 스텝에 마음을 두어, 핵심 꾸밈새를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
GAN - 맞겨루기 생성기 넷트웍
논문: "Generative Adversarial Networks" (2014)
핵심: 맞겨루며 익히는 두 넷트웍(생성기와 판별기)
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

# ---------------------------------------------------------------------------
# GAN 익힘 회귀
# ---------------------------------------------------------------------------
# 가장 작게-가장 크게 겨루기는 다음을 번갈아 한다:
#   1. D 고치기: log D(x) + log(1 - D(G(z)))을 가장 크게
#   2. G 고치기: log(1 - D(G(z)))을 가장 작게  [또는 같은 뜻으로 log D(G(z))을 가장 크게]
# 참으로는 둘째 스텝에서 "잦아들지 않는" 잃음 -log(D(G(z)))을 쓴다
# 익힘 이른 판에 그래디언트가 더 세기 때문이다.

def update_D(X, Z, net_D, net_G, loss, trainer_D):
    """판별기 고치기: log D(x) + log(1 - D(G(z)))을 가장 크게."""
    batch_size = X.shape[0]
    ones = torch.ones(batch_size, 1, device=X.device)
    zeros = torch.zeros(batch_size, 1, device=X.device)
    trainer_D.zero_grad()
    real_Y = net_D(X)
    fake_X = net_G(Z)
    fake_Y = net_D(fake_X.detach())  # detach so G is not updated
    loss_D = loss(real_Y, ones) + loss(fake_Y, zeros)
    loss_D.backward()
    trainer_D.step()
    return float(loss_D)


def update_G(Z, net_D, net_G, loss, trainer_G):
    """생성기 고치기: log D(G(z))을 가장 크게(잦아들지 않는 잃음)."""
    batch_size = Z.shape[0]
    ones = torch.ones(batch_size, 1, device=Z.device)
    trainer_G.zero_grad()
    fake_X = net_G(Z)
    fake_Y = net_D(fake_X)
    loss_G = loss(fake_Y, ones)  # fool D into classifying fake as real
    loss_G.backward()
    trainer_G.step()
    return float(loss_G)


def train_gan(net_G, net_D, data_iter, num_epochs, latent_dim,
              lr_D=0.0002, lr_G=0.0002, device='cpu'):
    """잃음을 좇는 온전한 GAN 익힘 회귀."""
    loss = nn.BCELoss()
    net_G, net_D = net_G.to(device), net_D.to(device)
    trainer_D = torch.optim.Adam(net_D.parameters(), lr=lr_D, betas=(0.5, 0.999))
    trainer_G = torch.optim.Adam(net_G.parameters(), lr=lr_G, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        d_loss_sum, g_loss_sum, n = 0, 0, 0
        for X, _ in data_iter:
            X = X.to(device)
            batch_size = X.shape[0]
            Z = torch.randn(batch_size, latent_dim, device=device)
            d_loss_sum += update_D(X, Z, net_D, net_G, loss, trainer_D)
            g_loss_sum += update_G(Z, net_D, net_G, loss, trainer_G)
            n += 1
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"D loss: {d_loss_sum/n:.4f}, G loss: {g_loss_sum/n:.4f}")


if __name__ == "__main__":
    model = GAN()
    print(f"Generator Parameters: {sum(p.numel() for p in model.generator.parameters()):,}")
    print(f"Discriminator Parameters: {sum(p.numel() for p in model.discriminator.parameters()):,}")
```

**출력:**

```
Generator Parameters: 1,510,032
Discriminator Parameters: 533,505
```

## 2. 논의

이 구현은 클래스 3개(`Generator`, `Discriminator`, `GAN`)를 매기고, 이들이 어울려 온전한 생성기 모델 구조를 이룬다. 클래스마다 남다른 몫을 담아 코드를 묶음으로 나누고 넓히기 쉽게 한다. `forward` 방법이 PyTorch가 절로 미분하는 데 쓰는 계산 그림을 매긴다.

학습 회귀는 여느 PyTorch 무늬를 따른다. 앞으로 스텝으로 추론을 계산하고, 로스를 계산하고, 뒤로 스텝으로 그래디언트를 계산하고, 다듬개로 파라메터를 고친다. 판마다 자를 좇으면 모여드는 결이 드러나고 덜 맞추기나 지나치게 맞추기 같은 탈을 짚어내는 데 도움이 된다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 증강 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
기본 첫자리로 잡은 `Generator`의 배울 수 있는 파라메터를 모두 세어라. 짐과 바이어스를 아울러 켜마다 나누어 적어라.

</div>

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)`마다 짐 파라메터가 `in_features * out_features`개이고 바이어스가 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 짐이 `in_c * out_c * k * k`개, 바이어스가 `out_c`개다. `nn.Embedding(num, dim)`은 파라메터가 `num * dim`개다. 켜를 모두 더한다. `sum(p.numel() for p in model.parameters())`으로 따져 볼 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
다듬개를 Adam(`torch.optim.Adam`에 `lr=0.001`)으로 갈음하고 본디 다듬개와 학습이 모여드는 결을 비교하여라. 둘의 로스 굽이를 한 그림에 그려라.

</div>

??? success "연습문제 2 풀이"
    다듬개를 짓는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 갈음한다. Adam은 파라메터마다 맞추어 가는 러닝레이트와 여세 어림을 지니므로 이른 판에서 대체로 더 빨리 모여든다. Adam의 로스 굽이는 흔히 첫 몇 판에서 더 가파르게 떨어지지만, 마무리 값 언저리에서는 여세를 쓰는 SGD보다 조금 더 흔들릴 수 있다. 고르게 견주려면 둘을 같은 시드와 같은 판 수로 돌려라.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현이 무너질 만한 결 둘을 밝히고, 저마다 어떻게 짚어내고 고칠지 밝혀라.

</div>

??? success "연습문제 3 풀이"
    흔히 무너지는 결은 이렇다. (1) **그래디언트가 사라지거나 터짐** -- 그래디언트 크기를 지켜보아 짚어낸다(`torch.nn.utils.clip_grad_norm_`이나 켜마다 `param.grad.norm()` 적기). 그래디언트 자르기, 더 나은 첫자리 잡기(Xavier/Kaiming), 구조 고치기(나머지 이음, 잣대 잡기)로 고친다. (2) **지나치게 맞추기** -- 학습 로스는 줄어드는데 따짐 로스가 오르면 짚어낸다. 다독임(드롭아웃, 짐 줄이기, 데이터 증강)이나 모델 크기 줄이기로 고친다. 학습과 따짐 자를 늘 함께 지켜보아 이를 일찍 잡아야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`Generator`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 파라메터를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 구조를 짜라. 켜 2개, 4개, 8개로 시험하여라.

</div>

??? success "연습문제 4 풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch가 파라메터를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = Generator(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — GAN

이 구현은 클래스 3개(`Generator`, `Discriminator`, `GAN`)를 매기고, 이들이 어울려 온전한 생성기 모델 구조를 이룬다.

핵심 클래스는 `Generator`, `Discriminator`, `GAN`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
