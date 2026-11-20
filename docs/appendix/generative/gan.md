# GAN

GAN은 2014년 논문 "Generative Adversarial Networks"에서 나왔다. 네트워크 둘(제너레이터와 디스크리미네이터)이 적대적으로 학습된다 — 제너레이터는 잡음에서 가짜 이미지를 만들고, 디스크리미네이터는 진짜와 가짜를 가려내려 하며, 둘이 서로를 상대로 실력을 키운다.

여기 구현은 GAN을 짧게 줄인 학습용 예제다. 코드는 핵심 구조와 순전파(forward pass)에 초점을 두어, 핵심 구성 요소를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
GAN - 적대적 생성 네트워크
논문: "Generative Adversarial Networks" (2014)
핵심: 적대적으로 학습하는 두 네트워크(제너레이터와 디스크리미네이터)
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
# GAN 학습 루프
# ---------------------------------------------------------------------------
# 최소-최대(minimax) 게임은 다음을 번갈아 한다:
#   1. D 갱신: log D(x) + log(1 - D(G(z)))을 최대화
#   2. G 갱신: log(1 - D(G(z)))을 최소화  [또는 같은 뜻으로 log D(G(z))을 최대화]
# 실제로는 둘째 스텝에서 "비포화(non-saturating)" 로스 -log(D(G(z)))을 쓴다
# 학습 초기에 그래디언트가 더 크기 때문이다.

def update_D(X, Z, net_D, net_G, loss, trainer_D):
    """디스크리미네이터 갱신: log D(x) + log(1 - D(G(z)))을 최대화."""
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
    """제너레이터 갱신: log D(G(z))을 최대화(비포화 로스)."""
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
    """로스를 기록하는 완전한 GAN 학습 루프."""
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

이 구현은 클래스 3개(`Generator`, `Discriminator`, `GAN`)를 정의하고, 이들이 함께 완전한 생성 모델 구조를 이룬다. 클래스마다 고유한 역할을 맡아 코드를 모듈로 나누고 확장하기 쉽게 한다. `forward` 메서드가 PyTorch의 자동 미분이 쓰는 계산 그래프를 정의한다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 순전파로 예측을 계산하고, 로스를 계산하고, 역전파로 그래디언트를 계산하고, 옵티마이저로 파라메터를 갱신한다. 에포크마다 지표를 기록하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
기본 설정으로 초기화한 `Generator`의 학습 가능한 파라메터를 모두 세어라. 웨이트와 바이어스를 함께 레이어마다 나누어 적어라.

</div>

??? success "풀이"
    `nn.Linear(in_features, out_features)`마다 웨이트 파라메터가 `in_features * out_features`개이고 바이어스가 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 웨이트가 `in_c * out_c * k * k`개, 바이어스가 `out_c`개다. `nn.Embedding(num, dim)`은 파라메터가 `num * dim`개다. 레이어를 모두 더한다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다. 위 출력의 1,510,032개가 기본 설정 `Generator`의 답이다(`BatchNorm1d`의 스케일과 시프트도 학습 가능한 파라메터로 들어간다).

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`train_gan`의 두 옵티마이저를 Adam 기본 설정(`lr=0.001`, `betas=(0.9, 0.999)`)으로 바꾸고, 이 코드의 설정(`lr=0.0002`, `betas=(0.5, 0.999)`)과 학습의 수렴 양상을 비교하여라. 둘의 로스 곡선을 한 그림에 그려라.

</div>

??? success "풀이"
    옵티마이저를 만드는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`처럼 바꾼다(`model` 자리에 `net_D`와 `net_G`를 넣고, `betas`는 기본값으로 둔다). 러닝레이트가 다섯 배 커지고 1차 모멘텀 계수가 0.5에서 0.9로 커지면, 두 네트워크가 상대의 변화에 더 크게 반응해 D 로스와 G 로스가 진동하거나 한쪽이 무너지기 쉽다. DCGAN 논문이 `lr=0.0002`와 `betas=(0.5, 0.999)`를 권한 까닭이 이것이다. 공정하게 견주려면 둘을 같은 시드와 같은 에포크 수로 돌려라.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현에서 생길 수 있는 실패 양상 둘을 들고, 저마다 어떻게 진단하고 고칠지 설명하라.

</div>

??? success "풀이"
    흔한 실패 양상은 이렇다. (1) **그래디언트 소실 또는 폭발** -- 그래디언트 크기를 관찰해 진단한다(`torch.nn.utils.clip_grad_norm_`이나 레이어마다 `param.grad.norm()` 기록하기). 그래디언트 클리핑, 더 나은 초기화(Xavier/Kaiming), 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** -- 학습 로스는 줄어드는데 검증 로스가 오르면 과적합으로 진단한다. 정칙화(드롭아웃, 웨이트 감쇠, 데이터 어그멘테이션)나 모델 크기 줄이기로 고친다. 학습 지표와 검증 지표를 늘 함께 관찰해 이를 일찍 잡아야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`Generator`를 레이어나 블록의 수를 설정할 수 있게 확장하라. `__init__`에 `num_layers` 파라메터를 추가하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 구성하라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩된 레이어를 다음으로 대체한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 반복한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`를 써야 PyTorch가 파라메터를 모두 최적화 대상으로 등록한다. `for n in [2, 4, 8]: model = Generator(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`로 시험한다.

## 정리하며

**다룬 것** — GAN

이 구현은 클래스 3개(`Generator`, `Discriminator`, `GAN`)를 정의하고, 이들이 함께 완전한 생성 모델 구조를 이룬다.

핵심 클래스는 `Generator`, `Discriminator`, `GAN`이며 앞의 연습문제 4개로 스스로 점검해 볼 수 있다.
