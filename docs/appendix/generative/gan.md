# GAN

GAN was introduced in the 2014 paper "Generative Adversarial Networks." Two networks (Generator and Discriminator) in adversarial training.

This implementation provides a concise, educational reference for GAN. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## 코드

```python
#!/usr/bin/env python3
'''
GAN - Generative Adversarial Networks
Paper: "Generative Adversarial Networks" (2014)
Key: Two networks (Generator and Discriminator) in adversarial training
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
# GAN Training Loop
# ---------------------------------------------------------------------------
# The minimax game alternates between:
#   1. Update D: maximize log D(x) + log(1 - D(G(z)))
#   2. Update G: minimize log(1 - D(G(z)))  [or equivalently maximize log D(G(z))]
# In practice, step 2 uses the "non-saturating" loss: -log(D(G(z)))
# because it provides stronger gradients early in training.

def update_D(X, Z, net_D, net_G, loss, trainer_D):
    """Update discriminator: maximize log D(x) + log(1 - D(G(z)))."""
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
    """Update generator: maximize log D(G(z))  (non-saturating loss)."""
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
    """Full GAN training loop with loss tracking."""
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
    print(f"Discriminator Parameters: {sum(p.numel() for p in model.discriminator.parameters()):,}")```

## 논의

이 짜보기는 갈래 3개(`Generator`, `Discriminator`, `GAN`)를 매기고, 이들이 어울려 온전한 만들개 모형 얼개를 이룬다. 갈래마다 남다른 몫을 담아 코드를 묶음으로 나누고 넓히기 쉽게 한다. `forward` 방법이 PyTorch이 절로 미분하는 데 쓰는 셈 그림을 매긴다.

The training loop follows the standard PyTorch pattern: forward pass to compute predictions, loss calculation, backward pass for gradient computation, and parameter update via the optimizer. Tracking metrics across epochs reveals the convergence behavior and helps diagnose issues like underfitting or overfitting.

여기 실린 코드는 본보기 짜보기라 다듬기보다 알아보기 쉬움을 앞세운다. 서비스 얼개라면 흔히 섞인 촘촘함 익히기, 흩은 자료 나란히, 더 정교한 자료 불리기를 더한다. 그래도 여기서 보인 얼개의 고갱이 깨침은 크기와 상관없이 그대로다.

## 익힘 문제

**익힘 1.**
기본 첫자리로 잡은 `Generator`의 배울 수 있는 매개변수를 모두 세어라. 짐과 치우침을 아울러 켜마다 나누어 적어라.

??? success "익힘 1 풀이"
    `nn.Linear(in_features, out_features)`마다 짐 매개변수가 `in_features * out_features`개이고 치우침이 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 짐이 `in_c * out_c * k * k`개, 치우침이 `out_c`개다. `nn.Embedding(num, dim)`은 매개변수가 `num * dim`개다. 켜를 모두 더한다. `sum(p.numel() for p in model.parameters())`으로 따져 볼 수 있다.

---

**익힘 2.**
Replace the optimizer with Adam (use `torch.optim.Adam` with `lr=0.001`) and compare the training convergence with the original optimizer. Plot the loss curves for both on the same graph.

??? success "익힘 2 풀이"
    Replace the optimizer creation line with `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`. Adam typically converges faster in early epochs because it maintains per-parameter adaptive learning rates and momentum estimates. The loss curve with Adam usually drops more steeply in the first few epochs but may oscillate slightly more than SGD with momentum near the optimum. For a fair comparison, run both with the same random seed and number of epochs.

---

**익힘 3.**
이 짜보기가 무너질 만한 결 둘을 밝히고, 저마다 어떻게 짚어내고 고칠지 밝혀라.

??? success "익힘 3 풀이"
    흔히 무너지는 결은 이렇다. (1) **기울기가 사라지거나 터짐** -- 기울기 크기를 지켜보아 짚어낸다(`torch.nn.utils.clip_grad_norm_`이나 켜마다 `param.grad.norm()` 적기). 기울기 자르기, 더 나은 첫자리 잡기(Xavier/Kaiming), 얼개 고치기(나머지 이음, 잣대 잡기)로 고친다. (2) **지나치게 맞추기** -- 익힘 잃음은 줄어드는데 따짐 잃음이 오르면 짚어낸다. 다독임(드롭아웃, 짐 줄이기, 자료 불리기)이나 모형 크기 줄이기로 고친다. 익힘과 따짐 자를 늘 함께 지켜보아 이를 일찍 잡아야 한다.

---

**익힘 4.**
`Generator`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 얼개를 짜라. 켜 2개, 4개, 8개로 시험하여라.

??? success "익힘 4 풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch이 매개변수를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = Generator(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.
