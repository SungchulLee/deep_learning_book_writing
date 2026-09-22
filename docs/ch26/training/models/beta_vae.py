"""beta_vae — ch24/architecture/beta_vae.md 의 모델을 모듈로 쓸 수 있게
옮겨 놓은 것이다. 고칠 일이 있으면 그 페이지를 고치는 편이 낫다.
"""

from .conv_vae import ConvVAE

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


class ConvBetaVAE(ConvVAE):
    """누비기 beta-VAE.

    얼개는 ConvVAE와 같고, 손실에서 KL 항에 주는 무게 beta만 다르다.
    beta를 키우면 숨은 축이 서로 풀리는 쪽으로 더 세게 밀리는 대신
    다시 세우기가 흐려진다.
    """

    def __init__(self, latent_dim=128, img_channels=1, beta=4.0):
        super().__init__(latent_dim=latent_dim, img_channels=img_channels)
        self.beta = beta
