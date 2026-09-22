"""autoencoder — ch24/architecture/autoencoder.md 의 모델을 모듈로 쓸 수 있게
옮겨 놓은 것이다. 고칠 일이 있으면 그 페이지를 고치는 편이 낫다.
"""

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
