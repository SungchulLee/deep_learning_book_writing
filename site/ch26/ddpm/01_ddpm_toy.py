import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import math
import tqdm

# ==========================================
# 1️⃣ Define a Simple U-Net-like Model
# ==========================================
# This model takes in a noisy image (x_t) and tries to predict the added noise (ε)
# The model is intentionally small to make it easy to understand and train quickly.

class SimpleUNet(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        # Convolutional layers are like feature extractors.
        # Here, we stack 3 simple conv layers.
        self.conv1 = nn.Conv2d(1, channels, 3, 1, 1)   # input channel = 1 (for MNIST)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels, 1, 3, 1, 1)   # output 1 channel (same shape as input)

    def forward(self, x, t):
        # x: noisy image at time t
        # t: timestep (0 → T)

        # Normalize the timestep a bit for stability
        t_embed = t[:, None, None, None].float() / 1000.0

        # Very naive time conditioning:
        # Add t_embed to the input to give model info about which timestep this is
        h = F.relu(self.conv1(x) + t_embed)
        h = F.relu(self.conv2(h))
        return self.conv3(h)  # Predict the noise added to x_t


# ==========================================
# 2️⃣ Diffusion Process Utilities
# ==========================================
# In diffusion models, we gradually add Gaussian noise to an image x₀.
# The noise schedule is controlled by "β" (beta), which defines how much noise we add at each step.

def get_beta_schedule(T, start=1e-4, end=0.02):
    """
    Create a linear beta schedule: from small noise to larger noise.
    T: total number of timesteps.
    """
    return torch.linspace(start, end, T)

def forward_diffusion_sample(x0, t, beta, device):
    """
    Given a clean image x₀, return a noisy version x_t for timestep t.

    x_t = sqrt(α̂_t) * x₀ + sqrt(1 - α̂_t) * ε
    where ε ~ N(0, 1)
    """
    noise = torch.randn_like(x0).to(device)
    sqrt_alpha_hat = torch.sqrt(torch.cumprod(1 - beta, dim=0))[t][:, None, None, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - torch.cumprod(1 - beta, dim=0))[t][:, None, None, None]
    return sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * noise, noise


# ==========================================
# 3️⃣ Training Loop
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
T = 300  # number of timesteps
betas = get_beta_schedule(T).to(device)

model = SimpleUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Load MNIST dataset (28x28 grayscale digits)
transform = transforms.Compose([transforms.ToTensor()])
dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Training: predict the noise added to the image at each step
for epoch in range(3):  # short training for demo
    for x, _ in tqdm.tqdm(dataloader):
        x = x.to(device)
        # Randomly pick timesteps for each image in the batch
        t = torch.randint(0, T, (x.size(0),), device=device).long()

        # Create a noisy version of x₀ at time t
        x_t, noise = forward_diffusion_sample(x, t, betas, device)

        # Predict the noise added to x_t
        noise_pred = model(x_t, t)

        # The goal: make the predicted noise close to the actual noise (MSE loss)
        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")


# ==========================================
# 4️⃣ Reverse Diffusion (Sampling)
# ==========================================
# Once trained, we start from pure noise and gradually denoise it.
# This reverses the forward process.

@torch.no_grad()
def sample(model, T, betas, size, device):
    """
    Generate new samples from random noise using the learned model.
    """
    model.eval()
    # Start from pure Gaussian noise
    x = torch.randn(size).to(device)

    # Loop backward through the timesteps
    for t in reversed(range(T)):
        z = torch.randn_like(x) if t > 0 else 0  # random noise except at the last step
        beta_t = betas[t]
        alpha_t = 1 - beta_t
        alpha_hat_t = torch.cumprod(1 - betas, dim=0)[t]

        # Predict noise at this step
        eps_theta = model(x, torch.tensor([t]*x.size(0), device=device))

        # DDPM reverse formula:
        # x_{t-1} = 1/sqrt(α_t) * (x_t - (β_t / sqrt(1 - α̂_t)) * ε_θ) + sqrt(β_t) * z
        x = (1 / torch.sqrt(alpha_t)) * (x - beta_t / torch.sqrt(1 - alpha_hat_t) * eps_theta) + torch.sqrt(beta_t) * z

    return x  # the denoised (generated) image


# ==========================================
# 5️⃣ Generate Samples and Save
# ==========================================
samples = sample(model, T, betas, size=(16, 1, 28, 28), device=device)
torchvision.utils.save_image(samples, "diffusion_samples.png", nrow=4)
print("✅ Generated samples saved to diffusion_samples.png")
