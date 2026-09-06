# DDPM 장난감

이 단원은 깊은 만들어 내는 모델의 중요한 부품인 ddpm 장난감을 짠다. 이 짜기를 이해하면 요즘 만들어 내는 모델에 쓰이는 얼개의 결과 익히기 절차를 꿰뚫어 볼 수 있다. 이 코드는 연구와 실제 얼개에서 널리 쓰이는 쓸모 있는 재주를 보인다.

## 코드

```python
"""DDPM 장난감."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import math
import tqdm

# ==========================================
# 1️⃣ 단순한 U-Net 같은 모델 뜻매김하기
# ==========================================
# 이 모델은 잡음 섞인 그림(x_t)을 받아 더해진 잡음(ε)을 헤아리려 한다
# 이해하기 쉽고 빨리 익히도록 모델을 일부러 작게 만들었다.

class SimpleUNet(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        # 겹말기 층은 특징 뽑개 같은 것이다.
        # 여기서는 단순한 겹말기 층 3개를 쌓는다.
        self.conv1 = nn.Conv2d(1, channels, 3, 1, 1)   # 들임 채널 = 1(MNIST용)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels, 1, 3, 1, 1)   # 채널 1개를 내놓는다(들임과 같은 꼴)

    def forward(self, x, t):
        # x: 때 t의 잡음 섞인 그림
        # t: 때 걸음(0 → T)

        # 안정을 위해 때 걸음을 조금 고르게 맞춘다
        t_embed = t[:, None, None, None].float() / 1000.0

        # 아주 순진한 때 조건 주기:
        # 어느 때 걸음인지 알려 주려 들임에 t_embed을 더한다
        h = F.relu(self.conv1(x) + t_embed)
        h = F.relu(self.conv2(h))
        return self.conv3(h)  # x_t에 더한 잡음을 헤아린다


# ==========================================
# 2️⃣ 퍼짐 과정 도구
# ==========================================
# 퍼짐 모델에서는 그림 x₀에 정규 잡음을 차츰 더한다.
# 잡음 차례표는 걸음마다 잡음을 얼마나 더할지 정하는 "β"(베타)가 다스린다.

def get_beta_schedule(T, start=1e-4, end=0.02):
    """
    선형 베타 차례표를 만든다: 작은 잡음에서 큰 잡음으로.
    T: 온 때 걸음 수.
    """
    return torch.linspace(start, end, T)

def forward_diffusion_sample(x0, t, beta, device):
    """
    깨끗한 그림 x₀이 주어질 때 때 걸음 t의 잡음 섞인 판 x_t을 돌려준다.

    x_t = sqrt(α̂_t) * x₀ + sqrt(1 - α̂_t) * ε
    여기서 ε ~ N(0, 1)
    """
    noise = torch.randn_like(x0).to(device)
    sqrt_alpha_hat = torch.sqrt(torch.cumprod(1 - beta, dim=0))[t][:, None, None, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - torch.cumprod(1 - beta, dim=0))[t][:, None, None, None]
    return sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * noise, noise


# ==========================================
# 3️⃣ 익히기 되풀이
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
T = 300  # 때 걸음 수
betas = get_beta_schedule(T).to(device)

model = SimpleUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# MNIST 자료 묶음을 불러온다(28x28 회색 숫자)
transform = transforms.Compose([transforms.ToTensor()])
dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# 익히기: 걸음마다 그림에 더한 잡음을 헤아린다
for epoch in range(3):  # 보여 주려 짧게 익히기
    for x, _ in tqdm.tqdm(dataloader):
        x = x.to(device)
        # 묶음의 그림마다 때 걸음을 마구잡이로 고른다
        t = torch.randint(0, T, (x.size(0),), device=device).long()

        # 때 t에서 x₀의 잡음 섞인 판을 만든다
        x_t, noise = forward_diffusion_sample(x, t, betas, device)

        # x_t에 더한 잡음을 헤아린다
        noise_pred = model(x_t, t)

        # 목표: 헤아린 잡음을 실제 잡음에 가깝게 한다(평균 제곱 어긋남 손실)
        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")


# ==========================================
# 4️⃣ 뒤 퍼짐(뽑기)
# ==========================================
# 익히고 나면 순수 잡음에서 시작해 차츰 잡음을 없앤다.
# 이는 앞 과정을 거꾸로 돌린다.

@torch.no_grad()
def sample(model, T, betas, size, device):
    """
    배운 모델로 아무 잡음에서 새 표본을 만든다.
    """
    model.eval()
    # 순수 정규 잡음에서 시작한다
    x = torch.randn(size).to(device)

    # 때 걸음을 거슬러 되풀이한다
    for t in reversed(range(T)):
        z = torch.randn_like(x) if t > 0 else 0  # 마지막 걸음만 빼고 아무 잡음
        beta_t = betas[t]
        alpha_t = 1 - beta_t
        alpha_hat_t = torch.cumprod(1 - betas, dim=0)[t]

        # 이 걸음의 잡음을 헤아린다
        eps_theta = model(x, torch.tensor([t]*x.size(0), device=device))

        # DDPM 뒤 공식:
        # x_{t-1} = 1/sqrt(α_t) * (x_t - (β_t / sqrt(1 - α̂_t)) * ε_θ) + sqrt(β_t) * z
        x = (1 / torch.sqrt(alpha_t)) * (x - beta_t / torch.sqrt(1 - alpha_hat_t) * eps_theta) + torch.sqrt(beta_t) * z

    return x  # 잡음을 없앤(만든) 그림


# ==========================================
# 5️⃣ 표본 만들어 갈무리하기
# ==========================================
samples = sample(model, T, betas, size=(16, 1, 28, 28), device=device)
torchvision.utils.save_image(samples, "diffusion_samples.png", nrow=4)
print("✅ Generated samples saved to diffusion_samples.png")


if __name__ == "__main__":
    pass
```

## 논의

이 짜기는 ddpm 장난감에 대해 자리 잡은 가장 좋은 방식을 따른다. 코드는 모델 뜻매김, 익히기 논리, 도구 함수를 또렷이 갈라 짜였다. 핵심 설계 결정에는 깨움 함수 고르기, 고르게 맞추기 방책, 가장 좋게 하기 웃매개변수가 들며 모두 익히기의 안정과 내놓기 품질에 크게 영향을 준다.

이 얼개는 깊은 만들어 내는 모델에 흔한 중요한 결 여럿을 보인다. 곧 여러 신경망 층을 지나며 특징을 차츰 다루기, 모델이 곁 앎을 받아들이게 하는 조건 주기 얼개, 익히는 동안 기울기가 안정되게 흐르도록 하는 꼼꼼한 첫자리매김이다.

새 자료 묶음이나 문제 마당에서는 웃매개변수 고르기와 익히기 절차를 꼼꼼히 맞추어야 할 때가 많으므로 다루는 이들은 이에 마음을 써야 한다. 코드가 조각으로 나뉘어 있어 다른 얼개, 손실 함수, 익히기 방책을 실험하기 쉽다.

## 연습문제

**연습문제 1.**
구체적인 들임 텐서로 이 단원의 으뜸 모델의 앞먹임을 좇아라. 층마다 꼴이 어떻게 바뀌는지 적고 내놓기 차원이 바라던 것과 맞는지 확인하라.

??? success "연습문제 1 풀이"
    들임 텐서에서 시작해 층마다 바뀜을 따라가라. 겹말기 층에서는 공간 차원에 공식 $H_{out} = \lfloor(H_{in} + 2p - k) / s\rfloor + 1$을 쓴다. 선형 층에서는 특징 차원의 바뀜을 좇는다. 중간 꼴을 하나씩 적고 마지막 내놓기가 그 일(그림 만들어 내기, 가르기 등)에 바라던 목표 차원과 맞는지 확인하라.

---

**연습문제 2.**
이 짜기의 핵심 웃매개변수(배움 빠르기, 묶음 크기, 얼개 고르기)를 가려내라. 다른 것을 붙박아 두고 하나씩 바꾸어 웃매개변수마다 익히기가 얼마나 민감한지 재는 실험을 짜라.

??? success "연습문제 2 풀이"
    핵심 웃매개변수에는 배움 빠르기(흔히 $10^{-4}$에서 $10^{-3}$), 묶음 크기(64-256), 층과 채널의 수, 깨움 함수가 든다. 웃매개변수마다 값을 3~5가지로 바꾸어 모델을 익히고 알맞은 잣대(손실, 표본 품질, 모이는 빠르기)를 좇아라. 결과를 그려 어느 웃매개변수가 가장 큰 영향을 주는지 가려내라. 흔히 배움 빠르기와 얼개 깊이가 가장 세게 영향을 주고, 묶음 크기는 알맞은 범위 안에서는 웬만큼 영향을 준다.

---

**연습문제 3.**
이 짜기에 새 기능을 더해 넓혀라. 곧 기울기 자르기, 배움 빠르기 차례표, 다른 손실 함수를 더하라. 고치기 앞뒤의 익히기 움직임을 견주어라.

??? success "연습문제 3 풀이"
    기울기 자르기는 `optimizer.step()` 앞에 `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`을 더한다. 배움 빠르기 차례표는 `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)`을 쓰고 바퀴마다 `scheduler.step()`을 부른다. 익히기 손실 곡선, 모이는 빠르기, 마지막 모델 품질을 견주어라. 기울기 자르기는 흔히 익히기가 치솟는 것을 막고, 코사인 식히기는 뒤 바퀴에서 더 곱게 가장 좋게 하여 마지막 솜씨를 높일 수 있다.
