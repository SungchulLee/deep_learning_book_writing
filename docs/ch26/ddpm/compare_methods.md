# 방법 견주기

이 단원은 요즘 만들어 내는 모델의 핵심 부품인 방법 견주기을 짠다. 여기서 보이는 개념과 재주를 알면 퍼짐 모델과 점수 바탕 만들어 내는 방법을 다루는 데 꼭 필요한 앎을 얻는다. 이 짜기는 또렷함과 실제 쓸모의 균형을 맞추어 배우기에도 실험하기에도 알맞다.

## 1. 코드

```python
"""방법을 견준다."""
# ==========================================
# 퍼짐 방법 견주기 보여 주기
# DDPM과 DDIM의 뽑기 빠르기를 견준다
# ==========================================
import torch
import torch.nn as nn
import torchvision
import time
from tqdm import tqdm

# 견주기용 단순한 U-Net
class SimpleUNet(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels, 1, 3, 1, 1)

    def forward(self, x, t):
        import torch.nn.functional as F
        t_embed = t[:, None, None, None].float() / 1000.0
        h = F.relu(self.conv1(x) + t_embed)
        h = F.relu(self.conv2(h))
        return self.conv3(h)

# 차례표
def get_cosine_schedule(T):
    import math
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + 0.008) / (1 + 0.008) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-8, 0.999)

# DDPM 뽑기
@torch.no_grad()
def ddpm_sample(model, T, betas, device, num_samples=1):
    model.eval()
    x = torch.randn(num_samples, 1, 28, 28).to(device)
    
    for t in tqdm(reversed(range(T)), desc="DDPM Sampling", total=T):
        z = torch.randn_like(x) if t > 0 else 0
        beta_t = betas[t]
        alpha_t = 1 - beta_t
        alpha_hat_t = torch.cumprod(1 - betas, dim=0)[t]
        
        eps_theta = model(x, torch.tensor([t]*x.size(0), device=device))
        
        x = (1 / torch.sqrt(alpha_t)) * (x - beta_t / torch.sqrt(1 - alpha_hat_t) * eps_theta) + torch.sqrt(beta_t) * z
    
    return x

# DDIM 뽑기
@torch.no_grad()
def ddim_sample(model, T, betas, device, ddim_steps=50, num_samples=1):
    model.eval()
    
    # 부분 차례를 만든다
    c = T // ddim_steps
    ddim_timesteps = torch.arange(0, T, c).to(device)
    ddim_timesteps_prev = torch.cat([torch.tensor([0]).to(device), ddim_timesteps[:-1]])
    
    alphas_cumprod = torch.cumprod(1 - betas, dim=0)
    
    x = torch.randn(num_samples, 1, 28, 28).to(device)
    
    for i in tqdm(reversed(range(len(ddim_timesteps))), desc="DDIM Sampling", total=len(ddim_timesteps)):
        t = ddim_timesteps[i].item()
        t_prev = ddim_timesteps_prev[i].item()
        
        # 알파 값을 얻는다
        alpha_t = alphas_cumprod[t]
        alpha_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0).to(device)
        
        # 잡음을 헤아린다
        eps = model(x, torch.tensor([t]*x.size(0), device=device))
        
        # x0을 헤아린다
        pred_x0 = (x - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
        pred_x0 = pred_x0.clamp(-1, 1)
        
        # 방향
        dir_xt = torch.sqrt(1 - alpha_prev) * eps
        
        # 갱신
        x = torch.sqrt(alpha_prev) * pred_x0 + dir_xt
    
    return x

def main():
    print("=" * 70)
    print("🔬 DIFFUSION METHODS COMPARISON")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n📍 Using device: {device}")
    
    # 준비
    T = 1000
    DDIM_STEPS = 50
    model = SimpleUNet().to(device)
    betas = get_cosine_schedule(T).to(device)
    
    print(f"\n📊 Configuration:")
    print(f"   Total timesteps: {T}")
    print(f"   DDIM steps: {DDIM_STEPS}")
    print(f"   Speedup factor: {T / DDIM_STEPS}x")
    
    # 비교
    print("\n" + "=" * 70)
    print("Method 1: DDPM (Original)")
    print("=" * 70)
    start = time.time()
    ddpm_samples = ddpm_sample(model, T, betas, device, num_samples=4)
    ddpm_time = time.time() - start
    print(f"⏱️  Time: {ddpm_time:.2f}s")
    
    print("\n" + "=" * 70)
    print("Method 2: DDIM (Fast Sampling)")
    print("=" * 70)
    start = time.time()
    ddim_samples = ddim_sample(model, T, betas, device, ddim_steps=DDIM_STEPS, num_samples=4)
    ddim_time = time.time() - start
    print(f"⏱️  Time: {ddim_time:.2f}s")
    
    # 결과
    print("\n" + "=" * 70)
    print("📈 RESULTS")
    print("=" * 70)
    print(f"\n{'Method':<20} {'Time':<15} {'Steps':<15} {'Speed'}")
    print("-" * 70)
    print(f"{'DDPM':<20} {ddpm_time:>6.2f}s{'':<8} {T:<15} {'1.0x (baseline)'}")
    print(f"{'DDIM':<20} {ddim_time:>6.2f}s{'':<8} {DDIM_STEPS:<15} {f'{ddpm_time/ddim_time:.1f}x faster!'}")
    
    speedup = ddpm_time / ddim_time
    print(f"\n🚀 DDIM is {speedup:.1f}x faster than DDPM!")
    print(f"   DDPM: {T} denoising steps")
    print(f"   DDIM: {DDIM_STEPS} denoising steps")
    
    # 견줌을 갈무리한다
    ddpm_grid = (ddpm_samples + 1) * 0.5
    ddim_grid = (ddim_samples + 1) * 0.5
    
    comparison = torch.cat([ddpm_grid, ddim_grid], dim=0)
    torchvision.utils.save_image(comparison, "comparison_ddpm_vs_ddim.png", nrow=4)
    
    print(f"\n💾 Saved comparison to: comparison_ddpm_vs_ddim.png")
    print(f"   Top row: DDPM samples")
    print(f"   Bottom row: DDIM samples")
    
    print("\n" + "=" * 70)
    print("💡 KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. DDPM: 느리지만 본디 방법
       - 때 걸음 1000개를 모두 쓴다
       - 걸음마다 아무 잡음을 더한다
       - 품질은 높지만 시간이 걸린다
    
    2. DDIM: 빠르고 정해져 있다
       - 대부분의 때 걸음을 건너뛴다(50개만 쓴다)
       - 정해진 대로(같은 씨앗이면 같은 그림)
       - 품질은 비슷하면서 20배 빠르다!
    
    3. 언제 무엇을 쓰는가:
       - DDPM: 확률 뽑기를 바랄 때
       - DDIM: 빠르기나 되풀이할 수 있음이 필요할 때
       - 둘 다 품질이 비슷하다!
    """)

if __name__ == "__main__":
    main()```

## 2. 논의

방법 견주기의 짜기는 이 마당에 자리 잡은 방식을 따른다. 코드 짜임이 모델 뜻매김과 익히기 논리를 갈라 놓아 부품을 하나씩 고치기 쉽다. 얼개 고르기는 만들어 내는 모델 무리가 많은 실험에서 얻은 배움을 담고 있다.

이 짜기의 핵심에는 수치의 안정을 꼼꼼히 다루기, 고르게 맞추기 재주를 제대로 쓰기, 효율 좋은 셈 결이 든다. 익히기 절차에는 잡음 차례표, 기울기 다루기, 이따금의 따지기가 들며 모두 품질 높은 결과를 내는 데 결정적이다.

이 단원은 이론의 개념이 실제 짜기로 어떻게 옮겨지는지 보이며 만들어 내는 모델의 더 넓은 틀과 이어진다. 여기서 보이는 재주는 만들어 내는 모델이 이룰 수 있는 것의 가장자리를 넓히는 더 앞선 변형과 넓힘을 이해하는 바탕이 된다.

## 연습문제

**연습문제 1.**
구체적인 자료 묶음으로 이 단원의 으뜸 셈을 좇아라. 큰 걸음마다 텐서 꼴을 적고 모든 차원이 서로 맞는지 확인하라.

??? success "연습문제 1 풀이"
    모델에 알맞은 꼴의 들임 묶음에서 시작한다. 층이나 함수 부르기마다 셈을 따라가며 바뀜 뒤 텐서 꼴을 적는다. 겹말기 층에서는 내놓기 차원 공식을 쓴다. 눈길 얼개에서는 물음, 열쇠, 값의 차원이 맞는지 확인한다. 마지막 내놓기 꼴이 바라던 목표 차원과 맞는지 굳힌다. 이 익힘은 자료가 얼개를 어떻게 흐르는지에 대한 직관을 쌓아 준다.

---

**연습문제 2.**
이 단원에 쓰인 손실 함수를 가려내고 모델 매개변수에 대한 기울기를 이끌어 내라. 왜 이 손실 함수가 이 일에 알맞은지 설명하라.

??? success "연습문제 2 풀이"
    손실 함수는 모델이 헤아린 값과 목표 사이의 어긋남을 잰다. 잡음 헤아리기에서는 평균 제곱 어긋남 손실 $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$을 쓰는데, 이것이 로그 가능도의 변분 아래 한계에 맞물리기 때문이다. 매개변수 $\theta$에 대한 기울기는 $-2(\epsilon - \epsilon_\theta) \nabla_\theta \epsilon_\theta$이며 헤아림 어긋남을 줄이는 방향을 가리킨다. 이 손실을 가장 작게 하는 것이 퍼짐 모델에서 자료 로그 가능도의 아래 한계를 가장 크게 하는 것과 같으므로 알맞다.

---

**연습문제 3.**
다른 잡음 차례표를 받쳐 주도록 이 짜기를 고쳐라(예컨대 선형에서 코사인으로, 또는 그 반대로). 두 차례표의 익히기 움직임과 표본 품질을 견주어라.

??? success "연습문제 3 풀이"
    두 차례표를 모두 짜고 각각으로 모델을 익힌다. $\bar{\alpha}_t = \cos^2\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$으로 뜻매김한 코사인 차례표는 선형 차례표 $\beta_t = \beta_{\min} + t(\beta_{\max} - \beta_{\min})/T$에 견주어 잡음이 더 매끄럽게 늘어난다. 손실 곡선을 좇고 일정한 사이마다 표본을 만든다. 코사인 차례표는 신호 대 잡음비가 더 완만하게 줄어 때 걸음에 걸쳐 배움 신호가 더 고르므로 흔히 더 좋은 결과를 낸다.

## 정리하며

**다룬 것** — 방법 견주기

방법 견주기의 짜기는 이 마당에 자리 잡은 방식을 따른다.

고갱이 갈래는 `SimpleUNet`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
