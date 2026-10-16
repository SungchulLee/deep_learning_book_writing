# DDPM과 DDIM의 샘플링 호출 비교

같은 노이즈 예측기를 사용해도 샘플러에 따라 호출 수와 확률성이 달라진다. 이 예제는 1,000단계 DDPM과 50단계 결정적 DDIM의 실행 경로를 비교한다. 다만 모델을 학습하거나 체크포인트를 불러오지 않으므로 생성 품질 비교 실험은 아니다.

## 1. 코드와 보존된 출력

```python
"""방법을 견준다."""
# ==========================================
# 확산 방법 견주기 보여 주기
# DDPM과 DDIM의 뽑기 빠르기를 견준다
# ==========================================
import torch
import torch.nn as nn
import torchvision
import time
from tqdm import tqdm
# 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
# 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
torch.manual_seed(0)

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
    main()
```

```
======================================================================
🔬 DIFFUSION METHODS COMPARISON
======================================================================

📍 Using device: cpu

📊 Configuration:
   Total timesteps: 1000
   DDIM steps: 50
   Speedup factor: 20.0x

======================================================================
Method 1: DDPM (Original)
======================================================================
⏱️  Time: 79.89s

======================================================================
Method 2: DDIM (Fast Sampling)
======================================================================
⏱️  Time: 3.87s

======================================================================
📈 RESULTS
======================================================================

Method               Time            Steps           Speed
----------------------------------------------------------------------
DDPM                  79.89s         1000            1.0x (baseline)
DDIM                   3.87s         50              20.7x faster!

🚀 DDIM is 20.7x faster than DDPM!
   DDPM: 1000 denoising steps
   DDIM: 50 denoising steps

💾 Saved comparison to: comparison_ddpm_vs_ddim.png
   Top row: DDPM samples
   Bottom row: DDIM samples

======================================================================
💡 KEY TAKEAWAYS
======================================================================

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
    
```

출력의 79.89초와 3.87초는 기존 실행 기록이며 이번 편집에서 재측정한 값이 아니다. 약 20.7배의 비율은 그 실행 환경의 관측값이다. 출력에 포함된 “품질이 비슷하다”는 문장은 이 코드로 검증되지 않는다.

## 2. 비교에 사용하는 모델

SimpleUNet이라는 이름과 달리 여기서는 3×3 컨볼루션 세 개만 사용한다. 채널은 1→64→64→1이고 축소·확대 경로나 skip 연결이 없다. 입력과 출력은 B×1×28×28이며 시간 t/1000을 첫 피처 맵에 더한다. 파라미터는 38,145개다.

main은 이 모델을 무작위로 초기화한 뒤 바로 샘플러에 전달한다. 데이터셋, optimizer, 학습 로스나 저장 웨이트가 없다. 따라서 이미지 모양의 텐서가 반환되더라도 학습된 MNIST 분포에서 생성한 숫자라고 설명할 수 없다.

## 3. DDPM 경로

코사인 스케줄의 1,000개 인덱스를 999부터 0까지 모두 사용한다. 모델 호출은 단계마다 한 번이며, 인덱스 0을 제외한 단계에서 분산 βₜ의 노이즈를 추가한다. 이는 조건부 사후분산 β̃ₜ를 사용하는 변형과 구분된다.

alphas_cumprod를 매 반복마다 다시 계산하므로 스케줄 계산도 불필요하게 반복된다. 모델 호출 수뿐 아니라 이 구현상의 차이가 측정 시간에 들어간다. DDPM 쪽은 x̂₀ 자르기가 없고 DDIM 쪽은 있으므로 역전이 구성도 동일한 조건이 아니다.

## 4. DDIM 경로와 끝점

T=1000, ddim_steps=50이면 간격 c=20이며 사용하는 인덱스는 980,960,…,0이다. **최대 학습 인덱스 999는 포함하지 않는다.** 표준정규 초기값이 해당 시작 단계의 주변분포를 얼마나 잘 근사하는지도 확인해야 한다.

중간 노이즈를 넣지 않는 η=0 식으로 x̂₀와 예측 노이즈 방향을 결합한다. 하지만 마지막 이전 인덱스를 −1 같은 깨끗한 끝점이 아니라 **0으로 지정**해 alpha_prev=1로 처리하는 분기는 사용되지 않는다. 마지막 t=0→0은 자르기가 없으면 항등 업데이트이고, 자르기가 있어도 명시적인 깨끗한 끝점으로의 전이는 아니다.

일반적인 ddim_steps에서 실제 호출 수가 인수와 같지도 않다. T=1000, S=60이면 c=16이고 0부터 992까지 63개를 선택한다. S=0이나 S>T는 나눗셈 또는 간격 0 문제를 일으킨다. 출력 표의 단계 수를 무조건 실제 반복 수로 간주하지 않는다.

## 5. 시간·재현성·저장 이미지 해석

기본 조건에서 모델 호출 수는 배치당 1,000회 대 50회다. 한 호출이 동일 비용이고 부가 비용이 없다고 가정하면 호출 수 비율은 20이다. 실제 시간에는 진행 표시, 스케줄 계산, 장치 준비와 커널 실행이 포함된다. GPU에서는 비동기 연산과 동기화 위치도 영향을 주므로 time.time만으로 얻은 값을 정밀한 장치 벤치마크로 확대하지 않는다.

파일 상단의 manual_seed는 전체 난수열의 시작을 정하지만 두 샘플러는 연달아 서로 다른 초기 노이즈를 뽑는다. 따라서 위·아래 행이 같은 초기점에서의 대응 결과는 아니다. η=0도 모델 상태와 초기값이 고정되어야 같은 결과를 기대할 수 있다.

저장 이미지는 두 배치에 (x+1)/2를 적용해 위에는 DDPM, 아래에는 DDIM 네 개씩 배치한다. 저장 과정의 표시 범위 제한이 원래 텐서의 범위나 품질을 입증하지 않는다. 품질을 비교하려면 먼저 학습된 동일 웨이트와 적절한 끝점·평가 절차를 갖추어야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 파일이 DDPM과 DDIM의 MNIST 생성 품질을 비교한 실험인가?

</div>

??? success "연습문제 1 풀이"

    아니다. 모델은 컨볼루션 세 개로 된 무작위 초기화 CNN이고 학습·체크포인트 로드가 없다. 출력 모양과 실행 시간은 살필 수 있지만 숫자 생성 품질이나 분포 일치를 입증할 수 없다. 두 경로의 x̂₀ 자르기 조건도 다르다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
T=1000에서 S=50과 S=60의 실제 DDIM 인덱스 수를 구하라.

</div>

??? success "연습문제 2 풀이"

    S=50이면 c=20으로 0,20,…,980의 50개다. S=60이면 c=16으로 0,16,…,992의 63개다. 둘 다 999를 포함하지 않으며 마지막 이전 인덱스가 0이어서 깨끗한 끝점 계수 1로의 전이도 빠져 있다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
출력의 20.7배와 설정의 20배는 각각 무엇을 뜻하는가?

</div>

??? success "연습문제 3 풀이"

    20.7배는 보존된 실행 시간 79.89/3.87의 근삿값이고, 20배는 기본 설정의 모델 호출 수 1000/50이다. 실제 시간에는 반복 계수 계산·장치·부가 비용이 포함된다. 어느 수치도 학습된 모델의 품질 동등성이나 모든 환경에서의 속도 향상을 보장하지 않는다.



## 정리하며

이 예제는 무작위 CNN으로 두 샘플링 루프의 호출 수와 기존 시간 기록을 비교한다. 품질을 검증하지 않으며, DDIM의 실제 인덱스 수·시작점·깨끗한 끝점 누락과 서로 다른 자르기 조건을 확인해야 한다.
