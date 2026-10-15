# MNIST로 살펴보는 간단한 DDPM

DDPM의 기본 학습은 시간별 노이즈 관측을 만들고 그 관측에 섞인 가우시안 노이즈를 예측하는 회귀다. 이 페이지는 작은 합성곱 네트워크로 학습·역전이의 연결을 보여 준다. 코드의 모델 이름은 SimpleUNet이지만 실제 구조는 축소·확대나 건너뛰기 연결이 없는 세 층 CNN이다.

## 1. 코드

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
# 2️⃣ 확산 과정 도구
# ==========================================
# 확산 모델에서는 그림 x₀에 정규 잡음을 차츰 더한다.
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
        # 배치의 그림마다 때 걸음을 마구잡이로 고른다
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
# 4️⃣ 뒤 확산(뽑기)
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

## 2. 노이즈 예측 네트워크

입력 B×1×28×28은 채널 64의 두 합성곱과 마지막 1채널 합성곱을 거쳐 같은 크기로 돌아온다. 모두 커널 3, stride 1, padding 1이어서 공간 해상도가 바뀌지 않는다. 파라미터는 38,145개이고 출력 한 위치의 입력 수용영역은 7×7이다. U-Net의 다중 해상도 경로와 비교하면 전역 구조를 표현하는 방식이 제한적이다.

시간 t는 B×1×1×1로 바꾸고 1,000으로 나눠 첫 합성곱 출력의 모든 채널·공간 위치에 더한다. T=300이므로 조건값은 0~0.299다. 학습 가능한 시간 임베딩이 아니며 T를 바꿔도 분모 1,000은 바뀌지 않는다. 출력은 노이즈를 예측하는 실수 텐서이므로 이미지처럼 0~1로 제한하지 않는다.

## 3. 정방향과 학습 손실

β를 0.0001~0.02의 선형 목록으로 만들고 ᾱt=∏s≤t(1−βs)를 사용한다.

$$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\qquad \epsilon\sim N(0,I)$$

`forward_diffusion_sample`은 노이즈 관측과 실제 ε를 반환한다. MSE는 배치·채널·픽셀 전체 평균이며 표본별 t를 0~299에서 균등 선택한다. 코드 인덱스 0도 첫 노이즈를 포함해 깨끗한 원본과는 다르다. 누적곱을 함수 호출마다 두 번 계산하지만 같은 고정 스케줄의 값이다.

MNIST는 ToTensor의 0~1 범위이고 레이블은 사용하지 않는다. 60,000개 데이터를 배치 128로 3에포크 순회하므로 에포크당 469번, 총 1,407번 갱신한다. 마지막 배치는 96개다. 출력의 Epoch Loss는 **마지막 배치 하나의 손실**이지 에포크 평균이 아니다. 검증·체크포인트·시드 고정과 학습률 스케줄은 없다.

## 4. 종단 분포와 초기화

T=300의 이 선형 스케줄에서 마지막 누적곱은 약 0.0481이고 원본 신호 진폭 √ᾱ는 약 0.219다. 따라서 마지막 관측은 데이터 신호가 완전히 사라진 표준 정규분포와 같지 않다. sample은 N(0,I)에서 시작하므로 실제 종단 주변 분포와의 근사 차이를 고려해야 한다. 단순히 단계를 줄였다고 동일한 노이즈 종단 조건이 유지되지는 않는다.

## 5. 역전이와 저장 이미지

역평균은 통상적인 노이즈 예측 파라미터화다.

$$\mu_\theta(x_t,t)=\frac1{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)$$

t>0에서는 분산 βt의 잡음을 추가하고 마지막에는 평균만 반환한다. 전방향 사후분산 β̃t를 쓰는 선택과 구분한다. 16개 이미지를 300단계에 걸쳐 함께 생성하므로 모델 호출은 300번이며 각 호출의 배치는 16이다.

샘플러는 no_grad와 eval을 사용하지만 이전 모드를 복원하지 않는다. 마지막 상태를 0~1로 제한하지 않아 표시용 범위를 벗어날 수 있으며 저장 과정의 clipping은 생성 텐서의 분포와 구분한다. 네트워크·β·시간·입력의 장치가 일치해야 한다.

코드의 학습과 생성은 파일 최상위에 있으므로 import해도 실행되는 구조다. 마지막 main 가드의 pass가 이를 막지 않는다. 여기서는 코드를 실행하지 않으며 기존 코드도 수정하지 않는다. 저장 성공 메시지는 생성 파일의 품질·다양성이나 정확한 분포 학습을 검증한 결과가 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 모델이 U-Net과 다른 구조적 특징과 출력 모양을 설명하라.

</div>

??? success "연습문제 1 풀이"

    합성곱 세 개가 모두 같은 해상도에서 작동하고 축소·확대·건너뛰기 연결이 없다. B×1×28×28→B×64×28×28→B×64×28×28→B×1×28×28이다. 세 커널 3의 수용영역은 한 출력 위치당 7×7이다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
T를 1,000에서 300으로 줄이면서 β 시작·끝을 고정하면 종단 노이즈 조건도 같은가?

</div>

??? success "연습문제 2 풀이"

    아니다. 누적곱이 달라져 남는 원본 신호가 바뀐다. 현재 마지막 ᾱ는 약 0.0481로 원본 신호 진폭이 약 0.219 남는다. 종단 사전분포 근사와 학습·생성 조건을 함께 검토해야 하며 단계를 줄이는 것 자체가 올바른 가속 샘플러는 아니다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
출력된 에포크 손실과 저장 성공 메시지로 무엇을 알 수 있고 무엇은 알 수 없는가?

</div>

??? success "연습문제 3 풀이"

    에포크 손실은 마지막 배치의 노이즈 MSE이고, 저장 메시지는 파일 저장 호출이 끝났음을 나타낸다. 전체 평가 손실·생성 품질·모드 포괄성·일반화를 측정한 것은 아니다. 공통 검증 데이터와 생성 조건에서 별도의 평가가 필요하다.



## 정리하며

간단한 DDPM은 노이즈 예측 회귀와 가우시안 역전이를 연결한다. 이 모델은 세 층 CNN이며 짧은 학습·종단 사전 근사와 마지막 배치 손실의 한계를 갖는다. 출력 크기나 저장 성공만으로 생성 품질을 판단하지 않는다.
