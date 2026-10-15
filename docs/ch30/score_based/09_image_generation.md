# 이미지 스코어 학습의 간단한 예제

이미지의 스코어는 픽셀·채널마다 한 성분을 가지므로 입력과 같은 모양의 텐서다. 이 페이지는 MNIST 일부에 고정된 가우시안 노이즈를 더하고 합성곱 네트워크로 조건부 스코어를 회귀하는 학습 루프를 살펴본다. 제목과 달리 새 이미지를 생성하는 샘플러는 포함하지 않는다.

## 1. 코드와 기존 출력

```python
"""
FILE: 09_image_generation.py
어려움: 나아간 단계
걸리는 시간: 5~6시간
미리 알 것: 07-08, 겹말기 신경망, U-Net 구조

학습 목표:
    1. 그림을 위한 U-Net 점수 모델을 짠다
    2. MNIST 숫자 만들어 내기로 익힌다
    3. 미리 헤아리개-고치개 뽑기를 짠다
    4. 셈에서 살필 것을 이해한다

수학 바탕:
    그림에서는 점수 그물이 흔히 U-Net 구조를 쓴다.
    - 인코더: 줄이기 + 특징 뽑기
    - 디코더: 키우기 + 되짓기
    - 건너뛰기 이음: 공간의 앎을 지킨다
    
    점수 s_θ(x, t)은 그림 공간 위의 벡터 마당이다.
"""

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
# 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
# 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
torch.manual_seed(0)

# ========================================================================
# 메인
# ========================================================================


class SimpleUNet(nn.Module):
    """
    작은 그림(28x28)의 점수 나타내기를 위한 단순한 U-Net.
    
    이는 가르치기 위한 짜기이며 실제 코드는 더 복잡할 것이다.
    """
    
    def __init__(self, in_channels=1, base_channels=64, time_emb_dim=128):
        super().__init__()
        
        # 때 임베딩
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # 인코더
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels*2),
            nn.SiLU()
        )
        
        # 가운데
        self.middle = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1),
            nn.GroupNorm(8, base_channels*2),
            nn.SiLU()
        )
        
        # 디코더
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2),
            nn.GroupNorm(8, base_channels),
            nn.SiLU()
        )
        self.dec1 = nn.Conv2d(base_channels, in_channels, 3, padding=1)
    
    def forward(self, x, t):
        # 때 임베딩
        t_emb = self.time_mlp(t.view(-1, 1))
        
        # 부호화
        h1 = self.enc1(x)
        h2 = self.enc2(h1)
        
        # 가운데
        h = self.middle(h2)
        
        # 디코딩
        h = self.dec2(h)
        h = h + h1  # 건너뛰는 이음
        out = self.dec1(h)
        
        return out


def demo_mnist():
    """MNIST으로 점수 모델을 익힌다(단순한 보여 주기)."""
    print("Score-Based Image Generation on MNIST")
    print("=" * 80)
    
    # MNIST 불러오기
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # [-1, 1]으로 잣수를 맞춘다
    ])
    
    dataset = torchvision.datasets.MNIST(
        root='/tmp/mnist', train=True, download=True, transform=transform
    )
    
    # 빠른 보여 주기를 위해 일부만 쓴다
    subset = torch.utils.data.Subset(dataset, range(1000))
    loader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=True)
    
    print(f"Dataset: {len(subset)} images")
    
    # 모델 생성
    model = SimpleUNet(in_channels=1, base_channels=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 단순한 익히기(보여 주려 몇 바퀴만)
    print("\nTraining (demo with few epochs)...")
    for epoch in range(5):
        total_loss = 0
        for images, _ in loader:
            # 잡음을 더한다(단순한 잡음 없애는 점수 맞추기)
            noise = torch.randn_like(images) * 0.5
            noisy_images = images + noise
            
            # 아무 때
            t = torch.rand(len(images))
            
            # 점수 미리보기
            pred_score = model(noisy_images, t)
            target_score = -noise / (0.5 ** 2)
            
            loss = nn.functional.mse_loss(pred_score, target_score)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/5 | Loss: {total_loss/len(loader):.6f}")
    
    print("\n✓ Training complete!")
    print("\nNote: Full training would take several hours on GPU.")
    print("This demo shows the architecture and training loop.")


if __name__ == "__main__":
    demo_mnist()
```

```
Score-Based Image Generation on MNIST
================================================================================
Dataset: 1000 images

Training (demo with few epochs)...
Epoch 1/5 | Loss: 3.266946
Epoch 2/5 | Loss: 1.613611
Epoch 3/5 | Loss: 1.009080
Epoch 4/5 | Loss: 0.788321
Epoch 5/5 | Loss: 0.671718

✓ Training complete!

Note: Full training would take several hours on GPU.
This demo shows the architecture and training loop.
```

## 2. 공간 크기와 건너뛰기 연결

main의 base_channels=32에서는 B×1×28×28 입력이 다음 모양으로 변한다.

| 단계 | 출력 모양 |
|---|---|
| enc1 | B×32×28×28 |
| enc2 | B×64×14×14 |
| middle | B×64×14×14 |
| dec2 | B×32×28×28 |
| enc1 출력과 덧셈 | B×32×28×28 |
| dec1 | B×1×28×28 |

한 번 축소·확대하고 같은 해상도의 특징을 더하는 간단한 U-Net 형태다. 채널을 연결하는 concat 방식과 달리 두 텐서의 모양이 같아야 한다. 홀수 높이·너비에서는 stride 2와 두 배 확대 뒤 원래 크기가 맞지 않을 수 있다. GroupNorm(8,channels)도 채널 수가 8로 나누어지는 구성을 전제로 한다.

## 3. 사용되지 않는 시간 임베딩

forward는 time_mlp로 t_emb를 계산하지만 그 값을 **어느 특징에도 사용하지 않는다**. 따라서 입력 t를 바꿔도 현재 네트워크의 출력은 같으며 time_mlp 파라미터에 손실로부터의 기울기가 전달되지 않는다. 시간 조건 스코어 모델을 구현했다고 설명할 수 없다.

main 설정의 전체 파라미터는 81,409개지만 그중 시간 MLP 16,768개는 출력에 영향을 주지 않는다. 실제 이미지 경로는 64,641개다. 현재 학습은 고정 σ=0.5이므로 임의 t를 생성하더라도 여러 노이즈 수준을 학습하는 것이 아니다.

## 4. DSM 회귀 목표

입력 MNIST는 Normalize(0.5,0.5)로 −1~1에 놓인다. noise는 이미 0.5가 곱해진 가우시안 섭동 η이고 관측은 x̃=x+η다. 따라서 목표 −noise/0.5²는 올바른 조건부 관측 스코어다.

$$\nabla_{\widetilde x}\log q(\widetilde x\mid x)
=-\frac{\widetilde x-x}{0.5^2}$$

MSE는 배치·채널·공간의 모든 원소를 평균 낸다. 모집단 최적 출력은 σ=0.5로 평활화한 데이터 주변 밀도의 스코어다. 출력은 일반 실수값을 가지므로 이미지처럼 0~1로 자르거나 시그모이드를 붙이는 것이 목적은 아니다. 노이즈 관측이 −1~1을 벗어나는 것도 이 가우시안 학습의 일부다.

## 5. 실제 학습 범위

학습 데이터 앞 1,000개만 사용하고 배치 크기 128로 5에포크를 수행한다. 에포크마다 8배치, 전체 40번의 Adam 갱신이며 마지막 배치는 104개다. 에포크 손실은 배치 평균을 동일 가중해 표본 전체 가중 평균과 다르다. 숫자 레이블은 사용하지 않는다.

CPU 모델을 생성하며 CUDA 이동·검증·체크포인트와 샘플 생성은 없다. 특정 GPU에서 전체 학습이 몇 시간 걸린다는 출력 문장은 이 실행의 측정 결과가 아니다. 학습 손실만으로 이미지 품질이나 분포 포괄성을 입증할 수 없으며, 실제 생성으로 확장하려면 학습한 스코어와 맞는 노이즈 수준·샘플러를 별도로 설계해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
28×28 입력의 축소·확대 후 건너뛰기 덧셈이 가능한 이유를 설명하라.

</div>

??? success "연습문제 1 풀이"

    enc1은 28×28, enc2와 middle은 14×14를 유지한다. 커널 2·stride 2의 전치 합성곱으로 28×28과 채널 32를 복원해 enc1 출력과 덧셈할 수 있다. 홀수 크기에서는 같은 복원이 성립하지 않을 수 있다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
time_mlp를 정의하고 호출했는데도 시간 조건 모델이 아닌 이유는 무엇인가?

</div>

??? success "연습문제 2 풀이"

    계산한 t_emb가 이후 출력 계산에 쓰이지 않는다. 따라서 t 변화가 출력에 영향을 주지 않고 시간 MLP 파라미터는 학습 손실의 계산 그래프에 연결되지 않는다. 여러 노이즈 수준을 학습하려면 조건을 실제 특징에 반영하고 그 수준에 맞는 관측·목표를 사용해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 예제의 학습 손실과 학습량만으로 생성 이미지 성능을 평가할 수 있는가?

</div>

??? success "연습문제 3 풀이"

    없다. 고정 σ DSM을 1,000개 이미지에서 40번 갱신하며 생성 샘플러와 독립 평가가 없다. 모델의 목표는 노이즈 주변 스코어이고 원본 이미지 복원의 정확도나 FID와 동일하지 않다. 현재 코드에는 비교할 선형·코사인 노이즈 스케줄도 없다.



## 정리하며

간단한 이미지 DSM에서는 입력·출력 공간 크기와 조건부 노이즈 목표를 맞춘다. 이 코드의 시간 임베딩은 출력에 사용되지 않으며 실제 학습은 한 노이즈 수준이다. 학습 루프 시연과 생성·품질 평가를 구분한다.
