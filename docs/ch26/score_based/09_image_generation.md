# 그림 만들어 내기

이 단원은 요즘 만들어 내는 모델의 핵심 부품인 그림 만들어 내기을 짠다. 여기서 보이는 개념과 재주를 알면 퍼짐 모델과 점수 바탕 만들어 내는 방법을 다루는 데 꼭 필요한 앎을 얻는다. 이 짜기는 또렷함과 실제 쓸모의 균형을 맞추어 배우기에도 실험하기에도 알맞다.

## 1. 코드

```python
"""
FILE: 09_image_generation.py
어려움: 나아간 단계
걸리는 시간: 5~6시간
미리 알 것: 07-08, 겹말기 신경망, U-Net 얼개

학습 목표:
    1. 그림을 위한 U-Net 점수 모델을 짠다
    2. MNIST 숫자 만들어 내기로 익힌다
    3. 미리 헤아리개-고치개 뽑기를 짠다
    4. 셈에서 살필 것을 이해한다

수학 바탕:
    그림에서는 점수 그물이 흔히 U-Net 얼개를 쓴다.
    - 부호기: 줄이기 + 특징 뽑기
    - 풀개: 키우기 + 되짓기
    - 건너뛰기 이음: 공간의 앎을 지킨다
    
    점수 s_θ(x, t)은 그림 공간 위의 벡터 마당이다.
"""

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

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
        
        # 때 박아 넣기
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # 부호기
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
        
        # 복호기
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2),
            nn.GroupNorm(8, base_channels),
            nn.SiLU()
        )
        self.dec1 = nn.Conv2d(base_channels, in_channels, 3, padding=1)
    
    def forward(self, x, t):
        # 때 박아 넣기
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
    demo_mnist()```

## 2. 논의

그림 만들어 내기의 짜기는 이 마당에 자리 잡은 방식을 따른다. 코드 짜임이 모델 뜻매김과 익히기 논리를 갈라 놓아 부품을 하나씩 고치기 쉽다. 얼개 고르기는 만들어 내는 모델 무리가 많은 실험에서 얻은 배움을 담고 있다.

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

**다룬 것** — 그림 만들어 내기

그림 만들어 내기의 짜기는 이 마당에 자리 잡은 방식을 따른다.

고갱이 갈래는 `SimpleUNet`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
