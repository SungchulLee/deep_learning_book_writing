# 잡음 없애는 자기 부호기

잡음 없애는 자기 부호기(DAE)는 망가뜨린 들임에서 깨끗한 그림을 다시 세우도록 익혀, 그물이 신호와 잡음을 가르는 튼튼한 나타냄을 배우게 한다. 뻔한 항등 대응을 배울 수 있는 여느 자기 부호기와 달리 잡음 없애는 자기 부호기는 잡음을 없애려 뜻 있는 특징을 뽑아야 한다. 이 보기는 정규, 소금과 후추, 떨구기 잡음의 망가뜨리기 전략을 짜고 최대 신호 대 잡음비로 잡음 없애기 성능을 값매김한다.

## 코드

```python
"""잡음 없애는 자기 부호기."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import time

noise_type, noise_factor = 'gaussian', 0.3
batch_size, learning_rate, num_epochs = 128, 1e-3, 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def add_noise(images, noise_type='gaussian', noise_factor=0.3):
    if noise_type == 'gaussian':
        return torch.clamp(images + torch.randn_like(images) * noise_factor, 0., 1.)
    elif noise_type == 'salt_pepper':
        noisy = images.clone()
        mask = torch.rand_like(images)
        noisy[mask < noise_factor / 2] = 1.0
        noisy[(mask >= noise_factor / 2) & (mask < noise_factor)] = 0.0
        return noisy
    elif noise_type == 'dropout':
        return images * (torch.rand_like(images) > noise_factor).float()

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = DenoisingAutoencoder().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    for clean_images, _ in train_loader:
        clean_images = clean_images.to(device)
        noisy_images = add_noise(clean_images, noise_type, noise_factor)
        loss = criterion(model(noisy_images), clean_images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    pass
```

## 논의

여느 자기 부호기와의 핵심 차이는 들임은 망가뜨리되 목표는 깨끗이 둔다는 것이다. 곧 손실이 그물의 내놓기를 잡음 낀 들임이 아니라 본디 그림과 견준다. 그래서 부호기가 더한 잡음은 무시하고 바탕 숫자 짜임을 잡는 특징을 배우게 된다. 그렇게 얻은 나타냄은 흔히 여느 자기 부호기의 것보다 뒤따르는 일에 더 쓸모 있다.

세 잡음 갈래가 실제 세상의 서로 다른 망가짐을 나타낸다. 정규 잡음은 사진기와 과학 기기의 감지기 잡음을 흉내낸다. 소금과 후추 잡음은 디지털 그림의 죽은 화소와 전송 어긋남을 나타낸다. 떨구기 잡음(화소를 아무렇게나 0으로 만들기)은 가려짐과 빠진 자료를 흉내낸다. 잘 익힌 잡음 없애는 자기 부호기는 익힌 잡음 수준은 다루지만 다시 익히지 않으면 못 본 잡음 갈래에는 잘 통하지 않는다.

성능은 최대 신호 대 잡음비(PSNR)로 재며 $[0, 1]$ 신호에서 $\text{PSNR} = -10 \log_{10}(\text{MSE})$으로 정의한다. 잡음 낀 들임보다 흔히 5~10 dB 나아지며, 이는 다시 세운 그림이 훨씬 깨끗하다는 뜻이다. 실전 잡음 없애기에는 건너뛰는 이음을 갖춘 U-그물 같은 더 정교한 얼개가 대체로 여기 쓴 단순한 부호기-풀개 짜임보다 낫다.

## 연습문제

**연습문제 1.**
정규, 소금과 후추, 떨구기 잡음마다 따로 잡음 없애는 자기 부호기 셋을 익혀라. 모델마다 세 잡음 갈래로 모두 시험해 최대 신호 대 잡음비 나아짐의 3x3 표를 만들어라. 어느 모델이 잡음 갈래에 걸쳐 가장 두루 통하는가?

??? success "연습문제 1 풀이"
    모델마다 제 잡음 갈래로 익힌 뒤 셋 모두로 값매김한다. 정규 잡음이 가장 "두루 쓰이는" 망가뜨림이므로 정규로 익힌 모델이 흔히 다른 갈래에도 그런대로 통한다. 소금과 후추 모델은 특화되기 쉬워 정규 잡음에서 나쁘고, 떨구기로 익힌 모델은 갈래를 넘나드는 두루 통함이 어중간하다.

---

**연습문제 2.**
잡음 인수를 0.3에서 0.5, 0.7로 올려라. 어느 잡음 수준에서 잡음 없애는 자기 부호기가 알아볼 만한 다시 세우기를 못 내는가? 잡음 수준에 대한 평균 최대 신호 대 잡음비 나아짐을 그려라.

??? success "연습문제 2 풀이"
    ```python
    for nf in [0.1, 0.3, 0.5, 0.7, 0.9]:
        # noise_factor=nf로 잡음 없애는 자기 부호기를 익히고 최대 신호 대 잡음비 나아짐을 셈한다
        print(f"noise_factor={nf}: PSNR improvement = {improvement:.1f} dB")
    ```
    최대 신호 대 잡음비 나아짐은 흔히 잡음 인수 0.3~0.5쯤에서 가장 크다. 0.7 이상이면 들임이 너무 망가져 잡음 없애는 자기 부호기가 숫자 짜임을 믿을 만하게 알아내지 못하고 다시 세운 것이 흐릿한 평균이 된다. dB 나아짐은 여전히 양수일 수 있으나 절대 품질은 나쁘다.

---

**연습문제 3.**
부호기에서 풀개로 건너뛰는 이음을 더해(U-그물 꼴 얼개를 만들어) 여느 잡음 없애는 자기 부호기와 성능을 견주어라. 건너뛰는 이음이 잡음 없애기에 왜 도움이 되는가?

??? success "연습문제 3 풀이"
    건너뛰는 이음은 풀개가 앎 병목을 우회해 부호기 앞선 층의 해상도 높은 공간 세부에 닿게 한다. 잡음 없애기에 이로운 까닭은 부호기 앞선 층에 또렷한 모서리와 결을 다시 세우는 데 도움이 되는 잔 공간 앎이 있고, 병목은 신호와 잡음을 가르는 데 필요한 뜻을 잡기 때문이다. U-그물 얼개는 흔히 수수한 부호기-풀개 잡음 없애는 자기 부호기보다 최대 신호 대 잡음비를 1~3 dB 높인다.
