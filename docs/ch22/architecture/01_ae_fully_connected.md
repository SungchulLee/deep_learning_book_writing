# 온전히 이어진 자기 부호기

온전히 이어진 자기 부호기는 눌러 담은 자료 나타냄을 배우는 가장 단순한 신경망 얼개이다. 주성분 분석의 선형 쏘기와 달리 자기 부호기는 비선형 깨어남 함수로 더 풍부하고 차원이 낮은 부호를 찾아낸다. 이 보기는 부호기(784에서 32차원)와 풀개(32에서 784로)를 갖춘 온전한 자기 부호기를 세우고 두 값 엇갈린 엔트로피 손실로 MNIST에 익혀 이끌리지 않은 배움의 온전한 물길을 보인다.

## 코드

```python
"""온전히 이어진 자기 부호기."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time

# === 자리매김 ===========================================================
input_dim, hidden_dim, latent_dim = 784, 128, 32
batch_size, learning_rate, num_epochs = 128, 1e-3, 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 모델 ===================================================================
class FullyConnectedAutoencoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return self.decode(self.encode(x))

# === 익히기 ================================================================
model = FullyConnectedAutoencoder(input_dim, hidden_dim, latent_dim).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    for images, _ in train_loader:
        images_flat = images.to(device).view(images.size(0), -1)
        loss = criterion(model(images_flat), images_flat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    pass
```

## 논의

병목 층(32차원)이 그물로 하여금 잡음과 남아도는 것은 버리고 숫자 그림의 요긴한 특징을 잡는 눌러 담은 나타냄을 배우게 한다. 성분 32개를 쓴 주성분 분석과 비슷하지만 비선형 부호기와 풀개는 선형 방법이 놓치는 굽은 다양체 짜임을 잡을 수 있다. 에스자 내놓기 깨어남이 다시 세운 것을 $[0, 1]$에 두어 고른 화소 범위에 맞춘다.

$[0, 1]$ 자료에서는 두 값 엇갈린 엔트로피 손실이 평균 제곱 어긋남보다 낫다. 화소마다 베르누이 확률 변수로 보아 0이나 1에 가까운 화소에 더 센 기울기를 주기 때문이다. 차원을 차츰 줄이는(784에서 128, 64, 32로) 대칭 부호기-풀개 얼개는 갑작스러운 병목보다 가장 좋게 하기 쉬운 매끄러운 눌러 담기 길을 만든다.

익힌 자기 부호기의 숨은 공간은 숫자 그림을 눈에 보이는 닮음으로 갈무리한다. 곧 비슷한 숫자가 함께 무리 지고 숨은 부호 사이를 사이 끼움하면 숫자 모양이 매끄럽게 옮아간다. 다만 변분 자기 부호기와 달리 여느 자기 부호기의 숨은 공간에는 벌주기가 없어 풀었을 때 뜻 없는 것이 나오는 "구멍"이 있을 수 있다.

## 연습문제

**연습문제 1.**
숨은 차원 2, 8, 32, 128로 자기 부호기를 익혀라. 숨은 차원에 대한 시험 다시 세우기 평균 제곱 어긋남을 그리고 얻는 것이 줄어드는 지점을 찾아라.

??? success "연습문제 1 풀이"
    ```python
    for dim in [2, 8, 32, 128]:
        model = FullyConnectedAutoencoder(latent_dim=dim).to(device)
        # ... 20바퀴 익힌다 ...
        # ... 시험 평균 제곱 어긋남을 셈한다 ...
        print(f"latent_dim={dim}: test MSE = {test_mse:.6f}")
    ```
    평균 제곱 어긋남이 2에서 32차원까지 가파르게 떨어진 뒤 평평해진다. 64차원을 넘으면 나아짐이 미미하다. 풀개가 특징 32~64개로도 잘 다시 세울 담이를 갖췄기 때문이다.

---

**연습문제 2.**
두 값 엇갈린 엔트로피 손실을 평균 제곱 어긋남 손실로 갈음해 다시 익혀라. 다시 세운 것의 눈에 보이는 품질을 견주어라. 어느 손실이 더 또렷한 숫자 그림을 내며 왜 그런가?

??? success "연습문제 2 풀이"
    `nn.BCELoss()` 대신 `nn.MSELoss()`을 쓰면 흔히 다시 세운 것이 조금 더 흐릿하다. 두 값 엇갈린 엔트로피는 0과 1 가까이에서 기울기를 더 세게 주어 내놓기를 더 두 값답게(또렷한 검정과 흰색으로) 이끌지만, 평균 제곱 어긋남은 모든 어긋남에 똑같이 벌을 주어 "평균 낸" 회색 화소 값이 많아진다. 두 값에 가까운 그림 자료에서는 대체로 두 값 엇갈린 엔트로피가 더 또렷한 결과를 낸다.

---

**연습문제 3.**
부호기 층에 떨구기(비율 0.3)를 더하고 떨구기가 있을 때와 없을 때의 시험 다시 세우기 어긋남을 견주어라. 이 이끌리지 않은 상황에서 벌주기가 지나치게 맞춰지는 것을 막는 데 도움이 되는가?

??? success "연습문제 3 풀이"
    ```python
    self.encoder = nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(hidden_dim // 2, latent_dim),
    )
    ```
    떨구기를 쓰면 익히기 손실은 흔히 더 높지만 익히기와 시험 손실의 벌어짐이 줄어든다. MNIST 자기 부호기에서는 자료 묶음이 모델에 견주어 커서 지나치게 맞춰짐이 크지 않으므로 떨구기의 덕이 크지 않다. 자료 묶음이 작으면 벌주기 효과가 더 뚜렷할 것이다.
