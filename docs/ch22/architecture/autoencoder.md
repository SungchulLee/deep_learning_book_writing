# 자기 부호기

이 단원은 차원 줄이기와 다시 세우기를 위한 말끔하고 다시 쓸 수 있는 `SimpleAutoencoder` 갈래를 정한다. 얼개는 정류 선형 깨어남과 에스자 내놓기를 갖춘 세 층짜리 온전히 이어진 부호기와 풀개 그물을 쓴다. 잡음 없애는 자기 부호기나 변분 자기 부호기 같은 더 복잡한 얼개의 바탕 벽돌 노릇을 하며 `encode`, `decode`, `loss_function` 방법을 준다.

## 코드

```python
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
```

## 논의

`SimpleAutoencoder`은 부호기와 풀개의 층 너비가 맞는 대칭 설계를 따른다(input_dim에서 hidden_dim, hidden_dim, latent_dim으로, 그리고 그 반대). 이 대칭이 꼭 필요하지는 않고 대칭이 아닌 얼개도 잘 되지만, 모델의 담이를 따져 보기 쉬워지고 풀개가 적어도 부호기만큼의 표현력을 갖추게 한다.

평균 제곱 어긋남 손실에서 붙박이인 `'mean'` 대신 `reduction='sum'`을 쓰는 것은 기울기의 잣수에 영향을 주는 일부러 한 고름이다. 합으로 줄이면 손실이 묶음 크기와 들임 차원에 비례하므로 배움 빠르기가 사실상 그 값에 따라 잣수 맞춰진다. 변분 자기 부호기 짜기 여럿이 KL 벌어짐 항(이 또한 흔히 차원에 걸쳐 더한다)과 결을 맞추려 합으로 줄인다.

이 최소 짜기에 묶음 고르게 맞추기, 떨구기, 무게 잦아듦이 없다는 것은 다스려진 바탕으로 쓸 수 있다는 뜻이다. 변분 자기 부호기나 잡음 없애는 자기 부호기로 넓힐 때는 흔히 벌주기와 고르게 맞추기 층을 더하지만, 말끔한 바탕 갈래가 있으면 고침마다의 효과를 따로 떼어 보기 쉽다.

## 연습문제

**연습문제 1.**
갈래에 `get_latent_dim` 속성과 `count_parameters` 방법을 더하라. 그것으로 숨은 차원 8, 32, 128인 모델을 견주어라.

??? success "연습문제 1 풀이"
    ```python
    @property
    def get_latent_dim(self):
        return self.latent_dim

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    for dim in [8, 32, 128]:
        m = SimpleAutoencoder(latent_dim=dim)
        print(f"latent_dim={dim}: {m.count_parameters():,} parameters")
    ```
    병목에 이어진 선형 층이 비례해 커지므로 매개변수 수가 숨은 차원에 따라 선형으로 는다.

---

**연습문제 2.**
평균 제곱 어긋남 손실을 두 값 엇갈린 엔트로피 손실로 갈음하라. 두 값 엇갈린 엔트로피가 옳으려면 들임 자료가 어떤 제약을 채워야 하며, 에스자 내놓기 깨어남이 왜 그것을 보장하는가?

??? success "연습문제 2 풀이"
    두 값 엇갈린 엔트로피는 어림과 목표가 모두 $[0, 1]$에 있어야 하며 이를 베르누이 확률 변수의 확률로 본다. 에스자 깨어남이 풀개의 내놓기를 $(0, 1)$에 옮겨 이 제약을 채운다. 들임 자료도 $[0, 1]$으로 골라야 한다(MNIST 그림은 `ToTensor()` 뒤 그렇게 된다). 이 범위 밖 자료에 두 값 엇갈린 엔트로피를 쓰면 음의 손실이나 NaN 기울기가 나온다.

---

**연습문제 3.**
들임 벡터 둘을 받아 부호화하고 그 숨은 부호를 고르게 벌어진 10개 점에서 선형으로 사이 끼움한 뒤 푸는 `interpolate` 방법을 짜라. MNIST 숫자 둘로 시험하라.

??? success "연습문제 3 풀이"
    ```python
    def interpolate(self, x1, x2, n_steps=10):
        z1 = self.encode(x1.unsqueeze(0))
        z2 = self.encode(x2.unsqueeze(0))
        alphas = torch.linspace(0, 1, n_steps).unsqueeze(1)
        z_interp = (1 - alphas) * z1 + alphas * z2
        return self.decode(z_interp)
    ```
    숫자 사이 사이 끼움이 매끄러우면 숨은 공간이 뜻 있고 이어진 나타냄을 배웠다는 뜻이다. 갑작스러운 옮아감은 숨은 공간에 틈이나 끊김이 있음을 뜻한다.
