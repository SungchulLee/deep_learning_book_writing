# 조건부 변분 자기 부호기

조건부 변분 자기 부호기(cVAE)는 부호기와 풀개를 모두 갈래 이름표에 조건 지어 여느 변분 자기 부호기를 넓히고, 그래서 만들어 내기를 다스릴 수 있게 한다. 갈래 이름표가 주어지면 그 갈래의 다양한 표본을 만들 수 있고 하나만 뜨거운 이름표 벡터를 섞어 갈래 사이를 사이 끼움할 수도 있다. 특정 숫자 갈래 만들기, 모양새 옮기기, 자료 부풀리기 같은 쓰임새에 쓸모 있다.

## 1. 코드

```python
"""조건부 변분 자기 부호기(cVAE)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalVAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=32, num_classes=10):
        super().__init__()
        self.input_dim, self.latent_dim, self.num_classes = input_dim, latent_dim, num_classes
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid()
        )

    def encode(self, x, c):
        return self.fc_mu(self.encoder(torch.cat([x, c], dim=1))), \
               self.fc_logvar(self.encoder(torch.cat([x, c], dim=1)))

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z, c):
        return self.decoder(torch.cat([z, c], dim=1))

    def forward(self, x, labels):
        c = F.one_hot(labels, self.num_classes).float() if len(labels.shape) == 1 else labels
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

    def sample(self, class_label, num_samples, device='cpu'):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        labels = torch.tensor([class_label] * num_samples).to(device)
        c = F.one_hot(labels, self.num_classes).float()
        return self.decode(z, c)

if __name__ == '__main__':
    model = ConditionalVAE()
    x = torch.randn(32, 784)
    labels = torch.randint(0, 10, (32,))
    reconstruction, mu, logvar = model(x, labels)
    print(f"Reconstruction shape: {reconstruction.shape}")
```

## 2. 논의

조건 짓는 장치는 곧바르다. 곧 하나만 뜨거운 이름표를 부호기와 풀개의 들임에 잇는다. 부호기에서는 그물이 갈래마다 다른 부호화 분포를 배우게 돕는다. 풀개에서는 뽑은 숨은 부호로 옳은 숫자 갈래를 만드는 데 필요한 갈래 정체를 준다. 그래서 숨은 공간이 갈래 사이 흔들림(어느 숫자인가)이 아니라 갈래 안의 흔들림(모양새, 기울기, 굵기)을 잡는다.

`sample` 방법이 조건부 만들어 내기의 핵심 이점을 보인다. 곧 갈래 이름표를 정하고 $z \sim \mathcal{N}(0, I)$을 뽑으면 바라는 어떤 갈래의 다양한 것이든 만들 수 있다. `interpolate` 방법은 하나만 뜨거운 이름표를 선형으로 섞어 갈래 사이 매끄러운 옮아감(예컨대 "3"이 "8"로 바뀌는 것)을 낸다.

결정적인 설계 세부 하나는 부호기가 익히는 동안 이름표를 받는다는 것이다. 그래서 부호기가 이름표로 자료의 일부를 설명하며 "속임수"를 써서 앎이 덜 담긴 숨은 나타냄을 배울 수 있다. 어떤 얼개는 풀개만 조건 지어 숨은 공간이 모든 흔들림을 잡게 하지만 그러면 익히기가 더 어렵다.

## 연습문제

**연습문제 1.**
MNIST에 조건부 변분 자기 부호기를 익히고 숫자마다(0~9) 표본 10개를 만들어라. 가로줄마다 한 숫자 갈래가 되도록 10x10 격자로 보여라. 만든 표본이 목표 갈래와 맞는가?

??? success "연습문제 1 풀이"
    ```python
    model.eval()
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for digit in range(10):
        samples = model.sample(digit, 10).view(-1, 28, 28).detach()
        for j in range(10):
            axes[digit, j].imshow(samples[j].numpy(), cmap='gray')
            axes[digit, j].axis('off')
    plt.show()
    ```
    만든 표본이 목표 갈래와 또렷이 맞아야 하며, 이는 풀개가 조건 이름표로 내놓는 숫자를 다스리는 법을 배웠음을 보인다.

---

**연습문제 2.**
`interpolate` 방법으로 갈래 이름표 "3"과 "8" 사이를 사이 끼움하라. 옮아가는 동안 어떤 중간 숫자 모양이 나타나는가?

??? success "연습문제 2 풀이"
    사이 끼움은 흔히 양 끝에서 "3"과 "8"을 닮은 모양을 지나며, 중간 걸음에서는 "8"의 위아래 고리가 차츰 드러나는 섞인 꼴이 보인다. 쓰는 숨은 부호에 따라 중간 걸음이 "5"나 "6"을 닮을 수도 있다.

---

**연습문제 3.**
풀개만 조건 짓도록 조건부 변분 자기 부호기를 고쳐라(부호기에서 이름표 잇기를 없앤다). 숨은 공간의 배치와 표본 품질을 여느 조건부 변분 자기 부호기와 견주어라.

??? success "연습문제 3 풀이"
    부호기에 조건이 없으면 숨은 공간이 갈래 정체와 모양새 흔들림을 모두 잡아야 한다. 그러면 흔히 갈래마다 숨은 공간의 뚜렷한 자리를 차지하고(조건 없는 변분 자기 부호기와 비슷하다) 이름표에 조건 지어진 풀개가 또렷한 갈래 신호로 내놓기를 다듬는다. 풀개가 앎이 덜 담긴 숨은 부호를 받으므로 표본 품질이 조금 낮을 수 있지만 숨은 공간은 더 짜임새 있고 풀이하기 쉬워진다.

## 정리하며

**다룬 것** — 조건부 변분 자기 부호기

조건 짓는 장치는 곧바르다.

고갱이 갈래는 `ConditionalVAE`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
