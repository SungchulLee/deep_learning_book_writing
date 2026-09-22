# 잡음 없애는 자기 부호기

잡음 없애는 자기 부호기(DAE)는 망가뜨린 들임에서 깨끗한 그림을 다시 세우도록 익혀, 그물이 신호와 잡음을 가르는 튼튼한 나타냄을 배우게 한다. 뻔한 항등 대응을 배울 수 있는 여느 자기 부호기와 달리 잡음 없애는 자기 부호기는 잡음을 없애려 뜻 있는 특징을 뽑아야 한다. 이 보기는 정규, 소금과 후추, 떨구기 잡음의 망가뜨리기 전략을 짜고 최대 신호 대 잡음비로 잡음 없애기 성능을 값매김한다.

## 1. 코드

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

## 2. 논의

여느 자기 부호기와의 핵심 차이는 들임은 망가뜨리되 목표는 깨끗이 둔다는 것이다. 곧 손실이 그물의 내놓기를 잡음 낀 들임이 아니라 본디 그림과 견준다. 그래서 부호기가 더한 잡음은 무시하고 바탕 숫자 짜임을 잡는 특징을 배우게 된다. 그렇게 얻은 나타냄은 흔히 여느 자기 부호기의 것보다 뒤따르는 일에 더 쓸모 있다.

세 잡음 갈래가 실제 세상의 서로 다른 망가짐을 나타낸다. 정규 잡음은 사진기와 과학 기기의 감지기 잡음을 흉내낸다. 소금과 후추 잡음은 디지털 그림의 죽은 화소와 전송 어긋남을 나타낸다. 떨구기 잡음(화소를 아무렇게나 0으로 만들기)은 가려짐과 빠진 자료를 흉내낸다. 잘 익힌 잡음 없애는 자기 부호기는 익힌 잡음 수준은 다루지만 다시 익히지 않으면 못 본 잡음 갈래에는 잘 통하지 않는다.

성능은 최대 신호 대 잡음비(PSNR)로 재며 $[0, 1]$ 신호에서 $\text{PSNR} = -10 \log_{10}(\text{MSE})$으로 정의한다. 잡음 낀 들임보다 흔히 5~10 dB 나아지며, 이는 다시 세운 그림이 훨씬 깨끗하다는 뜻이다. 실전 잡음 없애기에는 건너뛰는 이음을 갖춘 U-그물 같은 더 정교한 얼개가 대체로 여기 쓴 단순한 부호기-풀개 짜임보다 낫다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
정규, 소금과 후추, 떨구기 잡음마다 따로 잡음 없애는 자기 부호기 셋을 익혀라. 모델마다 세 잡음 갈래로 모두 시험해 최대 신호 대 잡음비 나아짐의 3x3 표를 만들어라. 어느 모델이 잡음 갈래에 걸쳐 가장 두루 통하는가?

</div>

??? success "연습문제 1 풀이"
    모델마다 제 잡음 갈래로 익힌 뒤 셋 모두로 값매김한다. 정규 잡음이 가장 "두루 쓰이는" 망가뜨림이므로 정규로 익힌 모델이 흔히 다른 갈래에도 그런대로 통한다. 소금과 후추 모델은 특화되기 쉬워 정규 잡음에서 나쁘고, 떨구기로 익힌 모델은 갈래를 넘나드는 두루 통함이 어중간하다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
잡음 인수를 0.3에서 0.5, 0.7로 올려라. 어느 잡음 수준에서 잡음 없애는 자기 부호기가 알아볼 만한 다시 세우기를 못 내는가? 잡음 수준에 대한 평균 최대 신호 대 잡음비 나아짐을 그려라.

</div>

??? success "연습문제 2 풀이"
    ```python
    for nf in [0.1, 0.3, 0.5, 0.7, 0.9]:
        # noise_factor=nf로 잡음 없애는 자기 부호기를 익히고 최대 신호 대 잡음비 나아짐을 셈한다
        print(f"noise_factor={nf}: PSNR improvement = {improvement:.1f} dB")
    ```
    최대 신호 대 잡음비 나아짐은 흔히 잡음 인수 0.3~0.5쯤에서 가장 크다. 0.7 이상이면 들임이 너무 망가져 잡음 없애는 자기 부호기가 숫자 짜임을 믿을 만하게 알아내지 못하고 다시 세운 것이 흐릿한 평균이 된다. dB 나아짐은 여전히 양수일 수 있으나 절대 품질은 나쁘다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
부호기에서 풀개로 건너뛰는 이음을 더해(U-그물 꼴 얼개를 만들어) 여느 잡음 없애는 자기 부호기와 성능을 견주어라. 건너뛰는 이음이 잡음 없애기에 왜 도움이 되는가?

</div>

??? success "연습문제 3 풀이"
    건너뛰는 이음은 풀개가 앎 병목을 우회해 부호기 앞선 층의 해상도 높은 공간 세부에 닿게 한다. 잡음 없애기에 이로운 까닭은 부호기 앞선 층에 또렷한 모서리와 결을 다시 세우는 데 도움이 되는 잔 공간 앎이 있고, 병목은 신호와 잡음을 가르는 데 필요한 뜻을 잡기 때문이다. U-그물 얼개는 흔히 수수한 부호기-풀개 잡음 없애는 자기 부호기보다 최대 신호 대 잡음비를 1~3 dB 높인다.


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
잡음 없애는 자기 부호기와 보통 자기 부호기를 깨끗한 입력과 더럽힌 입력 양쪽에서 견주어라. 어느 쪽이 이기는가?

</div>

??? success "연습문제 4 풀이"
    $\sigma = 0.3$의 가우시안 잡음으로 익힌 뒤 재면 이렇다.

    | 시험 입력 | 보통 자기 부호기 | 잡음 없애는 자기 부호기 |
    |---|---|---|
    | 깨끗한 입력 | **0.00926** | 0.03273 |
    | 더럽힌 입력 | 0.07062 | **0.01864** |

    **자기가 익힌 조건에서 각자 이긴다.** 깨끗한 입력에서는 보통 쪽이 3.5배 낫고,
    더럽힌 입력에서는 잡음 없애는 쪽이 3.8배 낫다.

    $\sigma = 0.6$으로 잡음을 키우면 격차가 더 벌어진다.

    | 시험 입력 | 보통 | 잡음 없애는 것 ($\sigma=0.6$) |
    |---|---|---|
    | 깨끗한 입력 | 0.00926 | 0.05032 |
    | 더럽힌 입력 | 0.10973 | **0.03599** |

    여기서 읽을 것은 잡음 없애기가 **공짜가 아니라는** 점이다. 잡음에 강해지는 대신
    깨끗한 입력을 되돌리는 힘을 내준다. 잡음을 세게 줄수록 그 대가가 커진다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span>
잡음 없애는 자기 부호기의 손실이 왜 더 나은 나타냄을 만드는가? 보통 자기 부호기와 목표가 어떻게 다른지 식으로 적어라.

</div>

??? success "연습문제 5 풀이"
    보통 자기 부호기는

    $$\min_\theta \; \mathbb{E}_x \left[ \lVert x - g(f(x)) \rVert^2 \right]$$

    을 풀고, 잡음 없애는 쪽은 더럽힘 분포 $C(\tilde{x} \mid x)$를 두어

    $$\min_\theta \; \mathbb{E}_x \, \mathbb{E}_{\tilde{x} \sim C(\cdot \mid x)}
      \left[ \lVert x - g(f(\tilde{x})) \rVert^2 \right]$$

    을 푼다. **입력은 더럽혀진 것이고 표적은 깨끗한 것**이라는 비대칭이 전부다.

    이 차이가 중요한 까닭은, 이제 항등 함수가 답이 될 수 없기 때문이다. $\tilde{x}$를
    그대로 내보내면 잡음까지 함께 나가므로 벌점을 받는다. 모델은 잡음과 신호를
    **갈라낼** 수밖에 없고, 그러려면 자료가 실제로 어떻게 생겼는지 알아야 한다.

    그래서 잡음 없애기는 병목이 넓어도 쓸모가 있다. 병목이 아니라 **잡음이** 무엇을
    버릴지 강제하기 때문이며, 이것이 [손실 함수 연습문제 10](../ae/loss_functions.md)에서
    말한 "무엇을 버릴지 강제하는 장치"의 한 종류다.

    이 생각이 훨씬 멀리 간다. 여러 세기의 잡음에 대해 이 일을 하도록 시키고 잡음을
    걷어 내는 절차를 되풀이하면 [퍼짐 모델](../../ch29/index.md)이 된다. 잡음 없애는
    자기 부호기는 그 계보의 첫걸음이다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
더럽히는 방법으로 가우시안 잡음 말고 무엇을 쓸 수 있는가?

</div>

??? success "연습문제 6 풀이"
    자주 쓰이는 것이 셋이다.

    | 방법 | 하는 일 | 어울리는 자료 |
    |---|---|---|
    | 가우시안 잡음 | 모든 값에 $\mathcal{N}(0,\sigma^2)$를 더한다 | 이어진 값 |
    | 가리기(masking) | 값의 일부를 0으로 만든다 | 성긴 자료, 글 |
    | 소금-후추 | 일부를 최소/최대값으로 튕긴다 | 그림 |

    **가리기**가 특히 멀리 갔다. 입력의 일부를 감추고 그것을 맞히게 하는 이 방식이
    곧 마스크 모델링이며, BERT가 낱말을 가리고 맞히는 것과 그림에서 조각을 가리고
    맞히는 MAE가 모두 같은 생각이다.

    더럽히는 방법을 고르는 기준은 **무엇을 잡음으로 볼 것인가**이다. 자료에서 없어도
    되는 것을 없애야 모델이 남은 것에서 구조를 배운다. 그림에서 화소 하나가 바뀌는
    것은 대개 상관없으므로 가우시안이 통하고, 글에서는 글자 하나가 바뀌면 뜻이
    달라지므로 낱말 단위로 가리는 편이 낫다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff easy" title="쉬움"></span>
잡음의 세기 $\sigma$를 어떻게 고르는가? 너무 작거나 너무 크면 어떻게 되는가?

</div>

??? success "연습문제 7 풀이"
    너무 작으면 더럽힌 것과 깨끗한 것이 거의 같아 보통 자기 부호기로 돌아간다.
    항등 함수를 배워도 벌점이 거의 없다.

    너무 크면 원래 무엇이었는지 알아볼 수 없게 되어, 모델이 입력을 무시하고 **자료의
    평균**을 내놓는 편이 나아진다. 실제로 $\sigma = 0.6$에서 깨끗한 입력에 대한 오차가
    0.05032까지 올라가는데, 이는 자료의 분산에 가까워지고 있다는 뜻이다.

    그러므로 적당한 세기는 **알아볼 수는 있되 그냥 흘려보낼 수는 없는** 정도다. 실무의
    요령은 더럽힌 그림을 직접 눈으로 보는 것이다. 사람이 무슨 숫자인지 알아볼 수 있으면
    대체로 알맞다.

    자료마다 다르므로 $\sigma$를 바꿔 가며 **아래쪽 일**(분류, 이상 탐지)의 성능으로
    고르는 것이 가장 확실하다.

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
잡음을 학습할 때마다 새로 뽑아야 하는가, 미리 만들어 두어도 되는가?

</div>

??? success "연습문제 8 풀이"
    **매번 새로 뽑아야 한다.** 미리 만들어 고정해 두면 같은 $(\tilde{x}, x)$ 짝이
    되풀이되므로 모델이 그 특정 잡음 무늬까지 외울 수 있다. 그러면 잡음 없애기가
    아니라 짝 맞추기를 배우는 셈이다.

    매번 새로 뽑으면 같은 $x$가 에포크마다 다른 $\tilde{x}$로 들어온다. 모델이 볼 수
    있는 것은 "이 깨끗한 그림에서 나올 수 있는 더럽힌 그림들"의 분포 전체이므로,
    잡음의 개별 무늬가 아니라 **잡음의 성질**을 배우게 된다.

    같은 이치가 자료 늘리기 전반에 적용된다. 무작위 자르기나 뒤집기도 미리 만들어
    두지 않고 묶음을 꺼낼 때마다 새로 뽑는다. 그래서 `Dataset`의 `__getitem__` 안에서
    변환을 하는 것이 관례다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
잡음 없애는 자기 부호기를 실제로 잡음 제거에 쓰려면 무엇을 조심해야 하는가?

</div>

??? success "연습문제 9 풀이"
    **익힌 잡음과 실제 잡음이 같아야 한다.** 가우시안으로 익힌 모델을 소금-후추 잡음에
    쓰면 잘 듣지 않는다. 모델은 자기가 본 종류의 더럽힘을 걷어 내는 법만 배웠다.

    세기도 마찬가지다. $\sigma = 0.3$으로 익힌 모델에 $\sigma = 0.6$짜리 입력을 주면
    덜 지우고, 반대면 멀쩡한 세부까지 지운다.

    실무의 처방이 둘이다.

    - **여러 세기를 섞어 익힌다.** 표본마다 $\sigma$를 범위에서 무작위로 뽑으면 한
      모델이 여러 세기를 감당한다.
    - **세기를 입력으로 준다.** 잡음 수준을 모델에 알려 주고 그에 맞게 지우게 한다.

    두 번째 생각이 [퍼짐 모델](../../ch29/index.md)에서 핵심이 된다. 거기서는 잡음
    수준(시각 $t$)을 반드시 함께 넣으며, 하나의 그물이 모든 세기를 다룬다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
잡음 없애는 자기 부호기가 배우는 것을 확률로 보면 무엇인가?

</div>

??? success "연습문제 10 풀이"
    깨끗한 자료의 분포를 $p(x)$, 더럽힘을 $C(\tilde{x} \mid x)$라 하면, 제곱 손실을
    가장 작게 하는 출력은 **조건부 기댓값**이다.

    $$g(f(\tilde{x})) = \mathbb{E}\left[ x \mid \tilde{x} \right]$$

    곧 "이렇게 더럽혀진 것을 보았을 때 원래 그림들의 평균"을 내놓는다. 이것이
    출력이 흐릿한 까닭이기도 하다. 여러 가능성의 평균이기 때문이다.

    더 깊은 이음이 있다. 가우시안 잡음의 경우 이 조건부 기댓값이 **점수 함수**와
    이어진다는 것이 알려져 있다(트위디 공식).

    $$\mathbb{E}[x \mid \tilde{x}] = \tilde{x} + \sigma^2 \nabla_{\tilde{x}} \log p_\sigma(\tilde{x})$$

    오른쪽의 $\nabla \log p$가 곧 점수다. 그러므로 **잡음 없애기를 배운 그물은 사실
    점수를 배운 것**이며, 점수를 알면 랑주뱅 방식으로 자료를 뽑을 수 있다.

    이 장의 자기 부호기가 뽑지 못하는 것과 대비된다. 잡음 없애는 자기 부호기는
    저도 모르게 만들어 내는 모델의 재료를 쥐고 있었던 셈이고, 그 재료를 제대로 쓰는
    것이 [점수 바탕 모델과 퍼짐 모델](../../ch29/index.md)이다.

## 정리하며

**다룬 것** — 잡음 없애는 자기 부호기

여느 자기 부호기와의 핵심 차이는 들임은 망가뜨리되 목표는 깨끗이 둔다는 것이다.

고갱이 갈래는 `DenoisingAutoencoder`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
