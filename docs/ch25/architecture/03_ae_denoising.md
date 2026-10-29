# 디노이징 오토인코더

디노이징 오토인코더(DAE)는 오염된 입력 x̃에서 깨끗한 원본 x를 복원하도록 학습한다. 입력을 그대로 복사하는 것과 다른 목표를 주지만, 잡음을 완전히 구별하거나 모든 후속 작업에 좋은 피처를 배운다는 보장은 없다. 학습에 쓰는 오염 분포와 실제 사용 조건을 함께 정의한다.

## 1. 예제 코드

```python
"""잡음 없애는 오토인코더."""
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

## 2. 입력과 표적이 다른 학습 목표

일반 복원과 디노이징 목표는 각각 다음과 같이 쓸 수 있다.

$$\mathbb E_x[L(r(x),x)],\qquad
\mathbb E_x\mathbb E_{\tilde x\sim C(\cdot|x)}[L(r(\tilde x),x)].$$

코드는 매 배치의 깨끗한 MNIST 이미지에 잡음을 새로 만들고, 복원 출력과 **깨끗한 이미지**의 BCE를 계산한다. 레이블은 사용하지 않는다. 입력 x̃를 표적으로 바꾸면 다른 목적이 된다.

모델은 앞 CNN과 같은 B×1×28×28 → B×128×7×7 → B×1×28×28 구조이며 잠재 표현이 6272원소다. 작은 벡터 병목으로 정보를 제한하는 모델은 아니지만 오염 입력 복원이라는 학습 제약을 사용할 수 있다.

## 3. 세 오염 방식의 실제 의미

가우시안 분기에서는 표준편차 `noise_factor`의 독립 정규 잡음을 더하고 [0,1]로 자른다. 따라서 실제 입력에 남는 잡음은 경계에서 비가우시안이며 평균·분산도 달라진다. 원래 픽셀 값에 따른 포화가 생긴다.

salt-and-pepper 분기는 확률 `noise_factor/2`로 1, 같은 확률로 0을 대입한다. 원래 이미 그 값이었다면 선택되어도 변화가 없을 수 있다. dropout 분기는 확률 `noise_factor`로 픽셀을 0으로 만들고 크기 보정은 하지 않는다. 이는 학습용 `nn.Dropout`의 스케일 규칙과 다르다.

확률 분기에는 0≤noise_factor≤1, 가우시안 표준편차에는 비음수 조건을 정해야 한다. 같은 숫자 0.3이라도 표준편차와 오염 확률은 서로 다른 세기다. 알 수 없는 `noise_type`에는 명시적인 오류 대신 `None`이 반환되므로 호출 조건을 검증해야 한다.

## 4. 평가 코드가 포함하지 않는 것

본 코드는 20에포크 학습만 하고 테스트 로더, PSNR 계산과 비교 표를 만들지 않는다. 고정된 dB 개선이나 다른 잡음으로의 일반화 순위를 측정한 것으로 설명하지 않는다.

평가에는 독립된 깨끗한 이미지와 대응하는 오염 표본을 준비하고 같은 쌍으로 입력 기준선·일반 AE·DAE를 비교한다. 잡음 유형과 세기가 바뀌는 교차 평가도 분리한다. 테스트 잡음 시드를 고정하면 모델 간 차이를 비교하기 쉽고 여러 오염 실현값에서 불확실성도 볼 수 있다.

## 5. PSNR과 품질 해석

픽셀 범위의 최대 차이가 MAX이고 같은 픽셀에 대한 MSE를 사용하면 다음과 같다.

$$\operatorname{PSNR}=10\log_{10}(MAX^2/\operatorname{MSE}).$$

[0,1]에서는 −10log₁₀(MSE)이며 MSE=0이면 무한대다. 이미지별 PSNR 평균과 전체 MSE에서 계산한 PSNR은 로그의 비선형성 때문에 다르다. 무엇을 보고하는지 명시한다. 개선량이 양수여도 절대 품질이 낮을 수 있고, PSNR이 시각적 품질이나 의미 보존을 완전히 설명하지도 않는다.

잡음이 클수록 복원 불확실성이 커질 수 있다. 제곱 오차의 이상적인 복원 함수는 E[x|x̃]이므로 여러 가능성이 있으면 평균적인 출력이 된다. [DAE 연구](https://www.jmlr.org/beta/papers/v11/vincent10a.html)의 표현 학습 목적도 실제 태스크와 오염 조건에 맞추어 평가한다.

## 6. 점수 추정과 연결할 때의 조건

클리핑 없는 가우시안 오염 x̃=x+σε, ε∼N(0,I)와 충분한 정칙성에서 조건부 평균은 다음과 연결된다.

$$\mathbb E[x|\tilde x]=\tilde x+\sigma^2\nabla_{\tilde x}\log p_\sigma(\tilde x).$$

여기 p_σ는 가우시안으로 평활화한 데이터 밀도다. 적절한 목적·최적 함수의 디노이징 잔차를 점수로 해석할 수 있지만, 위 코드는 **클리핑된 잡음과 BCE, 유한 모델 학습**을 사용한다. 이 식을 현재 출력에 그대로 적용해 정확한 점수를 얻었다고 볼 수 없다. 관련 조건은 [점수 매칭과 DAE의 연결 연구](https://direct.mit.edu/neco/article-abstract/23/7/1661/7677)에 설명되어 있다.

한 번의 DAE 복원만으로 확산 생성 과정이 완성되지는 않는다. 여러 잡음 수준의 학습과 목표, 역과정 또는 샘플링 절차가 추가로 필요하다. 현재 코드에 없는 생성 품질을 가정하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
세 잡음으로 학습한 모델의 3×3 교차 평가를 공정하게 설계하라.

</div>

??? success "풀이"

    같은 깨끗한 보류 표본에 각 잡음 유형·세기를 정의해 같은 오염 실현값으로 모델들을 비교한다. 입력 자체의 오차와 복원 오차, 절대 PSNR과 개선량을 함께 기록한다. 표준편차와 오염 확률을 같은 수치라고 같은 난도로 보지 않는다. 일반화 순위는 측정해야 하며 가우시안이 항상 가장 우수하다고 정하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
아래 noise_factor 반복문이 성능 비교를 완성하지 못하는 이유를 설명하라.

</div>

??? success "풀이"

    모델 재학습이나 잡음 생성·복원·PSNR 계산이 없고 improvement도 정의되지 않았다. 세기별 고정 모델 평가와 세기별 재학습은 다른 실험이다. 어느 값에서 알아볼 수 없게 되는지는 데이터·모델·평가 기준에 따라 정해야 한다.

    ```python
    for nf in [0.1, 0.3, 0.5, 0.7, 0.9]:
        # noise_factor=nf로 잡음 없애는 오토인코더를 익히고 최대 신호 대 잡음비 나아짐을 셈한다
        print(f"noise_factor={nf}: PSNR improvement = {improvement:.1f} dB")
    ```


---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
디노이징에 skip connection을 추가할 때 이점과 비교 조건을 설명하라.

</div>

??? success "풀이"

    고해상도 피처를 전달해 세부 복원에 도움을 줄 수 있지만 잡음도 전달될 수 있다. 채널·공간 모양과 파라미터 비용을 맞추고 같은 오염·표적과 학습 예산에서 검증한다. 특정 1~3 dB 개선은 구현·데이터에 대한 측정 없이는 주장할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
일반 AE와 DAE의 깨끗한 입력·오염 입력 평가에서 비교할 네 조건을 설명하라.

</div>

??? success "풀이"

    일반 AE와 DAE 각각에 깨끗한 입력과 같은 오염 입력을 넣고 모두 깨끗한 원본을 기준으로 평가한다. 두 모델의 구조·학습 예산·데이터 분할을 맞춘다. 각자가 학습한 조건에서 반드시 이긴다는 법칙은 없고 입력 오염의 변화와 품질을 함께 본다.



---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span>
디노이징 목적이 일반 AE와 다른 제약을 주는 이유를 설명하라.

</div>

??? success "풀이"

    오염 입력을 그대로 출력하면 원본과 차이가 남아 로스를 받는다. 따라서 원본의 조건부 구조를 이용하는 복원이 유리할 수 있다. 하지만 오염이 아주 약하거나 모델·데이터가 제한되면 항등에 가까울 수 있어 의미 있는 피처를 반드시 배운다고 결론 내리지 않는다.



---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
코드의 dropout 잡음과 nn.Dropout의 차이를 설명하라.

</div>

??? success "풀이"

    이 코드는 픽셀을 확률 p로 0으로 만들고 남은 값을 유지한다. 일반 nn.Dropout은 학습 때 남은 값을 1/(1−p)로 스케일하고 평가 때 항등으로 동작한다. 오염 연산과 은닉 정규화의 목적을 구분한다. 여기 masking도 깨끗한 전체 이미지를 복원하며 가린 부분만 평가하는 다른 목표와는 다르다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff easy" title="쉬움"></span>
잡음 세기를 선택할 기준을 설명하라.

</div>

??? success "풀이"

    사용 조건을 대표하는 오염과 검증 데이터에서 복원·후속 태스크를 평가한다. 너무 약하면 입력과 표적 차이가 작고 너무 강하면 원본의 조건부 불확실성이 커질 수 있다. 사람이 알아볼 수 있다는 주관적 기준이나 고정 sigma만으로 최선이 정해지지는 않는다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
잡음을 매번 새로 뽑는 방식과 고정된 오염 쌍을 쓰는 방식은 각각 언제 유용한가?

</div>

??? success "풀이"

    새로 뽑으면 더 넓은 오염 실현값을 경험해 특정 쌍의 암기를 줄일 수 있지만 항상 필수는 아니다. 고정된 다수 오염 쌍도 유효한 학습 데이터이고 평가에서는 고정 잡음이 공정한 비교에 유용하다. 표본 수와 다양성, 재현성과 과적합을 함께 고려한다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
학습 잡음과 실제 잡음이 다를 때 무엇을 확인해야 하는가?

</div>

??? success "풀이"

    유형·세기·신호 의존성과 클리핑·범위가 다른지 평가한다. 정확히 같아야만 작동하는 것은 아니지만 분포 차이가 성능을 바꿀 수 있다. 여러 세기·유형을 섞거나 잡음 수준을 조건으로 줄 수 있으나 새 조건의 일반화를 별도로 검증한다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
조건부 평균과 점수 함수의 관계를 현재 코드에 바로 적용할 수 있는가?

</div>

??? success "풀이"

    제곱 오차의 이상적인 복원은 조건부 평균이고, 클리핑 없는 가우시안 오염에서는 트위디 식으로 평활 밀도의 점수와 연결된다. 현재 코드는 클리핑된 입력, BCE와 유한 모델을 사용하므로 같은 점수식을 그대로 보장하지 않는다. 생성에는 적절한 확률 모형·여러 잡음 수준과 샘플링 과정도 필요하다.



## 정리하며

DAE는 오염 입력과 깨끗한 표적의 관계를 학습한다. 잡음 유형별 인수 의미와 클리핑, 모델의 실제 잠재 표현 크기와 평가 범위를 구분한다. PSNR의 집계 방식과 분포 차이를 명시하고, 점수 추정·생성 모델과의 연결에는 필요한 오염·목적·최적화 조건을 함께 설명해야 한다.
