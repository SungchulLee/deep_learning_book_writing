# 컨볼루션 오토인코더

컨볼루션 오토인코더는 국소 연결과 공간 위치 간 웨이트 공유를 사용해 이미지를 복원한다. 이 예제는 컨볼루션·BatchNorm·최대 풀링 인코더와 최근접 업샘플링·컨볼루션 디코더를 MNIST에 학습한다. 공간 해상도를 줄이더라도 잠재 표현의 전체 원소 수가 입력보다 작은지는 별도로 확인해야 한다.

## 1. 예제 코드

```python
"""컨볼루션 오토인코더."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

batch_size, learning_rate, num_epochs = 128, 1e-3, 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConvolutionalAutoencoder(nn.Module):
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

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))

model = ConvolutionalAutoencoder().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    for images, _ in train_loader:
        images = images.to(device)
        loss = criterion(model(images), images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    pass
```

## 2. 실제 피처 맵과 파라미터

입력은 B×1×28×28이다. 첫 풀링 뒤 B×32×14×14, 둘째 풀링 뒤 B×64×7×7이며 마지막 컨볼루션이 B×128×7×7을 만든다. 잠재 표현의 표본당 원소 수는 **128×7×7=6272**개로 원래 784개 픽셀의 8배다. 채널 수 128을 잠재 차원으로 부르거나 이 구조를 작은 벡터 병목과 같은 압축률로 비교하지 않는다.

디코더는 B×64×7×7에서 업샘플링으로 14×14, B×32×14×14에서 28×28을 만든 뒤 한 채널로 복원한다. 전체 학습 파라미터는 바이어스와 BatchNorm의 affine 파라미터를 포함해 185,857개다. BatchNorm 실행 평균·분산 등 버퍼는 이 개수에서 제외한다.

## 3. 컨볼루션의 귀납적 바이어스와 BatchNorm

컨볼루션은 같은 필터를 위치마다 재사용한다. 적절한 경계 조건에서의 이동 등변성이 관련되지만 풀링·경계·서브샘플링까지 포함한 전체 모델이 임의 이동에 완전히 불변인 것은 아니다. 필터가 모서리·질감을 학습할 수 있어도 특정 의미를 자동으로 보장하지 않는다.

BatchNorm은 학습 배치의 통계와 학습 가능한 스케일·이동을 사용하고 평가 때에는 기본 설정에서 실행 통계를 사용한다. 학습 안정성에 도움이 될 수 있지만 이를 단지 “내부 공변량 이동을 없애기 때문”이라고 확정하지 않는다. 배치 크기와 데이터 분포, 평가 모드에 따라 복원도 달라질 수 있다.

## 4. 업샘플링과 전치 컨볼루션

최근접 업샘플링은 값을 복제하고 뒤의 컨볼루션이 복원을 학습한다. 전치 컨볼루션은 다른 학습 가능한 확대 연산이며 원래 컨볼루션의 역함수가 아니다. 출력 한 축의 크기는 다음과 같다.

$$H_{out}=(H_{in}-1)s-2p+d(k-1)+\operatorname{output\_padding}+1.$$

`output_padding`은 stride 때문에 생기는 출력 크기 선택의 모호성을 다루며 로스된 정보를 되살리는 값은 아니다. 출력 크기와 채널을 맞춘다고 학습된 웨이트를 그대로 교환할 수도 없다.

커널 크기가 stride로 나누어떨어지지 않으면 위치별 겹침이 달라져 checkerboard artifact가 생길 수 있다. k=3,s=2가 이런 예이고 홀수 커널이 일반적인 해결책은 아니다. k=4,s=2처럼 고른 겹침이나 resize 후 컨볼루션도 선택지지만 모든 격자 무늬가 사라진다는 보장은 없다. [원인과 대안을 분석한 연구](https://distill.pub/2016/deconv-checkerboard/)를 참고한다.

## 5. 압축·복원·피처 평가

본 코드는 BCE로 학습하지만 테스트 로더·로스 기록·복원 비교는 없다. 고정 테스트 MSE나 CNN과 완전 연결 모델의 우열을 여기서 측정했다고 설명할 수 없다. 공정한 비교에서는 잠재 표현 원소 수, 실제 파라미터와 전처리·학습·평가 조건을 맞춘다.

낮은 복원 오차는 입력보다 큰 피처 맵의 유용성을 부정하지도, 압축 표현의 우수성을 증명하지도 않는다. 압축이 목적이면 피처 맵을 더 줄이거나 명시적 벡터 병목을 두고, 바이트 수와 디코더에 필요한 모든 정보를 센다. skip connection이 있으면 전달되는 우회 피처도 포함해야 한다. 이런 우회는 병목만으로 정보를 제한하는 목적을 약화할 수 있지만 디노이징 등 다른 목표에는 유용할 수 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
첫 컨볼루션 필터를 표시하는 아래 코드의 입력과 해석 조건을 설명하라.

</div>

??? success "풀이"

    첫 웨이트는 32×1×3×3이고 각 필터의 3×3 값을 그린다. 아래 조각에는 matplotlib.pyplot을 plt로 별도 import해야 한다. 웨이트 모양만으로 검출 의미를 단정하지 말고 실제 활성화와 입력 반응도 본다. 본 모델을 충분히 학습했는지도 확인한다.

    ```python
    filters = model.encoder[0].weight.data.cpu()
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    for i in range(32):
        ax = axes[i // 8, i % 8]
        ax.imshow(filters[i, 0].numpy(), cmap='viridis')
        ax.axis('off')
    plt.suptitle("First Layer Filters")
    plt.show()
    ```


---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 모델과 앞 절 완전 연결 모델의 기본 파라미터와 잠재 표현 원소 수를 비교하라.

</div>

??? success "풀이"

    CNN은 185857개 파라미터와 6272개 잠재 표현 원소, 앞 완전 연결 기본 모델은 222384개 파라미터와 32개 잠재 표현 원소다. 파라미터가 적다는 것과 잠재 표현이 더 작다는 것은 다르다. 테스트 MSE는 별도로 측정해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
Upsample을 아래 ConvTranspose2d로 바꾸려면 어떤 연결 조건이 필요한가?

</div>

??? success "풀이"

    제시된 64→64, k=2,s=2 계층은 해당 위치의 채널과 출력 크기에 맞춰 넣어야 한다. 다른 업샘플 위치는 입력 채널이 32라 같은 계층을 그대로 쓸 수 없다. 새 웨이트를 포함해 다시 학습하고 평가한다. 짝수 커널이라서 더 불균일하다는 주장은 맞지 않으며 겹침은 커널과 stride의 관계에 달려 있다.

    ```python
    # nn.Upsample(scale_factor=2)을 다음으로 갈음한다:
    nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
    ```


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
16차원 잠재 표현으로 CNN과 완전 연결 모델을 비교하는 실험의 전제를 정하라.

</div>

??? success "풀이"

    현재 CNN 잠재 표현은 6272원소라 16차원 벡터 병목과 이에 대응하는 디코더를 새로 설계해야 한다. 변경 후 실제 파라미터 수를 계산하고 같은 데이터·로스·예산과 여러 시드에서 테스트한다. 이 페이지에 없는 61329개 구조나 고정 MSE를 현재 모델의 값으로 쓰지 않는다.



---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span>
k=3,s=2,p=1,d=1인 전치 컨볼루션으로 7을 14로 만들 때 output_padding을 구하라.

</div>

??? success "풀이"

    출력은 (7−1)×2−2+2+output_padding+1=13+output_padding이므로 1이다. 이것은 전치 컨볼루션을 추가할 경우의 계산이고 본 코드는 Upsample을 쓴다. 출력 크기를 맞춰도 실제 복원 정보가 자동으로 되돌아오지는 않는다.



---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
checkerboard artifact의 불균일 겹침 원인과 대안의 한계를 설명하라.

</div>

??? success "풀이"

    커널 크기가 stride의 배수가 아니면 위치별 기여 횟수가 다를 수 있다. 배수 커널이나 resize 후 컨볼루션은 그 원인을 줄이는 방법이다. 그러나 학습 필터·경계와 다른 연산도 무늬에 영향을 주므로 완전한 제거를 보장하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff easy" title="쉬움"></span>
현재 CNN 잠재 표현의 차원과 입력 대비 원소 비율을 계산하라.

</div>

??? success "풀이"

    잠재 표현은 128×7×7=6272원소이고 입력 1×28×28=784의 8배다. 공간 축만 줄었다고 차원 축소는 아니다. 다른 모델의 32×7×7 값을 현재 구조에 적용하지 않는다. 실제 저장은 dtype·양자화와 추가 정보도 포함해 계산한다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
풀링과 stride 컨볼루션을 비교할 때 보존 정보와 비용을 어떻게 평가하는가?

</div>

??? success "풀이"

    최대 풀링은 창의 최댓값만 남기고 stride 컨볼루션은 학습 필터로 서브샘플링한다. 둘 다 정보 로스·aliasing 가능성이 있고 어느 쪽이 항상 낫지는 않다. 풀링 인덱스를 디코더에 전달하면 그 저장 비용도 잠재 표현에 포함한다. 같은 품질·잠재 표현 예산에서 비교한다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
skip connection이 작은 병목 표현의 평가에 미치는 영향을 설명하라.

</div>

??? success "풀이"

    병목 외 피처가 디코더로 전달되면 복원에 쓰는 정보가 병목 잠재 표현에만 있지 않다. 벡터 병목만의 압축률·피처 품질을 주장하면 잘못이다. 다만 복원·디노이징 목표에는 유용할 수 있으므로 사용 자체를 금지할 이유는 없다. 우회 정보를 포함한 비용과 목적을 명시한다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
입력 이미지 크기가 커지면 컨볼루션 AE의 파라미터와 실행 비용이 모두 같은가?

</div>

??? success "풀이"

    같은 채널·커널의 컨볼루션 파라미터 수는 공간 크기와 무관하지만 활성화 메모리와 연산량은 커진다. Flatten 뒤 선형 레이어를 추가하면 그 파라미터도 공간 크기에 따라 커진다. 현재 두 번 풀링·두 번 확대 구조는 임의 입력 크기에서 원래 크기로 복원되는지 별도로 계산해야 한다.



## 정리하며

컨볼루션 AE는 공간 웨이트 공유를 사용하지만 잠재 표현의 전체 원소 수와 실제 압축은 별도로 계산해야 한다. 현재 코드는 입력보다 큰 피처 맵을 만들며 테스트 품질 비교는 포함하지 않는다. 업샘플링·전치 컨볼루션의 출력 크기와 겹침, BatchNorm 모드와 우회 정보까지 목적에 맞게 평가한다.
