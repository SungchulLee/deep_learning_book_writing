# 완전 연결 오토인코더

이 예제는 MNIST 픽셀을 32차원 잠재 표현으로 바꾸고 복원하는 완전 연결 오토인코더를 학습한다. 인코더는 784→128→64→32, 디코더는 역방향이며 ReLU와 마지막 Sigmoid를 사용한다. 비선형 표현을 학습할 수 있지만 병목만으로 유용한 피처가나 잡음 제거를 보장하지는 않는다.

## 1. 예제 코드

```python
"""온전히 이어진 오토인코더."""
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

## 2. 입력과 모델 크기

`ToTensor`로 [0,1]에 맞춘 B×1×28×28 이미지를 B×784로 편다. 잠재 표현은 B×32이고 복원 출력은 B×784다. `forward`는 차원이 2보다 크면 펼치지만 `encode`를 직접 호출할 때는 이미 평탄화된 입력이 필요하다. 비연속 텐서에서 `view`가 가능한지도 확인한다.

기본 파라미터 수는 222,384개다. k차원 병목에서는 218256+129k개이며 k=16이면 220,320개다. 다른 페이지의 784→512→256→16 구조의 개수를 이 모델에 적용하지 않는다. 대칭 너비지만 전치 웨이트를 공유하지 않는다.

## 3. BCE 학습의 의미

`BCELoss`는 기본적으로 배치와 픽셀의 모든 원소를 평균한다. Sigmoid는 복원 확률의 범위를 맞추지만 표적 범위는 데이터 전처리로 맞추어야 한다. 회색 픽셀을 소프트 표적으로 쓰는 BCE와 실제 이진 관측의 베르누이 우도 해석도 구분한다.

BCE가 모든 [0,1] 데이터에서 MSE보다 우수하거나 항상 선명한 이미지를 만든다는 보장은 없다. BCE의 로짓 미분은 p−y이고 Sigmoid 출력에 대한 MSE의 미분에는 p(1−p)가 더 들어간다는 차이가 있지만, 최종 품질은 모델·최적화·평가 척도에 달려 있다. 두 목적 모두 제한된 표현에서 평균적인 복원을 만들 수 있다.

## 4. 학습 코드의 범위

코드는 Adam, 러닝레이트 0.001과 웨이트 감쇠 10⁻⁵로 20에포크 학습한다. 클래스 레이블은 사용하지 않는다. 검증·테스트 로더, 로스 기록과 복원 시각화, 저장과 잠재 보간은 본 코드에 없다. 따라서 이 페이지에서 특정 테스트 MSE나 분류 성능을 측정했다고 설명할 수 없다.

대부분의 실행문이 최상위에 있어 파일을 import하면 모델 구성·데이터 로딩·학습이 수행된다. 마지막 `__main__`의 `pass`가 이 실행을 막지는 않는다.

## 5. 구조와 잠재 표현 해석의 조건

이미지를 펼치는 것은 픽셀 위치의 인덱스 정보를 지우는 연산이 아니다. 완전 연결 계층은 모든 위치를 구분하지만 컨볼루션의 국소 연결·웨이트 공유라는 귀납적 편향이 없다. CNN과 비교할 때 잠재 표현 원소 수, 파라미터 수와 학습 예산을 함께 맞춘다. 피처 맵 잠재 표현은 채널×높이×너비로 세어 입력보다 크지 않은지 확인한다.

병목에 ReLU를 추가하면 음수 잠재 표현을 제한하고 0 영역의 그래디언트를 바꾸지만 유효 정보 용량이 정확히 절반이나 2ᵏ배 감소한다는 계산은 성립하지 않는다. 비음수 잠재 표현도 연속값을 가질 수 있다. 잠재 축의 의미, 군집과 보간 품질은 복원 로스만으로 정해지지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
병목 차원별 테스트 MSE 비교를 설계하고 아래 코드의 누락을 설명하라.

</div>

??? success "풀이"

    아래 반복문은 모델을 새로 만들기만 하고 학습·테스트 계산을 하지 않으며 test_mse도 정의되지 않았다. 각 모델을 같은 조건으로 학습하고 독립 검증 구간에서 eval·no_grad로 원소 수 가중 MSE를 계산해야 한다. 팔꿈치 위치를 미리 32나 64로 정하지 않는다. 설정 선택 뒤 최종 테스트를 분리한다.

    ```python
    for dim in [2, 8, 32, 128]:
        model = FullyConnectedAutoencoder(latent_dim=dim).to(device)
        # ... 20바퀴 익힌다 ...
        # ... 시험 평균 제곱 어긋남을 셈한다 ...
        print(f"latent_dim={dim}: test MSE = {test_mse:.6f}")
    ```


---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
BCE와 MSE 재학습 결과를 공정하게 비교할 조건과 해석을 설명하라.

</div>

??? success "풀이"

    같은 데이터 분할·초기값 조건·구조와 평가 척도를 사용하고 각 로스에 적절한 러닝레이트·예산을 검증한다. 서로 다른 로스값 자체를 수치로 비교하지 않는다. BCE가 항상 더 선명하거나 MSE가 모든 오차를 동일하게 벌한다는 설명은 맞지 않다. MSE의 벌점은 오차 제곱으로 커진다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
아래 Dropout 조각을 적용하고 비교할 때 필요한 작업을 설명하라.

</div>

??? success "풀이"

    이 조각은 클래스 안의 인코더 교체에 해당하며 단독 실행에서는 self 등의 문맥이 필요하다. 기존 모델에 교체하면 새 파라미터와 옵티마이저 연결을 맞추고 다시 학습해야 한다. 평가에는 eval로 Dropout을 끈다. 검증 로스 개선과 학습·검증 차이를 보고 효과를 판단하며 항상 도움이 된다고 가정하지 않는다.

    ```python
    self.encoder = nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(hidden_dim // 2, latent_dim),
    )
    ```


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
같은 16차원 잠재 표현의 완전 연결 모델과 CNN을 비교할 때 어떤 조건을 맞추는가?

</div>

??? success "풀이"

    두 인코더의 잠재 표현 원소 수를 16으로 맞추고 실제 파라미터·전처리·로스·학습 예산·데이터 분할을 기록한다. 이 페이지 모델은 k=16에서 220320개다. CNN의 수는 실제 구조에서 따로 센다. 계산하지 않은 고정 MSE 표로 어느 쪽이 이긴다고 결론 내리지 않는다.



---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span>
32×7×7 피처 맵을 잠재 표현으로 사용할 때 차원과 압축 여부를 판단하라.

</div>

??? success "풀이"

    잠재 표현은 1568개 원소라 입력 784개 픽셀보다 많다. 같은 dtype의 단순 좌표 저장에서는 압축이 아니다. 그래도 디노이징·정규화 등 다른 목적에 쓸모없다고 단정할 수 없다. 파일 압축에는 잠재 표현의 양자화·엔트로피와 모델 저장 비용도 고려한다.



---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff easy" title="쉬움"></span>
깊이를 늘리면 재구성이 항상 개선되는가?

</div>

??? success "풀이"

    아니다. 표현 함수의 종류와 최적화·정규화가 바뀌어 개선될 수도 악화될 수도 있다. 고정 병목 차원은 잠재 표현 원소 수를 정하지만 유한 표본의 정보량이나 암기 한계를 그 숫자만으로 정하지는 않는다. 여러 시드와 공정한 학습 예산에서 독립 평가한다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
선형 AE의 tied weights가 제약하는 재구성 행렬과 비선형 확장 시 주의점을 설명하라.

</div>

??? success "풀이"

    단일 선형 인코더 E와 전치 디코더의 곱은 EᵀE로 대칭 양의 준정부호다. 정규직교 행을 가지면 PCA 투영을 표현할 수 있다. 여러 비선형 계층에서는 단일 행렬 곱으로 같은 해석을 적용할 수 없다. 실제 공유는 웨이트를 줄이지만 바이어스까지 포함한 전체 수가 꼭 절반은 아니다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
병목 ReLU의 효과를 부호 조합 수로 정량화하면 안 되는 이유를 설명하라.

</div>

??? success "풀이"

    잠재 표현은 연속값이며 부호 조합 수만으로 표현 정보량이 정해지지 않는다. ReLU는 비음수 제약과 0 영역의 그래디언트를 도입해 학습을 바꾸지만 용량을 정확히 절반으로 줄이는 것은 아니다. 희소성이나 해석 목적과 함께 독립 검증 결과로 선택한다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
잠재 축 하나를 바꾸면 반드시 의미 있는 시각적 피처가 바뀌는가?

</div>

??? success "풀이"

    그렇지 않다. 출력이 거의 변하지 않는 영역도 있고 여러 피처가 함께 바뀔 수 있다. 로스에 개별 축 의미를 식별할 추가 가정이 없기 때문이다. 선형 병목의 가역변환을 디코더 첫 계층에서 상쇄할 수 있는 등의 비유일성이 있다. 비지도 분리 표현의 가능성과 식별 조건을 구분하며 레이블만이 유일한 해결책이라고 단정하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
다른 데이터에 적용할 때 구조·출력·평가에서 무엇을 바꾸는가?

</div>

??? success "풀이"

    입력 차원과 펼침을 맞추고 데이터 범위·분포에 맞는 출력과 로스를 정한다. 음수 연속 표적에는 현재 Sigmoid 범위가 맞지 않을 수 있다. 병목과 은닉 너비, CNN 등 귀납적 바이어스도 비교한다. 기본 병목은 32이며 모든 데이터에 적절하다는 보장은 없다.



## 정리하며

완전 연결 AE는 비선형 인코더·디코더로 복원을 학습한다. 이 예제의 실제 너비와 파라미터 수, BCE 평균과 입력 범위를 기준으로 읽는다. 병목·활성화·정규화의 효과는 별도 학습과 평가가 필요하며 낮은 재구성 오차만으로 유용한 잠재 표현나 좋은 샘플 생성을 보장할 수 없다.
