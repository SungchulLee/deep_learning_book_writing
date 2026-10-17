# 재사용 가능한 오토인코더 모듈

`SimpleAutoencoder`는 벡터를 잠재 코드로 바꾸는 인코더와 원래 모양으로 복원하는 디코더를 제공한다. 이 페이지의 기본 구조는 784→256→256→32와 그 역방향이다. 클래스 정의와 무작위 입력의 순전파만 포함하며 학습된 압축 품질이나 샘플 생성 성능을 검증한 예제는 아니다.

## 1. 예제 코드와 출력

```python
"""단순한 오토인코더 - 정해진 인코더-디코더 기본 구조."""
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

```
Input shape: torch.Size([32, 784])
Reconstruction shape: torch.Size([32, 784])
Loss: 31782.0742
```

## 2. 모듈의 입력·출력과 크기

`encode`는 B×784 입력을 B×32 코드로, `decode`는 이를 B×784 복원값으로 바꾼다. `forward`는 두 연산을 연결한다. 이미지 모양을 자동으로 펼치지는 않으므로 이미지 배치를 사용한다면 호출하는 쪽에서 피처 차원을 맞추어야 한다.

각 경로에는 선형 계층이 세 개 있으며 은닉 계층 사이에는 ReLU, 디코더 마지막에는 Sigmoid가 있다. 대칭 너비는 설계 선택이며 인코더·디코더의 웨이트를 공유하지 않는다. 같은 너비라고 역함수나 동등한 유효 표현력을 보장하는 것도 아니다.

기본값의 파라미터 수는 인코더 274,976개와 디코더 275,728개, 합계 550,704개다. 입력 차원 d, 은닉 너비 h와 잠재 차원 k에서 합은 2dh+2h²+2hk+4h+k+d다. d=784,h=256이면 534288+513k로 k에 대해 선형 증가한다.

## 3. 출력 범위와 로스의 합산

Sigmoid는 복원값을 [0,1] 범위로 제한하는 설계다. 그러나 `main`의 입력은 `randn`으로 만든 정규 난수라 음수와 1을 넘는 값도 있다. MSE 계산 자체는 가능하지만 이 출력 범위로 모든 표적을 복원할 수 없다. 실제 픽셀 학습에는 범위가 맞는 전처리가 필요하다.

`loss_function`은 원소별 제곱 오차의 **합**이다. 같은 원소당 오차에서 배치 크기와 피처 수가 커지면 로스·그래디언트의 척도가 바뀐다. 원소 평균 MSE와 비교하려면 원소 수로 나누고, 표본 평균 합제곱 오차를 원하면 배치 크기로 나눈다. 다른 정규화 항을 더할 때도 합·평균의 축을 일관되게 정한다.

## 4. 인터페이스를 나누는 이유

인코더만으로 피처를 구하거나 디코더만으로 코드를 복원하고, 두 입력 코드 사이를 보간할 수 있다. 입력에서 만든 코드 밖의 영역에서 디코더 출력이 의미 있다는 보장은 없다. 연속적인 네트워크의 보간 출력이 매끄럽다는 사실만으로 잠재 분포 전체의 품질을 증명하지 않는다.

로스는 클래스 안이나 밖에 둘 수 있다. 변형 모델은 KL, 은닉 활성화나 야코비안처럼 추가 정보가 필요할 수 있으므로 반환값과 호출부를 함께 설계한다. 함수를 같은 이름으로 둔다고 일반 AE와 VAE 학습 코드가 자동으로 호환되지는 않는다.

## 5. 정규화와 확장의 조건

이 기본 클래스에는 Dropout·BatchNorm·별도 잠재 정규화가 없다. 추가 여부는 목적에 따라 검증한다. 디노이징 학습은 같은 구조를 유지한 채 오염 입력을 넣고 깨끗한 표적과 비교할 수 있다. 반면 VAE처럼 인코더의 분포 파라미터와 샘플링이 필요한 모델은 모듈과 로스의 차원을 함께 바꾸어야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
잠재 차원 조회 속성과 파라미터 수 함수를 클래스에 추가할 때 아래 조각의 적용 조건을 설명하라. 기본 h=256에서 k=8,32,128의 전체 개수도 구하라.

</div>

??? success "풀이"

    메서드 정의를 SimpleAutoencoder 클래스 안에 넣거나 명시적으로 연결해야 인스턴스에서 사용할 수 있다. 아래 독립 함수 정의만으로 m.count_parameters가 생기지는 않는다. 전체 수는 534288+513k이므로 각각 538392, 550704, 599952개다. 기존 latent_dim 속성도 직접 조회할 수 있다.

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


---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
MSE 대신 BCE를 사용할 때 출력과 표적에 필요한 조건을 설명하라.

</div>

??? success "풀이"

    확률 입력 BCELoss는 출력과 표적이 [0,1]이어야 한다. Sigmoid는 출력 범위만 맞추고 표적 범위를 고치지 않는다. 연속 회색 픽셀의 BCE는 소프트 표적 로스로 사용할 수 있으나 픽셀을 실제 베르누이 관측으로 해석하는 가정과는 구분한다. 범위 밖 입력은 구현에서 오류를 낼 수 있다. 로짓을 반환하면 BCEWithLogitsLoss를 사용할 수 있다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
두 입력 사이의 잠재 보간 함수에서 모양·장치와 해석의 조건을 설명하라.

</div>

??? success "풀이"

    아래 함수는 x1,x2가 배치 차원이 없는 길이 d 벡터라고 가정한다. 모델·입력·alphas의 장치와 dtype을 맞추고 평가 시 eval과 no_grad를 사용한다. n_steps=10은 양 끝을 포함한 열 점이다. 출력의 연속성은 네트워크 함수의 성질일 수 있어 의미 있는 중간 표본이나 잠재공간 전체의 품질을 증명하지 않는다.

    ```python
    def interpolate(self, x1, x2, n_steps=10):
        z1 = self.encode(x1.unsqueeze(0))
        z2 = self.encode(x2.unsqueeze(0))
        alphas = torch.linspace(0, 1, n_steps).unsqueeze(1)
        z_interp = (1 - alphas) * z1 + alphas * z2
        return self.decode(z_interp)
    ```


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
실제 기본 구조의 인코더와 디코더 파라미터 수를 계산하라.

</div>

??? success "풀이"

    인코더는 (784×256+256)+(256×256+256)+(256×32+32)=274976개다. 디코더는 (32×256+256)+(256×256+256)+(256×784+784)=275728개다. 합은 550704개이고 차이 752개는 출력 바이어스 784와 잠재 바이어스 32의 차이다. 다른 너비 512·256·16의 개수를 이 모델에 적용하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
encode·decode·forward의 분리가 어떤 활용을 가능하게 하는가?

</div>

??? success "풀이"

    encode로 입력 피처를 구하고 decode로 이미 가진 코드를 복원하며 forward로 전체 재구성을 계산할 수 있다. 이 인터페이스는 실험을 편하게 하지만 임의 코드가 실제 데이터처럼 복원되는지는 보장하지 않는다. VAE가 튜플을 반환하면 학습 호출부도 그 계약에 맞춰야 한다.



---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
로스를 모듈 안에 두면 VAE로 교체할 때 학습 루프를 항상 유지할 수 있는가?

</div>

??? success "풀이"

    그렇지 않다. 일반 AE는 복원값을 반환하지만 VAE는 복원과 분포 파라미터를 필요로 할 수 있다. 반환값·인수·합산 축을 통일한 인터페이스를 먼저 설계해야 한다. 로스를 외부 함수에 두고 필요한 값을 반환하는 설계도 가능하다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff easy" title="쉬움"></span>
마지막 Sigmoid를 제거할 때 로스 선택은 어떻게 달라지는가?

</div>

??? success "풀이"

    실수 범위 출력과 MSE를 사용할 수 있으며 표적 범위에 맞는 복원인지 검증한다. BCELoss에 로짓을 그대로 주면 확률 범위 조건을 위반한다. 로짓용 BCEWithLogitsLoss는 Sigmoid를 별도로 적용하지 않고 쓸 수 있다. 모든 활성화를 없애야 선형 AE이므로 Sigmoid가 남으면 PCA의 선형 동등성 조건에 맞지 않는다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
아래 디노이징 조각의 입력과 표적을 설명하라.

</div>

??? success "풀이"

    noisy가 모델 입력이고 깨끗한 x가 표적이다. sigma와 criterion을 미리 정의하고 데이터에 맞는 잡음·클리핑을 선택한다. 잡음 종류와 강도에 따라 배우는 복원 문제가 달라지며 이 간단한 입력 오염만으로 모든 디노이징 목적과 품질이 해결되는 것은 아니다.

    ```python
    noisy = (x + sigma * torch.randn_like(x)).clamp(0, 1)
    out = model(noisy)          # 입력은 더럽힌 것
    loss = criterion(out, x)    # 표적은 깨끗한 것
    ```


---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
아래 SimpleVAE 서브클래스의 잠재 차원 불일치를 지적하라.

</div>

??? success "풀이"

    부모를 latent_dim*2로 만들면 인코더 출력뿐 아니라 디코더 입력도 2k가 된다. encode는 이를 k차원 mu와 logvar로 나누고 샘플 z는 k차원이라 decode의 2k 입력 조건과 맞지 않는다. 디코더 입력을 k로 별도 구성하고 latent_dim 속성과 로스·forward 계약도 맞추어야 한다. 이 조각은 실행 가능한 완성 VAE가 아니며 특정 샘플 품질 개선 수치를 증명하지 않는다.

    ```python
    class SimpleVAE(SimpleAutoencoder):
        def __init__(self, input_dim=784, hidden_dim=256, latent_dim=32):
            super().__init__(input_dim, hidden_dim, latent_dim * 2)   # 1. 두 배로

        def encode(self, x):
            h = super().encode(x)
            return h.chunk(2, dim=1)                                  # mu, logvar

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()      # 2. 다시 뽑기
            return self.decode(z), mu, logvar

        def loss_function(self, out, x, mu, logvar):
            rec = F.binary_cross_entropy(out, x, reduction='sum') / x.size(0)
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum() / x.size(0)
            return rec + kl                                           # 3. KL 항
        ```


---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
입력 검증에서 모양·dtype·값 범위를 어떻게 정하는가? 아래 assert 조각의 적용 범위도 설명하라.

</div>

??? success "풀이"

    이 모델은 B×input_dim 입력과 파라미터에 호환되는 부동소수 dtype·장치가 필요하다. float32만 가능한 것은 아니며 모델 dtype에 따른다. Sigmoid 표적이나 BCE에는 [0,1] 범위가 필요하다. 아래 self는 메서드 안에서 유효하고 비어 있는 입력의 min/max도 별도 처리해야 한다. uint8을 선형 계층에 직접 넣으면 보통 dtype 오류가 먼저 나며 조용히 큰 로스만 나는 것은 아니다.

    ```python
    assert x.dim() == 2 and x.size(1) == self.input_dim, f"모양이 어긋난다: {x.shape}"
    assert x.dtype == torch.float32, f"float32여야 한다: {x.dtype}"
    assert 0.0 <= x.min() and x.max() <= 1.0, f"[0,1]을 벗어난다: [{x.min()}, {x.max()}]"
    ```


## 정리하며

재사용 모듈은 입력·코드·복원의 차원과 반환 계약을 분명히 해야 한다. 합제곱 로스와 평균 MSE, 출력·표적 범위를 구분한다. 보간·디노이징·VAE로 확장할 때도 장치와 차원, 학습 목표를 함께 맞추어야 하며 구조만으로 품질을 보장할 수 없다.
