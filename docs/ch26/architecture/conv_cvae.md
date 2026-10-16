# 컨볼루션 조건부 VAE

컨볼루션 조건부 VAE는 이미지의 공간 구조를 처리하면서 클래스 조건에 따라 생성 분포를 바꾼다. 이 구현은 인코더에서 클래스 임베딩을 이미지 채널로 연결하고, 디코더에서 원핫 조건을 잠재 벡터에 연결한다. 두 경로가 서로 다른 조건 표현을 사용한다는 점에 주목한다.

## 1. 구현

```python
"""
컨볼루션 조건부 변분 오토인코더(ConvCVAE)
컨볼루션 구조와 조건부 만들어 내기를 아우른다
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class ConvConditionalVAE(nn.Module):
    """
    조건부 그림 만들어 내기를 위한 컨볼루션 조건부 변분 오토인코더.
    
    인수:
        latent_dim (int): 숨은 공간 차원
        num_classes (int): 조건 지을 클래스의 수
        img_channels (int): 들임 그림 채널 수
        img_size (int): 들임 그림 크기(네모 그림이라 여긴다)
    """
    
    def __init__(self, latent_dim=128, num_classes=10, img_channels=1, img_size=28):
        super(ConvConditionalVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.img_size = img_size
        
        # 공간 조건 짓기를 위한 이름표 묻힘
        self.label_embedding = nn.Embedding(num_classes, img_size * img_size)
        
        # 인코더 - 그림 + 묻은 이름표를 채널 하나로 더 받는다
        self.encoder = nn.Sequential(
            # 들임: img_channels + 1(묻은 이름표용)
            nn.Conv2d(img_channels + 1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Flatten()
        )
        
        # 펼친 크기를 셈한다
        self.flatten_size = 128 * 4 * 4
        
        # 숨은 분포 파라메터
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # 디코더 들임: 숨은 것 + 하나만 뜨거운 클래스
        self.decoder_input = nn.Linear(latent_dim + num_classes, self.flatten_size)
        
        # 디코더
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 4, 4)),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x, labels):
        """
        클래스 이름표에 조건 지어 들임 그림을 부호화한다.
        
        인수:
            x: 들임 그림 텐서 [배치 크기, 채널, 높이, 너비]
            labels: 클래스 이름표 [배치 크기]
            
        반환값:
            mu: 숨은 분포의 평균
            logvar: 숨은 분포의 로그 흩어짐
        """
        batch_size = x.size(0)
        
        # 이름표를 묻고 공간 꼴로 바꾼다
        c_embedded = self.label_embedding(labels)
        c_embedded = c_embedded.view(batch_size, 1, self.img_size, self.img_size)
        
        # 그림과 묻은 이름표를 잇는다
        x_combined = torch.cat([x, c_embedded], dim=1)
        
        # 부호화
        h = self.encoder(x_combined)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        다시 파라메터화 재주.
        
        인수:
            mu: 숨은 분포의 평균
            logvar: 숨은 분포의 로그 흩어짐
            
        반환값:
            z: 뽑은 숨은 벡터
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, labels):
        """
        클래스 이름표에 조건 지어 숨은 나타냄을 푼다.
        
        인수:
            z: 숨은 벡터 [배치 크기, 숨은 차원]
            labels: 클래스 이름표 [배치 크기]
            
        반환값:
            reconstruction: 다시 세운 그림
        """
        # 이름표를 하나만 뜨겁게 부호화한다
        c_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        
        # 숨은 부호와 조건을 잇는다
        z_combined = torch.cat([z, c_onehot], dim=1)
        
        # 디코딩
        h = self.decoder_input(z_combined)
        reconstruction = self.decoder(h)
        return reconstruction
    
    def forward(self, x, labels):
        """
        조건을 곁들인 온전한 피드포워드.
        
        인수:
            x: 들임 그림 텐서
            labels: 클래스 이름표
            
        반환값:
            reconstruction: 다시 세운 그림
            mu: 숨은 분포의 평균
            logvar: 숨은 분포의 로그 흩어짐
        """
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z, labels)
        return reconstruction, mu, logvar
    
    def loss_function(self, reconstruction, x, mu, logvar, beta=1.0):
        """
        변분 오토인코더 로스 함수.
        
        인수:
            reconstruction: 다시 세운 내놓기
            x: 본디 들임
            mu: 숨은 분포의 평균
            logvar: 숨은 분포의 로그 흩어짐
            beta: KL 벌어짐 항의 무게
            
        반환값:
            loss: 전체 변분 오토인코더 로스
            bce: 다시 세우기 로스
            kld: KL 벌어짐
        """
        BCE = F.binary_cross_entropy(reconstruction, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + beta * KLD, BCE, KLD
    
    def sample(self, class_label, num_samples, device='cpu'):
        """
        특정 클래스에 조건 지어 표본을 만든다.
        
        인수:
            class_label: 만들 클래스(int)
            num_samples: 만들 표본의 개수
            device: 표본을 만들 기기
            
        반환값:
            samples: 만든 그림 표본
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        labels = torch.tensor([class_label] * num_samples).to(device)
        samples = self.decode(z, labels)
        return samples
    
    def interpolate_classes(self, z, class1, class2, num_steps=10):
        """
        숨은 부호를 고정한 채 두 클래스 사이를 사이 끼움한다.
        
        인수:
            z: 고정한 숨은 부호 [1, 숨은 차원]
            class1: 시작 클래스
            class2: 끝 클래스
            num_steps: 사이 끼움 걸음 수
            
        반환값:
            interpolations: 사이 끼움한 표본
        """
        device = z.device
        interpolations = []
        
        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            
            # 사이 끼움을 위한 부드러운 이름표를 만든다
            c1_onehot = F.one_hot(torch.tensor([class1]), num_classes=self.num_classes).float().to(device)
            c2_onehot = F.one_hot(torch.tensor([class2]), num_classes=self.num_classes).float().to(device)
            c_interpolated = (1 - alpha) * c1_onehot + alpha * c2_onehot
            
            # 디코딩
            z_combined = torch.cat([z, c_interpolated], dim=1)
            h = self.decoder_input(z_combined)
            sample = self.decoder(h)
            interpolations.append(sample)
        
        return torch.cat(interpolations, dim=0)


if __name__ == '__main__':
    # 모델을 시험한다
    model = ConvConditionalVAE(latent_dim=128, num_classes=10, img_channels=1, img_size=28)
    # BCE 로스는 목표가 [0,1]이어야 한다
    x = torch.rand(32, 1, 28, 28)
    labels = torch.randint(0, 10, (32,))
    
    reconstruction, mu, logvar = model(x, labels)
    loss, bce, kld = model.loss_function(reconstruction, x, mu, logvar)
    
    print(f"Input shape: {x.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Reconstruction Loss: {bce.item():.4f}")
    print(f"KL Divergence: {kld.item():.4f}")
    
    # 조건부 뽑기를 시험한다
    samples = model.sample(class_label=7, num_samples=10)
    print(f"Generated samples (class 7) shape: {samples.shape}")
```

기본 이미지 입력은 `(B,1,28,28)`, 레이블은 0~9 정수 인덱스 `(B,)`다. 인코더의 임베딩 표는 클래스마다 학습 가능한 784개 값을 저장한다. 이를 `(B,1,28,28)`로 바꿔 이미지에 연결하므로 첫 컨볼루션은 2채널을 받는다. 이 조건 지도는 원핫을 공간에 반복한 10개 채널과 다르며 위치마다 값이 같을 필요도 없다.

인코더는 28→14→7→4로 줄이고 2,048개 피처에서 평균·로그분산을 예측한다. 디코더는 `(z,one_hot(y))`를 선형 변환해 `(B,128,4,4)`로 만든 뒤 4→7→14→28로 확대한다. 표준정규 사전을 쓰므로 KL 식은 [조건부 VAE](conditional_vae.md)와 같다.

**기존 출력 기록:**

```
Input shape: torch.Size([32, 1, 28, 28])
Labels shape: torch.Size([32])
Reconstruction shape: torch.Size([32, 1, 28, 28])
Total Loss: 21192.6875
Reconstruction Loss: 20655.6250
KL Divergence: 537.0617
Generated samples (class 7) shape: torch.Size([10, 1, 28, 28])
```

위 실행부는 무작위 입력의 모양과 로스 계산을 확인한다. 학습된 모델의 클래스 제어 정확도나 생성 품질을 측정한 결과가 아니다.

## 2. 사용 조건과 구조의 한계

img_size는 임베딩 지도의 크기에는 반영되지만 인코더의 펼친 크기와 디코더 시작 지도는 128×4×4로 고정되어 있다. 다른 해상도를 지원하려면 이 선형 레이어들과 업샘플링 구조까지 함께 바꿔야 한다. 기본 설정의 학습 파라미터는 1,032,545개다. 무조건 ConvVAE보다 임베딩 7,840개, 첫 컨볼루션 512개, 디코더 조건 연결 20,480개가 더해진 값이다.

BCE와 KL은 배치 전체 합이며 반환값은 `(전체 로스,BCE,KL)`이다. BCE 타깃의 범위·모양과 모델의 장치·자료형을 맞춘다. BatchNorm은 학습·평가 모드에 따라 동작이 달라지지만 잠재 표본은 평가에서도 다시 뽑는다. sample은 eval·no_grad를 적용하지 않고 기본 CPU 잠재 코드를 생성하므로 호출부에서 평가 조건을 정해야 한다.

## 3. 클래스 보간의 해석

interpolate_classes는 하나의 잠재 코드 `(1,latent_dim)`를 고정하고 두 원핫 조건을 선형 혼합한다. 조건 벡터가 한 행이므로 일반 배치 z와 그대로 연결할 수 없다. num_steps≥2가 필요하고 클래스 인덱스도 유효해야 한다. 기본 반환 모양은 `(num_steps,img_channels,28,28)`이다.

원핫 끝점 사이의 혼합은 학습에서 보지 않은 조건일 수 있다. 따라서 부드러운 계산 경로가 의미 있는 숫자 변형이나 스타일 보존을 보장하지 않는다. 같은 코드의 의미가 클래스 사이에서 정렬되었는지도 별도로 살펴야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
기본 설정과 배치 크기 4에서 조건 연결을 포함한 모양을 계산하라.

</div>

??? success "연습문제 1 풀이"

    레이블 `(4,)`→임베딩 `(4,784)`→조건 지도 `(4,1,28,28)`이며 연결 입력은 `(4,2,28,28)`이다. 컨볼루션 피처는 `(4,32,14,14)`→`(4,64,7,7)`→`(4,128,4,4)`, 펼친 값은 `(4,2048)`이다. 평균·로그분산·잠재 표본은 `(4,128)`이다. 디코더에서는 원핫 `(4,10)`과 연결해 `(4,138)`→`(4,2048)`→`(4,128,4,4)`로 바꾼다. 전치 컨볼루션은 `(4,64,7,7)`→`(4,32,14,14)`→`(4,1,28,28)`을 만든다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
64×64 RGB 조건부 모델을 설계하려면 무엇을 바꿔야 하는가?

</div>

??? success "연습문제 2 풀이"

    img_channels=3,img_size=64로 조건 지도를 만들면 연결 입력은 4채널이다. 기존 인코더 스트라이드를 유지하면 마지막 지도는 128×8×8이다. 두 헤드와 디코더 입력 레이어를 이 크기로 맞추고 Unflatten도 바꾼다. 디코더의 세 레이어를 커널 4·스트라이드 2·패딩 1로 구성하면 8→16→32→64가 가능하다. 마지막 출력은 3채널이어야 한다. forward에는 이미지와 레이블이 모두 필요하고 반환은 튜플이므로 기존 풀이처럼 model(x).shape만 확인하는 방식은 맞지 않는다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
깊이별 분리 컨볼루션이 조건부 모델에서 줄일 수 있는 비용과 줄이지 못하는 비용을 설명하라.

</div>

??? success "연습문제 3 풀이"

    표준 컨볼루션의 웨이트 C_in C_out k²를 채널당 depthwise C_in k²와 pointwise C_in C_out로 대체하면 비는 1/C_out+1/k²다. 같은 출력 크기에서 곱셈 누적 횟수도 같은 비로 줄지만 실행 시간과 정확도는 별도다. 이 변경은 클래스 임베딩, 두 선형 헤드와 디코더 입력 레이어의 비용을 줄이지 않는다. 조건 채널을 추가한 실제 C_in으로 계산해야 한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
설정 가능한 블록 수를 추가할 때 조건 경로에서 유지할 조건을 설명하라.

</div>

??? success "연습문제 4 풀이"

    조건 지도의 해상도와 이미지 크기, 첫 블록 입력 채널, 마지막 피처 크기와 두 헤드를 함께 맞춘다. 디코더에서 잠재·원핫 연결 차원과 시작 지도를 정하고 목표 해상도로 도달하는 레이어를 설계한다. 아래 ModuleList는 등록 예시일 뿐 YourBlock 구현이나 모양 계산이 없다. 실제 클래스에 num_layers를 추가하지 않고 인수만 넘길 수는 없다. 블록을 등록한 뒤 forward에서 호출해야 한다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
본문의 조건 지도 방식과 아래 대안의 차이를 설명하라.

</div>

??? success "연습문제 5 풀이"

    본문은 클래스 임베딩을 공간적으로 변하는 단일 채널로 만든다. 아래 첫 코드는 이미 원핫인 y를 10개 상수 채널로 반복하는 대안이다. 이 경우 회색조 이미지와 연결한 첫 레이어는 11채널이어야 한다. 두 번째 코드는 잠재·조건 벡터를 연결해 피처 지도를 만드는 개념적 대안이며 본문의 속성명·채널·시작 크기와 일치하지 않는다. 컨볼루션 뒤 펼친 피처에 조건을 연결하는 방법도 있지만 그 경우 앞선 컨볼루션 연산에는 조건이 직접 전달되지 않는다.

    ```python
    # y: (B, 10) 원-핫 -> (B, 10, 28, 28) 로 펴서 채널에 붙인다
    c = y.view(-1, 10, 1, 1).expand(-1, 10, 28, 28)
    x = torch.cat([x, c], dim=1)          # (B, 11, 28, 28)
    ```
    ```python
    h = self.fc(torch.cat([z, y], dim=1)).view(-1, 64, 7, 7)
    ```


---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
입력 연결, 후단 연결과 피처별 조건 조절의 차이를 설명하라.

</div>

??? success "연습문제 6 풀이"

    입력 연결은 초기 피처부터 조건의 영향을 주고 후단 연결은 앞쪽 공유 피처를 재사용한다. 레이어별 조건 조절은 각 레이어의 피처를 조건에 따라 스케일·이동시킬 수 있다. 아래 FiLM 형태는 h'=γ(y)h+β(y)이며 자체로 정규화 연산을 요구하지 않는다. 조건부 BatchNorm처럼 정규화와 결합한 방식과 구분한다. 비용은 조건 폭·레이어 수·지도 크기에 달려 있어 후단이 항상 가장 싸거나 레이어별 방식이 항상 가장 비싸다고 단정할 수 없다. 조건 일치율·다양성·학습 안정성을 확인해 선택한다.

    ```python
    gamma, beta = self.film(y).chunk(2, dim=1)      # (B, C) 둘
    h = gamma.view(-1, C, 1, 1) * h + beta.view(-1, C, 1, 1)
    ```


---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
조건 사용 여부와 코드·클래스의 독립성을 어떻게 진단하는가?

</div>

??? success "연습문제 7 풀이"

    별도 검증 자료에서 잠재 코드만으로 클래스 예측이 가능한지 검사하고 빈도 기준선과 비교한다. 낮은 분류 성능만으로 독립성이 입증되지는 않는다. 동일 z에서 조건을 바꾸거나 조건을 제거·섞어 출력 변화를 보는 개입도 보완 자료다. 아래 코드의 model.enc는 본문의 model.encoder와 속성명이 달라 그대로 동작하지 않는다. 이름을 맞추어 웨이트 크기를 살펴도 임베딩 스케일·후속 레이어·디코더 조건 경로가 있으므로 작은 첫 레이어 웨이트만으로 조건 무시를 결론 낼 수 없다.

    ```python
    w = model.enc[0].weight            # (32, 11, 3, 3)
    print("그림 채널:", w[:, :1].abs().mean().item())
    print("조건 채널:", w[:, 1:].abs().mean().item())
    ```


---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
컨볼루션과 조건부 생성의 효과가 단순히 더해진다고 볼 수 있는가?

</div>

??? success "연습문제 8 풀이"

    두 선택은 상호작용할 수 있어 이득이 가산적이라고 보장할 수 없다. 완전 연결·컨볼루션 각각에서 조건 유무를 비교하는 네 조합 실험이 필요하다. 재구성 변화가 생성 품질 개선을 뜻하지 않으며 조건 추가가 사전 불일치를 자동으로 해결하지도 않는다. 본문에는 학습 비교가 없어 기존 다른 설정의 수치로 빈 조합의 성능을 추정할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
네 조합 비교에서 통제할 조건과 평가 지표를 설계하라.

</div>

??? success "연습문제 9 풀이"

    데이터 분할·전처리·관측 모형·잠재 차원·로스 합산 기준을 맞추고 파라미터 수·시간·메모리 차이를 보고한다. 동일 하이퍼파라미터 비교와 동일 튜닝 예산 비교는 서로 다른 질문에 답하므로 구분한다. 여러 초기화에서 재구성·표준 ELBO·차원별 KL·품질·다양성을 측정한다. 조건부 모델에는 지정 클래스 일치율도 추가한다. 무조건부 비교용 조건 분포 p(y)도 명시한다. 분류기 확신도 임계는 보조 지표로 표시하며 “숫자로 보이는 진짜 비율”로 해석하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
고정 z에서 0~9 조건을 바꿔 디코딩하는 코드의 입력 계약과 해석을 설명하라.

</div>

??? success "연습문제 10 풀이"

    아래 코드는 잠재 16차원과 원핫을 직접 받는 decode를 가정한다. 본문 기본 잠재는 128이고 decode는 원핫이 아닌 정수 레이블을 받아 내부에서 one_hot을 만들므로 그대로 호환되지 않는다. 실제 잠재 차원에 맞춘 동일 z의 10행과 정수 레이블 0~9를 전달해야 한다. 이런 비교는 클래스 제어와 스타일 변화를 살피는 도구다. 스타일 변화가 곧 특정 정보 혼합 원인을 증명하거나 잘못된 클래스가 반드시 조건 전달 버그를 뜻하는 것은 아니며 학습 부족·모형 한계도 점검한다.

    ```python
    z = torch.randn(1, 16).repeat(10, 1)              # 같은 코드 열 벌
    y = F.one_hot(torch.arange(10), 10).float()       # 0..9
    imgs = model.decode(z, y)                         # (10, 1, 28, 28)
    ```


## 정리하며

이 모델은 공간 임베딩 채널과 디코더 원핫 연결로 조건을 전달한다. 고정 해상도와 정수 레이블 계약을 지키고, 보간·표현 분리·조건 일치율은 각각의 실험으로 평가한다. 조건 경로의 형태와 비용을 구분하면 다른 구현의 예시를 그대로 적용하는 오류를 줄일 수 있다.
