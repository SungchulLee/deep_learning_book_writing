# 컨볼루션 VAE

컨볼루션 VAE는 이미지의 공간 구조와 위치 간 웨이트 공유를 이용해 근사 사후를 예측하고 이미지를 생성한다. 확률 모형과 KL 항의 역할은 완전 연결 VAE와 같으며 인코더·디코더의 뉴럴 넷트웍 구조가 달라진다. 컨볼루션을 사용한다는 사실만으로 재구성이나 생성 품질의 우열이 정해지지는 않는다.

## 1. 모듈 구현

```python
"""
컨볼루션 변분 오토인코더(ConvVAE)
공간 피처를 더 잘 뽑으려 컨볼루션 레이어를 쓴다
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class ConvVAE(nn.Module):
    """
    그림 자료를 위한 컨볼루션 변분 오토인코더.
    
    인수:
        latent_dim (int): 숨은 공간 차원
        img_channels (int): 들임 그림 채널 수(회색조 1, RGB 3)
        img_size (int): 들임 그림 크기(네모 그림이라 여긴다)
    """
    
    def __init__(self, latent_dim=128, img_channels=1, img_size=28):
        super(ConvVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.img_size = img_size
        
        # 인코더
        self.encoder = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 14x14 -> 7x7
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 7x7 -> 4x4
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Flatten()
        )
        
        # 펼친 크기를 셈한다
        self.flatten_size = 128 * 4 * 4
        
        # 숨은 공간 파라메터
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # 디코더 들임
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)
        
        # 디코더
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 4, 4)),
            
            # 4x4 -> 7x7
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 7x7 -> 14x14
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 14x14 -> 28x28
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """
        들임 그림을 숨은 분포 파라메터로 부호화한다.
        
        인수:
            x: 들임 그림 텐서 [배치 크기, 채널, 높이, 너비]
            
        반환값:
            mu: 숨은 분포의 평균
            logvar: 숨은 분포의 로그 흩어짐
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        뽑기를 위한 다시 파라메터화 재주.
        
        인수:
            mu: 숨은 분포의 평균
            logvar: 숨은 분포의 로그 흩어짐
            
        반환값:
            z: 뽑은 숨은 벡터
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """
        숨은 나타냄을 그림으로 푼다.
        
        인수:
            z: 숨은 벡터
            
        반환값:
            reconstruction: 다시 세운 그림
        """
        h = self.decoder_input(z)
        reconstruction = self.decoder(h)
        return reconstruction
    
    def forward(self, x):
        """
        온전한 피드포워드.
        
        인수:
            x: 들임 그림 텐서
            
        반환값:
            reconstruction: 다시 세운 그림
            mu: 숨은 분포의 평균
            logvar: 숨은 분포의 로그 흩어짐
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
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
        # 되살림 로스
        BCE = F.binary_cross_entropy(reconstruction, x, reduction='sum')
        
        # KL 발산
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return BCE + beta * KLD, BCE, KLD
    
    def sample(self, num_samples, device='cpu'):
        """
        숨은 공간에서 표본을 만든다.
        
        인수:
            num_samples: 만들 표본의 개수
            device: 표본을 만들 기기
            
        반환값:
            samples: 만든 그림 표본
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples


if __name__ == '__main__':
    # 모델을 시험한다
    model = ConvVAE(latent_dim=128, img_channels=1, img_size=28)
    # BCE 로스는 목표가 [0,1]이어야 한다
    x = torch.rand(32, 1, 28, 28)  # 회색조 28x28 그림 32개 배치
    
    reconstruction, mu, logvar = model(x)
    loss, bce, kld = model.loss_function(reconstruction, x, mu, logvar)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Reconstruction Loss: {bce.item():.4f}")
    print(f"KL Divergence: {kld.item():.4f}")
    
    # 뽑기를 시험한다
    samples = model.sample(num_samples=10)
    print(f"Generated samples shape: {samples.shape}")
```

기본 입력은 `(B,1,28,28)`이다. 인코더의 공간 크기는 28→14→7→4, 채널 수는 1→32→64→128로 바뀐다. 마지막 피처 2,048개에서 평균과 로그분산을 각각 128차원으로 예측한다. 디코더는 잠재 표현을 `(B,128,4,4)`로 확장하고 전치 컨볼루션으로 4→7→14→28을 만든다. 출력은 픽셀별 베르누이 확률이다.

**기존 출력 기록:**

```
Input shape: torch.Size([32, 1, 28, 28])
Reconstruction shape: torch.Size([32, 1, 28, 28])
Latent mu shape: torch.Size([32, 128])
Total Loss: 21310.0566
Reconstruction Loss: 20787.1797
KL Divergence: 522.8765
Generated samples shape: torch.Size([10, 1, 28, 28])
```

실행부는 무작위 입력의 순전파와 로스·샘플 출력 모양을 확인한다. 학습 반복문이 없으므로 위 기록에서 재구성 품질이나 완전 연결 모델 대비 성능을 판단할 수 없다.

## 2. 크기 설정과 학습 계약

`img_channels`는 첫·마지막 컨볼루션에 반영되지만 `img_size`는 저장만 된다. 펼친 크기와 디코더 시작 지도가 128×4×4로 고정되어 있어 img_size만 바꿔 임의의 해상도를 지원하지 못한다. 입력·출력 공간 크기를 모두 계산해 구조를 바꿔야 한다.

BCE와 KL은 배치 전체의 합이며 loss_function은 `(전체 로스,BCE,KL)`을 반환한다. 0~1 범위 타깃, 모양 일치와 장치·자료형 호환성이 필요하다. 회색조 BCE의 관측 모형 해석은 [VAE 모듈](vae.md)과 같다.

BatchNorm 때문에 학습·평가 모드는 정규화 통계 사용 방식에 영향을 준다. 다만 리파라메리제이션은 평가에서도 표본을 뽑는다. sample도 eval이나 no_grad를 자동 적용하지 않고 기본 CPU 표준정규 잠재 표현을 생성하므로 모델과 장치·자료형을 맞춰 호출해야 한다.

## 3. 구조 비교의 기준

컨볼루션 커널의 파라미터 수는 같은 채널 수·커널 크기에서 이미지 해상도에 직접 비례하지 않는다. 하지만 이 모듈의 선형 헤드와 디코더 입력 레이어는 피처 지도의 원소 수에 비례한다. 컨볼루션 모델 전체의 파라미터 수나 연산량이 해상도와 무관하다는 뜻은 아니다.

재구성 로스, 차원별 KL, 사전 표본 품질과 계산비용을 따로 비교해야 한다. 같은 전처리·잠재 차원·로스 합산 기준을 맞추고 각 구조의 학습 상태를 확인해야 차이를 해석할 수 있다. 분류기 확신도 하나만으로 생성 이미지의 유효률을 정할 수는 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
기본 설정과 배치 크기 4에서 인코더·디코더의 주요 텐서 모양을 계산하라.

</div>

??? success "풀이"

    인코더는 `(4,1,28,28)`→`(4,32,14,14)`→`(4,64,7,7)`→`(4,128,4,4)`→`(4,2048)`이다. 평균·로그분산과 잠재 표본은 각각 `(4,128)`이다. 디코더 입력 선형 레이어는 `(4,2048)`을 만들고 이를 `(4,128,4,4)`로 바꾼다. 이어 `(4,64,7,7)`→`(4,32,14,14)`→`(4,1,28,28)`이 된다. 이 구현에는 풀링 레이어가 없다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
64×64 RGB 입력을 지원하려면 어떤 구조 변경이 필요한가? 실행 대신 모양을 계산하라.

</div>

??? success "풀이"

    img_channels=3으로 바꾸고 세 인코더 컨볼루션을 유지하면 공간 크기는 64→32→16→8이다. 펼친 크기는 128×8×8로 바뀐다. 예를 들어 디코더를 8×8에서 시작해 커널 4·스트라이드 2·패딩 1인 세 전치 컨볼루션으로 8→16→32→64를 만들 수 있다. 두 인코더 헤드, 디코더 입력 선형 레이어, Unflatten과 마지막 채널 수를 함께 고쳐야 한다. img_size=64만 넘기거나 반환 튜플에 바로 shape를 묻는 것은 해결책이 아니다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
표준 컨볼루션과 depthwise separable convolution의 웨이트 수·곱셈 누적 횟수를 비교하라.

</div>

??? success "풀이"

    바이어스를 제외하고 표준 컨볼루션은 C_in C_out k²개다. 채널당 필터 하나의 depthwise와 1×1 pointwise를 결합하면 C_in k²+C_in C_out개이며 비는 1/C_out+1/k²다. 동일한 출력 공간 크기라면 곱셈 누적 횟수의 비도 같다. k=3,C_out=256이면 약 0.115로 약 8.7배 감소한다. 그러나 표현하는 함수와 실제 장치의 실행 효율이 달라 같은 정확도나 같은 속도 비율을 보장하지는 않는다. 이 비교를 선형 레이어나 전치 컨볼루션 전체에 그대로 적용하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
은닉 블록 수를 설정 가능하게 만들 때 필요한 크기 계산을 설명하라.

</div>

??? success "풀이"

    인코더 블록별 채널·스트라이드·패딩에서 마지막 공간 크기를 구해 두 헤드와 디코더 시작 지도를 정한다. 디코더의 크기 변화는 별도로 설계해 원래 이미지 크기를 복구해야 하며 레이어 수만 역순으로 늘린다고 자동으로 맞지 않는다. 아래 ModuleList는 등록 방식의 골격이다. YourBlock과 dim은 구체적으로 정의해야 하고 forward에서 등록한 블록을 호출해야 한다. 깊이 2·4·8 비교에는 유효한 공간 크기, 출력 모양과 파라미터 수를 먼저 확인한다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
본문 기본 모델의 파라미터 수를 계산하고 완전 연결 모델과 비교할 때 주의할 점을 설명하라.

</div>

??? success "풀이"

    기본 잠재 128에서 인코더 컨볼루션·BatchNorm은 107,680개, 두 헤드는 각각 262,272개, 디코더 입력 선형 레이어는 264,192개, 나머지 디코더는 107,297개다. 합은 1,003,713개다. BatchNorm의 학습 가능한 affine 파라미터는 포함하고 running 통계 버퍼는 제외한 값이다. 잠재 16이면 315,361개이며 같은 잠재 16인 앞 절 VAE의 546,608개와 비교할 수 있다. 앞 장 ConvAutoencoder의 185,857개도 구조가 다르다. 잠재 크기·공간 구조가 다른 모델의 숫자를 같은 조건의 압축률로 읽지 않는다.



---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
컨볼루션 모델의 재구성과 사전 표본 품질을 어떻게 비교해야 하는가?

</div>

??? success "풀이"

    동일한 평가 데이터·전처리·관측 모형에서 표본당 재구성 항과 KL을 계산하고, 별도로 사전 생성의 다양성·품질을 평가한다. 재구성이 좋아졌다는 사실만으로 사전 표본이 좋아졌다고 할 수 없다. 본문 실행에는 학습·평가 실험이 없으므로 특정 로스 수치나 승패를 이 구현의 결과로 제시할 근거가 없다. 기존 다른 페이지의 분류기 확신도 임계 비율도 진짜 생성 유효률이나 이 구조의 고유 성능은 아니다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
전치 컨볼루션의 대안과 출력 크기 식을 설명하라.

</div>

??? success "풀이"

    업샘플링 뒤 일반 컨볼루션을 적용할 수 있다. 아래의 두 방식은 해상도를 늘리는 서로 다른 연산이다. 전치 컨볼루션의 한 축 출력 크기는 (H−1)s−2p+d(k−1)+output_padding+1이다. 이 코드의 첫 레이어는 H=4,k=3,s=2,p=1에서 7이고 이후 k=4 레이어는 7→14→28이다. 커널 4·스트라이드 2처럼 겹침을 균일하게 만드는 선택은 일부 checkerboard 원인을 줄이지만 모든 아티팩트를 없애지는 않는다. 업샘플링과 컨볼루션을 분리해도 품질을 확인해야 한다. [전치 컨볼루션 아티팩트 분석](https://distill.pub/2016/deconv-checkerboard/)은 불균일한 겹침의 영향을 설명한다.

    ```python
    # 방법 1: 옮겨 컨볼루션
    nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)

    # 방법 2: 키운 뒤 컨볼루션
    nn.Upsample(scale_factor=2, mode='nearest')
    nn.Conv2d(64, 32, 3, padding=1)
    ```


---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
스트라이드 컨볼루션과 최대 풀링의 정보 보존을 비교하라.

</div>

??? success "풀이"

    스트라이드 컨볼루션은 학습 가능한 필터로 다운샘플링하고 최대 풀링은 영역별 최댓값을 선택한다. 둘 다 일반적으로 비가역적이며 스트라이드 컨볼루션의 웨이트에 각 입력의 잃어버린 정보가 저장되는 것은 아니다. 충분한 채널 등 구조 조건에 따라 보존 능력이 달라진다. 풀링 인덱스를 따로 전달해 unpooling할 수 있지만 이는 잠재 표현 외의 부가 정보이므로 압축·생성에서 사용할 정보 계약에 포함해야 한다. 풀링을 쓴 VAE도 설계할 수 있다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
평균 KL이 더 크면 잠재 정보와 사전 불일치가 모두 증가했다고 결론 낼 수 있는가?

</div>

??? success "풀이"

    아니다. E_x KL(q(z|x)||p(z))=I_q(X;Z)+KL(q(z)||p(z))이므로 합만으로 두 항의 변화를 식별할 수 없다. 재구성 개선이나 분류기 확신도 하락도 각 항의 증가를 증명하지 못한다. 집계 사후 불일치의 별도 추정, 입력에 따른 잠재 표현 변화와 디코더 의존성 등을 함께 진단해야 한다. 차원별 KL도 유용하지만 그 자체가 두 항의 정확한 분해는 아니다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
해상도가 높아지면 컨볼루션과 완전 연결 구조의 비용은 어떻게 달라지는가?

</div>

??? success "풀이"

    고정 은닉 폭의 완전 연결 첫 레이어는 입력 픽셀 수에 비례해 커진다. 단일 채널 28²에서 128²로 늘면 웨이트 수가 약 20.9배다. 컨볼루션 커널 수는 같은 채널·커널 설정에서 그대로지만 연산량과 엑티베이션 메모리는 공간 크기에 따라 커진다. 본문처럼 평탄화한 피처를 선형 레이어에 넣으면 그 레이어도 커진다. 공간적 웨이트 공유가 유리한 귀납 바이어스일 수 있어도 모든 큰 이미지 작업에서 컨볼루션이 우세하다는 정리는 아니다. 구조·학습 예산·평가 조건을 맞춰 판단해야 한다.



## 정리하며

컨볼루션 VAE의 확률적 목적은 완전 연결 VAE와 같지만 이미지 피처를 추출·복원하는 구조가 다르다. 본문은 28×28 입력에 고정된 구현이다. 모양과 파라미터 수를 정확히 계산하고 재구성 성능·KL·생성 품질을 서로 다른 지표로 평가한다.
