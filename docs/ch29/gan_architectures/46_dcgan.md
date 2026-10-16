# 28×28용 컨볼루션 GAN 모듈

DCGAN은 컨볼루션 구조와 GAN 학습 지침을 결합한 접근이다. [원 논문](https://arxiv.org/abs/1511.06434)은 공간 구조·웨이트 공유를 활용한 생성과 표현 학습을 연구했다. 이 절의 모듈은 그 아이디어를 응용하지만 생성기에 전치 컨볼루션이 아닌 업샘플링·일반 컨볼루션과 LeakyReLU를 사용한다. 원 논문의 구조를 그대로 재현하는 구현은 아니다.

## 1. 모듈 코드

```python
#!/usr/bin/env python3
'''
DCGAN - 깊은 겹말기 맞겨루기 생성기
논문: "Unsupervised Representation Learning with Deep Convolutional GANs" (2015)
핵심: 여러 레이어 뉴럴 넷트웍을 겹말기 레이어로 바꾼 맞겨루기 생성기의 구조 지침
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class DCGenerator(nn.Module):
    def __init__(self, latent_dim=100, channels=1):
        super().__init__()
        self.init_size = 7
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )
    
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class DCDiscriminator(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(channels, 16, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, 1, 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.adv_layer = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
    
    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

class DCGAN(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.generator = DCGenerator(latent_dim)
        self.discriminator = DCDiscriminator()
    
    def forward(self, z):
        return self.generator(z)

if __name__ == "__main__":
    model = DCGAN()
    print(f"Generator Parameters: {sum(p.numel() for p in model.generator.parameters()):,}")
    print(f"Discriminator Parameters: {sum(p.numel() for p in model.discriminator.parameters()):,}")
```

생성기는 `(B,100)`을 6,272개 피처로 투영해 `(B,128,7,7)`로 바꾼다. 두 차례의 nearest 업샘플링으로 14×14,28×28을 만들며 컨볼루션으로 채널을 128→128→64→1로 바꾼다. 마지막 Tanh는 −1~1의 단일 채널 이미지를 반환한다. channels를 지정하면 마지막 출력 채널은 바뀌지만 DCGAN 래퍼는 기본 1채널만 생성한다.

**기존 출력 기록:**

```
Generator Parameters: 856,065
Discriminator Parameters: 97,729
```

실행부는 파라미터 수만 출력한다. 생성기는 856,065개,판별기는 97,729개다. BatchNorm 학습 스케일·바이어스는 포함하고 running 통계는 제외한다. 기존 출력 숫자가 이 계산과 다르다면 별도의 구조 기록으로 구분해야 한다. 학습이나 판별기 순전파를 검사하는 실행부가 아니다.

## 2. 판별기의 공간 크기 불일치

28×28 이미지는 판별기에서 28→14→7→4→2로 줄어든다. 마지막 Conv2d는 커널 3·스트라이드 1·패딩 0이므로 4×4를 **2×2**로 만든다. 따라서 펼친 피처는 `(B,512)`인데 adv_layer는 Linear(128,1)이라 현재 기본 이미지의 판별 forward에서 차원이 맞지 않는다.

28×28에 맞추려면 실제 펼친 크기에 맞춘 선형 레이어나 공간 크기를 1×1로 줄이는 별도 구조가 필요하다. 어느 경우든 파라미터 수와 학습 계약도 바뀐다. 이 편집에서는 코드를 보존하고 불일치를 설명한다. 생성기 출력 모양과 파라미터 집계의 성공만으로 전체 GAN이 동작한다고 결론 내리지 않는다.

## 3. 컨볼루션·정규화·드롭아웃

생성기에는 BatchNorm2d와 LeakyReLU가,판별기에는 첫 레이어를 제외한 BatchNorm2d와 LeakyReLU·Dropout2d가 있다. Dropout2d는 학습 중 피처 채널을 확률적으로 제거하며 평가 모드에서는 비활성화된다. 파라미터를 step하지 않는 단계에도 학습 모드의 통계·드롭아웃 동작은 남을 수 있다.

정규화·활성화·업샘플링의 선택은 학습 신호와 비용에 영향을 주지만 모드 붕괴 방지나 품질 개선을 보장하지 않는다. 배치 정규화를 어느 위치에 둘지에 관한 DCGAN 지침은 원래 실험의 경험적 설계 원칙이며,어기면 반드시 실패하는 수학적 조건은 아니다.

## 4. 데이터·로스·평가 계약

Tanh 생성 출력과 실제 자료의 범위·전처리를 맞추고,현재 Sigmoid 판별기에는 확률 BCE를 사용한다. logits 로스로 바꾸려면 마지막 Sigmoid의 계약도 바꿔야 한다. 이 모듈은 로스·optimizer·업데이트 비율·학습 루프를 포함하지 않는다.

평가 피처 네트워크가 다른 범위를 기대하면 실제·생성 자료 모두 같은 변환으로 맞춘다. 이 페이지에는 FID·인셉션 점수 비교 실험이 없으므로 정규화·드롭아웃 효과나 완전 연결 대비 우열을 특정 수치로 단정할 수 없다. 먼저 크기 불일치를 해결하고 같은 평가 조건에서 품질·다양성·비용을 비교해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
생성기와 판별기의 공간 크기를 모두 추적하라.

</div>

??? success "연습문제 1 풀이"

    G는 `(B,100)`→`(B,6272)`→`(B,128,7,7)`→`(B,128,14,14)`→`(B,64,28,28)`→`(B,1,28,28)`이다. D는 1→16→32→64→128채널과 공간 28→14→7→4→2를 거쳐 `(B,512)`로 펼친다. 따라서 Linear(128,1)과 맞지 않는다. 생성기의 원하는 출력 모양만 확인하는 것은 두 모델 연결 검증에 충분하지 않다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력·출력 부근 BatchNorm 생략 지침을 어떻게 해석해야 하는가?

</div>

??? success "연습문제 2 풀이"

    원 DCGAN의 경험적 안정화 지침이다. BatchNorm이 표본 간 통계와 평가 running 통계를 사용하므로 입력 분포·출력 스케일·상대 모델 상태에 영향을 줄 수 있다. 그러나 출력 BatchNorm이 모든 픽셀 범위 생성을 수학적으로 불가능하게 하거나 입력 BatchNorm이 항상 필수 구분 정보를 없앤다고 단정할 수는 없다. 실제 배치 크기·레이어 위치·데이터에서 검증해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
64×64 RGB 구조의 모양을 설계하고 파라미터 비교 조건을 설명하라.

</div>

??? success "연습문제 3 풀이"

    예를 들어 4×4 피처에서 네 차례 2배 업샘플링해 8→16→32→64를 만들고 마지막 채널을 3으로 둘 수 있다. D는 그 해상도를 맞게 줄여 실제 펼친 피처와 출력 레이어를 연결한다. 이 페이지에는 DCGAN64Generator·Discriminator가 없어 이름만 호출할 수 없다. 현재 G 856,065개,D 97,729개와 비교하려면 새 채널 폭·커널·바이어스·정규화까지 정한 뒤 계산해야 한다. 정하지 않은 구조의 “몇 배 FLOPs”나 파라미터 수는 단정할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
원 DCGAN 지침과 이 코드가 다른 부분을 설명하라.

</div>

??? success "연습문제 4 풀이"

    원 지침에는 학습 가능한 다운샘플링·업샘플링,적절한 BatchNorm,불필요한 완전 연결 은닉 레이어의 축소,G의 ReLU·출력 Tanh와 D의 LeakyReLU 등이 포함된다. 여기 G는 선형 투영 뒤 nearest 업샘플링·Conv2d·LeakyReLU를 쓴다. 따라서 원 지침의 효과를 이 코드의 실험으로 확인했다고 할 수 없다. 스트라이드 컨볼루션도 일반적으로 비가역적이며 웨이트에 버린 입력별 정보가 그대로 보존되는 것은 아니다.



---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
Tanh 생성기의 전처리·표시·평가 변환을 구분하라.

</div>

??? success "연습문제 5 풀이"

    아래 Normalize(0.5,0.5)는 ToTensor의 0~1을 −1~1로 보낸다. 실제·생성 범위가 다르면 D가 단순한 스케일 차이를 이용할 수 있지만 반드시 학습 신호가 전혀 없어지는 것은 아니다. 다음 (g(z)+1)/2는 생성 출력을 0~1로 되돌리는 예다. 평가기는 다시 별도의 정규화·채널·해상도를 요구할 수 있어 양쪽에 같은 계약을 적용한다. 코드에는 필요한 transforms import와 g,z 정의 등이 별도로 필요하다.

    ```python
    transform = transforms.Compose([
        transforms.ToTensor(),                       # [0,1]
        transforms.Normalize((0.5,), (0.5,)),        # [-1,1]
    ])
    ```
    ```python
    imgs = (g(z) + 1) / 2
    ```


---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
본문 업샘플링 방식과 전치 컨볼루션의 차이를 설명하라.

</div>

??? success "연습문제 6 풀이"

    본문은 nearest Upsample로 7→14→28을 만든 뒤 stride=1 Conv2d를 쓴다. 전치 컨볼루션이 아니다. 아래는 같은 분리 방식의 다른 채널 예다. 전치 컨볼루션의 한 축 출력은 (H−1)s−2p+d(k−1)+output_padding+1이며 커널 4·스트라이드 2·패딩 1은 두 배 크기를 만들 수 있다. 겹침을 고르게 하는 설계는 일부 checkerboard 원인을 줄이지만 두 방식 모두 아티팩트가 전혀 없다는 보장은 없다.

    ```python
    nn.Upsample(scale_factor=2, mode='nearest')
    nn.Conv2d(64, 32, 3, padding=1)
    ```


---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
D의 Dropout2d는 어떤 역할을 하고 무엇을 보장하지 않는가?

</div>

??? success "연습문제 7 풀이"

    학습 중 피처 채널을 무작위 제거해 의존성을 분산시키는 정규화다. D의 모든 출력 확률에 상한을 강제하거나 비포화 G의 모든 그래디언트·안정성을 보장하지 않는다. 학습·평가 모드와 정규화의 상호작용을 기록하고 동일 설정에서 효과를 검증해야 한다. 이 페이지에는 드롭아웃 단독 효과를 분리한 비교 결과가 없다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
28×28 단일 채널과 64×64 RGB에 맞출 조건을 정리하라.

</div>

??? success "연습문제 8 풀이"

    시작 피처 크기·업샘플링 횟수·출력 채널,D의 공간 축소와 최종 선형 차원,자료 범위·평가 전처리를 함께 정한다. 7×7에서 두 배 두 번은 28×28을 만드는 한 설계일 뿐 유일한 방법은 아니다. 현재 D의 512 대128 불일치도 먼저 해결해야 한다. 특정 채널 폭이 MNIST에 항상 충분하거나 컨볼루션이 반드시 더 나은 품질을 준다는 보장은 없다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
두 네트워크의 용량·업데이트 균형을 어떻게 비교할 수 있는가?

</div>

??? success "연습문제 9 풀이"

    파라미터 수뿐 아니라 표현·데이터·정규화·로스·업데이트 빈도·시간 비용을 본다. D를 무조건 약하게 만드는 단일 처방은 적절하지 않다. 동일 검증 목적과 튜닝 예산에서 용량·러닝레이트·정규화 조합을 비교하고 여러 시드의 품질·다양성을 보고한다. 라벨 스무딩도 모델 출력의 강제 상한은 아니며 크기 비슷함 자체가 이론적 안정성 조건은 아니다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
아래 웨이트 초기화 예제가 적용하는 범위를 설명하라.

</div>

??? success "연습문제 10 풀이"

    Conv2d·ConvTranspose2d 웨이트는 N(0,0.02²),BatchNorm2d 스케일은 N(1,0.02²),바이어스는 0으로 초기화한다. 그러나 Conv bias나 Linear는 이 함수에서 초기화하지 않는다. 따라서 G의 큰 선형 투영과 D의 마지막 선형 레이어는 기존 초기화가 남는다. g,d도 실제로 정의해야 하며 본문 main에서는 이 함수가 적용되지 않는다. 초기화·시드와 실제 적용 범위를 기록하고 여러 시드로 평가한다.

    ```python
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

    g.apply(init_weights); d.apply(init_weights)
    ```


## 정리하며

이 모듈은 업샘플링·컨볼루션 G와 확률 판별기 D를 정의한다. 기본 28×28 입력에서 D의 펼친 크기가 출력 레이어와 맞지 않는 한계가 있다. 원 DCGAN 지침과 실제 코드,전처리·로스·평가 계약을 구분하고 크기부터 일관되게 맞춰야 한다.
