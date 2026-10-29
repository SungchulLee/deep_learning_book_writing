# 완전 연결 GAN 모듈

이 절은 잡음 벡터를 이미지로 바꾸는 생성기와 이미지의 실제·생성 여부를 분류하는 판별기를 정의한다. GAN 클래스는 두 모듈을 등록하고 forward에서는 생성기만 호출한다. 모듈 정의 자체가 적대적 학습을 수행하거나 생성 품질을 보장하는 것은 아니다.

## 1. 구현과 파라미터 출력

```python
#!/usr/bin/env python3
'''
GAN - 맞겨루기 생성기
논문: "Generative Adversarial Networks" (2014)
핵심: 맞겨루며 익히는 두 넷트웍(생성기와 판별기)
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_size=28):
        super().__init__()
        self.img_size = img_size
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, img_size * img_size * 1),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, self.img_size, self.img_size)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_size=28):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(img_size * img_size * 1, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

class GAN(nn.Module):
    def __init__(self, latent_dim=100, img_size=28):
        super().__init__()
        self.generator = Generator(latent_dim, img_size)
        self.discriminator = Discriminator(img_size)
    
    def forward(self, z):
        return self.generator(z)

if __name__ == "__main__":
    model = GAN()
    print(f"Generator Parameters: {sum(p.numel() for p in model.generator.parameters()):,}")
    print(f"Discriminator Parameters: {sum(p.numel() for p in model.discriminator.parameters()):,}")
```

기본 생성기는 `(B,100)` 잠재 표현을 128→256→512→1,024→784의 선형 레이어로 확장하고 `(B,1,28,28)`로 바꾼다. 중간 256·512·1,024차원에 BatchNorm1d,각 은닉 레이어에 LeakyReLU를 사용하며 마지막 Tanh는 출력 범위를 −1~1로 제한한다.

판별기는 입력 이미지를 펼쳐 784→512→256→1로 줄이고 Sigmoid 확률을 반환한다. 이는 BCELoss를 사용하는 예시와 맞는 계약이며 BCEWithLogitsLoss에는 마지막 Sigmoid 없는 logits 출력이 필요하다. 실제 자료도 생성기 범위·전처리와 맞춰야 한다.

**기존 출력 기록:**

```
Generator Parameters: 1,510,032
Discriminator Parameters: 533,505
```

실행부는 생성기·판별기의 파라미터 수만 출력한다. 학습 루프·로스·평가·시각화 결과는 없다. 생성기는 1,510,032개,판별기는 533,505개의 학습 파라미터를 가진다. BatchNorm affine은 포함하고 running 통계 버퍼는 제외한다. 기존 출력이 다른 값을 표시한다면 이 구조 계산과 구분해야 한다.

## 2. 활성화와 정규화의 역할

Tanh는 모델 출력의 단위를 정하는 선택이지 모든 GAN의 필수 요소는 아니다. 데이터가 0~1 범위라면 Sigmoid 등과 맞추는 다른 설계도 가능하다. Sigmoid 생성기가 항상 더 느리거나 Tanh가 항상 좋은 그래디언트를 준다는 일반 결론은 없다.

LeakyReLU는 음의 구간에도 기울기 0.2를 주지만 모든 뉴런·전체 모델의 그래디언트 소실을 막는 것은 아니다. BatchNorm은 학습 배치 통계와 평가 running 통계를 다르게 사용한다. 이 생성기는 학습 모드의 배치 1에서 BatchNorm1d 통계를 계산할 수 없으므로 단일 생성에는 eval 등 적절한 모드가 필요하다. eval은 그래디언트 기록을 중지하는 기능과 다르다.

## 3. 별도로 필요한 학습 계약

[적대적 학습](../gan_foundations/adversarial.md)에서는 D를 실제·생성 분류로, G를 D를 통과한 생성기 목적로 교대 업데이트한다. [원래·비포화 로스](../gan_losses/original.md)의 부호와 레이블이 다르므로 이 클래스만으로 어떤 목적을 쓸지 정해지지 않는다.

D 단계의 가짜 표본은 G 그래프에서 분리하고 G 단계는 D를 통해 G까지 그래디언트를 유지한다. optimizer 파라미터도 두 모델별로 정해야 한다. 파라미터 수의 비율만으로 판별기의 분류 난이도·학습 속도·안정성 균형을 판정할 수 없다.

생성 분포는 잠재 분포와 G로 정의되지만 일반적인 GAN은 계산 가능한 p_G(x)를 직접 제공하지 않는다. 낮은 D 로스·D 평균 0.5·안정된 로스 곡선도 생성 분포 일치의 증명은 아니다. 품질·다양성·독립 평가와 계산비용을 함께 확인한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
기본 구조의 레이어별 파라미터 수와 비교의 한계를 설명하라.

</div>

??? success "풀이"

    G의 각 선형·해당 BatchNorm 합은 12,928,33,536,132,608,527,360,803,600개로 총 1,510,032개다. 256→512 레이어는 선형 131,584개와 BatchNorm 1,024개를 합친 132,608개다. D는 401,920+131,328+257=533,505개다. G가 약 2.83배 많지만 이것만으로 D가 더 쉽거나 두 모델의 러닝레이트·업데이트 비율이 어떻게 되어야 하는지는 결정되지 않는다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
생성기의 Tanh와 판별기의 Sigmoid를 데이터·로스 계약과 연결해 설명하라.

</div>

??? success "풀이"

    Tanh 생성 출력은 −1~1이므로 실제 자료도 같은 단위로 변환해야 한다. 판별기 Sigmoid는 BCELoss가 받는 확률을 만든다. D를 Tanh로 바꾸면 음수가 나와 확률 BCE와 맞지 않는다. D를 logits로 바꾸고 BCEWithLogitsLoss를 쓰는 대안은 가능하다. G에 Sigmoid를 쓰려면 실제 자료 범위도 0~1로 맞춘다. 어느 활성화가 항상 더 좋은 생성·속도를 주는지는 구조·학습 조건에 달려 있다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
64×64 RGB로 확장할 때 필요한 차원과 파라미터를 계산하라.

</div>

??? success "풀이"

    출력 차원은 12,288이다. G 마지막 레이어는 Linear(1024,12288),reshape는 `(B,3,64,64)`,D 첫 레이어는 Linear(12288,512)로 바꿔야 한다. 현재 생성 reshape의 채널 1도 함께 수정해야 하며 img_size=64만으로 RGB가 되지 않는다. 나머지 구조를 유지하면 G는 13,301,632개,D는 6,423,553개다. 비용·공간 공유를 고려해 컨볼루션 구조를 비교할 수 있지만 단순 차원 확장이 품질을 보장하지는 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
두 네트워크의 역할과 GAN.forward의 동작을 설명하라.

</div>

??? success "풀이"

    G는 잠재 잡음에서 이미지를 만들고 D는 이미지에서 실제·생성 분류 점수를 계산한다. GAN.forward(z)는 G(z)만 반환하며 D 분류나 로스·교대 학습을 자동 수행하지 않는다. 사전 생성에는 G와 잠재 분포가 필요하다. D는 학습 재개·진단·별도 피처 작업을 위해 보관할 수 있지만 그 용도별 성능도 따로 검증해야 한다.



---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
D 로스가 작다는 사실에서 무엇을 추론할 수 있는가?

</div>

??? success "풀이"

    현재 로스·레이블·평가 자료 아래 실제·생성 구분을 잘한다는 신호일 수 있다. 원래 Sigmoid minimax 로스에서는 작은 가짜 확률이 G logit 그래디언트를 줄일 수 있지만 비포화 목적에서는 해당 경로가 다르다. 따라서 낮은 D 로스가 항상 좋은 생성이나 항상 학습 정지를 뜻하지 않는다. D·G 목적,출력·그래디언트,품질·다양성 지표를 함께 본다. D≈0.5도 약한 판별기일 수 있다.



---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
G가 잠재 입력을 거의 무시하는지 어떻게 진단할 수 있는가?

</div>

??? success "풀이"

    여러 z의 출력 차이·유사도·클래스 내부 다양성,잠재 표현 개입과 생성 포괄도를 함께 본다. 고정 z 격자는 학습 변화 추적에 좋지만 새로운 z도 평가해야 한다. 인셉션 점수 1은 평가기의 조건부 클래스 분포가 입력에 의존하지 않는 경우에도 나오므로 모든 이미지가 동일하거나 반드시 모드 붕괴라는 증명은 아니다. 클래스 엔트로피도 클래스 내 다양성을 보지 못한다. 로스만으로도 확정할 수 없어 여러 진단을 결합한다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
GAN·VAE의 표본을 공정하게 비교하려면 무엇이 필요한가?

</div>

??? success "풀이"

    동일한 평가 피처·전처리·표본 수·자료 분할·선택 예산과 변동을 보고한다. FID는 선택 피처의 평균·공분산 비교이고 MNIST 분류기 피처를 쓰면 표준 ImageNet Inception FID와 다른 측정 계약이다. 이 페이지에는 해당 비교 실험이 없어 일정한 점수 우열을 제시할 근거가 없다. VAE가 반드시 흐리고 GAN이 항상 날카로우며 우도에서는 무조건 순위가 뒤집힌다고 결론 내리지 않는다. GAN도 분포는 정의하지만 일반적으로 계산 가능한 밀도는 제공하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
최적 판별기 대입과 현재 G 업데이트를 구분하라.

</div>

??? success "풀이"

    제한 없는 D와 균형 잡힌 분류에서 D*=p_data/(p_data+p_G)이고,원래 전체 밸류의 최적 D 목적은 −log4+2JS다. 밀도 합이 0인 곳은 식이 D를 정하지 않는다. 실제 유한 D를 고정한 비포화·미니배치 G 업데이트가 항상 정확한 JS 경사라는 뜻은 아니다. 서로 특이한 분포에서는 JS가 log2가 될 수 있지만 고차원 생성 분포가 항상 참 분포와 겹치지 않거나 모든 대안 로스가 유용한 경사를 보장하는 것도 아니다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
D:G 업데이트 횟수와 러닝레이트를 어떻게 선택·보고할 수 있는가?

</div>

??? success "풀이"

    D 최적 반응의 근사 정도와 G의 학습 신호·비용을 조절하는 값으로 검증한다. D를 많이 업데이트하는 것이 항상 좋거나 나쁘다는 일반 규칙은 없다. 원래·비포화·Wasserstein 등 목적·제약·자료·정규화에 따라 상호작용한다. 고정 시간·반복·튜닝 예산 중 무엇을 맞췄는지 명시하고 여러 초기화에서 품질·다양성·수치 상태를 비교한다. 이 모듈에는 횟수나 실제 성능 결과가 없다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
생성에 필요한 저장 상태와 학습 재개 상태를 구분하라.

</div>

??? success "풀이"

    생성에는 G 구조·웨이트·BatchNorm 버퍼,잠재 분포·차원·이미지 전처리 계약이 필요하다. 아래는 웨이트 저장·생성 골격이며 n을 정의하고 모델·잡음 장치·자료형을 맞춰야 한다. 실제 생성에는 eval·no_grad 등 평가 폴리시도 정한다. 학습 재개에는 D와 두 optimizer·진행 위치·관련 난수 상태까지 필요하다. D를 다른 피처 작업에 쓴다고 성능이 자동 보장되는 것은 아니다.

    ```python
    torch.save(g.state_dict(), 'generator.pt')     # 이것만 있으면 된다
    z = torch.randn(n, latent_dim)
    imgs = g(z)
    ```


## 정리하며

이 모듈은 완전 연결 G와 확률 판별기 D를 정의하며 학습 자체는 포함하지 않는다. 실제 입력 범위·BCE 출력 계약·BatchNorm 모드와 파라미터 수를 확인한다. 생성 품질·다양성과 학습 균형은 별도의 학습·검증 절차로 평가해야 한다.
