# GAN의 적대적 학습

GAN(Generative Adversarial Network)은 표본을 만드는 생성기와 실제·생성 표본을 구분하는 판별기를 함께 학습한다. [Goodfellow 등의 원 논문](https://arxiv.org/abs/1406.2661)은 이 게임을 통해 생성 분포를 데이터 분포에 맞추는 틀을 제안했다. 경쟁이 항상 두 모델의 개선이나 수렴으로 이어지는 것은 아니며, 분포 일치에 관한 이상적 이론과 유한 뉴럴 넷트웍의 최적화를 구분해야 한다.

## 1. 생성 분포와 판별기

생성기 G는 z~p_Z를 데이터 공간의 G(z)로 보낸다. 이 표본 절차가 p_G를 정의하므로 모델 확률분포 자체가 없는 것은 아니다. 다만 일반적인 GAN에서는 데이터 x의 계산 가능한 밀도 p_G(x)를 직접 제공하지 않는다. 잠재 차원이 낮으면 데이터 공간의 보통 연속 밀도 자체가 없는 경우도 있다.

판별기 D(x)는 실제·생성 표본의 학습 분류 점수다. 두 종류를 같은 비중으로 섞은 이진 분류의 이상적 최적값에서만 다음 사후확률 비율을 얻는다.

$$D^*(x)=\frac{p_{data}(x)}{p_{data}(x)+p_G(x)}.$$

밀도는 공통 지배 측도에 대해 해석할 수 있고, 분모가 0인 영역의 D는 이 목적에서 정해지지 않는다. 학습 중 D가 보정된 “진짜 이미지일 절대 확률”을 내놓는다고 볼 수 없다. 클래스 비중·라벨 스무딩·다른 로스는 이 해석도 바꾼다.

## 2. 원래 최소최대 목적과 비포화 로스

원래 게임은 다음과 같다.

$$\min_G\max_D V(D,G)=\mathbb E_{p_{data}}\log D(x)
+\mathbb E_{p_Z}\log(1-D(G(z))).$$

D는 두 표본 분류의 로그우도를 키우고 G는 두 번째 항을 줄인다. D를 고정한 생성기 업데이트에서 실제 자료 항은 G와 무관하다. 이상적인 최적 판별기를 대입하면 −log 4+2 JS(p_data||p_G)를 얻지만, 매 미니배치의 뉴럴 넷트웍 업데이트가 이 최적 판별기 목적을 정확히 최적화하는 것은 아니다.

D(G(z))가 매우 작을 때 원래 생성기 목적은 포화될 수 있다. 실용적으로는 같은 고정점을 겨냥하는 **비포화 생성기 로스** −E_z log D(G(z))를 많이 사용한다. 아래 코드도 원래 minimax 생성기 로스가 아니라 이 비포화 로스다. 두 모델이 하나의 값을 반대 부호로 최적화하는 정확한 영합 게임과 실제 비포화 학습을 구분한다.

## 3. 교대 학습 구현

```python
import torch
import torch.nn as nn
import torch.optim as optim

def train_gan(generator, discriminator, dataloader, latent_dim, 
              n_epochs, device):
    """
    기본 맞겨루기 생성기 익히기 되풀이.
    
    인수:
        generator: 생성기 넷트웍 G(z) -> x
        discriminator: 판별기 넷트웍 D(x) -> [0, 1]
        dataloader: 실제 자료용 DataLoader
        latent_dim: 숨은 벡터 z의 차원
        n_epochs: 익히기 바퀴 수
        device: 익힐 장치(cpu 또는 cuda)
    """
    # 로스와 가장 좋게 하개
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    for epoch in range(n_epochs):
        for real_data, _ in dataloader:
            batch_size = real_data.size(0)
            real_data = real_data.to(device)
            
            # 이름표
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)
            
            # ==================
            # 판별기를 익힌다
            # ==================
            d_optimizer.zero_grad()
            
            # 실제 자료의 로스
            d_real = discriminator(real_data)
            d_loss_real = criterion(d_real, real_labels)
            
            # 가짜 자료의 로스
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_data = generator(z)
            d_fake = discriminator(fake_data.detach())  # G 그래디언트를 피하려 떼어 낸다
            d_loss_fake = criterion(d_fake, fake_labels)
            
            # 온 판별기 로스
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # ===============
            # 생성기를 익힌다
            # ===============
            g_optimizer.zero_grad()
            
            # 새 가짜 표본을 만든다
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_data = generator(z)
            d_fake = discriminator(fake_data)
            
            # G은 D이 가짜를 실제로 여기기를 바란다
            g_loss = criterion(d_fake, real_labels)
            g_loss.backward()
            g_optimizer.step()
```

모델·로더는 호출자가 제공한다. 판별기는 `(B,1)`의 0~1 확률을 반환해야 BCELoss와 레이블이 맞는다. logits를 반환하는 판별기라면 이 로스 계약과 다르다. 실제 자료와 생성 출력의 모양·범위, 모델·잡음의 장치·자료형을 맞춰야 한다. 이 함수는 두 모델을 device로 옮기거나 train 모드를 설정하지 않는다. z는 `(B,latent_dim)`이므로 다른 잠재 입력 형태의 생성기에는 조정이 필요하다.

D 단계는 실제 레이블 1·생성 레이블 0의 평균 BCE 두 개를 **합**한다. G 단계는 새 생성 표본에 레이블 1을 적용한다. Adam 러닝레이트 0.0002, betas=(0.5,0.999)는 예시 설정이며 모든 구조의 최적값은 아니다. 검증·체크포인트·실험 결과는 이 함수에 없다.

## 4. 그래프 연결과 “고정”의 의미

```python
d_fake = discriminator(fake_data.detach())
```

D 단계의 detach는 가짜 표본에서 G로 역전파하지 않게 한다. G를 호출하지 않거나 G의 BatchNorm 통계까지 고정하는 기능은 아니다. G가 학습 모드라면 이 단계에서도 내부 상태가 바뀔 수 있다.

G 단계는 D를 통과해 생성 데이터까지 그래디언트가 흐르도록 유지한다. 현재 코드에서는 D 파라미터의 grad도 계산되지만 D optimizer를 step하지 않는다. 다음 D 단계에서 초기화하므로 이 grad가 직접 다음 업데이트에 더해지지는 않지만 불필요한 계산은 남는다. 학습 모드의 D 버퍼·드롭아웃은 파라미터를 step하지 않는 것과 별도로 동작한다.

```python
# G 익히기용
z = torch.randn(batch_size, latent_dim)  # 새 잡음
fake_data = generator(z)  # 새 표본
```

이 골격은 새 잡음으로 G 표본을 만드는 예다. 새 z를 반드시 뽑아야만 올바른 업데이트가 되는 것은 아니다. D 업데이트 이후의 새 D forward와 G까지 연결된 유효한 그래프가 핵심이며, 필요하면 같은 z·가짜 표본을 적절히 재사용할 수도 있다. 별도 잡음을 쓰는 선택과 그래프 수명·상태 변경을 구분한다.

## 5. 균형과 로스 진단

무제한 표현과 적절한 최적화의 이상적 목표에서 p_G=p_data이고 D*=1/2가 데이터·생성 혼합의 거의 모든 곳에서 성립한다. 그러나 D가 항상 0.5를 내는 약한 분류기여도 같은 값이 보일 수 있어 그것만으로 분포 일치를 증명할 수 없다. 입력 공간 전체에서 반드시 0.5일 필요도 없다.

이 코드의 합산 관례에서 이상적 D 로스는 2 log 2≈1.386, 비포화 G 로스는 log 2≈0.693이다. 실제로 이 숫자를 유지하게 만드는 것이 학습 목표는 아니다. D 로스가 낮다고 항상 좋거나 항상 실패인 것도 아니며 분리 능력·유용한 그래디언트·표본 분포를 함께 봐야 한다. [내시 균형 절](nash.md)은 게임의 정지점과 수렴 문제를 구분한다.

## 6. 생성 품질과 다른 모델의 절충

판별기의 학습 신호는 고정된 픽셀 회귀 오차와 다른 분포 피처를 반영할 수 있다. 그러나 사람이 로스를 정하지 않아도 되는 것은 아니다. 게임 목적·모델 계열·정규화·자료 표현을 사람이 설계하고 판별기는 그 안에서 구분 피처를 학습한다.

일반적인 GAN은 잠재 표본과 생성기 순전파로 출력하지만 실제 속도는 구조에 달려 있다. VAE·흐름·확산과의 표본 품질·안정성·속도를 보편적인 순위로 비교할 수는 없다. 모드 붕괴·다양성 누락과 분류기 과적합도 가능하므로 [평가 절](../gan_evaluation/01_evaluation_basics.md)의 품질·다양성 지표와 대표 표본을 함께 사용한다. 특정 FID 수치는 피처 추출기·전처리·표본 수가 맞을 때만 비교할 수 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
원래 최소최대 GAN과 제공 코드의 목적을 비교하라.

</div>

??? success "연습문제 1 풀이"

    원래 D는 V를 최대화하고 G는 V의 생성 항을 최소화한다. 코드는 D에 실제·생성 BCE의 합을 최소화하고 G에는 −log D(G(z))를 최소화하는 비포화 목적을 쓴다. D가 고정된 한 단계와 판별기까지 최적화한 분포 목적은 다르다. 이상적 분포 일치가 유한 게임의 모든 정지점·실제 학습 수렴을 보장하는 것도 아니다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
최적 판별기를 유도하고 경계 사례를 설명하라.

</div>

??? success "연습문제 2 풀이"

    각 x에서 a=p_data(x),b=p_G(x)에 대해 a log D+b log(1−D)를 최대화한다. a,b>0이면 a/D−b/(1−D)=0에서 D=a/(a+b)이고 이차 미분이 음수다. a>0,b=0이면 경계 최적값은 1, a=0,b>0이면 0이다. 둘 다 0이면 목적이 D를 정하지 않는다. 이 결과는 같은 실제·생성 비중과 제한 없는 판별기 계열을 전제로 하며 실제 유한 판별기가 항상 그 값을 학습하는 것은 아니다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
모드 붕괴를 어떻게 진단하고 완화 전략을 평가할 수 있는가?

</div>

??? success "연습문제 3 풀이"

    생성 분포가 데이터의 일부 영역·변형을 거의 만들지 않는 현상이다. 클래스별 빈도뿐 아니라 클래스 내부 다양성·중복·학습 자료 암기와 피처 공간 포괄도를 살핀다. 배치 다양성 신호,로스·정규화·업데이트 계획의 변경 등을 같은 평가 조건에서 비교할 수 있지만 어느 하나가 붕괴를 항상 막지는 않는다. 같은 표본을 반복해도 판별기를 속이거나 단일 품질 점수가 높을 수 있어 품질과 다양성을 따로 확인한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
Sigmoid 판별기 logit a에서 포화·비포화 생성기 그래디언트를 비교하라.

</div>

??? success "연습문제 4 풀이"

    D=σ(a)이면 원래 log(1−D)의 a에 대한 미분은 −D, 비포화 −log D의 미분은 D−1이다. D≈0에서 앞 미분은 작고 뒤는 약 −1이므로 Sigmoid 로스 경로의 포화를 완화한다. 다만 생성기 파라미터까지의 그래디언트는 ∇_x a와 ∇_θG도 곱해져 유용한 방향·크기가 항상 보장되는 것은 아니다. D가 강하다는 말만으로 모든 로스의 그래디언트가 사라진다고 할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
GAN 로스 곡선을 단일 지도 학습 로스처럼 읽기 어려운 이유는 무엇인가?

</div>

??? success "연습문제 5 풀이"

    각 모델의 목적이 상대 모델의 변화에 따라 바뀐다. D의 분류 로스가 낮아지는 것이 생성 분포 개선과 같은 방향은 아니며 전체 학습이 단일 잠재 함수의 감소로 표현되지 않을 수도 있다. 지도 학습에서도 로스 정체가 항상 수렴·일반화를 뜻하지 않는다. 고정 잠재 표본과 새 표본,품질·다양성·검증 분류 동작을 함께 기록하고 단일 로스의 최솟값으로 G를 고르지 않는다.



---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
판별기를 “학습되는 로스”로 본다는 것은 무엇을 뜻하는가?

</div>

??? success "연습문제 6 풀이"

    판별기의 현재 피처와 출력이 생성기에 데이터 적합성 신호를 준다는 비유다. 목적 함수·표현·정규화까지 모두 자동으로 정해지는 것은 아니며 이 부분은 설계자가 정한다. 학습 신호가 고정 픽셀 오차보다 다른 분포 차이를 잡을 수 있지만 전체 데이터 분포나 사람이 보는 품질을 완벽히 평가하는 잣대는 아니다. 검증 지표도 판별기와 별도로 필요하다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff hard" title="어려움"></span>
픽셀 MSE의 조건부 평균 성질이 왜 모든 VAE의 흐릿함을 설명하지는 못하는가?

</div>

??? success "연습문제 7 풀이"

    고정 조건에서 단일 결정적 출력 c의 기대 제곱오차는 c=E[X|조건]에서 최소다. 불확실한 여러 이미지의 평균은 흐릿할 수 있다. 그러나 조건이 충분한 정보를 담거나 확률 디코더가 표본을 만들면 이야기가 달라지며 모든 픽셀 로스가 반드시 흐린 출력을 만든다는 결론은 아니다. L1의 최적값은 조건부 중앙값이며 보편적으로 검은 이미지가 되는 것도 아니다. GAN의 모드 붕괴와 평균 회귀는 다른 원인·평가를 가진 문제라 같은 축의 두 끝으로 단정하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
판별기 신호·지각 로스·FID의 차이를 설명하라.

</div>

??? success "연습문제 8 풀이"

    GAN 판별기는 현재 생성기와 실제 자료를 구분하도록 함께 학습하며 G의 업데이트 신호로 사용한다. 지각 로스는 보통 고정된 피처 공간의 비교를 목적에 넣고 FID는 피처 분포의 평균·공분산을 평가한다. 모두 뉴럴 피처를 사용할 수 있지만 학습·평가 역할과 통계적 의미가 다르다. D 로스는 해석할 수 없는 숫자가 아니라 현재 게임의 분류 진단 값이며 생성 품질의 단조 지표가 아닌 것이다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
GAN 학습과 적대적 예제에 대한 강건 학습의 공통점·차이를 설명하라.

</div>

??? success "연습문제 9 풀이"

    둘 다 상충하는 목적과 교대 최적화를 사용할 수 있다. GAN은 잠재 표본을 생성하는 G의 파라미터와 분류 D를 학습한다. 강건 분류의 대표 목적은 min_θ E[max_{δ∈S}ℓ(f_θ(x+δ),y)]로 허용 집합 안의 입력 교란에 대한 최악 로스를 줄인다. 입력 교란·레이블 보존·허용 집합의 계약은 GAN 생성과 다르며 표준 GAN을 학습했다고 적대적 강건성이 보장되지는 않는다. 모든 공격이 입력 그래디언트를 직접 사용하는 것도 아니다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
안정화 기법을 평가할 때 단일 시드 FID로 확정할 수 없는 것은 무엇인가?

</div>

??? success "연습문제 10 풀이"

    모델·전처리·피처 추출기·표본 수·훈련 예산을 맞추고 여러 초기화·체크포인트의 변동을 본다. 정규화·러닝레이트·라벨 스무딩의 효과는 구조와 목적에 따라 달라 단일 결과에서 필수 조건이나 최적 처방을 도출할 수 없다. 라벨 스무딩도 모든 출력에 강제된 확률 상한은 아니다. 큰 FID 차이도 측정 계약·실행 실패·표본 수 영향을 먼저 확인해야 하며 이 페이지에는 해당 비교 실험이 없다.



---

<div class="drillbox" markdown>

**연습문제 11.** <span class="diff hard" title="어려움"></span>
PCA·AE·VAE·GAN을 학습 목표 관점에서 비교하라.

</div>

??? success "연습문제 11 풀이"

    PCA는 선형 재구성,AE는 뉴럴 넷트웍 재구성,VAE는 확률 잠재 모형의 변분 목표,GAN은 적대적 분포 학습을 대표한다. 각 방법은 다른 가정과 목적의 절충이며 차례로 모든 이전 한계를 제거하는 보편적 우열 관계가 아니다. AE에도 잠재 분포를 붙여 생성할 수 있고 VAE가 반드시 흐리거나 GAN이 반드시 더 좋은 것도 아니다. GAN은 계산 가능한 우도를 보통 제공하지 않지만 확률분포 자체를 잃은 것은 아니다. 목적·자료·평가·비용에 맞춰 비교한다.



## 정리하며

GAN은 생성 표본과 실제 표본의 구분 신호로 생성 분포를 학습한다. 원래 최소최대 이론과 비포화 코드,그래프 분리와 상태 고정,판별 로스와 생성 품질을 구분해야 한다. 분포 일치의 이상적 조건이 실제 교대 학습의 안정성이나 다양성을 보장하지는 않는다.
