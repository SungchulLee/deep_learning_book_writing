# VAE 모듈 구현

이 절은 다층 퍼셉트론으로 대각 가우시안 인코더와 베르누이 디코더를 구성한다. 인코더가 평균과 로그분산을 예측하고, 리파라메리제이션으로 얻은 잠재 표본을 디코더에 전달한다. 재구성 항과 표준정규 사전에 대한 KL 항을 함께 최소화하는 것이 학습 목표다.

## 1. 구현과 입출력

```python
"""
변분 오토인코더(VAE)
다시 파라메터화 재주로 확률 숨은 공간을 짠다
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class VAE(nn.Module):
    """
    정규 숨은 공간을 가진 여느 변분 오토인코더.
    
    인수:
        input_dim (int): 들임 차원(예컨대 MNIST는 784)
        hidden_dim (int): 숨은 레이어 차원
        latent_dim (int): 숨은 공간 차원
    """
    
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=32):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # 인코더: 숨은 분포의 파라메터를 내놓는다
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 숨은 분포의 평균과 로그 흩어짐
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # 디코더
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """
        들임을 숨은 분포 파라메터로 부호화한다.
        
        인수:
            x: 입력 텐서
            
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
        다시 파라메터화 재주: z = mu + sigma * epsilon
        여기서 epsilon ~ N(0, 1)
        
        그러면 뒤먹임 퍼뜨리기를 위해 뽑기를 미분할 수 있다.
        
        인수:
            mu: 숨은 분포의 평균
            logvar: 숨은 분포의 로그 흩어짐
            
        반환값:
            z: 뽑은 숨은 벡터
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """
        숨은 나타냄을 내놓기로 푼다.
        
        인수:
            z: 숨은 벡터
            
        반환값:
            reconstruction: 다시 세운 내놓기
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        온전한 피드포워드: 부호화 -> 다시 파라메터화 -> 풀기
        
        인수:
            x: 입력 텐서
            
        반환값:
            reconstruction: 다시 세운 내놓기
            mu: 숨은 분포의 평균
            logvar: 숨은 분포의 로그 흩어짐
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def loss_function(self, reconstruction, x, mu, logvar, beta=1.0):
        """
        변분 오토인코더 로스 = 다시 세우기 로스 + β * KL 벌어짐
        
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
        # 다시 세우기 로스(두 값 엇갈린 엔트로피)
        BCE = F.binary_cross_entropy(reconstruction, x, reduction='sum')
        
        # KL 벌어짐: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return BCE + beta * KLD, BCE, KLD
    
    def sample(self, num_samples, device='cpu'):
        """
        숨은 공간에서 표본을 만든다.
        
        인수:
            num_samples: 만들 표본의 개수
            device: 표본을 만들 기기
            
        반환값:
            samples: 만든 표본
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples


if __name__ == '__main__':
    # 모델을 시험한다
    model = VAE(input_dim=784, latent_dim=32)
    # BCE 로스는 목표가 [0,1]이어야 한다. randn은 음수를 내므로 rand를 쓴다
    x = torch.rand(32, 784)
    
    reconstruction, mu, logvar = model(x)
    loss, bce, kld = model.loss_function(reconstruction, x, mu, logvar)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Reconstruction Loss: {bce.item():.4f}")
    print(f"KL Divergence: {kld.item():.4f}")
    
    # 뽑기를 시험한다
    samples = model.sample(num_samples=10)
    print(f"Generated samples shape: {samples.shape}")
```

기본 입력은 배치 크기 B에 대한 `(B, 784)` 실수 텐서다. 이미지는 호출 전에 펼쳐야 하며 이 `forward`는 `(B,1,28,28)`을 자동 변환하지 않는다. 모델과 입력의 장치·자료형도 맞아야 한다. BCE 타깃은 0~1 범위여야 한다.

공유 인코더는 784→256→256이고 두 출력 헤드가 각각 32차원 평균과 로그분산을 만든다. 디코더는 32→256→256→784이며 마지막 Sigmoid가 픽셀별 확률을 출력한다. 따라서 `forward`의 반환값은 재구성 확률 `(B,784)`, 평균 `(B,32)`, 로그분산 `(B,32)`이다.

**기존 출력 기록:**

```
Input shape: torch.Size([32, 784])
Reconstruction shape: torch.Size([32, 784])
Latent mu shape: torch.Size([32, 32])
Latent logvar shape: torch.Size([32, 32])
Total Loss: 17424.4863
Reconstruction Loss: 17419.7031
KL Divergence: 4.7822
Generated samples shape: torch.Size([10, 784])
```

위 실행부는 무작위 입력으로 순전파·로스 계산·사전 샘플링의 모양을 확인한다. 최적화 반복문이 없으므로 학습 성능이나 생성 품질을 보여 주는 실험은 아니다. 초기화와 잠재 표본에 따라 로스 수치도 달라진다.

## 2. 로스의 합산 기준

`loss_function`은 `(전체 로스, BCE, KL)` 튜플을 반환한다. BCE는 모든 배치·픽셀 원소의 합이고 KL도 배치·잠재 차원의 합이다. β=1에서 전체 로스는 이 관측 모형의 음의 ELBO를 한 번의 잠재 표본으로 추정한 값이다. 데이터셋 평균을 보고하려면 배치별 합을 누적한 뒤 표본 수로 나눈다. 픽셀별 평균 MSE나 BCE와 수치 크기를 직접 비교하지 않는다.

엄밀한 베르누이 우도는 이진 관측에 해당한다. 0~1 회색조를 그대로 BCE 타깃으로 쓰는 관례는 가능하지만, 그 값이 연속 픽셀에 대한 정규화된 확률밀도의 로그우도인 것은 아니다. 실제 이진화 여부와 전처리를 기록해야 ELBO나 우도 수치를 해석할 수 있다.

## 3. 재구성과 생성의 확률성

이 구현의 `reparameterize`는 평가 모드에서도 표본을 뽑는다. `model.eval()`만으로 재구성 결과가 결정적으로 바뀌지 않는다. 평균 μ를 디코딩하는 별도 평가는 반복 가능한 시각화에 유용하지만 E_q[p(x|z)]를 정확히 계산하는 것은 아니다.

`sample`은 표준정규 사전에서 코드를 뽑고 디코더의 베르누이 확률을 반환한다. 확률 이미지를 표시하는 것과 그 확률로 픽셀을 다시 샘플링하는 것은 다르다. 기본 `device='cpu'`는 모델의 장치를 자동 추적하지 않으며 생성 코드의 기본 자료형도 모델과 맞춰야 한다. 이 메서드 자체는 평가 모드 설정이나 그래디언트 기록 중지를 수행하지 않는다.

모듈의 구조, 로스 함수와 평가 규칙을 구분하면 β 조절, Free Bits 또는 중요도 가중 평가를 추가할 때 바뀌는 부분을 명확히 설명할 수 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
기본 설정에서 학습 가능한 파라미터 수를 레이어별로 계산하라.

</div>

??? success "풀이"

    공유 인코더의 두 레이어는 각각 784×256+256=200,960개와 256×256+256=65,792개다. 평균·로그분산 헤드는 각각 256×32+32=8,224개다. 디코더의 세 레이어는 32×256+256=8,448개, 65,792개, 256×784+784=201,488개다. 합은 558,928개이며 ReLU·Sigmoid·리파라메리제이션에는 학습 파라미터가 없다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 모듈과 BCE 로스에 필요한 입력 조건을 설명하라.

</div>

??? success "풀이"

    입력은 `(B,input_dim)`이고 모델의 선형 레이어와 호환되는 부동소수점 자료형·장치여야 한다. B>0, 모든 값이 유한하며 BCE 타깃이 0~1 범위인지도 확인한다. float32만을 보편적인 필수 조건으로 둘 필요는 없다. float64 모델이나 혼합 정밀도 등 실제 실행 폴리시에 맞춘다. 4차원 이미지는 호출 전에 펼치고, 설정한 input_dim과 원소 수가 맞는지 검사한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
사후 붕괴와 수치 불안정을 어떻게 구분하고 진단할 수 있는가?

</div>

??? success "풀이"

    사후 붕괴에서는 입력별 근사 사후가 사전에 가까워지고 디코더가 잠재 표현을 거의 쓰지 않을 수 있다. 전체 KL뿐 아니라 차원별 KL, μ의 입력 의존성, 코드를 바꾸었을 때의 출력 변화를 함께 본다. 수치 불안정은 비유한 로스·그래디언트, 극단적인 로그분산·출력 확률 등으로 진단한다. 두 문제가 같은 것은 아니다. 러닝레이트·정밀도·로그분산 처리와 KL 가중 계획을 원인에 맞게 점검하며, 로그분산을 제한하면 샘플링과 KL에 같은 값을 사용해야 한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
인코더와 디코더의 은닉 레이어 수를 설정 가능하게 만들 때 어떤 조건을 유지해야 하는가?

</div>

??? success "풀이"

    인코더 첫 레이어의 입력 차원, 연속 레이어 사이의 차원, 마지막 공유 피처에서 두 헤드로 가는 차원을 맞춘다. 디코더는 latent_dim에서 시작해 input_dim으로 끝나야 한다. 두 헤드와 마지막 Sigmoid를 은닉 레이어 수와 혼동하지 않도록 깊이 정의도 명시한다. 아래는 등록 방식의 골격이며 YourBlock은 실제 구현으로 대체해야 한다. ModuleList에 등록한 레이어는 forward에서 직접 순회한다. 구조별 파라미터 수와 출력 모양을 확인하는 것만으로 학습 성능의 우열이 결정되지는 않는다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
forward가 세 텐서를 반환하는 이유와 loss_function 호출 결과를 설명하라.

</div>

??? success "풀이"

    재구성 항에는 디코더 출력이, 해석적 KL에는 평균과 로그분산이 필요하다. 따라서 세 값을 반환한다. 아래 첫 코드는 이 흐름을 보여 준다. 두 번째 코드의 loss에는 스칼라가 아니라 `(전체 로스,BCE,KL)` 튜플이 들어간다. 역전파하려면 전체 로스를 먼저 꺼내야 하며 튜플에 바로 backward를 호출할 수 없다. 잠재 표본 z도 반환하도록 확장할 수 있지만 평균과 표본은 서로 다른 값이다.

    ```python
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar     # 셋
    ```
    ```python
    out, mu, logvar = model(x)                # 셋을 받는다
    loss = model.loss_function(out, x, mu, logvar)
    ```


---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
평가에서 평균만 디코딩하도록 리파라메리제이션을 바꾸면 무엇이 달라지는가?

</div>

??? success "풀이"

    아래는 평가 모드에서 μ를 반환하는 대안이다. 본문의 원래 구현에는 이 분기가 없으므로 eval만으로 표본 추출이 중단되지 않는다. 평균 디코딩은 결정적인 재구성 비교에 편리하지만 사후 기대 재구성이나 몬테카를로 ELBO와는 다르다. 학습 중에도 평균만 사용하면 재구성 목적의 추정 방식이 달라진다. 해석적 KL이 남아 있다면 분산에 대한 그래디언트도 남으므로 곧바로 보통 오토인코더와 같아진다고 할 수 없다. 학습·평가 모드와 샘플링 여부를 별도 인수로 지정하는 설계도 가능하다.

    ```python
    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu                    # 평가할 때는 뽑지 않는다
        std = (0.5 * logvar).exp()
        return mu + torch.randn_like(std) * std
    ```


---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
같은 공유 인코더·디코더를 가진 결정적 오토인코더와 파라미터 수를 비교하라.

</div>

??? success "풀이"

    은닉 256, 잠재 32인 앞 장의 SimpleAutoencoder는 550,704개다. VAE는 같은 크기의 로그분산 헤드 8,224개가 더해져 558,928개로 약 1.49% 증가한다. 잠재 16이면 각각 542,496개와 546,608개로 차이는 4,112개다. 이 비교는 구조를 맞춘 경우에만 유효하다. 생성 품질의 변화는 파라미터 증가율만으로 설명할 수 없고, 사전·관측 모형·KL 학습 목표가 함께 달라진 결과다. 특정 분류기 확신도 비율을 파라미터 수의 효과로 단정할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
β와 Free Bits는 로스 계산을 어떻게 바꾸는가?

</div>

??? success "풀이"

    원래 loss_function에도 beta 인수가 있다. 아래 대안은 배치 크기로 두 항을 나누지만 스칼라 하나를 반환하므로 원래 튜플 인터페이스와 다르다. 호출부도 그 차이에 맞춰야 한다. Free Bits 예시는 차원별 KL의 배치 평균을 구하고 λ 아래에서 상수로 대체한 뒤 합한다. 이 경우 해당 항의 KL 그래디언트가 사라질 뿐 실제 KL이 λ 이상이 되도록 강제하지는 않는다. 양의 λ 설정과 재구성 항의 배치 정규화를 일치시켜야 한다. 조건부 VAE는 여기에 조건 정보를 전달하는 구조·입력 계약도 필요하다.

    ```python
    def loss_function(self, out, x, mu, logvar, beta=1.0):
        rec = F.binary_cross_entropy(out, x, reduction='sum') / x.size(0)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum() / x.size(0)
        return rec + beta * kl                        # 여기
    ```
    ```python
    kl_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())   # (B, k)
    kl = torch.clamp(kl_dim.mean(0), min=lam).sum()
    ```


---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
아래 중요도 가중 평가량이 보통 ELBO와 어떻게 다른지 설명하라.

</div>

??? success "풀이"

    양의 정수 K에 대해 q(z|x)에서 독립 표본을 뽑고 w=p(x,z)/q(z|x)를 계산한다. 코드는 입력별 log((1/K)Σw)를 반환하며 배치 평균은 호출부의 몫이다. 이 확률적 추정량의 기대값이 중요도 가중 하한이다. 한 번 얻은 수치 자체가 항상 log p(x) 이하이거나 K가 작은 별도 실행보다 큰 것은 아니다. 적절한 지지집합·적분 조건에서 기대 하한은 K에 따라 비감소하고 극한이 로그 주변우도에 접근한다. K=1도 한 표본 로그 중요도이며 본문의 해석적 KL을 사용한 ELBO 추정과 표본별 값은 다를 수 있다.

    이 평가는 단순히 같은 ELBO를 여러 번 평균 내는 것과 다르다. 큰 K에서의 개선을 상각 간격만의 측정값으로 읽을 수도 없다. math import, `(B,input_dim)` 입력과 관측 모형 조건이 필요하며 no_grad가 eval 설정까지 수행하지는 않는다. [IWAE 원 논문](https://arxiv.org/abs/1509.00519)의 하한은 기대값에 대한 결과다.

    ```python
    @torch.no_grad()
    def iwae_bound(self, x, K=64):
        # K개 표본으로 중요도 무게를 준 하한. K=1이면 보통 ELBO.
        mu, logvar = self.encode(x)
        mu = mu.unsqueeze(0).expand(K, -1, -1)          # (K, B, k)
        logvar = logvar.unsqueeze(0).expand(K, -1, -1)
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        z = mu + std * eps
        out = self.decode(z.flatten(0, 1)).view(K, x.size(0), -1)
        log_pxz = -F.binary_cross_entropy(out, x.expand(K, -1, -1),
                                          reduction='none').sum(-1)
        log_pz = -0.5 * (z.pow(2) + math.log(2 * math.pi)).sum(-1)
        log_qz = -0.5 * (eps.pow(2) + math.log(2 * math.pi) + logvar).sum(-1)
        w = log_pxz + log_pz - log_qz                   # (K, B)
        return torch.logsumexp(w, 0) - math.log(K)      # 표본마다
    ```


---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
체크포인트에 웨이트 외에 무엇을 기록해야 하는가?

</div>

??? success "풀이"

    실제 input_dim·hidden_dim·latent_dim, 전처리와 이진화 여부, 관측 모형, β·Free Bits 설정, 로스 합산 기준을 기록한다. 재학습을 이어 가려면 최적화 상태와 진행 위치 등도 필요하다. 아래 저장 예의 latent_dim=16은 본문 기본 모델의 32와 다르므로 실제 모델 설정으로 맞춰야 한다. x/255 전처리도 이미 ToTensor로 0~1 변환한 입력에 다시 적용하지 않도록 실제 경로를 기록한다. path는 호출자가 정해야 한다.

    β나 Free Bits를 바꾼 훈련 로스를 표준 음의 ELBO와 같은 지표로 비교해서는 안 된다. β>1 등 일부 설정에서 여전히 더 보수적인 하한을 만들 수 있지만 일반적인 변형 전체가 표준 ELBO인 것은 아니다. 같은 관측 모형에서 평가용 β=1·수정 없는 ELBO를 따로 계산하면 비교 기준을 맞출 수 있다.

    ```python
    torch.save({'state_dict': model.state_dict(),
                'config': {'input_dim': 784, 'hidden_dim': 256, 'latent_dim': 16},
                'preprocess': 'x / 255.0',
                'beta': 1.0,                   # 어떤 무게로 익혔는가
                'free_bits': 0.0}, path)       # 자유 비트를 썼는가
    ```


## 정리하며

VAE 모듈은 분포 파라미터 예측, 리파라메리제이션, 확률 디코딩과 KL 계산을 연결한다. 입력 모양, 로스 반환 형식과 합산 축을 먼저 확인하고, 평균 재구성·확률 이미지·관측 표본·우도 평가를 구분해서 사용한다. 기본 실행은 구조 확인용이며 학습 결과를 제시하지 않는다.
