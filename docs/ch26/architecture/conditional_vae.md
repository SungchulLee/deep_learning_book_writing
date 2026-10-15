# 조건부 VAE

조건부 VAE는 조건 y가 주어졌을 때의 데이터 분포 p(x|y)를 모델링한다. 여기서는 숫자 클래스의 원핫 벡터를 인코더와 디코더 입력에 연결한다. 조건을 지정해 표본을 생성할 수 있지만 지정 클래스의 정확한 생성이나 클래스·스타일의 완전한 분리가 자동으로 보장되지는 않는다.

## 1. 완전 연결 구현

```python
"""조건부 변분 오토인코더(cVAE)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalVAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=32, num_classes=10):
        super().__init__()
        self.input_dim, self.latent_dim, self.num_classes = input_dim, latent_dim, num_classes
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid()
        )

    def encode(self, x, c):
        return self.fc_mu(self.encoder(torch.cat([x, c], dim=1))), \
               self.fc_logvar(self.encoder(torch.cat([x, c], dim=1)))

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z, c):
        return self.decoder(torch.cat([z, c], dim=1))

    def forward(self, x, labels):
        c = F.one_hot(labels, self.num_classes).float() if len(labels.shape) == 1 else labels
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

    def sample(self, class_label, num_samples, device='cpu'):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        labels = torch.tensor([class_label] * num_samples).to(device)
        c = F.one_hot(labels, self.num_classes).float()
        return self.decode(z, c)

if __name__ == '__main__':
    model = ConditionalVAE()
    x = torch.randn(32, 784)
    labels = torch.randint(0, 10, (32,))
    reconstruction, mu, logvar = model(x, labels)
    print(f"Reconstruction shape: {reconstruction.shape}")
```

입력 x는 `(B,input_dim)`이다. labels가 1차원이면 정수 클래스 인덱스 `(B,)`에서 원핫 `(B,num_classes)`를 만든다. 이미 2차원이면 그대로 조건 벡터로 사용한다. 배치 크기·장치·자료형과 클래스 범위를 호출부에서 맞춰야 한다. float()로 만든 원핫은 float32이므로 다른 정밀도의 모델과 사용할 때도 확인이 필요하다.

인코더의 첫 층은 input_dim+num_classes를, 디코더의 첫 층은 latent_dim+num_classes를 받는다. 기본 설정에서 각각 794→256, 42→256이다. 두 분포 헤드는 32차원이며 전체 학습 파라미터는 564,048개다. encode는 같은 조건부 공유 인코더를 두 번 계산한다. 현재 공유부가 결정적이므로 같은 특징을 얻지만 중복 계산이 있다.

**기존 출력 기록:**

```
Reconstruction shape: torch.Size([32, 784])
```

실행 예는 임의 입력의 순전파 모양만 확인하며 학습하지 않는다. randn 입력은 모양 확인에는 사용할 수 있지만 BCE 타깃으로 그대로 쓸 수는 없다. 클래스에는 loss_function이 정의되어 있지 않아 학습하려면 조건부 재구성·KL 계산을 호출부 등에서 별도로 구성해야 한다.

## 2. 조건부 ELBO와 사전

이 구현은 조건과 무관한 표준정규 사전을 사용한다. 조건부 ELBO는 다음과 같다.

$$\mathcal L(x,y)=\mathbb E_{q_\phi(z|x,y)}\log p_\theta(x|z,y)-D_{KL}(q_\phi(z|x,y)\|p(z)).$$

일반적으로 p(z|y)를 사용할 수도 있으며 이때 KL과 생성 시의 사전 샘플링을 함께 바꾼다. 조건부 잠재 모형의 접근은 [Sohn 등의 논문](https://proceedings.neurips.cc/paper/2015/file/8d55a249e6baa5c06772297520da2051-Paper.pdf)에서도 다룬다.

생성에는 x나 인코더가 필요하지 않다. sample은 지정 클래스와 사전 표본을 디코더에 전달하고 픽셀별 확률을 반환한다. 모델의 장치·자료형을 맞추고 필요에 따라 eval·no_grad를 적용해야 한다. 재매개변수화는 평가에서도 계속 표본을 뽑는다.

## 3. 조건 제어와 표현 해석

조건을 직접 제공하면 잠재 코드가 클래스 정보를 중복해서 저장할 필요를 줄일 수 있다. 그러나 학습 목표가 z와 y의 독립성이나 클래스 간 같은 z의 동일한 스타일 의미를 보장하지는 않는다. 조건 없이 인코딩하는 q(z|x)도 변분 분포로 가능하며 표현·추론 성능의 차이는 별도로 검증해야 한다.

원핫 조건을 선형 혼합해 디코더에 넣는 실험은 가능하지만 이 클래스에는 interpolate 메서드가 없다. 학습에서 보지 않은 혼합 조건에 대한 출력이나 매끄러운 의미 변화도 보장되지 않는다. 조건 지정 성공률, 생성 다양성과 재구성 성능을 각각 평가해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
숫자별 표본 격자를 만들 때 아래 코드의 전제와 평가 한계를 설명하라.

</div>

??? success "연습문제 1 풀이"

    학습된 모델과 matplotlib import가 필요하다. 기본 CPU 모델에서는 각 행에 한 클래스의 확률 이미지를 표시한다. GPU 텐서를 numpy로 바꾸려면 먼저 CPU로 이동해야 하며 sample의 device도 모델과 맞춰야 한다. eval은 기울기 기록을 끄지 않지만 detach는 표시용 텐서의 그래프 연결을 끊는다. 이 코드는 학습을 수행하지 않으며 목표 클래스 일치나 다양성은 결과를 관찰·측정해야 한다. 격자만으로 전체 생성 분포를 판단하지 않는다.

    ```python
    model.eval()
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for digit in range(10):
        samples = model.sample(digit, 10).view(-1, 28, 28).detach()
        for j in range(10):
            axes[digit, j].imshow(samples[j].numpy(), cmap='gray')
            axes[digit, j].axis('off')
    plt.show()
    ```


---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 클래스에서 3과 8의 조건 보간을 설계하고 한계를 설명하라.

</div>

??? success "연습문제 2 풀이"

    고정 잠재 코드에 c(t)=(1−t)e₃+t e₈을 연결해 decode에 넣을 수 있다. interpolate라는 메서드는 구현되어 있지 않으므로 호출만으로 동작하지 않는다. 여러 z에서 경로를 비교하되 혼합 원핫은 학습 조건 분포 밖일 수 있음을 밝힌다. 중간 이미지가 반드시 특정 숫자나 같은 스타일을 유지한다고 예측할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
인코더에 조건을 주지 않는 변형도 조건부 ELBO를 정의할 수 있는가?

</div>

??? success "연습문제 3 풀이"

    가능하다. q(z|x)를 q(z|x,y)의 제한된 계열로 사용해 E_q log p(x|z,y)−KL(q(z|x)||p(z))를 계산할 수 있다. 디코더가 여전히 y를 받으므로 코드가 반드시 클래스 정체를 모두 저장해야 하는 것은 아니다. 추론 계열의 제한이 재구성·우도·잠재 표현에 미치는 영향은 데이터와 구조에 달려 있으며 품질 저하나 해석성 향상을 단정하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
조건을 양쪽 네트워크에 넣는 역할을 구분하라.

</div>

??? success "연습문제 4 풀이"

    인코더 조건은 q(z|x,y)가 관측과 클래스에 따라 사후를 근사하게 하고, 디코더 조건은 p(x|z,y)가 지정 클래스의 출력을 만들도록 한다. 아래는 연결 방식의 개념적 골격이다. encoder가 두 값을 반환하도록 정의되어 있어야 하며 y는 연결 가능한 조건 벡터다. 이 설계만으로 코드에서 클래스 정보가 제거되거나 같은 z가 클래스 간 동일한 글씨체를 뜻하는 것은 아니다.

    ```python
    mu, logvar = encoder(torch.cat([x, y], dim=1))   # 인코더
    xhat       = decoder(torch.cat([z, y], dim=1))   # 디코더
    ```


---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
조건부·무조건부 생성의 품질을 공정하게 비교하려면 무엇을 맞춰야 하는가?

</div>

??? success "연습문제 5 풀이"

    같은 평가 분류기·전처리·표본 수에서 각 지표의 의미를 명시한다. 조건부 생성의 지정 클래스 판정률과 무조건부 생성의 최대 확률 문턱 통과율은 서로 다른 지표라 직접적인 개선율로 비교할 수 없다. 클래스별 빈도·다양성·품질도 함께 본다. KL이나 재구성 수치의 차이만으로 클래스 정보가 얼마나 제거되었는지 증명할 수 없으며 본문에는 비교 실험 결과가 없다.



---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
조건부 KL은 무엇을 비교하며 어떤 정보 분해가 가능한가?

</div>

??? success "연습문제 6 풀이"

    본문은 KL(q(z|x,y)||p(z))를 사용하며 사전은 클래스와 무관한 N(0,I)다. 기대 KL은 I_q(X;Z|Y)+E_y KL(q(z|y)||p(z))로 나뉜다. 따라서 조건이 주어진 상태에서 입력 정보와 조건별 집계 불일치를 함께 벌한다. 이것이 유한 학습에서 완전한 클래스 독립성을 보장하지는 않는다. p(z|y)로 바꾸면 두 번째 항의 기준도 조건부 사전으로 바뀌며 클래스별 잠재 영역이 반드시 서로 분리되는 것은 아니다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
원핫과 임베딩 조건을 비교하라.

</div>

??? success "연습문제 7 풀이"

    원핫에 편향 없는 선형 변환을 적용하는 것은 해당 가중치 열을 선택하는 임베딩 조회와 같다. 편향이 있으면 공통 오프셋이 추가된다. 따라서 원핫 입력 뒤 학습 가능한 층도 클래스별 관계를 표현할 수 있다. 큰 클래스 수에서는 임베딩이 긴 원핫의 명시적 저장을 피하고 차원·비용을 조절하는 데 유용하지만 의미적 유사성을 자동으로 보장하지는 않는다. 연속 조건에는 원값·주기 인코딩·학습 변환 등 조건의 구조에 맞는 표현을 선택한다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
잠재 코드에 클래스 정보가 남아 있는지 어떻게 진단할 수 있는가?

</div>

??? success "연습문제 8 풀이"

    학습에서 분리한 자료로 코드만 받는 분류기를 학습·평가하고 클래스 빈도 기준선과 비교한다. 균형 10클래스라면 무작위 기준이 10%지만 불균형이면 다르다. 높은 검증 정확도는 정보가 남아 있다는 근거다. 반면 낮은 정확도는 선택한 분류기·자료·최적화로 정보를 찾지 못했다는 뜻이며 독립성의 증명이 아니다. μ를 검사하는 것과 q 표본 z를 검사하는 것도 구분한다. 같은 z에서 조건만 바꾸는 시각화는 보완 자료이며 스타일 분리의 정량 증거를 대체하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
학습에 없던 클래스·색 조합을 조건으로 줄 수 있다는 것과 일반화 성공은 어떻게 다른가?

</div>

??? success "연습문제 9 풀이"

    클래스와 색을 분리해 표현하면 미관측 조합도 형식상 입력할 수 있다. 그러나 올바른 조합을 생성하는지는 데이터의 요인 관계와 학습 편향에 달려 있다. 조합별 단일 ID를 쓰더라도 미관측 ID가 사전에 정의되어 있으면 입력 자체는 가능하지만 해당 임베딩이 학습되지 않아 일반화를 기대할 근거가 약하다. 표현 가능성·학습된 조건의 범위·새 조합의 성능을 구분한다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
조건부 생성의 비용과 사용 목적을 설명하라.

</div>

??? success "연습문제 10 풀이"

    학습·추론에서 사용할 조건과 그 품질을 확보해야 한다. 조건은 사람의 클래스 주석에 한정되지 않고 다른 관측이나 속성일 수도 있다. 무조건부처럼 사용하려면 별도 p(y)에서 조건을 뽑아 p(x)=Σ_y p(y)p(x|y)로 혼합할 수 있다. 무조건부 모델도 실제 데이터의 클래스 비율을 정확히 재현한다고 보장되지는 않는다. 조건부 코드에 클래스 정보가 전혀 없거나 분류에 반드시 쓸모없다고 단정할 수도 없다. 생성 제어와 특징 재사용 중 목적을 정하고 각각 검증해야 한다.



## 정리하며

조건부 VAE는 클래스 정보를 입력으로 받아 조건별 생성 분포를 학습한다. 본문은 표준정규 사전과 원핫 연결을 사용하지만 학습 손실·반복문·조건 보간은 별도 구현이 필요하다. 조건 제어와 코드의 정보 분리는 구분해서 평가한다.
