# 수축 오토인코더

수축 오토인코더(CAE)는 재구성 로스에 인코더의 국소 입력 민감도 패널티를 더한다. 작은 입력 변화에도 안정된 코드를 얻으려는 목적이지만, 전역적인 수축 사상이나 의미 있는 모든 방향의 보존을 보장하지는 않는다.

## 1. 목적과 국소 민감도

인코더 f:ℝᵈ→ℝᵏ와 디코더 g에 대해 다음 목적을 사용한다.

$$L=\mathbb E_x[L_{rec}(g(f(x)),x)]+\lambda\mathbb E_x\|J_f(x)\|_F^2,\qquad\lambda\ge0.$$

J_f의 크기는 k×d이고 프로베니우스 제곱 노름은 모든 입력·출력 편미분의 제곱합이다. 작은 δ에 대해 f(x+δ)−f(x)≈J_f(x)δ이므로 국소 코드 변화를 제한하는 지표다. 최악 방향 변화율은 연산자 노름에 대응하고 프로베니우스 노름은 그 상계다. 노름과 제곱 노름의 숫자도 구분한다.

재구성과 민감도는 절충할 수 있으나 잡음 방향만 선택적으로 줄이는 정리는 아니다. λ가 너무 크면 유용한 차이를 버릴 수 있다. 자유로운 디코더가 작은 코드 척도를 큰 웨이트로 리워드할 수 있어 인코더 수축만으로 전체 복원이 안정적이라고 할 수도 없다. [CAE 원 논문](https://icml.cc/2011/papers/455_icmlpaper.pdf)의 표현 민감도 목적과 구분해 평가한다.

## 2. DAE와의 관계를 정확히 읽기

디노이징 AE는 오염 입력에서 깨끗한 표적을 맞추고 CAE는 인코더 야코비안을 직접 제한한다. 작은 가우시안 입력 잡음의 제곱 복원 로스를 테일러 전개하면 전체 재구성 r=g∘f의 야코비안과, 잔차에 가중된 2차 미분 항이 나타난다. 추가 가정 없이 이를 σ²||J_f||_F²로 바꾸어 λ=σ²라고 할 수 없다.

두 방법은 안정적인 표현을 추구하는 관련 아이디어지만 같은 로스의 정확한 구현은 아니다. 잡음 생성과 추가 학습도 계산 비용이며 DAE가 “공짜로 같은 패널티”를 계산하는 것은 아니다.

## 3. 인코더와 야코비안 계산

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad

class ContractiveAutoencoder(nn.Module):
    """
    야코비 벌주기를 갖춘 오그리는 오토인코더.
    
    내놓기를 가두려 인코더에 에스자 활성화를 쓰며,
    그래야 야코비가 얌전하다.
    """
    
    def __init__(self, input_dim=784, latent_dim=64):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # 내놓기를 가두려 에스자를 쓴 인코더
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, latent_dim),
            nn.Sigmoid()  # 야코비가 안정되도록 [0,1]로 가둔다
        )
        
        # 디코더
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

def compute_jacobian_penalty(model, x):
    """
    인코더 야코비 행렬의 프로베니우스 노름 제곱을 셈한다.
    
    J_f(x)_ij = ∂z_i / ∂x_j
    ||J_f||²_F = Σ_ij (∂z_i / ∂x_j)²
    
    저절로 미분하기로 세로줄마다 셈한다.
    """
    x = x.requires_grad_(True)
    z = model.encode(x)
    
    jacobian_norm_sq = 0.0
    
    for i in range(z.shape[1]):
        grad_outputs = torch.zeros_like(z)
        grad_outputs[:, i] = 1.0
        
        jacobian_col = grad(
            outputs=z,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]
        
        jacobian_norm_sq = jacobian_norm_sq + torch.sum(jacobian_col ** 2)
    
    return jacobian_norm_sq / x.shape[0]  # 배치에 걸친 평균
```

기본 인코더는 784→256→128→64이며 각 선형 출력에 Sigmoid를 사용한다. 디코더는 64→128→256→784로 복원한다. `compute_jacobian_penalty`는 코드 성분마다 역방향 벡터-야코비안 곱을 구해 제곱합을 누적한다.

현재 인코더는 표본끼리 독립이므로 같은 코드 성분의 배치 합을 미분해도 표본별 야코비안의 해당 행을 얻는다. BatchNorm처럼 배치 표본을 결합하는 연산이 들어가면 교차 표본 미분이 더해져 이 해석이 달라진다. `create_graph=True`는 패널티를 파라미터까지 다시 미분하기 위한 설정이다.

코드 차원 k번의 역방향 호출과 추가 그래프가 필요하므로 비용이 크지만 전체 학습 시간이 단순히 k배라고 확정할 수는 없다. 단일 Sigmoid 계층의 해석적 식, 벡터화와 확률적 노름 추정 등도 가능하다. 예를 들어 E[vvᵀ]=I인 무작위 v에서 E||Jᵀv||²=||J||_F²를 사용하면 추정 분산과 계산비용을 절충할 수 있다.

## 4. 학습과 집계의 단위

```python
def train_contractive_autoencoder(
    model, train_loader, device, 
    lambda_contractive=0.1, num_epochs=15
):
    """
    오그리는 오토인코더를 익힌다.
    
    로스 = 다시 세우기 + λ × ||J_f||²_F
    """
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    recon_criterion = nn.MSELoss()
    
    history = {'recon_loss': [], 'contractive_loss': [], 'total_loss': []}
    
    for epoch in range(num_epochs):
        model.train()
        
        epoch_recon = 0
        epoch_contractive = 0
        epoch_total = 0
        
        for images, _ in train_loader:
            images = images.view(images.size(0), -1).to(device)
            
            optimizer.zero_grad()
            
            # 순전파
            recon, z = model(images)
            
            # 되살림 로스
            recon_loss = recon_criterion(recon, images)
            
            # 오그림 벌주기
            contractive_loss = compute_jacobian_penalty(model, images)
            
            # 전체 로스
            total_loss = recon_loss + lambda_contractive * contractive_loss
            
            total_loss.backward()
            optimizer.step()
            
            epoch_recon += recon_loss.item()
            epoch_contractive += contractive_loss.item()
            epoch_total += total_loss.item()
        
        n_batches = len(train_loader)
        history['recon_loss'].append(epoch_recon / n_batches)
        history['contractive_loss'].append(epoch_contractive / n_batches)
        history['total_loss'].append(epoch_total / n_batches)
        
        print(f"Epoch {epoch+1}: Recon={epoch_recon/n_batches:.6f}, "
              f"Contract={epoch_contractive/n_batches:.6f}")
    
    return history
```

복원은 모든 픽셀·표본의 평균 MSE이고 수축 항은 입력·코드의 제곱 편미분 합을 표본 수로 나눈 값이다. λ를 비교할 때 이 축과 입력 스케일을 명시한다. 학습 함수는 모델을 장치로 옮기지 않으므로 호출하는 쪽이 맞추어야 한다.

기록은 배치별 평균의 단순 평균이어서 마지막 배치 크기가 다르면 전체 표본 가중 평균과 다르다. 빈 로더도 처리하지 않는다. 코드에는 실행을 위한 로더 구성·함수 호출이 없고 테스트 복원 MSE를 계산하는 함수도 없다. 고정된 성능 표를 이 정의만으로 측정된 결과라고 제시할 수 없다.

## 5. 잡음 민감도 비교의 범위

```python
def analyze_contraction(model, test_loader, device, noise_std=0.1):
    """
    숨은 값의 바뀜과 들임의 바뀜의 비를 재어 배운 나타냄이
    얼마나 오그라드는지 살핀다.
    
    오그리는 인코더는 민감함 비가 1보다 훨씬 작다.
    """
    model.eval()
    
    sensitivity_scores = []
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.view(images.size(0), -1).to(device)
            
            z_original = model.encode(images)
            
            noise = torch.randn_like(images) * noise_std
            z_noisy = model.encode(images + noise)
            
            input_change = torch.norm(noise, dim=1)
            latent_change = torch.norm(z_noisy - z_original, dim=1)
            
            # 민감함 = ||Δz|| / ||Δx||
            sensitivity = latent_change / (input_change + 1e-8)
            sensitivity_scores.extend(sensitivity.cpu().numpy())
            
            if len(sensitivity_scores) > 1000:
                break
    
    return np.array(sensitivity_scores)

def compare_with_standard_ae(train_loader, test_loader, device):
    """
    인코더의 민감함으로 오그리는 오토인코더와 여느 오토인코더를 견준다.
    """
    # 여느 오토인코더(구조는 같고 오그림 벌주기는 없다)
    standard_ae = ContractiveAutoencoder().to(device)
    optimizer = optim.Adam(standard_ae.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print("Training Standard AE...")
    for epoch in range(15):
        standard_ae.train()
        for images, _ in train_loader:
            images = images.view(images.size(0), -1).to(device)
            optimizer.zero_grad()
            recon, _ = standard_ae(images)
            loss = criterion(recon, images)
            loss.backward()
            optimizer.step()
    
    # 오그리는 오토인코더
    contractive_ae = ContractiveAutoencoder().to(device)
    
    print("\nTraining Contractive AE...")
    train_contractive_autoencoder(
        contractive_ae, train_loader, device, 
        lambda_contractive=0.1, num_epochs=15
    )
    
    # 민감함을 견준다
    print("\nAnalyzing sensitivity to noise...")
    
    std_sensitivity = analyze_contraction(standard_ae, test_loader, device)
    cae_sensitivity = analyze_contraction(contractive_ae, test_loader, device)
    
    print(f"Standard AE sensitivity: {np.mean(std_sensitivity):.4f} "
          f"± {np.std(std_sensitivity):.4f}")
    print(f"Contractive AE sensitivity: {np.mean(cae_sensitivity):.4f} "
          f"± {np.std(cae_sensitivity):.4f}")
    
    # 시각화한다
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(std_sensitivity, bins=50, alpha=0.7, label='Standard AE')
    axes[0].hist(cae_sensitivity, bins=50, alpha=0.7, label='Contractive AE')
    axes[0].set_xlabel('Sensitivity (||Δz|| / ||Δx||)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Encoder Sensitivity Distribution')
    axes[0].legend()
    
    axes[1].boxplot([std_sensitivity, cae_sensitivity], 
                    labels=['Standard', 'Contractive'])
    axes[1].set_ylabel('Sensitivity')
    axes[1].set_title('Sensitivity Comparison')
    
    plt.tight_layout()
    plt.savefig('contractive_comparison.png', dpi=150)
    plt.show()
    
    return standard_ae, contractive_ae
```

`analyze_contraction`은 각 표본에 한 번의 가우시안 변화 δ를 더한 뒤 ||f(x+δ)−f(x)||/(||δ||+ε)를 기록한다. 유한 크기의 무작위 방향 변화율이며 최악 방향 야코비안 노름이나 패널티 자체와 같지 않다. 입력 범위를 [0,1]로 자르지 않으므로 범위 밖 점도 평가한다. 1,000개를 초과하면 배치 단위로 멈춰 반환 표본 수가 정확히 1,000개는 아니다.

두 모델은 새로 초기화되고 잡음도 별도로 뽑으므로 공정한 반복 비교에는 공통 초기값과 대응 잡음, 여러 시드가 유용하다. 작은 코드 변화가 재구성·분류 안정성으로 이어지는지도 따로 평가한다. 단순히 코드가 상수에 가까워져도 이 비율은 작아질 수 있다.

## 6. 수축·기하·해석의 한계

엄밀한 수축 사상은 어떤 고정 c<1에 대해 모든 점 쌍의 거리가 ||f(x)−f(y)||≤c||x−y||를 만족한다. c=1의 부등식은 비확장 사상이다. 학습 표본에서의 평균 패널티가 1보다 작다는 사실은 이런 전역 조건을 증명하지 않는다. 볼록 영역 전체에서 연산자 노름을 같은 c로 제한하는 조건 등은 별개의 충분조건이다.

야코비안 특이값은 학습된 인코더가 어느 입력 방향에 민감한지 보여 준다. 큰 특이값 방향이 반드시 참 데이터 매니폴드의 접방향이거나 0이 아닌 값의 개수가 내재 차원인 것은 아니다. 스케일·노이즈·모델과 λ에 의존한다. 표본 불확실성과 보류 데이터에서의 안정성을 함께 살핀다.

금융 요인 같은 응용에서도 작은 입력 변화에 대한 코드 안정성과 위험 예측·공분산 추정의 품질은 구분한다. 시장의 실제 국면 변화를 잡음으로 눌러 유용한 정보를 잃을 수도 있다. 데이터 척도와 사용 조건, 후속 태스크를 기준으로 λ와 다른 정규화를 비교한다.

## 연습문제


<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
λ를 높이면 반드시 테스트 복원이 악화되고 안정성이 개선되는가?

</div>

??? success "풀이"

    학습 목적에서 수축 항의 비중을 높이지만 유한 최적화·표본 외 성능의 단조 변화는 보장하지 않는다. 동일 조건에서 복원, 실제 야코비안 제곱 노름과 유한 잡음 변화율을 나누어 측정한다. 코드 스케일 축소만으로 낮아진 민감도가 의미 있는 안정성인지도 확인한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
아주 큰 λ가 항상 상수 인코더와 평균 복원을 강제하는가?

</div>

??? success "풀이"

    연결 영역 전체에서 야코비안이 0인 미분 가능한 인코더는 상수다. 고정 상수 출력의 제곱 오차 최적값은 데이터 평균이다. 그러나 패널티는 유한 표본에서 측정하고 디코더 스케일 리워드도 가능하므로 λ를 키우는 학습이 반드시 이 극한에 도달한다고 할 수 없다. 최적해 존재·웨이트 제약과 최적화 조건이 필요하다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
아래 z.sum() 미분이 프로베니우스 제곱 노름을 정확히 주지 않는 이유를 설명하라.

</div>

??? success "풀이"

    배치 독립 모델에서 각 입력의 그래디언트는 코드 축 편미분을 합한 Jᵀ1이다. 그 제곱에는 서로 다른 코드 행의 교차항이 있어 Σ_j||J_j||²와 다르다. 본문의 성분별 루프나 적절한 벡터화·확률적 노름 추정을 사용해야 한다. 입력 x와 encoder 정의도 필요하다.

    ```python
    x = x.requires_grad_(True)
    z = encoder(x)
    g = torch.autograd.grad(z.sum(), x, create_graph=True)[0]
    penalty = g.pow(2).sum() / x.size(0)
    ```


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
디코더 야코비안에 패널티를 주면 무엇이 달라지는가?

</div>

??? success "풀이"

    입력 변화에 대한 코드 민감도 대신 코드 변화에 대한 복원 민감도를 제한한다. 디코더 함수가 국소적으로 천천히 바뀌도록 유도할 수 있지만 코드 분포가 어디에 놓이는지는 정하지 않는다. 빈 잠재 구간의 의미나 샘플 생성 품질을 이것만으로 보장하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
수축 패널티와 L2 웨이트 정규화의 관계를 선형·비선형에서 비교하라.

</div>

??? success "풀이"

    단일 선형 인코더 f(x)=Wx에서는 J_f=W라 제곱 프로베니우스 패널티가 해당 W의 L2와 같다. 비선형·다층에서는 입력과 활성화 미분에 의존해 달라진다. 여러 레이어의 웨이트 각각을 줄이는 것과 합성 야코비안을 줄이는 것은 다르며 AdamW 등 decoupled 감쇠와 L2 로스도 구분한다.



---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
야코비안 특이값으로 해석할 수 있는 것과 단정할 수 없는 것을 설명하라.

</div>

??? success "풀이"

    그 지점의 학습된 코드가 민감한 입력 방향과 크기를 알 수 있다. 하지만 코드 스케일과 패널티에 의존하므로 큰 방향이 자동으로 참 접공간, 개수가 내재 차원이라는 결론은 성립하지 않는다. 데이터 변화·모델·λ와 함께 안정성을 평가한다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff easy" title="쉬움"></span>
수축 사상과 비확장 사상을 구분하고 평균 패널티의 한계를 설명하라.

</div>

??? success "풀이"

    비확장은 Lipschitz 상수 1, 수축은 전 영역에서 하나의 상수 c<1로 제한되는 성질이다. 평균 제곱 야코비안이 작아도 일부 점·방향의 큰 변화율이 남을 수 있다. 평균 실측값 0.538만으로 전역 수축을 증명할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
인코더 패널티가 있으면 디코더 정규화가 필요 없다는 주장은 맞는가?

</div>

??? success "풀이"

    재구성은 학습된 코드에서의 출력만 제약하고 디코더의 모든 위치·방향을 제약하지 않는다. 작은 코드 스케일을 큰 디코더 웨이트로 리워드하는 자유도도 남는다. 웨이트·디코더 민감도 정규화가 필요한지는 목적과 보류 성능에 따라 검토한다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
미분 가능한 무제약 내부 정지점에서 복원·패널티 그래디언트의 관계를 적어라.

</div>

??? success "풀이"

    L=L_rec+λΩ이면 ∇L_rec=−λ∇Ω다. 가중된 두 벡터가 상쇄한다는 뜻이며 둘 다 0일 수도 있다. 최적화가 그 정지점에 도달했다는 보장은 별개다. 서로 다른 λ 실행 사이 로스 변화의 유한 비율이 반드시 λ와 같다는 뜻도 아니다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
CAE 적용 여부를 판단할 평가 기준을 설명하라.

</div>

??? success "풀이"

    표현 민감도를 직접 제어할 필요와 계산·메모리 예산을 확인한다. 복원·후속 태스크·코드 스케일과 잡음 변화율을 보류 데이터에서 비교한다. DAE와 정확히 같은 효과라거나 CAE 특이값이 내재 차원을 자동으로 알려 준다고 가정하지 않는다. 각 구현의 비용과 실제 목표에 맞는 효과를 측정한다.



## 정리하며

CAE는 인코더의 국소 야코비안 민감도를 직접 제한한다. 계산한 패널티, 유한 잡음 변화율과 전역 수축은 다른 양이다. DAE와의 관계에는 추가 조건이 필요하며 코드 스케일·디코더 리워드와 표본 외 품질을 함께 평가해야 한다.

**참고 문헌**

- Rifai, S., et al. (2011). "Contractive Auto-Encoders: Explicit Invariance During Feature Extraction." *ICML*.
- Alain, G., & Bengio, Y. (2014). "What Regularized Auto-Encoders Learn from the Data-Generating Distribution." *JMLR*.
