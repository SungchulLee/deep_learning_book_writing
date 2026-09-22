# 오그리는 자기 부호기
들임의 흔들림에 대한 부호기의 민감함에 대놓고 벌을 주어 튼튼한 나타냄을 배운다.

---

**배울 것:**

- 부호기의 야코비 행렬과 민감함을 재는 데서의 노릇
- 오그림 벌주기: 부호기 야코비 행렬의 프로베니우스 노름
- 다양체 배움으로 보는 기하 풀이
- 잡음 없애는 자기 부호기와의 이음
- PyTorch에서 효율 좋고 정확하게 야코비 행렬 셈하기

---

## 1. 수학적 바탕

### 오그림 벌주기

**오그리는 자기 부호기(CAE)**는 여느 다시 세우기 손실에 부호기 야코비 행렬의 프로베니우스 노름 벌주기를 더한다:

$$\mathcal{L} = \underbrace{\|x - g(f(x))\|^2}_{\text{다시 세우기}} + \underbrace{\lambda \|J_f(x)\|_F^2}_{\text{오그림 벌주기}}$$

여기서 각 기호는 다음과 같다.

- $f$: 들임을 숨은 나타냄에 옮기는 부호기 함수
- $g$: 숨은 것을 들임 공간으로 되돌리는 풀개 함수
- $J_f(x) = \frac{\partial f(x)}{\partial x} \in \mathbb{R}^{k \times d}$: 부호기의 야코비 행렬
- $\|J_f\|_F^2 = \sum_{ij} J_{ij}^2$: 프로베니우스 노름의 제곱(모든 편미분의 제곱합)
- $\lambda$: 다시 세우기와 오그림의 맞바꿈을 다스리는 벌주기 세기

### 직관

| 조각 | 효과 |
|-----------|--------|
| 다시 세우기 손실 | 들임을 충실히 다시 세우는 법을 배운다 |
| 야코비 벌주기 | 부호기가 들임의 흔들림에 무디게 한다 |

야코비 벌주기는 다음을 이끈다:

- **국소 불변:** 들임이 조금 바뀌면 숨은 부호도 조금만 바뀐다
- **튼튼한 나타냄:** 부호기가 요긴한 짜임은 잡고 잡음은 무시하는 법을 배운다
- **평평한 다양체:** 숨은 공간이 잡음 방향으로는 그때그때 일정하고 자료 다양체를 따라서는 바뀐다

### 맞바꿈

$$\text{작은 } \lambda \to \text{다시 세우기는 낫고 튼튼함은 덜하다}$$

$$\text{큰 } \lambda \to \text{튼튼함은 더하고 다시 세우기는 못하다}$$

다시 세우기 항은 (정확히 다시 세우려) $f$이 들임의 모든 흔들림에 민감하기를 바라고 오그림 항은 무디기를 바란다. 이 저울질이 부호기를 다시 세우는 데 중요한 방향, 곧 자료 다양체를 따라서만 민감하게 만든다.

---

## 2. 잡음 없애는 자기 부호기와의 이음

### 이론의 이음

흩어짐이 $\sigma^2$인 작은 정규 잡음에서 잡음 없애는 자기 부호기의 목표는 대략 다음을 가장 작게 한다:

$$\mathcal{L}_{DAE} \approx \|x - g(f(x))\|^2 + \sigma^2 \|J_f(x)\|_F^2$$

**핵심 눈썰미:** 정규 잡음으로 잡음을 없애는 것은 넌지시 오그림 벌주기를 쓰는 것이며 잡음 흩어짐 $\sigma^2$이 $\lambda$ 노릇을 한다.

### 견줌

| 갈래 | 잡음 없애는 자기 부호기 | 오그리는 자기 부호기 |
|--------|--------------|----------------|
| 벌주기 | 망가뜨린 들임으로 | 또렷한 야코비 벌주기로 |
| 셈하기 | 잡음을 곁들인 여느 앞먹임 | 야코비 셈하기가 필요하다 |
| 융통성 | 여러 잡음 갈래를 쓸 수 있다 | 오그림 세기를 곧바로 다스린다 |
| 풀이 | 잡음 없애는 법을 배운다 | 부호기의 민감함을 가장 작게 한다 |
| 익히기 값 | 여느 것 | 더 높다(야코비가 비싸다) |

---

## 3. PyTorch 구현

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad

class ContractiveAutoencoder(nn.Module):
    """
    야코비 벌주기를 갖춘 오그리는 자기 부호기.
    
    내놓기를 가두려 부호기에 에스자 깨어남을 쓰며,
    그래야 야코비가 얌전하다.
    """
    
    def __init__(self, input_dim=784, latent_dim=64):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # 내놓기를 가두려 에스자를 쓴 부호기
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, latent_dim),
            nn.Sigmoid()  # 야코비가 안정되도록 [0,1]로 가둔다
        )
        
        # 복호기
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
    부호기 야코비 행렬의 프로베니우스 노름 제곱을 셈한다.
    
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
    
    return jacobian_norm_sq / x.shape[0]  # 묶음에 걸친 평균
```

---

## 4. 학습

```python
def train_contractive_autoencoder(
    model, train_loader, device, 
    lambda_contractive=0.1, num_epochs=15
):
    """
    오그리는 자기 부호기를 익힌다.
    
    손실 = 다시 세우기 + λ × ||J_f||²_F
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
            
            # 되살림 손실
            recon_loss = recon_criterion(recon, images)
            
            # 오그림 벌주기
            contractive_loss = compute_jacobian_penalty(model, images)
            
            # 전체 손실
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

---

## 5. 민감함 살피기

```python
def analyze_contraction(model, test_loader, device, noise_std=0.1):
    """
    숨은 값의 바뀜과 들임의 바뀜의 비를 재어 배운 나타냄이
    얼마나 오그라드는지 살핀다.
    
    오그리는 부호기는 민감함 비가 1보다 훨씬 작다.
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
    부호기의 민감함으로 오그리는 자기 부호기와 여느 자기 부호기를 견준다.
    """
    # 여느 자기 부호기(얼개는 같고 오그림 벌주기는 없다)
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
    
    # 오그리는 자기 부호기
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

---

## 6. 기하학적 해석

### 다양체 배움의 관점

오그림 벌주기는 부호기가 다음과 같은 대응을 배우도록 이끈다:

1. **자료 다양체 방향 지킴:** 부호기가 자료가 실제로 놓인 방향을 따라 바뀐다
2. **잡음 방향 오그림:** 다양체 밖 방향(잡음)은 숨은 공간에서 거의 0의 바뀜으로 옮겨진다
3. **평평한 숨은 다양체:** 숨은 나타냄이 잡음 방향으로는 그때그때 일정하다

이것이 바로 차원 높은 공간에 묻힌 자료 다양체의 속 기하를 배우는 데 바라는 몸가짐이다.

---

## 7. 계량 금융에서의 응용

오그리는 자기 부호기는 금융에서 **안정된 요인 나타냄**을 배우는 데 값지다:

- **튼튼한 위험 요인:** 오그림 벌주기가 시장 자료의 작은 흔들림(호가 튐, 미시 짜임 잡음)이 뽑은 요인을 바꾸지 않게 해 더 안정된 위험 쪼개기를 낸다
- **국면에 흔들리지 않는 특징:** 민감함에 벌을 주므로 배운 특징이 잠깐의 시장 어긋남에 덜 흔들린다
- **벌주기를 갖춘 공분산 어림:** 오그리는 부호기가 배운 공분산 짜임에 넌지시 벌을 주어 차원 높은 상황에서 어림 어긋남을 줄인다

---

## 연습문제


<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
오그림 벌점의 세기 $\lambda$를 0에서 0.1로 올리면 야코비 노름과 다시 세우기 오차가 각각 어떻게 되는가?

</div>

??? success "연습문제 1 풀이"
    병목 16, 10 에포크로 재면 이렇다.

    | $\lambda$ | 0 | 0.1 |
    |---|---|---|
    | 야코비 노름 (평균) | 24.329 | **0.538** |
    | 시험 MSE | 0.01014 | 0.01067 |

    야코비 노름이 **45분의 1**로 줄어드는 동안 다시 세우기 오차는 5%만 나빠진다.

    이 비율이 이 방법의 값어치를 말해 준다. 민감도를 크게 낮추면서 되돌리는 힘은
    거의 잃지 않았다는 뜻이며, 눌린 방향이 애초에 자료를 되돌리는 데 별로 쓰이지
    않던 방향이었음을 보여 준다.

    $\lambda$를 더 넓게 쓸어 보면 그 값어치가 더 또렷해진다.

    | $\lambda$ | 0 | 0.001 | 0.01 | 0.1 | 1.0 |
    |---|---|---|---|---|---|
    | 야코비 노름 | 24.329 | 7.636 | 3.532 | 0.538 | **0.069** |
    | 시험 MSE | 0.01014 | **0.01014** | 0.01029 | 0.01067 | 0.01028 |

    $\lambda = 0.001$ 칸을 보라. 야코비가 **3분의 1**로 줄었는데 다시 세우기 오차는
    소수점 다섯째 자리까지 똑같다. **공짜로 얻는 튼튼함**이다.

    그리고 MSE가 $\lambda$에 대해 단조롭지 않다. $\lambda = 1.0$이 $\lambda = 0.1$보다
    오히려 낫다. 벌점이 정칙화로도 작용해서 생기는 일이며, 맞바꿈이 깔끔한 곡선일
    것이라는 기대가 늘 맞지는 않음을 보여 준다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
$\lambda$를 아주 크게 하면 무슨 일이 일어나겠는가? 그 극한에서 부호기는 무엇이 되는가?

</div>

??? success "연습문제 2 풀이"
    $\lambda \to \infty$이면 오그림 항이 손실을 지배하므로 야코비를 0으로 만드는 것이
    최선이 된다. 야코비가 어디서나 0인 함수는 **상수 함수**다.

    곧 부호기가 모든 입력을 같은 코드로 보낸다. 그러면 풀개는 입력이 무엇이든 같은
    그림 하나만 내놓을 수 있고, 그 최선은 학습 자료의 평균 그림이다.

    실제로 $\lambda$를 키워 가며 재어 보면 야코비 노름이 0으로 가면서 다시 세우기
    오차가 자료의 분산 쪽으로 올라간다. 이는 "아무것도 배우지 않음"에 해당하는 값이다.

    그러므로 $\lambda$는 **되돌릴 수 있을 만큼은 민감하게, 그 이상은 둔감하게**를
    가르는 손잡이다. 좋은 값은 자료에 따라 다르며, 대개 다시 세우기 오차가 눈에 띄게
    나빠지기 직전까지 올린다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
야코비 노름을 실제로 어떻게 셈하는가? 코드 차원이 $k$일 때 비용이 얼마나 드는가?

</div>

??? success "연습문제 3 풀이"
    ```python
    x = x.requires_grad_(True)
    z = encoder(x)
    g = torch.autograd.grad(z.sum(), x, create_graph=True)[0]
    penalty = g.pow(2).sum() / x.size(0)
    ```

    `create_graph=True`가 핵심이다. 벌점을 다시 미분해야 하므로 기울기 계산 자체가
    그래프에 남아야 한다.

    다만 위 코드는 지름길이다. `z.sum()`을 미분하면 야코비의 **열 합**이 나올 뿐
    각 성분의 제곱합이 아니다. 정확한 프로베니우스 노름을 얻으려면 코드 차원마다
    따로 역전파해야 하므로 $k$번의 역전파가 든다.

    $$\text{비용} \approx k \times (\text{한 번의 역전파})$$

    코드가 16이면 16배, 128이면 128배다. 오그리는 자기 부호기가 널리 쓰이지 않는
    가장 큰 까닭이 이 비용이며, 실무에서 [잡음 없애는 쪽](../architecture/03_ae_denoising.md)을
    고르는 이유이기도 하다. 잡음은 공짜로 같은 효과를 어림한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
오그림 벌점을 부호기가 아니라 **풀개**에 걸면 어떻게 되는가?

</div>

??? success "연습문제 4 풀이"
    뜻이 달라진다.

    부호기의 야코비 $\partial z / \partial x$를 누르면 "입력이 흔들려도 코드가 덜
    흔들린다"는 뜻이었다. 풀개의 야코비 $\partial \hat{x} / \partial z$를 누르면
    **"코드가 흔들려도 그림이 덜 흔들린다"**가 된다.

    이쪽은 오히려 숨은 공간을 **매끄럽게** 만드는 쪽으로 작용한다. 가까운 코드가
    비슷한 그림을 내놓게 되므로, 코드 사이를 이어 가며 뽑으면 그림이 급격히 바뀌지
    않는다.

    그런데 이것만으로는 [뽑기 문제](../limits/latent_sampling.md)가 풀리지 않는다.
    풀개를 매끄럽게 해도 코드가 **어디에 놓이는지**는 여전히 아무도 단속하지 않기
    때문이다. 빈 곳에서 뽑으면 매끄럽게 바뀌는 얼룩이 나올 뿐이다.

    그래도 이 생각은 살아 있다. 풀개의 야코비를 제어하는 것은 숨은 공간의 기하를
    다루는 여러 방법의 출발점이고, 특히 흐름 모델에서는 그 야코비의 행렬식이
    **모델의 정의 자체**에 들어간다([27장](../../ch27/index.md)).

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
오그림 벌점과 가중치 감쇠는 둘 다 무언가를 작게 만든다. 무엇이 다른가?

</div>

??? success "연습문제 5 풀이"
    누르는 대상이 다르다.

    | | 무엇을 작게 하는가 | 자료에 기대는가 |
    |---|---|---|
    | 가중치 감쇠 | **가중치** $\lVert W \rVert^2$ | 아니다 |
    | 오그림 | **야코비** $\lVert \partial z/\partial x \rVert^2$ | 그렇다 |

    가중치 감쇠는 자료를 보지 않고 매개변수만 본다. 어디서나 똑같이 가중치를 0 쪽으로
    민다.

    오그림 벌점은 **자료가 있는 자리**에서 야코비를 잰다. 그래서 자료가 몰려 있는
    곳에서는 세게 누르고, 자료가 없는 곳은 신경 쓰지 않는다. 앞의 연습문제에서 본
    "다양체를 따르는 방향은 살리고 벗어나는 방향만 누른다"가 가능한 것도 이 때문이다.

    선형 모델이라면 둘이 같아진다. $z = Wx$의 야코비가 $W$ 자체이므로 야코비를 누르는
    것이 곧 가중치를 누르는 것이다. 비선형이라야 둘이 갈라진다.


---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
오그림 벌점을 준 부호기의 야코비 특이값을 구해 보라. 무엇을 기대할 수 있는가?

</div>

??? success "연습문제 6 풀이"
    특이값이 **몇 개만 크고 나머지는 0에 가깝게** 갈린다.

    큰 쪽이 자료가 실제로 퍼져 있는 방향, 곧 다양체의 접방향이다. 그 방향으로는
    민감해야 자료를 되돌릴 수 있으므로 벌점을 무릅쓰고 살려 둔다. 0에 가까운 쪽은
    자료가 놓이지 않은 방향이며, 눌러도 잃을 것이 없으니 벌점이 시키는 대로 눌린다.

    그러므로 **0이 아닌 특이값의 개수가 그 자리에서 다양체의 차원을 어림한 값**이 된다.
    자리마다 다르게 나올 수 있고, 자료가 굽어 있으면 실제로 다르게 나온다.

    이것이 오그리는 자기 부호기가 다른 정칙화와 구별되는 점이다. 벌점을 주면서도
    **자료의 기하를 읽어 낼 수 있는 물건**을 남긴다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff easy" title="쉬움"></span>
이 방법의 이름이 '오그리는'(contractive)인 까닭은 무엇인가?

</div>

??? success "연습문제 7 풀이"
    수학에서 사상 $f$가 **오그림 사상**(contraction)이라는 것은 두 점 사이의 거리를
    늘리지 않는다는 뜻이다.

    $$\lVert f(x) - f(y) \rVert \le \lVert x - y \rVert$$

    야코비의 노름을 작게 하면 국소적으로 이 성질에 가까워진다. 입력이 조금 움직일 때
    코드가 그보다 덜 움직이므로, 부호기가 공간을 **오그린다.**

    다만 이름이 약속하는 것보다는 느슨하다. 벌점은 야코비를 작게 만들 뿐 1 이하로
    묶어 주지는 않으므로, 익힌 부호기가 엄밀한 뜻의 오그림 사상이라는 보장은 없다.
    실제로 $\lambda = 0.1$에서 잰 야코비 노름이 0.538이었으니 그 경우에는 성립하지만,
    $\lambda$가 작으면 1을 넘는다.

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
오그림 벌점은 부호기에만 걸고 풀개에는 걸지 않는다. 풀개는 정칙화하지 않아도 되는가?

</div>

??? success "연습문제 8 풀이"
    이 방법의 관심이 **나타냄**에 있기 때문이다. 무엇을 코드에 담고 무엇을 버릴지는
    부호기가 정하므로, 버리라는 압력도 부호기에 걸어야 한다.

    풀개를 정칙화하지 않아도 되는 까닭은, 풀개가 마음대로 해도 **다시 세우기 손실이
    이미 단속하고 있기** 때문이다. 풀개가 엉뚱한 함수가 되면 곧바로 오차로 드러난다.
    반면 부호기는 코드를 아무렇게나 흩어 놓아도 풀개가 따라가 주기만 하면 오차가
    늘지 않는다. 단속받지 않는 자유가 부호기 쪽에 있는 것이다.

    이 비대칭이 이 장 전체를 관통한다. [뽑기 문제](../limits/latent_sampling.md)도
    부호기가 코드를 어디에 놓든 벌점이 없다는 같은 사실에서 나온다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
오그림 벌점 $\lVert \partial z/\partial x \rVert_F^2$과 다시 세우기 손실이 맞설 때, 최적해에서 두 기울기 사이에 어떤 관계가 성립하는가?

</div>

??? success "연습문제 9 풀이"
    최적해에서는 전체 손실의 기울기가 0이므로

    $$\nabla_\theta \mathcal{L}_{\text{rec}} + \lambda \nabla_\theta \Omega = 0
    \qquad\Longrightarrow\qquad
    \nabla_\theta \mathcal{L}_{\text{rec}} = -\lambda \nabla_\theta \Omega$$

    곧 두 기울기가 **크기가 같고 방향이 반대**다. 다시 세우기를 조금 더 잘하려고
    움직이면 그만큼 정확히 오그림 벌점이 늘어나는 자리에서 멈춘다.

    여기서 $\lambda$의 뜻이 분명해진다. $\lambda$는 **교환 비율**이다. 오그림 벌점
    한 단위를 얻기 위해 다시 세우기 오차를 얼마나 내줄 용의가 있는지를 정한다.

    실제로 잰 값이 이 그림과 맞는다. $\lambda = 0.1$에서 벌점이 24.329에서 0.538로
    23.8 줄고 오차는 0.01014에서 0.01067로 0.00053 늘었다. 비가 약 $0.00053/23.8
    \approx 2 \times 10^{-5}$로 $\lambda$와 같은 자릿수는 아니지만, 이는 벌점이 선형이
    아니어서 평균 비율과 한계 비율이 다르기 때문이다. 최적해 **바로 그 자리**에서만
    비가 $\lambda$와 맞는다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
오그리는 자기 부호기를 실제 일에서 쓸 것인가? 쓴다면 어떤 자리인가?

</div>

??? success "연습문제 10 풀이"
    솔직히 말해 자주 쓰이지 않는다. 야코비를 셈하는 값이 코드 차원에 비례해 드는데,
    [잡음 없애는 자기 부호기](../architecture/03_ae_denoising.md)가 공짜로 비슷한
    효과를 내기 때문이다.

    그래도 값어치가 있는 자리가 있다.

    - **다양체의 차원을 재고 싶을 때.** 야코비의 특이값이 자리마다의 접공간 차원을
      알려 준다. 잡음 방식은 이런 물건을 남기지 않는다.
    - **벌점을 정확히 조절해야 할 때.** 잡음은 어림이라 세기를 정밀하게 다루기
      어렵지만, $\lambda$는 정확한 손잡이다.
    - **이론을 따질 때.** 다른 정칙화들이 무엇을 하는지 견주는 기준으로 쓰인다.

    곧 실무의 연장이라기보다 **무슨 일이 벌어지는지 이해하기 위한 연장**이다. 이 절을
    읽는 값어치도 거기에 있다.

## 정리하며

| 갈래 | 여느 자기 부호기 | 오그리는 자기 부호기 |
|--------|-------------|----------------|
| 손실 | 다시 세우기만 | 다시 세우기 + $\|J\|_F^2$ |
| 민감함 | 높음(제약 없음) | 낮음(설계상) |
| 튼튼함 | 제한됨 | 나아짐 |
| 셈하기 | 빠름 | 느림(야코비 셈하기) |
| 다양체 배움 | 넌지시 | 벌주기로 또렷이 |

**핵심 눈썰미:** 오그리는 자기 부호기는 부호기의 민감함에 곧바로 벌을 주어 들임의 흔들림에 튼튼한 나타냄을 배우는 원칙 있는 길을 준다. (작은 정규 잡음에서) 잡음 없애는 자기 부호기와 이론상 같다는 점이 겉보기에 다른 두 벌주기 전략을 하나의 얼거리로 아우른다.

---

**참고 문헌**

- Rifai, S., et al. (2011). "Contractive Auto-Encoders: Explicit Invariance During Feature Extraction." *ICML*.
- Alain, G., & Bengio, Y. (2014). "What Regularized Auto-Encoders Learn from the Data-Generating Distribution." *JMLR*.
