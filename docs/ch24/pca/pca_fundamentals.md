# 주성분 분석의 기초

주성분 분석(PCA)은 평균을 뺀 데이터에서 분산이 큰 직교 방향을 찾아 저차원 좌표로 나타내는 선형 차원 축소 방법이다. 같은 데이터의 유클리드 제곱 재구성 오차를 최소화하는 부분공간을 찾는 문제와 같다. 이 최적성은 분류 성능이나 모든 종류의 정보 보존을 뜻하지 않는다.

## 1. 중심화와 주성분

행에 표본을 둔 데이터 X∈ℝⁿˣᵈ에서 학습 평균 μ를 빼 X_c를 만든다. 이 절의 이론은 공분산을 Σ=X_cᵀX_c/n으로 정의하고 고윳값을 λ₁≥⋯≥λ_d≥0으로 정렬한다. 1/(n−1)을 쓰면 방향과 설명 분산 비율은 같지만 고윳값 크기가 달라진다.

단위 벡터 v 방향 점수의 분산은 vᵀΣv다. 이를 최대화하면 가장 큰 고윳값의 고유벡터를 얻는다. 앞선 방향과 직교하도록 반복하면 주성분 기저를 구성한다. 중복 고윳값에서는 해당 고유공간의 기저가 유일하지 않다.

$$W=[v_1,\ldots,v_k],\quad W^TW=I_k,\qquad Z=X_cW,\qquad
\hat{X}_c=ZW^T.$$

원래 단위로 재구성하려면 μ를 다시 더한다. 중심화하지 않은 XᵀX는 공분산이 아니라 평균 항 μμᵀ까지 포함한 2차 모멘트다. 그 첫 방향이 항상 평균과 정확히 일치하는 것은 아니다.

## 2. 표준화는 목적에 따른 선택

중심화와 피처별 표준화는 다르다. 표준화는 각 피처를 표준편차로 나누어 상대적인 변동을 비슷한 척도로 비교한다. 단위가 다르면 유용할 수 있지만 반드시 해야 하는 규칙은 아니다. 같은 단위여도 변동성 차이를 제거하고 싶을 수 있고, 서로 다른 단위의 크기 자체가 목적상 중요할 수도 있다. 미터로 표현한 값은 같은 물리량을 킬로미터로 표현한 값보다 숫자가 1,000배 커지므로 단위 선택이 결과에 영향을 준다.

표준화한 데이터의 PCA는 같은 분모 관례로 구한 상관행렬의 고유분해에 해당한다. 분산이 0인 피처는 별도로 처리한다. 학습 데이터에서 구한 평균·스케일을 검증·테스트에도 그대로 적용해 데이터 누출을 피한다.

```python
def preprocess(X):
    """주성분 분석을 위해 자료의 가운데를 맞추고 잣수를 맞춘다."""
    mu = X.mean(axis=0)
    X_centered = X - mu

    sigma = X_centered.std(axis=0)
    X_scaled = X_centered / (sigma + 1e-10)  # 0으로 나누는 것을 피한다

    return X_scaled, mu, sigma
```

위 함수는 호출마다 평균과 표준편차를 새로 추정한다. 테스트 데이터에 독립적으로 다시 호출하는 변환기가 아니다. ε를 더하므로 분산이 매우 작은 피처는 정확히 단위 분산이 되지 않으며, 이 처리가 잡음 증폭 문제를 자동으로 해결하지도 않는다.

## 3. 주성분 계수와 점수 해석하기

여기서는 W의 열을 주성분 방향 또는 로딩 벡터라 부른다. 표본의 점수는 z=Wᵀ(x−μ)다. 같은 학습 데이터에서 점수의 공분산은 diag(λ₁,…,λ_k)이므로 서로 상관이 없지만 일반적으로 독립인 것은 아니다.

로딩의 절댓값은 해당 좌표의 선형 결합 계수 크기다. 인과적 영향이나 변수 중요도의 보편적인 척도는 아니다. v와 −v는 모두 같은 주성분 방향을 나타내므로 점수의 부호도 함께 바뀐다. 점수와 로딩을 함께 뒤집으면 재구성은 같다.

피처 j의 표준편차를 σ_j>0라 하면 원래 피처와 k번째 점수의 상관계수는 v_jk√λ_k/σ_j다. 표준화된 피처에서만 v_jk√λ_k가 바로 상관 로딩이 된다. 남긴 상관 로딩의 제곱합은 그 피처의 분산 중 설명되는 비율이다.

```python
def correlation_loadings(loadings, eigenvalues):
    """주성분과의 상관을 나타내도록 실림의 잣수를 맞춘다."""
    return loadings * np.sqrt(eigenvalues)[:, np.newaxis]
```

이 함수는 로딩을 **성분×피처** 배열로 받는다. 위 수학 표기의 W는 피처×성분이므로 전치가 필요하다. 원래 피처가 표준화되지 않았다면 피처별 표준편차로 나누는 단계도 추가로 필요하다.

## 4. 재구성 오차와 성분 수

중심화된 학습 데이터와 Σ=X_cᵀX_c/n에 대해 다음이 성립한다.

$$\mathcal E_k=\frac1n\|X_c-X_cWW^T\|_F^2=\sum_{j>k}\lambda_j.$$

이는 표본별 제곱 거리의 평균이다. 모든 피처 원소에 대한 MSE는 여기에 d를 더 나눈 값이다. 고윳값이 1/(n−1) 관례이면 표본별 평균 오차는 (n−1)/n을 곱해야 한다. 같은 식을 다른 테스트 데이터의 오차에 그대로 대입할 수는 없다.

전체 분산이 양수일 때 성분별 설명 분산 비율은 λ_j/Σ_iλ_i, 누적 비율은 Σ_{j≤k}λ_j/Σ_iλ_i다. 95%를 남긴다는 것은 이 데이터의 제곱 거리 기준 분산 95%를 보존한다는 뜻이며 모든 예측 정보 95%를 보존한다는 뜻은 아니다.

```python
def choose_n_components(eigenvalues, threshold=0.95):
    """바라는 설명 흩어짐을 얻는 가장 작은 k을 찾는다."""
    total = eigenvalues.sum()
    cumsum = np.cumsum(eigenvalues)
    k = np.searchsorted(cumsum / total, threshold) + 1
    return k
```

함수는 고윳값이 내림차순이고 합이 양수이며 0<threshold≤1인 입력을 전제로 한다. 유효 범위 밖의 임계값과 전체 분산 0에는 별도 검증이 필요하다.

스크리 플롯의 팔꿈치는 후보 성분 수를 살피는 휴리스틱이다. 팔꿈치 뒤가 반드시 잡음이라는 판정 기준은 아니다. 분류 등 후속 작업이 목적이면 PCA를 각 학습 폴드 안에서 적합하고 검증 성능으로 k를 고를 수 있다.

```python
def choose_by_error(eigenvalues, max_error):
    """받아들일 만한 다시 세우기 어긋남을 얻는 가장 작은 k을 찾는다."""
    cumsum_discarded = eigenvalues.sum() - np.cumsum(eigenvalues)
    k = np.searchsorted(-cumsum_discarded, -max_error) + 1
    return k
```

이 함수는 표본별 제곱 거리 평균을 오차 예산으로 해석한다. 구현상 최소 1개 성분을 반환하므로 전체 분산 이상의 예산에서 k=0을 허용하는 최적화와는 다르다. 음수 예산이나 잘못된 고윳값에는 유효하지 않은 결과를 줄 수 있다.

```python
def plot_scree(eigenvalues):
    """고윳값과 쌓아 올린 설명 흩어짐을 그린다."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(eigenvalues, 'o-')
    ax1.set_xlabel('Component')
    ax1.set_ylabel('Eigenvalue')
    ax1.set_title('Scree Plot')

    cumsum = np.cumsum(eigenvalues) / eigenvalues.sum()
    ax2.plot(cumsum, 'o-')
    ax2.axhline(0.95, color='r', linestyle='--', label='95%')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.legend()

    plt.tight_layout()
    plt.show()
```

## 5. 선형 오토인코더와의 관계

중심화된 데이터에서 활성화·바이어스 없는 k차원 선형 오토인코더의 제곱 재구성 로스를 **전역 최적화**하면 PCA와 같은 최소 오차를 달성할 수 있다. 하지만 학습된 인코더가 직교 고유벡터를 분산 순으로 반환하는 것은 아니다.

예를 들어 가역행렬 A에 대해 디코더 W A와 인코더 A⁻¹Wᵀ는 같은 재구성 W Wᵀ를 만든다. 따라서 웨이트가 자동으로 묶이거나 각 축이 주성분과 같아진다고 할 수 없다. 유한 횟수의 경사 기반 학습이 전역 최적점에 도달한다는 보장도 별개다. [선형 오토인코더 연구](https://proceedings.mlr.press/v97/kunin19a.html)에서도 주부분공간과 개별 주방향의 차이를 다룬다.

```python
import torch
import torch.nn as nn

class LinearAutoencoder(nn.Module):
    """주성분 분석과 같은 선형 오토인코더.

    평균 제곱 어긋남 로스를 쓰고 활성화도 치우침도 없으면 익히기가
    주성분 분석 풀이로 모인다.
    """
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim, bias=False)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=False)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
```

위 두 선형 계층은 독립적인 웨이트를 가지며 직교성이나 공유를 강제하지 않는다. PCA도 GPU 선형대수로 계산할 수 있으므로 GPU 지원 여부만으로 두 방법을 구분하지 않는다. 계산 비용은 표본·피처 수, 목표 차원과 분해 알고리즘에 따라 달라진다.

## 6. 비선형 구조에서의 한계

PCA는 하나의 평평한 부분공간을 찾는다. 스위스 롤처럼 굽은 표면을 평면에 투영하면 표면을 따라 멀리 있는 점들이 겹칠 수 있다. 이 한계는 PCA의 선형 재구성 목표와 매니폴드 거리 보존 목표가 다르기 때문이다.

```python
from sklearn.datasets import make_swiss_roll

X, color = make_swiss_roll(n_samples=1000, noise=0.1)
# 속 차원은 2이지만 주성분 분석은 그것을 "펴지" 못한다
```

비선형 활성화를 넣은 오토인코더나 커널 PCA는 더 복잡한 구조를 표현할 수 있지만, 좋은 매니폴드나 분류 경계를 반드시 찾는 것은 아니다. 모델 선택과 데이터, 최적화와 검증이 필요하다.

```python
# 선형(≈ 주성분 분석)
encoder = nn.Linear(784, 32)
decoder = nn.Linear(32, 784)

# 비선형(다양체를 배울 수 있다)
encoder = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 32)
)
decoder = nn.Sequential(
    nn.Linear(32, 256), nn.ReLU(),
    nn.Linear(256, 784)
)
```

## 7. 적용 예제 읽기

### 2차원에서 1차원으로 투영

아래 코드는 상관된 2차원 데이터에서 PCA를 적합하고 역변환한다. `transform`은 중심화를 포함하고 `inverse_transform`은 평균을 복원한다.

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

np.random.seed(0)

# 상관 있는 2차원 자료를 만든다
x = np.random.normal(size=(200,))
y = 0.5 * x + 2 + 0.1 * np.random.normal(size=(200,))
X = np.column_stack([x, y])

# 1차원으로 줄이고 다시 세운다
pca = PCA(n_components=1).fit(X)
X_pca = pca.transform(X)
X_reconstructed = pca.inverse_transform(X_pca)

print(f"Original shape:      {X.shape}")           # (200, 2)
print(f"Projected shape:     {X_pca.shape}")        # (200, 1)
print(f"Reconstructed shape: {X_reconstructed.shape}")  # (200, 2)

# 본디 자료와 쏜 자료를 그린다
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(X[:, 0], X[:, 1], alpha=0.3, label="Original")
ax.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1],
           color='red', s=20, label="Projected")
ax.legend()
plt.show()
```

```
Original shape:      (200, 2)
Projected shape:     (200, 1)
Reconstructed shape: (200, 2)
```

### MNIST 차원 축소

PCA를 학습 데이터에 적합하고 테스트 이미지를 변환한다. 출력의 성분 수는 해당 데이터와 전처리에 대한 예시다. 좌표 수의 감소율은 압축 파일의 배율과 같지 않다. 모델의 평균·기저 저장과 각 좌표의 dtype도 필요하다.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision import datasets

# 주성분 분석 자체는 넘파이로 하므로 텐서가 아니라 배열이 필요하다.
# transform 없이 불러오면 .data가 (N, 28, 28) uint8 텐서이고,
# numpy()로 옮긴 뒤 (N, 784)로 펴서 쓴다
train_set = datasets.MNIST(root='./data', train=True, download=True)
test_set = datasets.MNIST(root='./data', train=False, download=True)
X_train = train_set.data.numpy().reshape(-1, 784).astype(np.float32)
X_test = test_set.data.numpy().reshape(-1, 784).astype(np.float32)

# 흩어짐을 95% 남기는 주성분 분석
pca = PCA(n_components=0.95, svd_solver='full').fit(X_train)
X_reduced = pca.transform(X_test)
X_recovered = pca.inverse_transform(X_reduced)

print(f"Original dim:    {X_test.shape[1]}")       # 784
print(f"Reduced dim:     {X_reduced.shape[1]}")     # ~150
print(f"Compression:     {784 / X_reduced.shape[1]:.1f}x")
```

```
Original dim:    784
Reduced dim:     154
Compression:     5.1x
```

### 잡음 제거

큰 분산의 방향에 신호가 집중되고 잡음이 여러 방향으로 퍼져 있다면 일부 성분을 버리는 것이 유용할 수 있다. 분산이 큰 잡음이나 작은 분산의 유용한 신호에서는 반대 결과도 가능하다. 아래 코드는 잡음이 있는 같은 학습 데이터에 PCA를 적합하고 복원하며 깨끗한 정답과의 오차를 평가하지 않는다.

```python
# MNIST에 정규 잡음을 더한다
X_noisy = X_train + 10.0 * np.random.normal(size=X_train.shape)

# 잡음 낀 자료에 흩어짐을 90% 남겨 주성분 분석을 맞춘다
pca = PCA(n_components=0.9, svd_solver='full').fit(X_noisy)
X_filtered = pca.inverse_transform(pca.transform(X_noisy))

print(f"Components used for denoising: {pca.n_components_}")
```

```
Components used for denoising: 104
```

### 고유얼굴

얼굴 이미지의 주성분 방향을 이미지로 표시한 것이 고유얼굴이다. 각 성분이 조명·자세·표정 중 하나만 독립적으로 나타낸다는 보장은 없다. 학습 데이터와 전처리에 의존하며, 코드의 62×47 reshape도 실제 데이터의 이미지 크기와 맞아야 한다.

```python
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA

faces = fetch_lfw_people(min_faces_per_person=60)
print(f"Dataset: {faces.data.shape}")  # (사람 수, 62*47)

# 성분 150개로 주성분 분석을 맞춘다
pca = PCA(n_components=150, svd_solver='randomized').fit(faces.data)

# 성분마다 "고유 얼굴"이다
eigenface_0 = pca.components_[0].reshape(62, 47)

# 성분 150개로 얼굴을 다시 세운다
components = pca.transform(faces.data)
reconstructed = pca.inverse_transform(components)
```

### PyTorch 선형 오토인코더

다음 예제는 MNIST의 [0,1] 픽셀을 그대로 사용하며 피처별 학습 평균을 빼지 않는다. 바이어스 없는 선형 계층이므로 중심화 PCA의 아핀 재구성과 정확히 같은 문제가 아니다. 비교하려면 같은 학습 평균을 빼고 같은 차원의 PCA와 원소별 MSE를 비교해야 한다.

```python
import torch
from torch import nn, optim
from torchvision import datasets, transforms

torch.manual_seed(0)

class PCAAutoencoder(nn.Module):
    """선형 오토인코더(활성화 없음, 치우침 없음) ≡ 주성분 분석."""
    def __init__(self, input_dim=784, latent_dim=20):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim, bias=False)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=False)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.decoder(self.encoder(out))
        return out.view(x.size())

# 학습
transform = transforms.ToTensor()
train_data = datasets.MNIST('./data', train=True, transform=transform,
                             download=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64,
                                            shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PCAAutoencoder(784, 20).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(100):
    epoch_loss = 0.0
    for batch_X, _ in train_loader:
        batch_X = batch_X.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(batch_X), batch_X)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss={epoch_loss / len(train_loader):.4f}")
```

```
Epoch 0: loss=0.0312
Epoch 20: loss=0.0256
Epoch 40: loss=0.0256
Epoch 60: loss=0.0256
Epoch 80: loss=0.0256
```

보존된 출력은 학습 로스의 예시이며 중심화 PCA와의 동등성이나 테스트 일반화 성능을 검증한 결과가 아니다. 배치별 평균을 다시 평균한 기록이므로 마지막 배치의 크기가 다르면 전체 원소 가중 평균과도 약간 다를 수 있다.

## 8. 결과 시각화와 계산 관례

바이플롯은 표본 점수와 피처 로딩을 함께 표시한다. 아래 함수는 보기 좋게 화살표 크기를 조정하는 시각화다. 화살표 길이를 표본 거리와 직접 비교하거나 모든 화살표 각도를 상관계수로 해석하지 않는다.

```python
import matplotlib.pyplot as plt

def biplot(scores, loadings, feature_names, labels=None):
    """주성분 겹그림: 점수는 흩뿌림으로, 실림은 화살표로."""
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(scores[:, 0], scores[:, 1],
                         c=labels, alpha=0.6, s=50)

    scale = np.abs(scores).max() * 0.8
    for load, name in zip(loadings.T, feature_names):
        ax.arrow(0, 0, load[0] * scale, load[1] * scale,
                 head_width=0.05 * scale, fc='red', ec='red', alpha=0.7)
        ax.text(load[0] * scale * 1.1, load[1] * scale * 1.1,
                name, fontsize=10, ha='center')

    ax.axhline(0, color='gray', ls='--', alpha=0.3)
    ax.axvline(0, color='gray', ls='--', alpha=0.3)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PCA Biplot')
    plt.tight_layout()
    plt.show()
```

```python
def interpret_loadings(loadings, feature_names, n_top=5):
    """성분마다 가장 많이 보태는 피처를 보인다."""
    for k in range(loadings.shape[0]):
        abs_load = np.abs(loadings[k])
        top_idx = np.argsort(abs_load)[::-1][:n_top]
        print(f"\n=== PC{k+1} ===")
        for idx in top_idx:
            print(f"  {feature_names[idx]}: {loadings[k, idx]:.4f}")
```

이 두 함수는 로딩 배열을 성분×피처로 받는다. 이는 scikit-learn의 `components_` 배치와 같고 수학 표기의 W와는 전치 관계다.

다음 PyTorch 구현은 중심화 후 1/(n−1) 공분산의 고유분해를 수행한다. n≥2와 유효한 목표 차원이 필요하며 고윳값은 오름차순으로 반환되는 `eigh` 결과를 역순으로 선택한다.

```python
import torch
import numpy as np

def pca(X, n_components):
    """
    고윳값 쪼개기로 하는 주성분 분석.

    인수:
        X: 자료 행렬 [표본 수, 피처 수]
        n_components: 주성분의 수

    반환값:
        W: 주성분(실림) [피처 수, 성분 수]
        Z: 쏜 자료(점수) [표본 수, 성분 수]
        eigenvalues: 성분마다 설명하는 흩어짐
    """
    # 자료의 가운데를 맞춘다
    X_centered = X - X.mean(dim=0)

    # 공분산 행렬
    cov = X_centered.T @ X_centered / (X.shape[0] - 1)

    # 고윳값 쪼개기(대칭 행렬에는 eigh)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    # 고윳값 내림차순으로 정렬한다
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx[:n_components]]
    W = eigenvectors[:, idx[:n_components]]

    # 쏜다(점수를 셈한다)
    Z = X_centered @ W

    return W, Z, eigenvalues
```

함수는 학습 평균을 반환하지 않으므로 새 데이터 변환과 원래 좌표 재구성에 필요한 평균을 따로 보관해야 한다. 반환 고윳값도 선택된 성분만 포함하므로 이 값의 합만으로 전체 분산 비율을 계산할 수 없다. SVD를 직접 사용하면 공분산을 만들 때 조건수가 제곱되는 수치 문제를 줄일 수 있다. 라이브러리의 분모와 속성 정의는 [scikit-learn PCA 문서](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)를 참고한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
첫 주성분을 분산 최대화 문제에서 라그랑주 승수로 유도하라.

</div>

??? success "풀이"

    중심화 공분산을 Σ라 하면 max wᵀΣw, wᵀw=1이다. 라그랑주 함수 wᵀΣw−λ(wᵀw−1)의 정지 조건은 Σw=λw다. 이때 목적값은 λ이므로 최대 고윳값의 단위 고유벡터를 고른다. 중복 최대 고윳값에서는 방향이 유일하지 않다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
SVD로 주성분 방향과 점수, 공분산 고윳값을 구하는 관계를 적어라.

</div>

??? success "풀이"

    X_c=USVᵀ이면 X_cᵀX_c/n=V(S²/n)Vᵀ다. V의 앞 k개 열이 방향이고 점수는 U_k S_k다. 1/(n−1) 관례에서는 고윳값이 s_j²/(n−1)로 바뀐다. 얇은 SVD는 피처 수가 큰 경우 공분산 전체를 만들지 않아도 된다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
커널 PCA가 도움이 될 수 있는 구조와 선형 PCA에 비해 추가로 검증해야 할 사항을 설명하라.

</div>

??? success "풀이"

    굽은 매니폴드나 동심원처럼 선형 투영으로 구조를 보존하기 어려운 데이터에서 비선형 커널이 유용할 수 있다. 그러나 커널 종류와 폭, 목표 차원에 따라 결과가 달라지며 분류가 자동으로 완벽해지는 것은 아니다. 후속 태스크 검증과 커널 행렬의 계산·메모리 비용을 고려한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
고윳값이 6, 3, 1일 때 분산 90%를 남기는 최소 성분 수와 표본별 재구성 오차를 구하라. Σ의 분모는 n이다.

</div>

??? success "풀이"

    전체 분산은 10이고 두 성분이 9를 설명하므로 k=2다. 버린 고윳값 합인 1이 표본별 제곱 거리 평균이며 원소별 MSE는 1/3이다. 이 비율은 레이블 정보의 보존율과 다르다.



---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
어떤 MNIST 전처리에서 분산 90%에 성분 87개가 필요했다고 가정하자. 원래 784차원에 대한 비율을 구하고 이 숫자의 적용 범위를 설명하라.

</div>

??? success "풀이"

    87/784≈11.1%이며 좌표 수는 약 9.01배 줄어든다. 이는 가정한 데이터·전처리의 결과다. 표준화, 데이터 분할이나 입력 스케일이 달라지면 필요한 수가 달라질 수 있다. 실제 저장 비율에는 기저·평균과 좌표 dtype도 포함해야 한다.



---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
성분 수를 고르는 분산 임계값, 스크리 플롯과 후속 태스크 검증을 비교하라.

</div>

??? success "풀이"

    분산 임계값은 명확한 수치 규칙이지만 예측에 중요한 작은 분산 방향을 버릴 수 있다. 스크리 플롯의 팔꿈치는 주관적이고 없을 수도 있다. 후속 태스크 검증은 목적에 맞지만 비용이 들며 각 폴드 안에서 PCA를 적합해야 한다. 시각화의 2·3차원 선택에서는 누락된 분산도 함께 표시한다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
학습 데이터 고윳값으로 테스트 재구성 MSE를 정확히 알 수 있는가?

</div>

??? success "풀이"

    일반적으로 알 수 없다. 버린 고윳값 합과 재구성 오차의 등식은 그 공분산을 구한 데이터와 일관된 평균·분모에서 성립한다. 테스트에는 다른 공분산과 학습 평균에 대한 평균 이동이 있을 수 있다. 테스트 오차가 더 작거나 크다는 사실만으로 과적합 여부를 단정할 수도 없다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
PCA 전에 중심화와 표준화를 각각 언제 적용하는지 설명하라.

</div>

??? success "풀이"

    평균 주위 변동의 PCA를 구하려면 중심화한다. 표준화는 피처의 상대적 변동을 비교할지 원래 단위의 분산을 보존할지에 따라 선택한다. 단위가 다르면 검토할 필요가 크지만 필수 규칙은 아니며, 같은 단위라고 항상 생략하는 것도 아니다. 통계는 학습 데이터에서만 추정한다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
중심화하지 않은 2차 모멘트와 공분산의 관계를 구하라.

</div>

??? success "풀이"

    학습 평균 μ에 대해 XᵀX/n=Σ+μμᵀ다. 평균 항은 표본 간 변동과 별개로 방향 선택에 영향을 준다. 이 행렬의 고윳값 비율은 평균 주위의 설명 분산 비율과 다르다. 첫 방향이 반드시 μ와 일치하거나 비율 차이가 항상 같은 크기라는 결론은 성립하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
선형 오토인코더와 PCA의 관계에서 부분공간, 기저와 최적화 조건을 구분하라.

</div>

??? success "풀이"

    중심화된 데이터의 제곱 재구성 로스에서 충분한 조건의 전역 최적해는 PCA의 최소 오차를 달성한다. 인코더 좌표는 가역변환으로 바뀔 수 있어 직교하거나 분산 순일 필요가 없고 웨이트 공유도 자동으로 성립하지 않는다. 경사 학습의 유한 실행이 그 최적해에 도달하는지는 별도로 확인해야 한다.



## 정리하며

PCA는 중심화된 데이터의 분산 최대화와 선형 제곱 재구성 오차 최소화를 연결한다. 분모 관례와 평균 복원, 로딩 배열의 방향을 일관되게 사용해야 한다. 표준화와 성분 수는 목적에 따라 정하며, 설명 분산·테스트 성능·저장 압축을 서로 다른 지표로 해석한다.
