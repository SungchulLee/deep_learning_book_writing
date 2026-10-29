# 주성분 분석

주성분 분석(PCA)은 평균을 뺀 데이터에서 분산이 큰 직교 방향을 찾아 저차원 좌표로 나타내는 선형 차원 축소 방법이다. 같은 데이터의 유클리드 제곱 재구성 오차를 최소화하는 부분공간을 찾는 문제와 **같다** — 이 쪽의 정리 2가 그 등식을 보인다. 다만 이 최적성은 분류 성능이나 모든 종류의 정보 보존을 뜻하지 않는다.

이 쪽은 세 결과를 정리로 세우고 증명을 접어 둔다. 유도를 건너뛰고 읽어도 쓰는 데 모자람이 없도록, 그러나 "왜 하필 고유벡터인가"가 궁금해지면 바로 펼쳐 볼 수 있도록 한 배치다.

---

## 1. 중심화와 주성분

행에 표본을 둔 데이터 X∈ℝⁿˣᵈ에서 학습 평균 μ를 빼 X_c를 만든다. 이 절의 이론은 공분산을 Σ=X_cᵀX_c/n으로 정의하고 고윳값을 λ₁≥⋯≥λ_d≥0으로 정렬한다. 1/(n−1)을 쓰면 방향과 설명 분산 비율은 같지만 고윳값 크기가 달라진다.

단위 벡터 v 방향 점수의 분산은 vᵀΣv다. 이를 최대화하면 가장 큰 고윳값의 고유벡터를 얻는다. 앞선 방향과 직교하도록 반복하면 주성분 기저를 구성한다. 중복 고윳값에서는 해당 고유공간의 기저가 유일하지 않다.

$$W=[v_1,\ldots,v_k],\quad W^TW=I_k,\qquad Z=X_cW,\qquad
\hat{X}_c=ZW^T.$$

원래 단위로 재구성하려면 μ를 다시 더한다. 중심화하지 않은 XᵀX는 공분산이 아니라 평균 항 μμᵀ까지 포함한 2차 모멘트다. 그 첫 방향이 항상 평균과 정확히 일치하는 것은 아니다.

X∈ℝⁿˣᵈ의 각 행은 학습 평균을 뺀 표본이며 Σ=XᵀX/n이다. Σ는 실수 대칭 양의 준정부호 행렬이므로 다음과 같이 분해된다.

$$\Sigma=V\Lambda V^T,\qquad V^TV=I,\qquad
\lambda_1\ge\cdots\ge\lambda_d\ge0.$$

여기서는 n으로 나눈 경험 공분산을 사용한다. n>1에서 1/(n−1)을 사용하면 고윳값이 n/(n−1)배 되지만 주성분 방향과 설명 분산 비율은 같다. 이후 표본별 평균 오차와의 등식은 n 관례를 기준으로 한다.

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

---

## 4. 왜 고유벡터인가

<div class="thmbox" markdown>

### 정리 1. 분산을 가장 크게 하는 방향 { .thm }

중심화된 데이터의 공분산을 $\Sigma$, 고윳값을 $\lambda_1\ge\cdots\ge\lambda_d\ge0$이라 하자.

1. 단위 벡터 $v$ 방향 점수의 분산 $v^T\Sigma v$를 최대로 만드는 것은 $\lambda_1$의 고유공간에 있는 단위 벡터이고, 그때의 분산은 $\lambda_1$이다.
2. 앞의 방향들과 직교하도록 차례로 고르면 직교 고유벡터 기저가 나온다.
3. $k$개를 한꺼번에 고르는 문제 $\max_{W^TW=I_k}\operatorname{tr}(W^T\Sigma W)$의 최댓값은 $\sum_{j\le k}\lambda_j$다.

</div>

??? proof "증명"

    단위 벡터 v에 대한 점수는 z_i=x_iᵀv다. 평균이 0이므로 경험 분산은 다음과 같다.

    $$\frac1n\sum_i z_i^2=v^T\Sigma v.$$

    제약 vᵀv=1이 없으면 v를 키워 목적을 무한히 키울 수 있다. 라그랑주 함수는 다음과 같다.

    $$\mathcal L(v,\lambda)=v^T\Sigma v-\lambda(v^Tv-1).$$

    정지 조건 2Σv−2λv=0으로부터 Σv=λv를 얻는다. 이때 목적값은 vᵀΣv=λ다. 정지 조건만으로 최대·최소를 구분한 것은 아니며, 목적을 최대화하려면 가장 큰 고윳값의 고유공간에서 단위 벡터를 선택한다. 단위 구가 콤팩트하고 목적이 연속이므로 최댓값은 존재한다.

    두 번째 성분에는 vᵀv₁=0을 추가한다. 정지 조건은 2Σv−2λv−μv₁=0이다. 왼쪽에 v₁ᵀ를 곱하면 대칭성과 직교성으로 μ=0을 얻는다. 따라서 남은 직교 공간에서 가장 큰 고윳값인 λ₂의 방향을 선택한다. 같은 절차로 직교 고유벡터를 구성한다.

    k개 방향을 한꺼번에 구하는 문제는 다음과 같다.

    $$\max_{W^TW=I_k}\operatorname{tr}(W^T\Sigma W).$$

    상위 k개 고유벡터를 열로 둔 W가 최댓값 Σ_{j≤k}λ_j를 달성한다. 하지만 이 **대각합 목적만으로** W의 열을 개별 주성분 순서로 고정하지는 않는다. 어떤 직교 Q에 대해서도 WQ는 같은 부분공간과 목적값을 가지며, 일반적인 Q에서는 점수 공분산이 대각행렬이 아닐 수 있다. PCA의 표준 기저는 공분산을 대각화하고 분산 순으로 정렬한 기저다.

3번의 단서를 놓치기 쉽다. **대각합을 최대로 만든다는 것만으로는 $W$의 열이 개별 주성분이 되지 않는다.** 분산 순으로 정렬된 고유벡터를 얻으려면 그 부분공간 안에서 공분산을 대각화하는 기저를 따로 골라야 한다. 8절이 이 구분을 다시 다룬다.

---

## 5. 분산 최대화와 재구성 오차 최소화는 같은 문제다

<div class="thmbox" markdown>

### 정리 2. 두 목적의 동등성 { .thm }

$W^TW=I_k$인 $W$에 대해

$$\frac1n\|X_c-X_cWW^T\|_F^2=\operatorname{tr}(\Sigma)-\operatorname{tr}(W^T\Sigma W)$$

이다. 첫 항이 $W$와 무관하므로 **재구성 오차를 최소로 만드는 $W$와 분산을 최대로 만드는 $W$가 같다.** 그 최솟값은 버린 고윳값의 합이다.

$$\mathcal E_k=\sum_{j>k}\lambda_j$$

</div>

??? proof "증명"

    WᵀW=I일 때 P=WWᵀ는 대칭이며 P²=P인 직교 투영 행렬이다. 피타고라스 관계로 다음이 성립한다.

    $$\|x-Px\|^2=\|x\|^2-x^TPx.$$

    표본에 대해 평균을 취하면 다음과 같다.

    $$\frac1n\|X-XWW^T\|_F^2=\operatorname{tr}(\Sigma)-\operatorname{tr}(W^T\Sigma W).$$

    첫 항은 W와 무관하므로 재구성 오차 최소화가 분산 최대화와 같다. PCA 기저에서 최소 오차는 다음과 같다.

    $$\mathcal E_k=\sum_{j>k}\lambda_j.$$

    왼쪽은 표본당 제곱 거리 평균이다. `mean()`으로 모든 n×d 원소의 제곱 오차를 평균하면 값은 E_k/d다. 1/(n−1) 고윳값을 쓴 경우에는 오차 환산에 (n−1)/n도 필요하다.

    에카르트–영–미르스키 정리는 X의 절단 SVD가 랭크 **k 이하** 행렬 중 프로베니우스 오차를 최소화함을 보인다. 이 대수적 최적성에는 가우시안 잡음 가정이 필요하지 않다. 가우시안 오차 모델은 제곱 오차를 확률적으로 해석하는 별도의 관점이다.

이는 표본별 제곱 거리의 평균이다. 모든 피처 원소에 대한 MSE는 여기에 d를 더 나눈 값이다. 고윳값이 1/(n−1) 관례이면 표본별 평균 오차는 (n−1)/n을 곱해야 한다. 같은 식을 다른 테스트 데이터의 오차에 그대로 대입할 수는 없다.

---

## 6. 성분 수 고르기

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

전체 분산이 양수일 때 누적 설명 분산 비율은 다음과 같다.

$$\operatorname{CEVR}_k=\frac{\sum_{j\le k}\lambda_j}{\operatorname{tr}(\Sigma)}
=1-\frac{\mathcal E_k}{\operatorname{tr}(\Sigma)}.$$

```python
def explained_variance_analysis(eigenvalues):
    """설명하는 흩어짐 비와 다시 세우기 어긋남을 셈한다."""
    total = eigenvalues.sum()
    evr = eigenvalues / total
    cumulative_evr = np.cumsum(evr)
    reconstruction_error = total - np.cumsum(eigenvalues)

    return evr, cumulative_evr, reconstruction_error
```

이 함수는 전체 스펙트럼을 내림차순으로 받는다는 전제다. 전체 분산이 0이면 비율이 정의되지 않는다. PCA 고유벡터 기저의 점수 공분산은 WᵀΣW=diag(λ₁,…,λ_k)다. 상관이 없다는 성질은 해당 데이터의 경험 공분산에 대한 것이며 좌표의 통계적 독립성이나 테스트 데이터에서의 무상관을 보장하지 않는다.

---

## 7. 선형 오토인코더와의 관계

<div class="thmbox" markdown>

### 정리 3. 선형 오토인코더의 전역 최적 { .thm }

활성화도 바이어스도 없는 $k$차원 선형 오토인코더의 제곱 재구성 로스를 **전역 최적화** 하면 PCA와 같은 최소 오차에 닿는다. 그러나 학습된 인코더가 직교 고유벡터를 분산 순으로 돌려주지는 **않는다.**

</div>

??? proof "증명"

    열벡터 표기에서 인코더 W_eᵀ와 디코더 W_d의 재구성은 W_dW_eᵀx다. 중심화된 데이터와 바이어스·활성화 없는 모델의 로스를 다음과 같이 둔다.

    $$L(W_e,W_d)=\frac1n\sum_i\|x_i-W_dW_e^Tx_i\|^2.$$

    W_e를 고정하고 W_d로 미분하면 정상 방정식을 얻는다.

    $$\Sigma W_e=W_d(W_e^T\Sigma W_e).$$

    괄호 안 행렬이 가역이면 다음과 같다.

    $$W_d=\Sigma W_e(W_e^T\Sigma W_e)^{-1}.$$

    특이한 경우에는 역행렬을 그대로 사용할 수 없으며 의사역행렬과 비유일성을 고려해야 한다. 양의 정부호 Σ에서 1≤k<d이고 λ_k>λ_{k+1}라면 최적 재구성은 상위 k차원 부분공간의 투영이다. 예를 들어 가역 A에 대해 W_e=W A⁻ᵀ, W_d=W A는 같은 최적 재구성을 준다. 따라서 두 웨이트가 같거나 인코더가 정규직교인 것은 필수 조건이 아니다.

    공분산이 특이하면 학습 데이터가 놓이지 않는 영공간에서의 동작은 로스로 정해지지 않는다. 중복 고윳값이 경계를 가로지르면 최적 부분공간 자체도 여러 개일 수 있다. [선형 오토인코더 연구](https://proceedings.mlr.press/v97/kunin19a.html)의 주부분공간 결과를 임의의 유한 학습 실행에 대한 수렴 보장으로 확대하지 않는다.

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

```python
import torch
import torch.nn as nn

class LinearAutoencoder(nn.Module):
    """그래디언트 내려가기로 주성분 분석 풀이를 배운다."""
    def __init__(self, d, k):
        super().__init__()
        self.encoder = nn.Linear(d, k, bias=False)
        self.decoder = nn.Linear(k, d, bias=False)

    def forward(self, x):
        return self.decoder(self.encoder(x))

def verify_equivalence(X, k, epochs=5000, lr=0.01):
    """주성분 분석과 선형 오토인코더의 다시 세우기 어긋남을 견준다."""
    X_centered = X - X.mean(axis=0)

    # 닫힌 꼴 주성분 분석
    Sigma = X_centered.T @ X_centered / X.shape[0]
    eigvals = np.linalg.eigh(Sigma)[0]
    eigvals = np.sort(eigvals)[::-1]
    pca_error = eigvals[k:].sum()

    # 선형 오토인코더
    X_t = torch.tensor(X_centered, dtype=torch.float32)
    model = LinearAutoencoder(X.shape[1], k)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        opt.zero_grad()
        loss = ((X_t - model(X_t)) ** 2).mean()
        loss.backward()
        opt.step()

    ae_error = loss.item() * X.shape[1]  # 평균 제곱 어긋남 -> 전체 어긋남

    print(f"PCA error:       {pca_error:.6f}")
    print(f"Linear AE error: {ae_error:.6f}")
```

`verify_equivalence`는 같은 중심화 데이터에서 PCA의 표본별 오차와 학습 로스를 비교하려 한다. 원소별 평균 로스에 d를 곱해 단위를 맞추지만, 기록한 `loss`는 마지막 옵티마이저 갱신 **전** 값이다. 유한 에포크의 근접한 값만으로 전역 수렴이나 기저의 일치를 증명하지는 못한다.

---

## 8. 부호와 유일성

각 고유벡터의 부호는 자유다. 점수와 방향을 함께 뒤집으면 재구성과 거리는 유지된다. 부호를 고정하는 표시 관례는 해석을 돕지만 학습된 방향의 의미를 추가로 보장하지 않는다.

λ_k>λ_{k+1}이면 상위 k차원 부분공간은 유일하다. 개별 방향까지 부호 외에 유일하려면 해당 고윳값이 단순해야 한다. 선택된 공간 내부의 중복은 기저만 비유일하게 만들고, 선택 경계의 중복은 어떤 k차원 부분공간을 고를지도 비유일하게 만든다. 거의 같은 고윳값은 작은 데이터 변화에서 개별 축을 불안정하게 만들 수 있다.

---

## 9. 계산: 고유분해와 SVD

고윳값을 내림차순으로 정렬하고 상위 k개 고유벡터 W를 선택하면 점수 Z=XW와 중심화 재구성 X̂=ZWᵀ를 얻는다. 원래 데이터에서 시작했다면 학습 평균을 빼고, 복원 뒤에는 다시 더한다. 새 데이터에도 학습 평균을 사용한다.

```python
import numpy as np

def pca(X, k):
    """주성분 분석: 흩어짐 가장 크게 하기 / 다시 세우기 어긋남 가장 작게 하기.

    인수:
        X: 가운데 맞춘 자료 [n, d]
        k: 성분의 수

    반환값:
        W: 실림 [d, k]
        Z: 점수 [n, k]
        eigenvalues: 성분마다의 흩어짐 [k]
    """
    n = X.shape[0]
    Sigma = X.T @ X / n

    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    idx = np.argsort(eigenvalues)[::-1]

    W = eigenvectors[:, idx[:k]]
    eigenvalues = eigenvalues[idx[:k]]
    Z = X @ W

    return W, Z, eigenvalues
```

이 함수는 중심화와 평균 저장을 하지 않으므로 입력 X가 이미 중심화되어 있어야 한다. 반환 고윳값도 상위 k개뿐이어서 전체 분산 비율이나 버린 고윳값 합을 계산하기에는 부족하다. 1≤k≤d와 유효한 표본 수를 확인해야 한다.

SVD X=USVᵀ를 직접 구하면 방향은 V의 열, 점수는 U_kS_k, 고윳값은 s_j²/n이다. 공분산을 만들면 0이 아닌 특잇값 기준 조건수가 제곱되므로 작은 성분의 수치 정확도에 불리할 수 있다. 반면 n이 매우 크고 d가 작으면 공분산 고유분해가 실용적일 수 있다. 크기와 목표 차원에 따라 얇은 SVD, 절단·무작위 분해 등을 선택한다.

---

## 10. 비선형 구조에서의 한계

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

---

## 11. 적용 예제 읽기

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

---

## 12. 결과 시각화와 계산 관례

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

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
고윳값이 $6, 3, 1$일 때 분산의 90%를 남기는 최소 성분 수와 그때의 표본별 재구성 오차를 구하라. $\Sigma$의 분모는 $n$이다.

</div>

??? success "풀이"
    전체 분산은 $6+3+1=10$이다. 누적 비율은 $0.6,\ 0.9,\ 1.0$이므로 90%를 남기는 최소 성분 수는 $k=2$다.

    정리 2에 따라 표본별 재구성 오차는 버린 고윳값의 합이므로 $\mathcal E_2=\lambda_3=1$이다.

    피처 원소별 MSE를 원한다면 $d=3$으로 한 번 더 나누어 $1/3$이다. 두 수는 단위가 다르므로 어느 쪽을 적는지 밝혀야 한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
주성분의 부호를 뒤집으면 무엇이 바뀌고 무엇은 그대로인가?

</div>

??? success "풀이"
    $v$와 $-v$는 같은 방향을 나타낸다. 점수의 부호도 함께 뒤집히므로 재구성 $zv^T$는 그대로이고, 재구성 오차와 설명 분산도 바뀌지 않는다.

    바뀌는 것은 **그림과 해석뿐** 이다. 산점도가 좌우로 뒤집히고, 로딩의 부호를 "이 피처가 늘면 PC1이 는다"처럼 읽던 문장이 반대가 된다.

    그래서 부호를 고정하는 표시 관례를 쓰기도 하지만, 그것은 보기를 맞추는 일이지 방향에 의미를 더해 주지는 않는다. 8절에서 본 대로 중복 고윳값이 있으면 부호 규칙만으로 해결되지 않는 회전 자유가 남는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
PCA 전에 중심화와 표준화를 각각 언제 적용하는지 설명하라.

</div>

??? success "풀이"
    **중심화는 거의 언제나 한다.** 중심화하지 않으면 공분산이 아니라 2차 모멘트의 고유분해가 되고(연습문제 7), 첫 방향이 평균 쪽으로 끌려간다.

    **표준화는 목적에 따른 선택이다.** 피처의 단위가 서로 다르면(미터와 킬로그램) 숫자가 큰 피처가 분산을 지배하므로 표준화가 유용하다. 같은 단위라면 변동성의 차이 자체가 정보일 수 있어 표준화가 오히려 그것을 지운다.

    어느 쪽이든 **평균과 스케일은 학습 데이터에서 구해 검증·테스트에 그대로 적용한다.** 테스트 데이터로 다시 추정하면 누출이다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
SVD로 주성분 방향과 점수, 공분산 고윳값을 구하는 관계를 적고, 공분산 고유분해와 견주어 수치·계산상 어느 쪽이 유리한지 설명하라.

</div>

??? success "풀이"
    중심화된 $X=USV^T$이면

    $$\Sigma=\frac{X^TX}{n}=V\frac{S^2}{n}V^T$$

    이므로 $V$의 열이 주방향, 점수는 $U_kS_k$, 고윳값은 $s_j^2/n$이다. $1/(n-1)$ 관례는 고윳값만 $s_j^2/(n-1)$로 바꾼다.

    **수치적으로는 직접 SVD가 유리하다.** $X^TX$를 만들면 0이 아닌 특잇값에 대한 조건수가 **제곱되므로** 작은 성분의 정확도가 나빠진다. 공분산을 명시적으로 만들지 않아도 된다는 점도 이점이다.

    **계산량은 모양에 달려 있다.** $n$이 매우 크고 $d$가 작으면 $d\times d$ 공분산의 고유분해가 더 싸다. 반대로 $d$가 크면 얇은 SVD나 절단·무작위 분해가 낫다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
성분 수를 고르는 세 가지 — 누적 분산 임계값, 스크리 플롯의 팔꿈치, 후속 태스크 검증 — 를 견주어라. 각각 무엇을 보존하는가?

</div>

??? success "풀이"
    **누적 분산 임계값**은 $\sum_{j\le k}\lambda_j/\operatorname{tr}(\Sigma)$가 문턱을 넘는 최소 $k$를 고른다. 보존하는 것은 **그 학습 데이터의 제곱 거리 기준 분산** 이다. "95%를 남겼다"가 "예측 정보의 95%를 남겼다"는 뜻은 아니다.

    **스크리 플롯의 팔꿈치**는 후보를 눈으로 살피는 휴리스틱이다. 팔꿈치 뒤가 잡음이라는 판정 기준이 아니고, 고윳값이 완만하게 줄면 팔꿈치 자체가 없다.

    **후속 태스크 검증**은 분류 정확도 같은 실제 목표로 $k$를 고른다. 목적에 가장 가깝지만 비싸고, **각 학습 폴드 안에서 PCA를 적합해야** 한다. 전체 데이터로 PCA를 적합한 뒤 교차검증하면 누출이다.

    셋은 서로 다른 질문에 답하므로 같은 $k$를 줄 이유가 없다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
학습 데이터의 고윳값으로 테스트 데이터의 재구성 MSE를 정확히 알 수 있는가?

</div>

??? success "풀이"
    없다. 정리 2의 $\mathcal E_k=\sum_{j>k}\lambda_j$는 **그 고윳값을 계산한 바로 그 데이터** 에 대한 등식이다.

    테스트 데이터에서는 학습 평균과 학습 기저를 쓰므로 투영이 그 데이터에 최적이 아니고, 오차는 대개 더 크다. 분포가 달라졌다면 훨씬 클 수도 있다.

    테스트 오차를 알고 싶으면 **재어야 한다** — 학습 $\mu$와 $W$로 변환·복원한 뒤 직접 계산한다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
중심화하지 않은 2차 모멘트 $X^TX/n$과 공분산 $\Sigma$의 관계를 구하고, 중심화를 건너뛰면 무엇이 달라지는지 설명하라.

</div>

??? success "풀이"
    $\mu$를 평균이라 하면

    $$\frac{X^TX}{n}=\Sigma+\mu\mu^T$$

    이다. 2차 모멘트는 공분산에 **평균의 외적을 더한 것** 이다.

    $\mu\mu^T$는 랭크 1이고 $\mu$ 방향으로만 분산을 보태므로, 평균이 원점에서 멀면 첫 방향이 $\mu$ 쪽으로 끌려간다. 그 성분은 데이터의 퍼짐이 아니라 **데이터가 어디에 놓여 있는지** 를 말한다.

    다만 첫 방향이 $\mu$와 **정확히** 같아지는 것은 아니다. $\Sigma$와 $\mu\mu^T$의 합의 고유벡터이므로 둘의 상대적 크기에 따라 기울어진다.

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
커널 PCA가 선형 PCA보다 좋은 결과를 줄 때, 그것이 정리 2의 최적성에 대한 반례가 아닌 까닭을 설명하라.

</div>

??? success "풀이"
    정리 2의 최적성에는 조건이 둘 붙어 있다. **원래 좌표에서의** 제곱 재구성 오차이고, **고정된 랭크** $k$의 선형 부분공간 가운데서의 최소다.

    커널 PCA는 비선형 피처 공간 $\phi(x)$로 옮겨 거기서 PCA를 한다. 공분산도 기하도 다른 문제이므로, 시각화나 분류가 좋아져도 원래 좌표의 선형 최적성과 겨루는 것이 아니다.

    덧붙여 커널 PCA에는 원래 공간으로 되돌리는 일이 간단하지 않다는 대가가 있다(프리이미지 문제). 선형 PCA는 $W$를 곱하면 그만이다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
정리 1의 증명은 정지 조건 $\Sigma v=\lambda v$까지만 보였다. 정지 조건을 만족하는 단위 벡터는 모든 고유벡터인데, 그중 어느 것이 최대인지는 어떻게 아는가? 최댓값이 존재한다는 것은 무엇으로 보장되는가?

</div>

??? success "풀이"
    정지 조건은 **후보** 만 걸러 낸다. 각 고유벡터 $v_j$에서 목적값이 $v_j^T\Sigma v_j=\lambda_j$이므로, 후보들의 목적값이 곧 고윳값이다. 따라서 가장 큰 것은 $\lambda_1$의 고유공간에 있는 단위 벡터다. 최소를 주는 것은 $\lambda_d$ 쪽이고, 둘 다 같은 정지 조건을 만족한다.

    **존재는 콤팩트성에서 나온다.** 제약 집합 $\{v:\|v\|=1\}$은 $\mathbb R^d$의 닫힌 유계 집합이라 콤팩트하고 목적 $v^T\Sigma v$는 연속이므로, 최대·최소가 모두 달성된다(최대·최소 정리).

    이 두 단계를 건너뛰고 "라그랑주를 풀면 첫 주성분이 나온다"로 적으면, 왜 최소가 아니라 최대인지와 애초에 최댓값이 있는지를 둘 다 빠뜨리게 된다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
$k$개 방향의 대각합을 한꺼번에 최대화하면 각 열이 반드시 분산 순으로 정렬된 고유벡터인가? 아니라면 개별 주성분은 어떻게 얻는가?

</div>

??? success "풀이"
    그렇지 않다. 최적 $W$와 임의의 직교 $Q$($Q^TQ=I_k$)에 대해

    $$\operatorname{tr}((WQ)^T\Sigma(WQ))=\operatorname{tr}(Q^TW^T\Sigma WQ)=\operatorname{tr}(W^T\Sigma W)$$

    이므로 $WQ$도 같은 값을 준다. 대각합의 순환성을 쓴 것이다. $WQ$의 열이 만드는 **부분공간도 같다.**

    곧 대각합 목적이 정하는 것은 **부분공간이지 그 안의 기저가 아니다.** 일반적인 $Q$에서는 점수의 공분산 $Q^T\operatorname{diag}(\lambda)Q$가 대각행렬이 아니므로 좌표끼리 상관이 생긴다.

    개별 주성분을 얻으려면 그 부분공간 안에서 **공분산을 대각화하고 고윳값 순으로 정렬** 해야 한다. 부분공간 최적성과 기저 선택은 다른 문제다 — 7절의 선형 오토인코더가 부분공간은 맞히면서 기저는 맞히지 못하는 것이 같은 이야기다.

---

<div class="drillbox" markdown>

**연습문제 11.** <span class="diff hard" title="어려움"></span>
주부분공간의 유일성과 개별 주성분의 유일성은 조건이 다르다. 각각을 적고, 거의 같은 고윳값이 실무에서 무엇을 일으키는지 설명하라.

</div>

??? success "풀이"
    **부분공간**: $\lambda_k>\lambda_{k+1}$이면 상위 $k$차원 부분공간이 유일하다. 경계에서 $\lambda_k=\lambda_{k+1}$이면 그 고유공간을 가르는 방법이 여럿이므로 최적 부분공간 자체가 여러 개다.

    **개별 방향**: 그 안에서 축 하나가 부호를 빼고 정해지려면 해당 고윳값이 **단순** 해야 한다. 공간 안에 중복 고윳값이 있으면 그 고유공간 안의 어떤 정규직교 기저든 똑같이 최적이다.

    두 조건은 독립이다. 부분공간은 유일한데 그 안의 축은 유일하지 않을 수 있다.

    **거의 같은 고윳값은 불안정을 낳는다.** $\lambda_2\approx\lambda_3$이면 수학적으로는 유일하더라도, 데이터가 조금만 흔들려도 두 축이 서로 자리를 바꾸거나 그 평면 안에서 회전한다. 표본을 나눠 두 번 적합해 축이 같은지 보는 것이 실무적인 점검이다.

---

<div class="drillbox" markdown>

**연습문제 12.** <span class="diff hard" title="어려움"></span>
PCA의 최적성에 가우시안 잡음 가정이 필요한가? 선형 오토인코더를 학습시켜 PCA와 같은 오차가 나왔다면 무엇까지 말할 수 있는가?

</div>

??? success "풀이"
    **필요하지 않다.** 정리 2는 유한 행렬에 대한 대수적 등식이고, 에카르트–영–미르스키 정리의 랭크 $k$ 이하 프로베니우스 근사 최적성도 분포 가정을 쓰지 않는다. 가우시안 모형은 제곱 오차를 확률적으로 **해석하는** 별개의 관점이지 최적성의 전제가 아니다.

    **선형 오토인코더 쪽은 말할 수 있는 것이 좁다.** 정리 3대로 전역 최적의 로스는 PCA와 같다. 그러므로 학습 로스가 $\sum_{j>k}\lambda_j$에 가깝다면 **그 부분공간을 거의 찾았다** 고는 말할 수 있다.

    말할 수 없는 것이 셋이다.

    - **기저가 같다**는 보장은 없다. 가역 $A$에 대해 $W_e=WA^{-T},\ W_d=WA$가 같은 로스를 주므로 열이 주성분일 이유가 없다(연습문제 10과 같은 자유도다)
    - **전역 최적에 닿았다**는 보장도 없다. 유한 에포크의 근접한 값일 뿐이다
    - **새 데이터에서도 같다**는 보장은 더더욱 없다. 둘 다 학습 데이터의 오차를 견준 것이다

    확인하려면 로스가 아니라 **부분공간을 견주어야 한다** — 두 기저가 펼치는 공간의 주각(principal angle)을 재는 것이 한 방법이다.

## 정리하며

1. **PCA는 두 얼굴을 가진 하나의 문제다**: 분산 최대화와 재구성 오차 최소화가 같은 $W$를 고른다(정리 2). 직교 투영의 피타고라스 분해가 그 다리다
2. **고유벡터가 답인 까닭은 정지 조건에 있지 않다**: 정지 조건은 모든 고유벡터를 후보로 내놓고, 그중 최대를 고르는 것은 목적값이 곧 고윳값이기 때문이다
3. **부분공간과 기저는 다른 문제다**: 대각합 목적은 부분공간만 정한다. 임의의 직교 $Q$에 대해 $WQ$가 같은 값을 준다
4. **선형 오토인코더는 부분공간을 맞히고 기저는 맞히지 않는다**(정리 3). 로스가 같다고 축이 주성분인 것이 아니다
5. **설명 분산 비율은 그 데이터의 제곱 거리에 대한 것이다**: 95%를 남겼다는 말이 예측 정보 95%를 남겼다는 뜻이 아니고, 테스트 오차는 학습 고윳값으로 알 수 없다
6. **분모와 단위를 밝힌다**: $n$과 $n-1$, 표본별 제곱 거리와 원소별 MSE는 서로 다른 수다
7. **비유일성의 출처는 셋이다**: 부호, 공간 안의 중복 고윳값, 선택 경계의 중복. 거의 같은 고윳값은 축을 불안정하게 만든다
