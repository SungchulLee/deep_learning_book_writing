# 주성분 분석의 유도

PCA의 분산 최대화와 재구성 오차 최소화는 같은 직교 부분공간을 선택하는 문제다. 이 절에서는 중심화된 유한 데이터에 대한 등식을 유도하고, 최적 부분공간과 그 안의 기저가 얼마나 유일한지 살펴본다.

## 1. 표기와 공분산

X∈ℝⁿˣᵈ의 각 행은 학습 평균을 뺀 표본이며 Σ=XᵀX/n이다. Σ는 실수 대칭 양의 준정부호 행렬이므로 다음과 같이 분해된다.

$$\Sigma=V\Lambda V^T,\qquad V^TV=I,\qquad
\lambda_1\ge\cdots\ge\lambda_d\ge0.$$

여기서는 n으로 나눈 경험 공분산을 사용한다. n>1에서 1/(n−1)을 사용하면 고윳값이 n/(n−1)배 되지만 주성분 방향과 설명 분산 비율은 같다. 이후 표본별 평균 오차와의 등식은 n 관례를 기준으로 한다.

## 2. 첫 주성분: 분산 최대화

단위 벡터 v에 대한 점수는 z_i=x_iᵀv다. 평균이 0이므로 경험 분산은 다음과 같다.

$$\frac1n\sum_i z_i^2=v^T\Sigma v.$$

제약 vᵀv=1이 없으면 v를 키워 목적을 무한히 키울 수 있다. 라그랑주 함수는 다음과 같다.

$$\mathcal L(v,\lambda)=v^T\Sigma v-\lambda(v^Tv-1).$$

정지 조건 2Σv−2λv=0으로부터 Σv=λv를 얻는다. 이때 목적값은 vᵀΣv=λ다. 정지 조건만으로 최대·최소를 구분한 것은 아니며, 목적을 최대화하려면 가장 큰 고윳값의 고유공간에서 단위 벡터를 선택한다. 단위 구가 콤팩트하고 목적이 연속이므로 최댓값은 존재한다.

## 3. 여러 성분과 부분공간

두 번째 성분에는 vᵀv₁=0을 추가한다. 정지 조건은 2Σv−2λv−μv₁=0이다. 왼쪽에 v₁ᵀ를 곱하면 대칭성과 직교성으로 μ=0을 얻는다. 따라서 남은 직교 공간에서 가장 큰 고윳값인 λ₂의 방향을 선택한다. 같은 절차로 직교 고유벡터를 구성한다.

k개 방향을 한꺼번에 구하는 문제는 다음과 같다.

$$\max_{W^TW=I_k}\operatorname{tr}(W^T\Sigma W).$$

상위 k개 고유벡터를 열로 둔 W가 최댓값 Σ_{j≤k}λ_j를 달성한다. 하지만 이 **대각합 목적만으로** W의 열을 개별 주성분 순서로 고정하지는 않는다. 어떤 직교 Q에 대해서도 WQ는 같은 부분공간과 목적값을 가지며, 일반적인 Q에서는 점수 공분산이 대각행렬이 아닐 수 있다. PCA의 표준 기저는 공분산을 대각화하고 분산 순으로 정렬한 기저다.

## 4. 재구성 오차 최소화와의 동등성

WᵀW=I일 때 P=WWᵀ는 대칭이며 P²=P인 직교 투영 행렬이다. 피타고라스 관계로 다음이 성립한다.

$$\|x-Px\|^2=\|x\|^2-x^TPx.$$

표본에 대해 평균을 취하면 다음과 같다.

$$\frac1n\|X-XWW^T\|_F^2=\operatorname{tr}(\Sigma)-\operatorname{tr}(W^T\Sigma W).$$

첫 항은 W와 무관하므로 재구성 오차 최소화가 분산 최대화와 같다. PCA 기저에서 최소 오차는 다음과 같다.

$$\mathcal E_k=\sum_{j>k}\lambda_j.$$

왼쪽은 표본당 제곱 거리 평균이다. `mean()`으로 모든 n×d 원소의 제곱 오차를 평균하면 값은 E_k/d다. 1/(n−1) 고윳값을 쓴 경우에는 오차 환산에 (n−1)/n도 필요하다.

에카르트–영–미르스키 정리는 X의 절단 SVD가 랭크 **k 이하** 행렬 중 프로베니우스 오차를 최소화함을 보인다. 이 대수적 최적성에는 가우시안 잡음 가정이 필요하지 않다. 가우시안 오차 모델은 제곱 오차를 확률적으로 해석하는 별도의 관점이다.

## 5. 계산과 평균 복원

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

## 6. 선형 오토인코더와의 관계

열벡터 표기에서 인코더 W_eᵀ와 디코더 W_d의 재구성은 W_dW_eᵀx다. 중심화된 데이터와 바이어스·활성화 없는 모델의 로스를 다음과 같이 둔다.

$$L(W_e,W_d)=\frac1n\sum_i\|x_i-W_dW_e^Tx_i\|^2.$$

W_e를 고정하고 W_d로 미분하면 정상 방정식을 얻는다.

$$\Sigma W_e=W_d(W_e^T\Sigma W_e).$$

괄호 안 행렬이 가역이면 다음과 같다.

$$W_d=\Sigma W_e(W_e^T\Sigma W_e)^{-1}.$$

특이한 경우에는 역행렬을 그대로 사용할 수 없으며 의사역행렬과 비유일성을 고려해야 한다. 양의 정부호 Σ에서 1≤k<d이고 λ_k>λ_{k+1}라면 최적 재구성은 상위 k차원 부분공간의 투영이다. 예를 들어 가역 A에 대해 W_e=W A⁻ᵀ, W_d=W A는 같은 최적 재구성을 준다. 따라서 두 웨이트가 같거나 인코더가 정규직교인 것은 필수 조건이 아니다.

공분산이 특이하면 학습 데이터가 놓이지 않는 영공간에서의 동작은 로스로 정해지지 않는다. 중복 고윳값이 경계를 가로지르면 최적 부분공간 자체도 여러 개일 수 있다. [선형 오토인코더 연구](https://proceedings.mlr.press/v97/kunin19a.html)의 주부분공간 결과를 임의의 유한 학습 실행에 대한 수렴 보장으로 확대하지 않는다.

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

## 7. 분산 비율과 상관 없는 좌표

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

## 8. 부호와 유일성

각 고유벡터의 부호는 자유다. 점수와 방향을 함께 뒤집으면 재구성과 거리는 유지된다. 부호를 고정하는 표시 관례는 해석을 돕지만 학습된 방향의 의미를 추가로 보장하지 않는다.

λ_k>λ_{k+1}이면 상위 k차원 부분공간은 유일하다. 개별 방향까지 부호 외에 유일하려면 해당 고윳값이 단순해야 한다. 선택된 공간 내부의 중복은 기저만 비유일하게 만들고, 선택 경계의 중복은 어떤 k차원 부분공간을 고를지도 비유일하게 만든다. 거의 같은 고윳값은 작은 데이터 변화에서 개별 축을 불안정하게 만들 수 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
첫 주성분이 최대 고윳값의 단위 고유벡터인 이유를 유도하라.

</div>

??? success "풀이"

    제약 목적은 max wᵀΣw, wᵀw=1이다. 라그랑주 정지 조건은 Σw=λw이고 목적값은 λ다. 따라서 가장 큰 고윳값의 단위 고유벡터를 고른다. 정지 조건만으로 모든 고유벡터가 최대라는 결론을 내리지는 않는다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
SVD와 공분산 고유분해의 관계를 설명하라.

</div>

??? success "풀이"

    중심화 X=USVᵀ이면 Σ=XᵀX/n=V(S²/n)Vᵀ다. V의 열이 주방향이고 점수는 U_kS_k다. 1/(n−1) 관례는 고윳값만 s_j²/(n−1)로 바꾼다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
커널 PCA가 선형 PCA의 최적성에 모순되지 않으면서 다른 결과를 주는 이유를 설명하라.

</div>

??? success "풀이"

    선형 PCA의 최적성은 원래 좌표의 제곱 재구성 오차와 고정 랭크에 대한 것이다. 커널 PCA는 비선형 피처 공간에서 다른 공분산과 기하를 사용한다. 따라서 분류나 시각화가 개선될 가능성은 있어도 원래 선형 최적성에 대한 반례는 아니다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
누적 설명 분산 비율로 성분 수를 정할 때 무엇이 보존되고 무엇은 보장되지 않는가?

</div>

??? success "풀이"

    양의 전체 분산에서 Σ_{j≤k}λ_j/Σ_jλ_j가 임계값에 도달하는 최소 k를 고른다. 보존 대상은 학습 데이터의 제곱 거리 기준 분산이다. 레이블 정보, 테스트 정확도와 파일 크기가 같은 비율로 보존되는 것은 아니다.



---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
분산 최대화와 재구성 오차 최소화가 같은 이유를 식으로 보여라.

</div>

??? success "풀이"

    WᵀW=I이면 P=WWᵀ는 직교 투영이므로 ||x||²=||Px||²+||x−Px||²다. 표본 평균을 취하면 tr(Σ)=tr(WᵀΣW)+E(W)다. 왼쪽이 W와 무관하므로 설명 분산을 최대화하면 오차가 최소화된다.



---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
한꺼번에 k개 방향의 대각합을 최대화하면 각 열이 반드시 분산 순으로 정렬된 고유벡터인가?

</div>

??? success "풀이"

    그렇지 않다. 최적 W에 임의 직교 Q를 곱한 WQ도 같은 부분공간과 대각합을 갖는다. 개별 PCA 축을 얻으려면 해당 공간에서 공분산을 대각화하고 고윳값 순으로 정렬한다. 부분공간 최적성과 기저 선택을 구분해야 한다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
공분산 고유분해와 직접 SVD의 수치·계산상 차이를 설명하라.

</div>

??? success "풀이"

    XᵀX를 만들면 0이 아닌 특잇값에 대한 조건수가 제곱되어 작은 성분을 구할 때 불리할 수 있다. 직접 SVD는 d×d 공분산을 명시적으로 만들지 않아도 된다. 다만 n이 매우 크고 d가 작으면 공분산 방식이 효율적일 수 있다. 데이터 크기와 필요한 성분 수에 맞추어 선택한다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
주성분 부호를 고정하는 관례의 역할과 아래 코드의 한계를 설명하라.

</div>

??? success "풀이"

    부호 고정은 반복 결과의 표시와 해석을 맞추는 관례이며 재구성이나 설명 분산을 바꾸지 않는다. 아래 코드는 V의 행이 성분이라는 전제이고 열이 성분인 행렬에는 그대로 적용할 수 없다. 행 합이 0이면 이 규칙으로 부호가 정해지지 않는다. 점수도 같은 부호로 바꾸어야 한다. 중복 고윳값의 공간 안 회전 자유는 부호 규칙만으로 해결되지 않는다.

    ```python
    for j in range(k):
        if V[j].sum() < 0:
            V[j] = -V[j]
    ```


---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
주부분공간과 개별 주성분의 유일성 조건을 구분하라.

</div>

??? success "풀이"

    λ_k>λ_{k+1}이면 상위 k차원 부분공간은 유일하다. 그 내부에 중복 고윳값이 있으면 개별 축은 유일하지 않다. 경계에서 λ_k=λ_{k+1}이면 그 고유공간을 가르는 최적 부분공간도 여러 개다. 개별 단순 고윳값의 축은 부호를 제외하고 정해진다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
PCA의 최적성에 가우시안 잡음 가정이 필요한가? 선형 오토인코더 학습과의 관계도 설명하라.

</div>

??? success "풀이"

    유한 행렬의 랭크 k 이하 프로베니우스 근사 최적성에는 확률적 잡음 가정이 필요하지 않다. 가우시안 모형은 제곱 오차의 별도 확률 해석이다. 적절한 선형 오토인코더의 전역 최적 로스는 PCA와 같지만, 임의 초기화와 유한 경사 학습의 도달 여부나 인코더 기저의 일치까지 보장하지 않는다.



## 정리하며

PCA는 직교 투영의 피타고라스 분해를 통해 분산 최대화와 재구성 오차 최소화를 연결한다. 고유벡터 기저, 최적 부분공간과 일반적인 선형 인코더는 구분해야 한다. 분모와 오차 평균 단위를 맞추고, 부호·중복 고윳값·특이 공분산에 따른 비유일성을 고려한다.
