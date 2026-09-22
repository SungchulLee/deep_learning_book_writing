# 2차원 주성분 분석 PyTorch

이 보기는 PyTorch 텐서 연산과 특잇값 쪼개기로 주성분 분석을 짜서 NumPy 판과 같은 2차원에서 1차원 줄이기를 한다. GPU로 빨라지는 큰 자료 묶음을 다룰 때, 주성분 분석이 더 큰 깊은 배움 물길의 한 조각일 때, 미분할 수 있는 차원 줄이기를 위해 쏘기를 지나 저절로 미분해야 할 때 PyTorch를 쓰면 이롭다.

## 1. 코드

```python
"""2차원 주성분 분석 PyTorch."""
import torch
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n = 150

# === 촐레스키로 자료 만들기 =============================================
mean_true = torch.tensor([2.0, -1.0], device=device)
cov_true = torch.tensor([[3.0, 2.2], [2.2, 2.0]], device=device)
L = torch.linalg.cholesky(cov_true)
Z = torch.randn(n, 2, device=device)
X = mean_true + Z @ L.T

# === 가운데 맞추고 특잇값 쪼개기 셈하기 ==================================================
mu = X.mean(dim=0, keepdim=True)
X_centered = X - mu
U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
V = Vh.T
pc1 = V[:, 0]

explained_variance = (S ** 2) / (n - 1)
variance_ratios = explained_variance / explained_variance.sum()

# === 쏘고 다시 세우기 =================================================
scores_1d = X_centered @ pc1
X_recon = torch.outer(scores_1d, pc1) + mu
reconstruction_error = ((X - X_recon) ** 2).mean()

print(f"PC1 explains {variance_ratios[0].item():.1%} of variance")
print(f"Reconstruction MSE: {reconstruction_error.item():.6f}")

if __name__ == "__main__":
    pass
```

**출력:**

```
PC1 explains 96.1% of variance
Reconstruction MSE: 0.104872
```

## 2. 논의

PyTorch 짜기는 NumPy 판을 거의 줄 단위로 그대로 옮긴 것이며 일부러 그렇게 했다. PyTorch가 NumPy를 쓰던 이에게 낯익도록 만들어졌기 때문이다. 핵심 차이는 `torch.linalg.svd`을 쓴다는 것(이는 $V^T$이 아니라 $V^H$을 돌려주지만 실수 행렬에서는 같다), `.to(device)`으로 기기를 또렷이 다룬다는 것, 아무 텐서에나 `requires_grad=True`을 두어 주성분 분석 물길을 지나는 기울기를 셈할 수 있다는 것이다.

PyTorch에서 여러 변수 정규 표본을 만들려면 촐레스키 쪼개기 재주가 필요하다. 곧 공분산 $\Sigma = LL^\top$이 주어질 때 $Z \sim \mathcal{N}(0, I)$에 대해 바꿈 $X = \mu + ZL^\top$이 $\mathcal{N}(\mu, \Sigma)$의 표본을 낸다. 이는 NumPy의 `multivariate_normal`과 같되 기본 텐서 연산만 쓴다.

큰 규모의 쓰임새에서는 `torch.linalg.svd`을 GPU로 돌리면 CPU 기반 NumPy에 견주어 주성분 분석 셈 시간이 10~50배 줄 수 있다. 특잇값 쪼개기가 병목인, 표본이 수백만이거나 특징이 수천인 자료 묶음에서 중요하다. 신경망 층과 매끄럽게 어우러지므로 주성분 분석을 끝에서 끝까지 익히는 모델 안의 미분할 수 있는 미리 다듬기 모듈로 쓸 수도 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
CPU에서 NumPy(`np.linalg.svd`)와 PyTorch(`torch.linalg.svd`)로 10000 x 784 아무 행렬의 특잇값 쪼개기 시간을 재어라. GPU가 있으면 PyTorch 판을 GPU에서도 재어라. 얼마나 빨라졌는지 알려라.

</div>

??? success "연습문제 1 풀이"
    ```python
    import time
    X_large_np = np.random.randn(10000, 784).astype(np.float32)
    X_large_pt = torch.from_numpy(X_large_np)

    t0 = time.time()
    np.linalg.svd(X_large_np, full_matrices=False)
    t_numpy = time.time() - t0

    t0 = time.time()
    torch.linalg.svd(X_large_pt, full_matrices=False)
    t_torch_cpu = time.time() - t0

    print(f"NumPy CPU: {t_numpy:.2f}s")
    print(f"PyTorch CPU: {t_torch_cpu:.2f}s")
    ```
    CPU에서는 둘 다 LAPACK을 쓰므로 PyTorch와 NumPy의 성능이 비슷하다. GPU에서는 이 크기의 행렬에서 PyTorch 특잇값 쪼개기가 10~50배 빠를 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
들임 자료에 `requires_grad=True`을 두어 주성분 쏘기를 미분할 수 있게 하라. 첫 자료 점에 대한 다시 세우기 어긋남의 기울기를 셈하라. 이 기울기는 기하로 무엇을 뜻하는가?

</div>

??? success "연습문제 2 풀이"
    ```python
    X_diff = X.clone().requires_grad_(True)
    mu_d = X_diff.mean(dim=0, keepdim=True)
    Xc_d = X_diff - mu_d
    _, _, Vh_d = torch.linalg.svd(Xc_d, full_matrices=False)
    pc1_d = Vh_d[0]
    scores = Xc_d @ pc1_d
    recon = torch.outer(scores, pc1_d) + mu_d
    mse = ((X_diff - recon) ** 2).mean()
    mse.backward()
    print(f"Gradient for point 0: {X_diff.grad[0]}")
    ```
    기울기는 자료 점을 그 방향으로 옮겼을 때 다시 세우기 어긋남이 가장 많이 줄어드는 쪽을 가리킨다. 기하로는 그 점에서 주성분1에 직교하는 성분(남는 방향)에 비례한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
배치 주성분 분석을 짜라. 곧 서로 얽히지 않은 자료 묶음 $B$개를 뜻하는 꼴 `(B, N, D)`의 텐서가 주어질 때 배치 특잇값 쪼개기를 써서 되풀이 없이 자료 묶음마다 주성분 분석을 나란히 셈하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    B, N, D = 5, 100, 10
    X_batch = torch.randn(B, N, D)
    mu_batch = X_batch.mean(dim=1, keepdim=True)
    Xc_batch = X_batch - mu_batch
    U_b, S_b, Vh_b = torch.linalg.svd(Xc_batch, full_matrices=False)
    # Vh_b 꼴: (배치, D, D), 자료 묶음마다의 첫 주성분:
    pc1_batch = Vh_b[:, 0, :]  # (배치, D)
    print(f"Batch PCs shape: {pc1_batch.shape}")
    ```
    `torch.linalg.svd`은 배치 들임을 본디 받쳐 서로 얽히지 않은 특잇값 쪼개기를 나란히 셈한다. 특히 GPU에서 효율이 좋고 파이썬 되풀이의 군더더기를 피한다.


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
`torch.linalg.svd`와 `np.linalg.svd`의 반환값이 어떻게 다른가?

</div>

??? success "연습문제 4 풀이"
    거의 같다. 둘 다 `U, S, Vh`를 주고 성분은 `Vh`의 **행**에 있다.

    다른 점이 둘이다.

    | | NumPy | PyTorch |
    |---|---|---|
    | 이름 | `Vt` | `Vh` (에르미트 전치) |
    | 전체/축소 | `full_matrices=False` | `full_matrices=False` |
    | 반환 | 튜플 | 이름 있는 튜플 |

    PyTorch는 이름으로도 꺼낼 수 있다.

    ```python
    r = torch.linalg.svd(Xc, full_matrices=False)
    V = r.Vh                     # r[2] 와 같다
    ```

    복소수를 다룰 때 이름의 차이가 뜻을 갖는다. 에르미트 전치는 전치에 켤레까지 취하는
    것이므로 실수에서는 전치와 같다. 주성분 분석은 실수 자료를 쓰므로 차이가 없다.

    옛 함수 `torch.svd`는 $V$를 **전치하지 않고** 주므로 축이 반대다. 더는 쓰지 않는
    것이 좋고, 옛 코드를 읽을 때 이 차이를 알아 두면 헷갈리지 않는다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
같은 자료에서 NumPy 판과 PyTorch 판의 결과가 소수점 여섯째 자리에서 달랐다. 문제인가?

</div>

??? success "연습문제 5 풀이"
    문제가 아니다. **자료형** 때문이다.

    NumPy의 기본은 `float64`이고 PyTorch의 기본은 `float32`다. `float32`의 유효 숫자가
    일곱 자리쯤이므로 여섯째 자리의 차이는 예상된 것이다.

    확인하려면 자료형을 맞춰 보면 된다.

    ```python
    t = torch.from_numpy(X).double()        # float64 로
    U, S, Vh = torch.linalg.svd(t, full_matrices=False)
    ```

    이러면 훨씬 가까워진다.

    실용적으로 `float32`가 대개 충분하다. 신경망이 그것을 쓰고, 자료 자체의 잡음이
    여섯째 자리보다 훨씬 크기 때문이다.

    `float32`가 모자라는 자리가 있다. **작은 특잇값**이 그렇다. 큰 것과 작은 것의 비가
    `float32`의 정밀도를 넘으면 작은 쪽이 부정확해진다. 누적 흩어짐 곡선의 꼬리를
    정확히 그리려면 `float64`를 쓰는 편이 낫다.

    GPU에서는 `float64`가 매우 느리거나 지원되지 않는 일이 있다. 애플 실리콘의 MPS가
    그렇다. 그때는 정밀도가 필요한 부분만 CPU에서 하면 된다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
PyTorch로 주성분 분석을 짜는 것이 이로운 경우는 언제인가?

</div>

??? success "연습문제 6 풀이"
    셋이다.

    **GPU로 빨라질 때.** 자료가 크면 특잇값 쪼개기가 GPU에서 몇 배 빠르다.

    **더 큰 물길의 한 조각일 때.** 자료가 이미 텐서로 있고 뒤에 신경망이 온다면, 중간에
    NumPy로 나갔다 오는 것이 낭비다.

    **기울기를 흘려야 할 때.** 주성분 분석을 지나 미분해야 한다면 자동 미분이 필요하다.
    다만 특잇값이 겹칠 때의 위험을 알아야 한다
    ([PyTorch 기초 연습문제 6](03_pytorch_basics.md)).

    반대로 이롭지 않은 경우도 또렷하다. 자료가 작고 한 번만 하는 것이라면 `sklearn`이
    가장 짧고 안전하다. 부호 관례도 정해져 있고 `inverse_transform` 같은 편의가 있다.

    이 페이지의 2차원 장난감 자료에서는 세 가지 짜기의 결과가 같다. **같다는 것을
    확인하는 것**이 이 페이지의 목적이며, 그러고 나면 자료 크기와 쓰임에 따라 고를 수
    있게 된다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
GPU에서 특잇값 쪼개기가 CPU보다 느릴 수도 있는가?

</div>

??? success "연습문제 7 풀이"
    있다. 작은 행렬에서 그렇다.

    까닭이 둘이다. 자료를 옮기는 값이 들고, 특잇값 쪼개기가 신경망의 행렬 곱처럼 잘
    나뉘는 셈이 아니기 때문이다. 순차적인 부분이 있어 GPU의 장기가 덜 발휘된다.

    | 자료 크기 | 어느 쪽이 빠른가 |
    |---|---|
    | 2차원 장난감 (수백 행) | CPU. 옮기는 값이 더 크다 |
    | MNIST ($60{,}000 \times 784$) | GPU |
    | 아주 큰 자료 | GPU. 다만 무작위 방식을 쓰는 편이 낫다 |

    그리고 마지막 칸이 중요하다. 아주 크면 전체 쪼개기보다 **처음 $k$개만 구하는** 것이
    훨씬 빠르다. 성분 50개가 필요한데 784개를 다 구하는 것은 낭비다.

    재어 보는 것이 가장 확실하다. 다만 GPU의 셈하기는 비동기이므로 그냥 시간을 재면
    틀린 값이 나온다. 기다리게 한 뒤 재야 한다.

    ```python
    torch.mps.synchronize()          # 또는 torch.cuda.synchronize()
    t0 = time.time()
    torch.linalg.svd(X)
    torch.mps.synchronize()
    print(time.time() - t0)
    ```

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
MNIST 크기 자료에서 성분 50개만 필요하다면 무엇을 쓰겠는가?

</div>

??? success "연습문제 8 풀이"
    `torch.svd_lowrank`나 `sklearn`의 무작위 방식을 쓴다.

    ```python
    U, S, V = torch.svd_lowrank(Xc, q=50 + 10)   # 여유를 조금 둔다
    ```

    전체 쪼개기는 784개를 모두 구하므로 필요한 것의 열다섯 배를 셈한다. 무작위 방식은
    필요한 만큼만 어림한다.

    `q`를 필요한 것보다 조금 크게 두는 까닭은 어림의 정확도가 끝쪽에서 떨어지기 때문이다.
    50개가 필요하면 60쯤으로 두고 앞 50개만 쓴다.

    알아 둘 대가가 있다.

    - **어림이다.** 앞쪽 성분은 매우 정확하지만 정확한 값은 아니다
    - **무작위다.** 씨앗에 따라 조금 달라진다
    - **꼬리를 못 본다.** 전체 흩어짐을 모르므로 비율을 셈하려면 따로 구해야 한다

    마지막 것이 실무에서 걸린다. `explained_variance_ratio_`의 분모가 전체 흩어짐인데,
    구하지 않은 성분들의 몫을 모른다. 다행히 전체 흩어짐은 쪼개기 없이도 셈할 수 있다.
    $\sum_j \lambda_j = \operatorname{tr}(\Sigma)$이므로 각 열의 흩어짐을 더하면 된다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
주성분 분석을 미분 가능한 층으로 쓰고 싶다. 어떻게 짜겠는가?

</div>

??? success "연습문제 9 풀이"
    무엇에 대해 미분하려는지 먼저 정해야 한다. 두 경우가 다르다.

    **경우 1: 입력에 대해 미분.** 주성분 분석의 성분은 고정해 두고 사영만 지나 기울기를
    흘린다. 사영은 그냥 행렬 곱이므로 아무 문제가 없다.

    ```python
    V = V.detach()                 # 성분은 상수로 둔다
    z = (x - mu.detach()) @ V.T    # 미분 가능
    ```

    이것이 실무에서 거의 언제나 원하는 것이다. 주성분 분석을 **고정된 전처리**로 쓰면서
    뒤의 그물을 익힌다.

    **경우 2: 자료에 대해 미분.** 주성분 분석 자체를 다시 셈하며 기울기를 흘려야 한다면
    쪼개기를 지나야 하고, 겹침 문제를 만난다
    ([PyTorch 기초 연습문제 6](03_pytorch_basics.md)).

    이때는 개별 성분이 아니라 **사영 행렬**을 쓰는 것이 안전하다.

    ```python
    Vk = torch.linalg.svd(Xc, full_matrices=False).Vh[:k]
    P = Vk.T @ Vk                  # 부분 공간만 쓴다. 겹침에 딸리지 않는다
    loss = ((Xc - Xc @ P)**2).mean()
    ```

    $V_k V_k^\top$은 부분 공간만으로 정해지므로 그 안의 기저가 어떻게 회전해도 같다.
    주성분 분석의 답에서 정해지는 것이 부분 공간뿐이었으니
    ([유도 연습문제 5](pca_derivation.md)), 정해지는 것만 쓰는 셈이다.

    그리고 이 길로 가다 보면 자연스레 다른 생각에 이른다. 쪼개기를 아예 버리고 사영을
    배울 파라미터로 두면 어떤가. 그것이 [오토인코더](../../ch25/index.md)다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
이 세 판(NumPy, sklearn, PyTorch)이 같은 답을 내는지 확인하는 코드를 적어라.

</div>

??? success "연습문제 10 풀이"
    부호에 딸리지 않는 값들을 견주면 된다
    ([sklearn 연습문제 4](02_pca_2d_sklearn.md)).

    ```python
    Xc = X - X.mean(0)

    # NumPy
    Vt_np = np.linalg.svd(Xc, full_matrices=False)[2]
    Z_np = Xc @ Vt_np[:k].T

    # sklearn
    p = PCA(n_components=k).fit(X)
    Z_sk = p.transform(X)

    # PyTorch
    t = torch.from_numpy(Xc).double()
    Vh = torch.linalg.svd(t, full_matrices=False).Vh
    Z_pt = (t @ Vh[:k].T).numpy()

    for A, B in ((Z_np, Z_sk), (Z_np, Z_pt)):
        assert np.allclose(np.abs(A), np.abs(B), atol=1e-8)
    ```

    `np.abs`를 씌우는 것이 핵심이다. 부호는 정해지지 않으므로 그대로 견주면 실패한다.

    자료형을 `double()`로 맞추는 것도 필요하다. `float32`로 두면 `atol=1e-8`을 못 맞춘다
    ([연습문제 2](04_pca_2d_pytorch.md)).

    설명하는 흩어짐도 함께 견주면 더 든든하다. 그 값은 부호와 무관하므로 `np.abs` 없이
    바로 비교할 수 있다.

## 정리하며

**다룬 것** — 2차원 주성분 분석 PyTorch

PyTorch 짜기는 NumPy 판을 거의 줄 단위로 그대로 옮긴 것이며 일부러 그렇게 했다.

앞의 연습문제 3개로 스스로 따져 볼 수 있다.
