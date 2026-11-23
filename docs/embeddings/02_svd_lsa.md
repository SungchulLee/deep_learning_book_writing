# 특잇값 분해 — 고윳값 분해를 아무 행렬로 넓힌다 · LSA

<div class="leadbox" markdown>

**들어가는 이야기** — `awful` 로 찾으면 `terrible` 이 걸리는가

영화평 6,000편 가운데 「끔찍하다」는 평을 찾고 싶다. 검색어로 `awful` 을 넣는다. 단어를 맞춰 보는 검색이라면 `awful` 이 든 358편이 걸린다.

그런데 `awful` 은 안 쓰고 **`terrible` 만 쓴 평이 280편** 있다. 뜻은 같은데 단어가 다르다. 단어를 맞춰 보는 검색에서 이 280편의 점수는 **정확히 0**이다. `awful` 과 공통 단어가 없는 글과 똑같이 취급된다.

이 쪽은 단어 × 문서 행렬 하나를 **특잇값 분해**로 64차원까지 줄인다. 그러고 같은 검색어를 넣으면 280편의 평균 **코사인 유사도**가 **0.267**이 되고 — 두 벡터가 이루는 각의 코사인이며, 방향이 같으면 1이고 수직이면 0이다 — 두 단어 다 없는 글은 0.115에 머문다. 사전도 동의어 목록도 주지 않았는데 「같은 뜻 다른 말」이 이어졌다. 이것이 **LSA**다.

어떻게 이어졌는지가 이 쪽의 물음이다. 그리고 그 답을 따라가다 보면, 같은 분해를 조금만 다르게 쓰면 긍정·부정 가르기에서 정확도가 **0.86에서 0.56으로** 떨어지는 자리를 만난다.

</div>

[5.1절의 고윳값 분해 쪽](../ch05/dimreduction/00_eigen.md)은 대칭인 정사각 행렬을 고유벡터로 풀었다. 그런데 이 장이 다루는 행렬은 대개 정사각이 아니다. 단어 8,000개 × 영화평 6,000편이 그렇다. 이 쪽은 고윳값 분해를 **아무 행렬로 넓힌** 특잇값 분해를 세우고, 둘이 어떻게 이어지는지 잰 뒤, 그 분해를 단어 × 문서 행렬에 쓰는 LSA를 본다. [다음 쪽](02_counting.md)은 같은 분해를 단어 × 단어 행렬에 쓴다.

---

## 1. 특잇값 분해

$m \times n$ 행렬 $X$ 는 어떤 것이든 셋의 곱으로 적힌다.

$$
X = U\,\Sigma\,V^\top
$$

- $U$ ($m \times m$) — 열이 서로 수직인 단위벡터다. 이런 열을 **정규직교**(orthonormal)라 하고, $U$ 의 열을 **왼쪽 특이벡터**라 한다.
- $V$ ($n \times n$) — 역시 열이 정규직교다. **오른쪽 특이벡터**라 한다.
- $\Sigma$ ($m \times n$) — 대각선에만 0 이상의 수 $\sigma_1 \ge \sigma_2 \ge \cdots$ 가 놓인다. 이것이 **특잇값**(singular value)이다.

이것이 **특잇값 분해**(singular value decomposition, SVD)다. 고윳값 분해 $A = Q\Lambda Q^\top$ 와 형태가 닮았는데, 양쪽 행렬이 **서로 다르다**는 것이 차이다. $X$ 는 $n$ 차원 벡터를 받아 $m$ 차원 벡터를 내놓으므로, 들어가는 쪽의 축($V$)과 나오는 쪽의 축($U$)이 같을 수가 없다. 그래서 고윳값 분해는 정사각 행렬에만 있지만 특잇값 분해는 모든 행렬에 있다.

읽는 법은 고윳값 분해와 같다. $Xv_i = \sigma_i u_i$ 다. 들어가는 쪽의 축 $v_i$ 는 $X$ 를 지나 나오는 쪽의 축 $u_i$ 가 되고, 길이가 $\sigma_i$ 배가 된다. 거꾸로 왼쪽에서 곱하면 $u_i^\top X = \sigma_i v_i^\top$ 이다 — 나오는 쪽의 축을 가로로 눕혀 곱하면 짝인 들어가는 쪽의 축이 나온다. 4절에서 검색어를 접어 넣을 때 쓰는 것이 이 형태다. 고윳값 분해와 마찬가지로 **조각의 합**으로도 적힌다.

$$
X = \sum_i \sigma_i\, u_i\, v_i^\top
$$

조각 하나 $u_i v_i^\top$ 는 $m$ 칸짜리 세로 벡터와 $n$ 칸짜리 가로 벡터를 곱한 $m \times n$ 행렬이다. 행렬 전체와 크기가 같은데, 들고 있는 수는 $m + n$ 개뿐이다. 두 분해를 나란히 놓으면 이렇다.

| | 고윳값 분해 $A = \sum \lambda_i q_i q_i^\top$ | 특잇값 분해 $X = \sum \sigma_i u_i v_i^\top$ |
|---|---|---|
| 되는 행렬 | 정사각, 그리고 깔끔하게 되려면 실수 대칭 | 아무 행렬 |
| 크기를 재는 수 | 고윳값 $\lambda_i$ — 실수, **음수도 된다** | 특잇값 $\sigma_i$ — **0 이상** |
| 축 | 한 벌 $\{q_i\}$, 서로 수직인 단위벡터 | 두 벌 $\{u_i\}$, $\{v_i\}$, 저마다 서로 수직인 단위벡터 |
| 한 축에서 하는 일 | $A q_i = \lambda_i q_i$ | $X v_i = \sigma_i u_i$, $u_i^\top X = \sigma_i v_i^\top$ |

## 2. 고윳값 분해와 어떻게 이어지는가

$X$ 에 제 전치를 곱하면 정사각이면서 대칭인 행렬이 된다. 그 둘에 특잇값 분해를 넣어 보면

$$
X^\top X = V\,\Sigma^\top U^\top U\,\Sigma\,V^\top = V\,(\Sigma^\top\Sigma)\,V^\top,
\qquad
X X^\top = U\,(\Sigma\Sigma^\top)\,U^\top
$$

이다. 둘 다 고윳값 분해의 형태 $Q\Lambda Q^\top$ 다. 그러니

- **$V$ 는 $X^\top X$ 의 고유벡터이고, $U$ 는 $X X^\top$ 의 고유벡터다.**
- **특잇값의 제곱 $\sigma_i^2$ 가 그 두 행렬의 (0이 아닌) 고윳값이다.**

[5.1절의 PCA](../ch05/dimreduction/01_pca.md)가 이 관계의 한 경우다. 중심화한 데이터 $X_c$ 에서 $X_c^\top X_c / (N-1)$ 이 공분산이므로, 공분산을 고윳값 분해하는 것과 $X_c$ 를 특잇값 분해하는 것이 같은 주성분을 준다. [고윳값 분해 쪽](../ch05/dimreduction/00_eigen.md)의 6절이 MNIST로 그것을 재어 두었다.

아래 프로그램의 (나)가 단어 × 문서 행렬에서 같은 것을 잰다. $XX^\top$ 는 8,000 × 8,000이고 $X^\top X$ 는 6,000 × 6,000으로 크기가 다른데, 앞 다섯 고윳값이 서로 같고 $\sigma^2$ 와도 같다.

| | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| $\sigma_i^2$ | 224,035.1 | 18,148.3 | 16,446.1 | 12,874.4 | 12,020.4 |
| $XX^\top$ 의 고윳값 | 224,035.1 | 18,148.3 | 16,446.1 | 12,874.4 | 12,020.4 |
| $X^\top X$ 의 고윳값 | 224,035.1 | 18,148.3 | 16,446.1 | 12,874.4 | 12,020.4 |

### 손으로 한 번 — 3 × 2 행렬

8,000 × 6,000은 손으로 풀 수 없으니, 같은 길을 작은 행렬로 한 번 걸어 본다.

$$
M = \begin{pmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{pmatrix},
\qquad
M^\top M = \begin{pmatrix} 14 & 32 \\ 32 & 77 \end{pmatrix}
$$

$M$ 은 정사각이 아니라 고윳값 분해를 할 수 없다. 그러나 $M^\top M$ 은 2 × 2 대칭 행렬이라 된다. 고윳값은 $\det(M^\top M - \lambda I) = 0$ 을 풀면 나온다.

$$
(14 - \lambda)(77 - \lambda) - 32 \cdot 32 = \lambda^2 - 91\lambda + 54 = 0,
\qquad
\lambda = \frac{91 \pm \sqrt{8065}}{2}
$$

곧 $\lambda_1 = 90.4027$, $\lambda_2 = 0.5973$ 이고, 특잇값은 그 제곱근 $\sigma_1 = 9.5080$, $\sigma_2 = 0.7729$ 다. 상수항 54를 $1078 - 1024$ 로 나누어 셈하는 자리에서 틀리기 쉽다. $\lambda_1 \lambda_2 = 54$ 이고 $\lambda_1 + \lambda_2 = 91$ 인지로 검산하면 된다.

아래 프로그램이 그 셈을 라이브러리의 답과 맞춘다. 이 블록은 홀로 돈다.

```python
"""특잇값 분해를 손으로 한 번 — 3 x 2 행렬.

M^T M 의 특성다항식을 풀어 특잇값을 얻고, 라이브러리의 답과 맞춘다.
"""
import math

import torch



def fmt(t):
    return "[" + ", ".join(f"{x:7.4f}" for x in t.tolist()) + "]"

if __name__ == "__main__":
    M = torch.tensor([[1., 4.], [2., 5.], [3., 6.]], dtype=torch.float64)

    # === 손으로: M^T M 의 고윳값 =============================================
    MtM = M.T @ M
    print("  M^T M =", MtM.tolist())
    tr, det = MtM.trace().item(), torch.linalg.det(MtM).item()
    print(f"  특성다항식  lambda^2 - {tr:.0f} lambda + {det:.0f} = 0")
    disc = tr * tr - 4 * det
    lam = [(tr + math.sqrt(disc)) / 2, (tr - math.sqrt(disc)) / 2]
    print(f"  판별식 {disc:.0f},  lambda = {lam[0]:.4f}, {lam[1]:.4f}")
    s_hand = [math.sqrt(l) for l in lam]
    print(f"  특잇값 = 루트 lambda = {s_hand[0]:.4f}, {s_hand[1]:.4f}")

    # === 라이브러리와 맞춘다 ================================================
    U, S, Vt = torch.linalg.svd(M, full_matrices=False)
    print(f"\n  torch.linalg.svd 의 특잇값  {S[0]:.4f}, {S[1]:.4f}")
    ev = torch.linalg.eigvalsh(M @ M.T).flip(0)
    print(f"  M M^T (3 x 3) 의 고윳값      {ev[0]:.4f}, {ev[1]:.4f}, "
          f"그리고 0 인가 {abs(ev[2].item()) < 1e-10}")

    # === 조각 둘의 합 ======================================================
    # 부호를 하나로 정해 둔다 — v 의 첫 칸이 양수가 되게
    for i in range(2):
        if Vt[i, 0] < 0:
            U[:, i], Vt[i] = -U[:, i], -Vt[i]
    for i in range(2):
        u, v = U[:, i], Vt[i]
        print(f"\n  조각 {i + 1}: s = {S[i]:.4f}   u = {fmt(u)}   v = {fmt(v)}")
        print(f"    u^T M = {fmt(u @ M)}           s v 와 같은가 "
              f"{torch.allclose(u @ M, S[i] * v)}")
        print(f"    M v   = {fmt(M @ v)}  s u 와 같은가 "
              f"{torch.allclose(M @ v, S[i] * u)}")
    P1 = S[0] * torch.outer(U[:, 0], Vt[0])
    P2 = S[1] * torch.outer(U[:, 1], Vt[1])
    print("\n  첫 조각 s1 u1 v1^T        둘째 조각 s2 u2 v2^T")
    for r in range(3):
        print(f"    {fmt(P1[r])}        {fmt(P2[r])}")
    print(f"  두 조각의 합이 M 인가  {torch.allclose(P1 + P2, M)}")

    # === 첫 조각만 남기면 ===================================================
    err = (M - P1).norm().item()
    print(f"\n  첫 조각만 남긴 오차 ||M - s1 u1 v1^T|| = {err:.4f}  (= s2 = {S[1]:.4f})")
    print(f"  제곱합 가운데 첫 조각이 담는 몫 {S[0]**2 / (S ** 2).sum():.2%}")
```

**출력:**

```
  M^T M = [[14.0, 32.0], [32.0, 77.0]]
  특성다항식  lambda^2 - 91 lambda + 54 = 0
  판별식 8065,  lambda = 90.4027, 0.5973
  특잇값 = 루트 lambda = 9.5080, 0.7729

  torch.linalg.svd 의 특잇값  9.5080, 0.7729
  M M^T (3 x 3) 의 고윳값      90.4027, 0.5973, 그리고 0 인가 True

  조각 1: s = 9.5080   u = [ 0.4287,  0.5663,  0.7039]   v = [ 0.3863,  0.9224]
    u^T M = [ 3.6731,  8.7699]           s v 와 같은가 True
    M v   = [ 4.0758,  5.3845,  6.6931]  s u 와 같은가 True

  조각 2: s = 0.7729   u = [-0.8060, -0.1124,  0.5812]   v = [ 0.9224, -0.3863]
    u^T M = [ 0.7129, -0.2986]           s v 와 같은가 True
    M v   = [-0.6229, -0.0869,  0.4492]  s u 와 같은가 True

  첫 조각 s1 u1 v1^T        둘째 조각 s2 u2 v2^T
    [ 1.5745,  3.7594]        [-0.5745,  0.2406]
    [ 2.0801,  4.9664]        [-0.0801,  0.0336]
    [ 2.5857,  6.1735]        [ 0.4143, -0.1735]
  두 조각의 합이 M 인가  True

  첫 조각만 남긴 오차 ||M - s1 u1 v1^T|| = 0.7729  (= s2 = 0.7729)
  제곱합 가운데 첫 조각이 담는 몫 99.34%
```

**손으로 푼 두 수가 `torch.linalg.svd` 와 넷째 자리까지 같다.** 3 × 3 인 $MM^\top$ 에서도 같은 두 수가 나오고, 남은 고윳값 하나는 0이다 — 2절의 「0이 아닌 고윳값이 같다」가 크기가 다른 두 행렬에서 그대로 보인다. $Mv_i = \sigma_i u_i$ 와 $u_i^\top M = \sigma_i v_i^\top$ 도 두 축 모두에서 맞는다.

![가로로 세 칸이 놓인 그림. 왼쪽 칸은 행렬 M 으로 1, 4, 2, 5, 3, 6 이 적혀 있고, 등호 뒤 가운데 칸은 첫 조각 s1 u1 v1 전치로 1.5745, 3.7594, 2.0801, 4.9664, 2.5857, 6.1735 이며 색이 M 과 거의 같다. 더하기 뒤 오른쪽 칸은 둘째 조각으로 -0.5745, 0.2406, -0.0801, 0.0336, 0.4143, -0.1735 이며 색이 거의 없다. 아래에 첫 조각이 제곱합의 99.34%를 담고 빠뜨린 것의 크기가 s2 = 0.7729 라고 적혀 있다](figures/svd_hand.svg)

그림이 3절을 미리 보여 준다. **첫 조각 하나가 $M$ 과 거의 같다.** 제곱합의 99.34%를 담고, 빠뜨린 것의 크기 $\|M - \sigma_1 u_1 v_1^\top\|$ 은 정확히 $\sigma_2$ 다. 첫 조각은 수 6개 대신 $\sigma_1$ 과 $u_1$(3칸), $v_1$(2칸)의 6개를 들고 있으니 이 작은 행렬에서는 아끼는 것이 없다. 아끼는 것은 행렬이 클 때다.

### 대칭 행렬이면 — 특잇값은 고윳값의 절댓값이다

$X$ 가 처음부터 대칭이면 둘이 더 가까워진다. $X = Q\Lambda Q^\top$ 이므로 $X^\top X = Q\Lambda^2 Q^\top$ 이고, 따라서 $\sigma_i = |\lambda_i|$ 다. 특이벡터도 고유벡터와 같은데, 한 가지가 다르다. 특잇값은 0 이상이어야 하므로 **고윳값이 음수인 방향에서는 $u_i$ 와 $v_i$ 중 하나가 부호를 뒤집어 그 음수를 떠안는다.** 곧 $\lambda_i < 0$ 이면 $v_i = -u_i$ 다.

공분산은 고윳값이 음수일 수 없으므로 이 일이 없다. 그런데 [다음 쪽](02_counting.md)이 만드는 PPMI(Positive Pointwise Mutual Information, 양의 점별 상호정보량) 행렬은 대칭이면서 **고윳값이 음수인 방향을 가진다.** (아)가 그 행렬을 같은 방식으로 만들어 잰다.

| | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 특잇값 $\sigma_i$ | 685.22 | 254.58 | 209.96 | 191.98 | 160.31 |
| 고윳값 $\lambda_i$ (절댓값 순) | 685.22 | **−254.58** | 209.96 | 191.98 | 160.31 |

위 64개에서 $\sigma_i = |\lambda_i|$ 가 모두 맞고, 고윳값이 음수인 방향은 **2번째와 49번째** 둘이다. $u_i$ 와 $v_i$ 의 부호가 반대인 자리도 정확히 그 둘이다.

음수 고윳값은 무엇을 뜻할까. 고유벡터 $q$ 에서 $q^\top M q = \lambda < 0$ 이라는 것은, $q$ 의 양 끝에 놓인 단어들이 **같은 끝끼리보다 서로 반대 끝과** 더 강하게 이어져 있다는 뜻이다. 2번 방향의 한쪽 끝은 `by, their, his, into, who, from` 이고 다른 끝은 `praise, affected, generations, relatives` 다. 같은 끝 50단어끼리의 PPMI 평균이 0.141 인데 두 끝 사이는 0.375 다. 기능어는 기능어끼리가 아니라 **그 뒤에 오는 내용어와** 같이 나온다(`his relatives`, `by generations`). 이 방향은 「비슷한 단어끼리」가 아니라 **「서로 짝을 이루는 두 무리」**를 담는다.

다음 쪽이 단어 벡터로 $U\Sigma^{1/2}$ 만 쓰고 $V$ 쪽을 버려도 되는 것은 이 두 방향을 빼면 $V = U$ 이기 때문이다. 그 두 방향에서는 버린 쪽에 부호가 실려 있으니, 엄밀히는 정보 한 조각을 버리는 셈이다.

## 3. 몇 개만 남긴다

1절의 합에서 큰 특잇값 $k$ 개만 남긴 $X_k = \sum_{i \le k} \sigma_i u_i v_i^\top$ 는 **랭크가 $k$ 인 행렬 가운데 $X$ 에 가장 가까운 것**이다. **랭크**(rank)는 행렬에서 서로 독립인 열의 개수이고, 조각 $u_i v_i^\top$ 하나가 랭크 1이다. 가깝다는 것은 칸마다의 차이를 제곱해 더한 값의 제곱근, 곧 **프로베니우스 노름**으로 잴 때 그렇고, 이것을 **Eckart–Young 정리**라 한다. 잃는 것은 버린 특잇값의 제곱합이다.

남긴 $X_k$ 는 $\sigma_i$, $u_i$, $v_i$ 를 $k$ 벌 들고 있으면 되므로 수 $k(m + n + 1)$ 개로 적힌다. 100 × 10 행렬을 두 조각으로 줄이면 1,000개 대신 222개, 이 쪽의 8,000 × 6,000을 64개로 줄이면 4,800만 개 대신 약 90만 개다. 특잇값 분해를 하는 까닭이 이것이다 — **행렬을 조각의 합으로 풀어 적고, 앞의 큰 조각만 쓴다.**

이 단어 × 문서 행렬에서 64개가 담는 것은 제곱합의 **14.4%**다(출력 (다)). 고윳값 분해 쪽의 MNIST 공분산이 64개로 86.2%를 담은 것과 크게 다르다. 손글씨는 퍼진 모양이 몇 방향에 몰려 있지만, 영화평의 단어 선택은 글마다 제각각이라 큰 방향 몇 개로 다 담기지 않는다. 그런데도 다음 절에서 보듯 쓸모가 있다. 남긴 14.4%가 단어들 사이의 큰 줄기이고, 버린 쪽에 글 하나하나의 우연한 단어 선택이 많이 들어 있기 때문이다.

## 4. LSA — 단어 × 문서 행렬을 쪼갠다

**LSA**(latent semantic analysis, 잠재 의미 분석)는 3절을 그대로 단어 × 문서 행렬에 쓴 것이다. 원래 목적은 정보 검색이었다(Deerwester 등, 1990). 들머리의 문제, 곧 같은 뜻을 다른 단어로 쓴 글을 놓치는 문제를 풀려고 나왔다.

**행렬을 채우는 법.** 칸 $(w, d)$ 에 단어 $w$ 가 글 $d$ 에 나온 횟수를 그대로 넣으면 `the` 같은 단어가 행렬을 휘어잡는다. 그래서 두 가지를 손본다.

- 횟수 대신 $\log(1 + \text{횟수})$ 를 쓴다. 한 글에서 열 번 나온 것이 한 번 나온 것의 열 배로 중요하지는 않다.
- **IDF**(inverse document frequency)를 곱한다. 단어 $w$ 가 나온 글이 $\mathrm{df}_w$ 편이면 $\log(6000 / \mathrm{df}_w)$ 다. 거의 모든 글에 나오는 단어는 글을 가르는 데 쓸모가 없으므로 0에 가까워진다. `the` 는 0.010, `terrible` 은 2.862 다.

이 둘을 곱한 것이 흔히 쓰는 **TF-IDF** 가중의 한 형태다. 이렇게 채운 행렬은 칸의 1.57%만 0이 아니다.

**쪼개고 나면.** $X \approx U_{64}\Sigma_{64}V_{64}^\top$ 에서 $U_{64}\Sigma_{64}$ 의 행이 **단어 벡터**, $V_{64}\Sigma_{64}$ 의 행이 **문서 벡터**다. 둘 다 64차원이다. 「잠재 의미」라는 이름은 이 64개 축이 사람이 정해 준 적 없는 숨은 주제 축이라는 데서 왔다.

![칸이 둘인 그림. 왼쪽은 얼개로, 8000 단어 곱하기 6000 영화평인 X가 U와 대각 행렬 S와 V 전치의 곱과 같다고 적고, 세 행렬에서 앞 64개 부분만 칠해 그것만 남긴다고 적었다. 단어 벡터는 U_64 S_64의 행, 영화평 벡터는 V_64 S_64의 행이다. 오른쪽은 awful을 검색어로 넣었을 때 64차원 LSA 공간에서 잰 코사인의 분포로, awful이 든 358편은 0.3 언저리에, terrible만 든 280편은 그보다 조금 왼쪽에, 두 단어 다 없는 5362편은 0 언저리에 몰려 있다. 단어를 맞춰 보는 검색이라면 뒤의 두 무리가 정확히 0이라고 적혀 있다](figures/svd_lsa.svg)

### 단어 벡터의 이웃

행렬의 행을 그대로 단어 벡터로 쓸 수도 있다. 단어마다 「어느 글에 얼마나 나왔나」 6,000개의 수다. 그것과 LSA의 64차원을 견주면 이렇다(출력 (라)).

| 단어 | 행 그대로 (6,000차원) | LSA (64차원) |
|---|---|---|
| `terrible` | this, bad, the, a, to, it | **awful, bad, acting, horrible, waste, worst** |
| `funny` | this, a, and, it, the, is | **laugh, comedy, laughed, laughs, skit, fart** |
| `she` | her, in, is, a, to, and | **her, herself, she's, woman, actress, lover** |
| `good` | a, the, and, to, is, of | very, but, really, it, this, the |
| `movie` | this, the, a, i, it, and | this, i, it, just, like, so |

행 그대로는 거의 다 기능어다. 기능어는 IDF가 작아도 **모든 글에 조금씩** 들어 있어서, 어떤 단어의 행과도 코사인이 0이 아니게 된다. LSA는 위 세 단어에서 그럴듯한 이웃을 낸다.

그런데 그 이웃은 [다음 쪽](02_counting.md)이 윈도 네 칸으로 세어 얻는 이웃과 **결이 다르다.** 거기서 `funny` 의 이웃은 `hilarious, amusing` 처럼 **바꿔 넣어도 말이 되는** 단어다. 여기서는 `laugh, comedy, laughed` 처럼 **같은 화제에 나오는** 단어다. 「같은 글에 나온다」는 기준이 「바로 곁에 나온다」보다 넓기 때문이다. 앞엣것은 연관을, 뒤엣것은 대체 가능성을 잡는다. `good` 과 `movie` 는 어느 글에나 나와서 화제가 없으므로 LSA도 기능어를 내놓는다.

### 같은 뜻 다른 말

들머리의 물음이 (마)다. 검색어도 하나의 「글」로 보고 문서 벡터와 같은 자리로 옮긴다. 문서 벡터가 $V\Sigma = X^\top U$ 이므로 검색어 $q$ 는 $q^\top U$ 로 옮겨진다.

| 코사인 평균 | `awful` 이 든 358편 | `terrible` 만 든 280편 | 둘 다 없는 5,362편 |
|---|---|---|---|
| 단어 그대로 | 0.091 | **0.000** | 0.000 |
| LSA | 0.332 | **0.267** | 0.115 |

단어 그대로는 `terrible` 만 쓴 글을 두 단어 다 없는 글과 가르지 못한다. 둘 다 0이다. LSA에서는 그 글들이 `awful` 이 든 글 쪽으로 올라온다. `awful` 과 `terrible` 이 **같은 글들에 함께** 잘 나오므로, 64개 축 가운데 같은 축들에 무게를 싣게 되었기 때문이다. LSA로 가장 가까운 300편 가운데 `awful` 이 든 것은 32.7%뿐인데 부정 평은 **99.0%**다. 단어가 아니라 뜻으로 찾은 셈이다.

### 첫째 축의 함정 — Σ 를 곱하느냐

LSA의 문서 벡터를 그대로 써서 긍정 평과 부정 평을 가르려 하면 이상한 일이 생긴다. 4,000편으로 긍정 평균 방향과 부정 평균 방향을 잡고, 나머지 2,000편을 더 가까운 쪽으로 붙인다. 나누는 시드를 다섯 번 바꿔 잰 것이 (사)다.

| 문서 벡터 | 정확도 평균 | 범위 (시드 5) |
|---|---|---|
| TF-IDF 그대로 (8,000차원) | 0.808 | 0.792~0.829 |
| LSA $V\Sigma$ (64차원) | **0.559** | 0.541~0.571 |
| LSA $V\Sigma$, 첫째 축을 뺀다 | 0.840 | 0.834~0.847 |
| LSA $V$ ($\Sigma$ 를 곱하지 않는다) | **0.858** | 0.851~0.864 |

$V\Sigma$ 는 동전 던지기보다 조금 나은 0.559다. 그런데 **첫째 축 하나를 빼거나 $\Sigma$ 를 곱하지 않으면** 0.84~0.86으로, 8,000차원 그대로보다도 낫다. 범위가 서로 겹치지 않으니 나누는 운이 아니다.

까닭은 (바)에 있다. **첫째 특잇값이 둘째의 12.3배**(제곱으로)이고, 첫째 오른쪽 특이벡터는 **글 길이와 상관이 0.988**이다. 첫째 왼쪽 특이벡터는 8,000칸이 **모두 같은 부호**다. 칸이 모두 0 이상인 행렬의 첫째 특이벡터는 늘 이렇게 한 부호로 잡힌다(페론–프로베니우스 정리). 곧 첫째 축은 「이 글에 단어가 얼마나 많은가」이고, 내용이 아니다. $\Sigma$ 를 곱하면 그 축이 다른 축보다 3.5배 넘게 길어져 코사인을 휘어잡는다. 그래서 긍정·부정이 아니라 긴 글·짧은 글로 갈라 버린다.

**$\Sigma$ 를 어떻게 나눠 붙이느냐는 계산이 아니라 선택이다.** 곱 $U\Sigma V^\top$ 은 어떻게 나눠도 같지만, 한쪽 벡터만 꺼내 쓰는 순간 그 선택이 결과를 바꾼다. [다음 쪽](02_counting.md)이 단어 벡터로 $U\Sigma^{1/2}$ 를 쓰는 것도 같은 선택의 하나다.

> 이 측정에도 한계가 있다. 가르는 법이 평균 방향 둘에 견주는 가장 단순한 것이다. 로지스틱 회귀처럼 축마다 무게를 배우는 분류기라면 첫째 축의 길이를 스스로 줄일 수 있으므로 $V\Sigma$ 와 $V$ 의 차이가 이만큼 크지 않을 것이다. 그것은 재지 않았다.

## 5. 다음 쪽으로

LSA의 「같이 나온다」는 **같은 글 안**이다. 글 한 편은 수백 단어라서 그 기준은 화제를 잡고 뜻의 결은 놓친다. [다음 쪽](02_counting.md)은 기준을 **앞뒤 네 칸 안**으로 좁혀 단어 × 단어 행렬을 세고, 흔한 단어 탓을 PPMI로 걷어 낸 뒤, 이 쪽과 **같은 특잇값 분해**로 줄인다. 2절에서 본 대로 그 행렬은 대칭이어서 특잇값 분해가 사실상 고윳값 분해다.

---

## 6. 코드

```python
"""특잇값 분해 — 고윳값 분해를 정사각이 아닌 행렬로 넓힌다. 그리고 LSA.

영화평 6,000편을 단어 × 문서 행렬(8,000 x 6,000)로 적고 특잇값 분해한다.
이 행렬은 정사각이 아니라 고윳값 분해를 할 수 없다.

  (가) 단어 × 문서 행렬을 TF-IDF 로 채운다
  (나) 특잇값 분해 X = U Σ Vᵀ — 그리고 X Xᵀ, Xᵀ X 의 고윳값과의 관계
  (다) 특잇값 64 개는 행렬의 얼마를 담는가
  (라) LSA: 단어 벡터 UΣ 의 이웃 — 행을 그대로 쓴 것과 견준다
  (마) LSA: awful 로 찾으면 terrible 만 쓴 글도 걸리는가
  (바) 첫째 성분은 무엇을 담는가
  (사) 문서 벡터로 긍정·부정을 가를 때 Σ 를 곱하느냐가 무엇을 바꾸나
  (아) 대칭 행렬(다음 쪽의 PPMI)에서는 특잇값 = |고윳값| 이다

자료는 다음 쪽과 같다 — IMDB 트레인 영화평 긍정 3,000 + 부정 3,000, 어휘 8,000.
실행: 이 쪽의 코드를 파일로 갈무리해 저장소 뿌리에서 돌린다.
"""
import re
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

SEED, VOCAB, K, WINDOW = 42, 8000, 64, 4
ROOT = Path("./data/aclImdb")
TOKEN = re.compile(r"[a-z']+")
WORDS = ("terrible", "funny", "good", "movie", "she")


def header(title):
    print(f"\n=== {title} ===")


def load_docs(n_docs=6000):
    """긍정 3,000 편, 부정 3,000 편. 라벨은 (사)에서만 쓴다."""
    docs, labels = [], []
    for lab, name in ((1, "pos"), (0, "neg")):
        for f in sorted((ROOT / "train" / name).glob("*.txt"))[: n_docs // 2]:
            text = f.read_text(encoding="utf-8").replace("<br />", " ").lower()
            docs.append(TOKEN.findall(text))
            labels.append(lab)
    return docs, np.array(labels)


def ordered_svds(A, k):
    """svds 는 작은 것부터 돌려준다. 큰 것부터로 돌리고, 각 방향의 부호를
    U 열의 합이 양수가 되게 맞춘다(특잇값 분해는 u 와 v 를 함께 뒤집어도 같다)."""
    U, S, Vt = sla.svds(A, k=k, random_state=SEED)
    o = np.argsort(-S)
    U, S, Vt = U[:, o], S[o], Vt[o]
    sgn = np.sign(U.sum(0))
    sgn[sgn == 0] = 1
    return U * sgn, S, Vt * sgn[:, None]


def neighbors(M, w, stoi, words, n=6):
    """M 의 행끼리 코사인으로 가장 가까운 n 개."""
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    s = Mn @ Mn[stoi[w]]
    s[stoi[w]] = -np.inf
    return [words[i] for i in np.argsort(-s)[:n]]


def centroid_accuracy(F, labels, seed):
    """4,000 편으로 긍정·부정의 평균 방향을 잡고, 나머지 2,000 편을 더 가까운 쪽으로."""
    p = np.random.default_rng(seed).permutation(len(labels))
    tr, te = p[:4000], p[4000:]
    F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
    cp = F[tr][labels[tr] == 1].mean(0)
    cn = F[tr][labels[tr] == 0].mean(0)
    pred = (F[te] @ cp > F[te] @ cn).astype(int)
    return (pred == labels[te]).mean()


if __name__ == "__main__":
    docs, labels = load_docs()
    counts = Counter(w for d in docs for w in d)
    words = [w for w, _ in counts.most_common(VOCAB)]
    stoi = {w: i for i, w in enumerate(words)}

    # === (가) 단어 × 문서 행렬 =================================================
    header("(가) 단어 x 문서 행렬 X — 칸은 TF-IDF")
    rows, cols, vals = [], [], []
    for j, d in enumerate(docs):
        for i, n in Counter(stoi[w] for w in d if w in stoi).items():
            rows.append(i); cols.append(j); vals.append(n)
    TF = sp.csr_matrix((np.array(vals, float), (rows, cols)), shape=(VOCAB, len(docs)))
    df = np.asarray((TF > 0).sum(1)).ravel()     # 그 단어가 나온 글의 수
    idf = np.log(len(docs) / df)                  # 어디에나 나오는 단어일수록 작다
    X = sp.diags(idf) @ TF.log1p()                # log(1 + 횟수) x idf
    print(f"  크기 {X.shape}   0 이 아닌 칸 {100 * X.nnz / np.prod(X.shape):.2f}%")
    print(f"  idf   the {idf[stoi['the']]:.3f}   movie {idf[stoi['movie']]:.3f}   "
          f"terrible {idf[stoi['terrible']]:.3f}   awful {idf[stoi['awful']]:.3f}")

    # === (나) 특잇값 분해와 고윳값 분해 ==========================================
    header("(나) X = U Σ Vᵀ — X Xᵀ 와 Xᵀ X 의 고윳값이 σ² 인가")
    U, S, Vt = ordered_svds(X, K)
    v0 = np.ones(VOCAB) / np.sqrt(VOCAB)
    ev_w = np.sort(sla.eigsh(X @ X.T, k=5, which="LA", v0=v0)[0])[::-1]
    v0 = np.ones(len(docs)) / np.sqrt(len(docs))
    ev_d = np.sort(sla.eigsh((X.T @ X), k=5, which="LA", v0=v0)[0])[::-1]
    print("  σ², 1~5            " + "  ".join(f"{v:10.1f}" for v in S[:5] ** 2))
    print("  X Xᵀ 의 고윳값     " + "  ".join(f"{v:10.1f}" for v in ev_w))
    print("  Xᵀ X 의 고윳값     " + "  ".join(f"{v:10.1f}" for v in ev_d))
    print(f"  셋이 상대 1e-8 안에서 같은가   "
          f"{np.allclose(S[:5] ** 2, ev_w, rtol=1e-8) and np.allclose(S[:5] ** 2, ev_d, rtol=1e-8)}")
    print(f"  X Xᵀ 는 {VOCAB} x {VOCAB}, Xᵀ X 는 {len(docs)} x {len(docs)} — 크기가 달라도 0 아닌 고윳값은 같다")

    # === (다) 몇 개만 남기면 ====================================================
    header("(다) 특잇값 64 개가 담는 몫 (Eckart-Young: 랭크 64 의 가장 좋은 근사)")
    total = X.multiply(X).sum()
    print(f"  제곱합에서 담은 몫 {(S ** 2).sum() / total:.1%}")
    print(f"  σ1² / σ2² = {S[0] ** 2 / S[1] ** 2:.1f}")

    # === (라) LSA 의 단어 벡터 =================================================
    header("(라) 단어 벡터 — X 의 행 그대로 (6,000 차원) 대 LSA 의 UΣ (64 차원)")
    Xd = X.toarray()
    W = U * S
    for w in WORDS:
        print(f"  {w:9s} 행 그대로  {', '.join(neighbors(Xd, w, stoi, words))}")
        print(f"  {'':9s} LSA        {', '.join(neighbors(W, w, stoi, words))}")

    # === (마) 같은 뜻 다른 말 ===================================================
    header("(마) awful 한 단어로 글을 찾는다")
    a, b = stoi["awful"], stoi["terrible"]
    has_a = np.asarray((TF[a] > 0).todense()).ravel()
    has_t = np.asarray((TF[b] > 0).todense()).ravel()
    only_t, neither = has_t & ~has_a, ~has_t & ~has_a
    q = np.zeros(VOCAB)
    q[a] = 1.0
    Xn = Xd / (np.linalg.norm(Xd, axis=0, keepdims=True) + 1e-12)
    raw = Xn.T @ q                                # 글마다 awful 칸의 몫
    D = Vt.T * S                                  # 문서 벡터 VΣ = Xᵀ U
    Dn = D / np.linalg.norm(D, axis=1, keepdims=True)
    qh = U.T @ q                                  # 질의도 같은 자리로: qᵀ U
    lsa = Dn @ (qh / np.linalg.norm(qh))
    print(f"  글의 수   awful 이 든 것 {has_a.sum()}   terrible 만 든 것 {only_t.sum()}   "
          f"둘 다 없는 것 {neither.sum()}")
    print(f"  코사인 평균      awful 이 든 글   terrible 만 든 글   둘 다 없는 글")
    print(f"  행 그대로        {raw[has_a].mean():.3f}            {raw[only_t].mean():.3f}"
          f"               {raw[neither].mean():.3f}")
    print(f"  LSA              {lsa[has_a].mean():.3f}            {lsa[only_t].mean():.3f}"
          f"               {lsa[neither].mean():.3f}")
    top = np.argsort(-lsa)[:300]
    print(f"  LSA 로 가장 가까운 300 편   awful 이 든 것 {has_a[top].mean():.1%}   "
          f"부정 평 {1 - labels[top].mean():.1%}")

    # === (바) 첫째 성분 =========================================================
    header("(바) 첫째 성분은 무엇을 담는가")
    length = np.asarray(TF.sum(0)).ravel()
    print(f"  U 첫째 열에서 양수인 칸의 몫 {(U[:, 0] > 0).mean():.3f}")
    print(f"  V 첫째 열과 글 길이의 상관 {np.corrcoef(Vt[0], length)[0, 1]:.3f}")

    # === (사) Σ 를 곱하느냐 =====================================================
    header("(사) 문서 벡터로 긍정·부정 가르기 — 평균 방향에 더 가까운 쪽 (나누기 시드 0~4)")
    feats = (("TF-IDF 그대로 (8,000 차원)", Xd.T),
             ("LSA  VΣ (64 차원)", Vt.T * S),
             ("LSA  VΣ, 첫째 성분을 뺀다", Vt[1:].T * S[1:]),
             ("LSA  V  (Σ 를 곱하지 않는다)", Vt.T))
    for name, F in feats:
        acc = [centroid_accuracy(F, labels, s) for s in range(5)]
        print(f"  {name:28s} 평균 {np.mean(acc):.3f}   범위 {min(acc):.3f}~{max(acc):.3f}")

    # === (아) 대칭 행렬 =========================================================
    header("(아) 대칭 행렬 — 다음 쪽의 PPMI (단어 x 단어, 윈도 4)")
    ids = np.array([stoi[w] for d in docs for w in d if w in stoi])
    r, c = [], []
    for off in range(1, WINDOW + 1):
        r += [ids[:-off], ids[off:]]
        c += [ids[off:], ids[:-off]]
    Cnt = sp.coo_matrix((np.ones(sum(len(x) for x in r)),
                         (np.concatenate(r), np.concatenate(c))),
                        shape=(VOCAB, VOCAB)).tocsr()   # 겹친 칸은 더해진다
    tot = Cnt.sum()
    pa = np.asarray(Cnt.sum(1)).ravel() / tot
    Cnt = Cnt.tocoo()
    pmi = np.log(Cnt.data / tot / (pa[Cnt.row] * pa[Cnt.col]))
    keep = pmi > 0
    M = sp.csr_matrix((pmi[keep], (Cnt.row[keep], Cnt.col[keep])), shape=(VOCAB, VOCAB))
    print(f"  대칭인가 {abs(M - M.T).max() == 0}")
    Um, Sm, Vtm = ordered_svds(M, K)
    lam, Qm = sla.eigsh(M, k=K + 6, which="LM", v0=np.ones(VOCAB) / np.sqrt(VOCAB))
    o = np.argsort(-np.abs(lam))
    lam, Qm = lam[o][:K], Qm[:, o][:, :K]
    print("  σ, 1~5     " + "  ".join(f"{v:7.2f}" for v in Sm[:5]))
    print("  λ, 1~5     " + "  ".join(f"{v:7.2f}" for v in lam[:5]))
    print(f"  64 개 모두 σ = |λ| 인가 (상대 1e-8)   {np.allclose(Sm, np.abs(lam), rtol=1e-8)}")
    neg = np.where(lam < 0)[0]
    flip = np.where(np.sum(Um * Vtm.T, axis=0) < 0)[0]
    print(f"  고윳값이 음수인 자리 {(neg + 1).tolist()}   λ = {np.round(lam[neg], 2).tolist()}")
    print(f"  u 와 v 의 부호가 반대인 자리 {(flip + 1).tolist()}")
    qn = Qm[:, neg[0]]
    qn = qn * np.sign(qn[np.argmax(np.abs(qn))])
    print(f"  {neg[0] + 1}번 방향의 한쪽 끝   {', '.join(words[i] for i in np.argsort(-qn)[:8])}")
    print(f"  {neg[0] + 1}번 방향의 다른 끝   {', '.join(words[i] for i in np.argsort(qn)[:8])}")
    pos_side, neg_side = np.argsort(-qn)[:50], np.argsort(qn)[:50]
    within = M[pos_side][:, pos_side].mean()
    across = M[pos_side][:, neg_side].mean()
    print(f"  두 끝의 50 단어끼리 PPMI 평균   같은 끝 안 {within:.3f}   두 끝 사이 {across:.3f}")
```

**출력:**

```
=== (가) 단어 x 문서 행렬 X — 칸은 TF-IDF ===
  크기 (8000, 6000)   0 이 아닌 칸 1.57%
  idf   the 0.010   movie 0.498   terrible 2.862   awful 2.819

=== (나) X = U Σ Vᵀ — X Xᵀ 와 Xᵀ X 의 고윳값이 σ² 인가 ===
  σ², 1~5              224035.1     18148.3     16446.1     12874.4     12020.4
  X Xᵀ 의 고윳값       224035.1     18148.3     16446.1     12874.4     12020.4
  Xᵀ X 의 고윳값       224035.1     18148.3     16446.1     12874.4     12020.4
  셋이 상대 1e-8 안에서 같은가   True
  X Xᵀ 는 8000 x 8000, Xᵀ X 는 6000 x 6000 — 크기가 달라도 0 아닌 고윳값은 같다

=== (다) 특잇값 64 개가 담는 몫 (Eckart-Young: 랭크 64 의 가장 좋은 근사) ===
  제곱합에서 담은 몫 14.4%
  σ1² / σ2² = 12.3

=== (라) 단어 벡터 — X 의 행 그대로 (6,000 차원) 대 LSA 의 UΣ (64 차원) ===
  terrible  행 그대로  this, bad, the, a, to, it
            LSA        awful, bad, acting, horrible, waste, worst
  funny     행 그대로  this, a, and, it, the, is
            LSA        laugh, comedy, laughed, laughs, skit, fart
  good      행 그대로  a, the, and, to, is, of
            LSA        very, but, really, it, this, the
  movie     행 그대로  this, the, a, i, it, and
            LSA        this, i, it, just, like, so
  she       행 그대로  her, in, is, a, to, and
            LSA        her, herself, she's, woman, actress, lover

=== (마) awful 한 단어로 글을 찾는다 ===
  글의 수   awful 이 든 것 358   terrible 만 든 것 280   둘 다 없는 것 5362
  코사인 평균      awful 이 든 글   terrible 만 든 글   둘 다 없는 글
  행 그대로        0.091            0.000               0.000
  LSA              0.332            0.267               0.115
  LSA 로 가장 가까운 300 편   awful 이 든 것 32.7%   부정 평 99.0%

=== (바) 첫째 성분은 무엇을 담는가 ===
  U 첫째 열에서 양수인 칸의 몫 1.000
  V 첫째 열과 글 길이의 상관 0.988

=== (사) 문서 벡터로 긍정·부정 가르기 — 평균 방향에 더 가까운 쪽 (나누기 시드 0~4) ===
  TF-IDF 그대로 (8,000 차원)        평균 0.808   범위 0.792~0.829
  LSA  VΣ (64 차원)              평균 0.559   범위 0.541~0.571
  LSA  VΣ, 첫째 성분을 뺀다           평균 0.840   범위 0.834~0.847
  LSA  V  (Σ 를 곱하지 않는다)        평균 0.858   범위 0.851~0.864

=== (아) 대칭 행렬 — 다음 쪽의 PPMI (단어 x 단어, 윈도 4) ===
  대칭인가 True
  σ, 1~5      685.22   254.58   209.96   191.98   160.31
  λ, 1~5      685.22  -254.58   209.96   191.98   160.31
  64 개 모두 σ = |λ| 인가 (상대 1e-8)   True
  고윳값이 음수인 자리 [2, 49]   λ = [-254.58, -94.49]
  u 와 v 의 부호가 반대인 자리 [2, 49]
  2번 방향의 한쪽 끝   by, their, his, into, who, from, her, an
  2번 방향의 다른 끝   praise, affected, generations, recognize, relatives, cared, lay, cheated
  두 끝의 50 단어끼리 PPMI 평균   같은 끝 안 0.141   두 끝 사이 0.375
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
$X = \begin{bmatrix} 3 & 0 \\ 0 & -2 \end{bmatrix}$ 의 고윳값 분해와 특잇값 분해를 각각 적어라. 둘은 어디가 같고 어디가 다른가?

</div>

??? success "풀이"
    대각 행렬이므로 고유벡터는 좌표축 $e_1, e_2$ 이고 고윳값은 $3, -2$ 다. $X = Q\Lambda Q^\top$ 에서 $Q = I$, $\Lambda = \mathrm{diag}(3, -2)$ 다.

    특잇값은 0 이상이어야 하므로 $\sigma = 3, 2$ 다. 음수를 어딘가에 실어야 하니 $U = I$, $\Sigma = \mathrm{diag}(3, 2)$, $V = \mathrm{diag}(1, -1)$ 로 둔다. 확인하면 $U\Sigma V^\top = \mathrm{diag}(3, -2) = X$ 다.

    **같은 것**은 축(좌표축 둘)과 크기($|\lambda| = \sigma$)다. **다른 것**은 둘째 축의 부호다. 고윳값 분해는 음수를 $\Lambda$ 에 두고, 특잇값 분해는 $\Sigma$ 를 0 이상으로 지키려고 그 음수를 $V$ 의 한 열로 옮긴다. 본문 2절의 「고윳값이 음수인 방향에서 $v_i = -u_i$」가 이것이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
4절은 검색어 $q$ (단어 8,000칸짜리 벡터)를 $q^\top U$ 로 옮겨 문서 벡터 $V\Sigma$ 와 견주었다. 문서 $d$ 의 열 $x_d$ 를 같은 식 $x_d^\top U$ 로 옮기면 정확히 $V\Sigma$ 의 $d$ 번째 행이 됨을 보여라. 그리고 64개만 남긴 $U_{64}$ 로 옮겨도 여전히 맞는가?

</div>

??? success "풀이"
    $X = U\Sigma V^\top$ 의 양변에 왼쪽에서 $U^\top$ 를 곱하면 $U^\top X = \Sigma V^\top$ 이고, 전치하면 $X^\top U = V\Sigma$ 다. $X^\top U$ 의 $d$ 번째 행은 $x_d^\top U$ 이므로 그것이 $V\Sigma$ 의 $d$ 번째 행이다.

    64개만 남겨도 맞다. $U_{64}$ 는 $U$ 의 앞 64열이므로 $x_d^\top U_{64}$ 는 $x_d^\top U$ 의 앞 64칸, 곧 $V\Sigma$ 의 앞 64칸 $= (V_{64}\Sigma_{64})$ 의 $d$ 번째 행이다. 그래서 **새 글이나 검색어도 행렬을 다시 쪼개지 않고** $U_{64}$ 하나만 곱해 같은 자리로 옮길 수 있다. LSA에서 이것을 「접어 넣기」(fold-in)라 부른다.

    위 프로그램에 이어 붙여 확인한다.

    ```python
    print(np.abs((Xd.T @ U) - Vt.T * S).max() / np.abs(Vt.T * S).max() < 1e-8)
    ```

    ```
    True
    ```

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
4절은 「$\Sigma$ 를 곱하면 첫째 축이 다른 축보다 3.5배 넘게 길어진다」고 했다. 출력의 $\sigma_1^2/\sigma_2^2 = 12.3$ 에서 이 3.5를 끌어내라. 그리고 $\Sigma^{1/2}$ 를 곱한 문서 벡터 $V\Sigma^{1/2}$ 로 같은 정확도를 재면 $V$ 와 $V\Sigma$ 사이 어디쯤에 올지 짐작한 뒤, 재어서 확인하라.

</div>

??? success "풀이"
    $V$ 의 열은 길이가 1이므로 $V\Sigma$ 에서 첫째 축의 크기는 $\sigma_1$, 둘째 축은 $\sigma_2$ 에 비례한다. 비는 $\sigma_1/\sigma_2 = \sqrt{12.3} = 3.5$ 다. 셋째 축부터는 그보다 더 작으므로 「3.5배 넘게」다.

    $\Sigma^{1/2}$ 를 곱하면 그 비가 $\sqrt{3.5} = 1.9$ 배로 줄어든다. 첫째 축이 여전히 가장 길지만 덜 휘어잡으니 $V\Sigma$ 와 $V$ 사이, 아마 $V$ 쪽에 가까운 자리를 짐작할 수 있다.

    ```python
    acc = [centroid_accuracy(Vt.T * np.sqrt(S), labels, s) for s in range(5)]
    print(f"  V Σ^(1/2)   평균 {np.mean(acc):.3f}   범위 {min(acc):.3f}~{max(acc):.3f}")
    ```

    ```
      V Σ^(1/2)   평균 0.795   범위 0.785~0.802
    ```

    **짐작대로 둘 사이이고 $V$ 쪽에 가깝다.** 0.795 는 $V\Sigma$ 의 0.559 보다 $V$ 의 0.858 에 훨씬 가깝다. 그래도 $V$ 보다 0.06 낮고 범위도 겹치지 않는다. 첫째 축이 1.9배만 길어도 그만큼 깎인다는 뜻이다. 다음 쪽이 단어 벡터로 쓰는 것이 바로 이 $\Sigma^{1/2}$ 쪽인데, 거기서는 단어 × 단어 PPMI 행렬이라 첫째 축이 글 길이를 담을 일이 없다. 같은 선택이 행렬에 따라 해가 되기도 하고 안 되기도 한다.
