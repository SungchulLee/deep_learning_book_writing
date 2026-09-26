"""비지도학습의 세 정리를 수로 확인한다.

증명은 "왜 그런가"를 말해 주지만 "정말 그런가"는 재어 보아야 안다.
셋 다 몇 줄로 확인된다.

    정리 1  K-평균의 J는 걸음마다 줄고 유한 걸음에 멈춘다
    정리 2  분산을 가장 크게 하는 방향은 v_1 이고 그 값은 lambda_1
    정리 3  J_k는 k에 대해 단조 감소하므로 J로는 k를 고를 수 없다
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# 자료는 내내 같은 것을 쓴다. 갈래가 셋인 2차원 점 300개.
# 씨앗을 박아 두었으므로 아래 수는 돌릴 때마다 같다.
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0,
                  n_features=2, random_state=0)


def wcss(X, labels, centers):
    """군집 내 제곱합 J = sum_j sum_{x in C_j} ||x - mu_j||^2."""
    return float(sum(((X[labels == j] - centers[j]) ** 2).sum()
                     for j in range(len(centers))))


def kmeans(X, k, seed, trace=False):
    """K-평균을 손으로 돌린다. trace=True면 걸음마다의 J를 함께 돌려준다.

    sklearn을 쓰지 않는 까닭이 있다. 정리 1은 **배정 단계와 갱신 단계**에
    대한 이야기이므로, 그 두 단계가 코드에 그대로 보여야 확인이 된다.
    """
    rng = np.random.RandomState(seed)
    centers = X[rng.choice(len(X), k, replace=False)]   # 첫 중심은 점 몇 개를 그냥 고른다
    history = []

    for step in range(100):
        # --- 배정 단계: 점마다 가장 가까운 중심을 고른다 ---
        #   (n, 1, 2) - (1, k, 2) -> (n, k, 2) -> 제곱합 -> 거리 (n, k)
        d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(2)
        labels = d.argmin(1)
        if trace:
            history.append(wcss(X, labels, centers))    # 배정 직후의 J

        # --- 갱신 단계: 중심을 제 군집의 평균으로 옮긴다 ---
        new_centers = np.array([X[labels == j].mean(0) if (labels == j).any()
                                else centers[j] for j in range(k)])
        if trace:
            history.append(wcss(X, labels, new_centers))  # 갱신 직후의 J

        if np.allclose(new_centers, centers):           # 더 움직이지 않으면 멈춘다
            break
        centers = new_centers

    return labels, centers, history, step + 1


# ==========================================================================
# 정리 1 — J는 걸음마다 줄고, 유한 걸음에 멈춘다
# ==========================================================================
print("=" * 62)
print("정리 1  K-평균의 J는 단조로 줄고 유한 걸음에 멈춘다")
print("=" * 62)

_, _, hist, n_step = kmeans(X, k=3, seed=0, trace=True)
print(f"  {n_step}걸음에 멈췄다. 걸음마다의 J (배정 뒤 / 갱신 뒤):")
for i in range(0, len(hist), 2):
    print(f"    {i//2 + 1}걸음   배정 {hist[i]:8.2f}   갱신 {hist[i+1]:8.2f}")

# 단조성을 눈이 아니라 수로 확인한다
diffs = np.diff(hist)
print(f"\n  J가 늘어난 적이 있는가: {bool((diffs > 1e-9).any())}")
print(f"  가장 많이 늘어난 값     : {diffs.max():.2e}  (0 이하라야 한다)")


# ==========================================================================
# 정리 2 — 분산을 가장 크게 하는 방향은 v_1 이고 그 값은 lambda_1
# ==========================================================================
print("\n" + "=" * 62)
print("정리 2  max w'Sw = lambda_1,  argmax = v_1")
print("=" * 62)

Xc = X - X.mean(0)                       # 중심을 옮긴다
S = (Xc.T @ Xc) / len(Xc)                # 표본 공분산
lam, V = np.linalg.eigh(S)               # eigh는 오름차순으로 준다
lam, V = lam[::-1], V[:, ::-1]           # 큰 것부터로 뒤집는다

print(f"  고윳값        lambda_1 = {lam[0]:.4f},  lambda_2 = {lam[1]:.4f}")
print(f"  v_1' S v_1             = {V[:, 0] @ S @ V[:, 0]:.4f}   (lambda_1과 같아야 한다)")

# 무작위 방향 2만 개를 던져 lambda_1을 넘는 것이 있는지 본다.
# 증명이 옳다면 하나도 없어야 한다.
rng = np.random.RandomState(0)
W = rng.randn(20000, 2)
W /= np.linalg.norm(W, axis=1, keepdims=True)     # 길이를 1로 맞춘다
vals = np.einsum("ij,jk,ik->i", W, S, W)          # 방향마다 w'Sw
print(f"  무작위 방향 2만 개 가운데 가장 큰 w'Sw = {vals.max():.4f}")
print(f"  lambda_1을 넘은 방향의 수              = {int((vals > lam[0] + 1e-9).sum())}")

# 가장 좋았던 무작위 방향이 v_1과 얼마나 나란한가 (|cos| = 1이면 같은 방향)
best = W[vals.argmax()]
print(f"  가장 좋았던 방향과 v_1의 |cos|         = {abs(best @ V[:, 0]):.4f}")


# ==========================================================================
# 정리 3 — J_k는 k에 대해 단조 감소한다. 그래서 J로는 k를 고를 수 없다
# ==========================================================================
print("\n" + "=" * 62)
print("정리 3  J_k는 k에 대해 단조 감소 — J로는 k를 고를 수 없다")
print("=" * 62)
print(f"  {'k':>3} | {'J_k':>9} | {'실루엣':>7}")
print("  " + "-" * 26)

prev = None
for k in range(1, 11):
    # 첫 중심을 열 번 달리 잡아 가장 좋은 것을 쓴다. 정리 1이 국소 최적만
    # 보장하므로, J_k를 '전역 최솟값'에 가깝게 두려면 이렇게 해야 한다.
    best = min((kmeans(X, k, s) for s in range(10)),
               key=lambda r: wcss(X, r[0], r[1]))
    labels, centers = best[0], best[1]
    J = wcss(X, labels, centers)

    # 실루엣은 군집이 둘 이상이라야 정의된다
    sil = f"{silhouette_score(X, labels):7.4f}" if k >= 2 else "      —"

    # 정리 3이 맞다면 J가 늘어나는 줄은 없어야 한다
    mark = "" if prev is None or J <= prev + 1e-9 else "   <- 늘었다!"
    print(f"  {k:>3} | {J:9.2f} | {sil}{mark}")
    prev = J

print("\n  J는 k를 키울수록 줄기만 한다 -> 가장 작은 J를 고르면 언제나 k=n이다.")
print("  실루엣은 그렇지 않다 -> 꼭대기가 있어 고를 수 있다.")
