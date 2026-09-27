"""'가진 자료를 분석하라'의 세 정리를 수로 확인한다.

    정리 1  무작위 배정이 있으면 평균 차이가 곧 인과 효과다
    정리 2  표준오차는 sigma/sqrt(n) — 오차를 반으로 줄이려면 자료가 네 배
    정리 3  자기 자료로 매긴 점수는 늘 후하다

특히 문제 1("관측 자료가 수백만 개면 실험 수백 개보다 나은가")의 답을
수로 보인다. 자료를 아무리 키워도 **편향은 줄지 않는다.**
"""

import numpy as np

rng = np.random.RandomState(0)


# ==========================================================================
# 정리 1 — 무작위 배정이 없으면 n을 키워도 틀린 값으로 수렴한다
# ==========================================================================
print("=" * 66)
print("정리 1  무작위 배정의 값어치 (보기 1의 약 이야기를 그대로 돌린다)")
print("=" * 66)

# 보기 1의 상황을 그대로 만든다.
#   위중도가 처치와 결과를 함께 민다 = 교란요인
#   의사는 위중한 환자에게 약을 준다
#   그런데 약은 사실 **이롭다** (사망 확률을 0.10 낮춘다)
TRUE_ATE = -0.10          # 약이 사망 확률을 10%포인트 낮춘다 (이롭다)


def make_patients(n, randomized, rng):
    """환자 n명을 만든다. randomized=True면 동전으로 약을 나눠 준다."""
    severe = rng.rand(n) < 0.5                      # 절반이 위중하다

    # 약을 주지 않았을 때의 사망 확률 — 위중하면 높다
    p0 = np.where(severe, 0.80, 0.20)
    p1 = p0 + TRUE_ATE                              # 약을 주면 0.10 낮아진다

    if randomized:
        T = rng.rand(n) < 0.5                       # 동전 던지기 — 위중도와 무관
    else:
        # 관측 자료: 의사가 위중한 환자에게 약을 몰아 준다
        T = rng.rand(n) < np.where(severe, 0.90, 0.10)

    # 실제로 관측되는 결과 하나만 남는다 (반사실은 볼 수 없다)
    died = rng.rand(n) < np.where(T, p1, p0)
    return T, died


print(f"  참 인과 효과 ATE = {TRUE_ATE:+.2f}  (음수 = 약이 이롭다)\n")
print(f"  {'n':>9} | {'관측 자료 추정':>14} | {'무작위 시험 추정':>16}")
print("  " + "-" * 46)

for n in (1_000, 10_000, 100_000, 1_000_000):
    T, d = make_patients(n, randomized=False, rng=rng)
    obs = d[T].mean() - d[~T].mean()                # 단순 평균 차이
    T, d = make_patients(n, randomized=True, rng=rng)
    rct = d[T].mean() - d[~T].mean()
    print(f"  {n:>9,} | {obs:>+14.4f} | {rct:>+16.4f}")

print(f"\n  관측 자료 쪽은 n을 천 배로 키워도 **양수**에 머문다.")
print(f"  곧 약이 해롭다고 말한다 — 참값은 {TRUE_ATE:+.2f}인데도.")
print("  무작위 시험 쪽은 n이 작아도 참값 언저리에 있고, 커질수록 붙는다.")
print("\n  이것이 문제 1의 답이다. 자료를 키우면 **분산**은 줄지만")
print("  **편향**은 그대로다. 틀린 값으로 더 정확히 수렴할 뿐이다.")


# ==========================================================================
# 정리 2 — 표준오차는 sigma/sqrt(n)
# ==========================================================================
print("\n" + "=" * 66)
print("정리 2  표준오차 = sigma/sqrt(n)")
print("=" * 66)

SIGMA = 3.0
N_REPEAT = 20000                       # 같은 실험을 2만 번 되풀이해 흩어짐을 잰다

print(f"  참 분포의 sigma = {SIGMA}\n")
print(f"  {'n':>6} | {'잰 표준오차':>11} | {'sigma/sqrt(n)':>13} | {'앞 줄 대비':>9}")
print("  " + "-" * 48)

prev = None
for n in (100, 400, 1600, 6400):
    # 크기 n인 표본을 2만 번 뽑아 평균을 내고, 그 평균들의 흩어짐을 잰다
    samples = SIGMA * rng.randn(N_REPEAT, n)      # (2만, n) — 평균 0, 표준편차 SIGMA
    means = samples.mean(1)                       # 실험마다의 표본평균
    measured = means.std(ddof=1)                  # 표본평균들이 얼마나 흩어지는가
    closed = SIGMA / np.sqrt(n)
    ratio = "" if prev is None else f"{measured/prev:>9.3f}"
    print(f"  {n:>6} | {measured:>11.4f} | {closed:>13.4f} | {ratio:>9}")
    prev = measured

print("\n  잰 값과 닫힌 꼴이 소수점 셋째 자리까지 맞는다.")
print("  그리고 n을 네 배로 할 때마다 오차가 0.5배가 된다 — 정리 2가 말한 대로다.")


# ==========================================================================
# 정리 3 — 자기 자료로 매긴 점수는 늘 후하다
# ==========================================================================
print("\n" + "=" * 66)
print("정리 3  훈련오차의 낙관성")
print("=" * 66)

# 신호가 **하나도 없는** 자료를 만든다. y는 x와 아무 관계가 없다.
# 그러므로 정직한 모형이라면 설명력이 0이어야 한다.
N, P = 100, 80


def r2(y, pred):
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


Xtr, ytr = rng.randn(N, P), rng.randn(N)
Xte, yte = rng.randn(N, P), rng.randn(N)

# 최소제곱으로 맞춘다
w, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
print(f"  자료: 표본 {N}개, 특성 {P}개, 그리고 **신호는 0**")
print(f"    학습 자료 R^2 = {r2(ytr, Xtr @ w):+.4f}   <- 아무 관계도 없는데 높다")
print(f"    시험 자료 R^2 = {r2(yte, Xte @ w):+.4f}   <- 참값인 0 언저리, 오히려 음수")

# --- 문제 2: 시험 집합에서 고른 뒤 그 점수를 알리면? ---
# 한 번만 해 보면 운에 휘둘린다. 고르는 절차 전체를 300번 되풀이해 평균낸다.
print("\n  문제 2: 모형 50개를 시험 집합에서 견주어 가장 좋은 것을 고르면?")

N_MODEL, N_TRIAL, K = 50, 300, 5
picked_val, picked_test = [], []

for _ in range(N_TRIAL):
    # 자료를 세 벌로 나눈다. 셋 다 신호가 없으므로 참 R^2는 모두 0이다.
    Xa, ya = rng.randn(N, P), rng.randn(N)     # 학습
    Xv, yv = rng.randn(N, P), rng.randn(N)     # 고르는 데 쓰는 집합
    Xt, yt = rng.randn(N, P), rng.randn(N)     # 아껴 둔 집합

    val, test = [], []
    for _m in range(N_MODEL):
        cols = rng.choice(P, K, replace=False)             # 특성 5개를 아무렇게나 고른다
        w, *_ = np.linalg.lstsq(Xa[:, cols], ya, rcond=None)
        val.append(r2(yv, Xv[:, cols] @ w))
        test.append(r2(yt, Xt[:, cols] @ w))

    b = int(np.argmax(val))                     # 고르는 집합에서 가장 좋은 것
    picked_val.append(val[b])
    picked_test.append(test[b])                 # 같은 모형을 아껴 둔 집합에서 다시 잰다

print(f"    (같은 절차를 {N_TRIAL}번 되풀이한 평균)")
print(f"    참값                          = {0.0:+.4f}   (신호가 없으므로)")
print(f"    고른 모형의 '고르는 집합' 점수 = {np.mean(picked_val):+.4f}   <- 골랐으니 후하다")
print(f"    같은 모형의 아껴 둔 집합 점수  = {np.mean(picked_test):+.4f}   <- 정직한 값")
print(f"    고르기가 부풀린 몫             = {np.mean(picked_val) - np.mean(picked_test):+.4f}")
print("\n  시험 집합으로 **고르는 순간** 그 집합도 학습 자료가 된다.")
print("  그래서 고른 뒤의 점수는 다시 후해지고, 정직한 값은 또 한 벌의 자료에서만 나온다.")
