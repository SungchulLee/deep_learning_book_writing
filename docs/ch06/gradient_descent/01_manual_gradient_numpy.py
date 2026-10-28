"""넘파이로 직접 하는 경사 하강법 — 자동 미분 없이.

자동 미분이 대신해 주는 일이 정확히 무엇인지 알려면 한 번은 손으로
해 보아야 한다. 절편 없는 선형 모델 하나를 놓고 네 단계를 직접 적는다.

    순전파 -> 로스 -> 경사(손으로 민 식) -> 갱신

자료는 x = [1,2,3,4,5], y = 2x 이므로 답이 w = 2 임을 미리 알고 있다.
답을 아는 문제로 시작하는 까닭은, 모르는 문제에서 틀렸을 때 그것이
알고리즘 탓인지 문제 탓인지 가릴 수 없기 때문이다.

실행: python 01_manual_gradient_numpy.py
"""
import numpy as np

X = np.array([1, 2, 3, 4, 5], dtype=np.float64)
Y = 2 * X                       # 참 관계: y = 2x, 곧 w* = 2


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. 세 조각을 손으로 적는다
# ========================================================================

def forward(x, w):
    """순전파 — 예측."""
    return w * x


def loss(y_true, y_pred):
    """평균제곱오차 L(w) = (1/N) sum (y_pred - y_true)^2."""
    return np.mean((y_pred - y_true) ** 2)


def gradient(x, y_true, y_pred):
    """dL/dw 를 손으로 민 식.

    L(w)  = (1/N) sum (w x_i - y_i)^2
    dL/dw = (1/N) sum 2 (w x_i - y_i) x_i
          = (2/N) sum x_i (y_pred_i - y_i)
    """
    return np.mean(2 * x * (y_pred - y_true))


# ========================================================================
# 2. 손으로 민 식이 맞는지 먼저 확인한다
# ========================================================================

def check_gradient():
    header("1. 손으로 민 식 vs 수치 미분")
    print("  믿고 쓰기 전에 차분으로 견준다. 이것이 경사 검사다")
    print("  w        손으로 민 식      중앙 차분         차이")
    h = 1e-6
    for w in (0.0, 0.5, 2.0, 3.5):
        analytic = gradient(X, Y, forward(X, w))
        numeric = (loss(Y, forward(X, w + h)) - loss(Y, forward(X, w - h))) / (2 * h)
        print(f"  {w:<8.1f} {analytic:+14.8f}  {numeric:+14.8f}  "
              f"{abs(analytic - numeric):.2e}")
    print("  w = 2 에서 경사가 0 이다. 거기가 최솟값이다")


# ========================================================================
# 3. 이 자료의 로스는 사실 포물선 하나다
# ========================================================================

def closed_form():
    header("2. 이 자료에서 로스의 꼴")
    m = np.mean(X ** 2)
    print(f"  mean(x^2) = {m:.1f}")
    print(f"  y = 2x 이므로 L(w) = mean(x^2) (w-2)^2 = {m:.0f}(w-2)^2")
    print(f"  따라서 L''(w) = 2 * {m:.0f} = {2*m:.0f} 이고, 이것이 곡률이다")
    print(f"  수렴 조건 0 < eta < 2/L'' = {2/(2*m):.6f}")
    print(f"  한 걸음에 끝내는 eta = 1/L'' = {1/(2*m):.6f}")

    print("\n  식이 맞는지 몇 점에서 확인한다")
    print("  w      직접 잰 로스      {:.0f}(w-2)^2".format(m))
    for w in (0.0, 1.0, 2.0, 3.0):
        print(f"  {w:<6.1f} {loss(Y, forward(X, w)):14.6f}  "
              f"{m * (w - 2) ** 2:14.6f}")


# ========================================================================
# 4. 학습 루프
# ========================================================================

def train(lr=0.01, steps=20, w0=0.0, verbose=True):
    w = w0
    ws, ls = [w], []
    if verbose:
        print("  걸음   w            로스          경사        (1-22*eta)^t")
    for t in range(steps):
        y_pred = forward(X, w)
        L = loss(Y, y_pred)
        g = gradient(X, Y, y_pred)
        w = w - lr * g
        ws.append(w)
        ls.append(L)
        if verbose and (t % 4 == 0 or t == steps - 1):
            # 오차가 매 걸음 (1 - 22*eta) 배가 된다는 예측
            pred = (w0 - 2) * (1 - 2 * np.mean(X ** 2) * lr) ** (t + 1) + 2
            print(f"  {t+1:>4}   {w:11.6f}  {L:11.6f}  {g:+10.4f}   "
                  f"{pred:11.6f}")
    return w, ws, ls


def training_loop():
    header("3. 학습 (lr = 0.01, 20 걸음, w0 = 0)")
    w, ws, ls = train()
    print(f"  마지막 w = {w:.6f},  목표 2.0,  오차 {abs(w-2):.6f}")
    print("  마지막 열은 손으로 푼 식 w_t - 2 = (1 - 22*eta)^t (w_0 - 2) 다")
    print("  루프가 찍는 값과 자리까지 같다 — 식과 코드가 같은 것을 한다")

    print("\n  배운 모델로 예측해 본다")
    print("  x     예측        참값      오차")
    for x, y in zip(X, Y):
        p = forward(x, w)
        print(f"  {x:<5.0f} {p:9.4f}  {y:8.1f}  {abs(p-y):.4f}")
    print("  자료에 없던 x 로도")
    for x in (6.0, 7.0, 8.0):
        print(f"  {x:<5.0f} {forward(x, w):9.4f}  {2*x:8.1f}  "
              f"{abs(forward(x, w)-2*x):.4f}")
    print("  x 가 클수록 오차가 크다. w 가 아직 2 에 못 미쳐서이고,")
    print("  그 모자람이 x 에 곱해지기 때문이다")


# ========================================================================
# 5. 러닝레이트를 바꿔 본다
# ========================================================================

def sweep_lr():
    header("4. 러닝레이트를 바꾸면 (20 걸음)")
    limit = 1 / np.mean(X ** 2)
    print(f"  2절에서 구한 경계는 eta < {limit:.6f} 였다")
    print("  eta         20 걸음 뒤 w        오차          1-22*eta")
    for lr in (0.001, 0.01, 1 / 22, 0.08, limit, 0.1):
        w, _, _ = train(lr=lr, steps=20, verbose=False)
        shown = f"{w:16.6f}" if abs(w) < 1e12 else f"{w:16.3e}"
        err = abs(w - 2)
        err_s = f"{err:.6f}" if err < 1e12 else f"{err:.3e}"
        print(f"  {lr:<11.6f} {shown}  {err_s:>12}  {1-22*lr:+.4f}")
    print(f"  eta = 1/22 = {1/22:.6f} 에서 한 걸음에 끝난다 (1-22*eta = 0)")
    print(f"  eta = {limit:.6f} 가 경계다. 거기서는 0 과 4 를 오가며 제자리다")
    print("  그 너머는 발산한다")


# ========================================================================
# 6. 자료에 잡음이 있으면
# ========================================================================

def with_noise():
    header("5. 잡음이 섞인 자료라면")
    rng = np.random.default_rng(0)
    print("  y 에 표준편차 1.0 인 잡음을 더하고 2000번 돌린다")
    print("  (자료가 바뀔 때마다 답도 바뀌므로 여러 번 재어야 한다)")
    found = []
    for _ in range(2000):
        Yn = Y + rng.normal(0, 1.0, size=Y.shape)
        # 최소제곱 해는 손으로도 나온다: w = sum(x y) / sum(x^2)
        found.append(float(np.sum(X * Yn) / np.sum(X ** 2)))
    found = np.array(found)
    print(f"  최소제곱 해 2000개:  평균 {found.mean():.4f}  "
          f"퍼짐 {found.min():.4f}~{found.max():.4f}")
    print(f"  표준편차 {found.std(ddof=1):.4f}")
    print("  이론값 sigma / sqrt(sum(x^2)) = "
          f"{1.0/np.sqrt(np.sum(X**2)):.4f}")
    print("  경사 하강법이 찾아 주는 것은 '참값 2' 가 아니라")
    print("  '이 자료의 최소제곱 해' 다. 자료가 흔들리면 그 답도 흔들린다")


def main():
    check_gradient()
    closed_form()
    training_loop()
    sweep_lr()
    with_noise()


if __name__ == "__main__":
    main()
