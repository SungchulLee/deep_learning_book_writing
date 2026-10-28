"""단순 선형 회귀 — nn.Module 과 옵티마이저로.

앞 두 쪽은 파라메터가 하나였다. 여기서는 기울기와 절편 둘을 배우고,
`nn.Linear` 와 `torch.optim` 에 맡긴다. 자료는 y = 3x + 2 + 잡음 이다.

재어 볼 것은 넷이다.

- 경사 하강법이 닿는 곳이 정규방정식의 해와 같은가
- 그 해가 자료를 만든 규칙(3, 2)과 같은가 — 다르다. 얼마나 다른가
- 지표 넷(MSE, RMSE, MAE, R^2)이 각각 무엇을 말하는가
- 잔차 그림이 무엇을 잡아내는가

실행: python 03_simple_linear_regression.py
"""
import torch
import torch.nn as nn

torch.manual_seed(42)

N = 100
TRUE_W, TRUE_B, NOISE = 3.0, 2.0, 1.0

X = torch.linspace(0, 10, N).unsqueeze(1)
Y = TRUE_W * X + TRUE_B + NOISE * torch.randn(N, 1)


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. 자료와 모델
# ========================================================================

def setup():
    header("1. 자료와 모델")
    print(f"  표본 {N}개,  생성 규칙 y = {TRUE_W}x + {TRUE_B} + N(0, {NOISE}^2)")
    print(f"  X 모양 {tuple(X.shape)},  Y 모양 {tuple(Y.shape)}")
    print(f"  X 범위 {X.min():.2f} ~ {X.max():.2f}")

    torch.manual_seed(0)
    model = nn.Linear(1, 1)
    print(f"  nn.Linear(1, 1) 의 파라메터")
    for name, p in model.named_parameters():
        print(f"    {name:<8} 모양 {tuple(p.shape)}  처음 값 "
              f"{p.detach().flatten().tolist()}")
    print("  weight 가 기울기, bias 가 절편이다. 둘 다 배운다")
    return model


# ========================================================================
# 2. 정답을 먼저 구해 둔다
# ========================================================================

def closed_form():
    header("2. 정규방정식 — 경사 하강법이 닿아야 할 곳")
    A = torch.cat([X, torch.ones(N, 1)], 1)
    sol = torch.linalg.lstsq(A, Y).solution.flatten()
    best = ((A @ sol.unsqueeze(1) - Y) ** 2).mean().item()
    print(f"  최소제곱 해   w = {sol[0]:.6f},  b = {sol[1]:.6f}")
    print(f"  그때의 MSE    {best:.6f}")
    print(f"  자료를 만든 값  w = {TRUE_W},  b = {TRUE_B}")
    print(f"  차이          dw = {sol[0]-TRUE_W:+.6f},  "
          f"db = {sol[1]-TRUE_B:+.6f}")
    print("  같지 않다. 잡음이 섞인 100개로 추정했기 때문이다")
    print("  경사 하강법이 찾아야 하는 것은 (3, 2) 가 아니라 이 해다")
    return sol, best


# ========================================================================
# 3. 학습
# ========================================================================

def train(model, epochs=1000, lr=0.01):
    header("3. 학습 (전체 배치, lr = 0.01)")
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    print("  에폭    로스(갱신 전)    w          b")
    for e in range(epochs):
        loss = nn.functional.mse_loss(model(X), Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if e % 200 == 0 or e == epochs - 1:
            print(f"  {e:>5}   {loss.item():12.6f}  "
                  f"{model.weight.item():9.6f}  {model.bias.item():9.6f}")
    print("  로스는 갱신 전, w 와 b 는 갱신 후 값이다 — 같은 줄의 세 수가")
    print("  같은 시점의 것이 아니다")
    return model


def compare(model, sol, best):
    header("4. 닿은 곳과 정규방정식의 해")
    w, b = model.weight.item(), model.bias.item()
    mse = nn.functional.mse_loss(model(X), Y).item()
    print(f"  경사 하강법   w = {w:.6f},  b = {b:.6f},  MSE = {mse:.6f}")
    print(f"  정규방정식    w = {sol[0]:.6f},  b = {sol[1]:.6f},  "
          f"MSE = {best:.6f}")
    print(f"  차이          dw = {w-sol[0].item():+.2e},  "
          f"db = {b-sol[1].item():+.2e},  dMSE = {mse-best:+.2e}")
    print("  b 쪽이 w 쪽보다 한참 덜 수렴했다. 조건수 때문이다")

    # 조건수를 직접 구해 본다
    A = torch.cat([X, torch.ones(N, 1)], 1)
    H = 2 * (A.T @ A) / N
    ev = torch.linalg.eigvalsh(H)
    print(f"  헤세의 고윳값 {ev[0]:.4f}, {ev[1]:.4f}  "
          f"조건수 {ev[1]/ev[0]:.1f}")
    print(f"  수렴률 (k-1)/(k+1) = "
          f"{(ev[1]/ev[0]-1)/(ev[1]/ev[0]+1):.6f} — 한 걸음에 이만큼만 준다")
    print(f"  X 를 중심화하면 (x - {X.mean():.2f}) 어떻게 되는지 본다")
    Ac = torch.cat([X - X.mean(), torch.ones(N, 1)], 1)
    Hc = 2 * (Ac.T @ Ac) / N
    evc = torch.linalg.eigvalsh(Hc)
    print(f"  중심화 뒤 고윳값 {evc[0]:.4f}, {evc[1]:.4f}  "
          f"조건수 {evc[1]/evc[0]:.1f}")
    print("  x 와 1 의 상관이 사라져 조건수가 크게 준다")


# ========================================================================
# 5. 지표 넷
# ========================================================================

def metrics(model):
    header("5. 지표 넷이 각각 말하는 것")
    with torch.no_grad():
        pred = model(X)
        res = (pred - Y).flatten()
    mse = (res ** 2).mean().item()
    rmse = mse ** 0.5
    mae = res.abs().mean().item()
    ss_res = (res ** 2).sum().item()
    ss_tot = ((Y - Y.mean()) ** 2).sum().item()
    r2 = 1 - ss_res / ss_tot
    print(f"  MSE   {mse:9.6f}   단위가 y 의 제곱이다")
    print(f"  RMSE  {rmse:9.6f}   y 와 같은 단위 — '평균 이만큼 틀린다'")
    print(f"  MAE   {mae:9.6f}   역시 y 의 단위. 큰 오차에 덜 민감하다")
    print(f"  R^2   {r2:9.6f}   단위가 없다. 1 이면 완벽")
    print(f"  자료의 잡음 표준편차가 {NOISE} 였다. RMSE 가 그 언저리다")
    print(f"  RMSE >= MAE 는 언제나 참이다 (여기서는 {rmse/mae:.3f} 배)")
    print(f"  R^2 는 '평균만 찍는 모델' 을 기준으로 삼는다")
    print(f"    평균만 찍을 때의 MSE = {ss_tot/N:.6f}")
    print(f"    우리 모델의 MSE      = {mse:.6f}")
    print(f"    1 - 비 = {1 - mse/(ss_tot/N):.6f}  — R^2 와 같다")

    print("\n  R^2 가 큰 것에 속지 말 것 — x 범위를 좁히면")
    for lo, hi in ((0, 10), (0, 3), (4, 5)):
        m = (X.flatten() >= lo) & (X.flatten() <= hi)
        with torch.no_grad():
            r = (model(X[m]) - Y[m]).flatten()
        tot = ((Y[m] - Y[m].mean()) ** 2).sum().item()
        print(f"    x in [{lo},{hi}]  표본 {int(m.sum()):>3}개  "
              f"RMSE {(r**2).mean().item()**0.5:.4f}  "
              f"R^2 {1 - (r**2).sum().item()/tot:+.4f}")
    print("  RMSE 는 거의 그대로인데 R^2 는 무너진다. 모델이 나빠진 것이")
    print("  아니라 견주는 기준(평균만 찍기)이 좋아져서다")


# ========================================================================
# 6. 잔차가 무엇을 잡아내는가
# ========================================================================

def residuals():
    header("6. 잔차 — 직선으로 충분한지 가려낸다")
    A = torch.cat([X, torch.ones(N, 1)], 1)

    def fit_and_report(Yd, label):
        sol = torch.linalg.lstsq(A, Yd).solution
        res = (A @ sol - Yd).flatten()
        # 구간별 잔차 평균 — 구조가 있으면 여기서 드러난다
        bins = []
        for i in range(5):
            m = (X.flatten() >= i * 2) & (X.flatten() < (i + 1) * 2 + 1e-9)
            bins.append(res[m].mean().item())
        print(f"  {label}")
        print(f"    RMSE {(res**2).mean().item()**0.5:.4f}   "
              f"R^2 {1 - (res**2).sum().item()/((Yd-Yd.mean())**2).sum().item():.4f}")
        print("    구간별 잔차 평균 " +
              "  ".join(f"{v:+.3f}" for v in bins))
        return bins

    b1 = fit_and_report(Y, "(가) 정말 직선인 자료")
    torch.manual_seed(1)
    Y2 = 0.3 * X ** 2 + NOISE * torch.randn(N, 1)
    b2 = fit_and_report(Y2, "(나) 사실은 이차인 자료")

    print(f"  (가) 의 구간별 잔차는 부호가 뒤섞여 있다 "
          f"(최대 {max(abs(v) for v in b1):.3f})")
    print(f"  (나) 는 -, +, +, +, - 로 뚜렷한 곡선 꼴이다 "
          f"(최대 {max(abs(v) for v in b2):.3f})")
    print("  (나) 의 R^2 도 높게 나온다. R^2 만 보면 못 잡아낸다")
    print("  잔차를 x 에 대해 그려 보아야 보인다 — 그것이 잔차 그림이다")


def main():
    model = setup()
    sol, best = closed_form()
    train(model)
    compare(model, sol, best)
    metrics(model)
    residuals()


if __name__ == "__main__":
    main()
