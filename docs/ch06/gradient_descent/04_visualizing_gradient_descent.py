"""경사 하강법을 그림으로 보기 — 무엇을 어떤 축에 그릴 것인가.

경사 하강법을 그린 그림은 흔하지만, 대개 '로스가 내려간다' 는 것만
보여 주고 끝난다. 그것은 이미 아는 사실이다.

이 쪽은 **축을 어떻게 고르면 무엇이 보이는가** 를 다룬다.

- 로그 축으로 바꾸면 수렴률이 직선의 기울기로 드러난다
- 선형 축은 첫 몇 걸음 뒤를 전부 가린다
- 파라메터 공간의 경로는 조건수를 눈으로 보여 준다
- 로스 곡선만으로는 발산과 느린 수렴을 가릴 수 없다

그림은 figures/ 에 SVG 로 갈무리한다.

실행: python 04_visualizing_gradient_descent.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"

import matplotlib.pyplot as plt
import numpy as np
import torch

FIG = Path(__file__).resolve().parent / "figures"
INK = "#2f3337"
COLORS = ["#2f6f4e", "#8a4b2a", "#8a2a4b", "#2a4b8a"]


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. 1차원 — 네 러닝레이트
# ========================================================================

def run_1d(lr, steps=30, w0=7.0):
    """L(w) = (w-3)^2,  dL/dw = 2(w-3)."""
    w, traj = w0, [w0]
    for _ in range(steps):
        w = w - lr * 2 * (w - 3)
        traj.append(w)
    return np.array(traj)


def one_dim():
    header("1. L(w) = (w-3)^2 에서 네 러닝레이트")
    print("  오차가 매 걸음 (1 - 2*eta) 배가 된다")
    print("  eta    1-2*eta   30 걸음 뒤 |w-3|      판정")
    runs = {}
    for lr in (0.1, 0.5, 0.9, 1.1):
        t = run_1d(lr)
        runs[lr] = t
        err = abs(t[-1] - 3)
        shown = f"{err:.6e}" if err < 1e30 else "넘침"
        verdict = ("한 걸음에 끝" if abs(1 - 2 * lr) < 1e-12
                   else "수렴" if abs(1 - 2 * lr) < 1 else "발산")
        print(f"  {lr:<6} {1-2*lr:+7.2f}   {shown:>16}   {verdict}")

    print("\n  0.1 과 0.9 는 |1-2*eta| 가 둘 다 0.8 이다")
    print(f"  30 걸음 뒤 오차:  0.1 -> {abs(runs[0.1][-1]-3):.6e}")
    print(f"                    0.9 -> {abs(runs[0.9][-1]-3):.6e}")
    print("  같은 수다. 가는 모양만 다르고 속도는 같다")
    print("  그림에서 이것은 로그 축으로 보아야 드러난다 —")
    print("  선형 축에서는 둘 다 '금세 0 에 붙는 선' 으로만 보인다")
    return runs


def figure_1d(runs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 3.5))

    # eta = 0.1 과 0.9 의 |오차| 는 정확히 같아서 로그 판에서 겹친다.
    # 뒤에 그리는 쪽을 점선 + 빈 네모로 두어야 둘 다 보인다.
    styles = {0.1: ("o-", 3, 1.6), 0.5: ("o-", 3, 1.1),
              0.9: ("s--", 4.5, 1.1), 1.1: ("o-", 3, 1.1)}
    for (lr, t), c in zip(runs.items(), COLORS):
        st, ms, lw = styles[lr]
        ax1.plot(t[:16], st, color=c, ms=ms, lw=lw, mfc="none",
                 label=f"eta = {lr}")
        err = np.abs(t - 3)
        err[err < 1e-16] = 1e-16            # 로그 축에 0 을 못 찍는다
        ax2.semilogy(err, st, color=c, ms=ms, lw=lw, mfc="none",
                     label=f"eta = {lr}")

    ax1.axhline(3, color=INK, ls="--", lw=0.8, alpha=0.6)
    ax1.set_ylim(-3, 12)
    ax1.set_xlabel("step", fontsize=9)
    ax1.set_ylabel("w", fontsize=9)
    ax1.set_title("linear axis: w per step", fontsize=9.5, weight="bold",
                  color=INK, pad=7)
    ax1.legend(fontsize=7.5, frameon=False, ncol=2)

    ax2.set_ylim(1e-10, 1e3)
    ax2.set_xlabel("step", fontsize=9)
    ax2.set_ylabel("|w - 3|  (log)", fontsize=9)
    ax2.set_title("log axis: the rate becomes a slope", fontsize=9.5,
                  weight="bold", color=INK, pad=7)
    ax2.legend(fontsize=7.5, frameon=False, ncol=2)

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)
    fig.tight_layout()
    fig.savefig(FIG / "gd_1d.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print("  figures/gd_1d.svg 를 썼다")


# ========================================================================
# 2. 로그 축의 기울기가 수렴률이다
# ========================================================================

def log_slope(runs):
    header("2. 로그 축의 기울기를 재어 본다")
    print("  |w_t - 3| = |1-2*eta|^t |w_0 - 3| 이므로")
    print("  log10|w_t - 3| 는 t 에 대해 직선이고 기울기가 log10|1-2*eta| 다")
    print("  eta    잰 기울기      log10|1-2*eta|    차이")
    for lr in (0.1, 0.5, 0.9):
        if abs(1 - 2 * lr) < 1e-12:
            print(f"  {lr:<6} (한 걸음에 끝나 직선이 없다)")
            continue
        t = run_1d(lr, steps=30)
        err = np.abs(t - 3)
        k = np.arange(5, 26)
        slope = np.polyfit(k, np.log10(err[k]), 1)[0]
        want = np.log10(abs(1 - 2 * lr))
        print(f"  {lr:<6} {slope:+12.6f}  {want:+14.6f}  "
              f"{abs(slope-want):.2e}")
    print("  두 기울기가 소수점 아래 여섯 자리까지 같다. 그림의 로그 판에서")
    print("  0.1 의 선과 0.9 의 선은 나란한 정도가 아니라 **포개진다** —")
    print("  그래서 0.9 를 점선으로 그려야 둘 다 보인다")
    print("  선형 판에서는 둘이 전혀 다르게 생겼다. 하나는 매끄럽게 내려오고")
    print("  하나는 좌우로 튕긴다. 같은 속도라는 것을 거기서는 알 수 없다")


# ========================================================================
# 3. 두 파라메터 — 경로가 조건수를 보여 준다
# ========================================================================

def two_dim():
    header("3. 파라메터 공간의 경로 (선형 회귀)")
    torch.manual_seed(42)
    N = 50
    x = torch.linspace(0, 10, N)
    y = 3 * x + 2 + torch.randn(N)

    def path(center, lr, steps=60, start=(0.0, 0.0)):
        xx = x - x.mean() if center else x
        w, b = start
        out = [(w, b)]
        for _ in range(steps):
            pred = w * xx + b
            gw = float((2 * xx * (pred - y)).mean())
            gb = float((2 * (pred - y)).mean())
            w, b = w - lr * gw, b - lr * gb
            out.append((w, b))
        return np.array(out), xx

    ratios, kappas = {}, {}
    for center, lab in ((False, "그대로  "), (True, "중심화  ")):
        A = torch.stack([x - x.mean() if center else x, torch.ones(N)], 1)
        sol = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.flatten()
        H = 2 * (A.T @ A) / N
        ev = torch.linalg.eigvalsh(H)
        p, _ = path(center, 0.01)
        d0 = np.linalg.norm(p[0] - sol.numpy())
        d = np.linalg.norm(p[-1] - sol.numpy())
        ratios[center], kappas[center] = d / d0, float(ev[1] / ev[0])
        print(f"  {lab} 조건수 {ev[1]/ev[0]:7.1f}   최적해 "
              f"({sol[0]:.4f}, {sol[1]:.4f})")
        print(f"           출발 거리 {d0:7.4f} -> 60 걸음 뒤 {d:7.4f}  "
              f"(남은 비율 {d/d0:.4f})")
    print("  중심화하면 절편의 뜻이 바뀌어(x=0 이 아니라 x=xbar 에서의 값)")
    print("  출발 거리가 달라진다. 그러니 남은 비율로 견주어야 한다")
    print(f"  {ratios[False]:.4f} 대 {ratios[True]:.4f} — 중심화 쪽이 더 많이")
    print(f"  줄였다. 조건수 {kappas[False]:.1f} 대 {kappas[True]:.1f} 이 그 까닭이다")
    print("  다만 60 걸음에서의 차이는 크지 않다. 조건수가 정하는 것은")
    print("  끝까지 가는 데 드는 걸음 수이고, 그 차이는 더 뒤에 벌어진다")
    print("  그림의 두 판이 비슷해 보이는 것도 그 때문이다. 둘 다 같은")
    print("  병을 앓는다 — w 쪽으로는 몇 걸음에 가 놓고 b 쪽으로는 60")
    print("  걸음을 써도 못 닿는다. 빠른 방향과 느린 방향이 따로 있다")
    print("  이것이 파라메터 공간의 경로를 그리는 까닭이다. 로스 곡선은")
    print("  두 방향을 하나의 수로 뭉개 놓아 이 구조를 보여 주지 못한다")

    return path, x, y, N


def figure_2d(path, x, y, N):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for ax, center, title in (
            (axes[0], False, "x as is  (condition number 137)"),
            (axes[1], True, "x centered  (condition number 8.7)")):
        p, xx = path(center, 0.01)
        A = torch.stack([xx, torch.ones(N)], 1)
        sol = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.flatten().numpy()

        lo_w, hi_w = min(p[:, 0].min(), sol[0]) - 0.6, max(p[:, 0].max(), sol[0]) + 0.6
        lo_b, hi_b = min(p[:, 1].min(), sol[1]) - 1.2, max(p[:, 1].max(), sol[1]) + 1.2
        W, B = np.meshgrid(np.linspace(lo_w, hi_w, 200),
                           np.linspace(lo_b, hi_b, 200))
        xn, yn = xx.numpy(), y.numpy()
        Z = np.mean((W[..., None] * xn + B[..., None] - yn) ** 2, axis=-1)
        # 등고선 높이를 경로가 지나간 로스에서 뽑는다. 그러지 않으면
        # 선들이 일이 벌어지지 않는 자리에 몰린다.
        along = np.mean((p[:, 0:1] * xn + p[:, 1:2] - yn) ** 2, axis=1)
        lv = np.unique(np.geomspace(max(along.min(), Z.min() + 1e-6),
                                    along.max(), 12))
        ax.contour(W, B, Z, levels=lv,
                   colors=INK, linewidths=0.6, alpha=0.55, zorder=1)
        ax.plot(p[:, 0], p[:, 1], "o-", color=COLORS[1], ms=2.6, lw=1.0,
                zorder=3)
        ax.plot([sol[0]], [sol[1]], "*", color=INK, ms=11, zorder=4)
        ax.set_xlabel("w (slope)", fontsize=9)
        ax.set_ylabel("b (intercept)", fontsize=9)
        ax.set_title(title, fontsize=9.5, weight="bold", color=INK, pad=7)
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)
    fig.tight_layout()
    fig.savefig(FIG / "gd_2d.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print("  figures/gd_2d.svg 를 썼다")


# ========================================================================
# 4. 로스 곡선이 가리는 것
# ========================================================================

def what_the_curve_hides():
    header("4. 로스 곡선만 보면 가려지는 것")
    print("  세 설정을 로스 곡선으로만 보면 모두 '잘 내려간다' 로 보인다")
    print("  설정              10 걸음 로스   30 걸음 로스   100 걸음 로스")
    rows = {}
    for lr, lab, tag in ((0.02, "느림 (eta=0.02) ", "eta = 0.02  (slow)"),
                         (0.5, "딱 맞음(eta=0.5) ", "eta = 0.5   (exact)"),
                         (0.97, "거의 경계(0.97) ", "eta = 0.97  (near bound)")):
        t = run_1d(lr, steps=100)
        ls = (t - 3) ** 2
        rows[tag] = ls          # 그림의 범례로 쓰므로 아스키라야 한다
        print(f"  {lab}  {ls[10]:12.6f}  {ls[30]:12.6f}  {ls[100]:13.6e}")
    print("  10 걸음에서 느린 쪽 7.07 과 거의 경계 쪽 4.64 는 둘 다 '아직")
    print("  내려가는 중' 으로 보인다. 30 걸음에서 1.38 과 0.39 다")
    print("  100 걸음에서 4.55e-03 과 6.76e-05 로 67 배 갈린다")
    print("  선형 축에서는 이 셋이 모두 바닥에 붙어 구별되지 않는다 —")
    print("  0.001 과 0.00001 이 눈으로 같은 자리다")
    print("  로그 축으로 바꾸면 세 직선의 기울기가 서로 다르게 드러난다")
    return rows


def figure_curve(rows):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 3.4))
    for (lab, ls), c in zip(rows.items(), COLORS):
        ax1.plot(ls, color=c, lw=1.4, label=lab)
        v = np.maximum(ls, 1e-16)
        ax2.semilogy(v, color=c, lw=1.4, label=lab)
    ax1.set_ylim(-1, 20)
    ax1.set_xlabel("step", fontsize=9)
    ax1.set_ylabel("loss", fontsize=9)
    ax1.set_title("linear axis: all three look done", fontsize=9.5,
                  weight="bold", color=INK, pad=7)
    ax2.set_xlabel("step", fontsize=9)
    ax2.set_ylabel("loss (log)", fontsize=9)
    ax2.set_title("log axis: they are nothing alike", fontsize=9.5,
                  weight="bold", color=INK, pad=7)
    for ax in (ax1, ax2):
        ax.legend(fontsize=7.5, frameon=False)
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)
    fig.tight_layout()
    fig.savefig(FIG / "gd_curves.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print("  figures/gd_curves.svg 를 썼다")


def main():
    FIG.mkdir(exist_ok=True)
    runs = one_dim()
    figure_1d(runs)
    log_slope(runs)
    path, x, y, N = two_dim()
    figure_2d(path, x, y, N)
    rows = what_the_curve_hides()
    figure_curve(rows)


if __name__ == "__main__":
    main()
