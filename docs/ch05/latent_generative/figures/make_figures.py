"""
================================================================================
make_figures.py - 5.2절 「그림으로 보는 AI」 상자의 그림을 만든다
================================================================================

만드는 그림:
    reparam_variance.svg  같은 그래디언트를 두 가지 방법으로 어림할 때 한 번
                          재서 나오는 값이 얼마나 널뛰는가. 왼쪽은 잠재 칸이
                          64개일 때의 분포 둘, 오른쪽은 잠재 칸 수에 따라
                          널뛰는 폭이 어떻게 벌어지는지다.
                          (01_vae.md 의 상자)

    beta_rulers.svg       왼쪽은 이 쪽이 실제로 잰 세 자가 beta 를 따라
                          어떻게 움직이는가(모두 02_beta_vae.md 의 출력),
                          오른쪽은 beta 를 만든 사람들이 보려던 자의 꼴.
                          오른쪽은 수가 없는 schematic 이라고 그림 안에 적는다.
                          (02_beta_vae.md 의 상자)

수의 출처:
    reparam_variance.svg  모두 이 파일이 돌려서 얻은 값이다. MNIST 와는
                          아무 상관이 없는 장난감 계산이며, 그 사실을 그림
                          안에도 적는다. 쓰는 세팅은
                              q(z) = N(mu, I_d),  f(z) = ||z - a||^2
                          이고 mu - a 는 길이 1의 첫 단위벡터로 고정한다.
                          재는 것은 mu 의 첫 좌표에 대한 그래디언트이며
                          참값은 2 다. 두 어림법 모두 편향이 없다.

    beta_rulers.svg       왼쪽 세 줄의 수는 02_beta_vae.md 의 표에서 옮긴
                          것이고 여기서 다시 재지 않았다. 오른쪽에는 수가
                          없다.

그림 규칙(CLAUDE.md):
    - SVG로 저장하고 svg.fonttype='path'로 글자를 외곽선으로 만든다
    - transparent=True
    - 그림 안의 글자는 모두 ASCII (기본 글꼴에 한글이 없다)

실행:
    python make_figures.py
================================================================================
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["svg.fonttype"] = "path"

C_REPARAM = "tab:green"
C_SCORE = "tab:red"


# === 장난감 계산: 같은 그래디언트를 두 가지로 어림한다 ========================
def gradient_samples(d, n_trial, rng):
    """mu 의 첫 좌표에 대한 그래디언트를, 한 번 뽑아 어림한 값들.

    q(z) = N(mu, I_d), f(z) = ||z - a||^2, c = mu - a = (1, 0, ..., 0).
    참값은 d_mu_1 E_q[f] = 2 * c_1 = 2 이며 두 어림법 모두 이 값을 평균으로
    가진다. 갈리는 것은 한 번 재서 나오는 값이 그 둘레에서 얼마나 널뛰는가다.

    (1) 무작위성을 옆으로 밀어내는 쪽:  z = mu + eps 로 적고 f 를 직접 미분
    (2) 뽑기를 그대로 두는 쪽:          f(z) * d_mu log q(z) 를 쓴다
    """
    eps = rng.standard_normal((n_trial, d))
    c = np.zeros(d)
    c[0] = 1.0

    r = c + eps                       # z - a
    reparam = 2.0 * r[:, 0]
    score = (r ** 2).sum(1) * eps[:, 0]
    return reparam, score


def fig_reparam_variance():
    """같은 값을 두 방법으로 어림했을 때 한 번 재서 나오는 값의 폭.

    왼쪽은 잠재 칸 64개에서 여러 번 뽑아 평균 낼 때 값이 가라앉는 모습,
    오른쪽은 칸 수에 따른 한 번 측정의 표준편차다. 이 쪽이 쓰는 두 잠재
    차원(2와 64)에 눈금을 둔다.
    """
    rng = np.random.default_rng(42)
    n_trial = 200_000
    dims = [2, 4, 8, 16, 32, 64]

    sd_rep, sd_sco = [], []
    for d in dims:
        a, b = gradient_samples(d, n_trial, rng)
        sd_rep.append(a.std())
        sd_sco.append(b.std())

    # 왼쪽 칸: 64차원에서 1번, 2번, ... 뽑아 평균 낸 값
    rep64, sco64 = gradient_samples(64, n_trial, np.random.default_rng(7))
    n = np.arange(1, n_trial + 1)
    run_rep = np.cumsum(rep64) / n
    run_sco = np.cumsum(sco64) / n

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.6, 4.4))

    # --- 왼쪽: 몇 번 뽑아야 값이 가라앉는가 ---------------------------------
    axL.axhline(2.0, color="0.3", lw=1.2, ls="--", zorder=4)
    axL.annotate("true value = 2", xy=(6000, 2.0), xytext=(400, 13.0),
                 fontsize=9, color="0.3",
                 arrowprops=dict(arrowstyle="->", color="0.55", lw=1.0))
    axL.plot(n, run_sco, color=C_SCORE, lw=1.3, zorder=2,
             label="keep the draw as it is")
    axL.plot(n, run_rep, color=C_REPARAM, lw=1.6, zorder=3,
             label="move the randomness aside")
    axL.set_xscale("log")
    axL.set_xlim(1, n_trial)
    axL.set_ylim(-14, 18)
    axL.set_xlabel("how many draws were averaged", fontsize=9.5)
    axL.set_ylabel("the measured gradient", fontsize=9.5)
    axL.legend(fontsize=9, loc="lower right", frameon=False)
    axL.grid(alpha=0.22, lw=0.6)
    axL.set_axisbelow(True)
    axL.spines[["top", "right"]].set_visible(False)
    axL.set_title("latent slots = 64", fontsize=10.5, pad=8)

    # --- 오른쪽: 칸이 늘어나면 한 번 재는 값의 폭이 어떻게 되는가 ----------
    axR.plot(dims, sd_sco, "-o", color=C_SCORE, lw=2.0, ms=6, zorder=3,
             label="keep the draw as it is")
    axR.plot(dims, sd_rep, "-o", color=C_REPARAM, lw=2.0, ms=6, zorder=3,
             label="move the randomness aside")
    for d, lo, hi in zip(dims, sd_rep, sd_sco):
        if d in (2, 64):
            axR.annotate("", xy=(d, lo * 1.06), xytext=(d, hi * 0.94),
                         arrowprops=dict(arrowstyle="<->", color="0.45",
                                         lw=1.1), zorder=4)
            side = -9 if d == dims[-1] else 9
            axR.annotate(f"{hi / lo:.0f} times wider",
                         xy=(d, (lo * hi) ** 0.5), xytext=(side, 0),
                         textcoords="offset points", fontsize=9, color="0.3",
                         va="center",
                         ha="right" if side < 0 else "left")
    axR.set_xscale("log", base=2)
    axR.set_yscale("log")
    axR.set_xticks(dims)
    axR.set_xticklabels([str(d) for d in dims])
    axR.set_yticks([2, 5, 10, 20, 50, 100])
    axR.set_yticklabels(["2", "5", "10", "20", "50", "100"])
    axR.set_xlim(1.75, 85)
    axR.set_ylim(1.4, 130)
    axR.set_xlabel("number of latent slots (this page uses 2 and 64)",
                   fontsize=9.5)
    axR.set_ylabel("spread of a single measurement\n(standard deviation)",
                   fontsize=9.5)
    axR.legend(fontsize=9, loc="upper left", frameon=False)
    axR.grid(alpha=0.22, which="major", lw=0.6)
    axR.set_axisbelow(True)
    axR.spines[["top", "right"]].set_visible(False)
    axR.set_title("how the wobble grows", fontsize=10.5, pad=8)

    fig.text(0.5, 0.035,
             "toy computation, not MNIST:  q = N(mu, I),  f(z) = ||z - a||^2,  "
             f"{n_trial:,} draws.  Both ways are unbiased.",
             ha="center", fontsize=8.5, color="0.45")

    fig.subplots_adjust(left=0.085, right=0.985, top=0.90, bottom=0.20,
                        wspace=0.30)
    fig.savefig("reparam_variance.svg", transparent=True)
    plt.close(fig)
    print("wrote reparam_variance.svg")
    # 폭이 r배 넓으면 같은 안정도를 얻는 데 r^2 배를 뽑아야 한다
    # (여러 번 뽑아 평균 내면 폭이 1/sqrt(n) 로 줄기 때문이다)
    for d, lo, hi in zip(dims, sd_rep, sd_sco):
        r = hi / lo
        print(f"  d={d:3d}  aside {lo:7.3f}   as-is {hi:8.3f}   "
              f"ratio {r:6.1f}   draws needed x{r ** 2:,.0f}")


# === beta 를 따라 움직이는 자들 ==============================================
def fig_beta_rulers():
    """왼쪽은 이 쪽이 잰 세 자, 오른쪽은 이 쪽에 없는 자.

    왼쪽 수는 모두 02_beta_vae.md 의 표에서 옮긴 것이다. 단위가 서로 다르므로
    복원 로스는 따로 칸을 두어 위에 쌓는다. 오른쪽은 수가 없는 schematic 이며,
    그렇게 그림 안에 적는다.
    """
    x = [0, 1, 2, 3]
    recon = [0.0546, 0.0577, 0.0750, 0.1351]
    balance = [0.541, 0.784, 0.877, 0.950]
    variety = [0.642, 0.647, 0.631, 0.575]
    xlabels = ["0\n(no KL)", "0.5", "1.0", "4.0"]

    fig = plt.figure(figsize=(10.8, 4.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0],
                          height_ratios=[1.0, 1.55], hspace=0.16, wspace=0.34,
                          left=0.085, right=0.985, top=0.90, bottom=0.14)
    axT = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[1, 0], sharex=axT)
    axR = fig.add_subplot(gs[:, 1])

    # --- 왼쪽 위: 복원 로스 -------------------------------------------------
    axT.plot(x, recon, marker="s", ls="--", color="tab:orange", lw=2.0, ms=6,
             zorder=3)
    for xi, v in zip(x, recon):
        axT.annotate(f"{v:.4f}", (xi, v), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=8.5,
                     color="tab:orange")
    axT.set_ylim(0.035, 0.175)
    axT.set_ylabel("reconstruction\nerror", fontsize=9.5)
    axT.set_title("measured on this page", fontsize=10.5, pad=8)
    axT.tick_params(labelbottom=False)
    axT.spines[["top", "right"]].set_visible(False)
    axT.grid(axis="y", alpha=0.22, lw=0.6)
    axT.set_axisbelow(True)
    axT.text(0.98, 0.10, "lower is better", transform=axT.transAxes,
             ha="right", fontsize=8.5, color="0.45")

    # --- 왼쪽 아래: 두 자 ---------------------------------------------------
    axB.plot(x, balance, "-o", color="tab:blue", lw=2.0, ms=6, zorder=3,
             label="class balance")
    axB.plot(x, variety, "-o", color="tab:purple", lw=2.0, ms=6, zorder=3,
             label="within-class variety")
    # beta=0 에서는 두 자의 자리가 뒤집혀 있어 라벨도 뒤집어 단다
    for xi, v in zip(x, balance):
        axB.annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                     xytext=(0, -16 if xi == 0 else 9), ha="center",
                     fontsize=8.5, color="tab:blue")
    for xi, v in zip(x, variety):
        axB.annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                     xytext=(0, 9 if xi == 0 else -16), ha="center",
                     fontsize=8.5, color="tab:purple")
    axB.set_ylim(0.47, 1.04)
    axB.set_xticks(x)
    axB.set_xticklabels(xlabels, fontsize=9.5)
    axB.set_xlim(-0.32, 3.32)
    axB.set_xlabel("beta", fontsize=10)
    axB.legend(fontsize=8.8, loc="center right", frameon=False)
    axB.spines[["top", "right"]].set_visible(False)
    axB.grid(axis="y", alpha=0.22, lw=0.6)
    axB.set_axisbelow(True)

    # --- 오른쪽: 재지 않았다 -------------------------------------------------
    t = np.linspace(0.0, 1.0, 300)
    curve = 1.0 - 2.6 * (t - 0.5) ** 2
    axR.plot(t, curve, color="0.35", lw=2.2, ls=(0, (5, 3)), zorder=3)
    axR.plot([0.5], [1.0], "o", color="0.35", ms=7, zorder=4)
    axR.text(0.02, 0.40, "too small:\nfactors stay mixed", fontsize=9,
             color="0.35", ha="left", va="top")
    axR.text(0.93, 0.40, "too large:\nfactors are lost again", fontsize=9,
             color="0.35", ha="right", va="top")
    axR.annotate("best beta sits here", xy=(0.5, 1.0), xytext=(0.30, 1.20),
                 fontsize=9, color="0.35",
                 arrowprops=dict(arrowstyle="->", color="0.55", lw=1.1))
    axR.set_xlim(-0.03, 1.03)
    axR.set_ylim(0.02, 1.33)
    axR.set_xticks([])
    axR.set_yticks([])
    axR.set_xlabel("beta", fontsize=10)
    axR.set_ylabel("one slot = one thing in the picture", fontsize=9.5)
    axR.set_title("the ruler this page does not have", fontsize=10.5, pad=8)
    axR.spines[["top", "right"]].set_visible(False)
    axR.text(0.5, 0.015,
             "schematic - shape only, no recorded numbers.\n"
             "Needs pictures whose ingredients are known in advance;\n"
             "MNIST digits do not come with any.",
             transform=axR.transAxes, ha="center", va="bottom", fontsize=8.8,
             color="0.45")

    fig.savefig("beta_rulers.svg", transparent=True)
    plt.close(fig)
    print("wrote beta_rulers.svg")


if __name__ == "__main__":
    fig_reparam_variance()
    fig_beta_rulers()
