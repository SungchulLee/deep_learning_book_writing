"""3.2절의 그림을 그리는 스크립트.

「그림으로 보는 AI」 상자의 그림은 사진을 빌려 오지 않고 형식만 빌려 이
책의 그림체로 다시 그린다. 그래야 라이선스 문제가 없고, 이 책의 다른 그림과
결이 같다. 규칙은 CLAUDE.md 의 「그림으로 보는 AI」 항에 적어 두었다.

    python3 make_figures.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from pathlib import Path

plt.rcParams["svg.fonttype"] = "path"      # 글자를 외곽선으로 — 폰트에 기대지 않는다
HERE = Path(__file__).parent
INK = "#2b2b2b"


def save(fig, name):
    fig.savefig(HERE / name, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


# ===========================================================================
def draw_mark_i():
    """Mark I 퍼셉트론의 구조를 다시 그린다.

    왼쪽부터 20x20 광전지(S), 고정된 무작위 배선 플러그보드, 연합 유닛(A),
    모터가 돌리는 가변저항, 응답 유닛(R)이다. 오른쪽에 이 쪽의 모델을 나란히
    두어 「배우는 부분이 어디까지인가」를 견준다.
    """
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(10.6, 4.3))
    ax.set_xlim(0, 13.2); ax.set_ylim(0, 5.6); ax.axis("off")

    # --- S: 광전지 20x20 --------------------------------------------------------
    for i in range(8):
        for j in range(8):
            ax.add_patch(Circle((0.45 + 0.26 * j, 1.5 + 0.26 * i), 0.075,
                                fill=False, ec=INK, lw=0.6))
    ax.add_patch(Rectangle((0.28, 1.32), 2.1, 2.1, fill=False, ec=INK, lw=1.2))
    ax.text(1.33, 3.62, "400 photocells", ha="center", fontsize=9, color=INK)
    ax.text(1.33, 1.02, "20 x 20 retina", ha="center", fontsize=8.5, color=INK)
    ax.text(1.33, 0.62, "S-units", ha="center", fontsize=8.5, color=INK, style="italic")

    # --- 플러그보드: 고정된 무작위 배선 ----------------------------------------
    ax.add_patch(Rectangle((3.3, 1.32), 1.5, 2.1, fill=False, ec=INK, lw=1.2))
    for _ in range(16):
        y0, y1 = rng.uniform(1.45, 3.3, 2)
        ax.plot([3.35, 4.75], [y0, y1], color=INK, lw=0.5, alpha=0.75)
    ax.text(4.05, 3.62, "plugboard", ha="center", fontsize=9, color=INK)
    ax.text(4.05, 1.02, "random, fixed", ha="center", fontsize=8.5, color=INK)
    ax.text(4.05, 0.62, "never adjusted", ha="center", fontsize=8.5, color=INK, style="italic")
    ax.add_patch(FancyArrowPatch((2.45, 2.37), (3.25, 2.37), arrowstyle="-|>",
                                 mutation_scale=10, color=INK, lw=1.0))

    # --- A 유닛 -----------------------------------------------------------------
    for i in range(9):
        ax.add_patch(Circle((5.75, 1.5 + 0.24 * i), 0.09, fill=False, ec=INK, lw=0.8))
    ax.text(5.75, 3.62, "512 A-units", ha="center", fontsize=9, color=INK)

    # --- 배우는 층: 모터가 돌리는 가변저항 --------------------------------------
    for i in range(9):
        for k in range(3):
            ax.plot([5.9, 8.9], [1.5 + 0.24 * i, 1.75 + 0.6 * k],
                    color=INK, lw=0.35, alpha=0.5)
    kx, ky = 7.4, 4.35
    ax.add_patch(Circle((kx, ky), 0.33, fill=False, ec=INK, lw=1.3))
    ax.plot([kx, kx + 0.23], [ky, ky + 0.23], color=INK, lw=1.5)
    ax.add_patch(Rectangle((kx - 0.26, ky - 0.95), 0.52, 0.42, fill=False, ec=INK, lw=1.1))
    ax.text(kx, ky - 0.76, "M", ha="center", va="center", fontsize=8, color=INK)
    ax.plot([kx, kx], [ky - 0.53, ky - 0.33], color=INK, lw=1.0)
    ax.text(kx + 0.55, ky + 0.05, "one weight = one potentiometer,", ha="left",
            fontsize=8.5, color=INK)
    ax.text(kx + 0.55, ky - 0.35, "turned by a motor", ha="left", fontsize=8.5, color=INK)
    ax.text(7.4, 1.02, "512 x 8 = 4,096 weights", ha="center", fontsize=8.5, color=INK)
    ax.text(7.4, 0.62, "the only part that learns", ha="center", fontsize=8.5,
            color=INK, style="italic")

    # --- R 유닛 -----------------------------------------------------------------
    for k in range(3):
        ax.add_patch(Circle((9.05, 1.75 + 0.6 * k), 0.12, fill=False, ec=INK, lw=0.9))
    ax.text(9.05, 3.62, "8 R-units", ha="center", fontsize=9, color=INK)

    # --- 오늘의 쪽과 견주기 ------------------------------------------------------
    ax.plot([10.1, 10.1], [0.95, 4.0], color=INK, lw=0.6, ls=(0, (3, 3)))
    ax.text(11.6, 3.62, "this page", ha="center", fontsize=9, color=INK)
    rows = [("input", "784"), ("output", "10"), ("learned", "7,850")]
    for n, (a, b) in enumerate(rows):
        y = 2.9 - 0.52 * n
        ax.text(10.5, y, a, ha="left", fontsize=8.5, color=INK)
        ax.text(12.75, y, b, ha="right", fontsize=8.5, color=INK)
    ax.text(11.6, 1.02, "all of it learns", ha="center", fontsize=8.5, color=INK)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    return fig


def draw_softmax_temperature():
    """3.2 5절 - 온도가 분포를 어떻게 바꾸는가.

    로짓 하나를 고정해 두고 T 만 바꾼다. T 가 작으면 argmax 로 몰리고
    크면 1/10 로 평평해진다. 엔트로피를 함께 적어 「딱딱해진다·
    부드러워진다」가 수로 무엇인지 보이게 했다.
    """
    y = np.array([3.2, 1.1, 0.4, 2.7, -0.3, 0.9, 1.8, -1.2, 0.2, 1.5])
    Ts = [0.25, 1.0, 4.0]
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.9), sharey=True)
    for ax, T in zip(axes, Ts):
        e = np.exp((y - y.max()) / T); pr = e / e.sum()
        H = -(pr * np.log(pr)).sum(); top = int(pr.argmax())
        ax.bar(range(10), pr, 0.62,
               color=["#b91c1c" if i == top else "#9ca3af" for i in range(10)])
        ax.axhline(0.1, color="#1d4ed8", ls=":", lw=1.3)
        ax.set_xticks(range(10)); ax.set_xticklabels(range(10), fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"T = {T}", fontsize=12.5)
        ax.text(4.5, 0.92, f"largest prob {pr.max():.3f}\nentropy {H:.3f} nat",
                ha="center", fontsize=10.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("probability", fontsize=11)
    axes[0].text(9.4, 0.12, "1/10", fontsize=9.5, color="#1d4ed8", ha="right")
    axes[1].set_xlabel("class", fontsize=11)
    fig.suptitle("same logits  y = (3.2, 1.1, 0.4, 2.7, -0.3, 0.9, 1.8, "
                 "-1.2, 0.2, 1.5)   -   only T changes", fontsize=11.5)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.76, bottom=0.14,
                        wspace=0.08)
    for T in Ts:
        e = np.exp((y - y.max()) / T); pr = e / e.sum()
        print(f"     T={T:<5} max {pr.max():.4f}  H {-(pr*np.log(pr)).sum():.4f}")
    return fig


def draw_neg_log_curve():
    """3.2 4절 - 교차 엔트로피가 왜 틀린 확신을 크게 벌하는가.

    로스는 정답 클래스에 준 확률 하나의 -log 다. p=1 이면 0 이고
    p 가 0 으로 갈수록 끝없이 커진다. log 10 = 2.3026 이 「열 클래스에
    고르게 1/10 씩 주었을 때」의 값이라 가로선으로 함께 긋는다.
    """
    pr = np.linspace(0.005, 1.0, 600)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(pr, -np.log(pr), color="#b91c1c", lw=2.4)
    for q, lab, c in ((1.0, "p = 1\nloss 0", "#15803d"),
                      (0.1, "p = 1/10\nloss 2.3026", "#1d4ed8"),
                      (0.01, "p = 1/100\nloss 4.6052", "#b91c1c")):
        ax.plot(q, -np.log(q), "o", ms=10, color=c, zorder=3)
        dx = (-14, 6) if q == 1.0 else (14, 6)
        ha = "right" if q == 1.0 else "left"
        ax.annotate(lab, (q, -np.log(q)), dx, textcoords="offset points",
                    fontsize=10.5, color=c, ha=ha)
    ax.axhline(np.log(10), color="#1d4ed8", ls=":", lw=1.3)
    ax.text(1.04, np.log(10) + 0.14, "log 10 = 2.3026, the loss before "
            "training starts", ha="right", fontsize=10, color="#1d4ed8")
    ax.set_xlim(0, 1.08); ax.set_ylim(0, 5.3)
    ax.set_xlabel("probability the model gave the correct class", fontsize=11)
    ax.set_ylabel("loss  =  -log p", fontsize=11)
    ax.set_title("being wrong is cheap; being confidently wrong is not",
                 fontsize=11.5)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ===========================================================================
if __name__ == "__main__":
    save(draw_mark_i(), "mark_i_perceptron.svg")
    save(draw_softmax_temperature(), "softmax_temperature.svg")
    save(draw_neg_log_curve(), "neg_log_curve.svg")
