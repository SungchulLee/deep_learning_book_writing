"""19.3 학습 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 읽지 않으므로 어디서 돌려도 된다.

실행: python docs/ch18/pgm_learning/figures/make_figures.py
"""
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
INK = "#2f3337"
ACCENT = "#8a4b2a"
MUTED = "#2d6a8f"

# 로빈슨(1973) 의 점화식이 맞게 셈하는지 대조할 알려진 값 (OEIS A003024)
KNOWN = {1: 1, 2: 3, 3: 25, 4: 543, 5: 29281, 6: 3781503}


def _dag_counts(n_max):
    """라벨 붙은 DAG 의 수. a(n) = sum_k (-1)^(k+1) C(n,k) 2^(k(n-k)) a(n-k)."""
    a = [1]
    for n in range(1, n_max + 1):
        a.append(sum((-1) ** (k + 1) * comb(n, k) * 2 ** (k * (n - k))
                     * a[n - k] for k in range(1, n + 1)))
    return a


# === 그림으로 보는 AI: 고를 것이 몇 개인가 ====================================
#
# 변수 n 개 위의 DAG 수를 로빈슨의 점화식으로 센다. n=10 이면 4.18e18 인데,
# 탐욕적 힐 클라이밍이 한 걸음에 보는 이웃은 n^2 쯤이다. 둘을 나란히 그린다.

def search_space():
    n_max = 15
    a = _dag_counts(n_max)
    for n, v in KNOWN.items():
        assert a[n] == v, f"점화식이 틀렸다: a({n})={a[n]}, 알려진 값 {v}"

    ns = list(range(2, n_max + 1))
    dags = [float(a[n]) for n in ns]
    moves = [float(n * n) for n in ns]
    assert dags[-1] / moves[-1] > 1e30, "두 곡선이 충분히 갈라지지 않았다"

    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    ax.semilogy(ns, dags, "o-", color=ACCENT, linewidth=2.2, ms=6,
                label="DAGs on n variables")
    ax.semilogy(ns, moves, "s--", color=INK, linewidth=1.8, ms=5,
                label="moves a greedy step looks at  (~n^2)")

    for n in (5, 10, 15):
        i = ns.index(n)
        ax.annotate(f"{a[n]:.2e}".replace("e+", " x 10^"),
                    xy=(n, dags[i]), xytext=(-8, -20 if n == 15 else 12),
                    textcoords="offset points", fontsize=8.2, color=ACCENT,
                    ha="right")

    ax.set_xlabel("number of variables", fontsize=9.5)
    ax.set_ylabel("how many (log scale)", fontsize=9.5)
    ax.set_title("the thing being searched, and the thing being looked at",
                 fontsize=10.5, color=INK, weight="bold")
    ax.legend(fontsize=8.6, loc="upper left", framealpha=0.92)
    ax.grid(alpha=0.13, which="both")
    ax.tick_params(labelsize=8, colors=INK)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             "counted with Robinson's 1973 recurrence, checked against the "
             "known values for n up to 6.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "search_space.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote search_space.svg   a(10) = {a[10]:,}   a(15) = {a[15]:.3e}")


if __name__ == "__main__":
    search_space()
