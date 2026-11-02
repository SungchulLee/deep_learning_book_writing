"""
================================================================================
make_figures.py - 1장 하드웨어 절의 그림을 만든다
================================================================================

만드는 그림:
    memory_hierarchy.svg  메모리 계층의 지연을 로그 눈금으로. 표로 적으면
                          1 ns 와 100 us 가 나란한 두 칸이지만 실제로는
                          다섯 자릿수 차이다. 오른쪽에 「1 클럭을 1 초로
                          늘리면 얼마인가」를 붙여 사람의 감각으로 옮긴다.
    data_path.svg         데이터가 지나는 길. 쪽에 ASCII 로 그려져 있던 것을
                          그대로 옮기되 대역폭 차이를 선 굵기로 보인다.

수는 모두 쪽의 표에 적힌 값이다. 사람 눈금 쪽은 그 값을 3 GHz(한 클럭
0.333 ns)로 나눈 것이며, 쪽에도 어림값임을 적어 둔다.

그림 규칙(CLAUDE.md): SVG, svg.fonttype='path', transparent=True,
그림 안 글자는 모두 ASCII.

실행:
    python make_figures.py
"""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

CLOCK_NS = 1.0 / 3.0                       # 3 GHz 를 가정한 한 클럭


def human(ns):
    """지연을 「1 클럭 = 1 초」로 늘렸을 때의 시간으로 옮긴다."""
    sec = ns / CLOCK_NS
    if sec < 90:
        return f"{sec:.0f} s"
    if sec < 5400:
        return f"{sec / 60:.0f} min"
    if sec < 172800:
        return f"{sec / 3600:.1f} h"
    return f"{sec / 86400:.0f} days"


def fig_memory_hierarchy():
    rows = [("register", 0.33, "in your hand"),
            ("L1 cache", 1.0, "on the desk"),
            ("L2/L3 cache", 10.0, "in the drawer"),
            ("RAM", 100.0, "on the bookshelf"),
            ("SSD", 100_000.0, "in the warehouse")]
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ys = range(len(rows))[::-1]
    for (name, ns, metaphor), y in zip(rows, ys):
        c = "#15803d" if ns <= 1 else ("#b45309" if ns <= 100 else "#b91c1c")
        ax.barh(y, ns, 0.5, color=c, alpha=0.75)
        lab = f"{ns:g} ns" if ns < 1000 else f"{ns / 1000:g} us"
        ax.text(ns * 1.35, y, lab, va="center", fontsize=11, color=c)
        ax.text(0.18, y + 0.42, metaphor, va="center", fontsize=9.5,
                color="0.45")
        ax.text(2.5e6, y, human(ns), va="center", ha="right", fontsize=11,
                color=c)
    ax.set_yticks(list(ys)); ax.set_yticklabels([r[0] for r in rows],
                                                fontsize=11.5)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 4e6)
    ax.set_xlabel("latency (log scale)", fontsize=11)
    ax.set_ylim(-0.7, len(rows) - 0.05)
    ax.text(2.5e6, len(rows) - 0.42, "if one clock took one second",
            ha="right", fontsize=10.5, color="0.3")
    ax.set_title("the table puts these in neighbouring rows - "
                 "they are five orders of magnitude apart", fontsize=11.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig("memory_hierarchy.svg", transparent=True)
    plt.close(fig); print("wrote memory_hierarchy.svg")


def fig_data_path():
    fig, ax = plt.subplots(figsize=(9.8, 4.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")

    boxes = [(4, 26, 20, 9, "SSD", "the dataset lives here", "#9ca3af"),
             (31, 26, 20, 9, "RAM", "a batch at a time", "#b45309"),
             (58, 26, 18, 9, "GPU memory", "", "#1d4ed8"),
             (58, 8, 18, 9, "GPU cores", "", "#15803d")]
    for x, y, w, h, title, sub, c in boxes:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=c, alpha=0.16,
                               edgecolor=c, lw=1.6))
        ax.text(x + w / 2, y + h / 2 + (1.1 if sub else 0), title,
                ha="center", va="center", fontsize=12, color=c)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 2.0, sub, ha="center",
                    va="center", fontsize=9, color="0.45")

    links = [(24, 30.5, 31, 30.5, "~1 GB/s", 1.6, "#b91c1c", "disk read"),
             (51, 30.5, 58, 30.5, "~10 GB/s", 3.0, "#b45309", "PCIe bus"),
             (67, 26, 67, 17, "~1000 GB/s", 7.0, "#15803d", "inside the GPU")]
    for x1, y1, x2, y2, lab, lw, c, note in links:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="<|-|>",
                     color=c, lw=lw, mutation_scale=13))
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        if x1 == x2:
            ax.text(mx + 2.2, my, f"{lab}\n{note}", va="center",
                    fontsize=10.5, color=c)
        else:
            ax.text(mx, my + 5.6, lab, ha="center", fontsize=10.5, color=c)
            ax.text(mx, my - 6.8, note, ha="center", fontsize=9, color="0.45")

    ax.text(50, 1.5, "line thickness is bandwidth - the narrowest link is "
            "the one you pay for", ha="center", fontsize=11, color="#b91c1c")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    fig.savefig("data_path.svg", transparent=True)
    plt.close(fig); print("wrote data_path.svg")


if __name__ == "__main__":
    fig_memory_hierarchy()
    fig_data_path()
