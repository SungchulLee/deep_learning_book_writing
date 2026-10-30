"""34.8 응용 예제 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 내려받지 않으므로 어디서 돌려도 된다.

실행: python docs/ch33/applications/figures/make_figures.py
"""
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


# === 그림으로 보는 AI: 분자 그래프가 담는 것과 담지 못하는 것 =================
#
# 구성(constitution) 은 담기고 배치(configuration) 는 담기지 않는다.
# 거울상 두 분자는 그래프로는 **같은 그래프**이고, 구조 이성질체는 다르다.
# 그림은 꼴로 보이되, 동형 여부는 코드가 직접 판정한다.

def what_the_graph_holds():
    import networkx as nx

    def chiral():
        G = nx.Graph()
        G.add_node(0, el="C")
        for i, e in enumerate(["H", "F", "Cl", "Br"], 1):
            G.add_node(i, el=e)
            G.add_edge(0, i)
        return G

    nm = nx.algorithms.isomorphism.categorical_node_match("el", None)
    same = nx.is_isomorphic(chiral(), chiral(), node_match=nm)

    A = nx.Graph([(0, 1), (1, 2), (2, 3)])      # 곧은 사슬
    B = nx.Graph([(0, 1), (1, 2), (1, 3)])      # 가지 친 사슬
    diff = not nx.is_isomorphic(A, B)

    # --- 그림이 주장하는 것을 먼저 확인한다 -------------------------------
    assert same, "거울상은 같은 그래프여야 한다"
    assert diff, "구조 이성질체는 다른 그래프여야 한다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.3))

    # --- 왼쪽: 거울상 — 그래프로는 같다 ------------------------------------
    labels = ["H", "F", "Cl", "Br"]
    for k, (cx, order) in enumerate(((0.26, [0, 1, 2, 3]),
                                     (0.74, [0, 2, 1, 3]))):
        ax1.scatter([cx], [0.60], s=150, color=INK, zorder=4,
                    edgecolors="white", linewidths=1.3)
        ax1.text(cx, 0.60, "C", color="white", fontsize=8.5, ha="center",
                 va="center", zorder=5, weight="bold")
        dirs = [(0.0, 0.17), (-0.145, -0.085), (0.145, -0.085), (0.0, -0.17)]
        for j, idx in enumerate(order):
            dx, dy = dirs[j]
            ax1.plot([cx, cx + dx], [0.60, 0.60 + dy], color=ACCENT,
                     linewidth=1.7, zorder=2)
            ax1.scatter([cx + dx], [0.60 + dy], s=110, color=ACCENT,
                        zorder=4, edgecolors="white", linewidths=1.1)
            ax1.text(cx + dx * 1.52, 0.60 + dy * 1.58, labels[idx],
                     fontsize=8.4, color=INK, ha="center", va="center")
    ax1.plot([0.50, 0.50], [0.33, 0.87], color=INK, linestyle=":",
             linewidth=1.2, alpha=0.65)
    ax1.text(0.50, 0.90, "mirror", ha="center", fontsize=8.6, color=INK,
             alpha=0.75, style="italic")
    ax1.text(0.5, 0.20, "same atoms, same bonds", ha="center", fontsize=9.5,
             color=INK)
    ax1.text(0.5, 0.08, "isomorphic as graphs: TRUE", ha="center",
             fontsize=10, color=ACCENT, weight="bold", family="monospace")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0.02, 0.98)
    ax1.axis("off")
    ax1.set_title("two mirror-image molecules", fontsize=10, color=INK,
                  weight="bold")

    # --- 오른쪽: 구조 이성질체 — 다르다 ------------------------------------
    posA = {0: (0.08, 0.62), 1: (0.22, 0.62), 2: (0.36, 0.62),
            3: (0.50, 0.62)}
    posB = {0: (0.62, 0.62), 1: (0.76, 0.62), 2: (0.90, 0.62),
            3: (0.76, 0.46)}
    for G, pos in ((A, posA), (B, posB)):
        for u, v in G.edges():
            ax2.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                     color=MUTED, linewidth=1.9, zorder=1)
        ax2.scatter([p[0] for p in pos.values()],
                    [p[1] for p in pos.values()], s=110, color=MUTED,
                    zorder=3, edgecolors="white", linewidths=1.1)
    ax2.text(0.29, 0.76, "straight chain", ha="center", fontsize=9,
             color=INK)
    ax2.text(0.76, 0.76, "branched", ha="center", fontsize=9, color=INK)
    ax2.text(0.5, 0.26,
             f"degrees {sorted(d for _, d in A.degree())}"
             f"   vs   {sorted(d for _, d in B.degree())}",
             ha="center", fontsize=9, color=INK, family="monospace")
    ax2.text(0.5, 0.12, "isomorphic as graphs: FALSE", ha="center",
             fontsize=10, color=MUTED, weight="bold", family="monospace")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0.02, 0.98)
    ax2.axis("off")
    ax2.set_title("two structural isomers", fontsize=10, color=INK,
                  weight="bold")

    fig.text(0.5, -0.02,
             "the graph holds which atom is bonded to which. it does not "
             "hold how the bonds are arranged in space.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "graph_holds.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote graph_holds.svg   mirror isomorphic {same}, "
          f"structural differ {diff}")


if __name__ == "__main__":
    what_the_graph_holds()
