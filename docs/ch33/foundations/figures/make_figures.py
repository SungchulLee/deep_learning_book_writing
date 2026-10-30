"""34.1 그래프 기초 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 내려받지 않으므로 어디서 돌려도 된다.

실행: python docs/ch33/foundations/figures/make_figures.py
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


# === 그림으로 보는 AI: 스펙트럼이 같아도 그래프는 다르다 ======================
#
# 가장 작은 동형 아닌 동일 스펙트럼 쌍은 K(1,4) 와 C4 + K1 이다. 노드 다섯.
# 그보다 작으면 그런 쌍이 없다는 것까지 그래프 아틀라스를 훑어 직접 센다.

def cospectral():
    import networkx as nx

    G1 = nx.star_graph(4)                                   # K(1,4)
    G2 = nx.disjoint_union(nx.cycle_graph(4), nx.empty_graph(1))

    w1 = np.linalg.eigvalsh(nx.to_numpy_array(G1))
    w2 = np.linalg.eigvalsh(nx.to_numpy_array(G2))

    # --- 그림이 주장하는 것을 먼저 확인한다 -------------------------------
    assert np.abs(w1 - w2).max() < 1e-9, "두 스펙트럼이 같아야 한다"
    assert not nx.is_isomorphic(G1, G2), "두 그래프는 동형이 아니어야 한다"

    # n 이 작을 때 그런 쌍이 정말 없는지 아틀라스로 센다
    counts = {}
    atlas = list(nx.graph_atlas_g())
    for n in (3, 4, 5):
        seen, clash = {}, 0
        for g in atlas:
            if g.number_of_nodes() != n:
                continue
            key = tuple(np.round(np.linalg.eigvalsh(nx.to_numpy_array(g)), 6))
            if key in seen and not nx.is_isomorphic(g, seen[key]):
                clash += 1
            seen.setdefault(key, g)
        counts[n] = clash
    assert counts[3] == 0 and counts[4] == 0 and counts[5] >= 1, \
        "다섯에서 처음 나타나야 이 그림의 이야기가 된다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.2),
                                   gridspec_kw={"width_ratios": [1.1, 1]})

    # --- 왼쪽: 두 그래프 ---------------------------------------------------
    pos1 = {0: (0.0, 0.0)}
    for i, a in enumerate(np.linspace(0, 2 * np.pi, 5)[:4]):
        pos1[i + 1] = (0.42 * np.cos(a), 0.42 * np.sin(a))
    pos1 = {k: (x - 0.85, y + 0.25) for k, (x, y) in pos1.items()}

    pos2 = {}
    for i, a in enumerate(np.linspace(0, 2 * np.pi, 5)[:4] + np.pi / 4):
        pos2[i] = (0.33 * np.cos(a) + 0.75, 0.33 * np.sin(a) + 0.25)
    pos2[4] = (1.55, 0.25)

    for G, pos, col, lab, deg in (
            (G1, pos1, ACCENT, "K(1,4)", [1, 1, 1, 1, 4]),
            (G2, pos2, MUTED, "C4 + K1", [0, 2, 2, 2, 2])):
        for u, v in G.edges():
            ax1.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                     color=col, linewidth=1.8, zorder=1)
        ax1.scatter([p[0] for p in pos.values()],
                    [p[1] for p in pos.values()], s=130, color=col,
                    zorder=3, edgecolors="white", linewidths=1.4)
        cx = np.mean([p[0] for p in pos.values()])
        ax1.text(cx, 0.88, lab, ha="center", fontsize=11, color=col,
                 weight="bold", family="monospace")
        ax1.text(cx, -0.42, f"degrees {deg}", ha="center", fontsize=8.4,
                 color=INK)
    ax1.text(0.35, -0.72, "nothing in common but the count",
             ha="center", fontsize=9, color=INK, style="italic")
    ax1.set_xlim(-1.5, 1.95)
    ax1.set_ylim(-0.95, 1.05)
    ax1.set_aspect("equal")
    ax1.axis("off")

    # --- 오른쪽: 같은 스펙트럼 ---------------------------------------------
    ax2.scatter(w1, [0.78] * len(w1), s=130, color=ACCENT, zorder=3,
                edgecolors="white", linewidths=1.2)
    ax2.scatter(w2, [0.60] * len(w2), s=130, color=MUTED, marker="s",
                zorder=3, edgecolors="white", linewidths=1.2)
    ax2.text(-2.6, 0.78, "K(1,4)", fontsize=9, color=ACCENT, va="center",
             ha="right", family="monospace")
    ax2.text(-2.6, 0.60, "C4+K1", fontsize=9, color=MUTED, va="center",
             ha="right", family="monospace")
    for v in (-2.0, 0.0, 2.0):
        ax2.text(v, 0.90, f"{v:.0f}", ha="center", fontsize=8.4, color=INK)
    ax2.text(0.0, 0.46, f"max difference {np.abs(w1 - w2).max():.0e}",
             ha="center", fontsize=9, color=ACCENT)

    ax2.text(0.0, 0.28, "cospectral but non-isomorphic pairs", ha="center",
             fontsize=9, color=INK, style="italic")
    txt = "     ".join(f"n={n}: {c}" for n, c in counts.items())
    ax2.text(0.0, 0.15, txt, ha="center", fontsize=10, color=INK,
             family="monospace")
    ax2.text(0.0, 0.03, "counted over every graph of that size",
             ha="center", fontsize=8.2, color=INK, alpha=0.8)

    ax2.set_xlim(-3.6, 3.2)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_yticks([])
    ax2.set_xlabel("adjacency eigenvalues", fontsize=9)
    ax2.tick_params(labelsize=8, colors=INK)
    for s in ax2.spines.values():
        s.set_visible(False)

    fig.tight_layout()
    fig.text(0.27, 1.0, "two different graphs", ha="center", fontsize=10,
             color=INK, weight="bold")
    fig.text(0.76, 1.0, "one spectrum", ha="center", fontsize=10,
             color=INK, weight="bold")
    fig.savefig(HERE / "cospectral.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote cospectral.svg   spectra {np.round(w1, 3)}, "
          f"atlas clashes {counts}")


# === 그림으로 보는 AI: 그래프 이론의 첫 정리는 "없다" 였다 ====================
#
# 오일러 1736. 쾨니히스베르크의 땅 넷과 다리 일곱. 차수가 모두 홀수라 그런
# 산책은 없다. 지리를 전부 버리고 **차수만** 남긴 것이 그래프라는 생각의 시작.
# 다리를 하나 더 놓으면 답이 뒤집히는 것까지 직접 판정한다.

def konigsberg():
    import networkx as nx

    G = nx.MultiGraph()
    G.add_edges_from([("A", "B"), ("A", "B"), ("A", "C"), ("A", "C"),
                      ("A", "D"), ("B", "D"), ("C", "D")])
    deg = dict(G.degree())
    odd = [v for v, d in deg.items() if d % 2 == 1]

    G2 = G.copy()
    G2.add_edge("B", "C")
    odd2 = [v for v, d in dict(G2.degree()).items() if d % 2 == 1]

    # --- 그림이 주장하는 것을 먼저 확인한다 -------------------------------
    assert G.number_of_edges() == 7 and G.number_of_nodes() == 4
    assert len(odd) == 4, "네 땅이 모두 홀수 차수여야 한다"
    assert not nx.has_eulerian_path(G), "산책이 없어야 한다"
    assert len(odd2) == 2 and nx.has_eulerian_path(G2), \
        "다리 하나로 답이 뒤집혀야 한다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.2),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # --- 왼쪽: 일곱 다리 ---------------------------------------------------
    pos = {"A": (0.50, 0.50), "B": (0.50, 0.86),
           "C": (0.50, 0.14), "D": (0.90, 0.50)}
    seen = {}
    for u, v in G.edges():
        key = tuple(sorted((u, v)))
        i = seen.get(key, 0)
        seen[key] = i + 1
        rad = 0.0 if i == 0 else 0.42
        ax1.annotate("", xy=pos[v], xytext=pos[u],
                     arrowprops=dict(arrowstyle="-", color=ACCENT,
                                     linewidth=1.9,
                                     connectionstyle=f"arc3,rad={rad}"))
    for v, (x, y) in pos.items():
        ax1.scatter([x], [y], s=620, color=INK, zorder=4,
                    edgecolors="white", linewidths=1.6)
        ax1.text(x, y, f"{deg[v]}", color="white", fontsize=12,
                 ha="center", va="center", zorder=5, weight="bold")
    ax1.text(0.50, 0.50 - 0.12, "island", fontsize=8.2, color=INK,
             ha="center", alpha=0.8)
    ax1.text(0.17, 0.50, "seven bridges,\nfour pieces of land",
             ha="center", va="center", fontsize=9, color=INK)
    ax1.text(0.60, 0.02, "the number in each circle is its degree",
             ha="center", fontsize=8.4, color=INK, alpha=0.8)
    ax1.set_xlim(-0.02, 1.08)
    ax1.set_ylim(-0.02, 1.0)
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_title("Konigsberg, 1736", fontsize=10, color=INK, weight="bold")

    # --- 오른쪽: 규칙과 뒤집힘 ---------------------------------------------
    rows = [("odd-degree nodes: 0", "a closed walk exists", MUTED),
            ("odd-degree nodes: 2", "an open walk exists", MUTED),
            ("anything else", "no such walk", ACCENT)]
    for i, (lhs, rhs, col) in enumerate(rows):
        y = 0.88 - i * 0.125
        ax2.add_patch(plt.Rectangle((0.02, y - 0.052), 0.96, 0.104,
                                    facecolor=col, alpha=0.12,
                                    edgecolor=col, linewidth=0.8))
        ax2.text(0.06, y, lhs, fontsize=8.8, color=INK, va="center",
                 family="monospace")
        ax2.text(0.94, y, rhs, fontsize=8.8, color=col, va="center",
                 ha="right")
    ax2.text(0.5, 0.515, f"Konigsberg has {len(odd)}", ha="center",
             fontsize=11, color=ACCENT, weight="bold")
    ax2.text(0.5, 0.425, "so Euler proved it cannot be done", ha="center",
             fontsize=9, color=INK)

    ax2.plot([0.08, 0.92], [0.345, 0.345], color=INK, linewidth=0.7,
             alpha=0.35)
    ax2.text(0.5, 0.265, "add one more bridge, B to C", ha="center",
             fontsize=9, color=INK, style="italic")
    ax2.text(0.5, 0.16, f"odd-degree nodes: {len(odd2)}", ha="center",
             fontsize=10, color=MUTED, family="monospace", weight="bold")
    ax2.text(0.5, 0.06, "and now the walk exists", ha="center",
             fontsize=9.5, color=MUTED, weight="bold")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0.02, 1.0)
    ax2.axis("off")
    ax2.set_title("counting, not walking", fontsize=10, color=INK,
                  weight="bold")

    fig.text(0.5, -0.02,
             "the rivers, the distances and the shapes were thrown away. "
             "only the degrees decided it.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "konigsberg.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote konigsberg.svg   degrees {deg}, odd {len(odd)} -> "
          f"{len(odd2)} after one bridge")


if __name__ == "__main__":
    cospectral()
    konigsberg()
