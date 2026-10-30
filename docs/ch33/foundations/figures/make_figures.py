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


if __name__ == "__main__":
    cospectral()
