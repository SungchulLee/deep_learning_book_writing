"""34.3-34.4 그래프 겹말기 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 내려받지 않으므로 어디서 돌려도 된다.

실행: python docs/ch33/convolutions/figures/make_figures.py
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


# === 그림으로 보는 AI: 1-WL 이 멈추는 자리 ===================================
#
# C6 와 삼각형 둘은 노드 수·엣지 수·차수가 모두 같다. 1-WL 은 색을 한 번도
# 가르지 못하고, 합 집계를 쓰는 최대 표현력 GNN 도 **같은 임베딩**을 낸다.
# 그런데 연결성분 수와 삼각형 수는 다르다. 수는 전부 여기서 직접 센다.

def wl_stops():
    import networkx as nx
    import torch
    import torch.nn as nn

    G1 = nx.cycle_graph(6)                       # C6
    G2 = nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3))   # 삼각형 둘

    # --- 그림이 주장하는 것을 먼저 확인한다 -------------------------------
    assert not nx.is_isomorphic(G1, G2), "두 그래프는 동형이 아니어야 한다"
    assert (G1.number_of_nodes() == G2.number_of_nodes()
            and G1.number_of_edges() == G2.number_of_edges()), "크기가 같아야 한다"
    assert (sorted(d for _, d in G1.degree())
            == sorted(d for _, d in G2.degree())), "차수 분포가 같아야 한다"

    def wl_rounds(G, rounds=4):
        c = {v: "0" for v in G.nodes()}
        out = []
        for _ in range(rounds):
            new = {v: (c[v], tuple(sorted(c[u] for u in G.neighbors(v))))
                   for v in G.nodes()}
            key = {k: str(i) for i, k in enumerate(sorted(set(new.values())))}
            c = {v: key[new[v]] for v in G.nodes()}
            out.append(len(set(c.values())))
        return out

    r1, r2 = wl_rounds(G1), wl_rounds(G2)
    assert r1 == r2 == [1, 1, 1, 1], "1-WL 이 색을 가르지 못해야 한다"

    def embed(G, seed=0):
        torch.manual_seed(seed)
        n = G.number_of_nodes()
        A = torch.zeros(n, n)
        for u, v in G.edges():
            A[u, v] = A[v, u] = 1.0
        d = 16
        inp = nn.Linear(1, d)
        mlps = [nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
                for _ in range(5)]
        h = inp(torch.ones(n, 1))
        for mlp in mlps:
            h = mlp(h + A @ h)                   # 합 집계
        return h.sum(0)                           # 합 readout

    with torch.no_grad():
        gap = (embed(G1) - embed(G2)).abs().max().item()
    assert gap == 0.0, "최대 표현력 GNN 도 두 그래프를 가르지 못해야 한다"

    tri1 = sum(nx.triangles(G1).values()) // 3
    tri2 = sum(nx.triangles(G2).values()) // 3
    cc1 = nx.number_connected_components(G1)
    cc2 = nx.number_connected_components(G2)
    assert (tri1, cc1) != (tri2, cc2), "달라지는 성질이 있어야 한다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.2),
                                   gridspec_kw={"width_ratios": [1, 1.1]})

    # --- 왼쪽: 두 그래프 ---------------------------------------------------
    def draw(ax, G, pos, color):
        for u, v in G.edges():
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color=color, linewidth=1.8, zorder=1)
        xs = [pos[v][0] for v in G.nodes()]
        ys = [pos[v][1] for v in G.nodes()]
        ax.scatter(xs, ys, s=120, color=color, zorder=3,
                   edgecolors="white", linewidths=1.4)

    ang = np.linspace(0, 2 * np.pi, 7)[:6] + np.pi / 2
    pos1 = {i: (0.5 * np.cos(a) - 1.15, 0.5 * np.sin(a) + 0.55)
            for i, a in enumerate(ang)}
    draw(ax1, G1, pos1, ACCENT)
    ax1.text(-1.15, 1.32, "C6", ha="center", fontsize=11, color=ACCENT,
             weight="bold", family="monospace")

    a3 = np.linspace(0, 2 * np.pi, 4)[:3] + np.pi / 2
    pos2 = {}
    for k in range(2):
        cx = 0.55 + k * 1.25
        for i, a in enumerate(a3):
            pos2[k * 3 + i] = (0.33 * np.cos(a) + cx, 0.33 * np.sin(a) + 0.55)
    draw(ax1, G2, pos2, MUTED)
    ax1.text(1.18, 1.32, "two triangles", ha="center", fontsize=11,
             color=MUTED, weight="bold", family="monospace")

    ax1.text(0.0, -0.20,
             f"both: {G1.number_of_nodes()} nodes, "
             f"{G1.number_of_edges()} edges, every degree 2",
             ha="center", fontsize=9, color=INK)
    ax1.set_xlim(-1.95, 2.25)
    ax1.set_ylim(-0.45, 1.6)
    ax1.set_aspect("equal")
    ax1.axis("off")

    # --- 오른쪽: 가르지 못하는 것과 가르는 것 ------------------------------
    ax2.text(0.5, 0.95, "what the methods see", ha="center", fontsize=9,
             color=INK, style="italic")
    rows_same = [
        ("1-WL colours, round 1", f"{r1[0]} / {r2[0]}"),
        ("1-WL colours, round 4", f"{r1[3]} / {r2[3]}"),
        ("sum-aggregation GNN", f"gap {gap:.1f}"),
    ]
    for i, (lab, val) in enumerate(rows_same):
        y = 0.80 - i * 0.115
        ax2.add_patch(plt.Rectangle((0.02, y - 0.047), 0.96, 0.094,
                                    facecolor=ACCENT, alpha=0.13,
                                    edgecolor=ACCENT, linewidth=0.8))
        ax2.text(0.06, y, lab, fontsize=9, color=INK, va="center")
        ax2.text(0.94, y, val, fontsize=9.5, color=ACCENT, va="center",
                 ha="right", family="monospace", weight="bold")
    ax2.text(0.5, 0.40, "identical, forever", ha="center", fontsize=9.5,
             color=ACCENT, style="italic")

    ax2.plot([0.05, 0.95], [0.335, 0.335], color=INK, linewidth=0.7,
             alpha=0.35)
    rows_diff = [("connected components", f"{cc1} / {cc2}"),
                 ("triangles", f"{tri1} / {tri2}")]
    for i, (lab, val) in enumerate(rows_diff):
        y = 0.25 - i * 0.105
        ax2.text(0.06, y, lab, fontsize=9, color=INK, va="center")
        ax2.text(0.94, y, val, fontsize=9.5, color=INK, va="center",
                 ha="right", family="monospace", weight="bold")
    ax2.text(0.5, 0.02, "but these were never equal", ha="center",
             fontsize=9.5, color=INK)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.03, 1.0)
    ax2.axis("off")

    fig.tight_layout()
    # 왼쪽은 aspect="equal" 로 축이 줄어 set_title 높이가 어긋난다. 그림 좌표로 맞춘다.
    fig.text(0.26, 1.01, "same by every count that matches", ha="center",
             fontsize=10, color=INK, weight="bold")
    fig.text(0.76, 1.01, "counted, not asserted", ha="center",
             fontsize=10, color=INK, weight="bold")
    fig.savefig(HERE / "wl_stops.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote wl_stops.svg   colours {r1} vs {r2}, gnn gap {gap}, "
          f"cc {cc1}/{cc2}, tri {tri1}/{tri2}")


if __name__ == "__main__":
    wl_stops()
