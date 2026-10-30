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


# === 그림으로 보는 AI: 자기 루프는 모델링이 아니라 수치 때문이었다 ============
#
# Kipf & Welling 은 I + D^-1/2 A D^-1/2 의 고유값이 [0,2] 라서 깊이 쌓으면
# 터지거나 사라진다고 적고, 자기 루프를 더해 다시 정규화했다. 두 연산자의
# 스펙트럼과 레이어를 쌓았을 때의 노름을 직접 잰다.

def renormalization():
    import networkx as nx

    G = nx.karate_club_graph()
    n = G.number_of_nodes()
    A = np.zeros((n, n))
    for u, v in G.edges():
        A[u, v] = A[v, u] = 1.0

    Dm = np.diag(A.sum(1) ** -0.5)
    P_naive = np.eye(n) + Dm @ A @ Dm                 # 논문 식 (7)
    At = A + np.eye(n)
    Dmt = np.diag(At.sum(1) ** -0.5)
    P_renorm = Dmt @ At @ Dmt                         # 재정규화, 식 (8)

    w_naive = np.linalg.eigvalsh(P_naive)
    w_renorm = np.linalg.eigvalsh(P_renorm)

    # --- 그림이 주장하는 것을 먼저 확인한다 -------------------------------
    assert abs(w_naive).max() > 1.9, "식 (7) 의 스펙트럼 반지름은 2 에 닿아야 한다"
    assert abs(w_renorm).max() <= 1.0 + 1e-9, "재정규화는 1 을 넘지 않아야 한다"

    H = np.random.RandomState(0).randn(n, 8)
    ks = [1, 2, 4, 8, 16, 32]
    nv = [np.linalg.norm(np.linalg.matrix_power(P_naive, k) @ H) for k in ks]
    rn = [np.linalg.norm(np.linalg.matrix_power(P_renorm, k) @ H) for k in ks]
    assert nv[-1] > 1e6 and rn[-1] < 10, "한쪽만 터져야 이 그림의 이야기가 된다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.0))

    # --- 왼쪽: 두 스펙트럼 --------------------------------------------------
    for i, (w, lab, col) in enumerate(
            ((w_naive, "I + D^-1/2 A D^-1/2", INK),
             (w_renorm, "renormalized (self-loops)", ACCENT))):
        y = 0.72 - i * 0.40
        ax1.scatter(w, [y] * len(w), s=26, color=col, alpha=0.75,
                    zorder=3)
        ax1.plot([min(w), max(w)], [y, y], color=col, linewidth=1.0,
                 alpha=0.35)
        ax1.text(0.5, y + 0.15, lab, fontsize=9, color=col, ha="center",
                 family="monospace", transform=ax1.get_yaxis_transform(),
                 clip_on=False)
        ax1.text(max(w), y - 0.13, f"max |lambda| = {abs(w).max():.3f}",
                 fontsize=8.6, color=col, ha="right")
    ax1.axvline(1.0, color=INK, linestyle=":", linewidth=1.0, alpha=0.5)
    ax1.text(1.0, 0.03, " 1", fontsize=8.4, color=INK, alpha=0.7)
    ax1.set_xlim(-0.75, 2.25)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_yticks([])
    ax1.set_xlabel("eigenvalues", fontsize=9)
    ax1.set_title("what the self-loop changes", fontsize=10, color=INK,
                  weight="bold")
    ax1.tick_params(labelsize=8, colors=INK)
    for s in ax1.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    # --- 오른쪽: 쌓으면 어떻게 되는가 --------------------------------------
    ax2.semilogy(ks, nv, "s--", color=INK, linewidth=1.9, ms=5.5,
                 label="eigenvalues up to 2")
    ax2.semilogy(ks, rn, "o-", color=ACCENT, linewidth=2.1, ms=5.5,
                 label="renormalized")
    ax2.annotate(f"{nv[-1]:.0e}", xy=(ks[-1], nv[-1]), xytext=(-46, -4),
                 textcoords="offset points", fontsize=8.6, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
    ax2.annotate(f"{rn[-1]:.2f}", xy=(ks[-1], rn[-1]), xytext=(-40, 8),
                 textcoords="offset points", fontsize=8.6, color=ACCENT,
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.8))
    ax2.set_xlabel("layers k", fontsize=9)
    ax2.set_ylabel("norm of the propagated features", fontsize=9)
    ax2.set_title("stacked, one of them leaves", fontsize=10, color=INK,
                  weight="bold")
    ax2.legend(fontsize=8.2, loc="upper left", framealpha=0.9)
    ax2.grid(alpha=0.13, which="both")
    ax2.tick_params(labelsize=8, colors=INK)
    for s in ax2.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    fig.tight_layout()
    fig.savefig(HERE / "renormalization.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote renormalization.svg   radius {abs(w_naive).max():.3f} vs "
          f"{abs(w_renorm).max():.3f}, k=32 norm {nv[-1]:.1e} vs {rn[-1]:.2f}")


# === 그림으로 보는 AI: 고유기저에서는 국소가 공짜가 아니다 ====================
#
# 고유기저에서 계수를 주는 필터는 임펄스 하나를 그래프 전체로 퍼뜨린다.
# 라플라시안의 K 차 다항식은 정확히 K 홉까지만 닿는다. 둘 다 직접 잰다.

def locality():
    import networkx as nx

    G = nx.karate_club_graph()
    n = G.number_of_nodes()
    A = np.zeros((n, n))
    for u, v in G.edges():
        A[u, v] = A[v, u] = 1.0
    L = np.diag(A.sum(1)) - A
    w, U = np.linalg.eigh(L)
    hops = nx.single_source_shortest_path_length(G, 0)
    x = np.zeros(n)
    x[0] = 1.0

    theta = np.random.RandomState(0).randn(n)
    y_spec = U @ (theta * (U.T @ x))
    nz_spec = np.abs(y_spec) > 1e-10

    Ln = L / np.abs(w).max()
    poly = {}
    for K in (1, 2, 3):
        P = sum(np.linalg.matrix_power(Ln, k) for k in range(K + 1))
        m = np.abs(P @ x) > 1e-10
        poly[K] = (int(m.sum()), max(hops[v] for v in range(n) if m[v]))

    # --- 그림이 주장하는 것을 먼저 확인한다 -------------------------------
    assert nz_spec.all(), "고유기저 필터는 모든 노드에 닿아야 한다"
    for K, (_, reach) in poly.items():
        assert reach == K, f"K={K} 다항식은 정확히 {K} 홉이어야 한다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.0),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    pos = nx.spring_layout(G, seed=3)
    for ax, mask, title in (
            (ax1, nz_spec, "a filter set in the eigenbasis"),
            (ax2, np.abs(sum(np.linalg.matrix_power(Ln, k)
                             for k in range(2)) @ x) > 1e-10,
             "a degree-1 polynomial in L")):
        for u, v in G.edges():
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color=INK, linewidth=0.6, alpha=0.25, zorder=1)
        on = [v for v in G.nodes() if mask[v]]
        off = [v for v in G.nodes() if not mask[v]]
        ax.scatter([pos[v][0] for v in off], [pos[v][1] for v in off],
                   s=60, color=INK, alpha=0.18, zorder=2)
        ax.scatter([pos[v][0] for v in on], [pos[v][1] for v in on],
                   s=70, color=ACCENT, zorder=3, edgecolors="white",
                   linewidths=1.0)
        ax.scatter([pos[0][0]], [pos[0][1]], s=230, facecolors="none",
                   edgecolors=INK, linewidths=1.8, zorder=4)
        ax.set_title(title, fontsize=10, color=INK, weight="bold")
        ax.text(0.5, -0.04, f"{len(on)} of {n} nodes touched",
                transform=ax.transAxes, ha="center", fontsize=9,
                color=ACCENT)
        ax.axis("off")

    ax1.annotate("impulse here", pos[0], textcoords="offset points",
                 xytext=(16, 16), fontsize=8.4, color=INK,
                 arrowprops=dict(arrowstyle="-", color=INK, lw=0.7))
    ax2.text(0.5, -0.125,
             "   ".join(f"K={K}: {r} hops" for K, (_, r) in poly.items()),
             transform=ax2.transAxes, ha="center", fontsize=8.8, color=INK,
             family="monospace")

    fig.tight_layout()
    fig.savefig(HERE / "locality.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote locality.svg   spectral {int(nz_spec.sum())}/{n}, "
          f"poly {poly}")


if __name__ == "__main__":
    wl_stops()
    renormalization()
    locality()
