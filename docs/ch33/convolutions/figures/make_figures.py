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


# === 그림으로 보는 AI: 학습하지 않은 어텐션도 확신에 차 보인다 ================
#
# GCN 의 계수는 차수로 정해져 **완전히 대칭**이고, GAT 의 알파는 수신 노드별
# softmax 라 비대칭이다. 그런데 **학습 전** 임의 웨이트만으로도 어텐션이 이미
# 균등에서 멀리 떨어져 있다 — 큰 알파 하나가 그 자체로는 아무 근거가 아니다.

def attention_at_init():
    import math

    import networkx as nx
    import torch

    torch.manual_seed(0)
    G = nx.karate_club_graph()
    n = G.number_of_nodes()
    edges = ([(u, v) for u, v in G.edges()] + [(v, u) for u, v in G.edges()]
             + [(i, i) for i in range(n)])
    src = torch.tensor([e[0] for e in edges])
    dst = torch.tensor([e[1] for e in edges])

    F = 8
    X = torch.randn(n, 4)
    W = torch.randn(4, F) * 0.5
    a = torch.randn(2 * F) * 0.5
    H = X @ W
    sc = torch.nn.functional.leaky_relu(
        (torch.cat([H[src], H[dst]], 1) * a).sum(1), 0.2)
    alpha = torch.zeros(len(edges))
    for d in range(n):
        m = dst == d
        alpha[m] = torch.softmax(sc[m], 0)

    At = np.zeros((n, n))
    for u, v in G.edges():
        At[u, v] = At[v, u] = 1.0
    At = At + np.eye(n)
    dt = At.sum(1)

    amap = {(u, v): alpha[i].item()
            for i, (u, v) in enumerate(zip(src.tolist(), dst.tolist()))}
    pairs = [(u, v) for (u, v) in amap if (v, u) in amap and u != v]
    gcn_uv = [1 / np.sqrt(dt[u] * dt[v]) for u, v in pairs]
    gcn_vu = [1 / np.sqrt(dt[v] * dt[u]) for u, v in pairs]
    gat_uv = [amap[(u, v)] for u, v in pairs]
    gat_vu = [amap[(v, u)] for u, v in pairs]

    asym_gcn = max(abs(x - y) for x, y in zip(gcn_uv, gcn_vu))
    asym_gat = max(abs(x - y) for x, y in zip(gat_uv, gat_vu))

    ratios = []
    for d in range(n):
        m = dst == d
        p = alpha[m]
        k = int(m.sum())
        if k > 1:
            ent = float(-(p * p.clamp(min=1e-12).log()).sum())
            ratios.append(ent / math.log(k))

    # --- 그림이 주장하는 것을 먼저 확인한다 -------------------------------
    assert asym_gcn < 1e-12, "GCN 계수는 완전히 대칭이어야 한다"
    assert asym_gat > 0.1, "GAT 어텐션은 비대칭이어야 한다"
    assert np.mean(ratios) < 0.95, \
        "학습 전 어텐션이 균등하면 이 그림의 이야기가 없다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.1))

    # --- 왼쪽: 양방향 계수 ------------------------------------------------
    lim = max(max(gat_uv), max(gat_vu)) * 1.08
    ax1.plot([0, lim], [0, lim], color=INK, linestyle=":", linewidth=1.1,
             alpha=0.6)
    ax1.scatter(gat_uv, gat_vu, s=26, color=ACCENT, alpha=0.75,
                label=f"GAT    max gap {asym_gat:.3f}")
    ax1.scatter(gcn_uv, gcn_vu, s=26, color=MUTED, alpha=0.85,
                marker="s", label=f"GCN    max gap {asym_gcn:.0e}")
    ax1.text(lim * 0.60, lim * 0.52, "symmetric", fontsize=8.4, color=INK,
             rotation=45, alpha=0.7)
    ax1.set_xlabel("coefficient u -> v", fontsize=9)
    ax1.set_ylabel("coefficient v -> u", fontsize=9)
    ax1.set_title("the same edge, both directions", fontsize=10, color=INK,
                  weight="bold")
    ax1.legend(fontsize=8.2, loc="upper left", framealpha=0.9)
    ax1.grid(alpha=0.13)

    # --- 오른쪽: 학습 전 어텐션의 쏠림 -------------------------------------
    ax2.hist(ratios, bins=12, color=ACCENT, alpha=0.8)
    ax2.axvline(1.0, color=INK, linestyle="--", linewidth=1.3)
    ax2.text(1.0, ax2.get_ylim()[1] * 0.93, " uniform", fontsize=8.6,
             color=INK, va="top")
    ax2.axvline(float(np.mean(ratios)), color=MUTED, linewidth=1.6)
    ax2.text(float(np.mean(ratios)) - 0.012, ax2.get_ylim()[1] * 0.55,
             f"mean {np.mean(ratios):.3f} ", fontsize=8.6, color=MUTED,
             ha="right")
    ax2.set_xlim(0, 1.12)
    ax2.set_xlabel("attention entropy, as a fraction of uniform",
                   fontsize=9)
    ax2.set_ylabel("nodes", fontsize=9)
    ax2.set_title("before any training at all", fontsize=10, color=INK,
                  weight="bold")
    ax2.grid(alpha=0.13, axis="y")

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.tight_layout()
    fig.savefig(HERE / "attention_at_init.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote attention_at_init.svg   asym gcn {asym_gcn:.0e} gat "
          f"{asym_gat:.3f}, entropy ratio {np.mean(ratios):.4f}")


# === 그림으로 보는 AI: 표에서 함수로 =========================================
#
# 노드마다 임베딩 한 줄을 들고 있으면 파라메터가 노드 수에 비례해 자라고,
# 새 노드에는 줄 자체가 없다. 이웃을 모으는 **함수**는 파라메터가 고정이고
# 새 노드에도 바로 답을 낸다. 둘 다 직접 센다.

def table_vs_function():
    import networkx as nx
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    G = nx.karate_club_graph()
    n, d, fin = G.number_of_nodes(), 16, 8

    table = nn.Embedding(n, d)
    sage = nn.Linear(2 * fin, d)
    p_table = sum(p.numel() for p in table.parameters())
    p_sage = sum(p.numel() for p in sage.parameters())

    X = torch.randn(n, fin)
    new = n
    G2 = G.copy()
    G2.add_node(new)
    for v in (0, 1, 33):
        G2.add_edge(new, v)

    has_row = new < table.num_embeddings
    nb = torch.stack([X[v] for v in G2.neighbors(new)]).mean(0, keepdim=True)
    emb = sage(torch.cat([torch.randn(1, fin), nb], 1))

    # --- 그림이 주장하는 것을 먼저 확인한다 -------------------------------
    assert not has_row, "새 노드는 표에 줄이 없어야 한다"
    assert emb.shape == (1, d), "함수는 새 노드에도 임베딩을 내야 한다"

    sizes = [34, 100, 1000, 10000, 100000]
    tbl = [s * d for s in sizes]
    fn = [p_sage] * len(sizes)
    assert tbl[-1] > fn[-1] * 100, "표가 훨씬 빨리 자라야 한다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.0))

    # --- 왼쪽: 표에는 새 노드의 줄이 없다 ---------------------------------
    for i in range(6):
        y = 0.80 - i * 0.105
        ax1.add_patch(plt.Rectangle((0.12, y - 0.040), 0.52, 0.080,
                                    facecolor=ACCENT, alpha=0.16,
                                    edgecolor=ACCENT, linewidth=0.8))
        ax1.text(0.07, y, f"{i}" if i < 5 else "...", fontsize=9,
                 color=INK, va="center", ha="right", family="monospace")
    y = 0.80 - 6 * 0.105
    ax1.add_patch(plt.Rectangle((0.12, y - 0.040), 0.52, 0.080,
                                facecolor="none", edgecolor=INK,
                                linewidth=1.3, linestyle="--"))
    ax1.text(0.07, y, f"{new}", fontsize=9, color=INK, va="center",
             ha="right", family="monospace", weight="bold")
    ax1.text(0.38, y, "no row", fontsize=9.5, color=INK, va="center",
             ha="center", style="italic")
    ax1.text(0.38, 0.92, "one row per node", ha="center", fontsize=9.5,
             color=ACCENT, weight="bold")
    ax1.text(0.38, 0.02,
             "a node that arrives after training\nhas nothing to look up",
             ha="center", fontsize=8.8, color=INK)
    ax1.set_xlim(0, 0.76)
    ax1.set_ylim(-0.02, 1.0)
    ax1.axis("off")
    ax1.set_title("the transductive answer", fontsize=10, color=INK,
                  weight="bold")

    # --- 오른쪽: 파라메터가 자라는가 --------------------------------------
    ax2.loglog(sizes, tbl, "s--", color=INK, linewidth=1.9, ms=5.5,
               label="embedding table")
    ax2.loglog(sizes, fn, "o-", color=ACCENT, linewidth=2.1, ms=5.5,
               label="aggregating function")
    ax2.annotate(f"{tbl[-1]:,}", xy=(sizes[-1], tbl[-1]), xytext=(-56, -4),
                 textcoords="offset points", fontsize=8.6, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
    ax2.annotate(f"{fn[-1]:,}", xy=(sizes[-1], fn[-1]), xytext=(-40, 10),
                 textcoords="offset points", fontsize=8.6, color=ACCENT,
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.8))
    ax2.set_xlabel("nodes in the graph", fontsize=9)
    ax2.set_ylabel("parameters", fontsize=9)
    ax2.set_title("what has to be stored", fontsize=10, color=INK,
                  weight="bold")
    ax2.legend(fontsize=8.4, loc="upper left", framealpha=0.9)
    ax2.grid(alpha=0.13, which="both")
    ax2.tick_params(labelsize=8, colors=INK)
    for s in ax2.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    fig.tight_layout()
    fig.savefig(HERE / "table_vs_function.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote table_vs_function.svg   table {p_table} vs fn {p_sage}, "
          f"new node row: {has_row}")


# === 그림으로 보는 AI: 고리에서는 푸리에와 정확히 같다, 기저를 맞추면 ========
#
# 고리 그래프에서 라플라시안 고유값은 2-2cos(2 pi k/n) 이고 복소 지수가
# 고유벡터다. 그 기저에서는 고유기저 곱이 **정확히** 순환 겹말기가 된다.
# 그런데 eigh 가 돌려주는 실수 기저에서는 그렇지 않다 — 고유값이 겹쳐 있어
# 고유공간 안의 기저가 임의이기 때문이다. 셋 다 직접 잰다.

def cycle_is_fourier():
    import networkx as nx

    n = 12
    C = nx.cycle_graph(n)
    L = np.asarray(nx.laplacian_matrix(C).todense()).astype(float)
    w, U = np.linalg.eigh(L)

    theory = np.sort([2 - 2 * np.cos(2 * np.pi * k / n) for k in range(n)])
    gap_eig = float(np.abs(np.sort(w) - theory).max())

    F = np.array([[np.exp(-2j * np.pi * i * k / n) for k in range(n)]
                  for i in range(n)]) / np.sqrt(n)
    lam = np.array([2 - 2 * np.cos(2 * np.pi * k / n) for k in range(n)])
    gap_vec = float(np.abs(L @ F - F @ np.diag(lam)).max())

    rng = np.random.RandomState(0)
    x, h = rng.randn(n), rng.randn(n)
    circ = np.real(np.fft.ifft(np.fft.fft(h) * np.fft.fft(x)))
    d_complex = float(np.abs(
        F @ ((F.conj().T @ h) * (F.conj().T @ x)) * np.sqrt(n) - circ).max())
    d_real = float(np.abs(U @ ((U.T @ h) * (U.T @ x)) - circ).max())

    uw, cnt = np.unique(np.round(w, 8), return_counts=True)
    n_double = int((cnt == 2).sum())

    # --- 그림이 주장하는 것을 먼저 확인한다 -------------------------------
    assert gap_eig < 1e-10, "고유값이 이론값과 같아야 한다"
    assert gap_vec < 1e-10, "복소 지수가 고유벡터여야 한다"
    assert d_complex < 1e-9, "복소 기저에서는 겹말기 정리가 성립해야 한다"
    assert d_real > 1.0, "실수 기저에서는 깨져야 이 그림의 이야기가 된다"
    assert n_double > 0, "겹친 고유값이 있어야 한다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.1),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # --- 왼쪽: 고유값이 코사인 곡선 위에 ----------------------------------
    kk = np.linspace(0, n, 400)
    ax1.plot(kk, 2 - 2 * np.cos(2 * np.pi * kk / n), color=INK,
             linewidth=1.3, alpha=0.5, label="2 - 2 cos(2 pi k / n)")
    ax1.scatter(range(n), [2 - 2 * np.cos(2 * np.pi * k / n)
                           for k in range(n)], s=55, color=ACCENT, zorder=3,
                edgecolors="white", linewidths=1.0,
                label="Laplacian eigenvalues")
    ax1.set_xlabel("k", fontsize=9)
    ax1.set_ylabel("eigenvalue", fontsize=9)
    ax1.set_title(f"a cycle of {n}: the spectrum is a cosine", fontsize=10,
                  color=INK, weight="bold")
    ax1.legend(fontsize=8.2, loc="lower center", framealpha=0.92)
    ax1.set_ylim(-0.55, 4.45)   # 범례가 꼭대기 점을 가리지 않게
    ax1.text(0.5, -0.255, f"matches to {gap_eig:.0e}   -   "
                          f"{n_double} eigenvalues are doubled",
             transform=ax1.transAxes, ha="center", fontsize=8.6, color=INK)
    ax1.grid(alpha=0.13)

    # --- 오른쪽: 겹말기 정리가 어디서 성립하나 ----------------------------
    labs = ["complex exponential\nbasis", "the basis eigh\nreturns"]
    vals = [d_complex, d_real]
    cols = [MUTED, ACCENT]
    ax2.bar(range(2), [max(v, 1e-16) for v in vals], color=cols, width=0.48,
            alpha=0.85)
    ax2.set_yscale("log")
    ax2.set_xticks(range(2))
    ax2.set_xticklabels(labs, fontsize=8.8)
    for i, v in enumerate(vals):
        ax2.text(i, v * 2.2, f"{v:.1e}" if v < 1 else f"{v:.2f}",
                 ha="center", fontsize=9.5, color=INK, weight="bold")
    ax2.set_ylabel("gap from true circular convolution", fontsize=9)
    ax2.set_ylim(1e-16, max(vals) * 1e3)
    ax2.set_title("same graph, same signals", fontsize=10, color=INK,
                  weight="bold")
    ax2.grid(alpha=0.13, axis="y", which="both")

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.text(0.5, -0.05,
             "the convolution theorem needs one particular basis. the "
             "eigensolver has no reason to pick it.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "cycle_is_fourier.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote cycle_is_fourier.svg   eig gap {gap_eig:.1e}, "
          f"complex {d_complex:.1e}, real {d_real:.3f}, doubled {n_double}")


if __name__ == "__main__":
    wl_stops()
    renormalization()
    locality()
    attention_at_init()
    table_vs_function()
    cycle_is_fourier()
