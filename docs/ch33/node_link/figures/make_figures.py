"""34.7 노드·링크 과제 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 내려받지 않으므로 어디서 돌려도 된다 (가라테 클럽은 networkx 에 들어 있다).

실행: python docs/ch33/node_link/figures/make_figures.py
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


# === 그림으로 보는 AI: 커뮤니티가 둘이라는 것은 누가 정했는가 ==================
#
# 가라테 클럽의 "정답"이 둘인 까닭은 1970-72 년에 실제로 동아리가 둘로 갈라졌기
# 때문이다. 개수를 알려 주면 스펙트럼 2분할이 한 명만 틀리는데, 개수를 스스로
# 고르는 방법들은 둘이 아니라 셋·넷을 내놓는다. 수는 전부 여기서 직접 센다.

def who_said_two():
    import networkx as nx
    from networkx.algorithms.community import (greedy_modularity_communities,
                                               louvain_communities)

    G = nx.karate_club_graph()
    n = G.number_of_nodes()
    truth = {v: G.nodes[v]["club"] for v in G.nodes()}
    labs = sorted(set(truth.values()))
    assert len(labs) == 2, "실제 쪼개짐은 둘이어야 한다"

    # --- 개수를 알려 준 경우: 피들러 벡터로 2분할 -------------------------
    L = np.asarray(nx.normalized_laplacian_matrix(G).todense())
    _, V = np.linalg.eigh(L)
    fiedler = V[:, 1]
    nodes = list(G.nodes())
    pred = {v: ("A" if fiedler[i] > 0 else "B") for i, v in enumerate(nodes)}

    def wrong(pred):
        best, bestmap = n, None
        for flip in (False, True):
            m = {labs[0]: "B" if flip else "A", labs[1]: "A" if flip else "B"}
            d = [v for v in nodes if pred[v] != m[truth[v]]]
            if len(d) < best:
                best, bestmap = len(d), d
        return best, bestmap

    n_wrong, wrong_nodes = wrong(pred)
    assert n_wrong <= 2, "스펙트럼 2분할은 거의 맞아야 한다"

    # --- 개수를 스스로 고르는 경우 ----------------------------------------
    k_greedy = len(greedy_modularity_communities(G))
    k_louvain = len(louvain_communities(G, seed=0))
    assert k_greedy != 2 or k_louvain != 2, \
        "스스로 고르는 방법이 둘을 내놓으면 이 그림의 이야기가 없다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.4),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # --- 왼쪽: 실제 쪼개짐 --------------------------------------------------
    pos = nx.spring_layout(G, seed=3)
    for u, v in G.edges():
        ax1.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                 color=INK, linewidth=0.6, alpha=0.3, zorder=1)
    for lab, col in zip(labs, (ACCENT, MUTED)):
        pts = [pos[v] for v in nodes if truth[v] == lab]
        ax1.scatter([p[0] for p in pts], [p[1] for p in pts], s=90,
                    color=col, zorder=3, edgecolors="white", linewidths=1.2,
                    label=f"{lab}  ({sum(1 for v in nodes if truth[v]==lab)})")
    for v in wrong_nodes:
        ax1.scatter([pos[v][0]], [pos[v][1]], s=320, facecolors="none",
                    edgecolors=INK, linewidths=1.8, zorder=4)
        ax1.annotate("the one it misses", pos[v], textcoords="offset points",
                     xytext=(14, 14), fontsize=8.4, color=INK,
                     arrowprops=dict(arrowstyle="-", color=INK, lw=0.7))
    ax1.legend(fontsize=8.4, loc="lower left", framealpha=0.9)
    ax1.axis("off")
    ax1.set_title(f"the club that actually split, {n} people",
                  fontsize=10, color=INK, weight="bold")

    # --- 오른쪽: 누가 개수를 정했는가 --------------------------------------
    ax2.text(0.5, 0.95, "how many communities?", ha="center", fontsize=9,
             color=INK, style="italic")
    rows = [("told it is 2, spectral split", f"{n_wrong} wrong of {n}", ACCENT),
            ("greedy modularity, free to choose", f"{k_greedy} found", INK),
            ("Louvain, free to choose", f"{k_louvain} found", INK)]
    for i, (lab, val, col) in enumerate(rows):
        y = 0.76 - i * 0.145
        ax2.add_patch(plt.Rectangle((0.02, y - 0.058), 0.96, 0.116,
                                    facecolor=col, alpha=0.12,
                                    edgecolor=col, linewidth=0.9))
        ax2.text(0.06, y, lab, fontsize=8.8, color=INK, va="center")
        ax2.text(0.94, y, val, fontsize=9.5, color=col, va="center",
                 ha="right", family="monospace", weight="bold")
    ax2.text(0.5, 0.26,
             "the answer 2 came from an\nethnography, not from the graph",
             ha="center", va="top", fontsize=9, color=ACCENT)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0.0, 1.0)
    ax2.axis("off")
    ax2.set_title("counted, not asserted", fontsize=10, color=INK,
                  weight="bold")

    fig.tight_layout()
    fig.savefig(HERE / "who_said_two.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote who_said_two.svg   spectral {n_wrong}/{n} wrong, "
          f"greedy {k_greedy}, louvain {k_louvain}")


# === 그림으로 보는 AI: 랜덤워크가 언어를 닮는 조건 ===========================
#
# DeepWalk 의 근거는 "차수가 거듭제곱 꼴이면 워크의 노드 출현 빈도도 거듭제곱
# 꼴이 되어 낱말 빈도를 닮는다" 는 것이다. 앞부분(빈도 ~ 차수)은 어느 그래프
# 에서나 성립하지만, 뒷부분(거듭제곱 꼴)은 그래프에 달렸다. 둘 다 직접 잰다.

def walks_like_language():
    import collections
    import random

    import networkx as nx

    random.seed(0)

    def counts(G, n_walks=200, length=40):
        c = collections.Counter()
        nodes = list(G.nodes())
        for _ in range(n_walks):
            v = random.choice(nodes)
            for _ in range(length):
                c[v] += 1
                nb = list(G.neighbors(v))
                if not nb:
                    break
                v = random.choice(nb)
        return c

    data = {}
    for nm, G in (("scale-free", nx.barabasi_albert_graph(400, 2, seed=0)),
                  ("karate club", nx.karate_club_graph())):
        c = counts(G)
        deg = np.array([G.degree(v) for v in G.nodes()], float)
        vis = np.array([c[v] for v in G.nodes()], float)
        corr = float(np.corrcoef(deg, vis)[0, 1])
        v, k = np.unique(deg, return_counts=True)
        slope = float(np.polyfit(np.log(v), np.log(k), 1)[0])
        data[nm] = (deg, vis, corr, v, k, slope)

    # --- 그림이 주장하는 것을 먼저 확인한다 -------------------------------
    for nm, (_, _, corr, _, _, _) in data.items():
        assert corr > 0.9, f"{nm}: 방문 빈도가 차수를 따라야 한다"
    assert data["scale-free"][5] < data["karate club"][5] - 0.5, \
        "두 그래프의 차수 분포 꼴이 달라야 이 그림의 이야기가 된다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.0))

    for nm, col, mk in (("scale-free", ACCENT, "o"),
                        ("karate club", MUTED, "s")):
        deg, vis, corr, v, k, slope = data[nm]
        ax1.scatter(deg, vis / vis.sum(), s=22, color=col, alpha=0.6,
                    marker=mk, label=f"{nm}   r = {corr:.3f}")
        ax2.loglog(v, k, mk + "-", color=col, ms=5, linewidth=1.4,
                   alpha=0.85, label=f"{nm}   slope {slope:.2f}")
    ax1.set_xlabel("degree", fontsize=9)
    ax1.set_ylabel("share of walk visits", fontsize=9)
    ax1.set_title("visits follow degree - on any graph", fontsize=10,
                  color=INK, weight="bold")
    ax1.legend(fontsize=8.2, loc="upper left", framealpha=0.9)
    ax1.grid(alpha=0.13)

    ax2.set_xlabel("degree", fontsize=9)
    ax2.set_ylabel("how many nodes", fontsize=9)
    ax2.set_title("but the shape is the graph's own", fontsize=10,
                  color=INK, weight="bold")
    ax2.legend(fontsize=8.2, loc="upper right", framealpha=0.9)
    ax2.grid(alpha=0.13, which="both")

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.text(0.5, -0.03,
             "the borrowing is justified by the right-hand picture, not the "
             "left one.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "walks_like_language.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote walks_like_language.svg   " + "  ".join(
        f"{nm}: r={d[2]:.3f} slope={d[5]:.2f}" for nm, d in data.items()))


# === 그림으로 보는 AI: 드문 이웃이 더 말해 준다 ==============================
#
# Adamic-Adar 의 1/log 웨이트는 그래프 이론이 아니라 홈페이지끼리 얼마나 닮았는지
# 재려던 데서 나왔다. 웨이트가 실제로 허브를 끌어내리는지, 그리고 그 보정이 이
# 그래프의 예측을 낫게 하는지를 따로 잰다. 뒤쪽 답은 "아니오" 다.

def rare_neighbours_say_more():
    import itertools
    import math

    import networkx as nx

    G = nx.karate_club_graph()
    nodes = list(G.nodes())
    deg = dict(G.degree())
    pairs = list(itertools.combinations(nodes, 2))
    hubs = sorted(deg, key=deg.get, reverse=True)[:2]

    # --- 허브가 공통 이웃 질량에서 차지하는 몫 -----------------------------
    nb = {v: set(G.neighbors(v)) for v in nodes}
    cn_all = cn_hub = 0.0
    aa_all = aa_hub = 0.0
    for u, v in pairs:
        for w in nb[u] & nb[v]:
            wt = 1.0 / math.log(deg[w]) if deg[w] > 1 else 0.0
            cn_all += 1
            aa_all += wt
            if w in hubs:
                cn_hub += 1
                aa_hub += wt
    cn_share, aa_share = cn_hub / cn_all, aa_hub / aa_all
    assert aa_share < cn_share, "1/log 웨이트는 허브의 몫을 줄여야 한다"

    # --- 그 보정이 예측을 낫게 하는가: 엣지 20% 를 가리고 열 번 ------------
    edges = list(G.edges())
    k_hold = len(edges) // 5
    scores = {"CN": [], "AA": []}
    for seed in range(10):
        rng = np.random.default_rng(seed)
        held = [edges[i] for i in rng.permutation(len(edges))[:k_hold]]
        H = G.copy()
        H.remove_edges_from(held)
        nh = {v: set(H.neighbors(v)) for v in nodes}
        truth = {frozenset(e) for e in held}
        present = {frozenset(e) for e in H.edges()}
        cand = [p for p in pairs if frozenset(p) not in present]
        fns = {"CN": lambda u, v: len(nh[u] & nh[v]),
               "AA": lambda u, v: sum(1 / math.log(len(nh[w]))
                                      for w in nh[u] & nh[v]
                                      if len(nh[w]) > 1)}
        for nm, f in fns.items():
            top = sorted(cand, key=lambda p: -f(*p))[:k_hold]
            scores[nm].append(sum(1 for p in top if frozenset(p) in truth))
    n_cand = len(cand)
    base = k_hold / n_cand
    mu = {nm: float(np.mean(s)) for nm, s in scores.items()}
    sd = {nm: float(np.std(s)) for nm, s in scores.items()}
    assert abs(mu["CN"] - mu["AA"]) < max(sd.values()), \
        "두 점수의 차가 흩어짐보다 크면 이 그림의 이야기가 아니다"
    assert min(mu.values()) / k_hold > 5 * base, "둘 다 무작위보다 훨씬 나아야 한다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0),
                                   gridspec_kw={"width_ratios": [1, 1.05]})

    # --- 왼쪽: 웨이트는 제 일을 한다 ---------------------------------------
    labs = ["plain count\nof common\nneighbours", "weighted by\n1 / log(degree)"]
    shares = [cn_share, aa_share]
    x = np.arange(2)
    ax1.bar(x, shares, 0.52, color=ACCENT, alpha=0.85,
            label=f"from the two hubs (degree {deg[hubs[0]]}, {deg[hubs[1]]})")
    ax1.bar(x, [1 - s for s in shares], 0.52, bottom=shares, color=INK,
            alpha=0.2, label="from the other 32 nodes")
    for i, s in enumerate(shares):
        ax1.text(i, s / 2, f"{s:.1%}", ha="center", va="center", fontsize=11,
                 color="white", weight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labs, fontsize=8.6)
    ax1.set_ylim(0, 1.3)
    ax1.set_yticks(np.arange(0, 1.01, 0.25))
    ax1.set_yticklabels([f"{t:.0%}" for t in np.arange(0, 1.01, 0.25)])
    ax1.set_ylabel("share of the evidence", fontsize=9)
    ax1.set_title("the weight does what it promises", fontsize=10, color=INK,
                  weight="bold")
    ax1.legend(fontsize=8, loc="upper center", framealpha=0.0,
               handlelength=1.3)

    # --- 오른쪽: 그래도 예측은 같다 ----------------------------------------
    y = np.arange(3)[::-1]
    vals = [mu["CN"] / k_hold, mu["AA"] / k_hold, base]
    errs = [sd["CN"] / k_hold, sd["AA"] / k_hold, 0.0]
    cols = [INK, ACCENT, MUTED]
    names = ["common neighbours", "Adamic-Adar", "pick at random"]
    ax2.barh(y, vals, 0.5, xerr=errs, color=cols, alpha=0.8,
             error_kw=dict(ecolor=INK, capsize=4, lw=1.1))
    for yy, v, e, nm in zip(y, vals, errs, names):
        txt = f"{v:.1%}" + (f"  +- {e:.1%}" if e else "")
        ax2.text(v + e + 0.012, yy, txt, va="center", fontsize=9, color=INK,
                 family="monospace")
    ax2.set_yticks(y)
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlim(0, 0.42)
    ax2.set_xticks(np.arange(0, 0.41, 0.1))
    ax2.set_xticklabels([f"{t:.0%}" for t in np.arange(0, 0.41, 0.1)])
    ax2.set_xlabel(f"share of the {k_hold} hidden edges recovered "
                   f"in the top {k_hold}", fontsize=8.8)
    ax2.set_title("but here it buys nothing", fontsize=10, color=INK,
                  weight="bold")

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8, colors=INK)
        ax.grid(alpha=0.13, axis="x" if ax is ax2 else "y")
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             f"karate club, 10 random hold-outs of {k_hold} edges; the gap "
             "between the top two is smaller than their spread.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "rare_neighbours_say_more.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote rare_neighbours_say_more.svg   hub share {cn_share:.3f} -> "
          f"{aa_share:.3f};  CN {mu['CN']:.2f}+-{sd['CN']:.2f}  "
          f"AA {mu['AA']:.2f}+-{sd['AA']:.2f}  of {k_hold}, "
          f"random {base*k_hold:.2f}")


if __name__ == "__main__":
    who_said_two()
    walks_like_language()
    rare_neighbours_say_more()
