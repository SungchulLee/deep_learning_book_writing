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


# === 그림으로 보는 AI: 1860 년에 분류해 둔 성질들 ============================
#
# 관계의 성질 - 역, 대칭, 합성 - 을 갈라 놓은 것은 De Morgan(1860)과 Peirce(1870)
# 이다. 이동 가정은 그 가운데 합성은 그대로 안고 대칭은 못 안는다. 흠 없는 자료로
# 학습해서 무엇이 무너지는지 직접 잰다. 자료가 깨끗하므로 탓할 것은 모델뿐이다.

def properties_catalogued_in_1860():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    D = 32

    def fit(triples, n_ent, n_rel, steps=3000, seed=0, neg=5):
        """이 쪽의 TransE 그대로 - 점수는 -||e_h + e_r - e_t||."""
        torch.manual_seed(seed)
        E = nn.Embedding(n_ent, D)
        R = nn.Embedding(n_rel, D)
        opt = torch.optim.Adam(list(E.parameters()) + list(R.parameters()),
                               lr=0.01)
        h, r, t = [torch.tensor(v) for v in zip(*triples)]
        for _ in range(steps):
            opt.zero_grad()
            nt = torch.randint(0, n_ent, (len(triples), neg))
            pos = -(E(h) + R(r) - E(t)).norm(dim=-1)
            negs = -(E(h).unsqueeze(1) + R(r).unsqueeze(1) - E(nt)).norm(dim=-1)
            (-F.logsigmoid(pos).mean() - F.logsigmoid(-negs).mean()).backward()
            opt.step()
        with torch.no_grad():
            resid = float((E(h) + R(r) - E(t)).norm(dim=-1).mean())
        return E, R, resid

    def typical_gap(E, n):
        d = torch.cdist(E.weight, E.weight)
        return float(d[~torch.eye(n, dtype=bool)].mean())

    # --- 대칭: r(a,b) 와 r(b,a) 가 모두 참 ---------------------------------
    pairs = [(2 * i, 2 * i + 1) for i in range(20)]
    sym = [(a, 0, b) for a, b in pairs] + [(b, 0, a) for a, b in pairs]
    Es, Rs, res_s = fit(sym, 40, 1)
    with torch.no_grad():
        rel_norm = float(Rs.weight[0].norm())
        paired = float(torch.stack([(Es.weight[a] - Es.weight[b]).norm()
                                    for a, b in pairs]).mean())
    gap_s = typical_gap(Es, 40)

    # --- 일대다: 머리 하나, 관계 하나, 꼬리 스물 --------------------------
    otm = [(0, 0, 1 + i) for i in range(20)]
    Eo, Ro, res_o = fit(otm, 21, 1)
    with torch.no_grad():
        tails = Eo.weight[1:]
        dm = torch.cdist(tails, tails)
        tail_gap = float(dm[~torch.eye(20, dtype=bool)].mean())
    gap_o = typical_gap(Es, 40)        # 같은 눈금으로 견준다

    # --- 합성: r1 다음 r2 가 r3 와 같다 (정직한 반대쪽) -------------------
    rng = np.random.default_rng(0)
    comp = []
    for _ in range(20):
        a, b, c = rng.choice(30, 3, replace=False)
        comp += [(int(a), 0, int(b)), (int(b), 1, int(c)), (int(a), 2, int(c))]
    Ec, Rc, res_c = fit(comp, 30, 3)
    with torch.no_grad():
        comp_gap = float((Rc.weight[0] + Rc.weight[1] - Rc.weight[2]).norm())
        comp_scale = float(Rc.weight.norm(dim=1).mean())

    # --- 그림이 주장하는 것을 먼저 확인한다 -------------------------------
    assert rel_norm < 0.01, "대칭 관계에서는 관계 벡터가 사라져야 한다"
    assert paired * 100 < gap_s, "짝지어진 두 엔티티가 한 점으로 모여야 한다"
    assert tail_gap * 10 < gap_o, "일대다에서 꼬리들이 한 점으로 모여야 한다"
    assert comp_gap / comp_scale < 0.1, "합성은 잘 되어야 정직한 그림이다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.3),
                                   gridspec_kw={"width_ratios": [1.1, 1]})

    # --- 왼쪽: 무엇이 한 점으로 모이는가 ----------------------------------
    items = [
        ("two entities a symmetric\nrelation links", paired, ACCENT),
        ("the 20 tails of one\none-to-many relation", tail_gap, ACCENT),
        ("two entities with no\nrelation between them", gap_s, MUTED),
    ]
    ys = np.arange(len(items))[::-1]
    ax1.barh(ys, [v for _, v, _ in items], 0.5,
             color=[c for _, _, c in items], alpha=0.84)
    for y, (lab, v, c) in zip(ys, items):
        ax1.text(v * 1.5, y, f"{v:.3f}", va="center", fontsize=10,
                 color=INK, weight="bold", family="monospace")
    ax1.set_xscale("log")
    ax1.set_xlim(2e-3, 60)
    ax1.set_yticks(ys)
    ax1.set_yticklabels([lab for lab, _, _ in items], fontsize=8.4)
    ax1.set_xlabel("distance between their embeddings (log scale)",
                   fontsize=8.8)
    ax1.set_title("what the translation assumption collapses",
                  fontsize=10, color=INK, weight="bold")
    ax1.annotate(f"{gap_s / paired:.0f}x closer than\nunrelated entities",
                 (paired, ys[0]), textcoords="offset points",
                 xytext=(132, -26), fontsize=8.4, color=ACCENT, ha="center",
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.9))

    # --- 오른쪽: 왜 그런가, 그리고 무엇은 되는가 --------------------------
    ax2.text(0.5, 0.95, "the algebra decides it before training does",
             ha="center", fontsize=9, color=INK, style="italic")
    lines = [
        (r"$e_a + e_r \approx e_b$  and  $e_b + e_r \approx e_a$", 0.80, INK),
        (r"$\Rightarrow\ 2 e_r \approx 0 \ \Rightarrow\ e_r \approx 0,\ "
         r"e_a \approx e_b$", 0.70, ACCENT),
        (rf"measured:  $\|e_r\| = {rel_norm:.4f}$", 0.60, ACCENT),
    ]
    for txt, y, col in lines:
        ax2.text(0.5, y, txt, ha="center", fontsize=10.2, color=col)
    ax2.plot([0.08, 0.92], [0.52, 0.52], color=INK, lw=0.7, alpha=0.35)
    lines2 = [
        (r"$e_a + e_{r_1} \approx e_b$,  $e_b + e_{r_2} \approx e_c$,  "
         r"$e_a + e_{r_3} \approx e_c$", 0.38, INK),
        (r"$\Rightarrow\ e_{r_1} + e_{r_2} \approx e_{r_3}$ - no conflict",
         0.27, "#4a7c59"),
        (rf"measured:  $\|e_{{r_1}} + e_{{r_2}} - e_{{r_3}}\| = "
         rf"{comp_gap:.4f}$,  against  $\|e_r\| \approx {comp_scale:.2f}$",
         0.16, "#4a7c59"),
    ]
    for txt, y, col in lines2:
        ax2.text(0.5, y, txt, ha="center", fontsize=9.6, color=col)
    ax2.text(0.5, 0.86, "symmetry", ha="center", fontsize=9.6, color=INK,
             weight="bold")
    ax2.text(0.5, 0.455, "composition", ha="center", fontsize=9.6, color=INK,
             weight="bold")
    ax2.text(0.5, 0.03, "composition survives; symmetry does not",
             ha="center", fontsize=9, color=INK, style="italic")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.0)
    ax2.axis("off")

    ax1.tick_params(labelsize=8, colors=INK)
    ax1.grid(alpha=0.13, axis="x")
    ax1.set_axisbelow(True)
    for s in ax1.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    fig.tight_layout()
    fig.text(0.5, -0.05,
             "every relation above is noise-free and perfectly consistent; "
             f"residuals after training are {res_s:.3f}, {res_o:.3f} and "
             f"{res_c:.3f}.  Nothing here is a training failure.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.savefig(HERE / "properties_catalogued_in_1860.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote properties_catalogued_in_1860.svg   symmetric: ||e_r||="
          f"{rel_norm:.4f}, pair gap {paired:.4f} vs unrelated {gap_s:.4f};  "
          f"one-to-many tail gap {tail_gap:.4f};  composition "
          f"{comp_gap:.4f} / {comp_scale:.3f}")


# === 그림으로 보는 AI: 이웃은 표본이 아니다 =================================
#
# Feld(1991) - 내 친구들은 나보다 친구가 많다. 아무 노드나 고르는 것과 아무 엣지를
# 따라가는 것은 다른 추출이고, 그 차이가 분산/평균이다. 이 쪽의 그래프에서 그
# 항등식이 정확히 맞는지, 그리고 한 계정이 보는 이웃이 전체와 얼마나 다른지를 잰다.

def a_neighbour_is_not_a_sample():
    import networkx as nx

    # 이 쪽 본문의 그래프 그대로
    G = nx.stochastic_block_model([80, 20], [[0.05, 0.01], [0.01, 0.15]],
                                  seed=42)
    n = 100
    is_bot = np.array([0] * 80 + [1] * 20)
    deg = np.array([G.degree(v) for v in range(n)], float)

    node_mean = float(deg.mean())
    nbr_mean = float((deg ** 2).sum() / deg.sum())
    identity = node_mean + float(deg.var()) / node_mean
    connected = [v for v in range(n) if G.degree(v) > 0]
    below = sum(1 for v in connected
                if deg[v] < np.mean([deg[u] for u in G.neighbors(v)]))

    bot_share_all = float(is_bot.mean())
    nbrs_of_normal = [u for v in range(80) for u in G.neighbors(v)]
    bot_share_seen = float(np.mean([is_bot[u] == 1 for u in nbrs_of_normal]))
    deg_normal, deg_bot = float(deg[:80].mean()), float(deg[80:].mean())

    # --- 그림이 주장하는 것을 먼저 확인한다 -------------------------------
    assert abs(nbr_mean - identity) < 1e-9, "Feld 항등식이 정확히 맞아야 한다"
    assert nbr_mean > node_mean, "이웃이 더 많이 이어져 있어야 한다"
    assert below > len(connected) / 2, "절반 넘는 계정이 제 이웃보다 적어야 한다"
    assert deg_bot < deg_normal, \
        "확률이 세 배라도 무리가 작아 평균 차수는 더 낮아야 한다"
    assert bot_share_seen < bot_share_all / 2, \
        "한 계정이 보는 봇 비율이 전체보다 훨씬 낮아야 한다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2),
                                   gridspec_kw={"width_ratios": [1, 1.1]})

    # --- 왼쪽: Feld ---------------------------------------------------------
    xs = np.arange(2)
    vals = [node_mean, nbr_mean]
    ax1.bar(xs, vals, 0.5, color=[MUTED, ACCENT], alpha=0.85)
    for x, v in zip(xs, vals):
        ax1.text(x, v + 0.12, f"{v:.3f}", ha="center", fontsize=11,
                 color=INK, weight="bold")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(["pick an account\nat random",
                         "follow an edge\nand look at who\nyou land on"],
                        fontsize=8.6)
    ax1.set_xlim(-0.6, 2.55)
    ax1.set_ylim(0, 6.6)
    ax1.set_ylabel("connections that account has", fontsize=8.8)
    ax1.set_title("the same graph, two ways of drawing from it",
                  fontsize=10, color=INK, weight="bold")
    ax1.annotate("", (1, node_mean), (1, nbr_mean),
                 arrowprops=dict(arrowstyle="<->", color=INK, lw=1.1))
    ax1.text(1.3, (node_mean + nbr_mean) / 2,
             f"gap = variance / mean\n= {deg.var():.2f} / {node_mean:.2f}\n"
             f"= {deg.var() / node_mean:.3f}",
             fontsize=8.4, color=INK, va="center")
    ax1.text(0.98, 5.75,
             f"{below} of the {len(connected)} connected accounts have fewer\n"
             f"connections than their own neighbours average",
             ha="center", fontsize=8.6, color=INK, style="italic")

    # --- 오른쪽: 한 계정이 보는 것 ----------------------------------------
    ys = np.array([1, 0])
    vals2 = [bot_share_all, bot_share_seen]
    ax2.barh(ys, vals2, 0.44, color=[MUTED, ACCENT], alpha=0.85)
    for y, v in zip(ys, vals2):
        ax2.text(v + 0.008, y, f"{v:.1%}", va="center", fontsize=11,
                 color=INK, weight="bold")
    ax2.set_yticks(ys)
    ax2.set_yticklabels(["bots, as a share of\nall 100 accounts",
                         "bots, as a share of what\na normal account sees"],
                        fontsize=8.6)
    ax2.set_xlim(0, 0.30)
    ax2.set_xticks(np.arange(0, 0.26, 0.05))
    ax2.set_xticklabels([f"{t:.0%}" for t in np.arange(0, 0.26, 0.05)])
    ax2.set_ylim(-1.5, 1.6)
    ax2.set_title("and what one round of message passing reaches",
                  fontsize=10, color=INK, weight="bold")
    ax2.text(0.005, -0.82,
             f"the bot group is wired at 0.15 against 0.05,\n"
             f"yet it is the smaller group: bots average\n"
             f"{deg_bot:.2f} connections, normal accounts "
             f"{deg_normal:.2f}.",
             fontsize=8.6, color=INK, va="center")

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8, colors=INK)
        ax.grid(alpha=0.13, axis="y" if ax is ax1 else "x")
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.tight_layout()
    fig.text(0.5, -0.045,
             "the page's own 100-account graph, seed 42.  Every number is "
             "counted from it directly.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.savefig(HERE / "a_neighbour_is_not_a_sample.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote a_neighbour_is_not_a_sample.svg   node {node_mean:.3f} vs "
          f"neighbour {nbr_mean:.3f} (identity {identity:.3f}); {below}/"
          f"{len(connected)} below; bots {bot_share_all:.1%} of all but "
          f"{bot_share_seen:.1%} of what a normal account sees; degrees "
          f"{deg_normal:.2f} normal / {deg_bot:.2f} bot")


if __name__ == "__main__":
    what_the_graph_holds()
    properties_catalogued_in_1860()
    a_neighbour_is_not_a_sample()
