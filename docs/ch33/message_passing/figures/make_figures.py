"""34.2 메시지 패싱 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 내려받지 않으므로 어디서 돌려도 된다.

실행: python docs/ch33/message_passing/figures/make_figures.py
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


# === 그림으로 보는 AI: 세 자리만 바꾸면 세 모델이 된다 ========================
#
# 뼈대 하나(메시지 -> 집계 -> 갱신)에 세 자리를 갈아 끼워 GCN·GIN·GraphSAGE 를
# 만들고, 각 모델을 직접 짠 것과 맞춰 본다. 어긋남은 float32 반올림 수준이다
# (연산을 묶는 차례가 달라 정확히 0 은 아니다 — 그대로 적는다).

def one_skeleton():
    import networkx as nx
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    G = nx.karate_club_graph()
    n, d = G.number_of_nodes(), 6
    A = torch.zeros(n, n)
    for u, v in G.edges():
        A[u, v] = A[v, u] = 1.0
    H = torch.randn(n, d)
    W = torch.randn(d, d) * 0.3

    def mpnn(H, msg, agg, upd):          # 뼈대. 이 세 줄이 전부다.
        return upd(H, agg(msg(H)))

    At = A + torch.eye(n)
    Dm = At.sum(1).pow(-0.5)
    deg = A.sum(1).clamp(min=1)
    eps = torch.tensor(0.17)
    mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
    Wc = torch.randn(2 * d, d) * 0.3

    gaps = {}
    gen = mpnn(H,
               lambda H: (At.unsqueeze(-1)
                          * (Dm.unsqueeze(1) * Dm.unsqueeze(0)).unsqueeze(-1)
                          * (H @ W).unsqueeze(0)),
               lambda M: M.sum(1),
               lambda H, m: torch.relu(m))
    direct = torch.relu((Dm.unsqueeze(1) * At * Dm.unsqueeze(0)) @ (H @ W))
    gaps["GCN"] = (gen - direct).abs().max().item()

    gen = mpnn(H, lambda H: A.unsqueeze(-1) * H.unsqueeze(0),
               lambda M: M.sum(1),
               lambda H, m: mlp((1 + eps) * H + m))
    gaps["GIN"] = (gen - mlp((1 + eps) * H + A @ H)).abs().max().item()

    gen = mpnn(H, lambda H: A.unsqueeze(-1) * H.unsqueeze(0),
               lambda M: M.sum(1) / deg.unsqueeze(1),
               lambda H, m: torch.relu(torch.cat([H, m], 1) @ Wc))
    direct = torch.relu(torch.cat([H, (A @ H) / deg.unsqueeze(1)], 1) @ Wc)
    gaps["GraphSAGE"] = (gen - direct).abs().max().item()

    # --- 그림이 주장하는 것을 먼저 확인한다 -------------------------------
    assert all(g < 1e-5 for g in gaps.values()), \
        "뼈대가 직접 구현과 맞아야 이 그림의 이야기가 성립한다"
    assert any(g > 0 for g in gaps.values()), \
        "정확히 0 이면 반올림 이야기를 할 수 없다"

    fig, ax = plt.subplots(figsize=(10.6, 4.3))

    cols = [0.015, 0.265, 0.505, 0.735]
    heads = ["", "message", "aggregate", "update"]
    for x, h in zip(cols, heads):
        ax.text(x, 0.88, h, fontsize=9.5, color=INK, weight="bold")
    ax.plot([0.0, 0.93], [0.845, 0.845], color=INK, linewidth=0.9)

    rows = [
        ("GCN", "W h_u / sqrt(dv du)", "sum", "relu(m)"),
        ("GIN", "h_u", "sum", "MLP((1+e) h_v + m)"),
        ("GraphSAGE", "h_u", "mean", "relu([h_v || m] Wc)"),
    ]
    for i, (name, a, b, c) in enumerate(rows):
        y = 0.72 - i * 0.165
        ax.add_patch(plt.Rectangle((0.0, y - 0.058), 0.93, 0.118,
                                   facecolor=ACCENT if i % 2 == 0 else INK,
                                   alpha=0.08, edgecolor="none"))
        ax.text(cols[0], y, name, fontsize=9.5, color=ACCENT, va="center",
                weight="bold")
        for x, t in zip(cols[1:], (a, b, c)):
            ax.text(x, y, t, fontsize=8.8, color=INK, va="center",
                    family="monospace")

    ax.text(0.465, 0.185, "one skeleton:   update( aggregate( message(H) ) )",
            ha="center", fontsize=10, color=ACCENT, family="monospace",
            weight="bold")
    worst = max(gaps.values())
    ax.text(0.465, 0.085,
            f"each row matches a direct implementation to {worst:.0e}  "
            f"- float32 round-off, not zero",
            ha="center", fontsize=8.8, color=INK)
    ax.text(0.465, 0.005, "only the three slots changed", ha="center",
            fontsize=8.8, color=INK, style="italic", alpha=0.85)

    ax.set_xlim(-0.01, 0.95)
    ax.set_ylim(-0.03, 0.95)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(HERE / "one_skeleton.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote one_skeleton.svg   " +
          "  ".join(f"{k} {v:.1e}" for k, v in gaps.items()))


# === 그림으로 보는 AI: 게이트가 치운 것은 기억이 아니라 제약이었다 ===========
#
# Scarselli 의 GNN 은 전파를 고정점까지 돌렸고, 그러려면 전파가 축약 사상이어야
# 했다. 그 제약이 둘을 강제한다 - 고정점이 초기값과 무관해지고, 영향이 거리에
# 따라 rho^delta 로 줄어든다. Li 등의 Appendix A 를 그대로 재어 본다.

def what_the_gate_removed():
    import networkx as nx

    n = 21
    A = nx.to_numpy_array(nx.path_graph(n), weight=None)
    S = A / np.maximum(A.sum(1, keepdims=True), 1)
    b = np.zeros(n)
    b[0] = 1.0
    rng = np.random.default_rng(0)

    # --- (1) 축약이면 고정점은 출발점을 지운다 -----------------------------
    mu_demo = 0.9
    starts = [np.zeros(n)] + [rng.normal(scale=10.0, size=n) for _ in range(49)]
    traj, fixed = [], []
    for h0 in starts:
        h, path = h0.copy(), [h0.copy()]
        for _ in range(2000):
            h = mu_demo * S @ h + b
            if len(path) < 400:
                path.append(h.copy())
        traj.append(np.array(path))
        fixed.append(h)
    F = np.array(fixed)
    spread = float(np.abs(F - F[0]).max())
    start_spread = float(np.abs(np.array(starts) - starts[0]).max())
    assert spread < 1e-12 < start_spread, "출발점은 달라도 고정점은 같아야 한다"

    # --- (2) 영향은 거리에 따라 기하적으로 줄고, 반복으로는 늘지 않는다 ----
    curves = {}
    for mu in (0.9, 0.7, 0.5):
        h = np.zeros(n)
        it = 0
        while True:
            new = mu * S @ h + b
            it += 1
            if np.abs(new - h).max() < 1e-15:
                h = new
                break
            h = new
        more = h.copy()
        for _ in range(10000):
            more = mu * S @ more + b
        curves[mu] = (h / h[0], it, float(np.abs(more - h).max()))
        assert curves[mu][2] < 1e-13, "고정점은 더 돌려도 움직이지 않아야 한다"
    assert curves[0.9][0][20] < 1e-3, "20 홉이면 영향이 사라져야 한다"

    # --- (3) 제약이 없으면 그대로 나를 수 있다 ------------------------------
    g = np.zeros(n)
    g[0] = 1.0
    open_gate = [1.0]
    for t in range(1, n):
        g = np.maximum(g, A @ g)
        open_gate.append(float(g[t]))
    assert min(open_gate) == 1.0, "열린 게이트는 값을 그대로 날라야 한다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2),
                                   gridspec_kw={"width_ratios": [1, 1.1]})

    # --- 왼쪽: 쉰 개의 출발점이 한 점으로 ----------------------------------
    star = F[0]
    for k, path in enumerate(traj):
        d = np.abs(path - star).max(1)
        d = np.maximum(d, 1e-17)
        ax1.plot(np.arange(len(d)), d, lw=0.8,
                 color=INK if k else ACCENT, alpha=0.95 if k == 0 else 0.22,
                 zorder=3 if k == 0 else 1)
    ax1.axhline(spread, color=MUTED, lw=1.3, ls="--")
    ax1.text(8, spread * 3.2,
             f"all 50 land within {spread:.0e} of each other",
             ha="left", fontsize=8.6, color=MUTED)
    ax1.plot([], [], lw=1.4, color=ACCENT, label="started from all zeros")
    ax1.plot([], [], lw=1.4, color=INK, alpha=0.4,
             label=f"49 random starts, up to {start_spread:.0f} away")
    ax1.set_yscale("log")
    ax1.set_ylim(1e-17, 1e3)
    ax1.set_xlim(0, 400)
    ax1.set_xlabel("iteration", fontsize=8.8)
    ax1.set_ylabel("distance from the fixed point", fontsize=8.8)
    ax1.set_title("a contraction erases where you began", fontsize=10,
                  color=INK, weight="bold")
    ax1.legend(fontsize=8.2, loc="upper right", framealpha=0.92)

    # --- 오른쪽: 거리에 따른 영향 ------------------------------------------
    hops = np.arange(n)
    for mu, col in zip((0.9, 0.7, 0.5), (ACCENT, MUTED, INK)):
        r, it, drift = curves[mu]
        ax2.plot(hops, np.maximum(r, 1e-16), "o-", ms=3.4, lw=1.4, color=col,
                 label=rf"contraction, $\rho={mu}$   (fixed point at step {it})")
    ax2.plot(hops, open_gate, "s-", ms=3.4, lw=1.6, color="#4a7c59",
             label="a gate held open - no contraction asked for")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-13, 6.0)
    ax2.set_xlim(-0.5, n - 0.5)
    ax2.set_xticks(range(0, n, 4))
    ax2.set_xlabel("distance in hops from the node carrying the signal",
                   fontsize=8.8)
    ax2.set_ylabel("influence, relative to the source", fontsize=8.8)
    ax2.set_title("and its reach is set by the constraint, not by the "
                  "step count", fontsize=9.6, color=INK, weight="bold")
    ax2.legend(fontsize=8.0, loc="lower left", framealpha=0.92)


    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8, colors=INK)
        ax.grid(alpha=0.13)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.tight_layout()
    fig.text(0.5, -0.05,
             f"path of {n} nodes, one unit of signal held at node 0.  Running "
             f"{10000:,} further iterations past the fixed point moves it by "
             f"at most {max(c[2] for c in curves.values()):.0e} - "
             "more steps buy no more reach.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.savefig(HERE / "what_the_gate_removed.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote what_the_gate_removed.svg   50 starts (spread "
          f"{start_spread:.0f}) -> one fixed point within {spread:.1e};  " +
          "; ".join(f"rho={m}: node20 {c[0][20]:.2e}, settles at {c[1]}, "
                    f"drift {c[2]:.0e}" for m, c in curves.items()))


if __name__ == "__main__":
    one_skeleton()
    what_the_gate_removed()
