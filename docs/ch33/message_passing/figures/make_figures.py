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


if __name__ == "__main__":
    one_skeleton()
