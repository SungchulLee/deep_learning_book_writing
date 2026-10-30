"""34.5 깊이와 확장 구조 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 내려받지 않으므로 어디서 돌려도 된다.

실행: python docs/ch33/advanced/figures/make_figures.py
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


# === 그림으로 보는 AI: 같은 곡선, 다른 까닭 ===================================
#
# 이 쪽 머리말이 "거리가 작아졌다는 사실만으로 원인을 단정할 수 없다"고 적어
# 두었다. 그것을 수로 보인다 — 전파로 정말 매끄러워진 경우와, 그냥 전체 크기를
# 줄인 경우가 **평균 거리에서는 구별되지 않는다.** 크기를 맞추고 재면 갈린다.

def two_causes():
    import networkx as nx
    import torch

    torch.manual_seed(0)
    G = nx.karate_club_graph()
    n = G.number_of_nodes()
    A = torch.zeros(n, n)
    for u, v in G.edges():
        A[u, v] = A[v, u] = 1.0
    A = A + torch.eye(n)                       # 자기 루프
    deg = A.sum(1)
    S = A / torch.sqrt(deg).unsqueeze(1) / torch.sqrt(deg).unsqueeze(0)
    H0 = torch.randn(n, 8)

    def mad(H):
        return (H.unsqueeze(0) - H.unsqueeze(1)).norm(dim=-1).mean().item()

    def mad_unit(H):                            # 전체 크기를 1 로 맞춘 뒤
        H = H - H.mean(0, keepdim=True)
        s = H.norm(dim=1).pow(2).mean().sqrt()
        return mad(H / (s + 1e-12))

    ks = list(range(0, 15))
    c = 0.78                                    # 크기만 줄이는 쪽의 비율
    prop = [mad(torch.matrix_power(S, k) @ H0) for k in ks]
    scal = [mad(H0 * (c ** k)) for k in ks]
    prop_u = [mad_unit(torch.matrix_power(S, k) @ H0) for k in ks]
    scal_u = [mad_unit(H0 * (c ** k)) for k in ks]

    # --- 그림이 주장하는 것을 먼저 확인한다 -------------------------------
    assert prop[-1] < prop[0] * 0.2 and scal[-1] < scal[0] * 0.2, \
        "둘 다 평균 거리가 무너져야 한다"
    assert max(scal_u) - min(scal_u) < 1e-6, \
        "크기만 줄인 쪽은 정규화하면 변하지 않아야 한다"
    assert prop_u[-1] < prop_u[0] - 1e-3, \
        "전파한 쪽은 정규화해도 줄어야 한다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.0))

    for ax, a, b, title, note in (
            (ax1, prop, scal, "mean pairwise distance",
             "both collapse - the metric cannot tell them apart"),
            (ax2, prop_u, scal_u, "the same, with scale divided out",
             "now only one of them actually changed shape")):
        ax.plot(ks, a, "o-", color=ACCENT, linewidth=2.0, ms=4.5,
                label="propagation  S^k H")
        ax.plot(ks, b, "s--", color=MUTED, linewidth=2.0, ms=4.5,
                label="weights shrinking  c^k H")
        ax.set_xlabel("layers k", fontsize=9)
        ax.set_title(title, fontsize=10, color=INK, weight="bold")
        ax.grid(alpha=0.13)
        ax.legend(fontsize=8.2, framealpha=0.9)
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)
        ax.text(0.5, -0.255, note, transform=ax.transAxes, ha="center",
                fontsize=8.4, color=INK, alpha=0.9)

    ax1.set_ylabel("mean pairwise distance", fontsize=9)
    ax2.set_ylabel("distance at unit scale", fontsize=9)
    ax2.set_ylim(min(prop_u) - 0.06, max(scal_u) + 0.06)

    fig.tight_layout()
    fig.savefig(HERE / "two_causes.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote two_causes.svg   raw {prop[0]:.3f}->{prop[-1]:.3f} / "
          f"{scal[0]:.3f}->{scal[-1]:.3f}   unit {prop_u[0]:.4f}->"
          f"{prop_u[-1]:.4f} / flat {scal_u[0]:.4f}")


if __name__ == "__main__":
    two_causes()
