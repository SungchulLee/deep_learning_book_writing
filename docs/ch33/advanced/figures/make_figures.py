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


# === 그림으로 보는 AI: 같은 깊이인데 보는 범위가 다르다 =======================
#
# JK 논문의 출발점은 "같은 레이어 수인데도 노드마다 끌어오는 범위가 다르다"
# 는 관찰이다. 랜덤워크로 그것을 직접 잰다. 차수만으로 설명되지 않는다는
# 것까지 상관계수로 확인한다.

def reach_varies():
    import networkx as nx

    G = nx.karate_club_graph()
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G)
    P = A / A.sum(1, keepdims=True)
    deg = dict(G.degree())
    hub = max(deg, key=deg.get)
    peri = min(deg, key=deg.get)

    ks = [1, 2, 3, 4]
    reach_hub, reach_per = [], []
    for k in ks:
        Pk = np.linalg.matrix_power(P, k)
        reach_hub.append(int((Pk[hub] > 1e-9).sum()))
        reach_per.append(int((Pk[peri] > 1e-9).sum()))

    P2 = np.linalg.matrix_power(P, 2)
    reach2 = [int((P2[v] > 1e-9).sum()) for v in range(n)]
    corr = float(np.corrcoef([deg[v] for v in range(n)], reach2)[0, 1])

    # --- 그림이 주장하는 것을 먼저 확인한다 -------------------------------
    assert max(reach2) - min(reach2) > 10, "같은 k 에서 범위가 갈려야 한다"
    assert reach_hub[0] > reach_per[0], "허브가 먼저 퍼져야 한다"
    assert abs(corr) < 0.8, "차수만으로 설명되면 이 그림의 이야기가 약해진다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.0))

    ax1.plot(ks, reach_hub, "o-", color=ACCENT, linewidth=2.1, ms=6,
             label=f"node {hub}  (degree {deg[hub]})")
    ax1.plot(ks, reach_per, "s--", color=MUTED, linewidth=2.1, ms=6,
             label=f"node {peri}  (degree {deg[peri]})")
    ax1.axhline(n, color=INK, linestyle=":", linewidth=1.0, alpha=0.6)
    ax1.text(ks[0], n + 0.6, f"the whole graph ({n})", fontsize=8.4,
             color=INK, alpha=0.75)
    ax1.set_xticks(ks)
    ax1.set_xlabel("layers k", fontsize=9)
    ax1.set_ylabel("nodes reached", fontsize=9)
    ax1.set_ylim(0, n + 4)
    ax1.set_title("two nodes, the same depth", fontsize=10, color=INK,
                  weight="bold")
    ax1.legend(fontsize=8.4, loc="lower right", framealpha=0.9)
    ax1.grid(alpha=0.13)

    ax2.hist(reach2, bins=range(min(reach2), max(reach2) + 2), color=ACCENT,
             alpha=0.82)
    ax2.set_xlabel("nodes reached at k = 2", fontsize=9)
    ax2.set_ylabel("how many nodes", fontsize=9)
    ax2.set_title("every node, at one depth", fontsize=10, color=INK,
                  weight="bold")
    ax2.text(0.97, 0.93,
             f"from {min(reach2)} to {max(reach2)} of {n}\n"
             f"correlation with degree {corr:.2f}",
             transform=ax2.transAxes, ha="right", va="top", fontsize=8.8,
             color=INK,
             bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                       edgecolor=INK, alpha=0.9))
    ax2.grid(alpha=0.13, axis="y")

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.text(0.5, -0.03,
             "one depth does not buy one receptive field - and degree alone "
             "does not predict which.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "reach_varies.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote reach_varies.svg   hub {reach_hub} peri {reach_per}, "
          f"k=2 spread {min(reach2)}-{max(reach2)}, corr {corr:.3f}")


if __name__ == "__main__":
    two_causes()
    reach_varies()
