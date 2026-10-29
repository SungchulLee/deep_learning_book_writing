"""2장 군집화 지표 쪽의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

수는 모두 여기서 실제로 돌려 얻는다.

실행: python docs/ch02/metrics/figures/make_figures.py
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
BLUE = "#2d6a8f"


def style(ax):
    ax.tick_params(labelsize=8, colors=INK)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)


# === 그림으로 보는 AI: 실루엣이 고르는 k 와 옳은 k ===========================
#
# 볼록한 덩어리에서는 실루엣이 참 군집 수를 맞춘다. 비볼록한 moons 에서는
# 가장 높은 실루엣이 ARI 가 가장 낮은 자리를 가리킨다. 수는 여기서 잰다.

def silhouette_picks():
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_moons, make_blobs
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from sklearn.preprocessing import StandardScaler

    cases = [
        ("blobs - convex, true k = 3", make_blobs(
            n_samples=600, centers=3, cluster_std=1.0, random_state=42), 3),
        ("moons - not convex, true k = 2", make_moons(
            n_samples=600, noise=0.06, random_state=42), 2),
    ]
    ks = list(range(2, 9))
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))

    for ax, (title, (X, y), true_k) in zip(axes, cases):
        X = StandardScaler().fit_transform(X)
        sil, ari = [], []
        for k in ks:
            lab = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).labels_
            sil.append(silhouette_score(X, lab))
            ari.append(adjusted_rand_score(y, lab))
        sil, ari = np.array(sil), np.array(ari)
        kbest = ks[int(sil.argmax())]

        ax.plot(ks, sil, "o-", color=ACCENT, linewidth=2.0, ms=5.5,
                label="silhouette (no labels needed)")
        ax.plot(ks, ari, "s--", color=BLUE, linewidth=1.8, ms=5,
                label="ARI against the true labels")
        ax.axvline(true_k, color=INK, linestyle=":", linewidth=1.2,
                   alpha=0.7)
        ax.annotate(f"true k = {true_k}", xy=(true_k, 0.04),
                    xytext=(true_k + 0.25, 0.04), fontsize=8.5, color=INK)
        ax.plot([kbest], [sil.max()], "*", color=ACCENT, ms=17, zorder=5)
        ax.annotate(f"silhouette picks k = {kbest}",
                    xy=(kbest, sil.max()),
                    xytext=(-10 if kbest > 5 else 10, 24),
                    textcoords="offset points", fontsize=8.5, color=ACCENT,
                    ha="right" if kbest > 5 else "left",
                    arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.9))
        ax.set_xlabel("k", fontsize=9.5)
        ax.set_ylim(0, 1.08)
        ax.set_title(title, fontsize=10.5, color=INK, weight="bold")
        ax.legend(fontsize=8, loc="upper right", framealpha=0.92)
        ax.grid(alpha=0.13)
        style(ax)
    axes[0].set_ylabel("score", fontsize=9.5)

    fig.text(0.5, -0.04,
             "same metric, same k-means, same seed. only the shape of the "
             "clusters changes.",
             ha="center", fontsize=8.5, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "silhouette_picks.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote silhouette_picks.svg")


if __name__ == "__main__":
    silhouette_picks()
