"""35.5 금융 응용 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

바깥 시장 자료를 읽지 않는다. 참 분포를 알아야 재앙 확률을 셀 수 있으므로
**만든 자산**을 쓴다. 그림 안에 만든 자료임을 적는다.

실행: python docs/ch34/finance/figures/make_figures.py
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

N_DRAW = 600_000
DISASTER = -0.15


# === 그림으로 보는 AI: 흔들림이 못 보는 꼬리 =================================
#
# 로이 1952 의 안전제일 기준(재앙 수준 아래로 떨어질 확률을 가장 작게)과
# 평균-분산 기준이 **다른 꾸러미**를 고르는 자리를 만든다. B 는 평소 아주
# 얌전한데 드물게 크게 무너진다. 수는 전부 여기서 셈한다.

def safety_first():
    rng = np.random.default_rng(1)
    mu = np.array([0.08, 0.11, 0.07])
    sd = np.array([0.14, 0.05, 0.12])
    x = rng.normal(mu, sd, (N_DRAW, 3))
    crash = rng.random(N_DRAW) < 0.03
    x[crash, 1] -= 0.70

    best_s = best_r = None
    for w1 in np.arange(0, 1.001, 0.02):
        for w2 in np.arange(0, 1.001 - w1 + 1e-9, 0.02):
            w = np.array([w1, w2, 1 - w1 - w2])
            r = x @ w
            sh = r.mean() / r.std()
            pr = float((r < DISASTER).mean())
            if best_s is None or sh > best_s[0]:
                best_s = (sh, w, pr, r.mean())
            if best_r is None or pr < best_r[0]:
                best_r = (pr, w, sh, r.mean())

    sh_s, w_s, p_s, m_s = best_s
    p_r, w_r, sh_r, m_r = best_r

    # 그림이 주장하는 것을 먼저 확인한다
    assert w_s[1] > w_r[1] * 1.5, "두 기준이 B 비중을 비슷하게 고르면 이야기가 없다"
    assert p_r < p_s / 2, "재앙 확률이 크게 줄지 않으면 이야기가 없다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.2),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    idx = np.arange(3)
    ax1.bar(idx - 0.19, w_s, width=0.36, color=INK, alpha=0.82,
            label="mean-variance (best Sharpe)")
    ax1.bar(idx + 0.19, w_r, width=0.36, color=ACCENT, alpha=0.88,
            label="Roy's safety first")
    ax1.set_xticks(idx)
    ax1.set_xticklabels(["A", "B\n(calm, then a cliff)", "C"], fontsize=9)
    ax1.set_ylabel("weight in the portfolio", fontsize=9.5)
    ax1.set_title("the two criteria pick differently", fontsize=10,
                  color=INK, weight="bold")
    ax1.legend(fontsize=8.4, framealpha=0.92)
    ax1.grid(alpha=0.13, axis="y")

    bars = [p_s, p_r]
    ax2.bar([0, 1], bars, color=[INK, ACCENT], width=0.5, alpha=0.88)
    for i, v in enumerate(bars):
        ax2.text(i, v + max(bars) * 0.03, f"{v * 100:.2f}%", ha="center",
                 fontsize=10.5, color=INK, weight="bold")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["mean-variance", "safety first"], fontsize=9)
    ax2.set_ylabel(f"chance of losing more than {abs(DISASTER):.0%}",
                   fontsize=9.5)
    ax2.set_ylim(0, max(bars) * 1.3)
    ax2.set_title(f"{(1 - p_r / p_s) * 100:.0f}% less often", fontsize=10,
                  color=ACCENT, weight="bold")
    ax2.grid(alpha=0.13, axis="y")

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.text(0.5, -0.06,
             f"made-up assets.  the price is expected return "
             f"{m_s:.4f} -> {m_r:.4f} and Sharpe {sh_s:.3f} -> {sh_r:.3f}.",
             ha="center", fontsize=8.5, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "safety_first.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote safety_first.svg   B {w_s[1]:.2f}->{w_r[1]:.2f}   "
          f"disaster {p_s:.4f}->{p_r:.4f}   sharpe {sh_s:.3f}->{sh_r:.3f}")




# === 그림으로 보는 AI: 흔한 말이 닮음을 가로채는 자리 =========================
#
# 스파크 존스 1972 의 요지 — 드문 말이 흔한 말보다 값지다 — 를 실제 글에서
# 잰다. IDF 없이 재면 문서 닮음의 대부분이 "the, and, of" 같은 말에서 온다.
#
# IMDB 평을 읽으므로 **저장소 뿌리에서** 돌린다. 30 초 안팎 걸린다.

N_DOCS, N_COMMON, MIN_DF = 3000, 20, 5


def idf_effect():
    import glob
    import math
    import re
    from collections import Counter

    files = (sorted(glob.glob("./data/aclImdb/train/pos/*.txt"))[:N_DOCS // 2]
             + sorted(glob.glob("./data/aclImdb/train/neg/*.txt"))[:N_DOCS // 2])
    docs = [re.findall(r"[a-z]{2,}",
                       open(f, encoding="utf-8", errors="ignore").read().lower())
            for f in files]
    n = len(docs)
    df = Counter()
    for d in docs:
        df.update(set(d))
    vocab = [w for w, c in df.items() if c >= MIN_DF]
    pos = {w: i for i, w in enumerate(vocab)}
    idf = np.array([math.log(n / df[w]) for w in vocab])

    tf = np.zeros((n, len(vocab)), dtype=np.float32)
    for i, d in enumerate(docs):
        for w, v in Counter(x for x in d if x in pos).items():
            tf[i, pos[w]] = v

    order = np.argsort(-np.array([df[w] for w in vocab]))
    common = order[:N_COMMON]

    def common_share(weights):
        wn = weights / (np.linalg.norm(weights, axis=1, keepdims=True) + 1e-9)
        rng = np.random.default_rng(0)
        q = rng.choice(n, 300, replace=False)
        sims = wn[q] @ wn.T
        contrib = (wn[q][:, common][:, :, None]
                   * wn.T[common][None, :, :]).sum(1)
        mask = np.ones_like(sims, dtype=bool)
        mask[np.arange(300), q] = False
        return float(contrib[mask].sum() / sims[mask].sum())

    s_tf, s_tfidf = common_share(tf), common_share(tf * idf)

    # 그림이 주장하는 것을 먼저 확인한다
    assert s_tf > 0.6, "IDF 없이도 흔한 말이 지배하지 않으면 이 그림의 이야기가 없다"
    assert s_tfidf < s_tf / 3, "IDF 가 그 몫을 크게 줄이지 못하면 이야기가 없다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.2),
                                   gridspec_kw={"width_ratios": [1, 1.1]})

    bars = [s_tf, s_tfidf]
    ax1.bar([0, 1], bars, color=[INK, ACCENT], width=0.5, alpha=0.88)
    for i, v in enumerate(bars):
        ax1.text(i, v + 0.02, f"{v * 100:.1f}%", ha="center", fontsize=11,
                 color=INK, weight="bold")
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["counts only", "counts x idf"], fontsize=9.5)
    ax1.set_ylabel(f"share of similarity from the {N_COMMON} commonest words",
                   fontsize=9)
    ax1.set_ylim(0, 1.0)
    ax1.set_title("what the similarity is actually made of", fontsize=10,
                  color=INK, weight="bold")
    ax1.grid(alpha=0.13, axis="y")

    top = [vocab[i] for i in common[:12]]
    ax2.axis("off")
    ax2.set_title(f"the {N_COMMON} commonest words", fontsize=10, color=INK,
                  weight="bold")
    for k, w in enumerate(top):
        r, c = divmod(k, 3)
        ax2.text(0.06 + c * 0.32, 0.80 - r * 0.17, w, fontsize=13,
                 color=INK, family="monospace")
    ax2.text(0.5, 0.06,
             "without idf, 'similar documents' means\n"
             "documents that use these at similar rates",
             ha="center", fontsize=9, color=ACCENT)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    ax1.tick_params(labelsize=8, colors=INK)
    for sp in ax1.spines.values():
        sp.set_color(INK)
        sp.set_linewidth(0.7)

    fig.text(0.5, -0.05,
             f"{n:,} film reviews, any word appearing in at least {MIN_DF} "
             f"of them.",
             ha="center", fontsize=8.5, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "idf_effect.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote idf_effect.svg   tf {s_tf:.3f}  tfidf {s_tfidf:.3f}")


if __name__ == "__main__":
    idf_effect()
    safety_first()
