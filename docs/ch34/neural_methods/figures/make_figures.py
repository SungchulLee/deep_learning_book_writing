"""35.2 뉴럴 넷트웍 방법 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

바깥 자료를 읽지 않는다. 숨은 인수로 만든 숨은 피드백을 쓴다 — 가린 것을
알아야 NDCG 를 잴 수 있기 때문이다. 그림 안에 만든 자료임을 적는다.

⚠️ beta 네 값마다 작은 VAE 를 학습시키므로 이 기계에서 2 분 안팎 걸린다.

실행: python docs/ch34/neural_methods/figures/make_figures.py
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

N_U, N_I, D_TRUE = 4000, 300, 16
BETAS = (0.0, 0.2, 0.5, 1.0)


# === 그림으로 보는 AI: 가장 잘하는 VAE 는 VAE 이기를 그만둔 것 ================
#
# Liang 외 2018 은 KL 항의 웨이트 beta 를 1 이 아닌 값으로 두자고 했고, 그러면
# 더 이상 하한을 최적화하는 것이 아니라고 스스로 적었다. 그 말이 맞는지
# 숨은 피드백을 만들어 직접 잰다. 수는 전부 여기서 셈한다.

def beta_sweep():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    rng = np.random.default_rng(0)
    p = rng.normal(0, 1, (N_U, D_TRUE))
    q = rng.normal(0, 1, (N_I, D_TRUE))
    logit = p @ q.T
    prob = 1 / (1 + np.exp(-(logit - np.quantile(logit, 0.93))))
    x = (rng.random((N_U, N_I)) < prob).astype(np.float32)

    x_tr = x.copy()
    held = []
    for u in range(N_U):
        idx = np.flatnonzero(x[u])
        if len(idx) < 6:
            held.append(np.array([], dtype=int))
            continue
        h = rng.choice(idx, max(1, len(idx) // 5), replace=False)
        x_tr[u, h] = 0
        held.append(h)
    x_tr_t = torch.tensor(x_tr)

    class MultVAE(nn.Module):
        def __init__(self, k=32):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(N_I, 128), nn.Tanh(),
                                     nn.Linear(128, 2 * k))
            self.dec = nn.Sequential(nn.Linear(k, 128), nn.Tanh(),
                                     nn.Linear(128, N_I))
            self.k = k

        def forward(self, inp):
            h = self.enc(F.normalize(inp, dim=1))
            mu, lv = h[:, :self.k], h[:, self.k:]
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) \
                if self.training else mu
            return self.dec(z), mu, lv

    def ndcg(scores, k=10):
        scores = scores.clone()
        scores[x_tr_t > 0] = -1e9
        top = scores.topk(k, 1).indices.numpy()
        out = []
        for u in range(N_U):
            if len(held[u]) == 0:
                continue
            rel = np.isin(top[u], held[u]).astype(float)
            dcg = (rel / np.log2(np.arange(2, k + 2))).sum()
            ideal = (1 / np.log2(np.arange(2, min(k, len(held[u])) + 2))).sum()
            out.append(dcg / ideal)
        return float(np.mean(out))

    scores = []
    for beta in BETAS:
        torch.manual_seed(1)
        model = MultVAE()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(60):
            perm = torch.randperm(N_U)
            for i in range(0, N_U, 512):
                xb = x_tr_t[perm[i:i + 512]]
                model.train()
                lg, mu, lv = model(xb)
                ll = -(F.log_softmax(lg, 1) * xb).sum(1).mean()
                kl = (-0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(1)).mean()
                (ll + beta * kl).backward()
                opt.step()
                opt.zero_grad()
        model.eval()
        with torch.no_grad():
            sc, _, _ = model(x_tr_t)
        scores.append(ndcg(sc))

    best = int(np.argmax(scores))
    # 그림이 주장하는 것을 먼저 확인한다
    assert BETAS[best] not in (0.0, 1.0), \
        "가장 좋은 beta 가 0 이나 1 이면 이 그림의 이야기가 없다"
    assert scores[-1] < max(scores), "제대로 된 VAE 가 이기면 이야기가 없다"

    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    cols = [ACCENT if i == best else INK for i in range(len(BETAS))]
    ax.bar(range(len(BETAS)), scores, color=cols, width=0.56, alpha=0.88)
    for i, s in enumerate(scores):
        ax.text(i, s + 0.004, f"{s:.4f}", ha="center", fontsize=9.6,
                color=INK, weight="bold")
    ax.set_xticks(range(len(BETAS)))
    notes = {0.0: "\n(a plain autoencoder)", 1.0: "\n(a proper VAE)"}
    labels = [f"beta = {b:g}" + notes.get(b, "\n(best)" if i == best else "")
              for i, b in enumerate(BETAS)]
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("NDCG@10 on held-out interactions", fontsize=9.5)
    ax.set_ylim(min(scores) * 0.90, max(scores) * 1.05)
    ax.set_title("the winner sits between the two named models",
                 fontsize=10.5, color=INK, weight="bold", pad=8)
    ax.grid(alpha=0.13, axis="y")
    ax.tick_params(labelsize=8, colors=INK)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    fig.text(0.5, -0.09,
             "made-up implicit feedback, 20% of each user's clicks held out. "
             "beta is the weight on the KL term.",
             ha="center", fontsize=8.5, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "beta_sweep.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote beta_sweep.svg   "
          + "  ".join(f"b={b:g}: {s:.4f}" for b, s in zip(BETAS, scores)))


if __name__ == "__main__":
    beta_sweep()
