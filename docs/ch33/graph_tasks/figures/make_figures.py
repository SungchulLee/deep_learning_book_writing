"""34.6 그래프 수준 과제 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 내려받지 않으므로 어디서 돌려도 된다.

실행: python docs/ch33/graph_tasks/figures/make_figures.py
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


# === 그림으로 보는 AI: 평균과 최대는 크기를 지운다 ============================
#
# 같은 노드 피처를 가진 그래프와 그것을 그대로 복제해 키운 그래프를 읽어 낸다.
# 합만 둘을 가르고 평균·최대는 사실상 같은 값을 돌려준다 (최대는 정확히 0,
# 평균은 나누는 수가 달라 float32 반올림만 남는다). 직접 잰다.

def size_blindness():
    import torch

    torch.manual_seed(0)
    h = torch.randn(1, 4)
    small = h.repeat(3, 1)          # 노드 3
    big = h.repeat(12, 1)           # 같은 피처, 노드 12

    reads = {"sum": lambda X: X.sum(0),
             "mean": lambda X: X.mean(0),
             "max": lambda X: X.max(0).values}
    gaps = {k: float((f(small) - f(big)).abs().max()) for k, f in reads.items()}

    torch.manual_seed(1)
    X1 = torch.randn(5, 4)
    X2 = torch.cat([X1, X1])        # 똑같은 다중집합을 두 배로
    gaps2 = {k: float((f(X1) - f(X2)).abs().max()) for k, f in reads.items()}

    # --- 그림이 주장하는 것을 먼저 확인한다 -------------------------------
    # 평균은 **정확히** 0 이 아니다 — 3 으로 나누느냐 12 로 나누느냐에서
    # float32 반올림이 남는다. 최대만 정확히 0 이다. 그대로 적는다.
    assert gaps["mean"] < 1e-5 and gaps["max"] == 0.0, \
        "평균과 최대는 크기를 못 보아야 한다"
    assert gaps["sum"] > 1.0, "합은 크기를 보아야 한다"
    assert gaps2["mean"] < 1e-5 and gaps2["max"] == 0.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.0),
                                   gridspec_kw={"width_ratios": [1, 1.15]})

    # --- 왼쪽: 같은 피처, 다른 크기 ---------------------------------------
    for k, (cnt, cx, lab) in enumerate(((3, 0.26, "3 nodes"),
                                        (12, 0.74, "12 nodes"))):
        cols = 2 if cnt == 3 else 4
        for i in range(cnt):
            r, c = divmod(i, cols)
            ax1.add_patch(plt.Circle((cx - 0.055 * (cols - 1) / 2 + c * 0.055,
                                      0.62 - r * 0.085), 0.021,
                                     facecolor=ACCENT, alpha=0.85,
                                     edgecolor="white", linewidth=0.9))
        ax1.text(cx, 0.80, lab, ha="center", fontsize=10, color=INK,
                 weight="bold")
    ax1.text(0.5, 0.20, "every node carries the\nsame feature vector",
             ha="center", fontsize=9, color=INK)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0.05, 0.92)
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_title("one graph, and a bigger copy of it", fontsize=10,
                  color=INK, weight="bold")

    # --- 오른쪽: 세 readout 이 본 차이 ------------------------------------
    names = list(reads)
    vals = [gaps[k] for k in names]
    cols = [ACCENT if v > 1e-5 else MUTED for v in vals]
    ax2.barh(range(3), [max(v, 0.0) for v in vals], color=cols, height=0.5,
             alpha=0.85)
    ax2.set_yticks(range(3))
    ax2.set_yticklabels(names, fontsize=10)
    ax2.invert_yaxis()
    for i, v in enumerate(vals):
        if v > 1e-5:
            ax2.text(v * 0.98, i, f"{v:.2f} ", va="center", ha="right",
                     fontsize=9.5, color="white", weight="bold")
        else:
            tag = "  exactly 0" if v == 0.0 else f"  {v:.0e}  (round-off)"
            ax2.text(max(vals) * 0.02, i, tag, va="center",
                     fontsize=9.5, color=MUTED, weight="bold")
    ax2.set_xlabel("difference between the two readouts", fontsize=9)
    ax2.set_xlim(0, max(vals) * 1.12)
    ax2.set_title("only one of them notices", fontsize=10, color=INK,
                  weight="bold")
    ax2.grid(alpha=0.13, axis="x")
    ax2.tick_params(labelsize=8, colors=INK)
    for s in ax2.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    fig.text(0.5, -0.03,
             "mean is not exactly zero - dividing by 3 or by 12 leaves "
             "float32 round-off. max is.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "size_blindness.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote size_blindness.svg   {gaps}  duplicate {gaps2}")


# === 그림으로 보는 AI: 40,320 가지로 읽어도 같은 답 ==========================
#
# "Order Matters"(ICLR 2016)가 보인 것은 집합을 차례가 있는 것처럼 읽히면 답이
# 차례를 따라 흔들린다는 것이었다. 노드 8개의 모든 차례 40,320 가지를 전부 돌려
# 이 쪽의 Set2Set 과 보통 LSTM 이 각각 몇 가지 답을 내는지 센다.

def every_ordering():
    import itertools

    import torch
    import torch.nn as nn

    class Set2Set(nn.Module):
        """이 쪽 본문의 구현을 그대로 옮긴 것."""

        def __init__(self, in_ch, processing_steps=3):
            super().__init__()
            self.steps = processing_steps
            self.lstm = nn.LSTMCell(2 * in_ch, in_ch)
            self.in_ch = in_ch

        def forward(self, x, batch):
            ng = batch.max().item() + 1
            q = torch.zeros(ng, self.in_ch)
            c = torch.zeros(ng, self.in_ch)
            r = torch.zeros(ng, self.in_ch)
            for _ in range(self.steps):
                q, c = self.lstm(torch.cat([q, r], dim=-1), (q, c))
                e = (x * q[batch]).sum(dim=-1)
                e_max = torch.full((ng,), float("-inf"))
                e_max.scatter_reduce_(0, batch, e, reduce="amax")
                a = torch.exp(e - e_max[batch])
                a_sum = torch.zeros(ng)
                a_sum.scatter_add_(0, batch, a)
                a = a / a_sum[batch].clamp(min=1e-10)
                r = torch.zeros(ng, self.in_ch)
                r.scatter_add_(0, batch.unsqueeze(1).expand_as(x),
                               x * a.unsqueeze(1))
            return torch.cat([q, r], dim=-1)

    torch.manual_seed(42)
    n, d = 8, 8
    x = torch.randn(n, d)
    batch = torch.zeros(n, dtype=torch.long)
    s2s = Set2Set(d, processing_steps=3)
    lstm = nn.LSTM(d, 2 * d, batch_first=True)

    perms = list(itertools.permutations(range(n)))
    rows_s, rows_l = [], []
    with torch.no_grad():
        for p in perms:
            idx = torch.tensor(p)
            rows_s.append(s2s(x[idx], batch)[0])
            rows_l.append(lstm(x[idx].unsqueeze(0))[0][0, -1])
    S = torch.stack(rows_s).numpy()
    L = torch.stack(rows_l).numpy()

    spread_s = float((S.max(0) - S.min(0)).max())
    spread_l = float((L.max(0) - L.min(0)).max())
    uniq_s = len({tuple(np.round(v, 6)) for v in S})
    uniq_l = len({tuple(np.round(v, 6)) for v in L})

    # --- 그림이 주장하는 것을 먼저 확인한다 -------------------------------
    assert len(perms) == 40320, "8! 가지를 모두 돌려야 한다"
    assert spread_s < 1e-5, "Set2Set 의 흔들림은 반올림 수준이어야 한다"
    assert spread_l > 100 * spread_s, "LSTM 은 비교할 수 없이 크게 흔들려야 한다"
    assert uniq_l == len(perms) and uniq_s < 10, \
        "LSTM 은 차례마다 다른 답을, Set2Set 은 사실상 한 답을 내야 한다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.2),
                                   gridspec_kw={"width_ratios": [1.3, 1]})

    # --- 왼쪽: 첫 좌표의 분포 ---------------------------------------------
    col = 0
    ax1.hist(L[:, col], bins=90, color=INK, alpha=0.55,
             label=f"an LSTM reading the same {n} nodes")
    centre = float(S[:, col].mean())
    ax1.axvline(centre, color=ACCENT, lw=2.4, zorder=5,
                label="Set2Set - every ordering, one line")
    ax1.annotate(f"all {len(perms):,} orderings\nland inside "
                 f"{spread_s:.0e} of each other",
                 (centre, ax1.get_ylim()[1] * 0.62),
                 textcoords="offset points", xytext=(38, 0), fontsize=8.6,
                 color=ACCENT, va="center",
                 bbox=dict(facecolor="white", alpha=0.86, edgecolor="none",
                           pad=2.5),
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.0))
    ax1.set_xlabel("first coordinate of the readout", fontsize=8.8)
    ax1.set_ylabel(f"orderings (of {len(perms):,})", fontsize=8.8)
    ax1.set_title(f"every one of the {n}! ways to present the set",
                  fontsize=10, color=INK, weight="bold")
    ax1.legend(fontsize=8.4, loc="upper left", framealpha=0.92)

    # --- 오른쪽: 서로 다른 답은 몇 개인가 ---------------------------------
    ys = np.array([1, 0])
    vals = [uniq_l, uniq_s]
    labs = ["LSTM", "Set2Set"]
    cols = [INK, ACCENT]
    ax2.barh(ys, vals, 0.46, color=cols, alpha=0.82)
    for y, v in zip(ys, vals):
        ax2.text(v * 1.35, y, f"{v:,}", va="center", fontsize=12, color=INK,
                 weight="bold")
    ax2.set_xscale("log")
    ax2.set_xlim(0.6, 4e5)
    ax2.set_yticks(ys)
    ax2.set_yticklabels(labs, fontsize=10)
    ax2.set_xlabel("distinct answers, rounded to 1e-6 (log scale)",
                   fontsize=8.8)
    ax2.set_title("how many different answers came out", fontsize=10,
                  color=INK, weight="bold")
    ax2.text(1.0, -0.62,
             f"the {uniq_s} are float32 round-off: adding the same numbers\n"
             "in a different order does not give the same bits",
             fontsize=8.3, color=ACCENT, style="italic", va="top")
    ax2.set_ylim(-1.1, 1.5)

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8, colors=INK)
        ax.grid(alpha=0.13)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.tight_layout()
    fig.text(0.5, -0.04,
             f"untrained modules, seed 42.  Across orderings the LSTM swings "
             f"by up to {spread_l:.2f} on one of its {2 * d} coordinates; "
             f"Set2Set by {spread_s:.0e} on all of them.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.savefig(HERE / "every_ordering.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote every_ordering.svg   {len(perms):,} orderings;  Set2Set "
          f"spread {spread_s:.3e} / {uniq_s} distinct;  LSTM spread "
          f"{spread_l:.3e} / {uniq_l:,} distinct")


# === 그림으로 보는 AI: 세는 양과 나누는 양 ===================================
#
# 이 쪽의 목표는 크기를 따라 늘어나는 양(extensive)인데 mean readout 은 크기로
# 나눈 양(intensive)만 낼 수 있다. Tolman 이 1917 년에 가른 그 구분이다. 쪽이
# 스스로 권한 비교 — 오라클 · 두 수 · mean · sum — 를 그대로 돌려 잰다.

def extensive_target():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    NOISE_VAR = 0.01          # 목표에 넣은 잡음의 분산. 줄일 수 없는 바닥이다.

    def make(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        gs = []
        for _ in range(200):
            n = np.random.randint(5, 15)
            x = torch.randn(n, 4)
            edges = [(j, j + 1) for j in range(n - 1)]
            for _ in range(n // 3):
                u, v = np.random.choice(n, 2, replace=False)
                edges.append((u, v))
            src = [e[0] for e in edges] + [e[1] for e in edges]
            dst = [e[1] for e in edges] + [e[0] for e in edges]
            ei = torch.tensor([src, dst], dtype=torch.long)
            y = (x.sum().item() * 0.1 + len(edges) * 0.05
                 + np.random.randn() * 0.1)
            gs.append((x, ei, y, float(x.sum()), len(edges), n))
        return gs

    class Reg(nn.Module):
        """이 쪽 본문의 모델. readout 한 줄만 갈아 끼울 수 있게 했다."""

        def __init__(self, in_ch, h, num_layers=3, readout="mean"):
            super().__init__()
            self.convs = nn.ModuleList(
                [nn.Linear(in_ch if i == 0 else h, h)
                 for i in range(num_layers)])
            self.predictor = nn.Sequential(nn.Linear(h, h), nn.ReLU(),
                                           nn.Linear(h, 1))
            self.readout = readout

        def forward(self, x, ei):
            n = x.shape[0]
            batch = torch.zeros(n, dtype=torch.long)
            loop = torch.arange(n)
            sa = torch.cat([ei[0], loop])
            da = torch.cat([ei[1], loop])
            deg = torch.zeros(n)
            deg.scatter_add_(0, da, torch.ones(da.shape[0]))
            norm = (deg[sa] * deg[da]).pow(-0.5)
            norm[norm == float("inf")] = 0
            for lin in self.convs:
                h = lin(x)
                msg = h[sa] * norm.unsqueeze(1)
                out = torch.zeros(n, h.shape[1])
                out.scatter_add_(0, da.unsqueeze(1).expand_as(msg), msg)
                x = F.relu(out)
            pool = torch.zeros(1, x.shape[1])
            pool.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
            if self.readout == "mean":
                pool = pool / float(n)
            return self.predictor(pool).squeeze(-1)

    def train_eval(gs, readout, seed):
        torch.manual_seed(seed + 1000)
        m = Reg(4, 32, readout=readout)
        opt = torch.optim.Adam(m.parameters(), lr=0.005)
        m.train()
        for _ in range(50):
            for x, ei, y, *_ in gs[:160]:
                opt.zero_grad()
                F.mse_loss(m(x, ei), torch.tensor([y])).backward()
                opt.step()
        m.eval()
        with torch.no_grad():
            err = [F.mse_loss(m(x, ei), torch.tensor([y])).item()
                   for x, ei, y, *_ in gs[160:]]
        n_par = sum(p.numel() for p in m.parameters())
        return float(np.mean(err)), n_par

    ROWS = ["the formula itself", "a line through two numbers",
            "mean readout (this page)", "sum readout"]
    mse = {k: [] for k in ROWS}
    npar = {}
    for seed in (42, 0, 1):
        gs = make(seed)
        Y = np.array([g[2] for g in gs])
        S = np.array([g[3] for g in gs])
        M = np.array([g[4] for g in gs])
        te = slice(160, 200)
        mse[ROWS[0]].append(float(np.mean(
            (Y[te] - (0.1 * S[te] + 0.05 * M[te])) ** 2)))
        npar[ROWS[0]] = 0
        A = np.c_[S[:160], M[:160], np.ones(160)]
        w = np.linalg.lstsq(A, Y[:160], rcond=None)[0]
        mse[ROWS[1]].append(float(np.mean(
            (Y[te] - np.c_[S[te], M[te], np.ones(40)] @ w) ** 2)))
        npar[ROWS[1]] = 3
        for row, ro in ((ROWS[2], "mean"), (ROWS[3], "sum")):
            e, p = train_eval(gs, ro, seed)
            mse[row].append(e)
            npar[row] = p

    mu = {k: float(np.mean(v)) for k, v in mse.items()}
    sd = {k: float(np.std(v)) for k, v in mse.items()}

    # --- 그림이 주장하는 것을 먼저 확인한다 -------------------------------
    assert abs(mu[ROWS[0]] - NOISE_VAR) < 0.005, "오라클은 잡음 바닥에 닿아야 한다"
    assert mu[ROWS[1]] < mu[ROWS[3]] < mu[ROWS[2]], \
        "두 수 < sum < mean 이어야 이 그림의 이야기가 된다"
    assert mu[ROWS[2]] > 5 * mu[ROWS[1]], "mean 과 두 수의 차가 커야 한다"
    assert npar[ROWS[2]] == npar[ROWS[3]] > 1000

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.3),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # --- 왼쪽: 네 가지가 낸 테스트 오차 ------------------------------------
    ys = np.arange(len(ROWS))[::-1]
    cols = [MUTED, MUTED, ACCENT, "#4a7c59"]
    ax1.barh(ys, [mu[r] for r in ROWS], 0.5,
             xerr=[sd[r] for r in ROWS], color=cols, alpha=0.82,
             error_kw=dict(ecolor=INK, capsize=4, lw=1.1))
    for y, r in zip(ys, ROWS):
        ax1.text(mu[r] + sd[r] + 0.004, y,
                 f"{mu[r]:.4f}   ({mu[r] / NOISE_VAR:.0f}x the floor)",
                 va="center", fontsize=8.8, color=INK, family="monospace")
    ax1.axvline(NOISE_VAR, color=INK, lw=1.3, ls=":")
    ax1.text(NOISE_VAR + 0.004, -0.75,
             "noise floor 0.01 - nothing can do better",
             fontsize=8.4, color=INK, style="italic", va="center")
    ax1.set_ylim(-1.25, len(ROWS) - 0.45)
    ax1.set_yticks(ys)
    ax1.set_yticklabels(ROWS, fontsize=9)
    ax1.set_xlim(0, 0.20)
    ax1.set_xlabel("test MSE, mean of 3 seeds", fontsize=8.8)
    ax1.set_title("the comparison this page asks for", fontsize=10,
                  color=INK, weight="bold")

    # --- 오른쪽: 파라미터가 늘수록 나빠진다 --------------------------------
    for r, c in zip(ROWS, cols):
        ax2.errorbar(max(npar[r], 0.6), mu[r], yerr=sd[r], fmt="o", ms=9,
                     color=c, alpha=0.9, capsize=4, lw=1.1, zorder=3)
    ax2.annotate("the formula, and a line\nthrough two numbers",
                 (3, mu[ROWS[1]]), textcoords="offset points",
                 xytext=(18, 26), fontsize=8.4, color=MUTED,
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
    ax2.annotate("same model,\nsum instead of mean",
                 (npar[ROWS[3]], mu[ROWS[3]]), textcoords="offset points",
                 xytext=(-18, 30), ha="right", fontsize=8.4, color="#4a7c59",
                 arrowprops=dict(arrowstyle="->", color="#4a7c59", lw=0.9))
    ax2.annotate("mean readout", (npar[ROWS[2]], mu[ROWS[2]]),
                 textcoords="offset points", xytext=(-12, 10), ha="right",
                 fontsize=8.4, color=ACCENT)
    ax2.axhline(NOISE_VAR, color=INK, lw=1.3, ls=":")
    ax2.set_xscale("log")
    ax2.set_xlim(0.4, 1e4)
    ax2.set_ylim(0, 0.16)
    ax2.set_xlabel("parameters fitted (log scale)", fontsize=8.8)
    ax2.set_ylabel("test MSE", fontsize=8.8)
    ax2.set_title("more parameters, further from the floor", fontsize=10,
                  color=INK, weight="bold")

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8, colors=INK)
        ax.grid(alpha=0.13)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.tight_layout()
    fig.text(0.5, -0.045,
             f"the page's own 200 graphs and 50 epochs, repeated for 3 seeds.  "
             f"Both networks fit {npar[ROWS[2]]:,} parameters over 8,000 "
             f"updates; the line fits 3 in closed form.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.savefig(HERE / "extensive_target.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote extensive_target.svg   " + "; ".join(
        f"{r}: {mu[r]:.4f}+-{sd[r]:.4f} ({npar[r]} par)" for r in ROWS))


if __name__ == "__main__":
    size_blindness()
    every_ordering()
    extensive_target()
