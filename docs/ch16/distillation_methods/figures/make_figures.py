"""17.5 증류 기반 방법 — 「그림으로 보는 AI」 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 내려받지 않으므로 어디서 돌려도 된다.

실행: python docs/ch16/distillation_methods/figures/make_figures.py
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


# === 그림으로 보는 AI: T^2 는 관례가 아니라 보정이다 ==========================
#
# 부드러운 목표에서 오는 경사는 1/T^2 로 작아진다 (Hinton 2015). 그래서 T^2 을
# 곱한다. **크기는 되돌아오지만 방향은 되돌아오지 않는다** — 그것이 이 쪽이
# "서로 다른 T 의 최적화가 완전히 같아지는 것은 아니다" 라고 적은 까닭이다.

def temperature_scaling():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(0)
    kl = nn.KLDivLoss(reduction="batchmean")
    n, c = 256, 10
    teacher = torch.randn(n, c) * 2.0
    ts = [1, 2, 3, 4, 6, 8]

    def grad(t, use_t2):
        z = torch.randn(n, c, generator=torch.Generator().manual_seed(7),
                        requires_grad=True)
        with torch.no_grad():
            p = torch.softmax(teacher / t, 1)
        loss = kl(torch.log_softmax(z / t, 1), p)
        if use_t2:
            loss = loss * t * t
        loss.backward()
        return z.grad.clone()

    raw = [grad(float(t), False).norm().item() for t in ts]
    fixed = [grad(float(t), True).norm().item() for t in ts]
    g1 = grad(1.0, True)
    cos = [F.cosine_similarity(g1.flatten(), grad(float(t), True).flatten(),
                               dim=0).item() for t in ts]

    # 1/T^2 를 실제로 따르는가
    ratio = [r / raw[0] * t * t for r, t in zip(raw, ts)]
    assert all(0.8 < v < 1.3 for v in ratio), (
        f"T^2 을 안 곱하면 경사가 1/T^2 로 줄어야 한다 (ratio={ratio})")
    assert max(fixed) / min(fixed) < 1.5, "T^2 을 곱하면 크기가 고르게 남아야 한다"
    assert cos[-1] < 0.9, "그래도 방향은 달라져야 한다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.1))

    # --- 왼쪽: 크기 ---------------------------------------------------------
    ax1.loglog(ts, raw, "s--", color=INK, linewidth=1.9, ms=6,
               label="the soft-target gradient")
    ax1.loglog(ts, [raw[0] / (t * t) for t in ts], ":", color=INK,
               linewidth=1.2, alpha=0.6, label="exactly 1 / T^2")
    ax1.loglog(ts, fixed, "o-", color=ACCENT, linewidth=2.2, ms=6.5,
               label="after multiplying by T^2")
    ax1.set_xlabel("temperature T", fontsize=9.5)
    ax1.set_ylabel("gradient magnitude", fontsize=9.5)
    ax1.set_title("the size comes back", fontsize=10, color=INK,
                  weight="bold")
    ax1.set_xticks(ts)
    ax1.set_xticklabels([str(t) for t in ts])
    ax1.legend(fontsize=8.2, loc="lower left", framealpha=0.9)
    ax1.grid(alpha=0.13, which="both")

    # --- 오른쪽: 방향 -------------------------------------------------------
    ax2.plot(ts, cos, "o-", color=ACCENT, linewidth=2.2, ms=7)
    ax2.axhline(1.0, color=INK, linestyle=":", linewidth=1.2, alpha=0.7)
    ax2.text(ts[-1], 1.012, "same direction", fontsize=8.4, color=INK,
             alpha=0.8, ha="right")
    ax2.annotate(f"{cos[-1]:.2f}", xy=(ts[-1], cos[-1]), xytext=(-8, 8),
                 textcoords="offset points", fontsize=11, color=ACCENT,
                 weight="bold", ha="right")
    ax2.set_xlabel("temperature T", fontsize=9.5)
    ax2.set_ylabel("cosine with the T = 1 gradient", fontsize=9.5)
    ax2.set_title("the direction does not", fontsize=10, color=ACCENT,
                  weight="bold")
    ax2.set_xticks(ts)
    ax2.set_ylim(min(cos) - 0.08, 1.06)
    ax2.grid(alpha=0.13)

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8.5, colors=INK)
        for sp in ax.spines.values():
            sp.set_color(INK)
            sp.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             "same teacher logits and same student logits throughout. "
             "T^2 restores the scale, not the objective.",
             ha="center", fontsize=8.5, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "temperature_t2.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote temperature_t2.svg   raw {raw[0]:.4f}->{raw[-1]:.5f}, "
          f"fixed {fixed[0]:.4f}->{fixed[-1]:.4f}, cos {cos[-1]:.3f}")


# === 그림으로 보는 AI: 0 이 된 로스가 뜻하는 것 ==============================
#
# 교사·학생 양쪽 투영을 제약 없이 함께 배우면 **둘 다 0 으로 보내는** 해가
# 있다. 로스는 정확히 0 이 되는데 남은 피처도 0 이다. 기준 한쪽을 고정해야
# 로스가 뜻을 가진다.

def projection_collapse():
    import torch
    import torch.nn as nn

    n, dt, ds, p = 256, 64, 48, 32
    torch.manual_seed(0)
    ft = torch.randn(n, dt)          # 고정 교사 피처
    fs = torch.randn(n, ds)          # 학생 피처

    def run(free_teacher, steps=1500):
        torch.manual_seed(1)
        ps = nn.Linear(ds, p, bias=False)
        pt = nn.Linear(dt, p, bias=False)
        params = list(ps.parameters())
        if free_teacher:
            params += list(pt.parameters())
        else:
            for q in pt.parameters():
                q.requires_grad_(False)
        opt = torch.optim.Adam(params, lr=0.01)
        for _ in range(steps):
            opt.zero_grad()
            loss = ((ps(fs) - pt(ft)) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            return loss.item(), ps(fs).norm().item(), pt(ft).norm().item()

    l_free, a_free, b_free = run(True)
    l_fix, a_fix, b_fix = run(False)

    assert l_free < l_fix * 1e-3, "양쪽을 풀면 로스가 0 으로 내려가야 한다"
    assert a_free < a_fix * 1e-2, "그런데 피처도 같이 0 이 되어야 한다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.0))
    labels = ["both projections\nlearnable", "teacher side\nfixed"]
    cols = [ACCENT, MUTED]

    ax1.bar(range(2), [l_free, l_fix], color=cols, width=0.5, alpha=0.88)
    for i, v in enumerate([l_free, l_fix]):
        ax1.text(i, v + l_fix * 0.04, f"{v:.6f}", ha="center", fontsize=10.5,
                 color=INK, weight="bold")
    ax1.set_xticks(range(2))
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("feature distillation loss", fontsize=9.5)
    ax1.set_ylim(0, l_fix * 1.3)
    ax1.set_title("the better-looking number", fontsize=10, color=ACCENT,
                  weight="bold")
    ax1.grid(alpha=0.13, axis="y")

    ax2.bar(range(2), [a_free, a_fix], color=cols, width=0.5, alpha=0.88)
    for i, v in enumerate([a_free, a_fix]):
        ax2.text(i, v + a_fix * 0.04, f"{v:.2f}", ha="center", fontsize=10.5,
                 color=INK, weight="bold")
    ax2.set_xticks(range(2))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("size of what is left to compare", fontsize=9.5)
    ax2.set_ylim(0, a_fix * 1.3)
    ax2.set_title("what was actually matched", fontsize=10, color=INK,
                  weight="bold")
    ax2.grid(alpha=0.13, axis="y")

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8.5, colors=INK)
        for sp in ax.spines.values():
            sp.set_color(INK)
            sp.set_linewidth(0.7)

    fig.text(0.5, -0.05,
             "a loss of exactly zero, reached by sending both sides to zero. "
             "nothing was preserved.",
             ha="center", fontsize=8.5, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "projection_collapse.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote projection_collapse.svg   free loss {l_free:.2e} "
          f"norm {a_free:.3f} | fixed loss {l_fix:.4f} norm {a_fix:.2f}")


# === 그림으로 보는 AI: 네 방법이 묶어 두는 수의 개수 =========================
#
# "무엇을 맞추는가" 를 **몇 개를 맞추는가** 로 바꿔 세면 네 방법의 차이가
# 자릿수로 드러난다. 어텐션 전달이 피처 증류의 채널을 덜어 낸 것이라는 점도
# 숫자로 그대로 나온다 (채널 수만큼 작아진다).

def how_many_numbers():
    import torch
    import torch.nn as nn

    b, n_cls = 32, 10
    torch.manual_seed(0)
    net = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    )
    h = torch.randn(b, 3, 32, 32)
    feats = []
    for layer in net:
        h = layer(h)
        if isinstance(layer, nn.MaxPool2d):
            feats.append(h)

    f = feats[-1]
    ch, hh, ww = f.shape[1], f.shape[2], f.shape[3]
    counts = [
        ("output\nlogits", b * n_cls, INK),
        ("attention\nmap", hh * ww * b, MUTED),
        ("relation\nB x B", b * b, "#9a6b4f"),
        ("feature map\none layer", ch * hh * ww * b, ACCENT),
        ("feature maps\nall three", sum(t[0].numel() for t in feats) * b,
         ACCENT),
    ]
    out = counts[0][1]

    # 어텐션은 피처에서 채널을 덜어 낸 것이다 — 정확히 채널 수만큼 작다
    assert counts[3][1] == counts[1][1] * ch, (
        "어텐션 지도는 피처 맵을 채널로 나눈 크기여야 한다")
    assert counts[4][1] > counts[3][1], "레이어를 더 쓰면 더 커져야 한다"

    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    xs = np.arange(len(counts))
    vals = [c[1] for c in counts]
    bars = ax.bar(xs, vals, color=[c[2] for c in counts], width=0.58,
                  alpha=0.88)
    ax.set_yscale("log")
    for x, v in zip(xs, vals):
        ax.text(x, v * 1.35, f"{v:,}\n{v / out:.0f}x", ha="center",
                fontsize=8.8, color=INK, weight="bold")

    ax.annotate("", xy=(1, counts[1][1] * 1.08), xytext=(3, counts[3][1] * 0.92),
                arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.2,
                                connectionstyle="arc3,rad=0.25"))
    ax.text(2.0, counts[3][1] * 0.22, f"summing away\n{ch} channels",
            ha="center", fontsize=8.4, color=ACCENT)

    ax.set_xticks(xs)
    ax.set_xticklabels([c[0] for c in counts], fontsize=8.6)
    ax.set_ylabel("numbers the teacher pins down (log)", fontsize=9.5)
    ax.set_ylim(top=max(vals) * 12)
    ax.set_title("same batch, same teacher - four very different grips",
                 fontsize=10.5, color=INK, weight="bold", pad=10)
    ax.grid(alpha=0.13, axis="y", which="both")
    ax.tick_params(labelsize=8.5, colors=INK)
    for sp in ax.spines.values():
        sp.set_color(INK)
        sp.set_linewidth(0.7)

    fig.text(0.5, -0.05,
             f"batch of {b}, {n_cls} classes, a small conv net. "
             "counting what is constrained, not what it costs to compute.",
             ha="center", fontsize=8.5, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "how_many_numbers.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote how_many_numbers.svg   " +
          ", ".join(f"{c[0].replace(chr(10),' ')}={c[1]:,}" for c in counts))


if __name__ == "__main__":
    temperature_scaling()
    projection_collapse()
    how_many_numbers()
