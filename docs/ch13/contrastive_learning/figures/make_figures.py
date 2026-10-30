"""14.1 대조 학습 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

CIFAR-10 을 읽으므로 **저장소 뿌리에서** 돌린다.

실행: python docs/ch13/contrastive_learning/figures/make_figures.py
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

N_CAND = 500
BINS = 16


# === 그림으로 보는 AI: 색 히스토그램이라는 지름길 =============================
#
# SimCLR 논문 3.1 절의 주장 — "color histograms alone suffice to distinguish
# images" — 을 직접 잰다. 같은 그림에서 뜯은 두 조각은 색 분포가 거의 같아서,
# 내용을 하나도 안 보고 색만 세어도 짝을 찾을 수 있다. 색을 흔들면 그 길이
# 막힌다. 수는 전부 여기서 셈한다.

def _hist(img):
    a = np.asarray(img, dtype=np.float32) / 255.0
    h = np.concatenate([np.histogram(a[:, :, c], bins=BINS, range=(0, 1))[0]
                        for c in range(3)])
    return h / (h.sum() + 1e-9)


def _match_rate(ds, crop, jit, color_distort, n=N_CAND):
    v1, v2 = [], []
    for i in range(n):
        im = ds[i][0]
        for store in (v1, v2):
            v = crop(im)
            if color_distort:
                v = jit(v)
            store.append(_hist(v))
    a, b = np.array(v1), np.array(v2)
    d = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return (d.argmin(1) == np.arange(n)).mean(), a, b


def color_shortcut():
    import torch
    from torchvision import datasets, transforms

    torch.manual_seed(0)
    np.random.seed(0)
    ds = datasets.CIFAR10(root="./data", train=True, download=False)
    crop = transforms.RandomResizedCrop(24, scale=(0.3, 1.0))
    jit = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)

    plain, a, b = _match_rate(ds, crop, jit, False)
    jittered, _, _ = _match_rate(ds, crop, jit, True)
    chance = 1.0 / N_CAND

    # 그림이 주장하는 것을 먼저 확인한다
    assert plain > 0.5, "색만으로 짝이 찾아지지 않으면 이 그림의 이야기가 없다"
    assert jittered < plain / 5, "색 변화가 지름길을 막지 못하면 이야기가 없다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.0),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # --- 왼쪽: 같은 그림의 두 조각은 색이 거의 같다 ------------------------
    x = np.arange(len(a[0]))
    ax1.plot(x, a[0], color=ACCENT, linewidth=1.8, label="a crop of image 1")
    ax1.plot(x, b[0], color=ACCENT, linewidth=1.8, linestyle="--", alpha=0.8,
             label="another crop of image 1")
    ax1.plot(x, b[1], color=MUTED, linewidth=1.5, alpha=0.85,
             label="a crop of image 2")
    for k in (BINS, 2 * BINS):
        ax1.axvline(k - 0.5, color=INK, linewidth=0.6, alpha=0.3)
    lo, hi = ax1.get_ylim()
    ax1.set_ylim(lo - (hi - lo) * 0.10, hi)          # 채널 이름 놓을 자리
    for k, lab in ((BINS * 0.5, "red"), (BINS * 1.5, "green"),
                   (BINS * 2.5, "blue")):
        ax1.text(k, lo - (hi - lo) * 0.055, lab, ha="center", va="center",
                 fontsize=8, color=INK, alpha=0.7)
    ax1.set_xlabel("colour histogram bins", fontsize=9)
    ax1.set_ylabel("share of pixels", fontsize=9)
    ax1.set_title("two crops, one image", fontsize=10, color=INK,
                  weight="bold")
    ax1.legend(fontsize=8, framealpha=0.9)
    ax1.grid(alpha=0.13)

    # --- 오른쪽: 색만 세어 짝 맞히기 ---------------------------------------
    bars = [plain, jittered]
    ax2.bar([0, 1], bars, color=[ACCENT, INK], width=0.52, alpha=0.88)
    ax2.axhline(chance, color=INK, linestyle=":", linewidth=1.2)
    ax2.text(-0.42, chance + 0.035, f"chance {chance:.3f}", fontsize=8,
             color=INK, va="bottom", ha="left")
    for i, v in enumerate(bars):
        ax2.text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=10,
                 color=INK, weight="bold")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["crop only", "crop +\ncolour jitter"], fontsize=9)
    ax2.set_ylabel(f"found the pair among {N_CAND}", fontsize=9)
    ax2.set_ylim(0, 1.0)
    ax2.set_title("using colour alone, no content", fontsize=10, color=INK,
                  weight="bold")
    ax2.grid(alpha=0.13, axis="y")

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             "a network could pass the task without looking at the picture. "
             "that is what the colour jitter is for.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "colour_shortcut.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote colour_shortcut.svg   crop {plain:.3f}  "
          f"crop+jitter {jittered:.3f}  chance {chance:.4f}")


if __name__ == "__main__":
    color_shortcut()
