"""26장 장 소개의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

이 그림의 수는 **직접 잰 것이 아니라 논문이 보고한 값**이다. 다시 돌려 얻은
것이 아니므로 그림 안에 출처를 적는다. Hinton & Salakhutdinov (2006) 의
Fig. 2B 캡션에 실린 세 값이다.

주의 — 이 디렉터리의 ae_vae_latent.svg 와 ae_vae_samples.svg 는 2026-09 에
들어온 것으로 **만든 코드가 여기에 없다.** 이 스크립트는 그 둘을 만들지
않는다.

실행: python docs/ch25/figures/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
INK = "#2f3337"
ACCENT = "#8a4b2a"
GREY = "#8d9499"

# Hinton & Salakhutdinov 2006, Science 313, 504-507, Fig. 2B 캡션.
# MNIST, 한 장을 30 개의 수로 줄였을 때 시험 영상 10,000 장의 평균 제곱 오차.
REPORTED = [
    ("deep autoencoder\n784-1000-500-250-30", 3.00, ACCENT),
    ("logistic PCA\n30 components", 8.01, GREY),
    ("standard PCA\n30 components", 13.87, GREY),
]


def hinton_2006():
    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    names = [n for n, _, _ in REPORTED]
    vals = [v for _, v, _ in REPORTED]
    cols = [c for _, _, c in REPORTED]

    bars = ax.bar(range(3), vals, color=cols, width=0.56, alpha=0.9)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.35, f"{v:.2f}", ha="center", fontsize=11,
                color=INK, weight="bold")

    ax.set_xticks(range(3))
    ax.set_xticklabels(names, fontsize=8.5, color=INK)
    ax.set_ylabel("average squared error per test image", fontsize=9)
    ax.set_title("MNIST, 30 numbers per image", fontsize=10.5, color=INK,
                 weight="bold", pad=9)
    ax.set_ylim(0, 18.6)
    ax.grid(alpha=0.13, axis="y")
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8.5, colors=INK)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    ratio = vals[2] / vals[0]
    ax.annotate(f"{ratio:.1f}x", xy=(1.0, 17.0), ha="center", fontsize=10,
                color=ACCENT, weight="bold")
    ax.annotate("", xy=(0, 16.4), xytext=(2, 16.4),
                arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.2))

    fig.text(0.5, -0.09,
             "numbers as reported in Hinton and Salakhutdinov 2006, Fig. 2B - "
             "not re-measured here.\n"
             "trained on 60,000 images, tested on 10,000 held out.",
             ha="center", fontsize=8.2, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "hinton_2006.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote hinton_2006.svg   ratio {ratio:.2f}x")


if __name__ == "__main__":
    hinton_2006()
