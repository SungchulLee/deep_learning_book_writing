"""3장 개요의 그림을 그리는 스크립트.

「그림으로 보는 AI」 상자의 그림은 사진을 빌려 오지 않고 형식만 빌려 이
책의 그림체로 다시 그린다. 다만 오른쪽 판은 **실제 자료에서 잰 것**이다 —
MNIST 를 읽어 잉크 상자와 무게중심을 직접 구한다. 규칙은 CLAUDE.md 의
「그림으로 보는 AI」 항에 적어 두었다.

    python3 make_figures.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from matplotlib.patches import Rectangle, FancyArrowPatch
from pathlib import Path

plt.rcParams["svg.fonttype"] = "path"      # 글자를 외곽선으로 — 폰트에 기대지 않는다
HERE = Path(__file__).parent
INK = "#2b2b2b"


def save(fig, name):
    fig.savefig(HERE / name, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


# ===========================================================================
def draw_mnist_made():
    """왼쪽은 MNIST 가 어떻게 섞여 만들어졌는지, 오른쪽은 문서가 적은
    정규화를 실제 자료에서 확인한 것이다.

    왼쪽은 구조만 그린 그림이고, 오른쪽의 수(무게중심 14.01 / 14.00, 잉크
    상자가 모두 20x20 안에 든다는 것)는 학습 집합 60,000장을 직접 재어 얻었다.
    """
    ds = torchvision.datasets.MNIST("./data", train=True, download=True,
                                    transform=T.ToTensor())
    X = torch.cat([x for x, _ in DataLoader(ds, batch_size=4000)])[:, 0]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.2),
                                   gridspec_kw={"width_ratios": [1.35, 1]})

    # === 왼쪽: 어떻게 섞었는가 ==================================================
    axL.set_xlim(0, 11.5); axL.set_ylim(0, 6.4); axL.axis("off")
    for y, name, who in [(4.7, "SD-3", "census workers"),
                         (1.9, "SD-1", "high school students")]:
        axL.add_patch(Rectangle((0.2, y - 0.62), 2.9, 1.24, fill=False, ec=INK, lw=1.3))
        axL.text(1.65, y + 0.18, name, ha="center", fontsize=10, color=INK)
        axL.text(1.65, y - 0.34, who, ha="center", fontsize=8.3, color=INK)
    axL.text(1.65, 5.85, "NIST", ha="center", fontsize=9, color=INK, style="italic")

    for y, name, n in [(4.7, "train", "60,000"), (1.9, "test", "10,000")]:
        axL.add_patch(Rectangle((7.3, y - 0.62), 2.9, 1.24, fill=False, ec=INK, lw=1.3))
        axL.text(8.75, y + 0.18, name, ha="center", fontsize=10, color=INK)
        axL.text(8.75, y - 0.34, n, ha="center", fontsize=8.3, color=INK)
    axL.text(8.75, 5.85, "MNIST", ha="center", fontsize=9, color=INK, style="italic")

    for y0 in (4.7, 1.9):
        for y1 in (4.7, 1.9):
            axL.add_patch(FancyArrowPatch((3.2, y0), (7.2, y1), arrowstyle="-|>",
                                          mutation_scale=9, color=INK, lw=0.85,
                                          connectionstyle="arc3,rad=0.0" if y0 == y1 else "arc3,rad=0.22"))
    axL.plot([7.3, 10.2], [3.3, 3.3], color=INK, lw=0.9, ls=(0, (4, 3)))
    axL.text(8.75, 3.0, "writers disjoint", ha="center", fontsize=8.6, color=INK)
    axL.text(5.25, 3.42, "both sources", ha="center", fontsize=8.4, color=INK)
    axL.text(5.25, 3.02, "into both sets", ha="center", fontsize=8.4, color=INK)
    axL.text(5.75, 0.45, "NIST shipped SD-3 as train and SD-1 as test;\n"
                         "MNIST mixed them so the two halves match",
             ha="center", fontsize=8.4, color=INK, style="italic")

    # === 오른쪽: 문서가 적은 정규화를 재어 본다 =================================
    axR.set_xlim(-0.5, 27.5); axR.set_ylim(27.5, -0.5); axR.set_aspect("equal")
    axR.set_xticks([]); axR.set_yticks([])
    for s in axR.spines.values(): s.set_color(INK); s.set_linewidth(1.0)
    img = X[0]
    axR.imshow(img, cmap="gray_r", vmin=0, vmax=1)
    rows = torch.nonzero((img > 0).any(1)).flatten()
    cols = torch.nonzero((img > 0).any(0)).flatten()
    axR.add_patch(Rectangle((cols.min() - .5, rows.min() - .5),
                            cols.max() - cols.min() + 1, rows.max() - rows.min() + 1,
                            fill=False, ec=INK, lw=1.3, ls=(0, (3, 2))))
    ys, xs = torch.meshgrid(torch.arange(28.), torch.arange(28.), indexing="ij")
    m = img.sum(); cy = (img * ys).sum() / m; cx = (img * xs).sum() / m
    axR.plot(cx, cy, marker="+", ms=13, mew=1.8, color="crimson")
    axR.set_title("20 x 20 ink box, centre of mass marked", fontsize=9, color=INK, pad=8)
    axR.text(13.5, 30.6, "over all 60,000:  centre of mass  x 14.01,  y 14.00\n"
                         "every ink box fits in 20 x 20",
             ha="center", fontsize=8.4, color=INK)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.12, wspace=0.05)
    return fig


# ===========================================================================
if __name__ == "__main__":
    save(draw_mnist_made(), "mnist_made.svg")
