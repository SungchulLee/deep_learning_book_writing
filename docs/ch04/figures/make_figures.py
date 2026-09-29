"""
================================================================================
make_figures.py - 4.1의 그림을 만든다
================================================================================

만드는 그림:
    class_mean_templates_both.svg   MNIST와 CIFAR-10의 클래스 평균 템플릿 스무 장.
                                    위가 읽히는 숫자, 아래가 색 얼룩이다

그림 규칙(CLAUDE.md):
    - SVG로 저장하고 svg.fonttype='path'로 글자를 외곽선으로 만든다
    - transparent=True
    - 그림 안의 글자는 모두 ASCII (기본 글꼴에 한글이 없다)

실행:
    python make_figures.py
    DATA_ROOT=~/data python make_figures.py        # 내려받아 둔 자료를 쓸 때
================================================================================
"""

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms

matplotlib.rcParams["svg.fonttype"] = "path"

DATA = os.environ.get("DATA_ROOT", "./data")
CURVES = os.environ.get("CURVE_ROOT", ".")
CIFAR_CLASSES = ["plane", "car", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]


def class_means(name):
    """클래스마다 학습 이미지를 평균 낸다. 정규화 없이 [0,1] 그대로 쓴다."""
    cls = getattr(torchvision.datasets, name)
    ds = cls(root=DATA, train=True, download=True,
             transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(ds, batch_size=1000, shuffle=False)

    tot, cnt = None, torch.zeros(10)
    for x, y in loader:
        if tot is None:
            tot = torch.zeros(10, *x.shape[1:])
        tot.index_add_(0, y, x)
        cnt.index_add_(0, y, torch.ones_like(y, dtype=torch.float))
    return tot / cnt.reshape(-1, *([1] * (tot.dim() - 1)))


# === 그림 1: 두 데이터셋의 클래스 평균 템플릿 ===============================
def fig_class_mean_templates():
    mnist = class_means("MNIST")        # (10, 1, 28, 28)
    cifar = class_means("CIFAR10")      # (10, 3, 32, 32)

    fig, axes = plt.subplots(2, 10, figsize=(13, 3.1))

    for k in range(10):
        ax = axes[0, k]
        ax.imshow(mnist[k, 0], cmap="gray")
        ax.set_title(str(k), fontsize=11, pad=4)
        ax.axis("off")

    for k in range(10):
        ax = axes[1, k]
        ax.imshow(cifar[k].permute(1, 2, 0))
        ax.set_title(CIFAR_CLASSES[k], fontsize=9, pad=4)
        ax.axis("off")

    # axis("off") 뒤에는 set_ylabel이 먹지 않으므로 줄 이름은 fig.text로 적는다
    fig.text(0.085, 0.72, "MNIST", ha="right", va="center",
             fontsize=11, fontweight="bold")
    fig.text(0.085, 0.28, "CIFAR-10", ha="right", va="center",
             fontsize=11, fontweight="bold")

    fig.subplots_adjust(left=0.09, right=0.99, top=0.9, bottom=0.02,
                        wspace=0.12, hspace=0.35)
    fig.savefig("class_mean_templates_both.svg", transparent=True)
    plt.close(fig)
    print("wrote class_mean_templates_both.svg")


# === 그림 2: VGG16이 ImageNet에서 배운 첫 층 필터 64장 ======================
def fig_vgg16_conv1():
    """3장 4절이 MNIST CNN에 들이댄 잣대를 그대로 VGG16에 들이댄다.

    필터는 3x3x3, 곧 장당 27개 가중치뿐이다. 3장이 경고했듯 이 정도로
    좁은 공간에서는 무작위 가중치도 모서리 검출기와 어느 정도 닮는다.
    그래서 그림 옆에 무작위 기준선과의 비교를 함께 둔다.
    """
    from torchvision.models import vgg16, VGG16_Weights

    W = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[0].weight.detach()

    fig, axes = plt.subplots(4, 16, figsize=(13, 3.6))
    for i, ax in enumerate(axes.flat):
        f = W[i]                                    # (3, 3, 3)
        f = (f - f.min()) / (f.max() - f.min())     # 보이게 [0,1]로 늘린다
        ax.imshow(f.permute(1, 2, 0))
        ax.axis("off")

    fig.suptitle("VGG16 conv1: 64 filters, 3x3x3 (27 weights each)",
                 fontsize=11, y=0.99)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02,
                        wspace=0.15, hspace=0.15)
    fig.savefig("vgg16_conv1_filters.svg", transparent=True)
    plt.close(fig)
    print("wrote vgg16_conv1_filters.svg")


# === 그림 3: PCA로 줄였다 되살린 그림 ======================================
def fig_pca_reconstructions():
    """주성분을 몇 개 쓰느냐에 따라 복원이 어떻게 달라지는지 보인다.

    PCA는 닫힌 꼴이라 돌릴 때마다 똑같은 그림이 나온다. 씨앗이 없다.
    """
    import numpy as np

    ds = torchvision.datasets.CIFAR10(root=DATA, train=True, download=True,
                                      transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(ds, batch_size=2000, shuffle=False)
    xs = [x for x, _ in loader]
    X = torch.cat(xs).flatten(1)                    # (50000, 3072), [0,1]

    mu = X.mean(0, keepdim=True)
    Xc = X - mu
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    # torch.linalg.eigh는 이 macOS 빌드에서 3072x3072에 실패한다
    ev, evec = np.linalg.eigh(cov.double().numpy())
    evec = torch.from_numpy(np.ascontiguousarray(evec[:, ::-1])).float()
    ev = torch.from_numpy(np.ascontiguousarray(ev[::-1])).float()

    idx = [4, 7, 12, 19, 25, 31, 33, 40]            # 보여 줄 여덟 장
    ks = [16, 64, 256]
    rows = [("original", X[idx])]
    for k in ks:
        V = evec[:, :k]
        rec = ((X[idx] - mu) @ V) @ V.T + mu
        share = (ev[:k].sum() / ev.sum()).item()
        rows.append((f"PCA-{k}  ({100*share:.0f}% var)", rec))

    fig, axes = plt.subplots(len(rows), len(idx), figsize=(10, 5.4))
    for r, (label, imgs) in enumerate(rows):
        for c in range(len(idx)):
            ax = axes[r, c]
            ax.imshow(imgs[c].reshape(3, 32, 32).permute(1, 2, 0).clamp(0, 1))
            ax.axis("off")
        axes[r, 0].text(-0.15, 0.5, label, transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=9)

    fig.subplots_adjust(left=0.17, right=0.99, top=0.98, bottom=0.01,
                        wspace=0.08, hspace=0.12)
    fig.savefig("pca_reconstructions.svg", transparent=True)
    plt.close(fig)
    print("wrote pca_reconstructions.svg")


def fig_depth_curves():
    """깊이별 학습 곡선. 씨 다섯 개의 최소~최대를 띠로 두른다.

    띠를 표준편차가 아니라 최소~최대로 그리는 까닭은 4.1절부터 이 책이
    써 온 '퍼짐'이 바로 그 정의이기 때문이다. 그림과 표가 같은 자를 쓴다.
    """
    import json

    runs = []
    for w in ("w1", "w2", "w3"):
        with open(f"{CURVES}/curve_{w}.json") as f:
            runs += json.load(f)

    # n_per -> {epoch: [씨마다의 값]}
    by_depth = {}
    for r in runs:
        d = by_depth.setdefault(r["n_per"], {"test": {}, "gap": {}})
        for pt in r["curve"]:
            d["test"].setdefault(pt["epoch"], []).append(pt["test"])
            d["gap"].setdefault(pt["epoch"], []).append(pt["train"] - pt["test"])

    LABEL = {1: "2 conv layers", 2: "4 conv layers", 3: "6 conv layers"}
    COLOR = {1: "#888888", 2: "#1f77b4", 3: "#d62728"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    for key, ax, ylab in [("test", axes[0], "test accuracy (%)"),
                          ("gap", axes[1], "train - test (%p)")]:
        for n_per in sorted(by_depth):
            pts = by_depth[n_per][key]
            eps = sorted(pts)
            lo = [min(pts[e]) for e in eps]
            hi = [max(pts[e]) for e in eps]
            mid = [sum(pts[e]) / len(pts[e]) for e in eps]
            ax.fill_between(eps, lo, hi, color=COLOR[n_per], alpha=0.18, linewidth=0)
            ax.plot(eps, mid, color=COLOR[n_per], linewidth=1.6, label=LABEL[n_per])
        ax.set_xlabel("epoch", fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.set_xlim(0, 100)
    axes[0].legend(fontsize=8, loc="lower right", frameon=False)
    # 표가 값을 적는 세 지점을 그림에도 표시한다
    for ax in axes:
        for e in (5, 30, 100):
            ax.axvline(e, color="#000000", alpha=0.15, linewidth=0.7, linestyle=":")

    fig.tight_layout()
    fig.savefig("depth_curves.svg", transparent=True)
    plt.close(fig)
    print("wrote depth_curves.svg")



# =============================================================================
# 4.3의 그림 둘
# =============================================================================
def _box(ax, x, y, w, h, label, sub=None, fc="#dfe6ee", ec="#5b6b7d", fs=7.5):
    """상자 하나와 그 안의 글자. 글자는 모두 ASCII 로 적는다.

    윗글과 아랫글의 간격은 상자 높이에 **비례**해야 한다. 고정값으로 두면
    상자가 조금만 높아져도 두 줄이 겹쳐 읽을 수 없게 된다.
    """
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=0.9))
    cy = y + h / 2
    if sub:
        ax.text(x + w / 2, cy + h * 0.17, label, ha="center", va="center", fontsize=fs)
        ax.text(x + w / 2, cy - h * 0.19, sub, ha="center", va="center",
                fontsize=fs - 1.3, color="#44525f")
    else:
        ax.text(x + w / 2, cy, label, ha="center", va="center", fontsize=fs)


def fig_vgg16_architecture():
    """VGG16을 **부피**로 그린다. 4걸음 CNN을 같은 자로 나란히 둔다.

    보는 이가 읽어야 할 것: 상자가 지날수록 **낮아지고 두꺼워진다**는 것.
    낮아지는 것은 공간 해상도(224 -> 7)이고 두꺼워지는 것은 채널 수(3 -> 512)다.
    납작한 칸을 늘어놓으면 이 맞바꿈이 보이지 않는다.

    상자의 높이는 한 변에 비례하고(로그로 눌렀다), 너비는 채널 수의 제곱근에
    비례한다. 채널을 그대로 쓰면 512가 3을 짓눌러 그림이 되지 않는다.
    """
    import math
    from matplotlib.colors import to_rgb

    # 앞면은 공간 크기(한 변), 앞으로 밀어낸 깊이는 채널 수다. 둘 다 제곱근으로
    # 눌렀다 -- 그대로 쓰면 7x7이 224x224 옆에서 점이 되고 3채널이 512 옆에서
    # 사라진다. 눌러도 "낮아지면서 두꺼워진다"는 맞바꿈은 그대로 보인다.
    def face_of(side): return 0.55 + 3.65 * math.sqrt(side / 224)
    def deep_of(ch):   return 0.30 + 2.30 * math.sqrt(ch / 512)

    CONV, POOL, FC = "#bcd6f0", "#f0cfb4", "#c8e6c4"
    EDGE = "#4a5a6b"
    VX, VY = 0.62, 0.46          # 앞으로 밀어내는 방향(등각)

    def shade(c, f):
        r, g, b = to_rgb(c)
        return (min(1, r * f), min(1, g * f), min(1, b * f))

    def volume(ax, x, yc, side, ch, color, label, fs=6.4):
        """상자 하나를 그리고 **오른쪽 끝 x**를 돌려준다."""
        h, e = face_of(side), deep_of(ch)
        vx, vy = VX * e, VY * e
        x0, y0 = x, yc - h / 2
        A, B = (x0, y0), (x0 + h, y0)
        C, D = (x0 + h, y0 + h), (x0, y0 + h)
        off = lambda p: (p[0] + vx, p[1] + vy)
        for pts, f in (([D, C, off(C), off(D)], 1.12),      # 윗면
                       ([B, C, off(C), off(B)], 0.80)):     # 옆면
            ax.add_patch(plt.Polygon(pts, closed=True, facecolor=shade(color, f),
                                     edgecolor=EDGE, linewidth=0.7))
        ax.add_patch(plt.Polygon([A, B, C, D], closed=True, facecolor=color,
                                 edgecolor=EDGE, linewidth=0.7))
        ax.text(x0 + h / 2 + vx / 2, y0 + h + vy + 0.30, label,
                ha="center", va="bottom", fontsize=fs, color="#2f3d49")
        # PyTorch 는 채널이 앞이다(NCHW). 224x224x3 은 텐서플로 차례이므로
        # 이 책의 코드가 내놓는 모양과 어긋난다 -- 512x7x7 로 적는다.
        ax.text(x0 + h / 2, y0 - 0.32, f"{ch}x{side}x{side}",
                ha="center", va="top", fontsize=fs - 0.7, color="#69757f")
        return x0 + h + vx

    def bar(ax, x, yc, n, label, fs=6.4):
        """FC 한 층. 부피가 아니라 벡터이므로 납작한 막대로 그린다."""
        w, h = 0.42, 2.3
        ax.add_patch(plt.Rectangle((x, yc - h / 2), w, h, facecolor=FC,
                                   edgecolor=EDGE, linewidth=0.7))
        ax.text(x + w / 2, yc + h / 2 + 0.30, label, ha="center", va="bottom",
                fontsize=fs, color="#2f3d49")
        ax.text(x + w / 2, yc - h / 2 - 0.32, str(n), ha="center", va="top",
                fontsize=fs - 0.7, color="#69757f")
        return x + w

    def row(ax, yc, vols, fcs, gap=0.62):
        x = 0.0
        for side, ch, color, label in vols:
            x = volume(ax, x, yc, side, ch, color, label) + gap
        for n, label in fcs:
            x = bar(ax, x, yc, n, label) + gap * 0.8
        return x - gap

    vgg = [(224, 3, "#e4e9ec", "input"),
           (224, 64, CONV, "conv x2"), (112, 64, POOL, "pool"),
           (112, 128, CONV, "conv x2"), (56, 128, POOL, "pool"),
           (56, 256, CONV, "conv x3"), (28, 256, POOL, "pool"),
           (28, 512, CONV, "conv x3"), (14, 512, POOL, "pool"),
           (14, 512, CONV, "conv x3"), (7, 512, POOL, "pool")]
    step4 = [(32, 3, "#e4e9ec", "input"),
             (32, 32, CONV, "conv"), (16, 32, POOL, "pool"),
             (16, 64, CONV, "conv"), (8, 64, POOL, "pool")]

    fig, ax = plt.subplots(figsize=(15, 6.4))
    ax.axis("off")

    Y_VGG, Y_S4 = 10.4, 2.0
    ax.text(0, Y_VGG + 4.3, "VGG16  —  13 conv layers, 138,357,544 params",
            fontsize=10, weight="bold")
    xv = row(ax, Y_VGG, vgg, [(4096, "FC"), (4096, "FC"), (1000, "FC")])

    ax.text(0, Y_S4 + 3.0, "Step 4 CNN  —  2 conv layers, 545,098 params (CIFAR-10)",
            fontsize=10, weight="bold")
    xs = row(ax, Y_S4, step4, [(128, "FC"), (10, "FC")])

    ax.annotate("", xy=(xv * 0.66, Y_VGG + 3.5), xytext=(0.4, Y_VGG + 3.5),
                arrowprops=dict(arrowstyle="->", color="#93a0ac", lw=1.0))
    ax.text(xv * 0.33, Y_VGG + 3.7,
            "front face shrinks 224 -> 7,   depth grows 3 -> 512",
            ha="center", fontsize=8, color="#7b8792")

    # 눈금은 **그린 것에서** 얻는다. 손으로 어림한 폭을 박아 두었더니 5번째
    # 블록과 FC 세 층이 통째로 오른쪽 밖으로 잘려 나가 있었다.
    ax.relim(); ax.autoscale_view()
    x_hi = max(xv, xs)
    ax.set_xlim(-0.8, x_hi + 0.8)
    ax.set_ylim(Y_S4 - 3.4, Y_VGG + 5.2)

    fig.savefig("vgg16_architecture.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print("  vgg16_architecture.svg")


def fig_transfer_learning():
    """전이의 얼개: 무엇을 얼리고 무엇을 새로 다는가, 그리고 그 몫.

    보는 이가 읽어야 할 것: 잘라 내는 자리가 마지막 FC 하나뿐이고,
    새로 배우는 매개변수가 전체의 0.03%밖에 되지 않는다는 점.
    """
    fig, ax = plt.subplots(figsize=(11, 3.9))
    ax.set_xlim(-1, 92); ax.set_ylim(0, 36); ax.axis("off")

    FROZEN, CUT, NEW = "#e2e6ea", "#f6d6d6", "#cfe8cf"

    ax.text(0, 33.2, "VGG16 as a frozen feature extractor", fontsize=9, weight="bold")

    # 얼린 몸통
    _box(ax, 0, 22.0, 34, 5.4, "conv blocks 1-5  (13 conv layers)", "FROZEN", FROZEN, fs=8)
    _box(ax, 35.5, 22.0, 11, 5.4, "FC 4096", "FROZEN", FROZEN, fs=8)
    _box(ax, 48.0, 22.0, 11, 5.4, "FC 4096", "FROZEN", FROZEN, fs=8)
    # 떼어 내는 머리
    _box(ax, 60.5, 22.0, 13, 5.4, "FC 1000", "REMOVED", CUT, fs=8)
    ax.plot([60.5, 73.5], [22.0, 27.4], color="#b05555", lw=1.1)
    ax.plot([60.5, 73.5], [27.4, 22.0], color="#b05555", lw=1.1)
    # 새로 다는 머리
    _box(ax, 76.0, 22.0, 13, 5.4, "FC 10", "TRAINED", NEW, fs=8)

    # 상자 **위로** 넘어가게 둔다. 상자 높이에 걸치면 떼어 낸 칸의 글자를 가린다
    ax.annotate("", xy=(82.5, 27.6), xytext=(53.0, 27.6),
                arrowprops=dict(arrowstyle="->", color="#4a7a4a", lw=1.2,
                                connectionstyle="arc3,rad=-0.35"))
    ax.text(67.5, 31.4, "replace the head", fontsize=8, color="#4a7a4a", ha="center")

    # 아래: 매개변수 몫 막대
    ax.text(0, 13.0, "Who chose these numbers?", fontsize=9, weight="bold")
    bar_y, bar_h, W = 6.0, 4.0, 89.0
    frac = 40970 / (134260544 + 40970)          # 0.0003
    ax.add_patch(plt.Rectangle((0, bar_y), W * (1 - frac), bar_h,
                               facecolor=FROZEN, edgecolor="#5b6b7d", lw=0.9))
    ax.add_patch(plt.Rectangle((W * (1 - frac), bar_y), max(W * frac, 0.45), bar_h,
                               facecolor=NEW, edgecolor="#4a7a4a", lw=0.9))
    ax.text(W / 2, bar_y + bar_h / 2, "ImageNet chose  134,260,544  (99.97%)",
            ha="center", va="center", fontsize=8)
    ax.annotate("we chose 40,970  (0.03%)", xy=(W, bar_y + bar_h / 2),
                xytext=(W - 26, bar_y - 4.2), fontsize=8, color="#3f6b3f",
                arrowprops=dict(arrowstyle="->", color="#4a7a4a", lw=1.0))

    fig.tight_layout()
    fig.savefig("transfer_learning.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print("  transfer_learning.svg")


if __name__ == "__main__":
    fig_class_mean_templates()
    fig_vgg16_conv1()
    fig_pca_reconstructions()
    fig_depth_curves()
    fig_vgg16_architecture()
    fig_transfer_learning()
