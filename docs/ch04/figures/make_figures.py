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


def fig_vgg16_layers():
    """VGG16을 **층 목록**으로 쌓아 그린다. 부피 그림과 짝이다.

    부피 그림은 "낮아지면서 두꺼워진다"를 보이지만 층마다의 정확한 수는
    읽히지 않는다. 이 그림은 그 반대다 -- 왼쪽에 층 이름, 가운데에 연산,
    오른쪽에 **그 층을 지나고 난 텐서 모양**을 적는다. 둘을 함께 두면
    구조와 모양이 한 자리에서 읽힌다.

    모양은 PyTorch 차례(NCHW)로 적는다. 흔히 도는 그림들은 224x224x3처럼
    텐서플로 차례로 적어 두는데, 이 책의 코드가 내놓는 것과 어긋난다.

    흔한 잘못 하나: 인터넷에 도는 VGG16 그림 가운데 conv3 를 두 층으로
    그려 둔 것이 많다. 그러면 합성곱이 12층이 되어 이름의 16(=13+3)과
    맞지 않는다. 블록마다 2-2-3-3-3 이다.
    """
    CONV = ("#fce4d2", "#d4813f")
    POOL = ("#dbe7f7", "#4a72b8")
    FCL  = ("#dfeadb", "#4e7a43")
    SMAX = ("#fbdada", "#c44f4f")
    INP  = ("#e6e6e6", "#333333")
    AVG  = ("#eef1f4", "#8a97a4")

    # (층 이름, 상자 글, 224를 넣었을 때 모양, 32를 넣었을 때 모양, 색)
    # 두 열을 나란히 두는 까닭: 224에서는 적응 풀링이 하는 일이 없어서
    # (7x7 -> 7x7) 쓸데없는 줄로 보인다. 32를 함께 적어야 그 줄이 무엇을
    # 하는 자리인지 드러난다 -- 1x1 을 7x7 로 맞춰 주는 것이 그 일이다.
    rows = [
        ("",         "Input",              "3x224x224",   "3x32x32",   INP),
        ("conv1-1",  "3x3 conv, 64",       "64x224x224",  "64x32x32",  CONV),
        ("conv1-2",  "3x3 conv, 64",       "64x224x224",  "64x32x32",  CONV),
        ("",         "Pool",               "64x112x112",  "64x16x16",  POOL),
        ("conv2-1",  "3x3 conv, 128",      "128x112x112", "128x16x16", CONV),
        ("conv2-2",  "3x3 conv, 128",      "128x112x112", "128x16x16", CONV),
        ("",         "Pool",               "128x56x56",   "128x8x8",   POOL),
        ("conv3-1",  "3x3 conv, 256",      "256x56x56",   "256x8x8",   CONV),
        ("conv3-2",  "3x3 conv, 256",      "256x56x56",   "256x8x8",   CONV),
        ("conv3-3",  "3x3 conv, 256",      "256x56x56",   "256x8x8",   CONV),
        ("",         "Pool",               "256x28x28",   "256x4x4",   POOL),
        ("conv4-1",  "3x3 conv, 512",      "512x28x28",   "512x4x4",   CONV),
        ("conv4-2",  "3x3 conv, 512",      "512x28x28",   "512x4x4",   CONV),
        ("conv4-3",  "3x3 conv, 512",      "512x28x28",   "512x4x4",   CONV),
        ("",         "Pool",               "512x14x14",   "512x2x2",   POOL),
        ("conv5-1",  "3x3 conv, 512",      "512x14x14",   "512x2x2",   CONV),
        ("conv5-2",  "3x3 conv, 512",      "512x14x14",   "512x2x2",   CONV),
        ("conv5-3",  "3x3 conv, 512",      "512x14x14",   "512x2x2",   CONV),
        ("",         "Pool",               "512x7x7",     "512x1x1",   POOL),
        ("",         "AdaptiveAvgPool 7x7","512x7x7",     "512x7x7",   AVG),
        ("fc6",      "FC 4096",            "4096",        "4096",      FCL),
        ("fc7",      "FC 4096",            "4096",        "4096",      FCL),
        ("fc8",      "FC 1000",            "1000",        "1000",      FCL),
        ("",         "Softmax",            "1000",        "1000",      SMAX),
    ]

    BW, BH, GAP = 6.4, 0.78, 0.17
    C1, C2 = BW + 0.30, BW + 3.05          # 두 모양 열의 왼쪽 끝
    fig, ax = plt.subplots(figsize=(9.6, 12.2))
    ax.axis("off")

    for i, (name, mid, s224, s32, (fill, edge)) in enumerate(rows):
        y = i * (BH + GAP)
        adaptive = mid.startswith("Adaptive")
        if adaptive:                        # 두 열이 갈라졌다 다시 만나는 줄
            ax.add_patch(plt.Rectangle((C2 - 0.34, y - GAP / 2), 2.75, BH + GAP,
                                       facecolor="#fff3cd", edgecolor="none",
                                       zorder=0))
        ax.add_patch(plt.Rectangle((0, y), BW, BH, facecolor=fill,
                                   edgecolor=edge, linewidth=1.3))
        ax.text(BW / 2, y + BH / 2, mid, ha="center", va="center",
                fontsize=9.5, color="#26323c")
        if name:
            ax.text(-0.28, y + BH / 2, name, ha="right", va="center",
                    fontsize=9.5, color="#26323c")
        ax.text(C1, y + BH / 2, s224, ha="left", va="center",
                fontsize=8.6, color="#6b7883")
        ax.text(C2, y + BH / 2, s32, ha="left", va="center", fontsize=8.6,
                color="#a06a10" if adaptive else "#6b7883",
                weight="bold" if adaptive else "normal")

    top = len(rows) * (BH + GAP)
    ax.text(BW / 2, -0.95, "VGG16", ha="center", fontsize=12, weight="bold")
    for x, head in ((C1, "input 224x224"), (C2, "input 32x32")):
        ax.text(x, top + 0.10, head, ha="left", va="bottom",
                fontsize=8.6, color="#5b6b7d", weight="bold")
    ax.text(C1, top + 0.72, "shape after the layer  (PyTorch, NCHW)",
            ha="left", va="bottom", fontsize=8.2, color="#8a97a4")
    ax.text(-0.28, top + 0.10, "13 conv + 3 FC = 16",
            ha="right", va="bottom", fontsize=8.2, color="#8a97a4")
    ax.text(C2 + 2.55, 19 * (BH + GAP) + BH / 2,
            "1x1 -> 7x7 here:\nthe only row where\nthe columns meet",
            fontsize=7.8, color="#a06a10", ha="left", va="center")

    ax.set_xlim(-2.5, C2 + 5.2)
    ax.set_ylim(-1.5, top + 1.5)
    fig.savefig("vgg16_layers.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print("  vgg16_layers.svg")


if __name__ == "__main__":
    fig_class_mean_templates()
    fig_vgg16_conv1()
    fig_pca_reconstructions()
    fig_depth_curves()
    fig_vgg16_layers()
    fig_transfer_learning()
