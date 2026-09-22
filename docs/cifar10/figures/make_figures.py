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


if __name__ == "__main__":
    fig_class_mean_templates()
