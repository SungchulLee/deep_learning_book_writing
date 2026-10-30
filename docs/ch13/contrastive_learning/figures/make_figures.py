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


# === 그림으로 보는 AI: 로스가 보는 자리와 우리가 쓰는 자리 ====================
#
# SimCLR 논문 4.2 절의 결과 — 투영 헤드 **앞**의 h 가 **뒤**의 z 보다 선형 평가에서
# 10% 넘게 낫다 — 를 작게 재현한다. 로스는 z 에만 걸리는데 쓰는 것은 h 다.
#
# ⚠️ 이 함수는 **학습을 한다.** 이 기계에서 3 에포크에 3 분 안팎 걸린다.
#    멈춘 것이 아니니 기다린다.

def projection_gap(epochs=3):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import datasets, transforms

    torch.manual_seed(0)
    np.random.seed(0)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    tf = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.3, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()])

    class TwoViews(torch.utils.data.Dataset):
        def __init__(self, ds, t):
            self.ds, self.t = ds, t

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, i):
            im, y = self.ds[i]
            return self.t(im), self.t(im), y

    base = datasets.CIFAR10("./data", train=True, download=False)
    dl = torch.utils.data.DataLoader(TwoViews(base, tf), batch_size=256,
                                     shuffle=True, num_workers=0,
                                     drop_last=True)

    class Net(nn.Module):
        def __init__(self, d=256, p=64):
            super().__init__()
            self.f = nn.Sequential(
                nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, d, 3, 1, 1), nn.BatchNorm2d(d), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.g = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, p))

        def forward(self, x):
            h = self.f(x)
            return h, self.g(h)

    def nt_xent(z1, z2, t=0.5):
        z = F.normalize(torch.cat([z1, z2]), dim=1)
        n = z1.shape[0]
        sim = z @ z.T / t
        sim.fill_diagonal_(-9e15)
        tgt = torch.cat([torch.arange(n, 2 * n),
                         torch.arange(0, n)]).to(z.device)
        return F.cross_entropy(sim, tgt)

    model = Net().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(epochs):
        tot, nb = 0.0, 0
        for v1, v2, _ in dl:
            _, za = model(v1.to(dev))
            _, zb = model(v2.to(dev))
            loss = nt_xent(za, zb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        losses.append(tot / nb)

    ev = transforms.ToTensor()
    tr = datasets.CIFAR10("./data", train=True, download=False, transform=ev)
    te = datasets.CIFAR10("./data", train=False, download=False, transform=ev)

    def feats(ds):
        hs, zs, ys = [], [], []
        model.eval()
        with torch.no_grad():
            for x, y in torch.utils.data.DataLoader(ds, batch_size=512):
                h, z = model(x.to(dev))
                hs.append(h.cpu())
                zs.append(z.cpu())
                ys.append(y)
        return torch.cat(hs), torch.cat(zs), torch.cat(ys)

    h_tr, z_tr, y_tr = feats(tr)
    h_te, z_te, y_te = feats(te)

    def probe(a, b):
        a = (a - a.mean(0)) / (a.std(0) + 1e-6)
        b = (b - b.mean(0)) / (b.std(0) + 1e-6)
        clf = nn.Linear(a.shape[1], 10).to(dev)
        o = torch.optim.Adam(clf.parameters(), lr=1e-2)
        a, b = a.to(dev), b.to(dev)
        yt, ye = y_tr.to(dev), y_te.to(dev)
        for _ in range(200):
            o.zero_grad()
            F.cross_entropy(clf(a), yt).backward()
            o.step()
        with torch.no_grad():
            return (clf(b).argmax(1) == ye).float().mean().item()

    acc_h, acc_z = probe(h_tr, h_te), probe(z_tr, z_te)
    assert acc_h > acc_z, "h 가 z 보다 낫지 않으면 이 그림의 이야기가 없다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.0),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # --- 왼쪽: 로스가 걸리는 자리 ------------------------------------------
    boxes = [(0.06, "image", INK), (0.30, "encoder f", INK),
             (0.56, "head g", ACCENT)]
    for x, lab, col in boxes:
        ax1.add_patch(plt.Rectangle((x, 0.52), 0.19, 0.17, facecolor=col,
                                    alpha=0.14, edgecolor=col, linewidth=1.1))
        ax1.text(x + 0.095, 0.605, lab, ha="center", va="center",
                 fontsize=9.5, color=INK, family="monospace")
    for x0, x1 in ((0.25, 0.30), (0.49, 0.56)):
        ax1.annotate("", xy=(x1, 0.605), xytext=(x0, 0.605),
                     arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax1.text(0.525, 0.46, "h", ha="center", fontsize=12, color=INK,
             weight="bold")
    ax1.text(0.82, 0.605, "z", ha="center", va="center", fontsize=12,
             color=ACCENT, weight="bold")
    ax1.annotate("", xy=(0.79, 0.605), xytext=(0.75, 0.605),
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.2))

    ax1.annotate("", xy=(0.82, 0.70), xytext=(0.82, 0.86),
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.4))
    ax1.text(0.82, 0.90, "the loss acts here", ha="center", fontsize=9,
             color=ACCENT, weight="bold")
    ax1.annotate("", xy=(0.525, 0.40), xytext=(0.525, 0.24),
                 arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax1.text(0.525, 0.19, "this is what you keep", ha="center", fontsize=9,
             color=INK, weight="bold")
    ax1.text(0.5, 0.05, "the head is thrown away after pretraining",
             ha="center", fontsize=8.4, color=INK, alpha=0.8, style="italic")
    ax1.set_xlim(0, 1.05)
    ax1.set_ylim(0, 1.0)
    ax1.axis("off")
    ax1.set_title("where the two live", fontsize=10, color=INK, weight="bold")

    # --- 오른쪽: 재어 보면 -------------------------------------------------
    ax2.bar([0, 1], [acc_h, acc_z], color=[INK, ACCENT], width=0.52,
            alpha=0.88)
    for i, v in enumerate((acc_h, acc_z)):
        ax2.text(i, v + 0.012, f"{v:.4f}", ha="center", fontsize=10,
                 color=INK, weight="bold")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["h\nbefore the head", "z\nafter the head"],
                        fontsize=9)
    ax2.set_ylabel("linear probe accuracy", fontsize=9)
    ax2.set_ylim(0, max(acc_h, acc_z) * 1.28)
    ax2.set_title(f"{(acc_h - acc_z) * 100:.1f} points apart", fontsize=10,
                  color=INK, weight="bold")
    ax2.grid(alpha=0.13, axis="y")
    ax2.tick_params(labelsize=8, colors=INK)
    for s in ax2.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    fig.text(0.5, -0.05,
             f"a small run, {epochs} epochs on CIFAR-10 - the size of the gap "
             f"is this machine's, the direction is the paper's.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "projection_gap.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote projection_gap.svg   h {acc_h:.4f}  z {acc_z:.4f}  "
          f"loss {losses[-1]:.4f}")


# === 그림으로 보는 AI: 75% 라는 수가 어디서 오는가 ============================
#
# MAE 논문 1절의 근거 — 그림은 공간적으로 겹치는 정보가 많아서 "a missing patch
# can be recovered from neighboring patches with little high-level understanding"
# — 을 직접 잰다. 마스크 비율이 낮으면 이웃 평균만으로 가린 자리를 꽤 맞힌다.
# 비율을 올리면 그 길이 끊긴다. 수는 전부 여기서 셈한다.

PATCH = 4
RATIOS = (0.15, 0.25, 0.50, 0.75, 0.90)


def mask_ratio():
    import torch
    from torchvision import datasets, transforms

    torch.manual_seed(0)
    np.random.seed(0)
    ds = datasets.CIFAR10("./data", train=True, download=False,
                          transform=transforms.ToTensor())
    x = torch.stack([ds[i][0] for i in range(300)])
    g_side = 32 // PATCH

    def patch_means(img):
        pv = img.unfold(2, PATCH, PATCH).unfold(3, PATCH, PATCH)
        return pv.mean((-1, -2))                       # (N,3,G,G)

    pm = patch_means(x)

    def errors(ratio, seed=1):
        gen = torch.Generator().manual_seed(seed)
        keep = (torch.rand(g_side * g_side, generator=gen) > ratio)
        k = keep.view(1, 1, g_side, g_side).float()
        num = torch.zeros_like(pm)
        den = torch.zeros_like(pm)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            num += torch.roll(pm * k, (dy, dx), dims=(2, 3))
            den += torch.roll(k.expand_as(pm), (dy, dx), dims=(2, 3))
        nb = num / (den + 1e-9)
        hid = ~keep.view(g_side, g_side)
        e_nb = ((nb - pm)[:, :, hid] ** 2).mean().sqrt().item()
        glob = pm.mean((2, 3), keepdim=True).expand_as(pm)
        e_g = ((glob - pm)[:, :, hid] ** 2).mean().sqrt().item()
        return e_nb, e_g, keep

    nb_err, base_err, keeps = [], [], {}
    for r in RATIOS:
        a, b, keep = errors(r)
        nb_err.append(a)
        base_err.append(b)
        keeps[r] = keep

    # 그림이 주장하는 것을 먼저 확인한다
    assert nb_err[0] < base_err[0] * 0.75, \
        "낮은 비율에서 이웃 보간이 크게 낫지 않으면 이 그림의 이야기가 없다"
    i75 = RATIOS.index(0.75)
    assert nb_err[i75] > base_err[i75], \
        "75% 에서 보간이 기준선보다 못하지 않으면 절벽 이야기가 없다"

    fig = plt.figure(figsize=(11.0, 4.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], hspace=0.45,
                          wspace=0.28)
    ax = fig.add_subplot(gs[:, 0])
    axa = fig.add_subplot(gs[0, 1])
    axb = fig.add_subplot(gs[1, 1])

    xs = np.arange(len(RATIOS))
    ax.plot(xs, base_err, "s:", color=INK, alpha=0.6, linewidth=1.5, ms=6,
            label="fill with the image's mean colour")
    ax.plot(xs, nb_err, "o-", color=ACCENT, linewidth=2.2, ms=7,
            label="fill from the visible neighbours")
    ax.axvline(i75, color=ACCENT, linestyle="--", linewidth=1.1, alpha=0.5)
    ax.annotate("MAE's 75%", (i75, min(nb_err) * 1.02),
                textcoords="offset points", xytext=(9, 0), fontsize=8.6,
                color=ACCENT, va="center")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r:.0%}" for r in RATIOS], fontsize=9)
    ax.set_xlabel("share of patches hidden", fontsize=9.5)
    ax.set_ylabel("error on the hidden patches", fontsize=9.5)
    ax.set_title("where the shortcut runs out", fontsize=10.5, color=INK,
                 weight="bold")
    ax.legend(fontsize=8.5, framealpha=0.92, loc="upper left")
    ax.grid(alpha=0.14)
    ax.tick_params(labelsize=8, colors=INK)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    # --- 오른쪽: 15% 와 75% 는 눈으로도 다르다 ----------------------------
    img = x[7]
    for axi, r in ((axa, 0.15), (axb, 0.75)):
        keep = keeps[r].view(g_side, g_side)
        shown = img.clone()
        for a_ in range(g_side):
            for b_ in range(g_side):
                if not keep[a_, b_]:
                    shown[:, a_ * PATCH:(a_ + 1) * PATCH,
                          b_ * PATCH:(b_ + 1) * PATCH] = 0.78
        axi.imshow(shown.permute(1, 2, 0).numpy())
        axi.set_title(f"{r:.0%} hidden", fontsize=9, color=INK)
        axi.set_xticks([])
        axi.set_yticks([])
        for s in axi.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             "at a low ratio the neighbours give the answer away. "
             "that is the argument for hiding most of the image.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.savefig(HERE / "mask_ratio.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote mask_ratio.svg   "
          + "  ".join(f"{r:.0%}: nb {a:.4f} base {b:.4f}"
                      for r, a, b in zip(RATIOS, nb_err, base_err)))


if __name__ == "__main__":
    color_shortcut()
    projection_gap()
    mask_ratio()
