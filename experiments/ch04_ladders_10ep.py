"""4.1절의 여덟 칸을 **10 에포크**로 다시 잰다.

쪽에 실린 규약은 모든 칸을 5 에포크로 돌렸다. 그런데 3장의 2걸음만 10
에포크였던 탓에, 4.1절의 경고 상자는 "에포크 10 → 5가 만든 진짜 차이"를
설명하는 데 한 줄을 쓰고 있다. 규약을 10으로 올리면 그 교란 요인이
사라진다 — 이 스크립트가 그 값을 만든다.

바뀌는 것은 EPOCHS 하나뿐이다. 모델도, 전처리도, 최적화기도, 배치도
쪽에 실린 것과 글자 하나 다르지 않다.

쓰는 법
-------
일감을 나누어 여러 번 돌릴 수 있게 되어 있다. 결과는 일감마다 JSON 한
파일로 떨어지므로, 서로 덮어쓰지 않는다.

    python experiments/ch04_ladders_10ep.py main      --dataset MNIST
    python experiments/ch04_ladders_10ep.py spread    --dataset CIFAR10 --rung cnn
    python experiments/ch04_ladders_10ep.py decomp    --which shuffle
    python experiments/ch04_ladders_10ep.py determinism
    python experiments/ch04_ladders_10ep.py device

`--out` 으로 결과를 떨굴 자리를 정한다(붙박이: experiments/ch04_10ep/).
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

SEED, EPOCHS, BATCH, LR = 42, 10, 100, 1e-3        # 5 였던 것을 10 으로
ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

STATS = {
    "MNIST":   ((0.1307,), (0.3081,)),
    "CIFAR10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}
SHAPE = {"MNIST": (1, 28), "CIFAR10": (3, 32)}     # (채널 수, 한 변)


def loaders(name, shuffle_gen=None):
    mean, std = STATS[name]
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(mean, std)])
    cls = getattr(torchvision.datasets, name)
    tr = cls(root=str(ROOT / "data"), train=True, download=True, transform=tf)
    te = cls(root=str(ROOT / "data"), train=False, download=True, transform=tf)
    train_loader = (DataLoader(tr, batch_size=BATCH, shuffle=True, generator=shuffle_gen)
                    if shuffle_gen is not None
                    else DataLoader(tr, batch_size=BATCH, shuffle=True))
    return (train_loader,
            DataLoader(tr, batch_size=1000, shuffle=False),
            DataLoader(te, batch_size=1000, shuffle=False))


# === 1걸음: 템플릿 학습 — 학습이 없다 ========================================
def step1_templates(tr_eval, te, C, S, device=DEVICE):
    tot = torch.zeros(10, C, S, S, device=device)
    cnt = torch.zeros(10, device=device)
    for x, y in tr_eval:
        x, y = x.to(device), y.to(device)
        tot.index_add_(0, y, x)
        cnt.index_add_(0, y, torch.ones_like(y, dtype=torch.float))
    tmpl = tot / cnt[:, None, None, None]

    correct = n = 0
    for x, y in te:
        x, y = x.to(device), y.to(device)
        d = ((x[:, None] - tmpl[None]) ** 2).flatten(2).sum(2)
        correct += (d.argmin(1) == y).sum().item()
        n += y.numel()
    return 100.0 * correct / n


# === 2~4걸음 ================================================================
def step2_linear(C, S):
    return nn.Sequential(nn.Flatten(), nn.Linear(C * S * S, 10))


def step3_mlp(C, S):
    return nn.Sequential(nn.Flatten(), nn.Linear(C * S * S, 128),
                         nn.ReLU(), nn.Linear(128, 10))


class Step4CNN(nn.Module):
    """3장의 구조 그대로. C와 S만 데이터셋을 따른다."""

    def __init__(self, C, S):
        super().__init__()
        self.conv1 = nn.Conv2d(C, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * (S // 4) * (S // 4), 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout(torch.relu(self.fc1(x.flatten(1))))
        return self.fc2(x)


BUILD = {"linear": step2_linear, "mlp": step3_mlp, "cnn": Step4CNN}
LABEL = {"linear": "2 선형", "mlp": "3 MLP", "cnn": "4 CNN"}


def train_and_test(model, tr, te, device=DEVICE):
    model = model.to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=LR)
    for _ in range(EPOCHS):
        model.train()
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()

    model.eval()
    correct = n = 0
    with torch.no_grad():
        for x, y in te:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            n += y.numel()
    return 100.0 * correct / n


def save(out, name, payload):
    out.mkdir(parents=True, exist_ok=True)
    payload["epochs"] = EPOCHS
    payload["device"] = str(DEVICE)
    (out / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"  -> {out / (name + '.json')}", flush=True)


# === 일감들 ==================================================================
def task_main(args, out):
    """씨앗 42로 그 데이터셋의 네 칸을 잰다. 쪽의 3절 코드와 같은 차례다."""
    name = args.dataset
    C, S = SHAPE[name]
    torch.manual_seed(SEED)
    tr, tr_eval, te = loaders(name)

    res = {"dataset": name, "cells": {}}
    t0 = time.time()
    acc = step1_templates(tr_eval, te, C, S)
    res["cells"]["1 템플릿"] = {"acc": acc, "params": None}
    print(f"{name} 1 템플릿  {acc:.2f}%  ({time.time()-t0:.0f}s)", flush=True)

    for key in ("linear", "mlp", "cnn"):
        t0 = time.time()
        torch.manual_seed(SEED)
        model = BUILD[key](C, S)
        acc = train_and_test(model, tr, te)
        p = sum(q.numel() for q in model.parameters())
        res["cells"][LABEL[key]] = {"acc": acc, "params": p}
        print(f"{name} {LABEL[key]}  {acc:.2f}%  매개변수 {p:,}  ({time.time()-t0:.0f}s)",
              flush=True)
    save(out, f"main_{name}", res)


def task_spread(args, out):
    """한 칸을 씨앗 0~4로 다섯 번 돌려 퍼짐을 잰다."""
    name, key = args.dataset, args.rung
    C, S = SHAPE[name]
    accs = []
    for seed in args.seeds:
        t0 = time.time()
        torch.manual_seed(seed)
        tr, _, te = loaders(name)
        torch.manual_seed(seed)
        acc = train_and_test(BUILD[key](C, S), tr, te)
        accs.append(acc)
        print(f"{name} {LABEL[key]} 씨앗 {seed}  {acc:.2f}%  ({time.time()-t0:.0f}s)",
              flush=True)
    save(out, f"spread_{name}_{key}", {
        "dataset": name, "rung": LABEL[key], "seeds": args.seeds, "accs": accs,
        "lo": min(accs), "hi": max(accs), "spread": max(accs) - min(accs),
    })


def task_decomp(args, out):
    """CIFAR-10 CNN에서 초기화·섞기·드롭아웃을 하나씩만 풀어 퍼짐을 잰다."""
    name, key = "CIFAR10", "cnn"
    C, S = SHAPE[name]
    which = args.which
    accs = []
    for seed in args.seeds:
        t0 = time.time()
        init_seed = seed if which in ("init", "all") else 42
        shuf_seed = seed if which in ("shuffle", "all") else 42
        drop_seed = seed if which in ("dropout", "all") else 42

        torch.manual_seed(init_seed)
        model = Step4CNN(C, S)
        g = torch.Generator(); g.manual_seed(shuf_seed)
        tr, _, te = loaders(name, shuffle_gen=g)
        torch.manual_seed(drop_seed)
        acc = train_and_test(model, tr, te)
        accs.append(acc)
        print(f"{which} 만 바꿈, 씨앗 {seed}  {acc:.2f}%  ({time.time()-t0:.0f}s)",
              flush=True)
    save(out, f"decomp_{which}", {
        "which": which, "seeds": args.seeds, "accs": accs,
        "lo": min(accs), "hi": max(accs), "spread": max(accs) - min(accs),
    })


def task_determinism(args, out):
    """세 씨앗을 모두 못박고 같은 것을 두 번 돌린다."""
    C, S = SHAPE["CIFAR10"]
    accs = []
    for run in (1, 2):
        t0 = time.time()
        torch.manual_seed(42)
        model = Step4CNN(C, S)
        g = torch.Generator(); g.manual_seed(42)
        tr, _, te = loaders("CIFAR10", shuffle_gen=g)
        torch.manual_seed(42)
        acc = train_and_test(model, tr, te)
        accs.append(acc)
        print(f"{run}번째  {acc:.2f}%  ({time.time()-t0:.0f}s)", flush=True)
    save(out, "determinism", {"accs": accs, "identical": accs[0] == accs[1]})


def task_device(args, out):
    """같은 씨앗으로 MNIST MLP를 MPS와 CPU에서 각각 돌린다."""
    C, S = SHAPE["MNIST"]
    res = {}
    for dev in ("mps", "cpu"):
        d = torch.device(dev)
        t0 = time.time()
        torch.manual_seed(SEED)
        tr, _, te = loaders("MNIST")
        torch.manual_seed(SEED)
        acc = train_and_test(step3_mlp(C, S), tr, te, device=d)
        res[dev] = acc
        print(f"{dev}  {acc:.2f}%  ({time.time()-t0:.0f}s)", flush=True)
    res["gap"] = abs(res["mps"] - res["cpu"])
    save(out, "device", res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task", choices=["main", "spread", "decomp", "determinism", "device"])
    ap.add_argument("--dataset", choices=["MNIST", "CIFAR10"], default="MNIST")
    ap.add_argument("--rung", choices=["linear", "mlp", "cnn"], default="cnn")
    ap.add_argument("--which", choices=["init", "shuffle", "dropout", "all"], default="all")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default=str(ROOT / "experiments" / "ch04_10ep"))
    a = ap.parse_args()
    outdir = Path(a.out)
    print(f"  에포크 {EPOCHS}, 장치 {DEVICE}", flush=True)
    {"main": task_main, "spread": task_spread, "decomp": task_decomp,
     "determinism": task_determinism, "device": task_device}[a.task](a, outdir)
