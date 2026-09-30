"""여섯 가지 등뼈를 같은 전이 얼개에 얹어 잰다.

규약은 4.6절과 같다. Resize -> ImageNet 정규화 -> 얼린 등뼈로 특징을 한 번만
뽑고 -> 선형 머리 하나를 Adam 1e-3, 5 에포크, 배치 100 으로 학습한다.

특징은 씨앗과 무관하다(등뼈가 eval 이고 드롭아웃이 없다). 그래서 등뼈마다
한 번만 뽑아 두고 머리만 씨앗 다섯으로 다시 학습한다. 퍼짐이 거의 공짜다.
"""
import json, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as M

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
OUT = Path("backbone_results.json")
SEEDS = [0, 1, 2, 3, 4]

# (이름, 만드는 함수, 가중치, 머리를 떼는 함수, 입력 크기)
def strip_vgg(m):      d = m.classifier[6].in_features; m.classifier = nn.Sequential(*list(m.classifier.children())[:-1]); return d
def strip_fc(m):       d = m.fc.in_features;            m.fc = nn.Identity();            return d
def strip_cls(m, i):   d = m.classifier[i].in_features; m.classifier[i] = nn.Identity(); return d
def strip_vit(m):      d = m.heads.head.in_features;    m.heads.head = nn.Identity();    return d

BACKBONES = [
    ("VGG16",          M.vgg16,               M.VGG16_Weights.IMAGENET1K_V1,               strip_vgg,                 224),
    ("Inception v3",   M.inception_v3,        M.Inception_V3_Weights.IMAGENET1K_V1,        strip_fc,                  299),
    ("ResNet18",       M.resnet18,            M.ResNet18_Weights.IMAGENET1K_V1,            strip_fc,                  224),
    ("MobileNetV3-L",  M.mobilenet_v3_large,  M.MobileNet_V3_Large_Weights.IMAGENET1K_V1,  lambda m: strip_cls(m, 3), 224),
    ("EfficientNet-B0",M.efficientnet_b0,     M.EfficientNet_B0_Weights.IMAGENET1K_V1,     lambda m: strip_cls(m, 1), 224),
    ("ViT-B/16",       M.vit_b_16,            M.ViT_B_16_Weights.IMAGENET1K_V1,            strip_vit,                 224),
]


def loaders(size):
    tf = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    tr = torchvision.datasets.CIFAR10("./data", train=True,  download=True, transform=tf)
    te = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=tf)
    return tr, te


@torch.no_grad()
def extract(net, ds, size):
    feats, labels = [], []
    for x, y in DataLoader(ds, batch_size=100, shuffle=False, num_workers=0):
        assert x.shape[1:] == (3, size, size)
        feats.append(net(x.to(device)).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def train_head(Xtr, ytr, Xte, yte, dim, seed):
    torch.manual_seed(seed)
    head = nn.Linear(dim, 10).to(device)
    opt = optim.Adam(head.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    ld = DataLoader(TensorDataset(Xtr, ytr), batch_size=100, shuffle=True, generator=g)
    for _ in range(5):
        head.train()
        for xb, yb in ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(head(xb), yb).backward(); opt.step()
    head.eval()
    with torch.no_grad():
        pred = head(Xte.to(device)).argmax(1).cpu()
    return 100.0 * (pred == yte).float().mean().item()


results = json.loads(OUT.read_text()) if OUT.exists() else {}
only = sys.argv[1:] or None
for name, ctor, weights, strip, size in BACKBONES:
    if name in results or (only and name not in only):
        continue
    t0 = time.time()
    net = ctor(weights=weights).to(device).eval()
    dim = strip(net)
    frozen = sum(p.numel() for p in net.parameters())
    tr, te = loaders(size)
    Xtr, ytr = extract(net, tr, size)
    Xte, yte = extract(net, te, size)
    sec = time.time() - t0
    del net
    accs = [train_head(Xtr, ytr, Xte, yte, dim, s) for s in SEEDS]
    results[name] = dict(dim=dim, size=size, frozen=frozen, head=dim * 10 + 10,
                         accs=accs, mean=sum(accs) / len(accs),
                         spread=max(accs) - min(accs), lo=min(accs), hi=max(accs),
                         extract_sec=sec)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    r = results[name]
    print(f"{name:16s} {r['mean']:6.2f}%  퍼짐 {r['spread']:.2f} "
          f"({r['lo']:.2f}~{r['hi']:.2f})  특징 {dim:5d}  "
          f"얼림 {frozen:11,d}  머리 {r['head']:6,d}  뽑기 {sec:.0f}s", flush=True)
    del Xtr, Xte
print("\n끝. 결과는", OUT.resolve())
