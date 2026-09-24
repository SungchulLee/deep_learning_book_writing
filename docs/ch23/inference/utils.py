"""utils — 이 절의 쪽들이 함께 쓰는 잣대들.

모델을 작게 만드는 방법(양자화·가지치기·증류)을 견주려면 세 가지를 같은
방법으로 재야 한다. **정확도**, **크기**, **매개변수 수**다. 쪽마다 다르게
재면 "무엇이 얼마를 아꼈는가"를 맞대어 볼 수 없다.
"""

import random

import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int = 42) -> None:
    """파이썬·넘파이·토치의 씨앗을 한 번에 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, dataloader, device=None) -> float:
    """시험 정확도를 %로 돌려준다."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    correct = total = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(total, 1)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """매개변수 개수. trainable_only면 학습되는 것만 센다."""
    ps = model.parameters()
    if trainable_only:
        ps = (p for p in ps if p.requires_grad)
    return sum(p.numel() for p in ps)


def get_model_size(model: nn.Module) -> dict:
    """가중치와 버퍼가 차지하는 바이트를 잰다.

    매개변수 **수**가 아니라 **바이트**라는 점이 중요하다. 양자화는 개수를
    그대로 두고 자료형만 float32에서 int8로 바꾸므로, 개수로 재면 아무것도
    아낀 것이 없어 보인다.
    """
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total = param_bytes + buffer_bytes
    return {
        "bytes": total,
        "kb": total / 1024,
        "mb": total / (1024 ** 2),
        "params": count_parameters(model),
    }


def compare_model_sizes(models: dict, baseline: str = None) -> dict:
    """{이름: 모델}을 받아 크기를 나란히 찍는다. baseline 대비 몇 배인지도."""
    sizes = {name: get_model_size(m) for name, m in models.items()}
    if baseline is None:
        baseline = next(iter(sizes))
    base_mb = sizes[baseline]["mb"] or 1e-12
    print(f"{'model':24s} {'MB':>9s} {'params':>12s} {'vs ' + baseline:>10s}")
    for name, s in sizes.items():
        print(f"{name:24s} {s['mb']:9.3f} {s['params']:12,} {s['mb']/base_mb:9.2f}x")
    return sizes


def compare_accuracies(results: dict, baseline: str = None) -> dict:
    """{이름: 정확도} 또는 {이름: {'accuracy': ...}}를 받아 나란히 찍는다."""
    accs = {k: (v["accuracy"] if isinstance(v, dict) else v) for k, v in results.items()}
    if baseline is None:
        baseline = next(iter(accs))
    base = accs[baseline]
    print(f"{'model':24s} {'accuracy':>10s} {'vs ' + baseline:>12s}")
    for name, a in accs.items():
        print(f"{name:24s} {a:9.2f}% {a - base:+11.2f}p")
    return accs
