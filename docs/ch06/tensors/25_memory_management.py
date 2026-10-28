"""튜토리얼 25: 기억 자리 다루기 — 무엇이 자리를 잡고 있는가."""
import time

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

torch.manual_seed(0)


def header(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


class Deep(nn.Module):
    """같은 블록을 n 번 쌓은 것. 체크포인트를 켜고 끌 수 있다."""

    def __init__(self, n=16, width=256):
        super().__init__()
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.Linear(width, width), nn.ReLU()) for _ in range(n)])

    def forward(self, x, every=0):
        for i, b in enumerate(self.blocks):
            use = every and (i % every == 0)
            x = checkpoint(b, x, use_reentrant=False) if use else b(x)
        return x


def saved_kb(net, x, every):
    """역전파가 끝날 때까지 **저장해 둔** 텐서의 크기를 센다."""
    sizes = []
    with torch.autograd.graph.saved_tensors_hooks(
            lambda t: (sizes.append(t.numel()), t)[1], lambda t: t):
        net(x, every=every).sum().backward()
    return sum(sizes) * 4 / 1024


def ms(net, x, every, reps=5):
    for _ in range(2):
        net(x, every=every).sum().backward()      # 데우기
    t0 = time.perf_counter()
    for _ in range(reps):
        net.zero_grad()
        net(x, every=every).sum().backward()
    return (time.perf_counter() - t0) / reps * 1e3


def main():
    header("1. 텐서 하나가 잡는 자리는 셈으로 나온다")
    x = torch.randn(1000, 1000)
    print(f"모양 {tuple(x.shape)}  원소 {x.nelement():,}개  하나당 {x.element_size()} 바이트")
    print(f"= {x.element_size() * x.nelement() / 1024**2:.2f} MB")
    print(f"float16 이면 {torch.randn(1000, 1000, dtype=torch.float16).element_size() * x.nelement() / 1024**2:.2f} MB")

    header("2. 제자리 연산은 새 자리를 잡지 않는다")
    x = torch.randn(1000, 1000)
    ptr, first = x.data_ptr(), x[0, 0].item()
    y = x + 1
    print(f"x + 1   -> 같은 저장소인가 {y.data_ptr() == ptr}   x 가 바뀌었나 {x[0,0].item() != first}")
    x.add_(1)
    print(f"x.add_(1) -> 같은 저장소인가 {x.data_ptr() == ptr}   x 가 바뀌었나 {x[0,0].item() != first}")

    header("3. 그래프를 들고 있는 것이 자리를 잡는다")
    x = torch.randn(100, 100, requires_grad=True)
    y = x ** 2
    print(f"y            requires_grad {y.requires_grad}  grad_fn {y.grad_fn is not None}")
    print(f"y.detach()   requires_grad {y.detach().requires_grad}  grad_fn {y.detach().grad_fn is not None}")
    with torch.no_grad():
        z = x ** 2
    print(f"no_grad 안   requires_grad {z.requires_grad}  grad_fn {z.grad_fn is not None}")

    header("4. 그래디언트 쌓기는 같은 값을 준다")
    m = nn.Linear(8, 1)
    X, y = torch.randn(32, 8), torch.randn(32, 1)
    m.zero_grad()
    nn.MSELoss()(m(X), y).backward()
    full = m.weight.grad.clone()
    m.zero_grad()
    for i in range(4):
        (nn.MSELoss()(m(X[i*8:(i+1)*8]), y[i*8:(i+1)*8]) / 4).backward()
    acc = m.weight.grad.clone()
    print(f"32 를 한 번에  {full[0, :3].tolist()}")
    print(f"8 씩 네 번     {acc[0, :3].tolist()}")
    print(f"같은가 {torch.allclose(full, acc, atol=1e-6)}   가장 큰 차이 {(full-acc).abs().max():.2e}")
    print("로스를 쌓는 횟수로 나누어 두는 것이 조건이다. 안 나누면 네 배가 된다.")

    header("5. 체크포인트 — 자리를 셈으로 바꾼다")
    net = Deep()
    x = torch.randn(64, 256, requires_grad=True)
    base_kb, base_ms = saved_kb(net, x, 0), ms(net, x, 0)
    print(f"{'체크포인트':>14}{'저장 KB':>10}{'자리':>9}{'ms':>9}{'셈':>9}")
    print(f"{'안 쓴다':>14}{base_kb:>10.0f}{'1.0배':>9}{base_ms:>9.2f}{'1.0배':>9}")
    for every, name in [(4, "네 블록마다"), (2, "두 블록마다"), (1, "모든 블록")]:
        k, t = saved_kb(net, x, every), ms(net, x, every)
        print(f"{name:>14}{k:>10.0f}{base_kb/k:>8.1f}배{t:>9.2f}{t/base_ms:>8.1f}배")

    header("6. 그 값은 모델 크기에 딸려 있다")
    print("5절의 '모든 블록'을 너비와 배치를 키워 가며 다시 잰다.")
    print(f"{'너비':>7}{'배치':>7}{'그냥 ms':>12}{'체크포인트 ms':>16}{'셈':>9}")
    for w, b in [(256, 64), (1024, 256), (2048, 512)]:
        torch.manual_seed(0)
        n2 = Deep(width=w)
        x2 = torch.randn(b, w, requires_grad=True)
        a, c = ms(n2, x2, 0, reps=3), ms(n2, x2, 1, reps=3)
        print(f"{w:>7}{b:>7}{a:>12.2f}{c:>16.2f}{c/a:>8.2f}배")


if __name__ == "__main__":
    main()
