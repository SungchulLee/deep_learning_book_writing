"""모델 속 들여다보기 — 무엇이 세어지고 무엇이 저장되는가.

`print(model)` 한 줄이 꽤 많은 것을 보여 주지만, 거기 안 나오는 것이
있다. 모델을 고치거나 저장할 때 알아야 하는 것은 셋이다.

- **모듈 나무** — `named_children()` 은 바로 아래만, `named_modules()`
  는 전부
- **파라미터** — 학습되는 텐서. `parameters()` 가 준다
- **버퍼** — 학습되지는 않는데 **저장은 되는** 텐서. 배치 정규화의
  `running_mean` 이 여기 있다

세 번째가 자주 잊힌다. 파라미터 수를 세면 버퍼는 빠지는데, 저장
파일에는 들어간다.

실행: python 13_print_layers.py
"""
import torch
import torch.nn as nn


def header(title):
    print(f"\n=== {title} ===")


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, d)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.fc(x))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(4, 8)
        self.blocks = nn.Sequential(Block(8), Block(8))
        self.head = nn.Linear(8, 2)
        self.register_buffer("steps", torch.zeros(1))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


# ========================================================================
# 1. print(model)
# ========================================================================

def show_repr(net):
    header("1. print(model)")
    print(net)
    print("  들여쓰기가 중첩을 보여 준다")
    print("  그런데 파라미터 수도, 출력 모양도, 버퍼도 안 나온다")


# ========================================================================
# 2. 바로 아래만 vs 전부
# ========================================================================

def tree(net):
    header("2. children 과 modules")
    print("  named_children (바로 아래만):",
          [n for n, _ in net.named_children()])
    print(f"  named_modules  (전부 {len(list(net.modules()))}개):")
    for name, m in net.named_modules():
        depth = name.count(".") + (0 if name == "" else 1)
        print(f"    {'  ' * depth}{name or '(root)':20s} {type(m).__name__}")
    print("  modules() 는 자기 자신도 센다 — 그래서 하나 더 많다")
    print("  이름이 점으로 이어진 꼴이 그대로 state_dict 의 키가 된다")


# ========================================================================
# 3. 파라미터와 버퍼
# ========================================================================

def params_and_buffers(net):
    header("3. 파라미터와 버퍼")
    for name, p in net.named_parameters():
        print(f"  param  {name:22s} {str(tuple(p.shape)):10s} "
              f"requires_grad={p.requires_grad}")
    for name, b in net.named_buffers():
        print(f"  buffer {name:22s} {str(tuple(b.shape)):10s} "
              f"requires_grad={b.requires_grad}")
    print("  파라미터 수:", sum(p.numel() for p in net.parameters()))
    print("  버퍼  수   :", sum(b.numel() for b in net.buffers()),
          "<- 위 셈에 안 들어간다")
    print("  state_dict 키:")
    for k in net.state_dict():
        print("   ", k)
    print("  버퍼가 맨 앞에 끼어 있다. 저장은 되고 학습은 안 된다")


# ========================================================================
# 4. 실제로 버퍼가 쓰이는 자리
# ========================================================================

def real_buffer():
    header("4. 배치 정규화의 버퍼")
    bn = nn.BatchNorm1d(4)
    print("  파라미터:", [n for n, _ in bn.named_parameters()])
    print("  버퍼    :", [n for n, _ in bn.named_buffers()])
    before = bn.running_mean.clone()
    bn.train()
    bn(torch.randn(8, 4))
    print("  순전파 한 번 뒤 running_mean 이 바뀌었나:",
          not torch.equal(before, bn.running_mean))
    print("  경사는 없다:", bn.running_mean.grad_fn, bn.running_mean.requires_grad)
    print("  학습되지 않는데 순전파가 고친다 — 그래서 버퍼다")


# ========================================================================
# 5. 쓸모 있는 요약 한 줄
# ========================================================================

def summary(net):
    header("5. 층마다 수를 세어 본다")
    print(f"  {'이름':24s} {'종류':12s} {'파라미터':>10s}")
    total = 0
    for name, m in net.named_modules():
        own = sum(p.numel() for p in m.parameters(recurse=False))
        if own or not list(m.children()):
            print(f"  {name or '(root)':24s} {type(m).__name__:12s} {own:>10,}")
            total += own
    print(f"  {'합':24s} {'':12s} {total:>10,}")
    print("  recurse=False 로 그 층이 '제 몫으로' 가진 것만 센다")
    print("  안 그러면 부모가 자식 것을 다시 세어 합이 부풀어 오른다")


def main():
    torch.manual_seed(0)
    net = Net()
    show_repr(net)
    tree(net)
    params_and_buffers(net)
    real_buffer()
    summary(net)


if __name__ == "__main__":
    main()
