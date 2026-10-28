"""nn.Module 을 상속해 쓰기 — Sequential 로는 못 하는 것들.

`nn.Sequential` 은 "한 줄로 지나가는" 모델만 적을 수 있다. 입력을
두 갈래로 보내거나, 건너뛰어 더하거나, 중간값을 꺼내 쓰려면
`forward` 를 손으로 적어야 한다.

속성 이름이 그대로 `state_dict` 의 키가 되는 것은 「OrderedDict 으로
레이어에 이름 붙이기」와 같다. 다른 것은 **그 사이에 무엇을 끼워 넣을
수 있느냐**다.

실행: python 15_layer_naming_with_custom_module.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. 속성 이름이 키가 된다
# ========================================================================

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(5, 3)
        self.head = nn.Linear(3, 1)

    def forward(self, x):
        return self.head(F.relu(self.encoder(x)))


def names():
    header("1. 속성 이름이 키가 된다")
    net = TinyNet()
    print("  state_dict 키:", list(net.state_dict().keys()))
    print("  ReLU 를 F.relu 로 적어 모듈이 아예 없다")
    print("  모듈 목록:", [n for n, _ in net.named_children()])
    print("  Sequential 과 달리 파라미터 없는 층을 안 만들어도 된다")


# ========================================================================
# 2. Sequential 로는 못 적는 것 — 건너뛰기
# ========================================================================

class Residual(nn.Module):
    def __init__(self, d, skip=True):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.skip = skip

    def forward(self, x):
        h = self.fc2(F.relu(self.fc1(x)))
        return x + h if self.skip else h          # 입력을 더한다


def skip():
    header("2. 건너뛰어 더하기")
    torch.manual_seed(0)
    block = Residual(4)
    x = torch.randn(2, 4)
    print("  입력 모양", tuple(x.shape), "-> 출력", tuple(block(x).shape))
    print("  x 를 한 번 더 쓰는 식은 Sequential 로 적을 수 없다")
    print("  Sequential 은 앞 층의 출력만 다음 층에 넘기기 때문이다")

    print("  블록 스무 개를 쌓고 첫 블록에 닿는 경사를 재어 본다")
    for use_skip in (True, False):
        torch.manual_seed(0)
        deep = nn.Sequential(*[Residual(16, use_skip) for _ in range(20)])
        deep(torch.randn(8, 16)).pow(2).mean().backward()
        first = deep[0].fc1.weight.grad.abs().mean().item()
        last = deep[-1].fc1.weight.grad.abs().mean().item()
        print(f"    지름길 {str(use_skip):5s} 첫 블록 {first:.3e}  "
              f"마지막 블록 {last:.3e}  비 {first / last:.2e}")
    print("  지름길이 없으면 첫 블록의 경사가 사라진다")
    print("  더하기의 미분이 1 이라 지름길이 경사를 그대로 흘려 보낸다")


# ========================================================================
# 3. 두 갈래로 보내기
# ========================================================================

class TwoHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Linear(4, 8)
        self.cls = nn.Linear(8, 3)
        self.reg = nn.Linear(8, 1)

    def forward(self, x):
        h = F.relu(self.trunk(x))
        return self.cls(h), self.reg(h)          # 둘을 돌려준다


def branching():
    header("3. 출력이 둘")
    torch.manual_seed(0)
    net = TwoHeads()
    a, b = net(torch.randn(2, 4))
    print("  분류 머리", tuple(a.shape), "  회귀 머리", tuple(b.shape))
    print("  Sequential 은 텐서 하나만 돌려줄 수 있다")

    loss = F.cross_entropy(a, torch.tensor([0, 1])) + b.pow(2).mean()
    loss.backward()
    print("  두 로스를 더해 한 번 역전파하면 trunk 가 둘 다에게서 받는다")
    print("  trunk.weight.grad 가 있는가:", net.trunk.weight.grad is not None)


# ========================================================================
# 4. 중간값을 꺼내 쓰기
# ========================================================================

class WithTap(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 8)
        self.b = nn.Linear(8, 8)
        self.c = nn.Linear(8, 2)

    def forward(self, x, return_hidden=False):
        h1 = F.relu(self.a(x))
        h2 = F.relu(self.b(h1))
        out = self.c(h2)
        return (out, h1, h2) if return_hidden else out


def tapping():
    header("4. 중간값 꺼내기")
    torch.manual_seed(0)
    net = WithTap()
    x = torch.randn(2, 4)
    print("  보통 부르면 :", tuple(net(x).shape))
    out, h1, h2 = net(x, return_hidden=True)
    print("  중간값까지  :", tuple(out.shape), tuple(h1.shape), tuple(h2.shape))
    print("  forward 에 인자를 더 받을 수 있는 것도 Sequential 과 다르다")
    print("  특징 뽑기, 증류, 시각화가 전부 이 자리에서 나온다")


# ========================================================================
# 5. 흔한 실수 — super().__init__() 을 빼먹으면
# ========================================================================

def forgot_super():
    header("5. super().__init__() 을 빼먹으면")

    class Broken(nn.Module):
        def __init__(self):
            self.fc = nn.Linear(2, 2)          # super() 를 안 불렀다

    try:
        Broken()
    except AttributeError as e:
        print("  만들 때:", str(e).splitlines()[0][:72])
    print("  nn.Module 이 속성을 가로채려면 먼저 제 살림을 차려야 한다")
    print("  그 살림이 _parameters, _modules 같은 사전들이다")


def main():
    names()
    skip()
    branching()
    tapping()
    forgot_super()


if __name__ == "__main__":
    main()
