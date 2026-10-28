"""torch.no_grad — 끄는 것은 역전파가 아니라 기록이다.

`with torch.no_grad():` 안에서는 새로 만드는 텐서에 `grad_fn` 이
붙지 않는다. 그래서 그래프가 안 세워지고, 역전파가 쓰려고 저장해 둘
중간값도 없다.

자주 어긋나는 짐작이 둘 있다.

- "no_grad 안에서는 backward 를 못 부른다" — 부를 수 있다. 그래프를
  **블록 밖에서 세웠다면** 멀쩡히 돈다
- "no_grad 가 텐서의 requires_grad 를 끈다" — 안 끈다. 블록 안에서
  **만드는** 것에만 영향을 준다

실행: python 10_with_torch_no_grad.py
"""
import torch
import torch.nn as nn


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. 블록 안에서 만든 것만 달라진다
# ========================================================================

def scope():
    header("1. 범위는 코드 블록이다")
    w = torch.randn(3, requires_grad=True)
    print("  들어가기 전 w.requires_grad :", w.requires_grad)
    with torch.no_grad():
        print("  블록 안 w.requires_grad     :", w.requires_grad, " <- 그대로다")
        inside = w * 2
        print("  블록 안에서 만든 것          :", inside.requires_grad)
    outside = w * 2
    print("  블록 밖에서 만든 것          :", outside.requires_grad)
    print("  no_grad 는 텐서가 아니라 '여기서 하는 일' 에 건다")


# ========================================================================
# 2. 막는 것은 그래프 세우기지 역전파가 아니다
# ========================================================================

def what_is_blocked():
    header("2. 무엇이 막히는가")
    w = torch.randn(2, requires_grad=True)
    loss = (w * 2).sum()                 # 그래프를 블록 밖에서 세운다
    with torch.no_grad():
        loss.backward()                  # 역전파는 블록 안에서
    print("  밖에서 세운 그래프를 안에서 역전파:", w.grad.tolist(), " 된다")

    w2 = torch.randn(2, requires_grad=True)
    with torch.no_grad():
        loss2 = (w2 * 2).sum()           # 그래프를 블록 안에서 세우면
    print("  안에서 만든 loss2.grad_fn        :", loss2.grad_fn)
    try:
        loss2.backward()
    except RuntimeError as e:
        print("  그것을 역전파하면               :",
              str(e).splitlines()[0][:52])
    print("  그러니 막히는 것은 세우기다. 걷기가 아니다")


# ========================================================================
# 3. 안쪽에서 다시 켤 수 있다
# ========================================================================

def nesting():
    header("3. 겹쳐 쓰기")
    w = torch.randn(2, requires_grad=True)
    print("  지금 추적 켜져 있나:", torch.is_grad_enabled())
    with torch.no_grad():
        print("    no_grad 안       :", torch.is_grad_enabled())
        with torch.enable_grad():
            print("    enable_grad 안   :", torch.is_grad_enabled())
            a = w * 2
        b = w * 3
    print("  enable_grad 안에서 만든 것:", a.requires_grad)
    print("  그 바깥에서 만든 것       :", b.requires_grad)
    print("  평가 루프 한가운데서 한 조각만 미분해야 할 때 쓴다")


# ========================================================================
# 4. 꾸밈씨로도 쓴다
# ========================================================================

def decorator():
    header("4. 함수 전체에 걸기")

    @torch.no_grad()
    def predict(model, x):
        return model(x)

    torch.manual_seed(0)
    net = nn.Linear(4, 2)
    out = predict(net, torch.randn(3, 4))
    print("  @torch.no_grad() 를 붙인 함수의 결과:", out.requires_grad)
    print("  평가 함수에 한 번 붙여 두면 부를 때마다 감쌀 필요가 없다")


# ========================================================================
# 5. 평가 루프에서 eval() 과 함께
# ========================================================================

def eval_loop():
    header("5. eval() 과 no_grad 는 다른 일이다")
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(8, 16), nn.Dropout(0.5), nn.Linear(16, 1))
    x = torch.randn(4, 8)

    net.train()
    with torch.no_grad():
        a, b = net(x), net(x)
    print("  train() + no_grad: 두 번이 같은가", torch.allclose(a, b),
          " <- 드롭아웃이 켜져 있다")
    net.eval()
    with torch.no_grad():
        c, d = net(x), net(x)
    print("  eval()  + no_grad: 두 번이 같은가", torch.allclose(c, d))
    net.eval()
    e = net(x)
    print("  eval() 만          : grad_fn =",
          type(e.grad_fn).__name__, " <- 그래프를 들고 있다")
    print("  eval() 은 '무엇을 셈할지', no_grad 는 '기록할지' 를 정한다")


def main():
    scope()
    what_is_blocked()
    nesting()
    decorator()
    eval_loop()


if __name__ == "__main__":
    main()
