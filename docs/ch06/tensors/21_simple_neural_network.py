"""튜토리얼 21: 단순한 뉴럴 넷트웍 — 텐서에서 학습 한 걸음까지."""
import torch
import torch.nn as nn

torch.manual_seed(0)


def header(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def main():
    header("1. 레이어 하나의 모양")
    layer = nn.Linear(3, 2)
    print(f"{layer}")
    print(f"weight {tuple(layer.weight.shape)}   bias {tuple(layer.bias.shape)}")
    print(f"(3,) 하나를 넣으면 -> {tuple(layer(torch.randn(1, 3)).shape)}")

    header("2. 파라메터를 레이어별로 센다")
    model = SimpleNet()
    total = 0
    for name, p in model.named_parameters():
        total += p.numel()
        print(f"  {name:12s} {str(tuple(p.shape)):10s} {p.numel():>5d}")
    print(f"  {'합':12s} {'':10s} {total:>5d}")

    header("3. 로스와 그래디언트")
    pred = torch.tensor([[2.5], [3.0], [1.5]], requires_grad=True)
    targ = torch.tensor([[2.0], [3.0], [2.0]])
    loss = nn.MSELoss()(pred, targ)
    loss.backward()
    print(f"예측 - 목표 = {(pred - targ).detach().flatten().tolist()}")
    print(f"MSE        = {loss.item():.4f}   (제곱의 평균)")
    print(f"d로스/d예측 = {pred.grad.flatten().tolist()}")
    print(f"            = 2(예측 - 목표)/n, n = 3")

    header("4. 한 걸음이 실제로 무엇을 바꾸는가")
    model = SimpleNet()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    X, y = torch.randn(32, 10), torch.randn(32, 1)
    w_before = model.fc3.weight.detach().clone()
    opt.zero_grad()
    nn.MSELoss()(model(X), y).backward()
    g = model.fc3.weight.grad.detach().clone()
    opt.step()
    w_after = model.fc3.weight.detach()
    print(f"fc3.weight 첫 세 개  이전 {w_before[0, :3].tolist()}")
    print(f"                     이후 {w_after[0, :3].tolist()}")
    print(f"바뀐 양               {(w_after - w_before)[0, :3].tolist()}")
    print(f"-lr * grad            {(-0.1 * g)[0, :3].tolist()}")
    print(f"둘이 같은가? {torch.allclose(w_after - w_before, -0.1 * g)}")

    header("5. eval() 과 no_grad() 는 다른 일을 한다")
    model = SimpleNet()
    x = torch.randn(5, 10)

    model.eval()
    out = model(x)
    print(f"eval() 뒤       requires_grad = {out.requires_grad}, grad_fn = {type(out.grad_fn).__name__}")
    with torch.no_grad():
        out2 = model(x)
    print(f"no_grad() 안    requires_grad = {out2.requires_grad}, grad_fn = {out2.grad_fn}")
    print("-> 그래디언트를 끄는 것은 no_grad() 다. eval() 은 끄지 않는다.")

    # 그러면 eval() 은 무엇을 바꾸는가 — 이 모델에서는 아무것도 바꾸지 않는다
    model.train(); a = model(x)
    model.eval();  b = model(x)
    print(f"\n이 모델에서 train 과 eval 의 출력이 같은가? {torch.equal(a, b)}")

    class WithDropout(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 1)
            self.drop = nn.Dropout(0.5)

        def forward(self, x):
            return self.drop(self.fc(x))

    d = WithDropout()
    d.train(); a = d(x)
    d.eval();  b = d(x)
    print(f"드롭아웃을 넣으면 같은가?                 {torch.equal(a, b)}")
    print("-> eval() 이 바꾸는 것은 드롭아웃·배치 정규화처럼 모드를 타는 레이어뿐이다.")

    header("6. 속성으로 둔 텐서는 파라메터가 아니다")
    class Two(nn.Module):
        def __init__(self):
            super().__init__()
            self.plain = torch.ones(3)                 # 그냥 속성
            self.param = nn.Parameter(torch.ones(3))   # 등록된 파라메터

    t = Two()
    print(f"parameters() 가 세는 수 : {sum(p.numel() for p in t.parameters())}")
    print(f"state_dict() 의 열쇠     : {list(t.state_dict().keys())}")
    print(f"plain.requires_grad      : {t.plain.requires_grad}")
    print("-> 클래스 속성에 넣었다고 학습되지 않는다. 옵티마이저도 저장도 지나친다.")


if __name__ == "__main__":
    main()
