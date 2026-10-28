"""튜토리얼 20: 맞춤 자동 미분 함수 — forward 와 backward 를 직접 적는다."""
import torch

torch.manual_seed(0)


def header(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


# ========================================================================
# 두 개의 맞춤 함수
# ========================================================================
class SquareFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)          # backward 가 x 를 쓴다
        return input ** 2

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * 2 * input        # VJP: 상류 그래디언트 x 2x


class ReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0             # x = 0 은 그대로 흘려보낸다
        return grad_input


def main():
    header("1. SquareFunction")
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    y = SquareFunction.apply(x)
    print(f"x      = {x.data.tolist()}")
    print(f"y = x^2 = {y.data.tolist()}   grad_fn = {type(y.grad_fn).__name__}")
    y.sum().backward()
    print(f"x.grad = {x.grad.tolist()}   (2x = {[2*v for v in x.data.tolist()]})")

    header("2. ReLUFunction")
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    y = ReLUFunction.apply(x)
    print(f"x        = {x.data.tolist()}")
    print(f"ReLU(x)  = {y.data.tolist()}")
    y.sum().backward()
    print(f"x.grad   = {x.grad.tolist()}")

    header("3. 맞춘 것이 정말 맞는가 — gradcheck")
    # gradcheck 는 수치 미분과 맞대어 본다. float64 와 작은 입력을 쓴다.
    a = torch.randn(4, dtype=torch.double, requires_grad=True)
    print(f"SquareFunction: {torch.autograd.gradcheck(SquareFunction.apply, (a,))}")
    # ReLU 는 0 에서 꺾이므로 0 근처를 피해 뽑는다
    b = torch.randn(4, dtype=torch.double).sign() * (torch.rand(4, dtype=torch.double) + 0.5)
    b.requires_grad_(True)
    print(f"ReLUFunction:   {torch.autograd.gradcheck(ReLUFunction.apply, (b,))}")

    header("4. 내장과 견주기 — forward 는 같고 backward 는 다르다")
    pts = torch.tensor([-1.0, 0.0, 1.0])

    xa = pts.clone().requires_grad_(True)
    ya = torch.relu(xa)
    ya.sum().backward()

    xb = pts.clone().requires_grad_(True)
    yb = ReLUFunction.apply(xb)
    yb.sum().backward()

    print(f"입력                {pts.tolist()}")
    print(f"forward  내장       {ya.data.tolist()}")
    print(f"forward  맞춤       {yb.data.tolist()}   같은가? {torch.equal(ya.data, yb.data)}")
    print(f"backward 내장       {xa.grad.tolist()}")
    print(f"backward 맞춤       {xb.grad.tolist()}   같은가? {torch.equal(xa.grad, xb.grad)}")
    print(f"\n갈리는 자리는 x = 0 한 점뿐이다: 내장 {xa.grad[1].item()}, 맞춤 {xb.grad[1].item()}")


if __name__ == "__main__":
    main()
