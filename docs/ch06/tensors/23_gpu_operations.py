"""튜토리얼 23: 가속기 다루기 — 언제 이기고 언제 지는가.

이 쪽의 수는 **기계마다 다르다.** 자리를 가르는 수(어느 크기에서 뒤집히는가)는
대개 비슷하게 나오지만, 밀리초 자체는 돌리는 기계의 것이다.
"""
import time

import torch
import torch.nn as nn

torch.manual_seed(0)


def header(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def pick_device():
    """있는 가속기를 고른다. 이 책은 어디서 돌든 같은 코드를 쓴다."""
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.cuda.synchronize
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.mps.synchronize
    return torch.device("cpu"), lambda: None


DEVICE, sync = pick_device()


def main():
    header("1. 어디서 도는가")
    print(f"cuda.is_available()        {torch.cuda.is_available()}")
    print(f"backends.mps.is_available() {torch.backends.mps.is_available()}")
    print(f"고른 장치                   {DEVICE}")

    header("2. 장치는 텐서에 붙는다")
    x = torch.randn(2, 3)
    y = x.to(DEVICE)
    print(f"x.device {x.device}   y.device {y.device}")
    print(f".to() 는 옮기지 않고 **새 텐서를 돌려준다**: x 는 그대로 {x.device}")
    try:
        _ = x + y
    except RuntimeError as e:
        print(f"장치가 다른 둘을 더하면: {type(e).__name__}")
        print(f"  {str(e).splitlines()[0][:72]}")

    header("3. 모델도 같은 장치에 있어야 한다")
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))
    model.to(DEVICE)                      # 모델은 제자리에서 옮겨진다
    devs = {p.device for p in model.parameters()}
    print(f"파라메터가 놓인 장치 {devs}")
    print(f"model.to() 는 돌려받지 않아도 된다 — nn.Module 은 제자리에서 옮긴다")

    header("4. 언제 이기고 언제 지는가")

    def bench(n, reps=20):
        a, b = torch.randn(n, n), torch.randn(n, n)
        for _ in range(3):
            a @ b                                   # 데우기
        t0 = time.perf_counter()
        for _ in range(reps):
            a @ b
        cpu = (time.perf_counter() - t0) / reps

        A, B = a.to(DEVICE), b.to(DEVICE)
        for _ in range(3):
            A @ B
        sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            A @ B
        sync()
        acc = (time.perf_counter() - t0) / reps
        return cpu, acc

    rows = {}
    print(f"{'n':>6}{'CPU ms':>10}{str(DEVICE) + ' ms':>12}{'빠르기':>11}")
    prev = None
    for n in (32, 128, 512, 1024, 2048):
        c, g = bench(n)
        rows[n] = (c, g)
        mark = "  <- 여기서 뒤집힌다" if prev is not None and prev < 1 <= c / g else ""
        prev = c / g
        print(f"{n:>6}{c*1e3:>10.3f}{g*1e3:>12.3f}{c/g:>10.2f}배{mark}")

    header("5. 작은 일에서 지는 까닭 — 바닥값")
    floor_cpu, floor_acc = rows[32][0], rows[32][1]
    print(f"가장 작은 n = 32 에서 걸린 때")
    print(f"  CPU   {floor_cpu*1e3:7.3f} ms")
    print(f"  {str(DEVICE):5s} {floor_acc*1e3:7.3f} ms   — {floor_acc/floor_cpu:.0f} 배")
    print()
    print("이 수는 셈이 아니라 **커널을 띄우고 기다리는 고정 비용**이다. 일을 아무리")
    print("줄여도 가속기는 이 아래로 내려가지 않는다. 그래서 CPU 가 그 바닥값만큼")
    print("걸릴 만큼 일이 커지기 전까지는 가속기가 이길 수 없다.")
    print()
    cross = next(n for n in (32, 128, 512, 1024, 2048) if rows[n][0] >= floor_acc)
    print(f"CPU 가 {floor_acc*1e3:.3f} ms 를 넘어서는 것은 n = {cross} 부터다.")

    header("6. 옮기는 값")
    n = 1024
    a = torch.randn(n, n)
    t0 = time.perf_counter(); A = a.to(DEVICE); sync()
    move = time.perf_counter() - t0
    for _ in range(3):
        A @ A
    sync()
    t0 = time.perf_counter()
    for _ in range(10):
        A @ A
    sync()
    mm = (time.perf_counter() - t0) / 10
    print(f"{n}x{n} 을 옮기는 데      {move*1e3:7.3f} ms")
    print(f"같은 크기 행렬곱 한 번   {mm*1e3:7.3f} ms")
    print(f"옮기는 값 = 행렬곱 {move/mm:.1f} 번치")


if __name__ == "__main__":
    main()
