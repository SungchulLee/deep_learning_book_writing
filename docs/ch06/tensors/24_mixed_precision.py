"""튜토리얼 24: 섞인 정밀도 — 왜 로스를 키워 두는가.

여기 나오는 것은 모두 **자료형의 성질**이라 가속기가 없어도 그대로 돌아간다.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)


def header(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEV = pick_device()


def main():
    header("1. 세 자료형이 담는 것")
    print(f"{'':10s}{'바이트':>7}{'가장 큰 수':>13}{'가장 작은 정규수':>18}{'eps':>12}")
    for name, dt in [("float32", torch.float32),
                     ("float16", torch.float16),
                     ("bfloat16", torch.bfloat16)]:
        i = torch.finfo(dt)
        b = torch.tensor([0], dtype=dt).element_size()
        print(f"{name:10s}{b:>7d}{i.max:>13.3e}{i.tiny:>18.3e}{i.eps:>12.3e}")
    print()
    print("float16 은 범위를 줄여 정밀도를 지켰고, bfloat16 은 반대로 했다.")
    print("bfloat16 의 범위는 float32 와 같고, 대신 eps 가 8 배 크다.")

    header("2. float16 에서 그래디언트가 사라진다")
    print(f"{'원래 값':>10}{'float16 으로':>16}{'0 이 되었나':>14}")
    for v in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
        h = torch.tensor([v], dtype=torch.float32).to(torch.float16)
        print(f"{v:>10.0e}{h.item():>16.3e}{str(h.item() == 0):>14s}")
    print()
    print("float16 의 가장 작은 정규수는 6.1e-05 다. 그 아래는 비정규수로")
    print("버티다가 1e-08 에서 0 이 된다. 그래디언트가 그만큼 작아지는 일은 흔하다.")

    header("3. 그래서 로스를 키워 둔다 — GradScaler 의 속셈")
    g = 1e-8
    print(f"그래디언트가 {g:.0e} 일 때")
    print(f"{'배율':>8}{'float16 에서':>16}{'되돌린 값':>14}{'살았나':>9}")
    for s in (1, 2**8, 2**16):
        h = torch.tensor([g * s], dtype=torch.float32).to(torch.float16)
        back = h.to(torch.float32).item() / s
        print(f"{s:>8d}{h.item():>16.3e}{back:>14.3e}{str(back > 0):>9s}")
    print()
    print("로스에 배율을 곱해 역전파하면 그래디언트가 모두 그만큼 커져 살아남고,")
    print("갱신하기 전에 도로 나눈다. 이것이 GradScaler 가 하는 일의 전부다.")
    print("bfloat16 은 범위가 float32 와 같아 이 장치가 필요 없다.")

    header("4. autocast 는 모델을 half 로 바꾸는 것이 아니다")
    x = torch.randn(8, 16)
    w = torch.randn(16, 16)
    with torch.amp.autocast("cpu", dtype=torch.float16):
        rows = [("matmul", (x @ w).dtype),
                ("softmax", torch.softmax(x, -1).dtype),
                ("layer_norm", nn.functional.layer_norm(x, (16,)).dtype),
                ("cross_entropy",
                 nn.functional.cross_entropy(x[:, :10],
                                             torch.zeros(8, dtype=torch.long)).dtype),
                ("더하기", (x + x).dtype)]
    for name, dt in rows:
        print(f"  {name:15s} -> {dt}")
    print()
    half = nn.Sequential(nn.Linear(16, 16), nn.LayerNorm(16)).half()
    print(f"  모델을 .half() 로 바꾸면 layer_norm 출력 -> {half(x.half()).dtype}")
    print()
    print("autocast 는 연산마다 고른다 — 곱은 내리고, 합·정규화·소프트맥스는")
    print("float32 로 둔다. .half() 는 그런 구별 없이 모두 내린다.")

    header("5. 오늘의 API")
    print(f"고른 장치: {DEV}")
    print("torch.amp.autocast(device_type, dtype=...) 와 torch.amp.GradScaler(device_type)")
    print("가 지금 쓰는 꼴이다. torch.cuda.amp.* 는 예전 이름이다.")
    scaler = torch.amp.GradScaler(DEV)
    print(f"GradScaler 가 켜졌는가: {scaler.is_enabled()}")
    print("장치 이름을 넘기면 그 장치에 맞게 켜지고 꺼진다 — 직접 끌 일은 거의 없다.")


if __name__ == "__main__":
    main()
