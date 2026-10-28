"""튜토리얼 26: 자료 먹이기 — Dataset 과 DataLoader.

num_workers 를 쓰는 측정이 들어 있어 이 쪽의 코드는 반드시
`if __name__ == "__main__":` 아래에서 돌아야 한다. 까닭은 5절에 적었다.
"""
import time

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

torch.manual_seed(0)


def header(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


class Points(Dataset):
    """세 가지만 있으면 Dataset 이다 — __init__, __len__, __getitem__."""

    def __init__(self, n=100):
        self.x = torch.randn(n, 4)
        self.y = (self.x.sum(1) > 0).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class IoDataset(Dataset):
    """__getitem__ 이 기다리는 자료 — 디스크에서 읽어 오는 일을 흉내 낸다."""

    def __init__(self, n=1000, wait=0.002):
        self.n, self.wait = n, wait

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        time.sleep(self.wait)              # 기다리는 동안 CPU 를 놓아준다
        return torch.zeros(64), 0


def main():
    header("1. Dataset 은 세 함수다")
    ds = Points(100)
    print(f"len(ds)  {len(ds)}")
    x0, y0 = ds[0]
    print(f"ds[0]    x {x0.tolist()}  y {y0.item()}")

    header("2. DataLoader 는 묶고 섞는다")
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    print(f"표본 100개를 32씩 -> 배치 {len(dl)}개")
    for i, (xb, yb) in enumerate(dl):
        print(f"  배치 {i}: {tuple(xb.shape)}")
    print("마지막 배치가 4개다. drop_last=True 로 두면 버린다.")
    print(f"drop_last=True 면 배치 {len(DataLoader(ds, batch_size=32, drop_last=True))}개")

    header("3. 나누기")
    tr, va = random_split(ds, [80, 20], generator=torch.Generator().manual_seed(0))
    print(f"학습 {len(tr)}  검증 {len(va)}")
    print("검증 쪽은 섞지 않는다 — 섞어도 점수는 같고, 같은 차례로 두면 견주기 쉽다.")

    header("4. 배치 크기가 2의 거듭제곱이어야 하는가")
    X, y = torch.randn(20000, 64), torch.randint(0, 10, (20000,))
    big = TensorDataset(X, y)
    print(f"{'배치 크기':>10}{'한 에포크':>12}{'배치 수':>10}")
    for bs in (100, 128, 200, 256):
        d = DataLoader(big, batch_size=bs, shuffle=False)
        for _ in d:
            break
        t0 = time.perf_counter()
        for _ in range(3):
            for xb, yb in d:
                pass
        print(f"{bs:>10}{(time.perf_counter()-t0)/3*1e3:>10.1f} ms{len(d):>10}")
    print("이 자료에서는 2의 거듭제곱이라고 더 빠르지 않다.")

    header("5. num_workers — 늘리면 빨라지는가")
    print("(가) 이미 메모리에 올라온 자료")
    for nw in (0, 2, 4):
        d = DataLoader(big, batch_size=128, shuffle=True, num_workers=nw)
        for _ in d:
            break
        t0 = time.perf_counter()
        for _ in range(2):
            for xb, yb in d:
                pass
        print(f"  num_workers={nw}:  한 에포크 {(time.perf_counter()-t0)/2*1e3:8.1f} ms")
    print("  -> 느려진다. 자료를 가져오는 데 드는 일이 없으니 얻을 것이 없고,")
    print("     일꾼을 띄우는 값만 치른다.")

    header("6. 그러면 일꾼은 언제 버는가")
    print("__getitem__ 이 2 ms 를 기다리는 자료 1,000개, 세 에포크")

    def run(nw, persistent):
        d_kw = dict(batch_size=64, num_workers=nw)
        if nw:
            d_kw["persistent_workers"] = persistent
        d = DataLoader(IoDataset(), **d_kw)
        t0 = time.perf_counter()
        for _ in range(3):
            for xb, yb in d:
                pass
        return time.perf_counter() - t0

    a = run(0, False)
    b = run(4, False)
    c = run(4, True)
    print(f"  num_workers=0                      {a:6.2f} 초")
    print(f"  num_workers=4                      {b:6.2f} 초   {b/a:.1f}배 느리다")
    print(f"  num_workers=4, persistent_workers  {c:6.2f} 초   {a/c:.1f}배 빠르다")
    print(f"  -> 같은 일꾼 넷인데 한 줄 차이로 {b/c:.0f} 배가 갈린다.")


if __name__ == "__main__":
    main()
