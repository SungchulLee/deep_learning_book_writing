"""MNIST 전체 파이프라인 — 이 장에서 배운 것을 한자리에.

앞 쪽들은 조각을 하나씩 다루었다. 이 쪽은 처음부터 끝까지 한 번
돌린다. 자료를 나누고, 모델을 세우고, 학습하고, 재고, 보고한다.

조각을 아는 것과 파이프라인을 세우는 것은 다른 일이고, 그 차이가
드러나는 자리를 재어 본다.

- 검증셋과 테스트셋은 왜 따로 있는가 (수로 보면)
- model.train() 과 model.eval() 이 실제로 바꾸는 것
- '배치 평균의 평균' 과 참 평균이 얼마나 다른가
- 한 번 돌린 결과는 측정이 아니다 (씨앗을 바꾸면)
- 스케줄러가 실제로 한 일

저장소 뿌리에서 돌린다 (./data).

실행: python 01_neural_network_mnist.py
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH = 128


def header(title):
    print(f"\n=== {title} ===")


def load():
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    train_full = datasets.MNIST("./data", train=True, download=False,
                                transform=tf)
    test = datasets.MNIST("./data", train=False, download=False, transform=tf)
    g = torch.Generator().manual_seed(0)
    train, val = random_split(train_full, [48000, 12000], generator=g)
    return train, val, test


TRAIN, VAL, TEST = load()


class Net(nn.Module):
    def __init__(self, p_drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(64, 10),                 # 로짓. Softmax 를 붙이지 않는다
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader, train_mode=False):
    """로스를 **표본 수로 가중해** 더한다. 배치 평균의 평균이 아니다."""
    model.train() if train_mode else model.eval()
    tot_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEV), y.to(DEV)
            out = model(x)
            tot_loss += nn.functional.cross_entropy(
                out, y, reduction="sum").item()
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return tot_loss / n, correct / n * 100


# ========================================================================
# 1. 자료를 셋으로 나눈다
# ========================================================================

def splits():
    header("1. 학습 / 검증 / 테스트")
    print(f"  MNIST 학습셋 60000 장을 48000 / 12000 으로 나눴다")
    print(f"  학습 {len(TRAIN)},  검증 {len(VAL)},  테스트 {len(TEST)}")
    print(f"  장치 {DEV}")

    # 라벨 분포가 고르게 갈렸는지 확인한다
    def counts(ds):
        ys = torch.tensor([ds[i][1] for i in range(0, len(ds), 20)])
        return torch.bincount(ys, minlength=10).float() / len(ys) * 100

    for name, ds in (("학습", TRAIN), ("검증", VAL), ("테스트", TEST)):
        c = counts(ds)
        print(f"  {name} 라벨 분포(%) " +
              " ".join(f"{v:4.1f}" for v in c) +
              f"   최소 {c.min():.1f} 최대 {c.max():.1f}")
    print("  무작위로 나누었는데도 분포가 고르다. 표본이 많기 때문이다")
    print("  자료가 적거나 라벨이 치우쳐 있으면 층화 추출을 해야 한다")


# ========================================================================
# 2. 학습
# ========================================================================

def train_once(seed, epochs=8, lr=0.1, use_sched=True, p_drop=0.2,
               verbose=False):
    torch.manual_seed(seed)
    model = Net(p_drop).to(DEV)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=1)
    g = torch.Generator().manual_seed(seed)
    tl = DataLoader(TRAIN, batch_size=BATCH, shuffle=True, generator=g)
    vl = DataLoader(VAL, batch_size=512)

    hist = []
    for ep in range(epochs):
        model.train()
        batch_means = []
        for x, y in tl:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            batch_means.append(loss.item())
        vloss, vacc = evaluate(model, vl)
        if use_sched:
            sched.step(vloss)
        hist.append({"ep": ep, "batch_mean": float(np.mean(batch_means)),
                     "vloss": vloss, "vacc": vacc,
                     "lr": opt.param_groups[0]["lr"]})
        if verbose:
            print(f"  {ep:>4}   {hist[-1]['batch_mean']:.4f}        "
                  f"{vloss:.4f}      {vacc:6.2f}%     {hist[-1]['lr']:.4f}")
    return model, hist


def training():
    header("2. 학습 (8 에폭, SGD+모멘텀 0.9, lr 0.1, ReduceLROnPlateau)")
    print("  에폭   학습 배치평균   검증 로스   검증 정확도   그때의 lr")
    model, hist = train_once(0, verbose=True)
    return model, hist


# ========================================================================
# 3. train() 과 eval() 이 바꾸는 것
# ========================================================================

def modes(model):
    header("3. model.train() vs model.eval()")
    vl = DataLoader(VAL, batch_size=512)
    el, ea = evaluate(model, vl, train_mode=False)
    tl_, ta = evaluate(model, vl, train_mode=True)
    print(f"  eval()  로스 {el:.4f}   정확도 {ea:.2f}%")
    print(f"  train() 로스 {tl_:.4f}   정확도 {ta:.2f}%")
    print(f"  차이          {tl_-el:+.4f}            {ta-ea:+.2f}%p")
    print("  같은 모델, 같은 자료인데 다르다. Dropout 이 켜져 있기 때문이다")

    print("\n  train() 모드는 돌릴 때마다 답이 달라진다")
    for i in range(3):
        torch.manual_seed(100 + i)
        l, a = evaluate(model, vl, train_mode=True)
        print(f"    {i+1}번째   로스 {l:.4f}   정확도 {a:.2f}%")
    print("  eval() 을 빼먹으면 재현되지 않는 평가를 하게 된다")

    print("\n  Dropout 을 끈 모델이라면 두 모드가 같은가")
    m2, _ = train_once(0, epochs=1, p_drop=0.0)
    a1 = evaluate(m2, vl, train_mode=False)
    a2 = evaluate(m2, vl, train_mode=True)
    print(f"    p_drop=0.0:  eval {a1[1]:.2f}%   train {a2[1]:.2f}%   "
          f"차이 {abs(a1[1]-a2[1]):.4f}%p")
    print("  같다. 이 모델에는 BatchNorm 이 없어 Dropout 만이 모드에 반응한다")
    print("  no_grad() 는 모드와 다른 일을 한다 — 기록을 끌 뿐 행동을 안 바꾼다")


# ========================================================================
# 4. '배치 평균의 평균' 은 참 평균이 아니다
# ========================================================================

def mean_of_means(hist):
    header("4. 배치 평균의 평균 vs 표본 가중 평균")
    print("  학습 로스를 적는 두 방법이 있다")
    print("   (가) 배치마다의 평균 로스를 모아 다시 평균 낸다 (흔한 방법)")
    print("   (나) 로스의 합을 모아 전체 표본 수로 나눈다 (참 평균)")

    n_full = len(TRAIN)
    n_batches = -(-n_full // BATCH)
    print(f"\n  학습셋 {n_full}장, 배치 {BATCH} -> 배치 {n_batches}개")
    print(f"  마지막 배치 크기 {n_full - (n_batches-1)*BATCH}")
    print("  48000 = 128 * 375 이라 딱 떨어진다. 그러면 (가) 와 (나) 가 같다")

    print("\n  일부러 나누어떨어지지 않게 해 본다 (배치 127)")
    g = torch.Generator().manual_seed(0)
    model, _ = train_once(0, epochs=1)
    for bs in (128, 127, 1000):
        dl = DataLoader(TRAIN, batch_size=bs)
        model.eval()
        means, tot, n = [], 0.0, 0
        with torch.no_grad():
            for x, y in dl:
                x, y = x.to(DEV), y.to(DEV)
                out = model(x)
                means.append(nn.functional.cross_entropy(out, y).item())
                tot += nn.functional.cross_entropy(
                    out, y, reduction="sum").item()
                n += len(y)
        last = n_full - (len(means) - 1) * bs
        print(f"  배치 {bs:<5} 배치수 {len(means):>4}  마지막 {last:>4}장  "
              f"(가) {np.mean(means):.8f}  (나) {tot/n:.8f}  "
              f"차이 {abs(np.mean(means)-tot/n):.2e}")
    print("  나누어떨어지면 차이가 0 이거나(128) 부동소수점 자리다(1000).")
    print("  127 에서만 9.8e-06 으로 뜬다 — 마지막 121장짜리 배치가 128장짜리와")
    print("  같은 무게로 평균에 들어가기 때문이다")
    print("  크기는 작다. 그래도 적을 때는 어느 쪽인지 밝히는 편이 낫다")
    print("  정확도는 이 문제가 없다 — 맞힌 수를 전체 수로 나누면 그만이다")


# ========================================================================
# 5. 한 번 돌린 결과는 측정이 아니다
# ========================================================================

def spread():
    header("5. 씨앗을 바꾸면 (5번)")
    vl = DataLoader(VAL, batch_size=512)
    tl = DataLoader(TEST, batch_size=512)
    vaccs, taccs = [], []
    print("  씨앗   검증 정확도   테스트 정확도   둘의 차")
    for s in range(5):
        m, h = train_once(s)
        _, va = evaluate(m, vl)
        _, ta = evaluate(m, tl)
        vaccs.append(va)
        taccs.append(ta)
        print(f"  {s:>4}   {va:9.2f}%   {ta:11.2f}%   {ta-va:+6.2f}%p")
    print(f"  검증   평균 {np.mean(vaccs):.2f}%  "
          f"퍼짐 {min(vaccs):.2f}~{max(vaccs):.2f}")
    print(f"  테스트 평균 {np.mean(taccs):.2f}%  "
          f"퍼짐 {min(taccs):.2f}~{max(taccs):.2f}")
    print("  퍼짐이 이만큼 있다. 두 설정의 차이가 이보다 작으면 차이가 아니다")

    best = int(np.argmax(vaccs))
    print(f"\n  검증으로 고르면 씨앗 {best} 를 고른다 "
          f"(검증 {vaccs[best]:.2f}%)")
    print(f"  그 씨앗의 테스트 정확도는 {taccs[best]:.2f}% 이고,")
    print(f"  다섯의 테스트 평균은 {np.mean(taccs):.2f}% 다")
    print(f"  차이 {taccs[best]-np.mean(taccs):+.2f}%p — 검증으로 고르는 행위")
    print("  자체가 검증셋에 맞추는 일이다. 그래서 테스트셋이 따로 있다")
    print("  검증셋으로 고르고 테스트셋으로 한 번만 보고한다")


# ========================================================================
# 6. 스케줄러가 한 일
# ========================================================================

def scheduler_effect():
    header("6. ReduceLROnPlateau 가 한 일")
    vl = DataLoader(VAL, batch_size=512)
    for use in (True, False):
        accs = []
        for s in range(3):
            m, h = train_once(s, use_sched=use)
            accs.append(evaluate(m, vl)[1])
        lrs = [r["lr"] for r in h]
        lab = "스케줄러 켬" if use else "스케줄러 끔"
        print(f"  {lab}  검증 정확도 평균 {np.mean(accs):.2f}%  "
              f"퍼짐 {min(accs):.2f}~{max(accs):.2f}")
        print(f"              마지막 씨앗의 에폭별 lr: " +
              " ".join(f"{v:.4f}" for v in lrs))
    print("  두 줄이 평균도 퍼짐도 완전히 같다. 스케줄러가 **한 번도 켜지지**")
    print("  않았기 때문이다 — lr 이 여덟 에폭 내내 0.1 이다")
    print("  검증 로스가 8 에폭 동안 계속 내려가서 plateau 가 없었다")
    print("  스케줄러를 넣었다고 일하는 것이 아니다. lr 기록을 함께 찍어")
    print("  실제로 켜졌는지 확인해야 한다")


def main():
    splits()
    model, hist = training()
    modes(model)
    mean_of_means(hist)
    spread()
    scheduler_effect()


if __name__ == "__main__":
    main()
