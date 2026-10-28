"""Adam — 손으로 짜서 torch.optim.Adam 과 맞춰 본다.

Adam 을 설명하는 글은 많지만 대개 식을 적고 끝난다. 식을 적을 수 있다고
아는 것이 아니므로, 이 쪽은 손으로 짜서 PyTorch 와 **한 비트까지**
맞춰 본다. 맞으면 이해한 것이고, 안 맞으면 어디가 다른지가 드러난다.

이어서 네 가지를 잰다.

- 바이어스 보정이 하는 일 (첫 걸음을 키우는가 줄이는가)
- Adam 의 서명은 무엇인가 (곡률이 다른 좌표를 같은 속도로 줄인다)
- 같은 러닝레이트에서 견주지 않은 비교는 무엇을 재는가
- Adam 과 AdamW 의 웨이트 감쇠가 어떻게 다른가

개념과 유도는 ch09 의 Adam 쪽에서 다룬다.

실행: python 01_adam_optimizer.py
"""
import math

import numpy as np
import torch
import torch.nn as nn


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. 손으로 짠 Adam
# ========================================================================

def hand_adam_step(p, g, state, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    """Adam 한 걸음. PyTorch 의 구현과 같은 꼴로 적는다."""
    state["t"] += 1
    t = state["t"]
    state["m"] = b1 * state["m"] + (1 - b1) * g
    state["v"] = b2 * state["v"] + (1 - b2) * g * g
    m_hat = state["m"] / (1 - b1 ** t)          # 바이어스 보정
    v_hat = state["v"] / (1 - b2 ** t)
    return p - lr * m_hat / (v_hat.sqrt() + eps)


def match_pytorch():
    header("1. 손으로 짠 Adam vs torch.optim.Adam")
    torch.manual_seed(0)
    X = torch.randn(64, 5)
    Y = torch.randn(64, 1)

    torch.manual_seed(1)
    m1 = nn.Linear(5, 1)
    torch.manual_seed(1)
    m2 = nn.Linear(5, 1)

    opt = torch.optim.Adam(m2.parameters(), lr=1e-3)
    states = [{"m": torch.zeros_like(p), "v": torch.zeros_like(p), "t": 0}
              for p in m1.parameters()]

    print("  걸음   손으로 짠 w[0]      PyTorch w[0]       차이")
    for step in range(1, 11):
        # 손으로
        for p in m1.parameters():
            if p.grad is not None:
                p.grad = None
        nn.functional.mse_loss(m1(X), Y).backward()
        with torch.no_grad():
            for p, st in zip(m1.parameters(), states):
                p.copy_(hand_adam_step(p, p.grad, st))
        # PyTorch
        opt.zero_grad()
        nn.functional.mse_loss(m2(X), Y).backward()
        opt.step()
        if step in (1, 2, 5, 10):
            a = m1.weight.detach().flatten()[0].item()
            b = m2.weight.detach().flatten()[0].item()
            print(f"  {step:>4}   {a:+.12f}   {b:+.12f}   {abs(a-b):.2e}")
    gap = max((p1 - p2).abs().max().item()
              for p1, p2 in zip(m1.parameters(), m2.parameters()))
    print(f"  10 걸음 뒤 모든 파라메터의 최대 차이 {gap:.2e}")
    print("  float32 의 반올림 자리다. 같은 알고리즘이라는 뜻이다")


# ========================================================================
# 2. 바이어스 보정이 하는 일
# ========================================================================

def bias_correction():
    header("2. 바이어스 보정 — 첫 걸음을 키우는가 줄이는가")
    b1, b2, eps, lr = 0.9, 0.999, 1e-8, 1e-3
    g = torch.tensor(2.0)                      # 아무 경사나

    print("  m 과 v 를 0 에서 시작하므로 처음 몇 걸음은 0 쪽으로 치우친다")
    print("  걸음   보정 있음 걸음폭   보정 없음 걸음폭   비")
    m = v = torch.zeros(())
    m2 = v2 = torch.zeros(())
    for t in range(1, 11):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        with_c = lr * (m / (1 - b1 ** t)) / ((v / (1 - b2 ** t)).sqrt() + eps)
        m2 = b1 * m2 + (1 - b1) * g
        v2 = b2 * v2 + (1 - b2) * g * g
        without = lr * m2 / (v2.sqrt() + eps)
        if t in (1, 2, 3, 5, 10):
            print(f"  {t:>4}   {with_c.item():.8f}       "
                  f"{without.item():.8f}       {without.item()/with_c.item():.4f}")
    print("  보정을 **빼면** 첫 걸음이 3.16 배 커진다. 줄어드는 것이 아니다")
    print(f"  까닭: 보정 없는 첫 걸음은 (1-b1)g / sqrt((1-b2)g^2)")
    print(f"       = {1-b1} / sqrt({1-b2}) = {(1-b1)/math.sqrt(1-b2):.4f} 배")
    print("  보정이 있으면 첫 걸음이 정확히 lr 이다 (m_hat/sqrt(v_hat) = 1)")
    print("  곧 바이어스 보정은 **초기의 과도한 걸음을 막는** 장치다")
    print("  흔히 '0 쪽으로 치우친 것을 키워 준다' 고 적지만, 분자와 분모가")
    print("  함께 치우쳐 있어 몫에서는 반대로 나온다")


# ========================================================================
# 3. Adam 이 이기는 자리 — 조건수가 나쁜 문제
# ========================================================================

def where_adam_wins():
    header("3. Adam 의 서명 — 좌표마다 같은 속도")
    print("  좌표마다 곡률이 크게 다른 이차함수를 만든다")
    print("  f(x) = sum_i lam_i x_i^2 / 2,  lam = [1, 10, 100, 1000]")
    lam = torch.tensor([1.0, 10.0, 100.0, 1000.0])
    print(f"  조건수 {lam.max()/lam.min():.0f}")

    def run(make, lr, steps=2000):
        x = torch.ones(4, requires_grad=True)
        o = make([x], lr)
        for _ in range(steps):
            o.zero_grad()
            (0.5 * (lam * x ** 2).sum()).backward()
            o.step()
        return x.detach().abs()

    print("\n  2000 걸음 뒤 좌표별 |x| (0 에 가까울수록 좋다)")
    print("  옵티마이저        lr        x1        x2        x3        x4")
    for name, make, lr in (
            ("SGD             ", lambda p, l: torch.optim.SGD(p, lr=l), 0.001),
            ("SGD(mom=0.9)    ", lambda p, l: torch.optim.SGD(
                p, lr=l, momentum=0.9), 0.001),
            ("Adam            ", lambda p, l: torch.optim.Adam(p, lr=l), 0.001),
            ("Adam            ", lambda p, l: torch.optim.Adam(p, lr=l), 0.01)):
        r = run(make, lr)
        print(f"  {name}  {lr:<8} " + "  ".join(f"{v:.2e}" for v in r))
    print("  SGD 는 lam=1 인 좌표(x1)를 1.35e-01 에 남겨 둔 채 끝난다.")
    print("  lr 을 키우면 lam=1000 인 좌표가 터지므로 키울 수도 없다")
    print("  모멘텀을 켜면 x1 도 2.16e-10 까지 내려간다 — 이 문제에서는")
    print("  모멘텀만으로 충분하다. Adam 이 꼭 이기는 것은 아니다")
    print("  Adam 의 특징은 이김이 아니라 **고름** 이다. lr=0.001 줄의 네")
    print("  값이 2.07e-02 로 전부 같다. 곡률이 1000 배 차이 나는 좌표들을")
    print("  같은 속도로 줄인다 — sqrt(v) 로 나누어 걸음을 맞추기 때문이다")
    print("  그래서 어느 좌표가 느린지 몰라도 되고, 그 점이 실무의 이점이다")


# ========================================================================
# 4. 공정하게 견주기 — 각자의 러닝레이트를 쓸어 본다
# ========================================================================

def fair_compare():
    header("4. 공정하게: 같은 출발점 + 각자 최적 lr + 씨앗 5개")
    torch.manual_seed(0)
    X = torch.linspace(-3, 3, 200).unsqueeze(1)
    Y = torch.sin(X * 2) + 0.1 * torch.randn(200, 1)

    def run(make, lr, seed, epochs=300):
        torch.manual_seed(seed)
        m = nn.Sequential(nn.Linear(1, 32), nn.Tanh(),
                          nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
        o = make(m.parameters(), lr)
        for _ in range(epochs):
            o.zero_grad()
            nn.functional.mse_loss(m(X), Y).backward()
            o.step()
        with torch.no_grad():
            return nn.functional.mse_loss(m(X), Y).item()

    opts = (("SGD         ", lambda p, l: torch.optim.SGD(p, lr=l)),
            ("SGD(mom=0.9)", lambda p, l: torch.optim.SGD(
                p, lr=l, momentum=0.9)),
            ("Adam        ", lambda p, l: torch.optim.Adam(p, lr=l)))
    grid = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3)

    print("  먼저 각자의 lr 을 씨앗 하나로 쓸어 가장 좋은 값을 고른다")
    print("  옵티마이저     " + "".join(f"{l:>10}" for l in grid))
    best = {}
    for name, make in opts:
        row = []
        for lr in grid:
            v = run(make, lr, 0)
            row.append(v if math.isfinite(v) else float("inf"))
        best[name] = grid[int(np.argmin(row))]
        print(f"  {name}  " + "".join(
            f"{(v if v < 1e4 else float('inf')):>10.4f}" for v in row))
    print("  고른 lr: " + ",  ".join(f"{k.strip()} {v}" for k, v in best.items()))

    print("\n  그 lr 로 씨앗 5개를 돌린다")
    print("  옵티마이저     lr       로스 평균     퍼짐")
    for name, make in opts:
        vals = [run(make, best[name], s) for s in range(5)]
        print(f"  {name}  {best[name]:<7}  {np.mean(vals):.6f}   "
              f"{min(vals):.6f}~{max(vals):.6f}")
    print("  셋 다 최적 lr 이 0.1 로 같게 나왔다. 그 lr 에서도 Adam 이")
    print("  0.008795 로 SGD 의 0.050810 보다 5.8 배 낮다 — 이 문제에서는")
    print("  러닝레이트를 맞춰 주어도 Adam 이 이긴다")
    print("  다만 쓸어 보기 전의 표를 보라. lr=0.001 에서는 0.4484 대")
    print("  0.0528 로 8.5 배였다. 어느 쪽이든 Adam 이 낫지만 '얼마나'")
    print("  는 lr 을 맞췄는지에 따라 크게 달라진다")
    print("  그리고 Adam 의 lr 열은 0.003 부터 0.3 까지 평평하다 — 아무")
    print("  값이나 집어도 쓸 만하다는 뜻이고, 이것이 실무의 큰 이점이다")


# ========================================================================
# 5. Adam 과 AdamW 의 웨이트 감쇠
# ========================================================================

def decay():
    header("5. Adam(weight_decay) vs AdamW")
    print("  Adam 의 weight_decay 는 경사에 wd*p 를 더한다 — 그러면 그 항도")
    print("  sqrt(v) 로 나뉜다. AdamW 는 갱신 뒤에 따로 p 를 줄인다")

    for wd in (0.0, 0.1):
        outs = []
        for make in (torch.optim.Adam, torch.optim.AdamW):
            torch.manual_seed(0)
            p = nn.Parameter(torch.tensor([1.0, 1.0]))
            o = make([p], lr=0.1, weight_decay=wd)
            for _ in range(50):
                o.zero_grad()
                # 좌표마다 경사 크기가 다른 로스
                (0.5 * (torch.tensor([0.01, 100.0]) * p ** 2).sum()).backward()
                o.step()
            outs.append(p.detach().clone())
        print(f"  wd = {wd}:  Adam {outs[0].tolist()}")
        print(f"             AdamW {outs[1].tolist()}")
        print(f"             차이 {(outs[0]-outs[1]).abs().max():.6f}")
    print("  wd = 0 이면 둘이 비트까지 똑같다. wd 를 켜야 갈린다")
    print("  그런데 갈리는 모양이 뜻밖이다. Adam 쪽은 wd 를 0 에서 0.1 로")
    print("  올려도 값이 -0.0048174 에서 -0.0048181 로 **거의 안 변한다**")
    print("  AdamW 쪽은 -0.0174 로 3.6 배 커진다")
    print("  까닭: Adam 은 wd*p 를 경사에 더하는데, 그 합이 다시 sqrt(v) 로")
    print("  나뉜다. 로스 경사가 큰 자리에서는 v 가 커서 감쇠 항이 묻힌다")
    print("  곧 Adam 의 weight_decay 는 **경사 크기에 따라 세기가 달라진다**")
    print("  AdamW 는 갱신 뒤에 따로 빼므로 언제나 적은 만큼 줄인다")
    print("  이것이 AdamW 가 따로 있는 까닭이다. 같은 wd 값을 주어도 두")
    print("  옵티마이저가 전혀 다른 양만큼 줄인다")


# ========================================================================
# 6. 이어서 학습하려면 옵티마이저 상태도 저장해야 한다
# ========================================================================

def resume():
    header("6. 학습을 이어 하려면")
    torch.manual_seed(0)
    X, Y = torch.randn(32, 4), torch.randn(32, 1)

    def train(m, o, steps):
        for _ in range(steps):
            o.zero_grad()
            nn.functional.mse_loss(m(X), Y).backward()
            o.step()

    torch.manual_seed(1)
    ref = nn.Linear(4, 1)
    o_ref = torch.optim.Adam(ref.parameters(), lr=0.01)
    train(ref, o_ref, 40)

    for save_opt in (True, False):
        torch.manual_seed(1)
        m = nn.Linear(4, 1)
        o = torch.optim.Adam(m.parameters(), lr=0.01)
        train(m, o, 20)
        ckpt = {"model": m.state_dict(),
                "opt": o.state_dict() if save_opt else None}

        torch.manual_seed(1)
        m2 = nn.Linear(4, 1)
        m2.load_state_dict(ckpt["model"])
        o2 = torch.optim.Adam(m2.parameters(), lr=0.01)
        if ckpt["opt"] is not None:
            o2.load_state_dict(ckpt["opt"])
        train(m2, o2, 20)

        d = max((a - b).abs().max().item()
                for a, b in zip(m2.parameters(), ref.parameters()))
        lab = "옵티마이저 상태까지 저장" if save_opt else "모델만 저장      "
        print(f"  {lab}  40 걸음 통째와의 차이 {d:.6e}")
    print("  모델만 저장하면 m, v, t 가 0 으로 돌아가 다시 1 걸음째가 된다")
    print("  2 절에서 본 대로 1 걸음째의 걸음폭은 특별하므로, 이어 하는")
    print("  자리에서 갑자기 큰 걸음을 떼게 된다")
    print("  SGD(모멘텀 없음)는 상태가 없어 이 문제가 없다. 그래서 잊기 쉽다")


def main():
    match_pytorch()
    bias_correction()
    where_adam_wins()
    fair_compare()
    decay()
    resume()


if __name__ == "__main__":
    main()
