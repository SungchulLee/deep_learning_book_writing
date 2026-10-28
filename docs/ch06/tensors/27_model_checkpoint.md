# 되짚음 저장 — 무엇이 저장되고 무엇이 저장되지 않는가

`torch.save(model.state_dict(), path)` 한 줄이면 저장은 끝난다. 그런데 **그 한 줄에 담기지 않는 것**이 있고, 그것을 모르면 되살린 모델이 조용히 다르게 움직인다.

이 쪽은 `state_dict`에 무엇이 들어가고 무엇이 빠지는지를 하나씩 재어 보고, 학습을 이어 가려면 무엇을 더 들고 다녀야 하는지를 수로 보인다.

## 1. 코드

```python
"""튜토리얼 27: 되짚음 저장 — 무엇이 저장되고 무엇이 저장되지 않는가."""
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

torch.manual_seed(0)
TMP = Path(tempfile.mkdtemp())


def header(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def make_model():
    return nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Dropout(0.5))


def main():
    header("1. state_dict 에 들어 있는 것")
    m = make_model()
    m(torch.randn(8, 4))                      # 배치 정규화 통계를 움직여 둔다
    for k, v in m.state_dict().items():
        print(f"  {k:22s} {tuple(v.shape)}")
    print("웨이트와 바이어스, 그리고 **버퍼**(running_mean 따위)가 들어간다.")

    header("2. 들어 있지 않은 것 — 모드")
    m.eval()
    print(f"저장하기 직전 m.training = {m.training}")
    torch.save(m.state_dict(), TMP / "sd.pt")
    print(f"state_dict 에 'training' 이라는 열쇠가 있는가: "
          f"{any('training' in k for k in m.state_dict())}")

    m2 = make_model()
    m2.load_state_dict(torch.load(TMP / "sd.pt", weights_only=True))
    print(f"불러온 뒤 m2.training = {m2.training}")
    print("eval 로 두고 저장해도 불러오면 train 이다. 모드는 저장되지 않는다.")
    print(f"버퍼는 따라왔는가: "
          f"{torch.equal(m.state_dict()['1.running_mean'], m2.state_dict()['1.running_mean'])}")

    header("3. 학습을 이어 가려면 네 가지를 함께 저장한다")
    model = nn.Linear(4, 1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(3):                        # Adam 의 모멘텀을 쌓아 둔다
        opt.zero_grad()
        nn.MSELoss()(model(torch.randn(8, 4)), torch.randn(8, 1)).backward()
        opt.step()

    ckpt = {"epoch": 3,
            "model": model.state_dict(),
            "optim": opt.state_dict(),
            "loss": 0.1234}
    torch.save(ckpt, TMP / "ck.pt")
    print(f"저장한 열쇠: {list(ckpt)}")
    exp_avg = opt.state_dict()["state"][0]["exp_avg"]
    print(f"옵티마이저가 들고 있는 상태: step={opt.state_dict()['state'][0]['step'].item():.0f}, "
          f"exp_avg 노름={exp_avg.norm():.4f}")
    print("옵티마이저를 빼고 저장하면 Adam 의 모멘텀이 0 부터 다시 쌓인다.")

    header("4. 옵티마이저를 빼먹으면 생기는 일")
    def first_step_after_resume(restore_optim):
        torch.manual_seed(1)
        m = nn.Linear(4, 1)
        o = torch.optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(3):
            o.zero_grad()
            nn.MSELoss()(m(torch.randn(8, 4)), torch.randn(8, 1)).backward()
            o.step()
        saved_m, saved_o = m.state_dict(), o.state_dict()

        m3 = nn.Linear(4, 1); m3.load_state_dict(saved_m)
        o3 = torch.optim.Adam(m3.parameters(), lr=1e-3)
        if restore_optim:
            o3.load_state_dict(saved_o)
        before = m3.weight.detach().clone()
        torch.manual_seed(99)
        o3.zero_grad()
        nn.MSELoss()(m3(torch.randn(8, 4)), torch.randn(8, 1)).backward()
        o3.step()
        return (m3.weight.detach() - before).norm().item()

    a = first_step_after_resume(True)
    b = first_step_after_resume(False)
    print(f"옵티마이저까지 되살린 첫 걸음의 크기 {a:.6f}")
    print(f"모델만 되살린 첫 걸음의 크기       {b:.6f}   {b/a:.1f}배")
    print("Adam 은 처음 몇 걸음을 보정하므로, 상태를 잃으면 첫 걸음이 커진다.")

    header("5. 통째로 저장하면 클래스에 묶인다")
    print("torch.save(model) 은 클래스를 가리키는 이름을 함께 절인다.")
    print("불러오는 쪽에 그 클래스가 같은 경로로 있어야 하고, 없으면")
    print("  AttributeError: Can't get attribute 'MyNet' on <module '__main__'>")
    print("이 난다. state_dict 는 숫자만 담으므로 그 묶임이 없다.")

    header("6. 저장한 뒤에는 되읽어 맞추어 본다")
    m4 = make_model()
    m4.load_state_dict(torch.load(TMP / "sd.pt", weights_only=True))
    same = all(torch.equal(a, b) for a, b in
               zip(m.state_dict().values(), m4.state_dict().values()))
    print(f"모든 텐서가 같은가: {same}")
    print("weights_only=True 로 읽는다 — 숫자만 받아들이므로 절인 코드가 돌지 않는다.")


if __name__ == "__main__":
    main()
```

**출력:**

```
======================================================================
1. state_dict 에 들어 있는 것
======================================================================
  0.weight               (4, 4)
  0.bias                 (4,)
  1.weight               (4,)
  1.bias                 (4,)
  1.running_mean         (4,)
  1.running_var          (4,)
  1.num_batches_tracked  ()
웨이트와 바이어스, 그리고 **버퍼**(running_mean 따위)가 들어간다.

======================================================================
2. 들어 있지 않은 것 — 모드
======================================================================
저장하기 직전 m.training = False
state_dict 에 'training' 이라는 열쇠가 있는가: False
불러온 뒤 m2.training = True
eval 로 두고 저장해도 불러오면 train 이다. 모드는 저장되지 않는다.
버퍼는 따라왔는가: True

======================================================================
3. 학습을 이어 가려면 네 가지를 함께 저장한다
======================================================================
저장한 열쇠: ['epoch', 'model', 'optim', 'loss']
옵티마이저가 들고 있는 상태: step=3, exp_avg 노름=0.2166
옵티마이저를 빼고 저장하면 Adam 의 모멘텀이 0 부터 다시 쌓인다.

======================================================================
4. 옵티마이저를 빼먹으면 생기는 일
======================================================================
옵티마이저까지 되살린 첫 걸음의 크기 0.001219
모델만 되살린 첫 걸음의 크기       0.002000   1.6배
Adam 은 처음 몇 걸음을 보정하므로, 상태를 잃으면 첫 걸음이 커진다.

======================================================================
5. 통째로 저장하면 클래스에 묶인다
======================================================================
torch.save(model) 은 클래스를 가리키는 이름을 함께 절인다.
불러오는 쪽에 그 클래스가 같은 경로로 있어야 하고, 없으면
  AttributeError: Can't get attribute 'MyNet' on <module '__main__'>
이 난다. state_dict 는 숫자만 담으므로 그 묶임이 없다.

======================================================================
6. 저장한 뒤에는 되읽어 맞추어 본다
======================================================================
모든 텐서가 같은가: True
weights_only=True 로 읽는다 — 숫자만 받아들이므로 절인 코드가 돌지 않는다.
```

## 2. 들어가는 것 — 숫자

1절이 보이듯 `state_dict`는 **이름과 텐서의 사전**이다. 학습하는 파라메터(`weight`, `bias`)만이 아니라 **버퍼**도 들어간다 — 배치 정규화의 `running_mean`·`running_var`·`num_batches_tracked`가 그것이다.

버퍼가 들어가는 것이 중요하다. 그 수들은 학습으로 정해지지만 그래디언트로 정해지지는 않으므로, 빠뜨리면 **되살린 모델이 추론에서 다른 값을 낸다.** [21쪽](21_simple_neural_network.md)에서 `register_buffer`가 `state_dict`에 들어간다고 적은 것이 여기서 쓰인다.

## 3. 빠지는 것 — 모드

2절이 그것이다. `model.eval()`을 부른 **뒤에** 저장해도 `state_dict`에 `training`이라는 열쇠가 없고, 되살린 모델은 `training = True`다.

```
저장하기 직전 m.training = False
불러온 뒤      m2.training = True
```

그러니 "추론용으로 저장한다"며 `eval()`을 부르고 저장하는 것은 **파일에 아무 자취도 남기지 않는다.** 모드는 저장되는 성질이 아니라 **쓰는 쪽에서 정하는 상태**다. 되살린 뒤에 `eval()`을 부르는 것이 유일한 방법이고, 드롭아웃이나 배치 정규화가 있는 모델이면 이것을 빠뜨린 채 평가한 점수는 틀린 점수다.

## 4. 학습을 이어 가려면 모델만으로는 모자란다

3절이 네 가지를 한 사전에 담는다 — 에포크, 모델, **옵티마이저**, 로스.

옵티마이저가 왜 필요한지는 4절이 잰다. Adam은 모멘텀 두 개를 쌓아 두고, 초기 몇 걸음을 보정한다. 그 상태를 버리고 모델만 되살리면 **첫 걸음이 1.6배 커진다.**

| | 되살린 뒤 첫 걸음의 크기 |
|---|---|
| 옵티마이저까지 되살림 | 0.001219 |
| 모델만 되살림 | **0.002000** |

터지지는 않는다. 로스도 내려간다. 다만 **이어 간 학습이 이어 간 것이 아니게** 되고, 중간에 저장하고 불러오기를 되풀이하면 그때마다 스텝이 한 번씩 튄다. 재현하려는 실험에서 이것을 빠뜨리면 같은 설정이 다른 곡선을 그린다.

## 5. 통째로 저장하면 코드에 묶인다

`torch.save(model)`은 텐서만이 아니라 **클래스를 가리키는 이름**을 함께 절여 둔다. 그래서 불러오는 쪽에 그 클래스가 **같은 이름, 같은 모듈 경로**로 있어야 한다. 없으면 이렇게 끝난다.

```
AttributeError: Can't get attribute 'MyNet' on <module '__main__'>
```

파일을 다른 스크립트에서 열거나, 클래스를 다른 파일로 옮기거나, 모듈 이름을 바꾸기만 해도 난다. `state_dict`는 **숫자만** 담으므로 그런 묶임이 없다 — 대신 되살릴 때 구조를 내가 만들어 주어야 하고, 그것이 치르는 값의 전부다.

## 6. `weights_only=True`로 읽는다

`torch.load`는 본디 파이썬의 절임(pickle)을 그대로 풀었다. 절임은 **푸는 동안 코드를 돌릴 수 있으므로**, 남이 준 파일을 그렇게 여는 것은 그 사람의 코드를 돌리는 것과 같다.

그래서 `weights_only=True`가 생겼다. 텐서와 몇 가지 기본 자료형만 받아들이고 나머지는 거절한다. torch 2.5에서는 이것을 켜지 않으면 `FutureWarning`이 나고, 앞으로는 이쪽이 기본이 된다. 이 책의 다른 쪽들이 모두 `weights_only=True`로 읽는 까닭이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
드롭아웃이 있는 모델을 저장하고, 되살린 뒤 `eval()`을 **부르지 않은 채** 같은 입력을 두 번 넣어 보라. 무엇이 관찰되는가? 그것이 평가 점수에 어떻게 나타나겠는가?

</div>

??? success "풀이"
    같은 입력에 **다른 답**이 나온다.

    ```python
    import torch, torch.nn as nn, tempfile
    from pathlib import Path
    torch.manual_seed(0)
    tmp = Path(tempfile.mkdtemp()) / "m.pt"

    m = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5))
    m.eval()
    torch.save(m.state_dict(), tmp)

    m2 = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5))
    m2.load_state_dict(torch.load(tmp, weights_only=True))

    x = torch.ones(1, 4)
    print("불러온 그대로:", torch.equal(m2(x), m2(x)))
    m2.eval()
    print("eval() 뒤     :", torch.equal(m2(x), m2(x)))
    ```

    ```
    불러온 그대로: False
    eval() 뒤     : True
    ```

    **평가 점수가 흔들린다.** 드롭아웃이 매번 다른 뉴런을 끄므로 같은 시험 집합을 두 번 돌리면 다른 정확도가 나오고, 평균적으로는 **낮게** 나온다 — 쓰지 않아야 할 정칙화를 평가에 걸어 둔 셈이기 때문이다.

    고약한 것은 오류가 나지 않는다는 점이다. 점수가 조금 낮고 돌릴 때마다 조금 다를 뿐이라, 모델이 원래 그 정도인 줄 알기 쉽다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
4절은 옵티마이저 상태를 잃으면 첫 걸음이 1.6배가 된다고 했다. SGD(모멘텀 없음)로 같은 실험을 하면 어떻게 되겠는가? 재어 보고 까닭을 말하라.

</div>

??? success "풀이"
    **차이가 없다.** 모멘텀 없는 SGD는 들고 다닐 상태가 없기 때문이다.

    ```python
    import torch, torch.nn as nn

    def first_step(make_opt, restore):
        torch.manual_seed(1)
        m = nn.Linear(4, 1); o = make_opt(m.parameters())
        for _ in range(3):
            o.zero_grad()
            nn.MSELoss()(m(torch.randn(8, 4)), torch.randn(8, 1)).backward(); o.step()
        sm, so = m.state_dict(), o.state_dict()
        m2 = nn.Linear(4, 1); m2.load_state_dict(sm)
        o2 = make_opt(m2.parameters())
        if restore: o2.load_state_dict(so)
        before = m2.weight.detach().clone()
        torch.manual_seed(99)
        o2.zero_grad()
        nn.MSELoss()(m2(torch.randn(8, 4)), torch.randn(8, 1)).backward(); o2.step()
        return (m2.weight.detach() - before).norm().item()

    for name, mk in [("SGD        ", lambda p: torch.optim.SGD(p, lr=1e-3)),
                     ("SGD+모멘텀 ", lambda p: torch.optim.SGD(p, lr=1e-3, momentum=0.9)),
                     ("Adam       ", lambda p: torch.optim.Adam(p, lr=1e-3))]:
        a, b = first_step(mk, True), first_step(mk, False)
        print(f"{name} 되살림 {a:.6f}   안 되살림 {b:.6f}   {b/a:.2f}배")
    ```

    ```
    SGD         되살림 0.002118   안 되살림 0.002118   1.00배
    SGD+모멘텀  되살림 0.005929   안 되살림 0.002107   0.36배
    Adam        되살림 0.001219   안 되살림 0.002000   1.64배
    ```

    세 줄이 서로 다른 말을 한다.

    - **SGD는 정확히 같다.** `state_dict`에 담을 상태가 없으니 잃을 것도 없다.
    - **모멘텀 SGD는 걸음이 작아진다.** 쌓아 둔 속도를 잃고 0에서 다시 시작하기 때문이다.
    - **Adam은 커진다.** 초기 보정(bias correction)이 `step`을 세어 초반에 걸음을 키우는데, `step`이 0으로 돌아가면 그 보정이 다시 걸린다.

    곧 "옵티마이저도 저장하라"는 말의 무게가 옵티마이저마다 다르다. 상태가 없는 쪽은 저장해도 그만이고, 있는 쪽은 빠뜨리면 **방향이 서로 다르게** 어긋난다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
`weights_only=True`는 "숫자만 받아들인다"고 했다. 그렇다면 3절처럼 `epoch`·`loss` 같은 파이썬 수를 섞어 담은 사전은 그렇게 읽을 수 있는가? 읽을 수 **없는** 것은 어떤 것인가? 하나 만들어 확인하라.

</div>

??? success "풀이"
    파이썬의 기본 자료형은 통과한다. 막히는 것은 **임의의 객체**다.

    ```python
    import torch, tempfile
    from pathlib import Path
    tmp = Path(tempfile.mkdtemp())

    ok = {"epoch": 3, "loss": 0.1234, "name": "run-a",
          "w": torch.zeros(2), "hist": [1.0, 0.5]}
    torch.save(ok, tmp / "ok.pt")
    print("기본 자료형만:", list(torch.load(tmp / "ok.pt", weights_only=True)))

    import argparse
    bad = {"w": torch.zeros(2), "args": argparse.Namespace(lr=0.1)}
    torch.save(bad, tmp / "bad.pt")
    try:
        torch.load(tmp / "bad.pt", weights_only=True)
    except Exception as e:
        print("객체를 섞으면:", type(e).__name__)
        print("  ", str(e).splitlines()[0][:78])
    ```

    ```
    기본 자료형만: ['epoch', 'loss', 'name', 'w', 'hist']
    객체를 섞으면: UnpicklingError
       Weights only load failed. This file can still be loaded, to do so you have two
    ```

    **`int`·`float`·`str`·`list`·`dict`·텐서는 통과하고, `Namespace` 같은 객체는 거절당한다.**

    그래서 되짚음 저장을 짤 때 요령이 하나 생긴다 — **설정을 객체째 담지 말고 풀어서 담는다.** `vars(args)`로 사전으로 바꾸어 넣으면 `weights_only=True`로 읽히고, 받는 쪽에서 다시 `Namespace(**d)`로 만들면 된다.

    거절당했을 때 `weights_only=False`로 되돌리는 것은 **내가 만든 파일일 때만** 할 일이다. 남이 준 파일이라면 그 선택은 "이 사람의 코드를 내 기계에서 돌려도 좋다"는 뜻이 된다.

## 정리하며

`state_dict`에 들어가는 것은 **숫자**다 — 파라메터와 버퍼. 들어가지 않는 것은 **구조**(클래스), **모드**(train/eval), **옵티마이저 상태**이고, 셋 다 빠뜨려도 오류가 나지 않는다.

| 빠뜨리면 | 생기는 일 | 알아채는가 |
|---|---|---|
| 구조 | 되살릴 수 없다 | 바로 터진다 |
| 모드 | 드롭아웃이 켜진 채 평가된다 | 점수가 조금 낮고 흔들린다 |
| 옵티마이저 | 첫 걸음이 어긋난다 | 거의 못 알아챈다 |

아래로 갈수록 조용하다. 그래서 저장한 다음에는 **되읽어 맞추어 보는 것**이 습관이어야 하고, 맞추어 볼 것은 텐서만이 아니라 `training` 플래그와 옵티마이저의 `step`까지다.
