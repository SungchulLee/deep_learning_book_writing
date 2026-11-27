# 경사 추적 끄기

<div class="leadbox" markdown>

**들어가는 이야기** — 「동결한다」는 말이 하는 일

남이 오래 트레인시켜 둔 모델을 가져다 내 일에 맞게 조금만 고치는 것을
**전이 학습**이라 한다. 그때 「앞쪽은 건드리지 말고 뒤쪽만 배우게 하자」는 말을
하는데, 그 「건드리지 말자」를 **동결한다**(freeze)고 부른다.

코드로는 그 레이어의 파라메터에 `requires_grad_(False)`를 거는 일이다. 동결한다는
말에서 떠오르는 그림은 아마 **벽**일 것이다. 거기서 막히고, 그 너머로는
아무것도 못 가는 벽. 그렇다면 동결한 레이어는 계산할 것도 줄어서 트레인이 빨라져야
한다.

둘 다 틀렸다. 동결한 레이어는 **길을 막지 않고 그래디언트를 통과시킨다.** 그래서 그 너머의 레이어도
멀쩡히 배우고, 역전파도 생각만큼 빨라지지 않는다. 막는 것과 안 쌓는 것은
다른 일이다.

이 차이는 평소에는 안 드러난다. 앞쪽을 동결하면 그 앞에 더 갈 곳이 없기
때문이다. **가운데 레이어를 동결해 보면** 바로 드러나고, 그때 비슷해 보이던 셋 —
`requires_grad_(False)` · `torch.no_grad()` · `detach()` — 이 서로 다른 답을
낸다.

</div>

`requires_grad`는 **그 텐서의 속성**이다. 「경사 추적을 언제 멈출 것인가」에서 본 네
가지 가운데 유일하게 **영구적**인 것이고, 블록을 나온다고 돌아오지 않는다.

전이 학습에서 레이어를 동결하는 일이 이 플래그 하나로 끝난다. 그런데 "동결한다"는 말이
생각보다 좁은 뜻이다 — **동결한 레이어도 그래디언트를 통과시킨다.** 쌓지 않을 뿐이다.

## 1. 코드

```python
"""requires_grad 플래그 — 동결과, 동결해도 흐르는 것.

`requires_grad` 는 **그 텐서의 속성**이다. 켜고 끄면 영구적이고,
블록을 나온다고 돌아오지 않는다. 전이 학습에서 레이어를 동결하는 일이
이 플래그 하나로 끝난다.

조심할 것이 두 가지다.

- **리프 텐서에만 바꿀 수 있다.** 계산으로 생긴 텐서에는 못 건다
- **동결한 레이어도 그래디언트를 통과시킨다.** 쌓지 않을 뿐이다. 그래서 중간 레이어를
  동결해도 그 앞 레이어는 멀쩡히 배운다

실행: 이 쪽의 코드를 파일로 갈무리해 돌린다
"""
# torch 는 파이토치 본체다. 텐서를 만들고 미분하는 일이 다 여기서 온다.
import torch
# torch.nn 은 레이어(nn.Linear)와 그것을 잇는 틀(nn.Sequential)을 모아 둔 곳이다.
# 이름이 길어 nn 으로 줄여 부른다.
import torch.nn as nn
# torch.nn.functional 은 로스 같은 것을 「함수」로 모아 둔 곳이다. 여기서는
# 평균제곱오차 F.mse_loss 하나만 쓴다.
import torch.nn.functional as F


def header(title):
    # 절 제목을 한 줄 찍는다. \n 은 줄바꿈이고, f"...{title}..." 은 문자열 안에
    # 변수의 값을 끼워 넣는 파이썬 문법(f-문자열)이다.
    print(f"\n=== {title} ===")


# ========================================================================
# 1. 어디에 걸 수 있는가
# ========================================================================

def where_allowed():
    header("1. 플래그를 걸 수 있는 텐서")
    # 난수의 시드를 고정한다. 아래에서 뽑는 텐서가 돌릴 때마다 같은 값이 된다.
    torch.manual_seed(0)
    # torch.randn(3) 은 표준정규분포에서 뽑은 실수 3개짜리 텐서를 만든다. 형태 (3,).
    # 텐서는 파이토치가 다루는 수의 묶음이다. 이 절은 값이 아니라 플래그만 보므로
    # 난수를 고정하지 않아도 출력이 같다.
    a = torch.randn(3)
    # requires_grad 는 「이 텐서로 미분할 것이니, 이 텐서가 들어간 계산을 기록하라」
    # 는 표시다. 이것을 경사 추적이라 하고, 기본값은 False(끔)다.
    #
    # a.requires_grad_(True) 는 a 의 그 표시를 「제자리에서」 켠다. 파이토치에서
    # 이름 끝의 밑줄 _ 은 새 텐서를 만들지 않고 그 텐서 자체를 고친다는 뜻이다.
    # 고친 a 를 그대로 돌려주므로 바로 뒤에 .requires_grad 를 붙여 읽을 수 있다.
    # a 는 사용자가 직접 만든 텐서, 곧 리프 텐서라서 켤 수 있다.
    print("  리프 텐서에 켜기:", a.requires_grad_(True).requires_grad)
    # 만들 때 requires_grad=True 를 주어도 같다. 켜는 길이 이렇게 둘이다.
    print("  만들 때 켜기:", torch.randn(3, requires_grad=True).requires_grad)

    # b 는 직접 만든 텐서가 아니라 「경사 추적 중인 텐서에 2 를 곱한 결과」다.
    # 이런 텐서는 리프 텐서가 아니고, 플래그가 재료에 따라 저절로 True 로 정해진다.
    b = torch.randn(3, requires_grad=True) * 2        # 계산으로 생긴 것
    # try: 안의 줄이 오류를 내면 프로그램이 죽는 대신 except: 로 넘어간다.
    # 일부러 오류를 내 보고 그 메시지를 찍으려는 것이다.
    try:
        b.requires_grad_(False)                       # 리프가 아니라서 RuntimeError
    except RuntimeError as e:
        # e 가 그 오류다. str(e) 로 메시지를 꺼내 splitlines()[0] 으로 첫 줄만,
        # [:66] 으로 앞의 66글자만 잘라 찍는다. 메시지가 길어서다.
        print("  리프가 아닌 것에:", str(e).splitlines()[0][:66])

    # torch.randint(0, 5, (3,)) 은 0 이상 5 미만의 「정수」 3개짜리 텐서다.
    # (3,) 은 형태이고, 자료형은 정수(int64)다. 라벨이나 인덱스가 이런 텐서다.
    c = torch.randint(0, 5, (3,))
    try:
        c.requires_grad_(True)                        # 리프지만 정수라서 RuntimeError
    except RuntimeError as e:
        print("  정수 텐서에  :", str(e).splitlines()[0][:60])
    print("  미분은 실수에서만 뜻이 있으므로 정수에는 못 건다")


# ========================================================================
# 2. 레이어를 동결한다
# ========================================================================

def freezing():
    header("2. 레이어 동결")
    # 난수 발생기의 시드를 0 으로 고정한다. 레이어의 초기화와 아래 randn 이 난수를
    # 쓰므로, 고정해 두어야 돌릴 때마다 같은 뉴럴 네트워크와 같은 데이터가 나온다.
    torch.manual_seed(0)
    # nn.Linear(4, 8) 은 입력 4개를 받아 출력 8개를 내는 레이어다. 웨이트(형태 (8, 4))
    # 와 바이어스(형태 (8,))라는 파라메터 둘을 가지며, 둘 다 requires_grad=True 로
    # 만들어진다 — 파라메터는 트레인으로 고칠 값이기 때문이다.
    # nn.ReLU() 는 음수를 0 으로 바꾸는 함수이고 파라메터가 없다.
    # nn.Sequential 은 이 셋을 차례로 이어 하나의 뉴럴 네트워크로 만든다.
    #   net[0] = Linear(4, 8),  net[1] = ReLU,  net[2] = Linear(8, 1)
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    # net[0].parameters() 는 0번 레이어의 파라메터(웨이트, 바이어스)를 차례로 내준다.
    # 그 하나하나의 경사 추적을 끄는 것이 「동결」의 전부다.
    for p in net[0].parameters():
        p.requires_grad_(False)                       # 앞쪽 레이어를 동결한다

    # 가짜 데이터다. x 는 입력 16개 × 피처 4개(형태 (16, 4)), y 는 맞혀야 할 값
    # 16개(형태 (16, 1))다. 16 이 배치 크기다.
    x, y = torch.randn(16, 4), torch.randn(16, 1)
    # 옵티마이저는 그래디언트를 보고 파라메터를 고쳐 주는 도구다. SGD 는
    # 「파라메터 ← 파라메터 - lr * 그래디언트」를 하고, lr 이 러닝레이트다.
    # net.parameters() 는 뉴럴 네트워크의 파라메터 넷을 모두 내준다 —
    # 0번 레이어의 웨이트·바이어스, 2번 레이어의 웨이트·바이어스 차례다.
    opt = torch.optim.SGD(net.parameters(), lr=0.1)   # 전부 넘겨도 된다
    # 스텝 뒤에 값이 바뀌었는지 보려고 지금 값을 떠 둔다. detach() 는 계산 그래프와
    # 끊긴 텐서를 주고(메모리는 원본과 같이 쓴다), clone() 이 그것을 따로 복사한다.
    # [... for p in ...] 는 파라메터마다 그렇게 해 리스트로 모으는 파이썬 문법이다.
    before = [p.detach().clone() for p in net.parameters()]
    # 트레인 한 스텝이다. 아래 세 줄이 늘 한 묶음으로 나온다.
    opt.zero_grad()                       # 앞서 쌓인 그래디언트를 지운다
    # net(x) 가 순전파다 — 출력 형태 (16, 1). F.mse_loss 는 그 출력과 y 의 차이를
    # 제곱해 평균 낸 로스(수 하나)다. backward() 가 역전파다 — 로스에서 거꾸로
    # 따라가며, 경사 추적이 켜진 파라메터의 .grad 에 그래디언트를 넣는다.
    F.mse_loss(net(x), y).backward()
    opt.step()                            # .grad 를 보고 파라메터를 고친다
    # zip 은 지금 파라메터와 떠 둔 옛 값을 짝짓고, enumerate 는 0, 1, 2, 3 번호를 붙인다.
    for i, (p, b0) in enumerate(zip(net.parameters(), before)):
        # !s 는 True/False 를 문자열로 바꾸고 :5s 는 다섯 칸에 맞춰 찍는다.
        # p.grad 는 역전파가 그래디언트를 넣어 주지 않았으면 None 이다.
        # torch.equal 은 두 텐서의 형태와 값이 모두 같으면 True 다.
        print(f"  파라메터 {i}  requires_grad={p.requires_grad!s:5s} "
              f"grad={'None' if p.grad is None else '있음'}  "
              f"바뀌었나 {not torch.equal(p.detach(), b0)}")
    # 파라메터 0, 1 이 동결한 0번 레이어의 웨이트와 바이어스다. 그래디언트가 None 인
    # 파라메터는 opt.step() 이 건너뛴다.
    print("  동결한 둘은 그래디언트도 안 받고 값도 안 바뀌었다")
    print("  옵티마이저에 전부 넘겨도 되지만, 거르는 편이 깔끔하다")


# ========================================================================
# 3. 동결해도 그래디언트는 지나간다
# ========================================================================

def gradient_flows_through():
    header("3. 동결한 레이어를 그래디언트가 지나간다")
    # 2절과 같은 시드, 같은 뉴럴 네트워크다. 동결하는 자리만 바꾼다.
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    for p in net[2].parameters():                     # 이번엔 뒤쪽을 동결한다
        p.requires_grad_(False)
    x, y = torch.randn(16, 4), torch.randn(16, 1)     # 형태 (16, 4) 와 (16, 1)
    # 순전파 → 로스 → 역전파. 로스에서 0번 레이어로 가려면 동결한 2번 레이어를
    # 거꾸로 지나야 한다. 거기서 막힌다면 0번 레이어의 .grad 는 None 으로 남는다.
    F.mse_loss(net(x), y).backward()
    print("  뒤쪽을 동결했을 때")
    # net[0].weight 는 0번 레이어의 웨이트다. 「A if 조건 else B」 는 조건이 참이면
    # A, 아니면 B 를 고르는 파이썬 문법이다.
    print("    앞쪽 레이어 grad :",
          "있음" if net[0].weight.grad is not None else "None")
    # 동결한 레이어의 웨이트는 그래디언트를 쌓지 않으므로 None 이 그대로 찍힌다.
    print("    동결 레이어 grad :", net[2].weight.grad)
    print("  그래디언트가 동결한 레이어를 통과해 앞쪽까지 갔다")
    print("  동결한다는 것은 '안 쌓는다' 이지 '길을 막는다' 가 아니다")


# ========================================================================
# 4. 동결과 detach 와 no_grad 를 한 자리에서
# ========================================================================

def three_tools_on_a_net():
    header("4. 가운데 레이어에 세 가지를 걸어 본다")
    # 세 실험이 같은 데이터를 쓰도록 맨 앞에서 한 번만 만든다.
    x, y = torch.randn(16, 4), torch.randn(16, 1)

    def fresh():
        # 실험마다 「똑같이 초기화된 새 뉴럴 네트워크」를 준다. 시드를 매번 0 으로
        # 되돌리므로 세 번 다 같은 웨이트에서 시작하고, 앞 실험의 .grad 도 안 남는다.
        # 이번에는 Linear 셋을 이었다 — 4 → 8 → 8 → 1. 가운데가 net[1] 이다.
        torch.manual_seed(0)
        return nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 8), nn.Linear(8, 1))

    def report(label, net):
        # 세 레이어의 웨이트가 그래디언트를 받았는지를 "있음"/"None" 으로 모은다.
        got = ["있음" if p.grad is not None else "None"
               for p in (net[0].weight, net[1].weight, net[2].weight)]
        # :26s 와 :4s 는 문자열을 26칸, 4칸에 맞춰 찍어 줄을 가지런히 한다.
        print(f"  {label:26s} 0번 레이어 {got[0]:4s}  1번 레이어 {got[1]:4s}  2번 레이어 {got[2]:4s}")

    # (가) 가운데 레이어의 플래그만 끈다
    # 2·3절에서 한 동결이다. 순전파는 평소처럼 net(x) 로 통째로 한다.
    net = fresh()
    for p in net[1].parameters():
        p.requires_grad_(False)
    F.mse_loss(net(x), y).backward()
    report("(가) requires_grad_(False)", net)

    # (나) 가운데 레이어의 출력을 detach 한다
    # 순전파를 손으로 나눠 적는다. net[0](x) 는 0번 레이어만 지난 값(형태 (16, 8))
    # 이고, 그것을 net[1] 에 넣은 결과에 .detach() 를 건다. detach() 는 값은 같되
    # 계산 그래프와 끊긴 텐서를 준다 — h 는 「어떻게 만들어졌는지」를 잊은 텐서다.
    net = fresh()
    h = net[1](net[0](x)).detach()
    # 로스에서 거꾸로 가면 2번 레이어까지는 가지만 h 에서 기록이 끝난다.
    F.mse_loss(net[2](h), y).backward()
    report("(나) 출력을 detach", net)

    # (다) 가운데 레이어를 no_grad 로 돌린다
    # with torch.no_grad(): 블록 안의 계산은 계산 그래프에 기록되지 않는다. 블록을
    # 나오면 기록이 다시 켜진다. 0번 레이어는 블록 밖에서 돌렸으니 h0 에는 기록이
    # 있지만, h 는 기록 없이 만들어져 h0 과 이어지지 않는다.
    net = fresh()
    h0 = net[0](x)                # 0번 레이어의 출력, 형태 (16, 8) — 기록된다
    with torch.no_grad():
        h = net[1](h0)            # 1번 레이어의 출력, 형태 (16, 8) — 기록되지 않는다
    F.mse_loss(net[2](h), y).backward()
    report("(다) no_grad 로 감싸기", net)

    print("  (가) 만 0번 레이어까지 그래디언트가 간다. 1번 레이어를 지나보내기 때문이다")
    print("  (나)(다) 는 1번 레이어에서 길이 끊겨 0번 레이어가 아무것도 못 받는다")
    print("  '가운데 레이어만 동결하고 싶다' 면 (가) 라야 한다")


def main():
    # 네 절을 차례로 돌린다.
    where_allowed()
    freezing()
    gradient_flows_through()
    three_tools_on_a_net()


# 이 파일을 직접 돌렸을 때만 main() 을 부른다. 다른 파일이 import 로 불러올 때는
# 돌지 않게 하는 파이썬의 관용구다.
if __name__ == "__main__":
    main()
```

**출력:**

```
=== 1. 플래그를 걸 수 있는 텐서 ===
  리프 텐서에 켜기: True
  만들 때 켜기: True
  리프가 아닌 것에: you can only change requires_grad flags of leaf variables. If you 
  정수 텐서에  : only Tensors of floating point dtype can require gradients
  미분은 실수에서만 뜻이 있으므로 정수에는 못 건다

=== 2. 레이어 동결 ===
  파라메터 0  requires_grad=False grad=None  바뀌었나 False
  파라메터 1  requires_grad=False grad=None  바뀌었나 False
  파라메터 2  requires_grad=True  grad=있음  바뀌었나 True
  파라메터 3  requires_grad=True  grad=있음  바뀌었나 True
  동결한 둘은 그래디언트도 안 받고 값도 안 바뀌었다
  옵티마이저에 전부 넘겨도 되지만, 거르는 편이 깔끔하다

=== 3. 동결한 레이어를 그래디언트가 지나간다 ===
  뒤쪽을 동결했을 때
    앞쪽 레이어 grad : 있음
    동결 레이어 grad : None
  그래디언트가 동결한 레이어를 통과해 앞쪽까지 갔다
  동결한다는 것은 '안 쌓는다' 이지 '길을 막는다' 가 아니다

=== 4. 가운데 레이어에 세 가지를 걸어 본다 ===
  (가) requires_grad_(False)  0번 레이어 있음    1번 레이어 None  2번 레이어 있음  
  (나) 출력을 detach             0번 레이어 None  1번 레이어 None  2번 레이어 있음  
  (다) no_grad 로 감싸기          0번 레이어 None  1번 레이어 None  2번 레이어 있음  
  (가) 만 0번 레이어까지 그래디언트가 간다. 1번 레이어를 지나보내기 때문이다
  (나)(다) 는 1번 레이어에서 길이 끊겨 0번 레이어가 아무것도 못 받는다
  '가운데 레이어만 동결하고 싶다' 면 (가) 라야 한다
```

## 2. 리프 텐서에만, 실수에만 걸 수 있다

```
리프가 아닌 것에: you can only change requires_grad flags of leaf variables
정수 텐서에  : only Tensors of floating point dtype can require gradients
```

**리프 텐서**(leaf tensor)에만 걸 수 있다 — 연산의 결과가 아니라 사용자가 직접 만든 텐서로, 계산 그래프의 리프에 해당한다(오류 메시지의 「leaf variables」가 이것이다). 계산으로 생긴 텐서의 플래그는 이미 그 위쪽에 의해 정해져
있으므로 손으로 바꿀 일이 아니다. 끊고 싶으면 `detach()`를 쓰라는 것이 오류
메시지가 이어서 하는 말이다.

**실수에만** 걸 수 있다. 정수에 미분은 뜻이 없다. 라벨이나 인덱스 텐서에
실수로 `requires_grad=True`를 걸려 하면 여기서 걸린다.

## 3. 레이어 동결

```
파라메터 0  requires_grad=False  grad=None  바뀌었나 False
파라메터 1  requires_grad=False  grad=None  바뀌었나 False
파라메터 2  requires_grad=True   grad=있음  바뀌었나 True
파라메터 3  requires_grad=True   grad=있음  바뀌었나 True
```

```python
# 0번 레이어의 파라메터(웨이트, 바이어스)를 하나씩 꺼내 경사 추적을 끈다.
for p in net[0].parameters():
    p.requires_grad_(False)       # 끝의 _ 는 p 자체를 제자리에서 고친다는 뜻이다
```

두 줄로 끝난다. 동결한 파라메터는 **그래디언트를 안 받고**(`grad=None`) 따라서 **값도
안 바뀐다.**

옵티마이저에 `net.parameters()`를 통째로 넘겨도 괜찮다 — 그래디언트가 `None`인
파라메터는 `step()`이 건너뛴다. 다만 거르는 편이 깔끔하다.

```python
# [p for p in ... if 조건] 은 조건이 참인 것만 골라 리스트로 모으는 파이썬 문법이다.
# 경사 추적이 켜진 파라메터, 곧 동결하지 않은 것만 옵티마이저에 넘긴다.
opt = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=0.1)
```

거르면 옵티마이저가 동결한 파라메터의 **상태(모멘텀 버퍼 같은 것)를 만들지
않으므로** 메모리도 조금 아낀다. 그리고 나중에 그 레이어의 동결을 풀(unfreeze) 때 옵티마이저를
다시 만들어야 한다는 것이 코드에 드러난다.

## 4. 동결해도 그래디언트는 지나간다

```
뒤쪽을 동결했을 때
  앞쪽 레이어 grad : 있음
  동결 레이어 grad : None
```

![레이어 0 에서 동결한 레이어 1 을 거쳐 로스로 가는 그림. 아래쪽으로 역전파되는 점선 화살표가 동결한 레이어를 그대로 통과해 레이어 0 까지 닿고, 맨 아래 두 칸 가운데 레이어 0 의 칸만 채워져 있다.](figures/frozen_passes_through.svg)

뒤쪽 레이어를 동결했는데 **앞쪽 레이어는 그래디언트를 받았다.** 그래디언트가 동결한 레이어를 통과해 거기까지
간 것이다. 그림에서 역전파 화살표가 점선 칸을 **뚫고 지나간다.**

당연한 일이다. 사슬 규칙이 앞쪽 레이어까지 가려면 뒤쪽 레이어의 **웨이트를 거쳐야**
하고, 그 웨이트를 쓰는 것과 그 웨이트에 대한 그래디언트를 쌓는 것은 다른 일이다.

**동결한다는 것은 "안 쌓는다"이지 "길을 막는다"가 아니다.** 그래서 동결한다고 해서
역전파가 빨라지지는 않는다 — 파라메터 그래디언트 계산 하나가 빠질 뿐, 엑티베이션 값에 대한
그래디언트는 그대로 계산된다.

## 5. 세 가지가 가운데 레이어에서 갈린다

| | 0번 레이어 | 1번 레이어 (손댄 레이어) | 2번 레이어 |
|---|---|---|---|
| `requires_grad_(False)` | **있음** | None | 있음 |
| 출력을 `detach()` | None | None | 있음 |
| `no_grad`로 감싸기 | None | None | 있음 |

![세 줄 그림. 줄마다 레이어 0, 레이어 1, 레이어 2, 로스가 이어지고 레이어마다 아래에 그래디언트를 받았는지가 칸으로 붙어 있다. 첫 줄은 레이어 1 만 점선이고 길이 이어져 레이어 0 의 칸이 채워져 있다. 둘째와 셋째 줄은 레이어 1 과 레이어 2 사이가 끊겨 레이어 0 의 칸이 비어 있다.](figures/three_tools_on_a_net.svg)

**`requires_grad_(False)`만 0번 레이어까지 그래디언트를 보낸다.** 나머지 둘은 1번 레이어에서 길이
끊겨 그 앞이 아무것도 못 받는다. 그림의 첫 줄만 화살표가 끊기지 않은 것이
그 차이다 — 둘째 줄에서 1번 레이어가 실선인 것은 **플래그는 그대로 서 있고 받을 것이
없을 뿐**이기 때문이다.

그러니 셋을 바꿔 쓸 수 없다.

- **가운데 레이어만 동결하고 앞뒤는 배우게** 하려면 → `requires_grad_(False)`
- **그 앞을 통째로 멈추려면** → `detach()`나 `no_grad`

전이 학습에서 앞쪽을 동결할 때 둘 다 쓸 수 있는 것은 **앞쪽이 입력과 맞닿아 있어**
더 거슬러 갈 곳이 없기 때문이다. 가운데에 걸면 차이가 바로 드러난다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
모델의 파라메터 가운데 **트레인되는 것의 개수**를 세는 한 줄을 적어라. 앞쪽 레이어를
동결한 전후로 세어 보라.

</div>

??? success "풀이"
    ```python
    # 본문 코드에 이어서 돌린다 — torch, nn 을 이미 불러온 상태다.
    torch.manual_seed(0)          # 시드 고정. 개수만 세므로 값은 결과에 닿지 않는다
    # 입력 784개 → 128개 → 10개. 파라메터 수를 손으로 세면
    #   0번 레이어: 웨이트 128*784 = 100,352  +  바이어스 128  = 100,480
    #   2번 레이어: 웨이트 10*128  =   1,280  +  바이어스 10   =   1,290
    #   합쳐서 101,770 (ReLU 는 파라메터가 없다)
    net = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))

    def count(m):
        # p.numel() 은 텐서 p 에 든 수의 개수다. 형태 (128, 784) 면 100,352 다.
        # 파라메터마다 그 개수를 내어 sum 으로 모두 더한다.
        total = sum(p.numel() for p in m.parameters())
        # 같은 합인데 경사 추적이 켜진 것, 곧 트레인되는 파라메터만 센다.
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return total, trainable

    # count(net) 이 돌려주는 (전체, 트레인) 을 * 로 풀어 format 의 두 칸에 넣는다.
    # {:,} 는 수를 세 자리마다 쉼표를 넣어 찍으라는 뜻이다.
    print("  동결 전 : 전체 {:,}  트레인 {:,}".format(*count(net)))
    for p in net[0].parameters():
        p.requires_grad_(False)   # 0번 레이어를 동결한다. 100,480개가 빠진다
    # 남는 것은 2번 레이어의 1,290개다.
    print("  앞쪽 동결한 뒤: 전체 {:,}  트레인 {:,}".format(*count(net)))
    ```

    ```
      동결 전 : 전체 101,770  트레인 101,770
      앞쪽 동결한 뒤: 전체 101,770  트레인 1,290
    ```

    **전체는 그대로이고 트레인되는 것만 줄었다.** 101,770개 중 1,290개, 1.3%만
    배운다.

    `sum(p.numel() for p in m.parameters() if p.requires_grad)` 한 줄은 모델을
    고친 뒤 **의도대로 되었는지 확인하는 가장 빠른 길**이다. 동결하는 코드를 쓰고
    이 수를 안 찍어 보면, 오타로 아무것도 안 동결했거나 전부 동결했어도 모른다.

    큰 모델을 미세조정할 때 이 비율이 핵심이다. 4.6절의 「VGG16 — 3×3 컨볼루션의
    반복」 쪽에서 하듯 분류기만 새로 다는 것이 곧 이 수를 몇 퍼센트로 낮추는 일이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
동결한 레이어는 그래디언트를 안 받는다. 그렇다면 **역전파가 빨라지는가?** 앞쪽 절반을 동결한
깊은 뉴럴 네트워크에서 재어라.

</div>

??? success "풀이"
    ```python
    # 본문 코드에 이어서 돌린다 — torch, nn 을 이미 불러온 상태다.
    import time                     # 시간을 재는 파이썬 표준 모듈

    def spread(fn, rep=15):
        # fn 을 rep 번 돌려 가장 빠른 시간과 가장 느린 시간을 밀리초로 돌려준다.
        for _ in range(3):          # 예열 — 첫 몇 번은 유난히 느리다
            fn()
        ts = []
        for _ in range(rep):
            # perf_counter() 는 지금 시각(초)이다. fn 을 돌리기 전과 뒤의 차이가
            # 걸린 시간이다. 한 번만 재면 우연에 흔들리므로 15번 재어 모은다.
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        return min(ts) * 1e3, max(ts) * 1e3     # 초 → 밀리초 (1e3 은 1000)

    def build(freeze_first_half):
        torch.manual_seed(0)        # 두 경우가 같은 웨이트에서 시작하게 한다
        # Linear(512, 512) 레이어 12개를 리스트로 만들고, * 로 풀어 Sequential 에
        # 하나씩 넘긴다. net[0] 부터 net[11] 까지다.
        net = nn.Sequential(*[nn.Linear(512, 512) for _ in range(12)])
        if freeze_first_half:
            for layer in net[:6]:                   # 앞의 여섯 레이어 net[0]~net[5]
                for p in layer.parameters():        # 그 레이어의 웨이트와 바이어스
                    p.requires_grad_(False)         # 경사 추적을 끈다 = 동결
        return net

    # 난수의 시드를 고정한다. 아래에서 뽑는 텐서가 돌릴 때마다 같은 값이 된다.
    torch.manual_seed(0)
    x = torch.randn(128, 512)       # 입력 128개 × 피처 512개, 형태 (128, 512)
    # 같은 일을 두 번 한다 — 한 번은 동결 없이(False), 한 번은 앞 절반을 동결해(True).
    for label, frozen in (("전부 트레인 ", False), ("앞 절반 동결", True)):
        net = build(frozen)

        def step():
            # 재려는 한 스텝이다. zero_grad() 는 앞서 쌓인 그래디언트를 지운다.
            net.zero_grad()
            # net(x) 가 순전파(출력 형태 (128, 512)), .pow(2).mean() 이 그 출력을
            # 제곱해 평균 낸 수 하나(여기서는 이것을 로스로 삼는다), backward() 가
            # 역전파다. 재는 시간에는 순전파와 역전파가 다 들어 있다.
            net(x).pow(2).mean().backward()

        lo, hi = spread(step)
        # :6.2f 는 여섯 칸에 소수점 아래 두 자리로 찍으라는 뜻이다.
        print(f"  {label}  {lo:6.2f} ~ {hi:6.2f} ms")
    ```

    ```
      전부 트레인     3.07 ~   5.54 ms
      앞 절반 동결    1.89 ~   3.01 ms
    ```

    빨라지기는 한다. 가장 빠른 값끼리 견주면 대략 **40%** 줄었다. 다만 두 범위가
    거의 맞닿아 있으므로 "절반에 못 미치게 빨라진다"고만 말할 수 있다.

    절반을 동결했는데 딱 절반이 빨라지지 않는 까닭이 4절이다. 동결한 레이어도 **엑티베이션 값에
    대한 그래디언트는 계산해서 넘겨준다.** 빠지는 것은 **웨이트에 대한 그래디언트**뿐이고,
    그것은 역전파가 하는 일의 일부다.

    정말로 앞쪽을 건너뛰고 싶으면 그 출력에 `detach()`를 걸어야 한다. 그러면
    역전파가 거기서 멈춘다 — 대신 5절에서 본 대로 그 앞은 영영 못 배운다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
레이어를 동결한 채로 트레인하다가 중간에 **동결을 풀려** 한다. `requires_grad_(True)`만
되돌리면 되는가? 옵티마이저를 어떻게 해야 하는지 확인하라.

</div>

??? success "풀이"
    ```python
    # 본문 코드에 이어서 돌린다 — torch, nn, F 를 이미 불러온 상태다.
    torch.manual_seed(0)          # 시드 고정 — 같은 초기화, 같은 데이터
    # 레이어 둘짜리 뉴럴 네트워크다. net[0] = Linear(4, 8), net[1] = Linear(8, 1).
    # 파라메터는 넷이다 — 레이어마다 웨이트와 바이어스.
    net = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 1))
    x, y = torch.randn(16, 4), torch.randn(16, 1)     # 입력 (16, 4), 맞혀야 할 값 (16, 1)

    # --- 0번 레이어를 동결하고, 「거른」 옵티마이저를 만든다 ---
    for p in net[0].parameters():
        p.requires_grad_(False)
    # 경사 추적이 켜진 파라메터만 골라 넘긴다. 지금은 1번 레이어의 둘뿐이다.
    opt = torch.optim.SGD([p for p in net.parameters() if p.requires_grad],
                          lr=0.1)
    # opt.param_groups 는 옵티마이저가 맡은 파라메터 묶음의 목록이고, 묶음마다
    # g["params"] 에 그 텐서들이 들어 있다. 그 길이를 더하면 맡은 텐서의 수다.
    print("  옵티마이저가 들고 있는 텐서 수:",
          sum(len(g["params"]) for g in opt.param_groups))

    # --- 플래그만 되돌리고, 옵티마이저는 그대로 쓴다 ---
    for p in net[0].parameters():            # 동결을 푼다
        p.requires_grad_(True)
    # 0번 레이어의 웨이트를 떠 둔다. detach() 로 계산 그래프와 끊고 clone() 으로
    # 따로 복사해 두어야, 뒤에 웨이트가 바뀌어도 before 는 옛 값으로 남는다.
    before = net[0].weight.detach().clone()
    opt.zero_grad()                          # 쌓인 그래디언트를 지운다
    F.mse_loss(net(x), y).backward()         # 순전파 → 로스 → 역전파
    opt.step()                               # opt 가 맡은 파라메터만 고친다
    # 플래그를 켰으니 역전파는 0번 레이어의 .grad 를 채운다.
    print("  동결을 푼 뒤 0번 레이어 grad:",
          "있음" if net[0].weight.grad is not None else "None")
    # 그런데 opt 의 목록에는 0번 레이어가 없다. torch.equal 은 두 텐서의 값이
    # 모두 같으면 True 이므로, not 을 붙이면 「바뀌었나」가 된다.
    print("  그런데 값이 바뀌었나:",
          not torch.equal(net[0].weight.detach(), before))

    # --- 옵티마이저를 다시 만든다 ---
    # 이번에는 net.parameters() 넷을 모두 넘긴다. 0번 레이어도 목록에 든다.
    opt2 = torch.optim.SGD(net.parameters(), lr=0.1)   # 다시 만든다
    opt2.zero_grad()
    F.mse_loss(net(x), y).backward()
    opt2.step()
    print("  옵티마이저를 다시 만든 뒤 바뀌었나:",
          not torch.equal(net[0].weight.detach(), before))
    ```

    ```
      옵티마이저가 들고 있는 텐서 수: 2
      동결을 푼 뒤 0번 레이어 grad: 있음
      그런데 값이 바뀌었나: False
      옵티마이저를 다시 만든 뒤 바뀌었나: True
    ```

    **플래그만 되돌리면 모자란다.**

    동결을 푼 뒤 그래디언트는 제대로 온다 — `grad`가 "있음"이다. 그런데 **값이 안 바뀐다.**
    옵티마이저를 만들 때 거른 파라메터는 그 목록에 아예 없고, 옵티마이저는 제
    목록에 있는 것만 갱신하기 때문이다.

    옵티마이저를 다시 만들면 움직인다. 그래서 점진적 동결 해제
    (gradual unfreezing)는 동결을 풀 때마다 옵티마이저를 다시 만들거나,
    `opt.add_param_group({"params": net[0].parameters()})`로 집어넣어야 한다.

    > 처음부터 `net.parameters()`를 통째로 넘겨 두었다면 이 문제가 없다. 동결한
    > 동안에는 그래디언트가 `None`이라 `step()`이 건너뛰고, 동결을 풀면 저절로 갱신되기
    > 시작한다. 3절에서 "통째로 넘겨도 된다"고 한 것이 이 때문이다. 메모리를
    > 조금 더 쓰는 대신 동결 해제가 간단해진다.
