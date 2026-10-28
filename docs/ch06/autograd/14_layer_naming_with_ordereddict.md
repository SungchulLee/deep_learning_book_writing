# `OrderedDict`로 레이어에 이름 붙이기 — 번호는 밀리고 이름은 안 밀린다

`nn.Sequential`에 층을 그냥 넣으면 `state_dict`의 키가 `0.weight`, `2.weight`
처럼 **번호**가 된다. `OrderedDict`으로 이름을 주면 `encoder.weight`가 된다.

보기 좋으라고 하는 일이 아니다. 번호는 **층을 하나 끼워 넣는 순간 전부
밀리고**, 밀리면 옛 체크포인트를 못 읽는다.

## 1. 코드

```python
"""Sequential 에 이름 붙이기 — 번호는 밀리고 이름은 안 밀린다.

`nn.Sequential` 에 층을 그냥 넣으면 키가 `0.weight`, `2.weight` 처럼
**번호**가 된다. `OrderedDict` 으로 이름을 주면 `encoder.weight` 가
된다.

보기 좋으라고 하는 일이 아니다. 번호는 **층을 하나 끼워 넣는 순간
전부 밀리고**, 그러면 옛 체크포인트를 못 읽는다.

실행: 이 쪽의 코드를 파일로 갈무리해 돌린다
"""
from collections import OrderedDict

import torch
import torch.nn as nn


def header(title):
    print(f"\n=== {title} ===")


def plain():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def labelled():
    return nn.Sequential(OrderedDict([
        ("encoder", nn.Linear(4, 8)),
        ("act", nn.ReLU()),
        ("head", nn.Linear(8, 2)),
    ]))


# ========================================================================
# 1. 키가 달라진다
# ========================================================================

def keys():
    header("1. state_dict 의 키")
    print("  번호 :", list(plain().state_dict().keys()))
    print("  이름 :", list(labelled().state_dict().keys()))
    print("  이름은 ReLU 를 건너뛴다 — 파라미터가 없는 층은 키가 없다")
    print("  번호 쪽에서 1 번이 비어 있는 것도 같은 까닭이다")


# ========================================================================
# 2. 층을 하나 끼우면
# ========================================================================

def inserting():
    header("2. 가운데에 층을 끼우면")
    old_plain = plain().state_dict()
    old_named = labelled().state_dict()

    new_plain = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                              nn.Dropout(0.1), nn.Linear(8, 2))
    new_named = nn.Sequential(OrderedDict([
        ("encoder", nn.Linear(4, 8)),
        ("act", nn.ReLU()),
        ("drop", nn.Dropout(0.1)),
        ("head", nn.Linear(8, 2)),
    ]))
    print("  번호 :", list(new_plain.state_dict().keys()), " <- 2 가 3 으로 밀렸다")
    print("  이름 :", list(new_named.state_dict().keys()), " <- 그대로다")

    for label, model, sd in (("번호", new_plain, old_plain),
                             ("이름", new_named, old_named)):
        try:
            model.load_state_dict(sd)
            print(f"  {label} 쪽에 옛 체크포인트 불러오기: 성공")
        except RuntimeError as e:
            print(f"  {label} 쪽에 옛 체크포인트 불러오기: 실패")
            for line in str(e).splitlines()[1:3]:
                print("     ", line.strip()[:66])


# ========================================================================
# 3. 이름으로 고르기
# ========================================================================

def select_by_name():
    header("3. 이름으로 골라 얼리기")
    model = labelled()
    for name, p in model.named_parameters():
        if name.startswith("encoder."):
            p.requires_grad_(False)
    for name, p in model.named_parameters():
        print(f"  {name:16s} requires_grad={p.requires_grad}")
    print("  번호였다면 name.startswith('0.') 라고 적어야 한다")
    print("  그 코드는 층을 끼우는 순간 엉뚱한 층을 얼린다")


# ========================================================================
# 4. 점으로 이어진 이름이 곧 키다
# ========================================================================

def nested():
    header("4. 중첩하면 점으로 이어진다")
    model = nn.Sequential(OrderedDict([
        ("features", nn.Sequential(OrderedDict([
            ("conv", nn.Linear(4, 8)),
            ("act", nn.ReLU()),
        ]))),
        ("classifier", nn.Linear(8, 2)),
    ]))
    for k in model.state_dict():
        print("  ", k)
    print("  model.features.conv 로 바로 닿을 수 있다:",
          type(model.features.conv).__name__)
    print("  이름이 코드의 길과 저장 파일의 키를 동시에 정한다")


def main():
    torch.manual_seed(0)
    keys()
    inserting()
    select_by_name()
    nested()


if __name__ == "__main__":
    main()
```

**출력:**

```
=== 1. state_dict 의 키 ===
  번호 : ['0.weight', '0.bias', '2.weight', '2.bias']
  이름 : ['encoder.weight', 'encoder.bias', 'head.weight', 'head.bias']
  이름은 ReLU 를 건너뛴다 — 파라미터가 없는 층은 키가 없다
  번호 쪽에서 1 번이 비어 있는 것도 같은 까닭이다

=== 2. 가운데에 층을 끼우면 ===
  번호 : ['0.weight', '0.bias', '3.weight', '3.bias']  <- 2 가 3 으로 밀렸다
  이름 : ['encoder.weight', 'encoder.bias', 'head.weight', 'head.bias']  <- 그대로다
  번호 쪽에 옛 체크포인트 불러오기: 실패
      Missing key(s) in state_dict: "3.weight", "3.bias".
      Unexpected key(s) in state_dict: "2.weight", "2.bias".
  이름 쪽에 옛 체크포인트 불러오기: 성공

=== 3. 이름으로 골라 얼리기 ===
  encoder.weight   requires_grad=False
  encoder.bias     requires_grad=False
  head.weight      requires_grad=True
  head.bias        requires_grad=True
  번호였다면 name.startswith('0.') 라고 적어야 한다
  그 코드는 층을 끼우는 순간 엉뚱한 층을 얼린다

=== 4. 중첩하면 점으로 이어진다 ===
   features.conv.weight
   features.conv.bias
   classifier.weight
   classifier.bias
  model.features.conv 로 바로 닿을 수 있다: Linear
  이름이 코드의 길과 저장 파일의 키를 동시에 정한다
```

## 2. 파라미터 없는 층은 키가 없다

```
번호 : ['0.weight', '0.bias', '2.weight', '2.bias']
이름 : ['encoder.weight', 'encoder.bias', 'head.weight', 'head.bias']
```

번호 쪽에서 **1번이 비어 있다.** 그 자리의 `ReLU`에 파라미터가 없기 때문이다.
이름 쪽에서도 `act`가 안 나오는 것이 같은 까닭이다.

그래서 번호가 **층의 번호이지 파라미터의 번호가 아니다.** `2.weight`를 보고
"세 번째 파라미터"라고 읽으면 틀린다.

## 3. 층을 하나 끼우면

```
번호 : ['0.weight', '0.bias', '3.weight', '3.bias']   <- 2 가 3 으로 밀렸다
이름 : ['encoder.weight', ..., 'head.weight', ...]    <- 그대로다

번호 쪽에 옛 체크포인트 불러오기: 실패
    Missing key(s) in state_dict: "3.weight", "3.bias".
    Unexpected key(s) in state_dict: "2.weight", "2.bias".
이름 쪽에 옛 체크포인트 불러오기: 성공
```

드롭아웃 하나를 끼웠을 뿐인데 **마지막 선형층의 키가 `2`에서 `3`으로 밀렸고**,
저장해 둔 가중치를 못 읽는다.

이름을 붙여 두면 아무 일도 없다. 끼운 층이 어디 있든 `head`는 `head`다.

> 다행히 **터져 준다.** 키가 안 맞으면 `load_state_dict`가 거절하므로,
> 엉뚱한 가중치를 조용히 읽는 일은 없다. 다만 `strict=False`로 불러오면
> 조용해진다 — 연습문제 2에서 그 자리를 본다.

## 4. 이름으로 골라 얼리기

```python
for name, p in model.named_parameters():
    if name.startswith("encoder."):
        p.requires_grad_(False)
```

```
encoder.weight   requires_grad=False
encoder.bias     requires_grad=False
head.weight      requires_grad=True
head.bias        requires_grad=True
```

번호였다면 `name.startswith("0.")`이라고 적어야 하는데, 그 코드는 **층을 끼우는
순간 엉뚱한 층을 얼린다.** 게다가 오류도 안 난다 — 0번 층은 여전히 있으니까.

「경사 추적 끄기」에서 본 얼리기가 실전에서 이 꼴로 쓰인다. 이름이 있어야 그
코드가 안전해진다.

## 5. 중첩하면 점으로 이어진다

```
features.conv.weight
features.conv.bias
classifier.weight
classifier.bias
```

`model.features.conv`로 코드에서 닿을 수 있고, 저장 파일의 키도 같은 글자다.
**이름 하나가 코드의 길과 파일의 키를 동시에 정한다.**

그래서 속성 이름을 바꾸는 것은 가볍게 할 일이 아니다. 「모델 체크포인트」에서
구조가 저장되지 않는다고 한 것이 여기서 실제 비용으로 나타난다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
이름을 붙인 `Sequential`에 **번호로도** 닿을 수 있는가? 둘이 같은 객체인가?

</div>

??? success "풀이"
    ```python
    m = nn.Sequential(OrderedDict([("a", nn.Linear(2, 2)),
                                   ("b", nn.ReLU()),
                                   ("c", nn.Linear(2, 2))]))
    print("  이름으로 :", type(m.a).__name__, type(m.c).__name__)
    print("  번호로도 :", type(m[0]).__name__, type(m[2]).__name__)
    print("  둘이 같은 객체인가:", m.a is m[0])

    p = nn.Sequential(nn.Linear(2, 2))
    try:
        p.foo
    except AttributeError as e:
        print("  번호만 있는 쪽에 이름으로 닿으면:", str(e).splitlines()[0][:56])
    ```

    ```
      이름으로 : Linear Linear
      번호로도 : Linear Linear
      둘이 같은 객체인가: True
      번호만 있는 쪽에 이름으로 닿으면: 'Sequential' object has no attribute 'foo'
    ```

    **둘 다 되고 같은 객체다.** 이름을 붙이면 번호를 잃는 것이 아니라 **길이
    하나 더 생긴다.**

    그러니 이름을 붙이는 데 드는 값이 없다. 안 붙일 까닭이 없다는 뜻이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이름을 `enc`에서 `encoder`로 **바꾸면** 옛 체크포인트를 어떻게 읽는가? 두 가지
길을 시험하라.

</div>

??? success "풀이"
    ```python
    old = nn.Sequential(OrderedDict([("enc", nn.Linear(4, 8)),
                                     ("head", nn.Linear(8, 2))]))
    sd = old.state_dict()
    renamed = nn.Sequential(OrderedDict([("encoder", nn.Linear(4, 8)),
                                         ("head", nn.Linear(8, 2))]))
    try:
        renamed.load_state_dict(sd)
    except RuntimeError as e:
        for line in str(e).splitlines()[:3]:
            print("  ", line.strip()[:72])

    fixed = OrderedDict((k.replace("enc.", "encoder."), v) for k, v in sd.items())
    renamed.load_state_dict(fixed)
    print("  키 이름을 바꿔 주면 성공")
    print("  느슨하게 불러오기:", renamed.load_state_dict(sd, strict=False))
    ```

    ```
       Error(s) in loading state_dict for Sequential:
       Missing key(s) in state_dict: "encoder.weight", "encoder.bias".
       Unexpected key(s) in state_dict: "enc.weight", "enc.bias".
      키 이름을 바꿔 주면 성공
      느슨하게 불러오기: _IncompatibleKeys(missing_keys=['encoder.weight',
        'encoder.bias'], unexpected_keys=['enc.weight', 'enc.bias'])
    ```

    두 길이 **전혀 다르다.**

    **키를 바꿔 주는 길**이 맞다. `state_dict`는 그냥 사전이므로 키 이름을
    고쳐 넘기면 된다. 다른 저장소에서 받은 가중치를 내 모델에 맞출 때 쓰는
    표준 방법이다.

    **`strict=False`는 위험하다.** 오류 대신 `_IncompatibleKeys`를 돌려주고
    넘어가는데, 그 결과는 **`encoder`가 안 채워진 모델**이다. 무작위 초기값
    그대로인 층을 학습된 모델이라 믿고 쓰게 된다.

    돌려받은 값을 반드시 확인해야 한다. `missing_keys`가 비어 있지 않으면
    뭔가 안 채워졌다는 뜻이다.

    ```python
    r = model.load_state_dict(sd, strict=False)
    assert not r.missing_keys, f"안 채워진 키: {r.missing_keys}"
    ```

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
층이 넷인 모델의 가중치를 층이 둘인 모델에 넣으면 어떻게 되는가? 반대는?
`strict=False`로 양쪽을 해 보고 **어느 쪽이 더 위험한지** 가려라.

</div>

??? success "풀이"
    ```python
    big = nn.Sequential(OrderedDict([(f"layer{i}", nn.Linear(8, 8))
                                     for i in range(4)]))
    small = nn.Sequential(OrderedDict([(f"layer{i}", nn.Linear(8, 8))
                                       for i in range(2)]))

    r = small.load_state_dict(big.state_dict(), strict=False)
    print("  큰 것을 작은 것에:", "missing", len(r.missing_keys),
          "unexpected", len(r.unexpected_keys))
    print("    unexpected:", r.unexpected_keys[:4])

    r2 = big.load_state_dict(small.state_dict(), strict=False)
    print("  작은 것을 큰 것에:", "missing", len(r2.missing_keys),
          "unexpected", len(r2.unexpected_keys))
    print("    missing:", r2.missing_keys[:4])
    ```

    ```
      큰 것을 작은 것에: missing 0 unexpected 4
        unexpected: ['layer2.weight', 'layer2.bias', 'layer3.weight', 'layer3.bias']
      작은 것을 큰 것에: missing 4 unexpected 0
        missing: ['layer2.weight', 'layer2.bias', 'layer3.weight', 'layer3.bias']
    ```

    **둘이 정반대이고, 위험한 정도가 다르다.**

    **큰 것 → 작은 것**은 `unexpected`만 나온다. 작은 모델의 두 층은 **전부 제대로
    채워졌고**, 남는 두 층 치의 가중치를 버린 것뿐이다. 모델은 의도대로 동작한다.

    **작은 것 → 큰 것**은 `missing`이 나온다. 큰 모델의 뒤쪽 두 층이 **무작위
    초기값 그대로** 남는다. 그런데 앞쪽 두 층은 학습된 값이라, **반쯤 학습된
    모델**이 된다.

    뒤쪽이 훨씬 나쁘다. 모델이 돌아가고 숫자도 나오는데 성능만 이상하기
    때문이다. 전이학습에서 "사전학습 가중치를 올렸는데 왜 성능이 나쁘지"의
    흔한 원인이 이것이다.

    그래서 `strict=False`를 쓸 때는 **어느 쪽을 기대하는지 적어 두는 편이**
    낫다.

    ```python
    r = model.load_state_dict(pretrained, strict=False)
    # 새로 단 분류기만 비어 있어야 한다
    assert all(k.startswith("classifier.") for k in r.missing_keys)
    ```
