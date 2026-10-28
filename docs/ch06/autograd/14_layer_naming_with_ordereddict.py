"""Sequential 에 이름 붙이기 — 번호는 밀리고 이름은 안 밀린다.

`nn.Sequential` 에 층을 그냥 넣으면 키가 `0.weight`, `2.weight` 처럼
**번호**가 된다. `OrderedDict` 으로 이름을 주면 `encoder.weight` 가
된다.

보기 좋으라고 하는 일이 아니다. 번호는 **층을 하나 끼워 넣는 순간
전부 밀리고**, 그러면 옛 체크포인트를 못 읽는다.

실행: python 14_layer_naming_with_ordereddict.py
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
