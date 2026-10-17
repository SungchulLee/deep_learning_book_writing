# 분산 학습

DataParallel 등으로 감싼 모델의 체크포인트를 일반 모델과 주고받을 때 생기는 키 접두사 문제를 다룬다. 이 페이지는 분산 학습 알고리즘 전체보다 상태 사전의 저장·복원에 초점을 둔다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
============================================================
분산 학습: 검사점의 저장과 불러오기
============================================================

DataParallel이나 DistributedDataParallel을 쓸 때 모델을 제대로
저장하고 불러오는 법을 배운다.

주제:
- DataParallel의 검사점
- DistributedDataParallel의 검사점
- 'module.' 접두사 다루기
- 다중 GPU의 좋은 관행
"""

import torch
import torch.nn as nn
# 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
# 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
torch.manual_seed(0)

print("=" * 70)
print("DISTRIBUTED TRAINING CHECKPOINTS")
print("=" * 70)

# ============================================================
# 시연을 위한 간단한 모델
# ============================================================

class SimpleModel(nn.Module):
    """시연을 위한 간단한 모델"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# ============================================================
# DataParallel 상황
# ============================================================

print("\n" + "=" * 70)
print("SCENARIO 1: DataParallel")
print("=" * 70)

# 모델 만들기
model = SimpleModel()

# GPU가 여럿인지 확인
if torch.cuda.device_count() > 1:
    print(f"\nUsing {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
    model_wrapped = True
else:
    print("\nSingle GPU or CPU mode")
    model_wrapped = False

# 학습 모의실험
print("\nSimulating training...")

# DataParallel과 함께 저장하기
print("\n--- SAVING ---")

checkpoint_path = "dataparallel_checkpoint.pth"

if model_wrapped:
    # module의 상태 사전을 저장한다 ('module.' 접두사가 빠진다)
    state_dict = model.module.state_dict()
    print("Saving model.module.state_dict() (without 'module.' prefix)")
else:
    state_dict = model.state_dict()
    print("Saving model.state_dict()")

torch.save({
    'model_state_dict': state_dict,
    'wrapped': model_wrapped,
}, checkpoint_path)

print(f"Checkpoint saved to '{checkpoint_path}'")

# 처음 몇 개의 키 출력
print("\nFirst few state dict keys:")
for i, key in enumerate(list(state_dict.keys())[:3]):
    print(f"  {key}")

# DataParallel과 함께 불러오기
print("\n--- LOADING ---")

checkpoint = torch.load(checkpoint_path)

# 새 모델을 만든다
new_model = SimpleModel()

# 상태 사전 불러오기
new_model.load_state_dict(checkpoint['model_state_dict'])
print("State dict loaded into fresh model")

# 필요하면 DataParallel으로 감싸기
if checkpoint.get('wrapped', False) and torch.cuda.device_count() > 1:
    new_model = nn.DataParallel(new_model)
    print("Model wrapped with DataParallel")

# 뒷정리
import os
if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)


# ============================================================
# 'module.' 접두사 다루기
# ============================================================

print("\n" + "=" * 70)
print("SCENARIO 2: Handling 'module.' Prefix")
print("=" * 70)

# 'module.' 접두사가 붙은 상태 사전 모의실험
model = SimpleModel()
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

state_dict = model.state_dict() if torch.cuda.device_count() > 1 else model.state_dict()

print("\nOriginal state dict keys:")
for i, key in enumerate(list(state_dict.keys())[:3]):
    print(f"  {key}")

# 방법 1: 저장할 때 'module.' 접두사 없애기
from collections import OrderedDict

def remove_module_prefix(state_dict):
    """상태 사전의 키에서 'module.' 접두사를 없앤다"""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # 'module.' 접두사가 있으면 없앤다
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    return new_state_dict

clean_state_dict = remove_module_prefix(state_dict)

print("\nCleaned state dict keys:")
for i, key in enumerate(list(clean_state_dict.keys())[:3]):
    print(f"  {key}")

# 방법 2: 불러올 때 'module.' 접두사 붙이기
def add_module_prefix(state_dict):
    """상태 사전의 키에 'module.' 접두사를 붙인다"""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # 'module.' 접두사가 없으면 붙인다
        name = f'module.{k}' if not k.startswith('module.') else k
        new_state_dict[name] = v
    return new_state_dict

# 깨끗한 판본 저장
checkpoint_path = "clean_checkpoint.pth"
torch.save({
    'model_state_dict': clean_state_dict,
}, checkpoint_path)

print(f"\nClean checkpoint saved to '{checkpoint_path}'")

# 감싼 모델에도, 감싸지 않은 모델에도 불러올 수 있다
# 선택 1: 감싸지 않은 모델에 불러오기
new_model = SimpleModel()
checkpoint = torch.load(checkpoint_path)
new_model.load_state_dict(checkpoint['model_state_dict'])
print("Loaded into unwrapped model successfully")

# 선택 2: 감싼 모델에 불러오기 (접두사를 붙인다)
if torch.cuda.device_count() > 1:
    wrapped_model = nn.DataParallel(SimpleModel())
    prefixed_state = add_module_prefix(checkpoint['model_state_dict'])
    wrapped_model.load_state_dict(prefixed_state)
    print("Loaded into wrapped model successfully")

# 뒷정리
if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)


# ============================================================
# 좋은 관행
# ============================================================

print("\n" + "=" * 70)
print("BEST PRACTICES FOR DISTRIBUTED TRAINING")
print("=" * 70)

print("""
SAVING:
-------
1. DataParallel에서는 늘 model.module.state_dict()를 저장하라
   - 'module.' 접두사가 자동으로 없어진다
   - 체크포인트를 옮겨 쓰기 좋게 만든다

2. 감싸개에만 있는 키 없이 저장한다
   - 불러오기가 더 유연하다

3. 감싸기에 대한 메타데이터를 담는다
   - 모델을 올바로 다시 만드는 데 도움이 된다

Example:
    if isinstance(model, nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    
    torch.save({
        'model_state_dict': state_dict,
        'is_parallel': isinstance(model, nn.DataParallel),
    }, path)


LOADING:
--------
1. 먼저 기본 모델에 불러온다
   - 필요하면 그다음 감싼다
   - 과정을 더 잘 다룰 수 있다

2. 유연하게 하려면 strict=False를 쓰라
   - 빠졌거나 예상 밖인 키를 처리한다

3. 'module.' 접두사가 있으면 없앤다
   - 정리에는 도우미 함수를 쓴다

Example:
    model = SimpleModel()
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 필요하면 감싸기
    if use_multi_gpu:
        model = nn.DataParallel(model)


흔한 문제:
--------------
1. RuntimeError: 키가 빠졌거나 예상 밖의 키
   → 'module.' 접두사가 맞는지 확인하라
   → remove_module_prefix() 함수를 쓰라

2. DataParallel로 저장하고 없이 불러오기
   → 상태 사전에 'module.' 접두사가 있다
   → strict=False로 불러오거나 접두사를 정리하라

3. DataParallel 없이 저장하고 함께 불러오기
   → 'module.' 접두사를 붙여야 한다
   → 또는 model.module에 불러온다

4. GPU 개수가 다르면 체크포인트가 불러와지지 않음
   → 감싸개 없는 깨끗한 상태 사전을 저장하라
   → 불러온 뒤에 감싸개를 다시 만들라
""")

print("\n" + "=" * 70)
print("TUTORIAL COMPLETE")
print("=" * 70)

print("\nKey Takeaways:")
print("1. Save model.module.state_dict() for DataParallel")
print("2. Remove 'module.' prefix when saving")
print("3. Load into base model, then wrap")
print("4. Use helper functions for prefix handling")
print("5. Save metadata about model wrapping")


if __name__ == "__main__":
    pass
```

??? note "전체 출력 (120줄)"

    ```
    ======================================================================
    DISTRIBUTED TRAINING CHECKPOINTS
    ======================================================================

    ======================================================================
    SCENARIO 1: DataParallel
    ======================================================================

    Single GPU or CPU mode

    Simulating training...

    --- SAVING ---
    Saving model.state_dict()
    Checkpoint saved to 'dataparallel_checkpoint.pth'

    First few state dict keys:
      fc1.weight
      fc1.bias
      fc2.weight

    --- LOADING ---
    State dict loaded into fresh model

    ======================================================================
    SCENARIO 2: Handling 'module.' Prefix
    ======================================================================

    Original state dict keys:
      fc1.weight
      fc1.bias
      fc2.weight

    Cleaned state dict keys:
      fc1.weight
      fc1.bias
      fc2.weight

    Clean checkpoint saved to 'clean_checkpoint.pth'
    Loaded into unwrapped model successfully

    ======================================================================
    BEST PRACTICES FOR DISTRIBUTED TRAINING
    ======================================================================

    SAVING:
    -------
    1. DataParallel에서는 늘 model.module.state_dict()를 저장하라
       - 'module.' 접두사가 자동으로 없어진다
       - 체크포인트를 옮겨 쓰기 좋게 만든다

    2. 감싸개에만 있는 키 없이 저장한다
       - 불러오기가 더 유연하다

    3. 감싸기에 대한 메타데이터를 담는다
       - 모델을 올바로 다시 만드는 데 도움이 된다

    Example:
        if isinstance(model, nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        
        torch.save({
            'model_state_dict': state_dict,
            'is_parallel': isinstance(model, nn.DataParallel),
        }, path)


    LOADING:
    --------
    1. 먼저 기본 모델에 불러온다
       - 필요하면 그다음 감싼다
       - 과정을 더 잘 다룰 수 있다

    2. 유연하게 하려면 strict=False를 쓰라
       - 빠졌거나 예상 밖인 키를 처리한다

    3. 'module.' 접두사가 있으면 없앤다
       - 정리에는 도우미 함수를 쓴다

    Example:
        model = SimpleModel()
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 필요하면 감싸기
        if use_multi_gpu:
            model = nn.DataParallel(model)


    흔한 문제:
    --------------
    1. RuntimeError: 키가 빠졌거나 예상 밖의 키
       → 'module.' 접두사가 맞는지 확인하라
       → remove_module_prefix() 함수를 쓰라

    2. DataParallel로 저장하고 없이 불러오기
       → 상태 사전에 'module.' 접두사가 있다
       → strict=False로 불러오거나 접두사를 정리하라

    3. DataParallel 없이 저장하고 함께 불러오기
       → 'module.' 접두사를 붙여야 한다
       → 또는 model.module에 불러온다

    4. GPU 개수가 다르면 체크포인트가 불러와지지 않음
       → 감싸개 없는 깨끗한 상태 사전을 저장하라
       → 불러온 뒤에 감싸개를 다시 만들라


    ======================================================================
    TUTORIAL COMPLETE
    ======================================================================

    Key Takeaways:
    1. Save model.module.state_dict() for DataParallel
    2. Remove 'module.' prefix when saving
    3. Load into base model, then wrap
    4. Use helper functions for prefix handling
    5. Save metadata about model wrapping
    ```


## 2. 논의

감싼 모델의 state_dict에는 일반적으로 module. 접두사가 붙는다. 감싸기 전 모델에 복원하려면 내부 모델의 state_dict를 저장하거나 복원 대상과 키를 맞춘다. model.module은 실제로 감싸진 객체에서만 사용하며 일반 모델에 무조건 접근할 수는 없다.

remove_module_prefix는 시작 부분의 module.만 제거한다. 실제 하위 모듈 이름도 module인 경우나 중첩된 감싸기가 있으면 이름만 보고 무조건 변환하지 말고 저장 구조와 대상 키를 비교한다. strict=False만 사용하는 방식은 원하는 웨이트가 하나도 복원되지 않은 채 오류를 피할 수 있으므로 해결책이 아니다. 누락·추가 키와 실제로 불러온 항목을 확인해야 한다.

실제 DDP 체크포인트에서는 저장을 맡을 rank, 각 프로세스의 장치 매핑과 필요한 동기화도 정해야 한다. 이 코드는 DDP 프로세스 그룹이나 DistributedSampler를 구성해 학습하는 예제가 아니다. 접두사 없는 모델 웨이트가 있어도 옵티마이저·스케줄러·난수·샘플러 상태까지 자동으로 이전되는 것은 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
model.module.state_dict()와 model.state_dict()는 언제 키가 달라지는가?

</div>

??? success "풀이"
    DataParallel이나 DDP처럼 module 아래에 원래 모델을 둔 감싸개에서는 바깥 state_dict에 module. 접두사가 붙는다. 내부 모델의 state_dict를 저장하면 그 감싸개 접두사를 피할 수 있다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
접두사가 맞지 않을 때 strict=False만 사용하면 왜 위험한가?

</div>

??? success "풀이"
    모든 원래 키가 추가 키로, 대상 키가 누락 키로 처리되어 학습된 값이 실제로 복원되지 않을 수 있다. 키를 올바르게 정렬하고 반환된 불일치 정보를 확인해야 한다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
여러 DDP 프로세스가 같은 체크포인트 파일을 동시에 쓰는 것을 어떻게 피할 수 있는가?

</div>

??? success "풀이"
    지정한 rank에서만 저장하고, 다른 프로세스가 그 파일을 읽는 시점에는 필요한 동기화를 둔다. 복원 시 각 rank의 장치 매핑과 이어갈 학습 상태도 따로 처리한다.

## 정리하며

분산 감싸개와 내부 모델의 키 구조를 구분하면 웨이트를 이식하기 쉽다. 접두사 정리만으로 분산 학습의 저장·재개 전체가 해결되지는 않는다.
