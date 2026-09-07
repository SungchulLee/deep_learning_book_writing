# 분산 학습

DataParallel이나 DistributedDataParallel로 분산 학습을 하면 상태 사전의 키에 `module.` 접두사가 붙는다. 검사점을 제대로 저장하고 불러오려면 이 접두사를 다루어 단일 GPU 설정과 다중 GPU 설정 사이를 오갈 수 있게 해야 한다. 이 스크립트는 분산 검사점 관리의 좋은 관행을 보인다.

## 코드

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

## 논의

DataParallel은 모델을 감싸 배치를 여러 GPU에 나누어 주지만, 상태 사전의 모든 키에 `module.` 접두사를 붙인다. DataParallel 모델에서 `model.state_dict()`을 저장한 뒤 감싸지 않은 모델에 불러오면 키가 맞지 않는다는 오류가 난다. 해결책은 언제나 `model.module.state_dict()`을 저장하는 것이다.

`remove_module_prefix` 도우미 함수는 모든 키에서 `module.` 접두사를 떼어 내어 단일 GPU 설정과 다중 GPU 설정 사이에서 검사점을 옮겨 쓸 수 있게 한다. 반대 함수인 `add_module_prefix`은 단일 GPU 검사점을 감싼 모델에 불러오는 드문 경우를 처리한다.

언제나 접두사 없는 깨끗한 상태 사전을 저장하고, 불러온 뒤에 DataParallel/DistributedDataParallel 감싸개를 다시 씌우는 것이 좋은 관행이다. 그러면 불러올 때 GPU가 몇 대이든 검사점을 쓸 수 있다.

## 연습문제

**연습문제 1.**
코드를 따라가며 쓰인 주요 자료 구조를 찾아라. 각각에 대해 자료형, (해당한다면) 모양, 파이프라인에서의 구실을 적어라.

??? success "연습문제 1 풀이"
    코드를 꼼꼼히 읽으며 변수 대입마다 살펴본다. 텐서는 `.shape`과 `.dtype`을 확인하고, 클래스는 `__init__`의 매개변수와 `forward`/`__call__`의 서명을 확인한다. 이름, 자료형, 모양, 구실을 열로 하는 표에 정리한다.

---


**연습문제 2.**
오류 처리와 입력 검증을 넣도록 코드를 고쳐라. 이 코드를 실전에 쓸 수 있게 하려면 어떤 검사를 더하겠는가?

??? success "연습문제 2 풀이"
    입력에 자료형 검사(`isinstance`), 모양 검증(`assert tensor.dim() == expected`), 값 범위 검사(예: 확률이 [0,1] 안인지)를 넣고, 입출력 연산은 try-except로 감싼다. 빈 배치나 NaN 같은 경계 상황에는 경고를 남긴다. 매개변수와 반환값의 자료형을 적은 독스트링을 붙인다.

---


**연습문제 3.**
직접 고른 새로운 쓰임새를 지원하도록 코드를 확장하라. 무엇을 왜 바꿀지 설명하라.

??? success "연습문제 3 풀이"
    알맞은 확장을 하나 고른다(예: 다른 데이터셋, 지표 추가, 새 모델 변형). 필요한 변경을 설명한다. 새 임포트, 클래스 정의 수정, 초매개변수 갱신, 새로운 시각화나 기록 등이다. 핵심 변경을 구현하고 간단한 시험으로 올바름을 확인한다.

