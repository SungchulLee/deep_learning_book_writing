# 전이 학습

사전 학습 모델의 분류기 교체, 학습할 파라메터 선택과 상태 사전의 부분 복원을 설명한다. 예제의 실제 초기화와 학습 범위를 확인하며 전이 학습의 구성 방식을 읽는다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
============================================================
전이 학습: 미리 학습된 모델의 저장과 불러오기
============================================================

미리 학습된 모델을 다루고, 미세 조정하고, 일부만 저장하는
좋은 관행을 배운다.

주제:
- 미리 학습된 모델 불러오기
- 레이어 얼리고 녹이기
- 미세 조정한 모델 저장하기
- 상태 사전 부분적으로 불러오기
"""

import torch
import torch.nn as nn
import torchvision.models as models
# 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
# 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
torch.manual_seed(0)

print("=" * 70)
print("TRANSFER LEARNING SAVE/LOAD TUTORIAL")
print("=" * 70)

# ============================================================
# 미리 학습된 모델 불러오기
# ============================================================

print("\n" + "=" * 70)
print("LOADING PRE-TRAINED MODELS")
print("=" * 70)

print("\nLoading ResNet18 with pre-trained weights...")

# 미리 학습된 ResNet18 불러오기
model = models.resnet18(pretrained=False)  # 시연을 위해 False로 둔다
print("Model loaded")

# 파라메터의 수 확인
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# ============================================================
# 전이 학습을 위해 고치기
# ============================================================

print("\n" + "=" * 70)
print("MODIFYING MODEL FOR TRANSFER LEARNING")
print("=" * 70)

# 마지막 레이어의 입력 피처 수 얻기
num_features = model.fc.in_features
print(f"\nOriginal classifier input features: {num_features}")
print(f"Original classifier output classes: {model.fc.out_features}")

# 새 과제에 맞게 마지막 레이어 바꾸기 (예: 1000개 대신 10개 클래스)
num_classes = 10
model.fc = nn.Linear(num_features, num_classes)
print(f"\nNew classifier output classes: {num_classes}")

# ============================================================
# 레이어 얼리기
# ============================================================

print("\n" + "=" * 70)
print("FREEZING LAYERS")
print("=" * 70)

print("\nFreezing all layers except final classifier...")

# 모든 파라메터 얼리기
for param in model.parameters():
    param.requires_grad = False

# 마지막 레이어 녹이기
for param in model.fc.parameters():
    param.requires_grad = True

# 학습 가능한 파라메터 세기
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")
print(f"Frozen parameters: {total_params - trainable_params:,}")

# ============================================================
# 미세 조정한 모델 저장하기
# ============================================================

print("\n" + "=" * 70)
print("SAVING FINE-TUNED MODEL")
print("=" * 70)

# 학습 가능한 파라메터에만 최적화기 만들기
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
)

# 전이 학습 정보와 함께 검사점 저장
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'num_classes': num_classes,
    'frozen_layers': True,
    'base_model': 'resnet18',
}

filepath = "transfer_learning_checkpoint.pth"
torch.save(checkpoint, filepath)
print(f"\nCheckpoint saved to '{filepath}'")

import os
file_size = os.path.getsize(filepath) / (1024 * 1024)
print(f"File size: {file_size:.2f} MB")

# ============================================================
# 미세 조정한 모델 불러오기
# ============================================================

print("\n" + "=" * 70)
print("LOADING FINE-TUNED MODEL")
print("=" * 70)

# 체크포인트를 불러온다
checkpoint = torch.load(filepath)

# 같은 수정을 거쳐 모델 다시 만들기
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
num_classes = checkpoint['num_classes']
model.fc = nn.Linear(num_features, num_classes)

# 상태 사전 불러오기
model.load_state_dict(checkpoint['model_state_dict'])
print("\nModel state loaded")

# 얼리기를 먼저 되살린다.
# 저장할 때의 최적화기는 '학습할 파라메터'만 들고 있었으므로, 같은 배치로
# 다시 만들어야 상태를 받을 수 있다. 얼리기를 나중에 하면 배치 크기가 어긋나
# ValueError: loaded state dict contains a parameter group that doesn't match
# 가 난다.
if checkpoint.get('frozen_layers', False):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    print("Layer freezing applied")

# 최적화기 다시 만들기 — 저장할 때와 똑같이 학습할 파라메터만 준다
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters())
)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
print("Optimizer state loaded")

# ============================================================
# 상태 사전 부분적으로 불러오기
# ============================================================

print("\n" + "=" * 70)
print("PARTIAL STATE DICT LOADING")
print("=" * 70)

print("\nLoading only specific layers...")

# 원래 상태 저장
full_state = model.state_dict()

# 새 모델 만들기
new_model = models.resnet18(pretrained=False)
new_model.fc = nn.Linear(num_features, num_classes)

# 컨볼루션 레이어만 불러오기 (분류기는 제외)
pretrained_dict = {k: v for k, v in full_state.items() if 'fc' not in k}

# 현재 상태 얻기
model_dict = new_model.state_dict()

# 미리 학습된 웨이트로 갱신
model_dict.update(pretrained_dict)

# 새 상태 사전 불러오기
new_model.load_state_dict(model_dict)
print("Partial state dict loaded (excluding fc layer)")

# ============================================================
# 없거나 뜻밖의 키 다루기
# ============================================================

print("\n" + "=" * 70)
print("HANDLING MISSING/UNEXPECTED KEYS")
print("=" * 70)

# 상태 사전이 맞지 않을 때 다루는 법을 보인다
model = models.resnet18(pretrained=False)
saved_state = model.state_dict()

# 모델 고치기
model.fc = nn.Linear(512, 20)  # 클래스 수가 다르다

# strict=False가 봐주는 것은 '없는 열쇠'와 '남는 열쇠'뿐이다.
# 두 쪽에 다 있으면서 모양만 다른 열쇠(여기서는 fc.weight, fc.bias)는
# strict와 무관하게 오류를 낸다. 그러므로 모양이 맞는 것만 골라 넘긴다.
model_state = model.state_dict()
compatible = {
    k: v for k, v in saved_state.items()
    if k in model_state and v.shape == model_state[k].shape
}
dropped = [k for k in saved_state if k not in compatible]
print(f"\nShape-mismatched keys skipped: {dropped}")

missing_keys, unexpected_keys = model.load_state_dict(
    compatible,
    strict=False
)

print(f"\nMissing keys: {len(missing_keys)}")
if missing_keys:
    print(f"  {missing_keys}")

print(f"Unexpected keys: {len(unexpected_keys)}")
if unexpected_keys:
    print(f"  {unexpected_keys}")

print("\nModel loaded with partial matching")

# 뒷정리
if os.path.exists(filepath):
    os.remove(filepath)
    print(f"\nCleaned up '{filepath}'")

print("\n" + "=" * 70)
print("TUTORIAL COMPLETE")
print("=" * 70)

print("\nKey Takeaways:")
print("1. Use pre-trained models as feature extractors")
print("2. Freeze early layers, train final layers")
print("3. Save full model state including modifications")
print("4. Use strict=False for partial loading")
print("5. Filter optimizer parameters for frozen layers")


if __name__ == "__main__":
    pass
```

**출력:**

```
======================================================================
TRANSFER LEARNING SAVE/LOAD TUTORIAL
======================================================================

======================================================================
LOADING PRE-TRAINED MODELS
======================================================================

Loading ResNet18 with pre-trained weights...
Model loaded
Total parameters: 11,689,512

======================================================================
MODIFYING MODEL FOR TRANSFER LEARNING
======================================================================

Original classifier input features: 512
Original classifier output classes: 1000

New classifier output classes: 10

======================================================================
FREEZING LAYERS
======================================================================

Freezing all layers except final classifier...
Trainable parameters: 5,130
Frozen parameters: 11,684,382

======================================================================
SAVING FINE-TUNED MODEL
======================================================================

Checkpoint saved to 'transfer_learning_checkpoint.pth'
File size: 42.73 MB

======================================================================
LOADING FINE-TUNED MODEL
======================================================================

Model state loaded
Layer freezing applied
Optimizer state loaded

======================================================================
PARTIAL STATE DICT LOADING
======================================================================

Loading only specific layers...
Partial state dict loaded (excluding fc layer)

======================================================================
HANDLING MISSING/UNEXPECTED KEYS
======================================================================

Shape-mismatched keys skipped: ['fc.weight', 'fc.bias']

Missing keys: 2
  ['fc.weight', 'fc.bias']
Unexpected keys: 0

Model loaded with partial matching

Cleaned up 'transfer_learning_checkpoint.pth'

======================================================================
TUTORIAL COMPLETE
======================================================================

Key Takeaways:
1. Use pre-trained models as feature extractors
2. Freeze early layers, train final layers
3. Save full model state including modifications
4. Use strict=False for partial loading
5. Filter optimizer parameters for frozen layers
```

## 2. 논의

코드는 모든 ResNet-18 생성에서 pretrained=False를 사용한다. 따라서 출력 문구와 달리 사전 학습된 피처를 불러오지 않으며, 실제 데이터로 미세 조정하는 학습 루프도 없다. 전이 학습의 성능 실험이 아니라 분류기 교체·동결·저장과 복원 방식의 시연이다.

512차원 피처를 10개 클래스로 보내는 새 분류기에는 512×10+10=5130개의 학습 파라메터가 있다. 다른 파라메터의 requires_grad를 False로 해도 BatchNorm의 이동 통계 버퍼까지 동결되지는 않는다. 실제 학습에서는 백본의 모드와 전처리, 사용할 사전 학습 웨이트의 입력 규약도 함께 정해야 한다.

strict=False는 누락되거나 추가된 키를 허용하지만 같은 키의 텐서 모양 불일치를 무시하지 않는다. 코드처럼 호환되는 모양을 걸러 내고 반환된 누락·추가 키를 확인한다. 키에서 fc라는 부분 문자열을 제외하는 방법은 이 모델의 이름에 의존한다. 단계적으로 동결을 해제한다면 새 파라메터가 옵티마이저 그룹에 실제로 포함되는지도 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
이 코드가 사전 학습된 ResNet의 전이 성능을 보여 주는가?

</div>

??? success "풀이"
    아니다. pretrained=False로 무작위 초기화하며 실제 미세 조정 학습을 수행하지 않는다. 저장과 구조 변경의 시연으로 해석해야 한다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
strict=False이면 1000클래스 분류기 웨이트를 10클래스 분류기에 그대로 넣을 수 있는가?

</div>

??? success "풀이"
    아니다. 동일 키의 텐서 모양이 다르면 오류가 난다. 해당 키를 제거하거나 호환되는 항목만 걸러 부분 복원하고 누락 키를 확인한다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
requires_grad=False로 백본을 동결한 뒤에도 BatchNorm 상태가 달라질 수 있는가?

</div>

??? success "풀이"
    학습 모드에서는 이동 평균·분산 버퍼가 갱신될 수 있다. 그래디언트 동결과 모드 설정을 구분하고 의도한 통계 사용 방식에 맞춰 설정한다.

## 정리하며

전이 학습에서는 사전 학습 웨이트 사용 여부, 분류기 차원과 동결 범위를 명확히 한다. 부분 복원의 키 허용과 모양 호환성은 별도 조건이다.
