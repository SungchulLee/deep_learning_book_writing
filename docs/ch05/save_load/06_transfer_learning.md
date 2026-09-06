# 전이 학습

전이 학습은 마지막 분류층을 바꾸고 필요하면 뼈대의 가중치를 얼려 미리 학습된 모델을 새 과제에 맞춘다. 이 스크립트는 미리 학습된 ResNet을 불러와 클래스 수에 맞게 고치고, 미세 조정한 검사점을 저장하고 불러오며, `strict=False`으로 상태 사전을 부분적으로 불러오는 법을 보인다.

## 코드

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
- 층 얼리고 녹이기
- 미세 조정한 모델 저장하기
- 상태 사전 부분적으로 불러오기
"""

import torch
import torch.nn as nn
import torchvision.models as models

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

# 매개변수의 수 확인
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# ============================================================
# 전이 학습을 위해 고치기
# ============================================================

print("\n" + "=" * 70)
print("MODIFYING MODEL FOR TRANSFER LEARNING")
print("=" * 70)

# 마지막 층의 입력 특징 수 얻기
num_features = model.fc.in_features
print(f"\nOriginal classifier input features: {num_features}")
print(f"Original classifier output classes: {model.fc.out_features}")

# 새 과제에 맞게 마지막 층 바꾸기 (예: 1000개 대신 10개 클래스)
num_classes = 10
model.fc = nn.Linear(num_features, num_classes)
print(f"\nNew classifier output classes: {num_classes}")

# ============================================================
# 층 얼리기
# ============================================================

print("\n" + "=" * 70)
print("FREEZING LAYERS")
print("=" * 70)

print("\nFreezing all layers except final classifier...")

# 모든 매개변수 얼리기
for param in model.parameters():
    param.requires_grad = False

# 마지막 층 녹이기
for param in model.fc.parameters():
    param.requires_grad = True

# 학습 가능한 매개변수 세기
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")
print(f"Frozen parameters: {total_params - trainable_params:,}")

# ============================================================
# 미세 조정한 모델 저장하기
# ============================================================

print("\n" + "=" * 70)
print("SAVING FINE-TUNED MODEL")
print("=" * 70)

# 학습 가능한 매개변수에만 최적화기 만들기
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

# 최적화기 다시 만들기
optimizer = torch.optim.Adam(model.parameters())
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
print("Optimizer state loaded")

# 필요하면 얼리기 적용
if checkpoint.get('frozen_layers', False):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    print("Layer freezing applied")

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

# 합성곱 층만 불러오기 (분류기는 제외)
pretrained_dict = {k: v for k, v in full_state.items() if 'fc' not in k}

# 현재 상태 얻기
model_dict = new_model.state_dict()

# 미리 학습된 가중치로 갱신
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

# 어긋남을 허용하려고 strict=False으로 불러오기
missing_keys, unexpected_keys = model.load_state_dict(
    saved_state,
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

## 논의

전이 학습은 미리 학습된 모델의 마지막 층을 원하는 클래스 수만큼 출력하도록 고친다. ResNet-18이라면 `model.fc = nn.Linear(512, num_classes)`으로 바꾸는 것이다. 뼈대의 매개변수는 `param.requires_grad = False`으로 얼리며, 10개 클래스 문제라면 학습 가능한 매개변수가 1100만 개에서 5,130개로 줄어든다.

상태 사전을 부분적으로 불러올 때에는 사전을 걸러 맞는 키만 복사한다. 'fc'를 포함하는 키를 빼면 미리 학습된 합성곱 가중치는 불러오면서 분류기는 무작위 초기화 상태로 남길 수 있다. `load_state_dict`의 `strict=False` 매개변수는 키가 모자라거나 남을 때 나는 오류를 눌러 준다.

층을 얼리는 일은 전부 아니면 전무가 아니다. 점진적으로 녹이는 방식, 즉 먼저 분류기만 학습시키고, 다음에 마지막 합성곱 블록을 녹여 미세 조정하고, 그다음에 더 많은 블록을 녹이는 방식이 처음부터 모든 층을 학습시키는 것보다 나은 결과를 낼 때가 많다. 특히 데이터셋이 작을 때 그렇다.

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

