# 기본적인 저장과 불러오기

모델을 저장하고 불러오는 일은 학습된 신경망을 배포하고 중단된 학습을 이어 가는 데 꼭 필요하다. PyTorch는 크게 두 가지 방법을 제공한다. 모델 전체를 저장하는 방법(간편하지만 깨지기 쉽다)과 상태 사전만 저장하는 방법(실전에서 권장)이다. 학습 검사점에는 제대로 이어 가려면 최적화기의 상태와 에포크 정보도 담아야 한다.

## 코드

```python
#!/usr/bin/env python3
"""
============================================================
PyTorch 모델의 저장과 불러오기 - 완전한 안내
============================================================

이 종합 튜토리얼은 PyTorch 모델을 저장하고 불러오는 데 꼭 필요한 방법을
좋은 관행과 흔한 함정과 함께 모두 다룬다.

핵심 개념:
1. torch.save() - pickle으로 파이썬 객체를 디스크에 저장한다
2. torch.load() - 저장한 객체를 디스크에서 불러온다
3. state_dict() - 모델의 매개변수를 담은 순서 있는 사전
4. load_state_dict() - 매개변수를 모델에 넣는다

출처: Patrick Loeber의 PyTorch 튜토리얼에 바탕을 둠
자세한 주석과 설명을 덧붙였다
"""

import torch
import torch.nn as nn
import os

# ============================================================
# 기억해 둘 핵심 메서드
# ============================================================

"""
꼭 알아야 할 세 함수:
--------------------------

1. torch.save(obj, path)
   - 어떤 파이썬 객체든 저장한다 (모델, 텐서, 사전 등)
   - 파이썬의 pickle 규약을 쓴다
   - 경로는 관례에 따라 .pt나 .pth로 끝내는 것이 좋다
   
2. torch.load(path)
   - 저장한 객체를 메모리로 다시 불러온다
   - 저장했던 바로 그 객체를 돌려준다
   - map_location 매개변수로 장치를 지정할 수 있다
   
3. model.load_state_dict(state_dict)
   - 상태 사전에서 모델의 매개변수를 불러온다
   - 더 유연하며 대부분의 경우에 권장된다
   - 모델의 구조를 따로 정의할 수 있게 해 준다
"""

# ============================================================
# 모델을 저장하는 두 가지 주된 방법
# ============================================================

"""
방법 1: 모델 전체 저장하기 (간편하지만 권장하지 않는다)
----------------------------------------------------------
장점:
  - 한 줄이면 된다
  - 시제품을 만들 때 빠르다
  
단점:
  - 파일이 더 크다
  - 덜 유연하다
  - 코드를 정리하면 깨질 수 있다
  - 불러올 때 모델 클래스를 쓸 수 있어야 한다
  - 실전에는 권장하지 않는다

사용법:
  torch.save(model, PATH)
  model = torch.load(PATH)
  model.eval()


방법 2: 상태 사전만 저장하기 (권장)
-----------------------------------------------
장점:
  - 파일이 더 작다 (가중치만 담고 구조는 담지 않는다)
  - 더 유연하고 옮겨 쓰기 좋다
  - 버전 관리에 더 알맞다
  - 다른 모델 구조에도 불러올 수 있다
  - 업계 표준이다
  
단점:
  - 모델 클래스를 따로 정의해야 한다
  - 코드가 조금 더 길다
  
사용법:
  torch.save(model.state_dict(), PATH)
  model = Model(*args, **kwargs)
  model.load_state_dict(torch.load(PATH))
  model.eval()
"""

# ============================================================
# 시연을 위한 간단한 모델 정의
# ============================================================

class Model(nn.Module):
    """
    시연을 위한 간단한 신경망.
    
    구조:
    - 선형층 하나
    - 시그모이드 활성화
    - 이진 분류에 알맞다
    
    인수:
        n_input_features (int): 입력 특징의 수
    """
    
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        # 선형층 하나 초기화
        # n_input_features -> 출력 1개로 보낸다 (이진 분류)
        self.linear = nn.Linear(n_input_features, 1)
    
    def forward(self, x):
        """
        신경망을 통과하는 순전파
        
        인수:
            x: 모양이 (batch_size, n_input_features)인 입력 텐서
            
        반환값:
            y_pred: 모양이 (batch_size, 1)인 출력 예측
                    시그모이드 활성화 덕분에 값이 0과 1 사이이다
        """
        # 선형 변환을 적용한 뒤 시그모이드
        # 시그모이드가 출력을 확률이 되도록 [0, 1]로 눌러 준다
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


# ============================================================
# 방법 1: 모델 전체를 저장하고 불러오기
# ============================================================

print("=" * 60)
print("METHOD 1: Saving and Loading Entire Model")
print("=" * 60)

# 모델 인스턴스 만들기
model = Model(n_input_features=6)

# 처음 모델 매개변수 보이기
print("\n📊 Initial Model Parameters:")
for name, param in model.named_parameters():
    print(f"{name}: shape {param.shape}")
    print(f"  Values: {param.data.flatten()[:5]}...")  # 처음 값 5개 보이기

# 실제로는 여기서 모델을 학습시킬 것이다
# 시연이므로 무작위로 초기화된 가중치를 쓴다
print("\n🔧 (In production: Train your model here)")

# 저장할 파일 경로 정하기
FILE = "model_complete.pth"

# 저장: 모델 전체 (구조 + 매개변수)
print(f"\n💾 Saving entire model to '{FILE}'...")
torch.save(model, FILE)
print(f"✅ Model saved successfully!")
print(f"   File size: {os.path.getsize(FILE) / 1024:.2f} KB")

# 불러오기: 모델 전체
print(f"\n📂 Loading model from '{FILE}'...")
loaded_model = torch.load(FILE)

# 중요: 모델을 평가 모드로 둔다
# 드롭아웃을 끄고 배치 정규화를 평가 모드로 바꾼다
loaded_model.eval()
print("✅ Model loaded successfully!")

# 매개변수가 같은지 확인
print("\n🔍 Verifying Loaded Model Parameters:")
for name, param in loaded_model.named_parameters():
    print(f"{name}: shape {param.shape}")
    print(f"  Values: {param.data.flatten()[:5]}...")

# 서로 맞는지 확인
params_match = all(
    torch.equal(p1, p2) 
    for p1, p2 in zip(model.parameters(), loaded_model.parameters())
)
print(f"\n✓ Parameters match: {params_match}")

# 뒷정리
if os.path.exists(FILE):
    os.remove(FILE)
    print(f"🗑️  Cleaned up '{FILE}'")


# ============================================================
# 방법 2: 상태 사전을 저장하고 불러오기 (권장)
# ============================================================

print("\n" + "=" * 60)
print("METHOD 2: Saving and Loading State Dict (RECOMMENDED)")
print("=" * 60)

# 새 모델 만들기
model = Model(n_input_features=6)

# 파일 경로 정하기
FILE = "model_state_dict.pth"

# 저장: 상태 사전만
print(f"\n💾 Saving model state dict to '{FILE}'...")
torch.save(model.state_dict(), FILE)
print(f"✅ State dict saved successfully!")
print(f"   File size: {os.path.getsize(FILE) / 1024:.2f} KB")

# 상태 사전에 무엇이 들었는지 보이기
print("\n📋 State Dict Contents:")
state_dict = model.state_dict()
for key, value in state_dict.items():
    print(f"  {key}: shape {value.shape}, dtype {value.dtype}")

# 불러오기: 상태 사전을 새 모델에
print(f"\n📂 Loading state dict from '{FILE}'...")

# 1단계: 모델 구조 만들기 (저장한 모델과 맞아야 한다)
loaded_model = Model(n_input_features=6)

# 2단계: 상태 사전 불러오기
# 참고: torch.load()은 사전을 돌려준다
# load_state_dict()이 그것을 모델에 넣는다
loaded_model.load_state_dict(torch.load(FILE))

# 3단계: 평가 모드로 두기
loaded_model.eval()
print("✅ State dict loaded successfully!")

# 불러온 상태 사전 확인
print("\n🔍 Verifying Loaded State Dict:")
loaded_state_dict = loaded_model.state_dict()
for key, value in loaded_state_dict.items():
    print(f"  {key}: shape {value.shape}")

# 서로 맞는지 확인
state_match = all(
    torch.equal(state_dict[key], loaded_state_dict[key])
    for key in state_dict.keys()
)
print(f"\n✓ State dicts match: {state_match}")

# 뒷정리
if os.path.exists(FILE):
    os.remove(FILE)
    print(f"🗑️  Cleaned up '{FILE}'")


# ============================================================
# 방법 3: 학습 검사점을 저장하고 불러오기
# ============================================================

print("\n" + "=" * 60)
print("METHOD 3: Saving and Loading Training Checkpoint")
print("=" * 60)

"""
검사점은 학습 상태 전체를 저장하여 다음을 할 수 있게 해 준다:
- 멈춘 곳에서 학습을 이어 간다
- 학습 중 가장 좋은 모델을 저장한다
- 죽거나 끊겨도 되살릴 수 있다

검사점에 흔히 담는 것:
1. 모델의 상태 사전
2. 최적화기의 상태 사전
3. 현재 에포크 번호
4. 학습 손실 이력
5. 학습률 일정의 상태
6. 난수 생성기의 상태 (재현성을 위해)
"""

# 모델과 최적화기 만들기
model = Model(n_input_features=6)
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 학습 진행 모의실험
print("\n🏋️  Simulating training progress...")
print(f"   Current epoch: 90")
print(f"   Learning rate: {learning_rate}")

# 최적화기의 상태 보이기
print("\n📊 Optimizer State Before Saving:")
print(f"   State dict keys: {list(optimizer.state_dict().keys())}")

# 빠짐없는 검사점 사전 만들기
checkpoint = {
    "epoch": 90,                          # 현재 에포크 번호
    "model_state": model.state_dict(),     # 모델의 매개변수
    "optim_state": optimizer.state_dict(), # 최적화기의 상태 (모멘텀 등)
    "loss": 0.123,                         # 선택: 현재 손실
    "accuracy": 0.95,                      # 선택: 현재 정확도
}

# 검사점 저장
FILE = "checkpoint.pth"
print(f"\n💾 Saving checkpoint to '{FILE}'...")
torch.save(checkpoint, FILE)
print(f"✅ Checkpoint saved successfully!")
print(f"   File size: {os.path.getsize(FILE) / 1024:.2f} KB")

print("\n📦 Checkpoint Contents:")
for key in checkpoint.keys():
    if isinstance(checkpoint[key], dict):
        print(f"  {key}: dict with {len(checkpoint[key])} items")
    else:
        print(f"  {key}: {checkpoint[key]}")

# 불러오기: 학습 상태 되살리기
print(f"\n📂 Loading checkpoint from '{FILE}'...")

# 1단계: 모델과 최적화기 다시 만들기
model = Model(n_input_features=6)
optimizer = torch.optim.SGD(model.parameters(), lr=0)  # 학습률이 되살아난다

# 2단계: 검사점 불러오기
checkpoint = torch.load(FILE)

# 3단계: 모델의 상태 되살리기
model.load_state_dict(checkpoint['model_state'])

# 4단계: 최적화기의 상태 되살리기
optimizer.load_state_dict(checkpoint['optim_state'])

# 5단계: 그 밖의 학습 매개변수 되살리기
epoch = checkpoint['epoch']
loss = checkpoint.get('loss', None)  # 선택적인 키에는 .get()을 쓴다
accuracy = checkpoint.get('accuracy', None)

print("✅ Checkpoint loaded successfully!")
print(f"\n📊 Restored Training State:")
print(f"   Epoch: {epoch}")
print(f"   Loss: {loss}")
print(f"   Accuracy: {accuracy}")
print(f"   Learning rate: {optimizer.param_groups[0]['lr']}")

# 중요: 알맞은 모드로 둔다
print("\n⚙️  Setting Model Mode:")
print("   For inference: model.eval()")
print("   For continued training: model.train()")

# 추론할 때
model.eval()
print("   Current mode: EVAL")

# 또는 학습을 이어 갈 때
# model.train()
# print("   Current mode: TRAIN")

# 뒷정리
if os.path.exists(FILE):
    os.remove(FILE)
    print(f"\n🗑️  Cleaned up '{FILE}'")


# ============================================================
# 장치에 따른 저장과 불러오기
# ============================================================

print("\n" + "=" * 60)
print("DEVICE-SPECIFIC SAVING AND LOADING")
print("=" * 60)

"""
GPU를 쓸 때에는 모델이 어디에 저장되고 어디로 불러와지는지
조심해야 한다. PyTorch는 흔한 모든 상황에 유연하게 대응한다.

"""

print("\n📱 Scenario 1: Save on GPU, Load on CPU")
print("-" * 60)
print("""
# GPU에서 학습
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)

# CPU에서 추론
device = torch.device('cpu')
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))
model.to(device)  # CPU에서는 꼭 필요하지는 않지만 좋은 습관이다
""")

print("\n🖥️  Scenario 2: Save on GPU, Load on GPU")
print("-" * 60)
print("""
# GPU에서 학습
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)

# 같은 GPU에서 추론
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))  # 같은 장치로 불러온다
model.to(device)

# 참고: 입력 텐서도 GPU로 옮겨야 한다!
# input_tensor = input_tensor.to(device)
""")

print("\n🔄 Scenario 3: Save on CPU, Load on GPU")
print("-" * 60)
print("""
# CPU에서 학습
torch.save(model.state_dict(), PATH)

# GPU에서 추론
device = torch.device("cuda")
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # GPU 지정
model.to(device)

# GPU 없는 로컬 기계에서 학습하고 GPU 서버에 배포할 때
# 흔히 있는 일이다
""")

print("\n🎯 Scenario 4: Multi-GPU Considerations")
print("-" * 60)
print("""
# 모델을 DataParallel이나 DistributedDataParallel으로 학습시켰다면:

# 저장: 'module.' 접두사가 있으면 없앤다
state_dict = model.state_dict()
if list(state_dict.keys())[0].startswith('module.'):
    # 'module.' 접두사 없애기
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # 'module.' 접두사 없애기
        new_state_dict[name] = v
    torch.save(new_state_dict, PATH)
else:
    torch.save(state_dict, PATH)

# 불러오기: 단일 GPU에서도 다중 GPU에서도 불러올 수 있다
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.to(device)

# 다중 GPU라면:
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
""")


# ============================================================
# 좋은 관행과 중요한 참고 사항
# ============================================================

print("\n" + "=" * 60)
print("BEST PRACTICES AND IMPORTANT NOTES")
print("=" * 60)

print("""
✅ DO's:
--------
1. Use .state_dict() approach for saving models in production
2. Always call model.eval() before inference
3. 데이터 손실을 막으려면 학습 중에 체크포인트를 저장하라
4. 버전과 에폭 정보를 담은 뜻 있는 파일 이름을 써라
   예: 'model_epoch50_acc0.95.pth'
5. 체크포인트와 함께 초매개변수를 저장하라
6. 다른 장치로 불러올 때는 map_location을 써라
7. 옛 체크포인트를 지우기 전에 불러온 모델이 도는지 확인하라

❌ DON'Ts:
----------
1. 실서비스 코드에서 모델 전체를 저장하지 마라
2. Don't forget to call model.eval() for inference
3. Don't mix up train() and eval() modes
4. 임시 텐서나 캐시 텐서를 체크포인트에 저장하지 마라
5. 불러올 때 장치가 같다고 가정하지 마라
6. 빠졌거나 예상 밖인 키에 대한 경고를 무시하지 마라

🔍 흔한 문제:
----------------
1. RuntimeError: 상태 사전 크기가 맞지 않음
   → 모델 구조가 저장된 상태 사전과 맞지 않는다
   
2. 불러올 때 CUDA 메모리 부족
   → Use map_location='cpu' to load to CPU first
   
3. 추론할 때와 학습할 때 동작이 다르다
   → Forgot to call model.eval()
   
4. 학습을 제대로 이어 갈 수 없다
   → 최적화기 상태를 저장하거나 불러오기를 잊었다
   
5. DataParallel의 모듈 접두사가 맞지 않음
   → 키의 'module.' 접두사를 처리해야 한다

📚 더 볼 자료:
-----------------------
- PyTorch Docs: https://pytorch.org/tutorials/beginner/saving_loading_models.html
- 체크포인트 관리: 고급 튜토리얼을 보라
- 모델 배포: ONNX와 TorchScript 튜토리얼을 보라
""")

print("\n" + "=" * 60)
print("TUTORIAL COMPLETE!")
print("=" * 60)
print("\n💡 Key Takeaways:")
print("   1. Use state_dict() for production models")
print("   2. Save checkpoints for training recovery")
print("   3. Always use model.eval() for inference")
print("   4. Be mindful of device placement (CPU/GPU)")
print("   5. Include training state in checkpoints")
print("\n📖 Next Steps:")
print("   - Check out advanced checkpoint management")
print("   - Learn about model versioning")
print("   - Explore ONNX export for deployment")
print("   - Study distributed training checkpoints")


if __name__ == "__main__":
    pass
```

## 논의

`torch.save(model, path)`으로 모델 전체를 저장하면 파이썬의 pickle로 모델의 구조와 매개변수를 함께 직렬화한다. 편리하지만 깨지기 쉽다. 불러올 때 모델 클래스를 임포트할 수 있어야 하고, 코드를 정리하다 보면(클래스 이름 바꾸기, 파일 옮기기) 저장한 모델을 못 쓰게 될 수 있다.

`torch.save(model.state_dict(), path)`으로 상태 사전만 저장하는 방법을 권한다. 매개변수와 구조를 떼어 놓기 때문이다. 불러올 때에는 먼저 모델 클래스를 다시 만든 뒤 `model.load_state_dict(torch.load(path))`을 부른다. 더 유연하고 파일도 작다.

학습 검사점에는 모델의 매개변수뿐 아니라 최적화기의 상태, 에포크 번호, 손실 이력, 스케줄러의 상태까지 담아야 한다. 최적화기의 상태가 없으면 모멘텀 버퍼와 적응형 학습률의 누적값이 사라져 학습을 이어 갈 때 품질이 떨어질 수 있다.

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

