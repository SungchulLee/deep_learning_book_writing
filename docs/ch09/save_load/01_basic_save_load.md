# 기본적인 저장과 불러오기

PyTorch 모델의 상태 사전, 전체 객체와 학습 체크포인트를 저장하고 복원하는 방법을 비교한다. 추론에 필요한 정보와 학습 재개에 필요한 상태를 구분하는 것이 핵심이다.

## 1. 코드

??? note "코드 (521줄)"

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
    3. state_dict() - 모델의 파라메터를 담은 순서 있는 사전
    4. load_state_dict() - 파라메터를 모델에 넣는다

    출처: Patrick Loeber의 PyTorch 튜토리얼에 바탕을 둠
    자세한 주석과 설명을 덧붙였다
    """

    import torch
    import torch.nn as nn
    import os
    # 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
    # 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
    torch.manual_seed(0)

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
       - map_location 파라메터로 장치를 지정할 수 있다

    3. model.load_state_dict(state_dict)
       - 상태 사전에서 모델의 파라메터를 불러온다
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
      - 파일이 더 작다 (웨이트만 담고 구조는 담지 않는다)
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
        시연을 위한 간단한 뉴럴 넷트웍.

        구조:
        - 선형 레이어 하나
        - 시그모이드 활성화
        - 이진 분류에 알맞다

        인수:
            n_input_features (int): 입력 피처의 수
        """

        def __init__(self, n_input_features):
            super(Model, self).__init__()
            # 선형 레이어 하나 초기화
            # n_input_features -> 출력 1개로 보낸다 (이진 분류)
            self.linear = nn.Linear(n_input_features, 1)

        def forward(self, x):
            """
            뉴럴 넷트웍을 통과하는 순전파

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

    # 처음 모델 파라메터 보이기
    print("\n📊 Initial Model Parameters:")
    for name, param in model.named_parameters():
        print(f"{name}: shape {param.shape}")
        print(f"  Values: {param.data.flatten()[:5]}...")  # 처음 값 5개 보이기

    # 실제로는 여기서 모델을 학습시킬 것이다
    # 시연이므로 무작위로 초기화된 웨이트를 쓴다
    print("\n🔧 (In production: Train your model here)")

    # 저장할 파일 경로 정하기
    FILE = "model_complete.pth"

    # 저장: 모델 전체 (구조 + 파라메터)
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

    # 파라메터가 같은지 확인
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
    4. 학습 로스 이력
    5. 러닝레이트 일정의 상태
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
        "model_state": model.state_dict(),     # 모델의 파라메터
        "optim_state": optimizer.state_dict(), # 최적화기의 상태 (모멘텀 등)
        "loss": 0.123,                         # 선택: 현재 로스
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
    optimizer = torch.optim.SGD(model.parameters(), lr=0)  # 러닝레이트가 되살아난다

    # 2단계: 검사점 불러오기
    checkpoint = torch.load(FILE)

    # 3단계: 모델의 상태 되살리기
    model.load_state_dict(checkpoint['model_state'])

    # 4단계: 최적화기의 상태 되살리기
    optimizer.load_state_dict(checkpoint['optim_state'])

    # 5단계: 그 밖의 학습 파라메터 되살리기
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
    1. 실서비스에서는 .state_dict() 방식으로 모델을 저장하라
    2. 추론 앞에서 늘 model.eval()을 부르라
    3. 데이터 로스를 막으려면 학습 중에 체크포인트를 저장하라
    4. 버전과 에폭 정보를 담은 뜻 있는 파일 이름을 써라
       예: 'model_epoch50_acc0.95.pth'
    5. 체크포인트와 함께 초파라메터를 저장하라
    6. 다른 장치로 불러올 때는 map_location을 써라
    7. 옛 체크포인트를 지우기 전에 불러온 모델이 도는지 확인하라

    ❌ DON'Ts:
    ----------
    1. 실서비스 코드에서 모델 전체를 저장하지 마라
    2. 추론할 때 model.eval() 부르기를 잊지 마라
    3. train() 모드와 eval() 모드를 헷갈리지 마라
    4. 임시 텐서나 캐시 텐서를 체크포인트에 저장하지 마라
    5. 불러올 때 장치가 같다고 가정하지 마라
    6. 빠졌거나 예상 밖인 키에 대한 경고를 무시하지 마라

    🔍 흔한 문제:
    ----------------
    1. RuntimeError: 상태 사전 크기가 맞지 않음
       → 모델 구조가 저장된 상태 사전과 맞지 않는다

    2. 불러올 때 CUDA 메모리 부족
       → 먼저 CPU로 불러오려면 map_location='cpu'를 쓰라

    3. 추론할 때와 학습할 때 동작이 다르다
       → model.eval()을 부르지 않았다

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


??? note "전체 출력 (232줄)"

    ```
    ============================================================
    METHOD 1: Saving and Loading Entire Model
    ============================================================

    📊 Initial Model Parameters:
    linear.weight: shape torch.Size([1, 6])
      Values: tensor([-0.0031,  0.2190, -0.3360, -0.3004, -0.1572])...
    linear.bias: shape torch.Size([1])
      Values: tensor([-0.0081])...

    🔧 (In production: Train your model here)

    💾 Saving entire model to 'model_complete.pth'...
    ✅ Model saved successfully!
       File size: 2.31 KB

    📂 Loading model from 'model_complete.pth'...
    ✅ Model loaded successfully!

    🔍 Verifying Loaded Model Parameters:
    linear.weight: shape torch.Size([1, 6])
      Values: tensor([-0.0031,  0.2190, -0.3360, -0.3004, -0.1572])...
    linear.bias: shape torch.Size([1])
      Values: tensor([-0.0081])...

    ✓ Parameters match: True
    🗑️  Cleaned up 'model_complete.pth'

    ============================================================
    METHOD 2: Saving and Loading State Dict (RECOMMENDED)
    ============================================================

    💾 Saving model state dict to 'model_state_dict.pth'...
    ✅ State dict saved successfully!
       File size: 1.58 KB

    📋 State Dict Contents:
      linear.weight: shape torch.Size([1, 6]), dtype torch.float32
      linear.bias: shape torch.Size([1]), dtype torch.float32

    📂 Loading state dict from 'model_state_dict.pth'...
    ✅ State dict loaded successfully!

    🔍 Verifying Loaded State Dict:
      linear.weight: shape torch.Size([1, 6])
      linear.bias: shape torch.Size([1])

    ✓ State dicts match: True
    🗑️  Cleaned up 'model_state_dict.pth'

    ============================================================
    METHOD 3: Saving and Loading Training Checkpoint
    ============================================================

    🏋️  Simulating training progress...
       Current epoch: 90
       Learning rate: 0.01

    📊 Optimizer State Before Saving:
       State dict keys: ['state', 'param_groups']

    💾 Saving checkpoint to 'checkpoint.pth'...
    ✅ Checkpoint saved successfully!
       File size: 1.85 KB

    📦 Checkpoint Contents:
      epoch: 90
      model_state: dict with 2 items
      optim_state: dict with 2 items
      loss: 0.123
      accuracy: 0.95

    📂 Loading checkpoint from 'checkpoint.pth'...
    ✅ Checkpoint loaded successfully!

    📊 Restored Training State:
       Epoch: 90
       Loss: 0.123
       Accuracy: 0.95
       Learning rate: 0.01

    ⚙️  Setting Model Mode:
       For inference: model.eval()
       For continued training: model.train()
       Current mode: EVAL

    🗑️  Cleaned up 'checkpoint.pth'

    ============================================================
    DEVICE-SPECIFIC SAVING AND LOADING
    ============================================================

    📱 Scenario 1: Save on GPU, Load on CPU
    ------------------------------------------------------------

    # GPU에서 학습
    device = torch.device("cuda")
    model.to(device)
    torch.save(model.state_dict(), PATH)

    # CPU에서 추론
    device = torch.device('cpu')
    model = Model(*args, **kwargs)
    model.load_state_dict(torch.load(PATH, map_location=device))
    model.to(device)  # CPU에서는 꼭 필요하지는 않지만 좋은 습관이다


    🖥️  Scenario 2: Save on GPU, Load on GPU
    ------------------------------------------------------------

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


    🔄 Scenario 3: Save on CPU, Load on GPU
    ------------------------------------------------------------

    # CPU에서 학습
    torch.save(model.state_dict(), PATH)

    # GPU에서 추론
    device = torch.device("cuda")
    model = Model(*args, **kwargs)
    model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # GPU 지정
    model.to(device)

    # GPU 없는 로컬 기계에서 학습하고 GPU 서버에 배포할 때
    # 흔히 있는 일이다


    🎯 Scenario 4: Multi-GPU Considerations
    ------------------------------------------------------------

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


    ============================================================
    BEST PRACTICES AND IMPORTANT NOTES
    ============================================================

    ✅ DO's:
    --------
    1. 실서비스에서는 .state_dict() 방식으로 모델을 저장하라
    2. 추론 앞에서 늘 model.eval()을 부르라
    3. 데이터 로스를 막으려면 학습 중에 체크포인트를 저장하라
    4. 버전과 에폭 정보를 담은 뜻 있는 파일 이름을 써라
       예: 'model_epoch50_acc0.95.pth'
    5. 체크포인트와 함께 초파라메터를 저장하라
    6. 다른 장치로 불러올 때는 map_location을 써라
    7. 옛 체크포인트를 지우기 전에 불러온 모델이 도는지 확인하라

    ❌ DON'Ts:
    ----------
    1. 실서비스 코드에서 모델 전체를 저장하지 마라
    2. 추론할 때 model.eval() 부르기를 잊지 마라
    3. train() 모드와 eval() 모드를 헷갈리지 마라
    4. 임시 텐서나 캐시 텐서를 체크포인트에 저장하지 마라
    5. 불러올 때 장치가 같다고 가정하지 마라
    6. 빠졌거나 예상 밖인 키에 대한 경고를 무시하지 마라

    🔍 흔한 문제:
    ----------------
    1. RuntimeError: 상태 사전 크기가 맞지 않음
       → 모델 구조가 저장된 상태 사전과 맞지 않는다
       
    2. 불러올 때 CUDA 메모리 부족
       → 먼저 CPU로 불러오려면 map_location='cpu'를 쓰라
       
    3. 추론할 때와 학습할 때 동작이 다르다
       → model.eval()을 부르지 않았다
       
    4. 학습을 제대로 이어 갈 수 없다
       → 최적화기 상태를 저장하거나 불러오기를 잊었다
       
    5. DataParallel의 모듈 접두사가 맞지 않음
       → 키의 'module.' 접두사를 처리해야 한다

    📚 더 볼 자료:
    -----------------------
    - PyTorch Docs: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    - 체크포인트 관리: 고급 튜토리얼을 보라
    - 모델 배포: ONNX와 TorchScript 튜토리얼을 보라


    ============================================================
    TUTORIAL COMPLETE!
    ============================================================

    💡 Key Takeaways:
       1. Use state_dict() for production models
       2. Save checkpoints for training recovery
       3. Always use model.eval() for inference
       4. Be mindful of device placement (CPU/GPU)
       5. Include training state in checkpoints

    📖 Next Steps:
       - Check out advanced checkpoint management
       - Learn about model versioning
       - Explore ONNX export for deployment
       - Study distributed training checkpoints
    ```
## 2. 논의

전체 모델 저장은 클래스 정의 자체를 독립적인 프로그램으로 묶는 방식이 아니다. pickle이 클래스의 모듈 경로 등에 의존하므로 복원 환경에서 호환되는 정의가 필요하다. 또한 PyTorch 2.6부터 pickle_module을 별도로 지정하지 않은 torch.load는 weights_only=True를 기본으로 사용한다. 이 페이지의 전체 객체를 인수 없이 읽는 예제는 이런 버전에서 그대로 동작하지 않을 수 있다. 전체 객체 복원이 필요할 때의 weights_only=False는 신뢰하는 파일에 한해서만 고려한다. [직렬화 문서](https://docs.pytorch.org/docs/stable/notes/serialization)를 함께 확인한다.

state_dict에는 파라메터와 영속 버퍼가 담기며 모델의 파이썬 구조나 입력 전처리는 포함하지 않는다. 복원할 모델을 먼저 만들고 키와 텐서 모양을 맞춰 load_state_dict에 전달한다. map_location은 저장된 텐서의 장치 매핑을 지정하며, 입력과 모델을 같은 장치로 옮기는 작업까지 대신하지는 않는다.

학습 재개에는 옵티마이저의 모멘텀·모멘트, 스케줄러, 혼합 정밀도 스케일러와 반복 위치 등이 추가로 필요할 수 있다. 정확한 경로 재현에는 난수 상태와 데이터 순서도 고려한다. state_dict()는 텐서 참조를 담는 얕은 복사이므로, 메모리에 최적 상태를 보관할 때는 deepcopy나 실제 직렬화가 필요하다. 평가 모드와 자동 미분 비활성화는 불러오기와 별도로 설정한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
state_dict에는 모델 구조도 들어 있는가?

</div>

??? success "연습문제 1 풀이"
    일반적인 nn.Module의 state_dict는 파라메터와 영속 버퍼를 담고 모델의 생성자·forward나 전처리 코드는 담지 않는다. 호환되는 구조를 먼저 만들어야 한다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
best_state=model.state_dict()만으로 최적 웨이트를 메모리에 보관하면 어떤 문제가 생기는가?

</div>

??? success "연습문제 2 풀이"
    사전에 담긴 텐서가 모델의 저장 공간을 참조하므로 이후 학습에 따라 값이 바뀔 수 있다. deepcopy로 독립 복사하거나 그 시점에 파일로 저장한다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
추론은 맞는데 학습 재개 후 경로가 달라진다면 어떤 상태를 점검해야 하는가?

</div>

??? success "연습문제 3 풀이"
    옵티마이저와 스케줄러 상태, 스케일러, 에포크·배치 위치, 난수 상태와 데이터 순서를 점검한다. 모델·입력의 장치, 자료형과 학습 모드도 확인한다.

## 정리하며

저장 형식은 복원 목적에 맞춰 선택한다. 구조와 전처리, 텐서 상태, 학습 상태의 저장 범위를 명시하고 사용 버전의 torch.load 동작을 확인해야 한다.
