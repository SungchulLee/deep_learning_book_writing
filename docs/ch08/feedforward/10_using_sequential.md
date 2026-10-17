# Sequential 사용하기

`nn.Sequential`은 앞 모듈의 출력을 다음 모듈의 입력으로 전달한다. 이 예제는 기본 순차 모델, 이름을 붙인 모델, 재사용 블록을 조합한 모델을 비교한다.

## 1. 코드

??? note "코드 (383줄)"

    ```python
    from collections import OrderedDict
    """
    ================================================================================
    04_using_sequential.py - nn.Sequential로 모델 빠르게 만들기
    ================================================================================

    nn.Sequential은 사용자 정의 클래스를 쓰지 않고 모델을 빠르게 만드는 PyTorch의 방법이다.
    간단한 순방향 구조에 안성맞춤이다!

    nn.Sequential을 쓸 때:
        ✓ 간단한 순방향 뉴럴 넷트웍
        ✓ 빠른 시제품 만들기
        ✓ 선형 레이어를 쌓은 구조
        ✗ 복잡하게 갈라지는 구조
        ✗ 여러 입력/출력
        ✗ 사용자 정의 순전파 논리

    학습 목표:
        1. nn.Sequential로 모델 만들기
        2. 사용자 정의 nn.Module과의 절충 이해하기
        3. Sequential 블록 조립하는 법 배우기
        4. 여러 구조 양식으로 연습하기

    난이도: ⭐⭐☆☆☆ (초급~중급)
    소요 시간: 15~20분
    ================================================================================
    """

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    # 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
    # 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
    torch.manual_seed(0)

    # ================================================================================
    # 1부: 간단한 Sequential 모델
    # ================================================================================
    print("=" * 80)
    print("PART 1: Building a Simple Sequential Model")
    print("=" * 80)

    # PyTorch에서 뉴럴 넷트웍을 만드는 가장 간단한 방법이다!
    # 각 레이어가 지정한 순서대로 차례차례 적용된다
    simple_model = nn.Sequential(
        nn.Linear(784, 256),    # 입력 레이어: 784 → 256
        nn.ReLU(),              # 활성화
        nn.Linear(256, 128),    # 은닉 레이어: 256 → 128
        nn.ReLU(),              # 활성화
        nn.Linear(128, 10)      # 출력 레이어: 128 → 10
    )

    print("Simple Sequential Model:")
    print(simple_model)
    print(f"\nTotal parameters: {sum(p.numel() for p in simple_model.parameters()):,}")

    # ================================================================================
    # 2부: 이름 붙인 레이어를 쓰는 Sequential
    # ================================================================================
    print("\n" + "=" * 80)
    print("PART 2: Sequential with Named Layers")
    print("=" * 80)

    # 레이어에 이름을 붙이면 디버깅과 이해에 도움이 된다
    # 나중에 특정 레이어에 접근하고 싶을 때 편리하다
    # nn.Sequential은 모듈을 죽 늘어놓거나 OrderedDict 하나를 받는다.
    # ('이름', 모듈) 짝을 그냥 나열하면 TypeError가 난다.
    named_model = nn.Sequential(OrderedDict([
        ('flatten', nn.Flatten()),              # 입력 펼치기
        ('fc1', nn.Linear(784, 256)),           # 첫 완전 연결 레이어
        ('relu1', nn.ReLU()),                   # 첫 활성화
        ('dropout1', nn.Dropout(0.2)),          # 정칙화를 위한 드롭아웃
        ('fc2', nn.Linear(256, 128)),           # 둘째 완전 연결 레이어
        ('relu2', nn.ReLU()),                   # 둘째 활성화
        ('dropout2', nn.Dropout(0.2)),          # 드롭아웃 한 번 더
        ('fc3', nn.Linear(128, 10))             # 출력 레이어
    ]))

    print("Named Sequential Model:")
    for name, module in named_model.named_children():
        print(f"  {name}: {module}")

    # ================================================================================
    # 3부: 모듈식 Sequential (블록 조립하기)
    # ================================================================================
    print("\n" + "=" * 80)
    print("PART 3: Composing Sequential Blocks")
    print("=" * 80)

    # 다시 쓸 수 있는 구성 블록을 만들 수 있다!
    def make_fc_block(in_features, out_features, dropout=0.2):
        """
        완전 연결 블록 만들기: 선형 → ReLU → 드롭아웃

        흔히 쓰는 양식이다. 함수로 감싸 두면
        코드가 더 깔끔해지고 관리하기 쉬워진다.
        """
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    # 블록으로 모델 만들기
    modular_model = nn.Sequential(
        nn.Flatten(),                          # 28×28을 784로 펼치기
        make_fc_block(784, 512, dropout=0.3),  # 블록 1
        make_fc_block(512, 256, dropout=0.3),  # 블록 2
        make_fc_block(256, 128, dropout=0.2),  # 블록 3
        nn.Linear(128, 10)                     # 출력 (로짓이므로 활성화 없음)
    )

    print("Modular Sequential Model:")
    print(modular_model)
    print(f"\nNumber of layers: {len(list(modular_model.children()))}")

    # ================================================================================
    # 4부: MNIST로 학습하기
    # ================================================================================
    print("\n" + "=" * 80)
    print("PART 4: Training Sequential Model on MNIST")
    print("=" * 80)

    # 장치 준비
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # MNIST 불러오기
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, transform=transform
    )

    # num_workers 를 0으로 둔다. 1 이상이면 자료를 나르는 일을 딴 프로세스에
    # 맡기는데, 맥과 윈도에서는 그 프로세스를 **새로 띄우며**(spawn) 이 각본을
    # 처음부터 다시 읽는다. 로더를 만드는 아래 줄이 맨 바깥에 있으므로 새 프로세스가
    # 또 로더를 만들고, 그것이 또 프로세스를 띄워 끝내 죽는다. 1 이상을 쓰려면
    # 이 아래 모든 것을 `if __name__ == "__main__":` 안으로 넣어야 한다.
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=0
    )

    # 학습에는 간단한 모델을 쓴다
    model = simple_model.to(device)

    # 로스와 최적화기
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 학습 함수
    def train_epoch(model, loader, criterion, optimizer, device):
        """한 에폭을 학습한다."""
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in loader:
            # 이미지를 (batch, 1, 28, 28)에서 (batch, 784)로 펼친다
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)

            # 순전파
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 통계
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return total_loss / len(loader), 100 * correct / total

    # 평가 함수
    def evaluate(model, loader, criterion, device):
        """시험 집합에서 평가한다."""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.view(images.size(0), -1).to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return total_loss / len(loader), 100 * correct / total

    # 학습 루프
    print("\nTraining...")
    num_epochs = 5
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

    # ================================================================================
    # 5부: 구조 비교하기
    # ================================================================================
    print("\n" + "=" * 80)
    print("PART 5: Architecture Comparison")
    print("=" * 80)

    # 여러 모델을 비교해 보자
    models_dict = {
        'Simple (3 layers)': simple_model,
        'Named (with dropout)': named_model,
        'Modular (4 layers)': modular_model
    }

    print("\nModel Comparison:")
    print("-" * 80)
    print(f"{'Model':<25} {'Parameters':<15} {'Layers':<10}")
    print("-" * 80)
    for name, model in models_dict.items():
        params = sum(p.numel() for p in model.parameters())
        layers = len([m for m in model.modules() if not isinstance(m, nn.Sequential)])
        print(f"{name:<25} {params:<15,} {layers:<10}")
    print("-" * 80)

    # ================================================================================
    # 6부: Sequential 모델에 접근하고 고치기
    # ================================================================================
    print("\n" + "=" * 80)
    print("PART 6: Accessing Sequential Model Components")
    print("=" * 80)

    # 인덱스로 레이어에 접근할 수 있다
    print("First layer of simple_model:")
    print(simple_model[0])

    print("\nThird layer (second ReLU):")
    print(simple_model[3])

    # 레이어를 훑어볼 수 있다
    print("\nAll layers:")
    for idx, layer in enumerate(simple_model):
        print(f"  Layer {idx}: {layer.__class__.__name__}")

    # Sequential 모델을 잘라 낼 수 있다
    print("\nFirst 3 layers:")
    feature_extractor = simple_model[:3]  # 처음 세 레이어를 얻는다
    print(feature_extractor)

    # Sequential 모델을 고칠 수 있다
    print("\nModifying model by adding a new layer:")
    extended_model = nn.Sequential(
        *simple_model,  # 기존 레이어 펼치기
        nn.ReLU(),      # 활성화 하나 더 추가
        nn.Linear(10, 5)  # 마지막 레이어 추가
    )
    print(f"Original output size: 10")
    print(f"Extended output size: 5")

    # ================================================================================
    # 7부: 시각화
    # ================================================================================
    print("\n" + "=" * 80)
    print("PART 7: Visualizing Training Progress")
    print("=" * 80)

    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 5))

    # 로스를 그린다
    ax1.plot(train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(test_losses, 'r-', label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Test Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 정확도를 그린다
    ax2.plot(train_accs, 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(test_accs, 'r-', label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Test Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([90, 100])

    plt.tight_layout()
    plt.savefig('04_sequential_training.png', dpi=150, bbox_inches='tight')
    print("Training progress saved as '04_sequential_training.png'")
    plt.show()

    # ================================================================================
    # 핵심 정리
    # ================================================================================
    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
    1. nn.Sequential은 단순한 순전파 구조에 안성맞춤이다
       - 깔끔하고 읽기 쉬운 코드
       - 빠른 시제품 제작
       - 맞춤 nn.Module보다 군더더기가 적다

    2. Sequential을 쓰는 세 가지 방법:
       - 단순: 레이어를 순서대로 넘기기만 한다
       - 이름 붙이기: 디버깅을 돕도록 튜플을 쓴다
       - 모듈화: 다시 쓸 수 있는 블록을 조립한다

    3. Sequential의 한계:
       ✗ 여러 입력이나 출력을 다룰 수 없다
       ✗ 맞춤 순전파 논리를 넣을 수 없다
       ✗ 조건부 실행이 안 된다
       → 이런 경우에는 맞춤 nn.Module을 쓰라

    4. Sequential 모델은 다음과 온전히 호환된다.
       ✓ 모든 PyTorch 학습 API
       ✓ 모델 저장과 불러오기
       ✓ 전이 학습
       ✓ 모델 점검 도구

    5. Sequential 모델은 접근, 자르기, 고치기가 쉽다
       - 인덱스로 접근: model[0]
       - Slice: model[:3]
       - 순회: for layer in model

    언제 쓰는가:
      - 레이어를 단순히 쌓을 때는 Sequential을 쓴다
      - 복잡한 구조에는 맞춤 Module을 쓴다
    """)

    # ================================================================================
    # 학생을 위한 연습문제
    # ================================================================================
    print("=" * 80)
    print("EXERCISES TO TRY")
    print("=" * 80)
    print("""
    1. Sequential로 5~6레이어짜리 더 깊은 모델을 만들어 보라
    2. 레이어 사이에 배치 정규화를 넣어 보라
    3. 드롭아웃 비율을 달리하여 실험해 보라
    4. "넓은" 뉴럴 넷트웍(레이어마다 뉴런이 많다)과 "깊은" 뉴럴 넷트웍(레이어가 많다)을 만들어 견주어 보라
    5. 설정 목록으로 Sequential 모델을 만들어 내는 함수를 짜 보라
    6. 자르기로 중간 피처를 뽑아 보라
    7. 여러 엑티베이션 함수(LeakyReLU, ELU 등)를 써 보라
    8. Sequential 모델의 앙상블을 만들어 보라
    9. 첫 레이어의 웨이트를 시각화해 보라
    10. 레이어를 없애는 방식으로 모델 가지치기를 구현해 보라
    """)


    if __name__ == "__main__":
        pass
    ```


??? note "전체 출력 (160줄)"

    ```
    ================================================================================
    PART 1: Building a Simple Sequential Model
    ================================================================================
    Simple Sequential Model:
    Sequential(
      (0): Linear(in_features=784, out_features=256, bias=True)
      (1): ReLU()
      (2): Linear(in_features=256, out_features=128, bias=True)
      (3): ReLU()
      (4): Linear(in_features=128, out_features=10, bias=True)
    )

    Total parameters: 235,146

    ================================================================================
    PART 2: Sequential with Named Layers
    ================================================================================
    Named Sequential Model:
      flatten: Flatten(start_dim=1, end_dim=-1)
      fc1: Linear(in_features=784, out_features=256, bias=True)
      relu1: ReLU()
      dropout1: Dropout(p=0.2, inplace=False)
      fc2: Linear(in_features=256, out_features=128, bias=True)
      relu2: ReLU()
      dropout2: Dropout(p=0.2, inplace=False)
      fc3: Linear(in_features=128, out_features=10, bias=True)

    ================================================================================
    PART 3: Composing Sequential Blocks
    ================================================================================
    Modular Sequential Model:
    Sequential(
      (0): Flatten(start_dim=1, end_dim=-1)
      (1): Sequential(
        (0): Linear(in_features=784, out_features=512, bias=True)
        (1): ReLU()
        (2): Dropout(p=0.3, inplace=False)
      )
      (2): Sequential(
        (0): Linear(in_features=512, out_features=256, bias=True)
        (1): ReLU()
        (2): Dropout(p=0.3, inplace=False)
      )
      (3): Sequential(
        (0): Linear(in_features=256, out_features=128, bias=True)
        (1): ReLU()
        (2): Dropout(p=0.2, inplace=False)
      )
      (4): Linear(in_features=128, out_features=10, bias=True)
    )

    Number of layers: 5

    ================================================================================
    PART 4: Training Sequential Model on MNIST
    ================================================================================
    Using device: cpu

    Training...
    Epoch [1/5] | Train Loss: 0.3512, Acc: 90.35% | Test Loss: 0.1639, Acc: 94.98%
    Epoch [2/5] | Train Loss: 0.1340, Acc: 96.01% | Test Loss: 0.1066, Acc: 96.69%
    Epoch [3/5] | Train Loss: 0.0894, Acc: 97.28% | Test Loss: 0.0879, Acc: 97.10%
    Epoch [4/5] | Train Loss: 0.0658, Acc: 97.92% | Test Loss: 0.0862, Acc: 97.21%
    Epoch [5/5] | Train Loss: 0.0492, Acc: 98.50% | Test Loss: 0.0766, Acc: 97.55%

    ================================================================================
    PART 5: Architecture Comparison
    ================================================================================

    Model Comparison:
    --------------------------------------------------------------------------------
    Model                     Parameters      Layers    
    --------------------------------------------------------------------------------
    Simple (3 layers)         235,146         5         
    Named (with dropout)      235,146         8         
    Modular (4 layers)        567,434         11        
    --------------------------------------------------------------------------------

    ================================================================================
    PART 6: Accessing Sequential Model Components
    ================================================================================
    First layer of simple_model:
    Linear(in_features=784, out_features=256, bias=True)

    Third layer (second ReLU):
    ReLU()

    All layers:
      Layer 0: Linear
      Layer 1: ReLU
      Layer 2: Linear
      Layer 3: ReLU
      Layer 4: Linear

    First 3 layers:
    Sequential(
      (0): Linear(in_features=784, out_features=256, bias=True)
      (1): ReLU()
      (2): Linear(in_features=256, out_features=128, bias=True)
    )

    Modifying model by adding a new layer:
    Original output size: 10
    Extended output size: 5

    ================================================================================
    PART 7: Visualizing Training Progress
    ================================================================================
    Training progress saved as '04_sequential_training.png'

    ================================================================================
    KEY TAKEAWAYS
    ================================================================================

    1. nn.Sequential은 단순한 순전파 구조에 안성맞춤이다
       - 깔끔하고 읽기 쉬운 코드
       - 빠른 시제품 제작
       - 맞춤 nn.Module보다 군더더기가 적다

    2. Sequential을 쓰는 세 가지 방법:
       - 단순: 레이어를 순서대로 넘기기만 한다
       - 이름 붙이기: 디버깅을 돕도록 튜플을 쓴다
       - 모듈화: 다시 쓸 수 있는 블록을 조립한다

    3. Sequential의 한계:
       ✗ 여러 입력이나 출력을 다룰 수 없다
       ✗ 맞춤 순전파 논리를 넣을 수 없다
       ✗ 조건부 실행이 안 된다
       → 이런 경우에는 맞춤 nn.Module을 쓰라

    4. Sequential 모델은 다음과 온전히 호환된다.
       ✓ 모든 PyTorch 학습 API
       ✓ 모델 저장과 불러오기
       ✓ 전이 학습
       ✓ 모델 점검 도구

    5. Sequential 모델은 접근, 자르기, 고치기가 쉽다
       - 인덱스로 접근: model[0]
       - Slice: model[:3]
       - 순회: for layer in model

    언제 쓰는가:
      - 레이어를 단순히 쌓을 때는 Sequential을 쓴다
      - 복잡한 구조에는 맞춤 Module을 쓴다

    ================================================================================
    EXERCISES TO TRY
    ================================================================================

    1. Sequential로 5~6레이어짜리 더 깊은 모델을 만들어 보라
    2. 레이어 사이에 배치 정규화를 넣어 보라
    3. 드롭아웃 비율을 달리하여 실험해 보라
    4. "넓은" 뉴럴 넷트웍(레이어마다 뉴런이 많다)과 "깊은" 뉴럴 넷트웍(레이어가 많다)을 만들어 견주어 보라
    5. 설정 목록으로 Sequential 모델을 만들어 내는 함수를 짜 보라
    6. 자르기로 중간 피처를 뽑아 보라
    7. 여러 엑티베이션 함수(LeakyReLU, ELU 등)를 써 보라
    8. Sequential 모델의 앙상블을 만들어 보라
    9. 첫 레이어의 웨이트를 시각화해 보라
    10. 레이어를 없애는 방식으로 모델 가지치기를 구현해 보라

    ```


## 2. 논의

`simple_model`, `named_model`, `modular_model`은 구성 방식과 너비·드롭아웃이 서로 다른 모델이다. 이름을 붙이거나 블록을 재사용한다고 계산이 자동으로 같아지는 것은 아니다. 동일성을 비교하려면 레이어와 상태를 맞춰야 한다.

평탄화가 모델 안에 있는지 학습 루프에 있는지 확인한다. `nn.Linear`는 마지막 축에 작용하므로 28×28 이미지를 784차원으로 바꾸는 책임이 어디에 있는지가 중요하다.

순차 모델은 단일 경로에 편리하다. 분기와 합산, 여러 출력을 다루려면 사용자 정의 `forward`가 더 명확할 수 있다. 이 예제는 매 에포크 테스트 지표를 확인하므로 그 결과로 설정을 조정할 때는 별도의 검증셋을 마련한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
경사를 초기화하지 않으면 현재 갱신이 어떻게 달라지는가?

</div>

??? success "풀이"
    이전 경사에 현재 경사가 더해진다. 파라메터가 매번 바뀌므로 누적값이 현재 경사에 단계 수를 곱한 값과 같지는 않다. 의도적인 누적이라면 로스의 스케일과 갱신 주기를 함께 설계한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
기본 모델과 이름을 붙인 모델의 파라메터 수가 같아도 출력이 달라질 수 있는 이유를 설명하라.

</div>

??? success "풀이"
    독립적으로 초기화되어 웨이트가 다르고, 이름을 붙인 모델에는 드롭아웃이 있다. 평가 모드에서 같은 웨이트를 옮기고 같은 전처리를 적용해야 해당 차이를 분리해 비교할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
기존 모델에 새 레이어를 붙일 때 모듈 공유에 주의해야 하는 이유를 설명하라.

</div>

??? success "풀이"
    기존 모듈 객체를 새 Sequential에 넣으면 같은 웨이트를 공유할 수 있다. 새 모델을 학습할 때 원래 모델의 상태도 바뀔 수 있으므로 독립 비교에는 별도 모델과 복사한 상태를 사용한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
조기 종료를 구현하라. 매 에포크 후 검증 로스를 추적하고, 10 에포크 연속으로 개선이 없으면 학습을 멈춘다. 가장 좋은 모델 웨이트를 저장하고 복원하라.

</div>

??? success "풀이"
    인내 횟수 카운터와 최저 로스 추적기를 추가한다.
    ```python
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    for epoch in range(num_epochs):
        # ... 학습 단계 ...
        val_loss = evaluate(model, val_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        if patience_counter >= 10:
            print(f'Early stopping at epoch {epoch}')
            model.load_state_dict(best_state)
            break
    ```
    `state_dict().copy()`는 얕은 복사라 텐서 저장 공간이 계속 갱신될 수 있으므로 최선 상태의 보존을 보장하지 않는다. 실제 구현에서는 상태를 깊은 복사하거나 파일로 저장한다. 위의 `evaluate` 호출도 본문의 함수 인수와 반환값에 맞춰야 한다. 검증 선택과 독립 테스트 평가를 구분한다.

## 정리하며

`Sequential`은 실행 순서를 간결하게 표현한다. 입력 평탄화, 레이어의 차원과 모듈 공유 여부를 확인하고, 분기나 복수 출력에는 사용자 정의 모듈을 고려한다.
