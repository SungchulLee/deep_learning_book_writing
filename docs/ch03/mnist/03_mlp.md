# 다층 퍼셉트론

[2단계](../linear_softmax/06_implementation.md)의 선형 모델은 92.51%에서 멈췄다. 결정 경계가 선형이라는 제약 때문이다. 자연스러운 다음 수는 층을 하나 더 쌓는 것이다. 784 → 128 → 10으로 가면 매개변수가 7850개에서 10만 개 남짓으로 늘어나니, 표현력도 그만큼 늘 것 같다.

그런데 층만 쌓아서는 **아무것도 얻지 못한다.** 이 절은 먼저 그 사실을 보이고, 무엇을 더해야 하는지를 밝힌다.

## 1. 왜 활성화 함수가 필요한가

### 선형 변환을 겹쳐도 선형 변환이다

층을 두 개 쌓되 그 사이에 아무것도 넣지 않으면 이렇게 된다.

$$
y = (x W_1 + b_1) W_2 + b_2
$$

괄호를 풀어 정리해 보자.

$$
y = x (W_1 W_2) + (b_1 W_2 + b_2)
$$

$W_1 W_2$은 그냥 하나의 행렬이고 $b_1 W_2 + b_2$은 그냥 하나의 벡터이다. 이를 각각 $W'$과 $b'$이라 두면 다음과 같다.

$$
y = x W' + b'
$$

**층 두 개짜리 신경망이 층 하나짜리 신경망과 정확히 같아졌다.** 곧 2단계의 모델과 표현력이 완전히 동일하다. 층을 백 개 쌓아도 마찬가지다. 선형 변환을 아무리 겹쳐도 그 합성은 여전히 선형 변환이다. $\square$

매개변수 수를 세어 보면 이 사실이 더 또렷해진다. 784 → 128 → 10에 저장되는 수는 101,770개이지만, 위 식이 말하듯 실제로 쓰이는 자유도는 $W' \in \mathbb{R}^{784 \times 10}$과 $b' \in \mathbb{R}^{10}$, 곧 **7850개**뿐이다. 2단계와 정확히 같은 수이다. 나머지 9만여 개는 서로를 상쇄하며 아무 일도 하지 않는다.

### 실제로 확인해 보기

말로만 볼 것이 아니라 재어 보자. 같은 784 → 128 → 10 신경망을 ReLU만 넣고 빼서 5 에포크씩 학습시킨 결과이다.

| 구조 | 저장된 매개변수 | 실효 자유도 | 시험 정확도 |
|---|---|---|---|
| 784 → 128 → 10, 활성화 **없음** | 101,770 | 7,850 | **91.58%** |
| 784 → 128 → 10, ReLU **있음** | 101,770 | 101,770 | **97.53%** |

활성화가 없으면 91.58%로, 2단계의 선형 모델(92.51%)과 사실상 같은 자리에 머문다. 매개변수를 13배 저장하고 학습에 그만큼 시간을 쓰고도 얻은 것이 없다.

학습이 끝난 무활성화 모델의 두 가중치 행렬을 실제로 곱해 $W' = W_1 W_2$을 만들고, 그 하나의 아핀 변환과 원래 2층 신경망의 출력을 견주면 최대 오차가 $7.6 \times 10^{-6}$이다. 부동소수점 오차 수준이며, 두 모델이 같은 함수라는 뜻이다.

### ReLU가 하는 일

$$
\mathrm{ReLU}(z) = \max(0, z)
$$

음수를 0으로 자르는 것이 전부이다. 이 단순한 꺾임 하나가 위의 상쇄를 깨뜨린다. $\mathrm{ReLU}(xW_1 + b_1)W_2$는 어떤 $W'$으로도 다시 쓸 수 없다.

기하로 보면 이렇다. 은닉 뉴런 하나하나가 입력 공간을 직선으로 가르고, ReLU가 그 한쪽을 0으로 눌러 조각을 만든다. 은닉 뉴런이 128개이면 입력 공간이 여러 조각으로 나뉘고, 조각마다 다른 선형 함수가 적용된다. 전체로 보면 여러 개의 선형 조각을 이어 붙인 함수가 되어, 곡선 모양의 결정 경계를 그릴 수 있다.

MNIST에서 이것이 중요한 까닭은 한 숫자를 쓰는 방식이 여럿이기 때문이다. 가로줄이 있는 `7`과 없는 `7`은 화소 공간에서 서로 멀리 떨어져 있어, 하나의 선형 경계로는 둘 다 `7`쪽에 두기 어렵다. 은닉 뉴런이 여럿이면 서로 다른 필체를 각기 다른 뉴런이 맡을 수 있다.

ReLU 말고도 시그모이드, tanh, GELU 등 여러 활성화 함수가 있고 저마다 성질이 다르다. 그 비교와 선택 기준은 [6장의 활성화 함수](../../ch06/index.md)에서 다룬다. 여기서 중요한 것은 **무엇을 쓰느냐가 아니라 반드시 있어야 한다는 것**이다.

---

## 2. 코드


```python
"""
================================================================================
03_mnist_basic.py - 완전한 MNIST 숫자 분류기
================================================================================

이 예제는 손글씨 숫자(0~9)로 이루어진 유명한 MNIST 데이터셋을 써서
완전한 이미지 분류 파이프라인을 구현한다.

데이터셋: MNIST
    - 학습 이미지 60,000장
    - 시험 이미지 10,000장
    - 28×28 화소 회색조 이미지
    - 클래스 10개 (숫자 0~9)

구조:
    입력 (784) → ReLU를 쓰는 은닉 (128) → 소프트맥스를 쓰는 출력 (10)

이것이 첫 실전 딥러닝 과제이다!

학습 목표:
    1. 실제 데이터셋을 불러오고 전처리하기
    2. 완전한 학습 파이프라인 만들기
    3. 알맞은 학습/시험 분할 구현하기
    4. 모델 성능 평가하기
    5. GPU 가속 쓰기
    6. 예측 시각화하기

난이도: ⭐⭐⭐☆☆ (초급~중급)
소요 시간: 30~45분
================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# ================================================================================
# 1부: 설정과 장치 준비
# ================================================================================
print("=" * 80)
print("STEP 1: Configuration and Device Setup")
print("=" * 80)

# 재현성을 위해 난수 씨앗 고정
# 이렇게 하면 실행할 때마다 같은 결과가 나온다
torch.manual_seed(42)
np.random.seed(42)

# 장치 설정
# PyTorch는 CPU에서도 GPU(CUDA)에서도 돌 수 있다
# GPU를 쓰면 학습이 크게 빨라진다 (10~100배)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# 초매개변수
# 이들이 학습 과정과 모델 구조를 좌우한다
config = {
    'input_size': 784,        # 28×28 = 펼친 화소 784개
    'hidden_size': 128,       # 은닉층의 뉴런 수
    'num_classes': 10,        # 숫자 0~9
    'num_epochs': 5,          # 데이터셋 전체를 몇 번 볼지
    'batch_size': 100,        # 학습 단계마다의 표본 수
    'learning_rate': 0.001,   # 최적화기의 걸음 크기
}

print(f"\nHyperparameters:")
for key, value in config.items():
    print(f"  {key:15s}: {value}")

# ================================================================================
# 2부: 데이터 불러오기와 전처리
# ================================================================================
print("\n" + "=" * 80)
print("STEP 2: Loading MNIST Dataset")
print("=" * 80)

# 변환: PIL 이미지를 PyTorch 텐서로 바꾼다
# ToTensor()는 화소값을 [0, 255]에서 [0, 1]로 자동 조정한다
transform = transforms.Compose([
    transforms.ToTensor(),  # 텐서로 바꾸고 [0, 1]로 조정
])

# 학습 데이터 내려받아 불러오기
# 데이터가 없으면 './data'에 자동으로 내려받는다
print("Loading training data...")
train_dataset = torchvision.datasets.MNIST(
    root='./data',           # 데이터를 저장할 곳
    train=True,              # 학습 분할 불러오기
    transform=transform,     # 변환 적용
    download=True            # 없으면 내려받기
)

# 시험 데이터 불러오기
print("Loading test data...")
test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,             # 시험 분할 불러오기
    transform=transform,
    download=True
)

print(f"\nDataset Statistics:")
print(f"  Training samples: {len(train_dataset)}")
print(f"  Test samples: {len(test_dataset)}")
print(f"  Image shape: {train_dataset[0][0].shape}")  # (채널, 높이, 너비)
print(f"  Number of classes: {len(train_dataset.classes)}")

# 데이터 로더 만들기
# DataLoader가 배치 묶기, 섞기, 병렬 적재를 처리한다
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,            # 에포크마다 학습 데이터 섞기
    num_workers=2,           # 데이터 적재에 하위 프로세스 2개 쓰기
    pin_memory=True          # CPU-GPU 전송 속도 높이기
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=config['batch_size'],
    shuffle=False,           # 시험 데이터는 섞지 않는다
    num_workers=2,
    pin_memory=True
)

print(f"\nDataLoader Info:")
print(f"  Training batches: {len(train_loader)}")
print(f"  Test batches: {len(test_loader)}")

# ================================================================================
# 3부: 표본 데이터 시각화
# ================================================================================
print("\n" + "=" * 80)
print("STEP 3: Visualizing Sample Images")
print("=" * 80)

# 시험 이미지 배치 하나 가져오기
examples = iter(test_loader)
example_data, example_labels = next(examples)

# 표본 이미지 12장 그리기
fig, axes = plt.subplots(2, 6, figsize=(12, 4))
for i, ax in enumerate(axes.flat):
    # 그리기 위해 (1, 28, 28)을 (28, 28)로 바꾼다
    image = example_data[i].squeeze()
    ax.imshow(image, cmap='gray')
    ax.set_title(f'Label: {example_labels[i]}')
    ax.axis('off')

plt.tight_layout()
plt.savefig('03_mnist_samples.png', dpi=150, bbox_inches='tight')
print("Sample images saved as '03_mnist_samples.png'")
plt.close()

# ================================================================================
# 4부: 신경망 정의
# ================================================================================
print("\n" + "=" * 80)
print("STEP 4: Building the Neural Network")
print("=" * 80)

class MNISTClassifier(nn.Module):
    """
    MNIST 분류를 위한 순방향 신경망.
    
    구조:
        입력 (784) → ReLU를 쓰는 은닉 (128) → 출력 (10)
    
    참고: CrossEntropyLoss가 내부에서 소프트맥스를 적용하므로
    여기서는 쓰지 않는다(그 편이 수치적으로 더 안정하다).
    """
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(MNISTClassifier, self).__init__()
        
        # 1층: 입력 → 은닉
        # 784 → 128 변환
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        # ReLU 활성화
        # 비선형성을 넣어 복잡한 양상을 배울 수 있게 한다
        self.relu = nn.ReLU()
        
        # 2층: 은닉 → 출력
        # 128 → 10 변환 (숫자 클래스마다 출력 하나)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        """
        신경망을 통과하는 순전파.
        
        인수:
            x: 모양이 (batch_size, 1, 28, 28)인 입력 텐서
        
        반환값:
            모양이 (batch_size, 10)인 출력 로짓
        """
        # 이미지 펼치기
        # (batch_size, 1, 28, 28)에서 (batch_size, 784)로
        # -1은 "이 차원은 알아서 정하라"는 뜻이다
        x = x.reshape(x.size(0), -1)
        
        # 1층: 선형 → ReLU
        hidden = self.fc1(x)           # (batch_size, 128)
        hidden = self.relu(hidden)     # (batch_size, 128)
        
        # 2층: 선형 (활성화 없음 - CrossEntropyLoss는 로짓을 받는다)
        output = self.fc2(hidden)      # (batch_size, 10)
        
        return output
    
    def predict(self, x):
        """
        예측하기 (로짓이 아니라 클래스 레이블을 돌려준다).
        
        인수:
            x: 모양이 (batch_size, 1, 28, 28)인 입력 텐서
        
        반환값:
            모양이 (batch_size,)인 예측 클래스 레이블
        """
        logits = self.forward(x)
        # torch.max는 (값, 인덱스)를 돌려준다
        # 우리는 인덱스(확률이 가장 높은 클래스)가 필요하다
        _, predicted = torch.max(logits, dim=1)
        return predicted

# 모델을 만들어 장치로 옮기기
model = MNISTClassifier(
    config['input_size'],
    config['hidden_size'],
    config['num_classes']
).to(device)

# 매개변수 세기
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model: MNISTClassifier")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Parameters breakdown:")
print(f"    Layer 1: {config['input_size']} × {config['hidden_size']} + {config['hidden_size']} = {config['input_size'] * config['hidden_size'] + config['hidden_size']:,}")
print(f"    Layer 2: {config['hidden_size']} × {config['num_classes']} + {config['num_classes']} = {config['hidden_size'] * config['num_classes'] + config['num_classes']:,}")

# ================================================================================
# 5부: 손실과 최적화기 정의
# ================================================================================
print("\n" + "=" * 80)
print("STEP 5: Setting Up Training Components")
print("=" * 80)

# 손실 함수: 교차 엔트로피 손실
# 다중 클래스 분류에 안성맞춤이다
# LogSoftmax와 NLLLoss를 한 단계로 합친다
# 날것의 로짓을 받는다 (소프트맥스를 적용하지 않는다)
criterion = nn.CrossEntropyLoss()

# 최적화기: Adam
# 적응형 학습률 최적화기
# 대부분의 문제에서 별다른 손질 없이 잘 통한다
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

print(f"Loss function: CrossEntropyLoss")
print(f"Optimizer: Adam")
print(f"Learning rate: {config['learning_rate']}")

# ================================================================================
# 6부: 학습 루프
# ================================================================================
print("\n" + "=" * 80)
print("STEP 6: Training the Model")
print("=" * 80)

# 학습 기록
train_losses = []
train_accuracies = []

# 전체 단계 수
total_steps = len(train_loader)

print(f"\nStarting training for {config['num_epochs']} epochs...")
print(f"Steps per epoch: {total_steps}")
print("-" * 80)

for epoch in range(config['num_epochs']):
    # 모델을 학습 모드로
    # 드롭아웃이나 배치 정규화 같은 층에 영향을 준다 (여기서는 안 쓰지만 좋은 습관이다)
    model.train()
    
    epoch_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        # 데이터를 장치(GPU/CPU)로 옮긴다
        images = images.to(device)
        labels = labels.to(device)
        
        # ----------------------------------------
        # 순전파
        # ----------------------------------------
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # ----------------------------------------
        # 역전파와 최적화
        # ----------------------------------------
        optimizer.zero_grad()  # 이전 기울기 지우기
        loss.backward()         # 기울기 계산
        optimizer.step()        # 가중치 갱신
        
        # ----------------------------------------
        # 통계 기록
        # ----------------------------------------
        epoch_loss += loss.item()
        
        # 예측을 얻는다
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 100 배치마다 진행 상황 출력
        if (batch_idx + 1) % 100 == 0:
            current_acc = 100 * correct / total
            print(f"Epoch [{epoch+1}/{config['num_epochs']}], "
                  f"Step [{batch_idx+1}/{total_steps}], "
                  f"Loss: {loss.item():.4f}, "
                  f"Accuracy: {current_acc:.2f}%")
    
    # 에포크 통계 계산
    avg_loss = epoch_loss / total_steps
    epoch_accuracy = 100 * correct / total
    train_losses.append(avg_loss)
    train_accuracies.append(epoch_accuracy)
    
    print(f"\nEpoch [{epoch+1}/{config['num_epochs']}] Summary:")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Training Accuracy: {epoch_accuracy:.2f}%")
    print("-" * 80)

print("\nTraining completed!")

# ================================================================================
# 7부: 시험 집합에서의 평가
# ================================================================================
print("\n" + "=" * 80)
print("STEP 7: Evaluating on Test Set")
print("=" * 80)

# 모델을 평가 모드로 바꾼다
# 드롭아웃을 끄고, 배치 정규화는 이동 통계를 쓰게 한다
model.eval()

# 효율을 위해 기울기 계산 끄기
# 추론 중에는 기울기가 필요 없다
with torch.no_grad():
    correct = 0
    total = 0
    
    # 클래스별 정확도 기록
    class_correct = [0] * config['num_classes']
    class_total = [0] * config['num_classes']
    
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 클래스별 정확도
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# 전체 정확도
overall_accuracy = 100 * correct / total
print(f"Overall Test Accuracy: {overall_accuracy:.2f}%")
print(f"Correct predictions: {correct}/{total}")

# 클래스별 정확도
print("\nPer-Class Accuracy:")
print("-" * 40)
for i in range(config['num_classes']):
    class_acc = 100 * class_correct[i] / class_total[i]
    print(f"  Digit {i}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")
print("-" * 40)

# ================================================================================
# 8부: 예측 시각화
# ================================================================================
print("\n" + "=" * 80)
print("STEP 8: Visualizing Predictions")
print("=" * 80)

# 시험 이미지 배치 하나 가져오기
model.eval()
examples = iter(test_loader)
example_data, example_labels = next(examples)
example_data = example_data.to(device)
example_labels = example_labels.to(device)

with torch.no_grad():
    outputs = model(example_data)
    _, predictions = torch.max(outputs, 1)
    
    # 확률 얻기 (로짓의 소프트맥스)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)

# 그림을 그리기 위해 CPU로 되돌린다
example_data = example_data.cpu()
example_labels = example_labels.cpu()
predictions = predictions.cpu()
probabilities = probabilities.cpu()

# 예측 그리기
fig, axes = plt.subplots(3, 6, figsize=(15, 8))
for i, ax in enumerate(axes.flat):
    if i < 18:
        image = example_data[i].squeeze()
        true_label = example_labels[i].item()
        pred_label = predictions[i].item()
        confidence = probabilities[i][pred_label].item() * 100
        
        ax.imshow(image, cmap='gray')
        
        # 색 규칙: 맞으면 초록, 틀리면 빨강
        color = 'green' if pred_label == true_label else 'red'
        ax.set_title(f'True: {true_label}, Pred: {pred_label}\nConf: {confidence:.1f}%',
                    color=color, fontsize=10)
        ax.axis('off')

plt.tight_layout()
plt.savefig('03_mnist_predictions.png', dpi=150, bbox_inches='tight')
print("Predictions saved as '03_mnist_predictions.png'")
plt.close()

# ================================================================================
# 9부: 학습 과정 시각화
# ================================================================================
print("\n" + "=" * 80)
print("STEP 9: Training Progress Visualization")
print("=" * 80)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 손실 그리기
ax1.plot(range(1, config['num_epochs'] + 1), train_losses, 'b-', linewidth=2, marker='o')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Average Loss', fontsize=12)
ax1.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 정확도 그리기
ax2.plot(range(1, config['num_epochs'] + 1), train_accuracies, 'g-', linewidth=2, marker='s')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Training Accuracy Over Time', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 100])

plt.tight_layout()
plt.savefig('03_mnist_training_progress.png', dpi=150, bbox_inches='tight')
print("Training progress saved as '03_mnist_training_progress.png'")
plt.show()

# ================================================================================
# 핵심 정리
# ================================================================================
print("\n" + "=" * 80)
print("KEY TAKEAWAYS")
print("=" * 80)
print(f"""
1. 완전한 기계학습 파이프라인:
   ✓ 자료 불러오기와 미리 다듬기
   ✓ 모델 구조 설계
   ✓ 감시를 곁들인 학습 루프
   ✓ 따로 떼어 둔 시험 집합에서의 평가
   ✓ 결과 시각화

2. 단순한 2층 신경망으로 약 {overall_accuracy:.1f}%의 정확도를 얻었다!
   - 최고 수준의 CNN은 약 99.7%에 이른다
   - 이 기준선도 꽤 훌륭하다

3. 다중 클래스 분류에 쓰는 CrossEntropyLoss
   - LogSoftmax와 NLLLoss를 합친다
   - 따로 계산하는 것보다 수치적으로 안정적이다

4. GPU 가속은 학습을 훨씬 빠르게 한다
   - 모델과 데이터를 모두 장치로 옮겨야 한다
   - 텐서와 모델에는 .to(device)를 쓴다

5. 학습 모드와 평가 모드:
   - model.train(): 드롭아웃과 배치 정규화의 학습 동작을 켠다
   - model.eval(): 추론을 위해 그것들을 끈다

다음: 2단계에서는 PyTorch의 기능과 더 나은 구조를 다룬다!
""")

# ================================================================================
# 학생을 위한 연습문제
# ================================================================================
print("=" * 80)
print("EXERCISES TO TRY")
print("=" * 80)
print("""
1. hidden_size를 256이나 512로 늘려 보라. 정확도가 나아지는가?
2. 은닉층을 하나 더 넣어 3층 신경망을 만들어 보라
3. SGD, RMSprop, AdaGrad 등 여러 최적화기를 써 보라
4. 학습률 0.0001, 0.01, 0.1로 실험해 보라
5. 더 많은 에폭(10~20)으로 학습해 보라. 과적합을 살피라
6. 검증 손실을 기준으로 조기 종료를 구현해 보라
7. 학습한 모델을 저장하라: torch.save(model.state_dict(), 'model.pth')
8. 무작위 회전과 이동 같은 데이터 증강을 더해 보라
9. 첫 층의 가중치를 그려 신경망이 배운 것을 살펴보라
10. 혼동 행렬을 만들어 어떤 숫자가 헷갈리는지 보라
""")


if __name__ == "__main__":
    pass
```


**출력:**

```
Overall Test Accuracy: 97.42%
```

2단계의 92.51%에서 **97.42%**로 올랐다. 더한 것은 은닉층 하나와 ReLU뿐이다. 그 하나로 결정 경계가 선형이라는 제약이 풀리면서, 한 클래스 안의 서로 다른 필체를 각기 다른 은닉 뉴런이 맡을 수 있게 된다.

남은 약점은 첫 줄에 있다. 이 모델도 이미지를 784차원 벡터로 펼치고 시작하므로 화소의 이웃 관계를 쓰지 못한다. 마지막 걸음이 그것을 되찾는다.

### 그림으로 보기

위 코드가 그리는 그림 셋을 차례로 본다. 아래 그림들은 같은 코드를 다시 돌려 얻은 것이며, 그때의 마지막 정확도는 97.19%였다(실행마다 조금씩 흔들린다).

먼저 넣는 자료다.

![MNIST 시험 이미지 열두 장과 그 이름표](figures/mlp_samples.svg)

다음은 학습이 진행되는 모습이다. 실무에서 가장 자주 보게 되는 그림이 이쪽이다.

![에포크에 따른 학습 손실과 시험 정확도](figures/mlp_training_progress.svg)

| 에포크 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 학습 손실 | 0.3872 | 0.1780 | 0.1257 | 0.0960 | 0.0765 |
| 시험 정확도 | 93.92% | 95.66% | 96.76% | 97.07% | 97.19% |

두 곡선이 서로 다른 이야기를 한다. **손실은 계속 가파르게 내려가는데 정확도는 거의 평평해진다.** 1에포크에서 이미 93.92%이고, 남은 네 에포크가 3.3%포인트를 더할 뿐이다.

어긋나 보이지만 그렇지 않다. 손실은 정답에 준 **확률**을 재고 정확도는 **가장 큰 로짓이 맞았는지**만 센다. 이미 맞힌 표본의 확률이 0.8에서 0.95로 올라가면 손실은 뚜렷하게 줄지만 정확도는 한 톨도 움직이지 않는다. 곧 뒤쪽 에포크에서 모델이 하는 일은 새로 맞히는 것이 아니라 **이미 맞힌 것을 더 확신하게 되는 것**이다.

여기서 실무의 요령이 하나 나온다. 손실만 보고 있으면 아직 좋아지는 중이라고 착각하기 쉽다. 두 값을 함께 보아야 하며, 정확도가 평평해졌다면 에포크를 늘리는 것으로는 얻을 것이 적다.

마지막은 예측 결과다. 확신도를 함께 적었고, 틀린 것이 있으면 제목이 붉게 나온다.

![시험 이미지에 대한 예측과 그 확신도](figures/mlp_predictions.svg)

맞힌 예측의 확신도가 대부분 99%를 넘는다. [3.2절 연습문제 8](../linear_softmax/02_softmax.md)에서 선형 모델의 확신도를 재었을 때 맞힌 예측의 평균이 0.9420이었던 것과 견주면, 층 하나를 더한 모델이 훨씬 단호해졌음을 알 수 있다.

## 3. 논의

`MNISTClassifier` 클래스는 PyTorch의 `nn.Module` 인터페이스를 사용하여 모델 구조를 감싼다. `forward` 메서드가 계산 그래프를 정의하므로, 학습 중에 PyTorch의 autograd 체계가 경사 계산을 자동으로 처리한다. 이런 모듈식 설계 덕분에 개별 구성 요소를 고치거나 모델을 더 큰 파이프라인에 넣기가 쉬워진다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 딥러닝의 기초 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

!!! note "아래 풀이의 수치에 대하여"
    풀이에 적힌 값은 모두 이 쪽의 설정(ToTensor만 적용, 묶음 100, Adam $10^{-3}$, 5 에포크, 씨앗 42)으로 실제로 재어 얻은 것이다. 초기 가중치와 자료를 섞는 차례가 실행마다 달라 정확도는 0.1~0.4%포인트쯤 흔들리므로, 본문이 보고하는 97.42%와 마지막 자리가 다를 수 있다. 한 표 안의 값들은 모두 같은 조건에서 잰 것이므로 서로 견주는 데에는 문제가 없다.

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
784 → 128 → 10 신경망의 매개변수를 층별로 세어 101,770개가 되는지 확인하라.

</div>

??? success "연습문제 1 풀이"
    ```python
    for name, p in model.named_parameters():
        print(f"{name:12s} {tuple(p.shape)}  {p.numel():>7,}")
    ```

    | 층 | 가중치 | 편향 | 합 |
    |---|---|---|---|
    | 1층 (784→128) | $784 \times 128 = 100{,}352$ | 128 | $100{,}480$ |
    | 2층 (128→10) | $128 \times 10 = 1{,}280$ | 10 | $1{,}290$ |
    | 합 | | | **$101{,}770$** |

    1층이 전체의 98.7%를 차지한다. 입력이 784차원이라 크기 때문이다.

    [3.2절](../linear_softmax/01_linear_model.md)의 7,850개와 견주면 약 13배다.
    그런데 활성화 함수가 없으면 이 13배가 아무 구실도 하지 못한다는 것이 1절의
    요점이었다. 연습문제 3과 5에서 그 까닭을 다시 본다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
ReLU를 식으로 적고 도함수를 구하라. $z = 0$에서 미분할 수 있는가? PyTorch는 그 자리에서 무엇을 돌려주는가?

</div>

??? success "연습문제 2 풀이"
    $$\mathrm{ReLU}(z) = \max(0, z), \qquad
      \mathrm{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z < 0 \end{cases}$$

    $z = 0$에서는 **미분할 수 없다.** 왼쪽에서 온 기울기는 0이고 오른쪽에서 온
    기울기는 1이어서 둘이 맞지 않는다. 꺾인 점이기 때문이다.

    PyTorch는 그 자리에서 **0**을 돌려준다.

    ```python
    x = torch.zeros(1, requires_grad=True)
    torch.relu(x).backward()
    print(x.grad)      # tensor([0.])
    ```

    이것이 문제가 되지 않는 까닭은, 부동소수점에서 $z$가 **정확히** 0이 되는 일이
    거의 없기 때문이다. 설령 생겨도 한 점에서 어느 값을 고르든 경사 하강법의
    진행에는 영향이 없다. 이렇게 꺾인 점에서 아무 값이나 하나 고른 것을
    **하위 기울기**(subgradient)라 부르며, ReLU를 비롯한 여러 함수가 이 방식으로
    잘 학습된다.

    ReLU가 미분 불가능한 점을 가진다는 사실이 오히려 이 장의 요점이다. 그 꺾임이
    바로 선형 상쇄를 깨뜨리는 장치다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
활성화 함수가 없는 784 → 128 → 10 신경망은 매개변수를 101,770개 저장하는데 실효 자유도는 7,850개뿐이다. 이 7,850이라는 수가 어디서 나오는가?

</div>

??? success "연습문제 3 풀이"
    1절에서 보았듯 활성화가 없으면 두 층이 하나로 합쳐진다.

    $$y = (x W_1 + b_1) W_2 + b_2 = x \underbrace{(W_1 W_2)}_{W'} + \underbrace{(b_1 W_2 + b_2)}_{b'}$$

    모델이 실제로 내놓는 함수는 $(W', b')$으로 완전히 결정된다. 그런데
    $W' \in \mathbb{R}^{784 \times 10}$이고 $b' \in \mathbb{R}^{10}$이므로
    $784 \times 10 + 10 = $ **7,850**이다.

    곧 저장한 101,770개의 수가 서로 다른 함수 101,770개어치를 만들지 않는다.
    $(W_1, W_2)$를 다르게 고르고도 곱이 같으면 같은 함수다. 예컨대 $W_1$을 2배
    하고 $W_2$를 절반으로 하면 완전히 같은 모델이다.

    이 7,850이 3.2절의 선형 모델과 **정확히 같은 수**라는 점이 요점이다. 층을
    쌓아도 활성화가 없으면 3.2절에서 한 걸음도 나아가지 못한다. 연습문제 4와 5에서
    이를 수치로 확인하고, 연습문제 11에서 이 진술에 붙는 예외를 찾는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
활성화 없이 학습시킨 2층 신경망의 두 가중치를 실제로 곱해 $W' = W_1 W_2$을 만들고, 그 하나의 아핀 변환이 원래 신경망과 같은 값을 내는지 확인하라.

</div>

??? success "연습문제 4 풀이"
    ```python
    W1, b1 = model[1].weight.detach(), model[1].bias.detach()
    W2, b2 = model[2].weight.detach(), model[2].bias.detach()
    Wp, bp = W2 @ W1, W2 @ b1 + b2

    x = X_test[:2000].flatten(1)
    print((model(X_test[:2000]) - (x @ Wp.T + bp)).abs().max())
    ```

    최대 오차가 **$1.3 \times 10^{-5}$**이다. `float32`로 100,352번의 곱셈을
    누적한 결과이므로 이 정도는 반올림 오차이며, 두 계산이 **같은 함수**라는
    뜻이다.

    눈여겨볼 것은 $W'$의 꼴이 $(10, 784)$라는 점이다. 128차원 은닉층을 거쳤는데도
    남는 것은 $784 \to 10$ 변환 하나다. 은닉층은 계산 도중에 잠깐 들렀다 가는
    자리일 뿐 표현력을 더하지 않는다.

    이 확인이 중요한 까닭은, 1절의 증명이 **학습이 끝난 실제 가중치에도** 적용됨을
    보이기 때문이다. 이론상 같은 것과 코드가 실제로 같은 것은 다른 이야기이고,
    이 실험은 후자를 확인한다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
연습문제 4에서 만든 $W'$의 계수(rank)를 구하라. 왜 그 값이 되는가? 이 사실이 연습문제 3의 7,850과 어떻게 이어지는가?

</div>

??? success "연습문제 5 풀이"
    ```python
    print(torch.linalg.matrix_rank(Wp))          # 10
    print((torch.linalg.svdvals(Wp) > 1e-4).sum())   # 10
    ```

    계수는 **10**이고 특이값 10개가 모두 0에서 떨어져 있다.

    까닭은 간단하다. $W' = W_2 W_1$에서 $W_2 \in \mathbb{R}^{10 \times 128}$이므로

    $$\mathrm{rank}(W') \le \min\big(\mathrm{rank}(W_1), \mathrm{rank}(W_2)\big)
      \le \min(128, 10) = 10$$

    이고, $(10, 784)$ 행렬의 계수는 어차피 10을 넘을 수 없다. 곧 **은닉층이
    아무 제약도 걸지 않는다.** 은닉 너비가 10 이상이면 $W'$이 가질 수 있는
    행렬의 범위는 $784 \to 10$ 아핀 변환 전체와 같고, 자유도가 7,850이 된다.

    그렇다면 은닉 너비가 10보다 **작으면** 어떻게 되는가? 그때는 계수가 진짜로
    묶인다. 연습문제 11이 그 경우를 다룬다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
ReLU 자리에 시그모이드와 tanh를 넣어 보고, 활성화가 아예 없는 경우와 함께 견주어라.

</div>

??? success "연습문제 6 풀이"
    | 활성화 | 없음 | 시그모이드 | tanh | ReLU |
    |---|---|---|---|---|
    | 시험 정확도 | 92.11% | 95.78% | **97.10%** | **97.06%** |

    무엇을 쓰든 **있는 것이 없는 것보다 훨씬 낫다**는 것이 첫째 관찰이다. 5%포인트
    가까운 차이가 활성화의 존재 자체에서 온다.

    둘째, tanh와 ReLU가 사실상 같고 시그모이드가 1.3%포인트 뒤진다. 시그모이드는
    출력이 $(0, 1)$이라 중심이 0이 아니고, 양끝에서 도함수가 0에 가까워져 기울기가
    잘 흐르지 않는다. tanh는 출력이 $(-1, 1)$로 중심이 0이라 그 문제가 덜하다.

    이 두 줄짜리 층에서는 차이가 작다. 층을 깊이 쌓을수록 벌어져서, 시그모이드를
    여러 겹 쌓으면 기울기가 곱해지며 사그라들어 학습이 아예 되지 않는다. ReLU가
    표준이 된 까닭은 표현력이 더 좋아서가 아니라 **기울기를 잘 흘려보내서**다.
    자세한 비교는 [6장](../../ch06/index.md)에 있다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
은닉층의 너비를 16, 32, 64, 128, 256, 512, 1024로 바꾸어 정확도를 재어라. 매개변수를 늘리는 값어치가 어떻게 변하는가?

</div>

??? success "연습문제 7 풀이"
    | 너비 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
    |---|---|---|---|---|---|---|---|
    | 매개변수 | 12,730 | 25,450 | 50,890 | 101,770 | 203,530 | 407,050 | 814,090 |
    | 정확도 | 93.72% | 95.34% | 96.53% | 97.14% | 97.40% | 97.62% | 97.79% |

    오르기는 계속 오르지만 **값어치가 빠르게 준다.** 매개변수를 두 배씩 늘릴 때
    얻는 양을 보자.

    | 두 배 | 16→32 | 32→64 | 64→128 | 128→256 | 256→512 | 512→1024 |
    |---|---|---|---|---|---|---|
    | 얻은 양 | +1.62 | +1.19 | +0.61 | +0.26 | +0.22 | +0.17 |

    16에서 32로 갈 때 얻는 1.62%포인트를 512에서 1024로 갈 때는 0.17%포인트밖에
    얻지 못한다. 매개변수 40만 개를 더 넣고 얻은 값이다.

    이것이 다음 걸음이 필요한 까닭이다. 너비를 늘려 97%대 후반까지는 갈 수 있지만
    99%로 가려면 매개변수를 천문학적으로 늘려야 한다. [3.4 합성곱 신경망](04_cnn.md)은
    매개변수를 42만 개만 쓰면서 99.2%에 이른다. **모델을 키우는 것과 모델을 알맞게
    만드는 것은 다른 일이다.**

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
은닉층을 하나 더 쌓아 784 → 128 → 128 → 10, 그리고 하나 더 쌓아 784 → 128 → 128 → 128 → 10으로 만들어 보라. 깊이가 늘면 좋아지는가?

</div>

??? success "연습문제 8 풀이"
    | 은닉층 수 | 1 | 2 | 3 |
    |---|---|---|---|
    | 시험 정확도 | 97.14% | 97.19% | **97.02%** |

    좋아지지 않는다. 2층에서 0.05%포인트 오르고 3층에서는 오히려 내려간다.
    실행마다 생기는 흔들림 안에 들어가는 차이다.

    뜻밖으로 보일 수 있다. 깊은 신경망이 강하다고들 하니 말이다. 그러나 깊이가
    저절로 이득을 주지는 않는다. MNIST에서 이 얼개로는 얻을 것이 이미 거의 다
    나왔고, 층을 더하면 오히려 최적화가 어려워진다. 기울기가 지나갈 길이 길어지고
    매개변수가 늘어 과적합할 여지도 커진다.

    깊이가 값어치를 내려면 층마다 **뜻이 있는 특징**을 쌓아 올려야 한다. 날것의
    화소를 펼쳐 넣은 완전 연결층을 여러 겹 쌓는 것으로는 그렇게 되지 않는다.
    3.4절의 합성곱 신경망은 층마다 점점 넓은 영역의 무늬를 보게 되어 깊이가
    뜻을 갖는다. 깊이를 살리는 장치(잔차 연결, 정규화)는 뒤의 장들에서 다룬다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
학습이 끝난 뒤 은닉 뉴런 128개 가운데 몇 개가 **모든** 시험 입력에 대해 0을 내놓는지 세어라. 또 은닉 활성값 전체에서 0의 비율은 얼마인가?

</div>

??? success "연습문제 9 풀이"
    ```python
    with torch.no_grad():
        h = torch.relu(model[1](X_test.flatten(1)))
    alive = (h > 0).any(dim=0).sum()
    print(alive, (h == 0).float().mean())
    ```

    - 완전히 죽은 뉴런: **128개 중 1개**
    - 은닉 활성값 가운데 0인 비율: **40.9%**

    두 수를 구별해서 읽어야 한다.

    **죽은 뉴런**은 어떤 입력에도 반응하지 않으므로 매개변수를 785개 낭비하고 있고,
    기울기가 0이라 되살아날 길도 없다. 이것이 "죽은 ReLU" 문제다. 다행히 여기서는
    1개뿐이다. 학습률이 크거나 층이 깊으면 훨씬 많아진다.

    반면 활성값의 40.9%가 0인 것은 **문제가 아니라 기능**이다. 입력마다 다른
    뉴런 집합이 반응한다는 뜻이며, 이를 희소성(sparsity)이라 한다. 입력에 따라
    서로 다른 선형 함수가 적용되는 것이 바로 ReLU 망이 비선형인 방식이다.
    1절에서 "은닉 뉴런이 입력 공간을 조각으로 나눈다"고 한 것을 수치로 본 것이다.

    같은 40.9%를 뒤집어 보면 입력 하나가 평균 76개 뉴런을 켠다는 뜻이다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
활성화가 없는 2층 신경망(92.11%)이 3.2절의 단층 선형 모델보다 오히려 조금 낮다. 같은 설정에서 단층 선형 모델은 92.34%다. 두 모델이 같은 함수 모둠을 가진다면 왜 값이 다른가?

</div>

??? success "연습문제 10 풀이"
    | 모델 | 매개변수 | 표현할 수 있는 함수 | 정확도 |
    |---|---|---|---|
    | 단층 선형 | 7,850 | $784 \to 10$ 아핀 전체 | **92.34%** |
    | 2층, 활성화 없음 | 101,770 | $784 \to 10$ 아핀 전체 (같음) | 92.11% |

    표현할 수 있는 함수의 모둠이 **같다.** 그러므로 차이는 표현력이 아니라
    **최적화**에서 온다.

    같은 함수를 $W'$으로 직접 나타내느냐 $W_1 W_2$로 쪼개어 나타내느냐에 따라
    경사 하강법이 지나가는 길이 달라진다. 쪼개어 놓으면

    - 손실면이 **비볼록**해진다. 단층 선형의 교차 엔트로피는 볼록인데, 곱으로
      나타내면 그렇지 않다(연습문제 13).
    - 기울기가 사슬 규칙으로 곱해지므로 두 층의 크기 균형에 따라 실효 학습률이
      달라진다.
    - 초기화가 곱으로 들어가 출력의 크기가 달라진다.

    그래서 같은 5 에포크 안에 도달하는 자리가 조금 다르다. 0.23%포인트는 작은
    차이이며 이 방향이 늘 같다고 말할 수도 없다. 요점은 **표현력이 같아도 학습
    결과가 같지 않다**는 것이다.

    뒤집어 말하면, 모델을 볼 때 "무엇을 나타낼 수 있는가"와 "경사 하강법으로
    거기에 닿을 수 있는가"를 따로 물어야 한다. 딥러닝에서 이 둘이 어긋나는 일이
    아주 흔하다.

---

<div class="drillbox" markdown>

**연습문제 11.** <span class="diff hard" title="어려움"></span>
연습문제 5는 은닉 너비가 10 이상이면 활성화 없는 2층 모델이 단층 선형 모델과 같은 함수 모둠을 가진다고 했다. 너비가 10보다 **작으면** 어떻게 되는가? 계수 논증으로 예측한 뒤 너비 2, 5, 10에서 실제로 재어라.

</div>

??? success "연습문제 11 풀이"
    **예측.** $W' = W_2 W_1$에서 은닉 너비를 $h$라 하면

    $$\mathrm{rank}(W') \le \min(h, 10)$$

    이다. $h \ge 10$이면 이 한계가 $(10, 784)$ 행렬의 원래 한계와 같아 아무
    제약이 아니지만, $h < 10$이면 **계수가 $h$로 묶인다.** 곧 로짓 10개를
    $h$차원 공간을 거쳐 만들어야 하므로, 표현할 수 있는 아핀 변환이 진짜로
    줄어든다. 이때 은닉층은 **병목**이다.

    **측정.**

    | 은닉 너비 $h$ | 2 | 5 | 10 | 128 |
    |---|---|---|---|---|
    | $\mathrm{rank}(W')$ 상한 | 2 | 5 | 10 | 10 |
    | 활성화 없음 | **68.46%** | **89.56%** | **92.40%** | 92.11% |

    예측대로다. $h = 2$에서 68.46%로 무너지고, $h = 5$에서 89.56%, $h = 10$에서
    92.40%로 단층 선형 모델(92.34%)을 따라잡는다. 그 뒤로는 더 넓혀도 오르지
    않는다(128에서 92.11%).

    이 실험이 값진 까닭은 1절의 "층을 쌓아도 얻는 것이 없다"는 말에 **정확한
    조건**을 붙여 주기 때문이다. 정확히 말하면 이렇다. 활성화 없는 2층 모델은
    은닉 너비가 출력 차원 이상일 때 단층 모델과 같고, 그보다 좁으면 단층 모델보다
    **약하다.** 층을 쌓아 얻는 것이 없거나, 있으면 손해다.

    계수가 묶인 선형 변환은 그 자체로 쓸모가 있다. 이것이 저계수 근사이며,
    큰 모델을 줄이는 LoRA 같은 기법이 바로 이 구조를 일부러 쓴다. 다만 그것은
    표현력을 **줄이려는** 의도일 때의 이야기다.

---

<div class="drillbox" markdown>

**연습문제 12.** <span class="diff hard" title="어려움"></span>
연습문제 11에서 너비 2일 때 활성화 없는 모델이 68.46%였다. 같은 너비에서 ReLU를 넣으면 56.86%로 오히려 **더 낮아진다.** 너비 128에서는 ReLU가 5%포인트 이득을 주었는데 왜 여기서는 해로운가?

</div>

??? success "연습문제 12 풀이"
    | 은닉 너비 | 2 | 5 | 10 | 128 |
    |---|---|---|---|---|
    | 활성화 없음 | **68.46%** | 89.56% | 92.40% | 92.11% |
    | ReLU | **56.86%** | 89.07% | 92.49% | **97.06%** |
    | ReLU의 이득 | **−11.6** | −0.5 | +0.1 | **+5.0** |

    ReLU의 이득이 너비에 따라 부호를 바꾼다.

    까닭은 ReLU가 **공짜가 아니라는** 데 있다. ReLU는 음수를 0으로 버린다.
    은닉 뉴런 128개 가운데 40%가 0이 되는 것은(연습문제 9) 감당할 수 있는 손실이며,
    그 대가로 비선형성을 얻으니 남는 장사다.

    그런데 은닉 뉴런이 2개뿐이라면, 그중 하나가 어떤 입력에 대해 0이 되는 순간
    그 입력에 대해 쓸 수 있는 정보 통로의 **절반**이 막힌다. 통로가 두 개뿐인데
    수시로 하나가 닫힌다. 게다가 닫힌 뉴런은 기울기도 0이라 학습 신호가 끊긴다.
    비선형성으로 얻는 것보다 용량으로 잃는 것이 크다.

    정리하면 ReLU의 값어치는 **남는 용량이 있을 때만** 실현된다. 이 표는 그
    맞바꿈이 너비 5~10 사이에서 균형을 이루고 그 위에서 이득으로 돌아서는 것을
    보여 준다.

    실무의 규칙 하나가 여기서 나온다. 좁은 병목층에는 ReLU를 두지 않는 것이
    보통이다. 자기 부호기의 가장 좁은 층이나 어텐션의 사영층이 흔히 선형으로
    남는 까닭이다. 좁은 곳에서 절반을 버릴 여유가 없기 때문이다.

---

<div class="drillbox" markdown>

**연습문제 13.** <span class="diff hard" title="어려움"></span>
활성화가 없는 2층 모델의 손실이 매개변수에 대해 **비볼록**임을 보여라. $W' = W_1 W_2$이라는 곱셈 구조만으로 충분하다. 가장 간단한 반례를 만들어라.

</div>

??? success "연습문제 13 풀이"
    스칼라 두 개로 줄여도 곱셈 구조는 그대로 남는다. 목표를 $w_1 w_2 = 1$이라 하고
    손실을 다음과 같이 두자.

    $$f(w_1, w_2) = (w_1 w_2 - 1)^2$$

    이제 두 점을 잡는다.

    $$A = (1, 1), \qquad B = (-1, -1)$$

    둘 다 $w_1 w_2 = 1$이므로 $f(A) = f(B) = 0$으로 **전역 최소점**이다.
    그런데 중점은 $\left(\tfrac{1-1}{2}, \tfrac{1-1}{2}\right) = (0, 0)$이고

    $$f(0, 0) = (0 \cdot 0 - 1)^2 = 1$$

    이다. 볼록함수라면

    $$f\!\left(\frac{A+B}{2}\right) \le \frac{f(A) + f(B)}{2} = 0$$

    이어야 하는데 $1 > 0$이다. 따라서 $f$는 볼록하지 않다. $\square$

    ```python
    f = lambda a, b: (a * b - 1) ** 2
    print(f(1, 1), f(-1, -1), f(0, 0))      # 0.0 0.0 1.0
    ```

    헤세 행렬로도 확인된다. $(0.5, 0.5)$에서

    $$\nabla^2 f = \begin{pmatrix} 0.5 & -1 \\ -1 & 0.5 \end{pmatrix},
      \qquad \text{고윳값} = -0.5,\ 1.5$$

    로 부호가 섞여 있다. 볼록하다면 모든 고윳값이 0 이상이어야 한다.

    이 반례는 딥러닝의 손실면이 왜 비볼록인지를 가장 단순하게 보여 준다. 원인은
    활성화 함수가 아니라 **층을 곱으로 쌓는다는 사실 자체**다. 활성화를 모두
    없애도 비볼록하다. 그래서 두 개의 전역 최소점을 이은 선분 위에 더 나쁜 점이
    있을 수 있고, 3.2절과 달리 "극소점이면 최소점"이라는 보장이 사라진다.

    연습문제 10에서 표현력이 같은데도 결과가 달랐던 까닭이 바로 이것이다.

---

<div class="drillbox" markdown>

**연습문제 14.** <span class="diff hard" title="어려움"></span>
보편 근사 정리는 은닉층 하나로 충분히 넓으면 어떤 연속 함수든 원하는 만큼 가깝게 흉내 낼 수 있다고 말한다. 그렇다면 3.4절의 합성곱 신경망은 왜 필요한가?

</div>

??? success "연습문제 14 풀이"
    정리를 먼저 정확히 적어 두자. 대략 이렇다. $K \subset \mathbb{R}^n$이 옹골찬
    집합이고 $f: K \to \mathbb{R}$이 연속이며 활성화 $\sigma$가 다항식이 아니면,
    임의의 $\varepsilon > 0$에 대해 은닉층 하나를 가진 신경망 $g$가 있어

    $$\sup_{x \in K} |f(x) - g(x)| < \varepsilon$$

    을 만족한다. 곧 연습문제 7에서 너비를 늘린 것이 원리상 끝까지 통한다는 말이다.

    그런데도 합성곱이 필요한 까닭은, 이 정리가 말하지 **않는** 것이 셋이나 되기
    때문이다.

    첫째, **너비를 말해 주지 않는다.** 존재만 보장하고 몇 개가 필요한지는 함수에
    따라 지수적으로 커질 수 있다. 연습문제 7의 표가 그 실상이다. 너비를 64배
    늘려 얻은 것이 4%포인트였다.

    둘째, **찾을 수 있다고 말하지 않는다.** 그런 가중치가 존재한다는 것과 경사
    하강법이 유한한 시간에 거기 닿는다는 것은 전혀 다른 진술이다. 연습문제 10과
    13이 보인 것처럼 손실면은 비볼록하다.

    셋째, 그리고 가장 중요하게, **일반화를 말하지 않는다.** 정리는 학습 자료 위에서
    함수를 흉내 내는 이야기다. 우리가 원하는 것은 **처음 보는** 숫자를 맞히는
    것이다. 매개변수를 늘려 학습 자료에 맞추는 힘을 키우면 외우기 쉬워진다.

    합성곱이 이기는 곳이 정확히 셋째다. 이미지에는 공간 구조가 있다는 참인 가정을
    미리 심어 두어, 같은 매개변수로 **더 잘 일반화**한다. 3.4절 연습문제 9에서
    매개변수가 같은 다층 퍼셉트론이 1.5%포인트 뒤지는 것이 그 증거다.

    그러니 이 정리는 "무엇이 가능한가"의 답이고, 딥러닝의 실제 물음은 "무엇이
    적은 자료와 적은 계산으로 배워지는가"다. 보편 근사 정리는 후자에 대해 아무
    말도 하지 않는다.

## 정리하며

**다룬 것** — 왜 활성화 함수가 필요한가

층을 쌓는 것만으로는 아무것도 얻지 못한다. 선형 변환을 겹치면 그 합성이 다시 선형 변환이므로, 784 → 128 → 10 신경망은 매개변수를 101,770개 저장하고도 실효 자유도가 3.2절과 똑같은 7,850개다. 학습된 두 가중치를 실제로 곱해 보면 단일 아핀 변환과 오차 $10^{-5}$ 안에서 같다.

ReLU의 꺾임 하나가 그 상쇄를 깨뜨린다. 같은 구조에 ReLU만 넣어 97%대로 올라가며, 그 이득의 크기는 5%포인트에 이른다. 곧 이 절이 더한 생각은 **비선형성**이다.

그러나 여기서 두 가지를 함께 배워 두는 것이 좋다. 첫째, 비선형성은 공짜가 아니다. 은닉층이 좁으면 ReLU가 오히려 해롭다. 둘째, 너비를 늘려 얻는 값어치는 빠르게 줄어든다. 너비를 64배 늘려 얻는 것이 4%포인트뿐이다.

그래서 다음 걸음은 모델을 더 키우는 쪽이 아니다. 이 절의 모델은 여전히 첫 줄에서 이미지를 펼쳐 화소의 이웃 관계를 버리고 있다. [3.4 합성곱 신경망](04_cnn.md)이 그것을 되찾는다.

앞의 연습문제 14개로 직접 확인할 수 있다.
