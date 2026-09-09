# 3단계: 다층 퍼셉트론

[2단계](02_linear_softmax.md)의 선형 모델은 92.51%에서 멈췄다. 결정 경계가 선형이라는 제약 때문이다. 자연스러운 다음 수는 층을 하나 더 쌓는 것이다. 784 → 128 → 10으로 가면 매개변수가 7850개에서 10만 개 남짓으로 늘어나니, 표현력도 그만큼 늘 것 같다.

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

ReLU 말고도 시그모이드, tanh, GELU 등 여러 활성화 함수가 있고 저마다 성질이 다르다. 그 비교와 선택 기준은 [4장의 활성화 함수](../../ch05/index.md)에서 다룬다. 여기서 중요한 것은 **무엇을 쓰느냐가 아니라 반드시 있어야 한다는 것**이다.

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

## 3. 논의

`MNISTClassifier` 클래스는 PyTorch의 `nn.Module` 인터페이스를 사용하여 모델 구조를 감싼다. `forward` 메서드가 계산 그래프를 정의하므로, 학습 중에 PyTorch의 autograd 체계가 경사 계산을 자동으로 처리한다. 이런 모듈식 설계 덕분에 개별 구성 요소를 고치거나 모델을 더 큰 파이프라인에 넣기가 쉬워진다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 딥러닝의 기초 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
기본 초기화 상태의 `MNISTClassifier`에서 학습 가능한 매개변수의 총 개수를 계산하라. 가중치와 편향을 모두 포함하여 층별로 나누어 세어라.

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)` 각각에는 `in_features * out_features`개의 가중치 매개변수와 (`bias=False`가 아닌 한) `out_features`개의 편향 매개변수가 있다. `nn.Conv2d(in_c, out_c, k)`에는 `in_c * out_c * k * k`개의 가중치와 `out_c`개의 편향이 있다. `nn.Embedding(num, dim)`에는 `num * dim`개의 매개변수가 있다. 모든 층에 대해 더하면 된다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

---

**연습문제 2.**
최적화기를 Adam으로 바꾸고(`torch.optim.Adam`에 `lr=0.001`을 쓴다) 원래 최적화기와 학습 수렴을 비교하라. 두 손실 곡선을 같은 그래프에 그려라.

??? success "연습문제 2 풀이"
    최적화기를 만드는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 바꾼다. Adam은 매개변수마다 적응적인 학습률과 운동량 추정값을 유지하므로 초반 에폭에서 대체로 더 빠르게 수렴한다. Adam의 손실 곡선은 보통 처음 몇 에폭에서 더 가파르게 떨어지지만, 최적점 근처에서는 운동량을 쓴 SGD보다 조금 더 흔들릴 수 있다. 공정한 비교를 위해 둘을 같은 난수 씨앗과 같은 에폭 수로 실행하라.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
층이나 블록의 개수를 설정할 수 있도록 `MNISTClassifier`를 확장하라. `__init__`에 `num_layers` 매개변수를 추가하고 `nn.ModuleList`로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 순회한다. (평범한 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 PyTorch가 모든 매개변수를 최적화 대상으로 등록한다. 시험은 다음과 같이 한다. `for n in [2, 4, 8]: model = MNISTClassifier(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.

## 정리하며

**다룬 것** — MNIST 기본

`MNISTClassifier` 클래스는 PyTorch의 `nn.Module` 인터페이스를 사용하여 모델 구조를 감싼다.

핵심 클래스는 `MNISTClassifier`이며 앞의 연습문제 4개로 직접 확인할 수 있다.
