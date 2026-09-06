# 57모듈: 이어 배우기

57모듈: 이어 배우기 - 초급 파일 1: 파국적 잊음 보여 주기

이어 배우기는 앞서 배운 앎을 잊지 않으면서 잇단 과제로 모델을 익히는 어려움을 다룬다. 이 구현은 말랑함을 지키면서 파국적 잊음을 누그러뜨리는 핵심 전략을 보여 준다.

## 코드

```python
"""
57모듈: 이어 배우기 - 초급
파일 1: 파국적 잊음 보여 주기

이 스크립트는 과제를 차례로 배울 때 신경망에 생기는 파국적 잊음 문제를 보여 준다.
단순한 망을 여러 과제로 익히면서
앞선 과제의 성능이 어떻게 떨어지는지 그려 본다.

학습 목표:
1. 파국적 잊음이 무엇인지 이해한다
2. 구체적인 보기로 문제를 그려 본다
3. 잊음을 수로 잰다
4. 이어 배우기 방법의 밑금을 세운다

수학적 바탕:
- 과제 T₁, T₂, ..., T_n에 대해 다음을 잰다.
  * 과제 j을 배운 뒤 과제 i의 정확도: Acc_{i,j}
  * 잊음: F_i = Acc_{i,i} - Acc_{i,n}
  * 평균 잊음: (1/n) Σ F_i
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import copy

# ========================================================================
# 메인
# ========================================================================

# 재현성을 위한 난수 시드 설정
torch.manual_seed(42)
np.random.seed(42)


class SimpleNetwork(nn.Module):
    """
    파국적 잊음을 보여 주기 위한 단순한 앞먹임 망.
    
    구조:
    - 입력: 28x28을 편 784차원
    - 숨은 층: ReLU 활성을 쓴 층 둘
    - 출력: 부류 10개(MNIST 숫자)
    
    이 단순한 구조가 파국적 잊음을 더 또렷이 드러낸다.
    """
    
    def __init__(self, input_size: int = 784, hidden_size: int = 256, 
                 num_classes: int = 10):
        super(SimpleNetwork, self).__init__()
        
        # 망의 층을 정한다
        # 층 1: 입력에서 첫 숨은 층으로
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        
        # 층 2: 첫 숨은 층에서 둘째 숨은 층으로
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        
        # 층 3: 둘째 숨은 층에서 출력 층으로
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        신경망을 통과하는 순전파.
        
        인수:
            x: 꼴이 (batch_size, 784)인 입력 텐서
        
        반환값:
            꼴이 (batch_size, num_classes)인 출력 로짓
        """
        # 필요하면 입력을 편다([batch, 1, 28, 28]에서 [batch, 784]로)
        x = x.view(x.size(0), -1)
        
        # 앞으로 퍼뜨리기
        x = self.relu1(self.fc1(x))  # ReLU를 쓴 첫 숨은 층
        x = self.relu2(self.fc2(x))  # ReLU를 쓴 둘째 숨은 층
        x = self.fc3(x)               # 출력 층(로짓)
        
        return x


def create_split_mnist_tasks(num_tasks: int = 5) -> List[List[int]]:
    """
    숫자 10개를 무리로 나누어 Split MNIST 과제를 만든다.
    
    Split MNIST은 MNIST 숫자 10개를 여러 이진(또는 다중) 분류 과제로
    쪼개는 고전적인 이어 배우기 잣대이다.
    
    과제 5개일 때의 보기:
    - 과제 0: 숫자 0과 1
    - 과제 1: 숫자 2와 3
    - 과제 2: 숫자 4와 5
    - 과제 3: 숫자 6과 7
    - 과제 4: 숫자 8과 9
    
    인수:
        num_tasks: 만들 과제의 개수(기본값 5)
    
    반환값:
        리스트의 리스트이며, 안쪽 리스트마다 그 과제의 숫자 부류를 담는다
    """
    all_digits = list(range(10))
    classes_per_task = 10 // num_tasks
    
    tasks = []
    for i in range(num_tasks):
        # 이 과제에 쓸 부류를 고른다
        task_classes = all_digits[i * classes_per_task:(i + 1) * classes_per_task]
        tasks.append(task_classes)
    
    return tasks


def create_task_dataset(full_dataset, task_classes: List[int]) -> TensorDataset:
    """
    부류를 걸러 특정 과제의 데이터셋을 만든다.
    
    이 함수는 다음을 한다.
    1. 지정한 부류만 남도록 데이터셋을 거른다
    2. 이름표를 이어지게(0, 1, 2, ...) 다시 매긴다
    3. 새 TensorDataset을 되돌린다
    
    인수:
        full_dataset: 온전한 데이터셋(이를테면 MNIST)
        task_classes: 이 과제에 넣을 부류 이름표 목록
    
    반환값:
        지정한 부류의 보기만 담은 TensorDataset
    """
    # 과제 부류에 드는 보기의 첨자를 찾는다
    indices = []
    for idx in range(len(full_dataset)):
        _, label = full_dataset[idx]
        if label in task_classes:
            indices.append(idx)
    
    # 데이터셋의 부분집합을 만든다
    subset = Subset(full_dataset, indices)
    
    # 데이터와 이름표를 모두 뽑아낸다
    data_list = []
    label_list = []
    
    for idx in range(len(subset)):
        img, label = subset[idx]
        data_list.append(img)
        
        # 이름표를 이어지게 다시 매긴다(이를테면 부류 [4,5]가 [0,1]이 된다)
        new_label = task_classes.index(label)
        label_list.append(new_label)
    
    # 텐서로 쌓는다
    data_tensor = torch.stack(data_list)
    label_tensor = torch.tensor(label_list, dtype=torch.long)
    
    return TensorDataset(data_tensor, label_tensor)


def train_on_task(model: nn.Module, 
                  train_loader: DataLoader,
                  device: torch.device,
                  epochs: int = 5,
                  learning_rate: float = 0.001) -> List[float]:
    """
    과제 하나로 모델을 익힌다.
    
    학습 절차:
    1. 모델을 학습 모드로 둔다
    2. 시대마다 다음을 한다.
        a. 배치를 훑는다
        b. 손실을 셈한다(교차 엔트로피)
        c. 기울기를 거꾸로 퍼뜨린다
        d. 최적화기로 가중치를 고친다
    3. 살펴보려고 학습 손실을 좇는다
    
    인수:
        model: 익힐 신경망
        train_loader: 학습 데이터의 DataLoader
        device: 익힐 장치(CPU나 GPU)
        epochs: 학습 에포크 수
        learning_rate: 최적화기의 학습률
    
    반환값:
        시대마다의 평균 손실 목록
    """
    # 손실 함수를 정한다(분류에는 교차 엔트로피)
    criterion = nn.CrossEntropyLoss()
    
    # 최적화기를 정한다(Adam이 튼튼하고 널리 쓰인다)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 학습 손실을 좇는다
    losses = []
    
    model.train()  # 학습 모드로 둔다(드롭아웃 등을 켠다)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # 데이터를 장치로 옮기기
            data, target = data.to(device), target.to(device)
            
            # 앞선 되풀이의 기울기를 0으로 만든다
            optimizer.zero_grad()
            
            # 앞먹임: 예측을 셈한다
            output = model(data)
            
            # 손실을 계산한다
            loss = criterion(output, target)
            
            # 역전파: 경사를 계산한다
            loss.backward()
            
            # 가중치를 갱신한다
            optimizer.step()
            
            # 통계 기록
            epoch_loss += loss.item()
            num_batches += 1
        
        # 이 시대의 평균 손실
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return losses


def evaluate_on_task(model: nn.Module,
                    test_loader: DataLoader,
                    device: torch.device) -> Tuple[float, float]:
    """
    과제에서 모델의 성능을 평가한다.
    
    평가 지표:
    - 정확도: 맞힌 비율
    - 손실: 평균 교차 엔트로피 손실
    
    인수:
        model: 평가할 신경망
        test_loader: 시험 데이터의 DataLoader
        device: 평가할 장치
    
    반환값:
        (정확도, 손실) 짝
    """
    criterion = nn.CrossEntropyLoss()
    
    model.eval()  # 평가 모드로 둔다(드롭아웃 등을 끈다)
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    # 효율을 위해 기울기 계산 끄기
    with torch.no_grad():
        for data, target in test_loader:
            # 데이터를 장치로 옮기기
            data, target = data.to(device), target.to(device)
            
            # 순전파
            output = model(data)
            
            # 손실을 계산한다
            loss = criterion(output, target)
            total_loss += loss.item()
            
            # 정확도를 계산한다
            _, predicted = torch.max(output, 1)  # 확률이 가장 높은 부류를 얻는다
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    # 지표를 계산한다
    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(test_loader)
    
    return accuracy, avg_loss


def demonstrate_catastrophic_forgetting(num_tasks: int = 5,
                                       epochs_per_task: int = 5,
                                       batch_size: int = 128):
    """
    파국적 잊음을 보여 주는 주 함수.
    
    과정:
    1. Split MNIST 과제를 만든다
    2. 새 모델을 초기화한다
    3. 과제마다 차례로 익힌다
    4. 과제를 마칠 때마다 앞선 모든 과제에서 평가한다
    5. 성능이 어떻게 떨어지는지 그려 본다
    
    핵심 관찰:
    새 과제를 배울수록 앞선 과제의 정확도가 크게 떨어진다.
    이것이 파국적 잊음이다!
    
    인수:
        num_tasks: Split MNIST의 과제 개수
        epochs_per_task: 과제마다의 학습 시대 수
        batch_size: 학습에 쓸 배치 크기
    """
    # 장치 지정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # MNIST 데이터셋 불러오기
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST 고르기
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Split MNIST 과제를 만든다
    task_classes = create_split_mnist_tasks(num_tasks)
    print("Task Configuration:")
    for i, classes in enumerate(task_classes):
        print(f"  Task {i}: Classes {classes}")
    print()
    
    # 과제마다의 데이터셋을 만든다
    train_loaders = []
    test_loaders = []
    
    for classes in task_classes:
        train_task_dataset = create_task_dataset(train_dataset, classes)
        test_task_dataset = create_task_dataset(test_dataset, classes)
        
        train_loader = DataLoader(train_task_dataset, batch_size=batch_size, 
                                 shuffle=True)
        test_loader = DataLoader(test_task_dataset, batch_size=batch_size, 
                                shuffle=False)
        
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
    
    # 모형을 시작한다
    num_classes_per_task = len(task_classes[0])
    model = SimpleNetwork(input_size=784, hidden_size=256, 
                         num_classes=num_classes_per_task).to(device)
    
    print(f"Model Architecture:")
    print(model)
    print(f"\nTotal Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # 정확도를 담을 행렬: 행은 과제, 열은 학습 단계
    # accuracy_matrix[i][j] = 과제 j까지 익힌 뒤 과제 i의 정확도
    accuracy_matrix = np.zeros((num_tasks, num_tasks))
    
    # 과제마다 차례로 익힌다
    print("=" * 70)
    print("SEQUENTIAL TRAINING (Demonstrating Catastrophic Forgetting)")
    print("=" * 70)
    
    for task_id in range(num_tasks):
        print(f"\n{'=' * 70}")
        print(f"Training on Task {task_id} (Classes: {task_classes[task_id]})")
        print('=' * 70)
        
        # 지금 과제로 익힌다
        train_on_task(
            model=model,
            train_loader=train_loaders[task_id],
            device=device,
            epochs=epochs_per_task,
            learning_rate=0.001
        )
        
        # 지금까지 본 모든 과제에서 평가한다
        print(f"\nEvaluating on all tasks after training Task {task_id}:")
        print("-" * 50)
        
        for eval_task_id in range(task_id + 1):
            accuracy, loss = evaluate_on_task(
                model=model,
                test_loader=test_loaders[eval_task_id],
                device=device
            )
            
            # 정확도를 행렬에 담는다
            accuracy_matrix[eval_task_id][task_id] = accuracy
            
            # 결과 출력
            print(f"  Task {eval_task_id}: Accuracy = {accuracy:.2f}%")
            
            # 파국적 잊음을 두드러지게 보인다
            if eval_task_id < task_id:
                original_acc = accuracy_matrix[eval_task_id][eval_task_id]
                forgetting = original_acc - accuracy
                print(f"    (Original: {original_acc:.2f}%, "
                      f"Forgetting: {forgetting:.2f}%)")
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    # 잊음 지표를 셈한다
    print("\nAccuracy Matrix (rows = tasks, columns = after training task):")
    print("(Each entry shows accuracy on row's task after training column's task)")
    print()
    
    # 머리글을 찍는다
    print("Task |", end="")
    for j in range(num_tasks):
        print(f" After T{j} |", end="")
    print(" Forgetting")
    print("-" * (12 + 12 * num_tasks))
    
    # 잊음 셈과 함께 정확도 행렬을 찍는다
    total_forgetting = 0.0
    for i in range(num_tasks):
        print(f"  {i}  |", end="")
        for j in range(num_tasks):
            if j >= i:  # 과제를 배운 뒤의 값만 보인다
                print(f"  {accuracy_matrix[i][j]:5.1f}%  |", end="")
            else:
                print(f"    -    |", end="")
        
        # 이 과제의 잊음을 셈한다
        if i < num_tasks - 1:  # 마지막 과제에는 셈하지 않는다
            original = accuracy_matrix[i][i]
            final = accuracy_matrix[i][num_tasks - 1]
            forgetting = original - final
            total_forgetting += forgetting
            print(f"  {forgetting:5.1f}%")
        else:
            print("    N/A")
    
    # 평균 잊음을 셈한다
    avg_forgetting = total_forgetting / (num_tasks - 1) if num_tasks > 1 else 0
    print()
    print(f"Average Forgetting: {avg_forgetting:.2f}%")
    print(f"Final Average Accuracy: {np.mean(accuracy_matrix[:, -1]):.2f}%")
    
    # 결과를 그려 본다
    visualize_catastrophic_forgetting(accuracy_matrix, task_classes)
    
    return accuracy_matrix


def visualize_catastrophic_forgetting(accuracy_matrix: np.ndarray,
                                     task_classes: List[List[int]]):
    """
    파국적 잊음을 그려 본다.
    
    그림 둘을 만든다.
    1. 때에 따른 잊음을 보여 주는 정확도 행렬의 열 지도
    2. 과제마다의 정확도 자취를 보여 주는 선 그림
    
    인수:
        accuracy_matrix: 정확도 행렬(과제 × 학습 단계)
        task_classes: 과제마다의 부류 배정 목록
    """
    num_tasks = len(task_classes)
    
    # 부분 그림 둘을 담은 그림을 만든다
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 그림 1: 정확도 행렬의 열 지도
    ax1 = axes[0]
    im = ax1.imshow(accuracy_matrix, cmap='RdYlGn', aspect='auto', 
                    vmin=0, vmax=100)
    
    # 색 막대를 더한다
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Accuracy (%)', rotation=270, labelpad=20)
    
    # 이름표를 붙인다
    ax1.set_xlabel('After Training Task', fontsize=12)
    ax1.set_ylabel('Evaluated Task', fontsize=12)
    ax1.set_title('Catastrophic Forgetting Heatmap', fontsize=14, fontweight='bold')
    
    # 눈금을 매긴다
    ax1.set_xticks(range(num_tasks))
    ax1.set_yticks(range(num_tasks))
    ax1.set_xticklabels([f'T{i}' for i in range(num_tasks)])
    ax1.set_yticklabels([f'T{i}' for i in range(num_tasks)])
    
    # 글자 주석을 추가한다
    for i in range(num_tasks):
        for j in range(num_tasks):
            if j >= i:  # 쓸모 있는 자리에만 주석을 단다
                text = ax1.text(j, i, f'{accuracy_matrix[i, j]:.1f}',
                              ha="center", va="center", color="black", fontsize=9)
    
    # 그림 2: 정확도 자취를 보여 주는 선 그림
    ax2 = axes[1]
    
    for task_id in range(num_tasks):
        # 이 과제의 정확도 자취를 뽑아낸다
        accuracies = [accuracy_matrix[task_id][j] for j in range(task_id, num_tasks)]
        stages = list(range(task_id, num_tasks))
        
        # 선을 그린다
        ax2.plot(stages, accuracies, marker='o', linewidth=2, 
                label=f'Task {task_id} (Classes {task_classes[task_id]})',
                markersize=8)
        
        # 처음 정확도를 두드러지게 보인다
        ax2.scatter([task_id], [accuracies[0]], s=150, 
                   marker='*', zorder=5)
    
    ax2.set_xlabel('Training Stage (After Task)', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy Degradation Over Time', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])
    ax2.set_xticks(range(num_tasks))
    ax2.set_xticklabels([f'T{i}' for i in range(num_tasks)])
    
    plt.tight_layout()
    plt.savefig('catastrophic_forgetting_demonstration.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'catastrophic_forgetting_demonstration.png'")
    plt.show()


if __name__ == "__main__":
    """
    주 실행: Split MNIST에서 파국적 잊음을 보여 준다
    
    기대되는 관찰:
    1. 과제마다 처음에는 정확도가 95~99% 남짓이다
    2. 새 과제를 배울수록 옛 과제의 정확도가 크게 떨어진다
    3. 평균 잊음은 대개 40~60%이다
    4. 이는 신경망이 그대로 두면 파국적으로 잊는다는 것을 보여 준다
    
    핵심 요점:
    보통의 신경망 학습은 이어 배우기에 **알맞지 않다**.
    파국적 잊음을 막으려면 특별한 기법이 필요하다!
    """
    
    print("=" * 70)
    print("CATASTROPHIC FORGETTING DEMONSTRATION")
    print("=" * 70)
    print("\nThis script demonstrates how neural networks forget previous")
    print("tasks when learning new tasks sequentially.")
    print("\nWe'll train on Split MNIST (5 tasks, 2 classes each) and watch")
    print("how performance on earlier tasks degrades dramatically.")
    print("=" * 70)
    
    # 보여 주기를 돌린다
    accuracy_matrix = demonstrate_catastrophic_forgetting(
        num_tasks=5,
        epochs_per_task=5,
        batch_size=128
    )
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("\nCatastrophic forgetting is a fundamental problem in neural networks!")
    print("As we train on new tasks, the model's weights are updated to minimize")
    print("the new task's loss, which destroys information needed for old tasks.")
    print("\nIn the following scripts, we'll explore methods to prevent this:")
    print("  - Script 02: Naive Sequential Learning (baseline)")
    print("  - Script 03: Simple Experience Replay")
    print("  - Intermediate scripts: EWC, LWF, and more!")
    print("=" * 70)```

## 논의

`SimpleNetwork` 클래스는 파이토치의 `nn.Module` 인터페이스로 모델 구조를 감싼다. `forward` 메서드가 계산 그래프를 정하여, 학습 중에 파이토치의 자동 미분 체계가 기울기 계산을 알아서 하게 한다. 이 모듈 방식의 설계 덕분에 낱낱의 부품을 고치거나 모델을 더 큰 파이프라인에 끼워 넣기가 쉽다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 본새는 더 복잡한 상황으로도 자연스럽게 넓어진다. 초매개변수, 구조의 변형, 여러 데이터셋을 두고 실험해 보면 이해가 깊어지고 평생 학습 과제에 대한 실전 감각이 쌓인다.

## 연습문제

**연습문제 1.**
기본 초기화 상태의 `SimpleNetwork`에 든 학습 가능한 매개변수의 총 개수를 셈하라. 가중치와 편향을 모두 넣어 층별로 나누어 보여라.

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
`SimpleNetwork`이 층이나 블록의 개수를 설정할 수 있도록 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`를 써서 깊이를 바꿀 수 있는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 되풀이하라. (그냥 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 파이토치가 최적화 대상 매개변수를 모두 등록한다. `for n in [2, 4, 8]: model = SimpleNetwork(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`로 시험하라.
