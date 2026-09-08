# 57모듈: 이어 배우기

57모듈: 이어 배우기 - 초급 파일 2: 소박한 차례 학습(밑금)

이어 배우기는 앞서 배운 앎을 잊지 않으면서 잇단 과제로 모델을 익히는 어려움을 다룬다. 이 구현은 말랑함을 지키면서 파국적 잊음을 누그러뜨리는 핵심 전략을 보여 준다.

## 1. 코드

```python
"""
57모듈: 이어 배우기 - 초급
파일 2: 소박한 차례 학습(밑금)

이 스크립트는 이어 배우기 실험의 밑금이 되는
소박한 차례 학습 방법을 구현한다. 잊음을 수로 나타내는
제대로 된 평가 지표도 함께 구현한다.

학습 목표:
1. 제대로 된 이어 배우기 평가 규약을 구현한다
2. 표준 이어 배우기 지표를 셈한다
3. 방법을 견줄 밑금을 이해한다
4. 이어 배우기 실험을 어떻게 짜는지 배운다

수학적 지표:
1. 평균 정확도(AA): (1/T) Σ Acc_{i,T}
2. 뒤로의 옮김(BWT): (1/(T-1)) Σ (Acc_{i,T} - Acc_{i,i})
3. 앞으로의 옮김(FWT): (1/(T-1)) Σ (Acc_{i,i-1} - Acc_{i,init})
4. 배움 정확도(LA): (1/T) Σ Acc_{i,i}
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
import time
from dataclasses import dataclass

# ========================================================================
# 메인
# ========================================================================


@dataclass
class ContinualLearningMetrics:
    """
    이어 배우기 지표를 모두 담는 데이터 클래스.
    
    속성:
        average_accuracy: 마지막에 모든 과제에 걸친 평균 정확도
        backward_transfer: 잊음의 재기(음수면 잊었다는 뜻)
        forward_transfer: 새 과제로 앎이 옮겨 간 정도
        learning_accuracy: 막 배운 직후의 평균 정확도
        forgetting_per_task: 과제마다의 잊음 정도
        accuracy_matrix: 모든 정확도를 담은 온전한 행렬
    """
    average_accuracy: float
    backward_transfer: float
    forward_transfer: float
    learning_accuracy: float
    forgetting_per_task: List[float]
    accuracy_matrix: np.ndarray


class ContinualLearner:
    """
    이어 배우기 실험의 바탕 클래스.
    
    이 클래스는 다음을 준다.
    - 표준 평가 규약
    - 지표 셈하기
    - 실험 좇기
    - 시각화 도구
    
    하위 클래스가 특정 메서드를 갈아 끼워 서로 다른
    이어 배우기 전략을 구현할 수 있다.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: torch.device,
                 learning_rate: float = 0.001):
        """
        이어 배우는 학습기를 초기화한다.
        
        인수:
            model: 신경망 모델
            device: 익힐 장치(CPU나 GPU)
            learning_rate: 최적화기의 학습률
        """
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        
        # 좇을 변수
        self.current_task = 0
        self.task_train_loaders = []
        self.task_test_loaders = []
        self.accuracy_matrix = None
        
    def train_task(self, 
                   train_loader: DataLoader,
                   task_id: int,
                   epochs: int = 5) -> List[float]:
        """
        과제 하나로 익힌다.
        
        이것이 소박한 방법이다. 그저 지금 과제의 데이터로
        보통의 지도 학습을 한다. 잊음을 막을 특별한 기법은
        쓰지 않는다.
        
        인수:
            train_loader: 지금 과제의 DataLoader
            task_id: 지금 과제의 번호
            epochs: 학습 에포크 수
        
        반환값:
            시대마다의 손실 목록
        """
        # 이 과제에 쓸 새 최적화기를 만든다
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.model.train()
        losses = []
        
        print(f"\n{'=' * 60}")
        print(f"Training Task {task_id}")
        print('=' * 60)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                # 장치로 옮긴다
                data, target = data.to(self.device), target.to(self.device)
                
                # 경사 초기화
                optimizer.zero_grad()
                
                # 순전파
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # 역전파
                loss.backward()
                optimizer.step()
                
                # 통계 기록
                epoch_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            
            # 지표를 계산한다
            avg_loss = epoch_loss / len(train_loader)
            accuracy = 100.0 * correct / total
            losses.append(avg_loss)
            
            print(f"  Epoch {epoch + 1}/{epochs} - "
                  f"Loss: {avg_loss:.4f}, "
                  f"Train Acc: {accuracy:.2f}%")
        
        return losses
    
    def evaluate_task(self, 
                     test_loader: DataLoader,
                     task_id: int) -> Tuple[float, float]:
        """
        과제 하나에서 모델을 평가한다.
        
        인수:
            test_loader: 시험 데이터의 DataLoader
            task_id: 평가할 과제의 번호
        
        반환값:
            (정확도, 손실) 짝
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(test_loader)
        
        return accuracy, avg_loss
    
    def train_continual(self,
                       train_loaders: List[DataLoader],
                       test_loaders: List[DataLoader],
                       epochs_per_task: int = 5) -> ContinualLearningMetrics:
        """
        이어 배우기의 주 되돌이.
        
        과정:
        1. 과제마다 차례로 다음을 한다.
           a. 과제 데이터로 익힌다
           b. 모든 과제(앞선 것까지)에서 평가한다
           c. 정확도를 적는다
        2. 이어 배우기 지표를 셈한다
        3. 두루 갖춘 결과를 되돌린다
        
        인수:
            train_loaders: 학습용 DataLoader 목록
            test_loaders: 시험용 DataLoader 목록
            epochs_per_task: 과제마다 익힐 시대 수
        
        반환값:
            지표를 모두 담은 ContinualLearningMetrics 객체
        """
        num_tasks = len(train_loaders)
        self.task_train_loaders = train_loaders
        self.task_test_loaders = test_loaders
        
        # 정확도 행렬을 초기화한다
        # accuracy_matrix[i,j] = 과제 j까지 익힌 뒤 과제 i의 정확도
        self.accuracy_matrix = np.zeros((num_tasks, num_tasks))
        
        # 처음 무작위 정확도를 좇는다(익히기 전)
        print("\n" + "=" * 70)
        print("INITIAL EVALUATION (Before Any Training)")
        print("=" * 70)
        for task_id in range(num_tasks):
            acc, _ = self.evaluate_task(test_loaders[task_id], task_id)
            print(f"Task {task_id}: {acc:.2f}% (random guess ~ {100/2:.1f}%)")
        
        # 과제마다 차례로 익힌다
        for task_id in range(num_tasks):
            # 지금 과제로 익힌다
            self.train_task(
                train_loader=train_loaders[task_id],
                task_id=task_id,
                epochs=epochs_per_task
            )
            
            # 지금까지 본 모든 과제에서 평가한다
            print(f"\n{'=' * 60}")
            print(f"Evaluation after Task {task_id}")
            print('=' * 60)
            
            for eval_task_id in range(task_id + 1):
                acc, loss = self.evaluate_task(
                    test_loader=test_loaders[eval_task_id],
                    task_id=eval_task_id
                )
                
                # 정확도 행렬에 담는다
                self.accuracy_matrix[eval_task_id, task_id] = acc
                
                # 해당하면 잊음 정보와 함께 찍는다
                if eval_task_id == task_id:
                    print(f"  Task {eval_task_id}: {acc:.2f}% (just learned)")
                elif eval_task_id < task_id:
                    original_acc = self.accuracy_matrix[eval_task_id, eval_task_id]
                    forgetting = original_acc - acc
                    print(f"  Task {eval_task_id}: {acc:.2f}% "
                          f"(was {original_acc:.2f}%, forgot {forgetting:.2f}%)")
        
        # 지표를 셈해 되돌린다
        metrics = self.calculate_metrics()
        return metrics
    
    def calculate_metrics(self) -> ContinualLearningMetrics:
        """
        정확도 행렬에서 이어 배우기 지표를 모두 셈한다.
        
        지표 풀이:
        
        1. 평균 정확도(AA):
           모든 과제를 배운 뒤 모든 과제의 평균 정확도
           AA = (1/T) Σ_{i=1}^T Acc_{i,T}
        
        2. 뒤로의 옮김(BWT):
           잊음을 잰다. 곧 옛 과제의 정확도가 얼마나 바뀌었는가
           BWT = (1/(T-1)) Σ_{i=1}^{T-1} (Acc_{i,T} - Acc_{i,i})
           BWT가 음수면 잊음, 양수면 나아짐
        
        3. 앞으로의 옮김(FWT):
           지난 앎을 새 과제에 쓰는 힘을 잰다
           과제 i을 익히기 전 그 과제의 정확도를 잰다
           FWT = (1/(T-1)) Σ_{i=2}^T (Acc_{i,i-1} - 밑금)
        
        4. 배움 정확도(LA):
           과제를 막 배운 직후의 평균 정확도
           LA = (1/T) Σ_{i=1}^T Acc_{i,i}
        
        반환값:
            ContinualLearningMetrics 객체
        """
        num_tasks = self.accuracy_matrix.shape[0]
        
        # 1. 평균 정확도(마지막 열)
        final_accuracies = self.accuracy_matrix[:, -1]
        average_accuracy = np.mean(final_accuracies)
        
        # 2. 뒤로의 옮김(잊음)
        # 마지막 정확도를 막 배운 직후의 정확도와 견준다
        backward_transfer = 0.0
        forgetting_per_task = []
        
        for i in range(num_tasks - 1):  # 마지막 과제는 뺀다(잊을 틈이 없다)
            initial_acc = self.accuracy_matrix[i, i]  # 막 배운 직후
            final_acc = self.accuracy_matrix[i, num_tasks - 1]  # 모든 과제를 마친 뒤
            forgetting = final_acc - initial_acc  # 음수면 잊었다는 뜻
            backward_transfer += forgetting
            forgetting_per_task.append(forgetting)
        
        if num_tasks > 1:
            backward_transfer /= (num_tasks - 1)
        
        # 3. 앞으로의 옮김(소박한 학습에는 해당 없음)
        # 소박한 학습에는 앞으로 옮길 장치가 없다
        # 그러려면 과제 i을 익히기 전에 그 과제에서 평가해야 한다
        forward_transfer = 0.0  # 소박한 밑금을 위한 자리 채우개
        
        # 4. 배움 정확도(행렬의 대각선)
        learning_accuracy = np.mean(np.diag(self.accuracy_matrix))
        
        return ContinualLearningMetrics(
            average_accuracy=average_accuracy,
            backward_transfer=backward_transfer,
            forward_transfer=forward_transfer,
            learning_accuracy=learning_accuracy,
            forgetting_per_task=forgetting_per_task,
            accuracy_matrix=self.accuracy_matrix
        )


def create_simple_model(input_size: int = 784,
                       hidden_size: int = 256,
                       num_classes: int = 2) -> nn.Module:
    """
    단순한 앞먹임 망을 만든다.
    
    인수:
        input_size: 입력 차원
        hidden_size: 숨은 층의 크기
        num_classes: 출력 클래스의 수
    
    반환값:
        파이토치 모델
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, num_classes)
    )
    return model


def create_split_mnist_tasks(num_tasks: int = 5) -> List[List[int]]:
    """Split MNIST 과제 설정을 만든다."""
    all_digits = list(range(10))
    classes_per_task = 10 // num_tasks
    
    tasks = []
    for i in range(num_tasks):
        task_classes = all_digits[i * classes_per_task:(i + 1) * classes_per_task]
        tasks.append(task_classes)
    
    return tasks


def create_task_dataset(full_dataset, task_classes: List[int]) -> TensorDataset:
    """부류를 걸러 특정 과제의 데이터셋을 만든다."""
    indices = []
    for idx in range(len(full_dataset)):
        _, label = full_dataset[idx]
        if label in task_classes:
            indices.append(idx)
    
    subset = Subset(full_dataset, indices)
    
    data_list = []
    label_list = []
    
    for idx in range(len(subset)):
        img, label = subset[idx]
        data_list.append(img)
        new_label = task_classes.index(label)
        label_list.append(new_label)
    
    data_tensor = torch.stack(data_list)
    label_tensor = torch.tensor(label_list, dtype=torch.long)
    
    return TensorDataset(data_tensor, label_tensor)


def print_metrics_summary(metrics: ContinualLearningMetrics, num_tasks: int):
    """
    이어 배우기 지표를 두루 간추려 찍는다.
    
    인수:
        metrics: ContinualLearningMetrics 객체
        num_tasks: 과제 개수
    """
    print("\n" + "=" * 70)
    print("CONTINUAL LEARNING METRICS SUMMARY")
    print("=" * 70)
    
    print(f"\n📊 Key Metrics:")
    print(f"   Average Accuracy (AA):    {metrics.average_accuracy:.2f}%")
    print(f"   Learning Accuracy (LA):   {metrics.learning_accuracy:.2f}%")
    print(f"   Backward Transfer (BWT):  {metrics.backward_transfer:.2f}%")
    print(f"   Forward Transfer (FWT):   {metrics.forward_transfer:.2f}%")
    
    print(f"\n📉 Forgetting Analysis:")
    for i, forgetting in enumerate(metrics.forgetting_per_task):
        status = "✓" if forgetting >= 0 else "✗"
        print(f"   Task {i}: {forgetting:+.2f}% {status}")
    
    print(f"\n📈 Final Accuracy per Task:")
    final_accs = metrics.accuracy_matrix[:, -1]
    for i, acc in enumerate(final_accs):
        print(f"   Task {i}: {acc:.2f}%")
    
    print(f"\n📋 Accuracy Matrix:")
    print(f"   (Rows = Tasks, Columns = After Training Stage)")
    print("   " + "-" * 60)
    print("   Task |", end="")
    for j in range(num_tasks):
        print(f"  T{j}  |", end="")
    print()
    print("   " + "-" * 60)
    
    for i in range(num_tasks):
        print(f"    {i}   |", end="")
        for j in range(num_tasks):
            if j >= i:
                print(f" {metrics.accuracy_matrix[i, j]:4.1f} |", end="")
            else:
                print(f"  --  |", end="")
        print()


def visualize_metrics(metrics: ContinualLearningMetrics, num_tasks: int):
    """
    이어 배우기 결과를 두루 그려 본다.
    
    인수:
        metrics: ContinualLearningMetrics 객체
        num_tasks: 과제 개수
    """
    fig = plt.figure(figsize=(18, 5))
    
    # 그림 1: 정확도 행렬 열 지도
    ax1 = plt.subplot(1, 3, 1)
    im = ax1.imshow(metrics.accuracy_matrix, cmap='RdYlGn', 
                    aspect='auto', vmin=0, vmax=100)
    plt.colorbar(im, ax=ax1, label='Accuracy (%)')
    ax1.set_xlabel('After Training Task', fontsize=11)
    ax1.set_ylabel('Evaluated Task', fontsize=11)
    ax1.set_title('Accuracy Matrix', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(num_tasks))
    ax1.set_yticks(range(num_tasks))
    
    # 글자 주석을 추가한다
    for i in range(num_tasks):
        for j in range(num_tasks):
            if j >= i:
                ax1.text(j, i, f'{metrics.accuracy_matrix[i, j]:.1f}',
                        ha="center", va="center", color="black", fontsize=9)
    
    # 그림 2: 과제별 잊음
    ax2 = plt.subplot(1, 3, 2)
    tasks = list(range(len(metrics.forgetting_per_task)))
    colors = ['red' if f < 0 else 'green' for f in metrics.forgetting_per_task]
    
    bars = ax2.bar(tasks, metrics.forgetting_per_task, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Task', fontsize=11)
    ax2.set_ylabel('Backward Transfer (%)', fontsize=11)
    ax2.set_title('Forgetting per Task', fontsize=12, fontweight='bold')
    ax2.set_xticks(tasks)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 막대에 값 이름표를 추가한다
    for i, (bar, val) in enumerate(zip(bars, metrics.forgetting_per_task)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', 
                va='bottom' if val >= 0 else 'top', fontsize=9)
    
    # 그림 3: 배움 정확도와 마지막 정확도
    ax3 = plt.subplot(1, 3, 3)
    tasks = list(range(num_tasks))
    learning_accs = np.diag(metrics.accuracy_matrix)
    final_accs = metrics.accuracy_matrix[:, -1]
    
    x = np.arange(num_tasks)
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, learning_accs, width, 
                    label='Learning Accuracy', color='skyblue', alpha=0.8)
    bars2 = ax3.bar(x + width/2, final_accs, width,
                    label='Final Accuracy', color='coral', alpha=0.8)
    
    ax3.set_xlabel('Task', fontsize=11)
    ax3.set_ylabel('Accuracy (%)', fontsize=11)
    ax3.set_title('Learning vs Final Accuracy', fontsize=12, fontweight='bold')
    ax3.set_xticks(tasks)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 105])
    
    # 값 이름표를 추가한다
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('naive_sequential_learning_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'naive_sequential_learning_results.png'")
    plt.show()


def main():
    """
    소박한 차례 학습 밑금을 돌리는 주 함수.
    
    이는 뒤이은 스크립트에서 이어 배우기 방법을 견줄
    밑금 성능을 세운다.
    """
    print("=" * 70)
    print("NAIVE SEQUENTIAL LEARNING BASELINE")
    print("=" * 70)
    print("\nThis script implements the naive baseline for continual learning:")
    print("  - Train on tasks sequentially")
    print("  - No special techniques to prevent forgetting")
    print("  - Comprehensive evaluation metrics")
    print("\nThis serves as the baseline to beat with continual learning methods.")
    print("=" * 70)
    
    # 설정
    num_tasks = 5
    epochs_per_task = 5
    batch_size = 128
    learning_rate = 0.001
    
    # 장치
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # MNIST 불러오기
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # 과제를 만든다
    task_classes = create_split_mnist_tasks(num_tasks)
    print(f"\nTask Configuration:")
    for i, classes in enumerate(task_classes):
        print(f"  Task {i}: Classes {classes}")
    
    # 데이터로더들을 만든다
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
    
    # 모델 생성
    num_classes_per_task = len(task_classes[0])
    model = create_simple_model(
        input_size=784,
        hidden_size=256,
        num_classes=num_classes_per_task
    ).to(device)
    
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 이어 배우는 학습기를 만든다
    learner = ContinualLearner(
        model=model,
        device=device,
        learning_rate=learning_rate
    )
    
    # 이어 배우기를 돌린다
    start_time = time.time()
    metrics = learner.train_continual(
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        epochs_per_task=epochs_per_task
    )
    total_time = time.time() - start_time
    
    # 결과 출력
    print_metrics_summary(metrics, num_tasks)
    
    print(f"\n⏱️  Total Training Time: {total_time:.2f} seconds")
    
    # 시각화한다
    visualize_metrics(metrics, num_tasks)
    
    print("\n" + "=" * 70)
    print("BASELINE ESTABLISHED")
    print("=" * 70)
    print("\nThis naive approach shows significant catastrophic forgetting.")
    print("In the next scripts, we'll implement continual learning methods")
    print("that preserve knowledge of previous tasks while learning new ones:")
    print("  - Script 03: Experience Replay")
    print("  - Intermediate: EWC, LWF, Synaptic Intelligence, etc.")
    print("=" * 70)


if __name__ == "__main__":
    main()```

## 2. 논의

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 본새는 더 복잡한 상황으로도 자연스럽게 넓어진다. 초매개변수, 구조의 변형, 여러 데이터셋을 두고 실험해 보면 이해가 깊어지고 평생 학습 과제에 대한 실전 감각이 쌓인다.

## 연습문제

**연습문제 1.**
학습 루프에서 `optimizer.zero_grad()` 호출을 없애면 어떤 일이 일어나는지 설명하라. 고친 코드를 실행하고 학습 손실의 수렴에 미치는 영향을 서술하라.

??? success "연습문제 1 풀이"
    `optimizer.zero_grad()`가 없으면 PyTorch가 새 경사를 기존 `.grad` 텐서에 덮어쓰지 않고 더하기 때문에 반복에 걸쳐 경사가 누적된다. 이는 사실상 학습률에 누적된 단계 수를 곱하는 셈이어서 최적화가 점점 크고 불규칙한 걸음을 내딛게 된다. 학습 손실은 매끄럽게 수렴하는 대신 심하게 진동하거나 발산한다. 해결책은 간단하다. `loss.backward()`를 호출하기 전에 언제나 경사를 0으로 만들어라.

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
조기 종료를 구현하라. 매 에폭 후 검증 손실을 추적하고, 10 에폭 연속으로 개선이 없으면 학습을 멈춘다. 가장 좋은 모델 가중치를 저장하고 복원하라.

??? success "연습문제 4 풀이"
    인내 횟수 카운터와 최저 손실 추적기를 추가한다.
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
    이렇게 하면 따로 떼어 둔 데이터에서 모델이 더 나아지지 않을 때 멈추므로 과적합을 막을 수 있다.

## 정리하며

**다룬 것** — 57모듈: 이어 배우기

학습 루프는 표준적인 PyTorch 패턴을 따른다.

핵심 클래스는 `ContinualLearningMetrics`, `ContinualLearner`이며 앞의 연습문제 4개로 직접 확인할 수 있다.
