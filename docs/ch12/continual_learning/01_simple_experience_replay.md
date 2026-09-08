# 57모듈: 이어 배우기

57모듈: 이어 배우기 - 초급 파일 3: 단순 경험 되살리기

이어 배우기는 앞서 배운 앎을 잊지 않으면서 잇단 과제로 모델을 익히는 어려움을 다룬다. 이 구현은 말랑함을 지키면서 파국적 잊음을 누그러뜨리는 핵심 전략을 보여 준다.

## 1. 코드

```python
"""
57모듈: 이어 배우기 - 초급
파일 3: 단순 경험 되살리기

이 스크립트는 가장 단순한 이어 배우기 기법인 경험 되살리기를 구현한다.
앞선 보기를 담은 작은 버퍼를 두고, 새 과제를 익히는 동안 그것을 되살려
잊음을 막자는 생각이다.

학습 목표:
1. 경험 되살리기의 개념을 이해한다
2. 무작위 뽑기를 갖춘 기억 버퍼를 구현한다
3. 소박한 밑금보다 얼마나 나아졌는지 잰다
4. 기억과 효율의 맞바꿈을 배운다

수식:
과제 τ에서 손실은 다음이 된다.
L_total = L_current + L_replay

여기서 각 기호는 다음과 같다.
- L_current = 지금 과제 데이터의 손실
- L_replay = 앞선 과제에서 되살린 보기의 손실

이렇게 단순히 섞으면 학습 중에 옛 과제의 기울기가 "살아 있어"
파국적 잊음이 막힌다.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import random
from collections import defaultdict
import time

# ========================================================================
# 메인
# ========================================================================


class MemoryBuffer:
    """
    지난 보기를 담아 두고 뽑아 쓰는 단순한 기억 버퍼.
    
    다음을 갖춘 기본 에피소드 기억을 구현한다.
    1. (입력, 이름표, 과제 번호) 짝을 담는다
    2. 되살리기를 위한 무작위 뽑기를 받쳐 준다
    3. 최대 용량이 붙박여 있다
    
    버퍼가 차면 새 보기가 옛 보기를 아무렇게나 갈아 끼운다
    (더 고르게 담으려면 저수지 뽑기를 쓸 수도 있다).
    """
    
    def __init__(self, max_size: int = 1000):
        """
        기억 버퍼를 초기화한다.
        
        인수:
            max_size: 담아 둘 보기의 최대 개수
        """
        self.max_size = max_size
        self.data = []
        self.labels = []
        self.task_ids = []
        
    def add_examples(self, 
                    data: torch.Tensor, 
                    labels: torch.Tensor,
                    task_id: int):
        """
        기억 버퍼에 보기를 더한다.
        
        버퍼가 차면 옛 보기를 아무렇게나 갈아 끼운다.
        이는 단순한 전략이며 더 정교한 방법도 있다.
        
        인수:
            data: 입력 데이터 텐서 (batch_size, ...)
            labels: 이름표 텐서 (batch_size,)
            task_id: 이 보기가 속한 과제의 번호
        """
        batch_size = data.size(0)
        
        for i in range(batch_size):
            example = data[i].cpu()
            label = labels[i].cpu()
            
            if len(self.data) < self.max_size:
                # 버퍼가 차지 않았으니 그냥 덧붙인다
                self.data.append(example)
                self.labels.append(label)
                self.task_ids.append(task_id)
            else:
                # 버퍼가 찼으니 무작위로 갈아 끼운다(저수지 뽑기 방식)
                # 그래야 뽑힐 확률이 고르다
                idx = random.randint(0, self.max_size - 1)
                self.data[idx] = example
                self.labels[idx] = label
                self.task_ids[idx] = task_id
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        기억 버퍼에서 배치를 아무렇게나 뽑는다.
        
        인수:
            batch_size: 뽑을 보기의 개수
        
        반환값:
            (데이터, 이름표, 과제 번호) 짝
        """
        if len(self.data) == 0:
            # 버퍼가 비었으니 빈 텐서를 되돌린다
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        
        # 되돌려 놓고 뽑는다(되돌리지 않고 뽑을 수도 있다)
        sample_size = min(batch_size, len(self.data))
        indices = random.sample(range(len(self.data)), sample_size)
        
        # 뽑은 보기를 모은다
        sampled_data = torch.stack([self.data[i] for i in indices])
        sampled_labels = torch.tensor([self.labels[i] for i in indices], 
                                     dtype=torch.long)
        sampled_task_ids = torch.tensor([self.task_ids[i] for i in indices],
                                       dtype=torch.long)
        
        return sampled_data, sampled_labels, sampled_task_ids
    
    def __len__(self) -> int:
        """버퍼에 든 보기의 개수를 되돌린다."""
        return len(self.data)
    
    def get_stats(self) -> Dict[int, int]:
        """
        버퍼가 어떻게 짜였는지 통계를 얻는다.
        
        반환값:
            과제 번호를 보기 개수로 옮기는 사전
        """
        stats = defaultdict(int)
        for task_id in self.task_ids:
            stats[task_id] += 1
        return dict(stats)


class ExperienceReplayLearner:
    """
    경험 되살리기를 쓰는 이어 배우는 학습기.
    
    학습 절차:
    1. 지금 과제 데이터의 배치마다 다음을 한다.
       a. 기억에서 되살릴 배치를 뽑는다
       b. 지금 배치의 손실을 셈한다
       c. 되살린 배치의 손실을 셈한다
       d. 합친 손실을 거꾸로 퍼뜨린다
       e. 가중치를 고친다
    2. 과제를 익힌 뒤 보기 얼마를 기억 버퍼에 더한다
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 memory_size: int = 1000,
                 examples_per_task: int = 200,
                 learning_rate: float = 0.001):
        """
        경험 되살리기 학습기를 초기화한다.
        
        인수:
            model: 신경망 모델
            device: 학습에 쓸 장치
            memory_size: 기억 버퍼의 전체 크기
            examples_per_task: 과제마다 담아 둘 보기의 개수
            learning_rate: 최적화기의 학습률
        """
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        
        # 기억 버퍼를 초기화한다
        self.memory = MemoryBuffer(max_size=memory_size)
        self.examples_per_task = examples_per_task
        
        # 추적
        self.accuracy_matrix = None
        self.task_train_loaders = []
        self.task_test_loaders = []
    
    def populate_memory(self, 
                       train_loader: DataLoader,
                       task_id: int,
                       num_examples: int):
        """
        지금 과제의 보기로 기억 버퍼를 채운다.
        
        전략: 학습 집합에서 보기를 아무렇게나 고른다.
        더 정교한 전략(몰이, 기울기 기반 고르기)을 쓰면
        성능이 더 좋아질 수 있다.
        
        인수:
            train_loader: 지금 과제의 DataLoader
            task_id: 지금 과제의 번호
            num_examples: 기억에 더할 보기의 개수
        """
        print(f"\n  Adding {num_examples} examples to memory buffer...")
        
        # 지금 과제의 보기를 모두 모은다
        all_data = []
        all_labels = []
        
        for data, labels in train_loader:
            all_data.append(data)
            all_labels.append(labels)
        
        # 이어 붙인다
        all_data = torch.cat(all_data, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # num_examples개를 아무렇게나 고른다
        num_available = all_data.size(0)
        num_to_select = min(num_examples, num_available)
        
        indices = torch.randperm(num_available)[:num_to_select]
        selected_data = all_data[indices]
        selected_labels = all_labels[indices]
        
        # 기억 버퍼에 더한다
        self.memory.add_examples(selected_data, selected_labels, task_id)
        
        # 버퍼 통계를 찍는다
        stats = self.memory.get_stats()
        print(f"  Memory buffer: {len(self.memory)}/{self.memory.max_size} examples")
        print(f"  Distribution: {stats}")
    
    def train_task(self,
                  train_loader: DataLoader,
                  task_id: int,
                  epochs: int = 5) -> List[float]:
        """
        경험 되살리기를 곁들여 과제 하나로 익힌다.
        
        소박한 학습과의 핵심 차이:
        - 배치마다 되살릴 보기도 뽑아 함께 익힌다
        - 그러면 옛 과제의 기울기가 지켜져 잊음이 막힌다
        
        인수:
            train_loader: 지금 과제의 DataLoader
            task_id: 지금 과제의 번호
            epochs: 학습 에포크 수
        
        반환값:
            시대마다의 손실 목록
        """
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.model.train()
        losses = []
        
        print(f"\n{'=' * 60}")
        print(f"Training Task {task_id} with Experience Replay")
        print('=' * 60)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            current_loss_sum = 0.0
            replay_loss_sum = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                # 지금 배치를 장치로 옮긴다
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                # ===== 지금 과제의 손실 =====
                output = self.model(data)
                current_loss = self.criterion(output, target)
                
                # ===== 되살리기 손실 =====
                replay_loss = torch.tensor(0.0).to(self.device)
                
                if len(self.memory) > 0:
                    # 기억 버퍼에서 뽑는다
                    replay_data, replay_labels, _ = self.memory.sample(
                        batch_size=data.size(0)
                    )
                    
                    if replay_data.size(0) > 0:
                        replay_data = replay_data.to(self.device)
                        replay_labels = replay_labels.to(self.device)
                        
                        # 되살리기 손실을 셈한다
                        replay_output = self.model(replay_data)
                        replay_loss = self.criterion(replay_output, replay_labels)
                
                # ===== 합친 손실 =====
                # 단순한 무게 합침(다른 무게를 쓸 수도 있다)
                total_loss = current_loss + replay_loss
                
                # 합친 손실에 대해 되돌린다
                total_loss.backward()
                optimizer.step()
                
                # 통계 기록
                epoch_loss += total_loss.item()
                current_loss_sum += current_loss.item()
                replay_loss_sum += replay_loss.item()
                
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            
            # 지표를 계산한다
            avg_loss = epoch_loss / len(train_loader)
            avg_current = current_loss_sum / len(train_loader)
            avg_replay = replay_loss_sum / len(train_loader)
            accuracy = 100.0 * correct / total
            losses.append(avg_loss)
            
            print(f"  Epoch {epoch + 1}/{epochs} - "
                  f"Total Loss: {avg_loss:.4f}, "
                  f"Current: {avg_current:.4f}, "
                  f"Replay: {avg_replay:.4f}, "
                  f"Train Acc: {accuracy:.2f}%")
        
        return losses
    
    def evaluate_task(self,
                     test_loader: DataLoader,
                     task_id: int) -> Tuple[float, float]:
        """과제에서 모델을 평가한다."""
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
                       epochs_per_task: int = 5):
        """
        경험 되살리기를 쓰는 이어 배우기의 주 되돌이.
        
        과제마다의 절차:
        1. 되살리기를 곁들여 익힌다
        2. 이 과제의 보기로 기억 버퍼를 채운다
        3. 모든 과제에서 평가한다
        """
        num_tasks = len(train_loaders)
        self.task_train_loaders = train_loaders
        self.task_test_loaders = test_loaders
        
        self.accuracy_matrix = np.zeros((num_tasks, num_tasks))
        
        print("\n" + "=" * 70)
        print("CONTINUAL LEARNING WITH EXPERIENCE REPLAY")
        print("=" * 70)
        print(f"Memory Buffer Size: {self.memory.max_size}")
        print(f"Examples per Task: {self.examples_per_task}")
        
        for task_id in range(num_tasks):
            # 지금 과제로 익힌다
            self.train_task(
                train_loader=train_loaders[task_id],
                task_id=task_id,
                epochs=epochs_per_task
            )
            
            # 이 과제의 보기로 기억을 채운다
            self.populate_memory(
                train_loader=train_loaders[task_id],
                task_id=task_id,
                num_examples=self.examples_per_task
            )
            
            # 모든 과제에서 평가한다
            print(f"\n{'=' * 60}")
            print(f"Evaluation after Task {task_id}")
            print('=' * 60)
            
            for eval_task_id in range(task_id + 1):
                acc, loss = self.evaluate_task(
                    test_loader=test_loaders[eval_task_id],
                    task_id=eval_task_id
                )
                
                self.accuracy_matrix[eval_task_id, task_id] = acc
                
                if eval_task_id == task_id:
                    print(f"  Task {eval_task_id}: {acc:.2f}% (just learned)")
                elif eval_task_id < task_id:
                    original_acc = self.accuracy_matrix[eval_task_id, eval_task_id]
                    forgetting = original_acc - acc
                    print(f"  Task {eval_task_id}: {acc:.2f}% "
                          f"(was {original_acc:.2f}%, change {forgetting:+.2f}%)")
        
        return self.calculate_metrics()
    
    def calculate_metrics(self):
        """이어 배우기 지표를 셈한다."""
        num_tasks = self.accuracy_matrix.shape[0]
        
        # 평균 정확도
        average_accuracy = np.mean(self.accuracy_matrix[:, -1])
        
        # 뒤로의 옮김(잊음)
        backward_transfer = 0.0
        forgetting_per_task = []
        
        for i in range(num_tasks - 1):
            initial_acc = self.accuracy_matrix[i, i]
            final_acc = self.accuracy_matrix[i, num_tasks - 1]
            forgetting = final_acc - initial_acc
            backward_transfer += forgetting
            forgetting_per_task.append(forgetting)
        
        if num_tasks > 1:
            backward_transfer /= (num_tasks - 1)
        
        # 배움 정확도
        learning_accuracy = np.mean(np.diag(self.accuracy_matrix))
        
        return {
            'average_accuracy': average_accuracy,
            'backward_transfer': backward_transfer,
            'learning_accuracy': learning_accuracy,
            'forgetting_per_task': forgetting_per_task,
            'accuracy_matrix': self.accuracy_matrix
        }


def create_simple_model(input_size: int = 784,
                       hidden_size: int = 256,
                       num_classes: int = 2) -> nn.Module:
    """단순한 앞먹임 망을 만든다."""
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
    """특정 과제의 데이터셋을 만든다."""
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


def visualize_comparison(replay_metrics: dict, 
                        baseline_metrics: dict,
                        num_tasks: int):
    """
    경험 되살리기를 밑금과 견준다.
    
    인수:
        replay_metrics: 경험 되살리기의 지표
        baseline_metrics: 소박한 밑금의 지표
        num_tasks: 과제 개수
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 그림 1: 정확도 행렬 견줌
    ax1 = axes[0]
    replay_acc = replay_metrics['accuracy_matrix']
    x = np.arange(num_tasks)
    width = 0.35
    
    replay_final = replay_acc[:, -1]
    baseline_final = baseline_metrics['accuracy_matrix'][:, -1]
    
    bars1 = ax1.bar(x - width/2, baseline_final, width, 
                    label='Naive Baseline', color='coral', alpha=0.8)
    bars2 = ax1.bar(x + width/2, replay_final, width,
                    label='Experience Replay', color='skyblue', alpha=0.8)
    
    ax1.set_xlabel('Task', fontsize=11)
    ax1.set_ylabel('Final Accuracy (%)', fontsize=11)
    ax1.set_title('Final Accuracy Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 105])
    
    # 값 이름표를 추가한다
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    # 그림 2: 잊음 견줌
    ax2 = axes[1]
    
    replay_forgetting = replay_metrics['forgetting_per_task']
    baseline_forgetting = baseline_metrics['forgetting_per_task']
    
    x = np.arange(len(replay_forgetting))
    
    bars1 = ax2.bar(x - width/2, baseline_forgetting, width,
                    label='Naive Baseline', color='coral', alpha=0.8)
    bars2 = ax2.bar(x + width/2, replay_forgetting, width,
                    label='Experience Replay', color='skyblue', alpha=0.8)
    
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Task', fontsize=11)
    ax2.set_ylabel('Backward Transfer (%)', fontsize=11)
    ax2.set_title('Forgetting Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 그림 3: 지표 간추림
    ax3 = axes[2]
    
    metrics_names = ['Avg Accuracy', 'Backward Transfer', 'Learning Accuracy']
    replay_values = [
        replay_metrics['average_accuracy'],
        replay_metrics['backward_transfer'],
        replay_metrics['learning_accuracy']
    ]
    baseline_values = [
        baseline_metrics['average_accuracy'],
        baseline_metrics['backward_transfer'],
        baseline_metrics['learning_accuracy']
    ]
    
    x = np.arange(len(metrics_names))
    
    bars1 = ax3.bar(x - width/2, baseline_values, width,
                    label='Naive Baseline', color='coral', alpha=0.8)
    bars2 = ax3.bar(x + width/2, replay_values, width,
                    label='Experience Replay', color='skyblue', alpha=0.8)
    
    ax3.set_ylabel('Value (%)', fontsize=11)
    ax3.set_title('Overall Metrics Comparison', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics_names, rotation=15, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    
    # 값 이름표를 추가한다
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', 
                    va='bottom' if height >= 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('experience_replay_comparison.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'experience_replay_comparison.png'")
    plt.show()


def main():
    """경험 되살리기를 보여 주는 주 함수."""
    print("=" * 70)
    print("EXPERIENCE REPLAY FOR CONTINUAL LEARNING")
    print("=" * 70)
    print("\nThis script implements a simple but effective continual learning")
    print("technique: storing and replaying past examples during training.")
    print("\nKey idea: Mix current task data with replayed past examples")
    print("to maintain gradients for old tasks and prevent forgetting.")
    print("=" * 70)
    
    # 설정
    num_tasks = 5
    epochs_per_task = 5
    batch_size = 128
    learning_rate = 0.001
    memory_size = 1000
    examples_per_task = 200  # 과제마다 보기 200개 = 과제 5개에 모두 1000개
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # 재현성을 위해 씨앗을 설정한다
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
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
    
    # 경험 되살리기 학습기를 만든다
    learner = ExperienceReplayLearner(
        model=model,
        device=device,
        memory_size=memory_size,
        examples_per_task=examples_per_task,
        learning_rate=learning_rate
    )
    
    # 경험 되살리기로 이어 배우기를 돌린다
    start_time = time.time()
    replay_metrics = learner.train_continual(
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        epochs_per_task=epochs_per_task
    )
    total_time = time.time() - start_time
    
    # 결과 출력
    print("\n" + "=" * 70)
    print("EXPERIENCE REPLAY RESULTS")
    print("=" * 70)
    print(f"\n📊 Key Metrics:")
    print(f"   Average Accuracy:     {replay_metrics['average_accuracy']:.2f}%")
    print(f"   Learning Accuracy:    {replay_metrics['learning_accuracy']:.2f}%")
    print(f"   Backward Transfer:    {replay_metrics['backward_transfer']:.2f}%")
    print(f"\n⏱️  Training Time: {total_time:.2f} seconds")
    
    # 견주기 위해 밑금을 만든다(흉내 낸 것)
    # 실제로는 소박한 밑금을 따로 돌릴 것이다
    # 여기서는 그림을 위해 어림한 밑금 지표를 만든다
    baseline_metrics = {
        'average_accuracy': 50.0,  # 흔한 밑금
        'backward_transfer': -40.0,  # 많이 잊음
        'learning_accuracy': 95.0,  # 처음에는 잘 배움
        'forgetting_per_task': [-35, -40, -45, -38],
        'accuracy_matrix': np.array([
            [95, 60, 55, 50, 48],
            [0, 96, 58, 52, 50],
            [0, 0, 97, 60, 52],
            [0, 0, 0, 95, 58],
            [0, 0, 0, 0, 96]
        ])
    }
    
    # 견줌을 그려 본다
    visualize_comparison(replay_metrics, baseline_metrics, num_tasks)
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("\n✓ Experience replay significantly reduces forgetting!")
    print(f"  - Backward transfer improved from {baseline_metrics['backward_transfer']:.1f}%")
    print(f"    to {replay_metrics['backward_transfer']:.1f}%")
    print(f"\n✓ Memory efficiency: Only {memory_size} examples stored")
    print(f"  - That's just {memory_size / (len(train_dataset)):.2%} of training data")
    print("\n✓ Simple to implement and computationally efficient")
    print("\n⚠️  Limitations:")
    print("  - Requires storing raw examples (privacy concerns)")
    print("  - Random sampling may not be optimal")
    print("  - Performance depends on buffer size")
    print("\nNext steps: Explore advanced methods (EWC, LWF, etc.)")
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

핵심 클래스는 `MemoryBuffer`, `ExperienceReplayLearner`이며 앞의 연습문제 4개로 직접 확인할 수 있다.
