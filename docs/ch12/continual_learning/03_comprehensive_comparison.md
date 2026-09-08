# 57모듈: 이어 배우기

57모듈: 이어 배우기 - 고급. 이 스크립트는 여러 이어 배우기 방법을 두루 견주어 본다

이어 배우기는 앞서 배운 앎을 잊지 않으면서 잇단 과제로 모델을 익히는 어려움을 다룬다. 이 구현은 말랑함을 지키면서 파국적 잊음을 누그러뜨리는 핵심 전략을 보여 준다.

## 1. 코드

```python
"""
57모듈: 이어 배우기 - 고급
파일: 방법의 두루 견줌

이 스크립트는 까다로운 잣대에서 여러 이어 배우기 방법을 두루 견준다.
맞바꿈을 이해하고 상황마다 알맞은 방법을 고르는 데
도움이 된다.

학습 목표:
1. 벌주기, 되살리기, 섞은 접근법을 견준다
2. 셈과 기억의 맞바꿈을 이해한다
3. 여러 지표에 걸쳐 성능을 뜯어본다
4. 이어 배우기 실험의 좋은 버릇을 배운다

견주는 방법:
1. 소박한 차례 학습(밑금)
2. 경험 되살리기(ER)
3. 탄성 가중치 다지기(EWC)
4. 잊지 않고 배우기(LWF)
5. 섞음: EWC + 경험 되살리기

평가 지표:
- 평균 정확도(AA)
- 뒤로의 옮김(BWT) - 잊음 재기
- 배움 정확도(LA) - 배우는 힘
- 기억 안정성 - 옛 과제 성능의 흩어짐
- 셈 비용 - 학습 시간과 기억
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import copy
import time
import random
from dataclasses import dataclass
from collections import defaultdict

# ========================================================================
# 메인
# ========================================================================


@dataclass
class ExperimentResult:
    """이어 배우기 방법의 결과를 두루 담아 둔다."""
    method_name: str
    accuracy_matrix: np.ndarray
    average_accuracy: float
    backward_transfer: float
    learning_accuracy: float
    forgetting_per_task: List[float]
    training_time: float
    memory_usage_mb: float
    
    def summary_dict(self) -> Dict:
        """간추림을 사전으로 되돌린다."""
        return {
            'method': self.method_name,
            'avg_accuracy': self.average_accuracy,
            'backward_transfer': self.backward_transfer,
            'learning_accuracy': self.learning_accuracy,
            'training_time': self.training_time,
            'memory_usage': self.memory_usage_mb
        }


class NaiveLearner:
    """밑금: 소박한 차례 학습."""
    
    def __init__(self, model, device, lr=0.001):
        self.model = model
        self.device = device
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
    
    def train_continual(self, train_loaders, test_loaders, epochs_per_task=5):
        num_tasks = len(train_loaders)
        accuracy_matrix = np.zeros((num_tasks, num_tasks))
        
        for task_id in range(num_tasks):
            # 학습
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            self.model.train()
            
            for epoch in range(epochs_per_task):
                for data, target in train_loaders[task_id]:
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    optimizer.step()
            
            # 평가한다
            for eval_task_id in range(task_id + 1):
                acc = self.evaluate(test_loaders[eval_task_id])
                accuracy_matrix[eval_task_id, task_id] = acc
        
        return accuracy_matrix
    
    def evaluate(self, test_loader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        return 100.0 * correct / total


class ExperienceReplayLearner:
    """기억 버퍼를 쓰는 경험 되살리기."""
    
    def __init__(self, model, device, memory_size=1000, examples_per_task=200, lr=0.001):
        self.model = model
        self.device = device
        self.memory_size = memory_size
        self.examples_per_task = examples_per_task
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.memory_data = []
        self.memory_labels = []
    
    def add_to_memory(self, train_loader):
        all_data, all_labels = [], []
        for data, labels in train_loader:
            all_data.append(data)
            all_labels.append(labels)
        all_data = torch.cat(all_data, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        num_available = all_data.size(0)
        num_to_select = min(self.examples_per_task, num_available)
        indices = torch.randperm(num_available)[:num_to_select]
        
        self.memory_data.extend(all_data[indices])
        self.memory_labels.extend(all_labels[indices])
        
        # 최대 크기를 넘으면 잘라낸다
        if len(self.memory_data) > self.memory_size:
            self.memory_data = self.memory_data[-self.memory_size:]
            self.memory_labels = self.memory_labels[-self.memory_size:]
    
    def sample_memory(self, batch_size):
        if len(self.memory_data) == 0:
            return None, None
        sample_size = min(batch_size, len(self.memory_data))
        indices = random.sample(range(len(self.memory_data)), sample_size)
        return torch.stack([self.memory_data[i] for i in indices]), \
               torch.tensor([self.memory_labels[i] for i in indices], dtype=torch.long)
    
    def train_continual(self, train_loaders, test_loaders, epochs_per_task=5):
        num_tasks = len(train_loaders)
        accuracy_matrix = np.zeros((num_tasks, num_tasks))
        
        for task_id in range(num_tasks):
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            self.model.train()
            
            for epoch in range(epochs_per_task):
                for data, target in train_loaders[task_id]:
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    
                    # 지금 손실
                    output = self.model(data)
                    current_loss = self.criterion(output, target)
                    
                    # 되살리기 손실
                    replay_loss = torch.tensor(0.0).to(self.device)
                    replay_data, replay_labels = self.sample_memory(data.size(0))
                    if replay_data is not None:
                        replay_data = replay_data.to(self.device)
                        replay_labels = replay_labels.to(self.device)
                        replay_output = self.model(replay_data)
                        replay_loss = self.criterion(replay_output, replay_labels)
                    
                    total_loss = current_loss + replay_loss
                    total_loss.backward()
                    optimizer.step()
            
            # 기억에 더한다
            self.add_to_memory(train_loaders[task_id])
            
            # 평가한다
            for eval_task_id in range(task_id + 1):
                acc = self.evaluate(test_loaders[eval_task_id])
                accuracy_matrix[eval_task_id, task_id] = acc
        
        return accuracy_matrix
    
    def evaluate(self, test_loader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        return 100.0 * correct / total


class EWCLearner:
    """탄성 가중치 다지기."""
    
    def __init__(self, model, device, ewc_lambda=5000, lr=0.001):
        self.model = model
        self.device = device
        self.ewc_lambda = ewc_lambda
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.fisher_dict = {}
        self.optpar_dict = {}
    
    def compute_fisher(self, data_loader, num_samples=1000):
        self.model.eval()
        fisher = {n: torch.zeros_like(p.data) for n, p in self.model.named_parameters() if p.requires_grad}
        
        total_samples = 0
        for data, target in data_loader:
            if total_samples >= num_samples:
                break
            data, target = data.to(self.device), target.to(self.device)
            
            for i in range(data.size(0)):
                self.model.zero_grad()
                output = self.model(data[i:i+1])
                log_probs = F.log_softmax(output, dim=1)
                log_prob = log_probs[0, target[i]]
                log_prob.backward()
                
                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        fisher[n] += p.grad.data ** 2
                
                total_samples += 1
                if total_samples >= num_samples:
                    break
        
        for n in fisher:
            fisher[n] /= total_samples
        
        return fisher
    
    def ewc_penalty(self):
        penalty = torch.tensor(0.0).to(self.device)
        for task_id in self.fisher_dict.keys():
            for n, p in self.model.named_parameters():
                if n in self.fisher_dict[task_id] and p.requires_grad:
                    penalty += (self.fisher_dict[task_id][n] * 
                              (p - self.optpar_dict[task_id][n]) ** 2).sum()
        return (self.ewc_lambda / 2) * penalty
    
    def train_continual(self, train_loaders, test_loaders, epochs_per_task=5):
        num_tasks = len(train_loaders)
        accuracy_matrix = np.zeros((num_tasks, num_tasks))
        
        for task_id in range(num_tasks):
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            self.model.train()
            
            for epoch in range(epochs_per_task):
                for data, target in train_loaders[task_id]:
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    
                    output = self.model(data)
                    current_loss = self.criterion(output, target)
                    
                    ewc_loss = torch.tensor(0.0).to(self.device)
                    if len(self.fisher_dict) > 0:
                        ewc_loss = self.ewc_penalty()
                    
                    total_loss = current_loss + ewc_loss
                    total_loss.backward()
                    optimizer.step()
            
            # 다진다
            fisher = self.compute_fisher(train_loaders[task_id])
            self.fisher_dict[task_id] = fisher
            self.optpar_dict[task_id] = {n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad}
            
            # 평가한다
            for eval_task_id in range(task_id + 1):
                acc = self.evaluate(test_loaders[eval_task_id])
                accuracy_matrix[eval_task_id, task_id] = acc
        
        return accuracy_matrix
    
    def evaluate(self, test_loader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        return 100.0 * correct / total


class LWFLearner:
    """잊지 않고 배우기."""
    
    def __init__(self, model, device, distill_lambda=1.0, temperature=2.0, lr=0.001):
        self.model = model
        self.device = device
        self.distill_lambda = distill_lambda
        self.temperature = temperature
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.old_model = None
    
    def distillation_loss(self, new_logits, old_logits):
        T = self.temperature
        new_log_probs = F.log_softmax(new_logits / T, dim=1)
        old_probs = F.softmax(old_logits / T, dim=1)
        return F.kl_div(new_log_probs, old_probs, reduction='batchmean') * (T ** 2)
    
    def train_continual(self, train_loaders, test_loaders, epochs_per_task=5):
        num_tasks = len(train_loaders)
        accuracy_matrix = np.zeros((num_tasks, num_tasks))
        
        for task_id in range(num_tasks):
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            self.model.train()
            if self.old_model is not None:
                self.old_model.eval()
            
            for epoch in range(epochs_per_task):
                for data, target in train_loaders[task_id]:
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    
                    new_logits = self.model(data)
                    new_loss = self.criterion(new_logits, target)
                    
                    distill_loss = torch.tensor(0.0).to(self.device)
                    if self.old_model is not None:
                        with torch.no_grad():
                            old_logits = self.old_model(data)
                        distill_loss = self.distillation_loss(new_logits, old_logits)
                    
                    total_loss = new_loss + self.distill_lambda * distill_loss
                    total_loss.backward()
                    optimizer.step()
            
            # 옛 모델을 고친다
            self.old_model = copy.deepcopy(self.model)
            self.old_model.to(self.device)
            self.old_model.eval()
            for param in self.old_model.parameters():
                param.requires_grad = False
            
            # 평가한다
            for eval_task_id in range(task_id + 1):
                acc = self.evaluate(test_loaders[eval_task_id])
                accuracy_matrix[eval_task_id, task_id] = acc
        
        return accuracy_matrix
    
    def evaluate(self, test_loader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        return 100.0 * correct / total


def calculate_metrics(accuracy_matrix):
    """표준 이어 배우기 지표를 셈한다."""
    num_tasks = accuracy_matrix.shape[0]
    
    avg_accuracy = np.mean(accuracy_matrix[:, -1])
    learning_accuracy = np.mean(np.diag(accuracy_matrix))
    
    backward_transfer = 0.0
    forgetting_per_task = []
    for i in range(num_tasks - 1):
        initial = accuracy_matrix[i, i]
        final = accuracy_matrix[i, -1]
        change = final - initial
        backward_transfer += change
        forgetting_per_task.append(change)
    
    if num_tasks > 1:
        backward_transfer /= (num_tasks - 1)
    
    return avg_accuracy, backward_transfer, learning_accuracy, forgetting_per_task


def get_memory_usage():
    """지금 GPU 기억 씀씀이를 MB 단위로 얻는다."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def run_experiment(learner_class, model, device, train_loaders, test_loaders,
                  epochs_per_task, **kwargs):
    """주어진 학습기로 실험을 한 번 돌린다."""
    learner = learner_class(model, device, **kwargs)
    
    start_time = time.time()
    start_memory = get_memory_usage()
    
    accuracy_matrix = learner.train_continual(train_loaders, test_loaders, epochs_per_task)
    
    end_time = time.time()
    end_memory = get_memory_usage()
    
    training_time = end_time - start_time
    memory_usage = end_memory - start_memory
    
    avg_acc, bwt, learning_acc, forgetting = calculate_metrics(accuracy_matrix)
    
    return ExperimentResult(
        method_name=learner_class.__name__.replace('Learner', ''),
        accuracy_matrix=accuracy_matrix,
        average_accuracy=avg_acc,
        backward_transfer=bwt,
        learning_accuracy=learning_acc,
        forgetting_per_task=forgetting,
        training_time=training_time,
        memory_usage_mb=memory_usage
    )


def create_model():
    """새 모델을 만든다."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 2)
    )


def visualize_comparison(results: List[ExperimentResult], num_tasks: int):
    """견줌을 두루 그려 본다."""
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 그림 1: 마지막 정확도 견줌
    ax1 = fig.add_subplot(gs[0, 0])
    methods = [r.method_name for r in results]
    final_accs = [r.average_accuracy for r in results]
    colors = plt.cm.Set3(range(len(methods)))
    
    bars = ax1.bar(range(len(methods)), final_accs, color=colors, alpha=0.8)
    ax1.set_ylabel('Average Accuracy (%)', fontsize=11)
    ax1.set_title('Final Average Accuracy', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=15, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 100])
    
    for bar, val in zip(bars, final_accs):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=9)
    
    # 그림 2: 뒤로의 옮김 견줌
    ax2 = fig.add_subplot(gs[0, 1])
    bwts = [r.backward_transfer for r in results]
    bars = ax2.bar(range(len(methods)), bwts, color=colors, alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_ylabel('Backward Transfer (%)', fontsize=11)
    ax2.set_title('Forgetting Measure', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=15, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, bwts):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.1f}%', ha='center',
                va='bottom' if val >= 0 else 'top', fontsize=9)
    
    # 그림 3: 학습 시간 견줌
    ax3 = fig.add_subplot(gs[0, 2])
    times = [r.training_time for r in results]
    bars = ax3.bar(range(len(methods)), times, color=colors, alpha=0.8)
    ax3.set_ylabel('Training Time (seconds)', fontsize=11)
    ax3.set_title('Computational Efficiency', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(len(methods)))
    ax3.set_xticklabels(methods, rotation=15, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, times):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(times)*0.02,
                f'{val:.1f}s', ha='center', fontsize=9)
    
    # 그림 4: 정확도 행렬 열 지도(여러 칸)
    for idx, result in enumerate(results):
        row = 1
        col = idx
        if col >= 3:
            continue  # 앞의 세 방법의 행렬만 보인다
        
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(result.accuracy_matrix, cmap='RdYlGn',
                      aspect='auto', vmin=0, vmax=100)
        ax.set_xlabel('After Task', fontsize=9)
        ax.set_ylabel('Eval Task', fontsize=9)
        ax.set_title(f'{result.method_name}', fontsize=10, fontweight='bold')
        ax.set_xticks(range(num_tasks))
        ax.set_yticks(range(num_tasks))
        
        # 주석
        for i in range(num_tasks):
            for j in range(num_tasks):
                if j >= i:
                    ax.text(j, i, f'{result.accuracy_matrix[i, j]:.0f}',
                           ha="center", va="center", color="black", fontsize=8)
    
    plt.savefig('continual_learning_comparison.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'continual_learning_comparison.png'")
    plt.show()


def print_comparison_table(results: List[ExperimentResult]):
    """자세한 견줌 표를 찍는다."""
    print("\n" + "=" * 120)
    print("COMPREHENSIVE METHOD COMPARISON")
    print("=" * 120)
    
    # 머리글
    print(f"\n{'Method':<20} {'Avg Acc':<12} {'BWT':<12} {'Learn Acc':<12} {'Time (s)':<12} {'Memory (MB)':<12}")
    print("-" * 120)
    
    # 결과
    for result in results:
        print(f"{result.method_name:<20} "
              f"{result.average_accuracy:>10.2f}% "
              f"{result.backward_transfer:>10.2f}% "
              f"{result.learning_accuracy:>10.2f}% "
              f"{result.training_time:>10.2f} "
              f"{result.memory_usage_mb:>10.2f}")
    
    print("\n" + "=" * 120)


def create_split_mnist_tasks(num_tasks: int = 5) -> List[List[int]]:
    """Split MNIST 과제 설정을 만든다."""
    all_digits = list(range(10))
    classes_per_task = 10 // num_tasks
    return [all_digits[i * classes_per_task:(i + 1) * classes_per_task]
            for i in range(num_tasks)]


def create_task_dataset(full_dataset, task_classes: List[int]) -> TensorDataset:
    """특정 과제의 데이터셋을 만든다."""
    indices = [idx for idx in range(len(full_dataset))
               if full_dataset[idx][1] in task_classes]
    subset = Subset(full_dataset, indices)
    
    data_list, label_list = [], []
    for idx in range(len(subset)):
        img, label = subset[idx]
        data_list.append(img)
        label_list.append(task_classes.index(label))
    
    return TensorDataset(torch.stack(data_list),
                        torch.tensor(label_list, dtype=torch.long))


def main():
    """주 견줌 실험."""
    print("=" * 70)
    print("COMPREHENSIVE CONTINUAL LEARNING METHOD COMPARISON")
    print("=" * 70)
    print("\nComparing:")
    print("  1. Naive Sequential Learning (baseline)")
    print("  2. Experience Replay (memory-based)")
    print("  3. Elastic Weight Consolidation (regularization-based)")
    print("  4. Learning Without Forgetting (knowledge distillation)")
    print("=" * 70)
    
    # 설정
    num_tasks = 5
    epochs_per_task = 3  # 빠른 견줌을 위해 줄임
    batch_size = 128
    learning_rate = 0.001
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
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
    train_loaders = [DataLoader(create_task_dataset(train_dataset, classes),
                                batch_size=batch_size, shuffle=True)
                    for classes in task_classes]
    test_loaders = [DataLoader(create_task_dataset(test_dataset, classes),
                              batch_size=batch_size, shuffle=False)
                   for classes in task_classes]
    
    # 실험을 실행한다
    results = []
    
    print("\n" + "=" * 70)
    print("Running Experiments...")
    print("=" * 70)
    
    # 1. 소박한 방법
    print("\n[1/4] Naive Sequential Learning...")
    model = create_model().to(device)
    result = run_experiment(NaiveLearner, model, device, train_loaders, test_loaders,
                           epochs_per_task, lr=learning_rate)
    results.append(result)
    print(f"  Avg Acc: {result.average_accuracy:.2f}%, BWT: {result.backward_transfer:.2f}%")
    
    # 2. 경험 되살리기
    print("\n[2/4] Experience Replay...")
    model = create_model().to(device)
    result = run_experiment(ExperienceReplayLearner, model, device, train_loaders, test_loaders,
                           epochs_per_task, memory_size=1000, examples_per_task=200, lr=learning_rate)
    results.append(result)
    print(f"  Avg Acc: {result.average_accuracy:.2f}%, BWT: {result.backward_transfer:.2f}%")
    
    # 3. EWC
    print("\n[3/4] Elastic Weight Consolidation...")
    model = create_model().to(device)
    result = run_experiment(EWCLearner, model, device, train_loaders, test_loaders,
                           epochs_per_task, ewc_lambda=5000, lr=learning_rate)
    results.append(result)
    print(f"  Avg Acc: {result.average_accuracy:.2f}%, BWT: {result.backward_transfer:.2f}%")
    
    # 4. LWF
    print("\n[4/4] Learning Without Forgetting...")
    model = create_model().to(device)
    result = run_experiment(LWFLearner, model, device, train_loaders, test_loaders,
                           epochs_per_task, distill_lambda=1.0, temperature=2.0, lr=learning_rate)
    results.append(result)
    print(f"  Avg Acc: {result.average_accuracy:.2f}%, BWT: {result.backward_transfer:.2f}%")
    
    # 견줌을 찍는다
    print_comparison_table(results)
    
    # 시각화한다
    visualize_comparison(results, num_tasks)
    
    # 분석
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("\n📊 Performance Ranking (by Avg Accuracy):")
    sorted_results = sorted(results, key=lambda x: x.average_accuracy, reverse=True)
    for i, r in enumerate(sorted_results, 1):
        print(f"  {i}. {r.method_name}: {r.average_accuracy:.2f}%")
    
    print("\n💾 Memory Efficiency:")
    print("  - EWC & LWF: No example storage (privacy-preserving)")
    print("  - Experience Replay: Stores examples (memory overhead)")
    
    print("\n⚡ Computational Cost:")
    print("  - Naive & ER: Single forward/backward per batch")
    print("  - EWC: Extra Fisher computation per task")
    print("  - LWF: Double forward pass per batch")
    
    print("\n🎯 When to Use Each Method:")
    print("  - Experience Replay: When memory is available, best performance")
    print("  - EWC: Privacy concerns, memory constraints")
    print("  - LWF: Task domains similar, good knowledge transfer")
    print("  - Hybrid approaches: Combine strengths of multiple methods")
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

핵심 클래스는 `ExperimentResult`, `NaiveLearner`, `ExperienceReplayLearner`, `EWCLearner`이며 앞의 연습문제 4개로 직접 확인할 수 있다.
