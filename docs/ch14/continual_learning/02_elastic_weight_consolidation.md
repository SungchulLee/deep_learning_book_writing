# 57모듈: 이어 배우기

57모듈: 이어 배우기 - 중급 파일 1: 탄성 가중치 다지기(EWC)

이어 배우기는 앞서 배운 앎을 잊지 않으면서 잇단 과제로 모델을 익히는 어려움을 다룬다. 이 구현은 말랑함을 지키면서 파국적 잊음을 누그러뜨리는 핵심 전략을 보여 준다.

## 1. 코드

```python
"""
57모듈: 이어 배우기 - 중급
파일 1: 탄성 가중치 다지기(EWC)

이 스크립트는 가장 큰 영향을 끼친 이어 배우기 방법에 드는
탄성 가중치 다지기를 구현한다. EWC는 중요한 매개변수가 크게 바뀌지 않도록 지켜
파국적 잊음을 막는다.

학습 목표:
1. 피셔 정보와 그것이 매개변수 중요도를 재는 데 하는 몫을 이해한다
2. 이차 벌을 쓰는 EWC 손실을 구현한다
3. 피셔 정보를 효율적으로 셈한다
4. λ 초매개변수로 말랑함과 안정성의 균형을 잡는다

수학적 바탕:
과제 τ의 EWC 손실은 다음과 같다.

L(θ) = L_τ(θ) + (λ/2) Σ_i F_i (θ_i - θ*_{τ-1,i})²

여기서 각 기호는 다음과 같다.
- L_τ(θ): 지금 과제 τ의 손실
- F_i: 매개변수 i의 피셔 정보(중요도 무게)
- θ*_{τ-1}: 앞선 과제 뒤의 가장 좋은 매개변수
- λ: 벌주기 세기(초매개변수)

피셔 정보 행렬(대각선):
F_i = E_{x~D}[(∂log p(y|x,θ)/∂θ_i)²]

실험 표본을 쓴 어림:
F_i ≈ (1/N) Σ_{n=1}^N (∂L(x_n, θ)/∂θ_i)²

참고: Kirkpatrick 외, "Overcoming catastrophic forgetting in neural
networks," PNAS 2017
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
from typing import List, Tuple, Dict, Optional
import copy
from collections import defaultdict

# ========================================================================
# 메인
# ========================================================================


class EWCLearner:
    """
    탄성 가중치 다지기를 쓰는 이어 배우는 학습기.
    
    EWC 전략:
    1. 과제를 배운 뒤 피셔 정보 행렬을 셈한다
    2. 가장 좋은 매개변수와 피셔 값을 담아 둔다
    3. 새 과제를 배울 때 벌주기 항을 더해
       중요한 매개변수가 너무 많이 바뀌지 않게 한다
    
    핵심 통찰은 이것이다. 모든 매개변수가 똑같이 중요하지는 않다.
    앞선 과제에 결정적인 매개변수는 세게 지키고,
    덜 중요한 것은 마음대로 바꾸어도 된다.
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 ewc_lambda: float = 5000,
                 learning_rate: float = 0.001):
        """
        EWC 학습기를 초기화한다.
        
        인수:
            model: 신경망 모델
            device: 학습에 쓸 장치
            ewc_lambda: 벌주기 세기(λ)
                       값이 클수록 옛 매개변수를 더 세게 지킨다
                       흔한 범위: [100, 10000]
            learning_rate: 최적화기의 학습률
        """
        self.model = model
        self.device = device
        self.ewc_lambda = ewc_lambda
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        
        # 과제마다 피셔 정보와 가장 좋은 매개변수를 담아 둔다
        # fisher_dict: Dict[task_id -> Dict[param_name -> 피셔 값]]
        # optpar_dict: Dict[task_id -> Dict[param_name -> 가장 좋은 매개변수]]
        self.fisher_dict = {}
        self.optpar_dict = {}
        
        # 추적
        self.accuracy_matrix = None
        self.current_task = 0
    
    def compute_fisher_information(self,
                                   data_loader: DataLoader,
                                   task_id: int,
                                   num_samples: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        피셔 정보 행렬을 셈한다(대각 어림).
        
        피셔 정보는 모델의 예측이 매개변수마다에 얼마나 기대는지를
        수로 나타낸다. 피셔 값이 크면 그 매개변수가 지금 과제에
        중요하다는 뜻이다.
        
        수학적 자세한 내용:
        교차 엔트로피 손실을 쓰는 분류에서는 다음과 같다.
        F_i = E[(∂log p(y|x,θ)/∂θ_i)²]
        
        이를 실험 표본으로 어림한다.
        1. 보기 (x, y)마다 다음을 한다.
           a. 로그 확률을 셈한다: log p(y|x, θ)
           b. 기울기를 셈한다: ∂log p(y|x,θ)/∂θ_i
           c. 기울기를 제곱한다: (∂log p(y|x,θ)/∂θ_i)²
        2. 모든 보기에 걸쳐 평균 낸다
        
        이로써 온전한 피셔 행렬의 대각 어림을 얻으며,
        이는 셈으로 다룰 수 있다.
        
        인수:
            data_loader: 피셔를 셈할 DataLoader
            task_id: 지금 과제의 번호
            num_samples: 쓸 표본의 개수(None이면 모두)
        
        반환값:
            매개변수 이름을 피셔 값으로 옮기는 사전
        """
        print(f"\n  Computing Fisher information for Task {task_id}...")
        
        self.model.eval()
        
        # 피셔 정보 사전을 초기화한다
        fisher = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data)
        
        # 표본에서 피셔 정보를 쌓는다
        num_batches = 0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(data_loader):
            # 표본을 넉넉히 다루었는지 살핀다
            if num_samples is not None and total_samples >= num_samples:
                break
            
            data, target = data.to(self.device), target.to(self.device)
            
            # 경사 초기화
            self.model.zero_grad()
            
            # 순전파
            output = self.model(data)
            
            # 로그 확률을 셈한다
            # 분류에서는 log p(y|x) = log_softmax(output)[y]
            log_probs = F.log_softmax(output, dim=1)
            
            # 보기마다 로그 확률의 기울기를
            # 매개변수에 대해 셈한다
            batch_size = data.size(0)
            
            for i in range(batch_size):
                # 이 보기의 기울기를 0으로 만든다
                self.model.zero_grad()
                
                # 참 부류의 로그 확률을 얻는다
                log_prob = log_probs[i, target[i]]
                
                # 기울기를 셈한다: ∂log p(y|x)/∂θ
                log_prob.backward(retain_graph=(i < batch_size - 1))
                
                # 기울기 제곱 누적
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        # 제곱하여 피셔에 더한다
                        fisher[name] += param.grad.data ** 2
            
            total_samples += batch_size
            num_batches += 1
        
        # 모든 표본에 걸쳐 평균 낸다
        # Fisher[i] = (1/N) Σ_n (∂log p(y_n|x_n)/∂θ_i)²
        for name in fisher:
            fisher[name] /= total_samples
        
        print(f"  Fisher computed using {total_samples} samples")
        
        # 피셔 값의 통계를 찍는다
        all_fisher_values = torch.cat([f.flatten() for f in fisher.values()])
        print(f"  Fisher statistics:")
        print(f"    Mean: {all_fisher_values.mean().item():.6f}")
        print(f"    Std:  {all_fisher_values.std().item():.6f}")
        print(f"    Max:  {all_fisher_values.max().item():.6f}")
        print(f"    Min:  {all_fisher_values.min().item():.6f}")
        
        return fisher
    
    def ewc_penalty(self) -> torch.Tensor:
        """
        EWC 벌주기 항을 셈한다.
        
        이 벌은 중요한 매개변수가 너무 많이 바뀌지 않게 한다.
        
        벌 = (λ/2) Σ_{과제} Σ_i F_i(과제) (θ_i - θ*_i(과제))²
        
        여기서 각 기호는 다음과 같다.
        - 앞선 모든 과제에 걸쳐 더한다
        - 모든 매개변수에 걸쳐 더한다
        - F_i(과제): 그 과제에서 매개변수 i의 피셔 정보
        - θ_i: 지금 매개변수 값
        - θ*_i(과제): 그 과제를 배운 뒤의 가장 좋은 값
        
        반환값:
            EWC 벌 항(스칼라)
        """
        penalty = torch.tensor(0.0).to(self.device)
        
        # 앞선 과제를 모두 훑는다
        for task_id in self.fisher_dict.keys():
            fisher = self.fisher_dict[task_id]
            optpar = self.optpar_dict[task_id]
            
            # 매개변수마다
            for name, param in self.model.named_parameters():
                if name in fisher and param.requires_grad:
                    # 가장 좋은 값과의 차이를 제곱해 셈한다
                    # 피셔 정보로 무게를 준다
                    # (λ/2) * F_i * (θ_i - θ*_i)²
                    penalty += (fisher[name] * (param - optpar[name]) ** 2).sum()
        
        # 벌주기 세기를 씌운다
        penalty = (self.ewc_lambda / 2) * penalty
        
        return penalty
    
    def train_task(self,
                  train_loader: DataLoader,
                  task_id: int,
                  epochs: int = 5) -> List[float]:
        """
        EWC 벌주기를 곁들여 과제 하나로 익힌다.
        
        손실 함수:
        L_total = L_current + EWC_penalty
        
        여기서 각 기호는 다음과 같다.
        - L_current: 지금 과제의 교차 엔트로피 손실
        - EWC_penalty: 매개변수 변화에 대한 이차 벌
        
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
        print(f"Training Task {task_id} with EWC (λ={self.ewc_lambda})")
        print('=' * 60)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            current_loss_sum = 0.0
            ewc_loss_sum = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                # 순전파
                output = self.model(data)
                
                # 지금 과제의 손실
                current_loss = self.criterion(output, target)
                
                # EWC 벌(앞선 과제가 있을 때만)
                ewc_loss = torch.tensor(0.0).to(self.device)
                if len(self.fisher_dict) > 0:
                    ewc_loss = self.ewc_penalty()
                
                # 전체 손실
                total_loss = current_loss + ewc_loss
                
                # 역전파
                total_loss.backward()
                optimizer.step()
                
                # 통계 기록
                epoch_loss += total_loss.item()
                current_loss_sum += current_loss.item()
                ewc_loss_sum += ewc_loss.item()
                
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            
            # 지표를 계산한다
            avg_loss = epoch_loss / len(train_loader)
            avg_current = current_loss_sum / len(train_loader)
            avg_ewc = ewc_loss_sum / len(train_loader)
            accuracy = 100.0 * correct / total
            losses.append(avg_loss)
            
            print(f"  Epoch {epoch + 1}/{epochs} - "
                  f"Total Loss: {avg_loss:.4f}, "
                  f"Current: {avg_current:.4f}, "
                  f"EWC: {avg_ewc:.4f}, "
                  f"Train Acc: {accuracy:.2f}%")
        
        return losses
    
    def consolidate_task(self, 
                        train_loader: DataLoader,
                        task_id: int):
        """
        과제를 배운 뒤 앎을 다진다.
        
        여기에는 다음이 든다.
        1. 피셔 정보 행렬을 셈하기
        2. 지금의 가장 좋은 매개변수를 담아 두기
        
        이는 앞으로의 과제를 배울 때 제약으로 쓰인다.
        
        인수:
            train_loader: 피셔를 셈할 DataLoader
            task_id: 다질 과제의 번호
        """
        print(f"\n{'=' * 60}")
        print(f"Consolidating Task {task_id}")
        print('=' * 60)
        
        # 피셔 정보를 셈한다
        fisher = self.compute_fisher_information(
            data_loader=train_loader,
            task_id=task_id,
            num_samples=1000  # 효율을 위해 부분집합을 쓴다
        )
        
        # 피셔 정보를 담아 둔다
        self.fisher_dict[task_id] = fisher
        
        # 가장 좋은 매개변수를 담아 둔다
        optpar = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                optpar[name] = param.data.clone()
        
        self.optpar_dict[task_id] = optpar
        
        print(f"  Stored Fisher and optimal parameters for Task {task_id}")
    
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
        EWC를 쓰는 이어 배우기의 주 되돌이.
        
        과제마다의 절차:
        1. EWC 벌주기를 곁들여 익힌다
        2. 다진다: 피셔를 셈하고 매개변수를 담아 둔다
        3. 모든 과제에서 평가한다
        """
        num_tasks = len(train_loaders)
        self.accuracy_matrix = np.zeros((num_tasks, num_tasks))
        
        print("\n" + "=" * 70)
        print("CONTINUAL LEARNING WITH ELASTIC WEIGHT CONSOLIDATION (EWC)")
        print("=" * 70)
        print(f"EWC Lambda (λ): {self.ewc_lambda}")
        
        for task_id in range(num_tasks):
            self.current_task = task_id
            
            # 지금 과제로 익힌다
            self.train_task(
                train_loader=train_loaders[task_id],
                task_id=task_id,
                epochs=epochs_per_task
            )
            
            # 다지기: 피셔를 셈하고 매개변수를 담아 둔다
            self.consolidate_task(
                train_loader=train_loaders[task_id],
                task_id=task_id
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
                    change = acc - original_acc
                    print(f"  Task {eval_task_id}: {acc:.2f}% "
                          f"(was {original_acc:.2f}%, change {change:+.2f}%)")
        
        return self.calculate_metrics()
    
    def calculate_metrics(self):
        """이어 배우기 지표를 셈한다."""
        num_tasks = self.accuracy_matrix.shape[0]
        
        average_accuracy = np.mean(self.accuracy_matrix[:, -1])
        
        backward_transfer = 0.0
        forgetting_per_task = []
        
        for i in range(num_tasks - 1):
            initial_acc = self.accuracy_matrix[i, i]
            final_acc = self.accuracy_matrix[i, num_tasks - 1]
            change = final_acc - initial_acc
            backward_transfer += change
            forgetting_per_task.append(change)
        
        if num_tasks > 1:
            backward_transfer /= (num_tasks - 1)
        
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
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, num_classes)
    )


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


def visualize_ewc_results(ewc_metrics: dict, num_tasks: int):
    """EWC 결과를 그려 본다."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 그림 1: 정확도 행렬 열 지도
    ax1 = axes[0]
    im = ax1.imshow(ewc_metrics['accuracy_matrix'], cmap='RdYlGn',
                    aspect='auto', vmin=0, vmax=100)
    plt.colorbar(im, ax=ax1, label='Accuracy (%)')
    ax1.set_xlabel('After Training Task')
    ax1.set_ylabel('Evaluated Task')
    ax1.set_title('EWC: Accuracy Matrix', fontweight='bold')
    ax1.set_xticks(range(num_tasks))
    ax1.set_yticks(range(num_tasks))
    
    for i in range(num_tasks):
        for j in range(num_tasks):
            if j >= i:
                ax1.text(j, i, f'{ewc_metrics["accuracy_matrix"][i, j]:.1f}',
                        ha="center", va="center", color="black", fontsize=9)
    
    # 그림 2: 과제별 뒤로의 옮김
    ax2 = axes[1]
    tasks = list(range(len(ewc_metrics['forgetting_per_task'])))
    colors = ['green' if f >= 0 else 'red' 
              for f in ewc_metrics['forgetting_per_task']]
    
    bars = ax2.bar(tasks, ewc_metrics['forgetting_per_task'], 
                   color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Task')
    ax2.set_ylabel('Backward Transfer (%)')
    ax2.set_title('EWC: Forgetting per Task', fontweight='bold')
    ax2.set_xticks(tasks)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, ewc_metrics['forgetting_per_task']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center',
                va='bottom' if val >= 0 else 'top', fontsize=9)
    
    # 그림 3: 배움 정확도와 마지막 정확도
    ax3 = axes[2]
    tasks_range = list(range(num_tasks))
    learning_accs = np.diag(ewc_metrics['accuracy_matrix'])
    final_accs = ewc_metrics['accuracy_matrix'][:, -1]
    
    x = np.arange(num_tasks)
    width = 0.35
    
    ax3.bar(x - width/2, learning_accs, width,
            label='Learning Acc', color='skyblue', alpha=0.8)
    ax3.bar(x + width/2, final_accs, width,
            label='Final Acc', color='coral', alpha=0.8)
    
    ax3.set_xlabel('Task')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('EWC: Learning vs Final Accuracy', fontweight='bold')
    ax3.set_xticks(tasks_range)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig('ewc_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'ewc_results.png'")
    plt.show()


def main():
    """EWC를 보여 주는 주 함수."""
    print("=" * 70)
    print("ELASTIC WEIGHT CONSOLIDATION (EWC)")
    print("=" * 70)
    print("\nEWC prevents forgetting by:")
    print("  1. Identifying important parameters using Fisher information")
    print("  2. Adding quadratic penalty to prevent their modification")
    print("  3. Balancing learning new tasks with preserving old knowledge")
    print("=" * 70)
    
    # 설정
    num_tasks = 5
    epochs_per_task = 5
    batch_size = 128
    learning_rate = 0.001
    ewc_lambda = 5000  # 핵심 초매개변수 - 잊음과 배움의 맞바꿈을 다스린다
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
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
    
    # 모델 생성
    model = create_simple_model(784, 256, 2).to(device)
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # EWC 학습기를 만든다
    learner = EWCLearner(
        model=model,
        device=device,
        ewc_lambda=ewc_lambda,
        learning_rate=learning_rate
    )
    
    # 이어 배우기를 돌린다
    import time
    start_time = time.time()
    metrics = learner.train_continual(train_loaders, test_loaders, epochs_per_task)
    total_time = time.time() - start_time
    
    # 결과 출력
    print("\n" + "=" * 70)
    print("EWC RESULTS")
    print("=" * 70)
    print(f"\n📊 Key Metrics:")
    print(f"   Average Accuracy:     {metrics['average_accuracy']:.2f}%")
    print(f"   Learning Accuracy:    {metrics['learning_accuracy']:.2f}%")
    print(f"   Backward Transfer:    {metrics['backward_transfer']:.2f}%")
    print(f"\n⏱️  Training Time: {total_time:.2f} seconds")
    
    visualize_ewc_results(metrics, num_tasks)
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("\n✓ EWC reduces forgetting by protecting important parameters")
    print("✓ No need to store previous examples (privacy-preserving)")
    print("✓ Computationally efficient during training")
    print("\n⚠️  Considerations:")
    print("  - λ hyperparameter is task-dependent")
    print("  - Fisher computation adds overhead after each task")
    print("  - Diagonal Fisher approximation may be too restrictive")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

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

핵심 클래스는 `EWCLearner`이며 앞의 연습문제 4개로 직접 확인할 수 있다.
