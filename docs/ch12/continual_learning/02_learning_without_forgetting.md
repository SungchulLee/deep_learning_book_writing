# 57모듈: 이어 배우기

57모듈: 이어 배우기 - 중급 파일 2: 잊지 않고 배우기(LWF)

이어 배우기는 앞서 배운 앎을 잊지 않으면서 잇단 과제로 모델을 익히는 어려움을 다룬다. 이 구현은 말랑함을 지키면서 파국적 잊음을 누그러뜨리는 핵심 전략을 보여 준다.

## 코드

```python
"""
57모듈: 이어 배우기 - 중급
파일 2: 잊지 않고 배우기(LWF)

이 스크립트는 보기를 하나도 담아 두지 않고 앎 증류로 앞선 과제의 앎을 지키는
이어 배우기 방법인 잊지 않고 배우기를
구현한다.

학습 목표:
1. 이어 배우기를 위한 앎 증류를 이해한다
2. 부드러운 목표를 쓰는 LWF 손실을 구현한다
3. 새 과제 배우기와 옛 앎 지키기의 균형을 잡는다
4. 부드러운 목표를 위한 온도 눈금을 배운다

수학적 바탕:
LWF 손실은 다음을 합친다.

L_total = L_new + λ * L_distill

여기서 각 기호는 다음과 같다.
- L_new: 새 과제의 교차 엔트로피 손실
- L_distill: 옛 과제의 예측을 지키는 증류 손실

증류 손실:
L_distill = KL(softmax(z_old/T) || softmax(z_new/T))

여기서 각 기호는 다음과 같다.
- z_old: 옛 모델(얼린 것)의 로짓
- z_new: 새 모델(익히는 중)의 로짓
- T: 온도(T가 클수록 확률이 부드럽다)
- KL: 쿨백-라이블러 벌어짐

온도 눈금:
softmax_T(z_i) = exp(z_i/T) / Σ_j exp(z_j/T)

T가 클수록 → 확률이 부드러워지고 → 모델이 맞추기 쉬워진다

참고: Li & Hoiem, "Learning without Forgetting," IEEE TPAMI 2017
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

# ========================================================================
# 메인
# ========================================================================


class LWFLearner:
    """
    잊지 않고 배우기를 쓰는 이어 배우는 학습기.
    
    LWF 전략:
    1. 새 과제를 배우기 전에 지금 모델의 사본을 담아 둔다
    2. 새 과제를 익히는 동안 다음을 한다.
       a. 지금 모델로 새 데이터를 앞먹임한다
       b. 옛 모델(얼린 것)으로 새 데이터를 앞먹임한다
       c. 옛 모델의 예측에 맞추도록 증류 손실을 셈한다
       d. 새 과제의 손실을 셈한다
       e. 합친 손실을 거꾸로 퍼뜨린다
    
    핵심 통찰: 옛 모델의 부드러운 예측에는 앞선 과제에 대한 앎이
    담겨 있다. 새 과제 데이터에서 이 예측에 맞추면
    옛 앎이 넌지시 지켜진다.
    
    이점:
    - 앞선 보기를 담아 두지 않는다(사생활을 지킨다)
    - 어떤 신경망 구조에서도 굴러간다
    - 셈이 효율적이다
    
    한계:
    - 배치마다 앞먹임을 두 번 돌려야 한다
    - 새 과제의 데이터가 옛 과제와 아주 다르면 힘겨울 수 있다
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 distill_lambda: float = 1.0,
                 temperature: float = 2.0,
                 learning_rate: float = 0.001):
        """
        LWF 학습기를 초기화한다.
        
        인수:
            model: 신경망 모델
            device: 학습에 쓸 장치
            distill_lambda: 증류 손실의 무게
                           값이 클수록 옛 앎을 더 많이 지킨다
                           흔한 범위: [0.5, 5.0]
            temperature: 확률 분포를 부드럽게 하는 온도
                        온도가 높을수록 분포가 부드럽다
                        흔한 범위: [1.0, 4.0]
            learning_rate: 최적화기의 학습률
        """
        self.model = model
        self.device = device
        self.distill_lambda = distill_lambda
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        
        # 증류에 쓸 앞선 모델을 담아 둔다
        # 과제를 마칠 때마다 고쳐진다
        self.old_model = None
        
        # 추적
        self.accuracy_matrix = None
        self.current_task = 0
    
    def distillation_loss(self,
                         new_logits: torch.Tensor,
                         old_logits: torch.Tensor,
                         temperature: float) -> torch.Tensor:
        """
        앎 증류 손실을 셈한다.
        
        증류 손실은 다음 둘 사이의 KL 벌어짐이다.
        - 옛 모델의 부드러운 예측(스승)
        - 새 모델의 부드러운 예측(제자)
        
        수학적 자세한 내용:
        KL(P || Q) = Σ_i P(i) * log(P(i) / Q(i))
        
        수치 안정성을 위해 다음을 쓴다.
        KL = Σ_i P(i) * (log P(i) - log Q(i))
        
        온도 눈금:
        - 소프트맥스 전에 로짓을 T로 나누면 분포가 부드러워진다
        - T가 클수록 → 분포가 고루 퍼지고 → 맞추기 쉬워진다
        - 인자 T²이 눈금을 메운다(끌어내기는 논문을 보라)
        
        인수:
            new_logits: 지금 모델(제자)의 로짓
            old_logits: 옛 모델(스승)의 로짓
            temperature: 눈금을 위한 온도
        
        반환값:
            증류 손실(스칼라)
        """
        # 온도 눈금을 씌우고 로그 확률을 셈한다
        # log_softmax_T(z) = log(exp(z/T) / Σ exp(z/T))
        #                  = z/T - log(Σ exp(z/T))
        new_log_probs = F.log_softmax(new_logits / temperature, dim=1)
        
        # 옛 모델에서 부드러운 목표를 셈한다
        # softmax_T(z) = exp(z/T) / Σ exp(z/T)
        old_probs = F.softmax(old_logits / temperature, dim=1)
        
        # 온도를 메운 KL 벌어짐
        # 인자 T²이 온도 눈금을 메운다
        # (논문 "Distilling the Knowledge in a Neural Network"을 보라)
        loss = F.kl_div(
            new_log_probs,
            old_probs,
            reduction='batchmean'
        ) * (temperature ** 2)
        
        return loss
    
    def train_task(self,
                  train_loader: DataLoader,
                  task_id: int,
                  epochs: int = 5) -> List[float]:
        """
        LWF을 곁들여 과제 하나로 익힌다.
        
        과제 τ > 0의 손실 함수:
        L = L_new + λ * L_distill
        
        과제 0(첫 과제)에는 증류가 없다.
        L = L_new
        
        인수:
            train_loader: 지금 과제의 DataLoader
            task_id: 지금 과제의 번호
            epochs: 학습 에포크 수
        
        반환값:
            시대마다의 손실 목록
        """
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.model.train()
        
        # 옛 모델을 평가 모드로 둔다(얼린다)
        if self.old_model is not None:
            self.old_model.eval()
        
        losses = []
        
        print(f"\n{'=' * 60}")
        print(f"Training Task {task_id} with LWF")
        if task_id > 0:
            print(f"Distillation: λ={self.distill_lambda}, T={self.temperature}")
        else:
            print("First task: No distillation needed")
        print('=' * 60)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            new_loss_sum = 0.0
            distill_loss_sum = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                # ===== 지금 모델로 앞먹임 =====
                new_logits = self.model(data)
                
                # ===== 새 과제의 손실 =====
                new_loss = self.criterion(new_logits, target)
                
                # ===== 증류 손실 =====
                distill_loss = torch.tensor(0.0).to(self.device)
                
                if self.old_model is not None:
                    # 옛 모델로 앞먹임한다(얼려 둔 채)
                    with torch.no_grad():
                        old_logits = self.old_model(data)
                    
                    # 증류 손실을 셈한다
                    distill_loss = self.distillation_loss(
                        new_logits=new_logits,
                        old_logits=old_logits,
                        temperature=self.temperature
                    )
                
                # ===== 합친 손실 =====
                total_loss = new_loss + self.distill_lambda * distill_loss
                
                # 역전파
                total_loss.backward()
                optimizer.step()
                
                # 통계 기록
                epoch_loss += total_loss.item()
                new_loss_sum += new_loss.item()
                distill_loss_sum += distill_loss.item()
                
                _, predicted = torch.max(new_logits, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            
            # 지표를 계산한다
            avg_loss = epoch_loss / len(train_loader)
            avg_new = new_loss_sum / len(train_loader)
            avg_distill = distill_loss_sum / len(train_loader)
            accuracy = 100.0 * correct / total
            losses.append(avg_loss)
            
            if task_id > 0:
                print(f"  Epoch {epoch + 1}/{epochs} - "
                      f"Total: {avg_loss:.4f}, "
                      f"New: {avg_new:.4f}, "
                      f"Distill: {avg_distill:.4f}, "
                      f"Acc: {accuracy:.2f}%")
            else:
                print(f"  Epoch {epoch + 1}/{epochs} - "
                      f"Loss: {avg_loss:.4f}, "
                      f"Acc: {accuracy:.2f}%")
        
        return losses
    
    def update_old_model(self):
        """
        과제를 배운 뒤 옛 모델(스승)을 고친다.
        
        지금 모델을 깊이 복사하여 다음 과제의
        스승으로 삼는다.
        """
        print("\n  Updating old model for next task...")
        
        # 지금 모델을 깊이 복사한다
        self.old_model = copy.deepcopy(self.model)
        
        # 장치로 옮기고 평가 모드로 둔다
        self.old_model.to(self.device)
        self.old_model.eval()
        
        # 매개변수를 얼린다(기울기를 셈할 필요가 없다)
        for param in self.old_model.parameters():
            param.requires_grad = False
        
        print("  Old model updated and frozen")
    
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
        LWF을 쓰는 이어 배우기의 주 되돌이.
        
        과제마다의 절차:
        1. 증류 손실을 곁들여 익힌다(첫 과제가 아니면)
        2. 다음 과제를 위해 옛 모델을 고친다
        3. 모든 과제에서 평가한다
        """
        num_tasks = len(train_loaders)
        self.accuracy_matrix = np.zeros((num_tasks, num_tasks))
        
        print("\n" + "=" * 70)
        print("CONTINUAL LEARNING WITH LEARNING WITHOUT FORGETTING (LWF)")
        print("=" * 70)
        print(f"Distillation Lambda (λ): {self.distill_lambda}")
        print(f"Temperature (T): {self.temperature}")
        
        for task_id in range(num_tasks):
            self.current_task = task_id
            
            # 지금 과제로 익힌다
            self.train_task(
                train_loader=train_loaders[task_id],
                task_id=task_id,
                epochs=epochs_per_task
            )
            
            # 다음 과제를 위해 옛 모델을 고친다
            self.update_old_model()
            
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


def visualize_lwf_results(lwf_metrics: dict, num_tasks: int):
    """LWF 결과를 그려 본다."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 그림 1: 정확도 행렬 열 지도
    ax1 = axes[0]
    im = ax1.imshow(lwf_metrics['accuracy_matrix'], cmap='RdYlGn',
                    aspect='auto', vmin=0, vmax=100)
    plt.colorbar(im, ax=ax1, label='Accuracy (%)')
    ax1.set_xlabel('After Training Task')
    ax1.set_ylabel('Evaluated Task')
    ax1.set_title('LWF: Accuracy Matrix', fontweight='bold')
    ax1.set_xticks(range(num_tasks))
    ax1.set_yticks(range(num_tasks))
    
    for i in range(num_tasks):
        for j in range(num_tasks):
            if j >= i:
                ax1.text(j, i, f'{lwf_metrics["accuracy_matrix"][i, j]:.1f}',
                        ha="center", va="center", color="black", fontsize=9)
    
    # 그림 2: 뒤로의 옮김
    ax2 = axes[1]
    tasks = list(range(len(lwf_metrics['forgetting_per_task'])))
    colors = ['green' if f >= 0 else 'red'
              for f in lwf_metrics['forgetting_per_task']]
    
    bars = ax2.bar(tasks, lwf_metrics['forgetting_per_task'],
                   color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Task')
    ax2.set_ylabel('Backward Transfer (%)')
    ax2.set_title('LWF: Forgetting per Task', fontweight='bold')
    ax2.set_xticks(tasks)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, lwf_metrics['forgetting_per_task']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center',
                va='bottom' if val >= 0 else 'top', fontsize=9)
    
    # 그림 3: 배움 정확도와 마지막 정확도
    ax3 = axes[2]
    learning_accs = np.diag(lwf_metrics['accuracy_matrix'])
    final_accs = lwf_metrics['accuracy_matrix'][:, -1]
    
    x = np.arange(num_tasks)
    width = 0.35
    
    ax3.bar(x - width/2, learning_accs, width,
            label='Learning Acc', color='skyblue', alpha=0.8)
    ax3.bar(x + width/2, final_accs, width,
            label='Final Acc', color='coral', alpha=0.8)
    
    ax3.set_xlabel('Task')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('LWF: Learning vs Final Accuracy', fontweight='bold')
    ax3.set_xticks(range(num_tasks))
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig('lwf_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'lwf_results.png'")
    plt.show()


def main():
    """LWF을 보여 주는 주 함수."""
    print("=" * 70)
    print("LEARNING WITHOUT FORGETTING (LWF)")
    print("=" * 70)
    print("\nLWF prevents forgetting by:")
    print("  1. Saving predictions from the model before learning new task")
    print("  2. Using knowledge distillation to preserve these predictions")
    print("  3. No need to store previous examples (privacy-preserving)")
    print("=" * 70)
    
    # 설정
    num_tasks = 5
    epochs_per_task = 5
    batch_size = 128
    learning_rate = 0.001
    distill_lambda = 1.0  # 증류 손실의 무게
    temperature = 2.0     # 부드러운 목표의 온도
    
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
    
    # LWF 학습기를 만든다
    learner = LWFLearner(
        model=model,
        device=device,
        distill_lambda=distill_lambda,
        temperature=temperature,
        learning_rate=learning_rate
    )
    
    # 이어 배우기를 돌린다
    import time
    start_time = time.time()
    metrics = learner.train_continual(train_loaders, test_loaders, epochs_per_task)
    total_time = time.time() - start_time
    
    # 결과 출력
    print("\n" + "=" * 70)
    print("LWF RESULTS")
    print("=" * 70)
    print(f"\n📊 Key Metrics:")
    print(f"   Average Accuracy:     {metrics['average_accuracy']:.2f}%")
    print(f"   Learning Accuracy:    {metrics['learning_accuracy']:.2f}%")
    print(f"   Backward Transfer:    {metrics['backward_transfer']:.2f}%")
    print(f"\n⏱️  Training Time: {total_time:.2f} seconds")
    
    visualize_lwf_results(metrics, num_tasks)
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("\n✓ LWF preserves knowledge without storing examples")
    print("✓ Knowledge distillation is privacy-preserving")
    print("✓ Simple to implement with any architecture")
    print("\n⚠️  Considerations:")
    print("  - Requires two forward passes per batch")
    print("  - λ and T are important hyperparameters")
    print("  - May struggle if task domains are very different")
    print("\n📝 Hyperparameter Tuning Tips:")
    print("  - λ ∈ [0.5, 5.0]: Higher = more preservation")
    print("  - T ∈ [1.0, 4.0]: Higher = softer targets")
    print("=" * 70)


if __name__ == "__main__":
    main()```

## 논의

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
