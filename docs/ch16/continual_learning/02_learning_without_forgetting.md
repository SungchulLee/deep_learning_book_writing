# Split MNIST의 LwF

새 과제 입력에서 이전 모델의 출력 분포를 맞추는 LwF 예제다. 각 과제 학습 후 모델을 고정 교사로 복사하고 다음 과제의 지도 로스에 출력 증류를 더한다. 과거 원본 데이터의 재학습과는 다른 보존 방식이다.

## 1. 코드

??? note "코드 (621줄)"

    ```python
    """
    57모듈: 이어 배우기 - 중급
    파일 2: 잊지 않고 배우기(LWF)

    이 스크립트는 보기를 하나도 담아 두지 않고 앎 증류로 앞선 과제의 앎을 지키는
    이어 배우기 방법인 잊지 않고 배우기를
    구현한다.

    학습 목표:
    1. 이어 배우기를 위한 앎 증류를 이해한다
    2. 부드러운 목표를 쓰는 LWF 로스를 구현한다
    3. 새 과제 배우기와 옛 앎 지키기의 균형을 잡는다
    4. 부드러운 목표를 위한 온도 눈금을 배운다

    수학적 바탕:
    LWF 로스는 다음을 합친다.

    L_total = L_new + λ * L_distill

    여기서 각 기호는 다음과 같다.
    - L_new: 새 과제의 교차 엔트로피 로스
    - L_distill: 옛 과제의 예측을 지키는 증류 로스

    증류 로스:
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
           a. 지금 모델로 새 데이터를 피드포워드한다
           b. 옛 모델(얼린 것)으로 새 데이터를 피드포워드한다
           c. 옛 모델의 예측에 맞추도록 증류 로스를 셈한다
           d. 새 과제의 로스를 셈한다
           e. 합친 로스를 거꾸로 퍼뜨린다

        핵심 통찰: 옛 모델의 부드러운 예측에는 앞선 과제에 대한 앎이
        담겨 있다. 새 과제 데이터에서 이 예측에 맞추면
        옛 앎이 넌지시 지켜진다.

        이점:
        - 앞선 보기를 담아 두지 않는다(사생활을 지킨다)
        - 어떤 뉴럴 넷트웍 구조에서도 굴러간다
        - 셈이 효율적이다

        한계:
        - 배치마다 피드포워드를 두 번 돌려야 한다
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
                model: 뉴럴 넷트웍 모델
                device: 학습에 쓸 장치
                distill_lambda: 증류 로스의 무게
                               값이 클수록 옛 앎을 더 많이 지킨다
                               흔한 범위: [0.5, 5.0]
                temperature: 확률 분포를 부드럽게 하는 온도
                            온도가 높을수록 분포가 부드럽다
                            흔한 범위: [1.0, 4.0]
                learning_rate: 최적화기의 러닝레이트
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
            앎 증류 로스를 셈한다.

            증류 로스는 다음 둘 사이의 KL 벌어짐이다.
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
                증류 로스(스칼라)
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

            과제 τ > 0의 로스 함수:
            L = L_new + λ * L_distill

            과제 0(첫 과제)에는 증류가 없다.
            L = L_new

            인수:
                train_loader: 지금 과제의 DataLoader
                task_id: 지금 과제의 번호
                epochs: 학습 에포크 수

            반환값:
                시대마다의 로스 목록
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

                    # ===== 지금 모델로 피드포워드 =====
                    new_logits = self.model(data)

                    # ===== 새 과제의 로스 =====
                    new_loss = self.criterion(new_logits, target)

                    # ===== 증류 로스 =====
                    distill_loss = torch.tensor(0.0).to(self.device)

                    if self.old_model is not None:
                        # 옛 모델로 피드포워드한다(얼려 둔 채)
                        with torch.no_grad():
                            old_logits = self.old_model(data)

                        # 증류 로스를 셈한다
                        distill_loss = self.distillation_loss(
                            new_logits=new_logits,
                            old_logits=old_logits,
                            temperature=self.temperature
                        )

                    # ===== 합친 로스 =====
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

            # 파라메터를 얼린다(그래디언트를 셈할 필요가 없다)
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
            LWF을 쓰는 이어 배우기의 주 회귀.

            과제마다의 절차:
            1. 증류 로스를 곁들여 익힌다(첫 과제가 아니면)
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
        """단순한 피드포워드 망을 만든다."""
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

        # 그림 3: 러닝 정확도와 마지막 정확도
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
        distill_lambda = 1.0  # 증류 로스의 무게
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
        main()
    ```


??? note "전체 출력 (157줄)"

    ```
    ======================================================================
    LEARNING WITHOUT FORGETTING (LWF)
    ======================================================================

    LWF prevents forgetting by:
      1. Saving predictions from the model before learning new task
      2. Using knowledge distillation to preserve these predictions
      3. No need to store previous examples (privacy-preserving)
    ======================================================================

    Using device: cpu

    Task Configuration:
      Task 0: Classes [0, 1]
      Task 1: Classes [2, 3]
      Task 2: Classes [4, 5]
      Task 3: Classes [6, 7]
      Task 4: Classes [8, 9]

    Model: 267,266 parameters

    ======================================================================
    CONTINUAL LEARNING WITH LEARNING WITHOUT FORGETTING (LWF)
    ======================================================================
    Distillation Lambda (λ): 1.0
    Temperature (T): 2.0

    ============================================================
    Training Task 0 with LWF
    First task: No distillation needed
    ============================================================
      Epoch 1/5 - Loss: 0.0264, Acc: 99.19%
      Epoch 2/5 - Loss: 0.0022, Acc: 99.92%
      Epoch 3/5 - Loss: 0.0016, Acc: 99.95%
      Epoch 4/5 - Loss: 0.0038, Acc: 99.89%
      Epoch 5/5 - Loss: 0.0025, Acc: 99.91%

      Updating old model for next task...
      Old model updated and frozen

    ============================================================
    Evaluation after Task 0
    ============================================================
      Task 0: 99.91% (just learned)

    ============================================================
    Training Task 1 with LWF
    Distillation: λ=1.0, T=2.0
    ============================================================
      Epoch 1/5 - Total: 1.2142, New: 0.6438, Distill: 0.5704, Acc: 63.98%
      Epoch 2/5 - Total: 1.0966, New: 0.5527, Distill: 0.5438, Acc: 60.93%
      Epoch 3/5 - Total: 1.0840, New: 0.5419, Distill: 0.5422, Acc: 61.10%
      Epoch 4/5 - Total: 1.0751, New: 0.5336, Distill: 0.5416, Acc: 61.29%
      Epoch 5/5 - Total: 1.0689, New: 0.5277, Distill: 0.5412, Acc: 61.25%

      Updating old model for next task...
      Old model updated and frozen

    ============================================================
    Evaluation after Task 1
    ============================================================
      Task 0: 99.95% (was 99.91%, change +0.05%)
      Task 1: 59.60% (just learned)

    ============================================================
    Training Task 2 with LWF
    Distillation: λ=1.0, T=2.0
    ============================================================
      Epoch 1/5 - Total: 0.4970, New: 0.3085, Distill: 0.1885, Acc: 89.98%
      Epoch 2/5 - Total: 0.4630, New: 0.2787, Distill: 0.1843, Acc: 90.71%
      Epoch 3/5 - Total: 0.4585, New: 0.2754, Distill: 0.1831, Acc: 90.91%
      Epoch 4/5 - Total: 0.4559, New: 0.2727, Distill: 0.1832, Acc: 90.88%
      Epoch 5/5 - Total: 0.4549, New: 0.2722, Distill: 0.1826, Acc: 90.82%

      Updating old model for next task...
      Old model updated and frozen

    ============================================================
    Evaluation after Task 2
    ============================================================
      Task 0: 98.11% (was 99.91%, change -1.80%)
      Task 1: 76.00% (was 59.60%, change +16.41%)
      Task 2: 90.23% (just learned)

    ============================================================
    Training Task 3 with LWF
    Distillation: λ=1.0, T=2.0
    ============================================================
      Epoch 1/5 - Total: 0.6409, New: 0.3877, Distill: 0.2531, Acc: 85.83%
      Epoch 2/5 - Total: 0.6168, New: 0.3703, Distill: 0.2466, Acc: 86.69%
      Epoch 3/5 - Total: 0.6161, New: 0.3696, Distill: 0.2464, Acc: 86.71%
      Epoch 4/5 - Total: 0.6140, New: 0.3683, Distill: 0.2457, Acc: 86.69%
      Epoch 5/5 - Total: 0.6142, New: 0.3678, Distill: 0.2464, Acc: 86.76%

      Updating old model for next task...
      Old model updated and frozen

    ============================================================
    Evaluation after Task 3
    ============================================================
      Task 0: 99.34% (was 99.91%, change -0.57%)
      Task 1: 75.76% (was 59.60%, change +16.16%)
      Task 2: 71.29% (was 90.23%, change -18.94%)
      Task 3: 84.99% (just learned)

    ============================================================
    Training Task 4 with LWF
    Distillation: λ=1.0, T=2.0
    ============================================================
      Epoch 1/5 - Total: 0.5922, New: 0.3981, Distill: 0.1940, Acc: 89.43%
      Epoch 2/5 - Total: 0.5687, New: 0.3682, Distill: 0.2005, Acc: 92.58%
      Epoch 3/5 - Total: 0.5623, New: 0.3609, Distill: 0.2014, Acc: 93.05%
      Epoch 4/5 - Total: 0.5615, New: 0.3585, Distill: 0.2030, Acc: 93.47%
      Epoch 5/5 - Total: 0.5600, New: 0.3565, Distill: 0.2036, Acc: 93.74%

      Updating old model for next task...
      Old model updated and frozen

    ============================================================
    Evaluation after Task 4
    ============================================================
      Task 0: 62.08% (was 99.91%, change -37.83%)
      Task 1: 88.64% (was 59.60%, change +29.04%)
      Task 2: 38.15% (was 90.23%, change -52.08%)
      Task 3: 96.42% (was 84.99%, change +11.43%)
      Task 4: 93.29% (just learned)

    ======================================================================
    LWF RESULTS
    ======================================================================

    📊 Key Metrics:
       Average Accuracy:     75.72%
       Learning Accuracy:    85.61%
       Backward Transfer:    -12.36%

    ⏱️  Training Time: 2.43 seconds

    Visualization saved as 'lwf_results.png'

    ======================================================================
    KEY INSIGHTS
    ======================================================================

    ✓ LWF preserves knowledge without storing examples
    ✓ Knowledge distillation is privacy-preserving
    ✓ Simple to implement with any architecture

    ⚠️  Considerations:
      - Requires two forward passes per batch
      - λ and T are important hyperparameters
      - May struggle if task domains are very different

    📝 Hyperparameter Tuning Tips:
      - λ ∈ [0.5, 5.0]: Higher = more preservation
      - T ∈ [1.0, 4.0]: Higher = softer targets
    ======================================================================
    ```


## 2. 논의

첫 과제는 교사가 없으므로 지도 로스만 쓴다. 이후에는 이전 과제 종료 시 복사해 고정한 old_model을 평가 모드·no_grad로 실행한다. 학생의 로그 소프트맥스와 교사 확률로 KL(교사‖학생)을 계산하고 T² 및 distill_lambda를 곱한다. 교사에 역전파하지 않아도 저장과 순전파 비용은 남는다.

모든 과제는 두 숫자를 로컬 라벨 0·1로 바꾸고 같은 2개 출력 전체를 증류한다. 과제별 기존 클래스 헤드를 따로 보존하거나 전체 10개 클래스 출력으로 확장하는 구현은 아니다. 새 입력에서 이전 예측을 제한하는 것이 과거 입력 전체의 함수를 유지하는 보장은 아니며 새 지도 목표와 충돌할 수도 있다.

temperature는 양수여야 하고 T=1에서도 교사 목표는 일반적으로 확률 분포다. 계수·온도·러닝레이트는 과거와 현재의 검증 성능으로 선택한다. 정확도는 백분율, BWT와 과제별 변화는 퍼센트포인트이며 음수가 성능 하락이다. 저장된 모델도 정보를 포함할 수 있어 원본 미저장을 개인정보 보호 보장으로 읽지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
교사는 언제 갱신되고 첫 과제의 증류 로스는 얼마인가?

</div>

??? success "연습문제 1 풀이"
    과제 학습을 마친 뒤 학생을 deepcopy해 다음 과제의 교사로 만든다. 첫 과제에는 old_model이 없어 증류 로스는 0이다. 새 과제를 학습하는 동안 교사를 함께 갱신하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
학생·교사의 KL 발산에서 확률과 로그확률의 입력 위치를 설명하라.

</div>

??? success "연습문제 2 풀이"
    F.kl_div에는 학생의 log_softmax를 첫 인수로, 교사의 softmax 확률을 두 번째로 넣는다. 이때 KL(교사‖학생)이 계산된다. 같은 클래스 순서와 정규화 범위 및 양의 온도를 사용해야 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
새 입력에서 교사 로스가 작으면 이전 테스트 정확도도 반드시 유지되는가?

</div>

??? success "연습문제 3 풀이"
    아니다. 증류는 현재 입력에서의 출력만 직접 제약하며 이전 입력 분포의 함수는 관측하지 못할 수 있다. 실제 이전 과제의 독립적인 평가가 필요하다. 새 과제 적응과의 충돌도 함께 측정한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 조기 종료 조각의 체크포인트 저장·복원과 지속 학습 검증 기준을 검토하라.

</div>

??? success "연습문제 4 풀이"
    아래는 검토용 조각이다. state_dict().copy()는 얕은 사전 복사여서 내부 텐서가 이후 갱신과 함께 바뀔 수 있다. 고정 체크포인트는 deepcopy하거나 각 텐서를 분리해 복사해야 한다.
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
    현재 과제만의 검증 로스로 조기 종료하면 과거 성능과의 균형을 놓칠 수 있다. 사용할 과제별 검증 지표와 집계 규칙을 정하고 테스트셋은 선택에 사용하지 않는다. 인내 횟수에 도달하지 않고 종료된 경우에도 선택한 최적 상태를 복원하는 경로가 필요하다.

## 정리하며

LwF는 새 입력의 고정 교사 출력을 이용해 이전 함수의 변화를 제한한다. 이 예제의 단일 로컬 이진 출력 규약과 KL 방향을 확인하고, 보존 효과와 교사 유지 비용을 함께 평가한다.
