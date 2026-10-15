# 단순 순차 학습 기준 모델

단순 순차 학습은 망각 방지 항 없이 새 과제의 교차 엔트로피만 최적화하는 기준 방법이다. Split MNIST의 로컬 이진 분류 규약에서 평균 정확도와 학습 직후 성능, 후방 전이를 계산한다.

## 1. 코드

??? note "코드 (644줄)"

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
    # 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
    # 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
    torch.manual_seed(0)

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
        main()
    ```


??? note "전체 출력 (160줄)"

    ```
    ======================================================================
    NAIVE SEQUENTIAL LEARNING BASELINE
    ======================================================================

    This script implements the naive baseline for continual learning:
      - Train on tasks sequentially
      - No special techniques to prevent forgetting
      - Comprehensive evaluation metrics

    This serves as the baseline to beat with continual learning methods.
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
    INITIAL EVALUATION (Before Any Training)
    ======================================================================
    Task 0: 24.92% (random guess ~ 50.0%)
    Task 1: 47.01% (random guess ~ 50.0%)
    Task 2: 46.21% (random guess ~ 50.0%)
    Task 3: 41.89% (random guess ~ 50.0%)
    Task 4: 40.95% (random guess ~ 50.0%)

    ============================================================
    Training Task 0
    ============================================================
      Epoch 1/5 - Loss: 0.0277, Train Acc: 99.04%
      Epoch 2/5 - Loss: 0.0026, Train Acc: 99.91%
      Epoch 3/5 - Loss: 0.0026, Train Acc: 99.92%
      Epoch 4/5 - Loss: 0.0018, Train Acc: 99.92%
      Epoch 5/5 - Loss: 0.0017, Train Acc: 99.93%

    ============================================================
    Evaluation after Task 0
    ============================================================
      Task 0: 99.91% (just learned)

    ============================================================
    Training Task 1
    ============================================================
      Epoch 1/5 - Loss: 0.2449, Train Acc: 92.49%
      Epoch 2/5 - Loss: 0.0492, Train Acc: 98.30%
      Epoch 3/5 - Loss: 0.0251, Train Acc: 99.08%
      Epoch 4/5 - Loss: 0.0171, Train Acc: 99.43%
      Epoch 5/5 - Loss: 0.0099, Train Acc: 99.66%

    ============================================================
    Evaluation after Task 1
    ============================================================
      Task 0: 83.88% (was 99.91%, forgot 16.03%)
      Task 1: 98.92% (just learned)

    ============================================================
    Training Task 2
    ============================================================
      Epoch 1/5 - Loss: 0.1038, Train Acc: 97.07%
      Epoch 2/5 - Loss: 0.0063, Train Acc: 99.80%
      Epoch 3/5 - Loss: 0.0030, Train Acc: 99.93%
      Epoch 4/5 - Loss: 0.0013, Train Acc: 99.96%
      Epoch 5/5 - Loss: 0.0003, Train Acc: 100.00%

    ============================================================
    Evaluation after Task 2
    ============================================================
      Task 0: 12.72% (was 99.91%, forgot 87.19%)
      Task 1: 87.41% (was 98.92%, forgot 11.51%)
      Task 2: 99.95% (just learned)

    ============================================================
    Training Task 3
    ============================================================
      Epoch 1/5 - Loss: 0.1691, Train Acc: 96.40%
      Epoch 2/5 - Loss: 0.0033, Train Acc: 99.88%
      Epoch 3/5 - Loss: 0.0011, Train Acc: 99.98%
      Epoch 4/5 - Loss: 0.0002, Train Acc: 99.99%
      Epoch 5/5 - Loss: 0.0001, Train Acc: 100.00%

    ============================================================
    Evaluation after Task 3
    ============================================================
      Task 0: 76.22% (was 99.91%, forgot 23.69%)
      Task 1: 75.76% (was 98.92%, forgot 23.16%)
      Task 2: 49.79% (was 99.95%, forgot 50.16%)
      Task 3: 99.80% (just learned)

    ============================================================
    Training Task 4
    ============================================================
      Epoch 1/5 - Loss: 0.1764, Train Acc: 95.27%
      Epoch 2/5 - Loss: 0.0314, Train Acc: 98.97%
      Epoch 3/5 - Loss: 0.0182, Train Acc: 99.35%
      Epoch 4/5 - Loss: 0.0104, Train Acc: 99.68%
      Epoch 5/5 - Loss: 0.0081, Train Acc: 99.71%

    ============================================================
    Evaluation after Task 4
    ============================================================
      Task 0: 16.17% (was 99.91%, forgot 83.74%)
      Task 1: 64.89% (was 98.92%, forgot 34.04%)
      Task 2: 14.67% (was 99.95%, forgot 85.27%)
      Task 3: 79.46% (was 99.80%, forgot 20.34%)
      Task 4: 99.45% (just learned)

    ======================================================================
    CONTINUAL LEARNING METRICS SUMMARY
    ======================================================================

    📊 Key Metrics:
       Average Accuracy (AA):    54.93%
       Learning Accuracy (LA):   99.60%
       Backward Transfer (BWT):  -55.85%
       Forward Transfer (FWT):   0.00%

    📉 Forgetting Analysis:
       Task 0: -83.74% ✗
       Task 1: -34.04% ✗
       Task 2: -85.27% ✗
       Task 3: -20.34% ✗

    📈 Final Accuracy per Task:
       Task 0: 16.17%
       Task 1: 64.89%
       Task 2: 14.67%
       Task 3: 79.46%
       Task 4: 99.45%

    📋 Accuracy Matrix:
       (Rows = Tasks, Columns = After Training Stage)
       ------------------------------------------------------------
       Task |  T0  |  T1  |  T2  |  T3  |  T4  |
       ------------------------------------------------------------
        0   | 99.9 | 83.9 | 12.7 | 76.2 | 16.2 |
        1   |  --  | 98.9 | 87.4 | 75.8 | 64.9 |
        2   |  --  |  --  | 99.9 | 49.8 | 14.7 |
        3   |  --  |  --  |  --  | 99.8 | 79.5 |
        4   |  --  |  --  |  --  |  --  | 99.4 |

    ⏱️  Total Training Time: 311.50 seconds

    Visualization saved as 'naive_sequential_learning_results.png'

    ======================================================================
    BASELINE ESTABLISHED
    ======================================================================

    This naive approach shows significant catastrophic forgetting.
    In the next scripts, we'll implement continual learning methods
    that preserve knowledge of previous tasks while learning new ones:
      - Script 03: Experience Replay
      - Intermediate: EWC, LWF, Synaptic Intelligence, etc.
    ======================================================================
    ```


## 2. 논의

각 과제의 숫자 쌍을 0·1로 다시 매겨 같은 출력층을 사용한다. task_id는 평가 함수에 전달되지만 모델 계산에 사용되지 않는다. 전체 10개 숫자의 클래스 증분 학습과 구별한다. 각 과제 시작마다 Adam을 새로 만들고 모델을 학습 모드로 바꾼다.

행렬은 행=평가 과제, 열=학습 단계다. forgetting_per_task는 이름과 달리 최종 정확도−학습 직후 정확도를 저장하므로 음수가 성능 저하다. backward_transfer는 그 평균이다. forward_transfer는 0.0 상수이며 측정한 전방 전이가 아니다. 초기 평가는 출력하지만 저장하지 않고, 각 새 과제 직전의 사전 평가도 하지 않는다.

AA는 최종 열 평균, LA는 대각선 평균이며 T개 과제에서 AA=LA+(T−1)BWT/T가 성립한다. 코드의 손실은 배치 평균을 다시 평균 내고, 정확도는 표본 수로 평균 낸다. 모델 선택을 추가한다면 독립적인 검증 로더를 사용해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
각 배치에서 zero_grad를 생략하면 경사가 어떻게 달라지는가?

</div>

??? success "연습문제 1 풀이"
    이전 .grad에 현재 경사가 더해진다. 과거 경사가 현재 경사와 같은 방향·크기라는 보장은 없어 단순히 학습률에 누적 단계 수를 곱한 것과 같지는 않다. 의도한 경사 누적을 할 때는 누적 구간·평균 손실과 optimizer.step 시점을 정해야 한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
forward_transfer=0.0을 성능 결과로 해석할 수 있는가?

</div>

??? success "연습문제 2 풀이"
    측정값이 아니라 자리 표시자다. 전방 전이는 새 과제를 학습하기 직전 성능과 같은 과제의 학습 전 기준을 비교해야 한다. 현재 코드는 필요한 단계별 사전 평가를 저장하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
BWT가 음수일 때 forgetting_per_task의 부호를 어떻게 읽는가?

</div>

??? success "연습문제 3 풀이"
    이 목록은 최종−직후 값을 사용하므로 음수는 성능 하락, 양수는 개선이다. 직후−최종을 쓰는 망각량과 부호가 반대다. 명칭만 보지 말고 계산식을 확인한다.

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
    현재 과제만의 검증 손실로 조기 종료하면 과거 성능과의 균형을 놓칠 수 있다. 사용할 과제별 검증 지표와 집계 규칙을 정하고 테스트셋은 선택에 사용하지 않는다. 인내 횟수에 도달하지 않고 종료된 경우에도 선택한 최적 상태를 복원하는 경로가 필요하다.

## 정리하며

단순 순차 학습은 다른 방법을 비교할 기준을 제공한다. 로컬 라벨과 출력 규약, 후방 전이의 부호를 명시하고 미측정 전방 전이를 결과로 제시하지 않는다.
