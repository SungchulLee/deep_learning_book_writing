# Split MNIST의 파국적 망각 관찰

MNIST를 다섯 개의 숫자 쌍으로 나누어 같은 모델에 순서대로 학습시키고 이전 과제 정확도를 기록한다. 망각을 줄이는 기법을 적용하기 전의 관찰용 예제이며 출력과 그림은 보존된 결과다.

## 1. 코드

??? note "코드 (577줄)"

    ```python
    """
    57모듈: 이어 배우기 - 초급
    파일 1: 파국적 잊음 보여 주기

    이 스크립트는 과제를 차례로 배울 때 뉴럴 넷트웍에 생기는 파국적 잊음 문제를 보여 준다.
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
        파국적 잊음을 보여 주기 위한 단순한 피드포워드 망.

        구조:
        - 입력: 28x28을 편 784차원
        - 숨은 레이어: ReLU 활성을 쓴 레이어 둘
        - 출력: 부류 10개(MNIST 숫자)

        이 단순한 구조가 파국적 잊음을 더 또렷이 드러낸다.
        """

        def __init__(self, input_size: int = 784, hidden_size: int = 256, 
                     num_classes: int = 10):
            super(SimpleNetwork, self).__init__()

            # 망의 레이어를 정한다
            # 레이어 1: 입력에서 첫 숨은 레이어로
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu1 = nn.ReLU()

            # 레이어 2: 첫 숨은 레이어에서 둘째 숨은 레이어로
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.relu2 = nn.ReLU()

            # 레이어 3: 둘째 숨은 레이어에서 출력 레이어로
            self.fc3 = nn.Linear(hidden_size, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            뉴럴 넷트웍을 통과하는 순전파.

            인수:
                x: 꼴이 (batch_size, 784)인 입력 텐서

            반환값:
                꼴이 (batch_size, num_classes)인 출력 로짓
            """
            # 필요하면 입력을 편다([batch, 1, 28, 28]에서 [batch, 784]로)
            x = x.view(x.size(0), -1)

            # 앞으로 퍼뜨리기
            x = self.relu1(self.fc1(x))  # ReLU를 쓴 첫 숨은 레이어
            x = self.relu2(self.fc2(x))  # ReLU를 쓴 둘째 숨은 레이어
            x = self.fc3(x)               # 출력 레이어(로짓)

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
            b. 로스를 셈한다(교차 엔트로피)
            c. 그래디언트를 거꾸로 퍼뜨린다
            d. 최적화기로 웨이트를 고친다
        3. 살펴보려고 학습 로스를 좇는다

        인수:
            model: 익힐 뉴럴 넷트웍
            train_loader: 학습 데이터의 DataLoader
            device: 익힐 장치(CPU나 GPU)
            epochs: 학습 에포크 수
            learning_rate: 최적화기의 러닝레이트

        반환값:
            시대마다의 평균 로스 목록
        """
        # 로스 함수를 정한다(분류에는 교차 엔트로피)
        criterion = nn.CrossEntropyLoss()

        # 최적화기를 정한다(Adam이 튼튼하고 널리 쓰인다)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 학습 로스를 좇는다
        losses = []

        model.train()  # 학습 모드로 둔다(드롭아웃 등을 켠다)

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                # 데이터를 장치로 옮기기
                data, target = data.to(device), target.to(device)

                # 앞선 되풀이의 그래디언트를 0으로 만든다
                optimizer.zero_grad()

                # 피드포워드: 예측을 셈한다
                output = model(data)

                # 로스를 계산한다
                loss = criterion(output, target)

                # 역전파: 경사를 계산한다
                loss.backward()

                # 웨이트를 갱신한다
                optimizer.step()

                # 통계 기록
                epoch_loss += loss.item()
                num_batches += 1

            # 이 시대의 평균 로스
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
        - 로스: 평균 교차 엔트로피 로스

        인수:
            model: 평가할 뉴럴 넷트웍
            test_loader: 시험 데이터의 DataLoader
            device: 평가할 장치

        반환값:
            (정확도, 로스) 짝
        """
        criterion = nn.CrossEntropyLoss()

        model.eval()  # 평가 모드로 둔다(드롭아웃 등을 끈다)

        total_loss = 0.0
        correct = 0
        total = 0

        # 효율을 위해 그래디언트 계산 끄기
        with torch.no_grad():
            for data, target in test_loader:
                # 데이터를 장치로 옮기기
                data, target = data.to(device), target.to(device)

                # 순전파
                output = model(data)

                # 로스를 계산한다
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
        4. 이는 뉴럴 넷트웍이 그대로 두면 파국적으로 잊는다는 것을 보여 준다

        핵심 요점:
        보통의 뉴럴 넷트웍 학습은 이어 배우기에 **알맞지 않다**.
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
        print("=" * 70)
    ```


??? note "전체 출력 (152줄)"

    ```
    ======================================================================
    CATASTROPHIC FORGETTING DEMONSTRATION
    ======================================================================

    This script demonstrates how neural networks forget previous
    tasks when learning new tasks sequentially.

    We'll train on Split MNIST (5 tasks, 2 classes each) and watch
    how performance on earlier tasks degrades dramatically.
    ======================================================================
    Using device: cpu

    Task Configuration:
      Task 0: Classes [0, 1]
      Task 1: Classes [2, 3]
      Task 2: Classes [4, 5]
      Task 3: Classes [6, 7]
      Task 4: Classes [8, 9]

    Model Architecture:
    SimpleNetwork(
      (fc1): Linear(in_features=784, out_features=256, bias=True)
      (relu1): ReLU()
      (fc2): Linear(in_features=256, out_features=256, bias=True)
      (relu2): ReLU()
      (fc3): Linear(in_features=256, out_features=2, bias=True)
    )

    Total Parameters: 267,266

    ======================================================================
    SEQUENTIAL TRAINING (Demonstrating Catastrophic Forgetting)
    ======================================================================

    ======================================================================
    Training on Task 0 (Classes: [0, 1])
    ======================================================================
      Epoch 1/5, Loss: 0.0264
      Epoch 2/5, Loss: 0.0022
      Epoch 3/5, Loss: 0.0016
      Epoch 4/5, Loss: 0.0038
      Epoch 5/5, Loss: 0.0025

    Evaluating on all tasks after training Task 0:
    --------------------------------------------------
      Task 0: Accuracy = 99.91%

    ======================================================================
    Training on Task 1 (Classes: [2, 3])
    ======================================================================
      Epoch 1/5, Loss: 0.2208
      Epoch 2/5, Loss: 0.0398
      Epoch 3/5, Loss: 0.0206
      Epoch 4/5, Loss: 0.0128
      Epoch 5/5, Loss: 0.0104

    Evaluating on all tasks after training Task 1:
    --------------------------------------------------
      Task 0: Accuracy = 74.99%
        (Original: 99.91%, Forgetting: 24.92%)
      Task 1: Accuracy = 99.46%

    ======================================================================
    Training on Task 2 (Classes: [4, 5])
    ======================================================================
      Epoch 1/5, Loss: 0.1221
      Epoch 2/5, Loss: 0.0071
      Epoch 3/5, Loss: 0.0034
      Epoch 4/5, Loss: 0.0022
      Epoch 5/5, Loss: 0.0010

    Evaluating on all tasks after training Task 2:
    --------------------------------------------------
      Task 0: Accuracy = 13.24%
        (Original: 99.91%, Forgetting: 86.67%)
      Task 1: Accuracy = 88.59%
        (Original: 99.46%, Forgetting: 10.87%)
      Task 2: Accuracy = 99.89%

    ======================================================================
    Training on Task 3 (Classes: [6, 7])
    ======================================================================
      Epoch 1/5, Loss: 0.1843
      Epoch 2/5, Loss: 0.0027
      Epoch 3/5, Loss: 0.0008
      Epoch 4/5, Loss: 0.0003
      Epoch 5/5, Loss: 0.0001

    Evaluating on all tasks after training Task 3:
    --------------------------------------------------
      Task 0: Accuracy = 76.36%
        (Original: 99.91%, Forgetting: 23.55%)
      Task 1: Accuracy = 81.44%
        (Original: 99.46%, Forgetting: 18.02%)
      Task 2: Accuracy = 49.31%
        (Original: 99.89%, Forgetting: 50.59%)
      Task 3: Accuracy = 99.80%

    ======================================================================
    Training on Task 4 (Classes: [8, 9])
    ======================================================================
      Epoch 1/5, Loss: 0.2381
      Epoch 2/5, Loss: 0.0348
      Epoch 3/5, Loss: 0.0201
      Epoch 4/5, Loss: 0.0138
      Epoch 5/5, Loss: 0.0091

    Evaluating on all tasks after training Task 4:
    --------------------------------------------------
      Task 0: Accuracy = 22.98%
        (Original: 99.91%, Forgetting: 76.93%)
      Task 1: Accuracy = 74.05%
        (Original: 99.46%, Forgetting: 25.42%)
      Task 2: Accuracy = 15.53%
        (Original: 99.89%, Forgetting: 84.36%)
      Task 3: Accuracy = 82.43%
        (Original: 99.80%, Forgetting: 17.37%)
      Task 4: Accuracy = 99.45%

    ======================================================================
    RESULTS SUMMARY
    ======================================================================

    Accuracy Matrix (rows = tasks, columns = after training task):
    (Each entry shows accuracy on row's task after training column's task)

    Task | After T0 | After T1 | After T2 | After T3 | After T4 | Forgetting
    ------------------------------------------------------------------------
      0  |   99.9%  |   75.0%  |   13.2%  |   76.4%  |   23.0%  |   76.9%
      1  |    -    |   99.5%  |   88.6%  |   81.4%  |   74.0%  |   25.4%
      2  |    -    |    -    |   99.9%  |   49.3%  |   15.5%  |   84.4%
      3  |    -    |    -    |    -    |   99.8%  |   82.4%  |   17.4%
      4  |    -    |    -    |    -    |    -    |   99.4%  |    N/A

    Average Forgetting: 51.02%
    Final Average Accuracy: 58.88%

    Visualization saved as 'catastrophic_forgetting_demonstration.png'

    ======================================================================
    CONCLUSION
    ======================================================================

    Catastrophic forgetting is a fundamental problem in neural networks!
    As we train on new tasks, the model's weights are updated to minimize
    the new task's loss, which destroys information needed for old tasks.

    In the following scripts, we'll explore methods to prevent this:
      - Script 02: Naive Sequential Learning (baseline)
      - Script 03: Simple Experience Replay
      - Intermediate scripts: EWC, LWF, and more!
    ======================================================================
    ```


## 2. 논의

각 과제의 두 숫자를 로컬 라벨 0·1로 바꿔 같은 2개 출력 레이어를 사용한다. 예를 들어 출력 0은 과제에 따라 0·2·4·6·8 중 다른 숫자를 뜻한다. 전체 숫자 10개를 한 번에 구분하는 클래스 증분 평가가 아니며 이 프로토콜을 다른 결과와 혼동하지 않는다.

정확도 행렬의 행은 평가 과제, 열은 학습 단계다. 대각선은 학습 직후, 마지막 열은 모든 과제 학습 후의 성능이며 미관측 아래 삼각 영역은 0으로 남는다. 평균 망각은 이전 과제의 대각선과 마지막 값 차이로 계산하고, 정확도 차이는 퍼센트포인트다. 코드 주석의 예상 정확도·망각 범위가 모든 실행에 보장되는 것은 아니다.

SimpleNetwork의 생성자 기본 출력은 10개지만 주 실험은 num_classes=2로 생성한다. 과제 수를 임의로 바꾸면 정수 나눗셈 때문에 숫자가 누락될 수 있다. 과제마다 Adam을 새로 만들어 상태를 초기화하며 평가 로스는 배치 평균의 평균이어서 마지막 배치가 작으면 표본 평균과 다르다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
SimpleNetwork의 기본 설정과 주 실험 설정에서 파라메터 수를 구하라.

</div>

??? success "풀이"
    첫 레이어는 200,960개, 둘째 레이어는 65,792개다. 기본 10개 출력 레이어는 2,570개이므로 총 269,322개이고 실험의 2개 출력 레이어는 514개이므로 총 267,266개다. ReLU와 reshape는 학습 파라메터가 없다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 코드는 이미 Adam을 쓴다. SGD와 비교할 때 어떤 조건이 필요한가?

</div>

??? success "풀이"
    같은 초기 웨이트와 데이터 순서·과제 구성·표본 노출 예산을 사용한다. 러닝레이트는 각 방법에 대해 검증하고 새 과제 성능과 이전 과제 유지 성능을 함께 비교한다. 단순 학습 로스의 속도만으로 지속 학습의 우열을 정하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
정확도 98%에서 55%로 떨어졌다면 이 페이지의 망각 단위와 크기는 무엇인가?

</div>

??? success "풀이"
    차이는 43퍼센트포인트다. 43%의 상대 감소와는 다르다. 해당 과제의 학습 직후 값을 기준으로 한 차이며 학습 경로 전체의 최고 정확도 기준과는 다를 수 있다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 풀이의 `ModuleList` 코드가 원래 모델에 바로 적용될 수 있는지 검토하라. 입력·출력 차원, 활성화, 생성자와 `forward`에서 필요한 수정을 설명하라.

</div>

??? success "풀이"
    아래 코드는 깊이를 설정하는 아이디어를 위한 조각이다. 원래 모델의 생성자와 텐서 차원을 반영한 완성 구현은 아니다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `ModuleList`는 하위 모듈을 등록하지만 실행 순서나 활성화를 자동으로 정하지 않는다. 입력 레이어와 출력 레이어는 과제의 차원을 유지하고, 은닉 레이어 사이에 필요한 비선형성을 넣어야 한다. 선형 레이어만 반복하면 아핀 함수 하나로 합쳐진다. 원래 클래스에 없는 `num_layers` 인수를 그대로 넘길 수 없으므로 생성자와 `forward`를 함께 수정해야 한다.

## 정리하며

동일 모델의 과제별 학습 직후와 이후 성능을 비교해 망각을 관찰한다. 로컬 이진 출력의 의미와 행렬 축을 고정하고, 저장된 출력의 수치를 보편적인 기대값으로 해석하지 않는다.
