# 지속 학습 방법의 비교 실험

단순 순차 학습, 경험 리플레이, EWC와 LwF를 같은 Split MNIST 과제 구성에서 비교하는 예제다. 최종·학습 직후 정확도와 후방 전이, 실행 시간 및 메모리 지표를 수집한다. 비교가 통제한 조건과 아직 통제하지 않은 조건을 구분한다.

## 1. 코드

??? note "코드 (711줄)"

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
        main()
    ```


**출력:**

```
======================================================================
COMPREHENSIVE CONTINUAL LEARNING METHOD COMPARISON
======================================================================

Comparing:
  1. Naive Sequential Learning (baseline)
  2. Experience Replay (memory-based)
  3. Elastic Weight Consolidation (regularization-based)
  4. Learning Without Forgetting (knowledge distillation)
======================================================================

Using device: cpu

Task Configuration:
  Task 0: Classes [0, 1]
  Task 1: Classes [2, 3]
  Task 2: Classes [4, 5]
  Task 3: Classes [6, 7]
  Task 4: Classes [8, 9]

======================================================================
Running Experiments...
======================================================================

[1/4] Naive Sequential Learning...
  Avg Acc: 54.24%, BWT: -56.75%

[2/4] Experience Replay...
  Avg Acc: 94.80%, BWT: -5.88%

[3/4] Elastic Weight Consolidation...
  Avg Acc: 56.63%, BWT: -53.54%

[4/4] Learning Without Forgetting...
  Avg Acc: 76.42%, BWT: -13.67%

========================================================================================================================
COMPREHENSIVE METHOD COMPARISON
========================================================================================================================

Method               Avg Acc      BWT          Learn Acc    Time (s)     Memory (MB) 
------------------------------------------------------------------------------------------------------------------------
Naive                     54.24%     -56.75%      99.64%       1.55       0.00
ExperienceReplay          94.80%      -5.88%      99.50%       2.27       0.00
EWC                       56.63%     -53.54%      99.46%       4.49       0.00
LWF                       76.42%     -13.67%      87.35%       1.92       0.00

========================================================================================================================

Visualization saved as 'continual_learning_comparison.png'

======================================================================
KEY TAKEAWAYS
======================================================================

📊 Performance Ranking (by Avg Accuracy):
  1. ExperienceReplay: 94.80%
  2. LWF: 76.42%
  3. EWC: 56.63%
  4. Naive: 54.24%

💾 Memory Efficiency:
  - EWC & LWF: No example storage (privacy-preserving)
  - Experience Replay: Stores examples (memory overhead)

⚡ Computational Cost:
  - Naive & ER: Single forward/backward per batch
  - EWC: Extra Fisher computation per task
  - LWF: Double forward pass per batch

🎯 When to Use Each Method:
  - Experience Replay: When memory is available, best performance
  - EWC: Privacy concerns, memory constraints
  - LWF: Task domains similar, good knowledge transfer
  - Hybrid approaches: Combine strengths of multiple methods
======================================================================
```

## 2. 논의

네 방법은 같은 784→256→256→2 MLP와 로컬 이진 과제 구성을 사용한다. 다만 시드는 처음 한 번만 설정하고 방법마다 모델을 새로 생성하므로 같은 초기 가중치를 공유하지 않는다. 데이터 로더의 셔플·버퍼 추출·피셔 추정도 난수 상태를 소비해 표본 순서가 달라질 수 있다. 이 한 번의 실행을 일반적인 방법 순위로 해석하지 않는다.

리플레이는 과제마다 고른 표본을 리스트에 추가하고 용량 초과 시 앞부분을 제거한다. 텐서 뷰를 그대로 보관하므로 선택한 묶음의 저장 공간이 더 오래 남을 수 있다. EWC는 정답 기반 표본별 피셔와 과제별 기준점을 저장하고 LwF는 고정 교사 전체의 두 출력 분포를 맞춘다. 추가 과거 표본 학습과 피셔 추정 등 방법별 표본 노출·계산량도 다르다.

memory_usage_mb는 CUDA의 종료 시점 할당량에서 시작 할당량을 뺀 값이다. 피크 메모리나 CPU 버퍼·전체 모델·데이터 메모리를 뜻하지 않으며 CPU·MPS에서는 0을 반환하므로 메모리 사용이 없다고 해석하면 안 된다. 시간 측정에는 명시적 CUDA 동기화가 없어 장치 비동기 실행을 정확히 분리한 벤치마크가 아니다. 그래프의 BWT는 음수가 하락이고 열 지도는 앞 세 방법만 표시한다. AA·LA·BWT는 같은 행렬 정의에서 계산하지만 전방 전이는 측정하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
같은 시드를 실행 시작에 한 번 설정하면 모든 방법이 같은 초기값에서 시작하는가?

</div>

??? success "연습문제 1 풀이"
    각 모델 생성이 난수 상태를 소비하므로 다르다. 같은 초기 state_dict를 복사하고 데이터 순서·추출의 난수도 통제해야 한다. 같은 조건의 반복 실험으로 평균과 변동을 보고한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
CPU 실행에서 memory_usage_mb=0을 어떻게 해석해야 하는가?

</div>

??? success "연습문제 2 풀이"
    함수가 CUDA 할당량만 측정하기 때문에 CPU 사용량을 기록하지 않은 것이다. CPU 버퍼·텐서·모델의 저장량은 별도로 계산하거나 측정해야 한다. 종료−시작 차이도 피크 메모리와 다르다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
최종 평균 정확도만으로 어떤 문제가 숨겨질 수 있는가?

</div>

??? success "연습문제 3 풀이"
    새 과제를 잘 배우지 못한 경우와 잘 배운 뒤 잊은 경우가 같은 최종 평균을 가질 수 있다. 학습 직후 정확도와 과제별 변화, 마지막 행렬을 함께 보고한다. 표본 노출·시간·메모리의 차이도 고려해야 한다.

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

비교 실험은 같은 구조만으로 완성되지 않는다. 초기화·순서·학습 예산과 출력 규약을 맞추고 메모리·시간의 측정 범위를 명확히 해야 한다. 한 실행의 정확도 순위를 일반적인 방법의 우열로 확대하지 않는다.
