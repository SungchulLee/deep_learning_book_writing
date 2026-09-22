"""continual_metrics — ch14/continual_learning/evaluation_metrics.md 의 코드를
모듈로 쓸 수 있게 옮겨 놓은 것이다.
"""

def evaluate_continual_learning(model, test_loaders, train_loaders, 
                                train_fn, epochs_per_task, device):
    """
    이어 배우기의 표준 평가 절차.
    
    인수:
        model: 신경망 모델
        test_loaders: 과제마다의 시험 DataLoader 목록
        train_loaders: 과제마다의 학습 DataLoader 목록
        train_fn: 과제 하나로 익히는 함수
        epochs_per_task: 과제마다의 학습 시대 수
        device: 셈할 장치
    
    반환값:
        accuracy_matrix: T x T 정확도 행렬
    """
    num_tasks = len(train_loaders)
    accuracy_matrix = np.zeros((num_tasks, num_tasks))
    
    for task_id in range(num_tasks):
        # 지금 과제로 익힌다
        train_fn(model, train_loaders[task_id], epochs_per_task, device)
        
        # 지금까지 본 모든 과제에서 평가한다
        for eval_id in range(task_id + 1):
            accuracy_matrix[eval_id, task_id] = evaluate_single_task(
                model, test_loaders[eval_id], device
            )
    
    return accuracy_matrix

def evaluate_single_task(model, test_loader, device):
    """과제 하나의 정확도를 셈한다."""
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return 100.0 * correct / total

def average_accuracy(accuracy_matrix):
    """
    모든 과제에 걸친 평균 정확도를 셈한다.
    
    인수:
        accuracy_matrix: T x T 정확도 행렬
    
    반환값:
        마지막 평균 정확도(백분율)
    """
    return np.mean(accuracy_matrix[:, -1])

def backward_transfer(accuracy_matrix):
    """
    뒤로의 옮김을 셈한다(음수면 잊음).
    
    인수:
        accuracy_matrix: T x T 정확도 행렬
    
    반환값:
        평균 뒤로의 옮김(백분율)
    """
    T = accuracy_matrix.shape[0]
    if T <= 1:
        return 0.0
    
    bwt = 0.0
    for i in range(T - 1):
        bwt += accuracy_matrix[i, -1] - accuracy_matrix[i, i]
    
    return bwt / (T - 1)

def forward_transfer(accuracy_matrix, random_init_accuracies):
    """
    앞으로의 옮김을 셈한다.
    
    인수:
        accuracy_matrix: T x T 정확도 행렬
        random_init_accuracies: 과제마다 무작위 모델의 정확도
    
    반환값:
        평균 앞으로의 옮김(백분율)
    """
    T = accuracy_matrix.shape[0]
    if T <= 1:
        return 0.0
    
    fwt = 0.0
    for i in range(1, T):
        fwt += accuracy_matrix[i, i] - random_init_accuracies[i]
    
    return fwt / (T - 1)

def learning_accuracy(accuracy_matrix):
    """
    평균 배움 정확도를 셈한다(대각선의 평균).
    
    인수:
        accuracy_matrix: T x T 정확도 행렬
    
    반환값:
        평균 배움 정확도(백분율)
    """
    return np.mean(np.diag(accuracy_matrix))

def forgetting_measure(accuracy_matrix):
    """
    잊음 재기를 셈한다(과제마다 가장 큰 잊음).
    
    인수:
        accuracy_matrix: T x T 정확도 행렬
    
    반환값:
        평균 최대 잊음(백분율)
    """
    T = accuracy_matrix.shape[0]
    if T <= 1:
        return 0.0
    
    forgetting = 0.0
    for i in range(T - 1):
        # 어느 시점에서든 다다른 가장 높은 정확도
        max_acc = np.max(accuracy_matrix[i, i:])
        # 마지막 정확도
        final_acc = accuracy_matrix[i, -1]
        forgetting += max_acc - final_acc
    
    return forgetting / (T - 1)

def memory_stability(accuracy_matrix):
    """
    기억 안정성을 셈한다(흩어짐이 작을수록 안정적이다).
    
    인수:
        accuracy_matrix: T x T 정확도 행렬
    
    반환값:
        과제 성능의 평균 흩어짐
    """
    T = accuracy_matrix.shape[0]
    if T <= 1:
        return 0.0
    
    stability = 0.0
    for i in range(T - 1):
        # 때에 따른 과제 i 정확도의 흩어짐
        task_accs = accuracy_matrix[i, i:]
        stability += np.var(task_accs)
    
    return stability / (T - 1)

class ContinualLearningMetrics:
    """
    이어 배우기의 지표를 두루 셈하기.
    
    이 클래스는 정확도 행렬에서 표준 지표를 모두 셈해 담아 둔다.
    """
    
    def __init__(self, accuracy_matrix, random_init_accuracies=None):
        """
        정확도 행렬로 초기화한다.
        
        인수:
            accuracy_matrix: T x T 정확도 넘파이 배열
            random_init_accuracies: 선택할 수 있는 밑금 정확도
        """
        self.accuracy_matrix = accuracy_matrix
        self.T = accuracy_matrix.shape[0]
        self.random_init = random_init_accuracies
        
        # 모든 지표를 셈한다
        self._compute_metrics()
    
    def _compute_metrics(self):
        """모든 지표를 셈한다."""
        # 으뜸 지표
        self.average_accuracy = np.mean(self.accuracy_matrix[:, -1])
        self.learning_accuracy = np.mean(np.diag(self.accuracy_matrix))
        
        # 뒤로의 옮김
        if self.T > 1:
            bwt = sum(self.accuracy_matrix[i, -1] - self.accuracy_matrix[i, i] 
                     for i in range(self.T - 1))
            self.backward_transfer = bwt / (self.T - 1)
        else:
            self.backward_transfer = 0.0
        
        # 앞으로의 옮김(밑금이 주어지면)
        if self.random_init is not None and self.T > 1:
            fwt = sum(self.accuracy_matrix[i, i] - self.random_init[i] 
                     for i in range(1, self.T))
            self.forward_transfer = fwt / (self.T - 1)
        else:
            self.forward_transfer = None
        
        # 잊음 재기
        if self.T > 1:
            fm = 0.0
            for i in range(self.T - 1):
                max_acc = np.max(self.accuracy_matrix[i, i:])
                fm += max_acc - self.accuracy_matrix[i, -1]
            self.forgetting_measure = fm / (self.T - 1)
        else:
            self.forgetting_measure = 0.0
        
        # 과제별 잊음
        self.per_task_forgetting = []
        for i in range(self.T - 1):
            self.per_task_forgetting.append(
                self.accuracy_matrix[i, i] - self.accuracy_matrix[i, -1]
            )
        
        # 기억 안정성
        if self.T > 1:
            stability = sum(np.var(self.accuracy_matrix[i, i:]) 
                           for i in range(self.T - 1))
            self.memory_stability = stability / (self.T - 1)
        else:
            self.memory_stability = 0.0
    
    def summary(self):
        """간추림 사전을 되돌린다."""
        return {
            'average_accuracy': self.average_accuracy,
            'learning_accuracy': self.learning_accuracy,
            'backward_transfer': self.backward_transfer,
            'forward_transfer': self.forward_transfer,
            'forgetting_measure': self.forgetting_measure,
            'memory_stability': self.memory_stability,
            'per_task_forgetting': self.per_task_forgetting
        }
    
    def print_report(self):
        """모양을 갖춘 지표 보고를 찍는다."""
        print("=" * 60)
        print("CONTINUAL LEARNING METRICS REPORT")
        print("=" * 60)
        
        print(f"\n📊 Primary Metrics:")
        print(f"   Average Accuracy (AA):      {self.average_accuracy:.2f}%")
        print(f"   Learning Accuracy (LA):     {self.learning_accuracy:.2f}%")
        print(f"   Backward Transfer (BWT):    {self.backward_transfer:+.2f}%")
        if self.forward_transfer is not None:
            print(f"   Forward Transfer (FWT):     {self.forward_transfer:+.2f}%")
        
        print(f"\n📉 Forgetting Analysis:")
        print(f"   Forgetting Measure (FM):    {self.forgetting_measure:.2f}%")
        print(f"   Memory Stability (MS):      {self.memory_stability:.2f}")
        
        print(f"\n📋 Per-Task Forgetting:")
        for i, f in enumerate(self.per_task_forgetting):
            print(f"   Task {i}: {f:+.2f}%")
        
        print("=" * 60)
    
    def plot_metrics(self, save_path=None):
        """지표를 그려 본다."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 정확도 행렬 열 지도
        ax1 = axes[0, 0]
        im = ax1.imshow(self.accuracy_matrix, cmap='RdYlGn', 
                        vmin=0, vmax=100, aspect='auto')
        ax1.set_xlabel('After Training Task')
        ax1.set_ylabel('Evaluated Task')
        ax1.set_title('Accuracy Matrix', fontweight='bold')
        ax1.set_xticks(range(self.T))
        ax1.set_yticks(range(self.T))
        plt.colorbar(im, ax=ax1, label='Accuracy (%)')
        
        # 글자 주석을 추가한다
        for i in range(self.T):
            for j in range(self.T):
                if j >= i:
                    ax1.text(j, i, f'{self.accuracy_matrix[i,j]:.0f}',
                            ha='center', va='center', fontsize=9)
        
        # 2. 배움 정확도와 마지막 정확도
        ax2 = axes[0, 1]
        x = np.arange(self.T)
        width = 0.35
        learning = np.diag(self.accuracy_matrix)
        final = self.accuracy_matrix[:, -1]
        
        ax2.bar(x - width/2, learning, width, label='Learning Acc', 
                color='skyblue', alpha=0.8)
        ax2.bar(x + width/2, final, width, label='Final Acc',
                color='coral', alpha=0.8)
        ax2.set_xlabel('Task')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Learning vs Final Accuracy', fontweight='bold')
        ax2.legend()
        ax2.set_ylim([0, 105])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. 과제별 잊음
        ax3 = axes[1, 0]
        colors = ['red' if f > 0 else 'green' for f in self.per_task_forgetting]
        ax3.bar(range(len(self.per_task_forgetting)), 
                self.per_task_forgetting, color=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax3.set_xlabel('Task')
        ax3.set_ylabel('Forgetting (%)')
        ax3.set_title('Per-Task Forgetting', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. 정확도의 자취
        ax4 = axes[1, 1]
        for i in range(self.T):
            accs = [self.accuracy_matrix[i, j] if j >= i else np.nan 
                   for j in range(self.T)]
            ax4.plot(range(self.T), accs, marker='o', linewidth=2,
                    label=f'Task {i}')
        ax4.set_xlabel('Training Stage')
        ax4.set_ylabel('Accuracy (%)')
        ax4.set_title('Accuracy Trajectories', fontweight='bold')
        ax4.legend(loc='best', fontsize=9)
        ax4.set_ylim([0, 105])
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

def evaluate_with_confidence(run_experiment_fn, num_runs=5):
    """
    실험을 여러 번 돌려 믿음 구간을 셈한다.
    
    인수:
        run_experiment_fn: 정확도 행렬을 되돌리는 함수
        num_runs: 서로 독립인 실행의 횟수
    
    반환값:
        지표마다 평균과 표준편차를 담은 사전
    """
    all_metrics = []
    
    for run in range(num_runs):
        torch.manual_seed(run * 42)
        np.random.seed(run * 42)
        
        accuracy_matrix = run_experiment_fn()
        metrics = ContinualLearningMetrics(accuracy_matrix)
        all_metrics.append(metrics.summary())
    
    # 결과를 모은다
    results = {}
    for key in all_metrics[0].keys():
        if key == 'per_task_forgetting':
            continue
        values = [m[key] for m in all_metrics if m[key] is not None]
        if values:
            results[f'{key}_mean'] = np.mean(values)
            results[f'{key}_std'] = np.std(values)
    
    return results

def joint_training_baseline(model, all_loaders, test_loaders, 
                            epochs, device):
    """
    모든 과제를 함께 익힌다(위 한계 밑금).
    
    이는 이어 배우기의 제약 없이 다다를 수 있는
    가장 좋은 성능을 나타낸다.
    """
    from torch.utils.data import ConcatDataset
    
    # 학습 데이터를 모두 합친다
    combined_dataset = ConcatDataset([
        loader.dataset for loader in all_loaders
    ])
    combined_loader = DataLoader(
        combined_dataset, batch_size=128, shuffle=True
    )
    
    # 함께 익힌다
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs * len(all_loaders)):  # 시대 수를 조절한다
        for data, target in combined_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    # 모든 과제에서 평가한다
    return [evaluate_single_task(model, loader, device) 
            for loader in test_loaders]
