# 이어 배우기의 평가 지표
## 들어가며

이어 배우기 연구에서 제대로 된 평가는 매우 중요하다. 정확도 하나로 충분할 때가 많은 보통의 기계 학습과 달리, 이어 배우기는 성능의 여러 결을 담아내려면 지표가 여럿 필요하다. 곧 새 과제를 배우는 힘, 옛 과제를 잊는 성향, 그리고 과제 사이의 앎 옮김이다.

이 절은 이어 배우기에서 쓰는 표준 평가 규약과 지표를, 그 수학적 정의와 파이토치 구현과 함께 내보인다.

## 평가 규약

### 정확도 행렬

이어 배우기 평가의 바탕은 **정확도 행렬** $A \in \mathbb{R}^{T \times T}$이며, 여기서 다음과 같다.

$$
A_{i,j} = \text{과제 } j \text{까지 익힌 뒤 과제 } i \text{의 정확도}
$$

핵심 성질은 다음과 같다.

- **아래 삼각**: $j \geq i$인 자리만 정의된다(배우기 전에는 평가할 수 없다)
- **대각선**: $A_{i,i}$은 과제 $i$을 막 배운 직후의 정확도이다
- **마지막 열**: $A_{i,T}$은 과제 $i$의 마지막 정확도이다

과제 5개에 대한 정확도 행렬 보기이다.

```
        After Training Task
        T0      T1      T2      T3      T4
T0    [98.5%   62.3%   55.1%   51.8%   49.2%]
T1    [  -     97.8%   58.6%   52.4%   50.1%]
T2    [  -       -     98.2%   61.3%   53.7%]
T3    [  -       -       -     97.5%   58.9%]
T4    [  -       -       -       -     98.1%]
```

### 평가 절차

```python
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
```

## 으뜸 지표

### 평균 정확도(AA)

가장 근본이 되는 지표로, 학습이 끝난 뒤 모든 과제에 걸친 평균 정확도이다.

$$
\text{AA} = \frac{1}{T} \sum_{i=1}^{T} A_{i,T}
$$

**풀이**: 모든 과제에 걸친 전체 성능이다. 높을수록 좋다.

**한계**: 잊은 것인지 아예 배우지 못한 것인지를 가르지 못한다.

```python
def average_accuracy(accuracy_matrix):
    """
    모든 과제에 걸친 평균 정확도를 셈한다.
    
    인수:
        accuracy_matrix: T x T 정확도 행렬
    
    반환값:
        마지막 평균 정확도(백분율)
    """
    return np.mean(accuracy_matrix[:, -1])
```

### 뒤로의 옮김(BWT)

새 과제를 배우는 일이 옛 과제의 성능에 얼마나 영향을 주는지 잰다.

$$
\text{BWT} = \frac{1}{T-1} \sum_{i=1}^{T-1} (A_{i,T} - A_{i,i})
$$

**풀이**:

- **음의 BWT**: 잊음이 일어났다(흔한 경우)
- **0인 BWT**: 잊음이 없다
- **양의 BWT**: 새 과제를 배워 옛 과제의 성능이 좋아졌다(뒤로의 좋은 옮김)

```python
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
```

### 앞으로의 옮김(FWT)

앞선 배움이 새 과제에 얼마나 도움이 되는지 잰다.

$$
\text{FWT} = \frac{1}{T-1} \sum_{i=2}^{T} (A_{i,i} - A_i^{\text{rand}})
$$

여기서 $A_i^{\text{rand}}$은 아무렇게나 초기화한 모델의 과제 $i$ 정확도이다.

**풀이**:

- **양의 FWT**: 앞선 배움이 새 과제에 도움이 된다
- **0인 FWT**: 옮김이 없다
- **음의 FWT**: 앞선 배움이 새 과제 배우기를 해친다

```python
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
```

### 배움 정확도(LA)

과제를 처음 만났을 때 모델이 그것을 배우는 힘을 잰다.

$$
\text{LA} = \frac{1}{T} \sum_{i=1}^{T} A_{i,i}
$$

**풀이**: 배운 직후의 평균 정확도이다. 잘 배우는 모델이라면 높아야 한다.

```python
def learning_accuracy(accuracy_matrix):
    """
    평균 배움 정확도를 셈한다(대각선의 평균).
    
    인수:
        accuracy_matrix: T x T 정확도 행렬
    
    반환값:
        평균 배움 정확도(백분율)
    """
    return np.mean(np.diag(accuracy_matrix))
```

## 버금 지표

### 잊음 재기(FM)

과제마다 관찰된 가장 큰 잊음이다.

$$
F_i = \max_{j \in \{i, \ldots, T-1\}} (A_{i,j} - A_{i,T})
$$

$$
\text{FM} = \frac{1}{T-1} \sum_{i=1}^{T-1} F_i
$$

**참고**: FM은 마지막 낙폭이 아니라 가장 큰 낙폭을 쓰는데, 잊음이 한결같이 늘지 않는 본새까지 담기 위해서이다.

```python
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
```

### 뻣뻣함 재기(IM)

맨바닥부터 익히는 것에 견주어 새 과제를 배우지 못하는 정도를 잰다.

$$
\text{IM} = \frac{1}{T-1} \sum_{i=2}^{T} (A_i^{\text{joint}} - A_{i,i})
$$

여기서 $A_i^{\text{joint}}$은 과제 $i$을 맨바닥부터(또는 다른 과제와 함께) 익혔을 때의 정확도이다.

**풀이**: 앞선 배움이 새 과제를 익히는 데 얼마나 걸림돌이 되는가?

### 기억 안정성(MS)

때에 따라 옛 과제의 성능이 얼마나 흔들리는지이다.

$$
\text{MS}_i = \text{Var}(A_{i,i}, A_{i,i+1}, \ldots, A_{i,T})
$$

$$
\text{MS} = \frac{1}{T-1} \sum_{i=1}^{T-1} \text{MS}_i
$$

흩어짐이 작을수록 성능이 더 한결같다.

```python
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
```

## 두루 갖춘 평가 클래스

```python
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
```

## 지표 사이의 관계

지표들은 서로 이어져 있다.

$$
\text{AA} \approx \text{LA} + \text{BWT}
$$

이 어림이 성립하는 까닭은 다음과 같다.

- LA는 처음 배우는 힘을 잰다
- BWT는 그 뒤의 변화를 잰다(대개 음수이다)
- AA는 두 효과가 모두 작용한 뒤의 마지막 결과이다

!!! info "지표 조합 풀이하기"
    | LA | BWT | AA | 풀이 |
    |----|-----|-----|----------------|
    | 높음 | 0 근처 | 높음 | 잘 이어 배우는 모델 |
    | 높음 | 크게 음수 | 낮음 | 심한 잊음 |
    | 낮음 | 0 근처 | 낮음 | 배우는 힘이 모자람 |
    | 높음 | 양수 | 아주 높음 | 이로운 옮김 |

## 실용적인 고려

### 여러 번 돌리기

여러 무작위 씨앗에 걸친 평균 ± 표준편차를 늘 알려라.

```python
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
```

### 알림 지침

학계의 표준(Hsu 외, 2018)을 따른다.

1. **늘 알릴 것**: 적어도 AA, BWT, LA
2. **밑금 넣기**: 소박한 차례 학습과 함께 익히기의 위 한계
3. **같은 구조**: 똑같은 망 구조로 방법을 견주기
4. **같은 데이터 쪼갬**: 한결같은 과제 설정을 쓰기
5. **셈 비용**: 학습 시간과 기억 씀씀이를 알리기

## 함께 익히기와의 견줌

**함께 익히기의 위 한계**는 모든 과제를 한꺼번에 익힌다.

```python
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
```

## 요약

| 지표 | 식 | 재는 것 | 좋은 값 |
|--------|---------|----------|------------|
| AA | $\frac{1}{T}\sum_i A_{i,T}$ | 전체 성능 | 높음(80% 넘음) |
| LA | $\frac{1}{T}\sum_i A_{i,i}$ | 배우는 힘 | 높음(90% 넘음) |
| BWT | $\frac{1}{T-1}\sum_i (A_{i,T} - A_{i,i})$ | 잊음 | 0 근처이거나 양수 |
| FWT | $\frac{1}{T-1}\sum_i (A_{i,i} - A_i^{\text{rand}})$ | 앞으로의 옮김 | 양수 |
| FM | 과제마다 가장 큰 잊음 | 최악의 경우 잊음 | 낮음 |
| MS | 때에 따른 흩어짐 | 안정성 | 낮음 |

이 지표들로 제대로 평가하면 이어 배우기 방법을 공정하게 견줄 수 있고, 성능의 여러 결에 걸친 장단점을 짚어낼 수 있다.

## 참고 문헌

1. Lopez-Paz, D., & Ranzato, M. (2017). Gradient episodic memory for continual learning. *NeurIPS*.

2. Chaudhry, A., et al. (2018). Riemannian walk for incremental learning. *ECCV*.

3. Hsu, Y. C., Liu, Y. C., Ramasamy, A., & Kira, Z. (2018). Re-evaluating continual learning scenarios: A categorization and case for strong baselines. *NeurIPS Workshop*.

4. Díaz-Rodríguez, N., et al. (2018). Don't forget, there is more than forgetting: New metrics for continual learning. *NeurIPS Workshop*.

## 연습문제

**연습문제 1.**
이어 배우기의 평균 정확도, 뒤로의 옮김, 앞으로의 옮김을 정의하라.

??? success "연습문제 1 풀이"
    평균 정확도: $\bar{A} = \frac{1}{T}\sum_{i=1}^T a_{T,i}$(과제 $T$까지 익힌 뒤 모든 과제의 정확도). 뒤로의 옮김: $\text{BWT} = \frac{1}{T-1}\sum_{i=1}^{T-1}(a_{T,i} - a_{i,i})$(옛 과제 정확도의 변화). 앞으로의 옮김: $\text{FWT} = \frac{1}{T-1}\sum_{i=2}^T(a_{i-1,i} - b_i)$(앞선 배움 덕분에 얻는 앞으로의 과제에 대한 영 예시 성능).

---

**연습문제 2.**
정확도 행렬 $R_{ij}$과 그것에서 지표를 어떻게 끌어내는지 설명하라.

??? success "연습문제 2 풀이"
    성분 $R_{ij}$은 과제 $1, \ldots, i$까지 익힌 뒤 과제 $j$에서의 정확도이다. 대각선은 과제를 막 배운 직후의 정확도이다. 마지막 행은 모든 과제의 마지막 정확도이다. BWT는 마지막 행과 대각선의 차이를 쓴다. BWT가 음수이면 잊음이 있다는 뜻이다.

---

**연습문제 3.**
이어 배우기 방법을 공정하게 견줄 평가 규약을 설계하라.

??? success "연습문제 3 풀이"

    1. 과제 차례를 붙박아 두라(또는 여러 차례에 걸친 평균을 알려라). 2. 같은 등뼈 구조를 쓰라. 3. 세 지표(AA, BWT, FWT)를 모두 알려라. 4. 위 한계(모든 과제를 함께 익히기)와 아래 한계(이어 배우기 방법 없이 미세 조정하기)를 넣으라. 5. 정확도뿐 아니라 기억과 셈 비용도 알려라.

---

**연습문제 4.**
오늘날 이어 배우기 평가 잣대의 한계는 무엇인가?

??? success "연습문제 4 풀이"
    대부분의 잣대는 과제의 경계가 또렷한 단순한 데이터셋(MNIST, CIFAR)을 쓰는데, 이는 현실과 동떨어져 있다. 현실에서는 과제의 경계가 흐릿하고, 데이터 흐름이 멈추어 있지 않으며, 시험 때 과제 이름표도 없다. 지금의 지표는 셈 효율, 기억 씀씀이, 따로 떼어 둔 과제에서의 성능도 담아내지 못한다.
