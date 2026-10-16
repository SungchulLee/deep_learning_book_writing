# 지속 학습의 평가 지표

지속 학습은 새 과제의 학습 성능과 이전 과제의 유지 성능을 함께 평가한다. 정확도 행렬을 바탕으로 평균 정확도, 후방 전이와 망각을 계산하고, 아직 학습하지 않은 과제의 사전 평가로 전방 전이를 측정한다. 이 페이지에서는 행렬 축을 통일하고 수식과 보존된 코드의 차이를 설명한다.

## 1. 평가 규약

### 정확도 행렬

행은 평가 과제 i, 열은 마지막으로 학습한 과제 j다.

$$A_{i,j}=\text{과제 j까지 학습한 뒤 과제 i의 정확도}$$

j≥i인 학습 후 관측값은 **위 삼각 영역**에 놓인다. 대각선은 각 과제 학습 직후, 마지막 열은 모든 학습이 끝난 뒤의 성능이다. j<i인 값도 아직 학습하지 않은 과제를 평가하면 측정할 수 있다. 아래 평가 루프는 이미 학습한 과제만 측정하고 나머지를 0으로 남기므로, 그 0을 실제 사전 정확도라고 해석하지 않는다.

이 페이지 코드는 정확도를 0~100 백분율로 반환한다. 정확도 차이는 퍼센트포인트, 그 분산은 퍼센트포인트의 제곱 단위다. 0~1 비율과 혼용하지 않는다.

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

np와 torch 임포트, 모델·입력의 장치 설정을 별도로 준비한다. train_fn은 코드가 호출하는 인수 순서를 지원하고 매 과제 학습을 시작할 때 model.train()을 호출해야 한다. 평가 함수는 모델을 eval 상태로 남긴다. 테스트 로더가 비면 정확도의 분모가 0이 된다.

```python
def evaluate_continual_learning(model, test_loaders, train_loaders, 
                                train_fn, epochs_per_task, device):
    """
    이어 배우기의 표준 평가 절차.
    
    인수:
        model: 뉴럴 넷트웍 모델
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

---

## 2. 주요 지표

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

### 후방 전이 BWT

새 과제를 배우는 일이 옛 과제의 성능에 얼마나 영향을 주는지 잰다.

$$
\text{BWT} = \frac{1}{T-1} \sum_{i=1}^{T-1} (A_{i,T} - A_{i,i})
$$

**풀이**:

- **음수**: 평균적으로 학습 직후보다 성능이 낮아졌다.
- **0**: 평균 변화가 없다. 과제별 하락과 상승이 상쇄될 수 있다.
- **양수**: 평균적으로 이전 과제 성능이 개선되었다.

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

### 전방 전이 FWT

학습 이전 과제들이 아직 학습하지 않은 과제 i의 성능에 미친 영향을 기준 모델과 비교한다.

$$\mathrm{FWT}=\frac1{T-1}\sum_{i=2}^T(A_{i,i-1}-b_i)$$

b_i는 같은 출력·과제 조건에서 과제 i를 학습하기 전 기준 성능이다. 측정하려면 각 과제를 학습하기 직전에도 평가해야 한다. 과제별 새 헤드나 출력 확장 때문에 사전 평가 자체가 정의되지 않는 설정이라면 적용 방식을 먼저 정한다.

아래 forward_transfer와 뒤의 클래스는 A_{i,i}에서 무작위 초기화 정확도를 뺀다. 이는 해당 과제를 이미 학습한 성능을 사용하므로 위의 사전 전방 전이와 다르다. 현재 평가 루프가 채운 행렬만으로 위 FWT를 계산할 수 없다.

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

### 학습 정확도(LA)

과제를 처음 만났을 때 모델이 그것을 배우는 힘을 잰다.

$$
\text{LA} = \frac{1}{T} \sum_{i=1}^{T} A_{i,i}
$$

**풀이**: 배운 직후의 평균 정확도이다. 잘 배우는 모델이라면 높아야 한다.

```python
def learning_accuracy(accuracy_matrix):
    """
    평균 러닝 정확도를 셈한다(대각선의 평균).
    
    인수:
        accuracy_matrix: T x T 정확도 행렬
    
    반환값:
        평균 러닝 정확도(백분율)
    """
    return np.mean(np.diag(accuracy_matrix))
```

---

## 3. 추가 지표

### 망각 지표 FM

과제마다 관찰된 가장 큰 잊음이다.

$$
F_i = \max_{j\in\{i,\ldots,T\}} A_{i,j}-A_{i,T}
$$

$$
\text{FM} = \frac{1}{T-1} \sum_{i=1}^{T-1} F_i
$$

이 페이지 코드처럼 마지막 시점까지 최댓값에 포함하면 F_i는 음수가 되지 않는다. 마지막 이전의 최댓값만 사용하는 정의와 구분한다. 학습 직후 성능만 기준으로 삼는 음의 BWT와도 다르다.

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

### 비유연성 지표

동일한 시점까지의 데이터를 공동 학습한 기준 모델과 현재 과제 성능을 비교하여 새 과제 학습의 부족을 평가할 수 있다. 개별 과제만 처음부터 학습한 기준과 공동 학습 기준은 다르므로 하나를 명확히 선택한다.

$$
\text{IM} = \frac{1}{T-1} \sum_{i=2}^{T} (A_i^{\text{joint}} - A_{i,i})
$$

A_i^joint는 과제 1~i의 데이터로 별도 학습한 기준 모델을 과제 i에서 평가한 값이다. 같은 데이터·출력 공간과 학습 예산을 명시한다. 이 페이지에는 이 지표를 계산하는 함수가 없다.

**풀이**: 앞선 학습이 새 과제를 익히는 데 얼마나 걸림돌이 되는가?

### 성능 변동 MS

때에 따라 옛 과제의 성능이 얼마나 흔들리는지이다.

$$
\text{MS}_i = \text{Var}(A_{i,i}, A_{i,i+1}, \ldots, A_{i,T})
$$

$$
\text{MS} = \frac{1}{T-1} \sum_{i=1}^{T-1} \text{MS}_i
$$

분산이 작으면 관측 시점 사이의 변동이 작다. 항상 낮은 정확도도 분산이 0일 수 있으므로 보존 성능이나 정확도가 좋다는 뜻은 아니다. 코드의 np.var는 ddof=0으로 관측 경로의 분산을 계산한다.

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

---

## 4. 지표 계산 클래스

클래스의 forward_transfer는 앞서 설명한 비표준 학습 후 차이를 계산한다. per_task_forgetting은 대각선에서 최종 정확도를 뺀 값으로, 역사적 최댓값을 쓰는 forgetting_measure와 기준이 다르다. 두 출력을 같은 망각 정의로 섞지 않는다. 그래프의 미관측 셀은 0으로 저장되어 있어도 측정된 오답률로 읽지 않는다.

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
        
        # 2. 러닝 정확도와 마지막 정확도
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

---

## 5. 지표 사이의 관계

같은 T×T 행렬과 위 정의를 쓰면 근사가 아닌 다음 등식이 성립한다.

$$\mathrm{AA}=\mathrm{LA}+\frac{T-1}{T}\mathrm{BWT}$$

T=1이면 AA=LA이고 BWT는 편의상 0으로 둔다. 높은 LA와 낮은 AA는 학습 후 성능 저하를, 낮은 LA와 낮은 AA는 새 과제 학습 자체의 부족도 시사한다. 평균 BWT가 0이더라도 과제별 성능은 달라질 수 있으므로 전체 행렬과 경로도 확인한다.

## 6. 실용적인 고려

### 여러 번 돌리기

여러 초기화와 과제 순서에서 반복하고 평균·표준편차 및 반복 수를 보고한다. 아래 evaluate_with_confidence는 이름과 달리 신뢰구간이 아니라 np.std의 모집단 표준편차를 반환한다. 함수는 적어도 한 번 실행되어야 하며, 별도의 무작위 라이브러리나 데이터 로더의 시드도 사용 환경에 맞춰 관리한다.

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

### 결과 보고 항목

비교할 설정과 자원을 함께 보고한다.

1. 과제 순서, 경계와 테스트 시 과제 ID 사용 여부를 적는다.
2. 같은 클래스 분할·모델·출력 공간에서 비교한다.
3. 최종 정확도와 학습 직후 정확도, 망각·전이의 정확한 정의를 적는다.
4. 리플레이 버퍼와 추가 모델·통계의 메모리, 시간·학습 예산을 보고한다.
5. 순차 학습과 공동 학습을 기준으로 포함하되 이론적 하한·상한으로 단정하지 않는다.

---

## 7. 공동 학습 기준과 비교

공동 학습은 모든 과제 데이터를 함께 사용할 수 있을 때의 참고 기준이다. 실험적으로 강한 기준일 수 있지만 모든 지속 학습 방법의 이론적 상한은 아니다. 아래 함수는 합친 데이터셋을 epochs×과제 수만큼 순회하므로 과제별 epochs번 학습보다 표본 노출이 더 많다. 학습 예산을 맞춰 비교하고 학습 모드·모델 이동·임포트도 별도로 준비한다. 서로 다른 원본 클래스를 같은 로컬 라벨로 합쳤다면 공동 학습의 목표도 다시 확인해야 한다.

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

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
이 페이지의 A 행렬 규약에서 AA, BWT, FWT를 정의하라.

</div>

??? success "연습문제 1 풀이"
    AA는 마지막 열의 평균이다. BWT는 이전 과제의 마지막 열과 대각선 차이의 평균이다. FWT는 아직 학습하지 않은 과제 i를 i−1 단계에서 평가한 $A_{i,i-1}$과 학습 전 기준 $b_i$의 차이를 평균 낸다. 코드의 대각선 기반 forward_transfer와 구분한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
A 행렬의 행·열과 미관측 셀을 설명하라.

</div>

??? success "연습문제 2 풀이"
    행은 평가 과제이고 열은 마지막으로 학습한 과제다. 학습 후 관측값은 위 삼각 영역에 있고 최종 성능은 마지막 열이다. 현재 루프는 학습 전 평가를 하지 않으므로 아래 삼각 영역의 0은 미관측 표시이며 실제 정확도가 아니다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
같은 모델로 비교할 때도 추가로 통제할 항목을 설명하라.

</div>

??? success "연습문제 3 풀이"
    데이터 분할과 순서, 출력 공간, 과제 ID 제공 여부, 러닝레이트 선택과 표본 노출 예산을 맞춘다. 과거 표본·생성기·교사 모델·통계 저장도 모두 자원에 포함한다. 테스트 성능으로 하이퍼파라미터를 선택하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
망각 분산이 낮거나 BWT가 0이면 좋은 지속 학습인가?

</div>

??? success "연습문제 4 풀이"
    항상 낮은 성능도 분산이 0이며, 과제별 상승·하락이 상쇄되어 평균 BWT가 0일 수도 있다. 최종·학습 직후 성능과 과제별 경로를 함께 봐야 한다. 태스크·클래스·도메인 증분 등 서로 다른 프로토콜도 구분한다.

## 정리하며

행렬 축과 관측 시점을 먼저 고정하고 지표를 계산한다. AA·LA·BWT의 관계는 정확히 AA=LA+(T−1)BWT/T이며, 사전 FWT에는 별도의 학습 전 평가가 필요하다. 망각 정의와 정확도 단위, 반복 변동과 자원 사용을 함께 보고한다.

**참고 문헌**

1. Lopez-Paz, D., & Ranzato, M. (2017). Gradient episodic memory for continual learning. *NeurIPS*.

2. Chaudhry, A., et al. (2018). Riemannian walk for incremental learning. *ECCV*.

3. Hsu, Y. C., Liu, Y. C., Ramasamy, A., & Kira, Z. (2018). Re-evaluating continual learning scenarios: A categorization and case for strong baselines. *NeurIPS Workshop*.

4. Díaz-Rodríguez, N., et al. (2018). Don't forget, there is more than forgetting: New metrics for continual learning. *NeurIPS Workshop*.
