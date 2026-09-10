# 탄성 가중치 다지기(EWC)

**탄성 가중치 다지기(EWC)**는 Kirkpatrick 외(2017)가 들여온, 이어 배우기의 벌주기 기반 방법 가운데 가장 큰 영향을 끼친 것에 든다. EWC의 핵심 통찰은 앞서 배운 과제에서 신경망의 매개변수가 다 똑같이 중요하지는 않다는 것이다. 어떤 매개변수는 결정적이고 어떤 것은 마음대로 바꾸어도 된다. EWC는 **피셔 정보 행렬**로 중요한 매개변수를 짚어내어 지킨다.

!!! success "핵심 이점"

    - **데이터를 담아 두지 않음**: 사생활을 지킨다(앞선 보기를 담아 두지 않는다)
    - **붙박이 기억**: 과제가 늘어도 기억이 커지지 않는다
    - **원칙 있는 바탕**: 베이즈 추론과 정보 이론에 뿌리를 둔다

---

## 1. 이론적 바탕

### 베이즈의 눈

EWC는 배움을 보는 베이즈의 눈에서 나온다. 과제 A를 배운 뒤 매개변수 위의 뒤확률을 생각해 보자.

$$
p(\theta | \mathcal{D}_A) \propto p(\mathcal{D}_A | \theta) p(\theta)
$$

과제 B를 배울 때 우리는 B에도 좋으면서 A에서 배운 것과도 어긋나지 않는 매개변수를 찾고 싶다. 가장 좋은 길은 A의 뒤확률을 B의 앞확률로 쓰는 것이다.

$$
\log p(\theta | \mathcal{D}_A, \mathcal{D}_B) = \log p(\mathcal{D}_B | \theta) + \log p(\theta | \mathcal{D}_A) + \text{const}
$$

어려움은 신경망에서 $p(\theta | \mathcal{D}_A)$을 다룰 수 없다는 점이다.

### 라플라스 어림

EWC는 뒤확률 $p(\theta | \mathcal{D}_A)$을 가우스 분포로 어림한다.

$$
p(\theta | \mathcal{D}_A) \approx \mathcal{N}(\theta_A^*, F_A^{-1})
$$

여기서 각 기호는 다음과 같다.

- $\theta_A^*$은 과제 A 뒤의 가장 좋은 매개변수이다
- $F_A$은 피셔 정보 행렬이다

이 가우스 분포의 로그를 취하면 이차 벌이 나온다.

$$
\log p(\theta | \mathcal{D}_A) \approx -\frac{1}{2}(\theta - \theta_A^*)^T F_A (\theta - \theta_A^*) + \text{const}
$$

### 피셔 정보 행렬

피셔 정보 행렬은 데이터가 매개변수마다에 대해 얼마나 많은 정보를 주는지를 수로 나타낸다.

$$
F = \mathbb{E}_{x \sim p(x)} \left[ \nabla_\theta \log p(y|x, \theta) \nabla_\theta \log p(y|x, \theta)^T \right]
$$

**핵심 성질**: 피셔 정보가 큰 매개변수는 조금만 바뀌어도 모델의 예측이 크게 달라지는 매개변수이며, 바로 이들이 지켜야 할 중요한 매개변수이다.

#### 대각 어림

매개변수가 $d$개이면 온전한 피셔 행렬은 $d \times d$이라 셈으로 다룰 수 없다. EWC는 대각 어림을 쓴다.

$$
F_i = \mathbb{E}_{x \sim p(x)} \left[ \left( \frac{\partial \log p(y|x, \theta)}{\partial \theta_i} \right)^2 \right]
$$

이로써 저장이 $O(d^2)$에서 $O(d)$으로 줄어든다.

---

## 2. EWC의 손실 함수

과제 $\tau$을 배울 때 온전한 EWC 손실은 다음과 같다.

$$
\mathcal{L}(\theta) = \mathcal{L}_\tau(\theta) + \frac{\lambda}{2} \sum_{i} F_i (\theta_i - \theta_{\tau-1,i}^*)^2
$$

여기서 각 기호는 다음과 같다.

- $\mathcal{L}_\tau(\theta)$: 지금 과제의 손실
- $F_i$: 매개변수 $i$의 피셔 정보(앞선 과제에서 구한 것)
- $\theta_{\tau-1}^*$: 앞선 과제 뒤의 가장 좋은 매개변수
- $\lambda$: 벌주기 세기(초매개변수)

!!! info "직관"
    벌 항은 매개변수마다를 앞서의 가장 좋은 값에 이어 주는 용수철 노릇을 한다. 용수철 상수(뻣뻣함)는 피셔 정보가 정한다. 중요한 매개변수는 뻣뻣한 용수철을, 중요하지 않은 매개변수는 느슨한 용수철을 갖는다.

---

## 3. PyTorch 구현

### 온전한 EWC 학습기 클래스

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Optional, List
import copy

class EWCLearner:
    """
    이어 배우기를 위한 탄성 가중치 다지기.
    
    EWC는 다음으로 파국적 잊음을 막는다.
    1. 과제를 마칠 때마다 피셔 정보를 셈한다(매개변수 중요도)
    2. 중요한 매개변수를 지키려고 이차 벌을 더한다
    3. 새 과제 배우기와 옛 앎 지키기의 균형을 잡는다
    
    참고: Kirkpatrick 외, "Overcoming catastrophic forgetting
               in neural networks," PNAS 2017
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 ewc_lambda: float = 5000.0,
                 learning_rate: float = 0.001,
                 fisher_sample_size: Optional[int] = None):
        """
        EWC 학습기를 초기화한다.
        
        인수:
            model: 신경망 모델
            device: 셈할 장치(CPU나 GPU)
            ewc_lambda: 벌주기 세기(λ)
                       값이 클수록 옛 매개변수를 더 세게 지킨다
                       흔한 범위: [100, 10000]
            learning_rate: 최적화기의 학습률
            fisher_sample_size: 피셔를 셈하는 데 쓸 표본의 개수
                               (None이면 데이터셋 전체를 쓴다)
        """
        self.model = model
        self.device = device
        self.ewc_lambda = ewc_lambda
        self.learning_rate = learning_rate
        self.fisher_sample_size = fisher_sample_size
        
        self.criterion = nn.CrossEntropyLoss()
        
        # 피셔 정보와 가장 좋은 매개변수를 담을 곳
        # 열쇠: task_id, 값: Dict[param_name -> 텐서]
        self.fisher_dict: Dict[int, Dict[str, torch.Tensor]] = {}
        self.optpar_dict: Dict[int, Dict[str, torch.Tensor]] = {}
        
        self.current_task = 0
    
    def compute_fisher_information(self, 
                                   data_loader,
                                   task_id: int) -> Dict[str, torch.Tensor]:
        """
        대각 피셔 정보 행렬을 셈한다.
        
        매개변수 θ_i의 피셔 정보 F_i은 다음과 같다.
        F_i = E[(∂log p(y|x,θ)/∂θ_i)²]
        
        이를 실험으로 어림한다.
        F_i ≈ (1/N) Σ_n (∂L(x_n, θ)/∂θ_i)²
        
        인수:
            data_loader: 피셔를 셈할 DataLoader
            task_id: 지금 과제의 식별자
        
        반환값:
            매개변수 이름을 피셔 값으로 옮기는 사전
        """
        self.model.eval()
        
        # 피셔를 쌓을 그릇을 초기화한다
        fisher = {name: torch.zeros_like(param.data)
                 for name, param in self.model.named_parameters()
                 if param.requires_grad}
        
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(data_loader):
            # 표본 한계를 살핀다
            if (self.fisher_sample_size is not None and 
                total_samples >= self.fisher_sample_size):
                break
            
            data, target = data.to(self.device), target.to(self.device)
            batch_size = data.size(0)
            
            # 순전파
            self.model.zero_grad()
            output = self.model(data)
            
            # 로그 확률을 셈한다
            log_probs = F.log_softmax(output, dim=1)
            
            # 표본마다 기울기의 제곱을 셈한다
            for i in range(batch_size):
                if (self.fisher_sample_size is not None and 
                    total_samples >= self.fisher_sample_size):
                    break
                
                self.model.zero_grad()
                
                # 참 부류의 로그 확률
                log_prob = log_probs[i, target[i]]
                
                # 거꾸로 퍼뜨린다
                log_prob.backward(retain_graph=(i < batch_size - 1))
                
                # 기울기 제곱 누적
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fisher[name] += param.grad.data ** 2
                
                total_samples += 1
        
        # 표본에 걸쳐 평균 낸다
        for name in fisher:
            fisher[name] /= total_samples
        
        return fisher
    
    def compute_ewc_penalty(self) -> torch.Tensor:
        """
        EWC 벌주기 항을 셈한다.
        
        Penalty = (λ/2) Σ_tasks Σ_i F_i(task) (θ_i - θ*_i(task))²
        
        반환값:
            EWC 벌 항(스칼라 텐서)
        """
        penalty = torch.tensor(0.0, device=self.device)
        
        # 앞선 모든 과제에 걸쳐 더한다
        for task_id in self.fisher_dict.keys():
            for name, param in self.model.named_parameters():
                if name in self.fisher_dict[task_id]:
                    fisher = self.fisher_dict[task_id][name]
                    optpar = self.optpar_dict[task_id][name]
                    
                    # 피셔로 무게 준 이차 벌
                    penalty += (fisher * (param - optpar) ** 2).sum()
        
        return (self.ewc_lambda / 2) * penalty
    
    def train_on_task(self,
                      train_loader,
                      test_loader,
                      epochs: int = 5,
                      verbose: bool = True) -> Dict:
        """
        EWC 벌주기를 곁들여 과제 하나로 익힌다.
        
        인수:
            train_loader: 학습 데이터 로더
            test_loader: 평가에 쓸 시험 데이터 로더
            epochs: 학습 에포크 수
            verbose: 학습 진행을 찍는다
        
        반환값:
            학습 통계를 담은 사전
        """
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        train_losses = []
        ewc_losses = []
        task_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_ewc = 0.0
            epoch_task = 0.0
            num_batches = 0
            
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                # 순전파
                output = self.model(data)
                
                # 과제마다의 손실
                task_loss = self.criterion(output, target)
                
                # EWC 벌(앞선 과제가 있을 때만)
                if len(self.fisher_dict) > 0:
                    ewc_loss = self.compute_ewc_penalty()
                else:
                    ewc_loss = torch.tensor(0.0, device=self.device)
                
                # 전체 손실
                total_loss = task_loss + ewc_loss
                
                # 역전파
                total_loss.backward()
                optimizer.step()
                
                # 통계 기록
                epoch_loss += total_loss.item()
                epoch_ewc += ewc_loss.item()
                epoch_task += task_loss.item()
                num_batches += 1
            
            # 손실을 평균 낸다
            avg_loss = epoch_loss / num_batches
            avg_ewc = epoch_ewc / num_batches
            avg_task = epoch_task / num_batches
            
            train_losses.append(avg_loss)
            ewc_losses.append(avg_ewc)
            task_losses.append(avg_task)
            
            if verbose:
                print(f"  Epoch {epoch+1}/{epochs}: "
                      f"Total={avg_loss:.4f}, Task={avg_task:.4f}, EWC={avg_ewc:.4f}")
        
        return {
            'train_losses': train_losses,
            'ewc_losses': ewc_losses,
            'task_losses': task_losses
        }
    
    def consolidate_task(self, data_loader, task_id: int):
        """
        과제를 배운 뒤 앎을 다진다.
        
        여기서 다음을 셈해 담아 둔다.
        1. 피셔 정보(매개변수 중요도)
        2. 가장 좋은 매개변수(벌의 기준점)
        
        인수:
            data_loader: 피셔를 셈할 데이터 로더
            task_id: 과제 식별자
        """
        print(f"  Consolidating Task {task_id}...")
        
        # 피셔 정보를 셈한다
        fisher = self.compute_fisher_information(data_loader, task_id)
        self.fisher_dict[task_id] = fisher
        
        # 가장 좋은 매개변수를 담아 둔다
        optpar = {name: param.data.clone()
                 for name, param in self.model.named_parameters()
                 if param.requires_grad}
        self.optpar_dict[task_id] = optpar
        
        # 피셔 통계를 찍는다
        all_fisher = torch.cat([f.flatten() for f in fisher.values()])
        print(f"  Fisher stats - Mean: {all_fisher.mean():.6f}, "
              f"Max: {all_fisher.max():.6f}, "
              f"Sparsity: {(all_fisher < 1e-6).float().mean():.2%}")
    
    def evaluate(self, test_loader) -> float:
        """과제에서 모델의 정확도를 평가한다."""
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
    
    def train_continual(self,
                        train_loaders: List,
                        test_loaders: List,
                        epochs_per_task: int = 5) -> Dict:
        """
        온전한 이어 배우기 파이프라인.
        
        인수:
            train_loaders: 학습 DataLoader 목록
            test_loaders: 시험 DataLoader 목록
            epochs_per_task: 과제마다의 학습 시대 수
        
        반환값:
            정확도 행렬과 지표를 담은 사전
        """
        num_tasks = len(train_loaders)
        accuracy_matrix = np.zeros((num_tasks, num_tasks))
        
        for task_id in range(num_tasks):
            print(f"\n{'='*60}")
            print(f"Task {task_id}")
            print('='*60)
            
            # 지금 과제로 익힌다
            self.train_on_task(
                train_loaders[task_id],
                test_loaders[task_id],
                epochs=epochs_per_task
            )
            
            # 다진다(피셔를 셈하고 매개변수를 담아 둔다)
            self.consolidate_task(train_loaders[task_id], task_id)
            
            # 지금까지 본 모든 과제에서 평가한다
            print(f"\n  Evaluation:")
            for eval_id in range(task_id + 1):
                acc = self.evaluate(test_loaders[eval_id])
                accuracy_matrix[eval_id, task_id] = acc
                
                if eval_id < task_id:
                    original = accuracy_matrix[eval_id, eval_id]
                    change = acc - original
                    print(f"    Task {eval_id}: {acc:.1f}% "
                          f"(was {original:.1f}%, change: {change:+.1f}%)")
                else:
                    print(f"    Task {eval_id}: {acc:.1f}%")
        
        return {
            'accuracy_matrix': accuracy_matrix,
            'num_tasks': num_tasks
        }
```

### 쓰는 보기

```python
import numpy as np

# 설정
num_tasks = 5
epochs_per_task = 5
ewc_lambda = 5000.0  # 핵심 초매개변수

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 생성
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 2)  # 과제마다 이진 분류
).to(device)

# EWC 학습기를 초기화한다
ewc_learner = EWCLearner(
    model=model,
    device=device,
    ewc_lambda=ewc_lambda,
    learning_rate=0.001
)

# 데이터를 불러와 과제 로더를 만든다(앞 절에서 보인 대로)
# train_loaders, test_loaders = create_split_mnist_tasks(...)

# 이어 배우기를 돌린다
results = ewc_learner.train_continual(
    train_loaders, 
    test_loaders,
    epochs_per_task=epochs_per_task
)

# 지표를 계산한다
from continual_metrics import ContinualLearningMetrics
metrics = ContinualLearningMetrics(results['accuracy_matrix'])
metrics.print_report()
```

---

## 4. 초매개변수 분석

### λ 매개변수

벌주기 세기 $\lambda$은 안정성과 말랑함의 맞바꿈을 다스린다.

| λ 값 | 효과 |
|---------|--------|
| 너무 낮음(이를테면 10) | 약한 지킴 → 많이 잊음 |
| 알맞음(이를테면 1000~5000) | 지킴과 배움의 균형 |
| 너무 높음(이를테면 100000) | 지나친 지킴 → 새 과제를 배우지 못함 |

```python
def analyze_lambda_sensitivity(train_loaders, test_loaders, 
                               lambda_values, device):
    """
    λ 초매개변수에 대한 민감도를 뜯어본다.
    
    인수:
        train_loaders: 학습 데이터 로더
        test_loaders: 시험 데이터 로더
        lambda_values: 시험할 λ 값 목록
        device: 셈할 장치
    
    반환값:
        λ을 지표로 옮기는 사전
    """
    results = {}
    
    for ewc_lambda in lambda_values:
        print(f"\nTesting λ = {ewc_lambda}")
        
        # λ마다 새 모델
        model = create_model().to(device)
        
        learner = EWCLearner(
            model=model,
            device=device,
            ewc_lambda=ewc_lambda
        )
        
        result = learner.train_continual(
            train_loaders, test_loaders, epochs_per_task=5
        )
        
        metrics = ContinualLearningMetrics(result['accuracy_matrix'])
        results[ewc_lambda] = {
            'avg_accuracy': metrics.average_accuracy,
            'backward_transfer': metrics.backward_transfer,
            'learning_accuracy': metrics.learning_accuracy
        }
        
        print(f"  AA: {metrics.average_accuracy:.1f}%, "
              f"BWT: {metrics.backward_transfer:+.1f}%")
    
    return results
```

### λ 효과 그려 보기

```python
def plot_lambda_analysis(results):
    """λ이 여러 지표에 미치는 영향을 그린다."""
    lambdas = list(results.keys())
    aa = [results[l]['avg_accuracy'] for l in lambdas]
    bwt = [results[l]['backward_transfer'] for l in lambdas]
    la = [results[l]['learning_accuracy'] for l in lambdas]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 평균 정확도
    axes[0].semilogx(lambdas, aa, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('λ (log scale)')
    axes[0].set_ylabel('Average Accuracy (%)')
    axes[0].set_title('AA vs λ')
    axes[0].grid(True, alpha=0.3)
    
    # 뒤로의 옮김
    axes[1].semilogx(lambdas, bwt, 'o-', linewidth=2, markersize=8, color='red')
    axes[1].axhline(y=0, color='black', linestyle='--')
    axes[1].set_xlabel('λ (log scale)')
    axes[1].set_ylabel('Backward Transfer (%)')
    axes[1].set_title('BWT vs λ')
    axes[1].grid(True, alpha=0.3)
    
    # 배움 정확도
    axes[2].semilogx(lambdas, la, 'o-', linewidth=2, markersize=8, color='green')
    axes[2].set_xlabel('λ (log scale)')
    axes[2].set_ylabel('Learning Accuracy (%)')
    axes[2].set_title('LA vs λ')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ewc_lambda_analysis.png', dpi=150)
    plt.show()
```

---

## 5. EWC의 변형

### 온라인 EWC

과제마다 피셔 행렬을 따로 담아 두는 대신 온라인 EWC는 흐르는 어림값 하나를 지닌다.

$$
\tilde{F} = \gamma \tilde{F}_{\text{old}} + F_{\text{new}}
$$

여기서 $\gamma \in [0, 1]$은 감쇠 인자이다.

```python
class OnlineEWC(EWCLearner):
    """
    흐르는 피셔 어림값을 쓰는 온라인 EWC.
    
    과제마다 행렬을 따로 담아 두는 대신 과제에 걸쳐 중요도를 쌓는
    피셔 행렬 하나를 지닌다.
    """
    
    def __init__(self, model, device, ewc_lambda=5000.0, 
                 gamma=0.9, **kwargs):
        super().__init__(model, device, ewc_lambda, **kwargs)
        self.gamma = gamma
        self.running_fisher = None
        self.running_optpar = None
    
    def consolidate_task(self, data_loader, task_id):
        """흐르는 피셔 어림값을 고친다."""
        new_fisher = self.compute_fisher_information(data_loader, task_id)
        
        if self.running_fisher is None:
            self.running_fisher = new_fisher
        else:
            # 옛 피셔를 줄이고 새 피셔를 더한다
            for name in self.running_fisher:
                self.running_fisher[name] = (
                    self.gamma * self.running_fisher[name] + 
                    new_fisher[name]
                )
        
        # 참조 매개변수를 고친다
        self.running_optpar = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
    
    def compute_ewc_penalty(self):
        """흐르는 피셔로 벌을 셈한다."""
        if self.running_fisher is None:
            return torch.tensor(0.0, device=self.device)
        
        penalty = torch.tensor(0.0, device=self.device)
        
        for name, param in self.model.named_parameters():
            if name in self.running_fisher:
                fisher = self.running_fisher[name]
                optpar = self.running_optpar[name]
                penalty += (fisher * (param - optpar) ** 2).sum()
        
        return (self.ewc_lambda / 2) * penalty
```

### EWC++

EWC++(Chaudhry 외, 2018)은 두 가지를 낫게 한다.

1. **과제마다 따로 벌주기**: 앞선 과제마다 다른 λ을 쓴다
2. **고른 피셔**: 피셔 값이 터지지 않도록 눈금을 맞춘다

```python
class EWCPlusPlus(EWCLearner):
    """
    고른 피셔와 과제마다의 람다를 쓰는 EWC++.
    """
    
    def consolidate_task(self, data_loader, task_id):
        """고른 피셔를 셈한다."""
        fisher = self.compute_fisher_information(data_loader, task_id)
        
        # 층마다 피셔를 고른다
        for name in fisher:
            f = fisher[name]
            if f.max() > 0:
                fisher[name] = f / f.max()
        
        self.fisher_dict[task_id] = fisher
        self.optpar_dict[task_id] = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
```

---

## 6. 계산에 대한 고려

### 시간 복잡도

| 연산 | 복잡도 | 언제 |
|-----------|------------|------|
| 학습 앞먹임과 되돌림 | $O(N \cdot d)$ | 배치마다 |
| EWC 벌 셈하기 | $O(T \cdot d)$ | 배치마다(T는 과제 수) |
| 피셔 셈하기 | $O(N \cdot d)$ | 과제를 마칠 때마다 |

### 메모리 복잡도

| 저장 | 크기 | 커짐 |
|---------|------|--------|
| 모델 매개변수 | $O(d)$ | 그대로 |
| 과제마다의 피셔 | $O(d)$ | T에 선형 |
| 과제마다의 가장 좋은 매개변수 | $O(d)$ | T에 선형 |
| **EWC의 전체 짐** | $O(T \cdot d)$ | 과제 수에 선형 |

!!! warning "규모 확장성"
    과제 차례가 아주 길면 기억을 그대로 유지하도록 온라인 EWC를 생각해 보라.

---

## 7. 장점과 한계

### 이점

1. **데이터를 담아 두지 않음**: 사생활을 지키며 앞선 보기를 담아 둘 필요가 없다
2. **이론적 바탕**: 베이즈 원리에서 나왔다
3. **셈 효율**: 학습의 짐이 아주 적다
4. **풀이 가능성**: 피셔 값이 어떤 매개변수가 중요한지 보여 준다

### 한계

1. **대각 어림**: 매개변수 사이의 상관을 무시한다
2. **쌓이는 제약**: 과제를 거치며 제약이 겹겹이 쌓인다
3. **과제마다의 λ**: 가장 좋은 λ이 과제마다 다를 수 있다
4. **용량의 한계**: 끝내 말랑한 매개변수가 바닥난다

---

## 8. 기대되는 결과

기본 초매개변수로 Split MNIST에서는 다음과 같다.

| 지표 | 소박한 밑금 | EWC |
|--------|----------------|-----|
| 평균 정확도 | 55% 남짓 | 85% 남짓 |
| 뒤로의 옮김 | -45% 남짓 | -12% 남짓 |
| 배움 정확도 | 98% 남짓 | 95% 남짓 |

EWC는 배우는 힘을 잘 지키면서 잊음을 크게 줄인다.

---

## 연습문제

**연습문제 1.**
베이즈의 눈으로 EWC 손실 함수를 끌어내라.

??? success "연습문제 1 풀이"
    과제 A 뒤의 뒤확률은 $p(\theta|D_A) \propto p(D_A|\theta)p(\theta)$이다. 과제 B에서는 $p(\theta|D_A, D_B) \propto p(D_B|\theta)p(\theta|D_A)$을 바란다. $\theta_A^*$ 언저리에서 라플라스 어림으로 $\log p(\theta|D_A)$을 어림하면 $\log p(\theta|D_A) \approx \text{const} - \frac{1}{2}(\theta - \theta_A^*)^\top F (\theta - \theta_A^*)$이며 $F$은 피셔 정보 행렬이다. 여기서 EWC 벌 $\frac{\lambda}{2}\sum_i F_i(\theta_i - \theta_{A,i}^*)^2$이 나온다.

---

**연습문제 2.**
피셔 정보 행렬이 중요한 가중치를 어떻게 짚어내는지 설명하라.

??? success "연습문제 2 풀이"
    피셔 행렬의 대각 성분 $F_i = \mathbb{E}[(\frac{\partial \log p(y|x,\theta)}{\partial \theta_i})^2]$은 가중치 $\theta_i$이 바뀔 때 로그 가능도가 얼마나 달라지는지를 잰다. $F_i$이 크면 그 가중치가 지금 과제의 예측에 중요하다는 뜻이다. EWC는 $F_i$이 큰 가중치가 바뀌면 더 세게 벌을 주어 중요한 앎을 지킨다.

---

**연습문제 3.**
파이토치로 EWC를 구현하라.

??? success "연습문제 3 풀이"
    ```python
    class EWC:
        def __init__(self, model, dataset, lambda_=1000):
            self.params = {n: p.clone() for n, p in model.named_parameters()}
            self.fisher = self._compute_fisher(model, dataset)
            self.lambda_ = lambda_
        def _compute_fisher(self, model, dataset):
            fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
            for x, y in dataset:
                model.zero_grad()
                F.cross_entropy(model(x), y).backward()
                for n, p in model.named_parameters():
                    fisher[n] += p.grad.data ** 2 / len(dataset)
            return fisher
        def penalty(self, model):
            loss = 0
            for n, p in model.named_parameters():
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
            return self.lambda_ / 2 * loss
    ```

---

**연습문제 4.**
EWC의 한계는 무엇인가?

??? success "연습문제 4 풀이"

    1. 대각 피셔 어림은 매개변수 사이의 상관을 무시한다. 2. 피셔를 한 점($\theta_A^*$)에서만 셈하므로 뒤확률 전체를 나타내지 못할 수 있다. 3. 규모를 키우기 나쁘다. 과제마다 피셔와 매개변수를 담아 두어야 한다. 4. 과제가 많아지면 '얼어붙은' 매개변수가 늘어 성능이 떨어진다. 5. 과제의 경계를 안다고 놓는다.

## 정리하며

탄성 가중치 다지기는 다음으로 원칙 있는 이어 배우기 방법을 준다.

1. 피셔 정보로 **중요한 매개변수를 짚어낸다**
2. 이차 벌로 **중요한 매개변수를 지킨다**
3. 중요하지 않은 매개변수에는 **자유를 준다**

이 방법은 데이터를 담아 둘 필요가 없어 사생활이 민감한 응용에 알맞지만, 과제가 많아지면 제약이 쌓일 수 있다.

**참고 문헌**

1. Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*, 114(13), 3521-3526.

2. Huszár, F. (2018). Note on the quadratic penalties in elastic weight consolidation. *PNAS*, 115(11), E2496-E2497.

3. Schwarz, J., et al. (2018). Progress & compress: A scalable framework for continual learning. *ICML*.

4. Chaudhry, A., et al. (2018). Riemannian walk for incremental learning: Understanding forgetting and intransigence. *ECCV*.
