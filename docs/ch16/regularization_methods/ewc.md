# 탄성 웨이트 통합(EWC)

EWC(elastic weight consolidation)는 이전 과제에서 추정한 피셔 정보로 웨이트 변화에 벌점을 주는 지속 학습 방법이다. Kirkpatrick 등(2017)이 제안했으며, 이전 예측에 민감한 파라메터의 큰 이동을 억제한다. 피셔 근사 방식과 과제별 통계 저장 여부에 따라 비용과 동작이 달라진다.

## 1. 이론적 기초

### 순차 베이즈 갱신

과제별 데이터가 파라메터를 조건으로 독립이면 이전 사후분포를 다음 과제의 사전분포로 사용할 수 있다.

$$\log p(\theta\mid D_A,D_B)=\log p(D_B\mid\theta)+\log p(\theta\mid D_A)+C$$

뉴럴 넷트웍의 이전 사후분포 전체를 저장하기 어려우므로 이전 해 부근에서 국소 가우시안 근사를 사용한다. 라플라스 근사의 정밀도는 원래 음의 로그 사후분포의 헤시안이다. EWC는 이를 피셔 정보의 대각 근사 등으로 대체하므로 정확한 사후분포와는 다르다.

### 피셔 정보

모델 피셔는 입력과 모델의 예측 라벨 분포에서 계산한다.

$$F=\mathbb E_{x,\,y\sim p_\theta(y\mid x)}[\nabla_\theta\log p_\theta(y\mid x)\nabla_\theta\log p_\theta(y\mid x)^\top]$$

대각 성분은 로그확률 미분의 제곱 평균이다. 해당 입력 분포에서 예측 분포가 웨이트 변화에 얼마나 민감한지 나타내는 국소적 척도다. 실제 정답 라벨로 계산하는 경험적 피셔는 모델 피셔와 일반적으로 다르다.

### 대각 근사

P개 파라메터의 전체 행렬 저장은 O(P²)이고 대각만 저장하면 O(P)다. 교차 성분을 생략하므로 여러 웨이트가 함께 변하는 효과를 충분히 나타내지 못할 수 있다. 0인 대각 성분이 있다고 그 방향의 임의 크기 변화가 항상 안전한 것은 아니다.

## 2. EWC의 로스 함수

과제별 기준점과 중요도를 저장하는 구현은 이전 과제 항을 합산한다.

$$L_t(\theta)=L_{\mathrm{task},t}(\theta)+\frac\lambda2\sum_{k<t}\sum_iF_{k,i}(\theta_i-\theta^*_{k,i})^2$$

각 항은 웨이트를 이전 기준점으로 끌어당기는 용수철처럼 작용한다. λ는 새 로스와 정규화의 상대 크기를 정하며 피셔의 정규화 방식·데이터 수와 함께 해석한다. 하나의 누적 중요도와 최근 기준점을 쓰는 온라인 방식은 별도의 근사다. 여러 과제의 벌점 합을 정확한 순차 베이즈 갱신과 동일시하지 않는다.

## 3. PyTorch 구현

### EWC 학습기 클래스

이 구현은 실제 정답의 로그확률을 표본별로 역전파하고 제곱 경사를 평균 내는 경험적 대각 피셔를 계산한다. 배치 평균 경사를 제곱하는 방식과 다르다. fisher_sample_size는 양수여야 하며 빈 로더에서는 0으로 나눈다. 모델 이동은 호출자가 수행해야 한다.

각 과제의 피셔와 기준 웨이트를 별도 사전에 저장하므로 과제 수에 따라 메모리가 늘어난다. 기준 웨이트는 clone하여 보관한다. train_on_task의 test_loader는 사용하지 않으며 테스트 지표를 이용한 모델 선택도 수행하지 않는다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Optional, List
import copy

class EWCLearner:
    """
    이어 배우기를 위한 탄성 웨이트 다지기.
    
    EWC는 다음으로 파국적 잊음을 막는다.
    1. 과제를 마칠 때마다 피셔 정보를 셈한다(파라메터 중요도)
    2. 중요한 파라메터를 지키려고 이차 벌을 더한다
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
            model: 뉴럴 넷트웍 모델
            device: 셈할 장치(CPU나 GPU)
            ewc_lambda: 벌주기 세기(λ)
                       값이 클수록 옛 파라메터를 더 세게 지킨다
                       흔한 범위: [100, 10000]
            learning_rate: 최적화기의 러닝레이트
            fisher_sample_size: 피셔를 셈하는 데 쓸 표본의 개수
                               (None이면 데이터셋 전체를 쓴다)
        """
        self.model = model
        self.device = device
        self.ewc_lambda = ewc_lambda
        self.learning_rate = learning_rate
        self.fisher_sample_size = fisher_sample_size
        
        self.criterion = nn.CrossEntropyLoss()
        
        # 피셔 정보와 가장 좋은 파라메터를 담을 곳
        # 열쇠: task_id, 값: Dict[param_name -> 텐서]
        self.fisher_dict: Dict[int, Dict[str, torch.Tensor]] = {}
        self.optpar_dict: Dict[int, Dict[str, torch.Tensor]] = {}
        
        self.current_task = 0
    
    def compute_fisher_information(self, 
                                   data_loader,
                                   task_id: int) -> Dict[str, torch.Tensor]:
        """
        대각 피셔 정보 행렬을 셈한다.
        
        파라메터 θ_i의 피셔 정보 F_i은 다음과 같다.
        F_i = E[(∂log p(y|x,θ)/∂θ_i)²]
        
        이를 실험으로 어림한다.
        F_i ≈ (1/N) Σ_n (∂L(x_n, θ)/∂θ_i)²
        
        인수:
            data_loader: 피셔를 셈할 DataLoader
            task_id: 지금 과제의 식별자
        
        반환값:
            파라메터 이름을 피셔 값으로 옮기는 사전
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
            
            # 표본마다 그래디언트의 제곱을 셈한다
            for i in range(batch_size):
                if (self.fisher_sample_size is not None and 
                    total_samples >= self.fisher_sample_size):
                    break
                
                self.model.zero_grad()
                
                # 참 부류의 로그 확률
                log_prob = log_probs[i, target[i]]
                
                # 거꾸로 퍼뜨린다
                log_prob.backward(retain_graph=(i < batch_size - 1))
                
                # 그래디언트 제곱 누적
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
                
                # 과제마다의 로스
                task_loss = self.criterion(output, target)
                
                # EWC 벌(앞선 과제가 있을 때만)
                if len(self.fisher_dict) > 0:
                    ewc_loss = self.compute_ewc_penalty()
                else:
                    ewc_loss = torch.tensor(0.0, device=self.device)
                
                # 전체 로스
                total_loss = task_loss + ewc_loss
                
                # 역전파
                total_loss.backward()
                optimizer.step()
                
                # 통계 기록
                epoch_loss += total_loss.item()
                epoch_ewc += ewc_loss.item()
                epoch_task += task_loss.item()
                num_batches += 1
            
            # 로스를 평균 낸다
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
        1. 피셔 정보(파라메터 중요도)
        2. 가장 좋은 파라메터(벌의 기준점)
        
        인수:
            data_loader: 피셔를 셈할 데이터 로더
            task_id: 과제 식별자
        """
        print(f"  Consolidating Task {task_id}...")
        
        # 피셔 정보를 셈한다
        fisher = self.compute_fisher_information(data_loader, task_id)
        self.fisher_dict[task_id] = fisher
        
        # 가장 좋은 파라메터를 담아 둔다
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
            
            # 다진다(피셔를 셈하고 파라메터를 담아 둔다)
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

### 사용 예

train_loaders와 test_loaders는 별도로 준비해야 한다. 예제의 출력 수는 2이므로 각 과제 라벨도 같은 규약에 맞아야 한다. continual_metrics 모듈의 검색 경로와 지표 정의를 확인한다.

```python
import numpy as np

# 설정
num_tasks = 5
epochs_per_task = 5
ewc_lambda = 5000.0  # 핵심 초파라메터

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

## 4. 하이퍼파라미터 분석

### 정규화 계수

λ는 안정성과 가소성의 균형을 조절한다. 피셔의 크기와 로스 평균 방식에 따라 유효한 범위가 달라지므로 10이나 5000 같은 수치를 보편적인 기준으로 해석하지 않는다. 민감도 분석은 최종 테스트가 아닌 검증 과제에서 수행하고 각 후보의 초기화·데이터 순서·학습 예산을 맞춘다.

| λ 크기 | 예상되는 경향 |
|---|---|
| 상대적으로 작음 | 중요 웨이트 이동에 대한 제약이 약함 |
| 상대적으로 큼 | 이전 기준점에 더 강하게 묶여 새 과제 적응이 제한될 수 있음 |

아래 분석 조각의 create_model은 정의되어 있지 않다. 결과로 선택한 λ를 같은 테스트셋에서 다시 최적화하지 않도록 데이터 역할을 구분한다.

```python
def analyze_lambda_sensitivity(train_loaders, test_loaders, 
                               lambda_values, device):
    """
    λ 초파라메터에 대한 민감도를 뜯어본다.
    
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

### 정규화 효과 시각화

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
    
    # 러닝 정확도
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

아래 하위 클래스는 running_fisher만 갱신하지만 상속한 train_on_task는 fisher_dict가 비어 있으면 벌점을 호출하지 않는다. 따라서 이 코드 그대로는 온라인 벌점이 학습에 적용되지 않는다. 병합 수식과 실제 호출 조건을 함께 확인해야 한다.

온라인 EWC는 과제별 통계를 하나의 누적 대각 중요도로 병합하고 기준 웨이트를 최근 해로 갱신한다. 아래 γ는 과거 중요도에 곱하는 감쇠 계수다.

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
        
        # 참조 파라메터를 고친다
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

EWC++는 온라인 피셔 추정으로 계산 부담을 줄이는 방법으로 제안되었다. 아래 EWCPlusPlus는 과제 후 피셔를 구한 뒤 파라메터 텐서별 최댓값으로 나누는 변형이며 원 EWC++를 그대로 구현하지 않는다. 과제별 λ도 따로 사용하지 않는다. 텐서별 재스케일은 레이어 사이의 상대 중요도를 바꾸므로 λ만 바꾸는 것과도 다르다.

```python
class EWCPlusPlus(EWCLearner):
    """
    고른 피셔와 과제마다의 람다를 쓰는 EWC++.
    """
    
    def consolidate_task(self, data_loader, task_id):
        """고른 피셔를 셈한다."""
        fisher = self.compute_fisher_information(data_loader, task_id)
        
        # 레이어마다 피셔를 고른다
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

이 구현은 표본별 피셔 경사를 구하므로 선택한 표본 수만큼 역전파 작업이 필요하다. 정확한 비용은 네트워크 연산량과 표본 수에 따라 달라지며 파라메터 수만으로 일반적인 순전파 시간을 정할 수 없다.

P개 파라메터와 T개 이전 과제의 통계를 모두 저장하면 중요도·기준점의 추가 메모리는 O(TP), 매 학습 단계의 벌점 계산도 O(TP)다. 하나의 통계와 기준점으로 병합하면 O(P)로 유지할 수 있다. 모델·옵티마이저 상태와 엑티베이션 메모리는 별도로 포함한다.

## 7. 장점과 한계

과제 정리가 끝난 뒤 이전 표본 대신 중요도와 기준 웨이트만으로 벌점을 계산할 수 있다. 다만 모델·통계 저장도 개인정보를 포함할 수 있으므로 데이터 미저장이 곧 개인정보 보호 보장은 아니다.

피셔의 국소·대각 근사, 과제 누적 제약과 λ 선택이 성능에 영향을 준다. 과제별 항을 누적하면 저장과 계산 비용이 증가하고, 강한 제약은 새 과제를 배우기 어렵게 할 수 있다. BatchNorm 상태나 바뀐 출력 레이어도 별도 관리해야 한다.

## 8. 기대되는 결과

EWC의 효과는 과제 구성과 출력 공간, 피셔 추정 표본·방법, λ와 학습 예산에 따라 달라진다. 고정된 Split MNIST 정확도나 망각 감소율을 보장할 수 없다.

같은 초기화·과제 순서에서 단순 순차 학습과 비교하고, 새 과제의 학습 직후 정확도 및 모든 과제의 최종 정확도를 함께 보고한다. 검증에서 선택한 설정으로 여러 번 반복해 변동도 확인한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
순차 베이즈 갱신에서 EWC의 이차 벌점이 나오는 과정을 설명하라.

</div>

??? success "풀이"
    새 데이터의 로그우도에 이전 로그 사후분포를 더한다. 이전 해 근처의 음의 로그 사후분포를 이차식으로 근사하고 그 정밀도를 대각 피셔로 대체하면 가중 제곱 차이 벌점을 얻는다. 데이터 수·사전분포·스케일을 생략한 근사이므로 피셔가 정확한 사후 헤시안과 항상 같다고 해석하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
표본별 피셔 추정과 현재 데이터의 일반 로스 경사는 어떻게 다른가?

</div>

??? success "풀이"
    피셔 추정은 각 표본의 로그확률 경사를 제곱한 뒤 평균 낸다. 일반 학습의 배치 평균 경사는 부호가 상쇄될 수 있어 그 제곱이 같은 추정량이 아니다. 정답을 사용하는 경험적 피셔와 예측 분포에서 라벨을 추출하는 모델 피셔도 구분한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 간단한 EWC 조각의 입력 단위와 기준 웨이트 저장 방식을 검토하라.

</div>

??? success "풀이"
    dataset이 개별 표본을 반환하면 model(x)와 교차 엔트로피에 맞는 배치 차원을 별도로 넣어야 한다. 배치를 반환하면 배치 평균 경사의 제곱이므로 표본별 피셔와 다르다. 기준점의 p.clone()은 원본 파라메터와 미분 경로가 연결되어 있어 고정 참조로 쓰려면 detach한 복사본을 보관해야 한다. 장치·평가 모드·미사용 파라메터의 grad=None 처리도 필요하다.

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

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
이 페이지의 OnlineEWC에서 벌점이 적용되지 않는 원인을 설명하라.

</div>

??? success "풀이"
    기본 학습 루프는 fisher_dict가 비어 있지 않을 때만 compute_ewc_penalty를 부른다. 하위 클래스는 그 사전 대신 running_fisher를 채우므로 조건이 계속 거짓이다. 벌점 수식이 옳아도 학습 경로에서 호출되지 않으면 망각 완화 효과를 기대할 수 없다.

## 정리하며

EWC는 이전 과제에서 추정한 국소 중요도로 새 웨이트 이동을 제한한다. 경험적·모델 피셔와 과제별·병합 통계를 구분하고, 추정과 저장 비용을 함께 보고한다. 하위 클래스의 벌점이 실제 학습 루프에서 호출되는지와 고정 기준점의 미분 분리도 확인한다.

**참고 문헌**

1. Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*, 114(13), 3521-3526.

2. Huszár, F. (2018). Note on the quadratic penalties in elastic weight consolidation. *PNAS*, 115(11), E2496-E2497.

3. Schwarz, J., et al. (2018). Progress & compress: A scalable framework for continual learning. *ICML*.

4. Chaudhry, A., et al. (2018). Riemannian walk for incremental learning: Understanding forgetting and intransigence. *ECCV*.
