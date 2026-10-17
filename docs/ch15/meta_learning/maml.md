# MAML: 모델에 독립적인 메타 학습

MAML은 서포트 데이터로 몇 번 갱신한 뒤의 쿼리 성능이 좋아지도록 초기 웨이트를 학습한다. Finn 등(2017)이 제안한 방법으로, 특정한 네트워크 구조를 요구하지 않지만 내부 로스를 경사로 최적화할 수 있어야 한다. 과제별 최적점에 단순히 가까운 초기화보다, 적은 갱신으로 적응하기 좋은 초기화를 찾는다는 점이 핵심이다.

## 1. MAML 알고리즘

### 문제 정식화

과제 분포에서 여러 과제를 추출하고, 각 과제를 서포트와 쿼리로 나눈다. 서포트 로스는 내부 적응에, 쿼리 로스는 초기 웨이트의 외부 갱신에 쓴다. 최종 메타 테스트의 쿼리 정답은 학습에 사용하지 않는다.

### 이중 수준 최적화

MAML은 최적화 루프 두 겹을 쓴다.

**내부 루프(과제 맞춤)**:
서포트 집합이 $\mathcal{D}_i^{\text{train}}$인 과제 $\mathcal{T}_i$이 주어지면 파라메터를 맞춘다.

$$\theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta; \mathcal{D}_i^{\text{train}})$$

**외부 루프(메타 갱신)**:
맞춘 파라메터가 쿼리 집합에서 낸 성능으로 초기화를 고친다.

$$\theta \leftarrow \theta - \beta \nabla_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}(\theta'_i; \mathcal{D}_i^{\text{test}})$$

### 메타 그래디언트

적응 후 쿼리 로스를 적응 전 웨이트에 대해 미분한다. 아래 두 식에서는 같은 과제 로스 기호를 사용하지만 내부 경사는 서포트에서, 외부 로스는 쿼리에서 계산한다.

$$\nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta'_i) = \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta))$$

연쇄 법칙을 쓰면 다음과 같다.

$$\nabla_\theta L_{Q_i}(\theta'_i)=\left(\frac{\partial\theta'_i}{\partial\theta}\right)^\top\nabla_{\theta'_i}L_{Q_i}(\theta'_i)$$

여기서 각 기호는 다음과 같다.

$$\frac{\partial \theta'_i}{\partial \theta} = I - \alpha \nabla^2_\theta \mathcal{L}_{\mathcal{T}_i}(\theta)$$

여기에는 서포트 로스의 헤시안이 포함된다. 자동 미분으로 필요한 헤시안·벡터 곱을 계산할 수 있으므로 전체 헤시안 행렬을 직접 만들 필요는 없다. 2차 미분을 사용한다는 점과 뉴턴법처럼 헤시안의 역행렬로 갱신한다는 것은 다르다.

---

## 2. 수학적 분석

### 그래디언트 계산

안쪽 스텝이 하나일 때 메타 그래디언트는 다음과 같다.

$$\nabla_\theta L_{Q_i}(\theta'_i)=(I-\alpha H_{S_i})^\top\nabla_{\theta'_i}L_{Q_i}(\theta'_i)$$

여기서 H_S는 적응 전 웨이트에서 계산한 서포트 로스의 헤시안이다. 열벡터 경사 규약에 맞춰 야코비안의 전치를 곱한다. 여러 내부 단계에서는 각 갱신의 야코비안을 연쇄적으로 통과한다.

안쪽 스텝이 여럿이면 계산이 더 복잡해지지만, 맞춤 자취를 거슬러 되돌려 퍼뜨린다는 같은 원리를 따른다.

### 1차 근사 FOMAML

FOMAML은 내부 경사의 웨이트 의존성을 생략하고 적응 후 쿼리 경사를 초기 웨이트에 전달한다. 아래의 근사는 내부 갱신의 야코비안을 단위행렬로 보는 것이다.

$$\nabla_\theta \mathcal{L}(\theta'_i) \approx \nabla_{\theta'_i} \mathcal{L}(\theta'_i)$$

이 어림은 $\frac{\partial \theta'_i}{\partial \theta} \approx I$을 놓는데, 다음일 때 그럴듯하다.

- $\alpha$이 작다
- 로스 지형이 그 언저리에서 평평하다(헤시안 행렬이 작다)

근사의 오차는 내부 러닝레이트와 곡률, 적응 경로에 따라 달라진다. 여러 단계를 누적하면 각 단계의 작은 차이가 커질 수 있으므로 FOMAML과 MAML의 성능 차이는 실제 검증 과제에서 비교한다.

### Reptile

Reptile(Nichol 외, 2018)은 또 다른 일차 변형이다.

$$\theta \leftarrow \theta + \epsilon \cdot \frac{1}{n} \sum_{i=1}^n (\theta'_i - \theta)$$

이는 과제들에 걸쳐 평균 내면서 맞춘 파라메터 쪽으로 움직이는 것으로 풀이할 수 있다.

---

## 3. PyTorch 구현

### MAML 기본 구현

inner_loop는 원래 파라메터의 clone에서 시작한다. clone은 미분 연결을 유지하며, create_graph 설정으로 2차 항의 포함 여부를 정한다. 모델은 외부 파라메터 사전을 실제 forward 연산에 사용해야 한다. ConvMAMLModel은 평가 모드에서도 배치 통계를 직접 계산하므로 쿼리 배치 구성에 결과가 의존할 수 있다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional
import copy

class MAMLModel(nn.Module):
    """
    함수형 피드포워드를 갖춘 MAML의 바탕 모델 클래스.
    
    파라메터를 드러내어 넘길 수 있게 하여 안쪽 회귀 맞춤을 꿰뚫는
    그래디언트 셈이 되게 한다.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        params: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        파라메터를 갈아 끼울 수 있는 피드포워드.
        
        인수:
            x: 입력 텐서
            params: self.parameters() 대신 쓸 파라메터 사전
                   열쇠는 state_dict의 열쇠와 맞아야 한다
        """
        if params is None:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        else:
            x = F.relu(F.linear(x, params['fc1.weight'], params['fc1.bias']))
            x = F.relu(F.linear(x, params['fc2.weight'], params['fc2.bias']))
            x = F.linear(x, params['fc3.weight'], params['fc3.bias'])
        return x
    
    def get_params(self) -> Dict[str, torch.Tensor]:
        """파라메터를 사전으로 되돌린다."""
        return OrderedDict(self.named_parameters())

class ConvMAMLModel(nn.Module):
    """
    그림 분류를 위한 컨볼루션 MAML 모델.
    """
    
    def __init__(
        self, 
        in_channels: int = 1, 
        hidden_dim: int = 64, 
        output_dim: int = 5
    ):
        super().__init__()
        
        # 컨볼루션 블록
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim, track_running_stats=False)
        
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim, track_running_stats=False)
        
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_dim, track_running_stats=False)
        
        self.conv4 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(hidden_dim, track_running_stats=False)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        params: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """함수형 피드포워드."""
        if params is None:
            params = dict(self.named_parameters())
        
        # 컨볼루션 블록 1
        x = F.conv2d(x, params['conv1.weight'], params['conv1.bias'], padding=1)
        x = F.batch_norm(x, None, None, params['bn1.weight'], params['bn1.bias'], training=True)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # 컨볼루션 블록 2
        x = F.conv2d(x, params['conv2.weight'], params['conv2.bias'], padding=1)
        x = F.batch_norm(x, None, None, params['bn2.weight'], params['bn2.bias'], training=True)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # 컨볼루션 블록 3
        x = F.conv2d(x, params['conv3.weight'], params['conv3.bias'], padding=1)
        x = F.batch_norm(x, None, None, params['bn3.weight'], params['bn3.bias'], training=True)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # 컨볼루션 블록 4
        x = F.conv2d(x, params['conv4.weight'], params['conv4.bias'], padding=1)
        x = F.batch_norm(x, None, None, params['bn4.weight'], params['bn4.bias'], training=True)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # 전역 평균 풀링과 가려내개
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = F.linear(x, params['fc.weight'], params['fc.bias'])
        
        return x

class MAML:
    """
    모델 가리지 않는 메타 학습 알고리즘.
    
    새 과제에 재빨리 맞추어 갈 수 있게 하는 초기화를 배운다.
    """
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        inner_steps: int = 5,
        first_order: bool = False
    ):
        """
        인수:
            model: 메타 학습할 모델
            inner_lr: 안쪽 회귀(과제 맞춤)의 러닝레이트
            meta_lr: 바깥 회귀(메타 갱신)의 러닝레이트
            inner_steps: 안쪽 회귀의 그래디언트 스텝 수
            first_order: True이면 FOMAML을 쓴다(이차 항을 무시한다)
        """
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=meta_lr
        )
    
    def inner_loop(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        params: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        받침 집합에 대해 안쪽 회귀 맞춤을 한다.
        
        인수:
            support_x: 받침 입력
            support_y: 받침 이름표
            params: 처음 파라메터(맞추어질 것이다)
        
        반환값:
            adapted_params: 맞춘 뒤의 파라메터
        """
        # 맞춤을 위해 파라메터를 복제한다
        adapted_params = OrderedDict(
            (name, param.clone()) for name, param in params.items()
        )
        
        for step in range(self.inner_steps):
            # 순전파
            logits = self.model(support_x, adapted_params)
            loss = F.cross_entropy(logits, support_y)
            
            # 경사를 계산한다
            grads = torch.autograd.grad(
                loss,
                adapted_params.values(),
                create_graph=not self.first_order,
                allow_unused=True
            )
            
            # 파라메터 갱신
            adapted_params = OrderedDict(
                (name, param - self.inner_lr * grad if grad is not None else param)
                for (name, param), grad in zip(adapted_params.items(), grads)
            )
        
        return adapted_params
    
    def meta_train_step(
        self,
        tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Tuple[float, float]:
        """
        과제 배치에 대해 메타 학습 스텝을 한 번 밟는다.
        
        인수:
            tasks: (support_x, support_y, query_x, query_y) 짝의 목록
        
        반환값:
            meta_loss: 과제에 걸친 평균 물음 로스
            meta_accuracy: 과제에 걸친 평균 물음 정확도
        """
        self.meta_optimizer.zero_grad()
        
        meta_loss = 0.0
        meta_accuracy = 0.0
        
        # 처음 파라메터를 얻는다
        init_params = OrderedDict(self.model.named_parameters())
        
        for support_x, support_y, query_x, query_y in tasks:
            # 안쪽 회귀: 과제에 맞춘다
            adapted_params = self.inner_loop(support_x, support_y, init_params)
            
            # 물음 집합에서 평가한다
            query_logits = self.model(query_x, adapted_params)
            task_loss = F.cross_entropy(query_logits, query_y)
            
            meta_loss += task_loss
            
            # 정확도를 계산한다
            with torch.no_grad():
                predictions = query_logits.argmax(dim=1)
                accuracy = (predictions == query_y).float().mean()
                meta_accuracy += accuracy.item()
        
        # 과제들에 걸쳐 평균 낸다
        meta_loss = meta_loss / len(tasks)
        meta_accuracy = meta_accuracy / len(tasks)
        
        # 메타 갱신
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item(), meta_accuracy
    
    def adapt_and_evaluate(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
        inner_steps: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        새 과제에 맞추고 평가한다.
        
        인수:
            support_x, support_y: 맞춤에 쓸 받침 집합
            query_x, query_y: 평가에 쓸 물음 집합
            inner_steps: 맞춤 스텝 수를 갈아 끼운다
        
        반환값:
            물음 집합에서의 로스와 정확도
        """
        if inner_steps is not None:
            original_steps = self.inner_steps
            self.inner_steps = inner_steps
        
        self.model.eval()
        
        with torch.no_grad():
            # 처음 파라메터를 얻는다
            init_params = OrderedDict(
                (name, param.clone()) 
                for name, param in self.model.named_parameters()
            )
        
        # 맞춘다(안쪽 회귀에만 그래디언트를 쓴다)
        adapted_params = OrderedDict(
            (name, param.requires_grad_(True))
            for name, param in init_params.items()
        )
        
        for step in range(self.inner_steps):
            logits = self.model(support_x, adapted_params)
            loss = F.cross_entropy(logits, support_y)
            
            grads = torch.autograd.grad(loss, adapted_params.values())
            
            adapted_params = OrderedDict(
                (name, param - self.inner_lr * grad)
                for (name, param), grad in zip(adapted_params.items(), grads)
            )
        
        # 평가한다
        with torch.no_grad():
            query_logits = self.model(query_x, adapted_params)
            query_loss = F.cross_entropy(query_logits, query_y)
            
            predictions = query_logits.argmax(dim=1)
            accuracy = (predictions == query_y).float().mean()
        
        if inner_steps is not None:
            self.inner_steps = original_steps
        
        return query_loss.item(), accuracy.item()
```

### FOMAML과 Reptile 변형

FOMAML은 first_order=True를 전달한다. 아래 Reptile 코드는 쿼리 정확도 계산 중 원본 모델을 적응 웨이트로 덮어쓰고 복원하지 않는다. 다음 과제가 같은 초기화에서 출발하지 않으며 마지막 외부 갱신도 마지막 적응 상태에 더해진다. 따라서 제시한 Reptile 수식을 그대로 구현한 것으로 읽으면 안 된다. 반환 로스 0.0도 측정한 쿼리 로스가 아니다.

```python
class FOMAML(MAML):
    """
    일차 MAML - 이차 그래디언트를 무시한다.
    """
    
    def __init__(self, model: nn.Module, **kwargs):
        super().__init__(model, first_order=True, **kwargs)

class Reptile:
    """
    Reptile 메타 학습 알고리즘.
    
    MAML보다 단순하다. 그저 맞춘 파라메터 쪽으로 움직인다.
    """
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        meta_lr: float = 0.1,
        inner_steps: int = 5
    ):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps
    
    def inner_loop(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        받침 집합에서 모델 파라메터를 맞춘다.
        계산 그래프 없이 맞춘 파라메터를 되돌린다.
        """
        # 과제별 학습을 위해 모델을 복제한다
        task_model = copy.deepcopy(self.model)
        optimizer = torch.optim.SGD(task_model.parameters(), lr=self.inner_lr)
        
        for _ in range(self.inner_steps):
            logits = task_model(support_x)
            loss = F.cross_entropy(logits, support_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return OrderedDict(task_model.named_parameters())
    
    def meta_train_step(
        self,
        tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Tuple[float, float]:
        """
        Reptile 메타 갱신: 맞춘 파라메터 쪽으로 움직인다.
        """
        # 처음 파라메터를 담아 둔다
        init_params = OrderedDict(
            (name, param.clone())
            for name, param in self.model.named_parameters()
        )
        
        # 모든 과제에서 맞춘 파라메터를 모은다
        adapted_params_list = []
        meta_accuracy = 0.0
        
        for support_x, support_y, query_x, query_y in tasks:
            # 과제에 맞춘다
            adapted_params = self.inner_loop(support_x, support_y)
            adapted_params_list.append(adapted_params)
            
            # 평가한다(기록용)
            with torch.no_grad():
                # 맞춘 파라메터로 임시 모델을 만든다
                for name, param in self.model.named_parameters():
                    param.data.copy_(adapted_params[name])
                
                logits = self.model(query_x)
                predictions = logits.argmax(dim=1)
                accuracy = (predictions == query_y).float().mean()
                meta_accuracy += accuracy.item()
        
        meta_accuracy /= len(tasks)
        
        # Reptile 갱신: 맞춘 파라메터의 평균 쪽으로 움직인다
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                # 맞춘 파라메터의 평균
                avg_adapted = torch.stack([
                    adapted[name] for adapted in adapted_params_list
                ]).mean(dim=0)
                
                # 갱신: θ ← θ + ε(θ' - θ)
                param.data.add_(self.meta_lr * (avg_adapted - init_params[name]))
        
        return 0.0, meta_accuracy  # Reptile은 메타 로스를 셈하지 않는다
```

---

## 4. 학습 절차

### 학습 루프의 구조

sample_task는 이 페이지에 정의되어 있지 않다. 데이터셋에서 클래스·표본을 추출하고 로컬 라벨을 생성하는 함수를 따로 제공해야 한다. eval_interval은 이름과 달리 검증을 수행하지 않고 학습 배치 지표만 출력한다. 장치는 옵티마이저 생성 전에 모델에 설정하는 편이 명확하며 MetaSGD의 별도 러닝레이트 텐서도 같은 장치로 이동해야 한다.

```python
def train_maml(
    maml: MAML,
    train_dataset,
    n_way: int = 5,
    k_shot: int = 5,
    n_query: int = 15,
    meta_batch_size: int = 4,
    n_iterations: int = 60000,
    eval_interval: int = 100,
    device: str = 'cuda'
):
    """
    온전한 MAML 학습 회귀.
    """
    maml.model.to(device)
    
    for iteration in range(n_iterations):
        # 과제 배치를 뽑는다
        tasks = []
        for _ in range(meta_batch_size):
            support_x, support_y, query_x, query_y = sample_task(
                train_dataset, n_way, k_shot, n_query
            )
            tasks.append((
                support_x.to(device),
                support_y.to(device),
                query_x.to(device),
                query_y.to(device)
            ))
        
        # 메타 학습 스텝
        loss, acc = maml.meta_train_step(tasks)
        
        if (iteration + 1) % eval_interval == 0:
            print(f"Iteration {iteration + 1}: Loss = {loss:.4f}, Acc = {acc:.4f}")
    
    return maml
```

### 하이퍼파라미터 설정

| 설정 | 역할 | 선택 기준 |
|---|---|---|
| 내부 러닝레이트 | 서포트 적응의 갱신 크기 | 곡률과 적응 단계 수에 맞춰 검증 |
| 메타 러닝레이트 | 초기 웨이트의 외부 갱신 크기 | 독립적인 검증 과제에서 선택 |
| 내부 단계 수 | 과제별 적응 예산 | 학습·테스트 조건을 함께 기록 |
| 메타 배치 크기 | 외부 갱신당 과제 수 | 메모리와 경사 변동을 고려 |
| 외부 반복 수 | 메타 학습 예산 | 검증 성능으로 종료 시점 선택 |

---

## 5. 변형과 확장

### Meta-SGD

Meta-SGD는 초기 웨이트와 함께 파라메터별 갱신 계수를 학습한다. 계수에 양수 제약이 없으므로 갱신 크기뿐 아니라 방향도 바꿀 수 있다. 이 예제는 계수를 여러 내부 단계에서 공유한다. 다만 상속한 adapt_and_evaluate는 고정 inner_lr를 사용하므로 학습한 계수로 평가하려면 별도의 적응 경로가 필요하다.

```python
class MetaSGD(MAML):
    """
    Meta-SGD: 파라메터마다의 안쪽 러닝레이트를 배운다.
    """
    
    def __init__(self, model: nn.Module, **kwargs):
        super().__init__(model, **kwargs)
        
        # 파라메터마다 배울 수 있는 러닝레이트
        self.inner_lrs = nn.ParameterDict({
            name.replace('.', '_'): nn.Parameter(
                torch.ones_like(param) * self.inner_lr
            )
            for name, param in model.named_parameters()
        })
        
        # 러닝레이트를 최적화기에 더한다
        self.meta_optimizer = torch.optim.Adam(
            list(model.parameters()) + list(self.inner_lrs.values()),
            lr=self.meta_lr
        )
    
    def inner_loop(self, support_x, support_y, params):
        adapted_params = OrderedDict(
            (name, param.clone()) for name, param in params.items()
        )
        
        for step in range(self.inner_steps):
            logits = self.model(support_x, adapted_params)
            loss = F.cross_entropy(logits, support_y)
            
            grads = torch.autograd.grad(
                loss, adapted_params.values(),
                create_graph=not self.first_order
            )
            
            # 배운 러닝레이트를 쓴다
            adapted_params = OrderedDict(
                (name, param - self.inner_lrs[name.replace('.', '_')] * grad)
                for (name, param), grad in zip(adapted_params.items(), grads)
            )
        
        return adapted_params
```

### ANIL(내부 루프 거의 없음)

ANIL은 내부 루프에서 헤드만 갱신하고 외부 루프에서는 피처 추출기도 학습한다. head_names에는 fc3.weight처럼 정확한 파라메터 이름을 전달한다. 아래 코드는 head_params 사전을 첫 단계에만 만들고 갱신하지 않으므로 여러 내부 단계에서 현재 적응 웨이트가 아닌 이전 텐서를 미분하게 된다. 상속한 평가 함수도 모든 파라메터를 적응시키므로 이 예제만으로 일관된 ANIL 학습·평가가 완성되지는 않는다.

```python
class ANIL(MAML):
    """
    안쪽 회귀 거의 없음 - 가려내기 머리만 맞춘다.
    
    안쪽 회귀 동안 피처 뽑개는 얼려 두고
    마지막 가려내개만 맞춘다.
    """
    
    def __init__(self, model: nn.Module, head_names: List[str], **kwargs):
        """
        인수:
            head_names: 맞출 파라메터의 이름(이를테면 ['fc.weight', 'fc.bias'])
        """
        super().__init__(model, **kwargs)
        self.head_names = set(head_names)
    
    def inner_loop(self, support_x, support_y, params):
        # 머리 파라메터만 맞춘다
        adapted_params = OrderedDict(
            (name, param.clone() if name in self.head_names else param)
            for name, param in params.items()
        )
        
        # 그래디언트 셈에 쓸 머리 파라메터만 얻는다
        head_params = {
            name: param for name, param in adapted_params.items()
            if name in self.head_names
        }
        
        for step in range(self.inner_steps):
            logits = self.model(support_x, adapted_params)
            loss = F.cross_entropy(logits, support_y)
            
            grads = torch.autograd.grad(
                loss, head_params.values(),
                create_graph=not self.first_order
            )
            
            # 머리만 고친다
            for (name, grad) in zip(head_params.keys(), grads):
                adapted_params[name] = adapted_params[name] - self.inner_lr * grad
        
        return adapted_params
```

### 과제 적응 MAML

아래 TaskAdaptiveMAML은 과제 임베딩으로 갱신 계수를 조절하려는 구조의 시작 부분이다. 기본 MAMLModel과 ConvMAMLModel에는 encoder 속성이 없으며, inner_loop는 lr_scales만 계산하고 적응 파라메터를 반환하지 않는다. 완성된 실행 예제로 사용할 수 없다.

```python
class TaskAdaptiveMAML(MAML):
    """
    과제 적응 MAML - 러닝레이트를 과제에 맞추어 조건 짓는다.
    """
    
    def __init__(self, model: nn.Module, embed_dim: int = 64, **kwargs):
        super().__init__(model, **kwargs)
        
        # 과제 인코더
        self.task_encoder = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, len(list(model.parameters())))
        )
        
        # 최적화기에 더한다
        self.meta_optimizer = torch.optim.Adam(
            list(model.parameters()) + list(self.task_encoder.parameters()),
            lr=self.meta_lr
        )
    
    def compute_task_embedding(self, support_x, support_y):
        """받침 집합에서 과제 표현을 셈한다."""
        with torch.no_grad():
            features = self.model.encoder(support_x)
            task_embedding = features.mean(dim=0)
        return task_embedding
    
    def inner_loop(self, support_x, support_y, params):
        # 과제별 러닝레이트를 셈한다
        task_emb = self.compute_task_embedding(support_x, support_y)
        lr_scales = torch.sigmoid(self.task_encoder(task_emb))
        
        # ... 나머지는 MAML과 비슷하되 러닝레이트에 눈금을 씌운다
```

---

## 6. 이론적 통찰

### 적응 후 로스를 최적화하기

MAML은 보통의 여러 과제 학습과 달리 각 과제에서 적응한 뒤의 로스를 최적화한다.

$$\min_\theta\mathbb E_\tau L_{Q_\tau}(U_{S_\tau}(\theta))$$

여기서 U는 정해진 수의 내부 갱신이다. 내부 루프가 과제 로스의 전역 최솟값에 수렴해야 하는 것은 아니다. 제한된 데이터와 갱신 예산에서 좋은 쿼리 성능을 내는 것이 목표다.

### 표현력과 실제 성능

충분한 모델 용량과 특정한 조건에서 메타 학습기의 표현력을 분석할 수 있지만, 실제 학습이 임의의 학습 알고리즘을 찾아낸다는 뜻은 아니다. 새로운 과제에 대한 성능은 과제 분포와 모델, 내부 갱신 방식에 따라 평가한다.

## 7. 실용적인 고려

### MAML에서의 배치 정규화

BatchNorm은 웨이트 외에 통계 상태도 다루므로 과제 사이에 어떤 통계를 공유하는지 정해야 한다. 서포트 통계만 사용할지 쿼리 배치 통계도 사용할지에 따라 평가 조건이 달라진다. 아래 속성 변경만으로 이미 만들어진 running_mean과 running_var 버퍼를 제거하지는 않으므로 평가 동작까지 확인해야 한다.

```python
# 모든 BatchNorm 레이어에 track_running_stats=False를 준다
for module in model.modules():
    if isinstance(module, nn.BatchNorm2d):
        module.track_running_stats = False
```

### 그래디언트 클리핑

메타 로스의 backward 후, 외부 optimizer.step 전에 경사 노름을 제한할 수 있다. 아래 호출은 해당 위치에 삽입하는 코드 조각이다. 클리핑은 큰 갱신을 줄이지만 잘못된 미분 경로나 모든 학습 불안정을 해결하지는 않는다.

```python
# meta_loss.backward() 뒤에
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
```

### 메모리 효율

체크포인팅은 중간 엑티베이션 저장을 줄이는 대신 역전파에서 재계산한다. 아래 조각은 params_flat, unflatten과 adaptation_step의 본문이 없어 완성된 구현이 아니다. 내부에서 autograd.grad를 사용하는 계산에 맞는 체크포인트 방식과 난수·상태 처리도 정해야 한다.

```python
from torch.utils.checkpoint import checkpoint

def inner_loop_checkpointed(self, support_x, support_y, params):
    def adaptation_step(params_flat, support_x, support_y):
        # 편 것을 되돌리고 한 스텝 밟는다
        ...
    
    for step in range(self.inner_steps):
        params_flat = checkpoint(adaptation_step, params_flat, support_x, support_y)
    
    return unflatten(params_flat)
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
MAML의 내부 적응과 외부 갱신에서 어떤 데이터를 사용하는가?

</div>

??? success "풀이"
    내부 갱신은 각 과제의 서포트 로스로 적응 웨이트를 만든다. 외부 갱신은 적응 웨이트에서 계산한 쿼리 로스를 원래 초기 웨이트에 대해 미분한다. 메타 학습 과제의 쿼리 정답은 외부 학습에 사용하지만 최종 메타 테스트 정답은 평가에만 사용한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
MAML의 2차 미분과 FOMAML 근사를 설명하라.

</div>

??? success "풀이"
    한 단계 내부 갱신의 야코비안은 I−αH_S다. 정확한 메타 경사는 이 야코비안의 전치를 쿼리 경사에 곱하며, 자동 미분으로 헤시안·벡터 곱을 계산할 수 있다. FOMAML은 야코비안을 I로 근사한다. α가 작아도 곡률이 크거나 단계가 많으면 오차가 커질 수 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 내부 루프 조각에서 갱신된 params가 실제 모델 예측에 사용되는지 검토하라.

</div>

??? success "풀이"
    아래 코드는 params를 새 텐서 목록으로 바꾸지만 model(support_x)는 계속 원래 파라메터를 사용한다. 첫 단계 뒤 새 텐서는 다음 로스 그래프에 사용되지 않아 미분에 실패할 수 있다. 함수형 forward에 갱신된 이름별 파라메터를 전달해야 한다. 한 단계만 수행해도 반환값으로 쿼리 예측을 계산하는 경로가 따로 필요하다.

    ```python
    def inner_loop(model, support_x, support_y, inner_lr, inner_steps):
        params = list(model.parameters())
        for _ in range(inner_steps):
            loss = F.cross_entropy(model(support_x), support_y)
            grads = torch.autograd.grad(loss, params, create_graph=True)
            params = [p - inner_lr * g for p, g in zip(params, grads)]
        return params  # 맞춘 파라메터
    ```

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
MAML의 계산 비용과 평가에서 주의할 점을 설명하라.

</div>

??? success "풀이"
    내부 갱신 경로를 보존하면 단계 수에 따라 메모리와 계산량이 늘어난다. FOMAML은 2차 경사를 생략하지만 여전히 내부 적응이 필요하다. 서포트·쿼리 분리, 메타 테스트 과제의 독립성, BatchNorm 통계와 평가 적응 방식을 맞춰야 한다.

## 정리하며

MAML은 적응 후 쿼리 로스를 통해 초기 웨이트를 학습한다. 내부 갱신의 함수형 계산과 미분 경로가 핵심이며, FOMAML은 그 경로의 2차 항을 근사한다. 이 페이지의 기본 구현과 Reptile·Meta-SGD·ANIL·과제 적응 변형은 완성도가 다르므로 설명한 학습·평가 경로의 한계를 구분한다.

**참고 문헌**

1. Finn, C., et al. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." ICML 2017.
2. Nichol, A., et al. "On First-Order Meta-Learning Algorithms." arXiv 2018.
3. Li, Z., et al. "Meta-SGD: Learning to Learn Quickly for Few-Shot Learning." arXiv 2017.
4. Raghu, A., et al. "Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML." ICLR 2020.
5. Finn, C., et al. "Online Meta-Learning." ICML 2019.
