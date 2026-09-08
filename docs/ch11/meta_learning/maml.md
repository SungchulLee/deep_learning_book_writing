# 모델 가리지 않는 메타 학습(MAML)

Finn 외(2017)가 들여온 모델 가리지 않는 메타 학습(MAML)은 거리 기반 방법과 견주어 소수 예시 학습에 근본부터 다르게 다가간다. 붙박이 묻힘 공간을 배우는 대신, MAML은 기울기 걸음 몇 번만으로 새 과제에 재빨리 맞추어 갈 수 있게 하는 모델 매개변수의 **초기화**를 배운다.

핵심 통찰은 우아하다. 서로 다른 여러 과제의 최적 매개변수에 가까운 처음 매개변수를 찾을 수 있다면, 그 이웃에 있는 어떤 새 과제든 기울기 걸음 몇 번으로 맞추어 갈 수 있다.

---

## 1. MAML 알고리즘

### 문제 정식화

MAML은 과제 위의 분포 $p(\mathcal{T})$을 쓸 수 있다고 놓는다. 과제 $\mathcal{T}_i$마다 손실 함수 $\mathcal{L}_{\mathcal{T}_i}$과 입출력 위 분포에서 뽑은 표본으로 이루어진다. 목표는 새 과제에 재빨리 맞출 수 있는 매개변수 $\theta$을 찾는 것이다.

### 두 층 최적화

MAML은 최적화 되돌이 두 겹을 쓴다.

**안쪽 되돌이(과제 맞춤)**:
받침 집합이 $\mathcal{D}_i^{\text{train}}$인 과제 $\mathcal{T}_i$이 주어지면 매개변수를 맞춘다.

$$\theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta; \mathcal{D}_i^{\text{train}})$$

**바깥 되돌이(메타 갱신)**:
맞춘 매개변수가 물음 집합에서 낸 성능으로 초기화를 고친다.

$$\theta \leftarrow \theta - \beta \nabla_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}(\theta'_i; \mathcal{D}_i^{\text{test}})$$

### 메타 기울기

핵심 수학 통찰은 맞춘 **뒤**의 손실을 맞추기 **앞**의 매개변수에 대해 최적화한다는 점이다. 그러려면 맞춤 과정을 꿰뚫는 기울기를 셈해야 한다.

$$\nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta'_i) = \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta))$$

연쇄 법칙을 쓰면 다음과 같다.

$$\nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta'_i) = \nabla_{\theta'_i} \mathcal{L}_{\mathcal{T}_i}(\theta'_i) \cdot \frac{\partial \theta'_i}{\partial \theta}$$

여기서 각 기호는 다음과 같다.

$$\frac{\partial \theta'_i}{\partial \theta} = I - \alpha \nabla^2_\theta \mathcal{L}_{\mathcal{T}_i}(\theta)$$

여기에 과제 손실의 **헤세 행렬**이 들어가므로 MAML은 이차 최적화 방법이 된다.

---

## 2. 수학적 분석

### 기울기 셈하기

안쪽 걸음이 하나일 때 메타 기울기는 다음과 같다.

$$\nabla_\theta \mathcal{L}(\theta'_i) = \nabla_{\theta'_i} \mathcal{L}(\theta'_i) \cdot (I - \alpha H_{\mathcal{T}_i})$$

여기서 $H_{\mathcal{T}_i} = \nabla^2_\theta \mathcal{L}_{\mathcal{T}_i}(\theta)$은 헤세 행렬이다.

안쪽 걸음이 여럿이면 셈이 더 복잡해지지만, 맞춤 자취를 거슬러 되돌려 퍼뜨린다는 같은 원리를 따른다.

### 일차 어림(FOMAML)

헤세 행렬을 온전히 셈하는 것은 값비싸다. FOMAML은 이차 항을 무시한다.

$$\nabla_\theta \mathcal{L}(\theta'_i) \approx \nabla_{\theta'_i} \mathcal{L}(\theta'_i)$$

이 어림은 $\frac{\partial \theta'_i}{\partial \theta} \approx I$을 놓는데, 다음일 때 그럴듯하다.

- $\alpha$이 작다
- 손실 지형이 그 언저리에서 평평하다(헤세 행렬이 작다)

실험으로 보면 FOMAML은 셈을 크게 줄이고도 온전한 MAML에 거의 맞먹는 성능을 낸다.

### Reptile

Reptile(Nichol 외, 2018)은 또 다른 일차 변형이다.

$$\theta \leftarrow \theta + \epsilon \cdot \frac{1}{n} \sum_{i=1}^n (\theta'_i - \theta)$$

이는 과제들에 걸쳐 평균 내면서 맞춘 매개변수 쪽으로 움직이는 것으로 풀이할 수 있다.

---

## 3. PyTorch 구현

### 온전한 MAML 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional
import copy

class MAMLModel(nn.Module):
    """
    함수형 앞먹임을 갖춘 MAML의 바탕 모델 클래스.
    
    매개변수를 드러내어 넘길 수 있게 하여 안쪽 되돌이 맞춤을 꿰뚫는
    기울기 셈이 되게 한다.
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
        매개변수를 갈아 끼울 수 있는 앞먹임.
        
        인수:
            x: 입력 텐서
            params: self.parameters() 대신 쓸 매개변수 사전
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
        """매개변수를 사전으로 되돌린다."""
        return OrderedDict(self.named_parameters())

class ConvMAMLModel(nn.Module):
    """
    그림 분류를 위한 합성곱 MAML 모델.
    """
    
    def __init__(
        self, 
        in_channels: int = 1, 
        hidden_dim: int = 64, 
        output_dim: int = 5
    ):
        super().__init__()
        
        # 합성곱 블록
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
        """함수형 앞먹임."""
        if params is None:
            params = dict(self.named_parameters())
        
        # 합성곱 블록 1
        x = F.conv2d(x, params['conv1.weight'], params['conv1.bias'], padding=1)
        x = F.batch_norm(x, None, None, params['bn1.weight'], params['bn1.bias'], training=True)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # 합성곱 블록 2
        x = F.conv2d(x, params['conv2.weight'], params['conv2.bias'], padding=1)
        x = F.batch_norm(x, None, None, params['bn2.weight'], params['bn2.bias'], training=True)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # 합성곱 블록 3
        x = F.conv2d(x, params['conv3.weight'], params['conv3.bias'], padding=1)
        x = F.batch_norm(x, None, None, params['bn3.weight'], params['bn3.bias'], training=True)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # 합성곱 블록 4
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
            inner_lr: 안쪽 되돌이(과제 맞춤)의 학습률
            meta_lr: 바깥 되돌이(메타 갱신)의 학습률
            inner_steps: 안쪽 되돌이의 기울기 걸음 수
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
        받침 집합에 대해 안쪽 되돌이 맞춤을 한다.
        
        인수:
            support_x: 받침 입력
            support_y: 받침 이름표
            params: 처음 매개변수(맞추어질 것이다)
        
        반환값:
            adapted_params: 맞춘 뒤의 매개변수
        """
        # 맞춤을 위해 매개변수를 복제한다
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
            
            # 매개변수 갱신
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
        과제 배치에 대해 메타 학습 걸음을 한 번 밟는다.
        
        인수:
            tasks: (support_x, support_y, query_x, query_y) 짝의 목록
        
        반환값:
            meta_loss: 과제에 걸친 평균 물음 손실
            meta_accuracy: 과제에 걸친 평균 물음 정확도
        """
        self.meta_optimizer.zero_grad()
        
        meta_loss = 0.0
        meta_accuracy = 0.0
        
        # 처음 매개변수를 얻는다
        init_params = OrderedDict(self.model.named_parameters())
        
        for support_x, support_y, query_x, query_y in tasks:
            # 안쪽 되돌이: 과제에 맞춘다
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
            inner_steps: 맞춤 걸음 수를 갈아 끼운다
        
        반환값:
            물음 집합에서의 손실과 정확도
        """
        if inner_steps is not None:
            original_steps = self.inner_steps
            self.inner_steps = inner_steps
        
        self.model.eval()
        
        with torch.no_grad():
            # 처음 매개변수를 얻는다
            init_params = OrderedDict(
                (name, param.clone()) 
                for name, param in self.model.named_parameters()
            )
        
        # 맞춘다(안쪽 되돌이에만 기울기를 쓴다)
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

```python
class FOMAML(MAML):
    """
    일차 MAML - 이차 기울기를 무시한다.
    """
    
    def __init__(self, model: nn.Module, **kwargs):
        super().__init__(model, first_order=True, **kwargs)

class Reptile:
    """
    Reptile 메타 학습 알고리즘.
    
    MAML보다 단순하다. 그저 맞춘 매개변수 쪽으로 움직인다.
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
        받침 집합에서 모델 매개변수를 맞춘다.
        계산 그래프 없이 맞춘 매개변수를 되돌린다.
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
        Reptile 메타 갱신: 맞춘 매개변수 쪽으로 움직인다.
        """
        # 처음 매개변수를 담아 둔다
        init_params = OrderedDict(
            (name, param.clone())
            for name, param in self.model.named_parameters()
        )
        
        # 모든 과제에서 맞춘 매개변수를 모은다
        adapted_params_list = []
        meta_accuracy = 0.0
        
        for support_x, support_y, query_x, query_y in tasks:
            # 과제에 맞춘다
            adapted_params = self.inner_loop(support_x, support_y)
            adapted_params_list.append(adapted_params)
            
            # 평가한다(기록용)
            with torch.no_grad():
                # 맞춘 매개변수로 임시 모델을 만든다
                for name, param in self.model.named_parameters():
                    param.data.copy_(adapted_params[name])
                
                logits = self.model(query_x)
                predictions = logits.argmax(dim=1)
                accuracy = (predictions == query_y).float().mean()
                meta_accuracy += accuracy.item()
        
        meta_accuracy /= len(tasks)
        
        # Reptile 갱신: 맞춘 매개변수의 평균 쪽으로 움직인다
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                # 맞춘 매개변수의 평균
                avg_adapted = torch.stack([
                    adapted[name] for adapted in adapted_params_list
                ]).mean(dim=0)
                
                # 갱신: θ ← θ + ε(θ' - θ)
                param.data.add_(self.meta_lr * (avg_adapted - init_params[name]))
        
        return 0.0, meta_accuracy  # Reptile은 메타 손실을 셈하지 않는다
```

---

## 4. 학습 절차

### 완전한 학습 루프

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
    온전한 MAML 학습 되돌이.
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
        
        # 메타 학습 걸음
        loss, acc = maml.meta_train_step(tasks)
        
        if (iteration + 1) % eval_interval == 0:
            print(f"Iteration {iteration + 1}: Loss = {loss:.4f}, Acc = {acc:.4f}")
    
    return maml
```

### 초매개변수 설정

| 매개변수 | 흔한 값 | 비고 |
|-----------|---------------|-------|
| 안쪽 학습률($\alpha$) | 0.01 ~ 0.1 | 단순한 과제일수록 크게 |
| 메타 학습률($\beta$) | 0.001 | Adam 최적화기 |
| 안쪽 걸음 | 1~10 | 학습에는 1~5, 시험에는 더 많이 |
| 메타 배치 크기 | 2~8 | 갱신마다의 과제 개수 |
| 학습 되풀이 | 3만~6만 | 모일 때까지 |

---

## 5. 변형과 확장

### Meta-SGD

매개변수마다의 학습률을 배운다.

```python
class MetaSGD(MAML):
    """
    Meta-SGD: 매개변수마다의 안쪽 학습률을 배운다.
    """
    
    def __init__(self, model: nn.Module, **kwargs):
        super().__init__(model, **kwargs)
        
        # 매개변수마다 배울 수 있는 학습률
        self.inner_lrs = nn.ParameterDict({
            name.replace('.', '_'): nn.Parameter(
                torch.ones_like(param) * self.inner_lr
            )
            for name, param in model.named_parameters()
        })
        
        # 학습률을 최적화기에 더한다
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
            
            # 배운 학습률을 쓴다
            adapted_params = OrderedDict(
                (name, param - self.inner_lrs[name.replace('.', '_')] * grad)
                for (name, param), grad in zip(adapted_params.items(), grads)
            )
        
        return adapted_params
```

### ANIL(안쪽 되돌이 거의 없음)

마지막 층만 맞춘다.

```python
class ANIL(MAML):
    """
    안쪽 되돌이 거의 없음 - 가려내기 머리만 맞춘다.
    
    안쪽 되돌이 동안 특징 뽑개는 얼려 두고
    마지막 가려내개만 맞춘다.
    """
    
    def __init__(self, model: nn.Module, head_names: List[str], **kwargs):
        """
        인수:
            head_names: 맞출 매개변수의 이름(이를테면 ['fc.weight', 'fc.bias'])
        """
        super().__init__(model, **kwargs)
        self.head_names = set(head_names)
    
    def inner_loop(self, support_x, support_y, params):
        # 머리 매개변수만 맞춘다
        adapted_params = OrderedDict(
            (name, param.clone() if name in self.head_names else param)
            for name, param in params.items()
        )
        
        # 기울기 셈에 쓸 머리 매개변수만 얻는다
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

과제 묻힘에 맞추어 적응을 조건 짓는다.

```python
class TaskAdaptiveMAML(MAML):
    """
    과제 적응 MAML - 학습률을 과제에 맞추어 조건 짓는다.
    """
    
    def __init__(self, model: nn.Module, embed_dim: int = 64, **kwargs):
        super().__init__(model, **kwargs)
        
        # 과제 부호기
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
        # 과제별 학습률을 셈한다
        task_emb = self.compute_task_embedding(support_x, support_y)
        lr_scales = torch.sigmoid(self.task_encoder(task_emb))
        
        # ... 나머지는 MAML과 비슷하되 학습률에 눈금을 씌운다
```

---

## 6. 이론적 통찰

### 숨은 기울기 내려가기

MAML의 안쪽 되돌이는 다음의 해를 어림하는 것으로 볼 수 있다.

$$\theta^*_{\mathcal{T}} = \argmin_\theta \mathcal{L}_{\mathcal{T}}(\theta)$$

초기화 $\theta_0$에서 기울기 걸음 몇 번을 밟아서 말이다.

### 여러 과제 학습과의 이음

MAML은 다음을 최적화한다.

$$\min_\theta \sum_{\mathcal{T}} \mathcal{L}_{\mathcal{T}}(\theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}}(\theta))$$

이는 여러 과제 학습과 이어지지만, 맞춘 뒤에 평가한다는 결정적 차이가 있다.

### 만능 함수 어림

어떤 조건 아래에서는 넉넉한 용량을 가진 MAML이 어떤 학습 알고리즘이든 어림하도록 배울 수 있어 만능 메타 학습기가 된다.

---

## 7. 실용적인 고려

### MAML에서의 배치 정규화

안쪽 되돌이에서 흐르는 통계를 고쳐서는 안 되므로 배치 정규화는 MAML에서 까다롭다.

```python
# 모든 BatchNorm 층에 track_running_stats=False를 준다
for module in model.modules():
    if isinstance(module, nn.BatchNorm2d):
        module.track_running_stats = False
```

### 기울기 자르기

이차 셈에서 기울기가 터지지 않게 막는다.

```python
# meta_loss.backward() 뒤에
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
```

### 기억 효율

큰 모델에서는 기울기 검사점 두기를 쓴다.

```python
from torch.utils.checkpoint import checkpoint

def inner_loop_checkpointed(self, support_x, support_y, params):
    def adaptation_step(params_flat, support_x, support_y):
        # 편 것을 되돌리고 한 걸음 밟는다
        ...
    
    for step in range(self.inner_steps):
        params_flat = checkpoint(adaptation_step, params_flat, support_x, support_y)
    
    return unflatten(params_flat)
```

---

## 연습문제

**연습문제 1.**
MAML 갱신 규칙을 끌어내고 안쪽 되돌이와 바깥 되돌이의 몫을 설명하라.

??? success "연습문제 1 풀이"
    안쪽 되돌이: 과제 $i$에 맞춘다. 곧 $\theta'_i = \theta - \alpha \nabla_\theta L_i^{\text{support}}(\theta)$이다. 바깥 되돌이: 초기화를 최적화한다. 곧 $\theta \leftarrow \theta - \beta \nabla_\theta \sum_i L_i^{\text{query}}(\theta'_i)$이다. 바깥 기울기는 안쪽 기울기 걸음을 꿰뚫어 미분해야 하므로 이차 도함수가 필요하다.

---

**연습문제 2.**
MAML은 왜 이차 기울기가 필요한가? 일차 MAML(FOMAML)은 언제 좋은 어림이 되는가?

??? success "연습문제 2 풀이"
    바깥 기울기 $\nabla_\theta L(\theta'_i)$은 헤세-벡터 곱인 $\frac{\partial \theta'_i}{\partial \theta} = I - \alpha \nabla^2 L$을 쓴다. FOMAML은 이 항을 버리고 $\nabla_{\theta'} L$을 그대로 쓴다. 안쪽 학습률이 작아 $\theta' \approx \theta$일 때 FOMAML이 잘 굴러가며 셈은 훨씬 싸다.

---

**연습문제 3.**
`torch.autograd.grad`를 써서 파이토치로 MAML 안쪽 되돌이를 구현하라.

??? success "연습문제 3 풀이"
    ```python
    def inner_loop(model, support_x, support_y, inner_lr, inner_steps):
        params = list(model.parameters())
        for _ in range(inner_steps):
            loss = F.cross_entropy(model(support_x), support_y)
            grads = torch.autograd.grad(loss, params, create_graph=True)
            params = [p - inner_lr * g for p, g in zip(params, grads)]
        return params  # 맞춘 매개변수
    ```

---

**연습문제 4.**
MAML의 한계는 무엇인가?

??? success "연습문제 4 풀이"
    한계: (1) 값비싸다. 이차 기울기와 안쪽 되돌이 걸음이 여럿 필요하다. (2) 기억을 많이 쓴다. 과제마다 계산 그래프를 담아 두어야 한다. (3) 안쪽 학습률과 걸음 수에 민감하다. (4) 모든 과제가 같은 구조를 나누어 쓴다고 놓는다. (5) 깊은 망에서는 흔들릴 수 있다. Reptile이나 ProtoNet 같은 대안이 이 가운데 일부를 다룬다.

## 정리하며

MAML은 소수 예시 학습에 힘 있는 최적화 기반 접근법을 준다.

1. **배우는 법 배우기**: 재빨리 맞추어 갈 수 있게 하는 초기화를 찾는다
2. **모델 가리지 않음**: 미분할 수 있는 모델이면 무엇이든 된다
3. **이론에 뿌리를 둠**: 이차 정보를 쓰는 두 층 최적화

주요 맞바꿈은 다음과 같다.

- 거리 학습보다 두루 쓸 수 있지만 셈이 값비싸다
- 이차 셈은 어림할 수 있다(FOMAML, Reptile)
- 배치 정규화를 비롯한 상태를 지닌 연산을 조심스레 다루어야 한다

**참고 문헌**

1. Finn, C., et al. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." ICML 2017.
2. Nichol, A., et al. "On First-Order Meta-Learning Algorithms." arXiv 2018.
3. Li, Z., et al. "Meta-SGD: Learning to Learn Quickly for Few-Shot Learning." arXiv 2017.
4. Raghu, A., et al. "Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML." ICLR 2020.
5. Finn, C., et al. "Online Meta-Learning." ICML 2019.
