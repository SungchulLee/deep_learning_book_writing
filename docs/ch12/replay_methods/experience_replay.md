# 경험 되살리기
## 들어가며

**경험 되살리기**는 이어 배우기에서 가장 쓸모 있고 직관적인 방법에 든다. 핵심 생각은 단순하다. 앞선 과제의 보기를 담은 기억 버퍼를 지니고, 새 과제를 익히는 동안 그것을 되살려 쓴다. 옛 보기와 새 보기를 섞으면 모델이 새 과제를 배우면서도 앞선 과제의 솜씨를 지킨다.

!!! success "핵심 이점"

    - **직관적인 장치**: 옛 보기를 되뇌어 잊음을 곧바로 다룬다
    - **든든한 성능**: 실전에서 가장 쓸모 있는 방법일 때가 많다
    - **융통성**: 다른 이어 배우기 기법과 섞을 수 있다
    - **단순함**: 구현하고 이해하기 쉽다

## 이론적 바탕

### 되살리기가 통하는 까닭

이어 배우는 동안의 손실 함수를 생각해 보자. 되살리기가 없으면 지금 과제의 손실만 가장 작게 한다.

$$
\mathcal{L}(\theta) = \mathcal{L}_\tau(\theta)
$$

되살리기를 쓰면 모든 과제를 함께 익히는 것을 어림하게 된다.

$$
\mathcal{L}(\theta) = \mathcal{L}_\tau(\theta) + \sum_{k=1}^{\tau-1} \hat{\mathcal{L}}_k(\theta)
$$

여기서 $\hat{\mathcal{L}}_k(\theta)$은 기억 표본으로 어림한 과제 $k$의 손실이다.

### 함께 익히기와의 이음

경험 되살리기는 함께 익히기의 어림으로 볼 수 있다. 기억이 무한하여 앞선 보기를 모두 담아 둘 수 있다면, 되살리기는 모든 데이터를 한꺼번에 익히는 것과 같아지며 그때는 잊음이 없다.

## 기억 버퍼 전략

### 저수지 뽑기

**저수지 뽑기**는 지금까지 본 보기를 고르게 나타내는 붙박이 크기의 버퍼를 지닌다.

```python
import random
import torch
from typing import List, Tuple, Optional

class ReservoirBuffer:
    """
    경험 되살리기를 위한 저수지 뽑기 버퍼.
    
    도착한 차례와 상관없이 지금까지 본 보기 전체에 대한
    고른 표본을 지닌다.
    """
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer: List[Tuple[torch.Tensor, int]] = []
        self.seen_count = 0
    
    def add(self, example: torch.Tensor, label: int):
        """저수지 뽑기로 보기를 더한다."""
        self.seen_count += 1
        
        if len(self.buffer) < self.max_size:
            self.buffer.append((example.clone(), label))
        else:
            prob = self.max_size / self.seen_count
            if random.random() < prob:
                idx = random.randint(0, self.max_size - 1)
                self.buffer[idx] = (example.clone(), label)
    
    def sample(self, batch_size: int) -> Tuple[Optional[torch.Tensor], 
                                                Optional[torch.Tensor]]:
        """버퍼에서 배치를 뽑는다."""
        if len(self.buffer) == 0:
            return None, None
        
        sample_size = min(batch_size, len(self.buffer))
        indices = random.sample(range(len(self.buffer)), sample_size)
        
        examples = torch.stack([self.buffer[i][0] for i in indices])
        labels = torch.tensor([self.buffer[i][1] for i in indices], 
                             dtype=torch.long)
        return examples, labels
```

### 부류를 고르게 맞춘 버퍼

부류마다 똑같은 비중으로 담는다.

```python
class ClassBalancedBuffer:
    """
    부류를 고르게 맞춘 기억 버퍼.
    
    기억에서 모든 부류가 똑같은 비중을 갖게 하며,
    부류 증분 학습에 중요하다.
    """
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.class_buffers: Dict[int, List[torch.Tensor]] = {}
    
    def add_task(self, examples: torch.Tensor, labels: torch.Tensor):
        """새 과제의 보기를 더한다."""
        unique_classes = labels.unique().tolist()
        
        for cls in unique_classes:
            mask = labels == cls
            class_examples = examples[mask]
            self.class_buffers[cls] = [ex.clone() for ex in class_examples]
        
        self._rebalance()
    
    def _rebalance(self):
        """부류가 고르도록 버퍼의 균형을 다시 잡는다."""
        num_classes = len(self.class_buffers)
        if num_classes == 0:
            return
        
        per_class = self.max_size // num_classes
        
        for cls in self.class_buffers:
            if len(self.class_buffers[cls]) > per_class:
                indices = random.sample(
                    range(len(self.class_buffers[cls])), per_class
                )
                self.class_buffers[cls] = [
                    self.class_buffers[cls][i] for i in indices
                ]
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """부류를 고르게 맞추어 뽑는다."""
        all_examples = []
        all_labels = []
        
        for cls, examples in self.class_buffers.items():
            for ex in examples:
                all_examples.append(ex)
                all_labels.append(cls)
        
        if len(all_examples) == 0:
            return None, None
        
        sample_size = min(batch_size, len(all_examples))
        indices = random.sample(range(len(all_examples)), sample_size)
        
        return (torch.stack([all_examples[i] for i in indices]),
                torch.tensor([all_labels[i] for i in indices], dtype=torch.long))
```

## 온전한 경험 되살리기 구현

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Dict

class ExperienceReplayLearner:
    """
    경험 되살리기를 쓰는 이어 배우는 학습기.
    
    전략:
    1. 앞선 보기를 담은 기억 버퍼를 지닌다
    2. 익히는 동안 지금 과제의 데이터와 되살린 표본을 섞는다
    3. 합친 손실이 새 과제를 배우면서도 잊음을 막는다
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 memory_size: int = 1000,
                 examples_per_task: int = 200,
                 replay_batch_ratio: float = 0.5,
                 learning_rate: float = 0.001):
        """
        되살리기 학습기를 초기화한다.
        
        인수:
            model: 신경망 모델
            device: 셈할 장치
            memory_size: 기억 버퍼의 전체 용량
            examples_per_task: 과제마다 담아 둘 보기 수
            replay_batch_ratio: 배치에서 되살린 것의 비율
            learning_rate: 최적화기의 학습률
        """
        self.model = model
        self.device = device
        self.memory_size = memory_size
        self.examples_per_task = examples_per_task
        self.replay_batch_ratio = replay_batch_ratio
        self.learning_rate = learning_rate
        
        self.criterion = nn.CrossEntropyLoss()
        
        # 기억 저장소
        self.memory_data: List[torch.Tensor] = []
        self.memory_labels: List[int] = []
    
    def add_to_memory(self, data_loader: DataLoader):
        """
        지금 과제의 보기를 기억에 더한다.
        
        대표할 만한 보기를 무작위로 고른다.
        
        인수:
            data_loader: 지금 과제의 DataLoader
        """
        # 과제의 데이터를 모두 모은다
        all_data, all_labels = [], []
        for data, labels in data_loader:
            all_data.append(data)
            all_labels.append(labels)
        
        all_data = torch.cat(all_data, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # 무작위로 고르기
        n_available = all_data.size(0)
        n_select = min(self.examples_per_task, n_available)
        indices = torch.randperm(n_available)[:n_select]
        
        # 기억에 더한다
        for idx in indices:
            self.memory_data.append(all_data[idx].clone())
            self.memory_labels.append(all_labels[idx].item())
        
        # 용량을 넘으면 잘라낸다
        if len(self.memory_data) > self.memory_size:
            self.memory_data = self.memory_data[-self.memory_size:]
            self.memory_labels = self.memory_labels[-self.memory_size:]
        
        print(f"  Memory size: {len(self.memory_data)} examples")
    
    def sample_memory(self, batch_size: int):
        """기억에서 배치를 뽑는다."""
        if len(self.memory_data) == 0:
            return None, None
        
        sample_size = min(batch_size, len(self.memory_data))
        indices = random.sample(range(len(self.memory_data)), sample_size)
        
        examples = torch.stack([self.memory_data[i] for i in indices])
        labels = torch.tensor([self.memory_labels[i] for i in indices],
                             dtype=torch.long)
        return examples, labels
    
    def train_on_task(self, 
                      train_loader: DataLoader,
                      epochs: int = 5,
                      verbose: bool = True) -> Dict:
        """
        경험 되살리기를 곁들여 과제로 익힌다.
        
        인수:
            train_loader: 학습 데이터 로더
            epochs: 학습 에포크 수
            verbose: 진행 상황 출력 여부
        
        반환값:
            학습 통계
        """
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        losses = {'total': [], 'current': [], 'replay': []}
        
        for epoch in range(epochs):
            self.model.train()
            epoch_losses = {'total': 0, 'current': 0, 'replay': 0}
            num_batches = 0
            
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                batch_size = data.size(0)
                
                optimizer.zero_grad()
                
                # 지금 과제의 손실
                output = self.model(data)
                current_loss = self.criterion(output, target)
                
                # 되살리기 손실
                replay_loss = torch.tensor(0.0, device=self.device)
                replay_size = int(batch_size * self.replay_batch_ratio)
                
                if replay_size > 0 and len(self.memory_data) > 0:
                    replay_data, replay_labels = self.sample_memory(replay_size)
                    replay_data = replay_data.to(self.device)
                    replay_labels = replay_labels.to(self.device)
                    
                    replay_output = self.model(replay_data)
                    replay_loss = self.criterion(replay_output, replay_labels)
                
                # 결합된 손실
                total_loss = current_loss + replay_loss
                
                # 되돌리고 최적화한다
                total_loss.backward()
                optimizer.step()
                
                # 손실을 좇는다
                epoch_losses['total'] += total_loss.item()
                epoch_losses['current'] += current_loss.item()
                epoch_losses['replay'] += replay_loss.item()
                num_batches += 1
            
            # 손실을 평균 낸다
            for key in epoch_losses:
                losses[key].append(epoch_losses[key] / num_batches)
            
            if verbose:
                print(f"  Epoch {epoch+1}/{epochs}: "
                      f"Total={losses['total'][-1]:.4f}, "
                      f"Current={losses['current'][-1]:.4f}, "
                      f"Replay={losses['replay'][-1]:.4f}")
        
        return losses
    
    def evaluate(self, test_loader: DataLoader) -> float:
        """과제에서 정확도를 평가한다."""
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
                        train_loaders: List[DataLoader],
                        test_loaders: List[DataLoader],
                        epochs_per_task: int = 5) -> Dict:
        """
        되살리기로 이어 배우기를 온전히 한다.
        
        인수:
            train_loaders: 학습 DataLoader 목록
            test_loaders: 시험 DataLoader 목록
            epochs_per_task: 과제마다의 학습 시대 수
        
        반환값:
            정확도 행렬을 담은 사전
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
                epochs=epochs_per_task
            )
            
            # 익힌 **뒤에** 기억에 더한다
            self.add_to_memory(train_loaders[task_id])
            
            # 모든 과제에서 평가한다
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
        
        return {'accuracy_matrix': accuracy_matrix}
```

## 초매개변수 분석

### 기억 크기의 효과

| 기억 크기 | 효과 | 맞바꿈 |
|-------------|--------|-----------|
| 작음(100) | 지킴이 약함 | 기억을 아낌 |
| 보통(500~1000) | 균형이 좋음 | 권함 |
| 큼(5000 이상) | 함께 익히기에 가까움 | 기억을 많이 씀 |

### 과제마다의 보기 수

`examples_per_task` 매개변수가 과제 사이의 균형을 다스린다.

```python
def analyze_memory_parameters(train_loaders, test_loaders, device):
    """기억 매개변수의 영향을 뜯어본다."""
    
    configs = [
        {'memory_size': 500, 'examples_per_task': 100},
        {'memory_size': 1000, 'examples_per_task': 200},
        {'memory_size': 2000, 'examples_per_task': 400},
    ]
    
    results = []
    for config in configs:
        model = create_model().to(device)
        learner = ExperienceReplayLearner(
            model=model,
            device=device,
            **config
        )
        
        result = learner.train_continual(
            train_loaders, test_loaders, epochs_per_task=5
        )
        
        metrics = ContinualLearningMetrics(result['accuracy_matrix'])
        results.append({
            **config,
            'avg_accuracy': metrics.average_accuracy,
            'backward_transfer': metrics.backward_transfer
        })
    
    return results
```

## 한 걸음 나아간 되살리기 전략

### 기울기 기반 고르기

기울기의 다양함이 가장 커지는 보기를 고른다.

```python
class GradientBasedBuffer:
    """
    기울기 정보를 바탕으로 보기를 고른다.
    
    기울기가 크거나 서로 다른 보기에 우선순위를 주는데,
    학습에 알맹이가 가장 많기 때문이다.
    """
    
    def __init__(self, model, max_size, device):
        self.model = model
        self.max_size = max_size
        self.device = device
        self.buffer = []
        self.gradients = []
    
    def compute_gradient_score(self, example, label):
        """보기 하나의 기울기 크기를 셈한다."""
        self.model.zero_grad()
        example = example.unsqueeze(0).to(self.device)
        label = torch.tensor([label], device=self.device)
        
        output = self.model(example)
        loss = F.cross_entropy(output, label)
        loss.backward()
        
        # 기울기 제곱의 합
        score = sum(p.grad.norm().item() ** 2 
                   for p in self.model.parameters() 
                   if p.grad is not None)
        
        return score
    
    def add_task(self, examples, labels, num_select):
        """기울기 점수로 우선순위를 매겨 보기를 더한다."""
        scores = []
        for i in range(len(examples)):
            score = self.compute_gradient_score(examples[i], labels[i].item())
            scores.append((score, i))
        
        # 기울기 기준으로 위 보기를 고른다
        scores.sort(reverse=True)
        selected = [scores[i][1] for i in range(min(num_select, len(scores)))]
        
        for idx in selected:
            if len(self.buffer) < self.max_size:
                self.buffer.append((examples[idx].clone(), labels[idx].item()))
```

### 손실 기반 고르기

손실이 큰 보기(어려운 보기)에 우선순위를 준다.

```python
class LossBasedBuffer:
    """
    손실 값을 바탕으로 보기를 고른다.
    
    어려운 보기(손실이 큰 것)가 판단 경계를 지키는 데
    더 값질 수 있다.
    """
    
    def select_by_loss(self, examples, labels, model, num_select, device):
        """손실이 가장 큰 보기를 고른다."""
        model.eval()
        
        with torch.no_grad():
            outputs = model(examples.to(device))
            losses = F.cross_entropy(outputs, labels.to(device), 
                                    reduction='none')
        
        # 손실이 가장 큰 보기를 고른다
        _, indices = torch.topk(losses, min(num_select, len(losses)))
        
        return examples[indices.cpu()], labels[indices.cpu()]
```

## 다른 방법과의 견줌

### 되살리기와 벌주기

| 갈래 | 경험 되살리기 | EWC |
|--------|------------------|-----|
| 기억 | 보기를 담음 | 피셔와 매개변수를 담음 |
| 사생활 | 낮음(데이터를 담음) | 높음(데이터 없음) |
| 성능 | 대체로 더 좋음 | 좋지만 한계가 있음 |
| 규모 확장성 | 기억이 커지거나 붙박임 | 과제 수에 선형 |
| 융통성 | 섞기 쉬움 | 홀로 씀 |

### 기대되는 결과

Split MNIST 잣대에서는 다음과 같다.

| 방법 | 평균 정확도 | BWT | 기억 |
|--------|--------------|-----|--------|
| 소박한 방법 | 55% 남짓 | -45% | 0 |
| ER(1000) | 92% 남짓 | -6% | 보기 1000개 |
| EWC | 85% 남짓 | -12% | 매개변수의 2배 |

## 실용적인 고려

### 경험 되살리기를 언제 쓸까

✓ **알맞은 곳:**

- 기억을 담아 두어도 괜찮을 때
- 사생활이 큰 걱정거리가 아닐 때
- 성능을 최대로 뽑아야 할 때
- 과제의 성격이 서로 닮았을 때

✗ **피할 곳:**

- 사생활 요구가 빠듯할 때
- 기억이 아주 빠듯할 때
- 잇단 과제가 수천 개일 때

### 구현 요령

1. **버퍼 갱신**: 과제를 익힌 뒤 기억에 더한다
2. **뽑기**: 대개 무작위 뽑기로 충분하다
3. **배치 섞기**: 지금 것과 되살린 것을 반반으로 두는 것이 좋은 기본값이다
4. **데이터 늘리기**: 되살린 표본에도 같은 늘리기를 쓴다

## 요약

경험 되살리기는 이어 배우기에 힘 있고 실전에 맞는 방법이다.

- **장치**: 앞선 보기를 담아 두고 되살린다
- **쓸모**: 가장 좋은 성능을 내는 방법일 때가 많다
- **맞바꿈**: 기억을 담아 두어야 한다
- **융통성**: 다른 방법과 섞기 쉽다

단순하고 성능이 든든해서 기억이 빠듯하지만 않다면 첫손에 꼽을 만하다.

## 참고 문헌

1. Robins, A. (1995). Catastrophic forgetting, rehearsal and pseudorehearsal. *Connection Science*.

2. Rolnick, D., et al. (2019). Experience replay for continual learning. *NeurIPS*.

3. Chaudhry, A., et al. (2019). On tiny episodic memories in continual learning. *ICML Workshop*.

4. Buzzega, P., et al. (2020). Dark experience for general continual learning: a strong, simple baseline. *NeurIPS*.

## 연습문제

**연습문제 1.**
이 방법의 핵심 생각과 그것이 파국적 잊음을 어떻게 다루는지 설명하라.

??? success "연습문제 1 풀이"
    이 방법은 새 과제를 배울 때 모델의 매개변수나 표현이 바뀌는 방식을 옥죄어 파국적 잊음을 누그러뜨린다. (벌주기, 되살리기, 증류, 구조 갈라두기로) 배운 함수의 중요한 대목을 지켜 냄으로써, 앞선 과제의 성능을 지키면서도 새 과제에 맞추어 갈 수 있게 한다.

---

**연습문제 2.**
이 접근법의 셈과 기억 요구는 무엇인가?

??? success "연습문제 2 풀이"
    요구는 변형마다 다르지만 대체로 (a) 매개변수의 중요도 무게, (b) 학습 보기의 일부, (c) 스승 모델의 출력, (d) 과제마다의 망 모듈 가운데 하나를 담아 두어야 한다. 기억 비용과 잊음 막기의 효과 사이에서 맞바꿈이 일어난다.

---

**연습문제 3.**
이 방법을 효과와 셈 비용 면에서 EWC와 견주어라.

??? success "연습문제 3 풀이"
    EWC는 대각 피셔 정보로 중요한 가중치를 짚어낸다. 이 방법은 다른 맞바꿈을 준다. 옛 과제의 성능을 더 잘 지킬 수 있고, 기억 요구가 다르며, 과제 짜임에 대한 가정도 다르다. 실험으로 견주어 보면 잣대에 따라 서로 보완되는 강점을 보일 때가 많다.

---

**연습문제 4.**
이 방법을 간추린 판으로 파이토치에 구현하라.

??? success "연습문제 4 풀이"
    구현은 대개 새 과제를 익히는 동안 보통의 교차 엔트로피 손실에 벌주기 항을 더한다. 핵심 부품은 (1) 앞선 과제 학습에서 제약을 셈하기, (2) 필요한 정보(가중치, 본보기, 스승 출력)를 담아 두기, (3) 새 과제 학습 중에 그 제약을 씌우기이다.
