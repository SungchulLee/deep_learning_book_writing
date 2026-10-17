# 경험 리플레이

경험 리플레이는 과거 표본을 버퍼에 저장하고 새 데이터 학습 중 다시 사용하는 지속 학습 방법이다. 과거 로스를 직접 학습 목표에 포함하지만, 유한 버퍼의 대표성과 샘플링·로스 웨이트에 따라 보존 효과가 달라진다. 이 페이지는 버퍼 전략과 보존된 학습기의 실제 동작을 구분한다.

## 1. 이론적 기초

### 현재 로스와 과거 로스

현재 데이터와 버퍼에서 로스를 각각 추정해 결합한다.

$$L=L_{\mathrm{current}}+\beta L_{\mathrm{memory}}$$

버퍼의 표본 분포, 샘플링 확률과 β가 과거 데이터에 주는 웨이트를 결정한다. 과제별 균등 평균과 전체 표본별 평균도 서로 다르다.

### 공동 학습과의 관계

과거 데이터를 모두 보관하더라도 새·과거 데이터를 어떤 비율로 얼마나 학습하는지에 따라 공동 학습의 목적과 경로가 달라질 수 있다. 모든 데이터에 접근한다는 사실만으로 같은 해에 도달하거나 망각이 사라지는 것은 아니다.

## 2. 기억 버퍼 전략

### 저수지 샘플링

저수지 샘플링은 지금까지 본 스트림의 각 표본이 같은 확률로 남도록 고정 용량 버퍼를 갱신한다. 충분한 표본을 본 뒤 각 표본의 포함 확률은 버퍼 용량/누적 표본 수다. 같은 표본을 여러 에포크마다 다시 넣으면 고유 표본이 아닌 반복 방문 스트림에 대한 균등성이 된다.

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

### 클래스 균형 버퍼

ClassBalancedBuffer는 클래스마다 같은 상한을 배정하고 초과 표본을 무작위로 줄인다. 기존 클래스가 다시 들어오면 해당 리스트를 새 표본으로 덮어쓴다. 정수 나눗셈의 나머지는 사용하지 않으며 클래스 수가 용량보다 많으면 클래스별 용량이 0이 된다. 과제별 로컬 라벨이 같은 숫자를 재사용하면 서로 다른 원본 클래스가 합쳐지므로 전역 ID가 필요하다.

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

---

## 3. 경험 리플레이 학습기

이 학습기는 앞의 ReservoirBuffer나 ClassBalancedBuffer를 사용하지 않고 자체 리스트를 관리한다. 과제 종료에 examples_per_task개를 고른 뒤 용량을 넘으면 앞부분을 제거하므로 오래된 과제가 먼저 사라지는 방식이다. memory_size=0에서는 [-0:]가 전체 리스트를 뜻하므로 버퍼를 비우지 못한다. 양의 용량·표본 수를 검증해야 한다.

replay_batch_ratio는 현재 배치 수에 대한 추가 리플레이 표본 수의 비율이다. 0.5이면 현재 B개와 과거 B/2개를 별도로 계산하므로 최종 데이터의 절반이 과거라는 뜻은 아니다. 두 로스를 각각 평균 내고 같은 웨이트로 더해 로스 기여도는 표본 수 비율과도 다르다.

모든 과제 데이터를 한 번 cat하여 버퍼 표본을 고르므로 버퍼 용량보다 큰 임시 메모리가 필요하다. 저장하는 텐서는 로더의 변환이 적용된 값이며 리플레이 때 새 어그멘테이션을 자동 수행하지 않는다. 모델 장치 이동과 현재·과거 라벨의 의미도 호출자가 맞춰야 한다.

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
    3. 합친 로스가 새 과제를 배우면서도 잊음을 막는다
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
            model: 뉴럴 넷트웍 모델
            device: 셈할 장치
            memory_size: 기억 버퍼의 전체 용량
            examples_per_task: 과제마다 담아 둘 보기 수
            replay_batch_ratio: 배치에서 되살린 것의 비율
            learning_rate: 최적화기의 러닝레이트
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
                
                # 지금 과제의 로스
                output = self.model(data)
                current_loss = self.criterion(output, target)
                
                # 되살리기 로스
                replay_loss = torch.tensor(0.0, device=self.device)
                replay_size = int(batch_size * self.replay_batch_ratio)
                
                if replay_size > 0 and len(self.memory_data) > 0:
                    replay_data, replay_labels = self.sample_memory(replay_size)
                    replay_data = replay_data.to(self.device)
                    replay_labels = replay_labels.to(self.device)
                    
                    replay_output = self.model(replay_data)
                    replay_loss = self.criterion(replay_output, replay_labels)
                
                # 결합된 로스
                total_loss = current_loss + replay_loss
                
                # 되돌리고 최적화한다
                total_loss.backward()
                optimizer.step()
                
                # 로스를 좇는다
                epoch_losses['total'] += total_loss.item()
                epoch_losses['current'] += current_loss.item()
                epoch_losses['replay'] += replay_loss.item()
                num_batches += 1
            
            # 로스를 평균 낸다
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

---

## 4. 하이퍼파라미터 분석

### 버퍼 용량과 과제별 표본 수

버퍼 용량은 최종 저장 상한이고 examples_per_task는 과제 종료 시 추가할 표본 수다. 자체 리스트 구현에서는 과제가 많아지면 앞의 표본이 제거되므로 과제별로 일정 수를 영구 보장하지 않는다. 버퍼 표본 수만으로 모든 문제의 적절한 용량을 정할 수 없다.

아래 민감도 분석의 create_model과 ContinualLearningMetrics는 별도로 준비해야 한다. 최종 테스트셋 대신 검증 순서를 사용해 설정을 선택하고 후보마다 같은 초기화·과제 순서를 맞춘다.

```python
def analyze_memory_parameters(train_loaders, test_loaders, device):
    """기억 파라메터의 영향을 뜯어본다."""
    
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

---

## 5. 표본 선택의 변형

### 그래디언트 크기 기반 선택

GradientBasedBuffer는 표본별 경사 노름 제곱이 큰 표본을 고른다. 경사 간 방향의 다양성을 최적화하지는 않는다. 버퍼가 가득 차면 새 표본을 교체하지 않고 gradients 목록도 쓰지 않는다. F 임포트와 모델 모드가 필요하며, 이 계산은 기존 학습 경사를 지우고 새 경사를 남기므로 학습 갱신과 분리해야 한다.

```python
class GradientBasedBuffer:
    """
    그래디언트 정보를 바탕으로 보기를 고른다.
    
    그래디언트가 크거나 서로 다른 보기에 우선순위를 주는데,
    학습에 커널이 가장 많기 때문이다.
    """
    
    def __init__(self, model, max_size, device):
        self.model = model
        self.max_size = max_size
        self.device = device
        self.buffer = []
        self.gradients = []
    
    def compute_gradient_score(self, example, label):
        """보기 하나의 그래디언트 크기를 셈한다."""
        self.model.zero_grad()
        example = example.unsqueeze(0).to(self.device)
        label = torch.tensor([label], device=self.device)
        
        output = self.model(example)
        loss = F.cross_entropy(output, label)
        loss.backward()
        
        # 그래디언트 제곱의 합
        score = sum(p.grad.norm().item() ** 2 
                   for p in self.model.parameters() 
                   if p.grad is not None)
        
        return score
    
    def add_task(self, examples, labels, num_select):
        """그래디언트 점수로 우선순위를 매겨 보기를 더한다."""
        scores = []
        for i in range(len(examples)):
            score = self.compute_gradient_score(examples[i], labels[i].item())
            scores.append((score, i))
        
        # 그래디언트 기준으로 위 보기를 고른다
        scores.sort(reverse=True)
        selected = [scores[i][1] for i in range(min(num_select, len(scores)))]
        
        for idx in selected:
            if len(self.buffer) < self.max_size:
                self.buffer.append((examples[idx].clone(), labels[idx].item()))
```

### 로스 기반 선택

로스가 큰 표본을 선택하지만 라벨 오류나 이상치에 집중할 수 있다. 이 함수는 평가 모드로 바꾼 뒤 원래 모드를 복원하지 않으므로 이후 학습에서 다시 train()을 호출해야 한다. 입력·라벨의 장치와 선택 인덱스도 일치시킨다.

```python
class LossBasedBuffer:
    """
    로스 값을 바탕으로 보기를 고른다.
    
    어려운 보기(로스가 큰 것)가 판단 경계를 지키는 데
    더 값질 수 있다.
    """
    
    def select_by_loss(self, examples, labels, model, num_select, device):
        """로스가 가장 큰 보기를 고른다."""
        model.eval()
        
        with torch.no_grad():
            outputs = model(examples.to(device))
            losses = F.cross_entropy(outputs, labels.to(device), 
                                    reduction='none')
        
        # 로스가 가장 큰 보기를 고른다
        _, indices = torch.topk(losses, min(num_select, len(losses)))
        
        return examples[indices.cpu()], labels[indices.cpu()]
```

---

## 6. 다른 방법과의 비교

리플레이는 과거 입력·정답을 저장해 다시 학습하고, EWC는 중요도와 기준 웨이트로 이동에 벌점을 준다. 두 방식은 조합할 수 있다. EWC의 메모리도 과제별 저장인지 병합인지에 따라 달라진다.

원본 표본 보관 여부는 데이터 관리 조건에 영향을 주지만 모델·통계도 정보 노출 가능성이 있어 단순한 개인정보 보호 순위를 정하지 않는다. 고정된 Split MNIST 정확도 표 대신 같은 출력 공간·데이터 순서·학습 예산에서 새 학습과 이전 유지 성능을 비교한다.

## 7. 실용적인 고려

버퍼의 갱신 시점과 분포를 명시한다. 과제 종료 갱신은 경계가 주어진 설정이며, 매 표본 갱신은 스트림 설정에 맞게 별도로 정의할 수 있다.

현재와 과거의 샘플링 비율, 로스 계수와 실제 표본 노출량을 구분해 기록한다. 동일한 전역 라벨 규약을 유지하고 데이터 어그멘테이션을 재적용할지, 이미 변환한 입력을 저장할지 정한다. 테스트 데이터는 버퍼에 넣지 않고 모델 선택은 독립적인 검증셋에서 수행한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
리플레이 버퍼가 커지면 항상 공동 학습과 같은 결과가 되는가?

</div>

??? success "연습문제 1 풀이"
    아니다. 샘플링 비율과 로스 웨이트, 표본 노출 수와 최적화 경로가 같아야 목적·결과를 비교할 수 있다. 전체 과거 자료를 보관해도 현재 자료를 과도하게 학습하면 분포 웨이트가 달라진다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
현재 배치 B=100, replay_batch_ratio=0.5일 때 표본 수와 로스 웨이트를 설명하라.

</div>

??? success "연습문제 2 풀이"
    충분한 버퍼가 있으면 현재 100개와 과거 50개를 별도로 사용한다. 과거는 표본의 1/3이지만 코드의 두 평균 로스는 계수 1로 합산되므로 평균 목표에서 같은 비중이다. 현재·과거 표본별 웨이트는 같지 않다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
자체 메모리 리스트와 저수지 샘플링은 어떻게 다른가?

</div>

??? success "연습문제 3 풀이"
    학습기는 과제마다 고른 표본을 뒤에 넣고 앞에서 제거하므로 오래된 과제가 사라진다. 저수지 샘플링은 누적 스트림에서 각 표본이 같은 확률로 남도록 교체한다. 클래스 균형 버퍼는 또 다른 목표 분포를 만든다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
과제별 라벨을 0·1로 재사용해 하나의 버퍼에 넣을 때 무엇을 확인해야 하는가?

</div>

??? success "연습문제 4 풀이"
    같은 번호가 같은 클래스 의미인지 확인한다. 다른 원본 클래스가 같은 로컬 번호를 쓰면 단일 출력 목표가 충돌할 수 있다. 전역 클래스 라벨이나 과제 ID와 헤드 선택을 함께 저장하고 평가 규약을 맞춰야 한다.

## 정리하며

경험 리플레이는 과거 표본을 다시 학습해 보존을 돕는다. 버퍼 갱신·샘플링 분포와 평균 로스 웨이트를 명시하고, 자체 리스트 구현을 저수지·클래스 균형 전략과 구분한다. 저장 용량뿐 아니라 임시 데이터 수집과 추가 학습 비용도 함께 계산한다.

**참고 문헌**

1. Robins, A. (1995). Catastrophic forgetting, rehearsal and pseudorehearsal. *Connection Science*.

2. Rolnick, D., et al. (2019). Experience replay for continual learning. *NeurIPS*.

3. Chaudhry, A., et al. (2019). On tiny episodic memories in continual learning. *ICML Workshop*.

4. Buzzega, P., et al. (2020). Dark experience for general continual learning: a strong, simple baseline. *NeurIPS*.
