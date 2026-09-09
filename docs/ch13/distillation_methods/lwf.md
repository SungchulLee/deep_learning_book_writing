# 잊지 않고 배우기(LwF)

**잊지 않고 배우기(LwF)**는 Li와 Hoiem(2017)이 들여온, 앎 증류에 바탕을 둔 이어 배우기 방법이다. 경험 되살리기와 달리 LwF은 앞선 보기를 담아 두지 않는다. 그 대신 지금 과제의 데이터를 써서 모델의 출력을 제 자신의 앞선 예측에 맞추어 옛 과제의 성능을 지킨다.

!!! success "핵심 이점"

    - **데이터를 담아 두지 않음**: 사생활을 지키는 방법
    - **붙박이 기억**: 과제가 늘어도 기억이 커지지 않는다
    - **지금 데이터를 끌어 씀**: 새 과제 데이터로 옛 앎을 지킨다
    - **단순한 구현**: 보통의 앎 증류에 바탕을 둔다

---

## 1. 이론적 바탕

### 앎 증류 되짚기

앎 증류는 딱딱한 이름표가 아니라 부드러운 확률 분포를 맞추어 "스승" 모델의 앎을 "제자" 모델로 옮긴다.

$$
\mathcal{L}_{\text{distill}} = \text{KL}\left( p_{\text{teacher}}(y|x; T) \| p_{\text{student}}(y|x; T) \right)
$$

여기서 $T$은 분포를 부드럽게 하는 온도 매개변수이다.

$$
p(y_i|x; T) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

온도가 높을수록 확률 분포가 부드러워져 부류 사이의 관계를 더 많이 드러낸다.

### LwF의 통찰

LwF은 새 과제를 익히기 전의 모델을 "스승"으로 삼아 새 과제를 배우는 동안 모델을 옥죈다. 핵심 통찰은 다음과 같다.

> **옛 과제의 데이터를 쓸 수 없더라도, 새 과제의 데이터로 모델이 옛 과제에서 무엇을 맞힐지 어림할 수 있다.**

새 과제의 그림을 옛 모델에 넣으면 옛 과제 출력에서 내는 예측이 앞서 배운 것을 비춘다. 고친 모델이 이 예측에 맞추도록 북돋우면 옛 앎이 지켜진다.

---

## 2. LwF의 손실 함수

과제 $\tau$에서 LwF 손실은 다음을 합친다.

1. 새 과제의 **교차 엔트로피**: 보통의 분류 손실
2. 옛 과제의 **증류**: 익히기 전의 부드러운 출력에 맞춘다

$$
\mathcal{L}_{\text{LwF}} = \mathcal{L}_{\text{CE}}(y_\tau, \hat{y}_\tau) + \lambda \cdot \mathcal{L}_{\text{distill}}(\hat{y}_{1:\tau-1}^{\text{old}}, \hat{y}_{1:\tau-1}^{\text{new}})
$$

여기서 각 기호는 다음과 같다.

- $\mathcal{L}_{\text{CE}}$: 지금 과제의 교차 엔트로피 손실
- $\hat{y}_{1:\tau-1}^{\text{old}}$: 익히기 전 모델(얼려 둔 것)의 부드러운 예측
- $\hat{y}_{1:\tau-1}^{\text{new}}$: 옛 과제 머리에서 지금 모델이 내는 예측
- $\lambda$: 균형 매개변수
- 증류에는 온도 $T$을 쓴다

### 증류 손실 자세히 보기

증류 손실은 온도 눈금을 씌운 KL 벌어짐을 쓴다.

$$
\mathcal{L}_{\text{distill}} = T^2 \cdot \text{KL}\left( \text{softmax}(z^{\text{old}}/T) \| \text{softmax}(z^{\text{new}}/T) \right)
$$

$T^2$ 인자는 온도가 높을 때 기울기가 작아지는 것을 메워 준다.

---

## 3. PyTorch 구현

### 온전한 LwF 학습기

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Dict, Optional
import copy

class LwFLearner:
    """
    이어 배우기를 위한 잊지 않고 배우기.
    
    LwF은 다음으로 잊음을 막는다.
    1. 새 과제를 익히기 전에 모델의 출력을 적어 둔다
    2. 앎 증류로 옛 출력에 맞춘다
    3. 증류 손실과 새 과제의 손실을 합친다
    
    참고: Li & Hoiem, "Learning Without Forgetting," TPAMI 2017
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 distill_lambda: float = 1.0,
                 temperature: float = 2.0,
                 learning_rate: float = 0.001):
        """
        LwF 학습기를 초기화한다.
        
        인수:
            model: 신경망 모델
            device: 셈할 장치
            distill_lambda: 증류 손실의 무게
            temperature: 부드러운 목표의 온도
                        클수록 분포가 부드럽다
            learning_rate: 최적화기의 학습률
        """
        self.model = model
        self.device = device
        self.distill_lambda = distill_lambda
        self.temperature = temperature
        self.learning_rate = learning_rate
        
        self.criterion = nn.CrossEntropyLoss()
        
        # 증류에 쓸 참조 모델을 담아 둔다(얼린 채)
        self.old_model: Optional[nn.Module] = None
        
        self.current_task = 0
    
    def distillation_loss(self, 
                          new_logits: torch.Tensor,
                          old_logits: torch.Tensor) -> torch.Tensor:
        """
        앎 증류 손실을 셈한다.
        
        부드러운 확률 분포 사이의 KL 벌어짐을 쓴다.
        L_distill = T² × KL(softmax(z_old/T) || softmax(z_new/T))
        
        인수:
            new_logits: 지금 모델의 로짓
            old_logits: 옛 (얼린) 모델의 로짓
        
        반환값:
            증류 손실(스칼라)
        """
        T = self.temperature
        
        # 부드러운 확률
        old_probs = F.softmax(old_logits / T, dim=1)
        new_log_probs = F.log_softmax(new_logits / T, dim=1)
        
        # KL 벌어짐(T² 눈금을 씌운 것)
        # 참고: F.kl_div은 첫 인자로 로그 확률을 받는다
        loss = F.kl_div(new_log_probs, old_probs, reduction='batchmean')
        
        # 기울기의 크기를 지키려고 T²을 곱한다
        return loss * (T ** 2)
    
    def save_old_model(self):
        """
        증류에 쓸 참조로 지금 모델을 담아 둔다.
        
        새 과제를 익히기 전에 부른다.
        """
        if self.current_task > 0:
            self.old_model = copy.deepcopy(self.model)
            self.old_model.eval()
            
            # 모든 매개변수 얼리기
            for param in self.old_model.parameters():
                param.requires_grad = False
            
            print("  Old model saved for distillation")
    
    def train_on_task(self,
                      train_loader: DataLoader,
                      epochs: int = 5,
                      verbose: bool = True) -> Dict:
        """
        LwF 증류를 곁들여 과제로 익힌다.
        
        인수:
            train_loader: 학습 데이터 로더
            epochs: 학습 에포크 수
            verbose: 진행 상황 출력 여부
        
        반환값:
            학습 통계
        """
        # 익히기 전에 옛 모델을 담아 둔다
        self.save_old_model()
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        losses = {'total': [], 'task': [], 'distill': []}
        
        for epoch in range(epochs):
            self.model.train()
            epoch_losses = {'total': 0, 'task': 0, 'distill': 0}
            num_batches = 0
            
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                # 지금 모델로 앞먹임한다
                output = self.model(data)
                
                # 과제마다의 손실(교차 엔트로피)
                task_loss = self.criterion(output, target)
                
                # 증류 손실(옛 모델이 있을 때)
                distill_loss = torch.tensor(0.0, device=self.device)
                
                if self.old_model is not None:
                    with torch.no_grad():
                        old_output = self.old_model(data)
                    
                    distill_loss = self.distillation_loss(output, old_output)
                
                # 결합된 손실
                total_loss = task_loss + self.distill_lambda * distill_loss
                
                # 역전파
                total_loss.backward()
                optimizer.step()
                
                # 손실을 좇는다
                epoch_losses['total'] += total_loss.item()
                epoch_losses['task'] += task_loss.item()
                epoch_losses['distill'] += distill_loss.item()
                num_batches += 1
            
            # 손실을 평균 낸다
            for key in epoch_losses:
                losses[key].append(epoch_losses[key] / num_batches)
            
            if verbose:
                print(f"  Epoch {epoch+1}/{epochs}: "
                      f"Total={losses['total'][-1]:.4f}, "
                      f"Task={losses['task'][-1]:.4f}, "
                      f"Distill={losses['distill'][-1]:.4f}")
        
        self.current_task += 1
        
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
        LwF으로 이어 배우기를 온전히 한다.
        
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
            
            # 증류를 곁들여 지금 과제로 익힌다
            self.train_on_task(
                train_loaders[task_id],
                epochs=epochs_per_task
            )
            
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

## 4. 초매개변수 분석

### 온도의 효과

온도 $T$은 확률 분포의 부드러움을 다스린다.

| 온도 | 효과 | 쓰임새 |
|-------------|--------|----------|
| T = 1 | 딱딱한 목표 | 보통의 학습 |
| T = 2~4 | 부드러운 목표 | 흔한 증류 |
| T > 4 | 매우 부드러움 | 가름에 쓸 정보를 잃을 수 있다 |

```python
def analyze_temperature(train_loaders, test_loaders, device):
    """온도가 LwF 성능에 미치는 영향을 뜯어본다."""
    temperatures = [1.0, 2.0, 3.0, 4.0, 5.0]
    results = []
    
    for T in temperatures:
        model = create_model().to(device)
        learner = LwFLearner(
            model=model,
            device=device,
            temperature=T,
            distill_lambda=1.0
        )
        
        result = learner.train_continual(
            train_loaders, test_loaders, epochs_per_task=5
        )
        
        metrics = ContinualLearningMetrics(result['accuracy_matrix'])
        results.append({
            'temperature': T,
            'avg_accuracy': metrics.average_accuracy,
            'backward_transfer': metrics.backward_transfer
        })
    
    return results
```

### 증류 무게(λ)

`distill_lambda` 매개변수는 새로 배우기와 앎 지키기의 균형을 잡는다.

| λ 값 | 효과 |
|---------|--------|
| λ = 0 | 증류 없음(소박한 학습) |
| λ = 0.5~1.0 | 균형 잡힘(흔한 선택) |
| λ > 2.0 | 세게 지킴, 새로 배우기를 막을 수 있다 |

---

## 5. LwF의 거동 이해하기

### 새 과제 데이터가 통하는 까닭

LwF은 새 과제의 데이터로 옛 과제의 증류 목표를 셈한다. 이것이 통하는 까닭은 다음과 같다.

1. **나누어 쓰는 표현**: 앞쪽 층은 두루 쓰이는 특징을 배울 때가 많다
2. **출력의 한결같음**: 모델은 어떤 입력에서도 예측을 크게 바꾸어서는 안 된다
3. **벌주기 효과**: 증류가 매개변수 변화에 대한 제약 노릇을 한다

### 한계

1. **과제가 닮았다는 가정**: 과제가 표현을 나누어 쓸 때 가장 잘 통한다
2. **영역 이동**: 새 과제가 아주 다르면 성능이 떨어진다
3. **옛 과제를 직접 익히지 않음**: 옛 과제의 성능을 곧바로 최적화할 수 없다
4. **파국적 잊음이 여전히 일어날 수 있음**: 닮지 않은 과제 차례에서 그렇다

---

## 6. 한 걸음 나아간 LwF 변형

### 여러 머리 구조를 쓰는 LwF

출력 머리를 따로 두는 과제 증분 학습을 위한 것이다.

```python
class MultiHeadLwFLearner(LwFLearner):
    """
    과제마다 출력 머리를 두는 LwF.
    
    특징 뽑개는 함께 쓰고 과제마다 제 출력 층을 갖는다.
    증류는 머리마다 적용한다.
    """
    
    def __init__(self, feature_extractor, num_classes_per_task, device, **kwargs):
        super().__init__(feature_extractor, device, **kwargs)
        
        self.feature_extractor = feature_extractor
        self.num_classes_per_task = num_classes_per_task
        self.task_heads = nn.ModuleList()
        
        # 특징 차원을 얻는다
        with torch.no_grad():
            dummy = torch.randn(1, 1, 28, 28).to(device)
            feat_dim = feature_extractor(dummy).shape[1]
        
        self.feat_dim = feat_dim
    
    def add_task_head(self):
        """과제마다의 새 출력 머리를 더한다."""
        head = nn.Linear(self.feat_dim, self.num_classes_per_task)
        head = head.to(self.device)
        self.task_heads.append(head)
    
    def forward(self, x, task_id=None):
        """과제를 지정할 수 있는 앞먹임."""
        features = self.feature_extractor(x)
        
        if task_id is not None:
            return self.task_heads[task_id](features)
        else:
            # 머리를 모두 되돌린다(증류용)
            return [head(features) for head in self.task_heads]
```

### 특징 증류를 곁들인 LwF

출력뿐 아니라 중간 특징까지 증류한다.

```python
class FeatureLwF(LwFLearner):
    """
    중간 특징 증류를 곁들인 LwF.
    
    출력뿐 아니라 중간 표현까지 옛 모델과 닮도록
    옥죈다.
    """
    
    def __init__(self, model, device, feature_lambda=0.1, **kwargs):
        super().__init__(model, device, **kwargs)
        self.feature_lambda = feature_lambda
        
        # 중간 특징을 잡아내도록 갈고리를 건다
        self.features = {}
        self.old_features = {}
    
    def feature_distillation_loss(self, new_features, old_features):
        """중간 특징 사이의 평균제곱오차 손실을 셈한다."""
        loss = 0.0
        for name in new_features:
            if name in old_features:
                loss += F.mse_loss(new_features[name], old_features[name])
        return loss
```

---

## 7. 다른 방법과의 견줌

### LwF과 EWC

| 갈래 | LwF | EWC |
|--------|-----|-----|
| 저장 | 옛 모델 사본 | 피셔 + 가장 좋은 매개변수 |
| 기억 | O(d) | O(T × d) |
| 셈 | 앞먹임 2배 | 앞먹임 1배 + 벌 |
| 필요한 데이터 | 없음 | 옛 데이터로 구한 피셔 |
| 가장 알맞은 곳 | 닮은 과제 | 어떤 과제 차례든 |

### LwF과 경험 되살리기

| 갈래 | LwF | 경험 되살리기 |
|--------|-----|-------------------|
| 데이터 저장 | 없음 | 보기 |
| 사생활 | 높음 | 낮음 |
| 성능 | 좋음 | 대체로 더 좋음 |
| 영역 이동 | 민감함 | 더 튼튼함 |

---

## 8. 기대되는 결과

Split MNIST 잣대에서는 다음과 같다.

| 방법 | 평균 정확도 | BWT |
|--------|--------------|-----|
| 소박한 방법 | 55% 남짓 | -45% |
| LwF | 80% 남짓 | -18% |
| EWC | 85% 남짓 | -12% |
| ER | 92% 남짓 | -6% |

LwF은 사생활을 지키면서도 소박한 학습보다 뜻있게 나은 성능을 준다.

---

## 9. 실용적인 고려

### LwF을 언제 쓸까

✓ **알맞은 곳:**

- 사생활이 민감한 응용
- 시각과 뜻의 특징을 나누어 쓰는 과제
- 과제 수가 그리 많지 않을 때
- 기억이 빠듯한 환경

✗ **피할 곳:**

- 과제끼리 아주 다를 때
- 최고 성능이 필요할 때
- 과제 차례가 길 것으로 보일 때

### 구현 요령

1. **온도 손질**: T=2에서 시작하고, 출력이 너무 뾰족하면 올려라
2. **람다 고르기**: 검증으로 distill_lambda를 손질하라
3. **학습률**: 보통의 학습보다 낮게 잡아야 할 수 있다
4. **모델 구조**: 표현을 나누어 쓸수록 더 잘 굴러간다

---

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

## 정리하며

잊지 않고 배우기는 사생활을 지키는 이어 배우기 방법을 준다.

- **장치**: 지금 과제의 데이터를 쓰는 앎 증류
- **이점**: 앞선 보기를 담아 두지 않는다
- **한계**: 과제가 표현을 나누어 쓴다고 놓는다
- **가장 알맞은 쓰임**: 과제가 서로 이어진, 사생활이 민감한 응용

**참고 문헌**

1. Li, Z., & Hoiem, D. (2017). Learning without forgetting. *IEEE TPAMI*.

2. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *NeurIPS Workshop*.

3. Dhar, P., et al. (2019). Learning without memorizing. *CVPR*.

4. Jung, H., et al. (2016). Less-forgetting learning in deep neural networks. *AAAI*.
