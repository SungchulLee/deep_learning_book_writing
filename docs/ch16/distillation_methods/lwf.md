# Learning without Forgetting(LwF)

Learning without Forgetting(LwF)은 새 과제 입력에서 이전 모델의 예측을 유지하도록 증류하는 지속 학습 방법이다. 이전 원본 표본을 다시 사용하지 않고 새 지도 로스와 이전 출력의 증류 로스를 함께 최적화한다. 다만 새 입력에서의 출력 유지가 과거 데이터 전체의 성능 보장을 뜻하지는 않는다.

## 1. 이론적 기초

### 새 입력에서 이전 출력을 보존하기

새 과제 학습을 시작하기 전 모델을 복사해 고정 교사로 둔다. 같은 새 입력 x에 대해 교사의 이전 과제 출력과 학생의 대응 출력을 비교한다. 교사는 정답 라벨을 제공하는 것이 아니라 이전 모델의 함수 값을 제공한다.

### 온도와 출력 분포

$$p_T(c\mid x)=\frac{\exp(z_c(x)/T)}{\sum_j\exp(z_j(x)/T)},\qquad T>0$$

T가 커지면 고정 로짓의 확률 분포가 더 완만해진다. T=1도 일반적인 소프트맥스 확률이며 원핫 정답처럼 항상 딱딱한 목표가 되는 것은 아니다.

## 2. LwF의 로스 함수

새 과제의 지도 로스와 이전 출력에 대한 증류를 합산한다.

$$L=L_{\mathrm{task}}+\lambda T^2\operatorname{KL}(p_{\mathrm{old},T}(x)\|p_{\mathrm{new},T}(x))$$

코드의 kl_div에는 학생의 로그확률과 교사의 확률을 이 순서로 넣고 batchmean으로 평균 낸다. T²는 온도가 커질 때의 경사 스케일 변화를 완화하는 관례이며, 서로 다른 T의 최적화가 완전히 같아지는 것은 아니다.

과제별 출력 헤드나 이전 클래스 부분집합이 있으면 어떤 열을 맞추고 어디에서 소프트맥스를 정규화할지 정해야 한다. 아래 기본 구현은 같은 크기의 단일 출력 전체를 비교하는 단순화다.

## 3. PyTorch 구현

### LwF 학습기

첫 과제 이후 각 과제 시작에 모델을 deepcopy하고 교사의 경사를 끄며 평가 모드로 둔다. 새 옵티마이저를 과제마다 생성하므로 Adam 상태는 이어지지 않는다. 클래스 확장이나 과제별 헤드 선택은 기본 학습기에 포함되지 않는다. 모델 이동은 호출자가 수행하고 빈 로더는 허용하지 않는다.

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
    3. 증류 로스와 새 과제의 로스를 합친다
    
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
            model: 뉴럴 넷트웍 모델
            device: 셈할 장치
            distill_lambda: 증류 로스의 무게
            temperature: 부드러운 목표의 온도
                        클수록 분포가 부드럽다
            learning_rate: 최적화기의 러닝레이트
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
        앎 증류 로스를 셈한다.
        
        부드러운 확률 분포 사이의 KL 벌어짐을 쓴다.
        L_distill = T² × KL(softmax(z_old/T) || softmax(z_new/T))
        
        인수:
            new_logits: 지금 모델의 로짓
            old_logits: 옛 (얼린) 모델의 로짓
        
        반환값:
            증류 로스(스칼라)
        """
        T = self.temperature
        
        # 부드러운 확률
        old_probs = F.softmax(old_logits / T, dim=1)
        new_log_probs = F.log_softmax(new_logits / T, dim=1)
        
        # KL 벌어짐(T² 눈금을 씌운 것)
        # 참고: F.kl_div은 첫 인자로 로그 확률을 받는다
        loss = F.kl_div(new_log_probs, old_probs, reduction='batchmean')
        
        # 그래디언트의 크기를 지키려고 T²을 곱한다
        return loss * (T ** 2)
    
    def save_old_model(self):
        """
        증류에 쓸 참조로 지금 모델을 담아 둔다.
        
        새 과제를 익히기 전에 부른다.
        """
        if self.current_task > 0:
            self.old_model = copy.deepcopy(self.model)
            self.old_model.eval()
            
            # 모든 파라메터 얼리기
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
                
                # 지금 모델로 피드포워드한다
                output = self.model(data)
                
                # 과제마다의 로스(교차 엔트로피)
                task_loss = self.criterion(output, target)
                
                # 증류 로스(옛 모델이 있을 때)
                distill_loss = torch.tensor(0.0, device=self.device)
                
                if self.old_model is not None:
                    with torch.no_grad():
                        old_output = self.old_model(data)
                    
                    distill_loss = self.distillation_loss(output, old_output)
                
                # 결합된 로스
                total_loss = task_loss + self.distill_lambda * distill_loss
                
                # 역전파
                total_loss.backward()
                optimizer.step()
                
                # 로스를 좇는다
                epoch_losses['total'] += total_loss.item()
                epoch_losses['task'] += task_loss.item()
                epoch_losses['distill'] += distill_loss.item()
                num_batches += 1
            
            # 로스를 평균 낸다
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

## 4. 하이퍼파라미터 분석

### 온도의 효과

온도는 클래스 간 로짓 차이를 어떤 스케일로 비교할지 정한다. T가 매우 크면 분포가 균등에 가까워지지만 T² 스케일과 로짓 상대 정보도 함께 고려해야 한다. 유효한 범위는 검증 과제에서 선택한다.

아래 민감도 분석의 create_model과 ContinualLearningMetrics는 별도로 준비해야 한다. 후보마다 같은 초기화·순서를 사용하고 test_loaders 자리에 모델 선택용 검증 로더를 두어 최종 테스트 누수를 피한다.

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

### 증류 계수

λ=0이면 증류 없이 새 과제를 학습한다. 큰 값은 이전 출력에 더 강한 제약을 주어 새 학습을 제한할 수 있다. 효과는 출력 클래스 수·로스 정규화·온도와 함께 달라지므로 λ=0.5~1 같은 고정 범위를 보편적으로 권하지 않는다.

---

## 5. LwF의 거동 이해하기

새 과제와 이전 과제의 입력·표현이 관련되어 있으면 새 입력에서 얻은 교사 신호가 보존에 도움이 될 수 있다. 그러나 관측하지 않은 과거 입력에서의 출력은 별도로 제약되지 않는다.

입력 분포가 크게 다르거나 이전 교사가 새 입력에 불확실한 경우 증류 신호가 충분하지 않을 수 있다. 이전 모델의 오류도 전달될 수 있으므로 새 학습 성능과 과거 유지 성능을 함께 측정한다.

## 6. 한 걸음 나아간 LwF 변형

### 여러 머리 구조를 쓰는 LwF

과제별 헤드가 있는 구조의 시작 부분이다. 아래 하위 클래스는 nn.Module이 아닌 학습기 래퍼를 상속하며, 기존 train_on_task는 self.model만 호출하므로 새 forward나 task_heads를 사용하지 않는다. 헤드 추가·옵티마이저 등록·이전 헤드의 교사 저장과 평가 선택을 연결해야 완성된다. 더미 입력의 크기와 피처 차원도 모델에 맞춰야 한다.

```python
class MultiHeadLwFLearner(LwFLearner):
    """
    과제마다 출력 머리를 두는 LwF.
    
    피처 뽑개는 함께 쓰고 과제마다 제 출력 레이어를 갖는다.
    증류는 머리마다 적용한다.
    """
    
    def __init__(self, feature_extractor, num_classes_per_task, device, **kwargs):
        super().__init__(feature_extractor, device, **kwargs)
        
        self.feature_extractor = feature_extractor
        self.num_classes_per_task = num_classes_per_task
        self.task_heads = nn.ModuleList()
        
        # 피처 차원을 얻는다
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
        """과제를 지정할 수 있는 피드포워드."""
        features = self.feature_extractor(x)
        
        if task_id is not None:
            return self.task_heads[task_id](features)
        else:
            # 머리를 모두 되돌린다(증류용)
            return [head(features) for head in self.task_heads]
```

### 피처 증류를 곁들인 LwF

아래 FeatureLwF는 피처 사전과 비교 함수만 선언한다. 피처 추출 훅을 등록하거나 train_on_task에 피처 로스를 더하지 않으므로 그대로는 피처 증류가 적용되지 않는다. 교사 피처는 분리하고 학생 피처는 미분 경로를 유지하도록 연결해야 한다.

```python
class FeatureLwF(LwFLearner):
    """
    중간 피처 증류를 곁들인 LwF.
    
    출력뿐 아니라 중간 표현까지 옛 모델과 닮도록
    옥죈다.
    """
    
    def __init__(self, model, device, feature_lambda=0.1, **kwargs):
        super().__init__(model, device, **kwargs)
        self.feature_lambda = feature_lambda
        
        # 중간 피처를 잡아내도록 갈고리를 건다
        self.features = {}
        self.old_features = {}
    
    def feature_distillation_loss(self, new_features, old_features):
        """중간 피처 사이의 평균제곱오차 로스를 셈한다."""
        loss = 0.0
        for name in new_features:
            if name in old_features:
                loss += F.mse_loss(new_features[name], old_features[name])
        return loss
```

---

## 7. 다른 방법과의 비교

| 방법 | 보존에 사용하는 정보 | 주요 비용 |
|---|---|---|
| LwF | 새 입력에 대한 이전 모델 출력 | 교사 모델 저장과 순전파 |
| EWC | 이전 웨이트와 중요도 | 중요도 추정·저장 및 벌점 |
| 경험 리플레이 | 저장한 과거 표본 | 버퍼와 추가 표본 학습 |

LwF의 추가 저장은 모델 크기에 비례하지만 출력 헤드가 늘면 모델 자체도 커질 수 있다. EWC도 온라인 병합인지 과제별 저장인지에 따라 비용이 다르다. 이전 원본 미저장이 곧 개인정보 보호를 보장하거나 정확도의 우열을 결정하지는 않는다.

## 8. 기대되는 결과

Split MNIST의 정확도·BWT는 출력 공간과 라벨 재매핑, 클래스 분할·학습 예산·계수에 따라 달라진다. 조건이 없는 수치 표를 LwF·EWC·리플레이의 고정 순위로 제시할 수 없다.

같은 조건에서 단순 순차 학습을 포함해 비교하고, 각 과제의 학습 직후 정확도와 이후 성능, 시간·메모리를 보고한다. 이번 편집에서는 코드를 실행하거나 새 결과를 측정하지 않았다.

## 9. 실용적인 고려

LwF를 적용하려면 교사·학생의 출력 의미와 입력 전처리가 대응해야 한다. 교사를 고정한 평가 모드에서 실행하고, 학생 학습 모드·새 지도 로스와 증류 로스의 스케일을 확인한다.

온도·증류 계수·러닝레이트는 독립적인 검증 과제에서 선택한다. 모델 선택에 쓸 이전 과제 검증 데이터를 접근할 수 있는지 명시하며, 그 데이터를 증류에 사용하면 새 입력만 쓰는 기본 설정과 구분한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
LwF에서 교사가 제공하는 목표와 새 지도 정답의 차이를 설명하라.

</div>

??? success "연습문제 1 풀이"
    새 정답은 현재 과제의 예측을 학습시킨다. 교사 출력은 같은 입력에 대한 이전 모델의 확률을 유지시키는 신호이며 실제 이전 과제 정답을 대신 관측한 것은 아니다. 두 로스는 다른 목적을 가지므로 계수를 검증한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
교사를 no_grad 안에서 실행하면 어떤 비용이 남는가?

</div>

??? success "연습문제 2 풀이"
    교사 역전파 그래프를 저장하지 않아 그 비용은 줄지만 교사 파라메터 저장과 순전파 계산은 남는다. 학생은 과제·증류 로스를 미분해야 한다. 출력 캐시를 쓰면 저장 입력·증강 조건이 일치해야 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
MultiHeadLwFLearner에 헤드를 추가하면 기본 학습 루프가 자동 학습하는가?

</div>

??? success "연습문제 3 풀이"
    아니다. 상속한 루프는 self.model인 피처 추출기를 호출하고 그 파라메터만 옵티마이저에 등록한다. task_heads를 통과하는 예측·지도 로스와 이전 헤드 증류, 헤드 파라메터 등록 및 평가 경로를 함께 구현해야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
FeatureLwF의 로스 함수를 정의하기만 하면 피처 증류가 적용되는가?

</div>

??? success "연습문제 4 풀이"
    학습 루프에서 실제 학생·교사 피처를 수집하고 로스를 호출해 total_loss에 더해야 한다. 현재 코드는 그 연결이 없다. 교사 피처는 고정 타깃으로 분리하고 학생 피처는 학습 가능한 경로에 연결한다.

## 정리하며

LwF는 새 입력에서 이전 모델의 출력을 맞추는 함수 수준의 제약이다. 과거 원본을 저장하지 않는 설정이지만 보존 효과는 입력 분포와 출력 대응에 의존한다. 단일 출력 기본 구현과 아직 학습 루프에 연결되지 않은 다중 헤드·피처 증류 변형을 구분한다.

**참고 문헌**

1. Li, Z., & Hoiem, D. (2017). Learning without forgetting. *IEEE TPAMI*.

2. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *NeurIPS Workshop*.

3. Dhar, P., et al. (2019). Learning without memorizing. *CVPR*.

4. Jung, H., et al. (2016). Less-forgetting learning in deep neural networks. *AAAI*.
