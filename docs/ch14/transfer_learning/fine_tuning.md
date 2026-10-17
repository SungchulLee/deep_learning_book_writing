# 전이 학습의 미세 조정 방법

사전 학습 모델의 계수를 전부 또는 일부 갱신하는 방법과 고정 피처 위의 헤드 학습을 비교한다. 단계적 동결 해제·레이어별 러닝레이트·망각 완화 기법의 조건을 살펴본다. 제시한 조각의 실제 모델 경로·옵티마이저 포함 여부를 먼저 확인해야 한다.

## 1. 학습 목표

- 전체 미세 조정과 피처 추출·선형 프로빙의 차이를 이해한다.
- 동결 해제 시점과 옵티마이저 계수 그룹을 연결한다.
- 새 과제 적응과 기존 과제 성능 저하를 함께 평가한다.
- 모델 구조·데이터·예산에 따라 갱신 범위를 선택한다.

## 2. 들어가며

미세 조정은 사전 학습 웨이트를 새 과제에서 갱신한다. 전체 계수, 일부 레이어 또는 어댑터 등 추가 계수만 학습하는 방식이 있다. 원천·목표 분포와 성능 보존 요구가 달라 어느 한 방식이 항상 최적이지는 않다.

---

## 3. 미세 조정의 스펙트럼

```
Feature Extraction ←――――――――――――――――――→ Full Fine-tuning
(Frozen encoder)                        (All params trainable)

Less adaptation                         More adaptation
Lower risk of forgetting               Higher risk of forgetting
Faster training                        Slower training
Lower performance ceiling              Higher performance ceiling
```

---

## 4. 전면 미세 조정

FullFineTuning은 AutoModel의 첫 토큰 출력을 분류한다. 모든 계수의 경사를 켜더라도 풀러처럼 로짓 경로에서 사용되지 않는 계수는 로스로 갱신되지 않을 수 있다. AutoModel마다 hidden_size·첫 토큰의 의미가 다르므로 모든 체크포인트에 같은 래퍼가 맞지는 않는다.

과제 데이터로 모델의 파라메터를 모두 갱신한다.

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class FullFineTuning(nn.Module):
    """
    전면 미세 조정: 모든 파라메터를 학습한다.
    
    다음일 때 가장 좋다.
    - 과제에 맞는 큰 데이터셋이 있을 때
    - 과제가 사전 학습과 크게 다를 때
    - 최고 성능이 필요할 때
    """
    
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # [CLS] 토큰
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

# 모든 파라메터를 학습한다
model = FullFineTuning("bert-base-uncased", num_labels=2)
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# 전면 미세 조정의 흔한 초파라메터
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
```

### 전면 미세 조정의 설정 선택

러닝레이트·배치·에포크·워밍업·감쇠는 모델·과제와 실제 갱신 예산에 맞춰 선택한다. BERT 분류 미세 조정의 수치를 CNN·다른 언어 모델·처음부터 학습에 그대로 적용하지 않는다. 학습·검증·테스트 역할을 분리해 선택한다.

---

## 5. 피처 추출과 인코더 동결

FeatureExtraction의 헤드는 E→E→C MLP라 선형 프로빙과 다르다. no_grad와 계수 동결은 경사만 막으며 model.train()이면 인코더 드롭아웃이 켜진다. 고정 피처 평가에는 인코더 eval 폴리시도 별도로 정해야 한다.

사전 학습된 웨이트를 얼리고 과제에 맞는 헤드만 학습한다.

```python
class FeatureExtraction(nn.Module):
    """
    피처 뽑기: 인코더를 얼리고 분류기만 학습한다.
    
    다음일 때 가장 좋다.
    - 데이터셋이 아주 작을 때
    - 과제가 사전 학습과 비슷할 때
    - 빠르게 되풀이해야 할 때
    - 사전 학습 지식을 지키는 것이 매우 중요할 때
    """
    
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # 인코더의 파라메터를 모두 얼린다
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # 분류기만 학습한다
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # 인코더의 그래디언트는 셈하지 않는다
        with torch.no_grad():
            outputs = self.encoder(input_ids, attention_mask=attention_mask)
        
        pooled = outputs.last_hidden_state[:, 0]
        return self.classifier(pooled)
    
    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

model = FeatureExtraction("bert-base-uncased", num_labels=2)
print(f"Trainable parameters: {model.num_trainable_params:,}")  # 약 59만 대 전체 1억 900만

# 작은 머리만 학습하므로 러닝레이트를 더 높일 수 있다
optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)
```

---

## 6. 단계적 동결 해제

이 예제의 model.encoder는 Hugging Face BertModel이고 실제 레이어는 model.encoder.encoder.layer에 있다. GradualUnfreezeScheduler는 model.encoder.layer 또는 layers만 찾아 기본 예제에서 ValueError가 난다. 아래 코드는 경로를 수정하기 전에는 실행 가능한 일정이 아니다.

requires_grad를 켜도 옵티마이저가 처음의 학습 계수만 필터링해 갖고 있으면 새 계수는 갱신되지 않는다. 계수 그룹·러닝레이트·모드 폴리시를 함께 정한다. 예시 루프는 메시지만 출력하며 실제 학습 단계를 호출하지 않는다.

학습 중에 (출력에 가장 가까운) 위에서부터 레이어를 차츰 녹인다.

```python
from typing import List

class GradualUnfreezeScheduler:
    """
    레이어를 위에서 아래로 차츰 녹인다.
    
    이치: 위 레이어는 과제에 더 맞고 아래 레이어는 더 일반적인 피처를 지닌다.
    위부터 녹이면 낮은 수준의 표현을 흐트러뜨리지 않고
    모델이 차츰 맞추어 갈 수
    있다.
    """
    
    def __init__(self, model: nn.Module, encoder_attr: str = 'encoder'):
        self.model = model
        self.encoder = getattr(model, encoder_attr)
        
        # 인코더 레이어를 얻는다 (BERT 같은 모델에서 통한다)
        if hasattr(self.encoder, 'layer'):
            self.layers = list(self.encoder.layer)
        elif hasattr(self.encoder, 'layers'):
            self.layers = list(self.encoder.layers)
        else:
            raise ValueError("Could not find encoder layers")
        
        self.num_layers = len(self.layers)
        self.unfrozen_count = 0
        
        # 처음에는 인코더 레이어를 모두 얼린다
        self._freeze_all()
    
    def _freeze_all(self):
        """인코더의 파라메터를 모두 얼린다."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_layer(self, layer_idx: int):
        """정한 레이어를 녹인다."""
        for param in self.layers[layer_idx].parameters():
            param.requires_grad = True
    
    def unfreeze_next(self) -> bool:
        """
        다음 레이어를 녹인다 (위에서 아래로).
        
        반환값:
            레이어를 녹였으면 True, 이미 모두 녹였으면 False
        """
        if self.unfrozen_count >= self.num_layers:
            return False
        
        # 위(색인이 큰 쪽)에서 아래로 녹인다
        layer_idx = self.num_layers - 1 - self.unfrozen_count
        self.unfreeze_layer(layer_idx)
        self.unfrozen_count += 1
        
        print(f"Unfroze layer {layer_idx} ({self.unfrozen_count}/{self.num_layers})")
        return True
    
    def unfreeze_n_layers(self, n: int):
        """레이어 n개를 한꺼번에 녹인다."""
        for _ in range(n):
            if not self.unfreeze_next():
                break
    
    def unfreeze_embeddings(self):
        """임베딩 레이어를 녹인다 (대개 마지막에 한다)."""
        if hasattr(self.encoder, 'embeddings'):
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = True
            print("Unfroze embeddings")

# 쓰는 보기
model = FullFineTuning("bert-base-uncased", num_labels=2)
scheduler = GradualUnfreezeScheduler(model)

num_epochs = 12
unfreeze_every = 2  # 두 세대마다 새 레이어를 하나씩 녹인다

for epoch in range(num_epochs):
    # 레이어를 차츰 녹인다
    if epoch > 0 and epoch % unfreeze_every == 0:
        scheduler.unfreeze_next()
    
    # 마지막에 임베딩도 녹일 수 있다
    if epoch == num_epochs - 2:
        scheduler.unfreeze_embeddings()
    
    # train_epoch(model, train_loader, optimizer)
    print(f"Epoch {epoch}: training...")
```

---

## 7. 레이어별 러닝레이트

get_discriminative_lr_params도 같은 BertModel 레이어 경로를 잘못 찾아 기본 예제에서 실패한다. 아래 표는 경로를 맞춘 뒤 num_layers=12에 코드의 지수식을 적용한 수치이며 실제 출력이 아니다. 풀러 등 그룹에서 빠진 계수와 중복·동결 계수도 확인한다.

레이어마다 다른 러닝레이트를 준다. (일반 피처인) 아래 레이어에는 낮게, (과제에 맞는) 위 레이어에는 높게 준다.

```python
from typing import List, Dict, Any

def get_discriminative_lr_params(
    model: nn.Module,
    base_lr: float = 2e-5,
    lr_decay: float = 0.95,
    weight_decay: float = 0.01
) -> List[Dict[str, Any]]:
    """
    레이어별 러닝레이트를 가진 파라메터 배치를 만든다.
    
    러닝레이트 일정:
    - 임베딩: base_lr * decay^num_layers (가장 낮다)
    - 0레이어: base_lr * decay^(num_layers-1)
    - 1레이어: base_lr * decay^(num_layers-2)
    - ...
    - N-1레이어: base_lr * decay (두 번째로 높다)
    - 분류기: base_lr (가장 높다)
    
    인수:
        model: 인코더와 분류기를 가진 모델
        base_lr: 맨 위 레이어와 분류기의 러닝레이트
        lr_decay: 레이어마다 곱하는 감쇠 (0 < decay < 1)
        weight_decay: 규제를 위한 웨이트 감쇠
    """
    param_groups = []
    
    # 인코더 레이어의 수를 얻는다
    if hasattr(model.encoder, 'layer'):
        encoder_layers = model.encoder.layer
    elif hasattr(model.encoder, 'layers'):
        encoder_layers = model.encoder.layers
    else:
        raise ValueError("Could not find encoder layers")
    
    num_layers = len(encoder_layers)
    
    # 임베딩 — 가장 낮은 러닝레이트
    if hasattr(model.encoder, 'embeddings'):
        emb_lr = base_lr * (lr_decay ** (num_layers + 1))
        param_groups.append({
            'params': model.encoder.embeddings.parameters(),
            'lr': emb_lr,
            'weight_decay': weight_decay,
            'name': 'embeddings'
        })
    
    # 인코더 레이어 — 차츰 높아지는 러닝레이트
    for i, layer in enumerate(encoder_layers):
        layer_lr = base_lr * (lr_decay ** (num_layers - i))
        param_groups.append({
            'params': layer.parameters(),
            'lr': layer_lr,
            'weight_decay': weight_decay,
            'name': f'layer_{i}'
        })
    
    # 분류기 머리 — 가장 높은 러닝레이트
    if hasattr(model, 'classifier'):
        param_groups.append({
            'params': model.classifier.parameters(),
            'lr': base_lr,
            'weight_decay': weight_decay,
            'name': 'classifier'
        })
    
    return param_groups

def print_lr_schedule(param_groups: List[Dict[str, Any]]):
    """러닝레이트 일정을 찍는다."""
    print("Learning Rate Schedule:")
    print("-" * 40)
    for group in param_groups:
        print(f"  {group.get('name', 'unnamed'):15s}: {group['lr']:.2e}")

# 사용법
model = FullFineTuning("bert-base-uncased", num_labels=2)
param_groups = get_discriminative_lr_params(model, base_lr=2e-5, lr_decay=0.9)
print_lr_schedule(param_groups)

optimizer = torch.optim.AdamW(param_groups)
```

### 러닝레이트 일정 보기 (BERT-base, 12레이어)

| 요소 | 지수 | base_lr=2e-5, decay=0.9 | decay=0.95 |
|---|---|---|---|
| 임베딩 | 13 | 5.08e-6 | 1.03e-5 |
| 0레이어 | 12 | 5.65e-6 | 1.08e-5 |
| 5레이어 | 7 | 9.57e-6 | 1.40e-5 |
| 11레이어 | 1 | 1.80e-5 | 1.90e-5 |
| 분류기 | 0 | 2.00e-5 | 2.00e-5 |

---

## 8. 파국적 망각

미세 조정의 중심 어려움은 **파국적 망각**이다. 모델이 특정 과제에 맞추어지면서 사전 학습 지식을 잃는 것이다.

### 평가와 가능한 원인

기존 과제 성능의 실제 저하를 독립 자료에서 확인해야 망각을 논할 수 있다. 새 과제 로스만으로 측정할 수 없고, 목표 과제 자체가 악화되는 음의 전이와도 구분한다.

1. **분포 이동**: 미세 조정 데이터가 사전 학습 데이터와 다르다
2. **그래디언트 간섭**: 과제의 그래디언트가 일반 지식을 덮어쓴다
3. **그릇 다시 나누기**: 모델이 새 과제를 위해 뉴런을 다른 용도로 쓴다

### 누그러뜨리는 방법

#### 1. L2-SP

기준 상태는 고정·독립 복사해 보관하고 이름·장치를 맞춘다. 현재 예제는 전체 model.state_dict를 기준으로 삼아 새 분류 헤드의 무작위 초기값에도 패널티를 줄 수 있다. “사전 학습된 계수만 보존”하려는 목적과 범위를 맞춰야 한다.

사전 학습 웨이트에서 벗어나는 것에 패널티를 준다.

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \sum_i \|\theta_i - \theta_i^{\text{pre}}\|^2
$$

```python
class L2SPRegularizer:
    """
    L2-SP: (사전 학습 웨이트인) 출발점 쪽으로의 L2 규제.
    """
    
    def __init__(self, model: nn.Module, pretrained_state: dict, lambda_l2sp: float = 0.01):
        self.model = model
        self.pretrained_state = {k: v.clone() for k, v in pretrained_state.items()}
        self.lambda_l2sp = lambda_l2sp
    
    def penalty(self) -> torch.Tensor:
        """사전 학습 웨이트에서의 L2 거리를 셈한다."""
        penalty = 0.0
        for name, param in self.model.named_parameters():
            if name in self.pretrained_state:
                penalty += torch.sum((param - self.pretrained_state[name].to(param.device)) ** 2)
        return self.lambda_l2sp * penalty

# 사용법
pretrained_state = {k: v.clone() for k, v in model.state_dict().items()}
regularizer = L2SPRegularizer(model, pretrained_state, lambda_l2sp=0.01)

# 학습 고리 안에서
loss = criterion(outputs, labels)
loss = loss + regularizer.penalty()
loss.backward()
```

#### 2. EWC

compute_fisher는 outputs.logits와 labels를 받는 모델을 가정하지만 위 FullFineTuning은 labels 인수가 없고 Tensor를 반환한다. 그대로 연결할 수 없다. 배치 평균 로스의 경사를 제곱해 표본 수로 나누는 방식도 표본별 점수 경사 제곱의 평균 Fisher 추정과 다르다. 빈 자료·샘플 상한·장치와 기준 상태의 독립 복사도 처리해야 한다.

penalty는 앞 수식의 lambda/2와 달리 lambda를 그대로 곱하므로 계수 관례가 다르다. Fisher가 크면 기존 과제에 중요하다는 해석 역시 추정 자료·근사 조건에 따른다.

피셔 정보로 중요한 파라메터에 더 큰 웨이트를 준다.

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \sum_i \frac{\lambda}{2} F_i (\theta_i - \theta_i^{\text{pre}})^2
$$

```python
class EWCRegularizer:
    """
    파국적 망각을 막기 위한 탄력적 웨이트 굳히기.
    """
    
    def __init__(
        self,
        model: nn.Module,
        fisher_dict: dict,
        pretrained_state: dict,
        lambda_ewc: float = 1000
    ):
        self.model = model
        self.fisher = fisher_dict
        self.pretrained = pretrained_state
        self.lambda_ewc = lambda_ewc
    
    @classmethod
    def compute_fisher(
        cls,
        model: nn.Module,
        dataloader,
        num_samples: int = 1000
    ) -> dict:
        """데이터에서 피셔 정보를 어림한다."""
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        model.eval()
        
        count = 0
        for batch in dataloader:
            if count >= num_samples:
                break
            
            outputs = model(**batch)
            # 피셔 정보에는 로그 가능도를 쓴다
            log_probs = torch.log_softmax(outputs.logits, dim=-1)
            labels = batch['labels']
            loss = -log_probs.gather(1, labels.unsqueeze(1)).mean()
            
            model.zero_grad()
            loss.backward()
            
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad ** 2
            
            count += batch['input_ids'].size(0)
        
        # 정규화
        for n in fisher:
            fisher[n] /= count
        
        return fisher
    
    def penalty(self) -> torch.Tensor:
        penalty = 0.0
        for n, p in self.model.named_parameters():
            if n in self.fisher and n in self.pretrained:
                penalty += (self.fisher[n] * (p - self.pretrained[n].to(p.device)) ** 2).sum()
        return self.lambda_ewc * penalty
```

#### 3. 리플레이

ReplayDataLoader는 작업 배치마다 확률 replay_ratio로 리플레이 배치를 하나 추가한다. 이는 전체 배치 중 정확히 그 비율이 리플레이라는 뜻은 아니다. 배치 크기·로스 웨이트가 다르면 실제 학습 비중도 달라진다. 빈 replay_loader와 반복기·샘플링 상태도 처리해야 한다.

과제 데이터에 사전 학습 분포의 표본을 섞는다.

```python
class ReplayDataLoader:
    """
    과제 데이터에 사전 학습 분포의 되풀이 데이터를 섞는다.
    """
    
    def __init__(
        self,
        task_loader,
        replay_loader,
        replay_ratio: float = 0.1
    ):
        self.task_loader = task_loader
        self.replay_loader = replay_loader
        self.replay_ratio = replay_ratio
        self.replay_iter = iter(replay_loader)
    
    def __iter__(self):
        for task_batch in self.task_loader:
            yield task_batch
            
            # 이따금 되풀이 배치를 내보낸다
            if torch.rand(1).item() < self.replay_ratio:
                try:
                    replay_batch = next(self.replay_iter)
                except StopIteration:
                    self.replay_iter = iter(self.replay_loader)
                    replay_batch = next(self.replay_iter)
                yield replay_batch
```

#### 4. 추가 계수 기반 적응

기반 웨이트를 동결해도 어댑터·LoRA를 켠 전체 출력에서 기존 과제 성능이 나빠질 수 있다. 원래 모델을 보존하는 것과 수정된 모델의 망각 방지는 다르다.

LoRA·어댑터는 갱신 계수를 제한하는 선택이며 항상 망각을 제거하거나 메모리·품질에서 우수하다는 보장은 없다.

---

## 9. 방법 고르기

### 판단의 틀

```
                    Dataset Size
                    Small (<1K)    Medium (1K-100K)    Large (>100K)
Task Similarity     
to Pretraining

High              Feature         Discriminative LR    Full Fine-tune
                  Extraction      + Gradual Unfreeze   (careful LR)

Medium            LoRA/Adapters   Gradual Unfreeze     Full Fine-tune
                                  + L2-SP              

Low               LoRA with       Full Fine-tune       Full Fine-tune
                  larger rank     + EWC/Replay         
```

### 빠른 참고

피처 추출, 일부 해제, 전체 미세 조정과 추가 계수 방식을 같은 검증 예산에서 비교한다. 데이터 크기·분포 거리·기존 지식 보존 요구 외에 전체 추론 비용과 사전 학습 체크포인트의 적합성도 고려한다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
전체 미세 조정·피처 추출·선형 프로빙을 비교하라.

</div>

??? success "풀이"
    피처 추출은 기반 계수를 고정하고 새 헤드를 학습하며 그 헤드가 MLP일 수도 있다. 선형 프로빙은 동결 피처 위의 선형 분류로 표현을 평가하는 특별한 경우다. 전체 미세 조정은 기반 계수도 갱신한다. 성능의 고정 순위는 없고 데이터·분할·모드·학습 예산을 맞춰야 한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
기반 레이어와 새 헤드에 다른 러닝레이트를 주는 목적과 한계를 설명하라.

</div>

??? success "풀이"
    기반 계수의 변화를 줄이면서 새 헤드를 더 빠르게 학습하려는 선택이다. 앞 레이어가 모든 과제에서 더 일반적이거나 특정 러닝레이트 비가 최적이라는 법칙은 없다. 옵티마이저 그룹에 모든 필요한 계수가 포함됐는지와 실제 그래디언트·갱신을 확인한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 단계적 동결 해제 조각이 가정하는 속성과 옵티마이저·실행 모드를 설명하라.

</div>

??? success "풀이"
    위 모델에는 layer_groups 속성이 없어 직접 사용할 수 없다. 실제 레이어 그룹을 구성하고 범위·빈 목록을 검증해야 한다. 해제한 계수가 옵티마이저에 들어 있는지, 기반 모드·드롭아웃·정규화 버퍼 폴리시와 러닝레이트 스케줄이 맞는지도 확인한다. requires_grad 변경만으로 갱신이 완성되지는 않는다.

    ```python
    for epoch in range(num_epochs):
        # 다음 레이어 배치를 녹인다
        for param in model.layer_groups[min(epoch, len(model.layer_groups)-1)].parameters():
            param.requires_grad = True
        # 지금까지 녹인 레이어를 모두 써서 학습한다
        train_one_epoch(model, optimizer, dataloader)
    ```

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
음의 전이를 어떻게 진단하고 대안을 비교해야 하는가?

</div>

??? success "풀이"
    같은 분할·학습 예산에서 새 모델 학습, 동결 피처와 미세 조정을 비교한다. 분포가 다르거나 데이터가 많다는 사실만으로 전이를 배제하지 않는다. 목표 성능·바이어스·캘리브레이션·비용에서 손해가 있는지 확인하고 다른 체크포인트·전처리·갱신 범위를 검증한다.

## 정리하며

계수 동결·해제와 실제 옵티마이저 갱신·모드를 함께 관리한다. 코드의 BERT 레이어 경로와 EWC 인터페이스·추정 문제를 구분하고, 계수 제한·정규화·리플레이를 망각이나 최고 성능의 보장으로 해석하지 않는다.

**참고 문헌**

1. Howard, J., & Ruder, S. (2018). "Universal Language Model Fine-tuning for Text Classification." ACL.
2. Kirkpatrick, J., et al. (2017). "Overcoming Catastrophic Forgetting in Neural Networks." PNAS.
3. Li, X., et al. (2018). "Explicit Inductive Bias for Transfer Learning with Convolutional Networks." ICML.
4. Peters, M., et al. (2019). "To Tune or Not to Tune? Adapting Pretrained Representations to Diverse Tasks."
