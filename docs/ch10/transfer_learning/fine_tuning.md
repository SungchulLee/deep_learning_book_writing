# 전이 학습의 미세 조정 방법
## 학습 목표

- 특징 뽑기에서 전면 미세 조정까지의 스펙트럼을 이해한다
- 차츰 녹이기와 층별 학습률을 구현한다
- 파국적 망각을 알아채고 누그러뜨린다
- 과제의 요구에 맞는 미세 조정 방법을 고른다

## 들어가며

미세 조정은 사전 학습된 모델을 특정 아래쪽 과제에 맞춘다. 핵심 어려움은 값진 사전 학습 지식을 지키면서 새 데이터에 맞추는 균형을 잡는 것이다. 이 문서는 일반적인 미세 조정 방법을 다룬다. LoRA나 어댑터 같은 매개변수를 아끼는 방법은 15.6절 효율적인 대형 언어 모델 미세 조정을 보라.

## 미세 조정의 스펙트럼

```
Feature Extraction ←――――――――――――――――――→ Full Fine-tuning
(Frozen encoder)                        (All params trainable)

Less adaptation                         More adaptation
Lower risk of forgetting               Higher risk of forgetting
Faster training                        Slower training
Lower performance ceiling              Higher performance ceiling
```

## 전면 미세 조정

과제 데이터로 모델의 매개변수를 모두 갱신한다.

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class FullFineTuning(nn.Module):
    """
    전면 미세 조정: 모든 매개변수를 학습한다.
    
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

# 모든 매개변수를 학습한다
model = FullFineTuning("bert-base-uncased", num_labels=2)
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# 전면 미세 조정의 흔한 초매개변수
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
```

### 전면 미세 조정의 모범

| 초매개변수 | 흔한 범위 | 비고 |
|----------------|---------------|-------|
| 학습률 | 1e-5에서 5e-5 | 사전 학습보다 낮게 |
| 배치 크기 | 16~32 | 기억이 넉넉하면 더 크게 |
| 세대 | 2~4 | 조기 종료를 권한다 |
| 예열 | 전체 단계의 6~10% | 초기에 갈라지는 것을 막는다 |
| 가중치 감쇠 | 0.01 | 규제 |

## 특징 뽑기 (부호기 얼리기)

사전 학습된 가중치를 얼리고 과제에 맞는 머리만 학습한다.

```python
class FeatureExtraction(nn.Module):
    """
    특징 뽑기: 부호기를 얼리고 분류기만 학습한다.
    
    다음일 때 가장 좋다.
    - 데이터셋이 아주 작을 때
    - 과제가 사전 학습과 비슷할 때
    - 빠르게 되풀이해야 할 때
    - 사전 학습 지식을 지키는 것이 매우 중요할 때
    """
    
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # 부호기의 매개변수를 모두 얼린다
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
        # 부호기의 기울기는 셈하지 않는다
        with torch.no_grad():
            outputs = self.encoder(input_ids, attention_mask=attention_mask)
        
        pooled = outputs.last_hidden_state[:, 0]
        return self.classifier(pooled)
    
    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

model = FeatureExtraction("bert-base-uncased", num_labels=2)
print(f"Trainable parameters: {model.num_trainable_params:,}")  # 약 59만 대 전체 1억 900만

# 작은 머리만 학습하므로 학습률을 더 높일 수 있다
optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)
```

## 차츰 녹이기

학습 중에 (출력에 가장 가까운) 위에서부터 층을 차츰 녹인다.

```python
from typing import List

class GradualUnfreezeScheduler:
    """
    층을 위에서 아래로 차츰 녹인다.
    
    이치: 위 층은 과제에 더 맞고 아래 층은 더 일반적인 특징을 지닌다.
    위부터 녹이면 낮은 수준의 표현을 흐트러뜨리지 않고
    모델이 차츰 맞추어 갈 수
    있다.
    """
    
    def __init__(self, model: nn.Module, encoder_attr: str = 'encoder'):
        self.model = model
        self.encoder = getattr(model, encoder_attr)
        
        # 부호기 층을 얻는다 (BERT 같은 모델에서 통한다)
        if hasattr(self.encoder, 'layer'):
            self.layers = list(self.encoder.layer)
        elif hasattr(self.encoder, 'layers'):
            self.layers = list(self.encoder.layers)
        else:
            raise ValueError("Could not find encoder layers")
        
        self.num_layers = len(self.layers)
        self.unfrozen_count = 0
        
        # 처음에는 부호기 층을 모두 얼린다
        self._freeze_all()
    
    def _freeze_all(self):
        """부호기의 매개변수를 모두 얼린다."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_layer(self, layer_idx: int):
        """정한 층을 녹인다."""
        for param in self.layers[layer_idx].parameters():
            param.requires_grad = True
    
    def unfreeze_next(self) -> bool:
        """
        다음 층을 녹인다 (위에서 아래로).
        
        반환값:
            층을 녹였으면 True, 이미 모두 녹였으면 False
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
        """층 n개를 한꺼번에 녹인다."""
        for _ in range(n):
            if not self.unfreeze_next():
                break
    
    def unfreeze_embeddings(self):
        """임베딩 층을 녹인다 (대개 마지막에 한다)."""
        if hasattr(self.encoder, 'embeddings'):
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = True
            print("Unfroze embeddings")

# 쓰는 보기
model = FullFineTuning("bert-base-uncased", num_labels=2)
scheduler = GradualUnfreezeScheduler(model)

num_epochs = 12
unfreeze_every = 2  # 두 세대마다 새 층을 하나씩 녹인다

for epoch in range(num_epochs):
    # 층을 차츰 녹인다
    if epoch > 0 and epoch % unfreeze_every == 0:
        scheduler.unfreeze_next()
    
    # 마지막에 임베딩도 녹일 수 있다
    if epoch == num_epochs - 2:
        scheduler.unfreeze_embeddings()
    
    # train_epoch(model, train_loader, optimizer)
    print(f"Epoch {epoch}: training...")
```

## 층별 학습률

층마다 다른 학습률을 준다. (일반 특징인) 아래 층에는 낮게, (과제에 맞는) 위 층에는 높게 준다.

```python
from typing import List, Dict, Any

def get_discriminative_lr_params(
    model: nn.Module,
    base_lr: float = 2e-5,
    lr_decay: float = 0.95,
    weight_decay: float = 0.01
) -> List[Dict[str, Any]]:
    """
    층별 학습률을 가진 매개변수 묶음을 만든다.
    
    학습률 일정:
    - 임베딩: base_lr * decay^num_layers (가장 낮다)
    - 0층: base_lr * decay^(num_layers-1)
    - 1층: base_lr * decay^(num_layers-2)
    - ...
    - N-1층: base_lr * decay (두 번째로 높다)
    - 분류기: base_lr (가장 높다)
    
    인수:
        model: 부호기와 분류기를 가진 모델
        base_lr: 맨 위 층과 분류기의 학습률
        lr_decay: 층마다 곱하는 감쇠 (0 < decay < 1)
        weight_decay: 규제를 위한 가중치 감쇠
    """
    param_groups = []
    
    # 부호기 층의 수를 얻는다
    if hasattr(model.encoder, 'layer'):
        encoder_layers = model.encoder.layer
    elif hasattr(model.encoder, 'layers'):
        encoder_layers = model.encoder.layers
    else:
        raise ValueError("Could not find encoder layers")
    
    num_layers = len(encoder_layers)
    
    # 임베딩 — 가장 낮은 학습률
    if hasattr(model.encoder, 'embeddings'):
        emb_lr = base_lr * (lr_decay ** (num_layers + 1))
        param_groups.append({
            'params': model.encoder.embeddings.parameters(),
            'lr': emb_lr,
            'weight_decay': weight_decay,
            'name': 'embeddings'
        })
    
    # 부호기 층 — 차츰 높아지는 학습률
    for i, layer in enumerate(encoder_layers):
        layer_lr = base_lr * (lr_decay ** (num_layers - i))
        param_groups.append({
            'params': layer.parameters(),
            'lr': layer_lr,
            'weight_decay': weight_decay,
            'name': f'layer_{i}'
        })
    
    # 분류기 머리 — 가장 높은 학습률
    if hasattr(model, 'classifier'):
        param_groups.append({
            'params': model.classifier.parameters(),
            'lr': base_lr,
            'weight_decay': weight_decay,
            'name': 'classifier'
        })
    
    return param_groups

def print_lr_schedule(param_groups: List[Dict[str, Any]]):
    """학습률 일정을 찍는다."""
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

### 학습률 일정 보기 (BERT-base, 12층)

| 부품 | 층 | 학습률 (감쇠=0.9) | 학습률 (감쇠=0.95) |
|-----------|-------|----------------|-----------------|
| 임베딩 | - | 5.7e-7 | 1.1e-6 |
| 0층 | 맨 아래 | 5.1e-6 | 6.9e-6 |
| 5층 | 가운데 | 1.2e-5 | 1.3e-5 |
| 11층 | 맨 위 | 1.8e-5 | 1.9e-5 |
| 분류기 | - | 2.0e-5 | 2.0e-5 |

## 파국적 망각

미세 조정의 중심 어려움은 **파국적 망각**이다. 모델이 특정 과제에 맞추어지면서 사전 학습 지식을 잃는 것이다.

### 왜 일어나는가

1. **분포 이동**: 미세 조정 데이터가 사전 학습 데이터와 다르다
2. **기울기 간섭**: 과제의 기울기가 일반 지식을 덮어쓴다
3. **그릇 다시 나누기**: 모델이 새 과제를 위해 뉴런을 다른 용도로 쓴다

### 누그러뜨리는 방법

#### 1. 규제에 바탕한 방법 (L2-SP)

사전 학습 가중치에서 벗어나는 것에 벌점을 준다.

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \sum_i \|\theta_i - \theta_i^{\text{pre}}\|^2
$$

```python
class L2SPRegularizer:
    """
    L2-SP: (사전 학습 가중치인) 출발점 쪽으로의 L2 규제.
    """
    
    def __init__(self, model: nn.Module, pretrained_state: dict, lambda_l2sp: float = 0.01):
        self.model = model
        self.pretrained_state = {k: v.clone() for k, v in pretrained_state.items()}
        self.lambda_l2sp = lambda_l2sp
    
    def penalty(self) -> torch.Tensor:
        """사전 학습 가중치에서의 L2 거리를 셈한다."""
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

#### 2. 탄력적 가중치 굳히기 (EWC)

피셔 정보로 중요한 매개변수에 더 큰 가중치를 준다.

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \sum_i \frac{\lambda}{2} F_i (\theta_i - \theta_i^{\text{pre}})^2
$$

```python
class EWCRegularizer:
    """
    파국적 망각을 막기 위한 탄력적 가중치 굳히기.
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

#### 3. 되풀이에 바탕한 방법

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

#### 4. 구조에 바탕한 방법

사전 학습 가중치를 얼리는, 매개변수를 아끼는 방법(LoRA, 어댑터)을 쓴다. 15.6절을 보라.

## 방법 고르기

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

| 상황 | 권하는 방법 |
|----------|---------------------|
| 계산 자원이 적을 때 | 특징 뽑기나 LoRA |
| 데이터셋이 작을 때 | 특징 뽑기 → 차츰 녹이기 |
| 데이터셋이 크고 과제가 비슷할 때 | 낮은 학습률의 전면 미세 조정 |
| 데이터셋이 크고 과제가 다를 때 | 예열을 곁들인 전면 미세 조정 |
| 지식을 지켜야 할 때 | LoRA, 어댑터, EWC |
| 과제가 여럿일 때 | 어댑터 (과제마다 하나) |

## 요약

| 방법 | 학습 매개변수 | 망각 위험 | 잘 맞는 곳 |
|--------|-----------------|-----------------|----------|
| 전면 미세 조정 | 100% | 높음 | 최고 성능 |
| 특징 뽑기 | 1% 미만 | 없음 | 데이터가 적고 빠르게 되풀이할 때 |
| 차츰 녹이기 | 점진적 | 보통 | 균형 잡힌 적응 |
| 층별 학습률 | 100% | 보통 | 낮은 수준 특징 지키기 |
| L2-SP / EWC | 100% | 낮음 | 이어지는 학습 |

매개변수를 아끼는 방법(LoRA, QLoRA, 접두 조정, 어댑터)은 15.6절 효율적인 대형 언어 모델 미세 조정을 보라.

## 참고 문헌

1. Howard, J., & Ruder, S. (2018). "Universal Language Model Fine-tuning for Text Classification." ACL.
2. Kirkpatrick, J., et al. (2017). "Overcoming Catastrophic Forgetting in Neural Networks." PNAS.
3. Li, X., et al. (2018). "Explicit Inductive Bias for Transfer Learning with Convolutional Networks." ICML.
4. Peters, M., et al. (2019). "To Tune or Not to Tune? Adapting Pretrained Representations to Diverse Tasks."

## 연습문제

**연습문제 1.**
전이 학습에서 전면 미세 조정과 선형 탐침과 특징 뽑기를 견주어라.

??? success "연습문제 1 풀이"
    특징 뽑기는 사전 학습 층을 모두 얼리고 새 분류기 머리만 학습한다. 선형 탐침은 같되 (표현의 질을 재려고) 선형 분류기를 쓴다. 전면 미세 조정은 작은 학습률로 모든 매개변수를 갱신한다. 성능은 미세 조정 > 특징 뽑기 > 선형 탐침이지만, 목표 데이터셋이 작으면 미세 조정은 과적합 위험이 있다.

---

**연습문제 2.**
미세 조정의 학습률 방법을 설명하라. 사전 학습 층에 왜 더 작은 학습률을 쓰는가?

??? success "연습문제 2 풀이"
    사전 학습 층은 좋은 가중치를 지녀 조금만 바뀌어야 한다. 학습률이 크면 사전 학습된 특징이 부서진다('파국적 망각'). 층별 미세 조정은 (더 일반적인 특징인) 앞 층에 더 작은 학습률을, (더 과제에 맞는) 뒤 층에 더 큰 학습률을 쓴다. 흔히 사전 학습 층은 1e-5, 새 머리는 1e-3이다.

---

**연습문제 3.**
파이토치에서 차츰 녹이기를 구현하라 (세대마다 층 묶음 하나씩 녹인다).

??? success "연습문제 3 풀이"
    ```python
    for epoch in range(num_epochs):
        # 다음 층 묶음을 녹인다
        for param in model.layer_groups[min(epoch, len(model.layer_groups)-1)].parameters():
            param.requires_grad = True
        # 지금까지 녹인 층을 모두 써서 학습한다
        train_one_epoch(model, optimizer, dataloader)
    ```

---

**연습문제 4.**
언제 전이 학습을 쓰지 **말아야** 하는가?

??? success "연습문제 4 풀이"
    원천 도메인과 목표 도메인이 아주 다를 때(이를테면 시각 특징이 너무 다르면 ImageNet에서 의료 X선으로는 도움이 안 될 수 있다), 목표 데이터셋이 아주 클 때(맨바닥부터 학습해도 똑같이 잘 될 수 있다), 또는 사전 학습 모델의 편향이 목표 과제에 바람직하지 않을 때이다.
