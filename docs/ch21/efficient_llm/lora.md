# LoRA: 낮은 계수 맞추기

---

## 1. 학습 목표

- 낮은 계수 맞추기의 수학 바탕을 이해한다
- LoRA를 맨바닥부터 짜고 변환기 모델에 쓴다
- LoRA의 웃매개변수(계수, 알파, 목표 단원)를 정한다
- 덧짐 없는 미룸을 위해 LoRA 무게를 어울린다

---

## 2. 들어가며

LoRA(낮은 계수 맞추기)는 미리 익힌 모델 무게를 얼리고 층마다 익힐 수 있는 낮은 계수 쪼갬 행렬을 끼워 넣는, 매개변수를 아끼는 곱게 다듬기 방법이다. 익힐 매개변수를 1만분의 1로 줄이면서 온전한 곱게 다듬기에 맞먹는 성능을 낸다.

---

## 3. 수학적 바탕

### 핵심 생각

무게 행렬 $W_0 \in \mathbb{R}^{d_{out} \times d_{in}}$을 곧바로 고치는 대신 LoRA은 그 고침을 낮은 계수 쪼갬으로 옭아맨다.

$$
W = W_0 + \Delta W = W_0 + BA
$$

여기서:

- $B \in \mathbb{R}^{d_{out} \times r}$(내림 되비춤)
- $A \in \mathbb{R}^{r \times d_{in}}$(오름 되비춤)
- $r \ll \min(d_{in}, d_{out})$ is the rank

### 낮은 계수가 통하는 까닭

연구에 따르면 곱게 다듬는 동안의 무게 고침은 "본디 계수"가 낮다. 곧 고침의 실효 차원이 온 매개변수 공간보다 훨씬 작다. LoRA는 고침을 낮은 계수 행렬로 드러내어 매개변수로 나타냄으로써 이를 써먹는다.

### 순전파

들임 $x$에 대해:

$$
h = W_0 x + \Delta W x = W_0 x + BAx
$$

본디 무게 $W_0$은 얼리고 $A$과 $B$만 익힌다.

### 잣수 인자

LoRA는 고침의 크기를 다스리려 잣수 인자를 쓴다:

$$
h = W_0 x + \frac{\alpha}{r} BAx
$$

여기서 $\alpha$은 상수다(흔히 $\alpha = 2r$이나 $\alpha = r$). 이렇게 잣대를 잡으면 다음이 보장된다.

- $\Delta W$의 크기가 계수를 어떻게 고르든 달라지지 않는다
- 웃매개변수 옮기기: 같은 $\alpha$이 여러 계수에서 두루 통한다
- 든든한 익히기 흐름

### 매개변수 아끼기

차수가 $d_{in} \times d_{out}$인 선형 층에서는

| 방법 | 매개변수 |
|--------|------------|
| 온전한 곱게 다듬기 | $d_{in} \times d_{out}$ |
| LoRA (rank $r$) | $r \times (d_{in} + d_{out})$ |

**Example**: For $d_{in} = d_{out} = 4096$, $r = 8$:

- 온전히: 매개변수 16,777,216개
- LoRA: 매개변수 65,536개(0.39%)

---

## 4. 구현

### 고갱이 LoRA 층

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Dict, Any

class LoRALayer(nn.Module):
    """
    이미 있는 선형 층을 감싸는 LoRA 층.
    
    짜기: h = W₀x + (α/r)BAx
    
    인수:
        original_layer: 맞출 선형 층
        rank: 낮은 계수 쪼개기의 계수
        alpha: 잣수(보통 alpha = 2*rank)
        dropout: LoRA 길에서의 떨구기 확률
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.original = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # 본디 무게를 얼린다
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
        
        # LoRA 행렬
        # A: 들임 특징 -> 계수(내림 쏘기)
        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        # B: 계수 -> 내놓기 특징(올림 쏘기)
        self.lora_B = nn.Parameter(torch.empty(rank, out_features))
        
        # 초기화한다
        self._init_weights()
        
        # 고를 수 있는 떨구기
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 좇기 위해
        self.merged = False
    
    def _init_weights(self):
        """
        LoRA 무게를 첫자리매김한다.
        
        A: 카이밍 고른 분포(nn.Linear 붙박이와 같다)
        B: 0(그래서 처음 ΔW = BA = 0)
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        앞먹임: W₀x + (α/r)BAx
        """
        if self.merged:
            # 무게를 합쳤으면 본디 층만 쓴다
            return self.original(x)
        
        # 본디 길(얼림)
        original_output = self.original(x)
        
        # LoRA 길: x @ A @ B * 잣수
        lora_output = self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        
        return original_output + lora_output
    
    def merge_weights(self):
        """
        미룸을 위해 LoRA 무게를 본디 층에 합친다.
        
        W' = W₀ + (α/r)BA^T
        
        합친 뒤에는 앞먹임에 군더더기가 없다.
        """
        if self.merged:
            return
        
        # ΔW = (α/r) * A @ B를 셈한 뒤 무게 꼴에 맞게 옮겨 놓는다
        delta_w = (self.lora_A @ self.lora_B * self.scaling).T
        self.original.weight.data += delta_w
        self.merged = True
    
    def unmerge_weights(self):
        """
        LoRA 무게 합치기를 되돌린다.
        
        익히기를 이어 가거나 맞춤개를 바꿀 때 쓸모 있다.
        """
        if not self.merged:
            return
        
        delta_w = (self.lora_A @ self.lora_B * self.scaling).T
        self.original.weight.data -= delta_w
        self.merged = False
    
    def get_delta_weight(self) -> torch.Tensor:
        """LoRA 무게 바뀜 ΔW를 돌려준다."""
        return (self.lora_A @ self.lora_B * self.scaling).T
    
    @property
    def num_parameters(self) -> int:
        """익힐 수 있는 LoRA 매개변수의 수."""
        return self.lora_A.numel() + self.lora_B.numel()

class LoRALinear(nn.Module):
    """
    홀로 서는 LoRA 선형 층(이미 있는 층을 감싸지 않는다).
    
    LoRA를 처음부터 품은 새 모델을 만들 때 쓸모 있다.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        bias: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank
        
        # 얼린 바탕 무게
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False) if bias else None
        
        # 익힐 수 있는 LoRA
        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        self.lora_B = nn.Parameter(torch.empty(rank, out_features))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora = self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        return base + lora
```

### 모델에 LoRA 쓰기

```python
from dataclasses import dataclass, field
from typing import Set

@dataclass
class LoRAConfig:
    """LoRA 맞추기의 자리매김."""
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    # LoRA를 적용할 모듈 이름
    target_modules: Set[str] = field(default_factory=lambda: {'q_proj', 'v_proj'})
    # target_modules에 맞더라도 빼는 모듈
    exclude_modules: Set[str] = field(default_factory=set)

def apply_lora_to_model(
    model: nn.Module,
    config: LoRAConfig
) -> nn.Module:
    """
    모델에서 맞는 모듈 모두에 LoRA를 적용한다.
    
    인수:
        model: 맞출 모델
        config: LoRA 자리매김
        
    반환값:
        LoRA 층을 갖춘 모델(본디 무게는 얼림)
    """
    # 바꿀 모듈을 모은다(되풀이 도중에 고칠 수 없다)
    replacements = []
    
    for name, module in model.named_modules():
        # 이 모듈에 LoRA를 붙일지 살핀다
        should_apply = (
            isinstance(module, nn.Linear) and
            any(target in name for target in config.target_modules) and
            not any(exclude in name for exclude in config.exclude_modules)
        )
        
        if should_apply:
            replacements.append((name, module))
    
    # 바꾸기를 적용한다
    for name, module in replacements:
        # 어버이 모듈로 찾아간다
        parts = name.rsplit('.', 1)
        if len(parts) == 2:
            parent_name, child_name = parts
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            child_name = name
        
        # LoRA 층을 만들어 앉힌다
        lora_layer = LoRALayer(
            module,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout
        )
        setattr(parent, child_name, lora_layer)
        
        print(f"Applied LoRA to {name}: {module.in_features} -> {module.out_features}, rank={config.rank}")
    
    print(f"\nTotal LoRA layers: {len(replacements)}")
    return model

def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    모델 상태 사전에서 LoRA 매개변수만 뽑는다.
    
    맞춤개 무게만 갈무리하거나 불러올 때 쓸모 있다.
    """
    return {
        name: param for name, param in model.state_dict().items()
        if 'lora_' in name
    }

def load_lora_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]):
    """LoRA 매개변수를 모델에 불러온다."""
    model_state = model.state_dict()
    
    for name, param in state_dict.items():
        if name in model_state:
            model_state[name].copy_(param)
        else:
            print(f"Warning: {name} not found in model")
    
    model.load_state_dict(model_state, strict=False)
```

### 익히기 도구

```python
def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """가장 좋게 하개에 쓸 LoRA 매개변수만 얻는다."""
    params = []
    for name, param in model.named_parameters():
        if 'lora_' in name:
            params.append(param)
    return params

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """익힐 수 있는 매개변수와 전체 매개변수를 센다."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    lora = sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n)
    
    return {
        'trainable': trainable,
        'total': total,
        'lora': lora,
        'trainable_percent': 100.0 * trainable / total,
        'lora_percent': 100.0 * lora / total
    }

def freeze_non_lora(model: nn.Module):
    """LoRA만 빼고 모든 매개변수를 얼린다."""
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False

def merge_lora_weights(model: nn.Module):
    """미룸을 위해 모델의 모든 LoRA 무게를 합친다."""
    for module in model.modules():
        if isinstance(module, LoRALayer):
            module.merge_weights()

def unmerge_lora_weights(model: nn.Module):
    """모델의 모든 LoRA 무게 합치기를 되돌린다."""
    for module in model.modules():
        if isinstance(module, LoRALayer):
            module.unmerge_weights()
```

---

## 5. 웃매개변수 길잡이

### 계수 고르기

| 계수 | 매개변수 | 좋음 | 쓰임새 |
|------|------------|---------|----------|
| 1~4 | 아주 적음 | 낮음 | 아주 단순한 일, 극단적인 눌러 담기 |
| 8 | 적음 | 좋음 | **붙박이**, 대부분의 일 |
| 16 | 가운데 | 더 좋음 | 복잡한 일, 큰 자료 뭉치 |
| 32~64 | 많음 | 온전한 곱게 다듬기에 가까움 | 담는 힘이 많이 필요한 일 |
| 128 이상 | 높음 | 온전한 곱게 다듬기와 비슷 | 온전한 곱게 다듬기에 다가갈 때 |

**어림 규칙**: 계수 8로 시작하고 덜 맞으면 늘리고, 지나치게 맞거나 기억 공간이 빠듯하면 줄인다.

### 알파 고르기

흔한 전략:

- $\alpha = r$: 조심스러운 잣대
- $\alpha = 2r$: **기본값**, 고르다
- $\alpha = 4r$: 세게 고치기

견줌 $\alpha/r$이 LoRA 매개변수의 실제 배움 빠르기를 정한다. 견줌이 클수록 크게 고친다.

### 목표 단원 고르기

변환기 모델에서는:

| 목표 | 단원 | 좋음 | 효율 |
|--------|---------|---------|------------|
| 가장 적게 | q_proj | 바탕 | 가장 높음 |
| **보통** | q_proj, v_proj | 좋음 | 높음 |
| 넓힘 | q_proj, k_proj, v_proj, o_proj | 더 좋음 | 가운데 |
| 온 눈길 | 모든 눈길 + 내놓음 | 가장 좋음 | 낮음 |
| 전부 | 눈길 + 다층 퍼셉트론 | 이득이 미미 | 가장 낮음 |

```python
# 흔한 자리매김
LORA_CONFIGS = {
    'minimal': LoRAConfig(rank=8, target_modules={'q_proj'}),
    'standard': LoRAConfig(rank=8, target_modules={'q_proj', 'v_proj'}),
    'extended': LoRAConfig(rank=8, target_modules={'q_proj', 'k_proj', 'v_proj', 'o_proj'}),
    'full': LoRAConfig(rank=16, target_modules={'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}),
}
```

### 배움 비율

LoRA는 온전한 곱게 다듬기보다 큰 배움 비율을 쓰는 것이 보통이다:

| 방법 | 흔한 배움 비율 |
|--------|------------|
| 온전한 곱게 다듬기 | 1e-5 ~ 5e-5 |
| LoRA | 1e-4 ~ 3e-4 |

---

## 6. 더 깊은 주제

### 여러 일에 쓰는 LoRA

서로 다른 어댑터를 갈무리하고 불러온다:

```python
class LoRAManager:
    """바탕 모델 하나에 붙은 여러 LoRA 맞춤개를 다스린다."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.adapters: Dict[str, Dict[str, torch.Tensor]] = {}
        self.current_adapter: Optional[str] = None
    
    def save_adapter(self, name: str):
        """지금 LoRA 무게를 이름 붙인 맞춤개로 갈무리한다."""
        self.adapters[name] = get_lora_state_dict(self.model)
    
    def load_adapter(self, name: str):
        """갈무리한 맞춤개를 불러온다."""
        if name not in self.adapters:
            raise ValueError(f"Adapter '{name}' not found")
        
        # 합쳐 있으면 지금 것을 되돌린다
        unmerge_lora_weights(self.model)
        
        # 새 맞춤개를 불러온다
        load_lora_state_dict(self.model, self.adapters[name])
        self.current_adapter = name
    
    def delete_adapter(self, name: str):
        """갈무리한 맞춤개를 지운다."""
        if name in self.adapters:
            del self.adapters[name]
```

### LoRA+(나아진 배움 비율)

LoRA+는 행렬 A와 B에 서로 다른 배움 비율을 쓴다:

```python
def get_lora_plus_params(model: nn.Module, lr: float, lr_ratio: float = 16.0):
    """
    A와 B에 다른 배움 빠르기를 주는 LoRA+ 매개변수 무리.
    
    B는 0으로 시작하므로 배움 빠르기를 더 크게 준다.
    """
    params_A = []
    params_B = []
    
    for name, param in model.named_parameters():
        if 'lora_A' in name:
            params_A.append(param)
        elif 'lora_B' in name:
            params_B.append(param)
    
    return [
        {'params': params_A, 'lr': lr},
        {'params': params_B, 'lr': lr * lr_ratio}
    ]
```

### 계수를 든든히 한 LoRA(rsLoRA)

높은 계수에서 더 든든하도록 잣수를 맞춘다:

$$
h = W_0 x + \frac{\alpha}{\sqrt{r}} BAx
$$

```python
class rsLoRALayer(LoRALayer):
    """제곱근 잣수를 쓴 계수 안정 LoRA."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 잣수에 r 대신 sqrt(r)을 쓴다
        self.scaling = self.alpha / math.sqrt(self.rank)
```

---

## 7. 완전한 학습 예제

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

def train_lora(
    model_name: str,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    config: LoRAConfig,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    device: str = 'cuda'
):
    """온전한 LoRA 익히기 물길."""
    
    # 모델을 불러온다
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = apply_lora_to_model(model, config)
    model = model.to(device)
    
    # 가장 좋게 하개를 마련한다(LoRA 매개변수만)
    lora_params = get_lora_parameters(model)
    optimizer = torch.optim.AdamW(lora_params, lr=learning_rate)
    
    # 매개변수 개수 세기
    param_counts = count_parameters(model)
    print(f"Trainable: {param_counts['trainable']:,} ({param_counts['trainable_percent']:.2f}%)")
    print(f"LoRA: {param_counts['lora']:,} ({param_counts['lora_percent']:.2f}%)")
    
    # 학습 루프
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # 평가
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                eval_loss += outputs.loss.item()
        
        print(f"  Eval Loss: {eval_loss / len(eval_dataloader):.4f}")
        model.train()
    
    # LoRA 무게를 갈무리한다
    lora_state = get_lora_state_dict(model)
    torch.save(lora_state, 'lora_weights.pt')
    
    # 미룸을 위해 합친다
    merge_lora_weights(model)
    
    return model

# 사용 예
if __name__ == "__main__":
    config = LoRAConfig(
        rank=8,
        alpha=16,
        dropout=0.05,
        target_modules={'q_proj', 'v_proj'}
    )
    
    # model = train_lora("meta-llama/Llama-2-7b-hf", train_dl, eval_dl, config)
```

---

## 연습문제

**연습문제 1.**
LoRA의 고갱이 생각을 밝혀라. 모델의 좋음을 지키면서 익힐 매개변수를 어떻게 줄이는가?

??? success "연습문제 1 풀이"
    LoRA(낮은 계수 맞추기)는 미리 익힌 무게 행렬 $W_0 \in \mathbb{R}^{d \times k}$을 얼리고 낮은 계수 고침 $\Delta W = BA$을 더한다. 여기서 $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$이고 $r \ll \min(d, k)$이다. 앞으로 걸음은 $h = (W_0 + BA)x$이 된다. $A$과 $B$만 익히므로 익힐 매개변수가 $dk$개에서 $r(d+k)$개로 준다. $d = k = 4096$이고 $r = 8$인 흔한 층이면 $256\times$ 줄어든다. 고갱이 눈썰미는 곱게 다듬는 동안의 무게 고침이 타고난 계수가 낮다는 것이며, 그래서 낮은 계수 쪼갬이 종요로운 맞춤을 담아낸다.

---

**연습문제 2.**
LoRA, 앞가지 다듬기, 어댑터 층을 견주어라. 기억 공간, 미룸 빠르기, 일의 성능에서 맞바꿈은 무엇인가?

??? success "연습문제 2 풀이"
    | 방법 | 익힐 매개변수 | 미룸 덧짐 | 기억 공간 | 여러 일 |
    |--------|-----------------|-------------------|--------|------------|
    | **LoRA** | $\sim$0.1~1% | 없음(무게를 어울린다) | 낮음 | $A, B$ 행렬을 갈아 끼운다 |
    | **머리말 맞추기** | $\sim$0.1% | 조금 있음(덧붙은 토막) | 낮음 | 머리말 벡터를 갈아 끼운다 |
    | **맞춤개** | $\sim$1~5% | 어느 정도 있음(덧붙은 층) | 보통 | 맞춤개 단원을 갈아 끼운다 |

    $BA$을 $W_0$에 어울릴 수 있으므로 LoRA는 미룸 덧짐이 0이다. 앞가지 다듬기는 맥락 창을 먹는 가상 토막을 더한다. 어댑터는 늦음을 늘리는 병목 층을 더한다. 셋 다 대부분의 일에서 온전한 곱게 다듬기에 가까운 성능을 내며, 단순하고 효율적이어서 LoRA가 가장 널리 쓰인다.

---

**연습문제 3.**
매개변수 700억 모델을 온전히 곱게 다듬는 것이 대부분의 조직에 왜 실전에 맞지 않는가? 기억 공간 요구량을 수로 나타내어라.

??? success "연습문제 3 풀이"
    fp16으로 된 700억 매개변수 모델은 무게만 해도 $70 \times 10^9 \times 2$바이트 = 140GB가 든다. 곱게 다듬으려면 여기에 더해 가장 좋게 하개 상태(Adam은 매개변수마다 상태 2개를 담으므로 fp32으로 280GB), 기울기(fp16으로 140GB), 되짚기용 살림 값이 든다. 모두 GPU 기억 자리가 $\sim$700GB 넘게 든다. 기울기 되짚음 저장과 섞인 촘촘함을 써도 A100 80GB GPU가 8장 넘게 든다. LoRA은 익힐 매개변수를 $\sim$7000만 개로 줄여 가장 좋게 하개 상태와 기울기 기억 자리를 $1000\times$ 줄이므로 GPU 한두 장으로도 곱게 다듬을 수 있다.

---

**연습문제 4.**
양자화를 헤아린 LoRA(QLoRA)란 무엇이며 큰 말 모델 곱게 다듬기를 누구나 하게 만드는 데 왜 뜻깊은가?

??? success "연습문제 4 풀이"
    QLoRA은 밑 모델을 4비트로 수 줄이고 LoRA 맞춤개는 fp16/bf16으로 두어 둘을 아우른다. 밑 무게 $W_0$은 4비트 NormalFloat 꼴로 담기므로(매개변수마다 $\sim$0.5바이트) 700억 모델의 기억 자리가 140GB에서 $\sim$35GB으로 준다. LoRA 맞춤개는 익힘이 든든하도록 더 촘촘한 꼴로 남는다. 여기에 겹 수 줄이기(수 줄이기 상수를 다시 수 줄이기)와 쪽 넘김 가장 좋게 하개(가장 좋게 하개 상태가 치솟을 때 CPU 기억 자리를 씀)라는 새로움이 더해진다. 그래서 48GB GPU(A6000) 한 장으로 650억 모델을 곱게 다듬을 수 있고, 값비싼 여러 GPU 무리 없이도 연구자와 작은 조직이 큰 말 모델을 맞출 수 있게 된다.

## 정리하며

| 살필 점 | 권하는 바 |
|--------|----------------|
| **계수** | 8로 시작해 일의 복잡도에 따라 맞춘다 |
| **알파** | 붙박이로 계수의 2배를 쓴다 |
| **목표** | 대부분의 일에 q_proj + v_proj |
| **배움 비율** | 1e-4 ~ 3e-4(온전한 곱게 다듬기보다 크다) |
| **떨구기** | 벌주기로 0.05~0.1 |

**참고 문헌**

1. Hu, E., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
2. Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs."
3. Hayou, S., et al. (2024). "LoRA+: Efficient Low Rank Adaptation of Large Models."
4. Kalajdzievski, D. (2023). "Rank-Stabilized LoRA: Unlocking the Potential of LoRA Fine-Tuning."
