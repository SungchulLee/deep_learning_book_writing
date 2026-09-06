# 어댑터 층
## 학습 목표

- 어댑터 얼개와 꾸밈 원리를 이해한다
- 변환기 모델에 어댑터를 짠다
- 어댑터 변종(이어진, 나란한, AdapterFusion)을 견준다
- 어댑터의 병목 크기와 놓을 자리를 정한다

## 들어가며

어댑터(Houlsby 외, 2019)는 얼린 미리 익힌 모델의 층 사이에 끼우는 작은 익힐 수 있는 단원이다. 어댑터마다 내리쬐기, 비선형, 올리쬐기와 잔차 이음으로 이루어진다. 이러면 바탕 모델을 얼린 채로 일에 맞출 수 있다.

## 구조

### 기본 어댑터 단원

어댑터는 병목 얼개를 따른다:

$$
\text{Adapter}(x) = x + f(xW_{down})W_{up}
$$

여기서:

- $W_{down} \in \mathbb{R}^{d \times r}$ projects to bottleneck dimension $r$
- $f$은 비선형이다(GELU, ReLU)
- $W_{up} \in \mathbb{R}^{r \times d}$ projects back to original dimension
- 잔차 이음이 처음 몸짓을 미리 익힌 모델과 맞게 한다

### 매개변수의 수

차원이 $d$이고 병목이 $r$인 모델에서:

$$
\text{Adapter params} = 2 \times d \times r + r + d
$$

층마다 눈길 뒤와 앞먹임 그물 뒤에 어댑터를 두면:

$$
\text{Total params} = L \times 2 \times (2dr + r + d)
$$

**보기**: $r=64$인 BERT-base($L=12$, $d=768$):

- Per adapter: $2 \times 768 \times 64 + 64 + 768 = 99,136$
- Total: $12 \times 2 \times 99,136 = 2,379,264$ (2.2% of 110M)

## 구현

### 고갱이 어댑터 단원

```python
import torch
import torch.nn as nn
from typing import Optional, List


class Adapter(nn.Module):
    """
    맞춤개 모듈: 내림 쏘기 → 깨어남 → 올림 쏘기 + 남는 이음.
    
    인수:
        input_dim: 들임/내놓기 차원(모델의 숨은 크기)
        bottleneck_dim: 병목 차원(보통 64~256)
        activation: 깨어남 함수
        init_scale: 내놓기 쏘기 첫자리매김의 잣수
    """
    
    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int = 64,
        activation: str = 'gelu',
        init_scale: float = 1e-3,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        
        # 내림 쏘기
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        
        # 활성화
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.SiLU()
        
        # 올림 쏘기
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 남는 이음이 안정되도록 첫자리매김한다
        self._init_weights(init_scale)
    
    def _init_weights(self, scale: float):
        """남는 이음이 안정되도록 무게를 첫자리매김한다."""
        nn.init.normal_(self.down_proj.weight, std=0.02)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.normal_(self.up_proj.weight, std=scale)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """잔차 이음을 쓴 앞먹임."""
        down = self.down_proj(x)
        activated = self.activation(down)
        activated = self.dropout(activated)
        up = self.up_proj(activated)
        return x + up
    
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ScaledAdapter(Adapter):
    """배울 수 있는 잣수를 곁들인 맞춤개."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down = self.down_proj(x)
        activated = self.activation(down)
        activated = self.dropout(activated)
        up = self.up_proj(activated)
        return x + self.scale * up
```

### 이어진 어댑터(처음 꾸밈)

```python
class SerialAdapterTransformerLayer(nn.Module):
    """
    밑층마다 뒤에 맞춤개를 잇달아 둔 변환기 층.
    
    구조:
        x → 눈길 → 맞춤개 → 더하고 고르게 → 앞먹임 그물 → 맞춤개 → 더하고 고르게
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        bottleneck_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 맞춤개
        self.adapter_attn = Adapter(d_model, bottleneck_dim)
        self.adapter_ffn = Adapter(d_model, bottleneck_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 눈길 + 맞춤개
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=attention_mask)
        attn_out = self.adapter_attn(attn_out)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 앞먹임 그물 + 맞춤개
        ffn_out = self.ffn(x)
        ffn_out = self.adapter_ffn(ffn_out)
        x = self.norm2(x + ffn_out)
        
        return x
```

### 나란한 어댑터

```python
class ParallelAdapterTransformerLayer(nn.Module):
    """
    나란한 맞춤개를 둔 변환기 층.
    
    맞춤개가 밑층 뒤가 아니라 밑층과 나란히 돈다.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        bottleneck_dim: int = 64,
        dropout: float = 0.1,
        adapter_scale: float = 1.0
    ):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.adapter_attn = Adapter(d_model, bottleneck_dim)
        self.adapter_ffn = Adapter(d_model, bottleneck_dim)
        self.adapter_scale = adapter_scale
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 나란히: 맞춤개와 눈길이 모두 x를 받는다
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=attention_mask)
        adapter_out = self.adapter_attn(x) - x  # 남는 이음을 뺀다
        x = self.norm1(x + self.dropout(attn_out) + self.adapter_scale * adapter_out)
        
        # 나란히: 맞춤개와 앞먹임 그물이 모두 x를 받는다
        ffn_out = self.ffn(x)
        adapter_out = self.adapter_ffn(x) - x
        x = self.norm2(x + ffn_out + self.adapter_scale * adapter_out)
        
        return x
```

## AdapterFusion: 여러 일 배우기

```python
class AdapterFusion(nn.Module):
    """
    일마다 다른 맞춤개 여럿을 아우르는 법을 배운다.
    
    맞춤개를 따로따로 익힌 뒤, 녹여 붙이기가 목표 일에
    가장 좋은 무게 매기기를 배운다.
    """
    
    def __init__(
        self,
        adapters: nn.ModuleList,
        input_dim: int
    ):
        super().__init__()
        
        self.adapters = adapters
        self.num_adapters = len(adapters)
        
        # 낱낱의 맞춤개를 얼린다
        for adapter in adapters:
            for param in adapter.parameters():
                param.requires_grad = False
        
        # 녹여 붙이는 눈길
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        
        # 맞춤개가 내놓는 것을 얻는다
        adapter_outputs = []
        for adapter in self.adapters:
            adapter_out = adapter(x) - x  # 남는 이음 없는 몫
            adapter_outputs.append(adapter_out)
        
        # 쌓기: [묶음, 차례, 맞춤개 수, 숨은]
        adapter_stack = torch.stack(adapter_outputs, dim=2)
        
        # 맞춤개에 대한 눈길
        query = self.query(x).unsqueeze(2)
        keys = self.key(adapter_stack)
        
        scores = (query * keys).sum(dim=-1) / (hidden_dim ** 0.5)
        scores = scores / self.temperature
        weights = torch.softmax(scores, dim=-1)
        
        # 무게를 준 아우름
        values = self.value(adapter_stack)
        fused = (weights.unsqueeze(-1) * values).sum(dim=2)
        
        return x + fused
```

## 여러 일 어댑터 모델

```python
class MultiTaskAdapterModel(nn.Module):
    """일마다 바꿔 끼울 수 있는 맞춤개를 갖춘 모델."""
    
    def __init__(
        self,
        base_model: nn.Module,
        task_names: List[str],
        hidden_dim: int,
        bottleneck_dim: int = 64
    ):
        super().__init__()
        
        self.base_model = base_model
        self.task_names = task_names
        
        # 바탕을 얼린다
        for param in base_model.parameters():
            param.requires_grad = False
        
        # 일마다 맞춤개 하나
        self.adapters = nn.ModuleDict({
            name: Adapter(hidden_dim, bottleneck_dim)
            for name in task_names
        })
        
        self.current_task: Optional[str] = None
    
    def set_task(self, task_name: str):
        """어떤 일의 맞춤개를 켠다."""
        if task_name not in self.task_names:
            raise ValueError(f"Unknown task: {task_name}")
        
        self.current_task = task_name
        
        # 지금 일의 맞춤개만 익힐 수 있다
        for name, adapter in self.adapters.items():
            for param in adapter.parameters():
                param.requires_grad = (name == task_name)
    
    def forward(self, x: torch.Tensor, task: Optional[str] = None):
        task = task or self.current_task
        if task is None:
            raise ValueError("No task specified")
        
        hidden = self.base_model(x)
        return self.adapters[task](hidden)
```

## 웃매개변수 길잡이

### 병목 차원

| 병목 | BERT-base 대비 % | 쓰임새 |
|------------|----------------|----------|
| 8 | 0.2% | 극단적인 효율 |
| 32 | 0.9% | 자원이 적을 때 |
| **64** | 1.8% | **붙박이** |
| 128 | 3.5% | 복잡한 일 |
| 256 | 7.0% | 담는 힘이 클 때 |

### 놓을 자리 전략

| 전략 | 층당 어댑터 | 권하는 바 |
|----------|----------------|----------------|
| 눈길에만 | 1 | 가장 적게 |
| 앞먹임 그물에만 | 1 | 흔히 넉넉하다 |
| **둘 다** | 2 | **권함** |

### 배움 비율

- 어댑터: 1e-4 ~ 1e-3
- 온전한 곱게 다듬기: 1e-5 ~ 5e-5

## 비교

| 방법 | 매개변수 | 미룸 덧짐 | 여러 일 | 단원별 |
|--------|--------|-------------------|------------|---------|
| 어댑터 | 1~5% | 있다 | 쉽다 | 그렇다 |
| LoRA | 0.1~1% | 없다(어울림) | 더 어렵다 | 그렇다 |
| 앞가지 다듬기 | 0.1~1% | 있다 | 쉽다 | 그렇다 |

**어댑터를 쓸 때:**

- 일을 바꿔 가며 여러 일을 배울 때
- 단원별로 펼칠 때
- 미룸 덧짐을 받아들일 수 있을 때

## 학습 예제

```python
def train_with_adapters(
    base_model: nn.Module,
    train_loader,
    hidden_dim: int,
    bottleneck_dim: int = 64,
    num_epochs: int = 5,
    lr: float = 1e-4,
    device: str = 'cuda'
):
    # 바탕 모델을 얼린다
    for param in base_model.parameters():
        param.requires_grad = False
    
    # 맞춤개를 더한다(간추림 - 실제로 붙이는 법은 모델마다 다르다)
    adapter = Adapter(hidden_dim, bottleneck_dim).to(device)
    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 바탕 + 맞춤개를 지나 앞먹임한다
            with torch.no_grad():
                hidden = base_model(**batch).last_hidden_state
            output = adapter(hidden)
            
            loss = compute_loss(output, batch['labels'])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")
    
    return adapter


def save_adapter(adapter: nn.Module, path: str):
    """맞춤개 무게를 갈무리한다."""
    torch.save(adapter.state_dict(), path)


def load_adapter(adapter: nn.Module, path: str):
    """맞춤개 무게를 불러온다."""
    adapter.load_state_dict(torch.load(path))
```

## 요약

| 갈래 | 자세히 |
|--------|---------|
| **얼개** | 내리쬐기 → 깨어남 → 올리쬐기 + 잔차 |
| **매개변수** | 모델의 1~5% |
| **놓을 자리** | 눈길 뒤 그리고/또는 앞먹임 그물 뒤 |
| **핵심 이점** | 단원별이고 여러 일이 쉽다 |
| **맞바꿈** | 미룸 덧짐 |

## 참고 문헌

1. Houlsby, N., et al. (2019). "Parameter-Efficient Transfer Learning for NLP." ICML.
2. Pfeiffer, J., et al. (2020). "AdapterHub: A Framework for Adapting Transformers." EMNLP.
3. Pfeiffer, J., et al. (2021). "AdapterFusion: Non-Destructive Task Composition." EACL.
4. He, J., et al. (2022). "Towards a Unified View of Parameter-Efficient Transfer Learning." ICLR.

## 연습문제

**연습문제 1.**
LoRA의 고갱이 생각을 밝혀라. 모델의 좋음을 지키면서 익힐 매개변수를 어떻게 줄이는가?

??? success "연습문제 1 풀이"
    LoRA (Low-Rank Adaptation) freezes the pretrained weight matrix $W_0 \in \mathbb{R}^{d \times k}$ and adds a low-rank update $\Delta W = BA$ where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and $r \ll \min(d, k)$. The forward pass becomes $h = (W_0 + BA)x$. Only $A$ and $B$ are trained, reducing trainable parameters from $dk$ to $r(d+k)$. For a typical layer with $d = k = 4096$ and $r = 8$, this is a $256\times$ reduction. The key insight: the weight updates during fine-tuning have low intrinsic rank, so a low-rank decomposition captures the essential adaptation.

---

**연습문제 2.**
LoRA, 앞가지 다듬기, 어댑터 층을 견주어라. 기억 공간, 미룸 빠르기, 일의 성능에서 맞바꿈은 무엇인가?

??? success "연습문제 2 풀이"
    | 방법 | 익힐 매개변수 | 미룸 덧짐 | 기억 공간 | 여러 일 |
    |--------|-----------------|-------------------|--------|------------|
    | **LoRA** | $\sim$0.1-1% | None (merge weights) | Low | Swap $A, B$ matrices |
    | **Prefix Tuning** | $\sim$0.1% | Slight (extra tokens) | Low | Swap prefix vectors |
    | **Adapters** | $\sim$1-5% | Moderate (extra layers) | Medium | Swap adapter modules |

    $BA$을 $W_0$에 어울릴 수 있으므로 LoRA는 미룸 덧짐이 0이다. 앞가지 다듬기는 맥락 창을 먹는 가상 토막을 더한다. 어댑터는 늦음을 늘리는 병목 층을 더한다. 셋 다 대부분의 일에서 온전한 곱게 다듬기에 가까운 성능을 내며, 단순하고 효율적이어서 LoRA가 가장 널리 쓰인다.

---

**연습문제 3.**
매개변수 700억 모델을 온전히 곱게 다듬는 것이 대부분의 조직에 왜 실전에 맞지 않는가? 기억 공간 요구량을 수로 나타내어라.

??? success "연습문제 3 풀이"
    A 70B model in fp16 requires $70 \times 10^9 \times 2$ bytes = 140 GB just for weights. Fine-tuning additionally requires: optimizer states (Adam stores 2 states per parameter: 280 GB in fp32), gradients (140 GB in fp16), and activations for backpropagation. Total: $\sim$700+ GB of GPU memory. Even with gradient checkpointing and mixed precision, this requires 8+ A100 80GB GPUs. LoRA reduces trainable parameters to $\sim$70M, cutting optimizer states and gradient memory by $1000\times$, making fine-tuning feasible on 1-2 GPUs.

---

**연습문제 4.**
양자화를 헤아린 LoRA(QLoRA)란 무엇이며 큰 말 모델 곱게 다듬기를 누구나 하게 만드는 데 왜 뜻깊은가?

??? success "연습문제 4 풀이"
    QLoRA combines 4-bit quantization of the base model with LoRA adapters in fp16/bf16. The base weights $W_0$ are stored in 4-bit NormalFloat format ($\sim$0.5 bytes per parameter), reducing memory for a 70B model from 140 GB to $\sim$35 GB. LoRA adapters remain in higher precision for stable training. Additional innovations: double quantization (quantizing the quantization constants) and paged optimizers (using CPU memory for optimizer state spikes). This enables fine-tuning a 65B model on a single 48GB GPU (A6000), making LLM adaptation accessible to researchers and small organizations without expensive multi-GPU clusters.
