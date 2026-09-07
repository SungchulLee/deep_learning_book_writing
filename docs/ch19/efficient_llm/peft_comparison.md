# 매개변수를 아끼는 곱게 다듬기(PEFT)
## 학습 목표

- 큰 말 모델에 매개변수를 아끼는 방법이 왜 꼭 필요한지 이해한다
- LoRA를 짜고 그 수학 바탕을 이해한다
- 여러 방법(LoRA, QLoRA, 앞가지 다듬기, 어댑터)을 견준다
- 제약과 요건에 따라 알맞은 방법을 고른다

## 들어가며

매개변수를 아끼는 곱게 다듬기(PEFT)는 매개변수의 아주 작은 몫만 익혀 큰 말 모델을 맞춘다. 온전한 곱게 다듬기가 엄두를 못 낼 만큼 비싸거나 아예 안 되는 큰 말 모델에 꼭 필요하다.

### 왜 매개변수를 아끼는가?

| 모델 | 매개변수 | 온전한 곱게 다듬기 기억 공간 | LoRA 기억 공간 |
|-------|------------|----------------|-------------|
| BERT-base | 1억 1000만 | 약 2 GB | 약 2 GB |
| LLaMA-7B | 70억 | 약 56 GB | 약 8 GB |
| LLaMA-65B | 650억 | 약 520 GB | 약 40 GB |

곱게 다듬기의 일반 이론(차츰 녹이기, 층마다 다른 배움 비율, 큰 잊음)은 9장 옮겨 배우기를 보라.

## LoRA(낮은 계수 맞추기)

### 수학 바탕

Instead of updating weight matrix $W \in \mathbb{R}^{d_{out} \times d_{in}}$ directly, LoRA adds a low-rank decomposition:

$$
W' = W + \Delta W = W + BA
$$

여기서:

- $B \in \mathbb{R}^{d_{out} \times r}$ (down-projection)
- $A \in \mathbb{R}^{r \times d_{in}}$ (up-projection)
- $r \ll \min(d_{in}, d_{out})$ is the rank

**Parameter savings**: Instead of $d_{in} \times d_{out}$ parameters, LoRA uses $r \times (d_{in} + d_{out})$.

For $d_{in} = d_{out} = 4096$ and $r = 8$:

- 온전히: 매개변수 1670만 개
- LoRA: 매개변수 6만 5천 개(0.4%)

### 잣수 인자

LoRA uses a scaling factor $\alpha$ to control the magnitude of updates:

$$
W' = W + \frac{\alpha}{r} BA
$$

Typically $\alpha = 2r$ or $\alpha = r$, so the effective scaling is constant regardless of rank choice.

### 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class LoRALayer(nn.Module):
    """
    LoRA(낮은 계수 맞추기) 층.
    
    이미 있는 선형 층을 감싸고 익힐 수 있는 낮은 계수 행렬을 더한다.
    본디 무게는 얼리고 A와 B만 익힌다.
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16,
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
        for param in self.original.parameters():
            param.requires_grad = False
        
        # 낮은 계수 행렬
        # A는 카이밍/허 첫자리매김으로 시작한다
        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        # B는 0으로 시작해 처음에 ΔW = BA = 0이 된다
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 본디 앞먹임(얼림)
        original_out = self.original(x)
        
        # LoRA 앞먹임
        # x @ A @ B * 잣수
        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        
        return original_out + lora_out
    
    def merge_weights(self) -> nn.Linear:
        """
        미룸을 위해 LoRA 무게를 본디 층에 합친다.
        
        군더더기 없는 여느 nn.Linear를 돌려준다.
        """
        merged = nn.Linear(
            self.original.in_features,
            self.original.out_features,
            bias=self.original.bias is not None
        )
        
        # W' = W + (α/r) * B @ A
        merged.weight.data = self.original.weight.data + \
            (self.lora_A @ self.lora_B).T * self.scaling
        
        if self.original.bias is not None:
            merged.bias.data = self.original.bias.data
        
        return merged


class LoRAConfig:
    """LoRA 맞추기의 자리매김."""
    
    def __init__(
        self,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.0,
        target_modules: list = None
    ):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        # 붙박이: 눈길 쏘기에 적용한다
        self.target_modules = target_modules or ['q_proj', 'v_proj']


def apply_lora(
    model: nn.Module,
    config: LoRAConfig
) -> nn.Module:
    """
    모델에서 정해진 모듈에 LoRA를 적용한다.
    
    인수:
        model: 맞출 모델
        config: LoRA 자리매김
        
    반환값:
        LoRA 층을 적용한 모델
    """
    for name, module in model.named_modules():
        # 이 모듈에 LoRA를 적용할지 살핀다
        if any(target in name for target in config.target_modules):
            if isinstance(module, nn.Linear):
                # 어버이 모듈과 속성 이름을 얻는다
                parts = name.rsplit('.', 1)
                if len(parts) == 2:
                    parent_name, child_name = parts
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                    child_name = name
                
                # LoRA 층으로 바꾼다
                lora_layer = LoRALayer(
                    module,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout
                )
                setattr(parent, child_name, lora_layer)
                print(f"Applied LoRA to {name}")
    
    return model


def get_lora_params(model: nn.Module) -> list:
    """가장 좋게 하기에 쓸 LoRA 매개변수만 얻는다."""
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_params.append(param)
    return lora_params


def count_parameters(model: nn.Module) -> dict:
    """익힐 수 있는 매개변수와 전체 매개변수를 센다."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        'trainable': trainable,
        'total': total,
        'percentage': 100 * trainable / total
    }
```

### 어느 층을 목표로 할까?

연구에 따르면 층마다 영향이 다르다:

| 목표 단원 | 좋음 | 매개변수 |
|----------------|---------|------------|
| 물음만(q) | 좋음 | 가장 적음 |
| 물음 + 값(q, v) | 더 좋음 | 적음 |
| 모든 눈길(q, k, v, o) | 가장 좋음 | 가운데 |
| 눈길 + 다층 퍼셉트론 | 이득이 미미 | 많음 |

**권하는 바**: 좋음과 매개변수의 맞바꿈이 가장 나은 `q_proj`와 `v_proj`로 시작하라.

### 계수 고르기

| 계수 | 매개변수 | 쓰임새 |
|------|------------|----------|
| 4 | 아주 적음 | 단순한 일, 기억 공간이 아주 빠듯할 때 |
| 8 | 적음 | 붙박이, 대부분의 일에 좋다 |
| 16 | 가운데 | 복잡한 일, 큰 자료 뭉치 |
| 64 | 많음 | 온전한 곱게 다듬기에 가까운 좋음 |
| 256 이상 | 높음 | 온전한 곱게 다듬기에 다가감 |

## QLoRA(양자화한 LoRA)

QLoRA(Dettmers 외, 2023)는 다음을 아울러 일반 하드웨어에서도 아주 큰 모델을 곱게 다듬게 한다:

1. 바탕 모델의 **4비트 NormalFloat(NF4) 양자화**
2. 온전한 정밀도(bfloat16)의 **LoRA 어댑터**
3. 양자화 상수의 **겹 양자화**
4. 기억 공간 살림을 위한 **쪽 나눈 가장 좋게 하개**

```python
import torch
import torch.nn as nn
from typing import Tuple


class NF4Linear(nn.Module):
    """
    간추린 NF4(4비트 정규 실수) 양자화 선형 층.
    
    참고: 이는 본보기 짜기이다. 실전에서는 bitsandbytes 같은
    가장 좋게 다듬은 꾸러미를 써야 한다.
    """
    
    # NF4 양자화 층계(N(0,1) 분포에 맞춰 고름)
    NF4_LEVELS = torch.tensor([
        -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
        0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0
    ])
    
    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor = None,
        block_size: int = 64
    ):
        super().__init__()
        
        self.out_features, self.in_features = weight.shape
        self.block_size = block_size
        self.bias = nn.Parameter(bias) if bias is not None else None
        
        # 무게를 양자화한다
        self.register_buffer('weight_quantized', self._quantize(weight))
        self.register_buffer('scales', self._compute_scales(weight))
    
    def _compute_scales(self, weight: torch.Tensor) -> torch.Tensor:
        """덩이마다의 잣수를 셈한다."""
        # 덩이 꼴로 바꾼다
        weight_flat = weight.reshape(-1)
        num_blocks = (weight_flat.numel() + self.block_size - 1) // self.block_size
        
        # 필요하면 덧댄다
        padded_size = num_blocks * self.block_size
        if weight_flat.numel() < padded_size:
            weight_flat = F.pad(weight_flat, (0, padded_size - weight_flat.numel()))
        
        weight_blocks = weight_flat.reshape(num_blocks, self.block_size)
        
        # 잣수 = 덩이마다의 최대 절댓값
        scales = weight_blocks.abs().max(dim=1).values
        return scales
    
    def _quantize(self, weight: torch.Tensor) -> torch.Tensor:
        """무게를 4비트 NF4로 양자화한다."""
        # 간추림: NF4_LEVELS의 번호만 담는다
        # 실제 짜기라면 바이트마다 값 둘을 담을 것이다
        weight_flat = weight.reshape(-1)
        
        # 덩이 잣수로 고르게 한다
        scales = self._compute_scales(weight)
        num_blocks = scales.numel()
        
        # 꼴을 바꾸고 고르게 한다
        padded_size = num_blocks * self.block_size
        if weight_flat.numel() < padded_size:
            weight_flat = F.pad(weight_flat, (0, padded_size - weight_flat.numel()))
        
        weight_blocks = weight_flat.reshape(num_blocks, self.block_size)
        weight_normalized = weight_blocks / (scales.unsqueeze(1) + 1e-8)
        
        # 가장 가까운 NF4 층계를 찾는다
        distances = (weight_normalized.unsqueeze(-1) - self.NF4_LEVELS.to(weight.device)).abs()
        indices = distances.argmin(dim=-1)
        
        return indices.to(torch.uint8)
    
    def _dequantize(self) -> torch.Tensor:
        """앞먹임을 위해 무게의 양자화를 되돌린다."""
        # 번호로 NF4 값을 얻는다
        weight_nf4 = self.NF4_LEVELS.to(self.weight_quantized.device)[self.weight_quantized.long()]
        
        # 잣수를 되돌린다
        weight_scaled = weight_nf4 * self.scales.unsqueeze(1)
        
        # 본디 꼴로 되돌린다
        weight = weight_scaled.reshape(-1)[:self.out_features * self.in_features]
        return weight.reshape(self.out_features, self.in_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 그때그때 양자화를 되돌린다
        weight = self._dequantize()
        output = F.linear(x, weight, self.bias)
        return output


class QLoRALayer(nn.Module):
    """
    QLoRA: 바탕 무게를 양자화한 LoRA.
    
    바탕 모델은 4비트로, LoRA 맞춤개는 bfloat16으로.
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.0
    ):
        super().__init__()
        
        # 본디 무게를 NF4로 양자화한다
        self.quantized = NF4Linear(
            original_layer.weight.data,
            original_layer.bias.data if original_layer.bias is not None else None
        )
        
        # LoRA는 온전한 정밀도로(bfloat16)
        self.rank = rank
        self.scaling = alpha / rank
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        self.lora_A = nn.Parameter(
            torch.randn(in_features, rank, dtype=torch.bfloat16) * 0.01
        )
        self.lora_B = nn.Parameter(
            torch.zeros(rank, out_features, dtype=torch.bfloat16)
        )
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 양자화한 앞먹임(그때그때 되돌린다)
        base_out = self.quantized(x)
        
        # bfloat16으로 하는 LoRA 앞먹임
        x_bf16 = x.to(torch.bfloat16)
        lora_out = (self.dropout(x_bf16) @ self.lora_A @ self.lora_B * self.scaling)
        
        return base_out + lora_out.to(base_out.dtype)
```

### QLoRA의 기억 공간 아끼기

| 모델 | 온전한 곱게 다듬기 | LoRA(16비트) | QLoRA(4비트) |
|-------|---------|---------------|---------------|
| 70억 | 56 GB | 약 16 GB | 약 6 GB |
| 130억 | 104 GB | 약 28 GB | 약 10 GB |
| 650억 | 520 GB | 약 130 GB | 약 40 GB |

## 앞가지 다듬기

익힐 수 있는 이어진 시킴말을 들임 앞에 붙여, 모델 무게를 바꾸지 않고 눈길을 고친다:

```python
class PrefixTuning(nn.Module):
    """
    앞가지 다듬기: 들임 앞에 붙는 이어진 시킴말을 배운다.
    
    따로 떨어진 시킴말 토막 대신, 눈길 층마다 열쇠-값 짝 앞에 붙는
    이어진 묻힘을 배운다.
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        prefix_length: int = 20,
        hidden_dim: int = 512
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.prefix_length = prefix_length
        
        # 전체 크기: 층 수 * 2(열쇠 + 값) * 머리 수 * 머리 차원
        prefix_size = num_layers * 2 * num_heads * head_dim
        
        # 배울 수 있는 앞가지 묻힘
        self.prefix_embedding = nn.Embedding(prefix_length, hidden_dim)
        
        # 온전한 앞가지 크기로 쏘는 여러 층 인식개(안정을 위해)
        self.prefix_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, prefix_size)
        )
        
        self.num_heads = num_heads
        self.head_dim = head_dim
    
    def forward(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        앞가지 열쇠-값 짝을 만든다.
        
        반환값:
            (앞가지 열쇠, 앞가지 값)의 짝. 저마다 꼴은
            [묶음, 층 수, 머리 수, 앞가지 길이, 머리 차원]
        """
        # 앞가지 번호를 얻는다
        prefix_ids = torch.arange(self.prefix_length, device=self.prefix_embedding.weight.device)
        prefix_ids = prefix_ids.unsqueeze(0).expand(batch_size, -1)
        
        # 묻고 쏜다
        prefix = self.prefix_embedding(prefix_ids)  # [묶음, 앞가지 길이, 숨은]
        prefix = self.prefix_mlp(prefix)  # [묶음, 앞가지 길이, 앞가지 크기]
        
        # [묶음, 앞가지 길이, 층 수, 2, 머리 수, 머리 차원] 꼴로 바꾼다
        prefix = prefix.view(
            batch_size,
            self.prefix_length,
            self.num_layers,
            2,  # 열쇠와 값
            self.num_heads,
            self.head_dim
        )
        
        # [묶음, 층 수, 머리 수, 앞가지 길이, 머리 차원] 꼴로 다시 늘어놓는다
        prefix = prefix.permute(0, 2, 4, 1, 5, 3)  # 층과 머리 차원을 옮긴다
        
        prefix_keys = prefix[..., 0]    # [묶음, 층, 머리, 앞가지 길이, 머리 차원]
        prefix_values = prefix[..., 1]  # [묶음, 층, 머리, 앞가지 길이, 머리 차원]
        
        return prefix_keys, prefix_values


class PrefixAttention(nn.Module):
    """
    앞가지 다듬기를 쓰도록 고친 눈길 층.
    """
    
    def __init__(self, original_attention: nn.Module, prefix_length: int):
        super().__init__()
        self.attention = original_attention
        self.prefix_length = prefix_length
        
        # 본디 눈길을 얼린다
        for param in self.attention.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        prefix_key: torch.Tensor,
        prefix_value: torch.Tensor,
        attention_mask: torch.Tensor = None
    ):
        """
        열쇠와 값 앞에 앞가지를 붙여 앞먹임한다.
        """
        # 숨은 상태에서 Q, K, V를 셈한다
        # (짜기는 눈길 얼개에 따라 다르다)
        
        # K와 V 앞에 앞가지를 붙인다
        # K = [앞가지 열쇠; K]
        # V = [앞가지 값; V]
        
        # 앞가지에 맞춰 눈길 가림막을 넓힌다(앞가지는 늘 본다)
        if attention_mask is not None:
            prefix_mask = torch.ones(
                attention_mask.shape[0], self.prefix_length,
                device=attention_mask.device
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        # 넓힌 K, V로 눈길을 셈한다
        # ...
        
        pass  # 온전한 짜기는 모델 얼개에 따라 다르다
```

## 어댑터 층

얼린 변환기 층 사이에 작은 익힐 수 있는 병목 단원을 끼운다:

```python
class Adapter(nn.Module):
    """
    맞춤개 모듈: 내림 쏘기 → 비선형 → 올림 쏘기 + 남는 이음.
    
    눈길이나 앞먹임 그물 밑층 뒤에 끼운다.
    """
    
    def __init__(
        self,
        hidden_size: int,
        bottleneck_size: int = 64,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        self.down_proj = nn.Linear(hidden_size, bottleneck_size)
        self.up_proj = nn.Linear(bottleneck_size, hidden_size)
        
        if activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.Tanh()
        
        # 안정된 시작을 위해 up_proj를 거의 0으로 첫자리매김한다
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 남는 이음을 곁들인 병목
        return x + self.up_proj(self.act(self.down_proj(x)))


class AdapterTransformerBlock(nn.Module):
    """
    맞춤개를 끼운 변환기 덩이.
    
    구조:
        x → 눈길 → 맞춤개 → 층 고르게 맞추기 → 앞먹임 그물 → 맞춤개 → 층 고르게 맞추기 → 내놓기
    """
    
    def __init__(
        self,
        original_block: nn.Module,
        bottleneck_size: int = 64
    ):
        super().__init__()
        
        self.original = original_block
        
        # 본디 덩이를 얼린다
        for param in original_block.parameters():
            param.requires_grad = False
        
        # 덩이에서 숨은 크기를 알아낸다
        hidden_size = self._get_hidden_size(original_block)
        
        # 눈길과 앞먹임 그물 뒤에 맞춤개를 더한다
        self.adapter_attn = Adapter(hidden_size, bottleneck_size)
        self.adapter_ffn = Adapter(hidden_size, bottleneck_size)
    
    def _get_hidden_size(self, block: nn.Module) -> int:
        """덩이 얼개에서 숨은 크기를 알아내려 한다."""
        for name, module in block.named_modules():
            if isinstance(module, nn.Linear):
                return module.out_features
        raise ValueError("Could not infer hidden size")
    
    def forward(self, hidden_states: torch.Tensor, **kwargs):
        # 이는 간추린 판이며, 실제 짜기는
        # 쓰는 변환기 얼개에 따라 다르다
        
        # 어텐션
        attn_output = self.original.attention(hidden_states, **kwargs)
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]
        attn_output = self.adapter_attn(attn_output)  # 맞춤개를 적용한다
        
        hidden_states = hidden_states + attn_output
        hidden_states = self.original.ln_1(hidden_states)
        
        # 순전파 신경망
        ffn_output = self.original.mlp(hidden_states)
        ffn_output = self.adapter_ffn(ffn_output)  # 맞춤개를 적용한다
        
        hidden_states = hidden_states + ffn_output
        hidden_states = self.original.ln_2(hidden_states)
        
        return hidden_states


def add_adapters(model: nn.Module, bottleneck_size: int = 64) -> nn.Module:
    """모든 변환기 덩이에 맞춤개를 더한다."""
    # 짜기는 모델 얼개에 따라 다르다
    # 이는 두루 쓰는 방식의 본이다
    pass
```

### 어댑터 변종

| 변종 | 놓을 자리 | 비고 |
|---------|-----------|-------|
| 이어진 어댑터 | 아래층 뒤 | 처음 꾸밈(Houlsby 외) |
| 나란한 어댑터 | 아래층과 나란히 | 어떤 일에는 더 낫다 |
| AdapterFusion | 어댑터 여럿을 아우름 | 여러 일 배우기 |

## 두루 살피는 견줌

### 매개변수 아끼기

| 방법 | 익힐 매개변수 | 기억 공간 덧짐 | 미룸 덧짐 |
|--------|-----------------|-----------------|-------------------|
| 온전한 곱게 다듬기 | 100% | 많음 | 없음 |
| LoRA | 0.1~1% | 적음 | 없음(어울린 뒤) |
| QLoRA | 0.1~1% | 아주 적음 | 조금(양자화) |
| 앞가지 다듬기 | 0.1% | 적음 | 조금(차례가 길어짐) |
| 어댑터 | 1~5% | 적음 | 조금(덧층) |

### 좋음과 효율의 맞바꿈

```
Quality
  ↑
  │     ┌─────────────────┐
  │     │ Full Fine-tuning │
  │     └────────┬────────┘
  │              │
  │     ┌────────┴────────┐
  │     │      LoRA       │
  │     └────────┬────────┘
  │              │
  │     ┌────────┴────────┐
  │     │     QLoRA       │
  │     └────────┬────────┘
  │              │
  │     ┌────────┴────────┐
  │     │    Adapters     │
  │     └────────┬────────┘
  │              │
  │     ┌────────┴────────┐
  │     │  Prefix Tuning  │
  │     └─────────────────┘
  └─────────────────────────────→ Efficiency
```

### 언제 무엇을 쓸까

| 장면 | 권하는 방법 |
|----------|-------------------|
| 일반 GPU(24GB 이하), 70억 모델 | QLoRA |
| 실전, 덧짐이 0이어야 함 | LoRA(어울린 것) |
| 여러 일, 모델 하나 | 어댑터 |
| 매개변수가 아주 빠듯 | 앞가지 다듬기 |
| 가장 좋은 품질, 넉넉한 자원 | 온전한 곱게 다듬기 |
| 650억 이상 모델, 하드웨어가 빠듯 | QLoRA |

## 방법 아우르기

필요에 따라 방법을 아우를 수 있다:

```python
# 기억 공간 효율을 가장 크게 하려는 QLoRA + 기울기 되짚기
# 눈길에는 LoRA + 여러 층 인식개에는 맞춤개
# 더 큰 담이를 위한 앞가지 다듬기 + LoRA
```

## 요약

| 방법 | 핵심 생각 | 알맞은 곳 |
|--------|----------|----------|
| **LoRA** | 낮은 계수 무게 고침 | 두루 쓰기, 실전 |
| **QLoRA** | 4비트 바탕 + LoRA | 큰 모델, 빠듯한 기억 공간 |
| **앞가지 다듬기** | 배울 수 있는 부드러운 시킴말 | 만들어 내기 일 |
| **어댑터** | 병목 단원 | 여러 일, 단원별 |

## 참고 문헌

1. Hu, E., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." ICLR.
2. Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS.
3. Li, X., & Liang, P. (2021). "Prefix-Tuning: Optimizing Continuous Prompts for Generation." ACL.
4. Houlsby, N., et al. (2019). "Parameter-Efficient Transfer Learning for NLP." ICML.
5. Pfeiffer, J., et al. (2020). "AdapterHub: A Framework for Adapting Transformers." EMNLP.

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
