# 파라미터 효율적 파인튜닝 비교

PEFT는 전체 웨이트를 갱신하는 대신 일부 파라메터나 작은 추가 모듈로 모델을 적응시킨다. LoRA·양자화 기반 LoRA·학습 prefix·병목 어댑터는 서로 다른 계산 경로를 사용한다. 학습 가능한 파라메터가 적다는 사실과 전체 학습 메모리·추론 속도·품질은 구분해야 한다.

## 1. LoRA의 갱신과 적용

열벡터 표기에서 W'=W₀+(α/r)BA이고 A는 다운, B는 업 프로젝션이다. 학습 웨이트는 r(d_in+d_out)개지만 동결 기반도 저장한다. 코드의 행벡터 구현은 A·B의 모양이 전치 관례이며 병합 시 곱을 전치한다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

class LoRALayer(nn.Module):
    """
    LoRA(낮은 계수 맞추기) 레이어.
    
    이미 있는 선형 레이어를 감싸고 익힐 수 있는 낮은 계수 행렬을 더한다.
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
        # 본디 피드포워드(얼림)
        original_out = self.original(x)
        
        # LoRA 피드포워드
        # x @ A @ B * 잣수
        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        
        return original_out + lora_out
    
    def merge_weights(self) -> nn.Linear:
        """
        미룸을 위해 LoRA 무게를 본디 레이어에 합친다.
        
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
        # 붙박이: 어텐션 쏘기에 적용한다
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
        LoRA 레이어를 적용한 모델
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
                
                # LoRA 레이어로 바꾼다
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
    """가장 좋게 하기에 쓸 LoRA 파라메터만 얻는다."""
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_params.append(param)
    return lora_params

def count_parameters(model: nn.Module) -> dict:
    """익힐 수 있는 파라메터와 전체 파라메터를 센다."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        'trainable': trainable,
        'total': total,
        'percentage': 100 * trainable / total
    }
```

이 적용 함수는 이름 부분 문자열이 맞는 Linear만 감싸고 그 레이어만 동결한다. 모델의 나머지 requires_grad를 끄는 기능은 없으며 모델별 이름·결합 QKV·장치·dtype을 확인해야 한다. α/r가 같아도 랭크와 초기화가 달라지면 갱신·최적화가 자동으로 동등해지는 것은 아니다.

merge_weights는 새 Linear를 반환하지만 원 모델의 래퍼를 자동 교체하지 않는다. dropout이 꺼진 선형 갱신에서 동등성을 확인해야 하며 bias.data는 원 텐서와 저장소를 공유할 수 있다. 실제 병합 상태·저장과 dtype을 검토해야 한다.

## 2. QLoRA와 양자화 저장

[QLoRA](https://arxiv.org/abs/2305.14314)는 동결 기반의 저비트 저장·계산 경로와 LoRA 학습을 결합한다. NF4·double quantization·paged optimizer는 각각 다른 메모리 요소를 다룬다. BF16은 16비트 부동소수점이며 FP32와 같은 ‘전체 정밀도’라는 뜻이 아니다.

```python
import torch
import torch.nn as nn
from typing import Tuple

class NF4Linear(nn.Module):
    """
    간추린 NF4(4비트 정규 실수) 양자화 선형 레이어.
    
    참고: 이는 본보기 짜기이다. 실전에서는 bitsandbytes 같은
    가장 좋게 다듬은 꾸러미를 써야 한다.
    """
    
    # NF4 양자화 레이어계(N(0,1) 분포에 맞춰 고름)
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
        
        # 가장 가까운 NF4 레이어계를 찾는다
        distances = (weight_normalized.unsqueeze(-1) - self.NF4_LEVELS.to(weight.device)).abs()
        indices = distances.argmin(dim=-1)
        
        return indices.to(torch.uint8)
    
    def _dequantize(self) -> torch.Tensor:
        """피드포워드를 위해 무게의 양자화를 되돌린다."""
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
        # 양자화한 피드포워드(그때그때 되돌린다)
        base_out = self.quantized(x)
        
        # bfloat16으로 하는 LoRA 피드포워드
        x_bf16 = x.to(torch.bfloat16)
        lora_out = (self.dropout(x_bf16) @ self.lora_A @ self.lora_B * self.scaling)
        
        return base_out + lora_out.to(base_out.dtype)
```

현재 NF4Linear는 16개 수준 중 가까운 인덱스를 고르지만 두 4비트 값을 한 바이트에 pack하지 않는다. 인덱스를 uint8 하나씩 저장하므로 실제 웨이트 코드 저장은 토큰당이 아니라 **웨이트당 8비트**이며 scale 저장도 추가된다. 이 구현을 곧바로 4비트 메모리 감소의 실증이라고 할 수 없다.

forward마다 전체 웨이트를 복원하며 double quantization·paged optimizer·최적화된 양자화 커널은 없다. bias는 새 trainable Parameter이고 LoRA 행렬은 BF16 CPU에 생성되므로 기반·입력과 장치·dtype을 맞춰야 한다. 양자화 저장 비트 수와 전체 학습 peak memory를 구분한다. 이 부품만으로 논문의 전체 QLoRA 학습이 재현되지는 않는다.

## 3. Prefix의 학습 값과 연결

```python
class PrefixTuning(nn.Module):
    """
    앞가지 다듬기: 들임 앞에 붙는 이어진 시킴말을 배운다.
    
    따로 떨어진 시킴말 토막 대신, 어텐션 레이어마다 열쇠-값 짝 앞에 붙는
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
        
        # 전체 크기: 레이어 수 * 2(열쇠 + 값) * 머리 수 * 머리 차원
        prefix_size = num_layers * 2 * num_heads * head_dim
        
        # 배울 수 있는 앞가지 묻힘
        self.prefix_embedding = nn.Embedding(prefix_length, hidden_dim)
        
        # 온전한 앞가지 크기로 쏘는 여러 레이어 인식개(안정을 위해)
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
            [배치, 레이어 수, 머리 수, 앞가지 길이, 머리 차원]
        """
        # 앞가지 번호를 얻는다
        prefix_ids = torch.arange(self.prefix_length, device=self.prefix_embedding.weight.device)
        prefix_ids = prefix_ids.unsqueeze(0).expand(batch_size, -1)
        
        # 묻고 쏜다
        prefix = self.prefix_embedding(prefix_ids)  # [배치, 앞가지 길이, 숨은]
        prefix = self.prefix_mlp(prefix)  # [배치, 앞가지 길이, 앞가지 크기]
        
        # [배치, 앞가지 길이, 레이어 수, 2, 머리 수, 머리 차원] 꼴로 바꾼다
        prefix = prefix.view(
            batch_size,
            self.prefix_length,
            self.num_layers,
            2,  # 열쇠와 값
            self.num_heads,
            self.head_dim
        )
        
        # [배치, 레이어 수, 머리 수, 앞가지 길이, 머리 차원] 꼴로 다시 늘어놓는다
        prefix = prefix.permute(0, 2, 4, 1, 5, 3)  # 레이어와 머리 차원을 옮긴다
        
        prefix_keys = prefix[..., 0]    # [배치, 레이어, 머리, 앞가지 길이, 머리 차원]
        prefix_values = prefix[..., 1]  # [배치, 레이어, 머리, 앞가지 길이, 머리 차원]
        
        return prefix_keys, prefix_values

class PrefixAttention(nn.Module):
    """
    앞가지 다듬기를 쓰도록 고친 어텐션 레이어.
    """
    
    def __init__(self, original_attention: nn.Module, prefix_length: int):
        super().__init__()
        self.attention = original_attention
        self.prefix_length = prefix_length
        
        # 본디 어텐션을 얼린다
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
        열쇠와 값 앞에 앞가지를 붙여 피드포워드한다.
        """
        # 숨은 상태에서 Q, K, V를 셈한다
        # (짜기는 어텐션 구조에 따라 다르다)
        
        # K와 V 앞에 앞가지를 붙인다
        # K = [앞가지 열쇠; K]
        # V = [앞가지 값; V]
        
        # 앞가지에 맞춰 어텐션 가림막을 넓힌다(앞가지는 늘 본다)
        if attention_mask is not None:
            prefix_mask = torch.ones(
                attention_mask.shape[0], self.prefix_length,
                device=attention_mask.device
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        # 넓힌 K, V로 어텐션을 셈한다
        # ...
        
        pass  # 온전한 짜기는 모델 구조에 따라 다르다
```

PrefixTuning은 MLP로 레이어별 K·V를 만든다. 이는 입력 임베딩에만 벡터를 붙이는 prompt tuning과 다르다. 학습 파라메터에는 embedding·MLP도 포함하므로 생성한 KV 값의 수만 세면 안 된다.

이 페이지의 PrefixAttention.forward는 마스크 일부를 구성한 뒤 pass로 끝나므로 None을 반환한다. 기존 어텐션에 prefix를 실제로 연결하는 완성 구현이 아니다. 기반 모델의 cache·head·마스크·위치 계약은 별도로 필요하다.

## 4. 어댑터의 구조 가정

```python
class Adapter(nn.Module):
    """
    맞춤개 모듈: 내림 쏘기 → 비선형 → 올림 쏘기 + 남는 이음.
    
    어텐션이나 피드포워드 넷트웍 밑레이어 뒤에 끼운다.
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
        x → 어텐션 → 맞춤개 → 레이어 고르게 맞추기 → 피드포워드 넷트웍 → 맞춤개 → 레이어 고르게 맞추기 → 내놓기
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
        
        # 어텐션과 피드포워드 넷트웍 뒤에 맞춤개를 더한다
        self.adapter_attn = Adapter(hidden_size, bottleneck_size)
        self.adapter_ffn = Adapter(hidden_size, bottleneck_size)
    
    def _get_hidden_size(self, block: nn.Module) -> int:
        """덩이 구조에서 숨은 크기를 알아내려 한다."""
        for name, module in block.named_modules():
            if isinstance(module, nn.Linear):
                return module.out_features
        raise ValueError("Could not infer hidden size")
    
    def forward(self, hidden_states: torch.Tensor, **kwargs):
        # 이는 간추린 판이며, 실제 짜기는
        # 쓰는 변환기 구조에 따라 다르다
        
        # 어텐션
        attn_output = self.original.attention(hidden_states, **kwargs)
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]
        attn_output = self.adapter_attn(attn_output)  # 맞춤개를 적용한다
        
        hidden_states = hidden_states + attn_output
        hidden_states = self.original.ln_1(hidden_states)
        
        # 순전파 뉴럴 넷트웍
        ffn_output = self.original.mlp(hidden_states)
        ffn_output = self.adapter_ffn(ffn_output)  # 맞춤개를 적용한다
        
        hidden_states = hidden_states + ffn_output
        hidden_states = self.original.ln_2(hidden_states)
        
        return hidden_states

def add_adapters(model: nn.Module, bottleneck_size: int = 64) -> nn.Module:
    """모든 변환기 덩이에 맞춤개를 더한다."""
    # 짜기는 모델 구조에 따라 다르다
    # 이는 두루 쓰는 방식의 본이다
    pass
```

이 Adapter는 업 프로젝션 웨이트·바이어스를 0으로 초기화하므로 시작할 때 정확히 x를 반환한다. 작은 난수로 초기화한 다른 예제와 다르다. 비선형 어댑터는 일반적으로 기반 선형 행렬 하나에 합칠 수 없다.

AdapterTransformerBlock은 original.attention·ln_1·mlp·ln_2가 존재한다고 가정하고 고정 순서의 잔차·정규화를 수행한다. 실제 모델의 속성·반환값·pre/post-norm 계약과 다르면 호환되지 않는다. 첫 Linear의 out_features만으로 hidden size를 추정하면 결합 QKV·중간 FFN 폭을 잘못 선택할 수 있다. add_adapters도 pass여서 실제 삽입을 구현하지 않는다.

## 5. 파라메터·메모리·추론 비교

| 방법 | 주로 학습하는 것 | 기반 외 추론 계산 |
|------|-------------------|-------------------|
| LoRA | 적용한 레이어의 저랭크 갱신 | 선형 병합 조건이 맞으면 제거 가능 |
| QLoRA | 양자화 동결 기반의 LoRA | 양자화·복원·커널과 병합 폴리시에 의존 |
| Prefix tuning | 레이어별 KV 또는 생성 모듈 | prefix 참조와 추가 KV |
| 어댑터 | 추가 비선형 병목 | 추가 레이어 계산 |

전체 메모리에는 기반 웨이트·학습 경사·옵티마이저 상태·활성화·임시 버퍼·cache와 분산 폴리시를 포함한다. 예를 들어 70억 BF16 기반만 약 14GB이므로 양자화나 offload 없이 전체 LoRA 학습이 항상 8GB라는 표는 성립하지 않는다. 비트 수·batch·길이·학습 가능한 부분과 측정 환경을 명시해야 한다.

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

기존 품질 도식은 개념 그림이며 전체 파인튜닝>LoRA>QLoRA>어댑터>prefix라는 보편적 실측 순위가 아니다. 방법·과제·자료·튜닝 예산에 따라 순위가 달라지고 파라미터 절감률도 고정 비율이 아니다.

## 6. 선택과 조합

작은 추가 체크포인트·과제 교체·배포 병합·메모리 제한 등 목표를 정하고 동일 품질 기준 아래 측정한다. 특정 GPU 용량과 파라미터 수만으로 어떤 학습도 반드시 가능하다고 보장하지 않는다. [LoRA](lora.md), [어댑터](adapters.md), [prefix tuning](prefix_tuning.md)의 구현 계약도 함께 확인한다.

```python
# 기억 공간 효율을 가장 크게 하려는 QLoRA + 그래디언트 되짚기
# 어텐션에는 LoRA + 여러 레이어 인식개에는 맞춤개
# 더 큰 담이를 위한 앞가지 다듬기 + LoRA
```

위 블록은 조합 아이디어를 주석으로 나열한 것이며 실제 통합 코드가 아니다. 함께 쓴다고 효과가 단순히 합산되거나 메모리가 항상 최소화되지는 않는다. 동결·모드·체크포인트·경사·dtype과 각 방법의 독립적인 효과를 검증해야 한다. 이 편집에서는 학습·양자화·벤치마크를 실행하지 않았다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
적은 학습 파라메터와 적은 총메모리가 같은 뜻이 아닌 이유를 설명하라.

</div>

??? success "풀이"

    동결 기반 웨이트도 저장하며 입력·중간 활성화·경사 전달·임시 버퍼가 남는다. 학습 부분의 옵티마이저·경사는 줄 수 있지만 dtype·길이·batch·분할 폴리시에 따라 전체 peak memory가 달라진다. 기반만 14GB인 BF16 70억 모델을 양자화 없이 총 8GB에 학습한다고 단정할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
NF4Linear의 uint8 인덱스가 4비트 packing과 다른 점을 설명하라.

</div>

??? success "풀이"

    16가지 수준은 4비트로 표현할 수 있지만 현재는 인덱스 하나당 uint8 한 바이트를 저장한다. 실제 4비트 저장은 두 인덱스를 한 바이트에 넣는 등의 packing과 복원 구현이 필요하다. scale·bias·활성화와 매번 전체 웨이트 복원 비용도 더해진다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 페이지의 prefix·어댑터 코드 중 미구현 부분과 모델별 가정을 설명하라.

</div>

??? success "풀이"

    PrefixAttention.forward와 add_adapters가 pass여서 실제 연결이 없다. 어댑터 블록은 특정 속성·정규화 순서를 가정하고 첫 Linear 폭으로 hidden size를 추정한다. 모델의 cache·레이어 반환·mask·위치·차원을 확인하지 않으면 범용 삽입 함수가 되지 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
어떤 PEFT 방법이 가장 좋은지 비교할 실험의 조건을 제시하라.

</div>

??? success "풀이"

    같은 기반·분할·입출력 길이·품질 기준과 튜닝 예산을 사용한다. 학습 파라메터·peak memory·시간·추론 지연·체크포인트/과제 교체 비용을 별도 측정한다. 단일 도식이나 고정 추천표를 일반적인 품질 순위로 사용하지 않는다.



## 정리하며

PEFT 방법은 갱신 대상과 실제 연결·저장·추론 경로가 다르다. 부분 구현·미구현 연결과 양자화 packing을 명확히 구분하고 같은 품질·환경에서 전체 비용을 평가해야 한다.

**참고 문헌**

1. Hu, E., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." ICLR.
2. Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS.
3. Li, X., & Liang, P. (2021). "Prefix-Tuning: Optimizing Continuous Prompts for Generation." ACL.
4. Houlsby, N., et al. (2019). "Parameter-Efficient Transfer Learning for NLP." ICML.
5. Pfeiffer, J., et al. (2020). "AdapterHub: A Framework for Adapting Transformers." EMNLP.
