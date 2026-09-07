# 큰 말 모델의 양자화
## 들어가며

양자화는 무게와 깨어남을 정밀도가 낮은 자료 갈래로 나타내어 모델 크기를 줄이고 미룸을 빠르게 한다. 큰 말 모델에서는 일반 하드웨어에 펼치고 내놓기 값을 줄이는 데 결정적이다.

## 근본

### 정밀도 꼴

| 꼴 | 비트 | 범위 | 쓰임새 |
|--------|------|-------|----------|
| FP32 | 32 | ±3.4e38 | 익히기(바탕) |
| FP16 | 16 | ±65504 | 섞인 정밀도 익히기 |
| BF16 | 16 | ±3.4e38 | 익히기(범위가 더 넓음) |
| INT8 | 8 | -128 ~ 127 | 미룸 |
| INT4 | 4 | -8 ~ 7 | 큰 말 모델 미룸 |
| FP8 | 8 | 여러 가지 | 떠오르는 표준 |

### 기억 공간 아끼기

매개변수 70억 모델에서:

| 정밀도 | 크기 | 기억 공간 |
|-----------|------|--------|
| FP32 | 28 GB | 약 32 GB |
| FP16/BF16 | 14 GB | 약 16 GB |
| INT8 | 7 GB | 약 8 GB |
| INT4 | 3.5 GB | 약 4 GB |

## 양자화 이론

### 선형 양자화

뜬소수점 값을 정수에 대응시킨다:

$$
x_q = \text{round}\left(\frac{x}{\Delta}\right) + z
$$

$$
\hat{x} = \Delta(x_q - z)
$$

여기서:

- $\Delta$ = scale factor
- $z$ = 영점
- $x_q$ = 양자화한 값

### 잣수와 영점 셈하기

**대칭 양자화**($z = 0$):

$$
\Delta = \frac{\max(|x|)}{2^{b-1} - 1}
$$

**대칭이 아닌 양자화**:

$$
\Delta = \frac{x_{max} - x_{min}}{2^b - 1}, \quad z = \text{round}\left(-\frac{x_{min}}{\Delta}\right)
$$

### 양자화 어긋남

양자화에서 오는 어긋남:

$$
\epsilon = x - \hat{x} = x - \Delta \cdot \text{round}\left(\frac{x}{\Delta}\right)
$$

칸 안의 값이 고르게 퍼져 있을 때:

$$
\mathbb{E}[\epsilon^2] = \frac{\Delta^2}{12}
$$

## 무게 양자화 방법

### 익힌 뒤 양자화(PTQ)

다시 익히지 않고 미리 익힌 모델을 양자화한다.

```python
import torch
import torch.nn as nn
from typing import Tuple, Optional


def compute_scale_zero(
    tensor: torch.Tensor,
    bits: int = 8,
    symmetric: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    양자화 매개변수를 셈한다.
    """
    if symmetric:
        max_val = tensor.abs().max()
        scale = max_val / (2 ** (bits - 1) - 1)
        zero_point = torch.zeros(1, device=tensor.device)
    else:
        min_val, max_val = tensor.min(), tensor.max()
        scale = (max_val - min_val) / (2 ** bits - 1)
        zero_point = torch.round(-min_val / scale)
    
    return scale, zero_point


def quantize_tensor(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    bits: int = 8
) -> torch.Tensor:
    """
    텐서를 붙박이 소수점 표현으로 양자화한다.
    """
    q_min = -(2 ** (bits - 1))
    q_max = 2 ** (bits - 1) - 1
    
    quantized = torch.round(tensor / scale) + zero_point
    quantized = torch.clamp(quantized, q_min, q_max)
    
    return quantized.to(torch.int8)


def dequantize_tensor(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor
) -> torch.Tensor:
    """
    다시 뜬소수점으로 되돌린다.
    """
    return scale * (quantized.float() - zero_point)


class QuantizedLinear(nn.Module):
    """
    무게를 양자화한 선형 층.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 8,
        group_size: Optional[int] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size or in_features
        
        # 묶음별 양자화의 묶음 수
        self.num_groups = in_features // self.group_size
        
        # 양자화한 무게(int8로 담음)
        self.register_buffer(
            'weight_quantized',
            torch.zeros(out_features, in_features, dtype=torch.int8)
        )
        
        # 묶음마다의 잣수
        self.register_buffer(
            'scale',
            torch.ones(out_features, self.num_groups)
        )
        
        # 묶음마다의 영점
        self.register_buffer(
            'zero_point',
            torch.zeros(out_features, self.num_groups)
        )
        
        # 고를 수 있는 치우침
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    @classmethod
    def from_float(
        cls,
        linear: nn.Linear,
        bits: int = 8,
        group_size: Optional[int] = None
    ) -> 'QuantizedLinear':
        """
        뜬소수점 층으로 양자화 층을 만든다.
        """
        quant_linear = cls(
            linear.in_features,
            linear.out_features,
            bits=bits,
            group_size=group_size
        )
        
        weight = linear.weight.data
        group_size = quant_linear.group_size
        
        # 묶음마다 양자화한다
        for i in range(quant_linear.num_groups):
            start = i * group_size
            end = (i + 1) * group_size
            
            group_weight = weight[:, start:end]
            scale, zp = compute_scale_zero(group_weight, bits)
            
            quant_linear.scale[:, i] = scale
            quant_linear.zero_point[:, i] = zp
            quant_linear.weight_quantized[:, start:end] = quantize_tensor(
                group_weight, scale, zp, bits
            )
        
        if linear.bias is not None:
            quant_linear.bias.data = linear.bias.data.clone()
        
        return quant_linear
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 무게의 양자화를 되돌린다
        weight = torch.zeros(
            self.out_features, self.in_features,
            device=x.device, dtype=x.dtype
        )
        
        for i in range(self.num_groups):
            start = i * self.group_size
            end = (i + 1) * self.group_size
            
            weight[:, start:end] = dequantize_tensor(
                self.weight_quantized[:, start:end],
                self.scale[:, i:i+1],
                self.zero_point[:, i:i+1]
            )
        
        return nn.functional.linear(x, weight, self.bias)
```

### GPTQ(정확한 익힌 뒤 양자화)

GPTQ는 2차 앎을 써서 양자화 어긋남을 가장 작게 한다:

$$
\arg\min_{W_q} \|WX - W_q X\|_2^2
$$

핵심 눈썰미: 한 번에 무게 하나씩 양자화하고 남은 무게를 맞춰 메운다.

```python
import torch
import torch.nn as nn
from typing import List


class GPTQ:
    """
    GPTQ 양자화 알고리즘.
    
    헤세 앎을 써서 내놓기 어긋남을 가장 작게 하며 무게를
    세로줄마다 양자화한다.
    """
    
    def __init__(
        self,
        layer: nn.Linear,
        bits: int = 4,
        group_size: int = 128,
        block_size: int = 128
    ):
        self.layer = layer
        self.bits = bits
        self.group_size = group_size
        self.block_size = block_size
        
        self.rows = layer.weight.shape[0]
        self.cols = layer.weight.shape[1]
        
        # 헤세 쌓개
        self.H = torch.zeros(self.cols, self.cols, device=layer.weight.device)
        self.nsamples = 0
    
    def add_batch(self, inp: torch.Tensor):
        """
        들임 묶음에서 헤세를 쌓는다.
        
        H = X^T X(들임의 바깥 곱)
        """
        if len(inp.shape) == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        
        self.H += inp.T @ inp
        self.nsamples += inp.shape[0]
    
    def quantize(self) -> torch.Tensor:
        """
        GPTQ 양자화를 한다.
        """
        W = self.layer.weight.data.clone()
        
        # 헤세를 마무리한다
        H = self.H / self.nsamples
        
        # 수치가 안정되도록 잦아듦을 더한다
        damp = 0.01 * torch.diag(H).mean()
        H += damp * torch.eye(self.cols, device=H.device)
        
        # 촐레스키 쪼개기
        H_inv = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(H_inv)
        
        # 양자화한 무게
        Q = torch.zeros_like(W)
        
        # 덩이 단위로 처리한다
        for i1 in range(0, self.cols, self.block_size):
            i2 = min(i1 + self.block_size, self.cols)
            
            W_block = W[:, i1:i2].clone()
            Q_block = torch.zeros_like(W_block)
            Err_block = torch.zeros_like(W_block)
            H_inv_block = H_inv[i1:i2, i1:i2]
            
            for i in range(i2 - i1):
                w = W_block[:, i]
                d = H_inv_block[i, i]
                
                # 이 세로줄의 묶음을 정한다
                group_idx = (i1 + i) // self.group_size
                
                # 묶음의 잣수를 셈한다
                group_start = (group_idx * self.group_size) - i1
                group_end = min(group_start + self.group_size, i2 - i1)
                
                if i == max(0, group_start):
                    # 묶음이 시작할 때 잣수를 셈한다
                    group_weights = W_block[:, max(0, group_start):group_end]
                    scale, zp = compute_scale_zero(group_weights, self.bits)
                
                # 양자화한다
                q = quantize_tensor(w, scale, zp, self.bits)
                Q_block[:, i] = q.float()
                
                # 어긋남을 셈한다
                err = (w - dequantize_tensor(q, scale, zp)) / d
                
                # 덩이에 남은 무게를 새로 고친다
                W_block[:, i:] -= err.unsqueeze(1) * H_inv_block[i, i:].unsqueeze(0)
            
            Q[:, i1:i2] = Q_block
        
        return Q
```

### AWQ(깨어남을 헤아린 무게 양자화)

AWQ는 깨어남의 크기로 두드러진 무게를 가려내어 지킨다:

$$
\text{saliency}_j = \|X_j\|_2
$$

딸린 깨어남의 크기가 큰 무게는 양자화 앞에서 키우고 미룸 때 되돌린다.

```python
def compute_awq_scales(
    weight: torch.Tensor,
    activations: torch.Tensor,
    bits: int = 4
) -> torch.Tensor:
    """
    AWQ 잣수를 셈한다.
    
    잣수 = 깨어남 크기^alpha. 여기서 alpha가 무게 양자화 어긋남과
    깨어남 양자화 어긋남을 저울질한다.
    """
    # 깨어남의 크기를 셈한다
    act_scales = activations.abs().mean(dim=0)
    
    # 무게의 크기를 셈한다
    weight_scales = weight.abs().mean(dim=0)
    
    # 가장 좋은 잣수(무게 어긋남과 깨어남 어긋남을 저울질한다)
    # 깨어남이 클수록 더 지킨다
    alpha = 0.5  # 다듬을 수 있는 웃매개변수
    scales = act_scales.pow(alpha) / weight_scales.pow(1 - alpha)
    
    # 잣수를 잘라 낸다
    scales = torch.clamp(scales, min=1e-5, max=1e5)
    
    return scales
```

## 깨어남 양자화

### 그때그때 하는 양자화

들임마다 돌아가는 도중에 잣수를 셈한다:

```python
class DynamicQuantizedLinear(nn.Module):
    """
    깨어남을 그때그때 양자화하는 선형 층.
    """
    
    def __init__(self, in_features: int, out_features: int, bits: int = 8):
        super().__init__()
        self.bits = bits
        # 미리 양자화한 무게
        self.weight_scale = nn.Parameter(torch.ones(1))
        self.register_buffer(
            'weight_q', 
            torch.zeros(out_features, in_features, dtype=torch.int8)
        )
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 들임을 그때그때 양자화한다
        x_scale, x_zp = compute_scale_zero(x, self.bits)
        x_q = quantize_tensor(x, x_scale, x_zp, self.bits)
        
        # 정수 행렬 곱하기(흉내)
        # 실전에서는 전용 int8 알맹이를 쓴다
        out_q = torch.matmul(x_q.float(), self.weight_q.T.float())
        
        # 내놓기의 양자화를 되돌린다
        out = out_q * (x_scale * self.weight_scale)
        
        return out + self.bias
```

### 붙박이 양자화

대표 자료로 미리 잣수의 눈금을 맞춘다:

```python
class CalibrationCollector:
    """
    붙박이 양자화를 위해 깨어남 통계를 모은다.
    """
    
    def __init__(self, method: str = 'minmax'):
        self.method = method
        self.min_vals = []
        self.max_vals = []
        self.histograms = []
    
    def collect(self, tensor: torch.Tensor):
        """텐서에서 통계를 적는다."""
        self.min_vals.append(tensor.min().item())
        self.max_vals.append(tensor.max().item())
        
        if self.method == 'entropy':
            # 엔트로피 눈금 맞추기를 위한 도수 그림
            hist = torch.histc(tensor, bins=2048)
            self.histograms.append(hist)
    
    def compute_scale(self, bits: int = 8) -> Tuple[float, float]:
        """마지막 잣수와 영점을 셈한다."""
        if self.method == 'minmax':
            min_val = min(self.min_vals)
            max_val = max(self.max_vals)
            
        elif self.method == 'percentile':
            # 99.9번째 백분위수를 쓴다
            all_mins = torch.tensor(self.min_vals)
            all_maxs = torch.tensor(self.max_vals)
            min_val = torch.quantile(all_mins, 0.001).item()
            max_val = torch.quantile(all_maxs, 0.999).item()
            
        scale = (max_val - min_val) / (2 ** bits - 1)
        zero_point = round(-min_val / scale)
        
        return scale, zero_point
```

## 큰 말 모델에 맞춘 재주

### 열쇠-값 곳간 양자화

기억 공간을 줄이려 갈무리한 열쇠와 값을 양자화한다:

```python
class QuantizedKVCache:
    """
    양자화해 담는 열쇠-값 곳간.
    """
    
    def __init__(self, bits: int = 8):
        self.bits = bits
        self.keys_q = None
        self.values_q = None
        self.key_scales = []
        self.value_scales = []
    
    def update(
        self,
        new_keys: torch.Tensor,
        new_values: torch.Tensor
    ):
        """새 열쇠와 값을 곳간에 더한다."""
        # 새 항목을 양자화한다
        k_scale, k_zp = compute_scale_zero(new_keys, self.bits)
        v_scale, v_zp = compute_scale_zero(new_values, self.bits)
        
        new_k_q = quantize_tensor(new_keys, k_scale, k_zp, self.bits)
        new_v_q = quantize_tensor(new_values, v_scale, v_zp, self.bits)
        
        # 곳간에 덧붙인다
        if self.keys_q is None:
            self.keys_q = new_k_q
            self.values_q = new_v_q
        else:
            self.keys_q = torch.cat([self.keys_q, new_k_q], dim=2)
            self.values_q = torch.cat([self.values_q, new_v_q], dim=2)
        
        self.key_scales.append((k_scale, k_zp))
        self.value_scales.append((v_scale, v_zp))
    
    def get_keys_values(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """양자화를 되돌린 열쇠와 값을 가져온다."""
        # 간단히 하려고 가장 최근 잣수를 쓴다
        # 실전에서는 토막마다나 묶음마다의 잣수를 쓴다
        k_scale, k_zp = self.key_scales[-1]
        v_scale, v_zp = self.value_scales[-1]
        
        keys = dequantize_tensor(self.keys_q, k_scale, k_zp)
        values = dequantize_tensor(self.values_q, v_scale, v_zp)
        
        return keys, values
```

### 섞인 정밀도 양자화

조각마다 다른 자릿수를 쓴다:

```python
MIXED_PRECISION_CONFIG = {
    'embedding': 8,      # 묻힘: 8비트
    'attention.qkv': 4,  # QKV 쏘기: 4비트
    'attention.out': 4,  # 내놓기 쏘기: 4비트
    'mlp.gate': 4,       # 여러 층 인식개 층: 4비트
    'mlp.up': 4,
    'mlp.down': 4,
    'lm_head': 8,        # 내놓는 머리: 8비트(민감함)
    'layernorm': 32,     # 온전한 정밀도를 지킨다
}


def quantize_model_mixed(
    model: nn.Module,
    config: dict
) -> nn.Module:
    """
    섞인 정밀도 양자화를 적용한다.
    """
    for name, module in model.named_modules():
        for pattern, bits in config.items():
            if pattern in name:
                if isinstance(module, nn.Linear) and bits < 32:
                    # 양자화한 판으로 바꾼다
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent = model.get_submodule(parent_name)
                    
                    quant_module = QuantizedLinear.from_float(module, bits=bits)
                    setattr(parent, child_name, quant_module)
                break
    
    return model
```

## 양자화를 헤아린 익히기(QAT)

### 곧바로 지나가기 어림개(STE)

양자화를 지나는 기울기:

$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{w}} \cdot \mathbf{1}_{|w| \leq \text{clip}}
$$

```python
class STEQuantize(torch.autograd.Function):
    """
    양자화를 위한 곧바로 지나가기 어림개.
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor, bits: int):
        q_min = -(2 ** (bits - 1))
        q_max = 2 ** (bits - 1) - 1
        
        # 양자화한다
        x_q = torch.round(x / scale)
        x_q = torch.clamp(x_q, q_min, q_max)
        
        # 양자화를 되돌린다
        x_dq = x_q * scale
        
        return x_dq
    
    @staticmethod
    def backward(ctx, grad_output):
        # 곧바로 지나가기: 기울기를 그대로 넘긴다
        return grad_output, None, None


class QATLinear(nn.Module):
    """
    양자화를 헤아린 익히기를 하는 선형 층.
    """
    
    def __init__(self, in_features: int, out_features: int, bits: int = 4):
        super().__init__()
        self.bits = bits
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # 배울 수 있는 잣수
        self.weight_scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 익히는 동안 무게를 흉내 양자화한다
        w_q = STEQuantize.apply(self.weight, self.weight_scale, self.bits)
        
        return nn.functional.linear(x, w_q, self.bias)
```

## 값매김과 견줌

### 헷갈림도에 미치는 영향

WikiText-2에서 헷갈림도가 늘어나는 흔한 정도:

| 방법 | 비트 | 헷갈림도 늘어남 |
|--------|------|--------------|
| 가장 가까운 값으로 반올림 | 8 | +0.1 |
| 가장 가까운 값으로 반올림 | 4 | +0.5~2.0 |
| GPTQ | 4 | +0.1~0.3 |
| AWQ | 4 | +0.1~0.3 |
| GPTQ | 3 | +0.3~1.0 |

### 빠르기 견줌

A100에서의 미룸 빨라짐(LLaMA-7B):

| 정밀도 | 초당 토막 | 기억 공간 |
|-----------|------------|--------|
| FP16 | 100 | 14 GB |
| INT8 | 150 | 7 GB |
| INT4(GPTQ) | 180 | 4 GB |
| INT4(AWQ) | 200 | 4 GB |

## 실무 지침

### 권하는 자리매김

| 쓰임새 | 방법 | 비트 | 비고 |
|----------|--------|------|-------|
| 서버(품질) | FP16/BF16 | 16 | 가장 좋은 품질 |
| 서버(균형) | GPTQ/AWQ | 4 | 좋은 맞바꿈 |
| 일반 GPU | GPTQ | 4 | 70억을 8GB에 담는다 |
| 가장자리 기기 | AWQ | 4 | 묶음 크기 128 |
| 극단적 눌러 담기 | GPTQ | 3 | 품질 떨어짐이 눈에 띈다 |

### 좋은 관행

1. **눈금 맞추기 자료**: 목표 분야를 대표하는 표본 128~512개를 쓴다
2. **묶음 크기**: 128이 좋은 균형을 준다(텐서마다나 채널마다와 견주어)
3. **민감한 층**: 묻힘과 내놓는 머리는 정밀도를 높게 둔다
4. **검증**: 늘 헷갈림도와 뒤따르는 일을 값매김한다
5. **재주 아우르기**: 양자화 + 열쇠-값 곳간 + 플래시 눈길

## 요약

양자화는 큰 말 모델 펼치기에 꼭 필요하다:

1. **4비트 무게**: GPTQ/AWQ로 거의 잃음 없이
2. **기억 공간 줄이기**: 모델이 4~8배 작아진다
3. **빠르기 나아짐**: 미룸이 1.5~2배 빠르다
4. **누구나 쓰기**: 일반 GPU에서 70억 모델을 돌린다

핵심 맞바꿈: **품질과 기억 공간과 빠르기**

## 참고 문헌

1. Frantar, E., et al. (2023). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers."
2. Lin, J., et al. (2023). "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration."
3. Dettmers, T., et al. (2022). "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale."
4. Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs."
5. Xiao, G., et al. (2023). "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models."

## 연습문제

**연습문제 1.**
열쇠-값 곳간이 자기되돌리기 풀기를 어떻게 빠르게 하는지 밝혀라. 기억 공간의 맞바꿈은 무엇인가?

??? success "연습문제 1 풀이"
    자기되돌리기로 글을 만들 때 새 토막마다 앞선 토막 모두에 눈길을 준다. 갈무리하지 않으면 토막 $t$을 만들 때 앞선 토막 $t-1$개의 열쇠와 값 되비춤을 다시 셈하므로 길이 $T$의 이음에 온 셈이 $O(t^2)$이 된다. 열쇠-값 갈무리를 쓰면 앞선 걸음의 열쇠와 값을 담아 두었다가 되쓰므로 걸음마다 새 토막의 열쇠와 값만 셈하면 되어 온 셈이 $O(T)$으로 준다. 맞바꿈은 이렇다. 열쇠-값 갈무리의 기억 자리가 $O(T \cdot L \cdot d)$으로 늘어난다. 여기서 $L$은 층 수, $d$은 숨은 차수다. 큰 모델로 긴 이음을 다루면 GPU 기억 자리를 크게 잡아먹을 수 있다.

---

**연습문제 2.**
플래시 눈길의 고갱이 생각을 설명하여라. 수학으로는 같은 셈을 하는데 왜 빨라지는가?

??? success "연습문제 2 풀이"
    플래시 눈길은 GPU 기억 자리의 층 얼개를 쓴다. 여느 눈길은 $N \times N$ 눈길 행렬을 HBM(느린 GPU 기억 자리)에 실제로 만들어 기억 자리에 매인 셈이 된다. 플래시 눈길은 셈을 SRAM(빠른 칩 안 기억 자리)에 들어가는 덩이로 쪼개어, 온전한 눈길 행렬을 한 번도 만들지 않고 덩이마다 눈길을 셈한다. 이어 가는 소프트맥스(달리는 최댓값과 합을 좇는다)로 딱 맞는 눈길을 조금씩 셈한다. 빨라지는 까닭은 뜨는 셈 횟수가 줄어서가 아니라 HBM 읽고 쓰기가 줄어서다(입출력 복잡도가 $O(N^2 d)$에서 $O(N^2 d^2 / M)$으로 떨어지며 $M$은 SRAM 크기다). 그래서 벽시계 시간이 2~4배 빨라지고 기억 자리는 $O(N)$이 된다.

---

**연습문제 3.**
미리 짚어 풀기란 무엇이며 내놓는 분포를 바꾸지 않고 어떻게 미룸을 빠르게 하는가?

??? success "연습문제 3 풀이"
    미리 짚어 풀기는 작고 빠른 "밑그림" 모델로 후보 토막 $K$개를 만든 뒤 큰 "목표" 모델로 그 $K$개를 나란히 확인한다. 목표 모델이 밑그림의 어림에 동의하면 목표 모델 앞먹임 한 번으로 $K$개를 모두 받아들인다(차례차례 $K$번 대신). 어긋나면 물리치기 표집으로 다룬다. 곧 처음 물리친 토막을 맞춘 분포에서 다시 뽑아 내놓는 분포가 목표 모델의 것과 같게 한다. 빨라짐은 밑그림 모델의 받아들임 비율에 달렸고, 좋은 밑그림 모델이면 2~3배가 보통이다. 핵심 눈썰미는 확인은 나란히 되지만 만들어 내기는 차례차례라는 것이다.

---

**연습문제 4.**
익힌 뒤 양자화(PTQ)와 양자화를 헤아린 익히기(QAT)를 견주어라. 저마다 언제 쓰는 것이 좋은가?

??? success "연습문제 4 풀이"
    **익힌 뒤 양자화**는 더 익히지 않고 눈금 맞추기 자료로 잣수 인자를 정해 미리 익힌 모델의 무게(그리고 원하면 깨어남)를 양자화한다. 빠르고 단순하지만 특히 8비트 아래에서는 정확도가 떨어질 수 있다. **양자화를 헤아린 익히기**는 기울기에 곧바로 지나가기 어림개를 써서 익히는 동안 양자화를 흉내내어 모델이 낮은 정밀도에 맞춰지게 한다. 낮은 자릿수(4비트, 2비트)에서 정확도가 더 낫지만 온전한 익히기를 한 번 더 돌려야 한다. 빠르기가 중요하고 8비트면 넉넉한 (펼치기) 장면에서는 **익힌 뒤 양자화가 낫다**. 4비트처럼 세게 양자화해야 하고 익힐 자원이 있으면 **양자화를 헤아린 익히기가 낫다**.
