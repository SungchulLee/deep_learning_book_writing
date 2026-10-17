# 대규모 언어 모델의 양자화

양자화는 웨이트나 활성화를 적은 비트로 표현해 저장 공간과 메모리 전송량을 줄이는 방법이다. 실제 추론 속도는 해당 표현으로 계산하는 커널과 장치, 배치 크기와 입력 길이에 달려 있다. 이 절의 코드는 개념을 설명하는 구현이며 완성된 GPTQ·AWQ 배포 라이브러리와 구분해서 읽는다.

## 1. 수치 형식과 메모리

| 형식 | 비트 수 | 피처 |
|---|---:|---|
| FP32 | 32 | 최대 유한값 약 3.4×10³⁸ |
| FP16 | 16 | 최대 유한값 65,504 |
| BF16 | 16 | FP32와 비슷한 지수 범위, 더 적은 유효 자릿수 |
| INT8 | 8 | 부호 있는 정수 −128~127 |
| INT4 | 4 | 부호 있는 정수 −8~7, 실제 패킹 필요 |
| FP8 | 8 | 지수·가수 비트 배분에 따라 범위와 정밀도가 다름 |

파라미터 70억 개의 웨이트 데이터만 계산하면 FP32는 28 GB, FP16/BF16은 14 GB, INT8은 7 GB, INT4는 3.5 GB다. GB는 10⁹바이트이며 GiB와 다르다. 양자화 스케일, 영점, 패킹 메타데이터, 높은 정밀도로 남긴 계층과 실행 중 활성화·KV 캐시는 별도다. 따라서 이 수치만으로 필요한 GPU 용량을 확정할 수 없다.

## 2. 균일 아핀 양자화

정수 범위 [q_min,q_max], 양의 스케일 Δ와 정수 영점 z를 사용하면 다음과 같다.

$$q=\operatorname{clip}\left(\operatorname{round}(x/\Delta)+z,q_{\min},q_{\max}\right),\qquad
\hat{x}=\Delta(q-z).$$

대칭 양자화에서 b≥2이고 z=0, 정수 범위를 [−(2ᵇ⁻¹−1),2ᵇ⁻¹−1]로 잡으면 Δ=max|x|/(2ᵇ⁻¹−1)로 정할 수 있다. 부호 있는 최소값 −2ᵇ⁻¹을 사용하지 않는 관례이며, 모든 값이 0인 경우는 별도 처리해야 한다.

비대칭 min-max 양자화의 한 설정은 다음과 같다.

$$\Delta=\frac{x_{\max}-x_{\min}}{q_{\max}-q_{\min}},\qquad
z=\operatorname{clip}\left(\operatorname{round}(q_{\min}-x_{\min}/\Delta),q_{\min},q_{\max}\right).$$

보통 실수 범위에 0을 포함하도록 조정하며 상수 텐서에서도 양의 스케일이 필요하다. z의 반올림과 범위 제한 때문에 양 끝이 항상 정확히 복원되는 것은 아니다. 포화가 없는 구간의 반올림 오차는 |x−x̂|≤Δ/2다. 오차가 [−Δ/2,Δ/2]에서 균일하다는 근사 아래 평균 제곱 오차는 Δ²/12이며, 이상치·클리핑·비균일 분포에는 그대로 적용할 수 없다.

## 3. 웨이트 양자화와 복원 오차

학습 후 양자화(PTQ)는 사전 학습된 웨이트에 추가 그래디언트 학습 없이 적용한다. 다음 코드는 그룹별 스케일을 구하고 양자화한 값을 순전파에서 실수로 복원한다.

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
    양자화 파라메터를 셈한다.
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
    무게를 양자화한 선형 레이어.
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
        
        # 배치별 양자화의 배치 수
        self.num_groups = in_features // self.group_size
        
        # 양자화한 무게(int8로 담음)
        self.register_buffer(
            'weight_quantized',
            torch.zeros(out_features, in_features, dtype=torch.int8)
        )
        
        # 배치마다의 잣수
        self.register_buffer(
            'scale',
            torch.ones(out_features, self.num_groups)
        )
        
        # 배치마다의 영점
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
        뜬소수점 레이어로 양자화 레이어를 만든다.
        """
        quant_linear = cls(
            linear.in_features,
            linear.out_features,
            bits=bits,
            group_size=group_size
        )
        
        weight = linear.weight.data
        group_size = quant_linear.group_size
        
        # 배치마다 양자화한다
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

`compute_scale_zero`의 비대칭 분기는 부호 없는 범위에 해당하는 영점을 계산하지만 `quantize_tensor`는 부호 있는 범위로 자른다. 두 관례가 맞지 않아 의도하지 않은 포화가 생길 수 있다. 영점 범위 제한과 0 스케일 처리도 없다. 아래 예제들이 기본으로 호출하는 대칭 분기에서도 0으로만 된 텐서는 별도 처리가 필요하다.

`QuantizedLinear`는 4비트를 지정해도 값을 **INT8 텐서**에 하나씩 저장한다. 4비트 패킹에 따른 8배 압축 구현이 아니다. 스케일은 각 입력 그룹의 모든 출력 행을 합쳐 하나 계산하고 행마다 같은 값을 기록한다. `in_features`가 `group_size`로 나누어떨어지지 않으면 나머지 열을 처리하지 않는다. 새 모듈의 장치와 dtype도 원본과 맞추어야 한다. 순전파는 전체 실수 웨이트를 다시 만들고 실수 선형 연산을 하므로 정수 커널의 속도 향상을 보여 주지 않는다.

### GPTQ: 계층 출력의 복원 오차 줄이기

[GPTQ](https://arxiv.org/abs/2210.17323)는 보정 입력 X에서 원래 계층의 출력을 보존하도록 웨이트를 순차 양자화하고 남은 웨이트에 오차를 리워드한다. 열벡터를 입력으로 쓰는 표기에서는 다음 목적을 생각할 수 있다.

$$\min_{W_q}\|WX-W_qX\|_F^2.$$

이는 계층 출력의 근사 목적이며 언어 모델의 최종 로스를 전역 최적화한다는 뜻은 아니다. 입력의 2차 통계와 수치 안정화가 중요하다.

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
        들임 배치에서 헤세를 쌓는다.
        
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
                
                # 이 세로줄의 배치를 정한다
                group_idx = (i1 + i) // self.group_size
                
                # 배치의 잣수를 셈한다
                group_start = (group_idx * self.group_size) - i1
                group_end = min(group_start + self.group_size, i2 - i1)
                
                if i == max(0, group_start):
                    # 배치가 시작할 때 잣수를 셈한다
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

위 `GPTQ` 클래스는 논문의 완전한 구현이 아니다. 보정 표본이 없으면 `nsamples`로 나누는 부분부터 유효하지 않으며, 특이한 입력에는 분해의 안정성도 고려해야 한다. 블록 사이의 오차 전파를 구현하지 않고, 역행렬에 기반한 갱신도 논문의 전체 절차와 같다고 볼 수 없다. 반환값 `Q`는 정수 코드 값을 실수 텐서에 담은 것이며 복원에 필요한 그룹별 스케일·영점을 함께 반환하지 않는다. 그대로 원본 웨이트를 대체하거나 압축 체크포인트로 사용할 수 없다.

### AWQ: 활성화 통계로 스케일 정하기

[AWQ](https://arxiv.org/abs/2306.00978)는 활성화 통계를 이용해 중요한 입력 채널을 고려하고, 채널 스케일을 조정해 저비트 웨이트의 출력 오차를 줄인다. 웨이트를 키운 만큼 입력 쪽에서 역스케일을 적용하면 양자화 전의 선형 연산은 유지된다. 실제로는 스케일 탐색과 양자화 오차 평가가 필요하다.

```python
def compute_awq_scales(
    weight: torch.Tensor,
    activations: torch.Tensor,
    bits: int = 4
) -> torch.Tensor:
    """
    AWQ 잣수를 셈한다.
    
    잣수 = 활성화 크기^alpha. 여기서 alpha가 무게 양자화 어긋남과
    활성화 양자화 어긋남을 저울질한다.
    """
    # 활성화의 크기를 셈한다
    act_scales = activations.abs().mean(dim=0)
    
    # 무게의 크기를 셈한다
    weight_scales = weight.abs().mean(dim=0)
    
    # 가장 좋은 잣수(무게 어긋남과 활성화 어긋남을 저울질한다)
    # 활성화가 클수록 더 지킨다
    alpha = 0.5  # 다듬을 수 있는 웃파라메터
    scales = act_scales.pow(alpha) / weight_scales.pow(1 - alpha)
    
    # 잣수를 잘라 낸다
    scales = torch.clamp(scales, min=1e-5, max=1e5)
    
    return scales
```

위 함수는 α=0.5의 스케일 후보를 계산할 뿐이다. `bits` 인수는 사용하지 않으며 스케일 탐색, 입력의 역변환과 웨이트 양자화도 수행하지 않는다. `activations`가 [표본 수, 입력 차원]이라는 전제가 있고, 3차원 입력을 그대로 넘기면 평균 축의 의미가 달라진다. 웨이트 통계가 0일 때 0/0으로 생기는 NaN은 마지막 `clamp`만으로 해결되지 않는다.

## 4. 동적·정적 활성화 양자화

동적 양자화는 실행 중 입력에서 활성화의 양자화 파라미터를 정한다. 정적 양자화는 대표 데이터로 미리 보정한 범위를 사용한다. 동적 방식에는 통계 계산 비용이 들고 정적 방식에는 보정 데이터와 실제 입력 분포의 차이가 영향을 준다. 웨이트만 양자화하는 방식과도 구분한다.

```python
class DynamicQuantizedLinear(nn.Module):
    """
    활성화를 그때그때 양자화하는 선형 레이어.
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
        # 실전에서는 전용 int8 커널을 쓴다
        out_q = torch.matmul(x_q.float(), self.weight_q.T.float())
        
        # 내놓기의 양자화를 되돌린다
        out = out_q * (x_scale * self.weight_scale)
        
        return out + self.bias
```

`DynamicQuantizedLinear`는 기본 대칭 설정으로 입력을 양자화하지만 행렬곱 전에 두 정수 텐서를 `float()`로 변환한다. 실제 INT8 누산 커널을 구현한 것은 아니다. 웨이트 버퍼가 0으로 초기화되고 사전 학습된 웨이트를 채우는 절차도 없다. `weight_scale`의 양수 제약과 스케일 추정은 별도로 필요하다.

```python
class CalibrationCollector:
    """
    붙박이 양자화를 위해 활성화 통계를 모은다.
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

`CalibrationCollector`의 `percentile`은 개별 활성화 값의 백분위수가 아니라 배치별 최솟값·최댓값의 백분위수다. `entropy`는 히스토그램만 모으고 `compute_scale`에서 범위를 정하지 않아 그대로 호출하면 실패한다. 각 히스토그램의 구간도 동일하게 정렬되어 있지 않다. 빈 수집 결과와 상수 텐서 처리, 이후 사용할 정수 범위와 영점의 일치도 확인해야 한다.

## 5. KV 캐시와 혼합 정밀도

KV 캐시를 저비트로 저장하면 긴 문맥의 메모리를 줄일 수 있다. 다만 캐시를 읽을 때의 복원 비용과 어텐션 오차를 평가해야 한다. 토큰·헤드·그룹별 스케일을 쓰면 각 데이터 조각과 대응하는 스케일을 함께 관리한다.

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
        # 실전에서는 토막마다나 배치마다의 잣수를 쓴다
        k_scale, k_zp = self.key_scales[-1]
        v_scale, v_zp = self.value_scales[-1]
        
        keys = dequantize_tensor(self.keys_q, k_scale, k_zp)
        values = dequantize_tensor(self.values_q, v_scale, v_zp)
        
        return keys, values
```

`QuantizedKVCache`는 갱신마다 다른 스케일로 양자화한 데이터를 연결하지만, 읽을 때는 **마지막 갱신의 스케일로 전체 캐시를 복원한다**. 앞선 조각의 스케일이 다르면 잘못된 값이 된다. 저장한 스케일 목록과 각 조각의 길이를 대응시켜 복원해야 한다. 또한 4비트를 지정해도 INT8 저장이며, 매번 `cat`으로 복사하고 전체 캐시를 실수로 복원하므로 그 비용도 고려해야 한다.

계층별 혼합 정밀도는 민감한 부분을 높은 정밀도로 남기는 선택이다. 어떤 계층이 민감한지는 모델과 작업에 따라 검증한다.

```python
MIXED_PRECISION_CONFIG = {
    'embedding': 8,      # 묻힘: 8비트
    'attention.qkv': 4,  # QKV 쏘기: 4비트
    'attention.out': 4,  # 내놓기 쏘기: 4비트
    'mlp.gate': 4,       # 여러 레이어 인식개 레이어: 4비트
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

이 함수는 이름 패턴에 맞는 `nn.Linear`만 교체한다. 설정에 `embedding`이나 `layernorm`이 있어도 임베딩과 정규화 모듈 자체를 양자화하지 않는다. 실제 모듈 이름이 패턴과 맞는지 확인해야 하며, 앞서 본 `QuantizedLinear`의 저장·장치·그룹 처리 제한도 그대로 적용된다.

## 6. 양자화 인지 학습과 STE

양자화 인지 학습(QAT)은 순전파에서 양자화·복원된 값을 사용하고 역전파에는 미분 불가능한 반올림을 대신할 근사 그래디언트를 사용한다. 직선 통과 추정기(STE)의 한 선택은 다음과 같다.

$$\frac{\partial L}{\partial w}\approx\frac{\partial L}{\partial\hat{w}}.$$

클리핑 구간 밖의 그래디언트를 0으로 만드는 변형도 있지만 아래 구현은 그렇게 하지 않는다.

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
        # 곧바로 지나가기: 그래디언트를 그대로 넘긴다
        return grad_output, None, None

class QATLinear(nn.Module):
    """
    양자화를 헤아린 익히기를 하는 선형 레이어.
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

`STEQuantize.backward`는 모든 위치에서 입력 그래디언트를 그대로 통과시키고 스케일에는 `None`을 반환한다. 따라서 `weight_scale`이 `nn.Parameter`여도 이 경로로 학습되지 않는다. 초기 스케일 1은 0.01 수준으로 초기화한 웨이트 대부분을 0으로 반올림할 수 있다. 이 예제는 웨이트의 fake quantization만 보여 주며 활성화 양자화, 학습 루프와 실제 저비트 커널로의 변환은 포함하지 않는다.

## 7. 품질과 속도 평가

퍼플렉서티 변화는 원본 모델, 토크나이저, 평가 전처리, 문맥 길이와 보정 데이터에 따라 달라진다. GPTQ나 AWQ 4비트라는 이름만으로 일정한 로스 폭을 보장할 수 없다. 동일한 모델과 평가 조건에서 기준값과 양자화 후 값을 함께 기록한다.

속도 비교에는 장치, 커널, 배치, 프롬프트 길이, 생성 길이와 정밀도를 명시한다. 프리필과 디코딩의 병목이 다르므로 첫 토큰 지연 시간과 토큰 간 지연 시간, 초당 토큰 수를 구분한다. 웨이트 메모리만을 전체 실행 메모리로 보고하지 않는다.

## 8. 적용 순서

먼저 지원되는 커널과 메모리 예산을 확인하고, PTQ 기준 모델을 만든 다음 대표 데이터에서 품질을 비교한다. 저비트에서 성능 저하가 크면 그룹 크기, 보정 범위, 높은 정밀도로 남길 계층 또는 QAT를 검토한다. 보정 표본 수에 보편적인 정답은 없다.

**양자화 그룹 크기**와 **추론 배치 크기**는 서로 다른 설정이다. 예를 들어 그룹 크기 128은 스케일을 공유하는 웨이트 묶음의 크기이지 동시에 처리하는 요청 수가 아니다. KV 캐시와 어텐션 최적화를 결합할 때도 dtype, 레이아웃과 지원 커널의 호환성을 확인한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
웨이트 70억 개를 FP16과 실제 패킹된 INT4로 저장할 때 웨이트 데이터 크기를 계산하라. 4비트 정수 코드를 INT8 텐서에 저장하면 어떤 차이가 있는가?

</div>

??? success "풀이"

    FP16은 70억×2=14 GB, INT4는 70억×0.5=3.5 GB다. 스케일 등 추가 정보는 제외한 값이다. 코드를 INT8 원소 하나에 하나씩 담으면 웨이트 코드만 7 GB이며, 값의 범위를 4비트로 제한했다고 저장 공간까지 4비트가 되는 것은 아니다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
부호 있는 비대칭 양자화에서 영점 식에 q_min이 필요한 이유를 설명하라. 모든 입력이 같은 값인 텐서는 어떻게 다루어야 하는가?

</div>

??? success "풀이"

    x_min이 q_min 근처로 대응되도록 z=round(q_min−x_min/Δ)를 사용한다. 부호 없는 범위의 식을 그대로 쓰면 의도한 범위가 이동한다. 상수 텐서는 min-max 차이가 0이므로 양의 최소 스케일이나 상수 전용 처리가 필요하다. 실수 0을 표현할 수 있는 범위 조정과 영점 제한도 함께 고려한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
첫 캐시 조각의 스케일이 0.1이고 두 번째 조각의 스케일이 0.2일 때 전체 정수 캐시를 0.2로 복원하면 어떤 문제가 생기는가? 영점은 0이라고 가정하라.

</div>

??? success "풀이"

    첫 조각에서 코드 q가 뜻하던 값은 0.1q지만 0.2q로 복원되어 두 배가 된다. 조각별 스케일과 토큰 범위를 함께 저장해 대응시켜 복원하거나 일관된 다른 스케일 폴리시를 사용해야 한다. 마지막 스케일 하나로는 이전 조각을 정확히 해석할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
PTQ와 QAT의 차이를 설명하고 이 절의 STE가 스케일을 학습하는지 판별하라.

</div>

??? success "풀이"

    PTQ는 추가 그래디언트 학습 없이 사전 학습된 모델을 변환하고, QAT는 학습 중 양자화 오차를 모사해 웨이트를 적응시킨다. QAT는 전체 사전 학습을 반드시 반복하는 방식이 아니며 파인튜닝으로 적용할 수도 있다. 여기서는 backward가 scale에 None을 반환하므로 해당 스케일은 이 경로에서 학습되지 않는다. QAT가 언제나 PTQ보다 우수하다는 보장도 없다.



## 정리하며

양자화는 정수 범위, 스케일, 영점과 복원 규칙을 일관되게 정해야 한다. 비트 수를 지정하는 것과 실제 패킹·저비트 연산을 구현하는 것은 별개다. PTQ와 QAT, 웨이트와 활성화·KV 캐시 양자화를 구분하고 같은 조건에서 품질, 저장 공간, 실행 메모리와 속도를 평가한다.

**참고 문헌**

1. Frantar, E., et al. (2023). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers."
2. Lin, J., et al. (2023). "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration."
3. Dettmers, T., et al. (2022). "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale."
4. Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs."
5. Xiao, G., et al. (2023). "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models."
