# LoRA: 저랭크 적응

[LoRA](https://arxiv.org/abs/2106.09685)는 기반 웨이트를 동결하고 작은 두 행렬로 표현한 웨이트 갱신을 학습하는 파라미터 효율적 파인튜닝 방법이다. 학습 파라메터와 옵티마이저 상태를 줄일 수 있지만 기반 모델·활성화 메모리가 사라지는 것은 아니다. 감소 비율과 전체 파인튜닝 대비 품질은 적용 위치·랭크·과제에 따라 달라진다.

## 1. 행렬 모양과 학습량

열벡터 표기에서 W₀∈R^(d_out×d_in)에 ΔW=(α/r)BA를 더한다. A∈R^(r×d_in)는 다운 프로젝션, B∈R^(d_out×r)는 업 프로젝션이다. ΔW의 랭크는 최대 r이며 실제로 정확히 r일 필요는 없다. 적응에 저차원 구조가 유용할 수 있다는 실증 결과이지 모든 최적 갱신이 저랭크라는 정리는 아니다.

행렬 웨이트만 비교하면 전체 갱신은 d_in d_out개, LoRA는 r(d_in+d_out)개다. 두 차원 4096·r=8이면 16,777,216개 대 65,536개로 256배 차이다. 모델 전체에서는 적용하지 않는 레이어·head와 bias 폴리시도 포함해 집계해야 한다.

## 2. 레이어 구현과 초기화

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Dict, Any

class LoRALayer(nn.Module):
    """
    이미 있는 선형 레이어를 감싸는 LoRA 레이어.
    
    짜기: h = W₀x + (α/r)BAx
    
    인수:
        original_layer: 맞출 선형 레이어
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
        # A: 들임 피처 -> 계수(내림 쏘기)
        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        # B: 계수 -> 내놓기 피처(올림 쏘기)
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
        피드포워드: W₀x + (α/r)BAx
        """
        if self.merged:
            # 무게를 합쳤으면 본디 레이어만 쓴다
            return self.original(x)
        
        # 본디 길(얼림)
        original_output = self.original(x)
        
        # LoRA 길: x @ A @ B * 잣수
        lora_output = self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        
        return original_output + lora_output
    
    def merge_weights(self):
        """
        미룸을 위해 LoRA 무게를 본디 레이어에 합친다.
        
        W' = W₀ + (α/r)BA^T
        
        합친 뒤에는 피드포워드에 군더더기가 없다.
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
        """익힐 수 있는 LoRA 파라메터의 수."""
        return self.lora_A.numel() + self.lora_B.numel()

class LoRALinear(nn.Module):
    """
    홀로 서는 LoRA 선형 레이어(이미 있는 레이어를 감싸지 않는다).
    
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

코드는 행벡터 x@A@B 표기를 써서 lora_A가 d_in×r, lora_B가 r×d_out이며 합칠 갱신은 그 곱의 전치다. 수식 표기의 이름·모양과 이 차이를 혼동하지 않는다. B를 0으로 초기화하므로 처음 갱신은 0이고 첫 역전파의 A 경사는 0일 수 있지만 B는 학습 신호를 받을 수 있다.

LoRALayer는 제공한 Linear를 감싸고 기반 웨이트·바이어스를 동결한다. LoRALinear는 새 무작위 기반 웨이트를 만든 뒤 동결하므로 그 자체가 사전 학습 레이어를 가져온 것은 아니다. 랭크는 양수여야 하며 새 행렬의 장치·dtype과 기존 레이어도 맞춰야 한다.

α/r는 갱신의 스케일 선택이지 랭크를 바꿔도 갱신 노름·러닝레이트·안정성이 같아진다는 보장이 아니다. 러닝레이트·초기화·드롭아웃·랭크와 함께 검증한다. 고정 랭크 8이나 α=2r가 모든 과제의 최적 기본값이라고 단정하지 않는다.

## 3. 적용 위치와 동결

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
        LoRA 레이어를 갖춘 모델(본디 무게는 얼림)
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
        
        # LoRA 레이어를 만들어 앉힌다
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
    모델 상태 사전에서 LoRA 파라메터만 뽑는다.
    
    맞춤개 무게만 갈무리하거나 불러올 때 쓸모 있다.
    """
    return {
        name: param for name, param in model.state_dict().items()
        if 'lora_' in name
    }

def load_lora_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]):
    """LoRA 파라메터를 모델에 불러온다."""
    model_state = model.state_dict()
    
    for name, param in state_dict.items():
        if name in model_state:
            model_state[name].copy_(param)
        else:
            print(f"Warning: {name} not found in model")
    
    model.load_state_dict(model_state, strict=False)
```

apply_lora_to_model은 이름 부분 문자열이 일치하는 nn.Linear를 바꾼다. q_proj·v_proj라는 이름이 없는 모델이나 결합 QKV·다른 레이어 종류는 자동 지원하지 않는다. 예상 레이어를 찾았는지와 중복 적용·의도하지 않은 이름 일치를 확인해야 한다. 이 함수는 감싼 레이어만 동결하고 모델의 나머지 파라메터는 동결하지 않는다.

```python
def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """가장 좋게 하개에 쓸 LoRA 파라메터만 얻는다."""
    params = []
    for name, param in model.named_parameters():
        if 'lora_' in name:
            params.append(param)
    return params

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """익힐 수 있는 파라메터와 전체 파라메터를 센다."""
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
    """LoRA만 빼고 모든 파라메터를 얼린다."""
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

freeze_non_lora를 별도로 호출해야 전체 기반의 requires_grad가 꺼진다. optimizer에 LoRA만 넣는 것과 기반 전체의 경사 계산·저장을 끄는 것은 다르다. frozen 기반도 중간 어댑터까지 경사를 전달하는 데 필요한 연산·활성화를 사용할 수 있어, 모든 기반 순전파를 no_grad로 감싸는 것 역시 일반적인 LoRA 학습의 대체가 아니다.

```python
# 흔한 자리매김
LORA_CONFIGS = {
    'minimal': LoRAConfig(rank=8, target_modules={'q_proj'}),
    'standard': LoRAConfig(rank=8, target_modules={'q_proj', 'v_proj'}),
    'extended': LoRAConfig(rank=8, target_modules={'q_proj', 'k_proj', 'v_proj', 'o_proj'}),
    'full': LoRAConfig(rank=16, target_modules={'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}),
}
```

설정 목록은 적용 위치의 예다. q·v만 적용하는 구성과 어텐션·MLP까지 적용하는 구성의 품질 우열은 데이터와 학습 예산에 달려 있다. 모듈 수·차원별로 추가 파라메터와 메모리를 계산하고 독립 검증 자료로 선택한다.

## 4. 병합과 어댑터 저장

추론에서 드롭아웃이 꺼진 선형 갱신은 W₀+(α/r)BA로 합쳐 추가 저랭크 연산을 제거할 수 있다. 비활성 경로를 제거한 배포와 래퍼만 남긴 상태의 실제 지연은 따로 측정한다. 학습 모드의 드롭아웃 경로와 병합 경로가 같다고 가정하지 않는다. 양자화 모델의 병합·재양자화도 별도 수치 폴리시가 필요하다.

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

현재 get_lora_state_dict는 state_dict 텐서의 저장소를 복제하지 않는다. LoRAManager.save_adapter의 메모리 사전은 이후 학습·로드에 따라 같이 바뀔 수 있어 독립 스냅샷이 아니다. 보존하려면 복제·장치·기반 모델 버전·랭크·α·대상 레이어 메타데이터를 함께 관리해야 한다.

병합한 상태에서 A·B를 바꾸거나 학습한 뒤 현재 행렬로 unmerge하면 원래 델타를 정확히 제거하지 못할 수 있다. 반복 가감의 반올림 오차와 merged 플래그의 저장·복원도 고려해야 한다. 일반 torch.save가 디스크에 기록한 상태와 복제 없는 메모리 참조를 구분한다.

## 5. 다른 스케일과 러닝레이트

```python
def get_lora_plus_params(model: nn.Module, lr: float, lr_ratio: float = 16.0):
    """
    A와 B에 다른 러닝 빠르기를 주는 LoRA+ 파라메터 무리.
    
    B는 0으로 시작하므로 러닝 빠르기를 더 크게 준다.
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

LoRA+ 예는 A·B를 다른 러닝레이트 그룹으로 나눈다. lr_ratio=16은 예시 선택이며 항상 더 낫거나 동일 품질을 보장하지 않는다.

```python
class rsLoRALayer(LoRALayer):
    """제곱근 잣수를 쓴 계수 안정 LoRA."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 잣수에 r 대신 sqrt(r)을 쓴다
        self.scaling = self.alpha / math.sqrt(self.rank)
```

[rsLoRA](https://arxiv.org/abs/2312.03732)는 여기서 α/√r 스케일을 사용한다. 이 변형과 기존 α/r의 비교는 랭크·α·러닝레이트·자료를 명시해야 한다. 스케일 이름만으로 수치 안정성이나 일반화가 보장되지는 않는다.

## 6. 학습 함수의 범위

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
    
    # 가장 좋게 하개를 마련한다(LoRA 파라메터만)
    lora_params = get_lora_parameters(model)
    optimizer = torch.optim.AdamW(lora_params, lr=learning_rate)
    
    # 파라메터 개수 세기
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

train_lora는 모델을 불러오고 레이어를 감싼 뒤 LoRA 파라메터만 optimizer로 갱신한다. 그러나 freeze_non_lora를 호출하지 않아 나머지 기반의 경사가 계산·누적될 수 있다. optimizer.zero_grad도 그 optimizer에 들어간 파라메터만 초기화하므로 의도한 메모리 절감과 경사 상태를 확인해야 한다.

입력 배치는 모델이 요구하는 labels·마스크와 장치를 갖춰야 한다. 배치 평균 로스의 단순 평균은 유효 토큰 수 가중 평균과 다를 수 있고 빈 loader도 처리해야 한다. 최종 모델은 마지막 epoch를 반환하며 최적 검증 체크포인트 선택이나 학습 재개 상태는 저장하지 않는다. main은 config만 만들고 이 함수를 호출하지 않는다. 이 편집에서는 다운로드·학습·병합을 실행하지 않았다.

## 7. 메모리와 QLoRA

700억 파라메터의 2바이트 웨이트는 140GB다. 같은 크기의 2바이트 경사는 140GB, FP32 Adam 모멘트 두 벌은 각각 280GB로 합계 560GB다. 이 세 부분만 840GB이며 FP32 master weight·활성화·임시 버퍼와 분할 저장 폴리시에 따라 달라진다. 특정 GPU 수를 이 계산만으로 고정할 수 없다.

[QLoRA](https://arxiv.org/abs/2305.14314)는 양자화한 동결 기반을 통해 경사를 전달하며 LoRA를 학습하는 접근이다. NF4·양자화 상수의 이중 양자화·paged optimizer를 제안했다. 4비트의 원시 700억 웨이트 비트 수는 약 35GB지만 메타데이터·비양자화 부분·어댑터·활성화 등을 합한 총메모리는 더 크다. 단순한 LoRA가 기반까지 4비트로 만드는 것은 아니며 이 페이지에는 QLoRA 구현이 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
4096×4096 레이어의 rank 8 LoRA 웨이트 수와 모델 전체 절감률을 구분하라.

</div>

??? success "연습문제 1 풀이"

    기반 행렬은 16,777,216개, 추가 행렬은 8×(4096+4096)=65,536개로 해당 행렬 대비 1/256이다. 기반은 그대로 저장되며 모델 전체 절감률은 적용 레이어·공유·추가 head와 학습 범위에 따라 다르다. 랭크만으로 품질 동등성을 보장하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
LoRA 병합이 학습 중 드롭아웃과 다중 어댑터 교체에서 요구하는 조건을 설명하라.

</div>

??? success "연습문제 2 풀이"

    eval의 선형 갱신은 기반에 합칠 수 있지만 학습 중 랜덤 dropout(x)의 경로와는 일반적으로 같지 않다. 병합을 풀 때는 실제로 더했던 델타를 사용해야 하고 그 사이 어댑터를 바꾸면 안 된다. 메모리 어댑터 스냅샷은 복제·구성·기반 버전을 보존해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
700억 파라미터에서 2바이트 웨이트·경사와 FP32 Adam 모멘트 두 벌의 메모리를 계산하라.

</div>

??? success "연습문제 3 풀이"

    각각 140GB·140GB·560GB로 합계 840GB다. FP32 master weight를 쓰면 280GB가 더해질 수 있고 활성화·임시 버퍼·분할 저장은 별도다. LoRA는 학습 가능한 부분의 경사·옵티마이저 상태를 줄이지만 기반 저장과 전체 활성화를 모두 없애지 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
현재 train_lora와 QLoRA, 기반 동결과 optimizer 선택의 차이를 설명하라.

</div>

??? success "연습문제 4 풀이"

    현재 함수는 양자화하지 않으므로 QLoRA가 아니다. LoRA 파라메터만 optimizer에 넣지만 감싸지 않은 기반의 requires_grad는 그대로 남아 경사 계산·누적이 가능하다. 올바른 동결·학습 상태와 장치·dtype·labels를 따로 구성해야 원하는 메모리 절감을 얻는다.



## 정리하며

LoRA는 동결 기반의 저랭크 갱신을 학습한다. 행렬 표기·전체 동결·스냅샷·병합 상태와 실제 메모리를 구분하고 랭크·스케일·적용 범위의 효과는 같은 조건에서 검증해야 한다.

**참고 문헌**

1. Hu, E., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
2. Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs."
3. Hayou, S., et al. (2024). "LoRA+: Efficient Low Rank Adaptation of Large Models."
4. Kalajdzievski, D. (2023). "Rank-Stabilized LoRA: Unlocking the Potential of LoRA Fine-Tuning."
