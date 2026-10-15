# 어댑터 층

[어댑터](https://arxiv.org/abs/1902.00751)는 사전 학습 모델의 표현에 작은 학습 모듈을 추가하는 파라미터 효율적 전이 방법이다. 기반을 공유하면서 과제별 추가 가중치를 저장할 수 있다. 삽입 위치·학습 범위·추론 비용은 구현에 따라 다르며, 작은 모듈을 선언하는 것만으로 기반 동결이나 과제 수행이 완성되지는 않는다.

## 1. 병목과 잔차

차원 d의 입력을 병목 r로 줄이고 비선형 변환 후 d로 복원해 입력에 더한다. 편향을 포함한 한 어댑터의 매개변수는 2dr+r+d다. 층마다 두 개씩 L층에 넣으면 2L(2dr+r+d)지만 head·정규화 추가 학습은 별도로 센다.

```python
import torch
import torch.nn as nn
from typing import Optional, List

class Adapter(nn.Module):
    """
    맞춤개 모듈: 내림 쏘기 → 활성화 → 올림 쏘기 + 남는 이음.
    
    인수:
        input_dim: 들임/내놓기 차원(모델의 숨은 크기)
        bottleneck_dim: 병목 차원(보통 64~256)
        activation: 활성화 함수
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

코드는 업 프로젝션을 작은 정규분포로 초기화하므로 대체로 항등 함수에 가까운 출발을 의도한다. 잔차가 있다는 이유만으로 정확히 항등인 것은 아니다. ScaledAdapter의 학습 스칼라 초기값도 0이 아니라 1이다. 알 수 없는 activation 문자열은 오류 대신 SiLU로 처리한다.

d=768, r=64이면 한 모듈 99,136개, L=12에 두 개씩이면 2,379,264개다. 1억 1000만 기반 대비 약 2.16%이며 어떤 위치에 몇 개를 넣는지에 따라 달라진다. 병목 64가 언제나 같은 비율이나 최적 성능을 뜻하지 않는다.

## 2. 직렬 삽입

```python
class SerialAdapterTransformerLayer(nn.Module):
    """
    밑층마다 뒤에 맞춤개를 잇달아 둔 변환기 층.
    
    구조:
        x → 어텐션 → 맞춤개 → 더하고 고르게 → 앞먹임 그물 → 맞춤개 → 더하고 고르게
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
        # 어텐션 + 맞춤개
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=attention_mask)
        attn_out = self.adapter_attn(attn_out)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 앞먹임 그물 + 맞춤개
        ffn_out = self.ffn(x)
        ffn_out = self.adapter_ffn(ffn_out)
        x = self.norm2(x + ffn_out)
        
        return x
```

SerialAdapterTransformerLayer는 어텐션·FFN의 출력에 어댑터를 적용하고 외부 잔차·LayerNorm으로 연결한다. 새로운 트랜스포머 층을 만드는 예이며 기존 사전 학습 가중치를 이식하거나 기반을 동결하지 않는다. 원 논문의 모든 세부·학습 정책을 그대로 재현하는 클래스도 아니다.

attention_mask는 nn.MultiheadAttention의 key_padding_mask로 전달하므로 bool True가 무시할 위치다. Hugging Face의 1=유효 토큰 마스크를 그대로 넘기는 계약과 다르다. 인과 마스크가 없어 양방향 인코더 형태이며 자기회귀 모델에 그대로 쓰면 미래 토큰을 볼 수 있다.

## 3. 병렬 삽입

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
        # 나란히: 맞춤개와 어텐션이 모두 x를 받는다
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=attention_mask)
        adapter_out = self.adapter_attn(x) - x  # 남는 이음을 뺀다
        x = self.norm1(x + self.dropout(attn_out) + self.adapter_scale * adapter_out)
        
        # 나란히: 맞춤개와 앞먹임 그물이 모두 x를 받는다
        ffn_out = self.ffn(x)
        adapter_out = self.adapter_ffn(x) - x
        x = self.norm2(x + ffn_out + self.adapter_scale * adapter_out)
        
        return x
```

ParallelAdapterTransformerLayer는 서브층과 나란히 입력에 대한 어댑터 갱신을 계산한다. Adapter 자체가 입력 잔차를 포함하므로 adapter(x)−x를 사용해 갱신만 더한다. 직렬과 병렬은 적용하는 표현·계산 경로가 다르며 어느 쪽이 항상 빠르거나 정확하다고 단정하지 않는다. 실제 효과는 삽입 위치·병목·드롭아웃·실행 환경에서 비교한다.

## 4. 여러 어댑터의 결합

[AdapterFusion](https://aclanthology.org/2021.eacl-main.39/)은 여러 과제로 학습한 어댑터의 정보를 결합하는 접근이다. 다음 코드는 입력별 어댑터 갱신에 어텐션 가중치를 주는 단순화된 모듈이다.

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
        
        # 녹여 붙이는 어텐션
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
        
        # 쌓기: [배치, 차례, 맞춤개 수, 숨은]
        adapter_stack = torch.stack(adapter_outputs, dim=2)
        
        # 맞춤개에 대한 어텐션
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

어댑터들은 동결하고 query·key·value 사영과 temperature를 학습한다. 하지만 requires_grad=False는 dropout을 eval로 바꾸지 않으므로 고정 어댑터도 train 모드에서 무작위 출력을 낼 수 있다. temperature는 제약 없는 Parameter여서 0이나 음수가 될 수 있고 빈 어댑터 목록도 처리하지 않는다.

가중치의 소프트맥스 합이 1이어도 value 사영과 bias를 거친 결과가 원래 입력과 같다는 보장은 없다. 모든 어댑터 갱신이 0이어도 value bias가 남을 수 있다. 기반·과제 어댑터·fusion의 학습 단계와 모드를 실제로 분리해야 한다.

## 5. 과제별 모듈 선택

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

MultiTaskAdapterModel은 기반 출력 뒤에 과제별 어댑터 하나를 붙인다. 모든 트랜스포머 층 내부에 과제별 어댑터를 삽입하는 모델이 아니다. base_model(x)가 바로 텐서를 반환한다고 가정하므로 Hugging Face 출력 객체에는 그대로 맞지 않는다. 과제별 분류 head·손실도 없다.

set_task는 과제별 requires_grad를 바꾸지만 forward(task=...)만 호출하면 그 전환을 수행하지 않는다. 현재 활성 과제와 실제 경사 대상이 다를 수 있어 학습·선택을 일치시켜야 한다. 기반 가중치를 동결해도 train 모드의 dropout과 상태 버퍼 갱신까지 막는 것은 아니다.

## 6. 학습 함수와 저장 범위

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

train_with_adapters는 기반의 마지막 표현을 no_grad로 얻고 그 뒤 어댑터만 학습하려는 예다. compute_loss가 정의되지 않았고 과제 head도 없어 완성된 학습 함수가 아니다. 기반 모델의 장치·eval 모드 설정도 없고 batch 전체에 labels를 포함해 넘기므로 encoder의 지원 인자와 맞지 않을 수 있다.

이런 마지막 어댑터는 기반 전체 no_grad로 학습할 수 있지만 중간 층에 학습 모듈을 삽입한 모델은 뒤쪽 기반 연산을 통해 경사를 전달해야 한다. 동결과 계산 그래프 제외를 혼동하지 않는다. 빈 loader·손실 집계·검증·최적 체크포인트와 재개 상태도 별도로 구성해야 한다.

저장 함수는 어댑터의 state_dict만 기록한다. 기반 버전·차원·활성화·삽입 위치·head·레이블과 기타 설정은 별도로 보존해야 한다. 기존 코드의 학습·저장 함수를 이번 편집에서 실행하지 않았다.

## 7. 비용과 방법 선택

어댑터는 비선형 층을 추가하므로 일반적으로 단일 기반 행렬에 합칠 수 없다. LoRA는 조건이 맞는 선형 갱신을 병합할 수 있고 prefix 방식은 어텐션의 추가 문맥을 사용한다. 매개변수 비율·추론 지연·전체 학습 메모리는 구조·길이·배치·dtype에 따라 달라진다.

병목·위치·학습률의 고정 추천표보다 실제 추가 매개변수와 품질·지연·과제 교체 비용을 같은 조건에서 비교한다. 어댑터가 많아지면 저장·버전·로딩과 과제별 평가도 관리해야 한다. 어떤 방법도 모든 과제에서 전체 파인튜닝과 같은 품질을 보장하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
d=768, r=64인 어댑터와 12층에 두 개씩 넣은 경우의 추가 매개변수를 계산하라.

</div>

??? success "연습문제 1 풀이"

    한 어댑터는 2×768×64+64+768=99,136개이고 24개는 2,379,264개다. 1억 1000만 기반 대비 약 2.16%이며 별도 head·정규화 학습이나 ScaledAdapter 스칼라는 추가로 센다. 작은 초기값은 근사 항등을 의도하지만 정확한 항등을 보장하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
직렬과 병렬 어댑터에서 입력과 잔차가 다른 점을 설명하라.

</div>

??? success "연습문제 2 풀이"

    직렬은 서브층 출력에 어댑터를 적용하고 병렬은 서브층과 같은 입력에서 갱신을 계산해 더한다. Adapter가 x를 포함해 반환하므로 병렬에서는 adapter(x)−x로 갱신만 추출한다. 단순히 둘을 바꾸어도 같은 함수라는 보장은 없으며 실제 연결·모드·마스크를 확인해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
동결한 어댑터와 eval 모드의 차이 및 AdapterFusion의 temperature 문제를 설명하라.

</div>

??? success "연습문제 3 풀이"

    requires_grad=False는 매개변수 경사 학습을 막지만 dropout의 난수 동작이나 일부 모듈의 버퍼 갱신을 끄지 않는다. eval 모드는 별도다. temperature도 제약 없이 학습하면 0 나눗셈이나 음수 스케일이 가능하므로 유효한 양수 매개변수화·검증을 설계해야 한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
train_with_adapters가 완성된 과제별 파인튜닝 함수가 아닌 이유를 설명하라.

</div>

??? success "연습문제 4 풀이"

    손실 함수·head가 없고 기반의 장치·입력 키·모드와 검증이 구성되어 있지 않다. 마지막 표현 뒤 어댑터와 중간 삽입 방식의 경사 경로도 다르다. state_dict 외에 기반·삽입·레이블 설정을 함께 보존해야 재현 가능한 과제 모델이 된다.



## 정리하며

어댑터는 작은 추가 모듈로 기반 표현을 조정한다. 삽입 경로·기반 동결·실행 모드·마스크·과제 head와 저장 구성을 맞추고 실제 품질·메모리·지연으로 방법을 비교한다.

**참고 문헌**

1. Houlsby, N., et al. (2019). "Parameter-Efficient Transfer Learning for NLP." ICML.
2. Pfeiffer, J., et al. (2020). "AdapterHub: A Framework for Adapting Transformers." EMNLP.
3. Pfeiffer, J., et al. (2021). "AdapterFusion: Non-Destructive Task Composition." EACL.
4. He, J., et al. (2022). "Towards a Unified View of Parameter-Efficient Transfer Learning." ICLR.
