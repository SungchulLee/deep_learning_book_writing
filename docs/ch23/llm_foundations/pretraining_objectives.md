# 대규모 언어 모델의 사전 학습 목표

사전 학습 목표는 어떤 입력에서 무엇을 예측할지 정한다. 같은 텍스트도 다음 토큰 예측, 마스킹 복원, 구간 복원이나 교체 탐지의 자료로 구성할 수 있다. 목표는 구조·계산비용·전이 성능과 상호작용하지만 생성과 이해 능력을 혼자 결정하지는 않는다. 이 페이지는 대표 목적과 부분 구현을 비교한다.

## 1. 인과 언어 모델링

인과 언어 모델링(CLM)은 다음 토큰 조건부 확률로 시퀀스 확률을 분해한다.

$$P_\theta(x_{1:n})=\prod_{t=1}^n P_\theta(x_t\mid x_{<t}),\qquad
\mathcal L_{\mathrm{CLM}}=-\sum_t\log P_\theta(x_t\mid x_{<t}).$$

첫 토큰의 시작 조건과 실제 로스에 포함할 위치를 정의해야 한다. 디코더 전용 구조에서 흔히 사용하지만 구조만으로 목적이 고정되는 것은 아니다. FIM처럼 순서를 바꾼 학습 자료도 인과적인 다음 토큰 예측에 사용할 수 있다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def create_causal_mask(seq_len: int) -> torch.Tensor:
    """
    인과 주의 가림을 만든다.
    
    자리 i는 i 이하 자리만 볼 수 있다.
    
    반환값:
        아래 세모 가림막 (차례 길이, 차례 길이)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask  # 1 = 본다, 0 = 가린다

# 차례 길이 4의 보기:
# [[1, 0, 0, 0],
#  [1, 1, 0, 0],
#  [1, 1, 1, 0],
#  [1, 1, 1, 1]]
```

create_causal_mask는 1을 허용, 0을 차단으로 표시한 개념적 행렬이다. 이 float 행렬을 PyTorch 어텐션의 가산 마스크로 그대로 넣으면 0이 차단을 뜻하지 않는다. API별로 bool의 True 의미나 0/−∞ 가산 규칙을 확인해 변환해야 한다.

```python
class CausalLMHead(nn.Module):
    """인과 말 나타내기 머리."""
    
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        labels: torch.Tensor = None
    ):
        """
        인수:
            hidden_states: (배치, 차례 길이, 숨은 크기)
            labels: (배치, 차례 길이) - 다음 토막 어림을 위해 1만큼 밀림
        """
        logits = self.lm_head(hidden_states)  # (배치, seq_len, vocab_size)
        
        loss = None
        if labels is not None:
            # 다음 토큰 맞히기를 위해 민다
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return {'loss': loss, 'logits': logits}

def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    홀로 서는 인과 말 나타내기 로스.
    
    인수:
        logits: [배치, 차례 길이, 낱말 곳간 크기]
        labels: [배치, 차례 길이]
    """
    # 다음 토큰 맞히기를 위해 민다
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100
    )
```

head는 이미 계산된 은닉 표현을 어휘 로짓으로 바꾸고 입력과 같은 정렬의 labels를 한 칸 이동해 로스를 계산한다. encoder나 인과 마스크 자체를 적용하는 모듈은 아니다. 이미 이동한 목표를 다시 전달하면 두 번 이동하므로 호출 계약을 맞춘다. −100은 로스 제외 ID이며 유효 목표가 하나도 없는 경우도 처리해야 한다.

## 2. 마스킹 언어 모델링

MLM은 일부 토큰 위치 M을 골라 교란한 입력 x̃에서 원래 토큰을 예측한다.

$$\mathcal L_{\mathrm{MLM}}=-\sum_{i\in M}\log P_\theta(x_i\mid\tilde x).$$

BERT 방식은 일반적으로 적격 토큰의 약 15%를 목표로 선택하고 그중 80%를 MASK, 10%를 무작위 토큰으로 바꾸며 10%는 그대로 둔다. 입력을 그대로 두는 목표도 있어 항상 순수하게 ‘나머지 토큰만’ 조건으로 보는 식과는 다르다. 이 비율이 모든 MLM의 고정 규칙은 아니다.

```python
import torch
import random
from typing import Tuple

def bert_masking(
    tokens: list,
    mask_token: int,
    vocab_size: int,
    mask_prob: float = 0.15
) -> Tuple[list, list]:
    """
    BERT 꼴 가리기: 토막의 15%를 고친다.
    
    가린 토막 가운데:
    - 80%는 [MASK]으로 바꾼다
    - 10%는 무작위 토큰으로 바꾼다
    - 10%는 그대로 둔다
    """
    masked_tokens = tokens.copy()
    labels = [-100] * len(tokens)  # -100 = 로스에서 무시한다
    
    for i in range(len(tokens)):
        if random.random() < mask_prob:
            labels[i] = tokens[i]  # 본디 토막이 이름표이다
            
            r = random.random()
            if r < 0.8:
                masked_tokens[i] = mask_token
            elif r < 0.9:
                masked_tokens[i] = random.randint(0, vocab_size - 1)
            # 그 밖에는 본디 것을 둔다(10%)
    
    return masked_tokens, labels

def create_mlm_batch(
    input_ids: torch.Tensor,
    vocab_size: int,
    mask_token_id: int,
    mask_prob: float = 0.15,
    special_token_ids: set = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    알맞은 텐서 연산으로 가린 말 모델 익히기 배치를 만든다.
    
    인수:
        input_ids: [배치, 차례 길이] 들임 토막 번호
        vocab_size: 낱말 곳간의 크기
        mask_token_id: [MASK] 토막의 번호
        mask_prob: 토막마다 가릴 확률
        special_token_ids: 결코 가리지 않을 토막 번호의 모임(예: [CLS], [SEP], [PAD])
    """
    labels = input_ids.clone()
    
    # 확률 행렬을 만든다
    probability_matrix = torch.full(input_ids.shape, mask_prob)
    
    # 특별 토큰은 가리지 않는다
    if special_token_ids:
        for token_id in special_token_ids:
            probability_matrix.masked_fill_(input_ids == token_id, 0.0)
    
    # 가릴 번호를 뽑는다
    masked_indices = torch.bernoulli(probability_matrix).bool()
    
    # 가린 토막에서만 로스를 셈한다
    labels[~masked_indices] = -100
    
    # 80% -> [MASK]
    indices_replaced = torch.bernoulli(
        torch.full(input_ids.shape, 0.8)
    ).bool() & masked_indices
    input_ids[indices_replaced] = mask_token_id
    
    # 10% -> 아무 토막
    indices_random = torch.bernoulli(
        torch.full(input_ids.shape, 0.5)
    ).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(vocab_size, input_ids.shape, dtype=input_ids.dtype)
    input_ids[indices_random] = random_words[indices_random]
    
    # 10% -> 그대로(고치지 않는 것으로 이미 다룸)
    
    return input_ids, labels
```

bert_masking은 특수 토큰을 제외하지 않는다. create_mlm_batch는 special_token_ids를 주면 해당 위치를 선택에서 제외하지만 입력 텐서를 직접 수정한다. 확률·난수 텐서는 CPU에 생성하므로 GPU 입력에 그대로 쓸 수 없고, 무작위 대체 토큰에 특수 ID가 들어갈 수도 있다. 목표가 하나도 선택되지 않은 배치의 로스도 정의해야 한다.

로스를 선택 위치에서만 계산하더라도 주변 입력 표현과 공유 파라메터에는 역전파가 흐른다. ‘15%만 그래디언트를 받는다’와 ‘15%를 직접 복원 목표로 삼는다’는 다르다. MLM은 인과 문장 우도와 같지 않으며 생성에는 목적에 맞는 별도 절차가 필요하다.

## 3. 구간 손상과 복원

[T5](https://arxiv.org/abs/1910.10683)는 연속 구간을 서로 구분되는 sentinel로 대체하고 그 구간들을 디코더 목표로 구성한다. 다음 도식은 한 구간의 개념 예다.

```
Input:  "The quick brown [X] the lazy dog"
Target: "[X] fox jumps over"
```

BART의 text infilling은 같은 MASK로 손상하고 원문 전체를 복원하는 구성이므로 T5의 구간별 sentinel 목표와 동일하지 않다. 로스는 손상 입력에 조건을 건 목표 시퀀스의 음의 자기회귀 로그우도다.

```python
import numpy as np
from typing import List, Tuple

def span_corruption(
    tokens: List[int],
    sentinel_start_id: int,
    mean_span_length: float = 3.0,
    corruption_rate: float = 0.15
) -> Tuple[List[int], List[int]]:
    """
    T5 꼴 구간 망가뜨리기.
    
    인수:
        tokens: 들임 토막 번호
        sentinel_start_id: 파수 토막의 첫 번호([X], [Y], ...)
        mean_span_length: 망가뜨린 구간의 평균 길이
        corruption_rate: 망가뜨릴 토막의 비율
        
    반환값:
        (망가뜨린 들임, 목표)의 짝
    """
    n = len(tokens)
    num_to_corrupt = int(n * corruption_rate)
    
    if num_to_corrupt == 0:
        return tokens, []
    
    # 구간의 수와 길이를 정한다
    num_spans = max(1, int(num_to_corrupt / mean_span_length))
    
    # 기하 분포에서 구간 길이를 뽑는다
    span_lengths = np.random.geometric(1.0 / mean_span_length, num_spans)
    span_lengths = np.clip(span_lengths, 1, n // num_spans)
    
    # 목표 망가뜨림에 맞게 조절한다
    total_length = span_lengths.sum()
    if total_length > num_to_corrupt:
        span_lengths = (span_lengths * num_to_corrupt / total_length).astype(int)
        span_lengths = np.maximum(span_lengths, 1)
    
    num_spans = len(span_lengths)
    
    # 겹치지 않는 구간 자리를 뽑는다
    # 차례를 num_spans 도막으로 나누고 도막마다 시작점 하나를 뽑는다
    segment_length = n // num_spans
    span_starts = []
    for i in range(num_spans):
        start = i * segment_length
        end = min((i + 1) * segment_length - span_lengths[i], n - span_lengths[i])
        if start < end:
            span_starts.append(np.random.randint(start, end))
        else:
            span_starts.append(start)
    
    # 구간을 자리로 정렬한다
    spans = sorted(zip(span_starts, span_lengths))
    
    # 망가뜨린 들임과 목표를 세운다
    input_tokens = []
    target_tokens = []
    sentinel_id = sentinel_start_id
    pos = 0
    
    for start, length in spans:
        # 이 구간 앞의 토막을 더한다
        input_tokens.extend(tokens[pos:start])
        
        # 들임에 파수 토막을 더한다
        input_tokens.append(sentinel_id)
        
        # 목표에 파수 토막 + 본디 구간을 더한다
        target_tokens.append(sentinel_id)
        target_tokens.extend(tokens[start:start + length])
        
        sentinel_id += 1
        pos = start + length
    
    # 남은 토막을 들임에 더한다
    input_tokens.extend(tokens[pos:])
    
    # 목표에 마지막 파수 토막을 더한다
    target_tokens.append(sentinel_id)
    
    return input_tokens, target_tokens

# 사용 예
if __name__ == "__main__":
    tokens = list(range(20))  # [0, 1, 2, ..., 19]
    corrupted, target = span_corruption(tokens, sentinel_start_id=100)
    print(f"Original: {tokens}")
    print(f"Corrupted: {corrupted}")
    print(f"Target: {target}")
```

현재 span_corruption은 geometric 길이와 구역별 시작 위치를 사용하는 단순화 예다. 요청한 손상률과 실제 손상 토큰 수가 정확히 일치하지 않을 수 있으며 T5의 전처리를 그대로 재현하지 않는다. sentinel ID를 증가시키는 규칙도 실제 토크나이저의 예약 ID와 맞춰야 한다. 길이·손상률·평균 구간 길이와 sentinel 수를 검증하고 필요한 EOS를 구성해야 한다.

**기존 출력:**

```
Original: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
Corrupted: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 100, 16, 17, 18, 19]
Target: [100, 13, 14, 15, 101]
```

기존 출력은 특정 표본이며 코드가 난수 시드를 고정하지 않으므로 항상 같은 구간이 선택된다는 뜻은 아니다. 이 편집에서는 다시 실행하지 않았다.

## 4. Prefix 언어 모델링

prefix LM은 제공된 prefix 내부에 양방향 어텐션을 허용하고 suffix는 prefix와 이전 suffix 토큰만 본다. 로스는 보통 suffix에 계산한다.

```
Prefix (bidirectional): "Translate English to French:"
Generation (causal):    " Le chat est sur le tapis"
```

이는 prefix에 양방향 어텐션을 적용하는 **마스크·목표**다. encoder–decoder 구조의 T5 전체를 그대로 prefix LM의 정의로 삼거나, 파라메터 효율 튜닝인 prefix tuning과 혼동하지 않는다.

```python
def prefix_lm_mask(seq_len: int, prefix_len: int) -> torch.Tensor:
    """
    앞가지 말 모델의 어텐션 가림막.
    
    - 앞가지 토막: 양쪽 어텐션(앞가지 토막을 모두 본다)
    - 만들어 내기 토막: 인과 어텐션(앞가지 + 앞서 만든 것을 본다)
    
    인수:
        seq_len: 전체 차례 길이
        prefix_len: 앞가지 몫의 길이
        
    반환값:
        어텐션 가림막 (차례 길이, 차례 길이)
    """
    mask = torch.zeros(seq_len, seq_len)
    
    # 앞가지: 앞가지 안에서는 어텐션이 온전하다
    mask[:prefix_len, :prefix_len] = 1
    
    # 만들어 내기: 앞가지 전부 + 만들어 내기 안에서는 인과로 본다
    for i in range(prefix_len, seq_len):
        mask[i, :prefix_len] = 1  # 앞가지를 모두 본다
        mask[i, prefix_len:i+1] = 1  # 만들어 내기 안에서는 인과로
    
    return mask

# 보기: seq_len=6, prefix_len=3
# [[1, 1, 1, 0, 0, 0],   <- 앞가지 토막 0
#  [1, 1, 1, 0, 0, 0],   <- 앞가지 토막 1
#  [1, 1, 1, 0, 0, 0],   <- 앞가지 토막 2
#  [1, 1, 1, 1, 0, 0],   <- 만든 토막 0(앞가지 + 자신을 본다)
#  [1, 1, 1, 1, 1, 0],   <- 만든 토막 1
#  [1, 1, 1, 1, 1, 1]]   <- 만든 토막 2
```

prefix_lm_mask도 1/0 허용 표시이므로 실제 어텐션 API에 맞춰 변환해야 한다. 0≤prefix_len≤seq_len을 확인하고 suffix가 미래를 보거나 prefix가 suffix를 보는 경로를 만들지 않아야 한다.

## 5. 교체 토큰 탐지

[ELECTRA](https://arxiv.org/abs/2003.10555)는 작은 생성기로 만든 대체 토큰을 입력하고 판별기가 각 위치의 원래 토큰 여부를 구별하도록 학습한다. 여기서는 y_t=1을 원문과 다른 토큰, D_t를 그 확률로 정의하면 이진 교차 엔트로피를 사용할 수 있다. 원문과 같은 토큰을 샘플링한 위치는 바뀐 위치로 세지 않아야 한다.

```python
class ELECTRA(nn.Module):
    """
    ELECTRA: 글 인코더를 판별기로 미리 익히기.
    
    작은 생성기로 글을 망가뜨리고, 으뜸 모델이 망가진 곳을 알아내는 법을 배운다.
    모든 토막이 신호를 주므로 가린 말 모델보다 표본을 아낀다.
    """
    
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        gen_weight: float = 1.0,
        disc_weight: float = 50.0
    ):
        super().__init__()
        self.generator = generator  # 작은 가린 언어 모형
        self.discriminator = discriminator  # 주 모형
        self.gen_weight = gen_weight
        self.disc_weight = disc_weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        masked_indices: torch.Tensor,
        labels: torch.Tensor
    ):
        """
        인수:
            input_ids: [MASK] 토막이 든 들임
            masked_indices: 망가뜨린 자리의 참거짓 가림막
            labels: 가린 자리의 본디 토막 번호
        """
        # 생성기가 가린 토막을 어림한다(가린 말 모델)
        gen_logits = self.generator(input_ids).logits
        gen_loss = F.cross_entropy(
            gen_logits[masked_indices],
            labels[masked_indices]
        )
        
        # 생성기에서 바꿔 넣을 것을 뽑는다
        with torch.no_grad():
            gen_probs = F.softmax(gen_logits, dim=-1)
            sampled = torch.multinomial(
                gen_probs.view(-1, gen_probs.size(-1)), 1
            ).view(input_ids.shape)
        
        # 망가뜨린 차례를 만든다
        corrupted = input_ids.clone()
        corrupted[masked_indices] = sampled[masked_indices]
        
        # 판별기가 어느 토막이 바뀌었는지 어림한다
        disc_logits = self.discriminator(corrupted).logits
        disc_labels = (corrupted != input_ids).float()
        
        disc_loss = F.binary_cross_entropy_with_logits(
            disc_logits.squeeze(-1),
            disc_labels
        )
        
        total_loss = self.gen_weight * gen_loss + self.disc_weight * disc_loss
        
        return {
            'loss': total_loss,
            'gen_loss': gen_loss,
            'disc_loss': disc_loss
        }
```

현재 코드는 generator에 주는 input_ids와 corrupted를 비교해 disc_labels를 만든다. input_ids가 MASK를 포함한 교란 입력이라면 원문과의 비교가 아니므로 판별기 정답이 잘못된다. 반대로 원문을 그대로 주면 생성기의 마스킹 예측 입력이 아니다. 원문·교란 입력·선택 위치를 분리해야 한다.

판별기는 위치별 단일 로짓을 반환해야 하고 generator는 어휘 로짓을 반환해야 한다. attention·padding 마스크, 유효 로스 위치와 선택된 목표가 없는 경우는 별도로 처리해야 한다. 기본 disc_weight=50은 예시 웨이트이며 모든 데이터·로스 집계에서 최적이라는 뜻은 아니다. 샘플링 토큰에는 일반적인 역전파가 없고 코드도 no_grad로 분리한다.

## 6. 손상 목표의 혼합과 UL2

문장 순서 섞기·토큰 삭제·텍스트 회전 등으로 원문 복원을 학습할 수 있다. BART의 텍스트 회전은 회전량 숫자를 맞히는 분류가 아니라 손상된 순서에서 원문을 복원하는 목적이다. 어떤 정보가 입력에 남고 어떤 목표를 예측하는지 구분해야 한다.

[UL2](https://arxiv.org/abs/2205.05131)는 서로 다른 손상 정도와 순차 복원 등의 목적을 혼합한다. 다음 R·S·X 설정과 혼합 웨이트는 개념을 단순화한 예이며 전체 논문 설정을 재현한 구현은 아니다.

```python
from dataclasses import dataclass
from typing import Optional
import random

@dataclass
class UL2Config:
    """UL2 잡음 없애기 목표의 자리매김."""
    name: str
    mean_span_length: Optional[float]
    corruption_rate: Optional[float]
    prefix: str  # 방식 토막을 들임에 더한다

UL2_OBJECTIVES = [
    UL2Config('R', mean_span_length=3.0, corruption_rate=0.15, prefix='[R]'),   # 보통
    UL2Config('S', mean_span_length=None, corruption_rate=None, prefix='[S]'),  # 차례차례(앞가지 말 모델)
    UL2Config('X', mean_span_length=32.0, corruption_rate=0.50, prefix='[X]'),  # 극단
]

UL2_WEIGHTS = [0.5, 0.25, 0.25]  # 표집 무게

def sample_ul2_objective() -> UL2Config:
    """UL2 익히기 목표를 뽑는다."""
    return random.choices(UL2_OBJECTIVES, weights=UL2_WEIGHTS)[0]

def ul2_transform(
    tokens: List[int],
    sentinel_start_id: int,
    mode_token_ids: dict
) -> Tuple[List[int], List[int]]:
    """
    차례에 UL2 바꿈을 적용한다.
    
    인수:
        tokens: 들임 토막 번호
        sentinel_start_id: 파수 토막의 첫 번호
        mode_token_ids: 방식 이름('R', 'S', 'X')을 토막 번호에 대응시킨 사전
    """
    config = sample_ul2_objective()
    
    if config.name == 'S':
        # 차례차례(앞가지 말 모델): 앞가지와 목표로 쪼갠다
        split_point = random.randint(len(tokens) // 4, 3 * len(tokens) // 4)
        input_tokens = [mode_token_ids['S']] + tokens[:split_point]
        target_tokens = tokens[split_point:]
    else:
        # R이나 X: 파라메터를 달리한 구간 망가뜨리기
        corrupted, target = span_corruption(
            tokens,
            sentinel_start_id,
            mean_span_length=config.mean_span_length,
            corruption_rate=config.corruption_rate
        )
        input_tokens = [mode_token_ids[config.name]] + corrupted
        target_tokens = target
    
    return input_tokens, target_tokens
```

실제 학습에는 mode 토큰·sentinel·EOS·길이 예산·디코더 목표의 이동과 마스크를 일치시켜야 한다. 빈 입력이나 손상이 전혀 없는 짧은 입력에서 비어 있는 목표를 어떻게 다룰지도 필요하다. 이 함수는 자료 변환이며 학습 루프를 포함하지 않는다.

## 7. 가운데 채우기

[FIM](https://arxiv.org/abs/2207.14255)은 prefix·middle·suffix로 나눈 자료를 재배열해 뒤 문맥도 제공한 채 가운데를 생성하도록 학습한다. 변환한 시퀀스에서 인과적인 다음 토큰 예측을 유지할 수 있다.

```python
def fill_in_middle_transform(
    code: str,
    fim_rate: float = 0.5,
    fim_spm_rate: float = 0.5
) -> str:
    """
    가운데 채우기 익히기를 위해 부호를 바꾼다.
    
    두 가지 꼴:
    - PSM(앞-뒤-가운데): <PRE>앞<SUF>뒤<MID>가운데
    - SPM(뒤-앞-가운데): <SUF>뒤<PRE>앞<MID>가운데
    
    인수:
        code: 본디 부호 글줄
        fim_rate: 가운데 채우기를 적용할 확률(여느 인과 말 모델 대비)
        fim_spm_rate: 가운데 채우기를 쓸 때 SPM 꼴이 될 확률
    """
    if random.random() > fim_rate:
        return code  # 여느 인과 말 모델
    
    # 아무 데나 자르는 점
    split = random.randint(0, len(code))
    prefix = code[:split]
    
    # 고를 수 있음: 가운데의 끝점을 아무 데나
    if random.random() < 0.5:
        end = random.randint(split, len(code))
    else:
        end = len(code)
    
    middle = code[split:end]
    suffix = code[end:]
    
    # 꼴을 고른다
    if random.random() < fim_spm_rate:
        # SPM 꼴
        return f"<SUF>{suffix}<PRE>{prefix}<MID>{middle}"
    else:
        # PSM 꼴
        return f"<PRE>{prefix}<SUF>{suffix}<MID>{middle}"

# 보기:
# 본디: "def foo():\n    return 42"
# FIM PSM:  "<PRE>def foo():\n<SUF>\n<MID>    return 42"
```

현재 함수는 문자 위치에서 자르고 사용자 정의 PRE·SUF·MID 문자열을 사용한다. 이 문자열이 기존 체크포인트의 특수 토큰이나 표준 FIM 직렬화와 자동으로 호환되는 것은 아니다. 절반의 분기에서 end를 원문 끝으로 정하므로 suffix가 비는 사례도 많다. 비율·분할·토큰화·표식 규칙을 학습과 추론에서 일치시켜야 한다.

## 8. 학습 신호와 계산 효율

```python
def effective_tokens_per_example(
    seq_len: int,
    objective: str,
    mask_rate: float = 0.15
) -> float:
    """
    차례마다의 실효 익히기 신호를 셈한다.
    
    모든 목표가 토막마다 그래디언트 신호를 주지는 않는다.
    """
    if objective == 'CLM':
        # 첫 토막만 빼고 모든 토막이 신호를 준다
        return seq_len - 1
    
    elif objective == 'MLM':
        # 가린 토막만 신호를 준다
        return seq_len * mask_rate
    
    elif objective == 'span_corruption':
        # 가린 말 모델과 비슷하나 맥락이 더 낫다
        return seq_len * mask_rate
    
    elif objective == 'prefix_lm':
        # 만들어 내는 몫만 신호를 준다
        # 앞가지가 50%쯤이라 치고
        return seq_len * 0.5
    
    elif objective == 'ELECTRA':
        # 모든 토막이 판별기 신호를 준다
        return seq_len

def training_equivalence(clm_tokens: int, mlm_mask_rate: float = 0.15) -> dict:
    """
    인과 말 모델의 익히기 신호에 맞추려면 가린 말 모델 토막이 얼마나 필요한지 셈한다.
    """
    return {
        'clm_effective': clm_tokens,
        'mlm_effective_per_token': mlm_mask_rate,
        'mlm_tokens_to_match': clm_tokens / mlm_mask_rate,
        'ratio': 1 / mlm_mask_rate  # 가린 말 모델 토막이 약 6.7배 더 필요하다
    }
```

이 함수들은 직접 로스 위치 수를 거칠게 세는 예다. CLM의 이동된 목표, MLM의 기대 선택 수, ELECTRA의 이진 판별은 난이도·정보량·연산이 서로 다르다. 구간 복원 목표에는 sentinel 등 추가 토큰도 들어간다. 1/0.15≈6.67배 위치가 필요하다는 계산을 같은 품질·FLOPs·학습시간의 필요 비율로 해석하지 않는다.

목표 비교는 같은 자료·구조·계산 예산·전이 평가에서 수행해야 한다. 사전 학습만으로 채팅 지시 준수·사실성·안전성이 완성되지는 않으며 데이터 구성과 후속 학습·평가도 함께 필요하다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
인과 마스크의 1/0 허용 표기를 실제 어텐션 API에 그대로 넣을 수 없는 이유를 설명하라.

</div>

??? success "풀이"

    가산 마스크의 0은 점수를 바꾸지 않고 −∞가 차단한다. bool 마스크의 True 의미도 API별로 확인해야 한다. 개념적 하삼각 1/0 행렬을 알맞게 변환하고 padding과 미래 토큰을 차단하는 모든 경로를 검토해야 한다. head의 목표 이동만으로 은닉 표현의 인과성이 보장되지는 않는다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
MLM의 80/10/10과 로스 위치 15%가 무엇을 뜻하는지 설명하라.

</div>

??? success "풀이"

    적격 위치 중 선택된 약 15%가 원래 토큰의 복원 목표다. 그 선택 위치의 80%는 MASK, 10%는 무작위, 10%는 그대로 입력한다. 로스는 선택 위치에서 계산하지만 주변 입력 표현·공유 파라메터에도 경사가 흐른다. 특수 토큰 제외와 목표가 없는 배치도 별도로 처리한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
현재 ELECTRA의 disc_labels 비교가 틀릴 수 있는 이유를 설명하라.

</div>

??? success "풀이"

    생성기에는 교란 입력을 줘야 하지만 판별기의 정답은 생성 후 토큰과 원문을 비교해야 한다. 코드처럼 교란 input_ids와 비교하면 MASK를 원문 토큰으로 복원해도 바뀐 것으로 표시할 수 있다. 원문과 같게 샘플링된 토큰은 원래 토큰으로 취급하며 padding은 로스에서 제외해야 한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
T5 구간 복원·prefix LM·FIM이 입력과 목표를 구성하는 방식을 구분하라.

</div>

??? success "풀이"

    T5는 sentinel로 대체한 입력에서 제거 구간과 표식을 생성한다. prefix LM은 prefix 내부 양방향·suffix 인과 어텐션을 사용하고 주로 suffix를 학습한다. FIM은 prefix·suffix를 먼저 제공하도록 자료를 재배열해 middle을 생성한다. 구조·마스크·자료 순서와 특수 토큰을 일치시켜야 하며 prefix tuning과도 다르다.



## 정리하며

사전 학습 목표는 입력 교란·가시 문맥·예측 위치의 조합이다. 직접 로스 토큰 수를 학습 효율과 동일시하지 않고, 각 부분 구현의 마스크·원문 정답·특수 토큰과 평가 조건을 확인해야 한다.

**참고 문헌**

1. Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners." (GPT-2)
2. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers."
3. Raffel, C., et al. (2020). "Exploring the Limits of Transfer Learning with T5."
4. Clark, K., et al. (2020). "ELECTRA: Pre-training Text Encoders as Discriminators."
5. Tay, Y., et al. (2022). "UL2: Unifying Language Learning Paradigms."
6. Bavarian, M., et al. (2022). "Efficient Training of Language Models to Fill in the Middle."
