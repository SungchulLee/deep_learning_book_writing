# 큰 말 모델의 미리 익히기 목표

---

## 1. 학습 목표

- 인과 말 나타내기(CLM)와 가린 말 나타내기(MLM)를 견준다
- 다음 토막 어림하기의 수학 바탕을 이해한다
- 잡음 없애기 목표와 구간 망가뜨리기를 살핀다
- 여러 미리 익히기 방식의 맞바꿈을 값매김한다
- UL2와 FIM을 비롯한 요즘 미리 익히기 전략을 짠다

---

## 2. 들어가며

미리 익히기 목표는 큰 말 모델이 이름표 없는 글에서 배우는 스스로 살피는 일을 정한다. 어떤 목표를 고르느냐가 모델의 능력을 근본에서 빚어, 만들어 내기에 뛰어난지 이해에 뛰어난지 아니면 둘 다인지를 가른다.

---

## 3. 인과 말 나타내기(CLM)

GPT, LLaMA, Mistral과 모든 풀개만의 모델이 쓴다.

### 자기되돌리기로 세우기

인과 말 나타내기는 글의 확률을 조건부 확률의 곱으로 나타낸다:

$$
P(x_1, x_2, \ldots, x_n) = \prod_{t=1}^{n} P(x_t | x_1, \ldots, x_{t-1})
$$

익히기 목표는 음의 로그 가능도를 가장 작게 한다:

$$
\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^{n} \log P_\theta(x_t | x_{<t})
$$

### 인과 눈길 마스크

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

### 구현

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
            hidden_states: (묶음, 차례 길이, 숨은 크기)
            labels: (묶음, 차례 길이) - 다음 토막 어림을 위해 1만큼 밀림
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
    홀로 서는 인과 말 나타내기 손실.
    
    인수:
        logits: [묶음, 차례 길이, 낱말 곳간 크기]
        labels: [묶음, 차례 길이]
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

### 인과 말 나타내기의 성질

| 갈래 | 성질 |
|--------|----------------|
| 눈길 | 한 방향(왼쪽에서 오른쪽) |
| 만들어 내기 | 자연스럽다(자기되돌리기 표집) |
| 맥락 | 지난 토막만 |
| 익히기 | 단순하고 토막마다 신호를 준다 |
| 모델 | GPT 계열, LLaMA, Mistral, Claude |

---

## 4. 가린 말 나타내기(MLM)

BERT, RoBERTa, DeBERTa와 부호기만의 모델이 쓴다.

### 정식화

가린 말 나타내기는 토막을 마구잡이로 가리고 두 방향 맥락으로 그것을 어림한다:

$$
\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P_\theta(x_i | x_{\backslash \mathcal{M}})
$$

여기서 $\mathcal{M}$은 가린 자리의 모음이고 $x_{\backslash \mathcal{M}}$은 가리지 않은 토막 모두를 나타낸다.

### BERT 방식 가리기 전략

토막의 15%를 어림 대상으로 고른다:

- 80%는 [MASK]으로 바꾼다
- 10%는 무작위 토큰으로 바꾼다
- 10%는 그대로 둔다

이 전략은 모델이 [MASK] 토막만 어림하면 된다고 배우는 것을 막는다.

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
    labels = [-100] * len(tokens)  # -100 = 손실에서 무시한다
    
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
    알맞은 텐서 연산으로 가린 말 모델 익히기 묶음을 만든다.
    
    인수:
        input_ids: [묶음, 차례 길이] 들임 토막 번호
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
    
    # 가린 토막에서만 손실을 셈한다
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

### 가린 말 나타내기의 성질

| 갈래 | 성질 |
|--------|----------------|
| 눈길 | 두 방향(온 맥락) |
| 만들어 내기 | 거듭 다듬거나 따로 풀개가 있어야 한다 |
| 맥락 | 가리지 않은 모든 토막 |
| 익히기 | 토막의 15%만 기울기 신호를 준다 |
| 모델 | BERT, RoBERTa, DeBERTa, ALBERT |

---

## 5. 구간 망가뜨리기(T5)

T5, BART와 부호기-풀개 모델이 쓴다.

### 잡음 없애기 목표

잇닿은 구간을 보초 토막으로 갈음한 뒤 본디 구간을 어림한다:

```
Input:  "The quick brown [X] the lazy dog"
Target: "[X] fox jumps over"
```

$$
\mathcal{L}_{\text{denoise}} = -\log P_\theta(\text{corrupted spans} | \text{context})
$$

### 구현

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

---

## 6. 앞가지 말 나타내기

### 섞은 방식

앞가지 말 모델은 앞가지에 두 방향 눈길을 쓰고 만들어 낼 때는 인과 눈길을 쓴다:

```
Prefix (bidirectional): "Translate English to French:"
Generation (causal):    " Le chat est sur le tapis"
```

$$
\mathcal{L} = -\sum_{t > L_{\text{prefix}}} \log P(x_t | x_1, \ldots, x_{t-1})
$$

### 눈길 무늬

```python
def prefix_lm_mask(seq_len: int, prefix_len: int) -> torch.Tensor:
    """
    앞가지 말 모델의 눈길 가림막.
    
    - 앞가지 토막: 양쪽 눈길(앞가지 토막을 모두 본다)
    - 만들어 내기 토막: 인과 눈길(앞가지 + 앞서 만든 것을 본다)
    
    인수:
        seq_len: 전체 차례 길이
        prefix_len: 앞가지 몫의 길이
        
    반환값:
        눈길 가림막 (차례 길이, 차례 길이)
    """
    mask = torch.zeros(seq_len, seq_len)
    
    # 앞가지: 앞가지 안에서는 눈길이 온전하다
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

---

## 7. 갈음된 토막 알아채기(ELECTRA)

ELECTRA는 작은 만들개가 갈음한 토막을 알아채도록 가름개를 익힌다:

$$
\mathcal{L} = -\sum_{t=1}^{T} \left[ y_t \log D(x_t) + (1-y_t) \log(1 - D(x_t)) \right]
$$

여기서 토막 $t$이 만들개에 갈음됐으면 $y_t = 1$이다.

```python
class ELECTRA(nn.Module):
    """
    ELECTRA: 글 부호기를 가름개로 미리 익히기.
    
    작은 만들개로 글을 망가뜨리고, 으뜸 모델이 망가진 곳을 알아내는 법을 배운다.
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
        # 만들개가 가린 토막을 어림한다(가린 말 모델)
        gen_logits = self.generator(input_ids).logits
        gen_loss = F.cross_entropy(
            gen_logits[masked_indices],
            labels[masked_indices]
        )
        
        # 만들개에서 바꿔 넣을 것을 뽑는다
        with torch.no_grad():
            gen_probs = F.softmax(gen_logits, dim=-1)
            sampled = torch.multinomial(
                gen_probs.view(-1, gen_probs.size(-1)), 1
            ).view(input_ids.shape)
        
        # 망가뜨린 차례를 만든다
        corrupted = input_ids.clone()
        corrupted[masked_indices] = sampled[masked_indices]
        
        # 가름개가 어느 토막이 바뀌었는지 어림한다
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

---

## 8. 그 밖의 잡음 없애기 목표

### 글월 돌리기
글월을 마구잡이 지점에서 돌리고 돌린 양을 어림한다.

### 월 자리바꿈
월을 섞고 본디 차례를 되살린다(BART가 쓴다).

### 토막 지우기
토막을 마구잡이로 지우고 본디 차례를 어림한다.

### 토막 메우기
구간을 마스크 토막 하나로 갈음한다(구간마다 마스크를 하나씩 쓰는 T5와 다르다).

---

## 9. 잡음 없애개 섞기(UL2)

구글의 UL2는 미리 익히는 동안 목표 여럿을 아우른다:

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
        # R이나 X: 매개변수를 달리한 구간 망가뜨리기
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

---

## 10. 가운데 채우기(FIM)

코드 모델에서 FIM은 자기되돌리기 익히기를 그대로 두면서 가운데를 메우는 능력을 준다:

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

---

## 11. 익히기 효율 견줌

### 실효 익힘 신호

```python
def effective_tokens_per_example(
    seq_len: int,
    objective: str,
    mask_rate: float = 0.15
) -> float:
    """
    차례마다의 실효 익히기 신호를 셈한다.
    
    모든 목표가 토막마다 기울기 신호를 주지는 않는다.
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
        # 모든 토막이 가름개 신호를 준다
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

---

## 12. 두루 살피는 견줌

| 목표 | 얼개 | 두 방향 | 만들어 내기 | 토막당 신호 | 알맞은 곳 |
|-----------|--------------|---------------|------------|--------------|----------|
| CLM | 풀개 | ✗ | 자연스럽다 | 100% | 만들어 내기, 채팅 |
| MLM | 부호기 | ✓ | 제한됨 | 15% | 이해, 묻힘 |
| 구간 망가뜨리기 | 부호기-풀개 | 일부 | ✓ | 15% | 차례에서 차례로, 옮김 |
| 앞가지 말 모델 | 풀개 | 일부 | ✓ | 약 50% | 조건을 건 만들어 내기 |
| ELECTRA | 부호기 | ✓ | 제한됨 | 100% | 효율적인 미리 익히기 |
| UL2 | 부호기-풀개 | 섞임 | ✓ | 섞임 | 두루 쓰기 |

---

## 13. 핵심 식 간추림

**인과 말 나타내기**:

$$
\boxed{\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^{n} \log P(x_t | x_{<t})}
$$

**가린 말 나타내기**:

$$
\boxed{\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P(x_i | x_{\backslash \mathcal{M}})}
$$

**ELECTRA 가름개**:

$$
\boxed{\mathcal{L}_{\text{disc}} = -\sum_{t=1}^{T} \left[ y_t \log D(x_t) + (1-y_t) \log(1 - D(x_t)) \right]}
$$

---

## 연습문제

**연습문제 1.**
큰 규모로 큰 말 모델을 익힐 때의 핵심 어려움을 밝혀라.

??? success "연습문제 1 풀이"
    주된 어려움: (1) **셈 값**: 1750억 매개변수 모델을 익히려면 GPU 수천 대를 몇 주 돌려야 하고 수백만 달러가 든다. (2) **자료의 좋음**: 웹에서 긁은 자료에는 잡음, 치우침, 겹침, 해로운 내용이 들어 있어 대대적인 거르기가 필요하다. (3) **익히기의 흔들림**: 규모가 커지면 손실이 튀거나 흩어지거나 기울기에 탈이 나는 일이 잦아져 배움 비율 일정 짜기와 기울기 자르기를 조심스레 해야 한다. (4) **흩뿌린 익히기**: GPU 수백 대에 걸친 모델 나란히 하기와 자료 나란히 하기는 주고받기 덧짐과 맞추기의 어려움을 낳는다. (5) **값매김**: 표준 잣대가 떠오르는 능력이나 어그러지는 방식을 담아내지 못할 수 있다.

---

**연습문제 2.**
미리 익히기 목표로서 인과 말 나타내기, 가린 말 나타내기, 앞가지 말 나타내기를 견주어라.

??? success "연습문제 2 풀이"
    **인과 말 모델**(GPT): 왼쪽 맥락으로 다음 토막을 어림한다. 자연스럽게 만들어 낼 수 있지만 오른쪽 맥락에 조건을 걸 수 없다. **가린 말 모델**(BERT): 두 방향 온 맥락으로 가린 토막을 어림한다. 이해에 아주 좋지만 자기되돌리기로 만들어 내지 못한다. **앞가지 말 모델**(T5의 부호기 + 인과 풀개): 부호기가 "앞가지"(들임)의 두 방향 온 맥락을 보고 풀개가 자기되돌리기로 만들어 낸다. 이해와 만들어 내기를 아우른다. 만들어 내기에 초점을 둔 모델에서는 인과 말 모델이 판을 잡는데, 자기되돌리기 만들어 내기 과정과 결이 맞고 규모를 키우기 좋기 때문이다.

---

**연습문제 3.**
큰 말 모델의 떠오르는 능력이란 무엇인가? 보기를 들고 그것이 참으로 떠오르는 것인지 논하여라.

??? success "연습문제 3 풀이"
    떠오르는 능력이란 큰 모델에는 나타나지만 작은 모델에는 없는 능력으로, 규모에 따른 상 바뀜을 시사한다. 보기: 생각의 사슬 따지기, 맥락 안에서 배우기, 코드 만들기, 여러 말을 따로 익히지 않고도 하는 옮김. 논쟁은 이렇다. 어떤 연구자는 떠오름이 매끄럽게 나아지는 로그 확률에 비선형 값매김 잣대(정확도)를 씌워 생긴 찌꺼기라고 본다. 이어진 잣대(로그 가능도)로 값매김하면 나아짐이 차츰차츰 보인다. 다른 이들은 여러 걸음 시킴을 따르는 능력 같은 질적인 능력 바뀜은 참으로 떠오르는 현상이라고 본다.

---

**연습문제 4.**
큰 말 모델을 값매김하는 흔한 잣대와 그 한계를 설명하여라.

??? success "연습문제 4 풀이"
    흔한 잣대: **MMLU**(57개 과목에 걸친 여러 일 객관식), **HellaSwag**(상식 따지기), **GSM8K**(초등 수학), **HumanEval**(코드 만들기), **TruthfulQA**(사실 정확도). **한계**: (1) 자료 오염 — 잣대 자료가 익힘 말뭉치에 들어 있어 점수가 부풀 수 있다. (2) 좁은 값매김 — 객관식은 만들어 낸 글의 좋음을 시험하지 못한다. (3) 잣대 맞추기 — 모델을 특정 잣대에 맞춰 다듬을 수 있다. (4) 포화 — 으뜸 모델이 어떤 잣대에서 100%에 가까워 가르는 힘이 줄어든다. (5) 빠진 갈래 — 창의, 안전, 실제 쓸모는 표준 잣대로 잘 재지 못한다.

## 정리하며

| 목표 | 핵심 모델 | 으뜸 쓰임새 |
|-----------|------------|------------------|
| **CLM** | GPT, LLaMA, Mistral, Claude | 글 만들어 내기, 채팅, 따짐 |
| **MLM** | BERT, RoBERTa, DeBERTa | 갈래 매기기, 자연어 이해, 묻힘 |
| **구간 망가뜨리기** | T5, BART, mT5 | 옮김, 간추리기, 차례에서 차례로 |
| **앞가지 말 모델** | PaLM(일부), UniLM | 조건을 건 만들어 내기 |
| **ELECTRA** | ELECTRA, DeBERTa v3 | 효율적인 부호기 미리 익히기 |
| **UL2** | Flan-UL2, PaLM 2 | 두루 쓰기, 여러 일 |
| **FIM** | CodeLLaMA, StarCoder | 코드 이어 쓰기, 가운데 메우기 |

**참고 문헌**

1. Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners." (GPT-2)
2. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers."
3. Raffel, C., et al. (2020). "Exploring the Limits of Transfer Learning with T5."
4. Clark, K., et al. (2020). "ELECTRA: Pre-training Text Encoders as Discriminators."
5. Tay, Y., et al. (2022). "UL2: Unifying Language Learning Paradigms."
6. Bavarian, M., et al. (2022). "Efficient Training of Language Models to Fill in the Middle."
