# 미리 짚어 풀기

미리 짚어 풀기는 더 작은 **밑그림 모델**로 토막 여럿을 내놓고 그것을 더 큰 **목표 모델**이 나란히 확인해 자기되돌리기 만들어 내기를 빠르게 하는 재주이다. 나란히 할 수 있어 확인이 만들어 내기보다 값싸다는 점을 써먹는다.

---

## 1. 자기되돌리기의 병목

### 보통의 만들어 내기

자기되돌리기 모델은 한 번에 토막 하나씩 만든다:

$$
p(x_{1:T}) = \prod_{t=1}^{T} p(x_t | x_{<t})
$$

토막마다 모델을 온전히 한 번 지나야 하므로 만들어 내기는:

- **기억 공간에 묶임**: 셈의 밀도가 낮다
- **차례차례**: 토막끼리 나란히 할 수 없다
- **느림**: 큰 모델은 토막마다 늦음이 크다

### 핵심 통찰

**확인이 만들어 내기보다 빠르다**: 밑그림 토막 $K$개가 주어지면 목표 모델이 앞먹임 한 번으로 모두 (나란히) 확인할 수 있지만, $K$개를 만들려면 차례차례 $K$번 지나야 한다.

---

## 2. 알고리즘

### 훑어보기

```
1. Draft model generates K candidate tokens quickly
2. Target model scores all K tokens in one forward pass
3. Accept tokens until first rejection
4. Sample correction token at rejection point
5. Repeat
```

### 수학 얼거리

$p(x)$을 목표 분포, $q(x)$을 밑그림 분포라 하자.

토막 $x_t$의 **받아들임 잣대**:

$$
\text{Accept with probability } \min\left(1, \frac{p(x_t | x_{<t})}{q(x_t | x_{<t})}\right)
$$

**물리치기 표집 바로잡기**: 물리쳤으면 남은 것에서 뽑는다:

$$
p'(x) = \text{norm}\left(\max\left(0, p(x) - q(x)\right)\right)
$$

이러면 내놓는 분포가 목표 모델의 것과 정확히 맞는다.

### 자세한 알고리즘

```
Algorithm: Speculative Decoding

Input: Target model p, Draft model q, Prompt x₀, Draft length K
Output: Generated sequence

while not done:
    # 1걸음: 밑그림 마디
    for i = 1 to K:
        Sample x̃ᵢ ~ q(· | x₀, x̃₁, ..., x̃ᵢ₋₁)
        Store q(x̃ᵢ | ...)
    
    # 2걸음: 확인 마디(앞먹임 한 번)
    Compute p(x̃₁ | x₀), p(x̃₂ | x₀, x̃₁), ..., p(x̃ₖ | x₀, ..., x̃ₖ₋₁)
    
    # 3걸음: 받아들이기/물리치기
    n = 0  # 받아들인 토막 수
    for i = 1 to K:
        r ~ Uniform(0, 1)
        if r < min(1, p(x̃ᵢ)/q(x̃ᵢ)):
            Accept x̃ᵢ
            n = n + 1
        else:
            # 남은 분포에서 뽑는다
            Sample x from norm(max(0, p(·) - q(·)))
            Append x to sequence
            break
    
    if all K tokens accepted:
        # 덤: p에서 토막 하나를 더 뽑는다
        Sample x ~ p(· | x₀, x̃₁, ..., x̃ₖ)
        Append x to sequence
    
    Update x₀ with accepted tokens
```

---

## 3. PyTorch 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class SpeculativeOutput:
    """미리 짚어 풀기 한 걸음이 내놓는 것."""
    tokens: torch.Tensor
    num_accepted: int
    num_drafted: int
    
    @property
    def acceptance_rate(self) -> float:
        return self.num_accepted / self.num_drafted if self.num_drafted > 0 else 0.0

class SpeculativeDecoder:
    """
    글 만들어 내기를 빠르게 하는 미리 짚어 풀기.
    
    작은 밑그림 모델로 토막을 내놓고 더 큰 목표 모델이 확인한다.
    """
    
    def __init__(
        self,
        target_model: nn.Module,
        draft_model: nn.Module,
        draft_length: int = 4,
        temperature: float = 1.0
    ):
        self.target = target_model
        self.draft = draft_model
        self.K = draft_length
        self.temperature = temperature
    
    @torch.no_grad()
    def generate_step(
        self,
        input_ids: torch.Tensor,
        target_cache: Optional[Tuple] = None,
        draft_cache: Optional[Tuple] = None
    ) -> Tuple[SpeculativeOutput, Optional[Tuple], Optional[Tuple]]:
        """
        미리 짚어 풀기의 한 걸음.
        
        받아들인 토막과 새로 고친 곳간을 돌려준다.
        """
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        # 1걸음: 밑그림 토막 K개를 만든다
        draft_tokens = []
        draft_probs = []
        current_ids = input_ids
        
        for _ in range(self.K):
            logits, draft_cache = self.draft(current_ids, past_caches=draft_cache)
            probs = F.softmax(logits[:, -1, :] / self.temperature, dim=-1)
            
            # 밑그림 분포에서 뽑는다
            token = torch.multinomial(probs, num_samples=1)
            draft_tokens.append(token)
            draft_probs.append(probs.gather(-1, token))
            
            current_ids = token
        
        draft_tokens = torch.cat(draft_tokens, dim=1)  # [묶음, K]
        draft_probs = torch.cat(draft_probs, dim=1)    # [묶음, K]
        
        # 2걸음: 목표 모델로 K개 토막을 모두 나란히 확인한다
        verify_ids = torch.cat([input_ids, draft_tokens], dim=1)
        target_logits, target_cache = self.target(verify_ids, past_caches=target_cache)
        
        # 밑그림 토막에 대한 목표 확률을 얻는다
        # target_logits[:, -K-1:-1, :]는 밑그림 토막마다 그 앞자리에 맞닿는다
        target_probs = F.softmax(target_logits[:, -self.K-1:, :] / self.temperature, dim=-1)
        
        # 3걸음: 알맞은 번호 매기기로 받아들이거나 물리친다
        accepted_tokens = []
        num_accepted = 0
        
        for i in range(self.K):
            # 밑그림으로 낸 토막의 목표 확률
            p = target_probs[:, i, :].gather(-1, draft_tokens[:, i:i+1]).squeeze(-1)
            q = draft_probs[:, i]
            
            # 받아들일 확률
            accept_prob = torch.clamp(p / q, max=1.0)
            
            # 받아들임을 뽑는다
            r = torch.rand(batch_size, device=device)
            accept = r < accept_prob
            
            if accept.all():
                accepted_tokens.append(draft_tokens[:, i:i+1])
                num_accepted += 1
            else:
                # 물리침: 남은 분포에서 뽑는다
                residual = torch.clamp(target_probs[:, i, :] - 
                                       F.softmax(logits[:, -1, :] / self.temperature, dim=-1), 
                                       min=0)
                residual = residual / residual.sum(dim=-1, keepdim=True)
                
                # 수치 문제를 다룬다
                if torch.isnan(residual).any():
                    correction = torch.multinomial(target_probs[:, i, :], num_samples=1)
                else:
                    correction = torch.multinomial(residual, num_samples=1)
                
                accepted_tokens.append(correction)
                num_accepted += 1
                break
        else:
            # K개 토막을 모두 받아들였다 - 덤 토막을 뽑는다
            bonus = torch.multinomial(target_probs[:, -1, :], num_samples=1)
            accepted_tokens.append(bonus)
            num_accepted += 1
        
        result_tokens = torch.cat(accepted_tokens, dim=1)
        
        return SpeculativeOutput(
            tokens=result_tokens,
            num_accepted=num_accepted,
            num_drafted=self.K
        ), target_cache, draft_cache
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100
    ) -> Tuple[torch.Tensor, dict]:
        """
        미리 짚어 풀기로 토막을 만든다.
        """
        generated = input_ids
        total_accepted = 0
        total_drafted = 0
        num_steps = 0
        
        target_cache = None
        draft_cache = None
        
        while generated.size(1) - input_ids.size(1) < max_new_tokens:
            output, target_cache, draft_cache = self.generate_step(
                generated, target_cache, draft_cache
            )
            
            generated = torch.cat([generated, output.tokens], dim=1)
            total_accepted += output.num_accepted
            total_drafted += output.num_drafted
            num_steps += 1
            
            # 기억 공간 문제를 피하려 곳간을 이따금 비운다
            if num_steps % 50 == 0:
                target_cache = None
                draft_cache = None
        
        stats = {
            'acceptance_rate': total_accepted / total_drafted if total_drafted > 0 else 0,
            'tokens_per_step': total_accepted / num_steps if num_steps > 0 else 0,
            'speedup_factor': total_accepted / num_steps if num_steps > 0 else 1
        }
        
        return generated[:, :input_ids.size(1) + max_new_tokens], stats

def speculative_sample(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token: torch.Tensor
) -> Tuple[torch.Tensor, bool]:
    """
    미리 짚어 뽑기의 고갱이 연산.
    
    인수:
        target_probs: [낱말 곳간 크기] 목표 모델의 확률
        draft_probs: [낱말 곳간 크기] 밑그림 모델의 확률
        draft_token: 밑그림 모델이 내놓은 토막
        
    반환값:
        (받아들인 토막, 받아들였는지 여부)
    """
    p = target_probs[draft_token]
    q = draft_probs[draft_token]
    
    # min(1, p/q)의 확률로 받아들인다
    if torch.rand(1) < p / q:
        return draft_token, True
    
    # 물리침: 남은 것에서 뽑는다
    residual = torch.clamp(target_probs - draft_probs, min=0)
    residual = residual / residual.sum()
    
    corrected = torch.multinomial(residual, num_samples=1)
    return corrected, False

# 가짜 모델로 보이기
class MockLanguageModel(nn.Module):
    """보이기 위한 단순한 가짜 말 모델."""
    
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_size, nhead=4, batch_first=True)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, past_caches=None):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(h), None

if __name__ == "__main__":
    print("Speculative Decoding Demo")
    print("=" * 50)
    
    vocab_size = 1000
    
    # 모델을 만든다(목표가 더 크다)
    target = MockLanguageModel(vocab_size, hidden_size=256, num_layers=6)
    draft = MockLanguageModel(vocab_size, hidden_size=128, num_layers=2)
    
    # 복호기 만들기
    decoder = SpeculativeDecoder(
        target_model=target,
        draft_model=draft,
        draft_length=4,
        temperature=1.0
    )
    
    # 생성
    prompt = torch.randint(0, vocab_size, (1, 10))
    output, stats = decoder.generate(prompt, max_new_tokens=50)
    
    print(f"Prompt length: {prompt.size(1)}")
    print(f"Output length: {output.size(1)}")
    print(f"Acceptance rate: {stats['acceptance_rate']:.2%}")
    print(f"Tokens per step: {stats['tokens_per_step']:.2f}")
    print(f"Theoretical speedup: {stats['speedup_factor']:.2f}x")
```

---

## 4. 이론적 분석

### 걸음마다의 기대 토막 수

받아들임 비율이 $\alpha$이면 걸음마다 받아들이는 토막의 바라는 수는 다음과 같다.

$$
\mathbb{E}[\text{tokens}] = \sum_{k=1}^{K} k \cdot \alpha^{k-1}(1-\alpha) + (K+1)\alpha^K
$$

$K=4$이고 $\alpha=0.8$이면 걸음마다 $\mathbb{E} \approx 3.36$개다.

### 빨라짐 살피기

다음이라 하자:

- $T_t$ = 목표 모델의 앞먹임 시간
- $T_d$ = 밑그림 모델의 앞먹임 시간
- $K$ = 밑그림 길이
- $\alpha$ = 받아들임 비율

**미리 짐작하지 않으면**: 토막 $N$개를 만드는 데 $N \cdot T_t$이 든다

**미리 짚을 때**:

- Steps needed: $\approx N / \mathbb{E}[\text{tokens}]$
- 걸음마다 드는 시간: $K \cdot T_d + T_t$(초안 걸음 K번 + 따짐 1번)

**빨라짐**:

$$
S = \frac{N \cdot T_t}{\frac{N}{\mathbb{E}[\text{tokens}]} \cdot (K \cdot T_d + T_t)} = \frac{\mathbb{E}[\text{tokens}] \cdot T_t}{K \cdot T_d + T_t}
$$

When $T_d \ll T_t$:

$$
S \approx \mathbb{E}[\text{tokens}]
$$

---

## 5. 실용적인 고려

### 밑그림 모델 고르기

| 방식 | 좋은 점 | 나쁜 점 |
|----------|------|------|
| 같은 갈래의 작은 모델 | 받아들임이 높다 | 그래도 따로 모델이 필요하다 |
| 양자화한 목표 모델 | 받아들임이 아주 높다 | 빨라짐이 제한된다 |
| n-그램 / 찾기 | 신경망 셈이 없다 | 받아들임이 낮다 |
| 일찍 빠져나가기 | 매개변수를 나눠 쓴다 | 얼개를 바꿔야 한다 |

### 받아들임 비율을 좌우하는 것

1. **분포의 결 맞음**: 밑그림이 목표에 가까울수록 → 받아들임이 높다
2. **온도**: 온도가 높을수록 → 더 고르게 → 받아들임이 높다
3. **분야 맞음**: 같은 분야의 밑그림 → 받아들임이 높다
4. **차례에서의 자리**: 뒤쪽 자리가 흔히 받아들임이 높다

### 기억 공간에서 헤아릴 점

- 두 모델을 모두 기억 공간에 두어야 한다
- 두 모델 모두에 열쇠-값 곳간이 필요하다
- 맞바꿈: 기억 공간과 빠르기

---

## 6. 변형

### 메두사

목표 모델에 어림 머리를 여럿 둔다:

- 따로 밑그림 모델이 없다
- 앞으로 올 토막 여럿을 나란히 어림한다
- 모델 하나로 기억 공간을 줄인다

### SpecInfer

나무 짜임의 미리 짚기:

- 밑그림 차례 여럿(나무)
- 나무 전체를 한 번에 확인한다
- 후보가 많을수록 받아들임이 높다

### 단계별 미리 짚어 풀기

점점 커지는 모델의 사슬:
```
Tiny → Small → Medium → Target
```

---

## 7. 다른 빠르게 하기 방법과의 견줌

| 방법 | 빨라짐 | 정확 | 기억 공간 | 복잡도 |
|--------|---------|-------|--------|------------|
| 미리 짚어 풀기 | 2~3배 | ✓ | 모델 2배 | 가운데 |
| 열쇠-값 곳간 | 약 N배 | ✓ | O(차례 길이) | 낮음 |
| 플래시 눈길 | 2~4배 | ✓ | O(N) | 낮음 |
| 양자화 | 2~4배 | ✗ | 0.25~0.5배 | 가운데 |
| 가지치기 | 들쭉날쭉 | ✗ | 1배 미만 | 높음 |

---

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

## 정리하며

미리 짚어 풀기는 다음으로 큰 말 모델 미룸을 빠르게 한다:

1. **나란한 확인**: 앞먹임 한 번으로 밑그림 토막 여럿을 살핀다
2. **정확한 뽑기**: 목표 분포를 수학으로 보장한다
3. **서로 채워 줌**: 열쇠-값 곳간, 플래시 눈길, 양자화와 함께 쓴다
4. **맞바꿈**: 밑그림 모델이 필요하고 얼마나 잘 듣는지는 결이 얼마나 맞느냐에 달렸다

흔한 빨라짐: 잘 맞는 밑그림 모델이면 **2~3배**.

**참고 문헌**

1. Leviathan, Y., et al. (2023). "Fast Inference from Transformers via Speculative Decoding." ICML.
2. Chen, C., et al. (2023). "Accelerating Large Language Model Decoding with Speculative Sampling."
3. Cai, T., et al. (2024). "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads."
4. Miao, X., et al. (2023). "SpecInfer: Accelerating Generative Large Language Model Serving with Speculative Inference."
