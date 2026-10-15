# Speculative decoding: 초안 생성과 검증

Speculative decoding은 저렴한 초안 모델이 제안한 토큰들을 목표 모델로 함께 평가하는 접근이다. 올바른 수용·거절 보정을 사용하면 목표 모델의 샘플링 분포를 유지하면서 여러 토큰을 전진할 수 있다. 이는 [Leviathan 등](https://proceedings.mlr.press/v202/leviathan23a.html)과 [Chen 등](https://arxiv.org/abs/2302.01318)의 연구에서 다루었다. 속도 향상은 초안 비용·분포 일치·검증·cache 관리에 달려 있다.

## 1. 제안과 목표 분포

```
1. Draft model generates K candidate tokens quickly
2. Target model scores all K tokens in one forward pass
3. Accept tokens until first rejection
4. Sample correction token at rejection point
5. Repeat
```

한 prefix에서 목표 분포 p와 실제 초안 샘플링 분포 q를 생각하자. q에서 제안한 x를 min(1,p(x)/q(x)) 확률로 수용한다. 거절하면 같은 prefix의 전체 분포를 사용해 r(x)∝max(0,p(x)−q(x))에서 보정 토큰을 뽑는다.

수용된 질량은 min(p(x),q(x))이고 보정이 나머지 p의 질량을 채운다는 것이 핵심이다. q는 제안 토큰 하나의 확률뿐 아니라 거절 위치의 전체 벡터가 필요하다. 이론에서 q(x)=0인 사건은 q의 제안으로 나오지 않으며 수치 구현에서는 정규화·underflow·유효 분포도 검토해야 한다.

분포 보존은 인과 모델·같은 prefix·올바른 p와 q·마스크·보정이 모두 맞는 조건의 결과다. 온도나 후보 필터를 사용하면 실제로 샘플링한 분포에 대해 비율·잔차를 계산해야 한다. 같은 난수 시드에서 표준 생성과 토큰열이 반드시 같다는 의미도 아니다.

## 2. 여러 토큰의 검증

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

K개 제안을 순차적으로 만들고 목표 모델은 그 접두사들의 조건부 확률과 마지막 보너스 분포를 한 번의 인과 계산으로 얻을 수 있다. 첫 거절까지의 제안만 수용하고 거절 지점에서는 보정 토큰을 넣는다. 전부 수용하면 마지막 목표 분포에서 보너스 토큰 하나를 더 뽑는다.

위 기존 의사코드의 q(x̃_i) 저장만으로 잔차 벡터가 준비되는 것은 아니다. 각 제안 prefix의 전체 q_i를 저장하거나 정확히 재계산해야 한다. 다음 단계에는 수용한 제안뿐 아니라 보정/보너스까지 확정 prefix에 포함하고, 거절 뒤의 미확정 cache는 제거해야 한다.

## 3. 현재 PyTorch 코드의 한계

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
        
        draft_tokens = torch.cat(draft_tokens, dim=1)  # [배치, K]
        draft_probs = torch.cat(draft_probs, dim=1)    # [배치, K]
        
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
    미리 짚어 뽑기의 핵심 연산.
    
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
    
    # 디코더 만들기
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

현재 generate_step은 제안된 토큰의 q_i 값만 저장한다. 거절 보정에서는 루프 끝에 남은 마지막 draft logits를 빼므로, 앞쪽에서 거절되면 그 위치의 q_i가 아니다. 잔차가 NaN이면 목표 분포로 대체하는 fallback도 원래 수용·거절 질량을 일반적으로 보존하지 않는다.

캐시가 있어도 generate는 전체 generated를 다시 전달하고, target도 전체 prefix와 후보를 기존 cache에 추가한다. 과거를 중복 처리하며 거절된 후보의 cache를 되돌리지도 않는다. 50단계마다 cache를 지우는 것은 올바른 매 단계 정렬의 대체가 아니다.

배치에서는 accept.all()로 한 행이라도 거절하면 모든 행에 보정 토큰을 뽑는다. 각 행의 수용·거절 사건이 다를 수 있어 올바른 행별 샘플링이 아니다. EOS 종료·K와 온도·어휘/토크나이저·장치 및 cache 인터페이스의 검증도 필요하다.

main의 MockLanguageModel은 인과 마스크와 위치 표현 없이 TransformerEncoderLayer를 쓰고 cache를 무시한다. 목표 검증에서는 미래 후보를 볼 수 있고, 초안의 뒤 토큰은 캐시 없는 한 토큰 입력에서 예측한다. eval 모드도 설정하지 않아 dropout이 남을 수 있다. 이 시연은 분포를 보존하는 인과 언어 모델의 구현이나 검증 결과가 아니다.

별도 speculative_sample 함수의 한 위치 수용·잔차 식은 핵심 원리에 가깝지만 1차원 분포·단일 토큰과 유효 q를 전제로 한다. torch.rand는 CPU에 생성되며 장치·정규화와 잔차의 수치 처리를 맞춰야 한다. 이 함수가 잘 맞는다고 전체 캐시·배치 구현이 자동으로 검증되는 것은 아니다.

**기존 출력:**

```
Speculative Decoding Demo
==================================================
Prompt length: 10
Output length: 60
Acceptance rate: 62.50%
Tokens per step: 2.50
Theoretical speedup: 2.50x
```

num_accepted는 실제 수용한 초안뿐 아니라 보정·보너스도 센다. 따라서 acceptance_rate는 진짜 제안 수용률이 아니고 K개 모두 수용하면 (K+1)/K로 100%를 넘을 수 있다. speedup_factor도 측정 시간의 비가 아니라 출력 토큰/단계 수다. 마지막 길이 제한에서 잘린 토큰도 통계에 남을 수 있다. 기존 출력은 재실행 없이 보존했다.

## 4. 기대 전진량과 지연

각 연속 제안의 조건부 수용 확률이 일정하게 α이고 거절 보정 또는 보너스 토큰까지 출력한다는 단순 모형에서 단계당 토큰 수의 기대값은 다음과 같다.

$$\mathbb E[M]=\sum_{i=0}^{K}\alpha^i=\frac{1-\alpha^{K+1}}{1-\alpha}\quad(\alpha\ne1).$$

α=1이면 K+1, K=4·α=0.8이면 3.3616개다. 이는 초안에서 수용한 수만의 기대값과 다르다. 관측 평균 수용률을 무조건 상수 조건부 α로 대입해 정확한 기대값이라고 할 수는 없다.

초안 단계 지연 T_d, 블록 검증 T_v(K), 기타 관리 비용 T_o, 표준 목표 한 토큰 지연 T_t이면 단순 예상 가속률은 E[M]T_t/[KT_d+T_v(K)+T_o]다. 블록 검증이 언제나 한 토큰 forward와 같은 시간인 것은 아니다. 초안 비용·낮은 수용·통신·cache 복사 때문에 더 느릴 수도 있다.

## 5. 제안 모델과 확장

같은 prefix의 평균 수용 확률은 Σ_x min(p(x),q(x))=1−TV(p,q)다. 분포 일치가 중요하지만 같은 계열·같은 분야·높은 온도가 항상 더 높은 수용률을 보장하지 않는다. 호환 어휘·실제 제안 비용과 목표의 비교가 필요하다.

다중 head·후보 트리·검색 기반 제안 등은 제안·검증·샘플링 절차가 다르다. 정확한 목표 분포를 유지하는지, 근사 수용을 쓰는지 각 방식에서 따로 확인해야 한다. 후보가 많아지면 무조건 이득이 커지는 것도 아니다.

```
Tiny → Small → Medium → Target
```

이 도식은 여러 단계 제안의 아이디어이며 실제 통합을 구현하지 않는다. 메모리는 목표와 초안의 크기·cache·공유 여부에 따라 달라져 정확히 두 배로 고정되지 않는다. KV cache·FlashAttention·양자화와도 결합할 수 있지만 원래 목표와 같은 분포라는 주장과 양자화된 다른 목표의 분포는 구분해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
수용 질량 min(p,q)와 거절 잔차가 목표 p를 복원하는 이유를 설명하라.

</div>

??? success "연습문제 1 풀이"

    q에서 제안하고 min(1,p/q)로 수용하면 사건별 수용 질량은 min(p,q)다. 전체 거절 질량만큼 정규화한 (p−q)의 양수 부분에서 뽑아 나머지를 채우면 합이 p가 된다. 각 위치의 같은 prefix에 대한 전체 p·q가 필요하다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
현재 거절 보정과 cache 사용이 잘못되는 이유를 설명하라.

</div>

??? success "연습문제 2 풀이"

    앞 위치의 q_i 전체를 저장하지 않고 마지막 draft logits로 보정한다. 전체 prefix를 기존 cache와 함께 반복 입력하고 거절된 후보 cache도 제거하지 않는다. 조건부 분포·확정 prefix·cache 길이가 맞지 않아 이론의 분포 보존을 적용할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
K=4, α=0.8의 기대 출력 토큰 수와 수용률·가속률을 구분하라.

</div>

??? success "연습문제 3 풀이"

    보정/보너스를 포함한 기대 전진량은 1+0.8+0.64+0.512+0.4096=3.3616이다. 제안 수용률과 다른 수치이며 실제 가속은 초안·블록 검증·관리 시간 대비 표준 지연으로 계산해야 한다. 현재 num_accepted는 보정/보너스까지 세어 진짜 수용률이 아니다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
배치 accept.all()과 MockLanguageModel의 구조가 정확한 시연이 아닌 이유를 설명하라.

</div>

??? success "연습문제 4 풀이"

    행별 수용 결과가 다른데 전부를 같은 거절 분기로 보내면 분포가 달라진다. Mock 모델도 미래를 차단하지 않고 cache를 무시해 올바른 인과 조건부 검증이 아니다. 배치·인과성·cache와 실제 분포의 검증 없이 출력 길이나 속도 이름만으로 정당성을 판단하지 않는다.



## 정리하며

Speculative decoding의 분포 보존은 정확한 제안 확률·거절 보정·인과 검증과 cache 정렬에 달려 있다. 기대 토큰 전진량과 실제 지연·수용률을 구분하고 부분 시연을 완성된 정확 알고리즘으로 설명하지 않는다.

**참고 문헌**

1. Leviathan, Y., et al. (2023). "Fast Inference from Transformers via Speculative Decoding." ICML.
2. Chen, C., et al. (2023). "Accelerating Large Language Model Decoding with Speculative Sampling."
3. Cai, T., et al. (2024). "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads."
4. Miao, X., et al. (2023). "SpecInfer: Accelerating Generative Large Language Model Serving with Speculative Inference."
