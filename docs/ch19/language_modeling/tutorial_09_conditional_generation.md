# 실습 09

길잡이 09: 다스린 글 만들어 내기 전략. 글 만들어 내기의 좋음을 다스리고 낫게 하는 앞선 재주.

자연어 다루기는 깊은 배움 방식으로 크게 달라졌다. 이 단원은 말 모델 재주를 보여 주며, 신경망이 글을 어떻게 다루고 만들어 내는지 밝히는 실전 짜기를 준다.

## 1. 코드

```python
"""
길잡이 09: 다스린 글 만들어 내기 전략
===================================================

글 만들어 내기의 좋음을 다스리고 낫게 하는 앞선 재주.

주제:
1. 표집 전략(욕심쟁이, 빔 찾기, 알갱이, 상위 k)
2. 길이 고르게 맞추기
3. 되풀이 벌주기
4. 제약을 둔 풀기
5. 다스릴 수 있는 만들어 내기(마음결, 결풍, 주제)

만들어 내기 전략:
----------------------

1. 욕심쟁이 풀기:
   w_t = argmax P(w | context)
   - 빠르고 늘 같다
   - 가장 좋지 않을 수 있고 되풀이되기 쉽다

2. 빔 찾기:
   - 상위 k개 가설 남기기
   - 점수 = log P / 길이 벌주기
   - 욕심쟁이보다 좋다
   - 셈이 값비싸다

3. 표집:
   - 마구잡이: P(w | 맥락)에서 뽑기
   - 상위 k: 가장 그럴듯한 k개에서 뽑기
   - 알갱이(상위 p): 쌓인 확률이 p 이상인 가장 작은 모음에서 뽑기

4. 온도 맞추기:
   P'(w) ∝ exp(logit / T)
   - T < 1: 더 정해진 대로
   - T > 1: 더 마구잡이로
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional

# ========================================================================
# 메인
# ========================================================================


class GenerationStrategies:
    """글 만들어 내기 전략 모음."""
    
    @staticmethod
    def greedy_search(model, input_ids, max_length=50, vocab=None):
        """욕심쟁이 풀기 — 늘 가장 그럴듯한 낱말을 고른다."""
        model.eval()
        generated = input_ids.clone()
        
        for _ in range(max_length):
            with torch.no_grad():
                if hasattr(model, 'lstm') or hasattr(model, 'rnn'):
                    logits, _ = model(generated)
                else:
                    logits = model(generated)
                
                # 마지막 자리의 로짓을 얻는다
                next_token_logits = logits[:, -1, :]
                
                # 욕심쟁이: 가장 그럴듯한 것 고르기
                next_token = torch.argmax(next_token_logits, dim=-1)
                
                generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
                
                # 끝 토막인지 살피기
                if vocab and next_token.item() == vocab.word_to_idx(vocab.END_TOKEN):
                    break
        
        return generated
    
    @staticmethod
    def top_k_sampling(model, input_ids, max_length=50, k=50, 
                      temperature=1.0, vocab=None):
        """가장 그럴듯한 상위 k개 토막에서 뽑기."""
        model.eval()
        generated = input_ids.clone()
        
        for _ in range(max_length):
            with torch.no_grad():
                if hasattr(model, 'lstm') or hasattr(model, 'rnn'):
                    logits, _ = model(generated)
                else:
                    logits = model(generated)
                
                next_token_logits = logits[:, -1, :]
                
                # 온도를 적용한다
                next_token_logits = next_token_logits / temperature
                
                # 상위 k 거르기
                top_k_logits, top_k_indices = torch.topk(next_token_logits, k)
                probs = F.softmax(top_k_logits, dim=-1)
                
                # 상위 k에서 뽑기
                next_token_idx = torch.multinomial(probs, 1)
                next_token = top_k_indices.gather(-1, next_token_idx)
                
                generated = torch.cat([generated, next_token], dim=-1)
                
                if vocab and next_token.item() == vocab.word_to_idx(vocab.END_TOKEN):
                    break
        
        return generated
    
    @staticmethod
    def nucleus_sampling(model, input_ids, max_length=50, p=0.95,
                        temperature=1.0, vocab=None):
        """
        핵(상위 p) 표집.
        쌓인 확률이 p 이상인 가장 작은 토막 모음에서 뽑기.
        """
        model.eval()
        generated = input_ids.clone()
        
        for _ in range(max_length):
            with torch.no_grad():
                if hasattr(model, 'lstm') or hasattr(model, 'rnn'):
                    logits, _ = model(generated)
                else:
                    logits = model(generated)
                
                next_token_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                
                # 확률 정렬
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                
                # 쌓인 확률 셈하기
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # 쌓인 확률이 문턱값을 넘는 토막 없애기
                sorted_indices_to_remove = cumulative_probs > p
                # 적어도 토막 하나는 남기기
                sorted_indices_to_remove[..., 0] = False
                
                # 가림막 만들기
                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                
                # 확률 거르기
                filtered_probs = probs.clone()
                filtered_probs[indices_to_remove] = 0
                filtered_probs = filtered_probs / filtered_probs.sum()
                
                # 뽑기
                next_token = torch.multinomial(filtered_probs, 1)
                generated = torch.cat([generated, next_token], dim=-1)
                
                if vocab and next_token.item() == vocab.word_to_idx(vocab.END_TOKEN):
                    break
        
        return generated
    
    @staticmethod
    def beam_search(model, input_ids, max_length=50, beam_width=5,
                   length_penalty=1.0, vocab=None):
        """
        빔 찾기 풀기.
        걸음마다 beam_width개의 가설을 지닌다.
        """
        model.eval()
        batch_size = input_ids.size(0)
        vocab_size = model.fc.out_features
        
        # 빔 첫자리매김: (batch_size * beam_width, seq_len)
        beams = input_ids.unsqueeze(1).repeat(1, beam_width, 1)
        beams = beams.view(batch_size * beam_width, -1)
        
        # 빔마다의 점수
        beam_scores = torch.zeros(batch_size, beam_width)
        beam_scores[:, 1:] = -float('inf')  # 처음에는 첫 빔만 살아 있다
        beam_scores = beam_scores.view(-1)
        
        for step in range(max_length):
            with torch.no_grad():
                if hasattr(model, 'lstm') or hasattr(model, 'rnn'):
                    logits, _ = model(beams)
                else:
                    logits = model(beams)
                
                next_token_logits = logits[:, -1, :]
                next_token_scores = F.log_softmax(next_token_logits, dim=-1)
                
                # 빔 점수에 더하기
                next_scores = beam_scores.unsqueeze(-1) + next_token_scores
                next_scores = next_scores.view(batch_size, -1)
                
                # 상위 beam_width개 후보 얻기
                top_scores, top_indices = torch.topk(next_scores, beam_width, dim=-1)
                
                # 어느 빔인지, 어느 토막인지 셈하기
                beam_indices = top_indices // vocab_size
                token_indices = top_indices % vocab_size
                
                # 빔 고치기
                new_beams = []
                new_scores = []
                
                for i in range(batch_size):
                    for j in range(beam_width):
                        beam_idx = i * beam_width + beam_indices[i, j]
                        new_beam = torch.cat([
                            beams[beam_idx],
                            token_indices[i, j].unsqueeze(0)
                        ])
                        new_beams.append(new_beam)
                        new_scores.append(top_scores[i, j])
                
                beams = torch.stack(new_beams)
                beam_scores = torch.tensor(new_scores)
        
        # 가장 좋은 빔 돌려주기
        best_beam_idx = beam_scores[:beam_width].argmax()
        return beams[best_beam_idx].unsqueeze(0)


class RepetitionPenalty:
    """로짓에 되풀이 벌주기 쓰기."""
    
    @staticmethod
    def apply(logits, generated_tokens, penalty=1.2):
        """
        되풀이된 토막의 로짓을 벌주기로 나누어 벌준다.
        
        인수:
            logits: (vocab_size,) 로짓
            generated_tokens: 앞서 만든 토막 번호의 목록
            penalty: 벌주기 인자(> 1.0)
        """
        for token in set(generated_tokens):
            logits[token] /= penalty
        return logits


def demonstrate_generation_strategies():
    """여러 만들어 내기 전략 견주기."""
    
    print("Text Generation Strategies Comparison")
    print("=" * 70)
    
    print("""
전략의 성질:
------------------------

1. 욕심쟁이 찾기:
   - 늘 같다
   - 빠르다
   - 되풀이 무늬에 갇힐 수 있다
   - 쓸 곳: 단순한 이어 쓰기, 사실을 담은 글

2. 빔 찾기:
   - 더 꼼꼼한 찾기
   - 욕심쟁이보다 좋다
   - 그래도 되풀이될 수 있다
   - 쓸 곳: 옮김, 간추리기

3. 상위 k 표집:
   - 확률에 맡기며 여러 갈래이다
   - 확률 낮은 낱말을 거른다
   - k=50이 흔히 잘 된다
   - 쓸 곳: 창작 글쓰기, 채팅

4. 알갱이(상위 p) 표집:
   - 그때그때 바뀌는 낱말 곳간 크기
   - 확률 분포에 맞춰진다
   - p=0.9에서 0.95를 권한다
   - 쓸 곳: 두루 쓰기, 창의적인 일

5. 온도 표집:
   - 마구잡이 정도를 다스린다
   - T=0.7: 더 초점이 잡힘
   - T=1.0: 보통
   - T=1.5: 더 창의적

전략 아우르기:
--------------------
가장 좋은 버릇: 알갱이 + 온도
- 좋음을 위해 상위 p=0.95
- 창의를 위해 온도=0.8
- 고리를 피하려 되풀이 벌주기=1.2
    """)


if __name__ == "__main__":
    demonstrate_generation_strategies()
    
    print("""
익힘 문제:
1. 만들어 내기에 되풀이 벌주기 짜기
2. 빔 너비를 달리해 빔 찾기 견주기
3. 빔 찾기의 길이 고르게 맞추기 짜기
4. 상위 k와 알갱이 표집을 아울러 보기
5. 제약을 둔 풀기 짜기(특정 낱말을 강제하기)
6. 앞가지 다듬기로 다스릴 수 있는 만들어 내기 만들기
7. 여러 갈래 빔 찾기 짜기(여러 갈래의 내놓음)
8. 되풀이를 피하는 덮음 얼개 더하기
    """)
```

## 2. 논의

여기 짠 것은 깔끔하고 읽기 좋은 PyTorch 코드로 말 모델의 핵심 개념을 보여 준다. 단원별로 나뉜 짜임 덕분에 낱낱의 조각을 익히고 다른 일이나 자료 뭉치에 맞게 고치기 쉽다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 자연어 처리 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
코드를 끝까지 읽고 핵심 꾸밈 결정을 가려내어라. 구체적인 짜기 고름 세 가지를 들고 저마다 왜 말 모델에 알맞은지 설명하여라.

??? success "연습문제 1 풀이"
    설계 결정은 구현마다 다르지만 흔히 다음이 포함된다. (1) 활성화 함수의 선택 — ReLU 계열은 포화되지 않는 경사를 주어 학습을 빠르게 한다. (2) 정규화 전략 — 배치 정규화는 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 — 있을 경우 건너뛰는 경로를 제공하여 깊은 신경망에서도 경사가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 절충을 반영한다.

---

**연습문제 2.**
입력이 기대하는 모양과 자료형을 갖는지 확인하도록 주 함수나 클래스에 입력 검증을 추가하라. 잘못된 입력에는 유익한 오류 메시지를 내라.

??? success "연습문제 2 풀이"
    `forward` 메서드(또는 해당 함수)의 첫머리에 다음과 같은 검사를 추가한다. `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`와 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. 모양을 검증할 때는 중요한 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 유익한 오류 메시지는 디버깅 속도를 크게 높이고 코드를 재사용하기에도 더 견고하게 만든다.

---

**연습문제 3.**
은닉 크기가 $h$이고 입력 크기가 $x$로 같을 때 LSTM 셀과 GRU 셀의 매개변수 개수를 비교하라. 어느 쪽이 더 적으며 그 이유는 무엇인가?

??? success "연습문제 3 풀이"
    LSTM에는 4개의 게이트(입력, 망각, 셀, 출력)가 있고 각 게이트가 입력과 은닉 상태 양쪽에 대한 가중치 행렬을 가지므로 $4 \times (x \cdot h + h \cdot h + h) = 4(xh + h^2 + h)$개의 매개변수를 갖는다. GRU에는 3개의 게이트(재설정, 갱신, 새 상태)가 있어 $3 \times (x \cdot h + h \cdot h + h) = 3(xh + h^2 + h)$개이다. GRU는 게이트를 4개 대신 3개 쓰고 셀 상태와 은닉 상태를 합치므로 LSTM의 75%에 해당하는 매개변수를 갖는다. 실무에서 GRU는 매개변수가 적은데도 LSTM에 견줄 만한 성능을 내는 경우가 많다.

---

**연습문제 4.**
실습 09 구현을 검증하는 종합 시험 함수를 작성하라. 빈 입력, 원소가 하나뿐인 입력, 아주 큰 입력, 극단적인 값(0, 아주 큰 수)을 가진 입력 같은 경계 상황을 시험하라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_generationstrategies():
        model = GenerationStrategies(...)
        # 보통의 입력
        assert model(normal_input).shape == expected_shape
        # 원소가 하나인 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값 (넘침을 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 경사의 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    경사의 흐름을 시험하는 것은 그 구조가 처음부터 끝까지 이어지는 학습을 지원하는지 확인하는 데 특히 중요하다.

## 정리하며

**다룬 것** — 실습 09

여기 짠 것은 깔끔하고 읽기 좋은 PyTorch 코드로 말 모델의 핵심 개념을 보여 준다.

고갱이 갈래는 `GenerationStrategies`, `RepetitionPenalty`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
