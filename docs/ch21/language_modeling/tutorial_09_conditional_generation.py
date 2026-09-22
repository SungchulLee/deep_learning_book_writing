"""tutorial_09_conditional_generation — tutorial_09_conditional_generation 페이지의 코드를 모듈로 쓸 수 있게 옮겨 놓은 것이다.

이 파일은 ch20/language_modeling/tutorial_09_conditional_generation.md 에서 자동으로 뽑아낸 것이므로, 고칠 일이 있으면
그 페이지를 고치고 다시 뽑아내는 편이 낫다.
"""

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
