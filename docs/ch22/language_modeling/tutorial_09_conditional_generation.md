# 조건부 텍스트 생성과 디코딩

조건부 생성은 프롬프트 같은 조건을 주고 다음 토큰 분포를 반복해서 사용하는 과정이다. 이 페이지는 이미 학습된 모델의 로짓을 그리디·top-k·top-p·빔 서치로 선택하는 보조 함수를 다룬다. 프롬프트 튜닝이나 조건부 모델의 학습을 구현하는 것은 아니다. 어떤 디코딩도 사실성·관련성·문장 품질을 자동으로 보장하지 않는다.

## 1. 코드와 기존 출력

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
                
                # 쌓인 확률이 임계값을 넘는 토막 없애기
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
        스텝마다 beam_width개의 가설을 지닌다.
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
   - 확률에 맡기며 여러 클래스이다
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
7. 여러 클래스 빔 찾기 짜기(여러 클래스의 내놓음)
8. 되풀이를 피하는 덮음 구조 더하기
    """)
```

**기존 출력:**

```
Text Generation Strategies Comparison
======================================================================

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
   - 확률에 맡기며 여러 클래스이다
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
    

익힘 문제:
1. 만들어 내기에 되풀이 벌주기 짜기
2. 빔 너비를 달리해 빔 찾기 견주기
3. 빔 찾기의 길이 고르게 맞추기 짜기
4. 상위 k와 알갱이 표집을 아울러 보기
5. 제약을 둔 풀기 짜기(특정 낱말을 강제하기)
6. 앞가지 다듬기로 다스릴 수 있는 만들어 내기 만들기
7. 여러 클래스 빔 찾기 짜기(여러 클래스의 내놓음)
8. 되풀이를 피하는 덮음 구조 더하기
    
```

## 2. 모델과 입력의 계약

입력은 비어 있지 않은 B×T 정수 토큰 ID이고 모델의 출력 로짓은 B×T×V여야 한다. RNN·LSTM 속성이 있으면 (logits,hidden)을, 그 밖에는 로짓 텐서를 직접 반환한다고 가정한다. Hugging Face의 출력 객체나 고정 문맥 순방향 모델이 자동으로 호환되는 인터페이스는 아니다.

각 반복에서 전체 생성 접두사를 다시 모델에 넣는다. RNN 상태를 이전 호출에서 넘기지 않으므로 접두사 중복 누적 오류는 없지만 계산을 재사용하지도 않는다. 트랜스포머의 KV cache도 구현하지 않는다. max_length는 이 파일에서는 새 토큰 추가 횟수다. 반환 텐서는 원래 프롬프트를 포함한다.

그리디·샘플링 함수의 종료 조건은 next_token.item()을 사용해 배치 크기 1을 전제로 한다. 여러 입력의 개별 종료와 padding 마스크는 구현하지 않는다. eval·no_grad는 추론 설정이지 반환 텍스트의 정확성 검증이 아니다.

## 3. 온도와 후보 필터링

top-k는 온도로 나눈 로짓의 상위 k개를 정규화해 샘플링한다. T>0, 1≤k≤V와 유한한 유효 로짓이 필요하다. 낮은 온도는 높은 확률에 집중시키고 높은 온도는 분포를 평평하게 한다. 모든 과제에 통하는 k·p·온도의 최적값은 없다.

표준 top-p는 확률 내림차순의 누적합이 p에 도달하는 후보까지 포함한다. 현재 nucleus_sampling은 cumulative_probs>p인 위치를 바로 제거하고 첫 후보만 예외로 남긴다. 임계값을 처음 넘는 후보를 유지하도록 마스크를 이동하지 않아 표준 정의보다 좁은 집합을 선택할 수 있다. 예를 들어 (0.6,0.3,0.1), p=0.8이면 표준은 앞 두 개지만 현재 코드는 첫 번째만 남긴다.

또한 filtered_probs.sum()은 배치 전체 합이다. 행별로 합 1인 확률을 얻는 정규화가 아니다. torch.multinomial은 행별 상대 웨이트로도 뽑을 수 있어 이 차이만으로 즉시 샘플이 잘못된다고 단정할 수는 없지만, 반환 확률의 의미와 수치 안정성은 별도로 맞춰야 한다. p 범위·모든 후보 제거·NaN도 검토한다.

## 4. 빔 서치의 실제 범위

빔 서치는 다음 토큰의 로그확률을 누적하고 각 단계에서 제한된 수의 가설을 남기는 근사 탐색이다. 이 구현은 EOS를 만나도 종료 가설을 보관하거나 확장을 중지하지 않으며 length_penalty와 vocab 인자도 사용하지 않는다. 최종 선택은 첫 입력의 빔들에서만 하므로 배치 전체 결과를 반환하는 함수가 아니다.

beam_scores를 CPU에서 만들고 새 점수 목록도 torch.tensor로 재구성하므로 GPU 로짓과 일관된 장치 처리가 없다. 모델에 fc.out_features가 있다는 구조 가정도 필요하다. 따라서 ‘빔 서치가 그리디보다 항상 좋다’거나 길이를 보정한 최적 완료 문장을 반환한다는 설명은 맞지 않는다.

## 5. 반복 페널티와 실제 시연

RepetitionPenalty.apply는 등장한 토큰의 로짓을 penalty로 나눈다. penalty>1일 때 양수 로짓은 줄지만 음수 로짓은 0에 가까워져 오히려 상대 확률이 커질 수 있다. 부호에 관계없이 반복을 억제하는 변환이 아니며, 이 메서드는 위 생성 함수에 실제로 연결되어 있지도 않다. 1차원 로짓과 정수 토큰 목록을 가정한다는 점도 명확히 해야 한다.

main은 전략의 설명 문구만 출력한다. 모델·자료·정량 비교나 생성 결과를 측정하지 않으므로 출력에 적힌 고정 설정과 품질 순위는 검증 결과가 아니다. 생성 방법은 동일 프롬프트·길이·평가 기준에서 비교해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 페이지가 조건부 학습·프롬프트 튜닝을 구현하는지, 실제 조건과 디코딩 입력은 무엇인지 설명하라.

</div>

??? success "연습문제 1 풀이"

    입력 프롬프트의 토큰 ID가 조건이며 이미 준비된 언어 모델의 다음 토큰 로짓을 사용한다. 학습 함수·프롬프트 파라메터 최적화·어댑터는 없다. 반환에는 프롬프트도 포함된다. main은 방법 설명을 출력할 뿐 실제 모델로 생성하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
생성 함수의 올바른 입력 검증 항목을 제시하고, 이미지용 float32·채널 검사로 대체할 수 없는 이유를 설명하라.

</div>

??? success "연습문제 2 풀이"

    입력은 비어 있지 않은 B×T 정수 ID로 어휘 범위 안에 있어야 한다. 모델과 장치가 같고 출력은 B×T×V여야 한다. T>0, 유효 k·p·길이·빔 폭을 검사하며 현재 EOS 종료는 배치 크기 1만 지원한다. 토큰 입력에 float32나 B×C×H×W 계약을 강제하는 것은 잘못이다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
로짓 (2,−2)에 penalty=2를 적용할 때 현재 반복 페널티의 문제를 설명하라.

</div>

??? success "연습문제 3 풀이"

    반복 토큰의 양수 로짓 2는 1로 줄지만 음수 −2는 −1로 커진다. 후자는 다른 로짓이 같을 때 해당 토큰의 소프트맥스 확률을 높이는 방향이다. 부호를 구분한 로짓 변환이나 빈도·존재 기반 감산 등 목적에 맞는 정의가 필요하며, 생성 루프에 실제 적용해야 한다. 페널티로 반복 감소나 품질 향상이 항상 보장되는 것은 아니다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 test_generationstrategies의 구조가 실제 보조 함수 인터페이스와 맞는지 검토하고 검증 대상을 제시하라.

</div>

??? success "연습문제 4 풀이"

    GenerationStrategies는 정적 생성 메서드의 모음이며 학습 파라메터를 가진 호출 가능한 모델이 아니다. 아래 생성자·forward·parameters·backward 검사는 자리표시자이고 완성된 테스트가 아니다. 검증할 대상은 고정 로짓 모델에서의 토큰 선택, top-p 경계 후보 포함, EOS 종료, 반환 프롬프트·길이, 배치별 결과·장치와 반복 페널티의 부호다. 추론 함수에 경사 존재를 요구하지 않는다.

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


## 정리하며

조건부 생성은 모델 분포와 토큰 선택 규칙을 함께 해석해야 한다. 현재 top-p 경계 처리·배치 종료·빔 반환과 반복 페널티에는 제약이 있으므로 설명 문구를 실제 품질 검증이나 완성된 생성 시스템으로 취급하지 않는다.
