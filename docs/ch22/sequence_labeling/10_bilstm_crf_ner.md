# BiLSTM-CRF 구성의 시작: 토큰별 점수

BiLSTM-CRF는 문맥 인코더의 태그 점수와 시퀀스 전이 모델을 연결하는 구조다. 다만 아래 `BiLSTM_CRF`는 embedding·BiLSTM·선형 레이어에서 emission을 반환하는 부분만 구현한다. 클래스 이름과 달리 CRF 정규화·정답 점수·로스·Viterbi는 포함하지 않는다.

## 1. 코드와 기존 출력

```python
"""
이름 알아보기를 위한 두 방향 LSTM-CRF
========================================

가장 앞선 차례 이름표 붙이기를 위해 CRF 레이어를 얹은 두 방향 LSTM.

구조:
- 묻힘 레이어
- 두 방향 LSTM
- 짜임 있는 어림을 위한 CRF 레이어

지은이: 러닝 목적
날짜: 2025
"""

import torch
import torch.nn as nn
from typing import List, Tuple

# ========================================================================
# 메인
# ========================================================================


class BiLSTM_CRF(nn.Module):
    """
    이름 알아보기를 위한 두 방향 LSTM-CRF 모델.
    
    구조:
    들임 → 묻힘 → 두 방향 LSTM → 선형 → CRF → 내놓음
    """
    
    def __init__(self, vocab_size: int, tag_size: int, 
                 embedding_dim: int = 100, hidden_dim: int = 200):
        """
        두 방향 LSTM-CRF 첫자리매김.
        
        인수:
            vocab_size: 낱말 곳간의 크기
            tag_size: 것 이름표의 개수
            embedding_dim: 낱말 임베딩의 차원
            hidden_dim: LSTM의 숨은 차원
        """
        super(BiLSTM_CRF, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_size = tag_size
        
        # 임베딩 레이어
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        
        # 두 방향 LSTM 레이어
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                           num_layers=1, bidirectional=True, batch_first=True)
        
        # 이름표 공간으로 내리쬐는 선형 레이어
        self.hidden2tag = nn.Linear(hidden_dim, tag_size)
        
        # CRF 넘어가기 파라메터
        # transitions[i, j] = 이름표 j에서 이름표 i로 넘어가는 점수
        self.transitions = nn.Parameter(torch.randn(tag_size, tag_size))
        
        # START 이름표로는 결코 넘어가지 않는다
        self.transitions.data[:, tag_size - 2] = -10000
        # END 이름표에서는 결코 넘어가지 않는다
        self.transitions.data[tag_size - 1, :] = -10000
    
    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        두 방향 LSTM에서 내보냄 점수 얻기.
        
        인수:
            sentence: 낱말 번호를 담은 [batch_size, seq_len] 텐서
            
        반환값:
            emission_scores: [batch_size, seq_len, tag_size]
        """
        # 묻힘을 얻는다
        embeds = self.word_embeds(sentence)  # [batch, seq_len, embedding_dim]
        
        # 두 방향 LSTM에 통과시키기
        lstm_out, _ = self.lstm(embeds)  # [batch, seq_len, hidden_dim]
        
        # 이름표 공간으로 내리쬐기
        emissions = self.hidden2tag(lstm_out)  # [batch, seq_len, tag_size]
        
        return emissions


if __name__ == "__main__":
    # 예
    vocab_size = 1000
    tag_size = 10
    
    model = BiLSTM_CRF(vocab_size, tag_size)
    
    # 예제 입력
    sentence = torch.randint(0, vocab_size, (1, 5))  # 배치 크기 1, 길이 5
    emissions = model(sentence)
    
    print(f"Emissions shape: {emissions.shape}")
    print("BiLSTM-CRF model created successfully!")
```

**기존 출력:**

```
Emissions shape: torch.Size([1, 5, 10])
BiLSTM-CRF model created successfully!
```

## 2. 실제 순전파와 파라메터

문장 토큰 ID는 B×L 정수 텐서다. embedding_dim100이면 B×L×100을 만들고, hidden_dim200의 방향별100차원 LSTM을 연결해 B×L×200을 얻는다. 선형 레이어는 B×L×K 태그 점수를 반환한다. 예제 B1·L5·K10에서 출력은1×5×10이며 확률이나 최적 태그 열은 아니다.

hidden_dim은 방향별 바닥 나눗셈과 선형 입력을 맞추려면 짝수여야 한다. 패딩 ID·길이 packing·mask가 없으므로 길이가 다른 문장의 패딩도 BiLSTM 문맥에 참여할 수 있다. 임의로 생성한 토큰 ID의 모양 확인은 NER 학습·정확도를 검증하지 않는다.

transitions는 K×K 학습 가능한 파라메터로 등록하지만 forward에서 사용하지 않는다. 따라서 이 출력으로 로스를 계산해도 현재 경로에서는 전이 파라메터에 경사가 생기지 않는다. 마지막 두 태그를 특별한 시작·종료로 간주한−10000 대입도 실제 CRF 계산에 적용되지 않는다. 태그 매핑·전이 방향·시작/종료 모델을 먼저 정의해야 한다.

## 3. CRF와 학습을 연결하려면

같은 전이 방향과 emission으로 정답 시퀀스 점수·log partition을 계산하고 차이를 최소화한다. 추론도 같은 점수의 Viterbi를 사용한다. 시작·종료·BIO 제약과 비어 있지 않은 유효 길이·패딩 태그의 계약이 필요하다. 별도의 [CRF 절](crf.md)과 [BiLSTM 절](bilstm_ner.md)의 코드에도 방향·마스크 한계가 설명되어 있으므로 단순 복사만으로 완성되었다고 가정하지 않는다.

단일 LSTM에서 PyTorch 기본 bias를 포함한 파라메터는 방향별4DH+4H²+8H다. 후보 셀 계산까지 포함한 네 묶음이며 모두가 sigmoid 게이트인 것은 아니다. 레이어·방향·입력 차원과 분류 head까지 맞춰 비교해야 한다.

## 4. 구조 이름과 실험 범위

코드의 성공 메시지는 객체 생성과 emission 모양의 시연이다. 가장 높은 성능이나 완성된 CRF 학습을 의미하지 않는다. 실제 비교에는 데이터·유형·태그·전처리·span 지표와 최고 체크포인트의 기록이 필요하다. 이 문서에서는 코드·출력을 유지하며 학습·추론 예제를 실행하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
예제 vocab_size1000·tag_size10에서 기본 구조의 파라메터를 계산하라.

</div>

??? success "풀이"

    embedding은100000개, 방향별 LSTM은80800개씩이라161600개, 선형은200×10+10=2010개, 전이는100개다. 합계263710개이며 전이는 등록되었지만 forward에서 사용하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력과 반환값의 차원·자료형 계약을 설명하라.

</div>

??? success "풀이"

    입력은 유효 어휘 범위의 B×L 정수 ID이고 반환은 B×L×K 실수 emission이다. 패딩·길이 mask는 현재 없고 마지막 두 태그의 특별한 의미도 외부 매핑이 필요하다. 영상형 float32 입력 검사는 맞지 않는다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
같은 입력D·은닉H인 PyTorch 단일 방향 LSTM과 GRU의 파라메터를 비교하라.

</div>

??? success "풀이"

    LSTM은4DH+4H²+8H, GRU는3DH+3H²+6H다. 같은 조건이면 GRU가3/4이지만 양방향·레이어별 입력·projection이 달라지면 다시 계산해야 한다. 파라메터 비율이 성능·지연시간 비율은 아니다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 ModuleList 조각으로 레이어만 늘리면 CRF가 완성되는가?

</div>

??? success "풀이"

    YourBlock 정의와 레이어별 입출력·양방향 연결 차원을 정해야 한다. 전이·partition·정답 로스·Viterbi·마스크가 별도로 필요하다. ModuleList는 모듈 등록 도구이며 사용하지 않은 transitions를 자동으로 계산 경로에 연결하지 않는다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

현재 예제는 BiLSTM의 토큰별 emission까지만 계산한다. 전이 파라메터의 등록과 실제 CRF 학습·디코딩은 다르며 입력 길이·태그 매핑·전이 방향·로스·추론을 일관되게 연결해야 한다.
