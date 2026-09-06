# 두 방향 LSTM-CRF 이름 알아보기

이름 알아보기를 위한 두 방향 LSTM-CRF. 가장 앞선 차례 이름표 붙이기를 위해 CRF 층을 얹은 두 방향 LSTM.

자연어 다루기는 깊은 배움 방식으로 크게 달라졌다. 이 단원은 차례 이름표 붙이기 재주를 보여 주며, 신경망이 글을 어떻게 다루고 만들어 내는지 밝히는 실전 짜기를 준다.

## 코드

```python
"""
이름 알아보기를 위한 두 방향 LSTM-CRF
========================================

가장 앞선 차례 이름표 붙이기를 위해 CRF 층을 얹은 두 방향 LSTM.

구조:
- 묻힘 층
- 두 방향 LSTM
- 짜임 있는 어림을 위한 CRF 층

지은이: 배움 목적
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
        
        # 임베딩 층
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        
        # 두 방향 LSTM 층
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                           num_layers=1, bidirectional=True, batch_first=True)
        
        # 이름표 공간으로 내리쬐는 선형 층
        self.hidden2tag = nn.Linear(hidden_dim, tag_size)
        
        # CRF 넘어가기 매개변수
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
    sentence = torch.randint(0, vocab_size, (1, 5))  # 묶음 크기 1, 길이 5
    emissions = model(sentence)
    
    print(f"Emissions shape: {emissions.shape}")
    print("BiLSTM-CRF model created successfully!")```

## 논의

`BiLSTM_CRF` 클래스는 PyTorch의 `nn.Module` 사이를 써서 모델 얼개를 감싼다. `forward` 메서드가 셈 그래프를 정하므로 익히는 동안 PyTorch의 자동 미분 체계가 기울기 셈을 알아서 다룬다. 이 단원별 꾸밈 덕분에 낱낱의 조각을 고치거나 모델을 더 큰 물길에 끼워 넣기가 쉽다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 자연어 처리 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
붙박이 첫자리매김일 때 `BiLSTM_CRF`의 배울 수 있는 매개변수 전체 개수를 셈하여라. 무게와 치우침을 모두 넣어 층별로 나누어 세어라.

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)` 각각에는 `in_features * out_features`개의 가중치 매개변수와 (`bias=False`가 아닌 한) `out_features`개의 편향 매개변수가 있다. `nn.Conv2d(in_c, out_c, k)`에는 `in_c * out_c * k * k`개의 가중치와 `out_c`개의 편향이 있다. `nn.Embedding(num, dim)`에는 `num * dim`개의 매개변수가 있다. 모든 층에 대해 더하면 된다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

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
층이나 덩이의 개수를 정할 수 있도록 `BiLSTM_CRF`을 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`로 깊이가 바뀌는 얼개를 만들어라. 층 2, 4, 8개로 시험하여라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 되풀이한다. (그냥 파이썬 목록이 아니라) `nn.ModuleList`를 써야 PyTorch가 가장 좋게 하기에 쓸 매개변수를 모두 등록한다. 시험: `for n in [2, 4, 8]: model = BiLSTM_CRF(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
