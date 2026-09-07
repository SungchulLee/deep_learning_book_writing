# 모델

글자 수준 자기 되돌이 말 모델. 이 단원은 다음을 배우는 신경 자기 되돌이 모델을 짠다

자기 되돌이 모델은 앞선 모든 낱개를 조건으로 삼아 낱개마다 미리 헤아려 자료를 만든다. 이 단원은 자기 되돌이 모델 부품의 짜기를 보이며 차례대로 만들어 내는 과정과 그 얼개의 요구를 그려 보인다.

## 코드

```python
"""
글자 수준 자기 되돌이 말 모델

이 단원은 글자 하나씩 글을 만들어 내는 신경 자기 되돌이 모델을 짠다.
곧 글자마다 앞선 글자를 바탕으로 헤아린다.
차례의 앞선 모든 글자를 바탕으로 한다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

# ========================================================================
# 메인
# ========================================================================


class CharRNN(nn.Module):
    """
    글 만들어 내기를 위한 글자 수준 되돌이 신경망.
    
    이는 다음과 같은 자기 되돌이 모델이다.
    1. 글자의 차례를 들임으로 받는다
    2. 되돌이 신경망(긴 짧은 기억)으로 다룬다
    3. 다음 글자를 헤아린다
    
    구조:
        박아 넣기 -> 긴 짧은 기억 -> 선형 -> 소프트맥스
    """
    
    def __init__(self, 
                 vocab_size: int,
                 embedding_dim: int = 64,
                 hidden_dim: int = 128,
                 n_layers: int = 2):
        """
        글자 되돌이 신경망을 첫자리매김한다.
        
        인수:
            vocab_size: 낱말 속 서로 다른 글자의 수
            embedding_dim: 글자 박아 넣기의 차원
            hidden_dim: 긴 짧은 기억의 숨은 낱개 수
            n_layers: 긴 짧은 기억 층의 수
        """
        super(CharRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # 박아 넣기 층: 글자 어깨수를 빽빽한 벡터로 바꾼다
        # 글자마다 배울 수 있는 박아 넣기 벡터를 갖는다
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 긴 짧은 기억: 박아 넣기의 차례를 다룬다
        # batch_first=True는 들임 꼴이 [묶음, 차례, 특징]이라는 뜻이다
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            n_layers,
            batch_first=True,
            dropout=0.2 if n_layers > 1 else 0  # 긴 짧은 기억 층 사이의 떨구기
        )
        
        # 내놓기 층: 긴 짧은 기억의 숨은 상태를 낱말 점수로 옮긴다
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, 
                x: torch.Tensor, 
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        모델을 지나는 앞먹임.
        
        인수:
            x: 꼴 [묶음 크기, 차례 길이]인 들임 텐서
               글자 어깨수를 담는다
            hidden: 앞선 앞먹임에서 온 숨은 상태(있으면)
                    긴 짧은 기억을 위한 (h_0, c_0) 짝
                    
        반환값:
            output: 꼴 [묶음 크기, 낱말 수]인 로짓
                   낱말 속 글자마다의 점수
            hidden: 갱신된 숨은 상태
        """
        # 묶음 크기(차례의 수)를 얻는다
        batch_size = x.size(0)
        
        # 1. 글자를 박아 넣는다
        # 들임: [묶음, 차례 길이]
        # 내놓기: [묶음, 차례 길이, 박아 넣기 차원]
        embedded = self.embedding(x)
        
        # 2. 긴 짧은 기억으로 다룬다
        # 숨은 상태를 주지 않으면 긴 짧은 기억이 0으로 첫자리매김한다
        # lstm_out: [묶음, 차례 길이, 숨은 차원]
        # hidden: (h_n, c_n) - 마지막 숨은 상태와 칸 상태
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # 3. 헤아리기에 마지막 때 걸음만 쓴다
        # 차례 다음에 오는 글자를 헤아리려 한다
        # 꼴: [묶음, 숨은 차원]
        last_output = lstm_out[:, -1, :]
        
        # 4. 낱말 점수로 옮긴다
        # 꼴: [묶음, 낱말 수]
        output = self.fc(last_output)
        
        return output, hidden
    
    def generate(self, 
                start_sequence: torch.Tensor,
                length: int,
                temperature: float = 1.0) -> torch.Tensor:
        """
        자기 회귀로 글을 짓는다.
        
        이것이 자기 되돌이 만들어 내기의 핵심 과정이다.
        1. 씨앗 차례에서 시작한다
        2. 다음 글자를 헤아린다
        3. 헤아린 값을 차례에 덧붙인다
        4. 걸음 2-3을 되풀이한다
        
        인수:
            start_sequence: 글자 어깨수의 씨앗 차례 [sequence_length]
            length: 만들 글자의 수
            temperature: 표본 추출의 온도 (높을수록 더 무작위)
                        1.0 = 보통, <1.0 = 더 조심스럽다, >1.0 = 더 마구잡이다
                        
        반환값:
            만든 글자 어깨수의 차례
        """
        self.eval()  # 따지기 모드로 둔다
        
        # 씨앗 차례에서 시작한다
        generated = start_sequence.clone().unsqueeze(0)  # 배치 차원을 더한다
        
        with torch.no_grad():
            hidden = None  # 숨은 상태 없이 시작한다
            
            for _ in range(length):
                # 다음 글자의 헤아림을 얻는다
                output, hidden = self.forward(generated, hidden)
                
                # 로짓에 온도를 쓴다
                # 온도가 마구잡이 정도를 다스린다:
                # - 낮은 온도(< 1): 더 자신 있고 덜 다양하다
                # - 높은 온도(> 1): 덜 자신 있고 더 다양하다
                output = output / temperature
                
                # 로짓을 확률로 바꾸기
                probs = F.softmax(output, dim=-1)
                
                # 확률 분포에서 다음 글자를 뽑는다
                # 그래서 만들어 내기가 정해진 것이 아니라 확률에 따르게(마구잡이가) 된다
                next_char = torch.multinomial(probs, 1)
                
                # 만들어진 수열에 덧붙인다
                generated = torch.cat([generated, next_char], dim=1)
        
        # 묶음 차원을 없애고 돌려준다
        return generated.squeeze(0)


class SimpleCharTransformer(nn.Module):
    """
    글자 수준 말 나타내기를 위한 단순한 변환기.
    
    이는 되돌이 대신 스스로 눈길을 쓰는 더 요즘다운 되돌이 신경망의 대안이다.
    변환기는 GPT 같은 모델의 바탕 얼개이다.
    GPT 같은 모델.
    """
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 64,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 max_seq_length: int = 100):
        """
        변환기 모델을 첫자리매김한다.
        
        인수:
            vocab_size: 서로 다른 글자의 수
            embedding_dim: 박아 넣기의 차원(n_heads으로 나누어떨어져야 한다)
            n_heads: 눈길 머리의 수
            n_layers: 변환기 층의 수
            max_seq_length: 최대 차례 길이(자리 부호화용)
        """
        super(SimpleCharTransformer, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # 토큰 박아 넣기: 글자 어깨수를 벡터로 바꾼다
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 자리 박아 넣기: 자리 앎을 더한다
        # 변환기에는 본디 차례가 없으므로 자리 앎을 더한다
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)
        
        # 변환기 부호기 층
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # 출력층
        self.fc = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        변환기를 지나는 앞먹임.
        
        인수:
            x: 들임 텐서 [묶음 크기, 차례 길이]
            
        반환값:
            내놓기 로짓 [묶음 크기, 낱말 수]
        """
        batch_size, seq_length = x.shape
        
        # 토큰 박아 넣기
        token_emb = self.token_embedding(x)
        
        # 자리 박아 넣기
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)
        
        # 토큰 박아 넣기와 자리 박아 넣기를 아우른다
        x = token_emb + pos_emb
        
        # 인과 가림막을 만든다: 자리마다 앞 자리만 볼 수 있다
        # 이는 자기 되돌이로 나타내는 데 결정적이다!
        mask = torch.triu(torch.ones(seq_length, seq_length, device=x.device), diagonal=1)
        mask = mask.bool()
        
        # 인과 가림막과 함께 변환기를 쓴다
        x = self.transformer(x, mask=mask)
        
        # 헤아리기에 마지막 자리를 쓴다
        x = x[:, -1, :]
        
        # 낱말로 옮긴다
        output = self.fc(x)
        
        return output


if __name__ == "__main__":
    """
    보여 주기: 흉내 자료로 모델을 시험한다
    """
    
    # 초매개변수
    vocab_size = 50  # 서로 다른 글자 50개
    batch_size = 16
    seq_length = 20
    
    # 임시 입력 만들기
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    print("=" * 60)
    print("Testing CharRNN")
    print("=" * 60)
    
    # 모형을 시작한다
    rnn_model = CharRNN(vocab_size, embedding_dim=32, hidden_dim=64, n_layers=2)
    
    # 매개변수 개수 세기
    n_params = sum(p.numel() for p in rnn_model.parameters())
    print(f"Number of parameters: {n_params:,}")
    
    # 순전파
    output, hidden = rnn_model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output represents scores for {vocab_size} possible next characters")
    
    # 만들어 내기를 시험한다
    seed = torch.randint(0, vocab_size, (10,))
    generated = rnn_model.generate(seed, length=20, temperature=1.0)
    print(f"\nGenerated sequence length: {len(generated)}")
    
    print("\n" + "=" * 60)
    print("Testing SimpleCharTransformer")
    print("=" * 60)
    
    # 변환기를 첫자리매김한다
    transformer_model = SimpleCharTransformer(
        vocab_size, 
        embedding_dim=64,  # n_heads으로 나누어떨어져야 한다
        n_heads=4,
        n_layers=2
    )
    
    n_params = sum(p.numel() for p in transformer_model.parameters())
    print(f"Number of parameters: {n_params:,}")
    
    # 순전파
    output = transformer_model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    print("\n✓ Both models working correctly!")```

## 논의

이 짜기는 갈래 2개(`CharRNN`, `SimpleCharTransformer`)를 뜻매김하며 이들이 함께 온전한 자기 되돌이 모델 얼개를 이룬다. 갈래마다 뚜렷이 구분되는 부품을 감싸므로 코드가 조각으로 나뉘고 넓히기 쉽다. `forward` 방법은 PyTorch가 자동 미분에 쓰는 셈 그래프를 뜻매김한다.

여기서 보인 결은 더 복잡한 경우로 자연스럽게 넓혀진다. 웃매개변수와 얼개 변형, 여러 자료 묶음을 실험하면 만들어 내는 모델 일에 대한 이해가 깊어지고 실제 직관이 쌓인다.

## 연습문제

**연습문제 1.**
기본 첫자리매김에서 `CharRNN`의 배울 수 있는 매개변수 총수를 셈하라. 무게와 치우침을 모두 넣어 층별로 나누어 적어라.

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)` 각각에는 `in_features * out_features`개의 가중치 매개변수와 (`bias=False`가 아닌 한) `out_features`개의 편향 매개변수가 있다. `nn.Conv2d(in_c, out_c, k)`에는 `in_c * out_c * k * k`개의 가중치와 `out_c`개의 편향이 있다. `nn.Embedding(num, dim)`에는 `num * dim`개의 매개변수가 있다. 모든 층에 대해 더하면 된다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

---

**연습문제 2.**
어텐션 가중치 뒤에(값과 곱하기 전에) 드롭아웃 층을 추가하라. 학습 중에는 드롭아웃 비율 0.1을 쓴다. 어텐션 드롭아웃이 정칙화에 도움이 되는 이유를 설명하라.

??? success "연습문제 2 풀이"
    `__init__`에 `self.attn_dropout = nn.Dropout(0.1)`을 추가하고 소프트맥스 뒤에 적용한다. `attn_weights = self.attn_dropout(F.softmax(scores, dim=-1))`이다. 어텐션 드롭아웃은 학습 중에 일부 어텐션 가중치를 무작위로 0으로 만들어, 모델이 특정 토큰 사이의 관계에 지나치게 기대지 않게 한다. 이는 모델이 어텐션을 더 고르게 분산시키고 더 견고한 표현을 배우도록 북돋우며, 표준 드롭아웃이 뉴런의 공적응을 막는 것과 비슷하다.

---

**연습문제 3.**
자기 어텐션의 계산 복잡도를 열의 길이 $n$과 모델 차원 $d$의 함수로 설명하라. 이것이 왜 긴 열에 대해 Longformer나 Linformer 같은 구조의 동기가 되는가?

??? success "연습문제 3 풀이"
    표준 자기 어텐션은 $n \times n$ 어텐션 행렬을 계산하므로 시간 복잡도가 $O(n^2 d)$이고 어텐션 가중치에 $O(n^2)$의 메모리가 든다. 열이 길면(예: $n = 4096$) 감당하기 어려워진다. Longformer는 국소적인 미끄럼창 어텐션($w$이 창 크기일 때 $O(n \cdot w \cdot d)$)과 선택된 토큰에 대한 희소한 전역 어텐션을 결합한다. Linformer는 키와 값을 더 낮은 차원 $k \ll n$으로 사영하여 복잡도를 $O(n \cdot k \cdot d)$으로 줄인다. 둘 다 표현력을 조금 내주고 긴 입력에서의 실용적인 효율을 얻는다.

---

**연습문제 4.**
`CharRNN`을 층이나 덩이의 수를 맞출 수 있게 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 얼개를 만들어라. 2, 4, 8층으로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되풀이하라. (여느 파이썬 목록이 아니라) `nn.ModuleList`을 쓰면 PyTorch가 모든 매개변수를 가장 좋게 하기에 등록한다. `for n in [2, 4, 8]: model = CharRNN(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험하라.
