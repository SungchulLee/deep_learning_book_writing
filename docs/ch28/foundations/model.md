# 문자 단위 LSTM과 Transformer

이 절은 문자 문맥에서 다음 문자 하나의 logits를 예측하는 두 대안 모델을 정의한다. CharRNN은 LSTM의 마지막 은닉 출력을 사용하고 SimpleCharTransformer는 위치 임베딩과 인과적 self-attention을 사용한다. 출력은 두 모델 모두 `(B,vocab_size)`이므로 [문자 데이터 도구](data_2.md)의 창당 타깃 하나와 대응한다.

## 1. 구현과 출력

```python
"""
글자 수준 자기 회귀 말 모델

이 단원은 글자 하나씩 글을 만들어 내는 뉴럴 자기 회귀 모델을 짠다.
곧 글자마다 앞선 글자를 바탕으로 헤아린다.
차례의 앞선 모든 글자를 바탕으로 한다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

# ========================================================================
# 메인
# ========================================================================


class CharRNN(nn.Module):
    """
    글 만들어 내기를 위한 글자 수준 회귀 뉴럴 넷트웍.
    
    이는 다음과 같은 자기 회귀 모델이다.
    1. 글자의 차례를 들임으로 받는다
    2. 회귀 뉴럴 넷트웍(긴 짧은 기억)으로 다룬다
    3. 다음 글자를 헤아린다
    
    구조:
        임베딩 -> 긴 짧은 기억 -> 선형 -> 소프트맥스
    """
    
    def __init__(self, 
                 vocab_size: int,
                 embedding_dim: int = 64,
                 hidden_dim: int = 128,
                 n_layers: int = 2):
        """
        글자 회귀 뉴럴 넷트웍을 첫자리매김한다.
        
        인수:
            vocab_size: 낱말 속 서로 다른 글자의 수
            embedding_dim: 글자 임베딩의 차원
            hidden_dim: 긴 짧은 기억의 숨은 낱개 수
            n_layers: 긴 짧은 기억 레이어의 수
        """
        super(CharRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # 임베딩 레이어: 글자 어깨수를 빽빽한 벡터로 바꾼다
        # 글자마다 배울 수 있는 임베딩 벡터를 갖는다
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 긴 짧은 기억: 임베딩의 차례를 다룬다
        # batch_first=True는 들임 꼴이 [배치, 차례, 피처]이라는 뜻이다
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            n_layers,
            batch_first=True,
            dropout=0.2 if n_layers > 1 else 0  # 긴 짧은 기억 레이어 사이의 떨구기
        )
        
        # 내놓기 레이어: 긴 짧은 기억의 숨은 상태를 낱말 점수로 옮긴다
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, 
                x: torch.Tensor, 
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        모델을 지나는 피드포워드.
        
        인수:
            x: 꼴 [배치 크기, 차례 길이]인 들임 텐서
               글자 어깨수를 담는다
            hidden: 앞선 피드포워드에서 온 숨은 상태(있으면)
                    긴 짧은 기억을 위한 (h_0, c_0) 짝
                    
        반환값:
            output: 꼴 [배치 크기, 낱말 수]인 로짓
                   낱말 속 글자마다의 점수
            hidden: 갱신된 숨은 상태
        """
        # 배치 크기(차례의 수)를 얻는다
        batch_size = x.size(0)
        
        # 1. 글자를 박아 넣는다
        # 들임: [배치, 차례 길이]
        # 내놓기: [배치, 차례 길이, 임베딩 차원]
        embedded = self.embedding(x)
        
        # 2. 긴 짧은 기억으로 다룬다
        # 숨은 상태를 주지 않으면 긴 짧은 기억이 0으로 첫자리매김한다
        # lstm_out: [배치, 차례 길이, 숨은 차원]
        # hidden: (h_n, c_n) - 마지막 숨은 상태와 칸 상태
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # 3. 헤아리기에 마지막 때 스텝만 쓴다
        # 차례 다음에 오는 글자를 헤아리려 한다
        # 꼴: [배치, 숨은 차원]
        last_output = lstm_out[:, -1, :]
        
        # 4. 낱말 점수로 옮긴다
        # 꼴: [배치, 낱말 수]
        output = self.fc(last_output)
        
        return output, hidden
    
    def generate(self, 
                start_sequence: torch.Tensor,
                length: int,
                temperature: float = 1.0) -> torch.Tensor:
        """
        자기 회귀로 글을 짓는다.
        
        이것이 자기 회귀 만들어 내기의 핵심 과정이다.
        1. 씨앗 차례에서 시작한다
        2. 다음 글자를 헤아린다
        3. 헤아린 값을 차례에 덧붙인다
        4. 스텝 2-3을 되풀이한다
        
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
        
        # 배치 차원을 없애고 돌려준다
        return generated.squeeze(0)


class SimpleCharTransformer(nn.Module):
    """
    글자 수준 말 나타내기를 위한 단순한 변환기.
    
    이는 회귀 대신 셀프 어텐션을 쓰는 더 요즘다운 회귀 뉴럴 넷트웍의 대안이다.
    변환기는 GPT 같은 모델의 바탕 구조이다.
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
            embedding_dim: 임베딩의 차원(n_heads으로 나누어떨어져야 한다)
            n_heads: 어텐션 머리의 수
            n_layers: 변환기 레이어의 수
            max_seq_length: 최대 차례 길이(자리 부호화용)
        """
        super(SimpleCharTransformer, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # 토큰 임베딩: 글자 어깨수를 벡터로 바꾼다
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 자리 임베딩: 자리 앎을 더한다
        # 변환기에는 본디 차례가 없으므로 자리 앎을 더한다
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)
        
        # 변환기 인코더 레이어
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # 출력 레이어
        self.fc = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        변환기를 지나는 피드포워드.
        
        인수:
            x: 들임 텐서 [배치 크기, 차례 길이]
            
        반환값:
            내놓기 로짓 [배치 크기, 낱말 수]
        """
        batch_size, seq_length = x.shape
        
        # 토큰 임베딩
        token_emb = self.token_embedding(x)
        
        # 자리 임베딩
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)
        
        # 토큰 임베딩과 자리 임베딩을 아우른다
        x = token_emb + pos_emb
        
        # 인과 가림막을 만든다: 자리마다 앞 자리만 볼 수 있다
        # 이는 자기 회귀로 나타내는 데 결정적이다!
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
    
    # 초파라메터
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
    
    # 파라메터 개수 세기
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
    
    print("\n✓ Both models working correctly!")
```

입력은 어휘 범위 안의 int64 ID `(B,T)`다. T는 1 이상이어야 한다. LSTM은 임베딩 `(B,T,E)`를 순환 처리하고 마지막 위치를 선형 분류한다. 반환값은 logits와 `(h,c)` 상태이며 각 상태 모양은 `(n_layers,B,H)`다. 상태를 전달하지 않으면 새 창을 기본 초기 상태에서 처리한다.

Transformer는 토큰 임베딩에 학습 가능한 위치 임베딩을 더하고 상삼각 bool 마스크로 미래 위치를 차단한다. 여기의 mask=True는 해당 attention 연결을 금지한다. embedding_dim은 n_heads로 나누어져야 하고 T≤max_seq_length여야 한다. 모든 위치의 피처를 계산하지만 마지막 위치의 logits만 반환하며 별도 generate 메서드는 없다.

**기존 출력 기록:**

```
============================================================
Testing CharRNN
============================================================
Number of parameters: 63,218
Input shape: torch.Size([16, 20])
Output shape: torch.Size([16, 50])
Output represents scores for 50 possible next characters

Generated sequence length: 30

============================================================
Testing SimpleCharTransformer
============================================================
Number of parameters: 112,818
Input shape: torch.Size([16, 20])
Output shape: torch.Size([16, 50])

✓ Both models working correctly!
```

실행 예는 무작위 ID와 무작위 웨이트의 모양 확인이다. LSTM 생성 시 길이 10의 시드에 20개를 추가해 총 30개를 반환하지만 생성 품질을 검증하는 실험은 아니다. 출력의 “working correctly” 문구는 아래 생성 상태 재사용 문제까지 확인한 증거가 아니다.

## 2. LSTM 생성 상태의 중복 처리

현재 generate는 매 스텝 이미 생성한 **전체 접두사**를 forward에 전달하면서 직전 hidden도 재사용한다. 첫 호출에서 abc를 읽고 다음 문자가 d라면, 다음 호출은 abc 뒤에 d만 이어 읽는 대신 이전 상태에서 abcd 전체를 다시 읽는다. 이는 학습에서 사용한 접두사 조건과 다른 은닉 상태를 만든다.

올바른 두 방식은 전체 접두사를 매번 초기 상태에서 다시 계산하거나, 시드를 한 번 처리한 뒤 새 토큰만 전달하며 상태를 이어 가는 것이다. 이 둘을 혼합하면 안 된다. 코드와 출력은 보존했으며 현재 메서드를 표준 LSTM 조상 샘플러로 설명하지 않는다.

생성은 eval·no_grad를 사용하고 온도로 logits를 나눈 뒤 범주형 표본을 뽑는다. 온도는 유한한 양수여야 하며 length는 추가할 문자 수다. EOS 중단·음수 길이 검사·문맥 길이 제한은 없다. 초기 시드의 장치와 ID 범위를 모델과 맞춰야 한다.

## 3. 정규화와 계산비용

LSTM은 레이어 수가 2 이상일 때 레이어 사이 dropout=0.2를 사용한다. 마지막 레이어의 모든 출력이나 순환 상태에 동일하게 dropout을 거는 설계와 다르다. TransformerEncoderLayer는 dropout=0.1을 설정하므로 attention·후속 경로의 내장 드롭아웃을 이미 포함한다. 평가 모드에서 드롭아웃은 비활성화된다.

인과적 마스크가 있어도 Transformer의 주어진 문맥 계산은 여러 위치에서 병렬 처리할 수 있다. LSTM은 시간축 상태 의존성이 남는다. 표준 dense attention의 곱셈 비용은 O(T²E), projection과 feed-forward에는 O(TE²)도 들며 어휘 분류 비용은 별도다. 실제 메모리와 속도는 배치·헤드·커널·자료형에 영향을 받는다. 이 작은 모델에서 구조만으로 성능 우열을 단정할 수 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
CharRNN의 파라미터 수를 계산하라.

</div>

??? success "연습문제 1 풀이"

    어휘 V, 임베딩 E, 은닉 H, 레이어 수 L에서 임베딩은 VE, 첫 LSTM 레이어는 4H(E+H+2), 후속 각 레이어는 4H(2H+2), 출력은 HV+V다. PyTorch LSTM에는 입력·순환 바이어스가 각각 있어 +2가 들어간다. 실행 예 V=50,E=32,H=64,L=2이면 63,218개다. 기본 E=64,H=128,L=2에서는 231,424+193V개다. vocab_size는 반드시 지정하며 기본값이 없다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
본문의 드롭아웃 설정과 attention 확률에 적용하는 드롭아웃의 의미를 설명하라.

</div>

??? success "연습문제 2 풀이"

    TransformerEncoderLayer(dropout=0.1)은 내장 attention 드롭아웃을 이미 설정한다. 별도 레이어를 무조건 더하면 중복 적용일 수 있다. 소프트맥스 뒤 드롭아웃은 일부 웨이트를 0으로 하고 남은 값을 보정해 기대값을 유지하지만 각 실행의 행 합이 반드시 1인 것은 아니다. 특정 연결에 대한 의존을 줄이는 정규화일 수 있어도 더 균일한 attention이나 성능 개선을 보장하지 않는다. LSTM의 레이어간 드롭아웃과도 구분한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
Transformer의 문맥 길이 한계와 attention 비용을 구분하라.

</div>

??? success "연습문제 3 풀이"

    본문에는 최대 길이 100의 위치 임베딩 표가 있어 T>100이면 계산비용 이전에 인덱스 범위 문제가 생긴다. 표를 늘리면 길이 범위를 확장할 수 있지만 학습·일반화는 별도다. Dense attention은 O(T²E), 선형 projection·feed-forward는 O(TE²)의 계산이 들며 단순 attention 웨이트 저장은 헤드별 T²에 비례한다. 일부 최적화 커널은 전체 행렬 저장을 피하지만 dense 곱셈의 의존 구조 자체를 희소 attention으로 바꾸는 것과는 다르다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
레이어 수를 비교하려면 어느 인수를 사용하며 생성 버그와 어떤 관계가 있는가?

</div>

??? success "연습문제 4 풀이"

    CharRNN과 Transformer 모두 이미 n_layers 인수가 있다. 새 num_layers를 추가하지 않고 기존 인수로 깊이를 바꿀 수 있다. nn.LSTM과 nn.TransformerEncoder는 내부 모듈 파라미터를 등록하므로 아래 ModuleList 골격을 다시 붙일 필요는 없다. YourBlock은 정의되지 않은 예시다. 깊이를 바꾸어도 전체 접두사와 hidden을 중복 처리하는 generate 계약 문제는 해결되지 않는다. 구조 모양·파라미터 수와 생성 상태 관리를 따로 검토해야 한다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

두 문자 모델은 문맥당 다음 문자 하나를 예측한다. 입력 ID·위치 한계·반환 형식과 드롭아웃을 확인하고, LSTM 생성에서는 전체 접두사 재계산과 상태 캐시를 혼합하지 않아야 한다. 무작위 출력 모양 확인은 학습·생성 정확도 검증을 대체하지 않는다.
