# 자기회귀 Transformer의 구조와 생성

자기회귀 Transformer는 인과적 self-attention으로 앞선 문맥에서 다음 토큰의 분포를 예측한다. 관측 시퀀스의 여러 위치를 병렬로 학습할 수 있지만 조상 샘플링의 다음 토큰 결정은 순차적이다. [원 Transformer 논문](https://arxiv.org/abs/1706.03762)의 attention을 바탕으로, 이 절에서는 decoder-only 형태의 학습용 GPT 구현과 여러 확장 골격을 구분해 살펴본다.

## 1. 인과적 self-attention

입력 X에서 Q=XW_Q,K=XW_K,V=XW_V를 만들고 헤드별로 다음을 계산한다.

$$A=\operatorname{softmax}\left(QK^T/\sqrt{d_k}+M\right),\qquad Y=AV,$$

$$M_{ij}=\begin{cases}0&j\le i\\-\infty&j>i.\end{cases}$$

위치 i의 입력 토큰까지 볼 수 있지만 그 출력을 다음 토큰 i+1의 타깃과 맞추어야 한다. attention만 마스킹하고 다른 경로에서 미래가 새면 자기회귀 계약이 성립하지 않는다. “인과적”이라는 구조 명칭은 통계적 예측에서 미래를 차단한다는 뜻이며 현실의 개입 인과성을 증명하는 것은 아니다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    """
    자기 되돌이 모델을 위한 인과(가린) 셀프 어텐션.
    
    자리마다 제 자신과 앞선 자리에만 주의할 수 있다.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Q, K, V의 선형 사영
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        # 출력 사영
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # 인과 마스크 만들기(하삼각)
        # 버퍼로 등록해 모델과 함께 갈무리되되 익혀지지는 않게 한다
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer('causal_mask', mask)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        인과 가림을 쓰는 앞먹임.
        
        인수:
            x: 입력 텐서 [batch_size, seq_len, d_model]
            attention_mask: 덧붙일 수 있는 가림막 [배치 크기, 차례 길이]
            
        반환값:
            출력 텐서 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Q, K, V를 셈한다
        Q = self.W_q(x)  # [배치, 차례 길이, d_model]
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 다중 머리 주의에 맞게 꼴을 바꾼다
        # [배치, 차례 길이, 머리 수, d_k] -> [배치, 머리 수, 차례 길이, d_k]
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 주의 점수를 셈한다
        # [배치, 머리 수, 차례 길이, 차례 길이]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 인과 가림을 적용한다
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # 있으면 어텐션 가림막을 쓴다(예컨대 채우기)
        if attention_mask is not None:
            # attention_mask: [배치, 차례 길이] -> [배치, 1, 1, 차례 길이]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # 소프트맥스와 드롭아웃
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 값에 어텐션 적용
        # [배치, 머리 수, 차례 길이, d_k]
        context = torch.matmul(attention_weights, V)
        
        # 다시 꼴 되돌리기
        # [배치, 차례 길이, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 출력 사영
        output = self.W_o(context)
        
        return output
```

입력 `(B,T,d_model)`에서 Q·K·V는 `(B,H,T,d_k)`, 점수는 `(B,H,T,T)`다. d_model은 양의 n_heads로 나누어져야 하며 T는 미리 만든 causal_mask 범위 이하여야 한다. attention_mask는 `(B,T)`의 0인 **키 위치**를 차단한다. 쿼리 위치의 손실을 제외하거나 출력 자체를 0으로 만들지는 않는다.

어떤 쿼리의 모든 키가 가려지면 −∞만 있는 행에 softmax를 적용해 NaN이 생길 수 있다. 특히 왼쪽 패딩·빈 문맥을 고려한 정책이 필요하다. 드롭아웃은 attention 확률 뒤에 적용되므로 학습 중 각 행 합이 반드시 1인 것은 아니며 eval에서 비활성화된다.

## 2. Pre-LN 블록과 잔차 경로

```python
class TransformerBlock(nn.Module):
    """
    변환기 디코더 덩이 하나.
    
    짜임:
        x -> 층 고르게 맞추기 -> 인과 어텐션 -> + -> 층 고르게 맞추기 -> 앞먹임 신경망 -> +
             |__________________________|      |___________________|
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int = None,
        dropout: float = 0.1,
        max_seq_len: int = 2048
    ):
        super().__init__()
        
        d_ff = d_ff or 4 * d_model  # 여느 방식: 앞먹임 신경망은 모델 차원의 4배
        
        # 층 고르게 맞추기(앞 층 고르게 맞추기 구조)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # 인과 셀프 어텐션
        self.attention = CausalSelfAttention(
            d_model, n_heads, max_seq_len, dropout
        )
        
        # 순방향 신경망
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        변환기 덩이를 지나는 앞먹임.
        
        인수:
            x: 들임 [배치 크기, 차례 길이, d_model]
            attention_mask: 쓸 수도 있는 가림막 [배치 크기, 차례 길이]
            
        반환값:
            내놓기 [배치 크기, 차례 길이, d_model]
        """
        # 잔차를 곁들인 자기 주의
        x = x + self.attention(self.ln1(x), attention_mask)
        
        # 잔차를 곁들인 순전파
        x = x + self.ffn(self.ln2(x))
        
        return x
```

블록은 x←x+Attention(LN(x)), 이어 x←x+FFN(LN(x))를 계산한다. 첫 업데이트 결과가 두 번째 잔차 경로의 기준이다. 토큰별 LayerNorm은 시간 위치 사이 통계를 섞지 않아 이 코드의 인과성을 유지한다. 잔차·정규화는 최적화에 유용하지만 깊이·학습률에 무관한 안정성 보장은 아니다.

아래 기존 도식은 post-LN·pre-LN의 배치 차이를 개념적으로 보여 준다. 화살표가 생략된 부분은 위 코드와 두 업데이트 식을 기준으로 읽는다.

```
x -> Attention -> + -> LayerNorm -> FFN -> + -> LayerNorm
     |____________|                 |______|
```

```
x -> LayerNorm -> Attention -> + -> LayerNorm -> FFN -> +
                               |                        |
                               x ---------------------->|
```

## 3. GPT 모듈과 손실

```python
class GPT(nn.Module):
    """
    GPT 꼴 자기 되돌이 변환기.
    
    구조:
        - 토큰 임베딩 + 자리 임베딩
        - 변환기 디코더 덩이 쌓기
        - 낱말로의 내놓기 쏘기
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: int = None,
        max_seq_len: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # 토큰 임베딩과 자리 임베딩
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # 트랜스포머 블록
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, max_seq_len)
            for _ in range(n_layers)
        ])
        
        # 마지막 층 정규화
        self.ln_f = nn.LayerNorm(d_model)
        
        # 내놓기 쏘기(흔히 토큰 임베딩과 묶는다)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 무게 묶기
        self.lm_head.weight = self.token_embedding.weight
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """GPT-2을 따라 무게를 첫자리매김한다."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        앞먹임.
        
        인수:
            input_ids: 토큰 어깨수 [배치 크기, 차례 길이]
            attention_mask: 쓸 수도 있는 가림막 [배치 크기, 차례 길이]
            
        반환값:
            로짓 [배치 크기, 차례 길이, 낱말 수]
        """
        batch_size, seq_len = input_ids.shape
        
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"
        
        # 묻힘을 얻는다
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        
        x = self.dropout(tok_emb + pos_emb)
        
        # 변환기 덩이를 지난다
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # 마지막 층 고르게 맞추기와 쏘기
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        다음 토큰 헤아리기의 어긋 엔트로피 손실을 셈한다.
        """
        # 다음 토큰 맞히기를 위해 민다
        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]
        
        # 로짓을 얻는다
        logits = self.forward(inputs, attention_mask[:, :-1] if attention_mask is not None else None)
        
        # 교차 엔트로피 손실
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100  # 채움을 무시한다
        )
        
        return loss
```

정수 input_ids `(B,T)`를 토큰·학습 위치 임베딩으로 바꾸고 블록을 거쳐 logits `(B,T,V)`를 반환한다. 최종 출력 가중치는 토큰 임베딩과 공유한다. 이 모델은 무작위 초기화되며 사전학습 가중치·토크나이저를 자동으로 불러오지 않는다.

compute_loss는 입력·타깃을 한 칸 이동해 모든 위치 평균 교차엔트로피를 계산한다. 첫 토큰에 조건부인 목적이며 BOS·EOS·패딩 정책은 호출부에서 정의해야 한다. attention_mask는 attention의 키만 가릴 뿐 패딩 타깃을 손실에서 제외하지 않는다. ignore_index=−100을 지정해도 targets를 실제로 −100으로 바꾸는 처리는 없다. input_ids 자체에 −100을 넣으면 입력 임베딩 범위를 벗어날 수도 있어 입력과 별도 labels의 계약이 필요하다. 유효 타깃이 없거나 T<2인 경우도 처리해야 한다.

```python
# GPT-2 짜임새
GPT2_CONFIGS = {
    'gpt2-small': {
        'vocab_size': 50257,
        'd_model': 768,
        'n_layers': 12,
        'n_heads': 12,
        'max_seq_len': 1024
    },
    'gpt2-medium': {
        'vocab_size': 50257,
        'd_model': 1024,
        'n_layers': 24,
        'n_heads': 16,
        'max_seq_len': 1024
    },
    'gpt2-large': {
        'vocab_size': 50257,
        'd_model': 1280,
        'n_layers': 36,
        'n_heads': 20,
        'max_seq_len': 1024
    },
    'gpt2-xl': {
        'vocab_size': 50257,
        'd_model': 1600,
        'n_layers': 48,
        'n_heads': 25,
        'max_seq_len': 1024
    }
}

def create_gpt2(config_name: str) -> GPT:
    """짜임새 이름으로 GPT-2 모델을 만든다."""
    config = GPT2_CONFIGS[config_name]
    return GPT(**config)
```

설정 사전은 GPT-2 계열의 폭·층·헤드·문맥 크기를 참고한 교육용 구조다. 편향·활성화·초기화·토크나이저·체크포인트 계약까지 원 모델과 완전히 같다고 볼 수 없고 create_gpt2는 학습된 GPT-2를 다운로드하는 함수가 아니다. 설정 이름만으로 원 논문 성능을 재현했다고 주장하지 않는다.

## 4. 위치 표현의 선택

학습 위치 임베딩은 인덱스별 학습 벡터를 더한다.

```python
self.position_embedding = nn.Embedding(max_seq_len, d_model)
```

이 구현의 위치 표는 max_seq_len 범위로 고정되어 있다. 표를 단순 확장하는 것과 학습보다 긴 문맥에 잘 일반화하는 것은 다른 문제다. 생성 중 창을 잘라 쓰면 위치를 어떻게 다시 부여하는지도 모델 정책의 일부다.

사인·코사인 위치 표현은 다음과 같다.

$$PE_{pos,2i}=\sin(pos/10000^{2i/d}),\qquad
PE_{pos,2i+1}=\cos(pos/10000^{2i/d}).$$

```python
class SinusoidalPositionalEncoding(nn.Module):
    """붙박인 사인 꼴 자리 부호화."""
    
    def __init__(self, d_model: int, max_seq_len: int = 5000):
        super().__init__()
        
        # 위치 인코딩 행렬을 만든다
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_seq_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """들임에 자리 부호 더하기."""
        return x + self.pe[:, :x.size(1)]
```

수식은 더 긴 위치에도 정의할 수 있지만 이 코드의 버퍼 길이는 유한하다. 또한 현재 cos 대입은 d_model이 홀수이면 길이가 맞지 않으므로 짝수 폭을 전제로 한다. 계산 가능한 위치 표현이 곧 학습 범위 밖 예측의 성공 보장은 아니다.

[RoPE](https://arxiv.org/abs/2104.09864)는 Q·K의 좌표쌍을 위치별 회전으로 변환한다. 적절히 짝지은 회전의 내적이 상대 위치 차이를 반영한다.

```python
class RotaryPositionalEmbedding(nn.Module):
    """
    회전 위치 임베딩(RoPE).
    
    물음과 열쇠 벡터를 돌려 자리를 담는다.
    길이를 더 잘 늘려 헤아릴 수 있게 한다.
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # 돌림 잦기를 셈한다
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # 코사인과 사인을 미리 셈한다
        self._precompute_cache(max_seq_len)
    
    def _precompute_cache(self, seq_len: int):
        """코사인과 사인 값을 미리 셈한다."""
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # [차례 길이, d_model/2]
        
        # 사인과 코사인 둘 다에 쓰려 겹친다
        emb = torch.cat([freqs, freqs], dim=-1)  # [차례 길이, d_model]
        
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int
    ) -> tuple:
        """
        물음과 열쇠에 돌림 임베딩을 쓴다.
        
        인수:
            q: 물음 텐서 [배치, 머리 수, 차례 길이, d_k]
            k: 열쇠 텐서 [배치, 머리 수, 차례 길이, d_k]
            seq_len: 수열 길이
            
        반환값:
            (돌린 q, 돌린 k)
        """
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        
        q_rot = self._apply_rotation(q, cos, sin)
        k_rot = self._apply_rotation(k, cos, sin)
        
        return q_rot, k_rot
    
    def _apply_rotation(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """텐서에 돌림을 적용한다."""
        # 두 반으로 가른다
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        
        # 회전
        rotated = torch.cat([-x2, x1], dim=-1)
        
        return x * cos + rotated * sin
```

이 예시는 벡터의 앞·뒤 절반을 짝짓는 형태다. 회전할 마지막 차원은 짝수여야 하며 보통 헤드별 Q·K에 적용할 때는 그 **헤드 차원**과 캐시의 폭을 맞춘다. d_model이라는 인수 이름만 보고 전체 모델 폭을 그대로 넘기면 모양이 맞지 않을 수 있다. 캐시 범위를 자동으로 늘리거나 과거 길이의 위치 오프셋을 처리하지 않으므로 증분 생성에 그대로 연결하기에는 추가 설계가 필요하다. 세 위치 표현은 기본 GPT에 자동 교체·통합되어 있지 않다.

## 5. 캐시 없는 생성

```python
@torch.no_grad()
def generate(
    model: GPT,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    자기 회귀로 글을 짓는다.
    
    인수:
        model: 익힌 GPT 모델
        prompt_ids: 출발 토큰 어깨수 [1, prompt_len]
        max_new_tokens: 만들 토큰의 최대 개수
        temperature: 표집 온도
        top_k: 상위 k 표집 매개변수
        top_p: 웃 p(핵) 뽑기 매개변수
        device: 만들어 낼 기기
        
    반환값:
        만든 토큰 어깨수 [1, prompt_len + max_new_tokens]
    """
    model.eval()
    model = model.to(device)
    
    generated = prompt_ids.clone().to(device)
    
    for _ in range(max_new_tokens):
        # 최대 차례 길이로 자른다
        context = generated[:, -model.max_seq_len:]
        
        # 다음 토큰의 로짓을 얻는다
        logits = model(context)
        next_token_logits = logits[:, -1, :]  # [1, 낱말 수]
        
        # 온도를 적용한다
        next_token_logits = next_token_logits / temperature
        
        # 상위 k 거르기를 적용한다
        if top_k is not None:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = float('-inf')
        
        # 상위 p(핵) 거르기를 적용한다
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # 쌓인 확률이 문턱값을 넘는 토막 없애기
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            next_token_logits[indices_to_remove] = float('-inf')
        
        # 분포에서 뽑는다
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # 만들어진 수열에 덧붙인다
        generated = torch.cat([generated, next_token], dim=1)
        
        # 차례 끝 토큰이 있는지 살핀다(뜻매김되어 있으면)
        # if next_token.item() == eos_token_id:
        #     break
    
    return generated
```

이 함수는 `(B,문맥 길이)` 프롬프트에서 지정 개수의 토큰을 추가한다. 기본 device는 cuda로 고정되어 있어 CUDA가 없는 환경에는 명시적 조정이 필요하다. 프롬프트는 비어 있지 않아야 하고 온도는 유한한 양수, top_k는 1~V, top_p는 보통 0~1 사이의 양수로 정한다. EOS 중단이나 패딩 프롬프트 마스크는 없다.

매 스텝 마지막 max_seq_len 토큰만 다시 계산하고 위치도 0부터 부여한다. 따라서 최대 길이 밖에서는 전체 과거 조건부가 아닌 잘린 문맥 정책을 사용한다. 원래 GPT의 compute_loss와 별도로 이 정책을 기록해야 한다. top-k의 동점 경계는 k개보다 많은 토큰을 남길 수 있고, top-p를 함께 쓰면 top-k 이후 재정규화 분포에 적용된다. 온도·필터가 있으면 기본 조건부에서의 표본과 달라진다.

## 6. KV cache의 계약

이전 토큰의 K·V를 저장하면 새 토큰 처리에서 중복 projection·블록 계산을 줄일 수 있다. 하지만 새 Q는 여전히 저장한 키에 attention하므로 전체 생성 비용이 토큰 길이에 무관해지는 것은 아니다.

```python
class CausalSelfAttentionWithCache(nn.Module):
    """효율 좋은 만들어 내기를 위한 열쇠-값 저장턱을 갖춘 셀프 어텐션."""
    
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        # 만들어 내기를 위한 저장턱
        self.cache_k = None
        self.cache_v = None
    
    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        past_cache: tuple = None
    ) -> tuple:
        """
        저장턱을 쓸 수도 있는 앞먹임.
        
        인수:
            x: 입력 [batch, seq_len, d_model]
            use_cache: 저장턱을 쓰거나 고칠지 여부
            past_cache: 앞선 (K, V) 저장턱
            
        반환값:
            (내놓기, 저장턱). use_cache이면 저장턱은 (K, V), 아니면 None
        """
        batch_size, seq_len, _ = x.shape
        
        # 새 토큰의 Q, K, V을 셈한다
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 쓰고 있으면 저장턱에 덧붙인다
        if past_cache is not None:
            past_k, past_v = past_cache
            K = torch.cat([past_k, K], dim=2)
            V = torch.cat([past_v, V], dim=2)
        
        # 다음 되풀이를 위해 저장턱을 담아 둔다
        cache = (K, V) if use_cache else None
        
        # 어텐션 계산
        total_len = K.size(2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 새 자리가 옛 자리를 볼 때만 인과 가림막을 쓴다
        # 지난 자리는 모두 쓸 수 있다
        if seq_len > 1:
            mask = torch.triu(torch.ones(seq_len, total_len, device=x.device), diagonal=total_len - seq_len + 1).bool()
            scores = scores.masked_fill(mask, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)
        
        return output, cache

@torch.no_grad()
def generate_with_cache(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    열쇠-값 저장턱을 쓴 효율 좋은 만들어 내기.
    
    맥락에는 저장턱의 K/V을 쓰고 새 토큰의 어텐션만 셈한다.
    """
    model.eval()
    
    # 채근 글에 대한 첫 앞먹임
    generated = prompt_ids.clone()
    past_cache = None
    
    for _ in range(max_new_tokens):
        # 새 토큰만 다룬다
        if past_cache is None:
            input_ids = generated
        else:
            input_ids = generated[:, -1:]
        
        # 저장턱과 함께 앞먹임한다
        logits, past_cache = model.forward_with_cache(input_ids, past_cache)
        
        # 다음 토큰을 뽑는다
        next_logits = logits[:, -1, :] / temperature
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        
        generated = torch.cat([generated, next_token], dim=1)
    
    return generated
```

캐시 attention 예시는 전달받은 past_cache에 새 K·V를 이어 붙인다. 여러 새 토큰의 마스크는 과거 길이만큼 오프셋을 두고 미래 새 토큰을 차단한다. 하나의 새 토큰이면 현재까지의 모든 키를 볼 수 있다. 캐시의 배치·헤드·차원·장치·자료형과 모델 상태가 일치해야 한다.

그러나 위 GPT에는 forward_with_cache가 없어 generate_with_cache가 그대로 동작하지 않는다. 실제 통합에는 층별 캐시 반환, 새 토큰만 처리하는 경로와 위치 오프셋, 문맥 한계·패딩·캐시 초기화 정책이 필요하다. 선언된 cache_k·cache_v 속성을 자동 사용하는 구현도 아니다. 전체 접두사와 과거 캐시를 동시에 다시 전달해 중복 처리하지 않아야 한다.

## 7. Attention과 FFN의 변형

[MQA](https://arxiv.org/abs/1911.02150)는 쿼리 헤드는 유지하고 K·V 헤드 하나를 공유해 KV 저장량·메모리 전송을 줄이는 접근이다.

```python
class MultiQueryAttention(nn.Module):
    """
    여러 물음 어텐션: 모든 머리가 K, V 하나를 나누어 쓴다.
    
    열쇠-값 저장턱 크기를 n_heads 갑절만큼 줄인다.
    """
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)  # 물음 머리 여럿
        self.W_k = nn.Linear(d_model, self.d_k, bias=False)  # K 하나
        self.W_v = nn.Linear(d_model, self.d_k, bias=False)  # V 하나
        self.W_o = nn.Linear(d_model, d_model, bias=False)
```

[GQA](https://arxiv.org/abs/2305.13245)는 여러 쿼리 헤드 그룹이 더 적은 K·V 헤드를 공유한다.

```python
class GroupedQueryAttention(nn.Module):
    """
    무리 물음 어텐션: 무리 안에서 K, V을 나누어 쓴다.
    
    n_kv_heads < n_heads이면 기억과 셈을 아낀다.
    """
    
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
```

위 두 클래스는 projection 생성자만 있고 forward가 없다. 헤드 매핑·마스크·캐시 연산을 구현하지 않아 완성된 교체 모듈이 아니다. 유효한 헤드 수와 d_model의 나눗셈 조건도 필요하다. 같은 길이·자료형에서 KV 캐시 크기는 KV 헤드 수에 비례하지만 Q·attention·출력 계산이 모두 같은 비율로 줄어드는 것은 아니다. 품질·속도 효과는 실험으로 확인한다.

[SwiGLU](https://arxiv.org/abs/2002.05202)는 두 입력 projection의 곱에 출력 projection을 적용하는 FFN 변형이다.

```python
class SwiGLU(nn.Module):
    """
    SwiGLU 깨움: 스위시 문 달린 선형 낱개.
    
    SwiGLU(x) = Swish(xW1) * (xW2)
    여기서 Swish(x) = x * 시그모이드(x)
    """
    
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff, bias=False)
        self.W2 = nn.Linear(d_model, d_ff, bias=False)
        self.W3 = nn.Linear(d_ff, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W3(F.silu(self.W1(x)) * self.W2(x))
```

편향 없는 이 구현의 가중치는 3d×d_ff개로 같은 은닉 폭의 일반 두 projection FFN 2d×d_ff보다 많다. 비교에서는 폭·파라미터·계산 예산을 맞춰야 한다. 기본 GPT의 GELU FFN에 자동 적용되어 있지 않다.

[RMSNorm](https://arxiv.org/abs/1910.07467)은 평균을 빼는 대신 제곱평균제곱근으로 크기를 정규화한다.

```python
class RMSNorm(nn.Module):
    """
    제곱평균제곱근 층 정규화.
    
    층 고르게 맞추기보다 단순하고 빠르다(평균을 빼지 않는다).
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

LayerNorm과 같은 연산은 아니며 이 구현은 학습 스케일만 갖고 편향은 없다. x.pow(2)의 정밀도와 극단값도 고려해야 한다. 계산 일부를 줄이는 의도와 실제 하드웨어 속도·학습 성능은 구분한다.

## 8. 학습률과 활성값 체크포인팅

```python
def get_lr_schedule(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float
) -> float:
    """
    몸 풀기를 갖춘 코사인 배움 빠르기 차례표.
    """
    if step < warmup_steps:
        # 선형 워밍업
        return max_lr * step / warmup_steps
    elif step > max_steps:
        return min_lr
    else:
        # 코사인 감쇠
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

학습률 함수는 선형 warm-up 뒤 cosine 감쇠를 사용한다. 0<warmup_steps<max_steps, 유한한 비음수 학습률과 유효한 step을 전제로 해야 분모·일정이 정의된다. step=0의 값은 0이고 max_steps 이후에는 min_lr다. 함수 정의만으로 optimizer에 자동 연결되지는 않는다.

```python
from torch.utils.checkpoint import checkpoint

class GPTWithCheckpointing(GPT):
    """기억을 줄이려 기울기 되짚을 자리를 둔 GPT."""
    
    def forward(self, input_ids, attention_mask=None):
        # ... 임베딩 ...
        
        for block in self.blocks:
            # 기억을 아끼려 덩이마다 되짚을 자리를 둔다
            x = checkpoint(block, x, attention_mask)
        
        # ... 내놓기 쏘기 ...
```

활성값 체크포인팅은 역전파 때 중간 계산을 다시 수행해 메모리와 계산을 교환한다. 가중치 파일 저장과 다른 개념이다. 위 GPTWithCheckpointing.forward는 x가 정의되지 않고 임베딩·최종 정규화·출력·return도 없어 실행 가능한 완성 예제가 아니다. 실제 적용에는 원래 forward 경로를 유지하고 [PyTorch checkpoint 문서](https://docs.pytorch.org/docs/stable/checkpoint.html)의 use_reentrant·난수 상태·재계산 계약을 맞춰야 한다.

## 9. 도메인 확장 골격의 범위

```python
class FinancialGPT(GPT):
    """
    돈살림 글 만들어 내기에 미세 조정한 GPT.
    
    쓰임새:
    - 보고서 간추리기
    - 되짚어 시험하기를 위한 소식 만들어 내기
    - 시나리오 이야기 만들어 내기
    """
    
    def __init__(self, financial_vocab_size: int, **kwargs):
        super().__init__(vocab_size=financial_vocab_size, **kwargs)
        
        # 덧붙인 돈살림 낱개 임베딩
        self.entity_embedding = nn.Embedding(1000, kwargs.get('d_model', 768))
```

FinancialGPT는 entity_embedding을 추가하지만 상속한 forward에서 사용하지 않는다. 따라서 선언만으로 기업·종목 정보에 조건부인 모델이 되거나 도메인 지식을 학습한 것은 아니다. 해당 ID·입력 경로와 학습 목적을 연결해야 한다.

```python
class TimeSeriesTransformer(nn.Module):
    """
    돈살림 시계열 내다보기를 위한 변환기.
    
    값의 지난 일을 이어진 값의 "문장"으로 본다.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        forecast_horizon: int = 10
    ):
        super().__init__()
        
        # 이어진 값을 d_model으로 쏜다
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 트랜스포머 블록
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads)
            for _ in range(n_layers)
        ])
        
        # 내놓기: 분포 매개변수를 헤아린다
        self.output_projection = nn.Linear(d_model, input_dim * 2)  # 평균, 표준 편차
        self.forecast_horizon = forecast_horizon
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        인수:
            x: 지난 값 [배치, 차례 길이, 들임 차원]
            
        반환값:
            내다보기 분포의 (평균, 표준 편차)
        """
        h = self.input_projection(x)
        
        for block in self.blocks:
            h = block(h)
        
        params = self.output_projection(h[:, -1, :])
        mean, log_std = params.chunk(2, dim=-1)
        
        return mean, F.softplus(log_std)
```

TimeSeriesTransformer는 마지막 특징에서 다음 벡터의 평균과 양의 스케일 후보를 출력한다. 두 번째 projection의 이름은 log_std지만 반환은 exp가 아닌 softplus를 적용한 값이다. 이를 표준편차로 해석하려면 확률분포·손실을 별도로 정의하고 수치적으로 양수 하한도 검토해야 한다. forecast_horizon은 저장만 되어 실제 여러 스텝 예측을 하지 않으며 위치 표현·시계열 분할·타깃 정렬·롤아웃도 별도다. 시간상 사용할 수 없는 미래 정보가 전처리나 입력에 새지 않도록 해야 한다.

## 연습문제

1. 같은 실제 접두사에서 미래 입력만 바꿀 때 과거 위치 logits가 유지되어야 하는 이유를 설명하라. 마스크뿐 아니라 dropout·패딩·정규화의 평가 조건도 정하라.
2. 학습 위치·사인 위치·RoPE를 같은 작은 작업에 적용하는 비교를 설계하라. 헤드 차원·캐시 위치 오프셋·학습 길이와 평가 길이를 구분하고 실행 결과를 가정하지 마라.
3. GPT에 층별 KV cache를 통합하는 입력·출력 계약을 설계하라. eval에서 같은 문맥의 전체 재계산과 증분 logits가 일치하려면 무엇이 필요한가? 캐시 저장량과 attention 계산량을 따로 계산하라.
4. 별도 도메인 텍스트로 모델을 학습·평가할 때 토큰화·자료 분할·중복·시간 정보 누출과 체크포인트 설정을 어떻게 통제할지 설명하라. 선언만 된 entity_embedding이 왜 충분하지 않은지도 설명하라.
5. MHA·MQA·GQA 또는 GELU·SwiGLU를 비교할 때 동일 폭,동일 파라미터 수,동일 학습 예산이 각각 어떤 다른 질문에 답하는지 설명하라. 제공 골격에서 구현되지 않은 경로도 식별하라.

## 정리하며

자기회귀 Transformer에서는 한 칸 이동한 타깃과 인과적 attention을 맞추고 패딩 손실을 별도로 처리해야 한다. 생성·KV cache·위치 오프셋의 계약을 함께 설계한다. 이 페이지의 일부 확장은 생성자나 개념적 골격에 그치므로 기본 GPT에 통합된 실행 경로와 구분해 이해한다.

**참고 문헌**

1. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.
2. Radford, A., et al. (2018). Improving Language Understanding by Generative Pre-Training. *OpenAI*.
3. Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners. *OpenAI*.
4. Brown, T., et al. (2020). Language Models are Few-Shot Learners. *NeurIPS*.
5. Touvron, H., et al. (2023). LLaMA: Open and Efficient Foundation Language Models. *arXiv*.
6. Su, J., et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. *arXiv*.

---
