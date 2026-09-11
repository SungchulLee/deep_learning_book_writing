"""attention_mechanisms — attention_mechanisms 페이지의 코드를 모듈로 쓸 수 있게 옮겨 놓은 것이다.

이 파일은 ch10/transformer_architecture/attention_mechanisms.md 에서 자동으로 뽑아낸 것이므로, 고칠 일이 있으면
그 페이지를 고치고 다시 뽑아내는 편이 낫다.
"""

"""
순환 신경망을 위한 주의 얼개
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================

class BahdanauAttention(nn.Module):
    """바다나우(더하기) 주의"""
    def __init__(self, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
    
    def forward(self, query, keys):
        # query: [batch, hidden] - 디코더의 숨은 상태
        # keys: [batch, seq_len, hidden] - 인코더의 출력
        scores = self.V(torch.tanh(
            self.W1(query).unsqueeze(1) + self.W2(keys)
        ))  # [batch, seq_len, 1]
        attention_weights = F.softmax(scores, dim=1)
        context = torch.sum(attention_weights * keys, dim=1)
        return context, attention_weights

class LuongAttention(nn.Module):
    """루옹(곱하기) 주의"""
    def __init__(self, hidden_size, method='dot'):
        super().__init__()
        self.method = method
        if method == 'general':
            self.W = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, query, keys):
        if self.method == 'dot':
            scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2))
        elif self.method == 'general':
            scores = torch.bmm(self.W(query).unsqueeze(1), keys.transpose(1, 2))
        
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attention_weights, keys).squeeze(1)
        return context, attention_weights


# ---------------------------------------------------------------------------
# 가린 소프트맥스 도구
# ---------------------------------------------------------------------------
# 많은 수열 과제에서 입력의 길이가 제각각이다. 주의 점수를 셈할 때
# 실제 수열 길이를 넘는 자리는 -inf로 가려서 소프트맥스가
# 그 자리에 확률 0을 주도록 해야 한다.
# 그러면 채움 토큰에 주의하지 않게 되며, 트랜스포머의 인코더 자기 주의와
# 교차 주의 모두에 꼭 필요하다.

def masked_softmax(X, valid_lens):
    """유효 길이를 넘는 자리를 가리고 소프트맥스를 한다.

    인수:
        X: 꼴이 (batch_size, num_queries, num_keys)인 3차원 텐서
        valid_lens: 1차원 텐서 (batch_size,) 또는 2차원 텐서 (batch_size, num_queries)
            성분마다 그 질의에 유효한 열쇠가 몇 개인지 알려 준다.
    반환값:
        X와 같은 꼴의 소프트맥스 출력이며 가린 자리는 0이다.
    """
    if valid_lens is None:
        return F.softmax(X, dim=-1)

    shape = X.shape
    if valid_lens.dim() == 1:
        # 배치 원소마다 모든 질의에 같은 유효 길이를 쓴다
        valid_lens = valid_lens.repeat_interleave(shape[1])
    else:
        valid_lens = valid_lens.reshape(-1)

    # 가림을 만든다: valid_len 이상인 자리에 -1e6을 주어 exp(-1e6) ≈ 0이 되게 한다
    X_flat = X.reshape(-1, shape[-1])
    maxlen = X_flat.size(1)
    mask = torch.arange(maxlen, device=X.device)[None, :] < valid_lens[:, None]
    X_flat[~mask] = -1e6

    return F.softmax(X_flat.reshape(shape), dim=-1)


if __name__ == "__main__":
    pass
