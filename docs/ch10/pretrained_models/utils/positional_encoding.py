"""자리 매김(positional encoding).

트랜스포머의 주의 셈은 차례를 모른다. 낱말을 섞어 넣어도 같은 값이 나오므로,
어디에 있는 낱말인지를 따로 알려 주어야 한다. 그 몫을 하는 것이 자리 매김이다.

원 논문의 방식은 자리마다 서로 다른 진동수의 사인과 코사인을 섞어 쓰는 것이다.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

이렇게 하면 학습할 매개변수가 늘지 않고, 익힐 때 본 적 없는 긴 차례에도
값이 정의된다.
"""

import math

import torch
import torch.nn as nn

__all__ = ["PositionalEncoding"]


class PositionalEncoding(nn.Module):
    """사인과 코사인으로 만든 자리 매김을 입력에 더한다.

    인수:
        d_model: 묻힘 차원
        max_len: 미리 만들어 둘 가장 긴 차례
        dropout: 더한 뒤 걸 드롭아웃 비율
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 10000^(2i/d) 를 로그 공간에서 셈한다. 그대로 거듭제곱하면
        # d가 클 때 값이 넘쳐 흐른다.
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        # d_model이 홀수이면 코사인 쪽이 한 칸 적다
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])

        # 학습하지 않지만 state_dict에는 들어가야 하므로 buffer로 둔다
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """x: (묶음, 차례 길이, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
