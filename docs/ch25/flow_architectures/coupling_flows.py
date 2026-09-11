"""짝 짓는 흐름(coupling flow)과 그 사이에 끼우는 묶음 고르개.

25.2 흐름 얼개 절의 MNIST 예제가 쓰는 두 층이다. 둘 다 flow_utils.Flow를
물려받으므로 FlowSequence에 그대로 끼워 넣을 수 있고, 앞뒤 바꿈과 함께
야코비 행렬식의 로그값을 돌려준다.

**짝 짓기의 요령**

입력을 두 쪽으로 가른 뒤 한쪽은 그대로 두고, 그 그대로 둔 쪽만 보고
나머지 쪽을 늘이고 옮긴다.

    x_a 그대로
    x_b <- x_b * exp(s(x_a)) + t(x_a)

s와 t가 아무리 복잡해도 야코비 행렬이 삼각행렬이 되어 행렬식이
exp(s(x_a))의 곱으로 간단히 떨어진다. 되돌리기도 나눗셈 한 번이면 된다.
이것이 흐름 모델이 복잡한 바꿈을 쓰면서도 가능도를 정확히 셈할 수 있는 까닭이다.
"""

import torch
import torch.nn as nn

from flow_utils import Flow

__all__ = ["CouplingLayer", "BatchNorm"]


class CouplingLayer(Flow):
    """RealNVP 방식의 아핀 짝 짓기 층.

    인수:
        dim: 자료 차원
        hidden_dim: s와 t를 셈하는 그물의 숨은 너비
        mask: 0/1 텐서, 꼴 (dim,). 1인 자리는 그대로 지나가고
              0인 자리가 늘어나고 옮겨진다. 층마다 뒤집어 쓰면
              모든 차원이 번갈아 바뀐다.
    """

    def __init__(self, dim, hidden_dim=256, mask=None):
        super().__init__()
        if mask is None:
            # 기본: 앞 절반은 그대로, 뒤 절반이 바뀐다
            mask = torch.zeros(dim)
            mask[: dim // 2] = 1
        self.register_buffer("mask", mask.float())

        def net():
            return nn.Sequential(
                nn.Linear(dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, dim),
            )

        self.scale_net = net()
        self.shift_net = net()
        # 처음에는 항등 바꿈이 되게 두면 익히기가 훨씬 안정된다
        nn.init.zeros_(self.scale_net[-1].weight)
        nn.init.zeros_(self.scale_net[-1].bias)
        nn.init.zeros_(self.shift_net[-1].weight)
        nn.init.zeros_(self.shift_net[-1].bias)
        # scale이 마구 커지지 않도록 tanh로 묶는다
        self.scale_cap = nn.Parameter(torch.zeros(1))

    def _s_t(self, x_masked):
        s = torch.tanh(self.scale_net(x_masked)) * self.scale_cap.exp()
        t = self.shift_net(x_masked)
        return s, t

    def forward(self, z):
        """숨은 공간 → 자료 공간."""
        z_keep = z * self.mask
        s, t = self._s_t(z_keep)
        s, t = s * (1 - self.mask), t * (1 - self.mask)
        x = z_keep + (1 - self.mask) * (z * torch.exp(s) + t)
        return x, s.sum(dim=1)

    def inverse(self, x):
        """자료 공간 → 숨은 공간."""
        x_keep = x * self.mask
        s, t = self._s_t(x_keep)
        s, t = s * (1 - self.mask), t * (1 - self.mask)
        z = x_keep + (1 - self.mask) * ((x - t) * torch.exp(-s))
        return z, -s.sum(dim=1)


class BatchNorm(Flow):
    """흐름 사이에 끼우는 묶음 고르개.

    보통의 BatchNorm과 하는 일은 같지만, 흐름에 쓰려면 야코비 행렬식의
    로그값을 함께 돌려주어야 한다. 늘이는 값이 1/sqrt(var + eps)이므로
    로그 행렬식은 -0.5 * sum(log(var + eps)) 이다.

    짝 짓기 층을 여러 겹 쌓으면 값의 크기가 층마다 흔들리는데, 그 사이에
    이 층을 끼우면 익히기가 훨씬 안정된다.
    """

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.log_gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))

    def _stats(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False) + self.eps
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * var)
        else:
            mean, var = self.running_mean, self.running_var
        return mean, var

    def forward(self, z):
        """숨은 공간 → 자료 공간: 고르게 한 것을 되돌린다."""
        mean, var = self._stats(z)
        x = (z - self.beta) / self.log_gamma.exp() * var.sqrt() + mean
        log_det = (0.5 * var.log() - self.log_gamma).sum()
        return x, log_det.expand(z.shape[0])

    def inverse(self, x):
        """자료 공간 → 숨은 공간: 고르게 한다."""
        mean, var = self._stats(x)
        z = (x - mean) / var.sqrt() * self.log_gamma.exp() + self.beta
        log_det = (self.log_gamma - 0.5 * var.log()).sum()
        return z, log_det.expand(x.shape[0])
