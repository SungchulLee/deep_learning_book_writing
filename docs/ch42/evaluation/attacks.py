"""attacks — 겨루는 보기를 만드는 두 가지 방법.

이 절은 방어가 정말 든든한지를 따진다. 그러려면 먼저 **제대로 된 공격**이
있어야 한다. 약한 공격에 버티는 것은 아무것도 증명하지 않는다.

둘 다 $\\ell_\\infty$ 공으로 흔듦을 가둔다. 곧 화소 하나가 움직일 수 있는
폭이 epsilon을 넘지 않는다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FGSM:
    """Fast Gradient Sign Method — 기울기의 부호로 한 걸음.

        x_adv = clip(x + epsilon * sign(grad_x loss))

    한 번만 셈하므로 싸다. 다만 손실이 그 자리에서 선형이라고 보는 셈이라,
    굽은 자리에서는 약하다. 그것이 PGD가 여러 걸음을 밟는 까닭이다.
    """

    def __init__(self, model: nn.Module, epsilon: float = 0.03,
                 clip_min: float = 0.0, clip_max: float = 1.0):
        self.model = model
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max

    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()                     # 드롭아웃 등이 기울기를 흔들지 않게

        x_adv = x.clone().detach().requires_grad_(True)
        loss = F.cross_entropy(self.model(x_adv), y)
        grad, = torch.autograd.grad(loss, x_adv)

        out = x + self.epsilon * grad.sign()
        out = out.clamp(self.clip_min, self.clip_max)

        if was_training:
            self.model.train()
        return out.detach()


class PGD:
    """Projected Gradient Descent — 작은 걸음을 여러 번 밟고 매번 공 안으로 되민다.

        x_0   = x + U(-epsilon, epsilon)          (무작위로 시작한다)
        x_t+1 = proj_ball( x_t + alpha * sign(grad) )

    무작위로 시작하는 것이 중요하다. 늘 x에서 출발하면 기울기가 0인 자리에
    걸려 공격이 실패하는 것을 "든든함"으로 잘못 읽게 된다.
    """

    def __init__(self, model: nn.Module, epsilon: float = 0.03,
                 alpha: float = None, num_iter: int = 40,
                 random_start: bool = True,
                 clip_min: float = 0.0, clip_max: float = 1.0):
        self.model = model
        self.epsilon = epsilon
        # 걸음 폭을 따로 주지 않으면 공을 두어 번 건널 만큼으로 잡는다
        self.alpha = alpha if alpha is not None else max(epsilon / 4, 2.5 * epsilon / num_iter)
        self.num_iter = num_iter
        self.random_start = random_start
        self.clip_min = clip_min
        self.clip_max = clip_max

    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()

        if self.random_start:
            delta = torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
            x_adv = (x + delta).clamp(self.clip_min, self.clip_max)
        else:
            x_adv = x.clone()
        x_adv = x_adv.detach()

        for _ in range(self.num_iter):
            x_adv.requires_grad_(True)
            loss = F.cross_entropy(self.model(x_adv), y)
            grad, = torch.autograd.grad(loss, x_adv)
            x_adv = x_adv.detach() + self.alpha * grad.sign()
            # epsilon 공 안으로 되민 뒤, 그림이 가질 수 있는 값으로 자른다
            x_adv = torch.min(torch.max(x_adv, x - self.epsilon), x + self.epsilon)
            x_adv = x_adv.clamp(self.clip_min, self.clip_max).detach()

        if was_training:
            self.model.train()
        return x_adv
