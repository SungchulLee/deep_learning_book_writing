# PPO

PPO was introduced in the 2017 paper "Proximal Policy Optimization Algorithms." - Policy gradient with a *clipped surrogate objective* to prevent large updates   - Often uses GAE advantages and minibatch epochs.

This implementation provides a concise, educational reference for PPO. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## 코드

```python
#!/usr/bin/env python3
"""
PPO - Proximal Policy Optimization
Paper: "Proximal Policy Optimization Algorithms" (2017)
Authors: John Schulman et al.
Key idea:
  - Policy gradient with a *clipped surrogate objective* to prevent large updates
  - Often uses GAE advantages and minibatch epochs

Clipped objective:
  r_t = pi(a|s) / pi_old(a|s)
  L_clip = E[ min( r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t ) ]

File: appendix/rl/ppo.py
Note: Educational implementation of PPO loss (discrete actions).
"""

import torch
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


def ppo_loss(
    logits_new,        # (B, A) current policy logits
    logits_old,        # (B, A) behavior/old policy logits (frozen)
    actions,           # (B,)
    advantages,        # (B,)
    returns,           # (B,)
    values,            # (B,) critic values from current network
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
):
    """
    Compute PPO losses:
      - clipped policy loss
      - value loss
      - entropy bonus
    """
    # Compute log probs under new and old policies
    logp_new = F.log_softmax(logits_new, dim=1)
    logp_old = F.log_softmax(logits_old, dim=1)

    logp_new_a = logp_new.gather(1, actions.long().unsqueeze(1)).squeeze(1)
    logp_old_a = logp_old.gather(1, actions.long().unsqueeze(1)).squeeze(1)

    # Importance ratio r_t = exp(log pi_new - log pi_old)
    ratio = torch.exp(logp_new_a - logp_old_a)

    # Advantages are typically standardized; detach so actor doesn't backprop into advantage
    A = advantages.detach()

    # Clipped surrogate objective
    unclipped = ratio * A
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * A
    policy_loss = -torch.min(unclipped, clipped).mean()

    # Value function loss
    value_loss = F.mse_loss(values, returns)

    # Entropy bonus (encourage exploration)
    p_new = torch.softmax(logits_new, dim=1)
    entropy = -(p_new * logp_new).sum(dim=1).mean()

    total = policy_loss + value_coef * value_loss - entropy_coef * entropy
    return total, {"policy_loss": policy_loss, "value_loss": value_loss, "entropy": entropy}


if __name__ == "__main__":
    pass```

## 논의

This implementation demonstrates key concepts in reinforcement learning using clean, readable PyTorch code. The modular structure makes it easy to study individual components and adapt them for different tasks or datasets.

여기 실린 코드는 본보기 짜보기라 다듬기보다 알아보기 쉬움을 앞세운다. 서비스 얼개라면 흔히 섞인 촘촘함 익히기, 흩은 자료 나란히, 더 정교한 자료 불리기를 더한다. 그래도 여기서 보인 얼개의 고갱이 깨침은 크기와 상관없이 그대로다.

## 익힘 문제

**익힘 1.**
코드를 읽고 고갱이가 되는 설계 판단을 짚어라. 짜기에서 고른 것 셋을 들고, 저마다 왜 북돋움 배움에 알맞은지 밝혀라.

??? success "익힘 1 풀이"
    설계 판단은 짜보기마다 다르나 흔히 이런 것이 있다. (1) 살림 함수 고르기 -- ReLU 갈래는 기울기가 잦아들지 않아 익히기가 빠르다. (2) 잣대 잡는 꾀 -- 묶음 잣대 잡기가 안쪽 함께 바뀌는 옮겨감을 줄여 익힘을 든든하게 한다. (3) 나머지 이음 -- 있으면 건너뛰는 길을 주어 깊은 그물에서 기울기가 흐르게 한다. 고른 것마다 드러내는 힘, 셈 값, 익힘의 든든함 사이의 맞바꿈을 보여 준다.

---

**익힘 2.**
들임의 꼴과 자료 갈래가 바라는 대로인지 살피는 들임 살피기를 으뜸 함수나 갈래에 더하여라. 올바르지 않은 들임에는 알아듣기 쉬운 어긋남 알림을 띄워라.

??? success "익힘 2 풀이"
    `forward` 방법(또는 알맞은 함수)의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 살핌을 더한다. 꼴을 살피려면 종요로운 차수를 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아듣기 쉬운 어긋남 알림은 벌레잡기를 크게 앞당기고 코드를 되쓰기 든든하게 한다.

---

**익힘 3.**
이 짜보기가 무너질 만한 결 둘을 밝히고, 저마다 어떻게 짚어내고 고칠지 밝혀라.

??? success "익힘 3 풀이"
    흔히 무너지는 결은 이렇다. (1) **기울기가 사라지거나 터짐** -- 기울기 크기를 지켜보아 짚어낸다(`torch.nn.utils.clip_grad_norm_`이나 켜마다 `param.grad.norm()` 적기). 기울기 자르기, 더 나은 첫자리 잡기(Xavier/Kaiming), 얼개 고치기(나머지 이음, 잣대 잡기)로 고친다. (2) **지나치게 맞추기** -- 익힘 잃음은 줄어드는데 따짐 잃음이 오르면 짚어낸다. 다독임(드롭아웃, 짐 줄이기, 자료 불리기)이나 모형 크기 줄이기로 고친다. 익힘과 따짐 자를 늘 함께 지켜보아 이를 일찍 잡아야 한다.

---

**익힘 4.**
Write a comprehensive test function that validates the PPO implementation. Test edge cases including empty inputs, single-element inputs, very large inputs, and inputs with extreme values (zeros, very large numbers).

??? success "익힘 4 풀이"
    금 언저리 조건을 두루 건드리는 시험 함수를 짓는다.
    ```python
    def test_ppo():
        model = PPO(...)
        # 여느 들임
        assert model(normal_input).shape == expected_shape
        # 원소 하나짜리 묶음
        assert model(single_input).shape == (1, ...)
        # 큰 값(넘침을 살핀다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 기울기 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    얼개가 끝에서 끝까지 익히기를 받치는지 알려면 기울기 흐름을 시험하는 것이 특히 종요롭다.
