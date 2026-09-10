# 방침 매개변수 나타내기

방침 매개변수 나타내기는 방침 기울기 바탕에서 종요로운 생각이다. 따로 떨어진 움직임 공간과 이어진 움직임 공간을 함께 다룬다. 이 구현은 여기에 걸린 종요로운 알고리즘과 자료 얼개를 손으로 만져 보이며, 이치 바탕과 실제로 서비스에 올릴 때 살필 것을 함께 보여 준다.

## 1. 코드

```python
"""
34.1.1장: 방침 매개변수 나타내기
========================================
따로 떨어진 움직임 공간과 이어진 움직임 공간을 위한 여러 방침
매개변수 나타내기 꾀의 구현.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Beta
import numpy as np
import gymnasium as gym

# ========================================================================
# 메인
# ========================================================================


# ---------------------------------------------------------------------------
# 무게 첫 값 매기기 도구
# ---------------------------------------------------------------------------

def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0):
    """PPO의 좋은 버릇을 따르는 직교 첫 값 매기기."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


# ---------------------------------------------------------------------------
# 따로 떨어진 방침 (소프트맥스 / 갈래)
# ---------------------------------------------------------------------------

class DiscretePolicy(nn.Module):
    """
    따로 떨어진 움직임 공간을 위한 갈래 방침.
    
    그물이 로짓을 내놓고 소프트맥스로 갈래 분포로 바꾼다. 셈이
    든든하도록 로그 소프트맥스를 쓴다.
    
    매개변수
    ----------
    obs_dim : int
        봄 공간의 차원.
    act_dim : int
        따로 떨어진 움직임의 개수.
    hidden_dim : int
        숨은 켜의 크기.
    """
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, act_dim), std=0.01),  # 거의 고르도록 작게 매긴다
        )
    
    def forward(self, obs: torch.Tensor):
        """움직임 로짓을 돌려준다."""
        return self.network(obs)
    
    def get_distribution(self, obs: torch.Tensor) -> Categorical:
        """움직임에 대한 갈래 분포를 돌려준다."""
        logits = self.forward(obs)
        return Categorical(logits=logits)
    
    def get_action(self, obs: torch.Tensor):
        """움직임을 뽑아 움직임, 로그 낌새, 엔트로피를 돌려준다."""
        dist = self.get_distribution(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """주어진 상태-움직임 짝의 로그 낌새와 엔트로피를 따진다."""
        dist = self.get_distribution(obs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy


# ---------------------------------------------------------------------------
# 가우스 방침 (이어진 것, 매이지 않음)
# ---------------------------------------------------------------------------

class GaussianPolicy(nn.Module):
    """
    이어진 움직임 공간을 위한 대각 가우스 방침.
    
    갈래 둘:
    - 상태와 매이지 않은 log_std: 배우는 매개변수(기본값, PPO에서 씀)
    - 상태에 딸린 log_std: 그물이 내놓음
    
    매개변수
    ----------
    obs_dim : int
        봄 공간의 차원.
    act_dim : int
        이어진 움직임 공간의 차원.
    hidden_dim : int
        숨은 켜의 크기.
    state_dependent_std : bool
        True이면 표준편차가 그물을 거쳐 상태에 딸린다.
    log_std_init : float
        로그 표준편차의 첫 값.
    """
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 64,
        state_dependent_std: bool = False,
        log_std_init: float = 0.0,
    ):
        super().__init__()
        self.state_dependent_std = state_dependent_std
        self.act_dim = act_dim
        
        # 함께 쓰는 특징 뽑개
        self.features = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        
        # 평균 머리
        self.mean_head = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
        
        if state_dependent_std:
            # 상태에 딸림: 그물이 log_std를 내놓는다
            self.log_std_head = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
        else:
            # 상태와 매이지 않음: 배우는 매개변수
            self.log_std = nn.Parameter(torch.full((act_dim,), log_std_init))
    
    def forward(self, obs: torch.Tensor):
        """가우스 방침의 평균과 log_std를 돌려준다."""
        features = self.features(obs)
        mean = self.mean_head(features)
        
        if self.state_dependent_std:
            log_std = self.log_std_head(features)
            log_std = torch.clamp(log_std, min=-20, max=2)  # 든든함을 위한 자르기
        else:
            log_std = self.log_std.expand_as(mean)
        
        return mean, log_std
    
    def get_distribution(self, obs: torch.Tensor) -> Normal:
        """방침의 정규 분포를 돌려준다."""
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        return Normal(mean, std)
    
    def get_action(self, obs: torch.Tensor):
        """움직임을 뽑아 움직임, 로그 낌새, 엔트로피를 돌려준다."""
        dist = self.get_distribution(obs)
        action = dist.sample()
        # 여러 변수일 때 움직임 차원에 걸쳐 로그 낌새를 더한다
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """주어진 상태-움직임 짝의 로그 낌새와 엔트로피를 따진다."""
        dist = self.get_distribution(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


# ---------------------------------------------------------------------------
# 눌러 담은 가우스 방침 (SAC 꼴, 매인 움직임)
# ---------------------------------------------------------------------------

class SquashedGaussianPolicy(nn.Module):
    """
    매인 이어진 움직임 공간을 위한 눌러 담은 가우스 방침.
    
    가우스에서 뽑고 tanh를 매겨 움직임을 [-1, 1]으로 매어 둔다.
    로그 낌새에 변수 바꿈 바로잡기를 매긴다.
    
    SAC(부드러운 행위자-비평가)에서 쓴다.
    
    매개변수
    ----------
    obs_dim : int
        봄 공간의 차원.
    act_dim : int
        움직임 공간의 차원.
    hidden_dim : int
        숨은 켜의 크기.
    action_scale : float
        움직임의 잣대 인자(참 움직임 매임으로 맞대려고 쓴다).
    action_bias : float
        움직임의 치우침.
    """
    
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2
    EPS = 1e-6
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        action_scale: float = 1.0,
        action_bias: float = 0.0,
    ):
        super().__init__()
        
        self.features = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
        )
        self.mean_head = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
        self.log_std_head = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
        
        # 움직임 다시 잣대기
        self.register_buffer("action_scale", torch.tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor(action_bias, dtype=torch.float32))
    
    def forward(self, obs: torch.Tensor):
        """평균과 log_std를 돌려준다."""
        features = self.features(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """
        tanh로 눌러 담아 움직임을 뽑는다.
        
        돌려주는 값
        -------
        action : Tensor
            눌러 담고 다시 잣댄 움직임.
        log_prob : Tensor
            변수 바꿈 바로잡기를 매긴 로그 낌새.
        mean : Tensor
            평균 움직임(붙박이로 따질 때 쓴다).
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        
        if deterministic:
            u = mean
        else:
            u = dist.rsample()  # 기울기가 흐르도록 다시 매개변수 매긴 뽑기
        
        # tanh로 눌러 담기
        action = torch.tanh(u)
        
        # 변수 바꿈 바로잡기를 매긴 로그 낌새
        log_prob = dist.log_prob(u)
        # 바로잡기: log|det(da/du)| = sum(log(1 - tanh^2(u)))
        log_prob -= torch.log(1 - action.pow(2) + self.EPS)
        log_prob = log_prob.sum(dim=-1)
        
        # 참 움직임 매임으로 다시 잣댄다
        action = action * self.action_scale + self.action_bias
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean_action


# ---------------------------------------------------------------------------
# 베타 방침 (본디부터 매여 있음)
# ---------------------------------------------------------------------------

class BetaPolicy(nn.Module):
    """
    매인 이어진 움직임 공간 [0, 1]을 위한 베타 분포 방침.
    
    tanh로 눌러 담기와 로그 낌새 바로잡기가 필요 없다.
    
    매개변수
    ----------
    obs_dim : int
        봄 공간의 차원.
    act_dim : int
        움직임 공간의 차원.
    hidden_dim : int
        숨은 켜의 크기.
    """
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.alpha_head = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
        self.beta_head = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
    
    def forward(self, obs: torch.Tensor):
        """알파와 베타 매개변수를 돌려준다."""
        features = self.network(obs)
        alpha = F.softplus(self.alpha_head(features)) + 1.0  # 봉우리가 하나이려면 > 1
        beta = F.softplus(self.beta_head(features)) + 1.0
        return alpha, beta
    
    def get_distribution(self, obs: torch.Tensor) -> Beta:
        alpha, beta = self.forward(obs)
        return Beta(alpha, beta)
    
    def get_action(self, obs: torch.Tensor):
        dist = self.get_distribution(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy


# ---------------------------------------------------------------------------
# 등뼈를 함께 쓰는 행위자-비평가
# ---------------------------------------------------------------------------

class ActorCriticShared(nn.Module):
    """
    등뼈를 함께 쓰는 행위자-비평가 그물.
    
    함께 쓰는 특징 뽑개에 방침 머리와 값 머리를 따로 둔다.
    따로 떨어진 움직임 공간과 이어진 움직임 공간을 모두 받쳐 준다.
    """
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 64,
        continuous: bool = False,
    ):
        super().__init__()
        self.continuous = continuous
        
        # 함께 쓰는 특징 뽑개
        self.features = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        
        # 방침 머리
        if continuous:
            self.mean_head = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
            self.log_std = nn.Parameter(torch.zeros(act_dim))
        else:
            self.policy_head = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
        
        # 값 머리
        self.value_head = layer_init(nn.Linear(hidden_dim, 1), std=1.0)
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.features(obs)).squeeze(-1)
    
    def get_action_and_value(self, obs: torch.Tensor, action=None):
        features = self.features(obs)
        value = self.value_head(features).squeeze(-1)
        
        if self.continuous:
            mean = self.mean_head(features)
            std = self.log_std.exp().expand_as(mean)
            dist = Normal(mean, std)
            if action is None:
                action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)
        else:
            logits = self.policy_head(features)
            dist = Categorical(logits=logits)
            if action is None:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        
        return action, log_prob, entropy, value


# ---------------------------------------------------------------------------
# 따로 둔 행위자-비평가 그물
# ---------------------------------------------------------------------------

class ActorCriticSeparate(nn.Module):
    """
    그물을 따로 두는 행위자-비평가.
    
    방침과 값에 서로 매이지 않은 그물을 써서 두 목표 사이에서
    기울기가 서로를 방해하지 않게 한다.
    """
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 64,
        continuous: bool = False,
    ):
        super().__init__()
        self.continuous = continuous
        
        # 행위자 그물
        if continuous:
            self.actor = GaussianPolicy(obs_dim, act_dim, hidden_dim)
        else:
            self.actor = DiscretePolicy(obs_dim, act_dim, hidden_dim)
        
        # 비평가 그물
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)
    
    def get_action_and_value(self, obs: torch.Tensor, action=None):
        value = self.get_value(obs)
        
        if action is None:
            action, log_prob, entropy = self.actor.get_action(obs)
        else:
            log_prob, entropy = self.actor.evaluate_actions(obs, action)
        
        return action, log_prob, entropy, value


# ---------------------------------------------------------------------------
# 온도로 잣댄 방침
# ---------------------------------------------------------------------------

class TemperatureScaledPolicy(nn.Module):
    """
    따로 떨어진 방침에 온도 잣대기를 매기는 감싸개.
    
    온도가 낮으면 더 욕심스럽다(써먹기)
    온도가 높으면 더 고르다(살펴보기)
    """
    
    def __init__(self, base_policy: DiscretePolicy, temperature: float = 1.0):
        super().__init__()
        self.base_policy = base_policy
        self.temperature = temperature
    
    def get_distribution(self, obs: torch.Tensor) -> Categorical:
        logits = self.base_policy(obs)
        scaled_logits = logits / self.temperature
        return Categorical(logits=scaled_logits)
    
    def get_action(self, obs: torch.Tensor):
        dist = self.get_distribution(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy


# ---------------------------------------------------------------------------
# 보여 주기
# ---------------------------------------------------------------------------

def demo_discrete_policy():
    """CartPole로 따로 떨어진 방침을 보인다."""
    print("=" * 60)
    print("Discrete Policy Demo (CartPole-v1)")
    print("=" * 60)
    
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    policy = DiscretePolicy(obs_dim, act_dim)
    print(f"Observation dim: {obs_dim}, Action dim: {act_dim}")
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    obs, _ = env.reset()
    obs_t = torch.FloatTensor(obs).unsqueeze(0)
    
    action, log_prob, entropy = policy.get_action(obs_t)
    print(f"\nSampled action: {action.item()}")
    print(f"Log probability: {log_prob.item():.4f}")
    print(f"Entropy: {entropy.item():.4f}")
    
    dist = policy.get_distribution(obs_t)
    print(f"Action probabilities: {dist.probs.detach().numpy().round(4)}")
    
    env.close()


def demo_gaussian_policy():
    """Pendulum으로 가우스 방침을 보인다."""
    print("\n" + "=" * 60)
    print("Gaussian Policy Demo (Pendulum-v1)")
    print("=" * 60)
    
    env = gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    policy = GaussianPolicy(obs_dim, act_dim)
    print(f"Observation dim: {obs_dim}, Action dim: {act_dim}")
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    obs, _ = env.reset()
    obs_t = torch.FloatTensor(obs).unsqueeze(0)
    
    action, log_prob, entropy = policy.get_action(obs_t)
    mean, log_std = policy(obs_t)
    
    print(f"\nMean: {mean.detach().numpy().round(4)}")
    print(f"Std: {log_std.exp().detach().numpy().round(4)}")
    print(f"Sampled action: {action.detach().numpy().round(4)}")
    print(f"Log probability: {log_prob.item():.4f}")
    print(f"Entropy: {entropy.item():.4f}")
    
    env.close()


def demo_squashed_gaussian():
    """눌러 담은 가우스 방침을 보인다."""
    print("\n" + "=" * 60)
    print("Squashed Gaussian Policy Demo")
    print("=" * 60)
    
    obs_dim, act_dim = 3, 1
    policy = SquashedGaussianPolicy(obs_dim, act_dim)
    
    obs = torch.randn(1, obs_dim)
    action, log_prob, mean_action = policy.get_action(obs)
    
    print(f"Sampled action (bounded): {action.detach().numpy().round(4)}")
    print(f"Mean action: {mean_action.detach().numpy().round(4)}")
    print(f"Log probability (with correction): {log_prob.item():.4f}")
    print(f"Action in [-1, 1]: {(action.abs() <= 1.0).all().item()}")


def demo_actor_critic():
    """행위자-비평가 얼개를 보인다."""
    print("\n" + "=" * 60)
    print("Actor-Critic Architecture Demo")
    print("=" * 60)
    
    obs_dim, act_dim = 4, 2
    
    # 등뼈를 함께 씀
    shared = ActorCriticShared(obs_dim, act_dim, continuous=False)
    obs = torch.randn(8, obs_dim)  # 묶음 크기 8
    action, log_prob, entropy, value = shared.get_action_and_value(obs)
    print(f"\nShared Actor-Critic:")
    print(f"  Actions shape: {action.shape}")
    print(f"  Log probs shape: {log_prob.shape}")
    print(f"  Values shape: {value.shape}")
    print(f"  Params: {sum(p.numel() for p in shared.parameters()):,}")
    
    # 그물을 따로 둠
    separate = ActorCriticSeparate(obs_dim, act_dim, continuous=False)
    action, log_prob, entropy, value = separate.get_action_and_value(obs)
    print(f"\nSeparate Actor-Critic:")
    print(f"  Actions shape: {action.shape}")
    print(f"  Log probs shape: {log_prob.shape}")
    print(f"  Values shape: {value.shape}")
    print(f"  Params: {sum(p.numel() for p in separate.parameters()):,}")


if __name__ == "__main__":
    demo_discrete_policy()
    demo_gaussian_policy()
    demo_squashed_gaussian()
    demo_actor_critic()
```

## 2. 논의

이 구현은 방침 매개변수 나타내기의 한가운데 논리를 담은 `DiscretePolicy`, `GaussianPolicy`, `SquashedGaussianPolicy` 클래스를 축으로 삼는다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 갈라놓는 조각 설계를 따른다.

보여 주기 함수는 이 조각들을 여느 힘 북돋우는 배움 잣대에 실제로 써 보인다. 그 출력을 살피면 매개변수 고름과 문제 얼개에 따라 알고리즘의 됨됨이가 어떻게 달라지는지 볼 수 있다.

쓰임의 눈으로 보면 이 구현은 날 성능보다 또렷함을 앞세운다. 서비스 시스템은 묶음 셈하기, GPU 빠르게 하기, 더 야무진 매개변수 벼리기 같은 다듬기를 더 넣는 것이 보통이다. 그렇더라도 여기서 보인 한가운데 알고리즘 생각은 큰 잣대의 쓰임새에 그대로 옮겨 간다.

## 연습문제

**연습문제 1.**
보여 주기 코드를 돌리고 종요로운 출력 재기를 적어라. 매개변수 하나(배움률, 숨은 차원, 켜 개수 따위)를 고쳐 열매가 어떻게 달라지는지 밝혀라.

??? success "연습문제 1 풀이"
    보여 주기를 돌린 뒤 다른 것을 붙박아 두고 고른 매개변수만 짜임 있게 바꾼다. 보기로 숨은 차원을 곱절로 늘리면 나타내는 그릇이 커지지만 셈하는 때가 는다. 배움률은 한결같지 않은 결과를 낳는다. 너무 작으면 더디게 모이고 너무 크면 들쭉날쭉해진다. 고른 매개변수의 서로 다른 값 적어도 셋에 대해 또렷한 수를 적어 두라.

---

**연습문제 2.**
이 구현에서 종요로운 얼개 고름이 맡은 몫을 풀어라. 왜 그런 활성 함수, 고르게 하기 꾀, 손실 함수를 쓰는가? 다른 것으로 바꾸면 무슨 일이 생기는가?

??? success "연습문제 2 풀이"
    이 얼개 고름은 방침 기울기 바탕에서 자리 잡은 좋은 버릇을 비춘다. 보기로 ReLU 활성은 곧지 않음을 주면서 0보다 큰 들임에서 기울기가 사라지는 것을 막는다. 손실 함수는 일감 갈래에 맞추어 고른다(갈래 나누기에는 사귐 엔트로피, 되돌이에는 평균 제곱 잘못). 다른 것으로 바꾸면(보기로 시그모이드 활성, L1 손실) 가장 좋게 하기 지형이 바뀌어 됨됨이가 나빠질 수 있으나, 어떤 자리에서는 바꾸는 것이 이로울 수도 있다.

---

**연습문제 3.**
이 구현을 더 만만치 않은 자리로 넓혀라. 더 큰 자료 뭉치, 다른 문제 갈래, 덧붙인 기능 가운데 하나를 고르라. 고친 바를 밝히고 됨됨이에 미친 바를 따져라.

??? success "연습문제 3 풀이"
    절로 떠오르는 넓히기 하나는 정칙화(드롭아웃, 무게 삭임)나 더 야무진 얼개(켜 더하기, 건너뛰는 이음)를 더하는 것이다. 고른 넓히기를 만들고 같은 자료로 익힌 뒤 앞뒤의 재기를 견주어라. 이 넓히기는 처음 알고리즘과 고친 바의 이치 밑뜻을 모두 아는 것을 보여야 한다.

## 정리하며

**다룬 것** — 방침 매개변수 나타내기

이 구현은 방침 매개변수 나타내기의 한가운데 논리를 담은 `DiscretePolicy`, `GaussianPolicy`, `SquashedGaussianPolicy` 클래스를 축으로 삼는다.

고갱이 갈래는 `DiscretePolicy`, `GaussianPolicy`, `SquashedGaussianPolicy`, `BetaPolicy`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
