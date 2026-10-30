# 폴리시 파라메터화

폴리시 파라메터 나타내기는 폴리시 경사 기초에서 중요한 생각이다. 따로 떨어진 액션 공간과 이어진 액션 공간을 함께 다룬다. 이 구현은 여기에 걸린 중요한 알고리즘과 자료 구조를 손으로 만져 보이며, 이치 기초와 실제로 서비스에 올릴 때 살필 것을 함께 보여 준다.

## 1. 코드

??? note "코드 (587줄)"

    ```python
    """
    34.1.1장: 폴리시 파라메터 나타내기
    ========================================
    따로 떨어진 액션 공간과 이어진 액션 공간을 위한 여러 폴리시
    파라메터 나타내기 꾀의 구현.
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
    # 따로 떨어진 폴리시 (소프트맥스 / 클래스)
    # ---------------------------------------------------------------------------

    class DiscretePolicy(nn.Module):
        """
        따로 떨어진 액션 공간을 위한 클래스 폴리시.

        넷트웍이 로짓을 내놓고 소프트맥스로 클래스 분포로 바꾼다. 셈이
        든든하도록 로그 소프트맥스를 쓴다.

        파라메터
        ----------
        obs_dim : int
            봄 공간의 차원.
        act_dim : int
            따로 떨어진 액션의 개수.
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
            """액션 로짓을 돌려준다."""
            return self.network(obs)

        def get_distribution(self, obs: torch.Tensor) -> Categorical:
            """액션에 대한 클래스 분포를 돌려준다."""
            logits = self.forward(obs)
            return Categorical(logits=logits)

        def get_action(self, obs: torch.Tensor):
            """액션을 뽑아 액션, 로그 확률, 엔트로피를 돌려준다."""
            dist = self.get_distribution(obs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            return action, log_prob, entropy

        def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
            """주어진 상태-액션 짝의 로그 확률과 엔트로피를 따진다."""
            dist = self.get_distribution(obs)
            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()
            return log_prob, entropy


    # ---------------------------------------------------------------------------
    # 가우스 폴리시 (이어진 것, 매이지 않음)
    # ---------------------------------------------------------------------------

    class GaussianPolicy(nn.Module):
        """
        이어진 액션 공간을 위한 대각 가우스 폴리시.

        클래스 둘:
        - 상태와 매이지 않은 log_std: 배우는 파라메터(기본값, PPO에서 씀)
        - 상태에 딸린 log_std: 넷트웍이 내놓음

        파라메터
        ----------
        obs_dim : int
            봄 공간의 차원.
        act_dim : int
            이어진 액션 공간의 차원.
        hidden_dim : int
            숨은 켜의 크기.
        state_dependent_std : bool
            True이면 표준편차가 넷트웍을 거쳐 상태에 딸린다.
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

            # 함께 쓰는 피처 뽑개
            self.features = nn.Sequential(
                layer_init(nn.Linear(obs_dim, hidden_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden_dim, hidden_dim)),
                nn.Tanh(),
            )

            # 평균 머리
            self.mean_head = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)

            if state_dependent_std:
                # 상태에 딸림: 넷트웍이 log_std를 내놓는다
                self.log_std_head = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
            else:
                # 상태와 매이지 않음: 배우는 파라메터
                self.log_std = nn.Parameter(torch.full((act_dim,), log_std_init))

        def forward(self, obs: torch.Tensor):
            """가우스 폴리시의 평균과 log_std를 돌려준다."""
            features = self.features(obs)
            mean = self.mean_head(features)

            if self.state_dependent_std:
                log_std = self.log_std_head(features)
                log_std = torch.clamp(log_std, min=-20, max=2)  # 든든함을 위한 자르기
            else:
                log_std = self.log_std.expand_as(mean)

            return mean, log_std

        def get_distribution(self, obs: torch.Tensor) -> Normal:
            """폴리시의 정규 분포를 돌려준다."""
            mean, log_std = self.forward(obs)
            std = log_std.exp()
            return Normal(mean, std)

        def get_action(self, obs: torch.Tensor):
            """액션을 뽑아 액션, 로그 확률, 엔트로피를 돌려준다."""
            dist = self.get_distribution(obs)
            action = dist.sample()
            # 여러 변수일 때 액션 차원에 걸쳐 로그 확률을 더한다
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            return action, log_prob, entropy

        def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
            """주어진 상태-액션 짝의 로그 확률과 엔트로피를 따진다."""
            dist = self.get_distribution(obs)
            log_prob = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            return log_prob, entropy


    # ---------------------------------------------------------------------------
    # 눌러 담은 가우스 폴리시 (SAC 꼴, 매인 액션)
    # ---------------------------------------------------------------------------

    class SquashedGaussianPolicy(nn.Module):
        """
        매인 이어진 액션 공간을 위한 눌러 담은 가우스 폴리시.

        가우스에서 뽑고 tanh를 매겨 액션을 [-1, 1]으로 매어 둔다.
        로그 확률에 변수 바꿈 바로잡기를 매긴다.

        SAC(부드러운 액터-크리틱)에서 쓴다.

        파라메터
        ----------
        obs_dim : int
            봄 공간의 차원.
        act_dim : int
            액션 공간의 차원.
        hidden_dim : int
            숨은 켜의 크기.
        action_scale : float
            액션의 잣대 인자(참 액션 매임으로 맞대려고 쓴다).
        action_bias : float
            액션의 치우침.
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

            # 액션 다시 잣대기
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
            tanh로 눌러 담아 액션을 뽑는다.

            돌려주는 값
            -------
            action : Tensor
                눌러 담고 다시 잣댄 액션.
            log_prob : Tensor
                변수 바꿈 바로잡기를 매긴 로그 확률.
            mean : Tensor
                평균 액션(붙박이로 따질 때 쓴다).
            """
            mean, log_std = self.forward(obs)
            std = log_std.exp()
            dist = Normal(mean, std)

            if deterministic:
                u = mean
            else:
                u = dist.rsample()  # 그래디언트가 흐르도록 다시 파라메터 매긴 뽑기

            # tanh로 눌러 담기
            action = torch.tanh(u)

            # 변수 바꿈 바로잡기를 매긴 로그 확률
            log_prob = dist.log_prob(u)
            # 바로잡기: log|det(da/du)| = sum(log(1 - tanh^2(u)))
            log_prob -= torch.log(1 - action.pow(2) + self.EPS)
            log_prob = log_prob.sum(dim=-1)

            # 참 액션 매임으로 다시 잣댄다
            action = action * self.action_scale + self.action_bias
            mean_action = torch.tanh(mean) * self.action_scale + self.action_bias

            return action, log_prob, mean_action


    # ---------------------------------------------------------------------------
    # 베타 폴리시 (본디부터 매여 있음)
    # ---------------------------------------------------------------------------

    class BetaPolicy(nn.Module):
        """
        매인 이어진 액션 공간 [0, 1]을 위한 베타 분포 폴리시.

        tanh로 눌러 담기와 로그 확률 바로잡기가 필요 없다.

        파라메터
        ----------
        obs_dim : int
            봄 공간의 차원.
        act_dim : int
            액션 공간의 차원.
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
            """알파와 베타 파라메터를 돌려준다."""
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
    # 등뼈를 함께 쓰는 액터-크리틱
    # ---------------------------------------------------------------------------

    class ActorCriticShared(nn.Module):
        """
        등뼈를 함께 쓰는 액터-크리틱 넷트웍.

        함께 쓰는 피처 뽑개에 폴리시 머리와 값 머리를 따로 둔다.
        따로 떨어진 액션 공간과 이어진 액션 공간을 모두 받쳐 준다.
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

            # 함께 쓰는 피처 뽑개
            self.features = nn.Sequential(
                layer_init(nn.Linear(obs_dim, hidden_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden_dim, hidden_dim)),
                nn.Tanh(),
            )

            # 폴리시 머리
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
    # 따로 둔 액터-크리틱 넷트웍
    # ---------------------------------------------------------------------------

    class ActorCriticSeparate(nn.Module):
        """
        넷트웍을 따로 두는 액터-크리틱.

        폴리시와 값에 서로 매이지 않은 넷트웍을 써서 두 목표 사이에서
        그래디언트가 서로를 방해하지 않게 한다.
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

            # 액터 넷트웍
            if continuous:
                self.actor = GaussianPolicy(obs_dim, act_dim, hidden_dim)
            else:
                self.actor = DiscretePolicy(obs_dim, act_dim, hidden_dim)

            # 크리틱 넷트웍
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
    # 온도로 잣댄 폴리시
    # ---------------------------------------------------------------------------

    class TemperatureScaledPolicy(nn.Module):
        """
        따로 떨어진 폴리시에 온도 잣대기를 매기는 감싸개.

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
        """CartPole로 따로 떨어진 폴리시를 보인다."""
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
        """Pendulum으로 가우스 폴리시를 보인다."""
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
        """눌러 담은 가우스 폴리시를 보인다."""
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
        """액터-크리틱 구조를 보인다."""
        print("\n" + "=" * 60)
        print("Actor-Critic Architecture Demo")
        print("=" * 60)

        obs_dim, act_dim = 4, 2

        # 등뼈를 함께 씀
        shared = ActorCriticShared(obs_dim, act_dim, continuous=False)
        obs = torch.randn(8, obs_dim)  # 배치 크기 8
        action, log_prob, entropy, value = shared.get_action_and_value(obs)
        print(f"\nShared Actor-Critic:")
        print(f"  Actions shape: {action.shape}")
        print(f"  Log probs shape: {log_prob.shape}")
        print(f"  Values shape: {value.shape}")
        print(f"  Params: {sum(p.numel() for p in shared.parameters()):,}")

        # 넷트웍을 따로 둠
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

이 구현은 폴리시 파라메터 나타내기의 한가운데 논리를 담은 `DiscretePolicy`, `GaussianPolicy`, `SquashedGaussianPolicy` 클래스를 축으로 삼는다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 갈라놓는 조각 설계를 따른다.

보여 주기 함수는 이 조각들을 여느 강화학습 잣대에 실제로 써 보인다. 그 출력을 살피면 파라메터 공정성과 문제 구조에 따라 알고리즘의 성능이 어떻게 달라지는지 볼 수 있다.

쓰임의 눈으로 보면 이 구현은 날 성능보다 또렷함을 앞세운다. 서비스 시스템은 배치 계산, GPU 빠르게 하기, 더 야무진 파라메터 벼리기 같은 다듬기를 더 넣는 것이 보통이다. 그렇더라도 여기서 보인 한가운데 알고리즘 생각은 큰 잣대의 쓰임새에 그대로 옮겨 간다.

<div class="storybox" markdown>

**그림으로 보는 AI 1.** 폭을 파라메터로 둔다는 것의 뜻

이어진 액션에 가우스 분포를 쓰면 파라메터가 둘이 된다 — 평균 $\mu$ 와 표준편차 $\sigma$ 다. 둘째가 있다는 것이 그냥 분포 하나를 고른 결과처럼 보이지만, 윌리엄스는 1992년 논문에서 이 점을 따로 짚었다. **$\sigma$ 를 다스리는 것은 곧 탐색을 다스리는 것**이라는 것이다 — "control over $\sigma$ is tantamount to control over the unit's exploratory behavior." 파라메터가 하나뿐인 분포를 쓰는 유닛과 달리, 어디를 살필지와 **얼마나 넓게 살필지**를 따로 정할 수 있다.

그러면 $\sigma$ 는 어떻게 움직이는가. 논문 8.2절이 규칙을 말로 풀어 두었는데, 한 줄로 줄이면 이렇다 — **뽑힌 액션이 다시 나오기 쉬워지는 쪽으로** 움직인다. 좋은 값이 평균 **가까이**($|y-\mu| < \sigma$)에서 나오면 $\sigma$ 가 줄고, 좋은 값이 **멀리**서 나오면 $\sigma$ 가 는다. 나쁜 값이면 반대다.

![왼쪽은 가우스 분포 곡선 위에 평균에서 표준편차 하나 안쪽 구간을 칠한 그림이다. 그 안에서 좋은 값이 나오면 폭이 좁아지고 바깥에서 나오면 넓어진다고 화살표로 적혀 있으며, 아래에 시그마는 언제나 뽑힌 액션이 다시 나오기 쉬워지는 쪽으로 바뀐다고 적혀 있다. 오른쪽은 씨앗 스무 개를 돌려 평균이 어디서 멈췄는지 찍은 그림으로, 베이스라인이 없으면 점들이 0.3에서 5.1까지 흩어지고 스무 번 가운데 한 번만 봉우리에 머무는 반면 베이스라인이 있으면 스무 번 모두 봉우리인 3에 붙어 있다.](figures/gaussian_sigma.svg)

**이 규칙에는 구멍이 하나 있고, 윌리엄스가 직접 적어 두었다.** 리워드가 늘 양수인데 베이스라인을 쓰지 않으면($b=0$), REINFORCE가 "$\mu$ 가 어떤 언덕 꼭대기에 닿기도 전에 $\sigma$ 를 0으로 보내 버릴 수 있다"는 것이다. 까닭은 간단하다 — $r>0$ 이면 모든 표본이 **자기 자신을 북돋운다.** 좋았는지 나빴는지를 견줄 기준이 없으니 그냥 방금 한 것을 굳힌다.

오른쪽 그림이 그것을 재어 본 것이다. 봉우리가 하나인 문제를 씨앗 스무 개로 돌렸더니, 두 경우 모두 학습 중에 **한 번은** 봉우리를 지나간다(20/20). 그런데 끝에 거기 **머무는** 것은 베이스라인이 있을 때 20번 모두인 반면, $b=0$ 일 때는 **스무 번 중 한 번**뿐이다. 나머지는 0.3에서 5.1 사이 아무 데서나 $\sigma$ 가 닫힌 채 멈춘다.

**그러니 $\sigma$ 가 학습된다는 말은 탐색이 알아서 잘 조절된다는 뜻이 아니다.** $\sigma$ 는 "지금 한 것을 더 하기 쉽게" 움직일 뿐이고, 무엇이 좋은지 판단할 기준이 없으면 **아무 자리에서나 닫힌다.** 위 코드에서 가우스 쪽이 소프트맥스 쪽보다 손이 더 가는 까닭이 여기에 있다 — 파라메터가 하나 더 늘어난 만큼 잘못 수렴할 길도 하나 더 늘었고, 그 길을 막는 것이 베이스라인이다. 쪽을 나누어 본다면 이 상자는 [베이스라인](baselines.md) 쪽과 한 몸이다.

_확인한 곳 — R. J. Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning", *Machine Learning* **8**, 229–256 (1992). $\sigma$ 를 다스리는 것이 탐색을 다스리는 것이라는 말은 6절, $\sigma$ 가 움직이는 규칙과 $b=0$ 일 때 0으로 수렴할 수 있다는 관찰은 8.2절이다. 오른쪽 그림의 수는 그림을 그리는 코드가 씨앗 스무 개로 직접 돌려 얻으며, 양쪽 모두 봉우리에 한 번은 닿는지를 먼저 확인한 뒤에만 그린다. 다만 여기서 본 것은 $\mu$ 가 봉우리에 닿기 전에 닫히는 것이 아니라 **닿았다가 벗어난 자리에서 닫히는** 꼴이다 — 같은 조기 수렴이되 윌리엄스가 적은 바로 그 모양은 아니다._

</div>

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
보여 주기 코드를 돌리고 중요한 출력 재기를 적어라. 파라메터 하나(러닝레이트, 숨은 차원, 켜 개수 따위)를 고쳐 열매가 어떻게 달라지는지 밝혀라.

</div>

??? success "풀이"
    보여 주기를 돌린 뒤 다른 것을 붙박아 두고 고른 파라메터만 구조 있게 바꾼다. 보기로 숨은 차원을 곱절로 늘리면 나타내는 그릇이 커지지만 셈하는 때가 는다. 러닝레이트는 한결같지 않은 결과를 낳는다. 너무 작으면 더디게 모이고 너무 크면 들쭉날쭉해진다. 고른 파라메터의 서로 다른 값 적어도 셋에 대해 또렷한 수를 적어 두라.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 구현에서 중요한 구조 공정성이 맡은 몫을 풀어라. 왜 그런 엑티베이션 함수, 고르게 하기 꾀, 로스 함수를 쓰는가? 다른 것으로 바꾸면 무슨 일이 생기는가?

</div>

??? success "풀이"
    이 구조 공정성은 폴리시 경사 기초에서 자리 잡은 좋은 버릇을 비춘다. 보기로 ReLU 활성은 곧지 않음을 주면서 0보다 큰 입력에서 그래디언트가 사라지는 것을 막는다. 로스 함수는 과제 클래스에 맞추어 고른다(분류에는 교차 엔트로피, 회귀에는 평균 제곱 오차). 다른 것으로 바꾸면(보기로 시그모이드 활성, L1 로스) 최적화 지형이 바뀌어 성능이 나빠질 수 있으나, 어떤 자리에서는 바꾸는 것이 이로울 수도 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
이 구현을 더 만만치 않은 자리로 넓혀라. 더 큰 데이터셋, 다른 문제 클래스, 덧붙인 기능 가운데 하나를 고르라. 고친 바를 밝히고 성능에 미친 바를 따져라.

</div>

??? success "풀이"
    절로 떠오르는 넓히기 하나는 정규화(드롭아웃, 웨이트 삭임)나 더 야무진 구조(켜 더하기, 건너뛰는 이음)를 더하는 것이다. 고른 넓히기를 만들고 같은 데이터로 학습된 뒤 앞뒤의 재기를 비교하여라. 이 넓히기는 처음 알고리즘과 고친 바의 이치 밑뜻을 모두 아는 것을 보여야 한다.

## 정리하며

**다룬 것** — 폴리시 파라메터 나타내기

이 구현은 폴리시 파라메터 나타내기의 한가운데 논리를 담은 `DiscretePolicy`, `GaussianPolicy`, `SquashedGaussianPolicy` 클래스를 축으로 삼는다.

핵심 클래스는 `DiscretePolicy`, `GaussianPolicy`, `SquashedGaussianPolicy`, `BetaPolicy`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
