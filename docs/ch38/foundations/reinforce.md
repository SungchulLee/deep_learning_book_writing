# REINFORCE 알고리즘

REINFORCE는 어림 쌓인 리워드를 곧바로 가장 좋게 하여 파라메터로 나타낸 폴리시를 배우는, 기초가 되는 몬테카를로 폴리시 경사 알고리즘이다. 온전한 에피소드를 모으고 깎은 리턴을 셈한 뒤, 리워드가 높은 액션의 확률을 올리는 쪽으로 폴리시 파라메터를 고친다. 이 구현은 맹탕 REINFORCE, 앞으로의 리워드로 분산을 줄이는 재주, 리턴 고르게 하기, 배치 고침을 다루며, 다듬을 때마다 따로 떨어진 다스리기와 이어진 다스리기 과제에서 학습이 어떻게 더 든든해지는지 보인다.

## 1. 코드

??? note "코드 (422줄)"

    ```python
    """
    34.1.3장: REINFORCE 알고리즘
    =====================================
    여러 클래스를 갖춘 온전한 REINFORCE 구현:
    - 맹탕 REINFORCE
    - 앞으로의 리워드를 쓰는 REINFORCE
    - 돌아옴 고르게 하기를 쓰는 REINFORCE
    - CartPole과 이어진 다스리기에서 익히기
    """

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical, Normal
    import numpy as np
    import gymnasium as gym
    from typing import List, Tuple, Optional
    from collections import deque

    # ========================================================================
    # 메인
    # ========================================================================


    # ---------------------------------------------------------------------------
    # 폴리시 넷트웍
    # ---------------------------------------------------------------------------

    class DiscretePolicyNetwork(nn.Module):
        """따로 떨어진 액션을 위한 폴리시 넷트웍."""

        def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, act_dim),
            )

        def forward(self, obs: torch.Tensor) -> Categorical:
            logits = self.net(obs)
            return Categorical(logits=logits)


    class ContinuousPolicyNetwork(nn.Module):
        """이어진 액션을 위한 폴리시 넷트웍."""

        def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.mean_head = nn.Linear(hidden_dim, act_dim)
            self.log_std = nn.Parameter(torch.zeros(act_dim))

        def forward(self, obs: torch.Tensor) -> Normal:
            features = self.net(obs)
            mean = self.mean_head(features)
            std = self.log_std.exp().expand_as(mean)
            return Normal(mean, std)


    # ---------------------------------------------------------------------------
    # REINFORCE 에이전트
    # ---------------------------------------------------------------------------

    class REINFORCE:
        """
        REINFORCE 폴리시 그래디언트 에이전트.

        파라메터
        ----------
        env : gym.Env
            Gymnasium 둘레.
        lr : float
            러닝레이트.
        gamma : float
            깎기 인자.
        hidden_dim : int
            숨은 켜의 크기.
        use_reward_to_go : bool
            True이면 까닭 매김(앞으로의 리워드)을 쓴다. 아니면 온 돌아옴을 쓴다.
        normalize_returns : bool
            True이면 돌아옴을 평균 0, 흩어짐 1로 고르게 한다.
        entropy_coef : float
            엔트로피 덤 계수(살펴보기를 북돋운다).
        """

        def __init__(
            self,
            env: gym.Env,
            lr: float = 1e-3,
            gamma: float = 0.99,
            hidden_dim: int = 128,
            use_reward_to_go: bool = True,
            normalize_returns: bool = True,
            entropy_coef: float = 0.01,
        ):
            self.env = env
            self.gamma = gamma
            self.use_reward_to_go = use_reward_to_go
            self.normalize_returns = normalize_returns
            self.entropy_coef = entropy_coef

            # 액션 공간 클래스를 알아낸다
            obs_dim = env.observation_space.shape[0]
            self.continuous = isinstance(env.action_space, gym.spaces.Box)

            if self.continuous:
                act_dim = env.action_space.shape[0]
                self.policy = ContinuousPolicyNetwork(obs_dim, act_dim, hidden_dim)
            else:
                act_dim = env.action_space.n
                self.policy = DiscretePolicyNetwork(obs_dim, act_dim, hidden_dim)

            self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        def select_action(self, obs: np.ndarray) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
            """지금 폴리시에서 액션을 고른다."""
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            dist = self.policy(obs_t)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            if self.continuous:
                log_prob = log_prob.sum(dim=-1)
                entropy = entropy.sum(dim=-1)
                return action.detach().numpy().flatten(), log_prob, entropy
            else:
                return action.item(), log_prob, entropy

        def compute_returns(self, rewards: List[float]) -> torch.Tensor:
            """깎은 돌아옴을 셈한다."""
            if self.use_reward_to_go:
                # 앞으로의 리워드: G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
                returns = []
                G = 0.0
                for r in reversed(rewards):
                    G = r + self.gamma * G
                    returns.insert(0, G)
                returns = torch.tensor(returns, dtype=torch.float32)
            else:
                # 온 자취 돌아옴: 모든 때 스텝에 R(τ)로 무게를 준다
                R = sum(self.gamma ** t * r for t, r in enumerate(rewards))
                returns = torch.full((len(rewards),), R, dtype=torch.float32)

            if self.normalize_returns and len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            return returns

        def collect_episode(self) -> Tuple[List, List, List, float]:
            """온전한 에피소드 하나를 모은다."""
            obs, _ = self.env.reset()
            log_probs, entropies, rewards = [], [], []

            done = False
            while not done:
                action, log_prob, entropy = self.select_action(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)

                log_probs.append(log_prob)
                entropies.append(entropy)
                rewards.append(reward)

                obs = next_obs
                done = terminated or truncated

            return log_probs, entropies, rewards, sum(rewards)

        def update(self, log_probs: List, entropies: List, rewards: List):
            """REINFORCE 고침을 한 번 벌인다."""
            returns = self.compute_returns(rewards)

            # 로그 확률과 엔트로피를 쌓는다
            log_probs_t = torch.stack(log_probs).squeeze()
            entropies_t = torch.stack(entropies).squeeze()

            # 폴리시 그래디언트 로스: -E[log π(a|s) · G_t]
            policy_loss = -(log_probs_t * returns).mean()

            # 살펴보기를 위한 엔트로피 덤
            entropy_loss = -entropies_t.mean()

            # 온 로스
            loss = policy_loss + self.entropy_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            # 든든함을 위한 그래디언트 자르기
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

            return policy_loss.item(), entropies_t.mean().item()

        def train(
            self,
            n_episodes: int = 1000,
            print_interval: int = 100,
            solved_reward: Optional[float] = None,
        ) -> List[float]:
            """
            REINFORCE로 에이전트를 익힌다.

            파라메터
            ----------
            n_episodes : int
                익힘 에피소드의 개수.
            print_interval : int
                나아감을 얼마나 자주 찍을지.
            solved_reward : float, 고를 수 있음
                주어지면 평균 리워드가 이를 넘을 때 익힘을 멈춘다.

            돌려주는 값
            -------
            episode_rewards : float의 목록
                리워드 내력.
            """
            episode_rewards = []
            recent_rewards = deque(maxlen=100)

            for episode in range(1, n_episodes + 1):
                log_probs, entropies, rewards, total_reward = self.collect_episode()
                policy_loss, avg_entropy = self.update(log_probs, entropies, rewards)

                episode_rewards.append(total_reward)
                recent_rewards.append(total_reward)
                avg_reward = np.mean(recent_rewards)

                if episode % print_interval == 0:
                    print(
                        f"Episode {episode:>5d} | "
                        f"Reward: {total_reward:>7.1f} | "
                        f"Avg(100): {avg_reward:>7.1f} | "
                        f"Loss: {policy_loss:>8.4f} | "
                        f"Entropy: {avg_entropy:>6.3f}"
                    )

                if solved_reward is not None and avg_reward >= solved_reward:
                    print(f"\nSolved in {episode} episodes! Avg reward: {avg_reward:.1f}")
                    break

            return episode_rewards


    # ---------------------------------------------------------------------------
    # 배치 REINFORCE (고침마다 에피소드 여럿)
    # ---------------------------------------------------------------------------

    class BatchREINFORCE(REINFORCE):
        """
        흩어짐을 줄이려 배치로 고치는 REINFORCE.

        그래디언트 고침을 벌이기 앞서 에피소드 여럿을 모아 배치에 걸쳐
        그래디언트를 고르게 한다.
        """

        def __init__(self, *args, batch_size: int = 10, **kwargs):
            super().__init__(*args, **kwargs)
            self.batch_size = batch_size

        def train(
            self,
            n_episodes: int = 1000,
            print_interval: int = 100,
            solved_reward: Optional[float] = None,
        ) -> List[float]:
            episode_rewards = []
            recent_rewards = deque(maxlen=100)

            episode = 0
            while episode < n_episodes:
                # 에피소드 배치를 모은다
                batch_log_probs = []
                batch_entropies = []
                batch_returns = []
                batch_rewards = []

                for _ in range(self.batch_size):
                    log_probs, entropies, rewards, total_reward = self.collect_episode()
                    returns = self.compute_returns(rewards)

                    batch_log_probs.extend(log_probs)
                    batch_entropies.extend(entropies)
                    batch_returns.append(returns)
                    batch_rewards.append(total_reward)

                    episode += 1
                    episode_rewards.append(total_reward)
                    recent_rewards.append(total_reward)

                # 온 넘어감을 잇는다
                all_log_probs = torch.stack(batch_log_probs).squeeze()
                all_entropies = torch.stack(batch_entropies).squeeze()
                all_returns = torch.cat(batch_returns)

                # 온 배치에 걸쳐 돌아옴을 고르게 한다
                if self.normalize_returns:
                    all_returns = (all_returns - all_returns.mean()) / (all_returns.std() + 1e-8)

                # 그래디언트 고침 한 번
                policy_loss = -(all_log_probs * all_returns).mean()
                entropy_loss = -all_entropies.mean()
                loss = policy_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

                avg_reward = np.mean(recent_rewards)
                if episode % print_interval < self.batch_size:
                    print(
                        f"Episode {episode:>5d} | "
                        f"Batch Avg: {np.mean(batch_rewards):>7.1f} | "
                        f"Avg(100): {avg_reward:>7.1f}"
                    )

                if solved_reward is not None and avg_reward >= solved_reward:
                    print(f"\nSolved in {episode} episodes! Avg reward: {avg_reward:.1f}")
                    break

            return episode_rewards


    # ---------------------------------------------------------------------------
    # 보여 주기
    # ---------------------------------------------------------------------------

    def train_cartpole():
        """CartPole-v1에서 REINFORCE를 익힌다."""
        print("=" * 60)
        print("REINFORCE on CartPole-v1")
        print("=" * 60)

        env = gym.make("CartPole-v1")

        agent = REINFORCE(
            env=env,
            lr=1e-3,
            gamma=0.99,
            hidden_dim=128,
            use_reward_to_go=True,
            normalize_returns=True,
            entropy_coef=0.01,
        )

        rewards = agent.train(n_episodes=1000, print_interval=100, solved_reward=475.0)
        env.close()
        return rewards


    def train_cartpole_batch():
        """CartPole-v1에서 배치 REINFORCE를 익힌다."""
        print("\n" + "=" * 60)
        print("Batch REINFORCE on CartPole-v1")
        print("=" * 60)

        env = gym.make("CartPole-v1")

        agent = BatchREINFORCE(
            env=env,
            lr=1e-3,
            gamma=0.99,
            hidden_dim=128,
            use_reward_to_go=True,
            normalize_returns=True,
            entropy_coef=0.01,
            batch_size=10,
        )

        rewards = agent.train(n_episodes=1000, print_interval=100, solved_reward=475.0)
        env.close()
        return rewards


    def compare_variants():
        """CartPole에서 REINFORCE 클래스들을 견준다."""
        print("\n" + "=" * 60)
        print("Comparing REINFORCE Variants")
        print("=" * 60)

        variants = {
            "Total Return": {"use_reward_to_go": False, "normalize_returns": False},
            "Reward-to-Go": {"use_reward_to_go": True, "normalize_returns": False},
            "RTG + Normalize": {"use_reward_to_go": True, "normalize_returns": True},
        }

        n_episodes = 500
        n_trials = 3

        for name, kwargs in variants.items():
            trial_rewards = []
            for trial in range(n_trials):
                env = gym.make("CartPole-v1")
                torch.manual_seed(trial)
                np.random.seed(trial)

                agent = REINFORCE(
                    env=env, lr=1e-3, gamma=0.99, hidden_dim=128,
                    entropy_coef=0.01, **kwargs
                )

                rewards = agent.train(n_episodes=n_episodes, print_interval=n_episodes + 1)
                trial_rewards.append(np.mean(rewards[-100:]))
                env.close()

            avg = np.mean(trial_rewards)
            std = np.std(trial_rewards)
            print(f"{name:<20}: Final Avg Reward = {avg:.1f} ± {std:.1f}")


    if __name__ == "__main__":
        train_cartpole()
        train_cartpole_batch()
        compare_variants()
    ```


## 2. 논의

이 구현은 REINFORCE 알고리즘의 한가운데 논리를 담은 `DiscretePolicyNetwork`, `ContinuousPolicyNetwork`, `REINFORCE` 클래스를 축으로 삼는다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 갈라놓는 조각 설계를 따른다.

보여 주기 함수는 이 조각들을 여느 강화학습 잣대에 실제로 써 보인다. 그 출력을 살피면 파라메터 공정성과 문제 구조에 따라 알고리즘의 성능이 어떻게 달라지는지 볼 수 있다.

쓰임의 눈으로 보면 이 구현은 날 성능보다 또렷함을 앞세운다. 서비스 시스템은 배치 계산, GPU 빠르게 하기, 더 야무진 파라메터 벼리기 같은 다듬기를 더 넣는 것이 보통이다. 그렇더라도 여기서 보인 한가운데 알고리즘 생각은 큰 잣대의 쓰임새에 그대로 옮겨 간다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
보여 주기 코드를 돌리고 중요한 출력 재기를 적어라. 파라메터 하나(러닝레이트, 숨은 차원, 켜 개수 따위)를 고쳐 열매가 어떻게 달라지는지 밝혀라.

</div>

??? success "연습문제 1 풀이"
    보여 주기를 돌린 뒤 다른 것을 붙박아 두고 고른 파라메터만 구조 있게 바꾼다. 보기로 숨은 차원을 곱절로 늘리면 나타내는 그릇이 커지지만 셈하는 때가 는다. 러닝레이트는 한결같지 않은 결과를 낳는다. 너무 작으면 더디게 모이고 너무 크면 들쭉날쭉해진다. 고른 파라메터의 서로 다른 값 적어도 셋에 대해 또렷한 수를 적어 두라.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 구현에서 중요한 구조 공정성이 맡은 몫을 풀어라. 왜 그런 엑티베이션 함수, 고르게 하기 꾀, 로스 함수를 쓰는가? 다른 것으로 바꾸면 무슨 일이 생기는가?

</div>

??? success "연습문제 2 풀이"
    이 구조 공정성은 폴리시 경사 기초에서 자리 잡은 좋은 버릇을 비춘다. 보기로 ReLU 활성은 곧지 않음을 주면서 0보다 큰 입력에서 그래디언트가 사라지는 것을 막는다. 로스 함수는 과제 클래스에 맞추어 고른다(분류에는 교차 엔트로피, 회귀에는 평균 제곱 오차). 다른 것으로 바꾸면(보기로 시그모이드 활성, L1 로스) 최적화 지형이 바뀌어 성능이 나빠질 수 있으나, 어떤 자리에서는 바꾸는 것이 이로울 수도 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
이 구현을 더 만만치 않은 자리로 넓혀라. 더 큰 데이터셋, 다른 문제 클래스, 덧붙인 기능 가운데 하나를 고르라. 고친 바를 밝히고 성능에 미친 바를 따져라.

</div>

??? success "연습문제 3 풀이"
    절로 떠오르는 넓히기 하나는 정규화(드롭아웃, 웨이트 삭임)나 더 야무진 구조(켜 더하기, 건너뛰는 이음)를 더하는 것이다. 고른 넓히기를 만들고 같은 데이터로 학습된 뒤 앞뒤의 재기를 비교하여라. 이 넓히기는 처음 알고리즘과 고친 바의 이치 밑뜻을 모두 아는 것을 보여야 한다.

## 정리하며

**다룬 것** — REINFORCE 알고리즘

이 구현은 REINFORCE 알고리즘의 한가운데 논리를 담은 `DiscretePolicyNetwork`, `ContinuousPolicyNetwork`, `REINFORCE` 클래스를 축으로 삼는다.

핵심 클래스는 `DiscretePolicyNetwork`, `ContinuousPolicyNetwork`, `REINFORCE`, `BatchREINFORCE`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
