# 베이스라인

베이스라인 방법은 폴리시 경사 기초에서 중요한 생각이다. 폴리시 경사 알고리즘에서 쓰인다. 이 구현은 여기에 걸린 중요한 알고리즘과 자료 구조를 손으로 만져 보이며, 이치 기초와 실제로 서비스에 올릴 때 살필 것을 함께 보여 준다.

## 1. 코드

??? note "코드 (442줄)"

    ```python
    """
    34.1.4장: 밑금 방법
    =================================
    폴리시 그래디언트 알고리즘에서 흩어짐을 줄이는 여러 밑금 방법의 구현.
    """

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    import numpy as np
    import gymnasium as gym
    from typing import List, Tuple
    from collections import deque

    # ========================================================================
    # 메인
    # ========================================================================


    # ---------------------------------------------------------------------------
    # 넷트웍
    # ---------------------------------------------------------------------------

    class ValueNetwork(nn.Module):
        """밑금으로 쓰는 상태 값 함수 V(s)."""

        def __init__(self, obs_dim: int, hidden_dim: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return self.net(obs).squeeze(-1)


    class PolicyNetwork(nn.Module):
        """따로 떨어진 액션을 위한 쉬운 클래스 폴리시."""

        def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, act_dim),
            )

        def forward(self, obs: torch.Tensor) -> Categorical:
            return Categorical(logits=self.net(obs))


    # ---------------------------------------------------------------------------
    # 밑금 꾀
    # ---------------------------------------------------------------------------

    class ConstantBaseline:
        """돌아옴의 흐르는 평균을 상수 밑금으로 쓴다."""

        def __init__(self, decay: float = 0.99):
            self.value = 0.0
            self.decay = decay
            self.initialized = False

        def update(self, returns: List[float]):
            avg = np.mean(returns)
            if not self.initialized:
                self.value = avg
                self.initialized = True
            else:
                self.value = self.decay * self.value + (1 - self.decay) * avg

        def get_baseline(self, states: torch.Tensor) -> torch.Tensor:
            return torch.full((states.shape[0],), self.value)


    class LearnedBaseline:
        """배운, 상태에 딸린 값 함수 밑금 V_phi(s)."""

        def __init__(self, obs_dim: int, hidden_dim: int = 128, lr: float = 1e-3, n_epochs: int = 5):
            self.value_net = ValueNetwork(obs_dim, hidden_dim)
            self.optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
            self.n_epochs = n_epochs

        def update(self, states: torch.Tensor, returns: torch.Tensor):
            for _ in range(self.n_epochs):
                values = self.value_net(states)
                loss = nn.functional.mse_loss(values, returns)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            return loss.item()

        def get_baseline(self, states: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return self.value_net(states)


    # ---------------------------------------------------------------------------
    # 밑금을 쓰는 REINFORCE 에이전트
    # ---------------------------------------------------------------------------

    class REINFORCEWithBaseline:
        """
        밑금 방법을 골라 쓸 수 있는 REINFORCE 에이전트.

        파라메터
        ----------
        env : gym.Env
            둘레.
        baseline_type : str
            'none', 'constant', 'learned' 가운데 하나.
        """

        def __init__(
            self,
            env: gym.Env,
            baseline_type: str = "learned",
            lr_policy: float = 1e-3,
            lr_value: float = 1e-3,
            gamma: float = 0.99,
            hidden_dim: int = 128,
            normalize_advantages: bool = True,
            entropy_coef: float = 0.01,
        ):
            self.env = env
            self.gamma = gamma
            self.normalize_advantages = normalize_advantages
            self.entropy_coef = entropy_coef
            self.baseline_type = baseline_type

            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.n

            self.policy = PolicyNetwork(obs_dim, act_dim, hidden_dim)
            self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr_policy)

            if baseline_type == "constant":
                self.baseline = ConstantBaseline()
            elif baseline_type == "learned":
                self.baseline = LearnedBaseline(obs_dim, hidden_dim, lr=lr_value)
            else:
                self.baseline = None

        def compute_returns(self, rewards: List[float]) -> torch.Tensor:
            returns = []
            G = 0.0
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            return torch.tensor(returns, dtype=torch.float32)

        def collect_episode(self):
            obs, _ = self.env.reset()
            states, actions, log_probs, entropies, rewards = [], [], [], [], []

            done = False
            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0)
                dist = self.policy(obs_t)
                action = dist.sample()

                states.append(obs)
                actions.append(action.item())
                log_probs.append(dist.log_prob(action))
                entropies.append(dist.entropy())

                obs, reward, terminated, truncated, _ = self.env.step(action.item())
                rewards.append(reward)
                done = terminated or truncated

            return states, actions, log_probs, entropies, rewards

        def update(self, states, actions, log_probs, entropies, rewards):
            returns = self.compute_returns(rewards)
            states_t = torch.FloatTensor(np.array(states))

            # 이점을 셈한다
            if self.baseline is None:
                advantages = returns.clone()
            elif self.baseline_type == "constant":
                self.baseline.update(returns.numpy().tolist())
                advantages = returns - self.baseline.get_baseline(states_t)
            elif self.baseline_type == "learned":
                advantages = returns - self.baseline.get_baseline(states_t)
                self.baseline.update(states_t, returns)

            if self.normalize_advantages and len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # 폴리시 로스
            log_probs_t = torch.stack(log_probs).squeeze()
            entropies_t = torch.stack(entropies).squeeze()

            policy_loss = -(log_probs_t * advantages.detach()).mean()
            entropy_loss = -entropies_t.mean()
            loss = policy_loss + self.entropy_coef * entropy_loss

            self.policy_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.policy_optimizer.step()

            return policy_loss.item(), entropies_t.mean().item()

        def train(self, n_episodes: int = 1000, print_interval: int = 100) -> List[float]:
            episode_rewards = []
            recent_rewards = deque(maxlen=100)

            for episode in range(1, n_episodes + 1):
                states, actions, log_probs, entropies, rewards = self.collect_episode()
                policy_loss, avg_entropy = self.update(states, actions, log_probs, entropies, rewards)

                total_reward = sum(rewards)
                episode_rewards.append(total_reward)
                recent_rewards.append(total_reward)

                if episode % print_interval == 0:
                    print(
                        f"Episode {episode:>5d} | "
                        f"Reward: {total_reward:>7.1f} | "
                        f"Avg(100): {np.mean(recent_rewards):>7.1f} | "
                        f"Loss: {policy_loss:>8.4f}"
                    )

            return episode_rewards


    # ---------------------------------------------------------------------------
    # 이점 어림 보여 주기
    # ---------------------------------------------------------------------------

    def compute_mc_advantages(rewards, values, gamma=0.99):
        """몬테카를로 이점: A_t = G_t - V(s_t)."""
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        return returns - values


    def compute_td_advantages(rewards, values, next_values, dones, gamma=0.99):
        """한 스텝 때 차이 이점: A_t = r_t + γV(s_{t+1}) - V(s_t)."""
        td_targets = torch.tensor(rewards) + gamma * next_values * (1 - torch.tensor(dones, dtype=torch.float32))
        return td_targets - values


    def compute_nstep_advantages(rewards, values, next_values, dones, gamma=0.99, n=5):
        """n스텝 때 차이 이점 어림."""
        T = len(rewards)
        advantages = torch.zeros(T)

        for t in range(T):
            G = 0.0
            for k in range(min(n, T - t)):
                G += gamma ** k * rewards[t + k]
                if dones[t + k]:
                    break
            else:
                if t + n < T:
                    G += gamma ** n * next_values[t + n].item()
            advantages[t] = G - values[t].item()

        return advantages


    def demo_advantage_comparison():
        """이점 어림 방법들을 견준다."""
        print("=" * 60)
        print("Advantage Estimation Comparison")
        print("=" * 60)

        # 흉내 낸 에피소드 자료
        T = 20
        torch.manual_seed(42)
        rewards = [1.0] * T  # 붙박인 리워드
        rewards[-1] = 10.0   # 끝 덤

        # 흉내 낸 값 어림(온전하지 않다)
        true_values = torch.tensor([
            sum(0.99 ** (k - t) * rewards[k] for k in range(t, T))
            for t in range(T)
        ])
        noise = torch.randn(T) * 0.5
        estimated_values = true_values + noise
        next_values = torch.cat([estimated_values[1:], torch.zeros(1)])
        dones = [False] * (T - 1) + [True]

        # 이점을 셈한다
        mc_adv = compute_mc_advantages(rewards, estimated_values, gamma=0.99)
        td_adv = compute_td_advantages(rewards, estimated_values, next_values, dones, gamma=0.99)
        n5_adv = compute_nstep_advantages(rewards, estimated_values, next_values, dones, gamma=0.99, n=5)

        print(f"\n{'Step':>4} {'MC Adv':>10} {'TD(0) Adv':>10} {'TD(5) Adv':>10}")
        print("-" * 38)
        for t in range(min(10, T)):
            print(f"{t:>4} {mc_adv[t]:>10.4f} {td_adv[t]:>10.4f} {n5_adv[t]:>10.4f}")

        print(f"\n{'Method':<12} {'Mean':>8} {'Std':>8} {'|Max|':>8}")
        print("-" * 38)
        for name, adv in [("MC", mc_adv), ("TD(0)", td_adv), ("TD(5)", n5_adv)]:
            print(f"{name:<12} {adv.mean():>8.4f} {adv.std():>8.4f} {adv.abs().max():>8.4f}")


    # ---------------------------------------------------------------------------
    # CartPole에서 밑금 견주기
    # ---------------------------------------------------------------------------

    def compare_baselines():
        """CartPole에서 여러 밑금을 견준다."""
        print("\n" + "=" * 60)
        print("Baseline Comparison on CartPole-v1")
        print("=" * 60)

        baselines = ["none", "constant", "learned"]
        n_episodes = 500
        n_trials = 3

        results = {}

        for bl_type in baselines:
            trial_final_rewards = []

            for trial in range(n_trials):
                torch.manual_seed(trial)
                np.random.seed(trial)
                env = gym.make("CartPole-v1")

                agent = REINFORCEWithBaseline(
                    env=env,
                    baseline_type=bl_type,
                    lr_policy=1e-3,
                    lr_value=1e-3,
                    gamma=0.99,
                    normalize_advantages=True,
                    entropy_coef=0.01,
                )

                rewards = agent.train(n_episodes=n_episodes, print_interval=n_episodes + 1)
                trial_final_rewards.append(np.mean(rewards[-100:]))
                env.close()

            results[bl_type] = trial_final_rewards

        print(f"\nResults after {n_episodes} episodes (avg of last 100, {n_trials} trials):")
        print(f"{'Baseline':<18} {'Mean':>8} {'Std':>8}")
        print("-" * 36)
        for bl_type, vals in results.items():
            print(f"{bl_type:<18} {np.mean(vals):>8.1f} {np.std(vals):>8.1f}")


    # ---------------------------------------------------------------------------
    # 흩어짐 살피기
    # ---------------------------------------------------------------------------

    def analyze_gradient_variance():
        """밑금에 따라 그래디언트 흩어짐을 잰다."""
        print("\n" + "=" * 60)
        print("Gradient Variance Analysis")
        print("=" * 60)

        env = gym.make("CartPole-v1")
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n

        torch.manual_seed(42)
        policy = PolicyNetwork(obs_dim, act_dim, hidden_dim=64)

        # 에피소드를 모은다
        n_episodes = 50
        all_episodes = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            episode = {"states": [], "actions": [], "rewards": []}
            done = False
            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    dist = policy(obs_t)
                action = dist.sample().item()
                next_obs, reward, terminated, truncated, _ = env.step(action)
                episode["states"].append(obs)
                episode["actions"].append(action)
                episode["rewards"].append(reward)
                obs = next_obs
                done = terminated or truncated
            all_episodes.append(episode)

        # 에피소드마다 밑금을 달리하여 그래디언트를 셈한다
        baselines_to_test = {
            "No baseline": lambda returns, states: torch.zeros_like(returns),
            "Mean return": lambda returns, states: torch.full_like(returns, returns.mean()),
            "Per-step mean": lambda returns, states: returns.mean().expand_as(returns),
        }

        for bl_name, bl_fn in baselines_to_test.items():
            grad_norms = []

            for ep in all_episodes:
                # 돌아옴을 셈한다
                G = 0.0
                returns = []
                for r in reversed(ep["rewards"]):
                    G = r + 0.99 * G
                    returns.insert(0, G)
                returns = torch.tensor(returns, dtype=torch.float32)
                states_t = torch.FloatTensor(np.array(ep["states"]))
                actions_t = torch.tensor(ep["actions"])

                baseline_vals = bl_fn(returns, states_t)
                advantages = returns - baseline_vals

                # 그래디언트를 셈한다
                policy.zero_grad()
                dist = policy(states_t)
                log_probs = dist.log_prob(actions_t)
                loss = -(log_probs * advantages.detach()).mean()
                loss.backward()

                grad_norm = sum(
                    p.grad.norm().item() ** 2
                    for p in policy.parameters()
                    if p.grad is not None
                ) ** 0.5
                grad_norms.append(grad_norm)

            mean_gn = np.mean(grad_norms)
            std_gn = np.std(grad_norms)
            cv = std_gn / (mean_gn + 1e-8)
            print(f"{bl_name:<20}: grad_norm = {mean_gn:.4f} ± {std_gn:.4f} (CV: {cv:.4f})")

        env.close()


    if __name__ == "__main__":
        demo_advantage_comparison()
        compare_baselines()
        analyze_gradient_variance()
    ```


## 2. 논의

이 구현은 베이스라인 방법의 한가운데 논리를 담은 `ValueNetwork`, `PolicyNetwork`, `ConstantBaseline` 클래스를 축으로 삼는다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 갈라놓는 조각 설계를 따른다.

보여 주기 함수는 이 조각들을 여느 강화학습 잣대에 실제로 써 보인다. 그 출력을 살피면 파라메터 공정성과 문제 구조에 따라 알고리즘의 성능이 어떻게 달라지는지 볼 수 있다.

쓰임의 눈으로 보면 이 구현은 날 성능보다 또렷함을 앞세운다. 서비스 시스템은 배치 계산, GPU 빠르게 하기, 더 야무진 파라메터 벼리기 같은 다듬기를 더 넣는 것이 보통이다. 그렇더라도 여기서 보인 한가운데 알고리즘 생각은 큰 잣대의 쓰임새에 그대로 옮겨 간다.

<div class="storybox" markdown>

**그림으로 보는 AI 1.** 평균을 빼는 것이 가장 좋은 선택은 아니다

베이스라인은 나중에 덧붙인 재주가 아니라 **처음부터 식 안에 있었다.** 로널드 윌리엄스(Ronald Williams)가 1992년에 REINFORCE를 정의한 식이

$$
\Delta w_{ij} = \alpha_{ij}\,(r - b_{ij})\,e_{ij}
$$

이고, 그 $b_{ij}$ 가 베이스라인이다. REINFORCE라는 이름 자체가 이 네 조각의 머리글자다 — "REward Increment = Nonnegative Factor × Offset Reinforcement × Characteristic Eligibility". 가운데의 **Offset Reinforcement** 가 바로 $r - b$ 다.

그런데 윌리엄스는 같은 논문 8.3절에서 **무엇을 빼야 하는지는 자기 분석이 말해 주지 않는다**고 적어 두었다. "One important limitation of the analysis given here is that it offers no basis for choosing among various choices of reinforcement baseline." 그러고는 분산을 가장 작게 만드는 선택을 따로 언급하며 이렇게 덧붙인다 — "This turns out to yield **not the mean reinforcement** as in the usual reinforcement comparison approach, but another quantity that is more difficult to estimate effectively."

![왼쪽은 액션 셋짜리 작은 문제에서 파라메터마다 분산을 가장 작게 하는 베이스라인을 점으로 찍은 그림이다. 세 점이 0.574, 1.237, 1.538 로 서로 다르고, 평균 리턴을 나타내는 세로 점선 1.301 과도 모두 어긋나 있다. 오른쪽은 무작위로 뽑은 배치 2,000 개, 파라메터 6,947 개에 대해 평균 베이스라인을 쓸 때의 분산을 최적 베이스라인을 쓸 때의 분산으로 나눈 값의 분포로, 중앙값이 4.7배이고 평균 베이스라인이 베이스라인을 아예 안 쓰는 것보다 나쁜 경우가 21.6퍼센트라고 적혀 있다.](figures/baseline_not_the_mean.svg)

그 "다른 양"이 무엇인지 정확히 셈해 보면 이렇다. 분산을 최소로 만드는 베이스라인은 **리턴을 점수의 제곱으로 웨이트 주어 평균 낸 것**이다.

$$
b^{*}_{i} \;=\; \frac{\mathbb{E}\!\left[g_i^2\, r\right]}{\mathbb{E}\!\left[g_i^2\right]},
\qquad g_i = \frac{\partial \log \pi}{\partial \theta_i}
$$

**값함수가 아니다.** 그리고 분모·분자에 $g_i$ 가 들어 있으므로 **파라메터마다 값이 다르다.** 왼쪽 그림의 작은 보기에서 세 파라메터의 $b^{*}$ 는 0.574, 1.237, 1.538로 흩어지는데, 평균 리턴은 1.301 하나뿐이다.

얼마나 손해인지 세어 보았다. 배치를 무작위로 2,000가지 뽑아 파라메터 6,947개를 보면, 평균 리턴을 베이스라인으로 쓸 때의 분산이 $b^{*}$ 를 쓸 때보다 **중앙값으로 4.7배** 크다. 그리고 더 뜻밖의 쪽이 있다 — **21.6%에서는 평균을 빼는 것이 아예 안 빼는 것보다 나빴다.** 빼기만 하면 분산이 준다는 보장은 없다.

**그렇다고 이 쪽의 `ValueNetwork` 가 틀린 것은 아니다.** 윌리엄스가 같은 문장에서 적어 둔 대로 $b^{*}$ 는 "more difficult to estimate effectively" 다 — $g_i^2$ 으로 웨이트를 준 평균이라 파라메터마다 따로, 그것도 경사의 크기까지 알아야 어림할 수 있다. 값함수는 그보다 훨씬 싸게 얻어지고 대개 충분히 잘 듣는다. 요점은 값함수를 쓰지 말라는 것이 아니라, **그것이 최적이어서 쓰는 것이 아니라 쌀 수 있어서 쓰는 것**임을 알고 쓰자는 데 있다. 분산이 줄지 않는 자리를 만났을 때 베이스라인 말고 다른 데를 뒤지게 되는 것이 그 차이다.

_확인한 곳 — R. J. Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning", *Machine Learning* **8**, 229–256 (1992). 머리글자 풀이는 234쪽, 베이스라인을 고를 근거가 없다는 말과 "not the mean reinforcement" 는 8.3절이다. 그림의 수는 그림을 그리는 코드가 작은 밴딧에서 기댓값을 닫힌 식으로 **정확히** 셈해 얻는다. 다만 4.7배와 21.6%는 **거기서 뽑은 배치에 대해 센 값**이지 정리가 아니다 — 다르게 뽑으면 수도 달라진다._

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

**다룬 것** — 베이스라인 방법

이 구현은 베이스라인 방법의 한가운데 논리를 담은 `ValueNetwork`, `PolicyNetwork`, `ConstantBaseline` 클래스를 축으로 삼는다.

핵심 클래스는 `ValueNetwork`, `PolicyNetwork`, `ConstantBaseline`, `LearnedBaseline`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
